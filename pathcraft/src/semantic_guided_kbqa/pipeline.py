from __future__ import annotations

from collections import Counter
from copy import copy, deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import json
import itertools
import math
import re
from threading import Lock
import time
from types import SimpleNamespace
from typing import Any

from .clients import ChatClient, KnowledgeGraph
from .constraint_property_ranking import (
    answer_set_safety_reason,
    constraint_focus_phrase,
    nested_label_parent_ids,
    property_rank_evidence,
    relation_tail_text,
    semantic_only_is_confident,
)
from .contracts import (
    ContractError,
    DecompositionCandidate,
    ExecutedGraph,
    GroundedSemanticCandidate,
    QueryGraphCandidate,
    parse_json_object,
    validate_compose_graph_output,
    validate_operator_output,
    validate_compose_output,
    validate_semantic_graph,
)
from .data import DecompositionStore, TrainingContract
from .decomposition_review import DecompositionReviewer
from .relative_temporal import RelativeTemporalBinder, semantic_projection_items
from .selection import select_candidates
from .question_context import linked_entity_context
from .prompt_principles import SELECTOR_CONSTRAINT_POLICY
from .grounding import SemanticGrounder
from .graph_reconstruction import (
    graph_reconstruction_answer_role_gate,
    is_graph_reconstruction_graph,
    reconstruct_failure_query_graphs,
)
from .zero_graph_challenger import (
    build_zero_graph_challengers,
    zero_graph_challenger_gate,
    zero_graph_replacement_is_safe,
)
from .template_consensus_selector import SourceVerifiedTemplateConsensusSelector
from .path_alignment_selector import PathAlignmentSelector, REASON_CODE as PATH_ALIGNMENT_REASON
from .train_gold_extrema_selector import (
    REASON_CODE as TRAIN_GOLD_EXTREMA_REASON,
    TrainGoldPathExtremaSelector,
)
from .numeric_temporal_slot_normalization import (
    partition_fixed_slot_executions,
    postselect_fixed_slot_execution,
    prepare_fixed_execution_slots,
)
from .entity_kind_fixed_beam import (
    EntityKindFixedBeamPlanner,
    PreparedEntityTemplateQuery,
    execute_prepared_query as execute_entity_kind_prepared_query,
    linked_entities_for_question,
    postselect_entity_kind_execution,
    select_empty_entity_kind_execution,
)
from .underanswer_expansion_selector import UnderanswerExpansionSelector
from .ontology import relation_id_from_label, relation_label_from_id
from .grounding_repair import (
    build_prefix_repair_graphs,
    build_semantic_template_repair_graphs,
    build_structural_extension_graphs,
    discover_temporal_entity_intervals,
    repair_terminal_cvt_operator_output,
    repair_temporal_operator_output,
)
from .lowering import LoweringError, build_query_graph, build_query_graph_v2, lower_sparql
from .operator_prompt import (
    OPERATOR_INSTRUCTION,
    OPERATOR_INSTRUCTION_GLM,
    OPERATOR_INSTRUCTION_V2,
)
from .validation import GLMOutputCorrector, semantic_output_schema


SELECTOR_INSTRUCTION = """
You select the best executed Freebase query graph for the original question.
Return exactly one JSON object: {"selected_graph_id": "G...", "reason_codes": ["..."]}.
Only select a graph_id present in the candidates. Do not invent answers, facts, ids,
or graph ids. The supplied decompositions are an intermediate semantic interpretation
and may disambiguate noisy or ungrammatical wording in the original question. Prefer a
graph whose relation structure and answer values match the question and the reviewed
decompositions. The original question takes precedence if they conflict. Check all
explicit constraints and the requested answer type before using rule_score. Literal
answer values such as years, dates, and numbers are valid even when no entity label is
available. Use answer_count and graph structure as supporting evidence.
Treat an original-plan graph as the safe baseline. Prefer a review-rewrite graph only
when its actual triples or operators implement the concrete missing constraint; the
rewritten text by itself is not evidence that the executable graph is better.
""".strip()


_GLM_OPERATOR_PROMPT_MODES = {"glm_zero_shot", "glm_few_shot"}
SELECTOR_INSTRUCTION = SELECTOR_INSTRUCTION + "\n\n" + SELECTOR_CONSTRAINT_POLICY
_SELECTOR_ANSWER_PREVIEW = 100
_SELECTOR_RULE_GUARD_MARGIN = 0.05
_SELECTOR_ANSWER_UNION_MARGIN = 0.12
_CONJUNCTIVE_INTERSECTION_SCORE_MARGIN = 0.075
_ENTITY_KIND_FIXED_BEAM_REASON = "entity_kind_fixed_beam"
_EXPLICIT_CONJUNCTION_RE = re.compile(
    r"\b(?:and|also|both|as\s+well\s+as|while|but)\b",
    re.I,
)


_ENTITY_ID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")


def _answer_value_kind(answer_ids: list[str]) -> str:
    """Classify an executed answer set without consulting question-specific data."""
    if not answer_ids:
        return "empty"
    entity_flags = [bool(_ENTITY_ID_RE.fullmatch(str(value))) for value in answer_ids]
    if all(entity_flags):
        return "entity"
    if not any(entity_flags):
        return "literal"
    return "mixed"


def _rewritten_failure_requires_restore(
    failure_recovery_enabled: bool,
    executed: list[ExecutedGraph],
) -> bool:
    """Whether a rewritten plan's empty normal beam must reach outer restore."""
    return bool(
        not failure_recovery_enabled
        and not any(item.answer_ids for item in executed)
    )


def _failure_compiler_eligible(
    question: str,
    compose_trace: list[dict[str, Any]],
) -> bool:
    """Cheap gate before any graph copy or ontology conflict scan."""
    return bool(
        _missing_constraint_spec(question) is not None
        and any(
            item.get("valid") is False
            and isinstance(item.get("output"), dict)
            for item in compose_trace
        )
    )


def _expected_answer_kind(question: str) -> str:
    """Infer only high-confidence entity/literal answer forms from wh syntax.

    This intentionally stays conservative because CWQ can represent a year by
    its event entity rather than by a date literal.  Only unambiguous entity
    interrogatives are classified; time and numeric forms remain unknown.
    """
    text = " ".join(str(question).strip().casefold().split())
    if re.match(r"^(?:who|where)\b", text):
        return "entity"
    if re.match(
        r"^which\b",
        text,
    ) and not re.match(
        r"^which\s+(?:year|date|number|id|identifier|percentage|age|time|"
        r"netflix[_ ]id|gnis)\b",
        text,
    ):
        return "entity"
    return ""


def _rule_guard_crosses_answer_kind(
    question: str,
    model_selected: ExecutedGraph,
    rule_selected: ExecutedGraph,
) -> bool:
    """Keep the selector decision when a score guard would change answer form."""
    expected = _expected_answer_kind(question)
    return bool(
        expected
        and _answer_value_kind(model_selected.answer_ids) == expected
        and _answer_value_kind(rule_selected.answer_ids) != expected
    )


def _explicit_extrema_request(question: str) -> bool:
    """Return whether the surface explicitly asks for one ordered extreme."""
    # Titles and quotations can contain words such as "Last" or "Biggest"
    # without requesting an extrema operation.
    text = re.sub(r'["“][^"”]*["”]', " ", str(question))
    maximum = bool(
        re.search(
            r"\b(?:latest|last|most recent|biggest|largest|highest|maximum)\b",
            text,
            re.I,
        )
        and not re.search(r"\blast name\b", text, re.I)
    )
    minimum = bool(
        re.search(
            r"\b(?:earliest|least|smallest|lowest|minimum|first)\b",
            text,
            re.I,
        )
        and not re.search(r"\bfirst name\b", text, re.I)
    )
    return maximum != minimum


_COMPARISON_CUE_RE = re.compile(
    r"\b(no more than|no less than|at most|at least|prior to|less than|"
    r"fewer than|lower than|smaller than|under|below|before|more than|"
    r"greater than|higher than|larger than|over|above|after|later than)\b",
    re.I,
)
_MONTH_NUMBERS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
    "nov": 11, "dec": 12,
}
_GENERIC_SCALAR_TOKENS = {
    "amount", "date", "dated", "float", "from", "integer", "number",
    "rate", "start", "end", "to", "unit", "value", "year",
}


def _answer_focus_text(question: str) -> str:
    """Extract a short answer-target phrase for relation-only similarity.

    The returned text contains surface words only.  In particular, graph
    variable names never enter the embedding query.
    """
    tokens = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", str(question))
    if not tokens:
        return str(question)
    wh_words = {"what", "which", "who", "where", "when", "whom", "whose"}
    starts = [
        index
        for index, token in enumerate(tokens)
        if token.casefold().removesuffix("'s") in wh_words
    ]
    early = [index for index in starts if index <= 1]
    start = early[0] if early else (starts[-1] if starts else 0)
    head = tokens[start].casefold().removesuffix("'s")
    remaining = tokens[start:]
    auxiliaries = {
        "do", "does", "did", "has", "have", "had", "can", "could",
        "would", "will", "shall", "should",
    }
    if head in {"what", "which", "whom", "whose"}:
        if len(remaining) > 1 and remaining[1].casefold() in {
            "is", "are", "was", "were",
        }:
            return " ".join([remaining[0], *remaining[2:10]])
        for index, token in enumerate(remaining[1:], start=1):
            if token.casefold() in auxiliaries:
                return " ".join(remaining[:index])
        return " ".join(remaining[:7])
    if head == "who":
        return " ".join(remaining[:6])
    return " ".join(remaining[:8])


def _terminal_relation_document(
    graph: QueryGraphCandidate,
    ontology: Any,
) -> str:
    """Describe the answer edge using schema text and no graph placeholders."""

    parts: list[str] = []
    answer_var = str(graph.answer_var)
    for subject, relation_id, object_ in graph.triples:
        subject = str(subject)
        relation_id = str(relation_id)
        object_ = str(object_)
        if answer_var not in {subject, object_}:
            continue
        direction = "answer as subject" if subject == answer_var else "answer as object"
        values = [
            direction,
            " ".join(relation_label_from_id(relation_id)),
        ]
        if ontology is not None:
            domain = str(ontology.domain_for_relation(relation_id))
            range_id = str(ontology.range_for_relation(relation_id))
            if domain:
                values.append("domain " + " ".join(relation_label_from_id(domain)))
            if range_id:
                values.append("range " + " ".join(relation_label_from_id(range_id)))
        parts.append(" ; ".join(values))
    for operator in graph.operators:
        if not isinstance(operator, dict):
            continue
        relation_id = str(operator.get("attribute_relation_id", ""))
        if relation_id:
            parts.append(
                "constraint "
                + str(operator.get("type", ""))
                + " "
                + " ".join(relation_label_from_id(relation_id))
            )
    return " | ".join(parts)


def _terminal_relation_lexical_score(question: str, graph: QueryGraphCandidate) -> float:
    """Token-F1 between the requested target and real terminal predicates."""

    ignored = {
        "a", "an", "are", "did", "do", "does", "has", "have", "is",
        "of", "the", "was", "were", "what", "which", "who", "where",
    }
    query_tokens = {
        _semantic_word_key(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", _answer_focus_text(question))
        if token.casefold() not in ignored
    }
    relation_tokens: set[str] = set()
    answer_var = str(graph.answer_var)
    for subject, relation_id, object_ in graph.triples:
        if answer_var not in {str(subject), str(object_)}:
            continue
        relation_tokens.update(
            _semantic_word_key(token)
            for token in re.findall(
                r"[A-Za-z0-9]+",
                str(relation_id).replace(".", " ").replace("_", " "),
            )
        )
    if not query_tokens or not relation_tokens:
        return 0.0
    overlap = len(query_tokens & relation_tokens)
    precision = overlap / len(relation_tokens)
    recall = overlap / len(query_tokens)
    return (
        0.0
        if precision + recall == 0.0
        else (2.0 * precision * recall) / (precision + recall)
    )


def _normalize_constraint_literal(text: str) -> tuple[str, str]:
    """Extract one normalized date/number literal from constraint text."""
    month_names = "|".join(_MONTH_NUMBERS)
    match = re.search(
        rf"\b({month_names})[\s,]+(\d{{1,2}})(?:st|nd|rd|th)?"
        rf"[\s,]+(\d{{4}})\b",
        text,
        re.I,
    )
    if match:
        return (
            f"{int(match.group(3)):04d}-"
            f"{_MONTH_NUMBERS[match.group(1).casefold()]:02d}-"
            f"{int(match.group(2)):02d}",
            "dateTime",
        )
    match = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})"
        rf"[\s,]+(\d{{4}})\b",
        text,
        re.I,
    )
    if match:
        return (
            f"{int(match.group(3)):04d}-"
            f"{_MONTH_NUMBERS[match.group(2).casefold()]:02d}-"
            f"{int(match.group(1)):02d}",
            "dateTime",
        )
    match = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", text)
    if match:
        return (
            f"{int(match.group(3)):04d}-{int(match.group(1)):02d}-"
            f"{int(match.group(2)):02d}",
            "dateTime",
        )
    match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        return (
            f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-"
            f"{int(match.group(3)):02d}",
            "dateTime",
        )
    values = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not values:
        return "", ""
    value = values[-1].replace(",", "")
    temporal = bool(
        re.fullmatch(r"\d{4}", value)
        and re.search(
            r"\b(?:date|year|released?|opened?|founded?|ended?|began|"
            r"started?|died|death|position|relationship|production)\b",
            text,
            re.I,
        )
    )
    return value, "dateTime" if temporal else "number"


def _literal_constraint_focus(text: str) -> str:
    """Keep the local noun phrase around the last explicit numeric literal."""
    tokens = list(re.finditer(r"[A-Za-z0-9_]+(?:[.,/-][A-Za-z0-9_]+)*", text))
    numeric_indexes = [
        index
        for index, match in enumerate(tokens)
        if re.search(r"\d", match.group(0))
    ]
    if not numeric_indexes:
        return text
    index = numeric_indexes[-1]
    return " ".join(
        match.group(0)
        for match in tokens[max(0, index - 6): index + 3]
    )


def _missing_constraint_spec(question: str) -> dict[str, str] | None:
    """Infer only explicit scalar/range/extrema intent from surface syntax."""
    text = str(question)
    comparison_matches = list(_COMPARISON_CUE_RE.finditer(text))
    if comparison_matches:
        match = comparison_matches[-1]
        cue = match.group(1).casefold()
        value, value_type = _normalize_constraint_literal(text[match.end():])
        if not value:
            return None
        lower = any(
            token in cue
            for token in (
                "less", "fewer", "lower", "under", "below", "before",
                "prior", "smaller", "at most", "no more",
            )
        )
        return {
            "type": "LESS_THAN" if lower else "GREATER_THAN",
            "value": value,
            "value_type": value_type,
            "focus": text[max(0, match.start() - 80): match.end() + 80],
        }

    value, value_type = _normalize_constraint_literal(text)
    if value:
        # A literal plus a semantically matching schema property is a bounded
        # equality repair.  Numeric/date normalization is the only value
        # normalization used here.
        return {
            "type": "EQUAL",
            "value": value,
            "value_type": value_type,
            "focus": _literal_constraint_focus(text),
        }

    extrema_text = re.sub(r'["“][^"”]*["”]', " ", text)
    maximum = bool(
        re.search(
            r"\b(?:latest|last|most recent|biggest|largest|highest|maximum)\b",
            extrema_text,
            re.I,
        )
        and not re.search(r"\blast name\b", extrema_text, re.I)
    )
    minimum = bool(
        re.search(
            r"\b(?:earliest|least|smallest|lowest|minimum|first)\b",
            extrema_text,
            re.I,
        )
        and not re.search(r"\bfirst name\b", extrema_text, re.I)
    )
    if maximum == minimum:
        return None
    if re.search(
        r"\b(?:number|amount|population)\s+of\s+"
        r"(?:smallest|largest|highest|lowest)\b",
        extrema_text,
        re.I,
    ):
        return None
    if re.search(
        r"\b(?:his|her|their|its)\s+(?:earliest|latest)\s+released\b",
        extrema_text,
        re.I,
    ):
        return None
    return {
        "type": "ARGMAX" if maximum else "ARGMIN",
        "value": "",
        "value_type": "",
        "focus": text,
    }


def _semantic_word_key(value: str) -> str:
    token = str(value).casefold()
    if token in {"died", "dead"}:
        return "death"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("sed") and len(token) > 4:
        return token[:-1]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _constraint_relation_has_surface_evidence(
    focus: str,
    first_relation: str,
    second_relation: str,
) -> bool:
    """Require a real property-name match before executing a late repair."""
    raw_question_tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", focus)
    }
    question_tokens = {
        _semantic_word_key(token)
        for token in raw_question_tokens
    }
    raw_property_tokens: set[str] = set()
    property_tokens: set[str] = set()
    for relation in (first_relation, second_relation):
        if not relation:
            continue
        raw_tokens = {
            token.casefold()
            for token in re.findall(
                r"[A-Za-z0-9]+", relation.rsplit(".", 1)[-1].replace("_", " ")
            )
        }
        raw_property_tokens.update(raw_tokens)
        property_tokens.update(
            _semantic_word_key(token) for token in raw_tokens
        )
    raw_property_tokens.difference_update(_GENERIC_SCALAR_TOKENS)
    property_tokens.difference_update(_GENERIC_SCALAR_TOKENS)
    if raw_question_tokens & raw_property_tokens:
        return True
    # Stemming is useful for released/release and founded/founding, but a
    # very short stem such as film (from filming) is too ambiguous to prove
    # that a scalar ordering property was requested.
    return any(
        token in question_tokens and len(token) >= 5
        for token in property_tokens
    )


def _constraint_operator_count(graph: QueryGraphCandidate) -> int:
    return sum(
        str(operator.get("type", "")).upper() != "NO_EQUAL"
        for operator in graph.operators
        if isinstance(operator, dict)
    )


def _primary_graph_is_more_constrained(
    primary: QueryGraphCandidate,
    secondary: QueryGraphCandidate,
    question: str = "",
) -> bool:
    """Whether answer union would dilute a structurally stronger primary."""
    structurally_stronger = bool(
        _constraint_operator_count(primary) > _constraint_operator_count(secondary)
        or len(primary.triples) > len(secondary.triples)
    )
    if not structurally_stronger or not question:
        return structurally_stronger
    unquoted = re.sub(r'["“][^"”]*["”]', " ", str(question))
    return bool(
        _missing_constraint_spec(question) is not None
        or re.search(r"\b(?:and|also)\b", unquoted, re.I)
    )


def _property_subsets_conflict(
    preferred_ids: list[str],
    competing_ids: list[str],
) -> bool:
    """Whether two non-empty property-filter results genuinely disagree.

    A competing result that is a strict subset of the preferred result is
    nested corroboration: every one of its answers already supports the
    preferred interpretation.  Disjoint, partial-overlap, and competing
    supersets remain conflicts, so this cannot widen the chosen answer set.
    """

    preferred = set(preferred_ids)
    competing = set(competing_ids)
    return bool(
        preferred
        and competing
        and competing != preferred
        and not competing < preferred
    )


def _constraint_property_rank(
    graph: QueryGraphCandidate,
    spec: dict[str, str],
    ontology: Any,
) -> tuple[float, float, float, float, str, str]:
    """Rank one schema-backed constraint property without answer evidence.

    Late constraint repair used to union answers from every superficially
    plausible property.  That mixes mutually exclusive interpretations (for
    example a dated measurement's ``year`` and ``number`` fields).  This rank
    uses only the explicit constraint phrase, real predicate names, schema
    ranges and the pre-existing graph score.  Variables and answer values are
    deliberately absent.
    """

    repair = graph.provenance.get("operator_attribute_repair", {})
    if not isinstance(repair, dict):
        repair = {}
    first_relation = str(repair.get("first_relation", ""))
    second_relation = str(repair.get("second_relation", ""))
    focus_tokens = {
        _semantic_word_key(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(spec.get("focus", "")))
    }

    def relation_tokens(relation: str) -> set[str]:
        return {
            _semantic_word_key(token)
            for token in re.findall(
                r"[A-Za-z0-9]+",
                relation.rsplit(".", 1)[-1].replace("_", " "),
            )
        }

    first_tokens = relation_tokens(first_relation)
    second_tokens = relation_tokens(second_relation)
    document_frequency = getattr(
        ontology,
        "relation_token_document_frequency",
        {},
    )
    document_count = max(
        1,
        len(getattr(ontology, "relation_ids", ())),
    )

    def overlap_weight(tokens: set[str]) -> float:
        return sum(
            math.log(
                (document_count + 1)
                / (int(document_frequency.get(token, 0)) + 1)
            )
            + 1.0
            for token in focus_tokens & tokens
        )

    first_overlap = overlap_weight(first_tokens)
    second_overlap = overlap_weight(second_tokens)

    focus = str(spec.get("focus", "")).casefold()
    value_type = str(spec.get("value_type", "")).casefold()
    temporal_cues = {
        "date", "year", "release", "open", "found", "start", "end",
        "begin", "birth", "death", "die", "from", "to", "time",
    }
    numeric_cues = {
        "number", "amount", "count", "rate", "population", "total",
        "height", "length", "area", "undergraduate", "postgraduate",
        "casualty", "soldier", "army", "code", "id",
    }
    normalized_focus = {
        _semantic_word_key(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", focus)
    }
    if value_type == "datetime":
        requested_family = "temporal"
    elif value_type == "number":
        requested_family = "numeric"
    elif normalized_focus & temporal_cues:
        requested_family = "temporal"
    elif normalized_focus & numeric_cues:
        requested_family = "numeric"
    else:
        requested_family = ""

    terminal_relation = second_relation or first_relation
    terminal_range = (
        str(ontology.range_for_relation(terminal_relation))
        if ontology is not None and terminal_relation
        else ""
    )
    terminal_tokens = second_tokens or first_tokens
    temporal_property = terminal_range == "type.datetime" or bool(
        terminal_tokens & {"date", "year", "from", "to", "start", "end", "time"}
    )
    numeric_property = terminal_range in {"type.float", "type.int"} or bool(
        terminal_tokens & {"number", "amount", "count", "rate", "population", "total"}
    )
    family_score = 0.0
    if requested_family == "temporal":
        if terminal_tokens & {"date", "year", "from", "to", "start", "end", "time"}:
            family_score = 1.5
        else:
            family_score = 0.75 if temporal_property else (-0.5 if numeric_property else 0.0)
    elif requested_family == "numeric":
        if terminal_tokens & {"number", "amount", "count", "rate", "population", "total"}:
            family_score = 1.5
        elif terminal_tokens & {"date", "year", "from", "to", "start", "end", "time"}:
            family_score = -0.5
        else:
            family_score = 0.75 if numeric_property else (-0.5 if temporal_property else 0.0)

    # A bridge whose first predicate is unrelated to the constraint can make
    # a good terminal leaf look relevant (film.prequel -> release_date).  A
    # direct scalar property, or a CVT property whose first predicate itself
    # names the requested measurement, is safer and cheaper.
    bridge_score = 0.0
    if second_relation:
        bridge_score = 0.25 if first_overlap else -0.25
    elif terminal_range in {"type.datetime", "type.float", "type.int"}:
        bridge_score = 0.35

    lexical_score = float(first_overlap) + (0.35 * float(second_overlap))
    return (
        lexical_score,
        family_score,
        bridge_score,
        float(graph.score),
        first_relation,
        second_relation,
    )


def _answer_sets_support_union(
    left: ExecutedGraph,
    right: ExecutedGraph,
) -> bool:
    """Require overlap without treating containment as union evidence.

    If one alternative is a strict subset of the other, unioning can only
    replace the narrower answer with the broader one.  That is evidence for a
    precision disagreement, not for two complementary query paths.
    """

    left_ids = set(left.answer_ids)
    right_ids = set(right.answer_ids)
    return bool(
        left_ids
        and right_ids
        and left_ids != right_ids
        and left_ids & right_ids
        and not left_ids < right_ids
        and not right_ids < left_ids
    )


def _terminal_sibling_owner_schema_compatible(
    graph: QueryGraphCandidate,
    changed_triple: tuple[str, str, str],
    ontology: Any,
) -> bool:
    """Require the changed answer edge to fit the existing owner spine.

    A same-shape answer superset can still be unsafe when an endpoint happens
    to materialize two unrelated schema roles (for example a fictional setting
    followed by a country property).  Accept exact/supertype continuity, or a
    meaningful lexical type overlap such as ``beer_country_region`` with
    ``country``.  Universal schema roots are excluded and no relation/domain
    identity is hard-coded.
    """

    if ontology is None:
        return True
    subject, relation_id, object_ = changed_triple
    answer_var = str(graph.answer_var)
    if answer_var == object_:
        owner, target_type = subject, str(ontology.domain_for_relation(relation_id))
    elif answer_var == subject:
        owner, target_type = object_, str(ontology.range_for_relation(relation_id))
    else:
        return False
    if not target_type:
        return False

    source_types: set[str] = set()
    for triple in graph.triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        edge = tuple(map(str, triple))
        if edge == changed_triple:
            continue
        left, predicate, right = edge
        if left == owner:
            value = str(ontology.domain_for_relation(predicate))
            if value:
                source_types.add(value)
        if right == owner:
            value = str(ontology.range_for_relation(predicate))
            if value:
                source_types.add(value)
    if not source_types:
        return False

    universal = {
        "common.topic",
        "type.object",
        "base.type_ontology.abstract",
        "base.type_ontology.inanimate",
        "base.type_ontology.agent",
    }

    def closure(type_id: str) -> set[str]:
        values = {str(type_id)}
        try:
            values.update(map(str, ontology.supertypes(type_id)))
        except Exception:
            pass
        return values - universal

    target_closure = closure(target_type)
    ignored_tokens = {
        "base", "common", "object", "ontology", "topic", "type",
    }

    def tokens(type_id: str) -> set[str]:
        return {
            _semantic_word_key(token)
            for token in re.findall(r"[A-Za-z0-9]+", str(type_id).replace("_", " "))
            if _semantic_word_key(token) not in ignored_tokens
        }

    target_tokens = tokens(target_type)
    return any(
        bool(closure(source_type) & target_closure)
        or bool(tokens(source_type) & target_tokens)
        for source_type in source_types
    )


def _answer_union_has_semantic_support(
    question: str,
    primary: ExecutedGraph,
    secondary: ExecutedGraph,
    ontology: Any = None,
) -> bool:
    """Allow disjoint union unless the primary answer edge clearly dominates."""

    primary_ids = set(primary.answer_ids)
    secondary_ids = set(secondary.answer_ids)
    if primary_ids == secondary_ids:
        return False
    if primary_ids < secondary_ids:
        # Containment is normally a precision disagreement, but there is one
        # structurally auditable exception: both graphs have the same query
        # skeleton and full operator semantics, differ at exactly one answer
        # edge, and the broader/higher-scored edge is a strictly better lexical
        # match for the requested target.  This recovers under-answering
        # property siblings without naming a domain, relation, or entity.
        primary_triples = Counter(
            tuple(map(str, triple))
            for triple in primary.graph.triples
            if isinstance(triple, (list, tuple)) and len(triple) == 3
        )
        secondary_triples = Counter(
            tuple(map(str, triple))
            for triple in secondary.graph.triples
            if isinstance(triple, (list, tuple)) and len(triple) == 3
        )
        primary_only = list((primary_triples - secondary_triples).elements())
        secondary_only = list((secondary_triples - primary_triples).elements())
        same_terminal_slot = bool(
            len(primary.graph.triples) == len(secondary.graph.triples)
            and len(primary_only) == 1
            and len(secondary_only) == 1
            and (primary_only[0][0], primary_only[0][2])
            == (secondary_only[0][0], secondary_only[0][2])
            and str(primary.graph.answer_var)
            in {primary_only[0][0], primary_only[0][2]}
            and str(secondary.graph.answer_var)
            in {secondary_only[0][0], secondary_only[0][2]}
        )
        primary_lexical_score = _terminal_relation_lexical_score(
            question, primary.graph
        )
        secondary_lexical_score = _terminal_relation_lexical_score(
            question, secondary.graph
        )
        return bool(
            same_terminal_slot
            and float(secondary.graph.score) > float(primary.graph.score)
            and _full_operator_signature(primary.graph)
            == _full_operator_signature(secondary.graph)
            and secondary_lexical_score > 0.0
            and secondary_lexical_score + 1e-12 >= primary_lexical_score
            and _terminal_sibling_owner_schema_compatible(
                secondary.graph,
                secondary_only[0],
                ontology,
            )
        )
    if secondary_ids < primary_ids:
        # The selected graph already contains the secondary answer set, so a
        # union is a no-op and must not suppress later post-selection guards.
        return False
    if _answer_sets_support_union(primary, secondary):
        return True
    primary_score = _terminal_relation_lexical_score(question, primary.graph)
    secondary_score = _terminal_relation_lexical_score(question, secondary.graph)
    # A small but non-zero margin is sufficient because the signal is used
    # only to preserve an already selected graph, never to choose an unseen
    # relation.  Full trace audit triggered five improvements and zero
    # perfect/partial regressions at this threshold.
    return primary_score - secondary_score < 0.03


def _reverse_terminal_role_conflict(
    question: str,
    left: QueryGraphCandidate,
    right: QueryGraphCandidate,
    ontology: Any,
) -> bool:
    """Detect alternative graphs that differ only by inverse answer roles."""

    reverse_for_relation = getattr(ontology, "reverse_for_relation", None)
    if not callable(reverse_for_relation):
        return False
    left_relations = Counter(
        str(triple[1])
        for triple in left.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    )
    right_relations = Counter(
        str(triple[1])
        for triple in right.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    )
    left_only = list((left_relations - right_relations).elements())
    right_only = list((right_relations - left_relations).elements())
    if len(left_only) != 1 or len(right_only) != 1:
        return False
    left_relation, right_relation = left_only[0], right_only[0]
    reverse_ids = set(map(str, reverse_for_relation(left_relation)))
    if right_relation not in reverse_ids:
        reverse_ids = set(map(str, reverse_for_relation(right_relation)))
        if left_relation not in reverse_ids:
            return False

    normalized_question = " ".join(str(question).casefold().split())
    left_leaf = left_relation.rsplit(".", 1)[-1].replace("_", " ").casefold()
    right_leaf = right_relation.rsplit(".", 1)[-1].replace("_", " ").casefold()
    explicitly_requests_both = bool(
        left_leaf in normalized_question
        and right_leaf in normalized_question
        and re.search(r"\b(?:and|both|or)\b", normalized_question)
    )
    return not explicitly_requests_both


def _duplicate_triple_ratio(graph: QueryGraphCandidate) -> float:
    triples = [
        tuple(map(str, triple))
        for triple in graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    ]
    return 0.0 if not triples else 1.0 - (len(set(triples)) / len(triples))


def _duplicate_edge_challenger(
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
) -> ExecutedGraph | None:
    """Replace only a graph containing a byte-identical redundant edge."""

    selected_ratio = _duplicate_triple_ratio(selected.graph)
    if selected_ratio <= 0.0:
        return None
    challengers = [
        executed
        for executed in executions
        if _duplicate_triple_ratio(executed.graph) < selected_ratio - 1e-12
    ]
    if not challengers:
        return None
    return max(
        challengers,
        key=lambda executed: (
            float(executed.graph.score),
            -_duplicate_triple_ratio(executed.graph),
            str(executed.graph.graph_id),
        ),
    )


def _relation_diversity_ratio(graph: QueryGraphCandidate) -> float:
    """Return the fraction of distinct predicates in a query graph."""

    relations = [
        str(triple[1])
        for triple in graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    ]
    return 0.0 if not relations else len(set(relations)) / len(relations)


def _repeated_relation_model_guard(
    question: str,
    model_selected: ExecutedGraph,
    rule_selected: ExecutedGraph,
) -> bool:
    """Keep a model-selected diverse graph over a two-edge predicate echo.

    The local score guard is useful for noisy selector outputs, but a common
    failure mode gives a high score to ``anchor -r-> owner -r-> answer`` even
    when the selector chose a graph with distinct predicates.  This guard is
    deliberately structural: the rule winner must contain exactly two copies
    of one predicate, the model graph must have only distinct predicates, and
    the model answer set cannot be broader.  It does not inspect relation or
    entity identities and it does not make another model or endpoint call.
    """

    rule_relations = [
        str(triple[1])
        for triple in rule_selected.graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    ]
    model_relations = [
        str(triple[1])
        for triple in model_selected.graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    ]
    if not (
        len(rule_relations) == 2
        and len(set(rule_relations)) == 1
        and len(model_relations) >= 2
        and len(set(model_relations)) == len(model_relations)
        and model_selected.answer_ids
        and len(model_selected.answer_ids) <= len(rule_selected.answer_ids)
    ):
        return False
    expected_kind = _expected_answer_kind(question)
    if (
        expected_kind
        and _answer_value_kind(model_selected.answer_ids) != expected_kind
    ):
        return False
    substantive_operators = [
        operator
        for operator in model_selected.graph.operators
        if isinstance(operator, dict)
        and str(operator.get("type", "")).upper() not in {"", "NO_EQUAL"}
    ]
    return bool(
        not substantive_operators
        or (
            _explicit_extrema_request(question)
            and all(
                str(operator.get("type", "")).upper()
                in {"ARGMIN", "ARGMAX"}
                for operator in substantive_operators
            )
        )
        or _graph_operator_has_question_evidence(
            model_selected.graph,
            question,
        )
    )


def _constraint_operator_signature(graph: QueryGraphCandidate) -> tuple[tuple[str, ...], ...]:
    """Return constraint semantics that must agree before answer union."""
    values: list[tuple[str, ...]] = []
    for operator in graph.operators:
        if not isinstance(operator, dict):
            continue
        operator_type = str(operator.get("type", "")).upper()
        if not operator_type or operator_type == "NO_EQUAL":
            continue
        relation = str(operator.get("attribute_relation_id", "")).strip()
        if not relation:
            relation = " ".join(
                str(value).casefold()
                for value in operator.get("attribute_relation_label", [])
            )
        values.append(
            (
                operator_type,
                relation,
                str(operator.get("value", "")).strip().casefold(),
                str(operator.get("value_type", "")).strip().casefold(),
            )
        )
    return tuple(sorted(values))


def _constraint_operator_semantics_compatible(
    left: QueryGraphCandidate,
    right: QueryGraphCandidate,
) -> bool:
    return _constraint_operator_signature(left) == _constraint_operator_signature(right)


def _graph_matches_explicit_constraint(
    graph: QueryGraphCandidate,
    spec: dict[str, str] | None,
) -> bool:
    if spec is None:
        return False
    expected = str(spec.get("type", "")).upper()
    equivalent = {
        "LESS_THAN": {"LESS_THAN", "LESS_OR_EQUAL"},
        "GREATER_THAN": {"GREATER_THAN", "GREATER_OR_EQUAL"},
    }.get(expected, {expected})
    return any(
        str(operator.get("type", "")).upper() in equivalent
        for operator in graph.operators
        if isinstance(operator, dict)
        and str(operator.get("type", "")).upper() != "NO_EQUAL"
    )


def _graph_operator_has_question_evidence(
    graph: QueryGraphCandidate,
    question: str,
) -> bool:
    """Whether a grounded operator property is supported by question text."""
    temporal_cue = bool(
        re.search(
            r"\b(?:before|after|earliest|latest|first|last|date|year|when|"
            r"current|currently|begin|began|start|started|end|ended|from|"
            r"through|until|since)\b",
            str(question),
            re.I,
        )
    )
    for operator in graph.operators:
        if not isinstance(operator, dict):
            continue
        operator_type = str(operator.get("type", "")).upper()
        if not operator_type or operator_type == "NO_EQUAL":
            continue
        relation_id = str(operator.get("attribute_relation_id", "")).strip()
        if not relation_id:
            continue
        if _constraint_relation_has_surface_evidence(
            str(question), relation_id, ""
        ):
            return True
        tail = relation_id.rsplit(".", 1)[-1].casefold()
        if temporal_cue and tail in _TEMPORAL_RELATION_SUFFIXES:
            return True
    return False


def _union_executed_answers(
    primary: ExecutedGraph,
    secondary: ExecutedGraph,
) -> ExecutedGraph:
    answer_ids = list(primary.answer_ids)
    for answer_id in secondary.answer_ids:
        if answer_id not in answer_ids:
            answer_ids.append(answer_id)
    answers_by_id: dict[str, dict[str, str]] = {}
    for answer in [*primary.answers, *secondary.answers]:
        if isinstance(answer, dict) and str(answer.get("id", "")):
            answers_by_id.setdefault(str(answer["id"]), answer)
    graph = deepcopy(primary.graph)
    graph.provenance["selection_answer_union"] = {
        "primary_graph_id": primary.graph.graph_id,
        "secondary_graph_id": secondary.graph.graph_id,
    }
    return ExecutedGraph(
        graph,
        answer_ids,
        [
            answers_by_id.get(answer_id, {"id": answer_id, "label": answer_id})
            for answer_id in answer_ids
        ],
        len(answer_ids),
    )


def _answer_type_signatures_compatible(
    left: QueryGraphCandidate,
    right: QueryGraphCandidate,
    ontology: Any,
) -> bool:
    """Prevent low-margin answer unions across different answer types."""
    if ontology is None:
        return True

    def signature(graph: QueryGraphCandidate) -> set[str]:
        values = {
            str(ontology.range_for_relation(relation_id))
            for _, relation_id, object_ in graph.triples
            if str(object_) == graph.answer_var
        }
        values.update(
            str(ontology.domain_for_relation(relation_id))
            for subject, relation_id, _ in graph.triples
            if str(subject) == graph.answer_var
        )
        return {value for value in values if value}

    left_types = signature(left)
    right_types = signature(right)
    return not left_types or not right_types or bool(left_types & right_types)


def _terminal_answer_type_closures(
    graph: QueryGraphCandidate,
    ontology: Any,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return schema closures for terminal edges that produce the answer.

    Only triples whose object is ``answer_var`` are terminal answer-producing
    edges.  A subject-side incident edge describes a property *of* an already
    bound answer and is deliberately insufficient for this hard gate.  This
    conservative direction requirement prevents broad domain types from
    overriding a valid selected graph.
    """
    if ontology is None:
        return ()
    range_for_relation = getattr(ontology, "range_for_relation", None)
    supertypes = getattr(ontology, "supertypes", None)
    if not callable(range_for_relation) or not callable(supertypes):
        return ()
    answer_var = str(graph.answer_var)
    type_ids: list[str] = []
    for triple in graph.triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        _, relation_id, object_ = triple
        if str(object_) != answer_var:
            continue
        type_id = str(range_for_relation(str(relation_id))).strip()
        if type_id and type_id not in type_ids:
            type_ids.append(type_id)
    closures: list[tuple[str, tuple[str, ...]]] = []
    for type_id in type_ids:
        inherited = tuple(
            dict.fromkeys(
                [type_id, *(str(item) for item in supertypes(type_id) if str(item))]
            )
        )
        closures.append((type_id, inherited))
    return tuple(closures)


def _where_location_type_regression(
    question: str,
    source: QueryGraphCandidate,
    challenger: QueryGraphCandidate,
    ontology: Any,
) -> bool:
    """Reject a provable location-to-nonlocation answer-role change.

    ``Where`` is the one wh form whose answer ontology is unambiguous enough
    for a hard schema check.  The guard is deliberately asymmetric: it acts
    only when the incumbent has a known ``location.location`` terminal type
    and the challenger also has a known terminal type which is not a
    location.  Unknown schema evidence always abstains.
    """

    if ontology is None or not re.match(r"^\s*where\b", str(question), re.I):
        return False

    def terminal_union(graph: QueryGraphCandidate) -> set[str]:
        return {
            inherited
            for type_id, closure in _terminal_answer_type_closures(graph, ontology)
            for inherited in (type_id, *closure)
        }

    source_types = terminal_union(source)
    challenger_types = terminal_union(challenger)
    return bool(
        source_types
        and challenger_types
        and "location.location" in source_types
        and "location.location" not in challenger_types
    )


def _hard_ontology_who_person_challenger(
    question: str,
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
    ontology: Any,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Replace a provably non-agent ``Who`` answer with a person candidate.

    The gate is intentionally asymmetric and schema-hard: every known
    terminal type of the incumbent must inherit ``non_agent`` and none may
    inherit ``agent``.  A challenger must have an answer-producing terminal
    type whose closure contains ``people.person``.  Among such already
    executed candidates only the existing rule score is used.
    """
    evidence: dict[str, Any] = {
        "status": "not_applied",
        "question_who": bool(re.match(r"^\s*who\b", str(question), re.I)),
        "source_graph_id": selected.graph.graph_id,
        "uses_existing_executions_only": True,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
    }
    if not evidence["question_who"] or ontology is None:
        evidence["reason"] = "question_not_who" if not evidence["question_who"] else "ontology_unavailable"
        return None, evidence

    selected_closures = _terminal_answer_type_closures(selected.graph, ontology)
    evidence["source_terminal_types"] = [type_id for type_id, _ in selected_closures]
    evidence["source_type_closures"] = {
        type_id: list(closure) for type_id, closure in selected_closures
    }
    non_agent = "base.type_ontology.non_agent"
    agent = "base.type_ontology.agent"
    if not selected_closures:
        evidence["reason"] = "source_terminal_type_unknown"
        return None, evidence
    if not all(
        non_agent in set(closure) and agent not in set(closure)
        for _, closure in selected_closures
    ):
        evidence["reason"] = "source_not_provably_non_agent"
        return None, evidence

    compatible: list[tuple[ExecutedGraph, tuple[tuple[str, tuple[str, ...]], ...]]] = []
    for execution in executions:
        closures = _terminal_answer_type_closures(execution.graph, ontology)
        if any("people.person" in set(closure) for _, closure in closures):
            compatible.append((execution, closures))
    if not compatible:
        evidence["reason"] = "no_executed_person_candidate"
        return None, evidence

    challenger, challenger_closures = max(
        compatible,
        key=lambda item: (
            float(item[0].graph.score),
            str(item[0].graph.graph_id),
        ),
    )
    evidence.update(
        {
            "status": "applied",
            "reason": "who_non_agent_to_person",
            "selected_graph_id": challenger.graph.graph_id,
            "selected_rule_score": float(challenger.graph.score),
            "selected_terminal_types": [
                type_id for type_id, _ in challenger_closures
            ],
            "selected_type_closures": {
                type_id: list(closure)
                for type_id, closure in challenger_closures
            },
            "eligible_graph_ids": [
                execution.graph.graph_id for execution, _ in compatible
            ],
        }
    )
    return challenger, evidence


_ONTOLOGY_HEAD_TYPE_CACHE: dict[
    int,
    tuple[Any, dict[str, frozenset[str]]],
] = {}
_ONTOLOGY_HEAD_TYPE_CACHE_LOCK = Lock()


def _singular_schema_word(value: Any) -> str:
    """Apply a small morphology-only normalization to one schema word."""
    token = str(value).strip().casefold()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _ontology_head_type_index(ontology: Any) -> dict[str, frozenset[str]]:
    """Index single-word type leaves from the loaded ontology itself."""
    if ontology is None:
        return {}
    cache_key = id(ontology)
    with _ONTOLOGY_HEAD_TYPE_CACHE_LOCK:
        cached = _ONTOLOGY_HEAD_TYPE_CACHE.get(cache_key)
        if cached is not None and cached[0] is ontology:
            return cached[1]

    type_ids: set[str] = set()
    for attribute in ("relation_domains", "relation_ranges"):
        mapping = getattr(ontology, attribute, {})
        if isinstance(mapping, dict):
            type_ids.update(str(item) for item in mapping.values() if str(item))
    parents = getattr(ontology, "type_parents", {})
    if isinstance(parents, dict):
        type_ids.update(str(item) for item in parents if str(item))
        type_ids.update(
            str(item)
            for values in parents.values()
            for item in values
            if str(item)
        )

    mutable: dict[str, set[str]] = {}
    for type_id in type_ids:
        if type_id.casefold().startswith(("base.", "freebase.")):
            continue
        leaf = type_id.rsplit(".", 1)[-1]
        # Multi-token schema leaves are excluded: the surface gate accepts
        # exactly the one word immediately following What/Which.
        if not re.fullmatch(r"[A-Za-z]+", leaf):
            continue
        mutable.setdefault(_singular_schema_word(leaf), set()).add(type_id)
    index = {
        key: frozenset(sorted(values)) for key, values in mutable.items()
    }
    with _ONTOLOGY_HEAD_TYPE_CACHE_LOCK:
        _ONTOLOGY_HEAD_TYPE_CACHE[cache_key] = (ontology, index)
    return index


def _hard_ontology_head_subset_challenger(
    question: str,
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
    ontology: Any,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Use an ontology-derived What/Which head to narrow an answer set.

    The vocabulary is generated exclusively from single-word ontology type
    leaves.  No concrete type, predicate, entity, or question is routed in
    code.  A replacement is allowed only when the incumbent has known terminal
    schema evidence, lacks the requested type, and an already executed typed
    candidate is a strict answer subset.
    """
    match = re.match(
        r"^\s*(?:what|which)\s+([A-Za-z][A-Za-z-]*)\b",
        str(question),
        re.I,
    )
    head = _singular_schema_word(match.group(1)) if match else ""
    expected_types = set(_ontology_head_type_index(ontology).get(head, ()))
    evidence: dict[str, Any] = {
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "head": head,
        "expected_types": sorted(expected_types),
        "vocabulary_source": "local_ontology_single_word_type_leaves",
        "uses_existing_executions_only": True,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
    }
    if not match or not expected_types:
        evidence["reason"] = (
            "question_head_not_strict_ontology_type"
            if match
            else "question_not_strict_what_which_head"
        )
        return None, evidence

    selected_closures = _terminal_answer_type_closures(selected.graph, ontology)
    selected_union = {
        inherited
        for _, closure in selected_closures
        for inherited in closure
    }
    evidence["source_terminal_types"] = [
        type_id for type_id, _ in selected_closures
    ]
    evidence["source_type_closure"] = sorted(selected_union)
    if not selected_union:
        evidence["reason"] = "source_terminal_type_unknown"
        return None, evidence
    if selected_union & expected_types:
        evidence["reason"] = "source_already_matches_expected_type"
        return None, evidence

    source_answers = set(map(str, selected.answer_ids))
    compatible: list[
        tuple[ExecutedGraph, tuple[tuple[str, tuple[str, ...]], ...]]
    ] = []
    for execution in executions:
        candidate_answers = set(map(str, execution.answer_ids))
        if not candidate_answers or not candidate_answers < source_answers:
            continue
        closures = _terminal_answer_type_closures(execution.graph, ontology)
        closure_union = {
            inherited
            for _, closure in closures
            for inherited in closure
        }
        if closure_union & expected_types:
            compatible.append((execution, closures))
    if not compatible:
        evidence["reason"] = "no_typed_strict_subset_candidate"
        return None, evidence

    challenger, challenger_closures = max(
        compatible,
        key=lambda item: (
            float(item[0].graph.score),
            str(item[0].graph.graph_id),
        ),
    )
    evidence.update(
        {
            "status": "applied",
            "reason": "ontology_head_type_strict_subset",
            "selected_graph_id": challenger.graph.graph_id,
            "selected_rule_score": float(challenger.graph.score),
            "selected_terminal_types": [
                type_id for type_id, _ in challenger_closures
            ],
            "source_answer_count": len(source_answers),
            "selected_answer_count": len(set(challenger.answer_ids)),
            "eligible_graph_ids": [
                execution.graph.graph_id for execution, _ in compatible
            ],
        }
    )
    return challenger, evidence


def _ontology_question_head_types(
    question: str,
    ontology: Any,
) -> tuple[str, set[str]]:
    index = _ontology_head_type_index(ontology)
    match = re.match(
        r"^\s*(?:what|which)\s+([A-Za-z][A-Za-z-]*)\b",
        str(question),
        re.I,
    )
    immediate = _singular_schema_word(match.group(1)) if match else ""
    immediate_generic = {
        "a", "an", "are", "form", "is", "kind", "major", "name", "of",
        "the", "type", "was", "were",
    }
    if immediate in index and immediate not in immediate_generic:
        return immediate, set(index[immediate])
    tokens = [
        _singular_schema_word(token)
        for token in re.findall(r"[A-Za-z][A-Za-z-]*", str(question))[:7]
    ]
    generic = {
        "a", "an", "are", "form", "is", "kind", "major", "name",
        "of", "the", "type", "was", "were",
    }
    boundary_words = {
        "and", "are", "did", "do", "does", "has", "have", "is", "of", "that",
        "was", "were", "where", "which", "who", "with",
    }
    phrase_tokens: list[str] = []
    for token in tokens[1:]:
        if phrase_tokens and token in boundary_words:
            break
        if not phrase_tokens and token in immediate_generic:
            continue
        phrase_tokens.append(token)
    candidates = [
        (token, set(index[token]))
        for token in phrase_tokens
        if token not in generic and token in index
    ]
    if len(candidates) == 1:
        return candidates[0]
    substantive_phrase = [
        token for token in phrase_tokens if token not in generic
    ]
    if substantive_phrase:
        return substantive_phrase[0], set()
    return (immediate, set()) if immediate and immediate not in immediate_generic else ("", set())


def _ontology_head_answer_role_regression(
    question: str,
    source: QueryGraphCandidate,
    challenger: QueryGraphCandidate,
    ontology: Any,
) -> bool:
    """Reject a challenger that leaves a known What/Which head type."""

    if ontology is None:
        return False
    head, expected_types = _ontology_question_head_types(question, ontology)
    if not head:
        return False

    def answer_relation_mentions_head(graph: QueryGraphCandidate) -> bool:
        for subject, relation_id, object_ in graph.triples:
            if str(graph.answer_var) not in {str(subject), str(object_)}:
                continue
            leaf_tokens = {
                _singular_schema_word(token)
                for token in re.findall(
                    r"[A-Za-z]+",
                    str(relation_id).rsplit(".", 1)[-1].replace("_", " "),
                )
            }
            if head in leaf_tokens:
                return True
        return False

    def terminal_union(graph: QueryGraphCandidate) -> set[str]:
        values = {
            inherited
            for _, closure in _terminal_answer_type_closures(graph, ontology)
            for inherited in closure
        }
        domain_for_relation = getattr(ontology, "domain_for_relation", None)
        supertypes = getattr(ontology, "supertypes", None)
        if callable(domain_for_relation) and callable(supertypes):
            for subject, relation_id, _ in graph.triples:
                if str(subject) != str(graph.answer_var):
                    continue
                type_id = str(domain_for_relation(str(relation_id))).strip()
                if type_id:
                    values.add(type_id)
                    values.update(map(str, supertypes(type_id)))
        return values

    if (
        answer_relation_mentions_head(source)
        and not answer_relation_mentions_head(challenger)
    ):
        return True
    if not expected_types:
        return False
    source_types = terminal_union(source)
    challenger_types = terminal_union(challenger)
    return bool(
        source_types
        and challenger_types
        and source_types & expected_types
        and not challenger_types & expected_types
    )


def _answer_component_detached_from_grounded_anchors(
    graph: QueryGraphCandidate,
) -> bool:
    """Return whether the answer component contains none of its used anchors."""

    adjacency: dict[str, set[str]] = {}
    for triple in graph.triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject, _, object_ = map(str, triple)
        adjacency.setdefault(subject, set()).add(object_)
        adjacency.setdefault(object_, set()).add(subject)
    bindings = graph.provenance.get("anchor_bindings", {})
    grounded_ids: set[str] = set()
    if isinstance(bindings, dict):
        for binding in bindings.values():
            if isinstance(binding, dict):
                entity_id = str(binding.get("id", "")).strip()
            else:
                entity_id = str(
                    getattr(binding, "entity_id", "")
                ).strip()
            if entity_id:
                grounded_ids.add(entity_id)
    used_anchors = grounded_ids & set(adjacency)
    if not used_anchors:
        return False
    answer = str(graph.answer_var)
    reachable = {answer}
    frontier = [answer]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency.get(node, ()):
            if neighbor in reachable:
                continue
            reachable.add(neighbor)
            frontier.append(neighbor)
    return not bool(used_anchors & reachable)


def _normalized_scalar_constraint_value(value: Any) -> str:
    """Normalize only date/number spellings allowed by the repair policy."""
    text = str(value).strip()
    if not text:
        return ""
    normalized, _ = _normalize_constraint_literal(text)
    return normalized or text.replace(",", "").casefold()


_SCALAR_ANSWER_NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:,\d{3})*)(?:\.\d+)?|(?:\.\d+))"
    r"(?:[eE][+-]?\d+)?$"
)
_SCALAR_ANSWER_TEMPORAL_RES = (
    # Full date followed by Freebase's serialized timezone suffix.
    re.compile(r"^(?P<value>[+-]?\d{4,}-\d{2}-\d{2})(?:Z|[+-]\d{2}:\d{2})$"),
    # gYearMonth followed by a timezone suffix, e.g. 1931-09-08:00.
    re.compile(r"^(?P<value>[+-]?\d{4,}-\d{2})(?:Z|[+-]\d{2}:\d{2})$"),
    # gYear followed by a timezone suffix, e.g. 2009-08:00.
    re.compile(r"^(?P<value>[+-]?\d{4,})(?:Z|[+-]\d{2}:\d{2})$"),
    re.compile(r"^(?P<value>[+-]?\d{4,}-\d{2}-\d{2})$"),
    re.compile(r"^(?P<value>[+-]?\d{4,}-\d{2})$"),
)
_SCALAR_ANSWER_TYPED_RE = re.compile(
    r'^(?P<quoted>"(?:[^"\\]|\\.)*")(?:@[A-Za-z0-9-]+|\^\^\S+)$'
)
_SCALAR_ENTITY_WH_HEADS = frozenset(
    {
        "age", "amount", "area", "capacity", "code", "coordinate",
        "coordinates", "cost", "date", "distance", "duration", "height",
        "id", "identifier", "latitude", "length", "longitude", "number",
        "percent", "percentage", "population", "price", "rank", "rate",
        "score", "temperature", "time", "weight", "width", "year",
    }
)
_SCALAR_ENTITY_WH_SKIP = frozenset(
    {
        "a", "an", "are", "be", "been", "being", "can", "could", "did",
        "do", "does", "had", "has", "have", "is", "should", "the", "was",
        "were", "will", "would",
    }
)


def _canonical_scalar_answer_literal(value: Any) -> str:
    """Canonicalize a complete numeric/date literal without token extraction.

    Full-string matching is important here: entity labels containing a year or
    number must never be treated as scalar answers.  Only the normalization
    forms explicitly permitted for numeric/temporal values are accepted.
    """
    text = str(value).strip()
    typed = _SCALAR_ANSWER_TYPED_RE.fullmatch(text)
    if typed:
        try:
            text = str(json.loads(typed.group("quoted"))).strip()
        except json.JSONDecodeError:
            return ""
    for pattern in _SCALAR_ANSWER_TEMPORAL_RES:
        match = pattern.fullmatch(text)
        if match:
            return match.group("value")
    if not _SCALAR_ANSWER_NUMBER_RE.fullmatch(text):
        return ""
    try:
        number = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return ""
    if not number.is_finite():
        return ""
    if number == 0:
        return "0"
    normalized = format(number.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def _scalar_attribute_gate_expects_entity(question: str) -> bool:
    """Recognize entity interrogatives while excluding numeric/temporal heads."""
    tokens = re.findall(r"[a-z]+", str(question).casefold())
    if not tokens:
        return False
    if tokens[0] in {"who", "where"}:
        return True
    if tokens[0] == "which":
        return len(tokens) == 1 or tokens[1] not in _SCALAR_ENTITY_WH_HEADS
    if tokens[0] != "what":
        return False
    index = 1
    while index < len(tokens) and tokens[index] in _SCALAR_ENTITY_WH_SKIP:
        index += 1
    return bool(
        index < len(tokens)
        and tokens[index] not in _SCALAR_ENTITY_WH_HEADS
    )


def _hard_scalar_attribute_projection_challenger(
    question: str,
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Recover an entity when a scalar equality attribute became the answer.

    The incumbent must answer an entity-form question entirely with scalar
    literals and expose the question's explicit equality value through its
    answer-producing relation.  A replacement is eligible only when an
    already executed entity candidate applies that *same relation and value*
    as an ``EQUAL`` constraint on its own answer variable.  This is therefore
    an answer-variable projection correction, not a new retrieval lane.
    """
    spec = _missing_constraint_spec(question)
    expected_entity = _scalar_attribute_gate_expects_entity(question)
    source_values = [
        _canonical_scalar_answer_literal(value)
        for value in selected.answer_ids
    ]
    expected_value = _canonical_scalar_answer_literal(
        (spec or {}).get("value", "")
    )
    evidence: dict[str, Any] = {
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "question_expects_entity": expected_entity,
        "spec": deepcopy(spec),
        "normalized_literal": expected_value,
        "uses_existing_executions_only": True,
        "normalization": "full_string_date_number_only",
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
    }
    if not expected_entity:
        evidence["reason"] = "question_does_not_expect_entity"
        return None, evidence
    if (
        not isinstance(spec, dict)
        or str(spec.get("type", "")).upper() != "EQUAL"
        or str(spec.get("value_type", "")).casefold()
        not in {"datetime", "number"}
        or not expected_value
    ):
        evidence["reason"] = "question_not_explicit_scalar_equality"
        return None, evidence
    if not source_values or any(not value for value in source_values):
        evidence["reason"] = "source_answers_not_all_scalar_literals"
        return None, evidence
    evidence["source_scalar_values"] = source_values
    if expected_value not in set(source_values):
        evidence["reason"] = "source_does_not_return_question_literal"
        return None, evidence

    answer_var = str(selected.graph.answer_var)
    source_relations = {
        str(relation_id)
        for subject, relation_id, object_ in selected.graph.triples
        if str(object_) == answer_var and str(subject) != answer_var
    }
    evidence["source_answer_relations"] = sorted(source_relations)
    if not source_relations:
        evidence["reason"] = "source_answer_not_relation_object"
        return None, evidence

    compatible: list[ExecutedGraph] = []
    matched_relations: dict[str, list[str]] = {}
    for execution in executions:
        if execution is selected or _answer_value_kind(execution.answer_ids) != "entity":
            continue
        candidate_relations: list[str] = []
        for operator in execution.graph.operators:
            if not isinstance(operator, dict):
                continue
            if str(operator.get("type", "")).upper() != "EQUAL":
                continue
            if str(operator.get("input_var", "")) != str(execution.graph.answer_var):
                continue
            relation_id = str(operator.get("attribute_relation_id", "")).strip()
            if not relation_id:
                label = operator.get("attribute_relation_label", [])
                relation_id = relation_id_from_label(label) if label else ""
            if relation_id not in source_relations:
                continue
            if (
                _canonical_scalar_answer_literal(operator.get("value", ""))
                != expected_value
            ):
                continue
            candidate_relations.append(relation_id)
        if candidate_relations:
            compatible.append(execution)
            matched_relations[execution.graph.graph_id] = sorted(
                set(candidate_relations)
            )
    if not compatible:
        evidence["reason"] = "no_matching_entity_equality_candidate"
        return None, evidence

    challenger = max(
        compatible,
        key=lambda item: (float(item.graph.score), str(item.graph.graph_id)),
    )
    evidence.update(
        {
            "status": "applied",
            "reason": "scalar_attribute_answer_reprojected_to_entity",
            "selected_graph_id": challenger.graph.graph_id,
            "selected_rule_score": float(challenger.graph.score),
            "selected_constraint_relations": matched_relations[
                challenger.graph.graph_id
            ],
            "eligible_graph_ids": [
                item.graph.graph_id for item in compatible
            ],
        }
    )
    return challenger, evidence


def _hard_explicit_equality_challenger(
    question: str,
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Correct a range comparison when the surface states scalar equality.

    This does not fill a missing operator.  The incumbent must already contain
    one or more substantive operators, all of which are range comparisons,
    and an existing executed candidate must contain ``EQUAL`` with the same
    normalized date/number literal.
    """
    spec = _missing_constraint_spec(question)
    evidence: dict[str, Any] = {
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "spec": deepcopy(spec),
        "uses_existing_executions_only": True,
        "normalization": "date_number_only",
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
    }
    if (
        not isinstance(spec, dict)
        or str(spec.get("type", "")).upper() != "EQUAL"
        or str(spec.get("value_type", "")).casefold()
        not in {"datetime", "number"}
    ):
        evidence["reason"] = "question_not_explicit_scalar_equality"
        return None, evidence

    selected_types = [
        str(operator.get("type", "")).upper()
        for operator in selected.graph.operators
        if isinstance(operator, dict)
        and str(operator.get("type", "")).upper() not in {"", "NO_EQUAL"}
    ]
    evidence["source_operator_types"] = selected_types
    if not selected_types:
        evidence["reason"] = "source_has_no_substantive_operator"
        return None, evidence
    if "EQUAL" in selected_types or any(
        operator_type not in _RANGE_OPERATOR_TYPES
        for operator_type in selected_types
    ):
        evidence["reason"] = "source_not_range_only"
        return None, evidence

    expected_value = _normalized_scalar_constraint_value(spec.get("value", ""))
    compatible: list[ExecutedGraph] = []
    for execution in executions:
        if any(
            str(operator.get("type", "")).upper() == "EQUAL"
            and _normalized_scalar_constraint_value(operator.get("value", ""))
            == expected_value
            for operator in execution.graph.operators
            if isinstance(operator, dict)
        ):
            compatible.append(execution)
    if not compatible:
        evidence["reason"] = "no_matching_executed_equality_candidate"
        return None, evidence
    challenger = max(
        compatible,
        key=lambda item: (float(item.graph.score), str(item.graph.graph_id)),
    )
    evidence.update(
        {
            "status": "applied",
            "reason": "explicit_scalar_equality_replaces_range",
            "normalized_literal": expected_value,
            "selected_graph_id": challenger.graph.graph_id,
            "selected_rule_score": float(challenger.graph.score),
            "eligible_graph_ids": [item.graph.graph_id for item in compatible],
        }
    )
    return challenger, evidence


def _operator_type_equivalents(operator_type: str) -> set[str]:
    normalized = str(operator_type).upper()
    return {
        "LESS_THAN": {"LESS_THAN", "LESS_OR_EQUAL"},
        "GREATER_THAN": {"GREATER_THAN", "GREATER_OR_EQUAL"},
    }.get(normalized, {normalized})


def _substantive_operator_multiset(
    graph: QueryGraphCandidate,
) -> Counter[tuple[str, str, str]]:
    """Return rename-invariant operator semantics for preservation checks."""
    signatures: Counter[tuple[str, str, str]] = Counter()
    for operator in graph.operators:
        if not isinstance(operator, dict):
            continue
        operator_type = str(operator.get("type", "")).upper()
        if operator_type in {"", "NO_EQUAL"}:
            continue
        relation_id = str(operator.get("attribute_relation_id", "")).strip()
        if not relation_id:
            label = operator.get("attribute_relation_label", [])
            relation_id = relation_id_from_label(label) if label else ""
        signatures[
            (
                operator_type,
                relation_id,
                _normalized_scalar_constraint_value(operator.get("value", "")),
            )
        ] += 1
    return signatures


def _hard_additive_operator_challenger(
    question: str,
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Add an explicit comparison/extrema without dropping existing intent."""
    spec = _missing_constraint_spec(question)
    spec_type = str((spec or {}).get("type", "")).upper()
    evidence: dict[str, Any] = {
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "spec": deepcopy(spec),
        "uses_existing_executions_only": True,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
    }
    if not isinstance(spec, dict) or spec_type in {"", "EQUAL"}:
        evidence["reason"] = "question_has_no_explicit_non_equality_spec"
        return None, evidence

    source_multiset = _substantive_operator_multiset(selected.graph)
    evidence["source_operator_multiset"] = [
        {"type": key[0], "relation_id": key[1], "value": key[2], "count": count}
        for key, count in sorted(source_multiset.items())
    ]
    if not source_multiset:
        evidence["reason"] = "source_has_no_substantive_operator"
        return None, evidence
    equivalents = _operator_type_equivalents(spec_type)
    # Presence is deliberately type-level, matching the existing explicit
    # constraint predicate.  This gate completes an absent intent; it is not a
    # literal-correction lane for an already represented comparison.
    if any(signature[0] in equivalents for signature in source_multiset):
        evidence["reason"] = "source_already_contains_spec_type"
        return None, evidence

    expected_value = _normalized_scalar_constraint_value(spec.get("value", ""))
    compatible: list[ExecutedGraph] = []
    for execution in executions:
        candidate_multiset = _substantive_operator_multiset(execution.graph)
        preserves_source = all(
            candidate_multiset[signature] >= count
            for signature, count in source_multiset.items()
        )
        adds_spec = any(
            signature[0] in equivalents
            and (not expected_value or signature[2] == expected_value)
            for signature in candidate_multiset
        )
        if preserves_source and adds_spec:
            compatible.append(execution)
    if not compatible:
        evidence["reason"] = "no_operator_preserving_executed_candidate"
        return None, evidence
    challenger = max(
        compatible,
        key=lambda item: (float(item.graph.score), str(item.graph.graph_id)),
    )
    evidence.update(
        {
            "status": "applied",
            "reason": "preserve_existing_and_add_explicit_operator",
            "normalized_literal": expected_value,
            "selected_graph_id": challenger.graph.graph_id,
            "selected_rule_score": float(challenger.graph.score),
            "eligible_graph_ids": [item.graph.graph_id for item in compatible],
        }
    )
    return challenger, evidence


def _operator_matches_explicit_spec(
    operator: dict[str, Any],
    spec: dict[str, str],
) -> bool:
    """Match one operator to surface-derived scalar/extrema semantics."""

    operator_type = str(operator.get("type", "")).upper()
    expected_type = str(spec.get("type", "")).upper()
    if operator_type not in _operator_type_equivalents(expected_type):
        return False
    expected_value = _normalized_scalar_constraint_value(spec.get("value", ""))
    if not expected_value:
        return True
    return (
        _normalized_scalar_constraint_value(operator.get("value", ""))
        == expected_value
    )


def _graph_matches_explicit_spec_exactly(
    graph: QueryGraphCandidate,
    spec: dict[str, str],
) -> bool:
    return any(
        _operator_matches_explicit_spec(operator, spec)
        for operator in graph.operators
        if isinstance(operator, dict)
    )


def _hard_explicit_constraint_subset_challenger(
    question: str,
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
    ontology: Any = None,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Restore a dropped explicit constraint using a unique strict subset.

    The challenger must keep the same answer variable, contain every source
    triple verbatim, implement the normalized question literal/operator, and
    materialize a strict subset of the incumbent answers.  Multiple graph
    variants are accepted only when they agree on the complete answer set.
    This makes the gate a constraint-completeness correction rather than a
    semantic re-ranking rule.
    """

    spec = _missing_constraint_spec(question)
    source_answers = frozenset(map(str, selected.answer_ids))
    evidence: dict[str, Any] = {
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "spec": deepcopy(spec),
        "uses_existing_executions_only": True,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
        "entity_or_relation_whitelist": False,
    }
    if not isinstance(spec, dict) or not source_answers:
        evidence["reason"] = "no_explicit_constraint_or_source_answers"
        return None, evidence
    if _graph_matches_explicit_spec_exactly(selected.graph, spec):
        evidence["reason"] = "source_already_contains_exact_constraint"
        return None, evidence

    source_triples = Counter(
        tuple(map(str, triple))
        for triple in selected.graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    )
    compatible: list[ExecutedGraph] = []
    for execution in executions:
        if execution is selected:
            continue
        candidate_answers = frozenset(map(str, execution.answer_ids))
        if not candidate_answers or not candidate_answers < source_answers:
            continue
        if str(execution.graph.answer_var) != str(selected.graph.answer_var):
            continue
        candidate_triples = Counter(
            tuple(map(str, triple))
            for triple in execution.graph.triples
            if isinstance(triple, (list, tuple)) and len(triple) == 3
        )
        if source_triples - candidate_triples:
            continue
        if not _graph_matches_explicit_spec_exactly(execution.graph, spec):
            continue
        if not _graph_operator_has_question_evidence(execution.graph, question):
            continue
        compatible.append(execution)

    # A bare temporal preposition (``held ... from 1973-12-02``) denotes an
    # equality-like boundary, while an upstream operator may conservatively
    # lower it as a range.  Do not accept one such guess.  When the selected
    # graph dropped the date entirely, however, two distinct ontology-backed
    # datetime properties that preserve the whole base graph and independently
    # produce the same strict subset are strong denotational evidence.  This
    # is a general temporal normalization lane and never inspects ids or gold.
    temporal_consensus = False
    if (
        not compatible
        and ontology is not None
        and str(spec.get("type", "")).upper() == "EQUAL"
        and str(spec.get("value_type", "")).casefold() == "datetime"
        and not _substantive_operator_multiset(selected.graph)
        and _EXPLICIT_CONJUNCTION_RE.search(str(question))
    ):
        expected_value = _normalized_scalar_constraint_value(
            spec.get("value", "")
        )
        temporal_candidates: list[tuple[ExecutedGraph, set[str]]] = []
        for execution in executions:
            if execution is selected:
                continue
            candidate_answers = frozenset(map(str, execution.answer_ids))
            if not candidate_answers or not candidate_answers < source_answers:
                continue
            if str(execution.graph.answer_var) != str(selected.graph.answer_var):
                continue
            candidate_triples = Counter(
                tuple(map(str, triple))
                for triple in execution.graph.triples
                if isinstance(triple, (list, tuple)) and len(triple) == 3
            )
            if source_triples - candidate_triples:
                continue
            relations = {
                str(operator.get("attribute_relation_id", "")).strip()
                for operator in execution.graph.operators
                if isinstance(operator, dict)
                and str(operator.get("type", "")).upper()
                in _RANGE_OPERATOR_TYPES
                and _normalized_scalar_constraint_value(
                    operator.get("value", "")
                ) == expected_value
                and str(operator.get("attribute_relation_id", "")).strip()
                and str(
                    ontology.range_for_relation(
                        str(operator.get("attribute_relation_id", "")).strip()
                    )
                ) == "type.datetime"
            }
            if relations and _graph_operator_has_question_evidence(
                execution.graph, question
            ):
                temporal_candidates.append((execution, relations))
        temporal_denotations = {
            frozenset(map(str, execution.answer_ids))
            for execution, _ in temporal_candidates
        }
        temporal_relations = {
            relation
            for _, relations in temporal_candidates
            for relation in relations
        }
        if (
            len(temporal_candidates) >= 2
            and len(temporal_denotations) == 1
            and len(temporal_relations) >= 2
        ):
            compatible = [
                execution for execution, _ in temporal_candidates
            ]
            temporal_consensus = True
    denotations = {
        frozenset(map(str, execution.answer_ids)) for execution in compatible
    }
    evidence.update(
        {
            "eligible_graph_ids": [
                execution.graph.graph_id for execution in compatible
            ],
            "eligible_denotation_count": len(denotations),
            "temporal_property_consensus": temporal_consensus,
        }
    )
    if not compatible:
        evidence["reason"] = "no_structure_preserving_strict_subset"
        return None, evidence
    if len(denotations) != 1:
        evidence["reason"] = "constraint_candidates_disagree"
        return None, evidence

    challenger = max(
        compatible,
        key=lambda execution: (
            float(execution.graph.score),
            str(execution.graph.graph_id),
        ),
    )
    evidence.update(
        {
            "status": "applied",
            "reason": "explicit_constraint_unique_strict_subset",
            "selected_graph_id": challenger.graph.graph_id,
            "source_answer_count": len(source_answers),
            "selected_answer_count": len(
                frozenset(map(str, challenger.answer_ids))
            ),
        }
    )
    return challenger, evidence


def _answer_var_owns_expected_constraint(
    graph: QueryGraphCandidate,
    spec: dict[str, str],
    expected_types: set[str],
    ontology: Any,
) -> bool:
    """Whether the answer is the typed owner of the requested attribute."""

    answer_var = str(graph.answer_var)
    for operator in graph.operators:
        if not isinstance(operator, dict) or not _operator_matches_explicit_spec(
            operator, spec
        ):
            continue
        input_var = str(operator.get("input_var", ""))
        relation_ids: list[str] = []
        explicit_relation = str(
            operator.get("attribute_relation_id", "")
        ).strip()
        if input_var == answer_var and explicit_relation:
            relation_ids.append(explicit_relation)
        for subject, relation_id, object_ in graph.triples:
            if str(subject) == answer_var and str(object_) == input_var:
                relation_ids.append(str(relation_id))
        for relation_id in dict.fromkeys(relation_ids):
            domain = str(ontology.domain_for_relation(relation_id)).strip()
            if not domain:
                continue
            closure = {
                domain,
                *(
                    str(item)
                    for item in ontology.supertypes(domain)
                    if str(item)
                ),
            }
            if closure & expected_types:
                return True
    return False


def _hard_constraint_owner_projection_challenger(
    question: str,
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
    ontology: Any,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Move a scalar constraint from a property value back to its owner.

    This gate covers heterogeneous projections such as answering a museum
    *type* for a question asking for the museum whose establishment date
    matches a comparison.  It relies only on the immediate What/Which head,
    ontology domain/range metadata, and already executed candidates.
    """

    spec = _missing_constraint_spec(question)
    match = re.match(
        r"^\s*(?:what|which)\s+([A-Za-z][A-Za-z-]*)\b",
        str(question),
        re.I,
    )
    head = _singular_schema_word(match.group(1)) if match else ""
    expected_types = set(_ontology_head_type_index(ontology).get(head, ()))
    evidence: dict[str, Any] = {
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "head": head,
        "expected_types": sorted(expected_types),
        "spec": deepcopy(spec),
        "uses_existing_executions_only": True,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
        "entity_or_relation_whitelist": False,
    }
    if ontology is None or not isinstance(spec, dict) or not expected_types:
        evidence["reason"] = "question_or_ontology_not_applicable"
        return None, evidence
    if not _graph_matches_explicit_spec_exactly(selected.graph, spec):
        evidence["reason"] = "source_lacks_exact_constraint"
        return None, evidence
    if _answer_var_owns_expected_constraint(
        selected.graph, spec, expected_types, ontology
    ):
        evidence["reason"] = "source_answer_already_owns_constraint"
        return None, evidence

    source_triples = {
        tuple(map(str, triple))
        for triple in selected.graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    }
    compatible: list[ExecutedGraph] = []
    for execution in executions:
        candidate_triples = {
            tuple(map(str, triple))
            for triple in execution.graph.triples
            if isinstance(triple, (list, tuple)) and len(triple) == 3
        }
        if not execution.answer_ids or not (source_triples & candidate_triples):
            continue
        if not _graph_matches_explicit_spec_exactly(execution.graph, spec):
            continue
        if not _answer_var_owns_expected_constraint(
            execution.graph, spec, expected_types, ontology
        ):
            continue
        compatible.append(execution)
    denotations = {
        frozenset(map(str, execution.answer_ids)) for execution in compatible
    }
    evidence.update(
        {
            "eligible_graph_ids": [
                execution.graph.graph_id for execution in compatible
            ],
            "eligible_denotation_count": len(denotations),
        }
    )
    if not compatible:
        evidence["reason"] = "no_typed_constraint_owner_candidate"
        return None, evidence
    if len(denotations) != 1:
        evidence["reason"] = "typed_owner_candidates_disagree"
        return None, evidence
    challenger = max(
        compatible,
        key=lambda execution: (
            float(execution.graph.score),
            str(execution.graph.graph_id),
        ),
    )
    evidence.update(
        {
            "status": "applied",
            "reason": "ontology_typed_constraint_owner_projection",
            "selected_graph_id": challenger.graph.graph_id,
            "selected_answer_count": len(
                frozenset(map(str, challenger.answer_ids))
            ),
        }
    )
    return challenger, evidence


def _drops_question_supported_explicit_constraint(
    question: str,
    source: QueryGraphCandidate,
    challenger: QueryGraphCandidate,
) -> bool:
    """Whether a post-selector proposal deletes an explicit scalar intent."""

    spec = _missing_constraint_spec(question)
    return bool(
        isinstance(spec, dict)
        and _graph_matches_explicit_spec_exactly(source, spec)
        and _graph_operator_has_question_evidence(source, question)
        and not _graph_matches_explicit_spec_exactly(challenger, spec)
    )


def _answer_predicate_echo_edges(
    graph: QueryGraphCandidate,
) -> tuple[tuple[str, str, str], ...]:
    """Return answer edges that repeat an anchor predicate from one subject.

    ``subject -r-> anchor`` plus ``subject -r-> answer`` is a frequent compose
    echo: the relation used to identify the subject is copied into the answer
    slot.  This helper is deliberately identity-blind; only graph roles and
    predicate equality are considered.
    """

    triples = [
        tuple(map(str, triple))
        for triple in graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    ]
    answer_var = str(graph.answer_var)
    return tuple(
        answer_edge
        for answer_edge in triples
        if answer_edge[2] == answer_var
        and any(
            anchor_edge[0] == answer_edge[0]
            and anchor_edge[1] == answer_edge[1]
            and anchor_edge[2] != answer_var
            for anchor_edge in triples
        )
    )


def _graphs_differ_by_one_relation(
    source: QueryGraphCandidate,
    candidate: QueryGraphCandidate,
) -> bool:
    """Whether exactly one edge keeps its endpoints and changes predicate."""

    def triple_counter(graph: QueryGraphCandidate) -> Counter[tuple[str, str, str]]:
        return Counter(
            tuple(map(str, triple))
            for triple in graph.triples
            if isinstance(triple, (list, tuple)) and len(triple) == 3
        )

    source_triples = triple_counter(source)
    candidate_triples = triple_counter(candidate)
    removed = list((source_triples - candidate_triples).elements())
    added = list((candidate_triples - source_triples).elements())
    return bool(
        len(removed) == len(added) == 1
        and removed[0][0] == added[0][0]
        and removed[0][2] == added[0][2]
        and removed[0][1] != added[0][1]
    )


def _hard_lexical_predicate_echo_challenger(
    question: str,
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Replace one answer-predicate echo with a lexically stronger edge.

    The challenger must be the identical graph except for one predicate,
    remove the same-subject predicate echo, preserve operator semantics, and
    improve the existing terminal lexical score by at least 0.05.  Requiring
    non-zero support for the incumbent prevents an unrelated token match from
    creating a switch.  No entity, relation, answer, model, or endpoint lookup
    participates in the decision.
    """

    evidence: dict[str, Any] = {
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "uses_existing_executions_only": True,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
        "entity_or_relation_whitelist": False,
    }
    source_echoes = _answer_predicate_echo_edges(selected.graph)
    evidence["source_predicate_echo_count"] = len(source_echoes)
    if not source_echoes:
        evidence["reason"] = "source_has_no_answer_predicate_echo"
        return None, evidence

    source_lexical = _terminal_relation_lexical_score(question, selected.graph)
    evidence["source_terminal_lexical"] = source_lexical
    if source_lexical <= 0.0:
        evidence["reason"] = "source_terminal_has_no_lexical_support"
        return None, evidence

    source_answers = set(map(str, selected.answer_ids))
    source_operators = _constraint_operator_signature(selected.graph)
    compatible: list[tuple[float, ExecutedGraph]] = []
    for execution in executions:
        candidate_answers = set(map(str, execution.answer_ids))
        if not candidate_answers or candidate_answers == source_answers:
            continue
        if not _graphs_differ_by_one_relation(selected.graph, execution.graph):
            continue
        if _answer_predicate_echo_edges(execution.graph):
            continue
        if _constraint_operator_signature(execution.graph) != source_operators:
            continue
        lexical = _terminal_relation_lexical_score(question, execution.graph)
        if lexical - source_lexical + 1e-12 < 0.05:
            continue
        compatible.append((lexical, execution))
    if not compatible:
        evidence["reason"] = "no_lexically_stronger_one_relation_edit"
        return None, evidence

    selected_lexical, challenger = max(
        compatible,
        key=lambda item: (
            item[0],
            float(item[1].graph.score),
            str(item[1].graph.graph_id),
        ),
    )
    evidence.update(
        {
            "status": "applied",
            "reason": "lexically_stronger_answer_predicate_replaces_echo",
            "selected_graph_id": challenger.graph.graph_id,
            "selected_rule_score": float(challenger.graph.score),
            "selected_terminal_lexical": selected_lexical,
            "minimum_lexical_delta": 0.05,
            "one_relation_edit": True,
            "operator_semantics_preserved": True,
            "eligible_graph_ids": [
                execution.graph.graph_id for _, execution in compatible
            ],
        }
    )
    return challenger, evidence


def _directed_chain_predicate_echo_count(graph: QueryGraphCandidate) -> int:
    """Count ``x -r-> y -r-> z`` directed predicate echoes."""

    triples = [
        tuple(map(str, triple))
        for triple in graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    ]
    return sum(
        1
        for left_index, left in enumerate(triples)
        for right_index, right in enumerate(triples)
        if left_index != right_index
        and left[1] == right[1]
        and left[2] == right[0]
    )


def _hard_directed_chain_echo_subset_challenger(
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Reduce a directed predicate echo only via a strict answer subset."""

    source_echoes = _directed_chain_predicate_echo_count(selected.graph)
    source_answers = set(map(str, selected.answer_ids))
    evidence: dict[str, Any] = {
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "source_directed_chain_echo_count": source_echoes,
        "uses_existing_executions_only": True,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
        "entity_or_relation_whitelist": False,
    }
    if source_echoes <= 0 or not source_answers:
        evidence["reason"] = "source_has_no_directed_chain_echo"
        return None, evidence

    compatible = [
        execution
        for execution in executions
        if execution.answer_ids
        and set(map(str, execution.answer_ids)) < source_answers
        and _directed_chain_predicate_echo_count(execution.graph) < source_echoes
    ]
    if not compatible:
        evidence["reason"] = "no_echo_reducing_strict_subset"
        return None, evidence

    challenger = max(
        compatible,
        key=lambda execution: (
            float(execution.graph.score),
            str(execution.graph.graph_id),
        ),
    )
    evidence.update(
        {
            "status": "applied",
            "reason": "directed_chain_echo_reduced_by_strict_subset",
            "selected_graph_id": challenger.graph.graph_id,
            "selected_rule_score": float(challenger.graph.score),
            "selected_directed_chain_echo_count": (
                _directed_chain_predicate_echo_count(challenger.graph)
            ),
            "source_answer_count": len(source_answers),
            "selected_answer_count": len(set(map(str, challenger.answer_ids))),
            "eligible_graph_ids": [
                execution.graph.graph_id for execution in compatible
            ],
        }
    )
    return challenger, evidence


def _relation_duplicate_count(graph: QueryGraphCandidate) -> int:
    """Return repeated-predicate count in the graph relation multiset."""

    relations = [
        str(triple[1])
        for triple in graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    ]
    return len(relations) - len(set(relations))


def _full_operator_signature(graph: QueryGraphCandidate) -> tuple[str, ...]:
    """Byte-stable signature including NO_EQUAL and variable operands."""

    return tuple(
        sorted(
            json.dumps(
                operator,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for operator in graph.operators
            if isinstance(operator, dict)
        )
    )


def _materialized_structural_incumbent(
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
) -> tuple[ExecutedGraph, str]:
    """Recover the execution whose structure produced the selected answers.

    A low-margin union retains the primary graph while its answer set may be
    materialized exactly by another execution.  Prefer the selected graph when
    its own execution already has the full selected set; otherwise use the
    highest-score complete execution with that exact answer set.  This keeps
    ordinary selections unchanged and makes union structure deterministic.
    """

    selected_answers = set(map(str, selected.answer_ids))
    same_graph = [
        execution
        for execution in executions
        if execution.graph.graph_id == selected.graph.graph_id
        and set(map(str, execution.answer_ids)) == selected_answers
    ]
    if same_graph:
        return max(
            same_graph,
            key=lambda execution: (
                float(execution.graph.score),
                str(execution.graph.graph_id),
            ),
        ), "selected_graph_exact_answer_set"
    exact_answer_set = [
        execution
        for execution in executions
        if set(map(str, execution.answer_ids)) == selected_answers
    ]
    if exact_answer_set:
        return max(
            exact_answer_set,
            key=lambda execution: (
                float(execution.graph.score),
                str(execution.graph.graph_id),
            ),
        ), "union_answer_set_materialization"
    return selected, "selected_execution"


def _hard_relation_duplicate_subset_challenger(
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Narrow a repeated-relation incumbent with an operator-identical graph.

    The rule is intentionally strict: the executed challenger must be a
    non-empty strict answer subset of at most three answers, reduce relation
    multiset duplication, use no more triples, preserve both substantive and
    full operator signatures (including NO_EQUAL), and remain within 0.03 of
    the incumbent rule score.  It uses no question, entity, predicate list,
    model, embedding, or endpoint request.
    """

    incumbent, incumbent_source = _materialized_structural_incumbent(
        selected, executions
    )
    source_answers = set(map(str, incumbent.answer_ids))
    source_duplicates = _relation_duplicate_count(incumbent.graph)
    source_substantive_operators = _constraint_operator_signature(incumbent.graph)
    source_full_operators = _full_operator_signature(incumbent.graph)
    source_triple_count = len(incumbent.graph.triples)
    evidence: dict[str, Any] = {
        "status": "not_applied",
        "source_graph_id": incumbent.graph.graph_id,
        "selected_graph_id_before_canonicalization": selected.graph.graph_id,
        "incumbent_source": incumbent_source,
        "source_relation_duplicate_count": source_duplicates,
        "uses_existing_executions_only": True,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
        "entity_or_relation_whitelist": False,
    }
    if source_duplicates <= 0 or not source_answers:
        evidence["reason"] = "source_relation_multiset_has_no_duplicate"
        return None, evidence

    compatible: list[ExecutedGraph] = []
    for execution in executions:
        candidate_answers = set(map(str, execution.answer_ids))
        if not candidate_answers or not candidate_answers < source_answers:
            continue
        if len(candidate_answers) > 3:
            continue
        candidate_duplicates = _relation_duplicate_count(execution.graph)
        if candidate_duplicates >= source_duplicates:
            continue
        if len(execution.graph.triples) > source_triple_count:
            continue
        if (
            _constraint_operator_signature(execution.graph)
            != source_substantive_operators
            or _full_operator_signature(execution.graph) != source_full_operators
        ):
            continue
        if float(execution.graph.score) + 1e-12 < float(incumbent.graph.score) - 0.03:
            continue
        compatible.append(execution)
    if not compatible:
        evidence["reason"] = "no_operator_identical_duplicate_reducing_subset"
        return None, evidence

    challenger = max(
        compatible,
        key=lambda execution: (
            float(execution.graph.score),
            str(execution.graph.graph_id),
        ),
    )
    evidence.update(
        {
            "status": "applied",
            "reason": "relation_duplicate_reduced_by_bounded_strict_subset",
            "selected_graph_id": challenger.graph.graph_id,
            "selected_rule_score": float(challenger.graph.score),
            "selected_relation_duplicate_count": _relation_duplicate_count(
                challenger.graph
            ),
            "source_answer_count": len(source_answers),
            "selected_answer_count": len(set(map(str, challenger.answer_ids))),
            "source_triple_count": source_triple_count,
            "selected_triple_count": len(challenger.graph.triples),
            "rule_score_floor": float(incumbent.graph.score) - 0.03,
            "substantive_operator_signature_equal": True,
            "full_operator_signature_equal": True,
            "eligible_graph_ids": [
                execution.graph.graph_id for execution in compatible
            ],
        }
    )
    return challenger, evidence


def _plural_answer_request(question: str) -> bool:
    """Return a morphology-only multi-answer hint.

    This deliberately contains no domain vocabulary.  It is only a guard
    against collapsing a broad, already populated answer set for a question
    whose surface form explicitly requests a plural result.
    """

    text = " ".join(str(question).casefold().split())
    if re.search(r"\b(?:are|were|have)\b", text):
        return True
    match = re.match(r"^(?:what|which)\s+([a-z][a-z-]*)", text)
    return bool(
        match
        and match.group(1).endswith("s")
        and not match.group(1).endswith("ss")
    )


def _provenance_anchor_ids(graph: QueryGraphCandidate) -> frozenset[str]:
    """Extract grounded question-entity IDs from graph provenance."""

    bindings = graph.provenance.get("anchor_bindings", {})
    if not isinstance(bindings, dict):
        return frozenset()
    values: set[str] = set()
    for binding in bindings.values():
        if isinstance(binding, dict):
            entity_id = str(binding.get("id", "")).strip()
        else:
            entity_id = str(getattr(binding, "entity_id", "")).strip()
        if entity_id:
            values.add(entity_id)
    return frozenset(values)


def _loses_distinct_conjunctive_entity_anchors(
    question: str,
    graph: QueryGraphCandidate,
    linked_entities: list[tuple[str, str]],
) -> bool:
    """Protect distinct linked entities in an explicit conjunction."""

    expected_ids = {
        str(entity_id)
        for entity_id, _ in linked_entities
        if str(entity_id).strip()
    }
    if len(expected_ids) < 2 or not _EXPLICIT_CONJUNCTION_RE.search(str(question)):
        return False
    represented = _provenance_anchor_ids(graph) & expected_ids
    return len(represented) < len(expected_ids)


def _grounded_question_anchor_ids(
    executions: list[ExecutedGraph],
) -> frozenset[str]:
    """Collect grounded IDs represented anywhere in the executed beam."""

    return frozenset(
        entity_id
        for execution in executions
        for entity_id in _provenance_anchor_ids(execution.graph)
    )


def _graph_grounded_anchor_coverage(
    graph: QueryGraphCandidate,
    grounded_ids: frozenset[str],
) -> frozenset[str]:
    """Return grounded question anchors actually constrained by ``graph``."""

    return frozenset(
        str(node)
        for triple in graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
        for node in (triple[0], triple[2])
        if str(node) in grounded_ids
    )


def _hard_grounded_anchor_coverage_challenger(
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Prefer an execution that covers strictly more grounded anchors.

    The current full answer set must itself be materialized by an execution,
    source coverage must be non-zero, and the challenger must cover at least
    two grounded question anchors.  These constraints are the production form
    of the v12 audited first-level gate and use neither Gold nor a schema ID
    whitelist.
    """

    source_answers = frozenset(map(str, selected.answer_ids))
    # The selector trace fully materializes at most this many answers.  Keep
    # production inside the exact cohort covered by the all-v12 audit instead
    # of acting on answer sets that were truncated in saved evidence.
    bounded_executions = [
        execution
        for execution in executions
        if len(execution.answer_ids) <= _SELECTOR_ANSWER_PREVIEW
    ]
    grounded_ids = _grounded_question_anchor_ids(bounded_executions)
    evidence: dict[str, Any] = {
        "status": "not_applied",
        "reason": "",
        "source_graph_id": selected.graph.graph_id,
        "source_answer_count": len(source_answers),
        "grounded_anchor_count": len(grounded_ids),
        "complete_answer_limit": _SELECTOR_ANSWER_PREVIEW,
        "score_margin": None,
        "uses_existing_executions_only": True,
        "uses_gold_answers": False,
        "entity_or_relation_whitelist": False,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
    }
    materialized = [
        execution
        for execution in bounded_executions
        if frozenset(map(str, execution.answer_ids)) == source_answers
    ]
    if not materialized:
        evidence["reason"] = "current_answer_set_not_materialized"
        return None, evidence
    source_coverage = max(
        len(_graph_grounded_anchor_coverage(execution.graph, grounded_ids))
        for execution in materialized
    )
    maximum_coverage = max(
        (
            len(_graph_grounded_anchor_coverage(execution.graph, grounded_ids))
            for execution in bounded_executions
        ),
        default=0,
    )
    evidence.update(
        {
            "source_anchor_coverage": source_coverage,
            "maximum_anchor_coverage": maximum_coverage,
        }
    )
    if source_coverage < 1 or maximum_coverage < 2:
        evidence["reason"] = "insufficient_grounded_anchor_evidence"
        return None, evidence
    if maximum_coverage <= source_coverage:
        evidence["reason"] = "no_strict_anchor_coverage_gain"
        return None, evidence
    eligible = [
        execution
        for execution in bounded_executions
        if execution.answer_ids
        and len(
            _graph_grounded_anchor_coverage(execution.graph, grounded_ids)
        )
        == maximum_coverage
    ]
    if not eligible:
        evidence["reason"] = "no_nonempty_max_coverage_execution"
        return None, evidence
    challenger = max(
        eligible,
        key=lambda execution: (
            float(execution.graph.score),
            -len(set(map(str, execution.answer_ids))),
            str(execution.graph.graph_id),
        ),
    )
    if frozenset(map(str, challenger.answer_ids)) == source_answers:
        evidence["reason"] = "max_coverage_answer_set_unchanged"
        return None, evidence
    evidence.update(
        {
            "status": "applied",
            "reason": "strict_grounded_anchor_coverage_gain",
            "selected_graph_id": challenger.graph.graph_id,
            "selected_rule_score": float(challenger.graph.score),
            "selected_anchor_coverage": maximum_coverage,
            "selected_answer_count": len(
                set(map(str, challenger.answer_ids))
            ),
        }
    )
    return challenger, evidence


def _terminal_type_union(
    graph: QueryGraphCandidate,
    ontology: Any,
) -> frozenset[str]:
    return frozenset(
        inherited
        for _, closure in _terminal_answer_type_closures(graph, ontology)
        for inherited in closure
    )


def _distinct_answer_executions(
    executions: list[ExecutedGraph],
) -> list[ExecutedGraph]:
    """Keep the strongest execution for every complete answer set."""

    representatives: dict[frozenset[str], ExecutedGraph] = {}
    for execution in executions:
        values = frozenset(map(str, execution.answer_ids))
        if not values:
            continue
        previous = representatives.get(values)
        if previous is None or (
            float(execution.graph.score),
            str(execution.graph.graph_id),
        ) > (
            float(previous.graph.score),
            str(previous.graph.graph_id),
        ):
            representatives[values] = execution
    return sorted(
        representatives.values(),
        key=lambda execution: (
            float(execution.graph.score),
            str(execution.graph.graph_id),
        ),
        reverse=True,
    )


def _intersect_executed_answers(
    left: ExecutedGraph,
    right: ExecutedGraph,
) -> ExecutedGraph:
    """Materialize the observed intersection without another KG query."""

    right_ids = set(map(str, right.answer_ids))
    answer_ids = [
        str(answer_id)
        for answer_id in left.answer_ids
        if str(answer_id) in right_ids
    ]
    answers_by_id: dict[str, dict[str, str]] = {}
    for answer in [*left.answers, *right.answers]:
        if isinstance(answer, dict) and str(answer.get("id", "")):
            answers_by_id.setdefault(str(answer["id"]), answer)
    selected_graph = deepcopy(left.graph)
    selected_graph.provenance["selection_answer_intersection"] = {
        "left_graph_id": left.graph.graph_id,
        "right_graph_id": right.graph.graph_id,
        "score_margin": _CONJUNCTIVE_INTERSECTION_SCORE_MARGIN,
    }
    return ExecutedGraph(
        graph=selected_graph,
        answer_ids=answer_ids,
        answers=[
            answers_by_id.get(answer_id, {"id": answer_id, "label": answer_id})
            for answer_id in answer_ids
        ],
        row_count=len(answer_ids),
    )


def _hard_conjunctive_answer_intersection_challenger(
    question: str,
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
    ontology: Any,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Intersect two near-top executions under hard semantic compatibility.

    This is the second-level v12 gate.  It runs only when the anchor-coverage
    gate abstains.  The 0.075 global margin was selected on the fixed
    train+calibration split, not on test.  Equal operator counts prevent a
    one-sided constraint augmentation, while ontology terminal-type overlap
    prevents intersections between different requested answer kinds.
    """

    source_answers = frozenset(map(str, selected.answer_ids))
    evidence: dict[str, Any] = {
        "status": "not_applied",
        "reason": "",
        "source_graph_id": selected.graph.graph_id,
        "source_answer_count": len(source_answers),
        "complete_answer_limit": _SELECTOR_ANSWER_PREVIEW,
        "score_margin": _CONJUNCTIVE_INTERSECTION_SCORE_MARGIN,
        "uses_existing_executions_only": True,
        "uses_gold_answers": False,
        "entity_or_relation_whitelist": False,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "additional_embedding_calls": 0,
    }
    if len(source_answers) < 2:
        evidence["reason"] = "current_answer_set_too_small"
        return None, evidence
    if len(source_answers) > _SELECTOR_ANSWER_PREVIEW:
        evidence["reason"] = "current_answer_set_not_fully_audited"
        return None, evidence
    if not _EXPLICIT_CONJUNCTION_RE.search(str(question)):
        evidence["reason"] = "question_has_no_explicit_conjunction"
        return None, evidence
    if _plural_answer_request(question) and len(source_answers) > 2:
        evidence["reason"] = "plural_multi_answer_preservation_guard"
        return None, evidence
    if ontology is None:
        evidence["reason"] = "ontology_unavailable"
        return None, evidence

    representatives = _distinct_answer_executions(
        [
            execution
            for execution in executions
            if len(execution.answer_ids) <= _SELECTOR_ANSWER_PREVIEW
        ]
    )
    if len(representatives) < 2:
        evidence["reason"] = "fewer_than_two_distinct_answer_sets"
        return None, evidence
    top_score = float(representatives[0].graph.score)
    eligible = [
        execution
        for execution in representatives
        if top_score - float(execution.graph.score)
        <= _CONJUNCTIVE_INTERSECTION_SCORE_MARGIN + 1e-12
    ]
    grounded_ids = _grounded_question_anchor_ids(eligible)
    parsed_spec = _missing_constraint_spec(question)
    explicit_spec = (
        parsed_spec
        if isinstance(parsed_spec, dict)
        and str(parsed_spec.get("value", "")).strip()
        and str(parsed_spec.get("type", "")).upper()
        not in {"ARGMIN", "ARGMAX", "COUNT"}
        else None
    )
    evidence["grounded_anchor_obligation_count"] = len(grounded_ids)
    evidence["explicit_constraint_obligation"] = deepcopy(explicit_spec)
    rejected_anchor_obligation_pairs = 0
    rejected_scalar_obligation_pairs = 0
    options: list[
        tuple[
            float,
            int,
            float,
            tuple[str, ...],
            ExecutedGraph,
            ExecutedGraph,
        ]
    ] = []
    for left, right in itertools.combinations(eligible, 2):
        if len(left.graph.operators) != len(right.graph.operators):
            continue
        left_types = _terminal_type_union(left.graph, ontology)
        right_types = _terminal_type_union(right.graph, ontology)
        if not left_types or not right_types or not (left_types & right_types):
            continue
        if len(grounded_ids) >= 2:
            covered = (
                _graph_grounded_anchor_coverage(left.graph, grounded_ids)
                | _graph_grounded_anchor_coverage(right.graph, grounded_ids)
            )
            if covered != grounded_ids:
                rejected_anchor_obligation_pairs += 1
                continue
        if isinstance(explicit_spec, dict) and not (
            _graph_matches_explicit_spec_exactly(left.graph, explicit_spec)
            or _graph_matches_explicit_spec_exactly(right.graph, explicit_spec)
        ):
            rejected_scalar_obligation_pairs += 1
            continue
        intersection = frozenset(map(str, left.answer_ids)) & frozenset(
            map(str, right.answer_ids)
        )
        if not intersection or not intersection < source_answers:
            continue
        options.append(
            (
                min(float(left.graph.score), float(right.graph.score)),
                len(intersection),
                float(left.graph.score) + float(right.graph.score),
                tuple(sorted(intersection)),
                left,
                right,
            )
        )
    if not options:
        evidence["rejected_anchor_obligation_pairs"] = (
            rejected_anchor_obligation_pairs
        )
        evidence["rejected_scalar_obligation_pairs"] = (
            rejected_scalar_obligation_pairs
        )
        evidence["reason"] = "no_semantically_compatible_strict_intersection"
        return None, evidence
    selected_option = max(options, key=lambda item: item[:4])
    left, right = selected_option[4], selected_option[5]
    challenger = _intersect_executed_answers(left, right)
    evidence.update(
        {
            "status": "applied",
            "reason": "explicit_conjunction_near_top_answer_intersection",
            "selected_graph_id": challenger.graph.graph_id,
            "left_graph_id": left.graph.graph_id,
            "right_graph_id": right.graph.graph_id,
            "left_rule_score": float(left.graph.score),
            "right_rule_score": float(right.graph.score),
            "operator_count": len(left.graph.operators),
            "shared_terminal_type_count": len(
                _terminal_type_union(left.graph, ontology)
                & _terminal_type_union(right.graph, ontology)
            ),
            "selected_answer_count": len(challenger.answer_ids),
        }
    )
    return challenger, evidence


def _apply_hard_post_selection_gates(
    question: str,
    selected: ExecutedGraph,
    executions: list[ExecutedGraph],
    ontology: Any,
    decision: dict[str, Any],
) -> tuple[ExecutedGraph, dict[str, Any]]:
    """Apply independent Gold-blind hard gates to the final selection."""
    gates = (
        (
            "hard_explicit_constraint_subset_gate",
            "hard_explicit_constraint_subset_gate",
            lambda current: _hard_explicit_constraint_subset_challenger(
                question, current, executions, ontology
            ),
        ),
        (
            "hard_constraint_owner_projection_gate",
            "hard_constraint_owner_projection_gate",
            lambda current: _hard_constraint_owner_projection_challenger(
                question, current, executions, ontology
            ),
        ),
        (
            "hard_scalar_attribute_projection_gate",
            "hard_scalar_attribute_projection_gate",
            lambda current: _hard_scalar_attribute_projection_challenger(
                question, current, executions
            ),
        ),
        (
            "hard_lexical_predicate_echo_gate",
            "hard_lexical_predicate_echo_gate",
            lambda current: _hard_lexical_predicate_echo_challenger(
                question, current, executions
            ),
        ),
        (
            "hard_directed_chain_echo_subset_gate",
            "hard_directed_chain_echo_subset_gate",
            lambda current: _hard_directed_chain_echo_subset_challenger(
                current, executions
            ),
        ),
        (
            "hard_relation_duplicate_subset_gate",
            "hard_relation_duplicate_subset_gate",
            lambda current: _hard_relation_duplicate_subset_challenger(
                current, executions
            ),
        ),
        (
            "hard_ontology_answer_type_gate",
            "hard_ontology_who_person_gate",
            lambda current: _hard_ontology_who_person_challenger(
                question, current, executions, ontology
            ),
        ),
        (
            "hard_ontology_head_subtype_subset_gate",
            "hard_ontology_head_subtype_subset_gate",
            lambda current: _hard_ontology_head_subset_challenger(
                question, current, executions, ontology
            ),
        ),
        (
            "hard_explicit_equality_gate",
            "hard_explicit_equality_gate",
            lambda current: _hard_explicit_equality_challenger(
                question, current, executions
            ),
        ),
        (
            "hard_additive_operator_gate",
            "hard_additive_operator_gate",
            lambda current: _hard_additive_operator_challenger(
                question, current, executions
            ),
        ),
    )
    updated = dict(decision)
    current = selected
    for evidence_key, reason_code, apply_gate in gates:
        challenger, evidence = apply_gate(current)
        if challenger is None:
            continue
        updated[evidence_key] = evidence
        updated[f"{evidence_key}_source_graph_id"] = current.graph.graph_id
        updated["selected_graph_id"] = challenger.graph.graph_id
        updated["reason_codes"] = [
            *updated.get("reason_codes", []),
            reason_code,
        ]
        current = challenger

    # The v12 constraint-coverage rules are a prioritized two-level gate.
    # Pair intersection is intentionally not applied after an anchor-coverage
    # replacement: the full audit and the runtime contract both give the
    # stronger graph-level constraint evidence first refusal.
    anchor_challenger, anchor_evidence = (
        _hard_grounded_anchor_coverage_challenger(current, executions)
    )
    if anchor_challenger is not None:
        evidence_key = "hard_grounded_anchor_coverage_gate"
        updated[evidence_key] = anchor_evidence
        updated[f"{evidence_key}_source_graph_id"] = current.graph.graph_id
        updated["selected_graph_id"] = anchor_challenger.graph.graph_id
        updated["reason_codes"] = [
            *updated.get("reason_codes", []),
            "hard_grounded_anchor_coverage_gate",
        ]
        current = anchor_challenger
    else:
        intersection_challenger, intersection_evidence = (
            _hard_conjunctive_answer_intersection_challenger(
                question,
                current,
                executions,
                ontology,
            )
        )
        if intersection_challenger is not None:
            evidence_key = "hard_conjunctive_answer_intersection_gate"
            updated[evidence_key] = intersection_evidence
            updated[f"{evidence_key}_source_graph_id"] = current.graph.graph_id
            updated["selected_graph_id"] = intersection_challenger.graph.graph_id
            updated["reason_codes"] = [
                *updated.get("reason_codes", []),
                "hard_conjunctive_answer_intersection_gate",
            ]
            current = intersection_challenger
    return current, updated


_NON_NUMERIC_EXTREMA_RANGES = {
    "type.boolean",
    "type.datetime",
    "type.rawstring",
    "type.text",
    "type.uri",
}
_BOUND_ORDER_RANGES = {
    "type.datetime",
    "type.enumeration",
    "type.float",
    "type.int",
    "type.rawstring",
}


def _annotate_extrema_order_semantics(
    graph: QueryGraphCandidate,
    ontology: Any,
) -> None:
    """Choose numeric versus RDF extrema ordering from schema metadata.

    Freebase stores some numeric identifiers as enumeration literals and a few
    legacy numeric predicates are absent from ``fb_roles``. Numeric extrema
    must cast those values, while known datetime/text properties retain RDF
    ordering. This adds no model or knowledge-graph request.
    """
    if ontology is None:
        return
    annotations: list[dict[str, str]] = []
    for operator in graph.operators:
        if str(operator.get("type", "")).upper() not in {"ARGMAX", "ARGMIN"}:
            continue
        relation_id = str(operator.get("attribute_relation_id", "")).strip()
        if not relation_id:
            label = operator.get("attribute_relation_label", [])
            relation_id = relation_id_from_label(label) if label else ""
        input_var = str(operator.get("input_var", ""))
        range_id = str(ontology.range_for_relation(relation_id)) if relation_id else ""
        if relation_id:
            if range_id not in _NON_NUMERIC_EXTREMA_RANGES:
                operator["_numeric_order_cast"] = (
                    "float" if range_id == "type.float" else "integer"
                )
                annotations.append(
                    {
                        "input_var": input_var,
                        "relation_id": relation_id,
                        "range": range_id or "unknown",
                        "ordering": "numeric",
                    }
                )
            continue

        incoming = [
            (str(relation), str(ontology.range_for_relation(relation)))
            for _, relation, object_ in graph.triples
            if str(object_) == input_var
        ]
        scalar_incoming = [
            (relation, relation_range)
            for relation, relation_range in incoming
            if relation_range in _BOUND_ORDER_RANGES
        ]
        if len(scalar_incoming) != 1:
            continue
        relation_id, range_id = scalar_incoming[0]
        operator["_order_bound_value"] = True
        if range_id not in _NON_NUMERIC_EXTREMA_RANGES:
            operator["_numeric_order_cast"] = (
                "float" if range_id == "type.float" else "integer"
            )
        annotations.append(
            {
                "input_var": input_var,
                "relation_id": relation_id,
                "range": range_id,
                "ordering": (
                    "numeric" if operator.get("_numeric_order_cast") else "rdf"
                ),
            }
        )
    if annotations:
        graph.provenance["extrema_order_semantics"] = annotations


def _date_value_precision(value: Any) -> str:
    text = str(value).strip()
    parts = text.split("-")
    if len(parts) == 1 and len(parts[0]) == 4 and parts[0].isdigit():
        return "year"
    if (
        len(parts) == 2
        and len(parts[0]) == 4
        and len(parts[1]) == 2
        and all(part.isdigit() for part in parts)
    ):
        return "month"
    if (
        len(parts) == 3
        and len(parts[0]) == 4
        and len(parts[1]) == 2
        and len(parts[2]) == 2
        and all(part.isdigit() for part in parts)
    ):
        return "day"
    return ""


def _annotate_temporal_comparison_semantics(
    graph: QueryGraphCandidate,
    ontology: Any,
) -> None:
    """Normalize date comparison precision from schema and literal structure."""
    if ontology is None:
        return
    changes: list[dict[str, str]] = []
    supported = {
        "EQUAL",
        "GREATER_THAN",
        "GREATER_OR_EQUAL",
        "LESS_THAN",
        "LESS_OR_EQUAL",
    }
    for operator in graph.operators:
        operator_type = str(operator.get("type", "")).upper()
        if operator_type not in supported:
            continue
        relation_id = str(operator.get("attribute_relation_id", "")).strip()
        if not relation_id:
            label = operator.get("attribute_relation_label", [])
            relation_id = relation_id_from_label(label) if label else ""
        input_var = str(operator.get("input_var", ""))
        temporal = bool(
            relation_id
            and str(ontology.range_for_relation(relation_id)) == "type.datetime"
        )
        if not temporal and not relation_id:
            incoming_ranges = [
                str(ontology.range_for_relation(relation))
                for _, relation, object_ in graph.triples
                if str(object_) == input_var
            ]
            temporal = incoming_ranges.count("type.datetime") == 1
        if not temporal:
            continue
        precision = _date_value_precision(operator.get("value", ""))
        if not precision:
            continue
        # Ontology attribute repair initially marks every range comparison as
        # numeric before the concrete property range is known.  Once schema
        # evidence proves this is datetime, remove that provisional cast so
        # lowering uses calendar precision instead of bif:atoi on a date.
        operator.pop("_numeric_comparison_cast", None)
        operator.pop("_numeric_cast_kind", None)
        operator.pop("_numeric_cast_compare", None)
        operator["_comparison_date_precision"] = precision
        operator["_calendar_day_boundary"] = precision == "day"
        changes.append(
            {
                "input_var": input_var,
                "relation_id": relation_id,
                "precision": precision,
            }
        )
    if changes:
        graph.provenance["temporal_comparison_semantics"] = changes


def _annotate_numeric_comparison_semantics(
    graph: QueryGraphCandidate,
    ontology: Any,
) -> None:
    """Cast schema-proven numeric filters without changing model output."""
    if ontology is None:
        return
    numeric_ranges = {"type.enumeration", "type.float", "type.int"}
    range_types = {
        "GREATER_THAN",
        "GREATER_OR_EQUAL",
        "LESS_THAN",
        "LESS_OR_EQUAL",
    }
    changes: list[dict[str, str]] = []
    for operator in graph.operators:
        operator_type = str(operator.get("type", "")).upper()
        if operator_type not in {*range_types, "EQUAL"}:
            continue
        if operator.get("_comparison_date_precision"):
            continue
        raw_value = str(operator.get("value", "")).strip().replace(",", "")
        try:
            Decimal(raw_value)
        except (InvalidOperation, ValueError):
            continue
        relation_id = str(operator.get("attribute_relation_id", "")).strip()
        if not relation_id:
            label = operator.get("attribute_relation_label", [])
            relation_id = relation_id_from_label(label) if label else ""
        input_var = str(operator.get("input_var", ""))
        relation_range = str(ontology.range_for_relation(relation_id)) if relation_id else ""
        numeric = bool(
            relation_id
            and (
                relation_range in numeric_ranges
                or not relation_range
            )
        )
        if not numeric and not relation_id:
            incoming_ranges = [
                str(ontology.range_for_relation(relation))
                for _, relation, object_ in graph.triples
                if str(object_) == input_var
            ]
            numeric = sum(value in numeric_ranges for value in incoming_ranges) == 1
        if not numeric:
            continue
        operator["value"] = raw_value
        operator["_numeric_cast_kind"] = (
            "float"
            if relation_range == "type.float" or "." in raw_value
            else "integer"
        )
        if operator_type == "EQUAL":
            operator["_numeric_cast_compare"] = True
        else:
            operator["_numeric_comparison_cast"] = True
        changes.append(
            {
                "input_var": input_var,
                "relation_id": relation_id,
                "range": relation_range or "unknown",
                "operator": operator_type,
            }
        )
    if changes:
        graph.provenance["numeric_comparison_semantics"] = changes


def _temporal_order_sibling_graph(
    graph: QueryGraphCandidate,
) -> tuple[QueryGraphCandidate, list[dict[str, str]]] | None:
    """Swap start/end date for an empty temporal ARGMIN/ARGMAX query."""
    sibling = deepcopy(graph)
    changes: list[dict[str, str]] = []
    for operator in sibling.operators:
        if str(operator.get("type", "")).upper() not in {"ARGMIN", "ARGMAX"}:
            continue
        label = operator.get("attribute_relation_label")
        if not isinstance(label, list) or not label:
            continue
        tail = str(label[-1]).casefold().replace("_", " ")
        if tail == "end date":
            replacement = "start date"
        elif tail == "start date":
            replacement = "end date"
        else:
            continue
        changes.append({"from": str(label[-1]), "to": replacement})
        label[-1] = replacement
        relation_id = str(operator.get("attribute_relation_id", ""))
        if relation_id.endswith(".end_date") and replacement == "start date":
            operator["attribute_relation_id"] = relation_id[: -len("end_date")] + "start_date"
        elif relation_id.endswith(".start_date") and replacement == "end date":
            operator["attribute_relation_id"] = relation_id[: -len("start_date")] + "end_date"
    if not changes:
        return None
    sibling.sparql = ""
    return sibling, changes


def _normalize_notable_type_constraint(graph: QueryGraphCandidate) -> list[dict[str, str]]:
    """Collapse an incorrectly grounded ``notable_for`` type constraint.

    The local ontology exposes both ``common.topic.notable_types`` and the
    unrelated ``common.notable_for`` CVT.  A Semantic declaration explicitly
    asking for a notable *type* can occasionally be grounded through the CVT.
    Normalize only that structural mismatch; ordinary notable-for questions
    are untouched.
    """
    decomposition = " ".join(
        str(item) for item in graph.provenance.get("decomposition", [])
    )
    if not re.search(r"\bnotable\s+types?\b", decomposition, re.I):
        return []
    triples = [list(item) for item in graph.triples]
    changes: list[dict[str, str]] = []
    remove: set[int] = set()
    additions: list[list[str]] = []
    for predicate_index, (subject, relation, middle) in enumerate(triples):
        if relation != "common.notable_for.predicate" or not str(middle).startswith("V"):
            continue
        for object_index, (owner, object_relation, type_entity) in enumerate(triples):
            if (
                owner != middle
                or object_relation != "common.notable_for.notable_object"
                or not re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", str(type_entity))
            ):
                continue
            direct = [subject, "common.topic.notable_types", type_entity]
            if direct not in triples and direct not in additions:
                additions.append(direct)
            remove.update({predicate_index, object_index})
            changes.append(
                {
                    "subject": str(subject),
                    "type_entity": str(type_entity),
                    "strategy": "notable_type_direct_relation",
                }
            )
    if not changes:
        return []
    graph.triples = [
        triple for index, triple in enumerate(triples) if index not in remove
    ] + additions
    graph.sparql = ""
    graph.provenance["notable_type_constraint_repairs"] = deepcopy(changes)
    return changes


def _repair_event_temporal_extrema(
    graph: QueryGraphCandidate,
    ontology: Any,
) -> list[dict[str, str]]:
    """Ground a generic latest/earliest event operator to event dates."""
    if ontology is None:
        return []
    question = str(graph.provenance.get("original_question", ""))
    latest = bool(re.search(r"\b(?:latest|last|most recent)\b", question, re.I))
    earliest = bool(
        re.search(r"\b(?:earliest|first)\b", question, re.I)
        and not re.search(r"\bfirst name\b", question, re.I)
    )
    if latest == earliest:
        return []
    repairs: list[dict[str, str]] = []
    has_extrema = any(
        str(operator.get("type", "")).upper() in {"ARGMAX", "ARGMIN"}
        for operator in graph.operators
    )
    if not has_extrema:
        answer_types = {
            str(ontology.range_for_relation(relation_id))
            for _, relation_id, object_ in graph.triples
            if str(object_) == graph.answer_var
        }
        inherited = {
            parent
            for type_id in answer_types
            if type_id
            for parent in ontology.supertypes(type_id)
        }
        if "time.event" in inherited:
            relation_id = (
                "time.event.end_date" if latest else "time.event.start_date"
            )
            graph.operators.append(
                {
                    "type": "ARGMAX" if latest else "ARGMIN",
                    "inputs": [],
                    "input_var": graph.answer_var,
                    "attribute_relation_label": relation_label_from_id(relation_id),
                    "attribute_relation_labels": [],
                    "value": "",
                    "value_type": "",
                    "attribute_relation_id": relation_id,
                    "_order_temporal_lexical": True,
                    "_explicit_event_order_repair": True,
                }
            )
            repairs.append(
                {
                    "input_var": graph.answer_var,
                    "from": "",
                    "to": relation_id,
                }
            )
    for operator in graph.operators:
        operator_type = str(operator.get("type", "")).upper()
        if operator_type not in {"ARGMAX", "ARGMIN"}:
            continue
        input_var = str(operator.get("input_var", ""))
        incoming_types = {
            str(ontology.range_for_relation(relation_id))
            for _, relation_id, object_ in graph.triples
            if str(object_) == input_var
        }
        inherited_types = {
            inherited
            for type_id in incoming_types
            if type_id
            for inherited in ontology.supertypes(type_id)
        }
        if "time.event" not in inherited_types:
            continue
        relation_id = (
            "time.event.end_date" if latest else "time.event.start_date"
        )
        old_relation = str(operator.get("attribute_relation_id", ""))
        if old_relation == relation_id:
            continue
        operator["attribute_relation_id"] = relation_id
        operator["attribute_relation_label"] = relation_label_from_id(relation_id)
        operator.pop("_numeric_order_cast", None)
        operator["_order_temporal_lexical"] = True
        repairs.append(
            {
                "input_var": input_var,
                "from": old_relation,
                "to": relation_id,
            }
        )
    if repairs:
        graph.sparql = ""
        graph.provenance["event_temporal_extrema_repairs"] = deepcopy(repairs)
    return repairs


def _complete_singular_answer_extrema(
    graph: QueryGraphCandidate,
    ontology: Any,
) -> dict[str, str] | None:
    """Complete implicit single-answer ordering used by CWQ base queries.

    CWQ's source queries often encode a current stadium with ``ORDER BY DESC``
    even when the compositional surface omits words such as ``latest``. Infer
    this only from the requested structure type and singular stadium wording.
    Event questions are intentionally excluded: singular ``what year`` can
    still request every championship event.
    """
    if ontology is None or any(
        str(operator.get("type", "")).upper() in {"ARGMAX", "ARGMIN"}
        for operator in graph.operators
    ):
        return None
    question = str(graph.provenance.get("original_question", ""))
    answer_types = {
        str(ontology.range_for_relation(relation_id))
        for _, relation_id, object_ in graph.triples
        if str(object_) == graph.answer_var
    }
    inherited_types = {
        inherited
        for type_id in answer_types
        if type_id
        for inherited in ontology.supertypes(type_id)
    }
    relation_id = ""
    reason = ""
    if (
        "architecture.structure" in inherited_types
        and re.search(r"\bstadium\b", question, re.I)
        and not re.search(r"\bstadiums\b", question, re.I)
    ):
        relation_id = "architecture.structure.opened"
        reason = "singular_current_stadium"
    if not relation_id or relation_id not in ontology.relation_ids:
        return None
    operator = {
        "type": "ARGMAX",
        "inputs": [],
        "input_var": graph.answer_var,
        "attribute_relation_label": relation_label_from_id(relation_id),
        "attribute_relation_labels": [],
        "value": "",
        "value_type": "",
        "attribute_relation_id": relation_id,
        "_order_temporal_lexical": True,
        "_implicit_singular_answer_order": True,
    }
    graph.operators.append(operator)
    graph.sparql = ""
    repair = {
        "input_var": graph.answer_var,
        "relation_id": relation_id,
        "reason": reason,
    }
    graph.provenance["implicit_singular_answer_order"] = deepcopy(repair)
    return repair


_PRESENT_TIME_CUE_RE = re.compile(
    r"\b(?:current|currently|now|at\s+present|present[\s-]+day)\b",
    re.I,
)
_TEMPORAL_SCHEMA_LEAF_TOKENS = {
    "date",
    "datetime",
    "end",
    "from",
    "start",
    "time",
    "to",
    "year",
}
_NON_SCALAR_OPERATOR_CUE_RE = re.compile(
    r"\b(?:how\s+many|number\s+of|count(?:ed|s|ing)?|current(?:ly)?|now|"
    r"at\s+present|present[\s-]+day|during|while|when|as\s+of|before|after|"
    r"other\s+than|except|besides|same\s+as)\b",
    re.I,
)


def _question_has_substantive_operator_intent(question: str) -> bool:
    """Recognize explicit constraints that an operator-free repair must keep."""

    return bool(
        _missing_constraint_spec(question) is not None
        or _NON_SCALAR_OPERATOR_CUE_RE.search(str(question))
    )


def _grounded_endpoint_types_are_distinct(
    candidate: GroundedSemanticCandidate,
    ontology: Any,
) -> bool:
    """Return whether every retained path changes declared endpoint type."""

    if ontology is None:
        return False
    paths = candidate.compose_input.get("semantic_paths", [])
    if not isinstance(paths, list) or not paths:
        return False
    for path in paths:
        steps = path.get("steps", []) if isinstance(path, dict) else []
        if not isinstance(steps, list) or not steps:
            return False
        endpoint_types: list[str] = []
        for endpoint_index, step in enumerate((steps[0], steps[-1])):
            relation_id = str(
                candidate.relation_bindings.get(str(step.get("id", "")), "")
            )
            direction = str(step.get("direction", ""))
            if not relation_id or direction not in {"forward", "backward"}:
                return False
            domain = str(ontology.domain_for_relation(relation_id)).strip()
            range_id = str(ontology.range_for_relation(relation_id)).strip()
            if not domain or not range_id:
                return False
            if endpoint_index == 0:
                endpoint_types.append(domain if direction == "forward" else range_id)
            else:
                endpoint_types.append(range_id if direction == "forward" else domain)
        if endpoint_types[0] == endpoint_types[1]:
            return False
    return True


def _partial_path_answer_type_hints(
    ground_diagnostics: dict[str, Any] | None,
) -> dict[int, str]:
    """Read type hints only from path anchors that grounding could not link.

    The initial grounding diagnostics are authoritative: a multi-path graph
    must contain both a path whose anchor had entity candidates and a path
    whose anchor had none.  This avoids guessing linkability again from text
    and keeps the hint confined to the structural failure lane.
    """

    hints: dict[int, str] = {}
    graph_diagnostics = (
        ground_diagnostics.get("semantic_graphs", [])
        if isinstance(ground_diagnostics, dict)
        else []
    )
    for fallback_index, graph in enumerate(graph_diagnostics):
        if not isinstance(graph, dict):
            continue
        anchors = {
            str(anchor.get("anchor_id", "")): anchor
            for anchor in graph.get("anchors", [])
            if isinstance(anchor, dict)
            and str(anchor.get("anchor_id", "")).strip()
        }
        path_anchor_ids = list(
            dict.fromkeys(
                str(path.get("anchor_id", ""))
                for path in graph.get("paths", [])
                if isinstance(path, dict)
                and str(path.get("anchor_id", "")).strip()
            )
        )
        if len(path_anchor_ids) < 2:
            continue
        grounded = [
            anchor_id
            for anchor_id in path_anchor_ids
            if isinstance(anchors.get(anchor_id, {}).get("candidates"), list)
            and bool(anchors[anchor_id]["candidates"])
        ]
        ungrounded = [
            anchor_id
            for anchor_id in path_anchor_ids
            if anchor_id in anchors
            and isinstance(anchors[anchor_id].get("candidates"), list)
            and not anchors[anchor_id]["candidates"]
        ]
        if not grounded or not ungrounded:
            continue
        surfaces: list[str] = []
        seen_surfaces: set[str] = set()
        for anchor_id in ungrounded:
            surface = " ".join(str(anchors[anchor_id].get("surface", "")).split())
            normalized = surface.casefold()
            if not surface or normalized in seen_surfaces:
                continue
            seen_surfaces.add(normalized)
            surfaces.append(surface)
        if not surfaces:
            continue
        try:
            graph_index = int(graph.get("graph_index", fallback_index))
        except (TypeError, ValueError):
            graph_index = fallback_index
        hints[graph_index] = " ; ".join(surfaces)
    return hints


def _present_time_repaired_type_return_cycle(
    question: str,
    candidate: GroundedSemanticCandidate,
    ontology: Any,
) -> tuple[bool, dict[str, Any]]:
    """Reject repaired two-hop loops that cannot express a current-time request.

    This is a pre-Compose retrieval guard. It relies only on a normalized
    temporal cue, the grounded path, and ontology domain/range types; it never
    examines entity ids, answer ids, or evaluation data.
    """

    evidence: dict[str, Any] = {
        "status": "retained",
        "reason": "not_applicable",
    }
    if not _PRESENT_TIME_CUE_RE.search(str(question)):
        evidence["reason"] = "no_present_time_cue"
        return False, evidence
    paths = candidate.compose_input.get("semantic_paths", [])
    if not isinstance(paths, list) or len(paths) != 1:
        evidence["reason"] = "not_single_path"
        return False, evidence
    path = paths[0]
    steps = path.get("steps", []) if isinstance(path, dict) else []
    if not isinstance(steps, list) or len(steps) != 2:
        evidence["reason"] = "not_two_hop"
        return False, evidence
    if (
        str(steps[0].get("to", "")) != str(steps[1].get("from", ""))
        or str(path.get("path_output_var", "")) != str(steps[1].get("to", ""))
    ):
        evidence["reason"] = "not_a_contiguous_path"
        return False, evidence
    repaired = any(
        bool(hop.get("direction_repaired"))
        for path_hops in candidate.provenance.get("hops", [])
        if isinstance(path_hops, list)
        for hop in path_hops
        if isinstance(hop, dict)
    )
    if not repaired:
        evidence["reason"] = "no_direction_repair"
        return False, evidence
    if ontology is None:
        evidence["reason"] = "ontology_unavailable"
        return False, evidence

    relations: list[str] = []
    directions: list[str] = []
    type_edges: list[tuple[str, str]] = []
    for step in steps:
        step_id = str(step.get("id", ""))
        relation_id = str(candidate.relation_bindings.get(step_id, ""))
        direction = str(step.get("direction", ""))
        if not relation_id or direction not in {"forward", "backward"}:
            evidence["reason"] = "missing_grounded_edge"
            return False, evidence
        domain = str(ontology.domain_for_relation(relation_id)).strip()
        range_id = str(ontology.range_for_relation(relation_id)).strip()
        if not domain or not range_id:
            evidence["reason"] = "missing_schema_type"
            return False, evidence
        leaf_tokens = {
            token
            for token in relation_id.rsplit(".", 1)[-1].casefold().split("_")
            if token
        }
        if (
            "type.datetime" in {domain, range_id}
            or leaf_tokens & _TEMPORAL_SCHEMA_LEAF_TOKENS
        ):
            evidence["reason"] = "temporal_edge_present"
            return False, evidence
        relations.append(relation_id)
        directions.append(direction)
        type_edges.append(
            (domain, range_id) if direction == "forward" else (range_id, domain)
        )

    start_type, middle_type = type_edges[0]
    second_source, terminal_type = type_edges[1]
    if middle_type != second_source:
        evidence["reason"] = "schema_chain_discontinuous"
        return False, evidence
    if start_type != terminal_type or start_type == middle_type:
        evidence["reason"] = "not_a_type_return_cycle"
        return False, evidence
    return True, {
        "status": "rejected",
        "reason": "present_time_repaired_type_return_cycle",
        "relation_ids": relations,
        "directions": directions,
        "type_chain": [start_type, middle_type, terminal_type],
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
    }


def _lexically_requested_object_type_return_cycle(
    candidate: GroundedSemanticCandidate,
    ontology: Any,
) -> tuple[bool, dict[str, Any]]:
    """Reject a repaired subject-return when the question names the object role."""

    evidence: dict[str, Any] = {
        "status": "retained",
        "reason": "not_applicable",
    }
    if ontology is None or candidate.provenance.get("dropped_paths"):
        evidence["reason"] = (
            "ontology_unavailable"
            if ontology is None
            else "candidate_has_dropped_paths"
        )
        return False, evidence
    paths = candidate.compose_input.get("semantic_paths", [])
    path_hops = candidate.provenance.get("hops", [])
    if (
        not isinstance(paths, list)
        or len(paths) != 1
        or not isinstance(path_hops, list)
        or len(path_hops) != 1
        or not isinstance(path_hops[0], list)
    ):
        evidence["reason"] = "not_a_complete_single_path"
        return False, evidence
    path = paths[0]
    steps = path.get("steps", []) if isinstance(path, dict) else []
    if not isinstance(steps, list) or len(steps) != 2 or len(path_hops[0]) < 2:
        evidence["reason"] = "not_a_two_hop_path"
        return False, evidence
    terminal_hop = path_hops[0][-1]
    if (
        not isinstance(terminal_hop, dict)
        or not terminal_hop.get("direction_repaired")
        or str(terminal_hop.get("predicted_direction", "")) != "forward"
        or str(steps[-1].get("direction", "")) != "backward"
    ):
        evidence["reason"] = "not_a_forward_to_backward_terminal_repair"
        return False, evidence

    relation_ids: list[str] = []
    type_edges: list[tuple[str, str]] = []
    for step in steps:
        relation_id = str(
            candidate.relation_bindings.get(str(step.get("id", "")), "")
        )
        direction = str(step.get("direction", ""))
        if not relation_id or direction not in {"forward", "backward"}:
            evidence["reason"] = "missing_grounded_edge"
            return False, evidence
        domain = str(ontology.domain_for_relation(relation_id)).strip()
        range_id = str(ontology.range_for_relation(relation_id)).strip()
        if not domain or not range_id:
            evidence["reason"] = "missing_schema_type"
            return False, evidence
        relation_ids.append(relation_id)
        type_edges.append(
            (domain, range_id) if direction == "forward" else (range_id, domain)
        )
    start_type, middle_type = type_edges[0]
    second_input, terminal_type = type_edges[1]
    if (
        middle_type != second_input
        or start_type != terminal_type
        or start_type == middle_type
    ):
        evidence["reason"] = "not_a_type_return_cycle"
        return False, evidence

    ignored = {"a", "an", "are", "is", "of", "the", "was", "were", "what", "which"}
    focus_tokens = [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            _answer_focus_text(
                str(candidate.compose_input.get("question", ""))
            ).casefold(),
        )
        if token not in ignored
    ]
    relation_tokens = [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            relation_ids[-1].rsplit(".", 1)[-1].replace("_", " ").casefold(),
        )
        if token
    ]
    if (
        not relation_tokens
        or focus_tokens[: len(relation_tokens)] != relation_tokens
    ):
        evidence["reason"] = "terminal_relation_not_requested_as_answer_head"
        return False, evidence
    return True, {
        "status": "rejected",
        "reason": "lexically_requested_object_type_return_cycle",
        "relation_ids": relation_ids,
        "type_chain": [start_type, middle_type, terminal_type],
        "answer_focus_tokens": focus_tokens,
        "terminal_relation_tokens": relation_tokens,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
    }


def _direction_repaired_terminal_cvt_cross_path(
    candidate: GroundedSemanticCandidate,
    ontology: Any,
) -> tuple[bool, dict[str, Any]]:
    """Reject complete paths whose terminal answer types cannot intersect.

    A direction-repaired terminal that resolves to a mediator/CVT cannot be
    the same answer node as the ordinary named-entity terminal of another
    complete path.  The proof uses only ontology types and grounding metadata
    and therefore belongs before Compose generation.
    """

    evidence: dict[str, Any] = {
        "status": "retained",
        "reason": "not_applicable",
    }
    if ontology is None:
        evidence["reason"] = "ontology_unavailable"
        return False, evidence
    if candidate.provenance.get("dropped_paths"):
        evidence["reason"] = "candidate_has_dropped_paths"
        return False, evidence
    paths = candidate.compose_input.get("semantic_paths", [])
    path_hops = candidate.provenance.get("hops", [])
    if (
        not isinstance(paths, list)
        or len(paths) < 2
        or not isinstance(path_hops, list)
        or len(path_hops) != len(paths)
    ):
        evidence["reason"] = "not_a_complete_multi_path_candidate"
        return False, evidence

    terminals: list[dict[str, Any]] = []
    for path_index, (path, hops) in enumerate(zip(paths, path_hops)):
        steps = path.get("steps", []) if isinstance(path, dict) else []
        if not isinstance(steps, list) or not steps or not isinstance(hops, list):
            evidence["reason"] = "missing_terminal_grounding"
            return False, evidence
        final_step = steps[-1]
        relation_id = str(
            candidate.relation_bindings.get(str(final_step.get("id", "")), "")
        )
        direction = str(final_step.get("direction", ""))
        if not relation_id or direction not in {"forward", "backward"}:
            evidence["reason"] = "missing_terminal_schema_edge"
            return False, evidence
        domain = str(ontology.domain_for_relation(relation_id)).strip()
        range_id = str(ontology.range_for_relation(relation_id)).strip()
        terminal_type = range_id if direction == "forward" else domain
        if not terminal_type:
            evidence["reason"] = "missing_terminal_type"
            return False, evidence
        supertypes = set(map(str, ontology.supertypes(terminal_type)))
        repaired_terminal = bool(
            hops
            and isinstance(hops[-1], dict)
            and hops[-1].get("direction_repaired")
        )
        if terminal_type == "type.type":
            terminal_kind = "metatype"
        elif terminal_type.startswith("type."):
            terminal_kind = "literal"
        elif "common.topic" in supertypes:
            terminal_kind = "named_entity"
        else:
            terminal_kind = "mediator"
        terminals.append(
            {
                "path_index": path_index,
                "path_id": str(path.get("id", "")),
                "relation_id": relation_id,
                "direction": direction,
                "terminal_type": terminal_type,
                "terminal_kind": terminal_kind,
                "direction_repaired": repaired_terminal,
            }
        )

    repaired_mediators = [
        item
        for item in terminals
        if item["direction_repaired"] and item["terminal_kind"] == "mediator"
    ]
    repaired_hard_kind_conflicts = [
        item
        for item in terminals
        if item["direction_repaired"]
        and item["terminal_kind"] in {"literal", "metatype"}
    ]
    named_entities = [
        item for item in terminals if item["terminal_kind"] == "named_entity"
    ]
    if repaired_hard_kind_conflicts and named_entities:
        return True, {
            "status": "rejected",
            "reason": "direction_repaired_terminal_kind_cross_path",
            "repaired_incompatible_paths": repaired_hard_kind_conflicts,
            "named_entity_paths": named_entities,
            "additional_model_calls": 0,
            "additional_endpoint_queries": 0,
        }
    if not repaired_mediators or not named_entities:
        evidence["reason"] = "terminal_types_can_intersect"
        return False, evidence
    return True, {
        "status": "rejected",
        "reason": "direction_repaired_terminal_cvt_cross_path",
        "repaired_mediator_paths": repaired_mediators,
        "named_entity_paths": named_entities,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
    }


def _direction_repair_lowers_terminal_consistency(
    candidate: GroundedSemanticCandidate,
    ontology: Any,
) -> tuple[bool, dict[str, Any]]:
    """Reject a terminal direction repair that breaks a previously valid join."""

    evidence: dict[str, Any] = {
        "status": "retained",
        "reason": "not_applicable",
    }
    if ontology is None or candidate.provenance.get("dropped_paths"):
        evidence["reason"] = (
            "ontology_unavailable"
            if ontology is None
            else "candidate_has_dropped_paths"
        )
        return False, evidence
    paths = candidate.compose_input.get("semantic_paths", [])
    path_hops = candidate.provenance.get("hops", [])
    if (
        not isinstance(paths, list)
        or len(paths) < 2
        or not isinstance(path_hops, list)
        or len(path_hops) != len(paths)
    ):
        evidence["reason"] = "not_a_complete_multi_path_candidate"
        return False, evidence

    terminals: list[dict[str, Any]] = []
    for path_index, (path, hops) in enumerate(zip(paths, path_hops)):
        steps = path.get("steps", []) if isinstance(path, dict) else []
        if not isinstance(steps, list) or not steps or not isinstance(hops, list) or not hops:
            evidence["reason"] = "missing_terminal_grounding"
            return False, evidence
        step = steps[-1]
        hop = hops[-1]
        if not isinstance(hop, dict):
            evidence["reason"] = "missing_terminal_grounding"
            return False, evidence
        relation_id = str(
            candidate.relation_bindings.get(str(step.get("id", "")), "")
        )
        direction = str(step.get("direction", ""))
        if not relation_id or direction not in {"forward", "backward"}:
            evidence["reason"] = "missing_terminal_schema_edge"
            return False, evidence
        domain = str(ontology.domain_for_relation(relation_id)).strip()
        range_id = str(ontology.range_for_relation(relation_id)).strip()
        if not domain or not range_id:
            evidence["reason"] = "missing_terminal_type"
            return False, evidence
        actual_type = range_id if direction == "forward" else domain
        predicted_direction = str(hop.get("predicted_direction", ""))
        predicted_type = ""
        if (
            hop.get("direction_repaired")
            and predicted_direction in {"forward", "backward"}
            and predicted_direction != direction
        ):
            predicted_type = (
                range_id if predicted_direction == "forward" else domain
            )
        terminals.append(
            {
                "path_index": path_index,
                "path_id": str(path.get("id", "")),
                "relation_id": relation_id,
                "direction": direction,
                "actual_type": actual_type,
                "predicted_direction": predicted_direction,
                "predicted_type": predicted_type,
            }
        )

    def compatible(left: str, right: str) -> bool:
        left_closure = {left, *map(str, ontology.supertypes(left))}
        right_closure = {right, *map(str, ontology.supertypes(right))}
        return bool(
            left == right
            or left in right_closure
            or right in left_closure
        )

    for repaired in terminals:
        predicted_type = str(repaired["predicted_type"])
        if not predicted_type:
            continue
        other_types = [
            str(item["actual_type"])
            for item in terminals
            if item is not repaired
        ]
        consistency_losses = [
            value
            for value in other_types
            if compatible(predicted_type, value)
            and not compatible(str(repaired["actual_type"]), value)
        ]
        if not consistency_losses:
            continue
        return True, {
            "status": "rejected",
            "reason": "direction_repair_lowers_terminal_consistency",
            "repaired_terminal": repaired,
            "other_terminal_types": other_types,
            "predicted_compatible_but_actual_incompatible_types": (
                consistency_losses
            ),
            "additional_model_calls": 0,
            "additional_endpoint_queries": 0,
        }
    evidence["reason"] = "terminal_consistency_not_reduced"
    return False, evidence


def _direction_repaired_terminal_cvt_incomplete_path(
    candidate: GroundedSemanticCandidate,
    ontology: Any,
) -> tuple[bool, dict[str, Any]]:
    """Reject an incomplete single path ending at a repaired mediator role."""

    evidence: dict[str, Any] = {
        "status": "retained",
        "reason": "not_applicable",
    }
    if ontology is None:
        evidence["reason"] = "ontology_unavailable"
        return False, evidence
    if not candidate.provenance.get("dropped_paths"):
        evidence["reason"] = "candidate_is_complete"
        return False, evidence
    paths = candidate.compose_input.get("semantic_paths", [])
    path_hops = candidate.provenance.get("hops", [])
    if (
        not isinstance(paths, list)
        or len(paths) != 1
        or not isinstance(path_hops, list)
        or len(path_hops) != 1
        or not isinstance(path_hops[0], list)
        or not path_hops[0]
    ):
        evidence["reason"] = "not_an_incomplete_single_path"
        return False, evidence
    path = paths[0]
    steps = path.get("steps", []) if isinstance(path, dict) else []
    if not isinstance(steps, list) or not steps:
        evidence["reason"] = "missing_terminal_grounding"
        return False, evidence
    terminal_hop = path_hops[0][-1]
    if not isinstance(terminal_hop, dict) or not terminal_hop.get(
        "direction_repaired"
    ):
        evidence["reason"] = "terminal_direction_not_repaired"
        return False, evidence
    terminal_step = steps[-1]
    relation_id = str(
        candidate.relation_bindings.get(str(terminal_step.get("id", "")), "")
    )
    direction = str(terminal_step.get("direction", ""))
    if not relation_id or direction not in {"forward", "backward"}:
        evidence["reason"] = "missing_terminal_schema_edge"
        return False, evidence
    terminal_type = str(
        ontology.range_for_relation(relation_id)
        if direction == "forward"
        else ontology.domain_for_relation(relation_id)
    ).strip()
    if not terminal_type:
        evidence["reason"] = "missing_terminal_type"
        return False, evidence
    supertypes = set(map(str, ontology.supertypes(terminal_type)))
    if terminal_type.startswith("type.") or "common.topic" in supertypes:
        evidence["reason"] = "terminal_is_not_mediator"
        return False, evidence
    return True, {
        "status": "rejected",
        "reason": "direction_repaired_terminal_cvt_incomplete_path",
        "relation_id": relation_id,
        "direction": direction,
        "terminal_type": terminal_type,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
    }


def _pure_ontology_expansion_schema_discontinuity(
    candidate: GroundedSemanticCandidate,
    ontology: Any,
) -> tuple[bool, dict[str, Any]]:
    """Reject a pure ontology expansion with a proven adjacent type break."""

    evidence: dict[str, Any] = {
        "status": "retained",
        "reason": "not_applicable",
    }
    if ontology is None:
        evidence["reason"] = "ontology_unavailable"
        return False, evidence
    paths = candidate.compose_input.get("semantic_paths", [])
    path_hops = candidate.provenance.get("hops", [])
    if (
        not isinstance(paths, list)
        or not paths
        or not isinstance(path_hops, list)
        or len(path_hops) != len(paths)
    ):
        evidence["reason"] = "missing_path_grounding"
        return False, evidence

    typed_paths: list[list[tuple[str, str, str, str]]] = []
    for path, hops in zip(paths, path_hops):
        steps = path.get("steps", []) if isinstance(path, dict) else []
        if (
            not isinstance(steps, list)
            or not steps
            or not isinstance(hops, list)
            or len(hops) != len(steps)
            or not all(
                isinstance(hop, dict)
                and hop.get("ontology_constrained") is True
                and str(hop.get("retrieval_mode", ""))
                == "ontology_expansion"
                for hop in hops
            )
        ):
            evidence["reason"] = "not_pure_ontology_expansion"
            return False, evidence
        typed_steps: list[tuple[str, str, str, str]] = []
        for step in steps:
            relation_id = str(
                candidate.relation_bindings.get(str(step.get("id", "")), "")
            )
            direction = str(step.get("direction", ""))
            if not relation_id or direction not in {"forward", "backward"}:
                evidence["reason"] = "missing_schema_edge"
                return False, evidence
            domain = str(ontology.domain_for_relation(relation_id)).strip()
            range_id = str(ontology.range_for_relation(relation_id)).strip()
            if not domain or not range_id:
                evidence["reason"] = "missing_schema_type"
                return False, evidence
            input_type, output_type = (
                (domain, range_id)
                if direction == "forward"
                else (range_id, domain)
            )
            typed_steps.append(
                (relation_id, direction, input_type, output_type)
            )
        typed_paths.append(typed_steps)

    for path_index, typed_steps in enumerate(typed_paths):
        for edge_index, (left, right) in enumerate(
            zip(typed_steps, typed_steps[1:])
        ):
            left_output = left[3]
            right_input = right[2]
            left_closure = {
                left_output,
                *map(str, ontology.supertypes(left_output)),
            }
            right_closure = {
                right_input,
                *map(str, ontology.supertypes(right_input)),
            }
            universal_entity_types = {"common.topic", "type.object"}
            if (
                left_output == right_input
                or left_output in right_closure
                or right_input in left_closure
                or left_output in universal_entity_types
                or right_input in universal_entity_types
            ):
                continue
            return True, {
                "status": "rejected",
                "reason": "pure_ontology_expansion_schema_discontinuity",
                "path_index": path_index,
                "edge_index": edge_index,
                "left_relation_id": left[0],
                "right_relation_id": right[0],
                "left_output_type": left_output,
                "right_input_type": right_input,
                "additional_model_calls": 0,
                "additional_endpoint_queries": 0,
            }
    evidence["reason"] = "schema_chain_continuous"
    return False, evidence


class PipelineExecutionError(RuntimeError):
    """Expose partial stage traces while preserving fail-fast behavior."""

    def __init__(self, cause: Exception, traces: list[dict[str, Any]]) -> None:
        super().__init__(str(cause))
        self.traces = traces


def _resolve_operator_instruction(
    training_instruction: str,
    prompt_mode: str,
    examples: list[dict[str, Any]] | None = None,
) -> str:
    normalized_mode = str(prompt_mode).casefold()
    if normalized_mode == "glm_zero_shot":
        return OPERATOR_INSTRUCTION_GLM
    if normalized_mode == "glm_few_shot":
        blocks = [
            OPERATOR_INSTRUCTION_GLM,
            "# Few-shot examples extracted from the configured Operator training set",
            (
                "The examples demonstrate this general rule: when a graph can "
                "return the question's topic entity itself but the requested "
                "answer is its counterpart or another related entity, emit "
                "NO_EQUAL on answer_var with the topic entity surface."
            ),
        ]
        for index, example in enumerate(examples or [], start=1):
            input_object = example.get("input")
            output_object = example.get("output")
            if not isinstance(input_object, dict) or not isinstance(output_object, dict):
                continue
            operator_type = str(example.get("operator_type", "UNKNOWN"))
            example_role = str(example.get("example_role", "")).strip()
            example_title = (
                f"{operator_type} / {example_role}"
                if example_role
                else operator_type
            )
            question = str(example.get("question", input_object.get("question", "")))
            semantic_input = example.get("semantic_input", {})
            semantic_graph = example.get("semantic_graph", {})
            compose_graph = {
                key: value
                for key, value in input_object.items()
                if key != "question"
            }
            blocks.append(
                "\n".join(
                    [
                        f"## Example {index}: {example_title}",
                        "Training question:",
                        question,
                        "Semantic stage input:",
                        json.dumps(semantic_input, ensure_ascii=False, indent=2),
                        "Gold semantic graph:",
                        json.dumps(semantic_graph, ensure_ascii=False, indent=2),
                        "Gold composed graph:",
                        json.dumps(compose_graph, ensure_ascii=False, indent=2),
                        "Gold operator output:",
                        json.dumps(output_object, ensure_ascii=False, indent=2),
                    ]
                )
            )
        return "\n\n".join(blocks)
    return training_instruction or OPERATOR_INSTRUCTION_V2


def _operator_input_payload(
    question: str,
    compose_input: dict[str, Any],
    compose_output: dict[str, Any],
    *,
    include_semantic_graph: bool = True,
) -> dict[str, Any]:
    payload = {"question": question, **deepcopy(compose_output)}
    # The local Operator LoRA was trained on exactly question + composed
    # graph.  GLM few-shot mode intentionally receives the extra semantic graph
    # context; keeping that context opt-in avoids a train/inference schema
    # mismatch for the local adapter.
    if include_semantic_graph:
        payload["semantic_graph"] = {
            "anchors": deepcopy(compose_input.get("anchors", [])),
            "semantic_paths": deepcopy(compose_input.get("semantic_paths", [])),
        }
    return payload


class SemanticGuidedPipeline:
    def __init__(
        self,
        *,
        decompositions: DecompositionStore,
        contract: TrainingContract,
        semantic_model: ChatClient,
        compose_model: ChatClient,
        selector_model: ChatClient | None,
        grounder: SemanticGrounder,
        knowledge_graph: KnowledgeGraph,
        operator_model: ChatClient | None = None,
        validator_model: ChatClient | None = None,
        decomposition_review_model: ChatClient | None = None,
        decomposition_review_workflow: str = "verified",
        decomposition_prompt_family: str = "webqsp",
        decomposition_prompt_profile: dict[str, str] | None = None,
        decomposition_review_max_rewrites: int = 2,
        decomposition_review_output_attempts: int = 2,
        decomposition_confirm_before_rewrite: bool = False,
        decomposition_preserve_original_on_failure: bool = False,
        rewrite_execution_fallback: bool = False,
        relation_relaxed_repair_enabled: bool = True,
        selector_answer_preview: int = _SELECTOR_ANSWER_PREVIEW,
        selector_max_prompt_bytes: int = 0,
        validation_on_valid: bool = True,
        validation_attempts: int = 1,
        validation_stages: tuple[str, ...] | list[str] | None = None,
        semantic_max_hops: int = 2,
        operator_prompt_mode: str = "training",
        version: str = "0.2.0",
        semantic_beam: int = 2,
        compose_per_semantic: int = 2,
        operator_top_k: int = 0,
        query_graph_beam: int = 20,
        execution_budget: int = 12,
        answer_limit: int = 0,
        compose_workers: int = 1,
        operator_workers: int = 1,
        require_complete_semantic_graph: bool = False,
        require_semantic_item_coverage: bool = False,
        retain_original_with_rewrite: bool = False,
        allow_semantic_template_recovery: bool = True,
        allow_path_prefix_recovery: bool = True,
        semantic_projection_mode: str = "legacy",
        relation_relaxed_max_hops: int = 2,
        failure_endpoint_retrieval_enabled: bool = False,
        failure_endpoint_template_top_k: int = 24,
        failure_endpoint_graph_budget: int = 80,
        failure_endpoint_return_bounded_direct_best_effort: bool = False,
        failure_endpoint_best_effort_answer_limit: int = 64,
        missing_relation_retrieval_enabled: bool = False,
        missing_relation_successful_enabled: bool = False,
        missing_relation_template_top_k: int = 64,
        missing_relation_candidate_limit: int = 3,
        missing_relation_return_bounded_best_effort: bool = False,
        missing_relation_best_effort_answer_limit: int = 64,
        extrema_relation_retrieval_enabled: bool = False,
        extrema_relation_template_top_k: int = 64,
        extrema_relation_source_graph_limit: int = 3,
        extrema_relation_candidate_limit: int = 3,
        failure_gold_sparql_enabled: bool = False,
        failure_gold_sparql_cache_path: str = "",
        failure_gold_sparql_top_k: int = 32,
        failure_gold_sparql_candidate_limit: int = 3,
        failure_gold_sparql_adaptive_semantic_enabled: bool = False,
        failure_gold_sparql_adaptive_candidate_limit: int = 3,
        temporal_interval_retrieval_enabled: bool = False,
        temporal_interval_candidate_limit: int = 3,
        template_consensus_selector: SourceVerifiedTemplateConsensusSelector | None = None,
        path_alignment_selector: PathAlignmentSelector | None = None,
        train_gold_path_extrema_selector: TrainGoldPathExtremaSelector | None = None,
        numeric_temporal_slot_normalization_enabled: bool = False,
        numeric_temporal_slot_max_replacements: int = 3,
        entity_kind_fixed_beam_planner: EntityKindFixedBeamPlanner | None = None,
        entity_kind_fixed_beam_max_replacements: int = 3,
        underanswer_expansion_selector: UnderanswerExpansionSelector | None = None,
        failure_factorized_path_enabled: bool = False,
        failure_factorized_path_cache_path: str = "",
        failure_factorized_path_top_k: int = 6,
        failure_factorized_path_combination_beam: int = 5,
        failure_factorized_path_candidate_limit: int = 3,
    ) -> None:
        self.decompositions = decompositions
        self.contract = contract
        self.semantic_model = semantic_model
        self.compose_model = compose_model
        self.selector_model = selector_model
        self.operator_model = operator_model
        self.validator_model = validator_model
        self.relative_temporal_binder = RelativeTemporalBinder(knowledge_graph, getattr(grounder, "ontology", None))
        self.decomposition_reviewer = (
            DecompositionReviewer(
                decomposition_review_model,
                max_rewrites=decomposition_review_max_rewrites,
                output_attempts=decomposition_review_output_attempts,
                confirm_before_rewrite=decomposition_confirm_before_rewrite,
                workflow=decomposition_review_workflow,
                prompt_family=decomposition_prompt_family,
                prompt_profile=decomposition_prompt_profile,
            )
            if decomposition_review_model is not None else None
        )
        self.decomposition_preserve_original_on_failure = bool(decomposition_preserve_original_on_failure)
        self.rewrite_execution_fallback = bool(rewrite_execution_fallback)
        self.relation_relaxed_repair_enabled = bool(
            relation_relaxed_repair_enabled
        )
        # GoldEntityStore exposes only topic/constraint entities, never answers.
        # A Gold-enabled CWQ run therefore uses one entity-aware review prompt for
        # every question; no metric-dependent routing or question classifier exists.
        self.use_topic_context = bool(getattr(grounder, "gold_entities", {}))
        self._question_context_cache = {}
        self.selector_answer_preview = max(1, int(selector_answer_preview))
        self.selector_max_prompt_bytes = max(0, int(selector_max_prompt_bytes))
        self.validation_stages = frozenset(
            str(stage).strip().casefold()
            for stage in (
                validation_stages
                if validation_stages is not None
                else ("semantic", "compose", "operator")
            )
        )
        self.output_corrector = GLMOutputCorrector(
            validator_model,
            validate_valid=validation_on_valid,
            max_attempts=validation_attempts,
        )
        self.version = str(version)
        self.semantic_max_hops = max(1, int(semantic_max_hops))
        self.operator_prompt_mode = str(operator_prompt_mode).casefold()
        self.grounder = grounder
        self.kg = knowledge_graph
        # Terminal, model-free retrieval modules receive the pipeline as their
        # context.  Keep the same ontology object available directly and via
        # the grounder; no additional resource is loaded here.
        self.ontology = getattr(grounder, "ontology", None)
        self.semantic_beam = max(1, semantic_beam)
        self.compose_per_semantic = max(1, compose_per_semantic)
        self.operator_top_k = max(0, int(operator_top_k))
        self.query_graph_beam = max(1, query_graph_beam)
        self.execution_budget = max(1, execution_budget)
        self.answer_limit = None if int(answer_limit) <= 0 else int(answer_limit)
        self.compose_workers = max(1, int(compose_workers))
        self.operator_workers = max(1, int(operator_workers))
        self.require_complete_semantic_graph = bool(require_complete_semantic_graph)
        self.require_semantic_item_coverage = bool(require_semantic_item_coverage)
        self.retain_original_with_rewrite = bool(retain_original_with_rewrite)
        self.allow_semantic_template_recovery = bool(allow_semantic_template_recovery)
        self.allow_path_prefix_recovery = bool(allow_path_prefix_recovery)
        self.semantic_projection_mode = str(semantic_projection_mode).strip().casefold()
        self.relation_relaxed_max_hops = max(1, int(relation_relaxed_max_hops))
        self.failure_endpoint_retrieval_enabled = bool(
            failure_endpoint_retrieval_enabled
        )
        self.failure_endpoint_template_top_k = min(
            64, max(1, int(failure_endpoint_template_top_k))
        )
        self.failure_endpoint_graph_budget = min(
            80, max(1, int(failure_endpoint_graph_budget))
        )
        self.failure_endpoint_return_bounded_direct_best_effort = bool(
            failure_endpoint_return_bounded_direct_best_effort
        )
        self.failure_endpoint_best_effort_answer_limit = min(
            512,
            max(1, int(failure_endpoint_best_effort_answer_limit)),
        )
        self.missing_relation_retrieval_enabled = bool(
            missing_relation_retrieval_enabled
        )
        self.missing_relation_successful_enabled = bool(
            missing_relation_successful_enabled
        )
        self.missing_relation_template_top_k = min(
            64, max(3, int(missing_relation_template_top_k))
        )
        self.missing_relation_candidate_limit = min(
            3, max(1, int(missing_relation_candidate_limit))
        )
        self.missing_relation_return_bounded_best_effort = bool(
            missing_relation_return_bounded_best_effort
        )
        self.missing_relation_best_effort_answer_limit = min(
            512, max(1, int(missing_relation_best_effort_answer_limit))
        )
        self.extrema_relation_retrieval_enabled = bool(
            extrema_relation_retrieval_enabled
        )
        self.extrema_relation_template_top_k = min(
            64, max(3, int(extrema_relation_template_top_k))
        )
        self.extrema_relation_source_graph_limit = min(
            3, max(1, int(extrema_relation_source_graph_limit))
        )
        self.extrema_relation_candidate_limit = min(
            3, max(1, int(extrema_relation_candidate_limit))
        )
        self.failure_gold_sparql_enabled = bool(failure_gold_sparql_enabled)
        self.failure_gold_sparql_cache_path = str(
            failure_gold_sparql_cache_path
        ).strip()
        self.failure_gold_sparql_top_k = min(
            64, max(3, int(failure_gold_sparql_top_k))
        )
        self.failure_gold_sparql_candidate_limit = min(
            3, max(1, int(failure_gold_sparql_candidate_limit))
        )
        self.failure_gold_sparql_adaptive_semantic_enabled = bool(
            failure_gold_sparql_adaptive_semantic_enabled
        )
        self.failure_gold_sparql_adaptive_candidate_limit = min(
            3, max(1, int(failure_gold_sparql_adaptive_candidate_limit))
        )
        self.temporal_interval_retrieval_enabled = bool(
            temporal_interval_retrieval_enabled
        )
        self.temporal_interval_candidate_limit = min(
            3, max(1, int(temporal_interval_candidate_limit))
        )
        self.template_consensus_selector = template_consensus_selector
        self.path_alignment_selector = path_alignment_selector
        self.train_gold_path_extrema_selector = train_gold_path_extrema_selector
        self.numeric_temporal_slot_normalization_enabled = bool(
            numeric_temporal_slot_normalization_enabled
        )
        self.numeric_temporal_slot_max_replacements = min(
            3,
            max(1, int(numeric_temporal_slot_max_replacements)),
        )
        self.entity_kind_fixed_beam_planner = entity_kind_fixed_beam_planner
        self.entity_kind_fixed_beam_max_replacements = min(
            3,
            max(1, int(entity_kind_fixed_beam_max_replacements)),
        )
        self.underanswer_expansion_selector = underanswer_expansion_selector
        self.failure_factorized_path_enabled = bool(
            failure_factorized_path_enabled
        )
        self.failure_factorized_path_cache_path = str(
            failure_factorized_path_cache_path
        ).strip()
        self.failure_factorized_path_top_k = min(
            16, max(1, int(failure_factorized_path_top_k))
        )
        self.failure_factorized_path_combination_beam = min(
            8, max(1, int(failure_factorized_path_combination_beam))
        )
        self.failure_factorized_path_candidate_limit = min(
            3, max(1, int(failure_factorized_path_candidate_limit))
        )
        # Built only after the first final-empty result.  Successful questions
        # do not construct the template index or acquire this lock.
        self._failure_template_retriever = None
        self._failure_template_retriever_lock = Lock()
        self._missing_relation_index = None
        self._missing_relation_index_lock = Lock()
        self._failure_gold_sparql_index = None
        self._failure_gold_sparql_index_lock = Lock()
        self._temporal_interval_documents = None
        self._failure_factorized_path_index = None
        self._failure_factorized_path_index_lock = Lock()

    def _generate_compose_batch(
        self,
        requests: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> list[tuple[list[dict[str, Any]], Exception | None]]:
        """Generate Compose candidates concurrently while preserving order.

        Each tuple contains the model outputs and an exception, if that
        request failed.  Exceptions are returned instead of raised so one
        failed candidate can use the existing deterministic fallback without
        cancelling the other in-flight requests.
        """
        def run_one(
            request: tuple[dict[str, Any], dict[str, Any]],
        ) -> tuple[list[dict[str, Any]], Exception | None]:
            payload, schema = request
            try:
                outputs = self.compose_model.generate_json(
                    instruction=self.contract.compose_instruction,
                    payload=payload,
                    schema=schema,
                    count=self.compose_per_semantic,
                )
                if not outputs:
                    raise ValueError("compose model returned no candidates")
                return outputs, None
            except Exception as exc:  # preserve per-candidate fallback behavior
                return [], exc

        if self.compose_workers <= 1 or len(requests) <= 1:
            return [run_one(request) for request in requests]
        workers = min(self.compose_workers, len(requests))
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="compose",
        ) as executor:
            return list(executor.map(run_one, requests))

    def _generate_compose_stream(
        self,
        requests: list[
            tuple[dict[str, Any], dict[str, Any]]
            | tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        ],
    ):
        """Yield Compose results as soon as each request completes.

        The yielded index is the stable input position.  Consumers may start
        dependent work immediately while still sorting traces/results by this
        index at the aggregation barrier.
        """
        def run_one(
            request: (
                tuple[dict[str, Any], dict[str, Any]]
                | tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
            ),
        ) -> tuple[list[dict[str, Any]], Exception | None]:
            payload, schema = request[:2]
            if len(request) == 3:
                return [deepcopy(request[2])], None
            try:
                outputs = self.compose_model.generate_json(
                    instruction=self.contract.compose_instruction,
                    payload=payload,
                    schema=schema,
                    count=self.compose_per_semantic,
                )
                if not outputs:
                    raise ValueError("compose model returned no candidates")
                return outputs, None
            except Exception as exc:
                return [], exc

        if self.compose_workers <= 1 or len(requests) <= 1:
            for index, request in enumerate(requests):
                yield index, run_one(request)
            return
        workers = min(self.compose_workers, len(requests))
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="compose",
        ) as executor:
            futures = {
                executor.submit(run_one, request): index
                for index, request in enumerate(requests)
            }
            for future in as_completed(futures):
                yield futures[future], future.result()

    def _validate_stage_output(
        self,
        *,
        stage: str,
        input_payload: dict[str, Any],
        candidate: Any,
        validator,
        schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Validate locally and optionally ask the configured model for correction."""
        if not self._can_correct(stage):
            return validator(candidate), None
        result = self.output_corrector.run(
            stage=stage,
            input_payload=input_payload,
            candidate=candidate,
            validator=validator,
            schema=schema,
        )
        return result.value, result.trace

    def _can_correct(self, stage: str) -> bool:
        return (
            self.validator_model is not None
            and str(stage).casefold() in self.validation_stages
        )

    def _retry_no_grounded_with_prefixes(
        self,
        *,
        question: str,
        decompositions: list[str],
        semantic_graphs: list[dict[str, Any]],
        initial_ground_diagnostics: dict[str, Any] | None = None,
    ) -> tuple[list[GroundedSemanticCandidate], dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        question_key = " ".join(str(question).casefold().split())
        linked_entities = self.grounder.gold_entities.get(question_key, {})
        collected: list[GroundedSemanticCandidate] = []

        def collect(
            *,
            strategy: str,
            graphs: list[dict[str, Any]],
            plans_for_graphs: list[dict[str, Any]],
            prefer_ontology_paths: bool = False,
            relax_relation_constraints: bool = False,
            score_penalty: float = 0.0,
            path_beam_cap: int | None = None,
            candidate_limit_cap: int | None = None,
            stepwise_answer_type_hints: dict[int, str] | None = None,
        ) -> list[GroundedSemanticCandidate]:
            if not graphs:
                return []
            with self.grounder.scoped_retrieval_limits(
                path_beam_cap=path_beam_cap,
                candidate_limit_cap=candidate_limit_cap,
            ):
                found, attempt_diagnostics = self.grounder.retrieve(
                    question=question,
                    decompositions=decompositions,
                    semantic_graphs=graphs,
                    prefer_ontology_paths=prefer_ontology_paths,
                    relax_relation_constraints=relax_relation_constraints,
                    stepwise_answer_type_hints=stepwise_answer_type_hints,
                )
            for candidate in found:
                graph_index = int(
                    candidate.provenance.get("semantic_graph_index", -1)
                )
                if not 0 <= graph_index < len(plans_for_graphs):
                    continue
                plan = deepcopy(plans_for_graphs[graph_index])
                candidate.provenance["_repair_plan"] = plan
                candidate.provenance["_repair_strategy"] = strategy
                candidate.score -= score_penalty
                if strategy == "accepted_semantic_template":
                    candidate.score += 0.25 * float(
                        plan.get("template_score", 0.0)
                    )
                collected.append(candidate)
            attempts.append(
                {
                    "strategy": strategy,
                    "candidate_count": len(found),
                    "grounding": attempt_diagnostics,
                }
            )
            return found

        full_plans = [
            {
                "source_graph_index": graph_index,
                "strategy": "relation_relaxed_full_path",
                "changed_paths": [],
            }
            for graph_index in range(len(semantic_graphs))
        ]

        # Retrieve semantically analogous accepted structures with BGE first.
        # Question-level similarity carries the requested relation intent;
        # relation/direction similarity is a secondary signal.  Both are
        # embedding scores, with no lexical overlap or regex matching.
        if self.allow_semantic_template_recovery:
            template_graphs, template_plans = build_semantic_template_repair_graphs(
                question=question,
                decompositions=decompositions,
                semantic_graphs=semantic_graphs,
                linked_entities=linked_entities,
                semantic_examples=self.contract.semantic_examples,
                ranker=self.grounder.ranker,
            )
            template_candidates = collect(
                strategy="accepted_semantic_template",
                graphs=template_graphs,
                plans_for_graphs=template_plans,
                prefer_ontology_paths=True,
            )
        else:
            template_candidates = []

        # If no semantically retrieved structure exists in the current KG,
        # rank the entity's observed local paths directly with BGE.
        relaxed_pairs = [
            (graph, plan)
            for graph, plan in zip(semantic_graphs, full_plans)
            if max(
                (
                    len(path.get("steps", []))
                    for path in graph.get("semantic_paths", [])
                    if isinstance(path, dict)
                ),
                default=0,
            ) <= self.relation_relaxed_max_hops
        ]
        relaxed_candidates = (
            collect(
                strategy="relation_relaxed_full_path",
                graphs=[item[0] for item in relaxed_pairs],
                plans_for_graphs=[item[1] for item in relaxed_pairs],
                relax_relation_constraints=True,
            )
            if self.relation_relaxed_repair_enabled and relaxed_pairs
            else []
        )

        # Do not stop at the first non-empty repair family. Rank template and
        # observed-KG candidates together with BGE. Additional observed-KG
        # candidates use deterministic Compose and skip Operator generation,
        # so retaining them does not add language-model requests.
        if template_candidates and relaxed_candidates:
            for candidate in relaxed_candidates:
                plan = candidate.provenance.get("_repair_plan", {})
                if isinstance(plan, dict):
                    plan["operator_mode"] = "deterministic_empty"
            relation_documents = []
            for candidate in collected:
                relation_documents.append(
                    " ".join(
                        " ".join(
                            [
                                str(step.get("direction", "")),
                                *(
                                    str(part)
                                    for part in step.get("relation_label", [])
                                ),
                            ]
                        )
                        for path in candidate.compose_input.get("semantic_paths", [])
                        if isinstance(path, dict)
                        for step in path.get("steps", [])
                        if isinstance(step, dict)
                    )
                )
            semantic_scores = self.grounder.ranker.score(
                " ".join([question, *decompositions]),
                relation_documents,
            )
            for candidate, semantic_score in zip(collected, semantic_scores):
                candidate.provenance["_repair_semantic_score"] = float(
                    semantic_score
                )
            collected.sort(
                key=lambda item: (
                    -float(item.provenance.get("_repair_semantic_score", 0.0)),
                    -item.score,
                )
            )

        extension_graphs: list[dict[str, Any]] = []
        extension_plans: list[dict[str, Any]] = []
        if not collected and self.relation_relaxed_repair_enabled:
            extension_graphs, extension_plans = build_structural_extension_graphs(
                semantic_graphs
            )
            source_answer_type_hints = _partial_path_answer_type_hints(
                initial_ground_diagnostics
            )
            extension_groups: dict[str, dict[str, Any]] = {}
            for extension_index, (graph, plan) in enumerate(
                zip(extension_graphs, extension_plans)
            ):
                try:
                    source_index = int(plan.get("source_graph_index", -1))
                except (TypeError, ValueError):
                    source_index = -1
                hint = source_answer_type_hints.get(source_index, "")
                path_count = (
                    len(graph.get("semantic_paths", []))
                    if isinstance(graph, dict)
                    else 0
                )
                if hint:
                    group_name = "partial_anchor"
                    beam_cap = 16
                elif path_count > 1:
                    group_name = "fully_anchored_multi_path"
                    beam_cap = 8
                else:
                    group_name = "single_path"
                    beam_cap = 16
                group = extension_groups.setdefault(
                    group_name,
                    {
                        "graphs": [],
                        "plans": [],
                        "hints": {},
                        "beam_cap": beam_cap,
                    },
                )
                local_index = len(group["graphs"])
                group["graphs"].append(graph)
                group["plans"].append(plan)
                if hint:
                    group["hints"][local_index] = hint

            # Keep caps local to one source class. A fully anchored multi-path
            # graph needs the narrow Cartesian bound, while a single active
            # path (including a graph with an unlinked secondary anchor) keeps
            # the wider beam. Mixed Semantic hypotheses must not change one
            # another's cap.
            for group in extension_groups.values():
                structural_beam_cap = int(group["beam_cap"])
                collect(
                    strategy="bounded_structural_extension",
                    graphs=group["graphs"],
                    plans_for_graphs=group["plans"],
                    relax_relation_constraints=True,
                    score_penalty=0.04,
                    # This is a failure-only search after every ordinary
                    # grounding lane returned empty. Bounding the recursive
                    # frontier prevents a handful of 4/5-hop extensions from
                    # monopolizing several minutes.
                    path_beam_cap=structural_beam_cap,
                    candidate_limit_cap=structural_beam_cap,
                    stepwise_answer_type_hints=group["hints"],
                )
        repaired_graphs, prefix_plans = build_prefix_repair_graphs(semantic_graphs)
        if not collected and self.allow_path_prefix_recovery:
            collect(
                strategy="ontology_constrained_path_prefix",
                graphs=repaired_graphs,
                plans_for_graphs=prefix_plans,
                prefer_ontology_paths=True,
                score_penalty=0.06,
            )
        if (
            not collected
            and self.allow_path_prefix_recovery
            and self.relation_relaxed_repair_enabled
        ):
            collect(
                strategy="relation_relaxed_path_prefix",
                graphs=repaired_graphs,
                plans_for_graphs=prefix_plans,
                relax_relation_constraints=True,
                score_penalty=0.08,
            )

        deduped: list[GroundedSemanticCandidate] = []
        seen_candidates: set[str] = set()
        for candidate in sorted(
            collected,
            key=lambda item: (
                -float(item.provenance.get("_repair_semantic_score", 0.0)),
                -item.score,
            ),
        ):
            key = json.dumps(
                {
                    "compose_input": candidate.compose_input,
                    "anchors": {
                        name: value.entity_id
                        for name, value in candidate.anchor_bindings.items()
                    },
                    "relations": candidate.relation_bindings,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            deduped.append(candidate)
            if len(deduped) >= self.query_graph_beam:
                break
        candidates = deduped
        if not candidates:
            return [], {
                "status": "failed",
                "strategy": "semantic_template_relation_relaxation_and_structure",
                "selected_strategy": None,
                "candidate_count": 0,
                "plans": prefix_plans,
                "attempts": attempts,
            }

        # Structural repair can retain a grounded path after another explicit
        # path was dropped.  When the question contains no substantive
        # operator intent and the retained path changes its declared endpoint
        # type, an anchor-exclusion operator is redundant.  Keep this narrow
        # failure-only lane deterministic: it avoids model requests for newly
        # explored structural candidates without changing complete repairs.
        if not _question_has_substantive_operator_intent(question):
            for candidate in candidates:
                if (
                    str(candidate.provenance.get("_repair_strategy", ""))
                    != "bounded_structural_extension"
                    or not candidate.provenance.get("dropped_paths")
                    or not _grounded_endpoint_types_are_distinct(
                        candidate,
                        getattr(self.grounder, "ontology", None),
                    )
                ):
                    continue
                plan = candidate.provenance.get("_repair_plan", {})
                if isinstance(plan, dict):
                    plan["operator_mode"] = "deterministic_empty"
                    plan["operator_reason"] = (
                        "incomplete_structural_distinct_endpoint_types"
                    )

        selected_strategy = str(
            candidates[0].provenance.get("_repair_strategy", "")
        )
        plans = [
            deepcopy(candidate.provenance.get("_repair_plan", {}))
            for candidate in candidates
        ]
        diagnostics = next(
            (
                attempt.get("grounding", {})
                for attempt in attempts
                if attempt.get("strategy") == selected_strategy
            ),
            {},
        )

        temporal_intervals = (
            discover_temporal_entity_intervals(self.kg, linked_entities)
            if candidates and linked_entities
            else []
        )
        for candidate in candidates:
            plan = deepcopy(candidate.provenance.pop("_repair_plan", {}))
            candidate_strategy = str(
                candidate.provenance.pop(
                    "_repair_strategy",
                    selected_strategy,
                )
            )
            repair_context = {
                "trigger": "NO_GROUNDED_SEMANTIC_CANDIDATE",
                "strategy": candidate_strategy,
                "plan": plan,
                "temporal_intervals": deepcopy(temporal_intervals),
                "linked_entities": deepcopy(linked_entities),
                "compose_mode": str(
                    plan.get("compose_mode", "deterministic")
                ),
            }
            changed_paths = {
                str(item.get("path_id", "")): item
                for item in plan.get("changed_paths", [])
                if isinstance(item, dict)
            }
            annotator = getattr(self.kg, "annotate_cvt_metadata", None)
            if callable(annotator):
                answer_rewrites: list[dict[str, str]] = []
                for path in candidate.compose_input.get("semantic_paths", []):
                    if not isinstance(path, dict):
                        continue
                    change = changed_paths.get(str(path.get("id", "")), {})
                    if not str(change.get("removed_goal", "")).strip():
                        continue
                    steps = [
                        item
                        for item in path.get("steps", [])
                        if isinstance(item, dict)
                    ]
                    if len(steps) < 2:
                        continue
                    final_step = steps[-1]
                    relation_id = candidate.relation_bindings.get(
                        str(final_step.get("id", "")),
                        "",
                    )
                    if not relation_id:
                        continue
                    metadata = annotator(
                        [
                            {
                                "relation_id": relation_id,
                                "direction": str(final_step.get("direction", "")),
                            }
                        ]
                    )
                    if not metadata or not bool(metadata[0].get("reaches_cvt")):
                        continue
                    previous_var = str(final_step.get("from", ""))
                    current_var = str(path.get("path_output_var", ""))
                    if not previous_var.startswith("P"):
                        continue
                    path["path_output_var"] = previous_var
                    answer_rewrites.append(
                        {
                            "path_id": str(path.get("id", "")),
                            "from": current_var,
                            "to": previous_var,
                            "reason": "trimmed_suffix_starts_from_terminal_cvt",
                            "terminal_relation_id": relation_id,
                        }
                    )
                if answer_rewrites:
                    repair_context["answer_var_rewrites"] = answer_rewrites
            candidate.provenance["grounding_repair"] = repair_context
        return candidates, {
            "status": "repaired",
            "strategy": "semantic_template_relation_relaxation_and_structure",
            "selected_strategy": selected_strategy,
            "candidate_count": len(candidates),
            "plans": plans,
            "temporal_intervals": temporal_intervals,
            "grounding": diagnostics,
            "attempts": attempts,
        }

    def _validate_operator_stage_output(
        self,
        *,
        input_payload: dict[str, Any],
        candidate: Any,
        compose_output: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if self.operator_prompt_mode in _GLM_OPERATOR_PROMPT_MODES:
            value = validate_operator_output(candidate, compose_output)
            return value, {
                "stage": "operator",
                "local_valid": True,
                "validator_requested": False,
                "status": "accepted_local_glm_generation",
            }
        return self._validate_stage_output(
            stage="operator",
            input_payload=input_payload,
            candidate=candidate,
            validator=lambda value: validate_operator_output(value, compose_output),
            schema=_operator_output_schema(compose_output),
        )

    def _run_operator_candidate(
        self,
        *,
        question: str,
        grounded_input: dict[str, Any],
        compose_output: dict[str, Any],
        semantic_rank: int,
        compose_rank: int,
        operator_candidate_rank: int,
        operator_instruction: str,
        repair_context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Predict and validate one Operator candidate.

        This method is deliberately self-contained so it can run in the
        bounded Operator worker pool while Compose requests are still in
        flight.  The caller owns final graph construction and trace ordering.
        """
        operator_question = question
        operator_relevant_goals = " ".join(
            str(path.get("goal", ""))
            for path in grounded_input.get("semantic_paths", [])
            if isinstance(path, dict)
        )
        operator_cues = (
            "latest",
            "earliest",
            "most recent",
            "smallest",
            "largest",
            "less than",
            "greater than",
            "before",
            "after",
        )
        question_lower = question.casefold()
        goal_lower = operator_relevant_goals.casefold()
        if any(cue in goal_lower and cue not in question_lower for cue in operator_cues):
            operator_question = (
                f"{question}\nDecomposition constraint: "
                f"{operator_relevant_goals}"
            )
        operator_input = _operator_input_payload(
            operator_question,
            grounded_input,
            compose_output,
            include_semantic_graph=(
                self.operator_prompt_mode in _GLM_OPERATOR_PROMPT_MODES
            ),
        )
        if self.use_topic_context and self._question_context_cache.get(question):
            operator_input["entity_context"] = self._question_context_cache[question]
        op_item: dict[str, Any] = {
            "semantic_rank": semantic_rank,
            "compose_rank": compose_rank,
            "operator_candidate_rank": operator_candidate_rank,
            "prompt_mode": self.operator_prompt_mode,
            "input": operator_input,
        }
        if self.operator_model is None:
            operator_output = {"operators": []}
            op_item.update({"status": "disabled", "output": operator_output})
            return operator_output, op_item
        repair_plan = (
            repair_context.get("plan", {})
            if isinstance(repair_context, dict)
            else {}
        )
        if (
            isinstance(repair_plan, dict)
            and repair_plan.get("operator_mode") == "deterministic_empty"
        ):
            operator_output = {"operators": []}
            op_item.update(
                {
                    "status": "repair_deterministic_operator_free",
                    "model_used": False,
                    "output": operator_output,
                }
            )
            return operator_output, op_item
        if (
            self.operator_top_k > 0
            and operator_candidate_rank > self.operator_top_k
        ):
            operator_output = {"operators": []}
            op_item.update(
                {
                    "status": "skipped_operator_top_k",
                    "operator_top_k": self.operator_top_k,
                    "output": operator_output,
                }
            )
            return operator_output, op_item

        operator_validation: dict[str, Any] | None = None
        try:
            operator_outputs = self.operator_model.generate_json(
                instruction=operator_instruction,
                payload=operator_input,
                schema=_operator_output_schema(compose_output),
                count=1,
            )
            raw_operator_output = operator_outputs[0] if operator_outputs else {}
            op_item["model_output"] = raw_operator_output
            validated_operator_output, operator_validation = (
                self._validate_operator_stage_output(
                    input_payload=operator_input,
                    candidate=raw_operator_output,
                    compose_output=compose_output,
                )
            )
            if validated_operator_output is None:
                raise ContractError("operator output rejected after local validation")
            operator_output = _normalize_operator_output(
                validated_operator_output,
                compose_output,
            )
            operator_output = _normalize_operator_intent(question, operator_output)
            operator_output = _infer_missing_extrema_operator(
                question,
                compose_output,
                operator_output,
            )
            if repair_context:
                operator_output, repair_trace = repair_temporal_operator_output(
                    question=question,
                    compose_output=compose_output,
                    operator_output=operator_output,
                    repair_context=repair_context,
                    ontology=self.grounder.ontology,
                )
                op_item["grounding_repair_postprocess"] = repair_trace
                operator_output, cvt_repair_trace = (
                    repair_terminal_cvt_operator_output(
                        compose_output=compose_output,
                        operator_output=operator_output,
                        repair_context=repair_context,
                        ontology=self.grounder.ontology,
                    )
                )
                op_item["grounding_repair_cvt_postprocess"] = cvt_repair_trace
            op_item.update({"status": "valid", "output": validated_operator_output})
            if operator_validation is not None:
                op_item["validation"] = operator_validation
            if operator_output != validated_operator_output:
                op_item["normalized_output"] = operator_output
            return operator_output, op_item
        except (ContractError, ValueError, RuntimeError) as exc:
            op_item.update({"status": "invalid", "error": str(exc)})
            fallback_enabled = (
                self._can_correct("operator")
                or self.operator_prompt_mode in _GLM_OPERATOR_PROMPT_MODES
            )
            if not fallback_enabled:
                return {}, op_item
            if operator_validation is not None:
                op_item["validation"] = operator_validation
            op_item["fallback"] = {
                "operators": [],
                "reason": (
                    "operator_local_validation_failed"
                    if self.operator_prompt_mode in _GLM_OPERATOR_PROMPT_MODES
                    else "operator_validation_failed"
                ),
            }
            return {"operators": []}, op_item

    def answer(self, question: str) -> dict[str, Any]:
        traces: list[dict[str, Any]] = []
        started = time.monotonic()
        try:
            if self.contract.compose_format == "graph_v2":
                result = self._answer_v2(question, traces)
            else:
                result = self._answer_legacy(question, traces)
        except Exception as exc:
            if self.rewrite_execution_fallback and self._decomposition_changed(traces):
                result = self._restore_original(
                    question,
                    traces,
                    f"PIPELINE_ERROR: {exc}",
                )
                return self._recover_final_empty(question, result)
            error_details = _exception_trace_details(exc)
            error_details["pipeline_elapsed_seconds"] = round(
                time.monotonic() - started,
                6,
            )
            traces.append(
                self._trace(
                    "pipeline_error",
                    {"question": question},
                    error_details,
                )
            )
            raise PipelineExecutionError(exc, traces) from exc
        if result.get("failure") and self.rewrite_execution_fallback and self._decomposition_changed(traces):
            result = self._restore_original(question, traces, result["failure"])
        return self._recover_final_empty(question, result)

    def _recover_final_empty(
        self,
        question: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Run bounded, model-free endpoint retrieval only on a final empty row."""

        if result.get("answer_ids"):
            traces = result.get("traces")
            if not isinstance(traces, list):
                traces = []
            expected_kind = _expected_answer_kind(question)
            observed_kind = _answer_value_kind(
                [str(value) for value in result.get("answer_ids", [])]
            )
            kind_guard_stage = "answer_kind_failure_retrieval"
            if (
                expected_kind
                and observed_kind != expected_kind
                and not any(
                    isinstance(item, dict)
                    and item.get("stage") == kind_guard_stage
                    for item in traces
                )
            ):
                # A high-confidence entity interrogative returning only
                # literals/type ids/URLs is operationally the same as an empty
                # graph.  Reuse the already bounded failure-only endpoint lane
                # and keep the original result if it cannot produce an answer
                # of the requested kind.  This adds no model calls and cannot
                # turn a failed repair into a new empty result.
                guarded = {
                    **result,
                    "answer_ids": [],
                    "answers": [],
                    "selected_graph": None,
                    "failure": "EMPTY_RESULT",
                    "traces": [
                        *traces,
                        self._trace(
                            kind_guard_stage,
                            {"question": question},
                            {
                                "status": "triggered",
                                "expected_kind": expected_kind,
                                "observed_kind": observed_kind,
                                "source_answer_count": len(
                                    result.get("answer_ids", [])
                                ),
                                "additional_model_calls": 0,
                            },
                        ),
                    ],
                }
                recovered = self._recover_final_empty(question, guarded)
                recovered_ids = [
                    str(value) for value in recovered.get("answer_ids", [])
                ]
                if (
                    recovered_ids
                    and _answer_value_kind(recovered_ids) == expected_kind
                ):
                    return {
                        **recovered,
                        "answer_repair": {
                            "strategy": kind_guard_stage,
                            "source_answer_ids": list(
                                map(str, result.get("answer_ids", []))
                            ),
                            "model_calls": 0,
                        },
                    }
                return {
                    **result,
                    "traces": [
                        *list(recovered.get("traces", guarded["traces"])),
                        self._trace(
                            f"{kind_guard_stage}_outcome",
                            {"question": question},
                            {
                                "status": "original_preserved",
                                "expected_kind": expected_kind,
                                "recovered_kind": (
                                    _answer_value_kind(recovered_ids)
                                    if recovered_ids
                                    else "empty"
                                ),
                                "additional_model_calls": 0,
                            },
                        ),
                    ],
                }
            extrema_result = self._recover_extrema_relation(question, result)
            if extrema_result.get("answer_repair"):
                return extrema_result
            return self._recover_missing_relation(
                question,
                extrema_result,
                final_empty=False,
            )
        traces = result.get("traces")
        if not isinstance(traces, list):
            traces = []
        if any(
            isinstance(item, dict)
            and item.get("stage") == "repair_only_detached_component_guard"
            for item in traces
        ):
            claimed, temporal_result = self._recover_failure_temporal_interval(
                question,
                result,
            )
            if claimed:
                return temporal_result
        if not self.failure_endpoint_retrieval_enabled:
            return self._recover_missing_relation(
                question,
                result,
                final_empty=True,
            )
        if any(
            isinstance(item, dict)
            and item.get("stage")
            in {
                "failure_endpoint_retrieval",
                "failure_endpoint_template_retrieval",
            }
            for item in traces
        ):
            return self._recover_missing_relation(
                question,
                result,
                final_empty=True,
            )
        source_failure = str(result.get("failure") or "FINAL_EMPTY_RESULT")
        try:
            from .failure_retrieval_plugin import retrieve

            context = SimpleNamespace(
                question=question,
                pipeline=self,
                trace_bundle={"decomposition": {"traces": traces}},
                verbose_diagnostics=False,
            )
            outcome = retrieve(context, self.kg)
        except Exception as exc:
            diagnostics = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "source_failure": source_failure,
                "trigger": "final_empty",
                "llm_calls": 0,
                "uses_gold_answers": False,
            }
            failed_result = {
                **result,
                "traces": [
                    *traces,
                    self._trace(
                        "failure_endpoint_retrieval",
                        {"question": question},
                        diagnostics,
                    ),
                ],
            }
            return self._recover_missing_relation(
                question,
                failed_result,
                final_empty=True,
            )

        diagnostics = dict(outcome.get("diagnostics", {}))
        diagnostics.update(
            {
                "source_failure": source_failure,
                "trigger": "final_empty",
                "llm_calls": 0,
                "uses_gold_answers": False,
            }
        )
        updated_traces = [
            *traces,
            self._trace(
                "failure_endpoint_retrieval",
                {"question": question},
                diagnostics,
            ),
        ]
        answer_ids = [str(value) for value in outcome.get("answer_ids", [])]
        if not answer_ids:
            return self._recover_missing_relation(
                question,
                {**result, "traces": updated_traces},
                final_empty=True,
            )
        if diagnostics.get("status") == "selected_bounded_direct_best_effort":
            # The direct best-effort lane has already paid for and cached its
            # endpoint work, but its answer is deliberately low confidence.
            # Give the independently verified <=3-query relation repair first
            # refusal; if it abstains, commit the saved direct outcome without
            # rerunning retrieval or any of its endpoint queries.
            relation_result = self._recover_missing_relation(
                question,
                {**result, "traces": updated_traces},
                final_empty=True,
                allow_bounded_best_effort=False,
            )
            if relation_result.get("answer_ids"):
                return relation_result
            updated_traces = list(relation_result.get("traces", updated_traces))
            return {
                **result,
                "answer_ids": answer_ids,
                "answers": list(outcome.get("answers", [])),
                "selected_graph": outcome.get("selected_graph"),
                "failure": None,
                "failure_recovery": {
                    "strategy": "failure_endpoint_bounded_direct_best_effort",
                    "source_failure": source_failure,
                    "model_calls": 0,
                    "relation_retrieval_precedence": "attempted_then_abstained",
                },
                "traces": updated_traces,
            }
        return {
            **result,
            "answer_ids": answer_ids,
            "answers": list(outcome.get("answers", [])),
            "selected_graph": outcome.get("selected_graph"),
            "failure": None,
            "failure_recovery": {
                "strategy": (
                    "failure_endpoint_one_hop_path_retrieval"
                    if diagnostics.get("selected_lane")
                    == "direct_endpoint_path"
                    else "failure_endpoint_template_retrieval"
                ),
                "source_failure": source_failure,
                "model_calls": 0,
            },
            "traces": updated_traces,
        }

    def _recover_extrema_relation(
        self,
        question: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the narrow successful-answer extrema relation challenger."""

        if not getattr(self, "extrema_relation_retrieval_enabled", False):
            return result
        traces = result.get("traces")
        if not isinstance(traces, list):
            traces = []
        if any(
            isinstance(item, dict)
            and item.get("stage") == "extrema_relation_retrieval"
            for item in traces
        ):
            return result
        try:
            from .extrema_relation_retrieval import retrieve_extrema_relation

            outcome = retrieve_extrema_relation(
                pipeline=self,
                question=question,
                result={**result, "traces": traces},
                template_top_k=getattr(
                    self, "extrema_relation_template_top_k", 64
                ),
                source_graph_limit=getattr(
                    self, "extrema_relation_source_graph_limit", 3
                ),
                candidate_limit=getattr(
                    self, "extrema_relation_candidate_limit", 3
                ),
            )
        except Exception as exc:
            outcome = {
                "answer_ids": [],
                "diagnostics": {
                    "status": "error",
                    "error": f"{type(exc).__name__}:{exc}",
                    "trigger": "successful_explicit_unique_extrema",
                    "uses_gold_answers": False,
                    "llm_calls": 0,
                    "candidate_query_limit": 3,
                    "endpoint_execution_queries": 0,
                },
            }
        diagnostics = dict(outcome.get("diagnostics", {}))
        updated_traces = [
            *traces,
            self._trace(
                "extrema_relation_retrieval",
                {"question": question},
                diagnostics,
            ),
        ]
        answer_ids = [str(value) for value in outcome.get("answer_ids", [])]
        if not answer_ids:
            return {**result, "traces": updated_traces}
        return {
            **result,
            "answer_ids": answer_ids,
            "answers": list(outcome.get("answers", [])),
            "selected_graph": outcome.get("selected_graph"),
            "failure": None,
            "traces": updated_traces,
            "answer_repair": {
                "strategy": "extrema_relation_retrieval",
                "source_answer_ids": list(map(str, result.get("answer_ids", []))),
                "model_calls": 0,
                "proof_branch": diagnostics.get("selected_proof_branch", ""),
            },
        }

    def _recover_missing_relation(
        self,
        question: str,
        result: dict[str, Any],
        *,
        final_empty: bool,
        allow_bounded_best_effort: bool | None = None,
    ) -> dict[str, Any]:
        """Run the isolated <=3-query relation replacement lane."""

        if not getattr(self, "missing_relation_retrieval_enabled", False):
            return (
                self._recover_failure_gold_sparql(question, result)
                if final_empty
                else result
            )
        if not final_empty and not getattr(
            self, "missing_relation_successful_enabled", False
        ):
            return result
        if final_empty and str(result.get("failure", "")) != "EMPTY_RESULT":
            return self._recover_failure_gold_sparql(question, result)
        traces = result.get("traces")
        if not isinstance(traces, list):
            traces = []
        if any(
            isinstance(item, dict)
            and item.get("stage") == "missing_relation_retrieval"
            for item in traces
        ):
            return (
                self._recover_failure_gold_sparql(question, result)
                if final_empty
                else result
            )
        try:
            from .missing_relation_retrieval import retrieve_missing_relation

            outcome = retrieve_missing_relation(
                pipeline=self,
                question=question,
                result={**result, "traces": traces},
                final_empty=final_empty,
                template_top_k=getattr(
                    self, "missing_relation_template_top_k", 64
                ),
                candidate_limit=getattr(
                    self, "missing_relation_candidate_limit", 3
                ),
                return_bounded_best_effort=getattr(
                    self,
                    "missing_relation_return_bounded_best_effort",
                    False,
                ) if allow_bounded_best_effort is None else bool(
                    allow_bounded_best_effort
                ),
                best_effort_answer_limit=getattr(
                    self,
                    "missing_relation_best_effort_answer_limit",
                    64,
                ),
                # A deferred direct answer is committed after this call when
                # ``allow_bounded_best_effort`` is explicitly false.  In that
                # mode terminal high-confidence repair keeps first refusal,
                # but the new internal low-confidence lane must not consume
                # or override the deferred direct outcome.
                allow_internal_edge_fallback=bool(
                    final_empty and allow_bounded_best_effort is not False
                ),
            )
        except Exception as exc:
            outcome = {
                "answer_ids": [],
                "diagnostics": {
                    "status": "error",
                    "error": f"{type(exc).__name__}:{exc}",
                    "trigger": (
                        "final_empty" if final_empty else "successful_challenger"
                    ),
                    "uses_gold_answers": False,
                    "llm_calls": 0,
                    "candidate_query_limit": 3,
                },
            }
        diagnostics = dict(outcome.get("diagnostics", {}))
        updated_traces = [
            *traces,
            self._trace(
                "missing_relation_retrieval",
                {"question": question},
                diagnostics,
            ),
        ]
        answer_ids = [str(value) for value in outcome.get("answer_ids", [])]
        if not answer_ids:
            empty_result = {**result, "traces": updated_traces}
            return (
                self._recover_failure_gold_sparql(question, empty_result)
                if final_empty
                else empty_result
            )
        repaired = {
            **result,
            "answer_ids": answer_ids,
            "answers": list(outcome.get("answers", [])),
            "selected_graph": outcome.get("selected_graph"),
            "failure": None,
            "traces": updated_traces,
        }
        if final_empty:
            repaired["failure_recovery"] = {
                "strategy": (
                    "missing_relation_internal_edge_best_effort"
                    if diagnostics.get("status")
                    == "selected_bounded_internal_relation_best_effort"
                    else (
                        "missing_relation_bounded_best_effort"
                        if diagnostics.get("status")
                        == "selected_bounded_relation_best_effort"
                        else "missing_relation_retrieval"
                    )
                ),
                "source_failure": "EMPTY_RESULT",
                "model_calls": 0,
                "confidence": diagnostics.get("confidence", "high"),
            }
        else:
            repaired["answer_repair"] = {
                "strategy": "missing_relation_retrieval",
                "source_answer_ids": list(map(str, result.get("answer_ids", []))),
                "model_calls": 0,
            }
        return repaired

    def _recover_failure_temporal_interval(
        self,
        question: str,
        result: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Claim the shared <=3-query budget for explicit interval rows."""

        if (
            not getattr(self, "temporal_interval_retrieval_enabled", False)
            or result.get("answer_ids")
        ):
            return False, result
        traces = result.get("traces")
        if not isinstance(traces, list):
            traces = []
        if any(
            isinstance(item, dict)
            and item.get("stage") == "failure_temporal_interval_retrieval"
            for item in traces
        ):
            return True, result
        context = SimpleNamespace(
            question=question,
            pipeline=self,
            trace_bundle={"decomposition": {"traces": traces}},
            verbose_diagnostics=False,
        )
        try:
            from .temporal_interval_retrieval import eligible, retrieve

            if not eligible(context):
                return False, result
            outcome = retrieve(context, self.kg)
        except Exception as exc:
            outcome = {
                "answer_ids": [],
                "diagnostics": {
                    "status": "error",
                    "error": f"{type(exc).__name__}:{exc}",
                    "trigger": "final_empty_exclusive_interval_route",
                    "eligible": True,
                    "claimed_query_budget": True,
                    "uses_evaluation_gold": False,
                    "llm_calls": 0,
                    "candidate_query_limit": 3,
                    "entity_or_relation_whitelist": False,
                },
            }
        diagnostics = dict(outcome.get("diagnostics", {}))
        source_failure = str(result.get("failure") or "FINAL_EMPTY_RESULT")
        diagnostics["source_failure"] = source_failure
        updated_traces = [
            *traces,
            self._trace(
                "failure_temporal_interval_retrieval",
                {"question": question},
                diagnostics,
            ),
        ]
        answer_ids = [str(value) for value in outcome.get("answer_ids", [])]
        if not answer_ids:
            return True, {**result, "traces": updated_traces}
        return True, {
            **result,
            "answer_ids": answer_ids,
            "answers": list(outcome.get("answers", [])),
            "selected_graph": outcome.get("selected_graph"),
            "failure": None,
            "failure_recovery": {
                "strategy": "failure_temporal_interval_splice",
                "source_failure": source_failure,
                "model_calls": 0,
                "endpoint_execution_queries": int(
                    diagnostics.get("endpoint_execution_queries", 0)
                ),
            },
            "traces": updated_traces,
        }

    def _recover_failure_gold_sparql(
        self,
        question: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the lazy, source-TRAIN-only raw-SPARQL lane last."""

        if result.get("answer_ids"):
            return result
        claimed, temporal_result = self._recover_failure_temporal_interval(
            question, result
        )
        if claimed:
            return (
                temporal_result
                if temporal_result.get("answer_ids")
                else self._recover_failure_factorized_path(
                    question, temporal_result
                )
            )
        if not getattr(self, "failure_gold_sparql_enabled", False):
            return self._recover_failure_factorized_path(question, result)
        traces = result.get("traces")
        if not isinstance(traces, list):
            traces = []
        if any(
            isinstance(item, dict)
            and item.get("stage") == "failure_source_gold_sparql_retrieval"
            for item in traces
        ):
            return self._recover_failure_factorized_path(question, result)
        source_failure = str(result.get("failure") or "FINAL_EMPTY_RESULT")
        try:
            from .failure_gold_sparql_retrieval import retrieve

            context = SimpleNamespace(
                question=question,
                pipeline=self,
                trace_bundle={"decomposition": {"traces": traces}},
                verbose_diagnostics=False,
            )
            outcome = retrieve(context, self.kg)
        except Exception as exc:
            outcome = {
                "answer_ids": [],
                "diagnostics": {
                    "status": "error",
                    "error": f"{type(exc).__name__}:{exc}",
                    "trigger": "final_empty_after_existing_failure_lanes",
                    "uses_evaluation_gold": False,
                    "llm_calls": 0,
                    "candidate_query_limit": 3,
                },
            }
        diagnostics = dict(outcome.get("diagnostics", {}))
        diagnostics["source_failure"] = source_failure
        updated_traces = [
            *traces,
            self._trace(
                "failure_source_gold_sparql_retrieval",
                {"question": question},
                diagnostics,
            ),
        ]
        answer_ids = [str(value) for value in outcome.get("answer_ids", [])]
        if not answer_ids:
            return self._recover_failure_factorized_path(
                question,
                {**result, "traces": updated_traces},
            )
        return {
            **result,
            "answer_ids": answer_ids,
            "answers": list(outcome.get("answers", [])),
            "selected_graph": outcome.get("selected_graph"),
            "failure": None,
            "failure_recovery": {
                "strategy": (
                    "failure_source_train_gold_sparql_adaptive_semantic"
                    if diagnostics.get("selection_lane") == "adaptive_semantic"
                    else "failure_source_train_gold_sparql_retrieval"
                ),
                "source_failure": source_failure,
                "model_calls": 0,
                "endpoint_execution_queries": int(
                    diagnostics.get("endpoint_execution_queries", 0)
                ),
            },
            "traces": updated_traces,
        }

    def _recover_failure_factorized_path(
        self,
        question: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the <=3-query factorized TRAIN-path lane strictly last."""

        if result.get("answer_ids") or not getattr(
            self, "failure_factorized_path_enabled", False
        ):
            return result
        traces = result.get("traces")
        if not isinstance(traces, list):
            traces = []
        if any(
            isinstance(item, dict)
            and item.get("stage") == "failure_factorized_path_retrieval"
            for item in traces
        ):
            return result
        source_failure = str(result.get("failure") or "FINAL_EMPTY_RESULT")
        try:
            from .failure_factorized_path_retrieval import retrieve

            context = SimpleNamespace(
                question=question,
                pipeline=self,
                trace_bundle={"decomposition": {"traces": traces}},
                verbose_diagnostics=False,
            )
            outcome = retrieve(context, self.kg)
        except Exception as exc:
            outcome = {
                "answer_ids": [],
                "diagnostics": {
                    "status": "error",
                    "error": f"{type(exc).__name__}:{exc}",
                    "trigger": (
                        "final_empty_after_temporal_direct_and_adaptive_lanes"
                    ),
                    "uses_evaluation_gold": False,
                    "llm_calls": 0,
                    "embedding_calls": 0,
                    "candidate_query_limit": 3,
                },
            }
        diagnostics = dict(outcome.get("diagnostics", {}))
        diagnostics["source_failure"] = source_failure
        updated_traces = [
            *traces,
            self._trace(
                "failure_factorized_path_retrieval",
                {"question": question},
                diagnostics,
            ),
        ]
        answer_ids = [str(value) for value in outcome.get("answer_ids", [])]
        if not answer_ids:
            return {**result, "traces": updated_traces}
        return {
            **result,
            "answer_ids": answer_ids,
            "answers": list(outcome.get("answers", [])),
            "selected_graph": outcome.get("selected_graph"),
            "failure": None,
            "failure_recovery": {
                "strategy": "failure_factorized_train_path_retrieval",
                "source_failure": source_failure,
                "model_calls": 0,
                "embedding_calls": 0,
                "endpoint_execution_queries": int(
                    diagnostics.get("endpoint_execution_queries", 0)
                ),
            },
            "traces": updated_traces,
        }

    @staticmethod
    def _decomposition_changed(traces: list[dict[str, Any]]) -> bool:
        original = next((item["output"] for item in traces if item["stage"] == "decompose_predictions"), [])
        reviewed = next((item["output"].get("final_candidates", []) for item in traces
                         if item["stage"] == "decomposition_review_and_rewrite"), original)
        return {tuple(item["decomposition"]) for item in original} != {
            tuple(item["decomposition"]) for item in reviewed
        }

    def _restore_original(self, question: str, traces: list[dict[str, Any]], failure: str) -> dict[str, Any]:
        original_pipeline = copy(self)
        original_pipeline.decomposition_reviewer = None
        original_pipeline.rewrite_execution_fallback = False
        # The outer invocation owns the single final-empty retrieval attempt.
        original_pipeline.failure_endpoint_retrieval_enabled = False
        original_pipeline.missing_relation_retrieval_enabled = False
        original_pipeline.extrema_relation_retrieval_enabled = False
        original_pipeline.failure_gold_sparql_enabled = False
        original_pipeline.failure_factorized_path_enabled = False
        guard = self._trace("decomposition_execution_guard", {"rewritten_failure": failure},
                            {"status": "restored_original", "uses_gold_answers": False})
        try:
            result = original_pipeline.answer(question)
        except PipelineExecutionError as exc:
            raise PipelineExecutionError(exc, [*traces, guard, *exc.traces]) from exc
        result["traces"] = [*traces, guard, *result["traces"]]
        result["rewrite_guard"] = {"fallback_used": True, "rewritten_failure": failure,
                                   "original_failure": result.get("failure")}
        return result

    def _prepare_decompositions(
        self, question: str, traces: list[dict[str, Any]]
    ) -> list[DecompositionCandidate]:
        candidates = self.decompositions.get(question)
        traces.append(self._trace(
            "decompose_predictions", {"question": question},
            [asdict(item) for item in candidates],
        ))
        if self.decomposition_reviewer is None:
            return candidates
        entity_context = []
        if self.use_topic_context:
            try:
                question_key = " ".join(str(question).casefold().split())
                linked_entities = getattr(self.grounder, "gold_entities", {}).get(
                    question_key,
                    {},
                )
                entity_context = linked_entity_context(self.kg, linked_entities)
            except Exception as exc:
                traces.append(self._trace("linked_entity_context", {}, {"error": str(exc)}))
            self._question_context_cache[question] = entity_context
            traces.append(self._trace("linked_entity_context", {"source": "existing topic linking + KG types/event dates"}, entity_context))
        approved: list[DecompositionCandidate] = []
        review_trace: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            if self.decomposition_reviewer.workflow == "two_call" and rank > 1:
                # Preserve extra input hypotheses without spending another
                # review budget. Current WebQSP input has one candidate per item.
                final = list(candidate.decomposition)
                item = {"status": "unreviewed_call_budget", "workflow": "two_call", "events": [],
                        "original_decomposition": list(final), "final_decomposition": list(final),
                        "rewrite_count": 0, "model_call_count": 0}
            else:
                final, item = self.decomposition_reviewer.run(question, candidate.decomposition, entity_context=entity_context)
            item.update({"candidate_rank": rank, "score": candidate.score})
            review_trace.append(item)
            if final is None:
                if not self.decomposition_preserve_original_on_failure:
                    continue
                item["review_status"] = item["status"]
                item["status"] = "preserved_original"
                item["final_decomposition"] = list(candidate.decomposition)
                final = candidate.decomposition
            duplicate = next((existing for existing in approved
                              if existing.decomposition == final), None)
            if duplicate is None:
                approved.append(
                    DecompositionCandidate(
                        list(final),
                        candidate.score,
                        (
                            "review_rewrite"
                            if list(final) != list(candidate.decomposition)
                            else candidate.source
                        ),
                    )
                )
            elif candidate.score > duplicate.score:
                duplicate.score = candidate.score
            if (
                self.retain_original_with_rewrite
                and list(final) != list(candidate.decomposition)
                and not any(
                    existing.decomposition == candidate.decomposition
                    for existing in approved
                )
            ):
                approved.append(
                    DecompositionCandidate(
                        list(candidate.decomposition),
                        candidate.score - 0.01,
                        "review_original",
                    )
                )
                item["original_hypothesis_retained"] = True
        traces.append(self._trace(
            "decomposition_review_and_rewrite",
            {"question": question, "candidate_count": len(candidates)},
            {"approved_count": len({tuple(item["final_decomposition"]) for item in review_trace
                                    if item["status"] in {"accepted", "accepted_after_confirmation", "rewritten"}}),
             "runtime_candidate_count": len(approved), "candidates": review_trace,
             "final_candidates": [asdict(item) for item in approved]},
        ))
        return approved

    def _answer_legacy(
        self,
        question: str,
        traces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        decomposition_candidates = self._prepare_decompositions(question, traces)
        if not decomposition_candidates:
            return self._result(question, traces, failure="NO_APPROVED_DECOMPOSITION")

        semantic_graphs: list[dict[str, Any]] = []
        semantic_graph_sources: list[str] = []
        semantic_outputs: list[dict[str, Any]] = []
        for decomposition in decomposition_candidates:
            if self.semantic_projection_mode == "full":
                projection, deferred = list(decomposition.decomposition), []
            else:
                projection, deferred = semantic_projection_items(
                    question,
                    decomposition.decomposition,
                )
            payload = {"decomposition": projection, "question": question}
            if deferred:
                traces.append(self._trace("semantic_time_conditions", {"question": question},
                                          {"original": decomposition.decomposition, "projection": projection,
                                           "deferred_to_operator": deferred}))
            outputs = self.semantic_model.generate_json(
                instruction=self.contract.semantic_instruction,
                payload=payload,
                count=self.semantic_beam,
            )
            for rank, raw in enumerate(outputs, start=1):
                validation: dict[str, Any] | None = None
                try:
                    graph, validation = self._validate_stage_output(
                        stage="semantic",
                        input_payload=payload,
                        candidate=raw,
                        validator=lambda value: validate_semantic_graph(
                            value,
                            max_hops=self.semantic_max_hops,
                        ),
                        schema=semantic_output_schema(),
                    )
                    if graph is None:
                        raise ContractError("semantic output rejected after validation")
                    if self.require_semantic_item_coverage:
                        _validate_semantic_item_coverage(projection, graph)
                except (ContractError, ValueError) as exc:
                    item = {
                        "decomposition": decomposition.decomposition,
                        "rank": rank,
                        "valid": False,
                        "error": str(exc),
                        "output": raw,
                    }
                    if self._can_correct("semantic"):
                        item["validation"] = validation
                    semantic_outputs.append(item)
                    continue
                semantic_graphs.append(graph)
                semantic_graph_sources.append(decomposition.source)
                item = {
                    "decomposition": decomposition.decomposition,
                    "rank": rank,
                    "valid": True,
                    "output": graph,
                }
                if self._can_correct("semantic"):
                    item["validation"] = validation
                semantic_outputs.append(item)
        traces.append(self._trace("semantic_path_generation", {"question": question}, semantic_outputs))
        if not semantic_graphs:
            return self._result(question, traces, failure="NO_VALID_SEMANTIC_PATH")

        operator_predictions: dict[int, dict[str, Any]] = {}
        if self.operator_model is not None:
            operator_trace: list[dict[str, Any]] = []
            for semantic_index, semantic_graph in enumerate(semantic_graphs):
                payload = {
                    "question": question,
                    "anchors": semantic_graph["anchors"],
                    "semantic_paths": semantic_graph["semantic_paths"],
                }
                item: dict[str, Any] = {
                    "semantic_graph_index": semantic_index,
                    "input": payload,
                }
                try:
                    outputs = self.operator_model.generate_json(
                        instruction=OPERATOR_INSTRUCTION,
                        payload=payload,
                        count=1,
                    )
                    prediction, validation = self._validate_stage_output(
                        stage="operator",
                        input_payload=payload,
                        candidate=outputs[0] if outputs else {},
                        validator=lambda value: _validate_operator_prediction(value, semantic_graph),
                    )
                    if prediction is None:
                        raise ContractError("operator output rejected after validation")
                    operator_predictions[semantic_index] = prediction
                    item.update({"valid": True, "output": prediction})
                    if self._can_correct("operator"):
                        item["validation"] = validation
                except (ContractError, ValueError, RuntimeError) as exc:
                    item.update({"valid": False, "error": str(exc)})
                operator_trace.append(item)
            traces.append(
                self._trace(
                    "glm_operator_prediction",
                    {"model": "operator_model", "count": len(semantic_graphs)},
                    operator_trace,
                )
            )

        ground_candidates, ground_diagnostics = self.grounder.retrieve(
            question=question,
            decompositions=[item for candidate in decomposition_candidates for item in candidate.decomposition],
            semantic_graphs=semantic_graphs,
        )
        for candidate in ground_candidates:
            graph_index = int(candidate.provenance.get("semantic_graph_index", -1))
            source = (
                semantic_graph_sources[graph_index]
                if 0 <= graph_index < len(semantic_graph_sources)
                else "unknown"
            )
            candidate.provenance["decomposition_source"] = source
            if source == "review_rewrite":
                candidate.score -= 0.03
        traces.append(self._trace("entity_relation_candidate_grounding", {"question": question}, ground_diagnostics))
        traces.append(
            self._trace(
                "semantic_guided_local_subgraph_expansion",
                {
                    "first_hop_top_k": self.grounder.first_hop_top_k,
                    "second_hop_top_k": self.grounder.second_hop_top_k,
                    "hop_top_k": self.grounder.hop_top_k,
                    "path_beam": self.grounder.path_beam,
                    "hop_query_limit": self.grounder.hop_query_limit,
                    "hop_values_per_relation": self.grounder.hop_values_per_relation,
                    "long_path_strategy": self.grounder.long_path_strategy,
                },
                [item.provenance for item in ground_candidates],
            )
        )
        if not ground_candidates:
            ground_candidates, repair_output = self._retry_no_grounded_with_prefixes(
                question=question,
                decompositions=[
                    item
                    for candidate in decomposition_candidates
                    for item in candidate.decomposition
                ],
                semantic_graphs=semantic_graphs,
                initial_ground_diagnostics=ground_diagnostics,
            )
            traces.append(
                self._trace(
                    "no_grounded_semantic_candidate_repair",
                    {"failure": "NO_GROUNDED_SEMANTIC_CANDIDATE"},
                    repair_output,
                )
            )
            if (
                ground_candidates
                and self.require_complete_semantic_graph
                and self._decomposition_changed(traces)
                and any(
                    str(item.provenance.get("_repair_strategy", ""))
                    == "accepted_semantic_template"
                    for item in ground_candidates
                )
            ):
                traces.append(
                    self._trace(
                        "rewritten_recovery_guard",
                        {"candidate_count": len(ground_candidates)},
                        {
                            "status": "rejected_provisional_recovery",
                            "reason": "a rewritten plan may not be completed by an unverified semantic template",
                        },
                    )
                )
                return self._result(
                    question,
                    traces,
                    failure="REWRITE_RECOVERY_REJECTED",
                )
        if not ground_candidates:
            return self._result(question, traces, failure="NO_GROUNDED_SEMANTIC_CANDIDATE")
        traces.append(
            self._trace(
                "grounded_semantic_candidates",
                {"count": len(ground_candidates)},
                [item.compose_input for item in ground_candidates],
            )
        )

        query_graphs: list[QueryGraphCandidate] = []
        compose_trace: list[dict[str, Any]] = []
        compose_results = self._generate_compose_batch(
            [
                (
                    grounded.compose_input,
                    _compose_output_schema(grounded.compose_input),
                )
                for grounded in ground_candidates
            ]
        )
        for semantic_rank, (grounded, result) in enumerate(
            zip(ground_candidates, compose_results),
            start=1,
        ):
            semantic_index = int(grounded.provenance.get("semantic_graph_index", -1))
            operator_prediction = operator_predictions.get(semantic_index)
            outputs, request_error = result
            compose_error = (
                str(request_error) if request_error is not None else ""
            )
            if compose_error and not outputs:
                outputs = [
                    _fallback_compose_output(
                        grounded.compose_input,
                        operator_prediction,
                    )
                ]
            for compose_rank, raw in enumerate(outputs, start=1):
                item: dict[str, Any] = {
                    "semantic_rank": semantic_rank,
                    "compose_rank": compose_rank,
                    "input": grounded.compose_input,
                    "output": raw,
                }
                if compose_error:
                    item["fallback"] = {
                        "source": "deterministic_semantic_paths",
                        "error": compose_error,
                    }
                try:
                    compose_output, validation = self._validate_stage_output(
                        stage="compose",
                        input_payload=grounded.compose_input,
                        candidate=raw,
                        validator=lambda value: _validate_legacy_compose_candidate(
                            value,
                            grounded,
                        ),
                        schema=_compose_output_schema(grounded.compose_input),
                    )
                    if compose_output is None:
                        raise ContractError("compose output rejected after validation")
                    if operator_prediction is not None:
                        compose_output = _apply_operator_prediction(
                            compose_output,
                            operator_prediction,
                            grounded.compose_input,
                        )
                        item["operator_override"] = {
                            "source": "glm_operator_prediction",
                            "semantic_graph_index": semantic_index,
                            "and_groups": operator_prediction["and_groups"],
                            "operators": operator_prediction["operators"],
                        }
                    graph = build_query_graph(
                        grounded,
                        compose_output,
                        graph_id=f"G{len(query_graphs)}",
                        pipeline_version=self.version,
                    )
                except (ContractError, LoweringError, ValueError) as exc:
                    item.update({"valid": False, "error": str(exc)})
                    if self._can_correct("compose"):
                        item["validation"] = validation
                    compose_trace.append(item)
                    continue
                if self._can_correct("compose"):
                    item["validation"] = validation
                graph.score -= 0.01 * (compose_rank - 1)
                query_graphs.append(graph)
                item.update({"valid": True, "graph_id": graph.graph_id})
                compose_trace.append(item)
        traces.append(self._trace("compose_global_graph_construction", {}, compose_trace))
        detached_graph_ids = [
            graph.graph_id
            for graph in query_graphs
            if _answer_component_detached_from_grounded_anchors(graph)
        ]
        if detached_graph_ids:
            detached_set = set(detached_graph_ids)
            query_graphs = [
                graph for graph in query_graphs if graph.graph_id not in detached_set
            ]
            traces.append(
                self._trace(
                    "detached_answer_component_guard",
                    {"candidate_count_before": len(query_graphs) + len(detached_set)},
                    {
                        "status": "rejected_detached_answer_components",
                        "rejected_graph_ids": detached_graph_ids,
                        "retained_count": len(query_graphs),
                        "additional_model_calls": 0,
                        "additional_endpoint_queries": 0,
                    },
                )
            )
        query_graphs = _dedupe_graphs(query_graphs, self.query_graph_beam)
        for index, graph in enumerate(query_graphs):
            graph.graph_id = f"G{index}"
        traces.append(
            self._trace(
                "query_graph_beam",
                {"beam_size": self.query_graph_beam},
                [self._graph_summary(graph) for graph in query_graphs],
            )
        )
        if not query_graphs:
            return self._result(question, traces, failure="NO_VALID_QUERY_GRAPH")

        decompositions = [
            item
            for candidate in decomposition_candidates
            for item in candidate.decomposition
        ]
        (
            execution_graphs,
            ordinary_execution_budget,
            entity_kind_queries,
            fixed_slot_diagnostics,
            entity_kind_diagnostics,
            fixed_lane_coordination,
        ) = self._prepare_fixed_execution_lanes(
            question,
            query_graphs,
            decompositions=decompositions,
            semantic_graphs=semantic_graphs,
        )
        if entity_kind_diagnostics is not None:
            traces.append(
                self._trace(
                    "entity_kind_fixed_beam_preparation",
                    {
                        "execution_budget": self.execution_budget,
                        "max_replacements": (
                            self.entity_kind_fixed_beam_max_replacements
                        ),
                    },
                    entity_kind_diagnostics,
                )
            )
        if fixed_slot_diagnostics is not None:
            traces.append(
                self._trace(
                    "fixed_slot_numeric_temporal_preparation",
                    {
                        "execution_budget": self.execution_budget,
                        "max_replacements": (
                            self.numeric_temporal_slot_max_replacements
                        ),
                    },
                    fixed_slot_diagnostics,
                )
            )
        if entity_kind_diagnostics is not None or fixed_slot_diagnostics is not None:
            traces.append(
                self._trace(
                    "fixed_execution_lane_coordination",
                    {"execution_budget": self.execution_budget},
                    fixed_lane_coordination,
                )
            )

        executed: list[ExecutedGraph] = []
        sparql_trace: list[dict[str, Any]] = []
        for graph in execution_graphs[:ordinary_execution_budget]:
            graph.provenance["original_question"] = question
            graph.provenance["decomposition"] = [item for candidate in decomposition_candidates for item in candidate.decomposition]
            execution, execution_trace = self._execute_graph_candidate(graph)
            sparql_trace.append(execution_trace)
            if execution is not None:
                executed.append(execution)
        entity_kind_variants, entity_kind_execution_trace = (
            self._execute_entity_kind_queries(entity_kind_queries)
        )
        if entity_kind_diagnostics is not None:
            traces.append(
                self._trace(
                    "entity_kind_fixed_beam_execution",
                    {
                        "reserved_query_count": len(entity_kind_queries),
                        "ordinary_query_budget": ordinary_execution_budget,
                    },
                    entity_kind_execution_trace,
                )
            )
        execution_trace_items = [*executed, *entity_kind_variants]
        executed, fixed_slot_variants = (
            self._partition_fixed_numeric_temporal_executions(executed)
        )
        if fixed_slot_diagnostics is not None:
            fixed_slot_diagnostics.update(
                {
                    "ordinary_execution_count": len(executed),
                    "isolated_variant_execution_count": len(
                        fixed_slot_variants
                    ),
                    "variants_excluded_from_ordinary_selection": True,
                }
            )
        traces.append(self._trace("deterministic_sparql_lowering", {}, sparql_trace))
        traces.append(
            self._trace(
                "freebase_execution",
                {"budget": self.execution_budget},
                [
                    {
                        "graph_id": item.graph.graph_id,
                        "answer_ids": item.answer_ids,
                        "answers": item.answers,
                        "row_count": item.row_count,
                    }
                    for item in execution_trace_items
                ],
            )
        )
        successful = [item for item in executed if item.answer_ids]
        if not successful and entity_kind_variants:
            entity_rescue, entity_rescue_evidence = (
                select_empty_entity_kind_execution(
                    question,
                    entity_kind_variants,
                )
            )
            traces.append(
                self._trace(
                    "entity_kind_empty_source_rescue",
                    {"ordinary_nonempty_count": 0},
                    entity_rescue_evidence,
                )
            )
            if entity_rescue is not None:
                return {
                    "question": question,
                    "answer_ids": entity_rescue.answer_ids,
                    "answers": entity_rescue.answers,
                    "selected_graph": self._graph_summary(entity_rescue.graph),
                    "failure": None,
                    "traces": traces,
                }
        if not successful:
            return self._result(question, traces, failure="EMPTY_RESULT")
        selected, selector_trace = self._select(
            question,
            successful,
            decompositions=decompositions,
            semantic_graphs=semantic_graphs,
            anchor_surfaces=[
                str(anchor.get("surface", ""))
                for anchor in semantic_graphs[0].get("anchors", [])
                if isinstance(anchor, dict) and str(anchor.get("surface", "")).strip()
            ]
            if semantic_graphs
            else [],
            fixed_slot_variants=fixed_slot_variants,
            entity_kind_variants=entity_kind_variants,
        )
        traces.append(self._trace("executed_graph_selection", {"question": question}, selector_trace))
        return {
            "question": question,
            "answer_ids": selected.answer_ids,
            "answers": selected.answers,
            "selected_graph": self._graph_summary(selected.graph),
            "failure": None,
            "traces": traces,
        }

    def _answer_v2(
        self,
        question: str,
        traces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the 0.2.0 pipeline: grounding -> Compose -> Operator -> merge."""
        decomposition_candidates = self._prepare_decompositions(question, traces)
        if not decomposition_candidates:
            return self._result(question, traces, failure="NO_APPROVED_DECOMPOSITION")

        semantic_graphs: list[dict[str, Any]] = []
        semantic_trace: list[dict[str, Any]] = []
        for decomposition in decomposition_candidates:
            if self.semantic_projection_mode == "full":
                projection, deferred = list(decomposition.decomposition), []
            else:
                projection, deferred = semantic_projection_items(
                    question,
                    decomposition.decomposition,
                )
            payload = {"decomposition": projection, "question": question}
            if deferred:
                traces.append(self._trace("semantic_time_conditions", {"question": question},
                                          {"original": decomposition.decomposition, "projection": projection,
                                           "deferred_to_operator": deferred}))
            outputs = self.semantic_model.generate_json(
                instruction=self.contract.semantic_instruction,
                payload=payload,
                count=self.semantic_beam,
            )
            for rank, raw in enumerate(outputs, start=1):
                item: dict[str, Any] = {
                    "decomposition": decomposition.decomposition,
                    "rank": rank,
                    "output": raw,
                }
                validation: dict[str, Any] | None = None
                try:
                    graph, validation = self._validate_stage_output(
                        stage="semantic",
                        input_payload=payload,
                        candidate=raw,
                        validator=lambda value: validate_semantic_graph(
                            value,
                            max_hops=self.semantic_max_hops,
                        ),
                        schema=semantic_output_schema(),
                    )
                    if graph is None:
                        raise ContractError("semantic output rejected after validation")
                    if self.require_semantic_item_coverage:
                        _validate_semantic_item_coverage(projection, graph)
                except (ContractError, ValueError) as exc:
                    item.update({"valid": False, "error": str(exc)})
                    if self._can_correct("semantic"):
                        item["validation"] = validation
                    semantic_trace.append(item)
                    continue
                semantic_graphs.append(graph)
                item["valid"] = True
                if self._can_correct("semantic"):
                    item["validation"] = validation
                semantic_trace.append(item)
        traces.append(self._trace("semantic_path_generation", {"question": question}, semantic_trace))
        if not semantic_graphs:
            return self._result(question, traces, failure="NO_VALID_SEMANTIC_PATH")

        ground_candidates, ground_diagnostics = self.grounder.retrieve(
            question=question,
            decompositions=[
                item
                for candidate in decomposition_candidates
                for item in candidate.decomposition
            ],
            semantic_graphs=semantic_graphs,
        )
        reconstruction_ground_candidates = list(ground_candidates)
        incomplete_ground_candidates: list[GroundedSemanticCandidate] = []
        if (
            self.require_complete_semantic_graph
            and self._decomposition_changed(traces)
        ):
            incomplete = [
                candidate
                for candidate in ground_candidates
                if candidate.provenance.get("dropped_paths")
            ]
            if incomplete:
                incomplete_ground_candidates = incomplete
                ground_candidates = [
                    candidate
                    for candidate in ground_candidates
                    if not candidate.provenance.get("dropped_paths")
                ]
                traces.append(
                    self._trace(
                        "complete_semantic_graph_guard",
                        {"candidate_count_before": len(incomplete) + len(ground_candidates)},
                        {
                            "status": "rejected_incomplete_candidates",
                            "rejected_count": len(incomplete),
                            "retained_count": len(ground_candidates),
                            "reason": "every Semantic path is an explicit query constraint",
                        },
                    )
                )
        traces.append(
            self._trace(
                "semantic_subgraph_grounding_and_ranking",
                {
                    "first_hop_top_k": self.grounder.first_hop_top_k,
                    "second_hop_top_k": self.grounder.second_hop_top_k,
                    "hop_top_k": self.grounder.hop_top_k,
                    "path_beam": self.grounder.path_beam,
                    "hop_query_limit": self.grounder.hop_query_limit,
                    "hop_values_per_relation": self.grounder.hop_values_per_relation,
                    "long_path_strategy": self.grounder.long_path_strategy,
                    "grounded_semantic_top_k": self.grounder.semantic_beam,
                },
                ground_diagnostics,
            )
        )
        traces.append(
            self._trace(
                "grounded_semantic_candidates",
                {"count": len(ground_candidates)},
                [
                    {
                        "score": item.score,
                        "compose_input": item.compose_input,
                        "anchor_bindings": {
                            key: value.entity_id
                            for key, value in item.anchor_bindings.items()
                        },
                        "relation_bindings": item.relation_bindings,
                        "provenance": item.provenance,
                    }
                    for item in ground_candidates
                ],
            )
        )
        if not ground_candidates:
            ground_candidates, repair_output = self._retry_no_grounded_with_prefixes(
                question=question,
                decompositions=[
                    item
                    for candidate in decomposition_candidates
                    for item in candidate.decomposition
                ],
                semantic_graphs=semantic_graphs,
                initial_ground_diagnostics=ground_diagnostics,
            )
            traces.append(
                self._trace(
                    "no_grounded_semantic_candidate_repair",
                    {"failure": "NO_GROUNDED_SEMANTIC_CANDIDATE"},
                    repair_output,
                )
            )
            reconstruction_ground_candidates.extend(ground_candidates)
        if not ground_candidates and incomplete_ground_candidates:
            ground_candidates = incomplete_ground_candidates
            for candidate in ground_candidates:
                candidate.score -= 0.5
                candidate.provenance["incomplete_graph_fallback"] = {
                    "status": "last_resort",
                    "dropped_paths": deepcopy(
                        candidate.provenance.get("dropped_paths", [])
                    ),
                }
            traces.append(
                self._trace(
                    "incomplete_semantic_graph_fallback",
                    {"candidate_count": len(incomplete_ground_candidates)},
                    {
                        "status": "restored_after_no_complete_candidate",
                        "score_penalty": 0.5,
                    },
                )
            )
            reconstruction_ground_candidates.extend(ground_candidates)
        ontology = getattr(self.grounder, "ontology", None)
        retained_ground_candidates: list[GroundedSemanticCandidate] = []
        type_return_rejections: list[dict[str, Any]] = []
        for semantic_rank, candidate in enumerate(ground_candidates, start=1):
            reject, evidence = _present_time_repaired_type_return_cycle(
                question,
                candidate,
                ontology,
            )
            if reject:
                type_return_rejections.append(
                    {"semantic_rank": semantic_rank, **evidence}
                )
            else:
                retained_ground_candidates.append(candidate)
        retained_reconstruction_candidates: list[GroundedSemanticCandidate] = []
        reconstruction_type_return_rejections = 0
        for candidate in reconstruction_ground_candidates:
            reject, _ = _present_time_repaired_type_return_cycle(
                question,
                candidate,
                ontology,
            )
            if reject:
                reconstruction_type_return_rejections += 1
            else:
                retained_reconstruction_candidates.append(candidate)
        if type_return_rejections or reconstruction_type_return_rejections:
            candidate_count_before = len(ground_candidates)
            ground_candidates = retained_ground_candidates
            reconstruction_ground_candidates = retained_reconstruction_candidates
            traces.append(
                self._trace(
                    "present_time_type_return_cycle_guard",
                    {"candidate_count_before": candidate_count_before},
                    {
                        "status": "rejected_repaired_type_return_cycles",
                        "rejected_count": len(type_return_rejections),
                        "reconstruction_rejected_count": (
                            reconstruction_type_return_rejections
                        ),
                        "retained_count": len(ground_candidates),
                        "rejections": type_return_rejections,
                        "additional_model_calls": 0,
                        "additional_endpoint_queries": 0,
                    },
                )
            )
        retained_ground_candidates = []
        terminal_cvt_rejections: list[dict[str, Any]] = []
        for semantic_rank, candidate in enumerate(ground_candidates, start=1):
            reject, evidence = _direction_repaired_terminal_cvt_cross_path(
                candidate,
                ontology,
            )
            if reject:
                terminal_cvt_rejections.append(
                    {"semantic_rank": semantic_rank, **evidence}
                )
            else:
                retained_ground_candidates.append(candidate)
        retained_reconstruction_candidates = []
        reconstruction_terminal_cvt_rejections = 0
        for candidate in reconstruction_ground_candidates:
            reject, _ = _direction_repaired_terminal_cvt_cross_path(
                candidate,
                ontology,
            )
            if reject:
                reconstruction_terminal_cvt_rejections += 1
            else:
                retained_reconstruction_candidates.append(candidate)
        if terminal_cvt_rejections or reconstruction_terminal_cvt_rejections:
            candidate_count_before = len(ground_candidates)
            ground_candidates = retained_ground_candidates
            reconstruction_ground_candidates = retained_reconstruction_candidates
            traces.append(
                self._trace(
                    "direction_repaired_terminal_cvt_guard",
                    {"candidate_count_before": candidate_count_before},
                    {
                        "status": "rejected_schema_incompatible_cross_paths",
                        "rejected_count": len(terminal_cvt_rejections),
                        "reconstruction_rejected_count": (
                            reconstruction_terminal_cvt_rejections
                        ),
                        "retained_count": len(ground_candidates),
                        "rejections": terminal_cvt_rejections,
                        "additional_model_calls": 0,
                        "additional_endpoint_queries": 0,
                    },
                )
            )
        retained_ground_candidates = []
        schema_impossibility_rejections: list[dict[str, Any]] = []
        schema_guards = (
            _direction_repair_lowers_terminal_consistency,
            _lexically_requested_object_type_return_cycle,
            _direction_repaired_terminal_cvt_incomplete_path,
            _pure_ontology_expansion_schema_discontinuity,
        )
        for semantic_rank, candidate in enumerate(ground_candidates, start=1):
            rejection: dict[str, Any] | None = None
            for guard in schema_guards:
                reject, evidence = guard(candidate, ontology)
                if reject:
                    rejection = evidence
                    break
            if rejection is None:
                retained_ground_candidates.append(candidate)
            else:
                schema_impossibility_rejections.append(
                    {"semantic_rank": semantic_rank, **rejection}
                )
        retained_reconstruction_candidates = []
        reconstruction_schema_impossibility_rejections = 0
        for candidate in reconstruction_ground_candidates:
            rejected = False
            for guard in schema_guards:
                reject, _ = guard(candidate, ontology)
                if reject:
                    rejected = True
                    break
            if rejected:
                reconstruction_schema_impossibility_rejections += 1
            else:
                retained_reconstruction_candidates.append(candidate)
        if (
            schema_impossibility_rejections
            or reconstruction_schema_impossibility_rejections
        ):
            candidate_count_before = len(ground_candidates)
            ground_candidates = retained_ground_candidates
            reconstruction_ground_candidates = retained_reconstruction_candidates
            traces.append(
                self._trace(
                    "retrieval_schema_impossibility_guard",
                    {"candidate_count_before": candidate_count_before},
                    {
                        "status": "rejected_schema_impossible_candidates",
                        "rejected_count": len(schema_impossibility_rejections),
                        "reconstruction_rejected_count": (
                            reconstruction_schema_impossibility_rejections
                        ),
                        "retained_count": len(ground_candidates),
                        "rejections": schema_impossibility_rejections,
                        "additional_model_calls": 0,
                        "additional_endpoint_queries": 0,
                    },
                )
            )
        if not ground_candidates:
            return self._result(question, traces, failure="NO_GROUNDED_SEMANTIC_CANDIDATE")
        if (
            self.require_complete_semantic_graph
            and self._decomposition_changed(traces)
            and any(
                str(
                    item.provenance.get("grounding_repair", {}).get(
                        "strategy",
                        item.provenance.get("_repair_strategy", ""),
                    )
                )
                == "accepted_semantic_template"
                for item in ground_candidates
            )
        ):
            traces.append(
                self._trace(
                    "rewritten_recovery_guard",
                    {"candidate_count": len(ground_candidates)},
                    {
                        "status": "rejected_provisional_recovery",
                        "reason": "a rewritten plan may not be completed by an unverified semantic template",
                    },
                )
            )
            return self._result(
                question,
                traces,
                failure="REWRITE_RECOVERY_REJECTED",
            )

        # A rewritten decomposition must be allowed to fail through the old
        # path so answer() can invoke _restore_original.  A local non-empty
        # compiler result here would otherwise mask the execution guard.
        failure_compiler_enabled = not self._decomposition_changed(traces)
        query_graphs: list[QueryGraphCandidate] = []
        compose_trace: list[dict[str, Any]] = []
        operator_trace: list[dict[str, Any]] = []
        operator_candidate_rank = 0
        operator_instruction = _resolve_operator_instruction(
            self.contract.operator_instruction,
            self.operator_prompt_mode,
            self.contract.operator_examples,
        )
        compose_requests = []
        for grounded in ground_candidates:
            request: tuple[Any, ...] = (
                grounded.compose_input,
                _compose_graph_output_schema(),
            )
            repair_context = grounded.provenance.get("grounding_repair")
            if (
                isinstance(repair_context, dict)
                and repair_context.get("compose_mode") != "model"
            ):
                request = (
                    *request,
                    _fallback_compose_graph_output(grounded.compose_input),
                )
            compose_requests.append(request)
        # Operator futures are submitted from the Compose result stream.  This
        # lets Operator work overlap with remaining Compose requests while the
        # final aggregation below still uses stable candidate order.
        operator_futures: list[tuple[Any, dict[str, Any], Any]] = []
        operator_executor = (
            ThreadPoolExecutor(
                max_workers=self.operator_workers,
                thread_name_prefix="operator",
            )
            if self.operator_model is not None
            else None
        )
        try:
            for compose_index, result in self._generate_compose_stream(compose_requests):
                grounded = ground_candidates[compose_index]
                semantic_rank = compose_index + 1
                compose_item: dict[str, Any] = {
                    "semantic_rank": semantic_rank,
                    "input": grounded.compose_input,
                }
                compose_outputs, request_error = result
                if request_error is not None:
                    exc = request_error
                    compose_item.update({"valid": False, "error": str(exc)})
                    if self._can_correct("compose"):
                        try:
                            compose_outputs = [_fallback_compose_graph_output(grounded.compose_input)]
                            compose_item["fallback"] = {
                                "source": "deterministic_semantic_paths",
                                "reason": "compose_model_unavailable",
                            }
                        except (ContractError, ValueError) as fallback_exc:
                            compose_item["fallback_error"] = str(fallback_exc)
                            compose_trace.append(compose_item)
                            continue
                    else:
                        compose_trace.append(compose_item)
                        continue
                for compose_rank, raw_compose in enumerate(compose_outputs, start=1):
                    repair_provenance = grounded.provenance.get("grounding_repair")
                    compose_item = {
                        "semantic_rank": semantic_rank,
                        "compose_rank": compose_rank,
                        "input": grounded.compose_input,
                        "output": raw_compose,
                    }
                    if (
                        isinstance(repair_provenance, dict)
                        and repair_provenance.get("compose_mode") != "model"
                    ):
                        compose_item["repair_fallback"] = {
                            "source": "grounded_semantic_paths",
                            "model_used": False,
                        }
                    validation: dict[str, Any] | None = None
                    try:
                        compose_output, validation = self._validate_stage_output(
                            stage="compose",
                            input_payload=grounded.compose_input,
                            candidate=raw_compose,
                            validator=lambda value: _validate_graph_compose_candidate(
                                value,
                                grounded,
                            ),
                            schema=_compose_graph_output_schema(),
                        )
                        if compose_output is None:
                            raise ContractError("compose output rejected after validation")
                        compose_item["valid"] = True
                        if self._can_correct("compose"):
                            compose_item["validation"] = validation
                    except (ContractError, ValueError) as exc:
                        compose_item.update({"valid": False, "error": str(exc)})
                        if self._can_correct("compose"):
                            compose_item["validation"] = validation
                        # A model may append a schema-valid scalar attribute
                        # that was not part of the grounded Semantic path. Do
                        # not discard the already verified core path. Compile
                        # it deterministically without another Operator/LLM
                        # request; this candidate still competes normally in
                        # the bounded graph beam.
                        if (
                            "unable to transfer grounded relation" in str(exc)
                            and not grounded.provenance.get("dropped_paths")
                        ):
                            try:
                                fallback_output = _fallback_compose_graph_output(
                                    grounded.compose_input
                                )
                                fallback_graph = build_query_graph_v2(
                                    grounded,
                                    fallback_output,
                                    graph_id=f"GC{len(query_graphs)}",
                                    pipeline_version=self.version,
                                )
                            except (LoweringError, ContractError, ValueError) as fallback_exc:
                                compose_item["grounded_core_fallback_error"] = str(
                                    fallback_exc
                                )
                            else:
                                fallback_graph.score -= 0.015
                                fallback_graph.provenance["graph_fallback"] = {
                                    "source": "grounded_core_after_untransferred_compose_extension",
                                    "model_used": False,
                                }
                                query_graphs.append(fallback_graph)
                                compose_item.update(
                                    {
                                        "grounded_core_fallback": True,
                                        "fallback_graph_id": fallback_graph.graph_id,
                                    }
                                )
                        compose_trace.append(compose_item)
                        continue
                    compose_trace.append(compose_item)
                    operator_candidate_rank += 1
                    metadata = {
                        "grounded": grounded,
                        "compose_output": compose_output,
                        "compose_item": compose_item,
                        "semantic_rank": semantic_rank,
                        "compose_rank": compose_rank,
                        "operator_candidate_rank": operator_candidate_rank,
                    }
                    if operator_executor is None:
                        operator_output, op_item = self._run_operator_candidate(
                            question=question,
                            grounded_input=grounded.compose_input,
                            compose_output=compose_output,
                            semantic_rank=semantic_rank,
                            compose_rank=compose_rank,
                            operator_candidate_rank=operator_candidate_rank,
                            operator_instruction=operator_instruction,
                            repair_context=grounded.provenance.get("grounding_repair"),
                        )
                        operator_futures.append((None, metadata, (operator_output, op_item)))
                    else:
                        future = operator_executor.submit(
                            self._run_operator_candidate,
                            question=question,
                            grounded_input=grounded.compose_input,
                            compose_output=compose_output,
                            semantic_rank=semantic_rank,
                            compose_rank=compose_rank,
                            operator_candidate_rank=operator_candidate_rank,
                            operator_instruction=operator_instruction,
                            repair_context=grounded.provenance.get("grounding_repair"),
                        )
                        operator_futures.append((future, metadata, None))
        finally:
            if operator_executor is not None:
                operator_executor.shutdown(wait=True)

        # Collect Operator results in submission order so graph IDs and traces
        # remain deterministic even though requests completed out of order.
        for future, metadata, immediate in operator_futures:
            operator_output, op_item = immediate if immediate is not None else future.result()
            operator_trace.append(op_item)
            if not operator_output:
                continue
            compose_output = metadata["compose_output"]
            merged = deepcopy(compose_output)
            merged["operators"] = deepcopy(operator_output["operators"])
            try:
                graph = build_query_graph_v2(
                    metadata["grounded"],
                    merged,
                    graph_id=f"G{len(query_graphs)}",
                    pipeline_version=self.version,
                )
            except (LoweringError, ContractError, ValueError) as exc:
                metadata["compose_item"].update(
                    {"graph_valid": False, "graph_error": str(exc)}
                )
                continue
            graph.score -= 0.01 * (metadata["compose_rank"] - 1)
            query_graphs.append(graph)
            metadata["compose_item"]["graph_id"] = graph.graph_id

        # Compose models occasionally leave independently anchored intersection
        # paths disconnected even though every path terminal denotes the same
        # answer candidate. Add a small deterministic candidate beam that only
        # reuses the already grounded paths. It is bounded to the first three
        # candidates and only enabled when the normal beam is predominantly
        # operator-free, so temporal/numeric programs and runtime stay intact.
        operator_free = sum(not graph.operators for graph in query_graphs)
        if query_graphs and operator_free >= max(1, len(query_graphs) // 2):
            for deterministic_rank, grounded in enumerate(ground_candidates[:3], start=1):
                if len(grounded.compose_input.get("semantic_paths", [])) < 2:
                    continue
                try:
                    deterministic_output = _deterministic_terminal_intersection_output(
                        grounded.compose_input
                    )
                    graph = build_query_graph_v2(
                        grounded,
                        deterministic_output,
                        graph_id=f"GI{deterministic_rank}",
                        pipeline_version=self.version,
                    )
                except (LoweringError, ContractError, ValueError) as exc:
                    compose_trace.append({
                        "semantic_rank": deterministic_rank,
                        "valid": False,
                        "deterministic_intersection_error": str(exc),
                    })
                    continue
                graph.score += 0.005
                graph.provenance["graph_repair"] = {
                    "strategy": "terminal_intersection",
                    "model_used": False,
                }
                query_graphs.append(graph)
                compose_trace.append({
                    "semantic_rank": deterministic_rank,
                    "valid": True,
                    "graph_id": graph.graph_id,
                    "deterministic_intersection": {
                        "model_used": False,
                        "path_count": len(grounded.compose_input.get("semantic_paths", [])),
                    },
                })

        failure_compiler_eligible = bool(
            failure_compiler_enabled
            and _failure_compiler_eligible(question, compose_trace)
        )

        # Preempt a non-empty beam only when every graph is locally impossible
        # under two hard RDF categories.  Ordinary entity-type disagreements
        # are never sufficient.  The failure compiler is invoked only after
        # this proof, so any schema-feasible normal graph keeps byte-for-byte
        # behavior and does not pay compiler work.
        if query_graphs and failure_compiler_eligible:
            hard_conflicts = [
                _query_graph_hard_schema_conflicts(
                    graph,
                    getattr(self.grounder, "ontology", None),
                )
                for graph in query_graphs
            ]
            if all(hard_conflicts):
                constrained_graph, constrained_diagnostics = (
                    _failure_constrained_recovery_graph(
                        question=question,
                        compose_trace=compose_trace,
                        grounded_candidates=ground_candidates,
                        ontology=getattr(self.grounder, "ontology", None),
                        pipeline_version=self.version,
                    )
                )
                query_graphs, preemption_trace = (
                    _hard_schema_conflict_preemption(
                        query_graphs,
                        constrained_graph,
                        getattr(self.grounder, "ontology", None),
                        conflicts=hard_conflicts,
                    )
                )
                if preemption_trace.get("status") == "preempted":
                    compose_trace.append({
                        "semantic_rank": int(
                            constrained_diagnostics.get("semantic_rank", 0)
                        ),
                        "valid": True,
                        "hard_schema_conflict_preemption": preemption_trace,
                        "failure_constrained_compile": constrained_diagnostics,
                    })

        # If Compose or Operator rejected every model candidate, first compile
        # a small intersection beam from all already-grounded paths.  The old
        # first-path fallback below cannot compile when additional grounded
        # relations remain unconsumed.  This recovery is bounded and makes no
        # model request.
        if not query_graphs and failure_compiler_eligible:
            constrained_graph, constrained_diagnostics = (
                _failure_constrained_recovery_graph(
                    question=question,
                    compose_trace=compose_trace,
                    grounded_candidates=ground_candidates,
                    ontology=getattr(self.grounder, "ontology", None),
                    pipeline_version=self.version,
                )
            )
            compose_trace.append({
                "semantic_rank": int(
                    constrained_diagnostics.get("semantic_rank", 0)
                ),
                "valid": constrained_graph is not None,
                "failure_constrained_compile": constrained_diagnostics,
            })
            if constrained_graph is not None:
                query_graphs.append(constrained_graph)

        if not query_graphs and failure_compiler_enabled:
            recovery_graphs, reconstruction_diagnostics = (
                reconstruct_failure_query_graphs(
                    reconstruction_ground_candidates,
                    ontology=getattr(self.grounder, "ontology", None),
                    pipeline_version=self.version,
                    limit=3,
                )
            )
            if not recovery_graphs:
                recovery_graphs, recovery_diagnostics = (
                    _terminal_intersection_recovery_graphs(
                        ground_candidates,
                        pipeline_version=self.version,
                        limit=3,
                    )
                )
            else:
                recovery_diagnostics = []
            query_graphs.extend(recovery_graphs)
            compose_trace.append({
                "semantic_rank": 0,
                "valid": bool(recovery_graphs),
                "graph_reconstruction_recovery": reconstruction_diagnostics,
            })
            compose_trace.extend(
                {
                    "semantic_rank": int(item.get("semantic_rank", 0)),
                    "valid": item.get("status") == "compiled",
                    "terminal_intersection_recovery": item,
                }
                for item in recovery_diagnostics
            )

        # Retain the first grounded path only when even the complete terminal
        # intersection could not be compiled.
        if not query_graphs and failure_compiler_enabled:
            for fallback_index, grounded in enumerate(ground_candidates):
                try:
                    fallback_output = _fallback_compose_graph_output(
                        grounded.compose_input
                    )
                    fallback_output = _infer_missing_extrema_operator(
                        question,
                        fallback_output,
                        {"operators": fallback_output.get("operators", [])},
                    ) | {
                        key: value
                        for key, value in fallback_output.items()
                        if key != "operators"
                    }
                    graph = build_query_graph_v2(
                        grounded,
                        fallback_output,
                        graph_id=f"GF{fallback_index}",
                        pipeline_version=self.version,
                    )
                except (LoweringError, ContractError, ValueError) as exc:
                    compose_trace.append(
                        {
                            "semantic_rank": fallback_index + 1,
                            "valid": False,
                            "fallback_error": str(exc),
                            "fallback_source": "grounded_semantic_paths_after_empty_graph_beam",
                        }
                    )
                    continue
                graph.provenance["graph_fallback"] = {
                    "source": "grounded_semantic_paths_after_empty_graph_beam",
                    "model_used": False,
                }
                query_graphs.append(graph)
                compose_trace.append(
                    {
                        "semantic_rank": fallback_index + 1,
                        "valid": True,
                        "graph_id": graph.graph_id,
                        "fallback": {
                            "source": "grounded_semantic_paths_after_empty_graph_beam",
                            "model_used": False,
                        },
                    }
                )

        reversed_chain_graphs: list[QueryGraphCandidate] = []
        if failure_compiler_enabled:
            for source_graph in query_graphs[:10]:
                fallback = source_graph.provenance.get("graph_fallback", {})
                if (
                    isinstance(fallback, dict)
                    and fallback.get("source")
                    in {
                        "terminal_intersection_after_unsuccessful_graphs",
                        "typed_spine_intersection_after_unsuccessful_graphs",
                        "rejected_compose_scalar_constraint",
                    }
                ):
                    # Keep failure-only recovery within its fixed execution budget.
                    continue
                reversed_chain_graphs.extend(
                    _reverse_constant_to_answer_chains(source_graph, limit=1)
                )
                if len(reversed_chain_graphs) >= 5:
                    break
        if reversed_chain_graphs:
            query_graphs.extend(reversed_chain_graphs[:5])
            compose_trace.append({
                "semantic_rank": 0,
                "valid": True,
                "anchored_chain_repairs": {
                    "candidate_count": min(5, len(reversed_chain_graphs)),
                    "model_used": False,
                },
            })

        compose_trace.sort(
            key=lambda item: (
                int(item.get("semantic_rank", 0)),
                int(item.get("compose_rank", 0)),
            )
        )
        operator_trace.sort(
            key=lambda item: int(item.get("operator_candidate_rank", 0))
        )
        traces.append(self._trace("compose_graph_generation", {}, compose_trace))
        traces.append(
            self._trace(
                "operator_prediction_and_program_merge",
                {
                    "prompt_mode": self.operator_prompt_mode,
                    "operator_top_k": self.operator_top_k,
                    "operator_workers": self.operator_workers,
                    "few_shot_types": [
                        str(example.get("operator_type", ""))
                        for example in self.contract.operator_examples
                    ]
                    if self.operator_prompt_mode == "glm_few_shot"
                    else [],
                },
                operator_trace,
            )
        )
        detached_graph_ids = [
            graph.graph_id
            for graph in query_graphs
            if _answer_component_detached_from_grounded_anchors(graph)
        ]
        if detached_graph_ids:
            detached_set = set(detached_graph_ids)
            query_graphs = [
                graph for graph in query_graphs if graph.graph_id not in detached_set
            ]
            traces.append(
                self._trace(
                    "detached_answer_component_guard",
                    {"candidate_count_before": len(query_graphs) + len(detached_set)},
                    {
                        "status": "rejected_detached_answer_components",
                        "rejected_graph_ids": detached_graph_ids,
                        "retained_count": len(query_graphs),
                        "additional_model_calls": 0,
                        "additional_endpoint_queries": 0,
                    },
                )
            )
        query_graphs = _dedupe_graphs(query_graphs, self.query_graph_beam)
        for index, graph in enumerate(query_graphs):
            graph.graph_id = f"G{index}"
        traces.append(
            self._trace(
                "final_graph_beam",
                {"beam_size": self.query_graph_beam},
                [self._graph_summary(graph) for graph in query_graphs],
            )
        )
        if not query_graphs:
            return self._result(question, traces, failure="NO_VALID_QUERY_GRAPH")

        # Execution and normal selection may annotate/mutate operator graphs.
        # Keep only the small substantive-operator subset as it appeared in the
        # final beam so the post-selection challenger sees the same fixed
        # runtime evidence as the validated trace replay.  Operator-free and
        # NO_EQUAL-only graphs are never consumed by that challenger.
        zero_graph_operator_snapshots = [
            deepcopy(graph)
            for graph in query_graphs
            if any(
                str(operator.get("type", "")).upper() != "NO_EQUAL"
                for operator in graph.operators
                if isinstance(operator, dict)
            )
        ]

        decompositions = [
            item
            for candidate in decomposition_candidates
            for item in candidate.decomposition
        ]
        (
            execution_graphs,
            ordinary_execution_budget,
            entity_kind_queries,
            fixed_slot_diagnostics,
            entity_kind_diagnostics,
            fixed_lane_coordination,
        ) = self._prepare_fixed_execution_lanes(
            question,
            query_graphs,
            decompositions=decompositions,
            semantic_graphs=semantic_graphs,
        )
        if entity_kind_diagnostics is not None:
            traces.append(
                self._trace(
                    "entity_kind_fixed_beam_preparation",
                    {
                        "execution_budget": self.execution_budget,
                        "max_replacements": (
                            self.entity_kind_fixed_beam_max_replacements
                        ),
                    },
                    entity_kind_diagnostics,
                )
            )
        if fixed_slot_diagnostics is not None:
            traces.append(
                self._trace(
                    "fixed_slot_numeric_temporal_preparation",
                    {
                        "execution_budget": self.execution_budget,
                        "max_replacements": (
                            self.numeric_temporal_slot_max_replacements
                        ),
                    },
                    fixed_slot_diagnostics,
                )
            )
        if entity_kind_diagnostics is not None or fixed_slot_diagnostics is not None:
            traces.append(
                self._trace(
                    "fixed_execution_lane_coordination",
                    {"execution_budget": self.execution_budget},
                    fixed_lane_coordination,
                )
            )

        executed: list[ExecutedGraph] = []
        sparql_trace: list[dict[str, Any]] = []
        for graph in execution_graphs[:ordinary_execution_budget]:
            graph.provenance["original_question"] = question
            graph.provenance["decomposition"] = [item for candidate in decomposition_candidates for item in candidate.decomposition]
            execution, execution_trace = self._execute_graph_candidate(graph)
            sparql_trace.append(execution_trace)
            if execution is None:
                continue
            executed.append(execution)

        entity_kind_variants, entity_kind_execution_trace = (
            self._execute_entity_kind_queries(entity_kind_queries)
        )
        if entity_kind_diagnostics is not None:
            traces.append(
                self._trace(
                    "entity_kind_fixed_beam_execution",
                    {
                        "reserved_query_count": len(entity_kind_queries),
                        "ordinary_query_budget": ordinary_execution_budget,
                    },
                    entity_kind_execution_trace,
                )
            )

        executed, fixed_slot_variants = (
            self._partition_fixed_numeric_temporal_executions(executed)
        )
        if fixed_slot_diagnostics is not None:
            fixed_slot_diagnostics.update(
                {
                    "ordinary_execution_count": len(executed),
                    "isolated_variant_execution_count": len(
                        fixed_slot_variants
                    ),
                    "variants_excluded_from_ordinary_selection": True,
                }
            )

        ordinary_successful = [item for item in executed if item.answer_ids]
        if not ordinary_successful and entity_kind_variants:
            entity_rescue, entity_rescue_evidence = (
                select_empty_entity_kind_execution(
                    question,
                    entity_kind_variants,
                )
            )
            traces.append(
                self._trace(
                    "entity_kind_empty_source_rescue",
                    {"ordinary_nonempty_count": 0},
                    entity_rescue_evidence,
                )
            )
            if entity_rescue is not None:
                traces.append(
                    self._trace(
                        "deterministic_sparql_lowering_and_execution",
                        {},
                        sparql_trace,
                    )
                )
                return {
                    "question": question,
                    "answer_ids": entity_rescue.answer_ids,
                    "answers": entity_rescue.answers,
                    "selected_graph": self._graph_summary(entity_rescue.graph),
                    "failure": None,
                    "traces": traces,
                }

        rewritten_restore_pending = _rewritten_failure_requires_restore(
            failure_compiler_enabled,
            executed,
        )

        # A failure-constrained graph is already the sole, fully specified
        # local recovery program.  Do not broaden it with attribute siblings,
        # operator-free queries, selector calls, or post-selection repairs.
        failure_constrained_only = bool(query_graphs) and all(
            _is_failure_constrained_recovery_graph(graph)
            for graph in query_graphs
        )
        if failure_constrained_only:
            traces.append(
                self._trace(
                    "deterministic_sparql_lowering_and_execution",
                    {},
                    sparql_trace,
                )
            )
            successful = [item for item in executed if item.answer_ids]
            if not successful:
                return self._result(question, traces, failure="EMPTY_RESULT")
            selected = successful[0]
            selector_trace = {
                "status": "deterministic_failure_constrained_recovery",
                "selected_graph_id": selected.graph.graph_id,
                "reason_codes": [
                    "rejected_compose_scalar_constraint",
                    "no_additional_model_call",
                    "single_query_budget",
                ],
            }
            traces.append(
                self._trace(
                    "glm_final_graph_ranking",
                    {"model": "local_failure_recovery", "candidate_count": 1},
                    selector_trace,
                )
            )
            return {
                "question": question,
                "answer_ids": selected.answer_ids,
                "answers": selected.answers,
                "selected_graph": self._graph_summary(selected.graph),
                "failure": None,
                "traces": traces,
            }

        # Repair an empty numeric/temporal operator against the local ontology
        # before considering an operator-free fallback. The search is bounded
        # and keeps the original graph structure and literal value.
        source_graphs = [
            graph for graph in query_graphs
            if self._operator_needs_ontology_repair(graph)
            and not is_graph_reconstruction_graph(graph)
        ]
        if not any(item.answer_ids for item in executed):
            source_graphs = [
                graph
                for graph in query_graphs
                if not is_graph_reconstruction_graph(graph)
            ]
        if source_graphs:
            attribute_siblings: list[QueryGraphCandidate] = []
            documents = [
                " ".join(
                    " ".join(relation_label_from_id(str(triple[1])))
                    for triple in graph.triples
                )
                for graph in source_graphs
            ]
            semantic_scores = self.grounder.ranker.score(question, documents)
            source_graphs = [
                graph
                for _, graph in sorted(
                    zip(semantic_scores, source_graphs),
                    key=lambda item: (
                        -float(item[0]),
                        -item[1].score,
                        item[1].graph_id,
                    ),
                )[:4]
            ]
            for graph in source_graphs:
                attribute_siblings.extend(
                    self._operator_attribute_sibling_graphs(graph, limit=8)
                )
            for sibling in attribute_siblings[:32]:
                if _answer_component_detached_from_grounded_anchors(sibling):
                    sparql_trace.append(
                        {
                            "graph_id": sibling.graph_id,
                            "source_graph_id": sibling.provenance.get(
                                "operator_attribute_repair", {}
                            ).get("source_graph_id"),
                            "fallback": "ontology_operator_attribute_repair",
                            "status": "rejected_detached_answer_component",
                            "additional_endpoint_queries": 0,
                        }
                    )
                    continue
                sibling_execution, sibling_trace = self._execute_graph_candidate(sibling)
                sibling_trace.update(
                    {
                        "source_graph_id": sibling.provenance.get(
                            "operator_attribute_repair", {}
                        ).get("source_graph_id"),
                        "fallback": "ontology_operator_attribute_repair",
                    }
                )
                sparql_trace.append(sibling_trace)
                if sibling_execution is not None and sibling_execution.answer_ids:
                    executed.append(sibling_execution)
                    if rewritten_restore_pending:
                        break

        # A reviewed rewrite normally restores the saved original branch when
        # every ordinary execution is empty.  First allow only the bounded,
        # schema-backed attribute repair above: it preserves the grounded path
        # and explicit literal and was previously made unreachable by this
        # guard.  Do not continue into broader operator-free/reconstruction
        # fallbacks when the narrow repair also remains empty.
        if _rewritten_failure_requires_restore(
            failure_compiler_enabled,
            executed,
        ):
            traces.append(
                self._trace(
                    "deterministic_sparql_lowering_and_execution",
                    {},
                    sparql_trace,
                )
            )
            return self._result(question, traces, failure="EMPTY_RESULT")

        # Retry operator-bearing graphs only after the complete normal beam
        # produced no answer.  This prevents a broad operator-free graph from
        # competing with a successful normal candidate.
        if not any(item.answer_ids for item in executed):
            for graph in query_graphs[: self.execution_budget]:
                if not graph.operators or is_graph_reconstruction_graph(graph):
                    continue
                fallback = _operator_free_graph(graph)
                fallback_execution, fallback_trace = self._execute_graph_candidate(fallback)
                fallback_trace.update(
                    {
                        "source_graph_id": graph.graph_id,
                        "fallback": "operator_free_after_empty_result",
                    }
                )
                sparql_trace.append(fallback_trace)
                if fallback_execution is not None and fallback_execution.answer_ids:
                    executed.append(fallback_execution)

        # A valid model graph can still be over-composed into an empty join.
        # Retry at most three complete grounded terminal intersections.  Skip
        # this when those recovery graphs were already the original beam.
        if (
            not any(item.answer_ids for item in executed)
            and not any(
                isinstance(graph.provenance.get("graph_fallback"), dict)
                and graph.provenance["graph_fallback"].get("source")
                in {
                    "terminal_intersection_after_unsuccessful_graphs",
                    "typed_spine_intersection_after_unsuccessful_graphs",
                }
                for graph in query_graphs
            )
        ):
            recovery_graphs, reconstruction_diagnostics = (
                reconstruct_failure_query_graphs(
                    reconstruction_ground_candidates,
                    ontology=getattr(self.grounder, "ontology", None),
                    pipeline_version=self.version,
                    limit=3,
                )
            )
            if not recovery_graphs:
                recovery_graphs, recovery_diagnostics = (
                    _terminal_intersection_recovery_graphs(
                        ground_candidates,
                        pipeline_version=self.version,
                        limit=3,
                    )
                )
            else:
                recovery_diagnostics = []
            constrained_graph: QueryGraphCandidate | None = None
            constrained_diagnostics: dict[str, Any] = {
                "status": "not_attempted_without_terminal_graph",
                "model_used": False,
            }
            if recovery_graphs and failure_compiler_eligible:
                constrained_graph, constrained_diagnostics = (
                    _failure_constrained_recovery_graph(
                        question=question,
                        compose_trace=compose_trace,
                        grounded_candidates=ground_candidates,
                        ontology=getattr(self.grounder, "ontology", None),
                        pipeline_version=self.version,
                    )
                )
            recovery_plan = _failure_terminal_recovery_plan(
                recovery_graphs,
                constrained_graph,
                limit=3,
            )
            for graph in recovery_plan:
                graph.provenance["original_question"] = question
                graph.provenance["decomposition"] = [
                    item
                    for candidate in decomposition_candidates
                    for item in candidate.decomposition
                ]
                recovery_execution, recovery_trace = (
                    self._execute_graph_candidate(graph)
                )
                recovery_trace.update(
                    {
                        "fallback": (
                            "rejected_compose_scalar_after_empty_result"
                            if _is_failure_constrained_recovery_graph(graph)
                            else (
                                "typed_spine_intersection_after_empty_result"
                                if is_graph_reconstruction_graph(graph)
                                else "terminal_intersection_after_empty_result"
                            )
                        ),
                        "model_used": False,
                    }
                )
                sparql_trace.append(recovery_trace)
                if (
                    recovery_execution is not None
                    and recovery_execution.answer_ids
                ):
                    executed.append(recovery_execution)
            if recovery_diagnostics or reconstruction_diagnostics.get("status") == "compiled":
                recovery_summary: dict[str, Any] = {
                    "fallback": "bounded_graph_reconstruction_recovery_summary",
                    "model_used": False,
                    "candidates": recovery_diagnostics,
                    "graph_reconstruction": reconstruction_diagnostics,
                }
                if failure_compiler_eligible:
                    recovery_summary.update({
                        "failure_constrained_compile": constrained_diagnostics,
                        "execution_budget": len(recovery_plan),
                    })
                sparql_trace.append(recovery_summary)
        traces.append(self._trace("deterministic_sparql_lowering_and_execution", {}, sparql_trace))
        successful = [item for item in executed if item.answer_ids]
        if _repair_only_anchor_echo_failure(successful):
            traces.append(
                self._trace(
                    "repair_only_anchor_echo_guard",
                    {
                        "successful_execution_count": len(successful),
                    },
                    {
                        "status": "rejected_repair_only_anchor_echo",
                        "rejected_graph_ids": [
                            item.graph.graph_id for item in successful
                        ],
                        "reason": (
                            "ontology operator-attribute repair returned only "
                            "the input anchor after the ordinary beam was empty"
                        ),
                        "model_calls": 0,
                        "endpoint_queries": 0,
                        "uses_gold": False,
                    },
                )
            )
            return self._result(question, traces, failure="EMPTY_RESULT")
        repair_only_detached = _repair_only_detached_constant_failure(successful)
        interval_eligible = False
        if (
            repair_only_detached
            and getattr(self, "temporal_interval_retrieval_enabled", False)
        ):
            try:
                from .temporal_interval_retrieval import eligible

                interval_eligible = bool(
                    eligible(
                        SimpleNamespace(
                            question=question,
                            pipeline=self,
                            trace_bundle={"decomposition": {"traces": traces}},
                            verbose_diagnostics=False,
                        )
                    )
                )
            except Exception:
                interval_eligible = False
        if repair_only_detached and interval_eligible:
            traces.append(
                self._trace(
                    "repair_only_detached_component_guard",
                    {"successful_execution_count": len(successful)},
                    {
                        "status": "routed_to_temporal_interval_retrieval",
                        "rejected_graph_ids": [
                            item.graph.graph_id for item in successful
                        ],
                        "reason": (
                            "a detached constant component cannot constrain "
                            "the answer component"
                        ),
                        "temporal_interval_eligible": True,
                        "model_calls": 0,
                        "endpoint_queries": 0,
                        "uses_gold": False,
                    },
                )
            )
            return self._result(question, traces, failure="EMPTY_RESULT")
        role_gate_rejections: list[dict[str, Any]] = []
        role_compatible: list[ExecutedGraph] = []
        reconstruction_ontology = getattr(self.grounder, "ontology", None)
        for execution in successful:
            accepted, evidence = graph_reconstruction_answer_role_gate(
                question,
                execution.graph,
                reconstruction_ontology,
                answer_count=len(execution.answer_ids),
            )
            if accepted:
                role_compatible.append(execution)
                continue
            role_gate_rejections.append(
                {
                    **evidence,
                    "graph_id": execution.graph.graph_id,
                    "answer_count": len(execution.answer_ids),
                }
            )
        if role_gate_rejections:
            traces.append(
                self._trace(
                    "graph_reconstruction_answer_role_gate",
                    {
                        "question": question,
                        "recovery_execution_count": len(successful),
                    },
                    {
                        "status": "rejected_incompatible_recoveries",
                        "rejected": role_gate_rejections,
                        "retained_count": len(role_compatible),
                        "model_calls": 0,
                        "endpoint_queries": 0,
                        "embedding_calls": 0,
                        "uses_gold": False,
                    },
                )
            )
            successful = role_compatible
        if not successful:
            return self._result(question, traces, failure="EMPTY_RESULT")
        decompositions = [
            item
            for candidate in decomposition_candidates
            for item in candidate.decomposition
        ]
        recovery_only = all(
            _is_terminal_intersection_recovery(item)
            or _is_graph_reconstruction_recovery(item)
            for item in successful
        )
        failure_terminal_recovery_only = bool(successful) and all(
            _is_terminal_intersection_recovery(item)
            or _is_graph_reconstruction_recovery(item)
            or _is_failure_constrained_recovery_graph(item.graph)
            for item in successful
        ) and any(
            _is_failure_constrained_recovery_graph(item.graph)
            for item in successful
        )
        retrieval_recovery_only = all(
            _is_retrieval_ontology_recovery(item) for item in successful
        )
        # Recovery graphs already bind the common terminal directly.  Skip
        # normal post-selection projections because they can issue more KG
        # queries and would weaken the strict three-query recovery budget.
        if failure_terminal_recovery_only:
            selected, selector_trace = _select_failure_terminal_recovery(
                successful
            )
            selected, constraint_repair_trace = (
                self._postselect_missing_explicit_constraint(
                    selected,
                    question=question,
                )
            )
            selector_trace["late_explicit_constraint_repair"] = (
                constraint_repair_trace
            )
        elif recovery_only:
            selected, selector_trace = _select_terminal_intersection_recovery(
                successful
            )
            selected, constraint_repair_trace = (
                self._postselect_missing_explicit_constraint(
                    selected,
                    question=question,
                )
            )
            selector_trace["late_explicit_constraint_repair"] = (
                constraint_repair_trace
            )
        elif retrieval_recovery_only:
            selected, selector_trace = _select_retrieval_ontology_recovery(
                successful
            )
            selected, constraint_repair_trace = (
                self._postselect_missing_explicit_constraint(
                    selected,
                    question=question,
                )
            )
            selector_trace["late_explicit_constraint_repair"] = (
                constraint_repair_trace
            )
            selected, scalar_projection_trace = (
                self._postselect_scalar_answer_projection(
                    selected,
                    question=question,
                )
            )
            if scalar_projection_trace.get("status") != "not_applicable":
                selector_trace["scalar_answer_projection"] = (
                    scalar_projection_trace
                )
            selected, mediator_projection_trace = (
                self._postselect_unlabeled_mediator_projection(
                    selected,
                    question=question,
                )
            )
            if mediator_projection_trace.get("status") != "not_applicable":
                selector_trace["unlabeled_mediator_projection"] = (
                    mediator_projection_trace
                )
        else:
            selected, selector_trace = self._select(
                question,
                successful,
                decompositions=decompositions,
                semantic_graphs=semantic_graphs,
                anchor_surfaces=[
                    str(anchor.get("surface", ""))
                    for anchor in semantic_graphs[0].get("anchors", [])
                    if isinstance(anchor, dict)
                    and str(anchor.get("surface", "")).strip()
                ]
                if semantic_graphs
                else [],
                fixed_slot_variants=fixed_slot_variants,
                entity_kind_variants=entity_kind_variants,
            )
            selected, constraint_repair_trace = (
                self._postselect_missing_explicit_constraint(
                    selected,
                    question=question,
                )
            )
            selector_trace["late_explicit_constraint_repair"] = (
                constraint_repair_trace
            )
            selected, scalar_projection_trace = (
                self._postselect_scalar_answer_projection(
                    selected,
                    question=question,
                )
            )
            if scalar_projection_trace.get("status") != "not_applicable":
                selector_trace["scalar_answer_projection"] = (
                    scalar_projection_trace
                )
            selected, mediator_projection_trace = (
                self._postselect_unlabeled_mediator_projection(
                    selected,
                    question=question,
                )
            )
            if mediator_projection_trace.get("status") != "not_applicable":
                selector_trace["unlabeled_mediator_projection"] = (
                    mediator_projection_trace
                )
        traces.append(
            self._trace(
                "glm_final_graph_ranking",
                {"model": "selector_model", "candidate_count": len(successful)},
                selector_trace,
            )
        )
        selected, challenger_trace = self._postselect_zero_graph_challenger(
            selected,
            question=question,
            grounded_candidates=ground_candidates,
            final_graphs=zero_graph_operator_snapshots,
        )
        traces.append(
            self._trace(
                "zero_graph_challenger",
                {"question": question, "max_endpoint_executions": 3},
                challenger_trace,
            )
        )
        return {
            "question": question,
            "answer_ids": selected.answer_ids,
            "answers": selected.answers,
            "selected_graph": self._graph_summary(selected.graph),
            "failure": None,
            "traces": traces,
        }

    def _postselect_zero_graph_challenger(
        self,
        selected: ExecutedGraph,
        *,
        question: str,
        grounded_candidates: list[GroundedSemanticCandidate],
        final_graphs: list[QueryGraphCandidate],
    ) -> tuple[ExecutedGraph, dict[str, Any]]:
        """Challenge only a high-evidence structural loss, with three queries max.

        The gate and generator are local and Gold-blind.  Candidate execution is
        deliberately direct: it neither invokes a model nor performs label or
        temporal-reference lookups.  The first non-empty candidate in the fixed
        score order wins; if all three are empty or fail, ``selected`` is returned
        unchanged.
        """

        eligible, reasons = zero_graph_challenger_gate(
            question=question,
            selected_graph=selected.graph,
            grounded_candidates=grounded_candidates,
            final_graphs=final_graphs,
        )
        diagnostics: dict[str, Any] = {
            "status": "not_applicable",
            "gate": eligible,
            "gate_reasons": reasons,
            "endpoint_executions": 0,
            "execution_budget": 3,
            "model_used": False,
            "uses_gold": False,
        }
        if not eligible:
            return selected, diagnostics

        candidates, generation = build_zero_graph_challengers(
            question=question,
            grounded_candidates=grounded_candidates,
            final_graphs=final_graphs,
            ontology=getattr(self.grounder, "ontology", None),
            pipeline_version=self.version,
            limit=3,
        )
        diagnostics.update(
            {
                "status": "all_challengers_empty",
                "generation": generation,
                "executions": [],
            }
        )
        for candidate in candidates[:3]:
            started = time.monotonic()
            execution_trace: dict[str, Any] = {
                "graph_id": candidate.graph.graph_id,
                "strategy": candidate.strategy,
                "score": candidate.score,
                "sparql": candidate.sparql,
            }
            diagnostics["endpoint_executions"] += 1
            try:
                rows = self.kg.execute(candidate.sparql)
                answer_ids = _answer_values(rows, candidate.graph.answer_var)
                execution_trace.update(
                    {
                        "answer_count": len(answer_ids),
                        "row_count": len(rows),
                    }
                )
            except Exception as exc:
                execution_trace.update(
                    {
                        "answer_count": 0,
                        "error": str(exc),
                        "error_details": _exception_trace_details(exc),
                    }
                )
                answer_ids = []
                rows = []
            execution_trace["elapsed_seconds"] = round(
                time.monotonic() - started,
                6,
            )
            diagnostics["executions"].append(execution_trace)
            if not answer_ids:
                continue

            replacement_safe, safety_reason = zero_graph_replacement_is_safe(
                selected_graph=selected.graph,
                selected_answer_ids=selected.answer_ids,
                challenger=candidate,
                challenger_answer_ids=answer_ids,
                gate_reasons=reasons,
            )
            if replacement_safe and _where_location_type_regression(
                question,
                selected.graph,
                candidate.graph,
                getattr(self.grounder, "ontology", None),
            ):
                replacement_safe = False
                safety_reason = "where_location_answer_role_regression"
            if replacement_safe and _ontology_head_answer_role_regression(
                question,
                selected.graph,
                candidate.graph,
                getattr(self.grounder, "ontology", None),
            ):
                replacement_safe = False
                safety_reason = "ontology_head_answer_role_regression"
            execution_trace.update(
                {
                    "replacement_safe": replacement_safe,
                    "replacement_safety_reason": safety_reason,
                }
            )
            if not replacement_safe:
                diagnostics.update(
                    {
                        "status": "challenger_rejected_by_answer_safety",
                        "rejected_graph_id": candidate.graph.graph_id,
                        "replacement_safety_reason": safety_reason,
                    }
                )
                return selected, diagnostics

            graph = deepcopy(candidate.graph)
            graph.sparql = candidate.sparql
            replacement = ExecutedGraph(
                graph=graph,
                answer_ids=answer_ids,
                # Reuse labels already present on the normal selection.  New
                # IDs fall back to their stable value; a label lookup would be
                # an additional endpoint request outside the three-query cap.
                answers=_answer_labels(answer_ids, selected.answers),
                row_count=len(rows),
            )
            diagnostics.update(
                {
                    "status": "selected_nonempty_challenger",
                    "selected_graph_id": graph.graph_id,
                    "replaced_graph_id": selected.graph.graph_id,
                    "answer_count": len(answer_ids),
                }
            )
            return replacement, diagnostics
        return selected, diagnostics

    def _operator_attribute_sibling_graphs(
        self,
        graph: QueryGraphCandidate,
        *,
        limit: int = 4,
    ) -> list[QueryGraphCandidate]:
        """Build bounded ontology-backed alternatives for an empty operator graph.

        Candidate properties are selected from the schema domain/range of graph
        variables and ranked by the embedding model. No entity, question, or
        relation whitelist is used. One-hop scalar properties and two-hop CVT
        properties are both supported.
        """
        ontology = getattr(self.grounder, "ontology", None)
        ranker = getattr(self.grounder, "ranker", None)
        if ontology is None or ranker is None or not graph.operators:
            return []

        scalar_ranges = {
            "type.float",
            "type.int",
            "type.datetime",
        }
        variable_types: dict[str, set[str]] = {}
        predecessors: dict[str, list[str]] = {}
        for subject, relation_id, object_ in graph.triples:
            domain = str(ontology.domain_for_relation(relation_id))
            range_id = str(ontology.range_for_relation(relation_id))
            if str(subject).startswith("V") and domain:
                variable_types.setdefault(str(subject), set()).add(domain)
            if str(object_).startswith("V") and range_id:
                variable_types.setdefault(str(object_), set()).add(range_id)
            if str(subject).startswith("V") and str(object_).startswith("V"):
                predecessors.setdefault(str(object_), []).append(str(subject))

        question = str(graph.provenance.get("original_question", ""))
        candidates: list[tuple[float, int, str, str, str, dict[str, Any]]] = []
        for operator_index, operator in enumerate(graph.operators):
            operator_type = str(operator.get("type", "")).upper()
            if operator_type not in {
                "EQUAL",
                "GREATER_THAN",
                "GREATER_OR_EQUAL",
                "LESS_THAN",
                "LESS_OR_EQUAL",
                "TC",
                "ARGMIN",
                "ARGMAX",
            }:
                continue
            predicted_label = " ".join(
                str(value)
                for value in operator.get("attribute_relation_label", [])
            )
            intent = " ".join(value for value in (question, predicted_label) if value)
            input_var = str(operator.get("input_var", ""))
            owner_vars = list(dict.fromkeys([
                input_var,
                graph.answer_var,
                *predecessors.get(input_var, []),
            ]))
            for owner_var in owner_vars:
                declared_types = variable_types.get(owner_var, set())
                owner_types = {
                    supertype
                    for type_id in declared_types
                    for supertype in (
                        ontology.supertypes(type_id)
                        if callable(getattr(ontology, "supertypes", None))
                        else (type_id,)
                    )
                }
                owner_types.update(
                    subtype
                    for type_id in declared_types
                    for subtype in (
                        ontology.subtypes(type_id, max_depth=2)
                        if callable(getattr(ontology, "subtypes", None))
                        else (type_id,)
                    )
                )
                if not owner_types:
                    continue
                expanded_owner_types = {
                    candidate_type
                    for owner_type in owner_types
                    for candidate_type in (
                        ontology.supertypes(owner_type)
                        if callable(getattr(ontology, "supertypes", None))
                        else (owner_type,)
                    )
                }
                direct: list[str] = []
                first_hops: list[str] = []
                for owner_type in expanded_owner_types:
                    for relation_id in ontology.relations_for_domain(owner_type):
                        range_id = str(ontology.range_for_relation(relation_id))
                        if range_id in scalar_ranges:
                            direct.append(relation_id)
                        elif range_id:
                            first_hops.append(relation_id)

                direct = list(dict.fromkeys(direct))
                first_hops = list(dict.fromkeys(first_hops))
                if first_hops:
                    first_documents = [
                        " ".join(relation_label_from_id(relation_id))
                        for relation_id in first_hops
                    ]
                    first_scores = ranker.score(intent, first_documents)
                    first_hops = [
                        relation_id
                        for _, relation_id in sorted(
                            zip(first_scores, first_hops),
                            key=lambda item: (-float(item[0]), item[1]),
                        )[:24]
                    ]

                path_specs: list[tuple[str, str]] = [
                    (relation_id, "") for relation_id in direct
                ]
                for first_relation in first_hops:
                    middle_type = str(ontology.range_for_relation(first_relation))
                    for second_relation in ontology.relations_for_domain(middle_type):
                        if str(ontology.range_for_relation(second_relation)) in scalar_ranges:
                            path_specs.append((first_relation, second_relation))
                path_specs = list(dict.fromkeys(path_specs))
                if not path_specs:
                    continue
                documents = [
                    " ".join([
                        *relation_label_from_id(first_relation),
                        *(
                            relation_label_from_id(second_relation)
                            if second_relation
                            else []
                        ),
                    ])
                    for first_relation, second_relation in path_specs
                ]
                scores = ranker.score(intent, documents)
                for score, (first_relation, second_relation) in sorted(
                    zip(scores, path_specs),
                    key=lambda item: (
                        -(
                            float(item[0])
                            + 0.08
                            * len(
                                {
                                    token
                                    for token in re.findall(
                                        r"[a-z]+",
                                        " ".join(
                                            relation_label_from_id(item[1][0])
                                            + (
                                                relation_label_from_id(item[1][1])
                                                if item[1][1]
                                                else []
                                            )
                                        ).casefold(),
                                    )
                                    if len(token) > 2
                                }
                                & {
                                    token
                                    for token in re.findall(r"[a-z]+", intent.casefold())
                                    if len(token) > 2
                                }
                            )
                        ),
                        item[1][0],
                        item[1][1],
                    ),
                )[: max(8, int(limit))]:
                    path_words = {
                        token
                        for token in re.findall(
                            r"[a-z]+",
                            " ".join(
                                relation_label_from_id(first_relation)
                                + (
                                    relation_label_from_id(second_relation)
                                    if second_relation
                                    else []
                                )
                            ).casefold(),
                        )
                        if len(token) > 2
                    }
                    intent_words = {
                        token
                        for token in re.findall(r"[a-z]+", intent.casefold())
                        if len(token) > 2
                    }
                    adjusted_score = float(score) + 0.08 * len(
                        path_words & intent_words
                    ) - (
                        0.15 if owner_var != input_var else 0.0
                    )
                    candidates.append((
                        adjusted_score,
                        operator_index,
                        owner_var,
                        first_relation,
                        second_relation,
                        operator,
                    ))

        candidates.sort(key=lambda item: (-item[0], item[2], item[3], item[4]))
        siblings: list[QueryGraphCandidate] = []
        seen: set[str] = set()
        for score, operator_index, owner_var, first_relation, second_relation, _ in candidates:
            sibling = deepcopy(graph)
            sibling.graph_id = f"{graph.graph_id}A{len(siblings)}"
            sibling.sparql = ""
            keep_indexes: list[int] = []
            for index, sibling_operator in enumerate(sibling.operators):
                if index == operator_index:
                    keep_indexes.append(index)
                    continue
                value = str(sibling_operator.get("value", "")).strip().casefold()
                input_ref = str(sibling_operator.get("input_var", ""))
                known_types = variable_types.get(input_ref, set())
                type_tokens = {
                    token
                    for type_id in known_types
                    for token in type_id.casefold().replace("_", " ").split(".")[-1].split()
                }
                redundant_type_equality = (
                    str(sibling_operator.get("type", "")).upper() == "EQUAL"
                    and str(sibling_operator.get("value_type", "")).casefold()
                    in {"string", "text", "entity", "type"}
                    and value
                    and value in type_tokens
                )
                if not redundant_type_equality:
                    keep_indexes.append(index)
            repaired_index = keep_indexes.index(operator_index)
            sibling.operators = [sibling.operators[index] for index in keep_indexes]
            repaired_operator = sibling.operators[repaired_index]
            old_input = str(repaired_operator.get("input_var", ""))
            if second_relation:
                variables = {
                    value
                    for triple in sibling.triples
                    for value in (str(triple[0]), str(triple[2]))
                    if value.startswith("V") and value[1:].isdigit()
                }
                next_index = max((int(value[1:]) for value in variables), default=-1) + 1
                middle_var = f"V{next_index}"
                sibling.triples.append([owner_var, first_relation, middle_var])
                repaired_operator["input_var"] = middle_var
                repaired_operator["attribute_relation_id"] = second_relation
                repaired_operator["attribute_relation_label"] = relation_label_from_id(second_relation)
            else:
                repaired_operator["input_var"] = owner_var
                repaired_operator["attribute_relation_id"] = first_relation
                repaired_operator["attribute_relation_label"] = relation_label_from_id(first_relation)
            if str(repaired_operator.get("type", "")).upper() == "EQUAL":
                repaired_operator["_numeric_cast_compare"] = True
            if str(repaired_operator.get("type", "")).upper() in _RANGE_OPERATOR_TYPES:
                repaired_operator["_numeric_comparison_cast"] = True

            # If Compose returned the compared leaf as the answer, restore its
            # unique predecessor as the entity answer and remove the obsolete
            # scalar edge. This is structural and applies only to filter
            # operators, never to COUNT or entity extrema.
            if (
                old_input == graph.answer_var
                and owner_var != graph.answer_var
                and str(repaired_operator.get("type", "")).upper()
                in {"EQUAL", *_RANGE_OPERATOR_TYPES}
            ):
                sibling.answer_var = owner_var
                sibling.triples = [
                    triple
                    for triple in sibling.triples
                    if not (
                        str(triple[0]) == owner_var
                        and str(triple[2]) == graph.answer_var
                    )
                ]
            key = json.dumps(
                {
                    "triples": sibling.triples,
                    "operators": sibling.operators,
                    "answer_var": sibling.answer_var,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            if key in seen:
                continue
            seen.add(key)
            sibling.score = graph.score - 0.02 + (0.01 * score)
            sibling.provenance = deepcopy(graph.provenance)
            sibling.provenance["operator_attribute_repair"] = {
                "source_graph_id": graph.graph_id,
                "operator_index": operator_index,
                "owner_var": owner_var,
                "first_relation": first_relation,
                "second_relation": second_relation,
                "embedding_score": score,
            }
            siblings.append(sibling)
            coverage_repair = bool(
                repaired_operator.get("_constraint_coverage_repair")
            )
            if (
                second_relation
                and not coverage_repair
                and len(siblings) < max(1, int(limit))
            ):
                flattened = deepcopy(sibling)
                flattened.graph_id = f"{graph.graph_id}A{len(siblings)}"
                flattened.triples = [
                    triple
                    for triple in flattened.triples
                    if not (
                        str(triple[0]) == owner_var
                        and str(triple[1]) == first_relation
                    )
                ]
                flattened_operator = flattened.operators[repaired_index]
                flattened_operator["input_var"] = owner_var
                flattened_operator["attribute_relation_id"] = second_relation
                flattened_operator["attribute_relation_label"] = relation_label_from_id(second_relation)
                flattened.provenance["operator_attribute_repair"] = {
                    **flattened.provenance["operator_attribute_repair"],
                    "schema_variant": "flattened_leaf",
                }
                flattened.score -= 0.01
                siblings.append(flattened)
            if (
                str(repaired_operator.get("type", "")).upper()
                in _RANGE_OPERATOR_TYPES
                and len(siblings) < max(1, int(limit))
            ):
                equality_sibling = deepcopy(sibling)
                equality_sibling.graph_id = f"{graph.graph_id}A{len(siblings)}"
                equality_operator = equality_sibling.operators[operator_index]
                equality_operator["type"] = "EQUAL"
                equality_operator["_numeric_cast_compare"] = True
                equality_operator.pop("_numeric_comparison_cast", None)
                equality_sibling.provenance["operator_attribute_repair"] = {
                    **equality_sibling.provenance["operator_attribute_repair"],
                    "operator_type_alternative": "EQUAL",
                }
                equality_sibling.score -= 0.005
                siblings.append(equality_sibling)
            if len(siblings) >= max(1, int(limit)):
                break
        return siblings

    def _postselect_missing_explicit_constraint(
        self,
        selected: ExecutedGraph,
        *,
        question: str,
        execution_limit: int = 8,
    ) -> tuple[ExecutedGraph, dict[str, Any]]:
        """Apply a bounded schema-backed constraint absent from the winner.

        This is a no-model late repair.  It is deliberately conservative:
        the question must contain an explicit literal/comparison/extremum, a
        real schema property name must match that constraint surface, and the
        repaired answers must be a strict subset of the original answers.
        """
        spec = _missing_constraint_spec(question)
        if spec is None or not selected.answer_ids:
            return selected, {"status": "not_applicable"}
        if len(selected.answer_ids) <= 1:
            # A filtering/order repair cannot improve precision for a
            # singleton.  Skipping it also keeps the common exact-answer path
            # at zero additional KG cost.
            return selected, {
                "status": "not_applicable",
                "reason": "singleton_answer",
            }
        requested_count = re.match(
            r"^\s*(?:what|which)\s+(\d+)\s+",
            str(question),
            re.I,
        )
        if (
            requested_count is not None
            and int(requested_count.group(1)) == len(selected.answer_ids)
        ):
            return selected, {
                "status": "not_applicable",
                "reason": "explicit_answer_count_already_satisfied",
            }
        if len(selected.answer_ids) > 256:
            return selected, {
                "status": "not_applicable",
                "reason": "source_answer_budget_exceeded",
            }
        operator_type = str(spec["type"]).upper()
        if (
            operator_type in {"ARGMAX", "ARGMIN"}
            and re.search(
                r"\b(?:romantic\s+)?relationships?|\bdat(?:ed|ing)\b",
                str(question),
                re.I,
            )
        ):
            # Relationship extrema must order the already-bound relationship
            # CVT, not attach a fresh relationship to the answer entity.  The
            # generic late attribute repair cannot prove that owner binding.
            return selected, {
                "status": "not_applicable",
                "reason": "relationship_extrema_requires_existing_cvt_owner",
            }
        equivalent_types = {
            "ARGMAX": {"ARGMAX"},
            "ARGMIN": {"ARGMIN"},
            "EQUAL": {"EQUAL"},
            "GREATER_THAN": {"GREATER_THAN", "GREATER_OR_EQUAL"},
            "LESS_THAN": {"LESS_THAN", "LESS_OR_EQUAL"},
        }[operator_type]
        if any(
            str(operator.get("type", "")).upper() in equivalent_types
            for operator in selected.graph.operators
            if isinstance(operator, dict)
        ):
            return selected, {
                "status": "not_applicable",
                "reason": "constraint_already_present",
            }

        probe = deepcopy(selected.graph)
        probe.graph_id = f"{probe.graph_id}C"
        probe.sparql = ""
        probe.provenance["original_question"] = question
        synthetic_index = len(probe.operators)
        probe.operators.append(
            {
                "type": operator_type,
                "inputs": [],
                "input_var": probe.answer_var,
                "attribute_relation_label": [question],
                "attribute_relation_labels": [],
                "value": str(spec["value"]),
                "value_type": str(spec["value_type"]),
                "_late_explicit_constraint_repair": True,
            }
        )
        siblings = self._operator_attribute_sibling_graphs(
            probe,
            # Search a wider schema beam, then execute only the bounded set
            # that has surface evidence.  Unrelated high-scoring ontology
            # paths must not crowd a valid lower-ranked property out.
            limit=max(16, max(1, int(execution_limit)) * 2),
        )
        ontology = getattr(self.grounder, "ontology", None)
        ranker = getattr(self.grounder, "ranker", None)
        candidates: list[dict[str, Any]] = []
        for sibling in siblings:
            repair = sibling.provenance.get("operator_attribute_repair", {})
            if (
                not isinstance(repair, dict)
                or int(repair.get("operator_index", -1)) != synthetic_index
                # An equality sibling is useful while repairing an uncertain
                # model-produced range operator, but an explicit surface
                # comparison must never be weakened from < or > to =.
                or bool(repair.get("operator_type_alternative"))
            ):
                continue
            first_relation = str(repair.get("first_relation", ""))
            second_relation = str(repair.get("second_relation", ""))
            terminal_relation = second_relation or first_relation
            candidates.append(
                {
                    "graph": sibling,
                    "first_relation": first_relation,
                    "second_relation": second_relation,
                    "schema_variant": str(repair.get("schema_variant", "")),
                    "terminal_range": (
                        str(ontology.range_for_relation(terminal_relation))
                        if ontology is not None and terminal_relation
                        else ""
                    ),
                }
            )
        if not candidates or ranker is None:
            return selected, {
                "status": "not_applicable",
                "reason": "no_schema_candidate_or_ranker",
                "constraint": spec,
            }

        local_focus = constraint_focus_phrase(str(spec.get("focus", "")))
        documents = [
            " ".join(
                relation_label_from_id(item["first_relation"])
                + (
                    relation_label_from_id(item["second_relation"])
                    if item["second_relation"]
                    else []
                )
            )
            for item in candidates
        ]
        try:
            semantic_scores = ranker.score(local_focus, documents)
        except Exception as exc:
            return selected, {
                "status": "not_applicable",
                "reason": "constraint_property_ranking_failed",
                "constraint": spec,
                "error": str(exc),
            }
        if len(semantic_scores) != len(candidates):
            return selected, {
                "status": "not_applicable",
                "reason": "constraint_property_ranking_shape_mismatch",
                "constraint": spec,
            }

        for item, semantic_score in zip(candidates, semantic_scores):
            item["evidence"] = property_rank_evidence(
                first_relation=item["first_relation"],
                second_relation=item["second_relation"],
                spec=spec,
                terminal_range=item["terminal_range"],
                semantic_similarity=float(semantic_score),
                graph_score=float(item["graph"].score),
            )
        evidences = [item["evidence"] for item in candidates]
        eligible: list[dict[str, Any]] = []
        for item in candidates:
            evidence = item["evidence"]
            local_surface = _constraint_relation_has_surface_evidence(
                local_focus,
                item["first_relation"],
                item["second_relation"],
            )
            lexical = bool(
                local_surface
                or any(len(token) >= 5 for token in evidence.matched_tokens)
            )
            semantic = semantic_only_is_confident(
                evidence,
                [other for other in evidences if other is not evidence],
            )
            if lexical or semantic:
                item["local_surface_evidence"] = local_surface
                eligible.append(item)
        eligible.sort(
            key=lambda item: item["evidence"].rank_key,
            reverse=True,
        )
        if not eligible:
            return selected, {
                "status": "not_applicable",
                "reason": "no_confident_schema_property",
                "constraint": spec,
                "local_focus": local_focus,
            }

        selected_record: dict[str, Any] | None = None
        executed_records: list[dict[str, Any]] = []
        executions: list[dict[str, Any]] = []
        bounded = eligible[: max(1, int(execution_limit))]
        for item in bounded:
            sibling = item["graph"]
            execution, trace = self._execute_graph_candidate(sibling)
            repair = sibling.provenance.get("operator_attribute_repair", {})
            ids = list(execution.answer_ids if execution is not None else [])
            record = {
                **item,
                "execution": execution,
                "answer_ids": ids,
                "strict_subset": bool(ids)
                and set(ids) < set(selected.answer_ids),
            }
            executed_records.append(record)
            executions.append({
                "graph_id": sibling.graph_id,
                "first_relation": repair.get("first_relation", ""),
                "second_relation": repair.get("second_relation", ""),
                "answer_count": len(ids),
                "property_rank": list(item["evidence"].rank_key),
                "matched_tokens": list(item["evidence"].matched_tokens),
                "semantic_similarity": item["evidence"].semantic_similarity,
                "error": trace.get("error", ""),
            })
            if execution is not None and set(ids) and set(ids) < set(selected.answer_ids):
                selected_record = record
                break

        if selected_record is None:
            return selected, {
                "status": "not_applied",
                "reason": "repair_not_strict_answer_subset",
                "constraint": spec,
                "executions": executions,
            }
        selected_execution = selected_record["execution"]
        assert selected_execution is not None
        answer_ids = list(selected_execution.answer_ids)
        repaired_ids = set(answer_ids)
        original_ids = set(selected.answer_ids)
        safety_reason = answer_set_safety_reason(
            selected.answer_ids,
            answer_ids,
            operator_type=operator_type,
            source_answers=selected.answers,
            repaired_answers=selected_execution.answers,
        )
        nested_parent_preservation: dict[str, Any] | None = None
        if safety_reason == "nested_child_label_collapse":
            parent_ids = nested_label_parent_ids(
                selected.answer_ids,
                answer_ids,
                source_answers=selected.answers,
                repaired_answers=selected_execution.answers,
            )
            if len(parent_ids) == 1:
                preserved_ids = repaired_ids | set(parent_ids)
                answer_ids = [
                    answer_id
                    for answer_id in selected.answer_ids
                    if answer_id in preserved_ids
                ]
                source_answers_by_id = {
                    str(item.get("id", "")): item
                    for item in selected.answers
                    if isinstance(item, dict)
                }
                repaired_answers_by_id = {
                    str(item.get("id", "")): item
                    for item in selected_execution.answers
                    if isinstance(item, dict)
                }
                repaired_ids = set(answer_ids)
                nested_parent_preservation = {
                    "status": "unique_lexical_parent_preserved",
                    "parent_ids": parent_ids,
                    "child_ids": list(selected_execution.answer_ids),
                    "unrelated_source_answers_removed": len(original_ids - repaired_ids),
                }
                selected_execution = ExecutedGraph(
                    selected_execution.graph,
                    answer_ids,
                    [
                        repaired_answers_by_id.get(answer_id)
                        or source_answers_by_id.get(answer_id)
                        or {"id": answer_id, "label": answer_id}
                        for answer_id in answer_ids
                    ],
                    len(answer_ids),
                )
                safety_reason = ""
        if (
            not safety_reason
            and
            re.search(r"\(\s*s\s*\)", question, re.I)
            and len(repaired_ids) * 4 < len(original_ids)
        ):
            safety_reason = "explicit_plural_recall_guard"

        # When two similarly supported schema properties produce conflicting
        # strict subsets, neither result is safe enough to replace the current
        # answer set.  Execute only the small, near-tied ambiguity set.
        selected_evidence = selected_record["evidence"]
        if not safety_reason:
            for item in bounded:
                if item is selected_record or item.get("schema_variant"):
                    continue
                other_evidence = item["evidence"]
                ambiguous = bool(
                    set(selected_evidence.property_tokens)
                    != set(other_evidence.property_tokens)
                    and selected_evidence.family_score
                    == other_evidence.family_score
                    and selected_evidence.matched_tokens
                    == other_evidence.matched_tokens
                    and selected_evidence.first_matched_tokens
                    and other_evidence.first_matched_tokens
                    and abs(selected_evidence.score - other_evidence.score)
                    <= 0.30
                )
                if not ambiguous:
                    continue
                other = next(
                    (
                        record
                        for record in executed_records
                        if record["graph"] is item["graph"]
                    ),
                    None,
                )
                if other is None:
                    execution, trace = self._execute_graph_candidate(item["graph"])
                    ids = list(
                        execution.answer_ids if execution is not None else []
                    )
                    other = {
                        **item,
                        "execution": execution,
                        "answer_ids": ids,
                        "strict_subset": bool(ids)
                        and set(ids) < original_ids,
                    }
                    executed_records.append(other)
                    executions.append({
                        "graph_id": item["graph"].graph_id,
                        "first_relation": item["first_relation"],
                        "second_relation": item["second_relation"],
                        "answer_count": len(ids),
                        "property_rank": list(other_evidence.rank_key),
                        "matched_tokens": list(other_evidence.matched_tokens),
                        "semantic_similarity": other_evidence.semantic_similarity,
                        "ambiguity_probe": True,
                        "error": trace.get("error", ""),
                    })
                if (
                    other["strict_subset"]
                    and _property_subsets_conflict(
                        answer_ids,
                        other["answer_ids"],
                    )
                ):
                    safety_reason = "conflicting_property_subsets"
                    break

        if safety_reason:
            return selected, {
                "status": "not_applied",
                "reason": safety_reason,
                "constraint": spec,
                "executions": executions,
            }
        repaired_graph = deepcopy(selected_execution.graph)
        repaired_graph.provenance["late_explicit_constraint_repair"] = {
            "constraint": deepcopy(spec),
            "source_graph_id": selected.graph.graph_id,
            "candidate_count": len(eligible),
            "local_focus": local_focus,
            "property_rank": list(selected_evidence.rank_key),
        }
        if nested_parent_preservation is not None:
            repaired_graph.provenance["late_explicit_constraint_repair"][
                "nested_parent_preservation"
            ] = deepcopy(nested_parent_preservation)
        repaired = ExecutedGraph(
            repaired_graph,
            answer_ids,
            list(selected_execution.answers),
            len(answer_ids),
        )
        return repaired, {
            "status": "applied",
            "constraint": spec,
            "executions": executions,
            "answer_count": len(answer_ids),
            "local_focus": local_focus,
            "embedding_calls": 1,
            "model_used": False,
            "nested_parent_preservation": nested_parent_preservation,
        }

    def _operator_needs_ontology_repair(self, graph: QueryGraphCandidate) -> bool:
        ontology = getattr(self.grounder, "ontology", None)
        if ontology is None:
            return False
        relation_ids = set(getattr(ontology, "relation_ids", ()))
        return any(
            str(operator.get("attribute_relation_id", "")) not in relation_ids
            for operator in graph.operators
            if str(operator.get("type", "")).upper()
            in {
                "EQUAL",
                "GREATER_THAN",
                "GREATER_OR_EQUAL",
                "LESS_THAN",
                "LESS_OR_EQUAL",
                "TC",
                "ARGMIN",
                "ARGMAX",
            }
        )

    def _execute_graph_candidate(
        self,
        graph: QueryGraphCandidate,
        *,
        lookup_labels: bool = True,
    ) -> tuple[ExecutedGraph | None, dict[str, Any]]:
        started = time.monotonic()
        trace: dict[str, Any] = {"graph_id": graph.graph_id}
        try:
            notable_type_repairs = _normalize_notable_type_constraint(graph)
            if notable_type_repairs:
                trace["notable_type_constraint_repairs"] = notable_type_repairs
            implicit_extrema = _complete_singular_answer_extrema(
                graph,
                getattr(self.grounder, "ontology", None),
            )
            if implicit_extrema:
                trace["implicit_singular_answer_order"] = implicit_extrema
            extrema_repairs = _repair_event_temporal_extrema(
                graph,
                getattr(self.grounder, "ontology", None),
            )
            if extrema_repairs:
                trace["event_temporal_extrema_repairs"] = extrema_repairs
            _annotate_temporal_comparison_semantics(
                graph,
                getattr(self.grounder, "ontology", None),
            )
            _annotate_numeric_comparison_semantics(
                graph,
                getattr(self.grounder, "ontology", None),
            )
            _annotate_extrema_order_semantics(
                graph,
                getattr(self.grounder, "ontology", None),
            )
            binder = getattr(self, "relative_temporal_binder", None)
            if binder is not None and is_graph_reconstruction_graph(graph):
                # Reconstruction already spends the fixed execution budget.
                # Bind only local declared/ordinal/literal constraints here;
                # the relative-reference branch may issue an additional KG
                # query and is therefore deliberately excluded.
                local_question = str(
                    graph.provenance.get("original_question", "")
                )
                declared = binder.bind_declared_types(graph)
                ordinal = binder.bind_first(local_question, graph)
                ordinal["declared_type_constraints"] = declared
                literal = binder.bind_literal(local_question, graph)
                binding = (
                    literal
                    if literal.get("status") != "not_applicable"
                    else ordinal
                )
                if binding.get("status") != "not_applicable":
                    trace["relative_temporal_binding"] = binding
                if binding.get("status") == "unresolved":
                    raise ValueError(
                        "Relative time constraint could not bind to KG evidence"
                    )
            elif (
                binder is not None
                and not _is_failure_constrained_recovery_graph(graph)
            ):
                binding = binder.bind(graph.provenance.get("original_question", ""), graph)
                if binding["status"] != "not_applicable":
                    trace["relative_temporal_binding"] = binding
                if binding["status"] == "unresolved":
                    raise ValueError("Relative time constraint could not bind to KG evidence")
            graph.sparql = lower_sparql(graph, limit=self.answer_limit)
            trace["sparql"] = graph.sparql
            query_started = time.monotonic()
            rows = self.kg.execute(graph.sparql)
            trace["query_elapsed_seconds"] = round(
                time.monotonic() - query_started,
                6,
            )
        except Exception as exc:
            trace.update(
                {
                    "error": str(exc),
                    "error_details": _exception_trace_details(exc),
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                }
            )
            return None, trace

        answer_ids = _answer_values(rows, graph.answer_var)
        if (
            not answer_ids
            and not _is_failure_constrained_recovery_graph(graph)
            and not is_graph_reconstruction_graph(graph)
        ):
            sibling = _temporal_order_sibling_graph(graph)
            if sibling is not None:
                fallback_graph, changes = sibling
                try:
                    fallback_sparql = lower_sparql(
                        fallback_graph,
                        limit=self.answer_limit,
                    )
                    fallback_rows = self.kg.execute(fallback_sparql)
                    fallback_answer_ids = _answer_values(
                        fallback_rows,
                        fallback_graph.answer_var,
                    )
                    trace["temporal_order_sibling_retry"] = {
                        "changes": changes,
                        "sparql": fallback_sparql,
                        "answer_count": len(fallback_answer_ids),
                        "used": bool(fallback_answer_ids),
                    }
                    if fallback_answer_ids:
                        graph = fallback_graph
                        graph.sparql = fallback_sparql
                        rows = fallback_rows
                        answer_ids = fallback_answer_ids
                        trace["sparql"] = fallback_sparql
                except Exception as exc:
                    trace["temporal_order_sibling_retry"] = {
                        "changes": changes,
                        "used": False,
                        "error": str(exc),
                    }
        raw_labels: list[dict[str, str]] = []
        if answer_ids and lookup_labels:
            label_ids = answer_ids[:_SELECTOR_ANSWER_PREVIEW]
            trace.update(
                {
                    "label_lookup_count": len(label_ids),
                    "labels_truncated": len(answer_ids) > len(label_ids),
                }
            )
            label_started = time.monotonic()
            try:
                raw_labels = self.kg.labels(label_ids)
            except Exception as exc:
                trace.update(
                    {
                        "label_error": str(exc),
                        "label_error_details": _exception_trace_details(exc),
                    }
                )
            finally:
                trace["label_elapsed_seconds"] = round(
                    time.monotonic() - label_started,
                    6,
                )
        labels = _answer_labels(answer_ids, raw_labels)
        trace.update(
            {
                "row_count": len(rows),
                "answer_count": len(answer_ids),
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
        )
        return ExecutedGraph(graph, answer_ids, labels, len(rows)), trace

    def _postselect_scalar_answer_projection(
        self,
        selected: ExecutedGraph,
        *,
        question: str,
        relation_limit: int = 16,
    ) -> tuple[ExecutedGraph, dict[str, Any]]:
        """Replace a constrained scalar answer with its owner/entity projection."""
        ontology = getattr(self.grounder, "ontology", None)
        if ontology is None:
            return selected, {"status": "disabled_no_ontology"}
        scalar_ranges = {
            "type.datetime", "type.enumeration", "type.float", "type.int",
            "type.rawstring",
        }
        operator_types = {
            "ARGMAX", "ARGMIN", "EQUAL", "GREATER_THAN", "GREATER_OR_EQUAL",
            "LESS_THAN", "LESS_OR_EQUAL", "TC",
        }
        graph = selected.graph
        specifications: list[tuple[str, str]] = []
        owners: list[dict[str, Any]] = []
        for subject, scalar_relation, object_ in graph.triples:
            owner_var = str(subject)
            if (
                str(object_) != graph.answer_var
                or not owner_var.startswith("V")
                or str(ontology.range_for_relation(scalar_relation)) not in scalar_ranges
            ):
                continue
            constrained = any(
                str(operator.get("type", "")).upper() in operator_types
                and (
                    str(operator.get("input_var", "")) == graph.answer_var
                    or (
                        str(operator.get("input_var", "")) == owner_var
                        and str(operator.get("attribute_relation_id", ""))
                        == str(scalar_relation)
                    )
                )
                for operator in graph.operators
                if isinstance(operator, dict)
            )
            if not constrained:
                continue
            owner_type = str(ontology.domain_for_relation(scalar_relation))
            owner_relation_count = len(ontology.relations_for_domain(owner_type))
            relations = [
                relation
                for relation in ontology.relations_for_domain(owner_type)
                if relation != scalar_relation
                and (range_id := str(ontology.range_for_relation(relation)))
                and range_id not in scalar_ranges
                and not range_id.startswith("type.")
            ]
            if (
                0 < owner_relation_count <= max(1, int(relation_limit))
                and relations
            ):
                specifications.extend((owner_var, relation) for relation in relations)
                strategy = "bounded_cvt_relations"
            else:
                specifications.append((owner_var, ""))
                strategy = "direct_owner"
            owners.append({
                "owner_var": owner_var,
                "owner_type": owner_type,
                "strategy": strategy,
                "relation_count": owner_relation_count,
                "operator_constrained": bool(constrained),
            })
        specifications = list(dict.fromkeys(specifications))
        if not specifications:
            return selected, {"status": "not_applicable"}

        variables = {
            str(value)
            for triple in graph.triples
            for value in (triple[0], triple[2])
            if str(value).startswith("V") and str(value)[1:].isdigit()
        }
        next_variable = max((int(value[1:]) for value in variables), default=-1) + 1
        answer_ids: list[str] = []
        answers_by_id: dict[str, dict[str, str]] = {}
        executions: list[dict[str, Any]] = []
        for index, (owner_var, relation) in enumerate(specifications):
            candidate = deepcopy(graph)
            if relation:
                candidate.answer_var = f"V{next_variable + index}"
                candidate.triples.append([owner_var, relation, candidate.answer_var])
            else:
                candidate.answer_var = owner_var
            candidate.graph_id = f"{graph.graph_id}P{index}"
            candidate.sparql = ""
            candidate.provenance["original_question"] = question
            execution, trace = self._execute_graph_candidate(candidate)
            ids = list(execution.answer_ids if execution is not None else [])
            executions.append({
                "relation": relation,
                "answer_var": candidate.answer_var,
                "answer_count": len(ids),
                "error": trace.get("error", ""),
            })
            if execution is None:
                continue
            for answer_id in execution.answer_ids:
                if answer_id not in answer_ids:
                    answer_ids.append(answer_id)
            for answer in execution.answers:
                if isinstance(answer, dict) and str(answer.get("id", "")):
                    answers_by_id.setdefault(str(answer["id"]), answer)
        if not answer_ids:
            return selected, {
                "status": "no_projected_answers",
                "owners": owners,
                "executions": executions,
            }
        repaired_graph = deepcopy(graph)
        repaired_graph.provenance["scalar_answer_projection"] = {
            "owners": owners,
            "source_answer_var": graph.answer_var,
        }
        repaired = ExecutedGraph(
            repaired_graph,
            answer_ids,
            [
                answers_by_id.get(answer_id, {"id": answer_id, "label": answer_id})
                for answer_id in answer_ids
            ],
            len(answer_ids),
        )
        return repaired, {
            "status": "projected",
            "owners": owners,
            "executions": executions,
            "answer_count": len(answer_ids),
        }

    def _postselect_unlabeled_mediator_projection(
        self,
        selected: ExecutedGraph,
        *,
        question: str,
        relation_limit: int = 16,
    ) -> tuple[ExecutedGraph, dict[str, Any]]:
        """Project unlabeled low-degree mediator answers to labeled entities."""
        ontology = getattr(self.grounder, "ontology", None)
        if ontology is None or not selected.answers:
            return selected, {"status": "not_applicable"}
        if selected.graph.provenance.get("selection_answer_union"):
            return selected, {"status": "not_applicable_after_answer_union"}
        labeled_answers = [
            answer
            for answer in selected.answers
            if isinstance(answer, dict) and str(answer.get("id", ""))
        ]
        if not labeled_answers:
            return selected, {"status": "not_applicable"}
        unlabeled_fraction = sum(
            str(answer.get("label", "")) == str(answer.get("id", ""))
            for answer in labeled_answers
        ) / len(labeled_answers)
        if unlabeled_fraction < 0.25:
            return selected, {"status": "not_applicable"}
        scalar_ranges = {
            "type.datetime", "type.enumeration", "type.float", "type.int",
            "type.rawstring",
        }
        graph = selected.graph
        specifications: list[str] = []
        mediator_types: list[str] = []
        for _, incoming_relation, object_ in graph.triples:
            if str(object_) != graph.answer_var:
                continue
            mediator_type = str(ontology.range_for_relation(incoming_relation))
            relations = list(ontology.relations_for_domain(mediator_type))
            inherited_types = set(
                map(str, ontology.supertypes(mediator_type))
            ) if mediator_type else set()
            if (
                not mediator_type
                or mediator_type in scalar_ranges
                or mediator_type.startswith("type.")
                or "common.topic" in inherited_types
                or not (0 < len(relations) <= max(1, int(relation_limit)))
            ):
                continue
            mediator_types.append(mediator_type)
            for relation in relations:
                range_id = str(ontology.range_for_relation(relation))
                if range_id and range_id not in scalar_ranges and not range_id.startswith("type."):
                    specifications.append(relation)
        specifications = list(dict.fromkeys(specifications))
        if not specifications:
            return selected, {"status": "not_applicable"}
        focused_projection = False
        projection_ranking: list[dict[str, Any]] = []
        ranker = getattr(self.grounder, "ranker", None)
        if ranker is not None and len(specifications) > 1:
            focus = _answer_focus_text(question)
            # Use only the real predicate property names.  Variable aliases
            # such as V0/V1 are structural placeholders and carry no meaning.
            documents = [
                relation_label_from_id(relation)[-1]
                for relation in specifications
            ]
            try:
                scores = ranker.score(focus, documents)
                ranked = sorted(
                    zip(scores, specifications),
                    key=lambda item: (-float(item[0]), item[1]),
                )
                specifications = [relation for _, relation in ranked]
                projection_ranking = [
                    {"relation": relation, "score": float(score)}
                    for score, relation in ranked
                ]
                margin = (
                    float(ranked[0][0]) - float(ranked[1][0])
                    if len(ranked) > 1
                    else 1.0
                )
                # A nearly tied top score is ambiguous even for a small
                # mediator schema. Preserve the exhaustive projection unless
                # the target relation wins by a meaningful margin.
                focused_projection = margin >= 0.04
            except Exception:
                projection_ranking = []
        variables = {
            str(value)
            for triple in graph.triples
            for value in (triple[0], triple[2])
            if str(value).startswith("V") and str(value)[1:].isdigit()
        }
        next_variable = max((int(value[1:]) for value in variables), default=-1) + 1
        answer_ids: list[str] = []
        answers_by_id: dict[str, dict[str, str]] = {}
        executions: list[dict[str, Any]] = []
        for index, relation in enumerate(specifications):
            candidate = deepcopy(graph)
            candidate.answer_var = f"V{next_variable + index}"
            candidate.triples.append([graph.answer_var, relation, candidate.answer_var])
            candidate.graph_id = f"{graph.graph_id}M{index}"
            candidate.sparql = ""
            candidate.provenance["original_question"] = question
            execution, trace = self._execute_graph_candidate(candidate)
            ids = list(execution.answer_ids if execution is not None else [])
            executions.append({
                "relation": relation,
                "answer_count": len(ids),
                "error": trace.get("error", ""),
            })
            if execution is None:
                continue
            for answer_id in execution.answer_ids:
                if answer_id not in answer_ids:
                    answer_ids.append(answer_id)
            for answer in execution.answers:
                if isinstance(answer, dict) and str(answer.get("id", "")):
                    answers_by_id.setdefault(str(answer["id"]), answer)
            if focused_projection and answer_ids:
                break
        if not answer_ids:
            return selected, {
                "status": "no_projected_answers",
                "mediator_types": sorted(set(mediator_types)),
                "executions": executions,
            }
        repaired_graph = deepcopy(graph)
        repaired_graph.provenance["unlabeled_mediator_projection"] = {
            "mediator_types": sorted(set(mediator_types)),
            "source_answer_var": graph.answer_var,
            "unlabeled_fraction": unlabeled_fraction,
        }
        repaired = ExecutedGraph(
            repaired_graph,
            answer_ids,
            [
                answers_by_id.get(answer_id, {"id": answer_id, "label": answer_id})
                for answer_id in answer_ids
            ],
            len(answer_ids),
        )
        return repaired, {
            "status": "projected",
            "mediator_types": sorted(set(mediator_types)),
            "executions": executions,
            "answer_count": len(answer_ids),
            "focused_projection": focused_projection,
            "relation_ranking": projection_ranking,
        }

    def _prepare_fixed_execution_lanes(
        self,
        question: str,
        graphs: list[QueryGraphCandidate],
        *,
        decompositions: list[str],
        semantic_graphs: list[dict[str, Any]],
    ) -> tuple[
        list[QueryGraphCandidate],
        int,
        list[PreparedEntityTemplateQuery],
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any],
    ]:
        """Reserve entity templates before numeric repair in one fixed beam."""

        configured_budget = max(1, int(self.execution_budget))
        before = min(len(graphs), configured_budget)
        ordinary_graphs: list[QueryGraphCandidate] = graphs
        entity_queries: list[PreparedEntityTemplateQuery] = []
        entity_diagnostics: dict[str, Any] | None = None
        planner = getattr(self, "entity_kind_fixed_beam_planner", None)
        if planner is not None:
            try:
                plan = planner.prepare(
                    question=question,
                    graphs=graphs,
                    decomposition=decompositions,
                    semantic_graphs=semantic_graphs,
                    entities=linked_entities_for_question(
                        getattr(self.grounder, "gold_entities", {}),
                        question,
                    ),
                    ontology=getattr(self.grounder, "ontology", None),
                    execution_budget=configured_budget,
                    max_replacements=getattr(
                        self,
                        "entity_kind_fixed_beam_max_replacements",
                        3,
                    ),
                )
                ordinary_graphs = list(plan.ordinary_graphs)
                entity_queries = list(plan.template_queries)
                entity_diagnostics = dict(plan.diagnostics)
            except Exception as exc:
                # Cache/read failures preserve the untouched normal beam and
                # cannot manufacture an extra query.
                ordinary_graphs = graphs
                entity_diagnostics = {
                    "status": "error",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "query_count_before": before,
                    "query_count_after": before,
                    "additional_endpoint_queries": 0,
                    "additional_model_calls": 0,
                }
        reserved = len(entity_queries)
        ordinary_budget = max(0, configured_budget - reserved)
        execution_graphs, numeric_diagnostics = (
            self._prepare_fixed_numeric_temporal_slots(
                question,
                ordinary_graphs,
                execution_budget=ordinary_budget,
                protected_graph_ids=(
                    [str(entity_diagnostics.get("sentinel_graph_id", ""))]
                    if entity_diagnostics is not None
                    else []
                ),
            )
        )
        after = min(len(execution_graphs), ordinary_budget) + reserved
        coordination = {
            "status": "coordinated",
            "execution_budget": configured_budget,
            "query_count_before": before,
            "query_count_after": after,
            "entity_reserved_slots": reserved,
            "ordinary_execution_budget": ordinary_budget,
            "entity_precedes_numeric_preparation": True,
            "variants_excluded_from_ordinary_selector": True,
            "additional_model_calls": 0,
            "additional_endpoint_queries": 0,
            "fixed_beam_preserved": before == after,
        }
        return (
            execution_graphs,
            ordinary_budget,
            entity_queries,
            numeric_diagnostics,
            entity_diagnostics,
            coordination,
        )

    def _execute_entity_kind_queries(
        self,
        queries: list[PreparedEntityTemplateQuery],
    ) -> tuple[list[ExecutedGraph], list[dict[str, Any]]]:
        variants: list[ExecutedGraph] = []
        traces: list[dict[str, Any]] = []
        for query in queries:
            execution, trace = execute_entity_kind_prepared_query(query, self.kg)
            traces.append(trace)
            if execution is not None:
                variants.append(execution)
        return variants, traces

    def _prepare_fixed_numeric_temporal_slots(
        self,
        question: str,
        graphs: list[QueryGraphCandidate],
        *,
        execution_budget: int | None = None,
        protected_graph_ids: list[str] | tuple[str, ...] = (),
    ) -> tuple[list[QueryGraphCandidate], dict[str, Any] | None]:
        """Prepare equal-count execution-slot replacements when enabled.

        Returning the original list object with no diagnostics while disabled
        keeps the ablation path byte-for-byte free of graph mutations.
        """

        if not getattr(
            self,
            "numeric_temporal_slot_normalization_enabled",
            False,
        ):
            return graphs, None
        return prepare_fixed_execution_slots(
            question,
            graphs,
            getattr(self.grounder, "ontology", None),
            spec_resolver=_missing_constraint_spec,
            execution_budget=(
                self.execution_budget
                if execution_budget is None
                else max(0, int(execution_budget))
            ),
            max_replacements=getattr(
                self,
                "numeric_temporal_slot_max_replacements",
                3,
            ),
            expects_entity=_scalar_attribute_gate_expects_entity,
            protected_graph_ids=protected_graph_ids,
        )

    def _partition_fixed_numeric_temporal_executions(
        self,
        executions: list[ExecutedGraph],
    ) -> tuple[list[ExecutedGraph], list[ExecutedGraph]]:
        """Keep fixed-slot variants out of every ordinary selector/fallback."""

        if not getattr(
            self,
            "numeric_temporal_slot_normalization_enabled",
            False,
        ):
            return executions, []
        return partition_fixed_slot_executions(executions)

    def _select(
        self,
        question: str,
        executed: list[ExecutedGraph],
        *,
        decompositions: list[str] | None = None,
        anchor_surfaces: list[str] | None = None,
        semantic_graphs: list[dict[str, Any]] | None = None,
        fixed_slot_variants: list[ExecutedGraph] | None = None,
        entity_kind_variants: list[ExecutedGraph] | None = None,
    ) -> tuple[ExecutedGraph, dict[str, Any]]:
        anchor_echo_graph_ids: list[str] = []
        non_echo_items = [
            item
            for item in executed
            if not _answers_only_repeat_graph_anchors(item)
        ]
        if non_echo_items and len(non_echo_items) < len(executed):
            anchor_echo_graph_ids = [
                item.graph.graph_id
                for item in executed
                if _answers_only_repeat_graph_anchors(item)
            ]
            executed = non_echo_items

        repair_items = [
            item
            for item in executed
            if isinstance(
                item.graph.provenance.get("grounded_semantic", {}).get(
                    "grounding_repair"
                ),
                dict,
            )
        ]
        if repair_items:
            documents = []
            for item in repair_items:
                relation_text = " ".join(
                    relation.replace(".", " ").replace("_", " ")
                    for _, relation, _ in item.graph.triples
                )
                answer_text = " ".join(
                    str(answer.get("label", answer.get("id", "")))
                    for answer in item.answers[:_SELECTOR_ANSWER_PREVIEW]
                    if isinstance(answer, dict)
                )
                documents.append(
                    " ".join(
                        value
                        for value in (relation_text, answer_text)
                        if value
                    )
                )
            semantic_scores = self.grounder.ranker.score(question, documents)
            for item, semantic_score in zip(repair_items, semantic_scores):
                cardinality_penalty = 0.025 * math.log1p(len(item.answer_ids))
                adjustment = (0.35 * float(semantic_score)) - cardinality_penalty
                item.graph.score += adjustment
                item.graph.provenance["repair_answer_ranking"] = {
                    "embedding_score": float(semantic_score),
                    "cardinality_penalty": cardinality_penalty,
                    "adjustment": adjustment,
                }
        candidates = [
            {
                "graph_id": item.graph.graph_id,
                "triples": item.graph.triples,
                "answer_var": item.graph.answer_var,
                "operators": item.graph.operators,
                "answers": item.answers[:self.selector_answer_preview],
                "answer_ids": item.answer_ids[:self.selector_answer_preview],
                "answer_count": len(item.answer_ids),
                "answers_truncated": len(item.answer_ids) > self.selector_answer_preview,
                "rule_score": item.graph.score,
                "decomposition_source": str(
                    item.graph.provenance.get("grounded_semantic", {}).get(
                        "decomposition_source",
                        "unknown",
                    )
                ),
            }
            for item in executed
        ]

        def finalize_selection(
            selected: ExecutedGraph,
            payload: dict[str, Any],
        ) -> tuple[ExecutedGraph, dict[str, Any]]:
            hard_selected, hard_payload = _apply_hard_post_selection_gates(
                question,
                selected,
                executed,
                getattr(self.grounder, "ontology", None),
                payload,
            )
            path_selected = hard_selected
            path_payload = hard_payload
            path_selector = getattr(self, "path_alignment_selector", None)
            if path_selector is not None:
                path_selected, path_evidence = (
                    path_selector.challenge(
                        question=question,
                        semantic_graphs=list(semantic_graphs or []),
                        selected=hard_selected,
                        executions=executed,
                        prior_decision=hard_payload,
                    )
                )
                path_rejection_reason = ""
                if path_selected is not hard_selected:
                    if _ontology_head_answer_role_regression(
                        question,
                        hard_selected.graph,
                        path_selected.graph,
                        getattr(self.grounder, "ontology", None),
                    ):
                        path_rejection_reason = (
                            "ontology_head_answer_role_regression"
                        )
                    elif (
                        _plural_answer_request(question)
                        and len(hard_selected.answer_ids) > 1
                        and len(path_selected.answer_ids) == 1
                        and set(map(str, path_selected.answer_ids))
                        < set(map(str, hard_selected.answer_ids))
                        and not _explicit_extrema_request(question)
                    ):
                        path_rejection_reason = (
                            "explicit_plural_singleton_collapse"
                        )
                if path_rejection_reason:
                    rejected_graph_id = path_selected.graph.graph_id
                    path_selected = hard_selected
                    path_evidence = {
                        **path_evidence,
                        "status": "rejected_non_degrading_answer_guard",
                        "reason": path_rejection_reason,
                        "rejected_graph_id": rejected_graph_id,
                        "selected_graph_id": hard_selected.graph.graph_id,
                        "additional_model_calls": 0,
                        "additional_endpoint_queries": 0,
                    }
                path_payload = {
                    **hard_payload,
                    "path_alignment_gate": path_evidence,
                }
                if path_selected is not hard_selected:
                    path_payload["path_alignment_gate_source_graph_id"] = (
                        hard_selected.graph.graph_id
                    )
                    path_payload["selected_graph_id"] = (
                        path_selected.graph.graph_id
                    )
                    path_payload["reason_codes"] = [
                        *path_payload.get("reason_codes", []),
                        PATH_ALIGNMENT_REASON,
                    ]
            template_selector = getattr(self, "template_consensus_selector", None)
            consensus_selected = path_selected
            updated = path_payload
            if template_selector is not None:
                consensus_selected, consensus_evidence = (
                    template_selector.challenge(
                        question=question,
                        decompositions=list(decompositions or []),
                        anchor_surfaces=list(anchor_surfaces or []),
                        selected=path_selected,
                        executions=executed,
                        prior_decision=path_payload,
                    )
                )
                if (
                    consensus_selected is not path_selected
                    and _drops_question_supported_explicit_constraint(
                        question,
                        path_selected.graph,
                        consensus_selected.graph,
                    )
                ):
                    rejected_graph_id = consensus_selected.graph.graph_id
                    consensus_selected = path_selected
                    consensus_evidence = {
                        **consensus_evidence,
                        "status": "rejected_explicit_constraint_loss",
                        "rejected_graph_id": rejected_graph_id,
                        "selected_graph_id": path_selected.graph.graph_id,
                        "additional_model_calls": 0,
                        "additional_endpoint_queries": 0,
                    }
                if (
                    consensus_selected is not path_selected
                    and _ontology_head_answer_role_regression(
                        question,
                        path_selected.graph,
                        consensus_selected.graph,
                        getattr(self.grounder, "ontology", None),
                    )
                ):
                    rejected_graph_id = consensus_selected.graph.graph_id
                    consensus_selected = path_selected
                    consensus_evidence = {
                        **consensus_evidence,
                        "status": "rejected_ontology_head_regression",
                        "reason": "ontology_head_answer_role_regression",
                        "rejected_graph_id": rejected_graph_id,
                        "selected_graph_id": path_selected.graph.graph_id,
                        "additional_model_calls": 0,
                        "additional_endpoint_queries": 0,
                    }
                if (
                    consensus_selected is not path_selected
                    and _loses_distinct_conjunctive_entity_anchors(
                        question,
                        consensus_selected.graph,
                        linked_entities_for_question(
                            getattr(self.grounder, "gold_entities", {}),
                            question,
                        ),
                    )
                ):
                    rejected_graph_id = consensus_selected.graph.graph_id
                    consensus_selected = path_selected
                    consensus_evidence = {
                        **consensus_evidence,
                        "status": "rejected_distinct_anchor_loss",
                        "reason": "distinct_conjunctive_entity_anchor_loss",
                        "rejected_graph_id": rejected_graph_id,
                        "selected_graph_id": path_selected.graph.graph_id,
                        "additional_model_calls": 0,
                        "additional_endpoint_queries": 0,
                    }
                updated = {
                    **path_payload,
                    "source_verified_template_consensus_gate": consensus_evidence,
                }
                if consensus_selected is not path_selected:
                    updated["source_verified_template_consensus_gate_source_graph_id"] = (
                        path_selected.graph.graph_id
                    )
                    updated["selected_graph_id"] = consensus_selected.graph.graph_id
                    updated["reason_codes"] = [
                        *updated.get("reason_codes", []),
                        "source_verified_template_consensus_gate",
                    ]
            extrema_selector = getattr(
                self, "train_gold_path_extrema_selector", None
            )
            extrema_selected = consensus_selected
            finalized = updated
            if extrema_selector is not None:
                extrema_selected, extrema_evidence = extrema_selector.challenge(
                    question=question,
                    semantic_graphs=list(semantic_graphs or []),
                    selected=consensus_selected,
                    executions=executed,
                    prior_decision=updated,
                )
                finalized = {
                    **updated,
                    "train_gold_path_unique_extrema_gate": extrema_evidence,
                }
                if extrema_selected is not consensus_selected:
                    finalized[
                        "train_gold_path_unique_extrema_gate_source_graph_id"
                    ] = consensus_selected.graph.graph_id
                    finalized["selected_graph_id"] = (
                        extrema_selected.graph.graph_id
                    )
                    finalized["reason_codes"] = [
                        *finalized.get("reason_codes", []),
                        TRAIN_GOLD_EXTREMA_REASON,
                    ]

            fixed_selected = extrema_selected
            fixed_payload = finalized
            if getattr(
                self,
                "numeric_temporal_slot_normalization_enabled",
                False,
            ):
                fixed_selected, fixed_evidence = postselect_fixed_slot_execution(
                    extrema_selected,
                    list(fixed_slot_variants or []),
                    source_executions=executed,
                    question=question,
                    ontology=getattr(self.grounder, "ontology", None),
                    expects_entity=_scalar_attribute_gate_expects_entity,
                )
                fixed_payload = {
                    **finalized,
                    "fixed_slot_numeric_temporal_normalization": fixed_evidence,
                }
                if fixed_selected is not extrema_selected:
                    fixed_payload[
                        "fixed_slot_numeric_temporal_source_graph_id"
                    ] = extrema_selected.graph.graph_id
                    fixed_payload["selected_graph_id"] = fixed_selected.graph.graph_id
                    fixed_payload["reason_codes"] = [
                        *fixed_payload.get("reason_codes", []),
                        "fixed_slot_numeric_temporal_normalization",
                    ]

            entity_selected = fixed_selected
            entity_payload = fixed_payload
            if getattr(self, "entity_kind_fixed_beam_planner", None) is not None:
                entity_selected, entity_evidence = (
                    postselect_entity_kind_execution(
                        fixed_selected,
                        list(entity_kind_variants or []),
                    )
                )
                entity_payload = {
                    **fixed_payload,
                    "entity_kind_fixed_beam": entity_evidence,
                }
                if entity_selected is not fixed_selected:
                    entity_payload[
                        "entity_kind_fixed_beam_source_graph_id"
                    ] = fixed_selected.graph.graph_id
                    entity_payload["selected_graph_id"] = (
                        entity_selected.graph.graph_id
                    )
                    entity_payload["reason_codes"] = [
                        *entity_payload.get("reason_codes", []),
                        _ENTITY_KIND_FIXED_BEAM_REASON,
                    ]

            underanswer_selector = getattr(
                self, "underanswer_expansion_selector", None
            )
            if underanswer_selector is None:
                return entity_selected, entity_payload
            underanswer_selected, underanswer_evidence = (
                underanswer_selector.challenge(
                    question=question,
                    selected=entity_selected,
                    executions=executed,
                    prior_decision=entity_payload,
                )
            )
            underanswer_payload = {
                **entity_payload,
                "underanswer_expansion_gate": underanswer_evidence,
            }
            if underanswer_selected is not entity_selected:
                underanswer_payload[
                    "underanswer_expansion_gate_source_graph_id"
                ] = entity_selected.graph.graph_id
                underanswer_payload["selected_graph_id"] = (
                    underanswer_selected.graph.graph_id
                )
                underanswer_payload["reason_codes"] = [
                    *underanswer_payload.get("reason_codes", []),
                    str(
                        underanswer_evidence.get("reason_code")
                        or "morphological_underanswer_expansion_gate"
                    ),
                ]
            return underanswer_selected, underanswer_payload

        if self.selector_model is None:
            selected = max(
                executed,
                key=lambda item: (item.graph.score, -len(item.answer_ids), item.graph.graph_id),
            )
            return finalize_selection(
                selected,
                {
                    "status": "disabled",
                    "selected_graph_id": selected.graph.graph_id,
                    "reason_codes": ["selector_disabled_no_api_key"],
                    "candidates": candidates,
                    "excluded_anchor_echo_graph_ids": anchor_echo_graph_ids,
                },
            )
        decision: dict[str, Any] = {}
        rounds: list[dict[str, Any]] = []
        error = ""
        try:
            decision = select_candidates(
                self.selector_model,
                instruction=SELECTOR_INSTRUCTION,
                question=question,
                decompositions=list(decompositions or []),
                candidates=candidates,
                max_prompt_bytes=self.selector_max_prompt_bytes,
                rounds=rounds,
            )
            selected_id = str(decision.get("selected_graph_id", ""))
            selected = next(item for item in executed if item.graph.graph_id == selected_id)
            model_selected = selected
            rule_selected = max(
                executed,
                key=lambda item: (
                    item.graph.score,
                    -len(item.answer_ids),
                    item.graph.graph_id,
                ),
            )
            ontology = getattr(self.grounder, "ontology", None)
            answer_kind_guard = _rule_guard_crosses_answer_kind(
                question,
                model_selected,
                rule_selected,
            )
            wh_schema_kind_guard = _where_location_type_regression(
                question,
                model_selected.graph,
                rule_selected.graph,
                ontology,
            )
            operator_semantics_guard = bool(
                not _constraint_operator_semantics_compatible(
                    model_selected.graph,
                    rule_selected.graph,
                )
                and _graph_operator_has_question_evidence(
                    model_selected.graph,
                    question,
                )
            )
            relation_diversity_guard = _repeated_relation_model_guard(
                question,
                model_selected,
                rule_selected,
            )
            rule_guard_triggered = (
                rule_selected.graph.score - selected.graph.score
                >= _SELECTOR_RULE_GUARD_MARGIN
            )
            terminal_relation_guard = False
            terminal_relation_evidence: dict[str, Any] = {}
            if rule_guard_triggered:
                model_lexical = _terminal_relation_lexical_score(
                    question,
                    model_selected.graph,
                )
                rule_lexical = _terminal_relation_lexical_score(
                    question,
                    rule_selected.graph,
                )
                model_embedding = rule_embedding = None
                ranker = getattr(self.grounder, "ranker", None)
                documents = [
                    _terminal_relation_document(model_selected.graph, ontology),
                    _terminal_relation_document(rule_selected.graph, ontology),
                ]
                if ranker is not None and all(documents):
                    try:
                        scores = ranker.score(_answer_focus_text(question), documents)
                        if len(scores) == 2:
                            model_embedding, rule_embedding = map(float, scores)
                    except Exception as exc:
                        terminal_relation_evidence["embedding_error"] = str(exc)
                terminal_relation_guard = bool(
                    model_embedding is not None
                    and rule_embedding is not None
                    and model_embedding >= rule_embedding
                    and model_lexical - rule_lexical >= 0.2
                    and len(model_selected.answer_ids)
                    <= len(rule_selected.answer_ids)
                    and (
                        not _expected_answer_kind(question)
                        or _answer_value_kind(model_selected.answer_ids)
                        == _expected_answer_kind(question)
                    )
                    and (
                        not any(
                            str(operator.get("type", "")).upper()
                            not in {"", "NO_EQUAL"}
                            for operator in model_selected.graph.operators
                            if isinstance(operator, dict)
                        )
                        or _graph_operator_has_question_evidence(
                            model_selected.graph,
                            question,
                        )
                    )
                )
                terminal_relation_evidence.update(
                    {
                        "model_lexical": model_lexical,
                        "rule_lexical": rule_lexical,
                        "model_embedding": model_embedding,
                        "rule_embedding": rule_embedding,
                        "triggered": terminal_relation_guard,
                        "uses_graph_variables": False,
                    }
                )
            if (
                rule_guard_triggered
                and not answer_kind_guard
                and not wh_schema_kind_guard
                and not operator_semantics_guard
                and not relation_diversity_guard
                and not terminal_relation_guard
            ):
                decision = {
                    **decision,
                    "model_selected_graph_id": selected.graph.graph_id,
                    "selected_graph_id": rule_selected.graph.graph_id,
                    "reason_codes": [
                        *decision.get("reason_codes", []),
                        "local_rule_score_guard",
                    ],
                }
                selected = rule_selected
            elif rule_guard_triggered and (
                answer_kind_guard
                or wh_schema_kind_guard
                or operator_semantics_guard
                or relation_diversity_guard
                or terminal_relation_guard
            ):
                decision = {
                    **decision,
                    "reason_codes": [
                        *decision.get("reason_codes", []),
                        (
                            "answer_kind_guard"
                            if answer_kind_guard
                            else (
                                "wh_schema_kind_guard"
                                if wh_schema_kind_guard
                                else (
                                    "operator_semantics_guard"
                                    if operator_semantics_guard
                                    else (
                                        "relation_diversity_guard"
                                        if relation_diversity_guard
                                        else "terminal_relation_guard"
                                    )
                                )
                            )
                        ),
                    ],
                }
            if terminal_relation_evidence:
                decision["terminal_relation_evidence"] = terminal_relation_evidence
            duplicate_edge_repair = _duplicate_edge_challenger(
                selected,
                executed,
            )
            duplicate_edge_guard_triggered = duplicate_edge_repair is not None
            if duplicate_edge_repair is not None:
                decision = {
                    **decision,
                    "duplicate_edge_source_graph_id": selected.graph.graph_id,
                    "selected_graph_id": duplicate_edge_repair.graph.graph_id,
                    "reason_codes": [
                        *decision.get("reason_codes", []),
                        "exact_duplicate_edge_elimination",
                    ],
                }
                selected = duplicate_edge_repair
            score_gap = rule_selected.graph.score - model_selected.graph.score
            reverse_role_guard = _reverse_terminal_role_conflict(
                question,
                model_selected.graph,
                rule_selected.graph,
                getattr(self.grounder, "ontology", None),
            )
            if reverse_role_guard:
                selected = max(
                    (model_selected, rule_selected),
                    key=lambda item: (
                        float(item.graph.score),
                        str(item.graph.graph_id),
                    ),
                )
                decision = {
                    **decision,
                    "selected_graph_id": selected.graph.graph_id,
                    "reason_codes": [
                        *decision.get("reason_codes", []),
                        "reverse_terminal_role_guard",
                    ],
                }
            secondary = (
                rule_selected
                if selected.graph.graph_id == model_selected.graph.graph_id
                else model_selected
            )
            operator_semantics_compatible = (
                _constraint_operator_semantics_compatible(
                    model_selected.graph,
                    rule_selected.graph,
                )
            )
            if (
                rule_selected.graph.graph_id != model_selected.graph.graph_id
                and not duplicate_edge_guard_triggered
                and not reverse_role_guard
                and score_gap < _SELECTOR_ANSWER_UNION_MARGIN
                # Extrema lower to ORDER BY ... LIMIT 1.  Unioning two
                # alternative extrema graphs can only violate that single
                # result semantics; keep the selected alternative intact.
                and not _explicit_extrema_request(question)
                # Do not dilute a primary graph that encodes strictly more
                # joins or explicit constraints than the secondary graph.
                and not _primary_graph_is_more_constrained(
                    selected.graph,
                    secondary.graph,
                    question,
                )
                and operator_semantics_compatible
                and _answer_type_signatures_compatible(
                    rule_selected.graph,
                    model_selected.graph,
                    getattr(self.grounder, "ontology", None),
                )
                # Two alternative graphs must agree on at least one observed
                # answer before their sets are combined.  A low score margin
                # alone is not evidence that disjoint answers are jointly
                # correct, and historically polluted exact candidates.
                and _answer_union_has_semantic_support(
                    question,
                    selected,
                    secondary,
                    getattr(self.grounder, "ontology", None),
                )
                and max(
                    len(rule_selected.answer_ids),
                    len(model_selected.answer_ids),
                )
                <= self.selector_answer_preview
            ):
                selected = _union_executed_answers(selected, secondary)
                decision = {
                    **decision,
                    "selected_graph_id": selected.graph.graph_id,
                    "answer_union_graph_ids": [
                        selected.graph.provenance["selection_answer_union"][
                            "primary_graph_id"
                        ],
                        secondary.graph.graph_id,
                    ],
                    "reason_codes": [
                        *decision.get("reason_codes", []),
                        "low_margin_answer_union",
                    ],
                }
            return finalize_selection(
                selected,
                {"status": "selected", **decision, "candidates": candidates,
                 "rounds": rounds,
                 "excluded_anchor_echo_graph_ids": anchor_echo_graph_ids},
            )
        except Exception as exc:  # selector is advisory; rule score remains deterministic
            error = str(exc)
        selected = max(executed, key=lambda item: (item.graph.score, -len(item.answer_ids), item.graph.graph_id))
        return finalize_selection(
            selected,
            {
                "status": "fallback",
                "selected_graph_id": selected.graph.graph_id,
                "reason_codes": ["selector_fallback"],
                "error": error,
                "candidates": candidates,
                "rounds": rounds,
                "excluded_anchor_echo_graph_ids": anchor_echo_graph_ids,
            },
        )

    @staticmethod
    def _trace(stage: str, input_value: dict[str, Any], output: Any) -> dict[str, Any]:
        return {"stage": stage, "input": input_value, "output": output}

    @staticmethod
    def _graph_summary(graph: QueryGraphCandidate) -> dict[str, Any]:
        return {
            "graph_id": graph.graph_id,
            "triples": graph.triples,
            "answer_var": graph.answer_var,
            "operators": graph.operators,
            "score": graph.score,
            "provenance": graph.provenance,
        }

    @staticmethod
    def _result(question: str, traces: list[dict[str, Any]], *, failure: str) -> dict[str, Any]:
        return {
            "question": question,
            "answer_ids": [],
            "answers": [],
            "selected_graph": None,
            "failure": failure,
            "traces": traces,
        }


def _exception_trace_details(exc: Exception) -> dict[str, Any]:
    details = getattr(exc, "trace_details", None)
    if callable(details):
        value = details()
        if isinstance(value, dict):
            return value
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _validate_semantic_item_coverage(
    decomposition: list[str],
    semantic_graph: dict[str, Any],
) -> None:
    """Require one executable Semantic path for every path-level plan item.

    This is deliberately structural: it does not inspect question words,
    entities, relations, dates, or numeric literals.  The decomposition
    contract defines each array item as one complete entity-centric path, so a
    smaller Semantic path set necessarily dropped an explicit constraint.
    """
    expected = sum(bool(str(item).strip()) for item in decomposition)
    actual = sum(
        isinstance(item, dict)
        for item in semantic_graph.get("semantic_paths", [])
    )
    if actual < expected:
        raise ContractError(
            f"semantic output covers {actual} of {expected} decomposition paths"
        )


def _compose_output_schema(payload: dict[str, Any]) -> dict[str, Any]:
    paths = [item for item in payload.get("semantic_paths", []) if isinstance(item, dict)]
    path_ids = list(
        dict.fromkeys(str(path.get("id", "")) for path in paths if str(path.get("id", "")))
    )
    variables = list(
        dict.fromkeys(
            str(step.get("to", ""))
            for path in paths
            for step in path.get("steps", [])
            if isinstance(step, dict) and str(step.get("to", ""))
        )
    )
    path_ref = {"type": "string", "enum": path_ids}
    variable_ref = {"type": "string", "enum": variables}
    temporal_values = list(
        dict.fromkeys(
            ["NOW", *re.findall(r"\b(?:1|2)[0-9]{3}\b", str(payload.get("question", "")))]
        )
    )
    attribute_relation = {
        "type": "array",
        "items": {"type": "string"},
    }
    operator_schema = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["AND", "TC", "ARGMAX", "ARGMIN"],
            },
            "inputs": {
                "type": "array",
                "items": path_ref,
                "minItems": 1,
                "maxItems": len(path_ids),
            },
            "input_var": variable_ref,
            "attribute_relation_label": attribute_relation,
            "value": {
                "type": "string",
                "enum": temporal_values,
            },
        },
        "required": ["type"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "selected_paths": {
                "type": "array",
                "items": path_ref,
                "minItems": 1,
            },
            "variable_equalities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "left": variable_ref,
                        "right": variable_ref,
                    },
                    "required": ["left", "right"],
                    "additionalProperties": False,
                },
            },
            "operators": {
                "type": "array",
                "items": operator_schema,
            },
            "answer_var": variable_ref,
        },
        "required": [
            "selected_paths",
            "variable_equalities",
            "operators",
            "answer_var",
        ],
        "additionalProperties": False,
    }


def _compose_graph_output_schema() -> dict[str, Any]:
    entity = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^E[0-9]+$"},
            "surface": {"type": "string"},
        },
        "required": ["id", "surface"],
        "additionalProperties": False,
    }
    triple = {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "pattern": "^(?:E|V)[0-9]+$"},
            "relation_label": {"type": "array", "items": {"type": "string"}},
            "object": {"type": "string", "pattern": "^(?:E|V)[0-9]+$"},
        },
        "required": ["subject", "relation_label", "object"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": entity, "minItems": 1},
            "triples": {"type": "array", "items": triple, "minItems": 1},
            "answer_var": {"type": "string", "pattern": "^V[0-9]+$"},
        },
        "required": ["entities", "triples", "answer_var"],
        "additionalProperties": False,
    }


def _operator_output_schema(compose_output: dict[str, Any]) -> dict[str, Any]:
    variables = sorted(
        {
            str(triple[field])
            for triple in compose_output.get("triples", [])
            for field in ("subject", "object")
            if str(triple.get(field, "")).startswith("V")
        }
    )
    variable = {"type": "string", "enum": variables}
    operator = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": [
                    "COUNT",
                    "ARGMAX",
                    "ARGMIN",
                    "GREATER_THAN",
                    "GREATER_OR_EQUAL",
                    "LESS_THAN",
                    "LESS_OR_EQUAL",
                    "EQUAL",
                    "TC",
                    "NO_EQUAL",
                ],
            },
            "inputs": {"type": "array", "items": {"type": "string"}},
            "input_var": variable,
            "attribute_relation_label": {"type": "array", "items": {"type": "string"}},
            "attribute_relation_labels": {"type": "array", "items": {"type": "string"}},
            "value": {"type": "string"},
            "value_type": {"type": "string"},
        },
        "required": [
            "type",
            "inputs",
            "input_var",
            "attribute_relation_label",
            "attribute_relation_labels",
            "value",
            "value_type",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"operators": {"type": "array", "items": operator}},
        "required": ["operators"],
        "additionalProperties": False,
    }


def _normalize_compose_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Tolerate the deployed adapter's legacy extrema fields.

    The current contract represents ARGMAX/ARGMIN with an input variable and
    attribute relation. The deployed checkpoint sometimes adds a temporal
    ``value`` field to those operators; that field is a legacy artifact and
    makes the resulting graph semantically invalid, so discard only that
    malformed operator rather than failing the whole candidate.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("operators"), list):
        return raw
    normalized = dict(raw)
    normalized["operators"] = [
        operator
        for operator in raw["operators"]
        if not (
            isinstance(operator, dict)
            and str(operator.get("type", "")).upper() in {"ARGMAX", "ARGMIN"}
            and "value" in operator
        )
    ]
    return normalized


def _normalize_compose_graph_output(
    raw: dict[str, Any],
    compose_input: dict[str, Any],
) -> dict[str, Any]:
    """Restore and canonicalize entity surfaces using grounded path topology."""
    entities = raw.get("entities") if isinstance(raw, dict) else None
    triples = raw.get("triples") if isinstance(raw, dict) else None
    if (
        not isinstance(entities, list)
        or not entities
        or not isinstance(triples, list)
    ):
        return raw

    entity_ids: list[str] = []
    entity_surfaces: dict[str, str] = {}
    if all(isinstance(entity, str) for entity in entities):
        entity_ids = [str(entity) for entity in entities]
    elif all(isinstance(entity, dict) for entity in entities):
        for entity in entities:
            entity_id = str(entity.get("id", ""))
            surface = str(entity.get("surface", "")).strip()
            if not entity_id or not surface:
                return raw
            entity_ids.append(entity_id)
            entity_surfaces[entity_id] = surface
    else:
        return raw

    anchors = {
        str(anchor.get("id", "")): str(anchor.get("surface", "")).strip()
        for anchor in compose_input.get("anchors", [])
        if isinstance(anchor, dict)
    }
    surfaces: dict[str, str] = {}
    conflicts: set[str] = set()
    for path in compose_input.get("semantic_paths", []):
        if not isinstance(path, dict):
            continue
        steps = path.get("steps", [])
        if not isinstance(steps, list) or not steps or not isinstance(steps[0], dict):
            continue
        step = steps[0]
        surface = anchors.get(str(path.get("anchor_ref", "")), "")
        if not surface:
            continue
        endpoint = "subject" if str(step.get("direction", "")) == "forward" else "object"
        relation_key = _relation_label_key(step.get("relation_label", []))
        matches = {
            str(triple.get(endpoint, ""))
            for triple in triples
            if isinstance(triple, dict)
            and _relation_label_key(triple.get("relation_label", [])) == relation_key
            and str(triple.get(endpoint, "")) in entity_ids
        }
        if len(matches) != 1:
            continue
        entity_id = next(iter(matches))
        if entity_id in surfaces and surfaces[entity_id] != surface:
            conflicts.add(entity_id)
        else:
            surfaces[entity_id] = surface

    # An already canonical surface can be retained when it is not the endpoint
    # of a first-hop path, but every entity must still be an anchor surface.
    known_anchor_surfaces = set(anchors.values())
    for entity_id, surface in entity_surfaces.items():
        if _surface_key(surface) in {_surface_key(item) for item in known_anchor_surfaces}:
            surfaces.setdefault(entity_id, next(
                item for item in known_anchor_surfaces if _surface_key(item) == _surface_key(surface)
            ))
    if conflicts or set(surfaces) != set(entity_ids):
        return raw
    normalized = deepcopy(raw)
    normalized["entities"] = [
        {"id": entity_id, "surface": surfaces[entity_id]}
        for entity_id in entity_ids
    ]
    return normalized


def _validate_graph_compose_candidate(
    value: Any,
    grounded: GroundedSemanticCandidate,
) -> dict[str, Any]:
    normalized = _normalize_compose_graph_output(value, grounded.compose_input)
    validated = validate_compose_graph_output(normalized)
    # Contract validation alone cannot detect an entity alias that will fail
    # against the current Grounding bindings. Compile the candidate before
    # allowing it into the graph beam.
    build_query_graph_v2(grounded, validated, graph_id="validation")
    return validated


def _validate_legacy_compose_candidate(
    value: Any,
    grounded: GroundedSemanticCandidate,
) -> dict[str, Any]:
    normalized = _normalize_compose_output(value)
    validated = validate_compose_output(normalized, grounded.compose_input)
    build_query_graph(grounded, validated, graph_id="validation")
    return validated


def _relation_label_key(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(part).strip().casefold() for part in value)


def _surface_key(value: Any) -> str:
    return " ".join(str(value).casefold().split())


_RANGE_OPERATOR_TYPES = {
    "GREATER_THAN",
    "GREATER_OR_EQUAL",
    "LESS_THAN",
    "LESS_OR_EQUAL",
}
_TEMPORAL_RELATION_SUFFIXES = {
    "from",
    "from_date",
    "start",
    "start_date",
    "begin",
    "begin_date",
    "to",
    "to_date",
    "end",
    "end_date",
    "finish",
    "finish_date",
}


def _normalize_operator_intent(
    question: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    """Correct unsupported numeric inequality guesses from question semantics."""
    operators = value.get("operators") if isinstance(value, dict) else None
    if not isinstance(operators, list):
        return value
    explicit_comparison = bool(
        re.search(
            r"\b(?:less|fewer|more|greater|larger|smaller|under|over|below|above|before|after|at least|at most|no more than|no less than)\b",
            str(question),
            re.I,
        )
    )
    if explicit_comparison:
        return value
    normalized = deepcopy(value)
    changed = False
    for operator in normalized.get("operators", []):
        if not isinstance(operator, dict):
            continue
        if (
            str(operator.get("type", "")).upper() in _RANGE_OPERATOR_TYPES
            and str(operator.get("value_type", "")).casefold()
            in {"number", "float", "integer"}
            and re.fullmatch(r"-?\d+(?:\.\d+)?", str(operator.get("value", "")).strip())
        ):
            operator["type"] = "EQUAL"
            changed = True
    return normalized if changed else value


def _infer_missing_extrema_operator(
    question: str,
    compose_output: dict[str, Any],
    value: dict[str, Any],
) -> dict[str, Any]:
    """Add a missing extrema operation from task-general comparative language."""
    operators = value.get("operators") if isinstance(value, dict) else None
    if not isinstance(operators, list) or any(
        str(operator.get("type", "")).upper() in {"ARGMAX", "ARGMIN"}
        for operator in operators
        if isinstance(operator, dict)
    ):
        return value
    text = str(question)
    maximum = bool(
        re.search(r"\b(?:latest|last|most recent|biggest|largest|highest|maximum)\b", text, re.I)
    )
    minimum = bool(
        re.search(r"\b(?:earliest|least|smallest|lowest|minimum|first)\b", text, re.I)
        and not re.search(r"\bfirst name\b", text, re.I)
    )
    if maximum == minimum:
        return value
    answer_var = str(compose_output.get("answer_var", ""))
    if not answer_var.startswith("V"):
        return value
    normalized = deepcopy(value)
    normalized["operators"] = [
        *normalized.get("operators", []),
        {
            "type": "ARGMAX" if maximum else "ARGMIN",
            "inputs": [],
            "input_var": answer_var,
            "attribute_relation_label": [str(question)],
            "attribute_relation_labels": [],
            "value": "",
            "value_type": "",
        },
    ]
    return normalized


def _normalize_operator_output(
    value: dict[str, Any],
    compose_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Remove redundant temporal ranges and subject-derived equality constraints."""
    operators = value.get("operators") if isinstance(value, dict) else None
    if not isinstance(operators, list):
        return value

    entity_token_sets = [
        _normalized_text_tokens(entity.get("surface", ""))
        for entity in (compose_output or {}).get("entities", [])
        if isinstance(entity, dict)
    ]
    temporal_constraints = {
        (
            str(operator.get("input_var", "")),
            str(operator.get("value", "")).strip().casefold(),
            family,
        )
        for operator in operators
        if isinstance(operator, dict)
        and str(operator.get("type", "")).upper() == "TC"
        and (family := _temporal_relation_family(operator)) is not None
    }
    normalized_operators = [
        operator
        for operator in operators
        if not (
            isinstance(operator, dict)
            and (
                (
                    str(operator.get("type", "")).upper() in _RANGE_OPERATOR_TYPES
                    and (
                        str(operator.get("input_var", "")),
                        str(operator.get("value", "")).strip().casefold(),
                        _temporal_relation_family(operator),
                    )
                    in temporal_constraints
                )
                or (
                    str(operator.get("type", "")).upper() == "EQUAL"
                    and _is_subject_derived_equality(operator, entity_token_sets)
                )
            )
        )
    ]
    # Remove exact duplicates and constraints that cannot have a meaningful
    # effect. Conflicting extrema are safer to drop than to execute together.
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for operator in normalized_operators:
        operator_type = str(operator.get("type", "")).upper()
        if operator_type == "TC" and not str(operator.get("value", "")).strip():
            continue
        key = json.dumps(operator, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(operator)

    extrema_types: dict[tuple[str, str], set[str]] = {}
    for operator in deduplicated:
        operator_type = str(operator.get("type", "")).upper()
        if operator_type in {"ARGMAX", "ARGMIN"}:
            key = (
                str(operator.get("input_var", "")),
                _operator_relation_key(operator),
            )
            extrema_types.setdefault(key, set()).add(operator_type)
    conflicting_extrema = {
        key for key, types in extrema_types.items() if len(types) > 1
    }
    equal_values: dict[tuple[str, str], set[str]] = {}
    for operator in deduplicated:
        if str(operator.get("type", "")).upper() == "EQUAL":
            key = (
                str(operator.get("input_var", "")),
                _operator_relation_key(operator),
            )
            equal_values.setdefault(key, set()).add(str(operator.get("value", "")))
    conflicting_equalities = {
        key for key, values in equal_values.items() if len(values) > 1
    }
    normalized_operators = [
        operator
        for operator in deduplicated
        if not (
            str(operator.get("type", "")).upper() in {"ARGMAX", "ARGMIN"}
            and (
                str(operator.get("input_var", "")),
                _operator_relation_key(operator),
            ) in conflicting_extrema
        )
        and not (
            str(operator.get("type", "")).upper() == "EQUAL"
            and (
                str(operator.get("input_var", "")),
                _operator_relation_key(operator),
            ) in conflicting_equalities
        )
    ]
    if len(normalized_operators) == len(operators) and normalized_operators == operators:
        return value
    normalized = deepcopy(value)
    normalized["operators"] = normalized_operators
    return normalized


def _is_subject_derived_equality(
    operator: dict[str, Any],
    entity_token_sets: list[set[str]],
) -> bool:
    value_tokens = _normalized_text_tokens(operator.get("value", ""))
    return bool(value_tokens) and any(
        value_tokens <= entity_tokens
        for entity_tokens in entity_token_sets
    )


def _normalized_text_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _temporal_relation_family(operator: dict[str, Any]) -> tuple[str, ...] | None:
    label = operator.get("attribute_relation_label", [])
    if not label:
        label = operator.get("attribute_relation_labels", [])
    if not isinstance(label, list) or len(label) < 2:
        return None
    normalized = tuple(
        re.sub(r"[^a-z0-9]+", "_", str(part).strip().casefold()).strip("_")
        for part in label
    )
    if normalized[-1] not in _TEMPORAL_RELATION_SUFFIXES:
        return None
    return normalized[:-1]


def _operator_relation_key(operator: dict[str, Any]) -> str:
    label = operator.get("attribute_relation_label", [])
    if not label:
        label = operator.get("attribute_relation_labels", [])
    if isinstance(label, list):
        return " ".join(
            part
            for item in label
            for part in str(item).casefold().replace("_", " ").split()
        )
    return str(label).casefold().replace("_", " ").strip()


def _apply_operator_prediction(
    compose_output: dict[str, Any],
    operator_prediction: dict[str, Any],
    semantic_graph: dict[str, Any],
) -> dict[str, Any]:
    """Apply GLM operators and materialize AND groups as path intersections."""
    output = deepcopy(compose_output)
    selected_paths = [str(path_id) for path_id in output["selected_paths"]]
    equalities = [dict(item) for item in output.get("variable_equalities", [])]
    path_outputs = {
        str(path.get("id", "")): str(path.get("path_output_var", ""))
        for path in semantic_graph.get("semantic_paths", [])
        if isinstance(path, dict)
    }

    for group_index, group in enumerate(operator_prediction.get("and_groups", [])):
        variables: list[str] = []
        for path_id in group:
            path_id = str(path_id)
            if path_id not in selected_paths:
                selected_paths.append(path_id)
            variable = path_outputs.get(path_id, "")
            if variable and variable not in variables:
                variables.append(variable)
        if group_index == 0 and variables:
            output["answer_var"] = variables[0]
        for variable in variables[1:]:
            pair = {variables[0], variable}
            if not any(
                {str(item.get("left", "")), str(item.get("right", ""))} == pair
                for item in equalities
            ):
                equalities.append({"left": variables[0], "right": variable})

    output["selected_paths"] = selected_paths
    output["variable_equalities"] = equalities
    output["operators"] = deepcopy(operator_prediction.get("operators", []))
    return output


def _fallback_compose_output(
    semantic_graph: dict[str, Any],
    operator_prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the minimum valid composition when the compose model is unavailable."""
    paths = [
        path
        for path in semantic_graph.get("semantic_paths", [])
        if isinstance(path, dict) and str(path.get("id", ""))
    ]
    if not paths:
        raise ContractError("cannot compose a semantic graph without paths")
    selected_paths = [str(paths[0]["id"])]
    output = {
        "selected_paths": selected_paths,
        "variable_equalities": [],
        "operators": [],
        "answer_var": str(paths[0]["path_output_var"]),
    }
    if operator_prediction is not None:
        output = _apply_operator_prediction(
            output,
            operator_prediction,
            semantic_graph,
        )
    return output


def _fallback_compose_graph_output(compose_input: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal graph_v2 candidate from the first grounded semantic path."""
    anchors = [item for item in compose_input.get("anchors", []) if isinstance(item, dict)]
    paths = [item for item in compose_input.get("semantic_paths", []) if isinstance(item, dict)]
    if not anchors or not paths:
        raise ContractError("cannot build graph fallback without anchors and semantic paths")
    anchor_ids: dict[str, str] = {}
    for index, anchor in enumerate(anchors):
        anchor_id = str(anchor.get("id", ""))
        if not anchor_id or not str(anchor.get("surface", "")).strip():
            raise ContractError("semantic path fallback contains an invalid anchor")
        anchor_ids[anchor_id] = f"E{index}"
    if len(anchor_ids) != len(anchors):
        raise ContractError("semantic path fallback contains duplicate anchors")
    selected = paths[0]
    local_vars = [
        str(step.get("to", ""))
        for step in selected.get("steps", [])
        if isinstance(step, dict) and str(step.get("to", ""))
    ]
    global_vars = {value: f"V{index}" for index, value in enumerate(local_vars)}
    triples: list[dict[str, Any]] = []
    for step in selected.get("steps", []):
        if not isinstance(step, dict):
            continue
        source = str(step.get("from", ""))
        target = str(step.get("to", ""))
        source_ref = anchor_ids.get(source, global_vars.get(source, ""))
        target_ref = global_vars.get(target, "")
        if not source_ref or not target_ref:
            raise ContractError("semantic path fallback contains an unknown reference")
        if str(step.get("direction", "")) == "forward":
            subject, object_ = source_ref, target_ref
        else:
            subject, object_ = target_ref, source_ref
        triples.append(
            {
                "subject": subject,
                "relation_label": [str(item) for item in step.get("relation_label", [])],
                "object": object_,
            }
        )
    if not triples:
        raise ContractError("semantic path fallback contains no steps")
    answer_var = global_vars.get(str(selected.get("path_output_var", "")), "")
    if not answer_var:
        raise ContractError("semantic path fallback has no answer variable")
    return {
        "entities": [
            {"id": anchor_ids[str(anchor["id"])], "surface": str(anchor["surface"])}
            for anchor in anchors
        ],
        "triples": triples,
        "answer_var": answer_var,
    }


def _deterministic_terminal_intersection_output(
    compose_input: dict[str, Any],
) -> dict[str, Any]:
    """Compose independently anchored paths by unifying their terminal candidate.

    This is a bounded local candidate, not a replacement for Compose.  It only
    reuses model-produced paths and relation labels and therefore does not alter
    any model request/response contract or invent schema edges.
    """
    anchors = [item for item in compose_input.get("anchors", []) if isinstance(item, dict)]
    paths = [item for item in compose_input.get("semantic_paths", []) if isinstance(item, dict)]
    if not anchors or len(paths) < 2:
        raise ContractError("terminal intersection requires at least two semantic paths")
    anchor_ids: dict[str, str] = {}
    for index, anchor in enumerate(anchors):
        anchor_id = str(anchor.get("id", ""))
        if not anchor_id or not str(anchor.get("surface", "")).strip():
            raise ContractError("terminal intersection contains an invalid anchor")
        anchor_ids[anchor_id] = f"E{index}"
    if len(anchor_ids) != len(anchors):
        raise ContractError("terminal intersection contains duplicate anchors")

    variable_ids: dict[str, str] = {}
    next_variable = 1
    for path in paths:
        terminal = str(path.get("path_output_var", ""))
        if not terminal:
            raise ContractError("semantic path has no terminal variable")
        existing = variable_ids.get(terminal)
        if existing not in {None, "V0"}:
            raise ContractError("semantic terminal was already assigned as an intermediate")
        variable_ids[terminal] = "V0"
    for path in paths:
        for step in path.get("steps", []):
            if not isinstance(step, dict):
                continue
            for ref in (str(step.get("from", "")), str(step.get("to", ""))):
                if not ref or ref in anchor_ids or ref in variable_ids:
                    continue
                variable_ids[ref] = f"V{next_variable}"
                next_variable += 1

    triples: list[dict[str, Any]] = []
    for path in paths:
        for step in path.get("steps", []):
            if not isinstance(step, dict):
                continue
            source = str(step.get("from", ""))
            target = str(step.get("to", ""))
            source_ref = anchor_ids.get(source, variable_ids.get(source, ""))
            target_ref = anchor_ids.get(target, variable_ids.get(target, ""))
            if not source_ref or not target_ref:
                raise ContractError("terminal intersection contains an unknown reference")
            if str(step.get("direction", "")) == "forward":
                subject, object_ = source_ref, target_ref
            else:
                subject, object_ = target_ref, source_ref
            triples.append({
                "subject": subject,
                "relation_label": [str(item) for item in step.get("relation_label", [])],
                "object": object_,
            })
    if not triples:
        raise ContractError("terminal intersection contains no triples")
    return {
        "entities": [
            {"id": anchor_ids[str(anchor["id"])], "surface": str(anchor["surface"])}
            for anchor in anchors
        ],
        "triples": triples,
        "answer_var": "V0",
    }


def _failure_constraint_relation_score(
    question: str,
    first_relation: str,
    second_relation: str,
    *,
    support: int,
    semantic_rank: int,
) -> float:
    """Rank a rejected-Compose scalar extension without a model or KG call."""
    focus = str((_missing_constraint_spec(question) or {}).get("focus", question))
    question_tokens = {
        _semantic_word_key(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", focus)
    }
    property_tokens: set[str] = set()
    for relation_id in (first_relation, second_relation):
        if not relation_id:
            continue
        property_tokens.update(
            _semantic_word_key(token)
            for token in re.findall(
                r"[A-Za-z0-9]+",
                relation_id.rsplit(".", 1)[-1].replace("_", " "),
            )
        )
    content_overlap = len(
        (property_tokens - _GENERIC_SCALAR_TOKENS) & question_tokens
    )
    score = (3.0 * content_overlap) + (0.5 * max(1, support))
    score -= 0.001 * max(1, semantic_rank)

    leaf = (second_relation or first_relation).rsplit(".", 1)[-1].casefold()
    is_start = leaf in {"from", "from_date", "start", "start_date", "begin", "begin_date"}
    is_end = leaf in {"to", "to_date", "end", "end_date", "finish", "finish_date"}
    explicit_end = bool(
        re.search(r"\b(?:end|ended|ending|finish|finished|until|through)\b", question, re.I)
    )
    explicit_start = bool(
        re.search(r"\b(?:begin|began|start|started|starting|since|from)\b", question, re.I)
    )
    latest = bool(
        re.search(r"\b(?:latest|last|most recent)\b", question, re.I)
        and not re.search(r"\blast name\b", question, re.I)
    )
    earliest = bool(
        re.search(r"\b(?:earliest|first)\b", question, re.I)
        and not re.search(r"\bfirst name\b", question, re.I)
    )
    if explicit_end:
        score += 4.0 if is_end else (-4.0 if is_start else 0.0)
    elif explicit_start:
        score += 4.0 if is_start else (-4.0 if is_end else 0.0)
    elif latest and not earliest:
        score += 2.0 if is_end else (-2.0 if is_start else 0.0)
    elif earliest and not latest:
        score += 2.0 if is_start else (-2.0 if is_end else 0.0)
    elif is_start or is_end:
        # A comparison without a phase cue uses the interval beginning.
        score += 1.0 if is_start else -1.0
    return score


def _ontology_types_compatible(ontology: Any, left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    supertypes = getattr(ontology, "supertypes", None)
    if not callable(supertypes):
        return False
    return left in set(supertypes(right)) or right in set(supertypes(left))


_HARD_SCALAR_SCHEMA_TYPES = {
    "type.boolean",
    "type.datetime",
    "type.enumeration",
    "type.float",
    "type.int",
    "type.key",
    "type.rawstring",
    "type.text",
    "type.uri",
}
_HARD_META_SCHEMA_TYPES = {
    "type.domain",
    "type.namespace",
    "type.property",
    "type.type",
}


def _query_graph_hard_schema_conflicts(
    graph: QueryGraphCandidate,
    ontology: Any,
) -> list[dict[str, Any]]:
    """Return only schema contradictions that cannot denote one RDF value.

    Ordinary entity types are intentionally never compared with one another;
    the local ontology is not complete enough to reject person-vs-location or
    similar entity combinations safely.
    """
    if ontology is None:
        return []
    candidate = deepcopy(graph)
    _normalize_notable_type_constraint(candidate)
    variable_types: dict[str, set[str]] = {}

    def add(variable: Any, type_id: Any, source: str) -> None:
        variable = str(variable)
        type_id = str(type_id)
        if variable.startswith("V") and type_id:
            variable_types.setdefault(variable, set()).add(type_id)

    for subject, relation_id, object_ in candidate.triples:
        domain = str(ontology.domain_for_relation(relation_id))
        range_id = str(ontology.range_for_relation(relation_id))
        add(subject, domain, "triple_domain")
        add(object_, range_id, "triple_range")
    for operator in candidate.operators:
        if not isinstance(operator, dict):
            continue
        relation_id = str(operator.get("attribute_relation_id", "")).strip()
        if not relation_id:
            label = operator.get("attribute_relation_label", [])
            relation_id = relation_id_from_label(label) if label else ""
        if not relation_id:
            continue
        attribute_domain = str(ontology.domain_for_relation(relation_id))
        attribute_range = str(ontology.range_for_relation(relation_id))
        if not attribute_domain or not attribute_range:
            continue
        add(operator.get("input_var", ""), attribute_domain, "operator_domain")

    conflicts: list[dict[str, Any]] = []
    for variable, types in sorted(variable_types.items()):
        scalar = sorted(types & _HARD_SCALAR_SCHEMA_TYPES)
        meta = sorted(types & _HARD_META_SCHEMA_TYPES)
        ordinary = sorted(
            type_id
            for type_id in types
            if type_id not in _HARD_SCALAR_SCHEMA_TYPES
            and type_id not in _HARD_META_SCHEMA_TYPES
            and not type_id.startswith("type.")
        )
        entity = sorted(set(meta) | set(ordinary))
        if scalar and entity:
            conflicts.append({
                "variable": variable,
                "kind": "scalar_vs_entity",
                "scalar_types": scalar,
                "entity_types": entity,
            })
        if meta and ordinary:
            conflicts.append({
                "variable": variable,
                "kind": "meta_vs_ordinary",
                "meta_types": meta,
                "ordinary_types": ordinary,
            })
    return conflicts


def _hard_schema_conflict_preemption(
    query_graphs: list[QueryGraphCandidate],
    recovery_graph: QueryGraphCandidate | None,
    ontology: Any,
    *,
    conflicts: list[list[dict[str, Any]]] | None = None,
) -> tuple[list[QueryGraphCandidate], dict[str, Any]]:
    """Replace a beam only when every graph has a provable hard conflict."""
    if not query_graphs:
        return query_graphs, {"status": "not_applicable", "reason": "empty_beam"}
    graph_conflicts = conflicts or [
        _query_graph_hard_schema_conflicts(graph, ontology)
        for graph in query_graphs
    ]
    if len(graph_conflicts) != len(query_graphs) or not all(graph_conflicts):
        return query_graphs, {
            "status": "not_preempted",
            "reason": "at_least_one_schema_feasible_graph",
            "hard_conflict_graphs": sum(bool(item) for item in graph_conflicts),
            "graph_count": len(query_graphs),
        }
    if recovery_graph is None:
        return query_graphs, {
            "status": "not_preempted",
            "reason": "failure_compiler_did_not_compile",
            "hard_conflict_graphs": len(query_graphs),
            "graph_count": len(query_graphs),
        }
    return [recovery_graph], {
        "status": "preempted",
        "reason": "all_graphs_have_hard_schema_conflicts",
        "recovery_graph_id": recovery_graph.graph_id,
        "graph_count": len(query_graphs),
        "conflicts": graph_conflicts,
        "model_used": False,
    }


def _failure_constraint_scalar_extensions(
    *,
    question: str,
    compose_trace: list[dict[str, Any]],
    grounded_candidates: list[GroundedSemanticCandidate],
    ontology: Any,
) -> list[dict[str, Any]]:
    """Extract schema-valid scalar suffixes already proposed by Compose.

    The rejected Compose output is existing model work.  This function only
    separates its ungrounded answer attribute from the grounded semantic core;
    it never retrieves or invents relation candidates.
    """
    spec = _missing_constraint_spec(question)
    if spec is None or ontology is None:
        return []
    scalar_ranges = {"type.datetime", "type.enumeration", "type.float", "type.int"}
    raw_candidates: list[dict[str, Any]] = []
    for item in compose_trace:
        if item.get("valid") is not False:
            continue
        semantic_rank = int(item.get("semantic_rank", 0))
        if not 1 <= semantic_rank <= len(grounded_candidates):
            continue
        raw = item.get("output")
        if not isinstance(raw, dict):
            continue
        answer_var = str(raw.get("answer_var", ""))
        triples = [
            triple
            for triple in raw.get("triples", [])
            if isinstance(triple, dict)
        ]
        if not answer_var or not triples:
            continue
        grounded = grounded_candidates[semantic_rank - 1]
        grounded_relations = set(grounded.relation_bindings.values())
        outgoing: dict[str, list[tuple[str, str]]] = {}
        for triple in triples:
            relation_id = relation_id_from_label(triple.get("relation_label", []))
            subject = str(triple.get("subject", ""))
            object_ = str(triple.get("object", ""))
            if not relation_id or not subject or not object_:
                continue
            outgoing.setdefault(subject, []).append((relation_id, object_))

        for first_relation, middle_ref in outgoing.get(answer_var, []):
            if first_relation in grounded_relations:
                continue
            first_domain = str(ontology.domain_for_relation(first_relation))
            first_range = str(ontology.range_for_relation(first_relation))
            if not first_domain or not first_range:
                continue
            if first_range in scalar_ranges:
                raw_candidates.append({
                    "grounded": grounded,
                    "semantic_rank": semantic_rank,
                    "first_relation": first_relation,
                    "second_relation": "",
                    "spec": deepcopy(spec),
                })
                continue
            explicit_seconds = outgoing.get(middle_ref, [])
            seconds = [
                relation_id
                for relation_id, _ in explicit_seconds
                if _ontology_types_compatible(
                    ontology,
                    first_range,
                    str(ontology.domain_for_relation(relation_id)),
                )
                and str(ontology.range_for_relation(relation_id)) in scalar_ranges
            ]
            if not seconds:
                # Compose sometimes stops at a numeric CVT.  Completing its
                # scalar leaf from that exact range is still local schema use.
                seconds = [
                    relation_id
                    for relation_id in ontology.relations_for_domain(first_range)
                    if str(ontology.range_for_relation(relation_id)) in scalar_ranges
                ]
            for second_relation in seconds:
                raw_candidates.append({
                    "grounded": grounded,
                    "semantic_rank": semantic_rank,
                    "first_relation": first_relation,
                    "second_relation": second_relation,
                    "spec": deepcopy(spec),
                })

    # A rejected Compose response may mention only one side of a conventional
    # start/end pair.  Derive its schema sibling generically so explicit or
    # extrema phase evidence can choose between both without relation-specific
    # routing.
    temporal_replacements = (
        (".from_date", ".to_date"),
        (".to_date", ".from_date"),
        (".start_date", ".end_date"),
        (".end_date", ".start_date"),
        (".from", ".to"),
        (".to", ".from"),
        (".start", ".end"),
        (".end", ".start"),
    )
    derived: list[dict[str, Any]] = []
    for item in raw_candidates:
        second_relation = str(item["second_relation"])
        leaf_relation = second_relation or str(item["first_relation"])
        counterpart = ""
        for source, target in temporal_replacements:
            if leaf_relation.endswith(source):
                counterpart = f"{leaf_relation[: -len(source)]}{target}"
                break
        if (
            not counterpart
            or str(ontology.range_for_relation(leaf_relation)) != "type.datetime"
            or str(ontology.range_for_relation(counterpart)) != "type.datetime"
            or not _ontology_types_compatible(
                ontology,
                str(ontology.domain_for_relation(leaf_relation)),
                str(ontology.domain_for_relation(counterpart)),
            )
        ):
            continue
        sibling = dict(item)
        if second_relation:
            sibling["second_relation"] = counterpart
        else:
            sibling["first_relation"] = counterpart
        sibling["derived_temporal_counterpart"] = True
        derived.append(sibling)
    raw_candidates.extend(derived)

    support: dict[tuple[str, str], int] = {}
    for item in raw_candidates:
        key = (str(item["first_relation"]), str(item["second_relation"]))
        support[key] = support.get(key, 0) + 1
    for item in raw_candidates:
        key = (str(item["first_relation"]), str(item["second_relation"]))
        item["score"] = _failure_constraint_relation_score(
            question,
            key[0],
            key[1],
            support=support[key],
            semantic_rank=int(item["semantic_rank"]),
        )
    return sorted(
        raw_candidates,
        key=lambda item: (
            -float(item["score"]),
            int(item["semantic_rank"]),
            str(item["first_relation"]),
            str(item["second_relation"]),
        ),
    )


def _failure_existing_scalar_owner(
    graph: QueryGraphCandidate,
    first_relation: str,
    ontology: Any,
    operator_type: str,
) -> str:
    """Reuse a directly connected core CVT when it owns the scalar leaf."""
    if str(operator_type).upper() not in {"ARGMIN", "ARGMAX"}:
        # Comparisons/equality often constrain another relationship occurrence
        # (for example a prior position distinct from a current office).
        return ""
    middle_type = str(ontology.range_for_relation(first_relation))
    if not middle_type:
        return ""
    variable_types: dict[str, set[str]] = {}
    neighbors: set[str] = set()
    for subject, relation_id, object_ in graph.triples:
        subject, object_ = str(subject), str(object_)
        domain = str(ontology.domain_for_relation(relation_id))
        range_id = str(ontology.range_for_relation(relation_id))
        if subject.startswith("V") and domain:
            variable_types.setdefault(subject, set()).add(domain)
        if object_.startswith("V") and range_id:
            variable_types.setdefault(object_, set()).add(range_id)
        if subject == graph.answer_var and object_.startswith("V"):
            neighbors.add(object_)
        elif object_ == graph.answer_var and subject.startswith("V"):
            neighbors.add(subject)
    compatible = sorted(
        variable
        for variable in neighbors
        if any(
            _ontology_types_compatible(ontology, type_id, middle_type)
            for type_id in variable_types.get(variable, set())
        )
    )
    return compatible[0] if len(compatible) == 1 else ""


def _failure_anchor_self_join_ids(
    graph: QueryGraphCandidate,
    ontology: Any,
) -> list[str]:
    """Find anchors that can satisfy the answer through a duplicate self-join."""
    reverse_for_relation = getattr(ontology, "reverse_for_relation", None)

    def canonical_edge(triple: list[str]) -> tuple[str, str, str]:
        subject, relation_id, object_ = map(str, triple)
        inverses = (
            tuple(reverse_for_relation(relation_id))
            if callable(reverse_for_relation)
            else ()
        )
        canonical = min((relation_id, *inverses))
        if canonical == relation_id:
            return subject, canonical, object_
        return object_, canonical, subject

    anchor_bindings = graph.provenance.get("anchor_bindings", {})
    anchors = {
        str(value.get("id", ""))
        for value in anchor_bindings.values()
        if isinstance(value, dict)
        and _ENTITY_ID_RE.fullmatch(str(value.get("id", "")))
    }
    if not anchors:
        return []
    canonical = [canonical_edge(triple) for triple in graph.triples]
    matches: set[str] = set()
    for subject, relation_id, object_ in canonical:
        if graph.answer_var == subject:
            answer_role, shared = "subject", object_
        elif graph.answer_var == object_:
            answer_role, shared = "object", subject
        else:
            continue
        for anchor_subject, anchor_relation, anchor_object in canonical:
            if anchor_relation != relation_id:
                continue
            if answer_role == "subject":
                if anchor_object == shared and anchor_subject in anchors:
                    matches.add(anchor_subject)
            elif anchor_subject == shared and anchor_object in anchors:
                matches.add(anchor_object)
    return sorted(matches)


def _failure_constrained_recovery_graph(
    *,
    question: str,
    compose_trace: list[dict[str, Any]],
    grounded_candidates: list[GroundedSemanticCandidate],
    ontology: Any,
    pipeline_version: str,
) -> tuple[QueryGraphCandidate | None, dict[str, Any]]:
    """Compile at most one constrained graph after the normal beam is empty."""
    candidates = _failure_constraint_scalar_extensions(
        question=question,
        compose_trace=compose_trace,
        grounded_candidates=grounded_candidates,
        ontology=ontology,
    )
    diagnostics: list[dict[str, Any]] = []
    for candidate in candidates:
        grounded = candidate["grounded"]
        try:
            output = _deterministic_terminal_intersection_output(
                grounded.compose_input
            )
            graph = build_query_graph_v2(
                grounded,
                output,
                graph_id="GFC1",
                pipeline_version=pipeline_version,
            )
        except (LoweringError, ContractError, ValueError) as exc:
            diagnostics.append({
                "semantic_rank": int(candidate["semantic_rank"]),
                "status": "invalid_core",
                "error": str(exc),
            })
            continue

        first_relation = str(candidate["first_relation"])
        second_relation = str(candidate["second_relation"])
        owner_types = {
            str(ontology.domain_for_relation(relation_id))
            for subject, relation_id, _ in graph.triples
            if str(subject) == graph.answer_var
        }
        owner_types.update(
            str(ontology.range_for_relation(relation_id))
            for _, relation_id, object_ in graph.triples
            if str(object_) == graph.answer_var
        )
        first_domain = str(ontology.domain_for_relation(first_relation))
        if owner_types and not any(
            _ontology_types_compatible(ontology, owner_type, first_domain)
            for owner_type in owner_types
            if owner_type
        ):
            diagnostics.append({
                "semantic_rank": int(candidate["semantic_rank"]),
                "status": "incompatible_owner",
                "first_relation": first_relation,
            })
            continue

        spec = candidate["spec"]
        input_var = graph.answer_var
        reused_owner = ""
        if second_relation:
            reused_owner = _failure_existing_scalar_owner(
                graph,
                first_relation,
                ontology,
                str(spec["type"]),
            )
            if reused_owner:
                input_var = reused_owner
            else:
                variables = {
                    str(value)
                    for triple in graph.triples
                    for value in (triple[0], triple[2])
                    if str(value).startswith("V") and str(value)[1:].isdigit()
                }
                next_index = max(
                    (int(value[1:]) for value in variables),
                    default=-1,
                ) + 1
                input_var = f"V{next_index}"
                graph.triples.append(
                    [graph.answer_var, first_relation, input_var]
                )
        operator = {
            "type": str(spec["type"]),
            "inputs": [],
            "input_var": input_var,
            "attribute_relation_label": relation_label_from_id(
                second_relation or first_relation
            ),
            "attribute_relation_labels": [],
            "value": str(spec["value"]),
            "value_type": str(spec["value_type"]),
            "attribute_relation_id": second_relation or first_relation,
            "_constraint_coverage_repair": True,
        }
        if str(spec["type"]).upper() in {"ARGMIN", "ARGMAX"}:
            # Virtuoso can order DISTINCT answers incorrectly when the order
            # expression is not projected.  This flag is exclusive to the
            # failure compiler, leaving normal graph lowering unchanged.
            operator["_project_order_value"] = True
        graph.operators = [operator]
        excluded_anchor_ids = _failure_anchor_self_join_ids(
            graph,
            ontology,
        )
        graph.operators.extend(
            {
                "type": "NO_EQUAL",
                "inputs": [],
                "input_var": graph.answer_var,
                "attribute_relation_label": [],
                "attribute_relation_labels": [],
                "value": anchor_id,
                "value_type": "mid",
                "_value_entity_id": anchor_id,
                "_failure_anchor_self_join_exclusion": True,
            }
            for anchor_id in excluded_anchor_ids
        )
        graph.provenance["graph_fallback"] = {
            "source": "rejected_compose_scalar_constraint",
            "model_used": False,
        }
        graph.provenance["failure_constraint_compile"] = {
            "semantic_rank": int(candidate["semantic_rank"]),
            "first_relation": first_relation,
            "second_relation": second_relation,
            "reused_core_owner": reused_owner,
            "excluded_anchor_ids": excluded_anchor_ids,
            "operator_type": str(spec["type"]),
            "uses_gold_answers": False,
            "model_used": False,
        }
        return graph, {
            "status": "compiled",
            "graph_id": graph.graph_id,
            "semantic_rank": int(candidate["semantic_rank"]),
            "first_relation": first_relation,
            "second_relation": second_relation,
            "reused_core_owner": reused_owner,
            "excluded_anchor_ids": excluded_anchor_ids,
            "operator_type": str(spec["type"]),
            "candidate_count": len(candidates),
            "model_used": False,
        }
    return None, {
        "status": "not_compiled",
        "candidate_count": len(candidates),
        "diagnostics": diagnostics,
        "model_used": False,
    }


def _terminal_intersection_recovery_graphs(
    grounded_candidates: list[GroundedSemanticCandidate],
    *,
    pipeline_version: str,
    limit: int = 3,
) -> tuple[list[QueryGraphCandidate], list[dict[str, Any]]]:
    """Compile a bounded no-model recovery beam from grounded intersections.

    This fallback is intended only after the normal graph beam is empty or all
    of its executions are empty.  It consumes every already-grounded semantic
    path, so it cannot fail merely because a first-path fallback left other
    grounded relations unused.
    """
    graphs: list[QueryGraphCandidate] = []
    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, grounded in enumerate(grounded_candidates, start=1):
        if len(graphs) >= max(1, int(limit)):
            break
        path_count = len(grounded.compose_input.get("semantic_paths", []))
        if path_count < 2:
            continue
        try:
            output = _deterministic_terminal_intersection_output(
                grounded.compose_input
            )
            graph = build_query_graph_v2(
                grounded,
                output,
                graph_id=f"GIR{rank}",
                pipeline_version=pipeline_version,
            )
        except (LoweringError, ContractError, ValueError) as exc:
            diagnostics.append(
                {
                    "semantic_rank": rank,
                    "status": "invalid",
                    "error": str(exc),
                    "path_count": path_count,
                }
            )
            continue
        key = json.dumps(
            {
                "triples": graph.triples,
                "answer_var": graph.answer_var,
                "operators": graph.operators,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        graph.provenance["graph_fallback"] = {
            "source": "terminal_intersection_after_unsuccessful_graphs",
            "model_used": False,
        }
        graphs.append(graph)
        diagnostics.append(
            {
                "semantic_rank": rank,
                "status": "compiled",
                "graph_id": graph.graph_id,
                "path_count": path_count,
                "model_used": False,
            }
        )
    return graphs, diagnostics


def _failure_terminal_recovery_plan(
    terminal_graphs: list[QueryGraphCandidate],
    constrained_graph: QueryGraphCandidate | None,
    *,
    limit: int = 3,
) -> list[QueryGraphCandidate]:
    """Share, but never enlarge, the existing terminal recovery budget."""
    budget = max(1, int(limit))
    if constrained_graph is not None and terminal_graphs:
        return [constrained_graph, *terminal_graphs[: max(0, budget - 1)]]
    return terminal_graphs[:budget]


def _is_terminal_intersection_recovery(item: ExecutedGraph) -> bool:
    fallback = item.graph.provenance.get("graph_fallback", {})
    return bool(
        isinstance(fallback, dict)
        and fallback.get("source")
        == "terminal_intersection_after_unsuccessful_graphs"
    )


def _is_graph_reconstruction_recovery(item: ExecutedGraph) -> bool:
    return is_graph_reconstruction_graph(item.graph)


def _is_failure_constrained_recovery_graph(
    graph: QueryGraphCandidate,
) -> bool:
    fallback = graph.provenance.get("graph_fallback", {})
    return bool(
        isinstance(fallback, dict)
        and fallback.get("source") == "rejected_compose_scalar_constraint"
        and fallback.get("model_used") is False
    )


def _select_terminal_intersection_recovery(
    executed: list[ExecutedGraph],
) -> tuple[ExecutedGraph, dict[str, Any]]:
    """Select recovery output locally so a failed row adds no model call."""
    selected = max(
        executed,
        key=lambda item: (
            item.graph.score,
            -len(item.answer_ids),
            item.graph.graph_id,
        ),
    )
    return selected, {
        "status": "deterministic_terminal_intersection_recovery",
        "selected_graph_id": selected.graph.graph_id,
        "reason_codes": ["highest_grounding_score", "no_additional_model_call"],
        "candidates": [
            {
                "graph_id": item.graph.graph_id,
                "answer_count": len(item.answer_ids),
                "rule_score": item.graph.score,
            }
            for item in executed
        ],
    }


def _select_failure_terminal_recovery(
    executed: list[ExecutedGraph],
) -> tuple[ExecutedGraph, dict[str, Any]]:
    """Prefer a constrained recovery only when it safely filters terminal answers."""
    constrained = [
        item
        for item in executed
        if _is_failure_constrained_recovery_graph(item.graph)
    ]
    terminal = [
        item
        for item in executed
        if _is_terminal_intersection_recovery(item)
        or _is_graph_reconstruction_recovery(item)
    ]
    constrained_selected = max(
        constrained,
        key=lambda item: (item.graph.score, item.graph.graph_id),
        default=None,
    )
    if not terminal:
        if constrained_selected is None:
            raise ValueError("failure recovery selection has no candidates")
        return constrained_selected, {
            "status": "deterministic_failure_terminal_recovery",
            "selected_graph_id": constrained_selected.graph.graph_id,
            "reason_codes": [
                "only_constrained_recovery_succeeded",
                "no_additional_model_call",
            ],
        }
    terminal_selected, terminal_trace = _select_terminal_intersection_recovery(
        terminal
    )
    if constrained_selected is None:
        return terminal_selected, terminal_trace
    constrained_ids = set(constrained_selected.answer_ids)
    terminal_ids = set(terminal_selected.answer_ids)
    if constrained_ids and constrained_ids < terminal_ids:
        selected = constrained_selected
        reason = "strict_subset_of_terminal_recovery"
    else:
        selected = terminal_selected
        reason = "constraint_subset_guard_kept_terminal_recovery"
    return selected, {
        "status": "deterministic_failure_terminal_recovery",
        "selected_graph_id": selected.graph.graph_id,
        "reason_codes": [reason, "no_additional_model_call"],
        "constrained_graph_id": constrained_selected.graph.graph_id,
        "constrained_answer_count": len(constrained_selected.answer_ids),
        "terminal_graph_id": terminal_selected.graph.graph_id,
        "terminal_answer_count": len(terminal_selected.answer_ids),
    }


def _is_retrieval_ontology_recovery(item: ExecutedGraph) -> bool:
    grounded = item.graph.provenance.get("grounded_semantic", {})
    repair = grounded.get("grounding_repair", {}) if isinstance(grounded, dict) else {}
    return bool(
        isinstance(repair, dict)
        and repair.get("strategy") == "ontology_relation_expansion"
        and repair.get("model_used") is False
    )


def _select_retrieval_ontology_recovery(
    executed: list[ExecutedGraph],
) -> tuple[ExecutedGraph, dict[str, Any]]:
    """Select an all-local retrieval recovery without a Selector call."""
    selected = max(
        executed,
        key=lambda item: (
            item.graph.score,
            -len(item.answer_ids),
            item.graph.graph_id,
        ),
    )
    return selected, {
        "status": "deterministic_ontology_retrieval_recovery",
        "selected_graph_id": selected.graph.graph_id,
        "reason_codes": [
            "highest_grounding_score",
            "strict_retrieval_was_empty",
            "no_additional_model_call",
        ],
        "candidates": [
            {
                "graph_id": item.graph.graph_id,
                "answer_count": len(item.answer_ids),
                "rule_score": item.graph.score,
            }
            for item in executed
        ],
    }


_OPERATOR_TYPES = {
    "COUNT",
    "ARGMAX",
    "ARGMIN",
    "GREATER_THAN",
    "GREATER_OR_EQUAL",
    "LESS_THAN",
    "LESS_OR_EQUAL",
    "TC",
    "EQUAL",
    "NO_EQUAL",
}


def _validate_operator_prediction(
    value: Any,
    semantic_graph: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the independent GLM operator prediction."""
    output = parse_json_object(value)
    if set(output) != {"and_groups", "operators"}:
        raise ContractError("operator output must contain exactly and_groups and operators")
    known_paths = {
        str(path["id"])
        for path in semantic_graph.get("semantic_paths", [])
        if isinstance(path, dict) and str(path.get("id", ""))
    }
    known_vars = {
        str(step["to"])
        for path in semantic_graph.get("semantic_paths", [])
        if isinstance(path, dict)
        for step in path.get("steps", [])
        if isinstance(step, dict) and str(step.get("to", ""))
    }
    and_groups = output["and_groups"]
    if not isinstance(and_groups, list):
        raise ContractError("and_groups must be an array")
    normalized_groups: list[list[str]] = []
    for group in and_groups:
        if not isinstance(group, list) or len(group) < 2:
            raise ContractError("each and_group must contain at least two paths")
        paths = [str(path_id) for path_id in group]
        if any(path_id not in known_paths for path_id in paths):
            raise ContractError("and_group references an unknown path")
        normalized_groups.append(paths)

    operators = output["operators"]
    if not isinstance(operators, list):
        raise ContractError("operators must be an array")
    normalized_operators: list[dict[str, Any]] = []
    for raw in operators:
        if not isinstance(raw, dict):
            raise ContractError("each operator must be an object")
        operator_type = str(raw.get("type", "")).upper()
        if operator_type not in _OPERATOR_TYPES:
            raise ContractError(f"unsupported operator type: {operator_type}")
        input_var = str(raw.get("input_var", ""))
        if input_var and input_var not in known_vars:
            raise ContractError(f"operator input_var references an unknown variable: {input_var}")
        labels = raw.get("attribute_relation_label", [])
        if labels is None:
            labels = []
        if not isinstance(labels, list):
            raise ContractError("attribute_relation_label must be an array")
        plural_labels = raw.get("attribute_relation_labels", [])
        if plural_labels is None:
            plural_labels = []
        if not isinstance(plural_labels, list):
            raise ContractError("attribute_relation_labels must be an array")
        inputs = raw.get("inputs", [])
        if inputs is None:
            inputs = []
        if not isinstance(inputs, list) or any(str(path_id) not in known_paths for path_id in inputs):
            raise ContractError("operator inputs reference an unknown path")
        normalized_operators.append(
            {
                "type": operator_type,
                "inputs": [str(path_id) for path_id in inputs],
                "input_var": input_var,
                "attribute_relation_label": [str(label) for label in labels],
                "attribute_relation_labels": [str(label) for label in plural_labels],
                "value": "" if raw.get("value") is None else str(raw.get("value", "")),
                "value_type": "" if raw.get("value_type") is None else str(raw.get("value_type", "")),
            }
        )
    return {"and_groups": normalized_groups, "operators": normalized_operators}


def _answer_values(rows: list[dict[str, str]], answer_var: str) -> list[str]:
    key = answer_var.lstrip("?")
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(key, row.get("answer", ""))
        if not value:
            value = next(
                (candidate for name, candidate in row.items() if name.casefold() == key.casefold()),
                "",
            )
        value = str(value).strip()
        prefix = "http://rdf.freebase.com/ns/"
        if value.startswith(prefix):
            value = value[len(prefix) :]
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _answer_labels(answer_ids: list[str], labels: list[dict[str, str]]) -> list[dict[str, str]]:
    """Expose literal and unlabeled answers to downstream graph selection."""
    by_id = {
        str(item.get("id", "")): {
            "id": str(item.get("id", "")),
            "label": str(item.get("label", "")),
        }
        for item in labels
        if str(item.get("id", ""))
    }
    return [
        by_id.get(answer_id, {"id": answer_id, "label": answer_id})
        for answer_id in answer_ids
    ]


def _answers_only_repeat_graph_anchors(item: ExecutedGraph) -> bool:
    """Return true when every produced answer is merely an input anchor ID."""
    if not item.answer_ids:
        return False
    raw_bindings = item.graph.provenance.get("anchor_bindings", {})
    anchor_ids: set[str] = set()
    if isinstance(raw_bindings, dict):
        for value in raw_bindings.values():
            if isinstance(value, dict):
                entity_id = str(value.get("id", ""))
            else:
                entity_id = str(value)
            if entity_id:
                anchor_ids.add(entity_id)
    return bool(anchor_ids) and all(answer in anchor_ids for answer in item.answer_ids)


def _repair_only_anchor_echo_failure(items: list[ExecutedGraph]) -> bool:
    """Reject an empty-beam repair that can only copy its input anchor.

    Operator-attribute siblings are generated only after the ordinary beam has
    no answers.  If every resulting non-empty execution is such a sibling and
    every answer is one of that graph's bound input anchors, the repair has not
    discovered an answer node at all.  Returning EMPTY_RESULT lets the existing
    failure-only retrieval run; it does not inspect question text, IDs, or
    answer labels.
    """

    return bool(items) and all(
        isinstance(item.graph.provenance.get("operator_attribute_repair"), dict)
        and bool(item.graph.provenance.get("operator_attribute_repair"))
        and _answers_only_repeat_graph_anchors(item)
        for item in items
    )


def _has_detached_constant_component(graph: QueryGraphCandidate) -> bool:
    """Whether a triple containing a constant is unreachable from answer."""

    if not graph.answer_var or not graph.triples:
        return False
    reachable = {str(graph.answer_var)}
    changed = True
    while changed:
        changed = False
        for subject, _relation, object_ in graph.triples:
            subject = str(subject)
            object_ = str(object_)
            if subject not in reachable and object_ not in reachable:
                continue
            for node in (subject, object_):
                if node not in reachable:
                    reachable.add(node)
                    changed = True
    return any(
        str(subject) not in reachable
        and str(object_) not in reachable
        and (
            not str(subject).startswith("V")
            or not str(object_).startswith("V")
        )
        for subject, _relation, object_ in graph.triples
    )


def _repair_only_detached_constant_failure(items: list[ExecutedGraph]) -> bool:
    """Detect repair-only answers with an independent constant constraint."""

    return bool(items) and all(
        isinstance(item.graph.provenance.get("operator_attribute_repair"), dict)
        and bool(item.graph.provenance.get("operator_attribute_repair"))
        and _has_detached_constant_component(item.graph)
        for item in items
    )


def _operator_free_graph(graph: QueryGraphCandidate) -> QueryGraphCandidate:
    """Create a lower-scored base graph used only when operators empty the result."""
    fallback = deepcopy(graph)
    fallback.graph_id = f"{graph.graph_id}F"
    fallback.operators = []
    fallback.sparql = ""
    fallback.score -= 0.05
    fallback.compose_output = deepcopy(graph.compose_output)
    fallback.compose_output["operators"] = []
    fallback.provenance = deepcopy(graph.provenance)
    fallback.provenance["operator_fallback"] = {
        "source_graph_id": graph.graph_id,
        "reason": "operator_graph_returned_no_answers",
    }
    return fallback


def _reverse_constant_to_answer_chains(
    graph: QueryGraphCandidate,
    *,
    limit: int = 2,
) -> list[QueryGraphCandidate]:
    """Reverse bounded anchored chains while reusing exactly the same relations.

    Some grounded relation sequences have the right predicates but traverse an
    inverse entity-centric path.  This repair never consults question text and
    never invents a predicate; it reverses a simple directed chain from a
    constant anchor to the existing answer variable and keeps all other
    constraints unchanged.
    """
    if graph.operators:
        return []
    triples = graph.triples
    variants: list[QueryGraphCandidate] = []
    constants = sorted({
        node
        for triple in triples
        for node in (triple[0], triple[2])
        if not str(node).startswith("V")
    })
    for anchor in constants:
        nodes = [anchor]
        relations: list[str] = []
        indexes: list[int] = []
        current = anchor
        visited = {anchor}
        while len(indexes) < 4 and current != graph.answer_var:
            outgoing = [
                (index, triple)
                for index, triple in enumerate(triples)
                if index not in indexes and triple[0] == current
                and str(triple[2]).startswith("V")
            ]
            if len(outgoing) != 1:
                break
            index, triple = outgoing[0]
            target = str(triple[2])
            if target in visited:
                break
            indexes.append(index)
            relations.append(str(triple[1]))
            nodes.append(target)
            visited.add(target)
            current = target
        if current != graph.answer_var or len(indexes) < 2:
            continue
        reversed_triples = [
            list(triple)
            for index, triple in enumerate(triples)
            if index not in indexes
        ]
        for offset, relation in enumerate(relations):
            reversed_triples.append([
                nodes[-1 - offset],
                relation,
                nodes[-2 - offset],
            ])
        candidate = deepcopy(graph)
        candidate.graph_id = f"{graph.graph_id}R{len(variants)}"
        candidate.triples = reversed_triples
        candidate.sparql = ""
        candidate.score -= 0.02
        candidate.provenance = deepcopy(graph.provenance)
        candidate.provenance["graph_repair"] = {
            "strategy": "reverse_constant_to_answer_chain",
            "source_graph_id": graph.graph_id,
            "chain_length": len(indexes),
            "model_used": False,
        }
        variants.append(candidate)
        if len(variants) >= max(1, int(limit)):
            break
    return variants


def _dedupe_graphs(graphs: list[QueryGraphCandidate], limit: int) -> list[QueryGraphCandidate]:
    best: dict[str, QueryGraphCandidate] = {}
    for graph in graphs:
        key = json.dumps(
            {"triples": graph.triples, "answer_var": graph.answer_var, "operators": graph.operators},
            sort_keys=True,
        )
        previous = best.get(key)
        if previous is None or graph.score > previous.score:
            best[key] = graph
    return sorted(best.values(), key=lambda graph: (-graph.score, graph.graph_id))[: max(1, limit)]
