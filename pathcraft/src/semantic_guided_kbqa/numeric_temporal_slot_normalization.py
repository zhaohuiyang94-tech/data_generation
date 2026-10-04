"""Fixed-budget ontology normalization for explicit scalar constraints.

The caller supplies an already parsed date/number/extrema specification.  The
module ranks schema-backed attribute paths using anchor-masked lexical
evidence and returns deterministic graph replacements.  It never accesses
Gold data, calls a model, or names an entity/relation/sample.  Callers replace
existing execution slots; they must not append beyond their fixed beam.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .constraint_property_ranking import (
    answer_set_safety_reason,
    constraint_focus_phrase,
    property_rank_evidence,
)
from .contracts import ExecutedGraph, QueryGraphCandidate
from .ontology import relation_label_from_id


EPS = 1e-12
SCALAR_RANGES = frozenset(
    {"type.datetime", "type.enumeration", "type.float", "type.int"}
)
INTERVAL_ROLE_RE = re.compile(
    r"\b(?:during|served?|president|governor|office|position|tenure|held|current)\b",
    re.I,
)
GENERIC_WORDS = frozenset(
    {
        "a", "an", "and", "are", "at", "by", "did", "do", "does",
        "for", "has", "have", "in", "is", "of", "on", "or", "the",
        "that", "to", "was", "were", "what", "when", "where", "which",
        "who", "with", "first", "last", "latest", "earliest", "largest",
        "smallest", "highest", "lowest", "most", "recent", "number",
        "value", "date", "year", "time", "event", "film", "movie",
        "country", "location", "organization", "organisation", "person",
        "school", "structure", "team", "war",
    }
)
TEMPORAL_MORPHOLOGY = frozenset(
    {"birth", "death", "end", "found", "open", "publication", "release", "runtime", "start"}
)
RANGE_TYPES = frozenset(
    {"GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL"}
)
ENTITY_ID_RE = re.compile(r"^(?:m|g)\.[A-Za-z0-9_]+$")
SCALAR_LITERAL_RE = re.compile(
    r"^(?:[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?|"
    r"[+-]?\d{4,}(?:-\d{2}(?:-\d{2})?)?(?:Z|[+-]\d{2}:\d{2})?)$"
)
MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
DATE_RE = re.compile(
    rf"(?:\b\d{{4}}[-/]\d{{1,2}}[-/]\d{{1,2}}\b|"
    rf"\b\d{{1,2}}[-/]\d{{1,2}}[-/]\d{{4}}\b|"
    rf"\b(?:{MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+\d{{4}}\b|"
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTH_NAMES})[,]?\s+\d{{4}}\b)",
    re.I,
)
SCALAR_CUE_RE = re.compile(
    r"\b(?:age|amount|area|army|attendance|capacity|casualt(?:y|ies)|code|"
    r"count|duration|episode|episodes|force|gdp|height|id|identifier|length|"
    r"minutes?|hours?|miles?|kilomet(?:er|re)s?|number|population|"
    r"postgraduates?|rate|runtime|size|soldiers?|tons?|undergraduates?|"
    r"visitors?|weight|width|percent(?:age)?)\b",
    re.I,
)
TEMPORAL_CUE_RE = re.compile(
    r"\b(?:date|year|during|served?|president|governor|office|position|tenure|"
    r"held|founded|established|opened|released?|ended?|began|started?|born|"
    r"died|death|current|until|since)\b",
    re.I,
)
COMPARISON_RE = re.compile(
    r"\b(?:no more than|no less than|at most|at least|prior to|less than|"
    r"fewer than|lower than|smaller than|under|below|before|more than|"
    r"greater than|higher than|larger than|over|above|after|later than)\b",
    re.I,
)


def lexical_tokens(value: str) -> set[str]:
    """Small generic morphology used only for numeric/temporal attributes."""

    aliases = {
        "died": "death", "dead": "death", "released": "release",
        "releasing": "release", "founded": "found", "founding": "found",
        "began": "start", "begun": "start", "ended": "end",
        "ending": "end", "visitors": "visitor", "minutes": "runtime",
        "minute": "runtime", "long": "runtime",
        "ids": "id",
    }
    output: set[str] = set()
    for raw in re.findall(r"[A-Za-z]+", str(value).casefold()):
        token = aliases.get(raw, raw)
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        output.add(token)
    return output


def _schema_leaf_exact_match(relation_id: str, focus: str) -> bool:
    """Match a complete schema leaf against the anchor-masked question.

    This is deliberately relation-agnostic.  Every token in the final schema
    component must occur in the question after the same small morphology used
    by scalar retrieval, and at least one token must carry non-generic lexical
    content.  It is used only to break otherwise equal property ranks.
    """

    leaf_tokens = lexical_tokens(str(relation_id).rsplit(".", 1)[-1])
    focus_tokens = lexical_tokens(focus)
    return bool(
        leaf_tokens
        and leaf_tokens <= focus_tokens
        and (leaf_tokens - GENERIC_WORDS - {"id"})
    )


def mask_anchor_labels(question: str, graph: QueryGraphCandidate) -> str:
    """Remove linked entity labels before recognizing a scalar literal."""

    text = str(question)
    bindings = graph.provenance.get("anchor_bindings", {})
    if isinstance(bindings, dict):
        for binding in bindings.values():
            label = str(binding.get("label", "")) if isinstance(binding, dict) else ""
            if label:
                text = re.sub(re.escape(label), " ", text, flags=re.I)
    text = re.sub(r"\bearlies\b", "earliest", text, flags=re.I)
    text = re.sub(r"\bmost\s+recently\b", "latest", text, flags=re.I)
    return " ".join(text.split())


def _equality_focus(text: str, value: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+(?:[.,/-][A-Za-z0-9]+)*", str(text))
    digits = re.sub(r"[^0-9]", "", str(value))
    indexes = [
        index
        for index, token in enumerate(tokens)
        if digits and re.sub(r"[^0-9]", "", token) == digits
    ]
    if not indexes:
        return constraint_focus_phrase(text)
    index = indexes[-1]
    preceding = tokens[index - 1].casefold() if index else ""
    end = index + 1 if preceding in {
        "at", "during", "from", "in", "on", "since", "throughout", "until",
    } else min(len(tokens), index + 4)
    return " ".join(tokens[max(0, index - 6):end])


def normalize_explicit_spec(
    question: str,
    graph: QueryGraphCandidate,
    resolver: Callable[[str], Mapping[str, Any] | None],
) -> dict[str, Any] | None:
    """Resolve one explicit constraint after entity-label/value masking."""

    text = mask_anchor_labels(question, graph)
    raw = resolver(text)
    if not isinstance(raw, Mapping):
        return None
    spec = dict(raw)
    operator_type = str(spec.get("type", "")).upper()
    if operator_type in {"ARGMIN", "ARGMAX"}:
        spec["focus"] = text
        return spec
    if operator_type in RANGE_TYPES:
        comparisons = list(COMPARISON_RE.finditer(text))
        if comparisons:
            cue = comparisons[-1]
            tail = text[cue.end():]
            date = DATE_RE.search(tail)
            number = re.search(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", tail)
            choices = [match for match in (date, number) if match is not None]
            if choices:
                literal = min(choices, key=lambda match: match.start())
                nearest = resolver(text[: cue.end()] + tail[: literal.end()])
                if (
                    isinstance(nearest, Mapping)
                    and str(nearest.get("type", "")).upper() in RANGE_TYPES
                ):
                    spec = dict(nearest)
        spec["focus"] = text
        return spec
    if operator_type != "EQUAL":
        return None

    values = re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", text)
    dates = list(DATE_RE.finditer(text))
    if len(values) != 1 and not (len(dates) == 1 and len(values) <= 3):
        return None
    value = str(spec.get("value", ""))
    spec["focus"] = _equality_focus(text, value)
    if dates:
        spec["value_type"] = "dateTime"
        return spec
    if re.fullmatch(r"[12]\d{3}", value) and TEMPORAL_CUE_RE.search(text):
        if INTERVAL_ROLE_RE.search(text) and re.search(
            rf"\b(?:in|during|throughout|at)\s+(?:the\s+year\s+)?{re.escape(value)}\b",
            text,
            re.I,
        ):
            spec["type"] = "TC"
            spec["value_type"] = "year"
        else:
            spec["value_type"] = "dateTime"
        return spec
    if SCALAR_CUE_RE.search(text):
        spec["value_type"] = "number"
        return spec
    return None


def _substantive_types(graph: QueryGraphCandidate) -> set[str]:
    return {
        str(operator.get("type", "")).upper()
        for operator in graph.operators
        if isinstance(operator, dict)
        and str(operator.get("type", "")).upper() not in {"", "NO_EQUAL"}
    }


def _equivalent_types(operator_type: str) -> set[str]:
    return {
        "LESS_THAN": {"LESS_THAN", "LESS_OR_EQUAL"},
        "GREATER_THAN": {"GREATER_THAN", "GREATER_OR_EQUAL"},
    }.get(str(operator_type).upper(), {str(operator_type).upper()})


def build_fixed_slot_variants(
    graph: QueryGraphCandidate,
    spec: Mapping[str, Any],
    ontology: Any,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return at most ``limit`` deterministic replacements for ``graph``.

    The result is ordered.  A caller may try the variants in order, but once
    a variant returns a nonempty answer set it must not fall through to a
    lower-ranked property merely to obtain a smaller set.
    """

    if ontology is None or not graph.triples or int(limit) < 1:
        return []
    variable_types: dict[str, set[str]] = {}
    predecessors: dict[str, list[str]] = {}
    for subject, relation, object_ in graph.triples:
        domain = str(ontology.domain_for_relation(relation))
        range_id = str(ontology.range_for_relation(relation))
        if str(subject).startswith("V") and domain:
            variable_types.setdefault(str(subject), set()).add(domain)
        if str(object_).startswith("V") and range_id:
            variable_types.setdefault(str(object_), set()).add(range_id)
        if str(subject).startswith("V") and str(object_).startswith("V"):
            predecessors.setdefault(str(object_), []).append(str(subject))

    owners = list(dict.fromkeys([
        str(graph.answer_var),
        *predecessors.get(str(graph.answer_var), []),
    ]))
    local_focus = constraint_focus_phrase(str(spec.get("focus", "")), window=9)
    question_tokens = lexical_tokens(local_focus)
    operator_type = str(spec.get("type", "")).upper()
    candidates: list[tuple[float, str, str, str, Any, bool, bool]] = []
    seen_paths: set[tuple[str, str, str]] = set()
    for owner in owners:
        declared = variable_types.get(owner, set())
        declared_namespaces = {
            value.split(".", 1)[0] for value in declared if "." in value
        }
        expanded: set[str] = set()
        for type_id in declared:
            expanded.update(ontology.supertypes(type_id))
            expanded.update(ontology.subtypes(type_id, max_depth=2))
        for owner_type in expanded:
            for first in ontology.relations_for_domain(owner_type):
                middle_type = str(ontology.range_for_relation(first))
                paths = [(first, "")] if middle_type in SCALAR_RANGES else []
                if middle_type and middle_type not in SCALAR_RANGES:
                    paths.extend(
                        (first, second)
                        for second in ontology.relations_for_domain(middle_type)
                        if str(ontology.range_for_relation(second)) in SCALAR_RANGES
                    )
                for first_relation, second_relation in paths:
                    path_key = (owner, first_relation, second_relation)
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)
                    terminal = second_relation or first_relation
                    relation_text = " ".join(
                        relation_label_from_id(first_relation)
                        + (
                            relation_label_from_id(second_relation)
                            if second_relation else []
                        )
                    )
                    relation_tokens = lexical_tokens(relation_text)
                    overlap = (
                        (question_tokens - GENERIC_WORDS)
                        & (relation_tokens - GENERIC_WORDS)
                    )
                    tail = terminal.rsplit(".", 1)[-1].casefold()
                    tc_start = bool(
                        operator_type == "TC"
                        and tail in {"from", "from_date", "start", "start_date"}
                        and INTERVAL_ROLE_RE.search(str(spec.get("focus", "")))
                    )
                    temporal_overlap = bool(
                        question_tokens & relation_tokens & TEMPORAL_MORPHOLOGY
                    )
                    evidence = property_rank_evidence(
                        first_relation=first_relation,
                        second_relation=second_relation,
                        spec=spec,
                        terminal_range=str(ontology.range_for_relation(terminal)),
                        semantic_similarity=0.0,
                        graph_score=float(graph.score),
                    )
                    if not evidence.matched_tokens and not tc_start and not temporal_overlap:
                        continue
                    score = (
                        4.0 * len(overlap)
                        + 2.0 * int(temporal_overlap)
                        + 1.5 * int(tc_start)
                        + 2.0 * int(
                            str(ontology.domain_for_relation(first_relation)).split(".", 1)[0]
                            in declared_namespaces
                        )
                        + evidence.family_score
                        + (0.20 if not second_relation else 0.0)
                        + evidence.score
                    )
                    candidates.append(
                        (
                            score,
                            owner,
                            first_relation,
                            second_relation,
                            evidence,
                            _schema_leaf_exact_match(
                                terminal,
                                str(spec.get("focus", "")),
                            ),
                            bool(
                                owner == str(graph.answer_var)
                                or any(
                                    str(subject) == owner
                                    and str(relation) == first_relation
                                    and str(object_) == str(graph.answer_var)
                                    for subject, relation, object_ in graph.triples
                                )
                            ),
                        )
                    )

    output: list[dict[str, Any]] = []
    seen_graphs: set[str] = set()
    exact_terminals = {
        second_relation or first_relation
        for _, _, first_relation, second_relation, _, exact, _ in candidates
        if exact
    }
    unique_exact_terminal = (
        next(iter(exact_terminals)) if len(exact_terminals) == 1 else ""
    )
    for (
        score,
        owner,
        first_relation,
        second_relation,
        evidence,
        exact,
        answer_path_bound,
    ) in sorted(
        candidates,
        key=lambda item: (
            -item[0],
            -int(bool(unique_exact_terminal) and (item[3] or item[2]) == unique_exact_terminal),
            item[1],
            item[2],
            item[3],
        ),
    ):
        sibling = deepcopy(graph)
        sibling.graph_id = f"{graph.graph_id}N{len(output)}"
        sibling.sparql = ""
        operator: dict[str, Any] = {
            "type": operator_type,
            "inputs": [],
            "input_var": owner,
            "attribute_relation_label": relation_label_from_id(
                second_relation or first_relation
            ),
            "attribute_relation_labels": [],
            "value": str(spec.get("value", "")),
            "value_type": str(spec.get("value_type", "")),
            "attribute_relation_id": second_relation or first_relation,
            "_fixed_slot_numeric_temporal_repair": True,
        }
        if second_relation:
            variables = {
                str(value)
                for triple in sibling.triples
                for value in (triple[0], triple[2])
                if re.fullmatch(r"V\d+", str(value))
            }
            number = max((int(value[1:]) for value in variables), default=-1) + 1
            middle = f"V{number}"
            sibling.triples.append([owner, first_relation, middle])
            operator["input_var"] = middle
        sibling.operators.append(operator)
        signature = json.dumps(
            {
                "triples": sibling.triples,
                "answer_var": sibling.answer_var,
                "operators": sibling.operators,
            },
            sort_keys=True,
        )
        if signature in seen_graphs:
            continue
        seen_graphs.add(signature)
        sibling.provenance = deepcopy(graph.provenance)
        sibling.provenance["fixed_slot_numeric_temporal_normalization"] = {
            "source_graph_id": graph.graph_id,
            "source_answer_var": graph.answer_var,
            "source_triples": deepcopy(graph.triples),
            "first_relation": first_relation,
            "second_relation": second_relation,
            "lexical_rank_score": score,
            "matched_tokens": list(evidence.matched_tokens),
            "unique_schema_leaf_exact_match": bool(
                unique_exact_terminal
                and (second_relation or first_relation) == unique_exact_terminal
            ),
            "answer_path_unique_schema_leaf_exact_match": bool(
                answer_path_bound
                and unique_exact_terminal
                and (second_relation or first_relation) == unique_exact_terminal
            ),
            "additional_model_calls": 0,
            "additional_endpoint_queries": 0,
        }
        output.append(
            {
                "graph": sibling,
                "first_relation": first_relation,
                "second_relation": second_relation,
                "evidence": evidence,
                "lexical_rank_score": score,
                "unique_schema_leaf_exact_match": bool(
                    unique_exact_terminal
                    and (second_relation or first_relation)
                    == unique_exact_terminal
                ),
                "answer_path_unique_schema_leaf_exact_match": bool(
                    answer_path_bound
                    and unique_exact_terminal
                    and (second_relation or first_relation)
                    == unique_exact_terminal
                ),
            }
        )
        if len(output) >= int(limit):
            break
    return output


def choose_fixed_slot_prediction(
    source_ids: Sequence[str],
    attempts: Sequence[Mapping[str, Any]],
    *,
    operator_type: str,
    source_answers: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[str], int | None, str]:
    """Choose the first safe variant, with nonempty top-property interlock."""

    for index, attempt in enumerate(attempts):
        answers = list(map(str, attempt.get("answer_ids", [])))
        reason = answer_set_safety_reason(
            source_ids,
            answers,
            operator_type=operator_type,
            source_answers=source_answers,
            repaired_answers=attempt.get("answers", []),
        )
        if not reason:
            return answers, index, "strict_subset"
        if answers:
            return [], None, f"top_nonempty_{reason}"
    return [], None, "all_variants_empty"


def prepare_fixed_execution_slots(
    question: str,
    graphs: Sequence[QueryGraphCandidate],
    ontology: Any,
    *,
    spec_resolver: Callable[[str], Mapping[str, Any] | None],
    execution_budget: int,
    max_replacements: int = 3,
    expects_entity: Callable[[str], bool] | None = None,
    protected_graph_ids: Iterable[str] = (),
) -> tuple[list[QueryGraphCandidate], dict[str, Any]]:
    """Insert normalized variants while keeping graph/query counts constant.

    The highest-scoring eligible source remains in the beam.  Its variants
    replace the same number of lowest-priority execution slots.  Variants are
    marked so the normal selector can ignore them; they are considered only by
    :func:`postselect_fixed_slot_execution` after an incumbent is chosen.
    """

    original = list(graphs)
    protected_ids = {str(value) for value in protected_graph_ids if str(value)}
    budget = min(len(original), max(0, int(execution_budget)))
    replacement_cap = min(max(0, int(max_replacements)), max(0, budget - 1))
    diagnostics: dict[str, Any] = {
        "status": "not_applied",
        "execution_budget": budget,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
        "uses_whitelist": False,
        "protected_graph_ids": sorted(protected_ids),
    }
    if ontology is None or budget < 2 or replacement_cap < 1:
        diagnostics["reason"] = "missing_ontology_or_execution_slots"
        return original, diagnostics

    require_entity = bool(expects_entity and expects_entity(question))
    eligible: list[tuple[float, int, QueryGraphCandidate, dict[str, Any]]] = []
    # Keep enough tail positions available for equal-count replacement.
    for index, graph in enumerate(original[:budget]):
        spec = normalize_explicit_spec(question, graph, spec_resolver)
        if spec is None:
            continue
        operator_type = str(spec.get("type", "")).upper()
        if _substantive_types(graph) & _equivalent_types(operator_type):
            continue
        if (
            operator_type in {"ARGMIN", "ARGMAX"}
            and (
                graph.provenance.get("grounded_semantic", {}) or {}
            ).get("dropped_paths")
        ):
            continue
        if require_entity:
            answer_types: set[str] = set()
            for subject, relation, object_ in graph.triples:
                if str(object_) == str(graph.answer_var):
                    answer_types.add(str(ontology.range_for_relation(relation)))
                if str(subject) == str(graph.answer_var):
                    answer_types.add(str(ontology.domain_for_relation(relation)))
            known = {value for value in answer_types if value}
            if known and all(value.startswith("type.") for value in known):
                continue
        eligible.append((float(graph.score), -index, graph, spec))
    if not eligible:
        diagnostics["reason"] = "no_eligible_source_graph"
        return original, diagnostics

    ranked_eligible = sorted(
        eligible,
        key=lambda item: (-item[0], -item[1], item[2].graph_id),
    )
    prepared_sources: list[
        tuple[float, int, QueryGraphCandidate, dict[str, Any], list[dict[str, Any]]]
    ] = []
    top_schema_score: float | None = None
    for score, negative_index, candidate, candidate_spec in ranked_eligible:
        # Once a schema-backed source exists, only near-score challengers can
        # displace it.  This bounds ontology work and protects normal ranking.
        if top_schema_score is not None and top_schema_score - score > 0.030000001:
            break
        candidate_variants = build_fixed_slot_variants(
            candidate,
            candidate_spec,
            ontology,
            limit=replacement_cap,
        )
        if not candidate_variants:
            continue
        if top_schema_score is None:
            top_schema_score = score
        prepared_sources.append(
            (
                score,
                negative_index,
                candidate,
                candidate_spec,
                candidate_variants,
            )
        )
    if not prepared_sources:
        diagnostics["reason"] = "no_schema_backed_variant"
        return original, diagnostics

    chosen = prepared_sources[0]
    source_selection_reason = "highest_score_schema_backed_source"
    for challenger in prepared_sources[1:]:
        incumbent_top = chosen[4][0]
        challenger_top = challenger[4][0]
        score_gap = chosen[0] - challenger[0]
        lexical_delta = float(challenger_top["lexical_rank_score"]) - float(
            incumbent_top["lexical_rank_score"]
        )
        direct_schema_upgrade = bool(
            score_gap <= 0.020000001
            and incumbent_top.get("second_relation")
            and not challenger_top.get("second_relation")
            and lexical_delta >= 0.4 - EPS
        )
        unique_leaf_upgrade = bool(
            score_gap <= 0.020000001
            and challenger_top.get(
                "answer_path_unique_schema_leaf_exact_match"
            )
            and not incumbent_top.get(
                "answer_path_unique_schema_leaf_exact_match"
            )
        )
        if direct_schema_upgrade or unique_leaf_upgrade:
            chosen = challenger
            source_selection_reason = (
                "near_score_direct_schema_leaf_upgrade"
                if direct_schema_upgrade
                else "near_score_unique_schema_leaf_exact_match"
            )

    _, negative_index, source, spec, variants = chosen
    source_index = -negative_index
    source_order = [
        chosen,
        *(item for item in prepared_sources if item is not chosen),
    ]
    diverse_source_count = max(
        1,
        min(
            len(source_order),
            replacement_cap,
            max(1, budget // 2),
        ),
    )
    active_source_order = source_order[:diverse_source_count]
    active_replacement_cap = min(
        replacement_cap,
        max(0, budget - diverse_source_count),
    )
    replacement_items: list[dict[str, Any]] = []
    replacement_keys: set[str] = set()

    def retain(item: dict[str, Any]) -> None:
        graph = item["graph"]
        key = json.dumps(
            {
                "triples": graph.triples,
                "answer_var": graph.answer_var,
                "operators": graph.operators,
            },
            sort_keys=True,
        )
        if key not in replacement_keys:
            replacement_keys.add(key)
            replacement_items.append(item)

    # Give each near-score source one slot before spending the remaining slots
    # on lower-ranked properties of a single source. Query count is unchanged.
    for prepared_source in active_source_order:
        retain(prepared_source[4][0])
        if len(replacement_items) >= active_replacement_cap:
            break
    if len(replacement_items) < active_replacement_cap:
        for prepared_source in active_source_order:
            for item in prepared_source[4][1:]:
                retain(item)
                if len(replacement_items) >= active_replacement_cap:
                    break
            if len(replacement_items) >= active_replacement_cap:
                break

    count = min(len(replacement_items), active_replacement_cap)
    evicted: list[int] = []
    diverse_source_ids = {
        str(item[2].graph_id) for item in active_source_order
    }
    # Never evict a graph ranked above the source merely to manufacture more
    # repair slots.  A low-priority source may consume only still-lower tail
    # slots; if it is already last, the fixed-budget lane abstains.
    for index in range(budget - 1, source_index, -1):
        if (
            str(original[index].graph_id) in protected_ids
            or str(original[index].graph_id) in diverse_source_ids
        ):
            continue
        evicted.append(index)
        if len(evicted) == count:
            break
    count = min(count, len(evicted))
    if not count:
        diagnostics["reason"] = "no_replaceable_tail_slot"
        return original, diagnostics
    evicted_set = set(evicted[:count])
    prefix = [
        graph for index, graph in enumerate(original[:budget])
        if index not in evicted_set
    ]
    replacement_graphs = [item["graph"] for item in replacement_items[:count]]
    prepared = [*prefix, *replacement_graphs, *original[budget:]]
    diagnostics.update(
        {
            "status": "prepared",
            "constraint": dict(spec),
            "source_graph_id": source.graph_id,
            "source_selection_reason": source_selection_reason,
            "slot_allocation": "near_score_source_diverse_round_robin",
            "schema_backed_sources_considered": [
                item[2].graph_id for item in prepared_sources
            ],
            "variant_graph_ids": [graph.graph_id for graph in replacement_graphs],
            "replaced_graph_ids": [original[index].graph_id for index in sorted(evicted_set)],
            "replacement_query_count": count,
            "query_count_before": budget,
            "query_count_after": min(len(prepared), budget),
        }
    )
    return prepared, diagnostics


def is_fixed_slot_variant(graph: QueryGraphCandidate) -> bool:
    return isinstance(
        graph.provenance.get("fixed_slot_numeric_temporal_normalization"),
        dict,
    )


def partition_fixed_slot_executions(
    executions: Sequence[ExecutedGraph],
) -> tuple[list[ExecutedGraph], list[ExecutedGraph]]:
    normal, variants = [], []
    for execution in executions:
        (variants if is_fixed_slot_variant(execution.graph) else normal).append(execution)
    return normal, variants


def _provenance_source_graph_ids(graph: QueryGraphCandidate) -> list[str]:
    """Return explicit source/base graph ids in stable provenance order."""

    output: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"source_graph_id", "base_graph_id"}:
                    graph_id = str(item).strip()
                    if graph_id and graph_id not in output:
                        output.append(graph_id)
                elif isinstance(item, (Mapping, list, tuple)):
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(graph.provenance)
    return output


def _triple_counter(graph: QueryGraphCandidate) -> Counter[tuple[str, str, str]]:
    return Counter(
        (str(subject), str(relation), str(object_))
        for subject, relation, object_ in graph.triples
    )


def _is_safe_source_derivation(
    selected: ExecutedGraph,
    source: ExecutedGraph,
) -> bool:
    """Verify that a derived execution is the same answered source core.

    Attribute-repair executions use ids such as ``G1A5`` even though their
    unchanged answer-producing spine comes from ``G1``.  A fixed-slot variant
    may challenge that derived execution only when the provenance/base id,
    answer variable, complete current answer set, and source core triples all
    agree.  This prevents a shared graph-id prefix from linking unrelated
    candidates.
    """

    if str(selected.graph.answer_var) != str(source.graph.answer_var):
        return False
    if set(map(str, selected.answer_ids)) != set(map(str, source.answer_ids)):
        return False
    source_core = _triple_counter(source.graph)
    selected_core = _triple_counter(selected.graph)
    return all(selected_core[triple] >= count for triple, count in source_core.items())


def _selected_source_graph_id(
    selected: ExecutedGraph,
    source_executions: Sequence[ExecutedGraph],
    available_source_ids: set[str],
) -> tuple[str, str]:
    """Resolve the selected execution to one safely equivalent beam source."""

    selected_id = str(selected.graph.graph_id)
    if selected_id in available_source_ids:
        return selected_id, "selected_is_source"

    explicit = _provenance_source_graph_ids(selected.graph)
    candidates = explicit
    evidence = "provenance_source_graph_id"
    if not candidates:
        match = re.match(r"^(G\d+)", selected_id)
        candidates = [match.group(1)] if match else []
        evidence = "canonical_graph_base"

    executions_by_id = {
        str(execution.graph.graph_id): execution
        for execution in source_executions
    }
    for source_id in candidates:
        if source_id not in available_source_ids:
            continue
        source = executions_by_id.get(source_id)
        if source is None:
            continue
        if _is_safe_source_derivation(selected, source):
            return source_id, evidence
    return "", "no_safe_selected_source"


def _unique_extrema_operator(question: str) -> str:
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
    if maximum == minimum:
        return ""
    return "ARGMAX" if maximum else "ARGMIN"


def _answer_terminal_types(
    graph: QueryGraphCandidate,
    ontology: Any,
) -> set[str]:
    if ontology is None:
        return set()
    output: set[str] = set()
    answer_var = str(graph.answer_var)
    for subject, relation, object_ in graph.triples:
        if str(object_) == answer_var:
            output.add(str(ontology.range_for_relation(relation)))
        if str(subject) == answer_var:
            output.add(str(ontology.domain_for_relation(relation)))
    return {value for value in output if value}


def _ontology_type_compatible(
    answer_types: set[str],
    attribute_domain: str,
    ontology: Any,
) -> bool:
    if not answer_types or not attribute_domain or ontology is None:
        return False
    attribute_closure = {attribute_domain}
    attribute_closure.update(ontology.supertypes(attribute_domain))
    for answer_type in answer_types:
        closure = {answer_type}
        closure.update(ontology.supertypes(answer_type))
        closure.update(ontology.subtypes(answer_type, max_depth=2))
        if attribute_domain in closure or answer_type in attribute_closure:
            return True
    return False


def _entity_projection_source(
    selected: ExecutedGraph,
    variants: Sequence[ExecutedGraph],
    source_executions: Sequence[ExecutedGraph],
    *,
    question: str,
    ontology: Any,
    expects_entity: Callable[[str], bool] | None,
) -> tuple[ExecutedGraph | None, str]:
    """Resolve a scalar extrema projection to one complete entity source."""

    entity_intent_texts = [str(question)]
    for source in source_executions:
        decomposition = source.graph.provenance.get("decomposition", [])
        if isinstance(decomposition, (list, tuple)):
            entity_intent_texts.extend(map(str, decomposition))
    if not expects_entity or not any(
        expects_entity(text) for text in entity_intent_texts
    ):
        return None, "question_does_not_expect_entity"
    if not selected.answer_ids or not all(
        SCALAR_LITERAL_RE.fullmatch(str(value).strip())
        for value in selected.answer_ids
    ):
        return None, "selected_answers_not_all_scalar_literals"
    extrema = _unique_extrema_operator(question)
    if not extrema:
        return None, "question_not_unique_extrema"

    variants_by_source: dict[str, list[ExecutedGraph]] = {}
    for execution in variants:
        repair = execution.graph.provenance.get(
            "fixed_slot_numeric_temporal_normalization",
            {},
        )
        source_id = str((repair or {}).get("source_graph_id", ""))
        if source_id:
            variants_by_source.setdefault(source_id, []).append(execution)

    eligible: list[ExecutedGraph] = []
    for source in source_executions:
        source_id = str(source.graph.graph_id)
        source_variants = variants_by_source.get(source_id, [])
        if not source_variants or not 2 <= len(set(source.answer_ids)) <= 256:
            continue
        if not all(ENTITY_ID_RE.fullmatch(str(value)) for value in source.answer_ids):
            continue
        answer_types = _answer_terminal_types(source.graph, ontology)
        if not answer_types or any(value in SCALAR_RANGES for value in answer_types):
            continue
        top_variant = max(
            source_variants,
            key=lambda execution: float(
                (
                    execution.graph.provenance.get(
                        "fixed_slot_numeric_temporal_normalization",
                        {},
                    )
                    or {}
                ).get("lexical_rank_score", 0.0)
            ),
        )
        repair_operator = next(
            (
                operator
                for operator in top_variant.graph.operators
                if isinstance(operator, dict)
                and operator.get("_fixed_slot_numeric_temporal_repair")
            ),
            None,
        )
        repair = top_variant.graph.provenance.get(
            "fixed_slot_numeric_temporal_normalization",
            {},
        )
        first_relation = str((repair or {}).get("first_relation", ""))
        if (
            not isinstance(repair_operator, Mapping)
            or str(repair_operator.get("type", "")).upper() != extrema
            or str(repair_operator.get("input_var", ""))
            != str(source.graph.answer_var)
            or not _ontology_type_compatible(
                answer_types,
                str(ontology.domain_for_relation(first_relation)),
                ontology,
            )
        ):
            continue
        eligible.append(source)

    if not eligible:
        return None, "no_complete_entity_extrema_source"
    eligible.sort(
        key=lambda execution: (
            -float(execution.graph.score),
            execution.graph.graph_id,
        )
    )
    source = eligible[0]
    if (
        len(eligible) > 1
        and abs(
            float(source.graph.score) - float(eligible[1].graph.score)
        ) <= EPS
    ):
        return None, "entity_source_top_score_not_unique"
    if float(source.graph.score) + EPS < float(selected.graph.score):
        return None, "entity_source_scores_below_scalar_selected"
    return source, "entity_projection_source"


def postselect_fixed_slot_execution(
    selected: ExecutedGraph,
    variants: Sequence[ExecutedGraph],
    *,
    source_executions: Sequence[ExecutedGraph] = (),
    question: str = "",
    ontology: Any = None,
    expects_entity: Callable[[str], bool] | None = None,
) -> tuple[ExecutedGraph, dict[str, Any]]:
    """Apply a fixed-slot variant only to its selected source graph."""

    diagnostics: dict[str, Any] = {
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "additional_model_calls": 0,
        "additional_endpoint_queries": 0,
    }
    variant_sources = {
        str(
            (
                execution.graph.provenance.get(
                    "fixed_slot_numeric_temporal_normalization", {}
                )
                or {}
            ).get("source_graph_id", "")
        )
        for execution in variants
    }
    variant_sources.discard("")
    source_graph_id, source_evidence = _selected_source_graph_id(
        selected,
        source_executions,
        variant_sources,
    )
    challenge_source = selected
    if not source_graph_id:
        sources_by_id = {
            str(execution.graph.graph_id): execution
            for execution in source_executions
        }
        cross_source: list[tuple[ExecutedGraph, ExecutedGraph]] = []
        for execution in variants:
            repair = execution.graph.provenance.get(
                "fixed_slot_numeric_temporal_normalization", {}
            ) or {}
            candidate_source_id = str(repair.get("source_graph_id", ""))
            source = sources_by_id.get(candidate_source_id)
            if source is None or not execution.answer_ids:
                continue
            if not repair.get(
                "answer_path_unique_schema_leaf_exact_match"
            ):
                continue
            if (
                float(selected.graph.score) - float(source.graph.score)
                > 0.030000001
            ):
                continue
            candidate_answers = set(map(str, execution.answer_ids))
            source_answers = set(map(str, source.answer_ids))
            if not candidate_answers or not candidate_answers < source_answers:
                continue
            cross_source.append((execution, source))
        cross_answer_sets = {
            tuple(sorted(set(map(str, execution.answer_ids))))
            for execution, _ in cross_source
        }
        if cross_source and len(cross_answer_sets) == 1:
            challenger, source = max(
                cross_source,
                key=lambda item: (
                    float(item[0].graph.score),
                    str(item[0].graph.graph_id),
                ),
            )
            diagnostics.update(
                {
                    "status": "applied",
                    "reason": "cross_source_unique_exact_constraint",
                    "source_resolution": "cross_source_exact_schema_leaf",
                    "source_graph_id": source.graph.graph_id,
                    "selected_graph_id": challenger.graph.graph_id,
                    "answer_count": len(challenger.answer_ids),
                    "variant_graph_ids": [
                        execution.graph.graph_id
                        for execution, _ in cross_source
                    ],
                }
            )
            return challenger, diagnostics
        entity_source, entity_reason = _entity_projection_source(
            selected,
            variants,
            source_executions,
            question=question,
            ontology=ontology,
            expects_entity=expects_entity,
        )
        if entity_source is not None:
            challenge_source = entity_source
            source_graph_id = str(entity_source.graph.graph_id)
            source_evidence = entity_reason
        elif not 2 <= len(set(selected.answer_ids)) <= 256:
            diagnostics["source_resolution"] = entity_reason
            diagnostics["reason"] = "source_cardinality_outside_gate"
            return selected, diagnostics
    diagnostics["source_resolution"] = source_evidence
    if not source_graph_id:
        diagnostics["reason"] = "no_variant_for_selected_source"
        return selected, diagnostics
    diagnostics["source_graph_id"] = source_graph_id
    matching = [
        execution
        for execution in variants
        if str(
            (
                execution.graph.provenance.get(
                    "fixed_slot_numeric_temporal_normalization", {}
                )
                or {}
            ).get("source_graph_id", "")
        ) == source_graph_id
    ]
    matching.sort(
        key=lambda execution: (
            -float(
                (
                    execution.graph.provenance.get(
                        "fixed_slot_numeric_temporal_normalization", {}
                    )
                    or {}
                ).get("lexical_rank_score", 0.0)
            ),
            execution.graph.graph_id,
        )
    )
    if not matching:
        diagnostics["reason"] = "no_variant_for_selected_source"
        return selected, diagnostics
    operator_type = next(
        (
            str(operator.get("type", "")).upper()
            for operator in matching[0].graph.operators
            if isinstance(operator, dict)
            and operator.get("_fixed_slot_numeric_temporal_repair")
        ),
        "",
    )
    prediction, selected_index, reason = choose_fixed_slot_prediction(
        challenge_source.answer_ids,
        [
            {"answer_ids": execution.answer_ids, "answers": execution.answers}
            for execution in matching
        ],
        operator_type=operator_type,
        source_answers=challenge_source.answers,
    )
    diagnostics.update(
        {
            "reason": reason,
            "operator_type": operator_type,
            "variant_graph_ids": [execution.graph.graph_id for execution in matching],
        }
    )
    if selected_index is None:
        return selected, diagnostics
    challenger = matching[selected_index]
    diagnostics.update(
        {
            "status": "applied",
            "selected_graph_id": challenger.graph.graph_id,
            "answer_count": len(prediction),
        }
    )
    return challenger, diagnostics
