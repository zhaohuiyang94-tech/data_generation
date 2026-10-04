"""Conservative, model-free repair for a morphology-signalled under-answer.

The selector only considers answers that have already been completely
materialised by normal query-graph execution.  It recognises one narrow,
generic structure: the incumbent returns one value through a singular answer
predicate while a near-top graph changes only that predicate, pluralises the
same question-mentioned relation head, and returns a strict superset.

No evaluation Gold, endpoint, embedding/model call, entity list, relation
list, or sample identifier is available to this module.  It is intentionally
independent of :mod:`pipeline`; callers can place it after the existing hard,
path, template, TRAIN-extrema and numeric/temporal selectors.
"""

from __future__ import annotations

from collections import Counter
import itertools
import re
import time
from typing import Any, Iterable, Sequence

from .contracts import ExecutedGraph, QueryGraphCandidate


SELECTOR_VERSION = "morphological-underanswer-expansion-production-v2"
REASON_CODE = "morphological_underanswer_expansion_gate"
EXPLICIT_PLURAL_REASON_CODE = "explicit_plural_underanswer_expansion_gate"
EPSILON = 1e-12
MIN_ANSWER_COUNT = 2
MAX_ANSWER_COUNT = 8
MIN_RULE_SCORE_DELTA = -0.08
EXPLICIT_PLURAL_MIN_RULE_SCORE_DELTA = -0.10

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_VARIABLE_RE = re.compile(r"^(?:V\d+|P\d+\.V\d+)$", re.IGNORECASE)
_ENTITY_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_DATE_RE = re.compile(r"^-?\d{1,4}(?:-\d{1,2}(?:-\d{1,2})?)?(?:T.*)?$")
_QUOTED_RE = re.compile(r'["“][^"”]*["”]')
_PLURAL_AUXILIARIES = frozenset(
    {"are", "were", "have", "has", "do", "does", "did", "can", "could", "would", "will"}
)

_BLOCKING_REASON_CODES = frozenset(
    {
        "path_alignment_gate",
        "source_verified_template_consensus_gate",
        "train_gold_path_unique_extrema_gate",
        "fixed_slot_numeric_temporal_normalization",
        "entity_kind_fixed_beam",
    }
)


def _raw_tokens(value: Any) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(str(value)))


def _is_plural_surface(token: str) -> bool:
    value = str(token).casefold()
    return bool(
        len(value) > 3
        and (
            (len(value) > 4 and value.endswith("ies"))
            or (
                value.endswith("s")
                and not value.endswith(("ss", "us", "is"))
            )
        )
    )


def _singular(token: str) -> str:
    value = str(token).casefold()
    if len(value) > 4 and value.endswith("ies"):
        return f"{value[:-3]}y"
    if _is_plural_surface(value):
        return value[:-1]
    return value


def _normalized_tokens(value: Any) -> frozenset[str]:
    return frozenset(_singular(token) for token in _raw_tokens(value))


def _relation_leaf(relation: str) -> str:
    return str(relation).split(".")[-1].replace("_", " ")


def _triples(graph: QueryGraphCandidate) -> list[tuple[str, str, str]]:
    return [
        tuple(map(str, triple))
        for triple in graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    ]


def _constants(graph: QueryGraphCandidate) -> frozenset[str]:
    return frozenset(
        value
        for subject, _, object_ in _triples(graph)
        for value in (subject, object_)
        if not _VARIABLE_RE.fullmatch(value)
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return str(value)


def _operator_signature(graph: QueryGraphCandidate) -> tuple[Any, ...]:
    return tuple(sorted((_freeze(operator) for operator in graph.operators), key=repr))


def _value_kind(value: Any) -> str:
    text = str(value)
    if _ENTITY_RE.fullmatch(text):
        return "entity"
    if _NUMBER_RE.fullmatch(text):
        return "number"
    if _DATE_RE.fullmatch(text):
        return "date"
    return "literal"


def _answer_kind(values: Iterable[Any]) -> str:
    kinds = {_value_kind(value) for value in values}
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def _type_closure(ontology: Any, value: str) -> frozenset[str]:
    if not value:
        return frozenset()
    try:
        parents = ontology.supertypes(value)
    except (AttributeError, KeyError, TypeError, ValueError):
        parents = ()
    return frozenset((str(value), *(str(item) for item in parents)))


def _terminal_edge(
    graph: QueryGraphCandidate, ontology: Any
) -> tuple[str, str, str, str, str] | None:
    answer = str(graph.answer_var)
    edges: list[tuple[str, str, str, str, str]] = []
    for subject, relation, object_ in _triples(graph):
        if object_ == answer:
            answer_type = str(ontology.range_for_relation(relation) or "")
            edges.append((subject, relation, object_, "in", answer_type))
        elif subject == answer:
            answer_type = str(ontology.domain_for_relation(relation) or "")
            edges.append((subject, relation, object_, "out", answer_type))
    return edges[0] if len(edges) == 1 else None


def _one_terminal_predicate_edit(
    incumbent: QueryGraphCandidate,
    challenger: QueryGraphCandidate,
) -> tuple[str, str] | None:
    if str(incumbent.answer_var) != str(challenger.answer_var):
        return None
    left, right = Counter(_triples(incumbent)), Counter(_triples(challenger))
    removed = list((left - right).elements())
    added = list((right - left).elements())
    answer = str(incumbent.answer_var)
    if not (
        len(removed) == len(added) == 1
        and removed[0][0] == added[0][0]
        and removed[0][2] == added[0][2]
        and removed[0][1] != added[0][1]
        and answer in (removed[0][0], removed[0][2])
        and answer in (added[0][0], added[0][2])
    ):
        return None
    return removed[0][1], added[0][1]


def _alpha_terminal_predicate_edit(
    incumbent: QueryGraphCandidate,
    challenger: QueryGraphCandidate,
    ontology: Any,
) -> dict[str, Any] | None:
    """Find one answer-edge predicate edit modulo variable renaming.

    The answer variable is fixed first and at most five remaining variables
    are permuted.  This hard bound keeps the branch well below its runtime
    budget while covering the short CWQ paths for which the gate was designed.
    """

    left, right = _triples(incumbent), _triples(challenger)
    incumbent_answer = str(incumbent.answer_var)
    challenger_answer = str(challenger.answer_var)
    left_variables = sorted(
        {
            value
            for subject, _, object_ in left
            for value in (subject, object_)
            if _VARIABLE_RE.fullmatch(value)
        }
    )
    right_variables = sorted(
        {
            value
            for subject, _, object_ in right
            for value in (subject, object_)
            if _VARIABLE_RE.fullmatch(value)
        }
    )
    if (
        len(left_variables) != len(right_variables)
        or incumbent_answer not in left_variables
        or challenger_answer not in right_variables
    ):
        return None
    left_other = [value for value in left_variables if value != incumbent_answer]
    right_other = [value for value in right_variables if value != challenger_answer]
    if len(left_other) > 5:
        return None
    target = Counter(right)
    for permutation in itertools.permutations(right_other):
        mapping = {
            incumbent_answer: challenger_answer,
            **dict(zip(left_other, permutation)),
        }
        mapped = Counter(
            (mapping.get(subject, subject), relation, mapping.get(object_, object_))
            for subject, relation, object_ in left
        )
        removed = list((mapped - target).elements())
        added = list((target - mapped).elements())
        if not (
            len(removed) == len(added) == 1
            and removed[0][0] == added[0][0]
            and removed[0][2] == added[0][2]
            and removed[0][1] != added[0][1]
            and challenger_answer in (removed[0][0], removed[0][2])
        ):
            continue
        direction = "out" if removed[0][0] == challenger_answer else "in"
        incumbent_type = str(
            (
                ontology.domain_for_relation(removed[0][1])
                if direction == "out"
                else ontology.range_for_relation(removed[0][1])
            )
            or ""
        )
        challenger_type = str(
            (
                ontology.domain_for_relation(added[0][1])
                if direction == "out"
                else ontology.range_for_relation(added[0][1])
            )
            or ""
        )
        if not (
            _type_closure(ontology, incumbent_type)
            & _type_closure(ontology, challenger_type)
        ):
            continue
        return {
            "incumbent_terminal_relation": removed[0][1],
            "challenger_terminal_relation": added[0][1],
            "answer_direction": direction,
            "incumbent_answer_type": incumbent_type,
            "challenger_answer_type": challenger_type,
            "alpha_variable_mapping": dict(sorted(mapping.items())),
        }
    return None


def _explicit_plural_request(question: str) -> dict[str, Any] | None:
    """Recognise a plural interrogative head immediately before an auxiliary.

    This syntactic shape distinguishes a noun such as ``movies`` in
    ``what fantasy movies has ...`` from third-person verbs such as
    ``which country includes ...`` without maintaining a vocabulary list.
    """

    tokens = list(_raw_tokens(_QUOTED_RE.sub(" ", str(question))))
    for question_index, token in enumerate(tokens[:4]):
        if token not in {"what", "which"}:
            continue
        for auxiliary_index in range(question_index + 2, min(len(tokens), question_index + 7)):
            if tokens[auxiliary_index] not in _PLURAL_AUXILIARIES:
                continue
            head = tokens[auxiliary_index - 1]
            if _is_plural_surface(head):
                return {
                    "plural_head": head,
                    "auxiliary": tokens[auxiliary_index],
                    "question_span": tokens[question_index : auxiliary_index + 1],
                }
            break
    return None


def _explicit_plural_expansion_candidates(
    question: str,
    selected: ExecutedGraph,
    executions: Sequence[ExecutedGraph],
    ontology: Any,
    *,
    min_answer_count: int,
    max_answer_count: int,
    min_rule_score_delta: float,
) -> list[tuple[ExecutedGraph, dict[str, Any]]]:
    """Gold-free explicit-plural lane, before the singleton safety guard."""

    plural = _explicit_plural_request(question)
    if plural is None:
        return []
    selected_answers = frozenset(map(str, selected.answer_ids))
    if not selected_answers:
        return []
    eligible: list[tuple[ExecutedGraph, dict[str, Any]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for execution in executions:
        answers = frozenset(map(str, execution.answer_ids))
        key = (str(execution.graph.graph_id), tuple(sorted(answers)))
        if key in seen:
            continue
        seen.add(key)
        rule_delta = float(execution.graph.score) - float(selected.graph.score)
        if not (
            selected_answers < answers
            and min_answer_count <= len(answers) <= max_answer_count
            and rule_delta + EPSILON >= min_rule_score_delta
            and _answer_kind(answers) == _answer_kind(selected_answers)
            and _operator_signature(execution.graph)
            == _operator_signature(selected.graph)
            and _constants(execution.graph) == _constants(selected.graph)
        ):
            continue
        alpha_edit = _alpha_terminal_predicate_edit(
            selected.graph, execution.graph, ontology
        )
        if alpha_edit is None:
            continue
        eligible.append(
            (
                execution,
                {
                    "branch": "explicit_plural_singleton",
                    "reason_code": EXPLICIT_PLURAL_REASON_CODE,
                    "graph_id": str(execution.graph.graph_id),
                    "answer_count": len(answers),
                    "rule_score_delta": rule_delta,
                    **alpha_edit,
                    **plural,
                },
            )
        )
    return eligible


def _plural_head_evidence(
    question: str, incumbent_relation: str, challenger_relation: str
) -> dict[str, Any] | None:
    question_tokens = _normalized_tokens(question)
    incumbent_raw = _raw_tokens(_relation_leaf(incumbent_relation))
    challenger_raw = _raw_tokens(_relation_leaf(challenger_relation))
    incumbent_tokens = frozenset(_singular(token) for token in incumbent_raw)
    challenger_tokens = frozenset(_singular(token) for token in challenger_raw)
    challenger_plural_heads = frozenset(
        _singular(token) for token in challenger_raw if _is_plural_surface(token)
    )
    incumbent_plural_heads = frozenset(
        _singular(token) for token in incumbent_raw if _is_plural_surface(token)
    )
    shared_plural_heads = (
        challenger_plural_heads
        & incumbent_tokens
        & challenger_tokens
        & question_tokens
    ) - incumbent_plural_heads
    incumbent_modifiers = (incumbent_tokens - challenger_tokens) & question_tokens
    if not shared_plural_heads or not incumbent_modifiers:
        return None
    return {
        "shared_plural_heads": sorted(shared_plural_heads),
        "question_matched_incumbent_modifiers": sorted(incumbent_modifiers),
    }


def _higher_priority_selected(decision: dict[str, Any]) -> str:
    reasons = {str(reason) for reason in decision.get("reason_codes", [])}
    if any(reason.startswith("hard_") for reason in reasons):
        return "preserve_hard_post_selection_gate"
    blocked = sorted(reasons & _BLOCKING_REASON_CODES)
    return f"preserve_{blocked[0]}" if blocked else ""


class UnderanswerExpansionSelector:
    """Select one uniquely supported, morphology-signalled answer superset."""

    def __init__(
        self,
        ontology: Any,
        *,
        min_answer_count: int = MIN_ANSWER_COUNT,
        max_answer_count: int = MAX_ANSWER_COUNT,
        min_rule_score_delta: float = MIN_RULE_SCORE_DELTA,
        explicit_plural_min_rule_score_delta: float = EXPLICIT_PLURAL_MIN_RULE_SCORE_DELTA,
    ) -> None:
        self.ontology = ontology
        self.min_answer_count = max(2, int(min_answer_count))
        self.max_answer_count = max(self.min_answer_count, int(max_answer_count))
        self.min_rule_score_delta = float(min_rule_score_delta)
        self.explicit_plural_min_rule_score_delta = float(
            explicit_plural_min_rule_score_delta
        )

    def challenge(
        self,
        *,
        question: str,
        selected: ExecutedGraph,
        executions: Sequence[ExecutedGraph],
        prior_decision: dict[str, Any],
    ) -> tuple[ExecutedGraph, dict[str, Any]]:
        started = time.perf_counter()
        evidence: dict[str, Any] = {
            "version": SELECTOR_VERSION,
            "status": "not_applicable",
            "model_calls": 0,
            "endpoint_queries": 0,
            "embedding_calls": 0,
            "uses_evaluation_gold": False,
            "entity_relation_or_sample_whitelist": False,
            "frozen_gate": {
                "min_answer_count": self.min_answer_count,
                "max_answer_count": self.max_answer_count,
                "minimum_rule_score_delta": self.min_rule_score_delta,
                "explicit_plural_minimum_rule_score_delta": (
                    self.explicit_plural_min_rule_score_delta
                ),
                "incumbent_answer_count": 1,
                "requires_strict_superset": True,
                "requires_unique_answer_set": True,
                "requires_exact_operator_signature": True,
                "requires_single_terminal_predicate_edit": True,
                "requires_ontology_type_compatibility": True,
                "requires_question_derived_plural_head": True,
            },
        }
        blocked = _higher_priority_selected(prior_decision)
        if blocked:
            evidence.update(
                {
                    "status": "skipped_existing_higher_priority_gate",
                    "reason": blocked,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        selected_answers = frozenset(map(str, selected.answer_ids))
        if len(selected_answers) != 1 or self.ontology is None:
            evidence.update(
                {
                    "status": "incumbent_not_singleton_or_ontology_missing",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        incumbent_terminal = _terminal_edge(selected.graph, self.ontology)
        explicit_plural = _explicit_plural_request(question)

        eligible: list[tuple[ExecutedGraph, dict[str, Any]]] = []
        explicit_eligible: list[tuple[ExecutedGraph, dict[str, Any]]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for execution in executions:
            answers = frozenset(map(str, execution.answer_ids))
            key = (str(execution.graph.graph_id), tuple(sorted(answers)))
            if key in seen:
                continue
            seen.add(key)
            if not (
                selected_answers < answers
                and self.min_answer_count <= len(answers) <= self.max_answer_count
                and _answer_kind(answers) == _answer_kind(selected_answers)
                and _operator_signature(execution.graph)
                == _operator_signature(selected.graph)
                and _constants(execution.graph) == _constants(selected.graph)
            ):
                continue
            rule_delta = float(execution.graph.score) - float(selected.graph.score)
            if (
                incumbent_terminal is not None
                and rule_delta + EPSILON >= self.min_rule_score_delta
            ):
                terminal = _terminal_edge(execution.graph, self.ontology)
                edit = _one_terminal_predicate_edit(selected.graph, execution.graph)
                if (
                    terminal is not None
                    and edit is not None
                    and terminal[3] == incumbent_terminal[3]
                ):
                    incumbent_types = _type_closure(
                        self.ontology, incumbent_terminal[4]
                    )
                    challenger_types = _type_closure(self.ontology, terminal[4])
                    morphology = _plural_head_evidence(question, edit[0], edit[1])
                    if (
                        incumbent_types
                        and challenger_types
                        and incumbent_types & challenger_types
                        and morphology is not None
                    ):
                        eligible.append(
                            (
                                execution,
                                {
                                    "branch": "relation_head_pluralization",
                                    "reason_code": REASON_CODE,
                                    "graph_id": str(execution.graph.graph_id),
                                    "answer_count": len(answers),
                                    "rule_score_delta": rule_delta,
                                    "incumbent_terminal_relation": edit[0],
                                    "challenger_terminal_relation": edit[1],
                                    "answer_direction": terminal[3],
                                    "incumbent_answer_type": incumbent_terminal[4],
                                    "challenger_answer_type": terminal[4],
                                    **morphology,
                                },
                            )
                        )
        if explicit_plural is not None:
            explicit_eligible = _explicit_plural_expansion_candidates(
                question,
                selected,
                executions,
                self.ontology,
                min_answer_count=self.min_answer_count,
                max_answer_count=self.max_answer_count,
                min_rule_score_delta=self.explicit_plural_min_rule_score_delta,
            )
        answer_sets = {
            frozenset(map(str, execution.answer_ids)) for execution, _ in eligible
        }
        if len(answer_sets) > 1:
            evidence.update(
                {
                    "status": "no_unique_supported_expansion",
                    "eligible_candidate_count": len(eligible),
                    "eligible_answer_set_count": len(answer_sets),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        selected_lane = eligible
        reason = "unique_question_derived_plural_terminal_expansion"
        if not selected_lane:
            explicit_answer_sets = {
                frozenset(map(str, execution.answer_ids))
                for execution, _ in explicit_eligible
            }
            if len(explicit_answer_sets) != 1:
                evidence.update(
                    {
                        "status": "no_unique_supported_expansion",
                        "eligible_candidate_count": len(explicit_eligible),
                        "eligible_answer_set_count": len(explicit_answer_sets),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                return selected, evidence
            selected_lane = explicit_eligible
            reason = "explicit_plural_singleton_alpha_equivalent_expansion"
        chosen, proposal = max(
            selected_lane,
            key=lambda item: (
                float(item[0].graph.score),
                str(item[0].graph.graph_id),
            ),
        )
        evidence.update(
            {
                "status": "applied",
                "reason": reason,
                "reason_code": proposal["reason_code"],
                "eligible_candidate_count": len(selected_lane),
                "proposal": proposal,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return chosen, evidence


__all__ = [
    "EXPLICIT_PLURAL_MIN_RULE_SCORE_DELTA",
    "EXPLICIT_PLURAL_REASON_CODE",
    "MAX_ANSWER_COUNT",
    "MIN_ANSWER_COUNT",
    "MIN_RULE_SCORE_DELTA",
    "REASON_CODE",
    "SELECTOR_VERSION",
    "UnderanswerExpansionSelector",
]
