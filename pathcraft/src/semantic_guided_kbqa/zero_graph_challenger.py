"""Gold-blind, model-free challengers for structurally lossy selected graphs.

The runtime gate and candidate generator use only the question, already-grounded
semantic paths, the final graph beam, and the local ontology.  They deliberately
have no dataset index or Gold-answer input.  Callers are expected to execute at
most the first three returned candidates after the gate fires.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import islice, product
import json
import re
from typing import Any, Iterable

from .contracts import GroundedSemanticCandidate, QueryGraphCandidate
from .lowering import LoweringError, lower_sparql


_VERSION = "zero-graph-challenger-v1"
_SCALAR_TYPES = {
    "type.boolean",
    "type.datetime",
    "type.enumeration",
    "type.float",
    "type.int",
    "type.key",
    "type.rawstring",
    "type.text",
}
_LOWER_CUE = re.compile(
    r"\b(?:lower|less|fewer|smaller)\s+than\b|\bprior\s+to\b|\bbefore\b|\bunder\b|\bbelow\b",
    re.I,
)
_UPPER_CUE = re.compile(
    r"\b(?:more|greater|higher|larger)\s+than\b|\bafter\b|\babove\b|\bover\b",
    re.I,
)
_ARGMIN_CUE = re.compile(r"\b(?:earliest|oldest|first|least|lowest|smallest)\b", re.I)
_ARGMAX_CUE = re.compile(
    r"\b(?:latest|newest|last|most\s+recent|highest|largest|greatest)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class ZeroGraphCandidate:
    """One ranked executable challenger produced without endpoint/model work."""

    key: str
    strategy: str
    score: float
    graph: QueryGraphCandidate
    sparql: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PathView:
    path_id: str
    anchor: str
    variables: tuple[str, ...]
    terminal: str
    triples: tuple[tuple[str, str, str], ...]
    node_types: dict[str, tuple[str, ...]]
    goal: str


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _relation_counter(grounded: GroundedSemanticCandidate) -> Counter[str]:
    return Counter(str(value) for value in grounded.relation_bindings.values())


def _counter_covers(candidate: Counter[str], required: Counter[str]) -> bool:
    return all(candidate[key] >= count for key, count in required.items())


def _grounded_path_shape(
    grounded: GroundedSemanticCandidate,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    paths = grounded.compose_input.get("semantic_paths", [])
    return (
        tuple(
            sorted(
                len(path.get("steps", []))
                for path in paths
                if isinstance(path, dict)
            )
        ),
        tuple(
            str(step.get("direction", ""))
            for path in paths
            if isinstance(path, dict)
            for step in path.get("steps", [])
            if isinstance(step, dict)
        ),
    )


def _repeated_relation_named_twice_in_goal(
    grounded: GroundedSemanticCandidate,
) -> bool:
    """Return whether a repeated binding is explicitly repeated in its goal.

    This is lexical evidence from the saved semantic path, not a relation or
    question whitelist.  It distinguishes a genuine same-property traversal
    (for example, ``currency used ... currency used``) from an accidental
    grounding that maps two semantically different steps to the anchor edge.
    """

    repeated_relations = {
        relation
        for relation, count in _relation_counter(grounded).items()
        if count > 1
    }
    if not repeated_relations:
        return False
    for path in grounded.compose_input.get("semantic_paths", []):
        if not isinstance(path, dict):
            continue
        goal = re.sub(r"\s+", " ", str(path.get("goal", "")).casefold())
        for step in path.get("steps", []):
            if not isinstance(step, dict):
                continue
            relation = str(
                grounded.relation_bindings.get(str(step.get("id", "")), "")
            )
            if relation not in repeated_relations:
                continue
            raw_label = step.get("relation_label", [])
            if isinstance(raw_label, list) and raw_label:
                label = str(raw_label[-1])
            else:
                label = str(raw_label)
            label = re.sub(r"[\s_]+", " ", label.casefold()).strip()
            if label and len(
                re.findall(rf"(?<!\w){re.escape(label)}(?!\w)", goal)
            ) >= 2:
                return True
    return False


def _expected_operator_type(question: str) -> str:
    if _LOWER_CUE.search(question) and not _UPPER_CUE.search(question):
        return "LESS_THAN"
    if _UPPER_CUE.search(question) and not _LOWER_CUE.search(question):
        return "GREATER_THAN"
    if _ARGMIN_CUE.search(question) and not _ARGMAX_CUE.search(question):
        return "ARGMIN"
    if _ARGMAX_CUE.search(question) and not _ARGMIN_CUE.search(question):
        return "ARGMAX"
    return ""


def _comparison_cue_has_following_scalar(question: str, expected: str) -> bool:
    cue = _LOWER_CUE if expected == "LESS_THAN" else _UPPER_CUE
    return any(
        re.search(r"\d", question[match.end() : match.end() + 40])
        for match in cue.finditer(question)
    )


def _normalized_value_in_question(value: Any, question: str) -> bool:
    """Numeric/date/string surface normalization; never an ID whitelist."""

    raw = str(value).strip().casefold()
    if not raw:
        return False

    def normalize_numeric_phrase(text: str) -> str:
        text = re.sub(
            r"\b(\d+)\s*(?:games?\s*)?(?:-|to)\s*(\d+)\b",
            r"\1-\2",
            text,
        )
        return re.sub(r"[\s,_-]+", "", text)

    compact = normalize_numeric_phrase(raw)
    question_compact = normalize_numeric_phrase(question.casefold())
    return len(compact) >= 2 and compact in question_compact


def _safe_scalar_operator_value(operator: dict[str, Any]) -> bool:
    value = str(operator.get("value", "")).strip()
    value_type = str(operator.get("value_type", "")).casefold()
    if value_type in {
        "number",
        "int",
        "integer",
        "float",
        "double",
        "date",
        "datetime",
        "type.datetime",
    }:
        return True
    return bool(
        re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value)
        or re.fullmatch(r"\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?", value)
        or re.fullmatch(r"\d+\s*-\s*\d+", value)
    )


def _repaired_operators(
    question: str,
    source_operators: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, bool]:
    expected = _expected_operator_type(question)
    result: list[dict[str, Any]] = []
    repaired = False
    value_supported = False
    for source in source_operators:
        operator = deepcopy(source)
        operator_type = str(operator.get("type", "")).upper()
        value = str(operator.get("value", "")).strip()
        if value and _normalized_value_in_question(value, question):
            value_supported = True
        if (
            expected in {"LESS_THAN", "GREATER_THAN"}
            and value
            and _safe_scalar_operator_value(operator)
            and _comparison_cue_has_following_scalar(question, expected)
            and operator_type
            in {
                "EQUAL",
                "LESS_THAN",
                "LESS_OR_EQUAL",
                "GREATER_THAN",
                "GREATER_OR_EQUAL",
            }
        ):
            if operator_type != expected:
                repaired = True
            operator["type"] = expected
            if operator.get("_numeric_cast_compare"):
                operator["_numeric_comparison_cast"] = True
            if str(operator.get("value_type", "")).casefold() in {
                "number",
                "int",
                "integer",
                "float",
                "double",
            }:
                operator["_numeric_comparison_cast"] = True
                operator.setdefault(
                    "_numeric_cast_kind",
                    "float" if "." in value else "integer",
                )
        elif expected in {"ARGMIN", "ARGMAX"} and operator_type in {
            "ARGMIN",
            "ARGMAX",
        }:
            if operator_type != expected:
                repaired = True
            operator["type"] = expected
        result.append(operator)
    return result, repaired, value_supported


def zero_graph_challenger_gate(
    *,
    question: str,
    selected_graph: QueryGraphCandidate,
    grounded_candidates: list[GroundedSemanticCandidate],
    final_graphs: list[QueryGraphCandidate],
) -> tuple[bool, list[str]]:
    """Conservative structural gate using runtime-visible state only."""

    if not grounded_candidates:
        return False, []
    reasons: list[str] = []
    selected_relations = Counter(
        str(triple[1])
        for triple in selected_graph.triples
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    )
    selected_score = float(selected_graph.score)
    ranked_grounded = sorted(
        grounded_candidates,
        key=lambda value: (
            -float(value.score),
            json.dumps(
                {
                    "compose_input": value.compose_input,
                    "anchor_bindings": sorted(
                        (key, candidate.entity_id)
                        for key, candidate in value.anchor_bindings.items()
                    ),
                    "relation_bindings": value.relation_bindings,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        ),
    )
    top = ranked_grounded[0]
    top_relations = _relation_counter(top)
    top_score = float(top.score)

    repeated = any(count > 1 for count in top_relations.values())
    top_signature = tuple(sorted(top_relations.elements()))
    signature_support = sum(
        tuple(sorted(_relation_counter(value).elements())) == top_signature
        for value in ranked_grounded
    )
    path_lengths, directions = _grounded_path_shape(top)
    selected_triple_count = sum(
        isinstance(triple, (list, tuple)) and len(triple) == 3
        for triple in selected_graph.triples
    )
    selected_fallback = bool(selected_graph.provenance.get("graph_fallback"))
    selected_has_substantive_operator = any(
        isinstance(operator, dict)
        and str(operator.get("type", "")).upper() != "NO_EQUAL"
        for operator in selected_graph.operators
    )
    score_gap = top_score - selected_score
    repeated_safe_shape = (
        (signature_support >= 2 and path_lengths == (2,))
        or (
            path_lengths == (2, 4)
            and not re.search(r"\b(?:1\d{3}|20\d{2})\b", question)
        )
        or (
            path_lengths == (2,)
            and directions == ("backward", "forward")
            and top_score >= 0.82
            and not re.search(r'["“”]', question)
        )
    )
    conservative_repeated_loss = (
        repeated
        and score_gap >= 0.05
        and repeated_safe_shape
        and not _counter_covers(selected_relations, top_relations)
    )
    five_edge_two_path_repeated_loss = (
        repeated
        and not _counter_covers(selected_relations, top_relations)
        and path_lengths == (2, 3)
        and score_gap >= 0.05
    )
    broadened_repeated_loss = (
        repeated
        and not _counter_covers(selected_relations, top_relations)
        and (
            # A five-edge two-path graph can lose one mediator-side edge
            # during composition.  The old gate covered [2,4] but not [2,3].
            five_edge_two_path_repeated_loss
            # Three independently anchored two-hop paths provide enough
            # agreement to admit a smaller score margin.
            or (path_lengths == (2, 2, 2) and score_gap >= 0.02)
            # Two symmetric two-hop paths are safe only when the selected
            # graph has the same four-edge capacity and the direction pattern
            # is fully explicit.
            or (
                path_lengths == (2, 2)
                and directions
                == ("backward", "forward", "backward", "forward")
                and selected_triple_count == 4
                and score_gap >= 0.005
            )
            # Require the semantic goal itself to name the repeated property
            # twice before lowering the single-path score margin.
            or (
                path_lengths == (2,)
                and directions == ("backward", "forward")
                and score_gap >= 0.03
                and _repeated_relation_named_twice_in_goal(top)
            )
        )
    )
    lower_rank_repeated_loss = False
    for rank, candidate in enumerate(ranked_grounded[:4], start=1):
        candidate_relations = _relation_counter(candidate)
        if (
            not any(count > 1 for count in candidate_relations.values())
            or _counter_covers(selected_relations, candidate_relations)
        ):
            continue
        candidate_lengths, candidate_directions = _grounded_path_shape(candidate)
        candidate_gap = float(candidate.score) - selected_score
        if (
            selected_fallback
            and selected_triple_count == 1
            and rank <= 2
            and candidate_lengths == (2,)
            and candidate_directions == ("backward", "forward")
            and candidate_gap >= 0.01
        ) or (
            selected_fallback
            and not selected_has_substantive_operator
            and candidate_lengths == (2, 4)
            and candidate_gap >= -0.005
            and not re.search(r"\b(?:1\d{3}|20\d{2})\b", question)
        ):
            lower_rank_repeated_loss = True
            break
    if conservative_repeated_loss or broadened_repeated_loss or lower_rank_repeated_loss:
        reasons.append("repeated_grounded_predicate_lost")
        if five_edge_two_path_repeated_loss:
            reasons.append("five_edge_two_path_repeated_loss")

    # A graph fallback can select a semantically different two-edge chain even
    # though a higher-scoring forward chain is available.  Never challenge an
    # operator-bearing selection here: path-only reconstruction would discard
    # the scalar/temporal constraint that made the original answer exact.
    if (
        selected_fallback
        and not selected_has_substantive_operator
        and path_lengths == (2,)
        and directions == ("forward", "forward")
        and score_gap >= 0.025
        # Sharing a relation means the fallback retained the grounded prefix
        # and changed only its CVT projection.  Replacing that partially useful
        # graph with the whole forward chain can discard correct answers.  This
        # lane is safe only when the selected graph missed the chain entirely.
        and not bool(selected_relations.keys() & top_relations.keys())
        and not _counter_covers(selected_relations, top_relations)
    ):
        reasons.append("operator_free_fallback_underuses_forward_chain")

    expected_operator = _expected_operator_type(question)
    selected_operators = [
        operator
        for operator in selected_graph.operators
        if isinstance(operator, dict)
        and str(operator.get("type", "")).upper() != "NO_EQUAL"
    ]
    selected_types = {
        str(operator.get("type", "")).upper() for operator in selected_operators
    }
    operator_graphs: list[tuple[float, bool, bool, bool]] = []
    for graph in final_graphs:
        operators, type_repaired, value_supported = _repaired_operators(
            question,
            [item for item in graph.operators if isinstance(item, dict)],
        )
        if any(str(item.get("type", "")).upper() != "NO_EQUAL" for item in operators):
            operator_graphs.append(
                (
                    float(graph.score),
                    type_repaired,
                    value_supported,
                    any(_safe_scalar_operator_value(item) for item in operators),
                )
            )
    if (
        expected_operator in {"LESS_THAN", "GREATER_THAN"}
        and expected_operator not in selected_types
        and any(
            type_repaired and value_supported and safe_scalar
            for _, type_repaired, value_supported, safe_scalar in operator_graphs
        )
    ):
        reasons.append("explicit_comparison_operator_conflict")
    if (
        not selected_operators
        and any(
            score >= selected_score + 0.05 and value_supported and safe_scalar
            for score, _, value_supported, safe_scalar in operator_graphs
        )
    ):
        reasons.append("explicit_value_constraint_dropped")
    return bool(reasons), reasons


def zero_graph_replacement_is_safe(
    *,
    selected_graph: QueryGraphCandidate,
    selected_answer_ids: Iterable[str],
    challenger: ZeroGraphCandidate,
    challenger_answer_ids: Iterable[str],
    gate_reasons: Iterable[str],
) -> tuple[bool, str]:
    """Guard a non-empty replacement using only observed runtime evidence.

    A missing scalar comparison is a narrowing operation.  When the current
    graph has no substantive operator and already returns multiple answers, a
    challenger is safe only if it strictly narrows that same answer set.  This
    prevents an unrelated reconstructed path from replacing a partially useful
    result merely because the intended operator graph happened to be empty.

    Multi-path repeated-predicate reconstruction is similarly conservative for
    an existing multi-answer result: it may refine the observed set, but may not
    replace it with a disjoint/different denotation.  Single-path sibling
    reconstruction remains unaffected; that is the validated high-yield lane.
    """

    selected = {str(value) for value in selected_answer_ids if str(value)}
    candidate = {str(value) for value in challenger_answer_ids if str(value)}
    reasons = set(map(str, gate_reasons))
    if not candidate:
        return False, "empty_challenger"

    selected_types = {
        str(operator.get("type", "")).upper()
        for operator in selected_graph.operators
        if isinstance(operator, dict)
        and str(operator.get("type", "")).upper() != "NO_EQUAL"
    }
    if (
        "explicit_comparison_operator_conflict" in reasons
        and not selected_types
        and len(selected) > 1
        and not candidate < selected
    ):
        return False, "comparison_result_does_not_strictly_narrow_current_answers"

    path_count = int(challenger.metadata.get("path_count", 0) or 0)
    if (
        "repeated_grounded_predicate_lost" in reasons
        and "five_edge_two_path_repeated_loss" not in reasons
        and path_count > 1
        and len(selected) > 1
        and not candidate < selected
    ):
        return False, "multipath_reconstruction_does_not_strictly_narrow_current_answers"
    return True, ""


def _type_compatibility(
    ontology: Any,
    left: Iterable[str],
    right: Iterable[str],
) -> float:
    left_values = {value for value in left if value}
    right_values = {value for value in right if value}
    if not left_values or not right_values:
        return 0.25
    best = -1.0
    for left_type in left_values:
        for right_type in right_values:
            if left_type == right_type:
                best = max(best, 4.0)
                continue
            left_supers = set(ontology.supertypes(left_type))
            right_supers = set(ontology.supertypes(right_type))
            if left_type in right_supers or right_type in left_supers:
                best = max(best, 3.0)
                continue
            informative = {
                value
                for value in left_supers & right_supers
                if value != "common.topic"
                and not value.startswith("base.type_ontology.")
                and value not in {"type.object", "type.entity"}
            }
            if informative:
                best = max(best, 2.0)
    return best


def _path_views(
    grounded: GroundedSemanticCandidate,
    ontology: Any,
) -> list[_PathView]:
    result: list[_PathView] = []
    for path in grounded.compose_input.get("semantic_paths", []):
        if not isinstance(path, dict):
            continue
        path_id = str(path.get("id", ""))
        anchor_ref = str(path.get("anchor_ref", ""))
        anchor_candidate = grounded.anchor_bindings.get(anchor_ref)
        anchor = str(anchor_candidate.entity_id) if anchor_candidate is not None else ""
        if not anchor:
            continue
        triples: list[tuple[str, str, str]] = []
        variables: list[str] = []
        node_types: dict[str, set[str]] = defaultdict(set)
        valid = True
        for step in path.get("steps", []):
            if not isinstance(step, dict):
                valid = False
                break
            relation = str(grounded.relation_bindings.get(str(step.get("id", "")), ""))
            source_ref = str(step.get("from", ""))
            target_ref = str(step.get("to", ""))
            source = anchor if source_ref == anchor_ref else source_ref
            target = target_ref
            if not relation or not source or not target:
                valid = False
                break
            if target not in variables:
                variables.append(target)
            if str(step.get("direction", "")) == "forward":
                subject, object_ = source, target
            elif str(step.get("direction", "")) == "backward":
                subject, object_ = target, source
            else:
                valid = False
                break
            triples.append((subject, relation, object_))
            if subject.startswith("P"):
                node_types[subject].add(str(ontology.domain_for_relation(relation)))
            if object_.startswith("P"):
                node_types[object_].add(str(ontology.range_for_relation(relation)))
        terminal = str(path.get("path_output_var", ""))
        if valid and triples and terminal in variables:
            result.append(
                _PathView(
                    path_id=path_id,
                    anchor=anchor,
                    variables=tuple(variables),
                    terminal=terminal,
                    triples=tuple(triples),
                    node_types={
                        key: tuple(sorted(value for value in values if value))
                        for key, values in node_types.items()
                    },
                    goal=str(path.get("goal", "")),
                )
            )
    return result


def _entity_question(question: str) -> bool:
    return bool(re.match(r"\s*(?:who|which|what)\b", question, re.I)) and not bool(
        re.match(r"\s*(?:when|how many|what (?:year|date|time))\b", question, re.I)
    )


def _word_tokens(value: str) -> set[str]:
    result: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9]+", value.casefold()):
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        result.add(token)
    return result


def _answer_semantic_bonus(
    question: str,
    paths: list[_PathView],
    base: _PathView,
    answer: str,
    ontology: Any,
) -> float:
    node_types = set(base.node_types.get(answer, ()))
    inherited = {
        parent
        for node_type in node_types
        for parent in ontology.supertypes(node_type)
    }
    score = 0.0
    if re.search(r"\bwho\b", question, re.I):
        score += 10.0 if "people.person" in inherited else (-2.0 if inherited else 0.0)
    focus_tokens = _word_tokens(" ".join([question, *(path.goal for path in paths)]))
    type_tokens = {
        token
        for type_id in inherited | node_types
        for token in _word_tokens(type_id.replace("_", " ").replace(".", " "))
    }
    score += 1.5 * len(focus_tokens & type_tokens)
    incident_relations = {
        relation
        for subject, relation, object_ in base.triples
        if answer in {subject, object_}
    }
    relation_tokens = {
        token
        for relation in incident_relations
        for token in _word_tokens(relation.rsplit(".", 1)[-1].replace("_", " "))
    }
    score += 3.0 * len(focus_tokens & relation_tokens)
    return score


def _compile(
    *,
    question: str,
    grounded: GroundedSemanticCandidate,
    paths: list[_PathView],
    merges: list[tuple[str, str]],
    answer: str,
    strategy: str,
    base_score: float,
    join_score: float,
    trim_at_join: bool,
    ontology: Any,
    pipeline_version: str,
) -> ZeroGraphCandidate | None:
    variables = {value for path in paths for value in path.variables}
    if answer not in variables:
        return None
    union = _UnionFind(variables)
    for left, right in merges:
        if left not in variables or right not in variables:
            return None
        union.union(left, right)
    retained: list[tuple[str, str, str]] = []
    join_nodes = {value for pair in merges for value in pair}
    for path in paths:
        triples = list(path.triples)
        if trim_at_join:
            relevant = [
                path.variables.index(value)
                for value in path.variables
                if value in join_nodes or value == answer
            ]
            if relevant:
                triples = triples[: max(relevant) + 1]
        retained.extend(triples)

    root_names: dict[str, str] = {}
    lowered: list[list[str]] = []
    for subject, relation, object_ in retained:
        nodes: list[str] = []
        for node in (subject, object_):
            if node in variables:
                root = union.find(node)
                if root not in root_names:
                    root_names[root] = f"V{len(root_names)}"
                nodes.append(root_names[root])
            else:
                nodes.append(node)
        if nodes[0] == nodes[1]:
            return None
        triple = [nodes[0], relation, nodes[1]]
        if triple not in lowered:
            lowered.append(triple)
    answer_var = root_names.get(union.find(answer), "")
    if not answer_var or not any(answer_var in (triple[0], triple[2]) for triple in lowered):
        return None

    terminal_support = sum(
        union.find(path.terminal) == union.find(answer) for path in paths
    )
    answer_local_nodes = {
        value for value in variables if union.find(value) == union.find(answer)
    }
    answer_types = {
        type_id
        for path in paths
        for node in answer_local_nodes
        for type_id in path.node_types.get(node, ())
    }
    answer_supertypes = {
        inherited
        for type_id in answer_types
        for inherited in ontology.supertypes(type_id)
    }
    score = base_score + join_score + (4.0 * terminal_support)
    if trim_at_join:
        score -= 0.5
    metadata = {
        "path_count": len(paths),
        "edge_count": len(lowered),
        "terminal_support": terminal_support,
        "trimmed": trim_at_join,
        "merge_count": len(merges),
        "answer_types": sorted(answer_types),
        "answer_relations": sorted(
            relation
            for subject, relation, object_ in lowered
            if answer_var in {subject, object_}
        ),
        "who_person_compatible": "people.person" in answer_supertypes,
    }
    graph = QueryGraphCandidate(
        graph_id="ZGC",
        triples=lowered,
        answer_var=answer_var,
        operators=[],
        score=score,
        compose_output={
            "source": "zero_graph_challenger",
            "strategy": strategy,
        },
        provenance={
            "version": str(pipeline_version),
            "anchor_bindings": {
                key: {
                    "id": value.entity_id,
                    "label": value.label,
                    "score": value.score,
                }
                for key, value in grounded.anchor_bindings.items()
            },
            "relation_bindings": dict(grounded.relation_bindings),
            "grounded_semantic": deepcopy(grounded.provenance),
            "zero_graph_challenger": {
                "version": _VERSION,
                "strategy": strategy,
                "uses_gold": False,
                "model_used": False,
            },
        },
    )
    try:
        sparql = lower_sparql(graph)
    except (LoweringError, ValueError, TypeError):
        return None
    key = json.dumps(
        [lowered, answer_var],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ZeroGraphCandidate(key, strategy, score, graph, sparql, metadata)


def _single_path_candidates(
    *,
    question: str,
    grounded: GroundedSemanticCandidate,
    ontology: Any,
    pipeline_version: str,
) -> list[ZeroGraphCandidate]:
    paths = _path_views(grounded, ontology)
    if len(paths) != 1:
        return []
    path = paths[0]
    result: list[ZeroGraphCandidate] = []
    for answer in path.variables:
        answer_types = set(path.node_types.get(answer, ()))
        if _entity_question(question) and answer_types and answer_types <= _SCALAR_TYPES:
            continue
        base_score = (
            10.0 * float(grounded.score)
            + _answer_semantic_bonus(question, paths, path, answer, ontology)
            + (12.0 if answer == path.terminal else 0.0)
        )
        for trimmed in (False, True):
            candidate = _compile(
                question=question,
                grounded=grounded,
                paths=paths,
                merges=[],
                answer=answer,
                strategy="grounded_path_answer_enumeration",
                base_score=base_score,
                join_score=0.0,
                trim_at_join=trimmed,
                ontology=ontology,
                pipeline_version=pipeline_version,
            )
            if candidate is None:
                continue
            metadata = {
                **candidate.metadata,
                "raw_score": float(grounded.score),
                "relation_multiset": sorted(_relation_counter(grounded).elements()),
                "answer_is_terminal": answer == path.terminal,
                "full_path": not trimmed,
            }
            result.append(replace(candidate, metadata=metadata))
    return result


def _multipath_candidates(
    *,
    question: str,
    grounded: GroundedSemanticCandidate,
    ontology: Any,
    pipeline_version: str,
    limit: int,
) -> list[ZeroGraphCandidate]:
    paths = _path_views(grounded, ontology)
    if len(paths) < 2:
        return []
    base_score = 10.0 * float(grounded.score)
    generated: list[ZeroGraphCandidate] = []
    terminal_merges = [(paths[0].terminal, path.terminal) for path in paths[1:]]
    for trimmed in (False, True):
        candidate = _compile(
            question=question,
            grounded=grounded,
            paths=paths,
            merges=terminal_merges,
            answer=paths[0].terminal,
            strategy="terminal_intersection",
            base_score=base_score,
            join_score=8.0 * (len(paths) - 1),
            trim_at_join=trimmed,
            ontology=ontology,
            pipeline_version=pipeline_version,
        )
        if candidate is not None:
            generated.append(candidate)

    for base_index, base in enumerate(paths):
        other_paths = [path for index, path in enumerate(paths) if index != base_index]
        attachment_options: list[list[tuple[float, str, str]]] = []
        for other in other_paths:
            options: list[tuple[float, str, str]] = []
            for other_node in other.variables:
                for base_node in base.variables:
                    compatibility = _type_compatibility(
                        ontology,
                        other.node_types.get(other_node, ()),
                        base.node_types.get(base_node, ()),
                    )
                    if compatibility < 0.0:
                        continue
                    terminal_bonus = 2.0 * (other_node == other.terminal)
                    terminal_bonus += 2.0 * (base_node == base.terminal)
                    options.append((compatibility + terminal_bonus, other_node, base_node))
            options.sort(key=lambda value: (-value[0], value[1], value[2]))
            attachment_options.append(options[:6])
        if any(not values for values in attachment_options):
            continue
        for selected in islice(product(*attachment_options), 216):
            merges = [(other_node, base_node) for _, other_node, base_node in selected]
            join_score = sum(value for value, _, _ in selected)
            for answer in base.variables:
                answer_types = set(base.node_types.get(answer, ()))
                if _entity_question(question) and answer_types and answer_types <= _SCALAR_TYPES:
                    continue
                answer_bonus = 5.0 if answer == base.terminal else 0.0
                answer_bonus += _answer_semantic_bonus(
                    question, paths, base, answer, ontology
                )
                for trimmed in (False, True):
                    candidate = _compile(
                        question=question,
                        grounded=grounded,
                        paths=paths,
                        merges=merges,
                        answer=answer,
                        strategy="typed_spine_attachment",
                        base_score=base_score + answer_bonus,
                        join_score=join_score,
                        trim_at_join=trimmed,
                        ontology=ontology,
                        pipeline_version=pipeline_version,
                    )
                    if candidate is not None:
                        generated.append(candidate)
    best: dict[str, ZeroGraphCandidate] = {}
    for candidate in generated:
        previous = best.get(candidate.key)
        if previous is None or candidate.score > previous.score:
            best[candidate.key] = candidate
    result: list[ZeroGraphCandidate] = []
    for candidate in sorted(best.values(), key=lambda item: (-item.score, item.key))[:limit]:
        metadata = {
            **candidate.metadata,
            "raw_score": float(grounded.score),
            "relation_multiset": sorted(_relation_counter(grounded).elements()),
        }
        result.append(replace(candidate, metadata=metadata))
    return result


def _hybrid_grounded(
    grounded_candidates: list[GroundedSemanticCandidate],
) -> list[GroundedSemanticCandidate]:
    groups: dict[tuple[Any, ...], list[GroundedSemanticCandidate]] = defaultdict(list)
    for grounded in grounded_candidates:
        paths = [
            path
            for path in grounded.compose_input.get("semantic_paths", [])
            if isinstance(path, dict)
        ]
        key = (
            tuple((str(path.get("id", "")), len(path.get("steps", []))) for path in paths),
            tuple(
                sorted(
                    (anchor_id, candidate.entity_id)
                    for anchor_id, candidate in grounded.anchor_bindings.items()
                )
            ),
        )
        if paths:
            groups[key].append(grounded)
    hybrids: list[GroundedSemanticCandidate] = []
    for values in groups.values():
        template = values[0]
        template_paths = template.compose_input.get("semantic_paths", [])
        choices: list[list[tuple[dict[str, Any], dict[str, str], float]]] = []
        for template_path in template_paths:
            path_id = str(template_path.get("id", ""))
            unique: dict[
                tuple[str, ...], tuple[dict[str, Any], dict[str, str], float]
            ] = {}
            for grounded in values:
                candidate_path = next(
                    (
                        path
                        for path in grounded.compose_input.get("semantic_paths", [])
                        if isinstance(path, dict) and str(path.get("id", "")) == path_id
                    ),
                    None,
                )
                if candidate_path is None:
                    continue
                step_ids = [
                    str(step.get("id", ""))
                    for step in candidate_path.get("steps", [])
                    if isinstance(step, dict)
                ]
                bindings = {
                    step_id: str(grounded.relation_bindings.get(step_id, ""))
                    for step_id in step_ids
                }
                signature = tuple(bindings[step_id] for step_id in step_ids)
                if not signature or not all(signature):
                    continue
                item = (candidate_path, bindings, float(grounded.score))
                if signature not in unique or item[2] > unique[signature][2]:
                    unique[signature] = item
            choices.append(sorted(unique.values(), key=lambda item: -item[2])[:4])
        if any(not values for values in choices):
            continue
        for combination in islice(product(*choices), 128):
            compose_input = dict(template.compose_input)
            compose_input["semantic_paths"] = [item[0] for item in combination]
            bindings: dict[str, str] = {}
            for _, path_bindings, _ in combination:
                bindings.update(path_bindings)
            hybrids.append(
                GroundedSemanticCandidate(
                    compose_input=compose_input,
                    anchor_bindings=dict(template.anchor_bindings),
                    relation_bindings=bindings,
                    score=sum(item[2] for item in combination) / len(combination),
                    provenance={
                        **dict(template.provenance),
                        "zero_graph_challenger_hybrid": {
                            "model_used": False,
                            "uses_gold": False,
                        },
                    },
                )
            )
    return hybrids


def _saved_operator_candidates(
    *,
    question: str,
    final_graphs: list[QueryGraphCandidate],
    pipeline_version: str,
) -> list[ZeroGraphCandidate]:
    expected = _expected_operator_type(question)
    result: list[ZeroGraphCandidate] = []
    for source in final_graphs:
        if not source.triples or not source.answer_var or not source.operators:
            continue
        operators, repaired, value_supported = _repaired_operators(
            question,
            [item for item in source.operators if isinstance(item, dict)],
        )
        substantive = [
            operator
            for operator in operators
            if str(operator.get("type", "")).upper() != "NO_EQUAL"
        ]
        if not substantive or not (repaired or value_supported or expected):
            continue
        score = (10.0 * float(source.score)) + (
            60.0 if repaired else 40.0 if value_supported else 4.0
        )
        graph = QueryGraphCandidate(
            graph_id="ZGC",
            triples=deepcopy(source.triples),
            answer_var=str(source.answer_var),
            operators=operators,
            score=score,
            compose_output=deepcopy(source.compose_output),
            provenance={
                **deepcopy(source.provenance),
                "zero_graph_challenger": {
                    "version": _VERSION,
                    "strategy": "saved_operator_relowering",
                    "source_graph_id": source.graph_id,
                    "uses_gold": False,
                    "model_used": False,
                },
            },
        )
        try:
            sparql = lower_sparql(graph)
        except (LoweringError, ValueError, TypeError):
            continue
        if any(
            str(operator.get("type", "")).upper() == "EQUAL"
            and str(operator.get("value_type", "")).casefold().startswith("lang:")
            for operator in operators
        ):
            sparql = re.sub(
                r'FILTER\((\?[A-Za-z0-9_]+) = ("(?:[^"\\]|\\.)*")\)',
                r"FILTER(STR(\1) = \2)",
                sparql,
            )
            for operator in operators:
                value = str(operator.get("value", "")).strip()
                if (
                    str(operator.get("type", "")).upper() == "EQUAL"
                    and re.fullmatch(r"\d+\s*-\s*\d+", value)
                ):
                    compact_value = re.sub(r"\s+", "", value)
                    sparql = re.sub(
                        rf'FILTER\(STR\((\?[A-Za-z0-9_]+)\) = "{re.escape(value)}"\)',
                        rf'FILTER(REPLACE(STR(\1), " ", "") = "{compact_value}")',
                        sparql,
                    )
        key = json.dumps(
            [graph.triples, graph.answer_var, graph.operators],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        metadata = {
            "source_graph_id": source.graph_id,
            "expected_operator_type": expected,
            "operator_type_repaired": repaired,
            "explicit_value_supported": value_supported,
            "operators": deepcopy(operators),
        }
        result.append(
            ZeroGraphCandidate(
                key,
                "saved_operator_relowering",
                score,
                graph,
                sparql,
                metadata,
            )
        )
    return result


def build_zero_graph_challengers(
    *,
    question: str,
    grounded_candidates: list[GroundedSemanticCandidate],
    final_graphs: list[QueryGraphCandidate],
    ontology: Any,
    pipeline_version: str,
    limit: int = 3,
    per_grounded_limit: int = 24,
) -> tuple[list[ZeroGraphCandidate], dict[str, Any]]:
    """Return the top three Gold-blind candidates after a caller-applied gate."""

    budget = max(0, min(3, int(limit)))
    if budget == 0 or ontology is None or not grounded_candidates:
        return [], {
            "status": "not_compiled",
            "reason": "missing_budget_ontology_or_grounding",
            "execution_budget": budget,
            "uses_gold": False,
            "model_used": False,
        }
    grounded_with_hybrids = [
        *grounded_candidates,
        *_hybrid_grounded(grounded_candidates),
    ]
    best: dict[str, ZeroGraphCandidate] = {}
    for candidate in _saved_operator_candidates(
        question=question,
        final_graphs=final_graphs,
        pipeline_version=pipeline_version,
    ):
        best[candidate.key] = candidate
    for grounded in grounded_with_hybrids:
        candidates = [
            *_single_path_candidates(
                question=question,
                grounded=grounded,
                ontology=ontology,
                pipeline_version=pipeline_version,
            ),
            *_multipath_candidates(
                question=question,
                grounded=grounded,
                ontology=ontology,
                pipeline_version=pipeline_version,
                limit=max(1, int(per_grounded_limit)),
            ),
        ]
        hybrid = "zero_graph_challenger_hybrid" in grounded.provenance
        for candidate in candidates:
            if hybrid:
                strategy = "hybrid_" + candidate.strategy
                graph = deepcopy(candidate.graph)
                graph.compose_output["strategy"] = strategy
                graph.provenance["zero_graph_challenger"]["strategy"] = strategy
                candidate = replace(candidate, strategy=strategy, graph=graph)
            relation_counts = Counter(candidate.metadata.get("relation_multiset", []))
            if any(count > 1 for count in relation_counts.values()):
                graph = deepcopy(candidate.graph)
                graph.score = candidate.score + 6.0
                candidate = replace(
                    candidate,
                    score=candidate.score + 6.0,
                    graph=graph,
                )
            previous = best.get(candidate.key)
            if previous is None or candidate.score > previous.score:
                best[candidate.key] = candidate
    ranked = sorted(best.values(), key=lambda item: (-item.score, item.key))
    selected: list[ZeroGraphCandidate] = []
    for rank, candidate in enumerate(ranked[:budget], start=1):
        graph = deepcopy(candidate.graph)
        graph.graph_id = f"ZGC{rank}"
        graph.sparql = candidate.sparql
        graph.provenance["zero_graph_challenger"].update(
            {"rank": rank, "execution_budget": budget}
        )
        selected.append(replace(candidate, graph=graph))
    return selected, {
        "status": "compiled" if selected else "not_compiled",
        "candidate_count": len(best),
        "returned_count": len(selected),
        "execution_budget": budget,
        "grounded_candidate_count": len(grounded_candidates),
        "hybrid_candidate_count": len(grounded_with_hybrids) - len(grounded_candidates),
        "uses_gold": False,
        "model_used": False,
    }
