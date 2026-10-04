"""Bounded training/ontology retrieval for a missing answer-edge relation.

This lane is deliberately independent from graph selection and every chat
model.  It edits one relation incident to the incumbent answer variable,
executes at most three candidates, and applies runtime-only confidence gates.
It can be invoked after a final ``EMPTY_RESULT`` or as a conservative
challenger to an existing successful answer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import heapq
import json
import math
import re
import time
from typing import Any, Iterable, Sequence

from .contracts import QueryGraphCandidate
from .lowering import LoweringError, lower_sparql
from .ontology import FreebaseOntology, relation_id_from_label
from .template_path_retrieval import mask_entities


__all__ = ["retrieve_missing_relation"]


_TOKEN_RE = re.compile(r"<entity>|[a-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text))]


def _relation_text(relation_id: str) -> str:
    return " ".join(
        part.replace("_", " ") for part in str(relation_id).split(".") if part
    )


def _leaf_tokens(relation_id: str) -> set[str]:
    return set(_tokens(str(relation_id).rsplit(".", 1)[-1].replace("_", " ")))


def _answer_values(rows: list[dict[str, str]], answer_var: str) -> list[str]:
    key = str(answer_var).lstrip("?")
    output: list[str] = []
    for row in rows:
        value = row.get(key, row.get("answer", ""))
        if not value:
            value = next(
                (
                    candidate
                    for name, candidate in row.items()
                    if str(name).casefold() == key.casefold()
                ),
                "",
            )
        value = str(value).strip()
        prefix = "http://rdf.freebase.com/ns/"
        if value.startswith(prefix):
            value = value[len(prefix) :]
        if value and value not in output:
            output.append(value)
    return output


def _trace_outputs(traces: Sequence[dict[str, Any]], stage: str) -> Iterable[Any]:
    for trace in traces:
        if isinstance(trace, dict) and trace.get("stage") == stage:
            yield trace.get("output")


def _decompositions(traces: Sequence[dict[str, Any]]) -> list[str]:
    reviewed: list[str] = []
    for output in _trace_outputs(traces, "decomposition_review_and_rewrite"):
        if not isinstance(output, dict):
            continue
        values = [
            str(item)
            for candidate in output.get("final_candidates", [])
            if isinstance(candidate, dict)
            for item in candidate.get("decomposition", [])
            if str(item).strip()
        ]
        if values:
            reviewed = values
    if reviewed:
        return list(dict.fromkeys(reviewed))
    values: list[str] = []
    for output in _trace_outputs(traces, "decompose_predictions"):
        for candidate in output if isinstance(output, list) else []:
            if isinstance(candidate, dict):
                values.extend(
                    str(item)
                    for item in candidate.get("decomposition", [])
                    if str(item).strip()
                )
    return list(dict.fromkeys(values))


def _anchor_surfaces(traces: Sequence[dict[str, Any]]) -> list[str]:
    for output in _trace_outputs(traces, "semantic_path_generation"):
        for candidate in output if isinstance(output, list) else []:
            graph = candidate.get("output", {}) if isinstance(candidate, dict) else {}
            values = [
                str(item.get("surface", ""))
                for item in graph.get("anchors", [])
                if isinstance(item, dict) and str(item.get("surface", "")).strip()
            ]
            if values:
                return list(dict.fromkeys(values))
    return []


def _grounded_union(traces: Sequence[dict[str, Any]]) -> Counter[str]:
    output: Counter[str] = Counter()
    for values in _trace_outputs(traces, "grounded_semantic_candidates"):
        for candidate in values if isinstance(values, list) else []:
            if not isinstance(candidate, dict):
                continue
            current = Counter(
                str(value)
                for value in (candidate.get("relation_bindings") or {}).values()
                if str(value)
            )
            output |= current
    return output


def _final_incumbent(traces: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for output in _trace_outputs(traces, "final_graph_beam"):
        if isinstance(output, list):
            candidates = [value for value in output if isinstance(value, dict)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda value: (
            float(value.get("score", 0.0)),
            -len(value.get("triples", [])),
            str(value.get("graph_id", "")),
        ),
    )


def _hard_post_selection_applied(traces: Sequence[dict[str, Any]]) -> bool:
    """Preserve every zero-cost hard-gate decision made by normal selection."""

    latest: dict[str, Any] = {}
    for output in _trace_outputs(traces, "glm_final_graph_ranking"):
        if isinstance(output, dict):
            latest = output
    reason_codes = {str(value) for value in latest.get("reason_codes", [])}
    return (
        any(value.startswith("hard_") for value in reason_codes)
        or "morphological_underanswer_expansion_gate" in reason_codes
        or any(str(key).startswith("hard_") for key in latest)
    )


@dataclass(slots=True)
class _Document:
    source_index: int
    length: int
    relations: Counter[str]
    answer_incidence: Counter[str]
    triple_count: int
    # Retain the aligned Compose topology in the shared cache.  The narrow
    # extrema-relation lane can then reuse this index instead of building a
    # second training index or invoking any model.
    triples: tuple[tuple[str, str, str], ...]
    answer_var: str
    tokens: list[str]


class _AlignedRelationIndex:
    def __init__(
        self,
        semantic_examples: Sequence[dict[str, Any]],
        compose_examples: Sequence[dict[str, Any]],
    ) -> None:
        self.documents: list[_Document] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        for source_index, (semantic, compose) in enumerate(
            zip(semantic_examples, compose_examples)
        ):
            triples: list[tuple[str, str, str]] = []
            for triple in compose.get("triples", []) if isinstance(compose, dict) else []:
                if not isinstance(triple, dict):
                    continue
                relation_id = relation_id_from_label(triple.get("relation_label", []))
                if relation_id:
                    triples.append(
                        (
                            str(triple.get("subject", "")),
                            relation_id,
                            str(triple.get("object", "")),
                        )
                    )
            answer_var = str(compose.get("answer_var", "")) if isinstance(compose, dict) else ""
            if not triples or not answer_var:
                continue
            graph = semantic.get("semantic_graph", {})
            surfaces = [
                str(item.get("surface", ""))
                for item in graph.get("anchors", [])
                if isinstance(item, dict) and str(item.get("surface", "")).strip()
            ]
            text = " ".join(
                [
                    mask_entities(str(semantic.get("question", "")), surfaces),
                    *(
                        mask_entities(str(value), surfaces)
                        for value in semantic.get("decomposition", [])
                    ),
                ]
            )
            frequencies = Counter(_tokens(text))
            relations = Counter(relation for _, relation, _ in triples)
            incidence = Counter(
                f"{relation}|{'out' if subject == answer_var else 'in' if object_ == answer_var else 'internal'}"
                for subject, relation, object_ in triples
            )
            index = len(self.documents)
            document = _Document(
                source_index=source_index,
                length=sum(frequencies.values()),
                relations=relations,
                answer_incidence=incidence,
                triple_count=len(triples),
                triples=tuple(triples),
                answer_var=answer_var,
                tokens=list(frequencies.elements()),
            )
            self.documents.append(document)
            document_frequency.update(frequencies)
            for token, frequency in frequencies.items():
                self.postings[token].append((index, frequency))
        count = max(1, len(self.documents))
        self.average_length = sum(value.length for value in self.documents) / count
        self.idf = {
            token: math.log(1.0 + ((count - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in document_frequency.items()
        }
        self.max_posting_length = max(64, int(len(self.documents) * 0.18))

    def retrieve(
        self,
        question: str,
        decompositions: Sequence[str],
        surfaces: Sequence[str],
        *,
        top_k: int,
    ) -> list[tuple[float, _Document]]:
        text = " ".join(
            [
                mask_entities(question, surfaces),
                *(mask_entities(value, surfaces) for value in decompositions),
            ]
        )
        query = Counter(_tokens(text))
        scores: dict[int, float] = defaultdict(float)
        informative = sorted(
            (
                (self.idf.get(token, 0.0), token, frequency)
                for token, frequency in query.items()
                if len(self.postings.get(token, ())) <= self.max_posting_length
            ),
            reverse=True,
        )[:24]
        for _, token, query_frequency in informative:
            inverse_frequency = self.idf.get(token, 0.0)
            for index, frequency in self.postings.get(token, ()):
                document = self.documents[index]
                normalization = 0.25 + 0.75 * document.length / max(
                    self.average_length, 1e-9
                )
                scores[index] += (
                    inverse_frequency
                    * ((frequency * 2.2) / (frequency + 1.2 * normalization))
                    * min(query_frequency, 2)
                )
        return [
            (score, self.documents[index])
            for score, index in heapq.nlargest(
                max(1, top_k), ((score, index) for index, score in scores.items())
            )
        ]


@dataclass(slots=True)
class _Evidence:
    relation_id: str
    best_template_rank: int
    normalized_bm25: float
    support_top8: int
    support_top32: int
    weighted_support: float
    best_relation_overlap: float
    preferred_orientation: str
    orientation_confidence: float
    preferred_size_delta: int
    grounded_union: bool
    reverse_of_terminal: bool
    incumbent_weighted_support: float
    template_support_margin: float
    template_support_ratio: float
    lexical_overlap: float
    bge_score: float = 0.0
    incumbent_bge_score: float = 0.0
    candidate_score: float = 0.0
    relation_support: dict[str, float] = field(default_factory=dict)

    def dump(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _relation_evidence(
    *,
    question: str,
    decompositions: Sequence[str],
    surfaces: Sequence[str],
    graph: dict[str, Any],
    traces: Sequence[dict[str, Any]],
    index: _AlignedRelationIndex,
    ontology: FreebaseOntology,
    top_k: int,
) -> tuple[list[_Evidence], list[str]]:
    triples = [
        tuple(map(str, value))
        for value in graph.get("triples", [])
        if isinstance(value, (list, tuple)) and len(value) == 3
    ]
    answer_var = str(graph.get("answer_var", ""))
    selected = Counter(relation for _, relation, _ in triples)
    terminal = {
        relation
        for subject, relation, object_ in triples
        if answer_var in {subject, object_}
    }
    grounded = _grounded_union(traces)
    neighbours = index.retrieve(
        question, decompositions, surfaces, top_k=max(3, int(top_k))
    )
    top_score = max((score for score, _ in neighbours), default=1.0)
    aggregate: dict[str, dict[str, Any]] = {}
    all_support: Counter[str] = Counter()
    for rank, (bm25, document) in enumerate(neighbours, 1):
        overlap = sum((document.relations & selected).values()) / max(
            1, sum(selected.values())
        )
        normalized = bm25 / max(top_score, 1e-9)
        document_weight = normalized * (0.35 + 0.65 * overlap)
        for relation_id, count in document.relations.items():
            all_support[relation_id] += document_weight * count
        for relation_id in document.relations - selected:
            value = aggregate.setdefault(
                relation_id,
                {
                    "rank": rank,
                    "bm25": bm25,
                    "support8": 0,
                    "support32": 0,
                    "weight": 0.0,
                    "overlap": 0.0,
                    "orientations": Counter(),
                    "sizes": Counter(),
                },
            )
            value["rank"] = min(value["rank"], rank)
            value["bm25"] = max(value["bm25"], bm25)
            value["support8"] += int(rank <= 8)
            value["support32"] += int(rank <= 32)
            value["weight"] += document_weight
            value["overlap"] = max(value["overlap"], overlap)
            value["sizes"][document.triple_count - len(triples)] += normalized
            for orientation in ("in", "out"):
                count = document.answer_incidence.get(
                    f"{relation_id}|{orientation}", 0
                )
                if count:
                    value["orientations"][orientation] += normalized * count
    incumbent_support = max(
        (float(all_support.get(value, 0.0)) for value in terminal), default=0.0
    )
    question_tokens = set(_tokens(question))
    output: list[_Evidence] = []
    for relation_id, value in aggregate.items():
        orientations: Counter[str] = value["orientations"]
        orientation = (
            max(orientations, key=lambda key: (orientations[key], key))
            if orientations
            else ""
        )
        orientation_total = sum(orientations.values())
        sizes: Counter[int] = value["sizes"]
        size_delta = max(sizes, key=lambda key: (sizes[key], -abs(key))) if sizes else 0
        relation_tokens = _leaf_tokens(relation_id)
        lexical = (
            len(relation_tokens & question_tokens) / len(relation_tokens)
            if relation_tokens
            else 0.0
        )
        output.append(
            _Evidence(
                relation_id=relation_id,
                best_template_rank=int(value["rank"]),
                normalized_bm25=float(value["bm25"] / max(top_score, 1e-9)),
                support_top8=int(value["support8"]),
                support_top32=int(value["support32"]),
                weighted_support=float(value["weight"]),
                best_relation_overlap=float(value["overlap"]),
                preferred_orientation=orientation,
                orientation_confidence=(
                    float(orientations.get(orientation, 0.0) / orientation_total)
                    if orientation_total
                    else 0.0
                ),
                preferred_size_delta=int(size_delta),
                grounded_union=relation_id in grounded,
                reverse_of_terminal=any(
                    relation_id in ontology.reverse_for_relation(old)
                    for old in terminal
                ),
                incumbent_weighted_support=incumbent_support,
                template_support_margin=float(value["weight"] - incumbent_support),
                template_support_ratio=float(
                    value["weight"] / max(incumbent_support, 1e-9)
                ),
                lexical_overlap=lexical,
                relation_support={
                    str(key): float(support)
                    for key, support in all_support.items()
                },
            )
        )
    return output, sorted(terminal)


def _type_compatibility(ontology: FreebaseOntology, left: str, right: str) -> float:
    if not left or not right:
        return 0.35
    if left == right:
        return 1.0
    left_supers, right_supers = set(ontology.supertypes(left)), set(ontology.supertypes(right))
    if left in right_supers or right in left_supers:
        return 0.8
    if left_supers & right_supers:
        return 0.55
    return 0.0


def _neighbour_type(
    ontology: FreebaseOntology, relation_id: str, orientation: str
) -> str:
    return (
        ontology.range_for_relation(relation_id)
        if orientation == "out"
        else ontology.domain_for_relation(relation_id)
    )


def _score_evidence(
    evidence: list[_Evidence],
    *,
    question: str,
    terminal: Sequence[str],
    ranker: Any,
) -> bool:
    relation_ids = list(dict.fromkeys([*terminal, *(item.relation_id for item in evidence[:24])]))
    try:
        scores = ranker.score(question, [_relation_text(value) for value in relation_ids])
    except Exception:
        return False
    by_relation = dict(zip(relation_ids, map(float, scores)))
    incumbent_bge = max((by_relation.get(value, 0.0) for value in terminal), default=0.0)
    max_support = max((item.weighted_support for item in evidence), default=1.0)
    for item in evidence:
        item.bge_score = by_relation.get(item.relation_id, 0.0)
        item.incumbent_bge_score = incumbent_bge
        support = item.weighted_support / max(max_support, 1e-9)
        item.candidate_score = (
            0.24 * item.normalized_bm25
            + 0.22 * support
            + 0.19 * item.best_relation_overlap
            + 0.22 * max(0.0, min(1.0, (item.bge_score + 1.0) / 2.0))
            + 0.07 * item.lexical_overlap
            + 0.04 * float(item.grounded_union)
            + 0.02 * item.orientation_confidence
        )
    evidence.sort(
        key=lambda item: (
            -item.candidate_score,
            item.best_template_rank,
            item.relation_id,
        )
    )
    return True


def _preliminary_gate(evidence: Sequence[_Evidence]) -> bool:
    if not evidence:
        return False
    best = evidence[0]
    return bool(
        best.best_relation_overlap >= 0.34
        and best.support_top32 >= 2
        and (
            best.bge_score - best.incumbent_bge_score >= -0.04
            or best.lexical_overlap > 0.0
            or best.grounded_union
        )
    )


def _local_evidence_possible(
    evidence: _Evidence,
    *,
    graph: dict[str, Any],
    incumbent_count: int,
    final_empty: bool,
) -> bool:
    """BGE-free necessary conditions for any production acceptance branch."""

    if final_empty:
        return bool(
            evidence.relation_id
            in {
                str(triple[1])
                for triple in graph.get("triples", [])
                if isinstance(triple, (list, tuple)) and len(triple) == 3
            }
            or evidence.reverse_of_terminal
            or (
                evidence.best_template_rank == 1
                and evidence.best_relation_overlap >= 2.0 / 3.0
                and evidence.orientation_confidence >= 0.99
            )
        )
    a_possible = bool(
        evidence.incumbent_weighted_support <= 10.0
        and evidence.support_top32 >= 20
        and evidence.orientation_confidence >= 0.99
    )
    b_possible = bool(
        incumbent_count >= 3
        and evidence.support_top32 >= 15
        and evidence.orientation_confidence >= 0.95
        and evidence.best_relation_overlap < 1.0
    )
    answer_var = str(graph.get("answer_var", ""))
    terminal_orientations = {
        "out" if str(triple[0]) == answer_var else "in"
        for triple in graph.get("triples", [])
        if isinstance(triple, (list, tuple))
        and len(triple) == 3
        and answer_var in {str(triple[0]), str(triple[2])}
    }
    d_possible = bool(
        5 <= incumbent_count <= 64
        and evidence.support_top8 >= 5
        and evidence.best_template_rank == 1
        and 0.0 < evidence.best_relation_overlap < 1.0
        and (
            not evidence.preferred_orientation
            or evidence.preferred_orientation in terminal_orientations
        )
    )
    e_possible = bool(
        incumbent_count > 0
        and evidence.best_template_rank == 1
        and 0.5 <= evidence.best_relation_overlap <= 2.0 / 3.0
        and (
            not evidence.preferred_orientation
            or evidence.preferred_orientation in terminal_orientations
        )
    )
    return a_possible or b_possible or d_possible or e_possible


def _pre_execution_candidate(
    candidate: _Candidate,
    *,
    incumbent_count: int,
    final_empty: bool,
) -> bool:
    """Apply every gate term known before spending an endpoint query."""

    evidence = candidate.evidence
    if final_empty:
        return bool(
            evidence.relation_id == candidate.replaced_relation
            or evidence.reverse_of_terminal
            or (
                evidence.best_template_rank == 1
                and evidence.best_relation_overlap >= 2.0 / 3.0
                and evidence.orientation_confidence >= 0.99
            )
        )
    a_possible = bool(
        evidence.incumbent_weighted_support <= 10.0
        and evidence.bge_score - evidence.incumbent_bge_score >= 0.0
        and evidence.support_top32 >= 20
        and evidence.orientation_confidence >= 0.99
    )
    b_possible = bool(
        candidate.graph.score >= 0.90
        and incumbent_count >= 3
        and evidence.support_top32 >= 15
        and evidence.orientation_confidence >= 0.95
        and evidence.best_relation_overlap < 1.0
    )
    d_possible = bool(
        5 <= incumbent_count <= 64
        and evidence.support_top8 >= 5
        and evidence.best_template_rank == 1
        and 0.0 < evidence.best_relation_overlap < 1.0
        and candidate.old_orientation == candidate.new_orientation
    )
    e_possible = bool(
        incumbent_count > 0
        and evidence.best_template_rank == 1
        and candidate.old_orientation == candidate.new_orientation
        and evidence.bge_score - evidence.incumbent_bge_score >= 0.0
        and 0.5 <= evidence.best_relation_overlap <= 2.0 / 3.0
    )
    return a_possible or b_possible or d_possible or e_possible


@dataclass(slots=True)
class _Candidate:
    graph: QueryGraphCandidate
    evidence: _Evidence
    replaced_relation: str
    old_orientation: str
    new_orientation: str
    answers: list[str]
    query_elapsed_seconds: float = 0.0
    error: str = ""
    edge_index: int = -1
    edge_role: str = "answer"
    reversed_edge: bool = False
    schema_compatibility: float = 0.0
    answer_distance: int = -1


def _node_distances_to_answer(
    triples: Sequence[Sequence[str]], answer_var: str
) -> dict[str, int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        if len(triple) != 3:
            continue
        subject, _, object_ = map(str, triple)
        adjacency[subject].add(object_)
        adjacency[object_].add(subject)
    distances = {str(answer_var): 0}
    frontier = [str(answer_var)]
    while frontier:
        node = frontier.pop(0)
        for neighbour in adjacency.get(node, set()):
            if neighbour in distances:
                continue
            distances[neighbour] = distances[node] + 1
            frontier.append(neighbour)
    return distances


def _endpoint_type_hints(
    ontology: FreebaseOntology,
    triples: Sequence[Sequence[str]],
    *,
    edge_index: int,
    node: str,
) -> list[str]:
    hints: list[str] = []
    for index, triple in enumerate(triples):
        if index == edge_index or len(triple) != 3:
            continue
        subject, relation, object_ = map(str, triple)
        value = ""
        if subject == node:
            value = str(ontology.domain_for_relation(relation))
        elif object_ == node:
            value = str(ontology.range_for_relation(relation))
        if value and value not in hints:
            hints.append(value)
    return hints


def _best_endpoint_compatibility(
    ontology: FreebaseOntology,
    expected: str,
    hints: Sequence[str],
) -> float:
    if not hints:
        return 0.35 if not expected else 0.50
    return max(_type_compatibility(ontology, expected, value) for value in hints)


def _internal_schema_compatibility(
    ontology: FreebaseOntology,
    triples: Sequence[Sequence[str]],
    *,
    edge_index: int,
    new_relation: str,
    reversed_edge: bool,
) -> float:
    subject, old_relation, object_ = map(str, triples[edge_index])
    new_domain = str(ontology.domain_for_relation(new_relation))
    new_range = str(ontology.range_for_relation(new_relation))
    if reversed_edge:
        new_domain, new_range = new_range, new_domain
    old_domain = str(ontology.domain_for_relation(old_relation))
    old_range = str(ontology.range_for_relation(old_relation))
    subject_local = _best_endpoint_compatibility(
        ontology,
        new_domain,
        _endpoint_type_hints(
            ontology, triples, edge_index=edge_index, node=subject
        ),
    )
    object_local = _best_endpoint_compatibility(
        ontology,
        new_range,
        _endpoint_type_hints(
            ontology, triples, edge_index=edge_index, node=object_
        ),
    )
    subject_score = max(
        subject_local,
        0.75 * _type_compatibility(ontology, new_domain, old_domain),
    )
    object_score = max(
        object_local,
        0.75 * _type_compatibility(ontology, new_range, old_range),
    )
    return 0.5 * (subject_score + object_score)


def _internal_candidate_possible(candidate: _Candidate) -> bool:
    """Generic Gold-blind pre-query gate for the final-empty internal lane."""

    evidence = candidate.evidence
    return bool(
        candidate.edge_role == "internal"
        and evidence.best_relation_overlap >= 0.5
        and evidence.support_top32 >= 2
        and (
            evidence.grounded_union
            or evidence.bge_score - evidence.incumbent_bge_score >= -0.04
            or evidence.lexical_overlap > 0.0
        )
    )


def _compile_internal(
    graph: dict[str, Any],
    evidence: Sequence[_Evidence],
    ontology: FreebaseOntology,
    *,
    limit: int,
) -> list[_Candidate]:
    """Compile one-edge internal repairs ranked by schema and answer proximity."""

    triples = [
        list(map(str, value))
        for value in graph.get("triples", [])
        if isinstance(value, (list, tuple)) and len(value) == 3
    ]
    answer_var = str(graph.get("answer_var", ""))
    internal_edges = [
        (index, value)
        for index, value in enumerate(triples)
        if answer_var not in {value[0], value[2]}
    ]
    if not internal_edges:
        return []
    distances = _node_distances_to_answer(triples, answer_var)
    output: list[_Candidate] = []
    seen: set[str] = set()
    for relation in evidence[:12]:
        support = relation.relation_support
        max_support = max(support.values(), default=1.0)
        for edge_index, edge in internal_edges:
            subject, old_relation, object_ = edge
            if relation.relation_id == old_relation:
                continue
            distance = min(
                distances.get(subject, len(triples) + 1),
                distances.get(object_, len(triples) + 1),
            )
            proximity = 1.0 / (1.0 + float(distance))
            support_delta = (
                float(support.get(relation.relation_id, 0.0))
                - float(support.get(old_relation, 0.0))
            ) / max(max_support, 1e-9)
            for reversed_edge in (False, True):
                compatibility = _internal_schema_compatibility(
                    ontology,
                    triples,
                    edge_index=edge_index,
                    new_relation=relation.relation_id,
                    reversed_edge=reversed_edge,
                )
                if compatibility <= 0.0:
                    continue
                revised = deepcopy(triples)
                revised[edge_index] = (
                    [object_, relation.relation_id, subject]
                    if reversed_edge
                    else [subject, relation.relation_id, object_]
                )
                signature = json.dumps(
                    {"triples": revised, "answer_var": answer_var},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                score = (
                    0.48 * relation.candidate_score
                    + 0.25 * compatibility
                    + 0.05 * (1.0 / (1.0 + math.exp(-4.0 * support_delta)))
                    + 0.04 * proximity
                    + 0.02 * float(not reversed_edge)
                )
                candidate_graph = QueryGraphCandidate(
                    graph_id=f"MRI{len(output)}",
                    triples=revised,
                    answer_var=answer_var,
                    operators=deepcopy(list(graph.get("operators", []))),
                    score=score,
                    compose_output={},
                    provenance=deepcopy(dict(graph.get("provenance", {}))),
                )
                candidate_graph.provenance["missing_relation_retrieval"] = {
                    "relation_id": relation.relation_id,
                    "replaced_relation": old_relation,
                    "edge_role": "internal",
                    "edge_index": edge_index,
                    "answer_distance": distance,
                    "uses_gold": False,
                    "model_calls": 0,
                }
                output.append(
                    _Candidate(
                        graph=candidate_graph,
                        evidence=relation,
                        replaced_relation=old_relation,
                        old_orientation="internal",
                        new_orientation="internal",
                        answers=[],
                        edge_index=edge_index,
                        edge_role="internal",
                        reversed_edge=reversed_edge,
                        schema_compatibility=compatibility,
                        answer_distance=distance,
                    )
                )
    output.sort(
        key=lambda item: (
            -item.graph.score,
            item.answer_distance,
            item.edge_index,
            item.graph.graph_id,
        )
    )
    return output[: max(1, min(24, int(limit)))]


def _compile(
    graph: dict[str, Any],
    evidence: Sequence[_Evidence],
    ontology: FreebaseOntology,
    *,
    limit: int,
) -> list[_Candidate]:
    triples = [
        list(map(str, value))
        for value in graph.get("triples", [])
        if isinstance(value, (list, tuple)) and len(value) == 3
    ]
    answer_var = str(graph.get("answer_var", ""))
    terminal_edges = [
        (index, value)
        for index, value in enumerate(triples)
        if answer_var in {value[0], value[2]}
    ]
    output: list[_Candidate] = []
    seen: set[str] = set()
    for relation in evidence[:12]:
        for edge_index, edge in terminal_edges:
            subject, old_relation, object_ = edge
            old_orientation = "out" if subject == answer_var else "in"
            new_orientation = relation.preferred_orientation or old_orientation
            new_subject, new_object = subject, object_
            if new_orientation != old_orientation:
                new_subject, new_object = object_, subject
            compatibility = _type_compatibility(
                ontology,
                _neighbour_type(ontology, old_relation, old_orientation),
                _neighbour_type(ontology, relation.relation_id, new_orientation),
            )
            if compatibility == 0.0:
                continue
            revised = deepcopy(triples)
            revised[edge_index] = [new_subject, relation.relation_id, new_object]
            signature = json.dumps(
                {"triples": revised, "answer_var": answer_var},
                sort_keys=True,
                separators=(",", ":"),
            )
            if signature in seen:
                continue
            seen.add(signature)
            score = (
                relation.candidate_score
                + 0.08 * compatibility
                + 0.04 * float(new_orientation == relation.preferred_orientation)
                + 0.03 * float(relation.preferred_size_delta == 0)
            )
            candidate_graph = QueryGraphCandidate(
                graph_id=f"MR{len(output)}",
                triples=revised,
                answer_var=answer_var,
                operators=deepcopy(list(graph.get("operators", []))),
                score=score,
                compose_output={},
                provenance=deepcopy(dict(graph.get("provenance", {}))),
            )
            candidate_graph.provenance["missing_relation_retrieval"] = {
                "relation_id": relation.relation_id,
                "replaced_relation": old_relation,
                "uses_gold": False,
                "model_calls": 0,
            }
            output.append(
                _Candidate(
                    graph=candidate_graph,
                    evidence=relation,
                    replaced_relation=old_relation,
                    old_orientation=old_orientation,
                    new_orientation=new_orientation,
                    answers=[],
                )
            )
    output.sort(key=lambda item: (-item.graph.score, item.graph.graph_id))
    return output[: max(1, min(12, int(limit)))]


def _execute(pipeline: Any, candidates: Sequence[_Candidate]) -> int:
    count = 0
    for candidate in candidates[:3]:
        started = time.monotonic()
        try:
            candidate.graph.sparql = lower_sparql(
                candidate.graph, limit=getattr(pipeline, "answer_limit", None)
            )
            count += 1
            rows = pipeline.kg.execute(candidate.graph.sparql)
            candidate.answers = _answer_values(rows, candidate.graph.answer_var)
        except Exception as exc:
            candidate.error = f"{type(exc).__name__}:{exc}"
        candidate.query_elapsed_seconds = time.monotonic() - started
    return count


def _answer_relation(candidate: Sequence[str], incumbent: Sequence[str]) -> str:
    left, right = set(map(str, candidate)), set(map(str, incumbent))
    if left == right:
        return "equal"
    if left < right:
        return "subset"
    if left > right:
        return "superset"
    if left & right:
        return "overlap"
    return "disjoint"


def _successful_branch(
    candidate: _Candidate, incumbent: Sequence[str]
) -> tuple[str, int]:
    evidence = candidate.evidence
    count, old_count = len(candidate.answers), len(incumbent)
    answer_relation = _answer_relation(candidate.answers, incumbent)
    disjoint = answer_relation == "disjoint"
    a = bool(
        evidence.incumbent_weighted_support <= 10.0
        and count <= 5
        and evidence.bge_score - evidence.incumbent_bge_score >= 0.0
        and evidence.support_top32 >= 20
        and evidence.orientation_confidence >= 0.99
        and disjoint
    )
    b = bool(
        candidate.graph.score >= 0.90
        and count <= 10
        and old_count >= 3
        and evidence.support_top32 >= 15
        and evidence.orientation_confidence >= 0.95
        and disjoint
        and count / max(1, old_count) <= 2.0
    )
    c = bool(
        count <= 3
        and old_count >= 5
        and evidence.support_top8 >= 4
        and disjoint
    )
    d = bool(
        c
        and evidence.best_template_rank == 1
        and candidate.old_orientation == candidate.new_orientation
        and evidence.support_top8 >= 5
        and 0.0 < evidence.best_relation_overlap < 1.0
        and old_count <= 64
    )
    e = bool(
        answer_relation in {"subset", "superset"}
        and evidence.best_template_rank == 1
        and candidate.old_orientation == candidate.new_orientation
        and evidence.bge_score - evidence.incumbent_bge_score >= 0.0
        and 0.5 <= evidence.best_relation_overlap <= 2.0 / 3.0
        and (
            (answer_relation == "superset" and count >= 3)
            or evidence.bge_score - evidence.incumbent_bge_score >= 0.08
            or evidence.template_support_ratio >= 3.0
        )
    )
    if e:
        return "E", 4
    if a:
        return "A", 3
    if d:
        return "D", 2
    if b and not c and evidence.best_relation_overlap < 1.0:
        return "B", 1
    return "", 0


def _empty_branch(candidate: _Candidate) -> tuple[str, int]:
    if len(candidate.answers) != 1:
        return "", 0
    evidence = candidate.evidence
    if evidence.relation_id == candidate.replaced_relation:
        return "same_relation_relowering", 3
    if evidence.reverse_of_terminal:
        return "ontology_reverse", 2
    if (
        evidence.best_template_rank == 1
        and evidence.best_relation_overlap >= 2.0 / 3.0
        and evidence.orientation_confidence >= 0.99
    ):
        return "top_template_consensus", 1
    return "", 0


def _index(pipeline: Any) -> _AlignedRelationIndex | None:
    cached = getattr(pipeline, "_missing_relation_index", None)
    if cached is not None:
        return cached
    semantic = getattr(pipeline.contract, "semantic_examples", [])
    compose = getattr(pipeline.contract, "compose_examples", [])
    if not semantic or len(semantic) != len(compose):
        return None
    lock = getattr(pipeline, "_missing_relation_index_lock", None)
    if lock is None:
        value = _AlignedRelationIndex(semantic, compose)
        pipeline._missing_relation_index = value
        return value
    with lock:
        cached = getattr(pipeline, "_missing_relation_index", None)
        if cached is None:
            cached = _AlignedRelationIndex(semantic, compose)
            pipeline._missing_relation_index = cached
    return cached


def _graph_dict(graph: QueryGraphCandidate) -> dict[str, Any]:
    return {
        "graph_id": graph.graph_id,
        "triples": deepcopy(graph.triples),
        "answer_var": graph.answer_var,
        "operators": deepcopy(graph.operators),
        "score": graph.score,
        "provenance": deepcopy(graph.provenance),
    }


def _anchor_ids(graph: dict[str, Any]) -> set[str]:
    bindings = (graph.get("provenance") or {}).get("anchor_bindings", {})
    if not isinstance(bindings, dict):
        return set()
    return {
        str(value.get("id", "") if isinstance(value, dict) else value)
        for value in bindings.values()
        if str(value.get("id", "") if isinstance(value, dict) else value)
    }


def _answer_incidence(graph: dict[str, Any]) -> set[str]:
    answer_var = str(graph.get("answer_var", ""))
    return {
        "out" if str(triple[0]) == answer_var else "in"
        for triple in graph.get("triples", [])
        if isinstance(triple, (list, tuple))
        and len(triple) == 3
        and answer_var in {str(triple[0]), str(triple[2])}
    }


def _broad_answer_kind(graph: dict[str, Any], ontology: Any) -> str:
    answer_var = str(graph.get("answer_var", ""))
    type_ids: set[str] = set()
    for triple in graph.get("triples", []):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject, relation, object_ = map(str, triple)
        if object_ == answer_var:
            type_ids.add(str(ontology.range_for_relation(relation)))
        elif subject == answer_var:
            type_ids.add(str(ontology.domain_for_relation(relation)))
    type_ids.discard("")
    scalar = {
        "type.boolean",
        "type.datetime",
        "type.enumeration",
        "type.float",
        "type.int",
        "type.key",
        "type.rawstring",
        "type.string",
        "type.text",
        "type.uri",
    }
    if type_ids and type_ids <= scalar:
        return "scalar"
    if type_ids:
        return "entity"
    return "unknown"


def _highest_scored_successful_fallback_graph(
    traces: Sequence[dict[str, Any]],
    selected: dict[str, Any],
    ontology: Any,
) -> dict[str, Any] | None:
    """Return a trace-proven compatible source after selected-source refusal."""

    final_graphs: list[dict[str, Any]] = []
    for output in _trace_outputs(traces, "final_graph_beam"):
        if isinstance(output, list):
            final_graphs = [item for item in output if isinstance(item, dict)]
    successful_ids: set[str] = set()
    for output in _trace_outputs(
        traces, "deterministic_sparql_lowering_and_execution"
    ):
        for item in output if isinstance(output, list) else []:
            if (
                isinstance(item, dict)
                and int(item.get("answer_count", 0)) > 0
                and str(item.get("graph_id", ""))
            ):
                successful_ids.add(str(item["graph_id"]))
    selected_id = str(selected.get("graph_id", ""))
    selected_anchors = _anchor_ids(selected)
    selected_incidence = _answer_incidence(selected)
    selected_kind = _broad_answer_kind(selected, ontology)
    selected_has_operators = bool(selected.get("operators"))
    selected_triple_count = len(selected.get("triples", []))
    candidates = [
        graph
        for graph in final_graphs
        if str(graph.get("graph_id", "")) in successful_ids
        and str(graph.get("graph_id", "")) != selected_id
        and _anchor_ids(graph) == selected_anchors
        and _answer_incidence(graph) == selected_incidence
        and _broad_answer_kind(graph, ontology) == selected_kind
        and bool(graph.get("operators")) == selected_has_operators
        # A source fallback is justified only when it removes structure from a
        # selected graph that could not produce any viable relation repair.
        # Switching laterally to an equally complex graph is an unconstrained
        # re-selection and can disturb an already-correct incumbent.
        and len(graph.get("triples", [])) < selected_triple_count
    ]
    return (
        max(
            candidates,
            key=lambda graph: (
                float(graph.get("score", 0.0)),
                -len(graph.get("triples", [])),
                str(graph.get("graph_id", "")),
            ),
        )
        if candidates
        else None
    )


def _retrieve_missing_relation_once(
    *,
    pipeline: Any,
    question: str,
    result: dict[str, Any],
    final_empty: bool,
    template_top_k: int = 64,
    candidate_limit: int = 3,
    return_bounded_best_effort: bool = False,
    best_effort_answer_limit: int = 64,
    allow_internal_edge_fallback: bool = False,
) -> dict[str, Any]:
    """Return a gated replacement outcome without consulting test Gold."""

    started = time.monotonic()
    diagnostics: dict[str, Any] = {
        "status": "abstained",
        "trigger": "final_empty" if final_empty else "successful_challenger",
        "uses_gold_answers": False,
        "uses_gold_query": False,
        "llm_calls": 0,
        "candidate_query_limit": min(3, max(1, int(candidate_limit))),
        "endpoint_execution_queries": 0,
        "terminal_execution_queries": 0,
        "internal_execution_queries": 0,
        "internal_edge_fallback_allowed": bool(
            final_empty and allow_internal_edge_fallback
        ),
        "bge_ranker_calls": 0,
    }
    ontology = getattr(getattr(pipeline, "grounder", None), "ontology", None)
    ranker = getattr(getattr(pipeline, "grounder", None), "ranker", None)
    traces = result.get("traces", [])
    if ontology is None or ranker is None or not isinstance(traces, list):
        diagnostics["reason"] = "missing_runtime_dependency"
        return {"answer_ids": [], "diagnostics": diagnostics}
    if not final_empty and _hard_post_selection_applied(traces):
        diagnostics["reason"] = "preserve_hard_post_selection_gate"
        diagnostics["selection_order"] = "hard_post_selection_before_relation_retrieval"
        return {"answer_ids": [], "diagnostics": diagnostics}
    graph = _final_incumbent(traces) if final_empty else result.get("selected_graph")
    if not isinstance(graph, dict):
        diagnostics["reason"] = "missing_incumbent_graph"
        return {"answer_ids": [], "diagnostics": diagnostics}
    relation_index = _index(pipeline)
    if relation_index is None:
        diagnostics["reason"] = "missing_aligned_training_templates"
        return {"answer_ids": [], "diagnostics": diagnostics}
    evidence, terminal = _relation_evidence(
        question=question,
        decompositions=_decompositions(traces),
        surfaces=_anchor_surfaces(traces),
        graph=graph,
        traces=traces,
        index=relation_index,
        ontology=ontology,
        top_k=min(64, max(3, int(template_top_k))),
    )
    if not evidence or not terminal:
        diagnostics["reason"] = "no_relation_proposal"
        return {"answer_ids": [], "diagnostics": diagnostics}
    incumbent = list(map(str, result.get("answer_ids", [])))
    evidence = [
        value
        for value in evidence
        if _local_evidence_possible(
            value,
            graph=graph,
            incumbent_count=len(incumbent),
            final_empty=final_empty,
        )
    ]
    diagnostics["local_prefilter_relation_count"] = len(evidence)
    if not evidence:
        diagnostics["reason"] = "local_pre_embedding_gate_rejected"
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"answer_ids": [], "diagnostics": diagnostics}
    diagnostics["bge_ranker_calls"] = 1
    if not _score_evidence(
        evidence,
        question=question,
        terminal=terminal,
        ranker=ranker,
    ):
        diagnostics["reason"] = "embedding_unavailable"
        return {"answer_ids": [], "diagnostics": diagnostics}
    candidates = _compile(
        graph,
        evidence,
        ontology,
        limit=12,
    )
    diagnostics["compiled_candidate_count"] = len(candidates)
    candidates = [
        candidate
        for candidate in candidates
        if _pre_execution_candidate(
            candidate,
            incumbent_count=len(incumbent),
            final_empty=final_empty,
        )
    ]
    candidates = candidates[: min(3, max(1, int(candidate_limit)))]
    diagnostics["pre_execution_eligible_count"] = len(candidates)
    if not candidates:
        diagnostics["reason"] = "pre_execution_gate_rejected"
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"answer_ids": [], "diagnostics": diagnostics}
    query_budget = min(3, max(1, int(candidate_limit)))
    terminal_query_count = _execute(pipeline, candidates)
    diagnostics["terminal_execution_queries"] = terminal_query_count
    diagnostics["endpoint_execution_queries"] = terminal_query_count
    accepted: list[tuple[int, _Candidate, str]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    for candidate in candidates:
        branch, priority = (
            _empty_branch(candidate)
            if final_empty
            else _successful_branch(candidate, incumbent)
        )
        candidate_diagnostics.append(
            {
                "relation_id": candidate.evidence.relation_id,
                "replaced_relation": candidate.replaced_relation,
                "score": candidate.graph.score,
                "answer_count": len(candidate.answers),
                "answer_relation_to_incumbent": _answer_relation(
                    candidate.answers, incumbent
                ),
                "accepted_branch": branch,
                "query_elapsed_seconds": candidate.query_elapsed_seconds,
                "error": candidate.error,
            }
        )
        if branch and candidate.answers:
            accepted.append((priority, candidate, branch))
    diagnostics["candidates"] = candidate_diagnostics
    internal_selected: _Candidate | None = None
    if (
        final_empty
        and allow_internal_edge_fallback
        and not accepted
        and terminal_query_count > 0
        and terminal_query_count < query_budget
        and all(not candidate.answers for candidate in candidates)
    ):
        remaining = query_budget - terminal_query_count
        internal_candidates = [
            candidate
            for candidate in _compile_internal(
                graph,
                evidence,
                ontology,
                limit=24,
            )
            if _internal_candidate_possible(candidate)
        ][:remaining]
        internal_query_count = _execute(pipeline, internal_candidates)
        diagnostics["internal_execution_queries"] = internal_query_count
        diagnostics["endpoint_execution_queries"] += internal_query_count
        diagnostics["internal_candidates"] = [
            {
                "graph_id": candidate.graph.graph_id,
                "relation_id": candidate.evidence.relation_id,
                "replaced_relation": candidate.replaced_relation,
                "edge_index": candidate.edge_index,
                "edge_role": candidate.edge_role,
                "reversed_edge": candidate.reversed_edge,
                "schema_compatibility": candidate.schema_compatibility,
                "answer_distance": candidate.answer_distance,
                "score": candidate.graph.score,
                "answer_count": len(candidate.answers),
                "answer_ids": list(candidate.answers),
                "query_elapsed_seconds": candidate.query_elapsed_seconds,
                "error": candidate.error,
            }
            for candidate in internal_candidates
        ]
        answer_cap = min(512, max(1, int(best_effort_answer_limit)))
        internal_selected = next(
            (
                candidate
                for candidate in internal_candidates
                if 0 < len(candidate.answers) <= answer_cap
            ),
            None,
        )
    elif final_empty:
        if not allow_internal_edge_fallback:
            diagnostics["internal_edge_fallback_reason"] = "disabled_by_caller"
        elif terminal_query_count <= 0:
            diagnostics["internal_edge_fallback_reason"] = (
                "no_executed_terminal_candidate"
            )
        elif terminal_query_count >= query_budget:
            diagnostics["internal_edge_fallback_reason"] = "shared_query_budget_exhausted"
        elif any(candidate.answers for candidate in candidates):
            diagnostics["internal_edge_fallback_reason"] = (
                "terminal_candidate_was_nonempty"
            )
    bounded_best_effort = False
    internal_best_effort = False
    if accepted:
        # Gate first, then prefer saved-grounding support, branch strength and
        # the pre-endpoint score. This cannot spend more queries than the fixed
        # beam.
        priority, selected, branch = max(
            accepted,
            key=lambda value: (
                bool(value[1].evidence.grounded_union),
                value[0],
                value[1].graph.score,
                value[1].evidence.relation_id,
            ),
        )
    elif internal_selected is not None:
        selected = internal_selected
        priority, branch = 0, "bounded_internal_relation_best_effort"
        bounded_best_effort = True
        internal_best_effort = True
    elif final_empty and return_bounded_best_effort:
        answer_cap = min(512, max(1, int(best_effort_answer_limit)))
        selected = next(
            (
                candidate
                for candidate in candidates
                if 0 < len(candidate.answers) <= answer_cap
            ),
            None,
        )
        if selected is None:
            diagnostics["reason"] = "confidence_gate_rejected"
            diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
            return {"answer_ids": [], "diagnostics": diagnostics}
        priority, branch = 0, "bounded_relation_best_effort"
        bounded_best_effort = True
    else:
        diagnostics["reason"] = "confidence_gate_rejected"
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"answer_ids": [], "diagnostics": diagnostics}
    raw_labels: list[dict[str, str]] = []
    try:
        raw_labels = pipeline.kg.labels(selected.answers)
    except Exception as exc:
        diagnostics["label_error"] = f"{type(exc).__name__}:{exc}"
    labels = {
        str(value.get("id", "")): str(value.get("label", ""))
        for value in raw_labels
        if isinstance(value, dict) and str(value.get("id", ""))
    }
    diagnostics.update(
        {
            "status": (
                "selected_bounded_internal_relation_best_effort"
                if internal_best_effort
                else (
                    "selected_bounded_relation_best_effort"
                    if bounded_best_effort
                    else "selected"
                )
            ),
            "confidence": "low" if bounded_best_effort else "high",
            "selected_branch": branch,
            "selected_relation_id": selected.evidence.relation_id,
            "selected_replaced_relation": selected.replaced_relation,
            "selected_edge_role": selected.edge_role,
            "selected_edge_index": selected.edge_index,
            "answer_count": len(selected.answers),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    )
    return {
        "answer_ids": list(selected.answers),
        "answers": [
            {"id": value, "label": labels.get(value, value)}
            for value in selected.answers
        ],
        "selected_graph": _graph_dict(selected.graph),
        "diagnostics": diagnostics,
    }


def retrieve_missing_relation(
    *,
    pipeline: Any,
    question: str,
    result: dict[str, Any],
    final_empty: bool,
    template_top_k: int = 64,
    candidate_limit: int = 3,
    return_bounded_best_effort: bool = False,
    best_effort_answer_limit: int = 64,
    allow_internal_edge_fallback: bool = False,
) -> dict[str, Any]:
    """Run relation repair, with one trace-proven source fallback on refusal."""

    kwargs = {
        "pipeline": pipeline,
        "question": question,
        "result": result,
        "final_empty": final_empty,
        "template_top_k": template_top_k,
        "candidate_limit": candidate_limit,
        "return_bounded_best_effort": return_bounded_best_effort,
        "best_effort_answer_limit": best_effort_answer_limit,
        "allow_internal_edge_fallback": allow_internal_edge_fallback,
    }
    primary = _retrieve_missing_relation_once(**kwargs)
    if final_empty or primary.get("answer_ids"):
        return primary
    diagnostics = primary.get("diagnostics", {})
    if (
        not isinstance(diagnostics, dict)
        or int(diagnostics.get("endpoint_execution_queries", 0)) != 0
        or int(diagnostics.get("compiled_candidate_count", -1)) != 0
        or int(diagnostics.get("pre_execution_eligible_count", -1)) != 0
    ):
        return primary
    traces = result.get("traces", [])
    selected = result.get("selected_graph")
    ontology = getattr(getattr(pipeline, "grounder", None), "ontology", None)
    if not isinstance(traces, list) or not isinstance(selected, dict) or ontology is None:
        return primary
    fallback_graph = _highest_scored_successful_fallback_graph(
        traces,
        selected,
        ontology,
    )
    if fallback_graph is None:
        return primary
    fallback_result = {**result, "selected_graph": fallback_graph}
    fallback = _retrieve_missing_relation_once(
        **{**kwargs, "result": fallback_result}
    )
    fallback_diagnostics = fallback.get("diagnostics", {})
    if isinstance(fallback_diagnostics, dict):
        fallback_diagnostics["source_graph_fallback"] = {
            "status": "selected_source_pre_execution_refusal",
            "selected_graph_id": str(selected.get("graph_id", "")),
            "fallback_graph_id": str(fallback_graph.get("graph_id", "")),
            "same_anchor_coverage": True,
            "same_answer_incidence": True,
            "same_broad_answer_kind": True,
            "strictly_simpler_source": True,
            "initial_bge_ranker_calls": int(
                diagnostics.get("bge_ranker_calls", 0)
            ),
            "initial_endpoint_execution_queries": 0,
        }
        fallback_diagnostics["bge_ranker_calls"] = int(
            fallback_diagnostics.get("bge_ranker_calls", 0)
        ) + int(diagnostics.get("bge_ranker_calls", 0))
    return fallback
