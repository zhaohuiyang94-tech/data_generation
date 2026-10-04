"""Narrow, model-free relation repair for explicit entity extrema.

The normal pipeline occasionally keeps the right ``ARGMIN``/``ARGMAX``
operator but attaches the answer variable through the wrong relation.  This
lane repairs exactly one answer edge on at most three graphs from the saved
``final_graph_beam``.  It reuses the aligned training/ontology/BGE index owned
by :mod:`missing_relation_retrieval`, executes at most three candidate
queries, and never consults test answers or an LLM.

The acceptance gate is intentionally much narrower than the research
generator.  A candidate must preserve the unique surface extrema, retain the
answer-edge direction, be schema compatible, and return one answer.  It also
needs either an ontology reverse proof on an object-side answer edge (R), or a
strong aligned-template/topology proof (X).
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
import time
from typing import Any, Iterable, Sequence

from .contracts import QueryGraphCandidate
from .lowering import LoweringError, lower_sparql
from .missing_relation_retrieval import (
    _AlignedRelationIndex,
    _Evidence,
    _anchor_surfaces,
    _answer_values,
    _best_endpoint_compatibility,
    _decompositions,
    _hard_post_selection_applied,
    _index,
    _relation_evidence,
    _score_evidence,
    _type_compatibility,
)


__all__ = ["retrieve_extrema_relation"]


_VARIABLE_RE = re.compile(r"^(?:V\d+|P\d+\.V\d+)$", re.IGNORECASE)
_MAXIMUM_RE = re.compile(
    r"\b(?:latest|last|most recent|biggest|largest|highest|maximum|newest|longest)\b",
    re.IGNORECASE,
)
_MINIMUM_RE = re.compile(
    r"\b(?:earliest|least|smallest|lowest|minimum|first|oldest|youngest|shortest)\b",
    re.IGNORECASE,
)
_NON_EXTREMA_NAME_RE = re.compile(r"\b(?:first|last)\s+name\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r'["“][^"”]*["”]')


def _unique_extrema_operator(question: str) -> str:
    """Return the one explicit extrema polarity, or ``""`` if ambiguous."""

    text = _QUOTED_RE.sub(" ", str(question))
    text = _NON_EXTREMA_NAME_RE.sub(" ", text)
    maximum = bool(_MAXIMUM_RE.search(text))
    minimum = bool(_MINIMUM_RE.search(text))
    if maximum == minimum:
        return ""
    return "ARGMAX" if maximum else "ARGMIN"


def _trace_outputs(traces: Sequence[dict[str, Any]], stage: str) -> Iterable[Any]:
    for trace in traces:
        if isinstance(trace, dict) and trace.get("stage") == stage:
            yield trace.get("output")


def _final_graph_beam(traces: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    latest: list[dict[str, Any]] = []
    for output in _trace_outputs(traces, "final_graph_beam"):
        if isinstance(output, list):
            values = [value for value in output if isinstance(value, dict)]
            if values:
                latest = values
    return latest[: max(1, min(3, int(limit)))]


def _selection_reason_codes(traces: Sequence[dict[str, Any]]) -> set[str]:
    latest: dict[str, Any] = {}
    for output in _trace_outputs(traces, "glm_final_graph_ranking"):
        if isinstance(output, dict):
            latest = output
    return {
        str(value)
        for value in latest.get("reason_codes", [])
        if str(value).strip()
    }


def _interlock_reason(result: dict[str, Any], traces: Sequence[dict[str, Any]]) -> str:
    """Protect prior hard/path/template decisions before doing any BGE work."""

    if _hard_post_selection_applied(traces):
        return "preserve_hard_post_selection_gate"
    reason_codes = _selection_reason_codes(traces)
    if "path_alignment_gate" in reason_codes:
        return "preserve_path_alignment_gate"
    if "source_verified_template_consensus_gate" in reason_codes:
        return "preserve_template_consensus_gate"
    blocked_stages = {
        "failure_endpoint_retrieval",
        "failure_endpoint_template_retrieval",
        "missing_relation_retrieval",
        "extrema_relation_retrieval",
    }
    if any(
        isinstance(value, dict) and value.get("stage") in blocked_stages
        for value in traces
    ):
        return "preserve_existing_retrieval_lane"
    recovery = result.get("failure_recovery")
    if isinstance(recovery, dict):
        strategy = str(recovery.get("strategy", ""))
        if "path" in strategy:
            return "preserve_failure_path_retrieval"
        if "template" in strategy:
            return "preserve_failure_template_retrieval"
    graph = result.get("selected_graph")
    provenance = graph.get("provenance", {}) if isinstance(graph, dict) else {}
    if isinstance(provenance, dict):
        if "failure_endpoint_path_retrieval" in provenance:
            return "preserve_failure_path_retrieval"
        if "failure_template_retrieval" in provenance:
            return "preserve_failure_template_retrieval"
    return ""


def _is_variable(value: str) -> bool:
    return bool(_VARIABLE_RE.fullmatch(str(value)))


def _multiset_f1(left: Counter[str], right: Counter[str]) -> float:
    left_size, right_size = sum(left.values()), sum(right.values())
    if not left_size or not right_size:
        return float(left_size == right_size)
    return 2.0 * sum((left & right).values()) / (left_size + right_size)


@dataclass(frozen=True, slots=True)
class _GraphFeatures:
    triples: tuple[tuple[str, str, str], ...]
    answer_var: str
    relations: Counter[str]
    answer_incidence: Counter[str]
    paths: tuple[tuple[tuple[str, str], ...], ...]
    abstract_paths: tuple[tuple[str, ...], ...]
    constant_count: int
    answer_degree: int

    @property
    def triple_count(self) -> int:
        return len(self.triples)


def _graph_features(
    triples: Iterable[Sequence[Any]], answer_var: Any
) -> _GraphFeatures:
    normalized = tuple(
        (str(value[0]), str(value[1]), str(value[2]))
        for value in triples
        if isinstance(value, (list, tuple)) and len(value) == 3 and str(value[1])
    )
    answer = str(answer_var)
    relations = Counter(relation for _, relation, _ in normalized)
    incidence = Counter(
        f"{relation}|{'out' if subject == answer else 'in' if object_ == answer else 'internal'}"
        for subject, relation, object_ in normalized
    )
    nodes = {
        node for subject, _, object_ in normalized for node in (subject, object_)
    }
    adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for subject, relation, object_ in normalized:
        adjacency[subject].append((object_, relation, "out"))
        adjacency[object_].append((subject, relation, "in"))
    constants = sorted(node for node in nodes if not _is_variable(node))
    paths: list[tuple[tuple[str, str], ...]] = []
    for constant in constants:
        queue: deque[tuple[str, tuple[tuple[str, str], ...]]] = deque(
            [(constant, ())]
        )
        visited = {constant}
        while queue:
            node, path = queue.popleft()
            if node == answer:
                paths.append(path)
                break
            for neighbour, relation, direction in sorted(adjacency.get(node, [])):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                queue.append((neighbour, (*path, (relation, direction))))
    ordered = tuple(sorted(paths))
    return _GraphFeatures(
        triples=normalized,
        answer_var=answer,
        relations=relations,
        answer_incidence=incidence,
        paths=ordered,
        abstract_paths=tuple(
            sorted(tuple(direction for _, direction in path) for path in paths)
        ),
        constant_count=len(constants),
        answer_degree=len(adjacency.get(answer, [])),
    )


def _path_similarity(
    left: tuple[tuple[str, str], ...], right: tuple[tuple[str, str], ...]
) -> float:
    if not left or not right:
        return float(left == right)
    denominator = max(len(left), len(right))
    aligned = sum(a == b for a, b in zip(left, right)) / denominator
    left_relations = Counter(value[0] for value in left)
    right_relations = Counter(value[0] for value in right)
    relation = _multiset_f1(left_relations, right_relations)
    direction = sum(a[1] == b[1] for a, b in zip(left, right)) / denominator
    return 0.55 * aligned + 0.25 * relation + 0.20 * direction


def _path_set_similarity(
    left: tuple[tuple[tuple[str, str], ...], ...],
    right: tuple[tuple[tuple[str, str], ...], ...],
) -> float:
    if not left or not right:
        return float(left == right)

    def directed(source: Any, target: Any) -> float:
        return sum(
            max(_path_similarity(path, other) for other in target)
            for path in source
        ) / len(source)

    return 0.5 * (directed(left, right) + directed(right, left))


def _topology_similarity(left: _GraphFeatures, right: _GraphFeatures) -> float:
    relation = _multiset_f1(left.relations, right.relations)
    incidence = _multiset_f1(left.answer_incidence, right.answer_incidence)
    paths = _path_set_similarity(left.paths, right.paths)
    abstract_exact = float(left.abstract_paths == right.abstract_paths)
    metadata = sum(
        (
            left.constant_count == right.constant_count,
            left.triple_count == right.triple_count,
            left.answer_degree == right.answer_degree,
        )
    ) / 3.0
    return (
        0.25 * relation
        + 0.20 * incidence
        + 0.35 * paths
        + 0.12 * abstract_exact
        + 0.08 * metadata
    )


def _weighted_train_features(
    neighbours: Sequence[tuple[float, Any]], selected: _GraphFeatures
) -> tuple[list[tuple[float, _GraphFeatures]], dict[str, float], dict[str, float]]:
    top = max((float(score) for score, _ in neighbours), default=1.0)
    weighted: list[tuple[float, _GraphFeatures]] = []
    relation_support: dict[str, float] = defaultdict(float)
    role_support: dict[str, float] = defaultdict(float)
    for bm25, document in neighbours:
        graph = _graph_features(document.triples, document.answer_var)
        normalized = float(bm25) / max(top, 1e-9)
        overlap = sum((graph.relations & selected.relations).values()) / max(
            1, sum(selected.relations.values())
        )
        weight = normalized * (0.35 + 0.65 * overlap)
        weighted.append((weight, graph))
        for relation, count in graph.relations.items():
            relation_support[relation] += weight * count
        for key, count in graph.answer_incidence.items():
            role_support[key] += weight * count
    return weighted, dict(relation_support), dict(role_support)


def _topology_fit(
    graph: _GraphFeatures,
    neighbours: Sequence[tuple[float, _GraphFeatures]],
) -> tuple[float, float]:
    values = sorted(
        (
            float(weight) * _topology_similarity(graph, document)
            for weight, document in neighbours
        ),
        reverse=True,
    )
    if not values:
        return 0.0, 0.0
    return values[0], sum(values[:3]) / min(3, len(values))


def _endpoint_hints(
    ontology: Any,
    triples: Sequence[Sequence[str]],
    edge_index: int,
    node: str,
) -> list[str]:
    output: list[str] = []
    for index, triple in enumerate(triples):
        if index == edge_index or len(triple) != 3:
            continue
        subject, relation, object_ = map(str, triple)
        value = ""
        if subject == node:
            value = str(ontology.domain_for_relation(relation))
        elif object_ == node:
            value = str(ontology.range_for_relation(relation))
        if value and value not in output:
            output.append(value)
    return output


def _edge_schema_compatibility(
    ontology: Any,
    triples: Sequence[Sequence[str]],
    edge_index: int,
    new_relation: str,
    reversed_edge: bool,
) -> tuple[float, dict[str, Any]]:
    subject, old_relation, object_ = map(str, triples[edge_index])
    new_domain = str(ontology.domain_for_relation(new_relation))
    new_range = str(ontology.range_for_relation(new_relation))
    if reversed_edge:
        new_domain, new_range = new_range, new_domain
    old_domain = str(ontology.domain_for_relation(old_relation))
    old_range = str(ontology.range_for_relation(old_relation))
    subject_hints = _endpoint_hints(ontology, triples, edge_index, subject)
    object_hints = _endpoint_hints(ontology, triples, edge_index, object_)
    subject_score = max(
        _best_endpoint_compatibility(ontology, new_domain, subject_hints),
        0.75 * _type_compatibility(ontology, new_domain, old_domain),
    )
    object_score = max(
        _best_endpoint_compatibility(ontology, new_range, object_hints),
        0.75 * _type_compatibility(ontology, new_range, old_range),
    )
    return 0.5 * (subject_score + object_score), {
        "new_domain": new_domain,
        "new_range": new_range,
        "old_domain": old_domain,
        "old_range": old_range,
        "subject_hints": subject_hints,
        "object_hints": object_hints,
        "subject_compatibility": subject_score,
        "object_compatibility": object_score,
    }


def _sigmoid(value: float) -> float:
    if value >= 30.0:
        return 1.0
    if value <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(slots=True)
class _Candidate:
    key: str
    source_graph_rank: int
    edge_index: int
    edge_role: str
    relation_id: str
    replaced_relation: str
    reversed_edge: bool
    graph: QueryGraphCandidate
    evidence: dict[str, Any]
    sparql: str
    answers: list[str]
    error: str = ""
    elapsed_seconds: float = 0.0


def _edge_role(triple: Sequence[str], answer_var: str) -> str:
    subject, _, object_ = map(str, triple)
    if subject == answer_var:
        return "out"
    if object_ == answer_var:
        return "in"
    return "internal"


def _node_distances(
    triples: Sequence[Sequence[str]], answer_var: str
) -> dict[str, int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        if len(triple) != 3:
            continue
        subject, _, object_ = map(str, triple)
        adjacency[subject].add(object_)
        adjacency[object_].add(subject)
    distances = {answer_var: 0}
    frontier = [answer_var]
    while frontier:
        node = frontier.pop(0)
        for neighbour in adjacency.get(node, set()):
            if neighbour in distances:
                continue
            distances[neighbour] = distances[node] + 1
            frontier.append(neighbour)
    return distances


def _compile_candidates(
    *,
    graph: dict[str, Any],
    evidence: Sequence[_Evidence],
    ontology: Any,
    neighbours: Sequence[tuple[float, Any]],
    source_graph_rank: int,
) -> list[_Candidate]:
    triples = [
        list(map(str, value))
        for value in graph.get("triples", [])
        if isinstance(value, (list, tuple)) and len(value) == 3
    ]
    answer_var = str(graph.get("answer_var", ""))
    if not triples or not answer_var:
        return []
    incumbent = _graph_features(triples, answer_var)
    weighted, relation_support, role_support = _weighted_train_features(
        neighbours, incumbent
    )
    incumbent_best, incumbent_mean = _topology_fit(incumbent, weighted)
    max_relation_support = max(relation_support.values(), default=1.0)
    max_role_support = max(role_support.values(), default=1.0)
    operators = deepcopy(list(graph.get("operators", [])))
    distances = _node_distances(triples, answer_var)
    output: list[_Candidate] = []
    seen: set[str] = set()

    for relation in evidence[:16]:
        for edge_index, triple in enumerate(triples):
            subject, old_relation, object_ = triple
            if relation.relation_id == old_relation:
                continue
            role = _edge_role(triple, answer_var)
            distance = min(
                distances.get(subject, len(triples) + 1),
                distances.get(object_, len(triples) + 1),
            )
            proximity = 1.0 / (1.0 + float(distance))
            orientations = [False]
            if (
                role != "internal"
                and relation.preferred_orientation
                and relation.preferred_orientation != role
            ):
                orientations.insert(0, True)
            elif role == "internal":
                orientations.append(True)
            for reversed_edge in orientations:
                compatibility, schema = _edge_schema_compatibility(
                    ontology,
                    triples,
                    edge_index,
                    relation.relation_id,
                    reversed_edge,
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
                    {
                        "triples": revised,
                        "answer_var": answer_var,
                        "operators": operators,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                revised_features = _graph_features(revised, answer_var)
                best_fit, mean_fit = _topology_fit(revised_features, weighted)
                topology_delta = 0.6 * (best_fit - incumbent_best) + 0.4 * (
                    mean_fit - incumbent_mean
                )
                support_delta = (
                    relation_support.get(relation.relation_id, 0.0)
                    - relation_support.get(old_relation, 0.0)
                ) / max(max_relation_support, 1e-9)
                new_role = (
                    "in" if role == "out" else "out"
                ) if reversed_edge and role != "internal" else role
                role_support_value = role_support.get(
                    f"{relation.relation_id}|{new_role}", 0.0
                ) / max(max_role_support, 1e-9)
                score = (
                    0.48 * relation.candidate_score
                    + 0.18 * best_fit
                    + 0.10 * mean_fit
                    + 0.25 * compatibility
                    + 0.07 * role_support_value
                    + 0.05 * _sigmoid(4.0 * support_delta)
                    + 0.06 * max(-0.5, min(0.5, topology_delta))
                    + 0.04 * proximity
                    + 0.02 * float(not reversed_edge)
                    - 0.012 * (source_graph_rank - 1)
                )
                candidate_graph = QueryGraphCandidate(
                    graph_id=f"ER{hashlib.sha256(signature.encode()).hexdigest()[:16]}",
                    triples=revised,
                    answer_var=answer_var,
                    operators=operators,
                    score=score,
                    compose_output={},
                    provenance=deepcopy(dict(graph.get("provenance", {}))),
                )
                candidate_graph.provenance["extrema_relation_retrieval"] = {
                    "source_graph_rank": source_graph_rank,
                    "edge_index": edge_index,
                    "edge_role": role,
                    "relation_id": relation.relation_id,
                    "replaced_relation": old_relation,
                    "uses_gold": False,
                    "model_calls": 0,
                }
                try:
                    sparql = lower_sparql(candidate_graph)
                except (LoweringError, ValueError):
                    continue
                output.append(
                    _Candidate(
                        key=candidate_graph.graph_id[2:],
                        source_graph_rank=source_graph_rank,
                        edge_index=edge_index,
                        edge_role=role,
                        relation_id=relation.relation_id,
                        replaced_relation=old_relation,
                        reversed_edge=reversed_edge,
                        graph=candidate_graph,
                        evidence={
                            **relation.dump(),
                            **schema,
                            "incumbent_topology_best": incumbent_best,
                            "candidate_topology_best": best_fit,
                            "incumbent_topology_mean3": incumbent_mean,
                            "candidate_topology_mean3": mean_fit,
                            "topology_delta": topology_delta,
                            "old_relation_support": relation_support.get(
                                old_relation, 0.0
                            ),
                            "new_relation_support": relation_support.get(
                                relation.relation_id, 0.0
                            ),
                            "normalized_support_delta": support_delta,
                            "normalized_role_support": role_support_value,
                            "schema_compatibility": compatibility,
                            "answer_distance": distance,
                            "answer_proximity": proximity,
                        },
                        sparql=sparql,
                        answers=[],
                    )
                )
    output.sort(
        key=lambda value: (
            -value.graph.score,
            value.source_graph_rank,
            value.edge_index,
            value.key,
        )
    )
    return output


def _proof_from_values(
    *,
    edge_role: str,
    reversed_edge: bool,
    operators: Sequence[dict[str, Any]],
    evidence: dict[str, Any],
    wanted_operator: str,
) -> str:
    """Shared production proof gate used by runtime and offline audit."""

    operator_types = {
        str(value.get("type", "")).upper()
        for value in operators
        if isinstance(value, dict)
    }
    opposite = "ARGMIN" if wanted_operator == "ARGMAX" else "ARGMAX"
    base = bool(
        edge_role in {"in", "out"}
        and not reversed_edge
        and str(evidence.get("preferred_orientation", "")) == edge_role
        and wanted_operator in operator_types
        and opposite not in operator_types
        and float(evidence.get("schema_compatibility", 0.0)) >= 0.75
    )
    if not base:
        return ""
    reverse = bool(
        edge_role == "in" and evidence.get("reverse_of_terminal")
    )
    template = bool(
        int(evidence.get("best_template_rank", 10**9)) == 1
        and int(evidence.get("support_top8", 0)) >= 8
        and evidence.get("grounded_union")
        and float(evidence.get("topology_delta", 0.0)) >= 0.20
        and float(evidence.get("normalized_support_delta", 0.0)) >= 0.90
    )
    return "R" if reverse else "X" if template else ""


def _proof_branch(candidate: _Candidate, wanted_operator: str) -> str:
    return _proof_from_values(
        edge_role=candidate.edge_role,
        reversed_edge=candidate.reversed_edge,
        operators=candidate.graph.operators,
        evidence=candidate.evidence,
        wanted_operator=wanted_operator,
    )


def _candidate_audit(candidate: _Candidate, branch: str) -> dict[str, Any]:
    return {
        "graph_id": candidate.graph.graph_id,
        "source_graph_rank": candidate.source_graph_rank,
        "edge_index": candidate.edge_index,
        "edge_role": candidate.edge_role,
        "relation_id": candidate.relation_id,
        "replaced_relation": candidate.replaced_relation,
        "reversed_edge": candidate.reversed_edge,
        "score": candidate.graph.score,
        "schema_compatibility": candidate.evidence.get(
            "schema_compatibility", 0.0
        ),
        "topology_delta": candidate.evidence.get("topology_delta", 0.0),
        "normalized_support_delta": candidate.evidence.get(
            "normalized_support_delta", 0.0
        ),
        "proof_branch": branch,
        "answer_count": len(candidate.answers),
        "answer_ids": list(candidate.answers),
        "error": candidate.error,
        "query_elapsed_seconds": candidate.elapsed_seconds,
    }


def retrieve_extrema_relation(
    *,
    pipeline: Any,
    question: str,
    result: dict[str, Any],
    template_top_k: int = 64,
    source_graph_limit: int = 3,
    candidate_limit: int = 3,
) -> dict[str, Any]:
    """Return a Gold-blind answer-edge extrema repair, or abstain."""

    started = time.monotonic()
    query_limit = min(3, max(1, int(candidate_limit)))
    diagnostics: dict[str, Any] = {
        "status": "abstained",
        "uses_gold_answers": False,
        "uses_gold_query": False,
        "uses_relation_or_sample_whitelist": False,
        "llm_calls": 0,
        "trigger": "successful_explicit_unique_extrema",
        "candidate_query_limit": query_limit,
        "endpoint_execution_queries": 0,
        "bge_ranker_calls": 0,
        "source_graph_limit": min(3, max(1, int(source_graph_limit))),
    }
    incumbent = list(map(str, result.get("answer_ids", [])))
    if not incumbent or result.get("failure") is not None:
        diagnostics["reason"] = "requires_successful_nonempty_incumbent"
        return {"answer_ids": [], "diagnostics": diagnostics}
    wanted = _unique_extrema_operator(question)
    diagnostics["wanted_operator"] = wanted
    if not wanted:
        diagnostics["reason"] = "surface_extrema_not_unique"
        return {"answer_ids": [], "diagnostics": diagnostics}
    traces = result.get("traces", [])
    if not isinstance(traces, list):
        diagnostics["reason"] = "missing_trace_bundle"
        return {"answer_ids": [], "diagnostics": diagnostics}
    interlock = _interlock_reason(result, traces)
    if interlock:
        diagnostics["reason"] = interlock
        diagnostics["interlock_applied"] = True
        return {"answer_ids": [], "diagnostics": diagnostics}
    ontology = getattr(getattr(pipeline, "grounder", None), "ontology", None)
    ranker = getattr(getattr(pipeline, "grounder", None), "ranker", None)
    if ontology is None or ranker is None:
        diagnostics["reason"] = "missing_runtime_dependency"
        return {"answer_ids": [], "diagnostics": diagnostics}
    relation_index: _AlignedRelationIndex | None = _index(pipeline)
    if relation_index is None:
        diagnostics["reason"] = "missing_aligned_training_templates"
        return {"answer_ids": [], "diagnostics": diagnostics}
    beam = _final_graph_beam(traces, source_graph_limit)
    diagnostics["source_graph_count"] = len(beam)
    if not beam:
        diagnostics["reason"] = "missing_final_graph_beam"
        return {"answer_ids": [], "diagnostics": diagnostics}

    decompositions = _decompositions(traces)
    surfaces = _anchor_surfaces(traces)
    all_candidates: list[_Candidate] = []
    graph_audit: list[dict[str, Any]] = []
    for graph_rank, graph in enumerate(beam, 1):
        evidence, terminal = _relation_evidence(
            question=question,
            decompositions=decompositions,
            surfaces=surfaces,
            graph=graph,
            traces=traces,
            index=relation_index,
            ontology=ontology,
            top_k=min(64, max(3, int(template_top_k))),
        )
        if not evidence or not terminal:
            continue
        diagnostics["bge_ranker_calls"] += 1
        if not _score_evidence(
            evidence,
            question=question,
            terminal=terminal,
            ranker=ranker,
        ):
            continue
        neighbours = relation_index.retrieve(
            question,
            decompositions,
            surfaces,
            top_k=min(64, max(3, int(template_top_k))),
        )
        compiled = _compile_candidates(
            graph=graph,
            evidence=evidence,
            ontology=ontology,
            neighbours=neighbours,
            source_graph_rank=graph_rank,
        )
        all_candidates.extend(compiled)
        graph_audit.append(
            {
                "source_graph_rank": graph_rank,
                "terminal_relations": list(terminal),
                "relation_evidence_count": len(evidence),
                "compiled_candidate_count": len(compiled),
            }
        )
    diagnostics["graph_audit"] = graph_audit
    all_candidates.sort(
        key=lambda value: (
            -value.graph.score,
            value.source_graph_rank,
            value.edge_index,
            value.key,
        )
    )
    eligible: list[tuple[_Candidate, str]] = []
    seen_sparql: set[str] = set()
    for candidate in all_candidates:
        if candidate.sparql in seen_sparql:
            continue
        seen_sparql.add(candidate.sparql)
        branch = _proof_branch(candidate, wanted)
        if branch:
            eligible.append((candidate, branch))
        if len(eligible) >= query_limit:
            break
    diagnostics["compiled_candidate_count"] = len(all_candidates)
    diagnostics["pre_execution_eligible_count"] = len(eligible)
    if not eligible:
        diagnostics["reason"] = "extrema_proof_gate_rejected"
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"answer_ids": [], "diagnostics": diagnostics}

    for candidate, _ in eligible:
        query_started = time.monotonic()
        try:
            diagnostics["endpoint_execution_queries"] += 1
            rows = pipeline.kg.execute(candidate.sparql)
            candidate.answers = _answer_values(rows, candidate.graph.answer_var)
        except Exception as exc:
            candidate.error = f"{type(exc).__name__}:{exc}"
        candidate.elapsed_seconds = time.monotonic() - query_started
    accepted = [
        (candidate, branch)
        for candidate, branch in eligible
        if not candidate.error and len(candidate.answers) == 1
    ]
    diagnostics["candidates"] = [
        _candidate_audit(candidate, branch) for candidate, branch in eligible
    ]
    if not accepted:
        diagnostics["reason"] = "singleton_answer_gate_rejected"
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"answer_ids": [], "diagnostics": diagnostics}
    selected, branch = accepted[0]
    diagnostics.update(
        {
            "selected_graph_id": selected.graph.graph_id,
            "selected_proof_branch": branch,
            "selected_relation_id": selected.relation_id,
            "selected_replaced_relation": selected.replaced_relation,
            "selected_edge_role": selected.edge_role,
            "selected_source_graph_rank": selected.source_graph_rank,
            "answer_count": 1,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    )
    if set(selected.answers) == set(incumbent):
        diagnostics["status"] = "unchanged_equal_answer"
        diagnostics["reason"] = "candidate_answer_equals_incumbent"
        return {"answer_ids": [], "diagnostics": diagnostics}
    labels: dict[str, str] = {}
    try:
        labels = {
            str(value.get("id", "")): str(value.get("label", ""))
            for value in pipeline.kg.labels(selected.answers)
            if isinstance(value, dict) and str(value.get("id", ""))
        }
    except Exception as exc:
        diagnostics["label_error"] = f"{type(exc).__name__}:{exc}"
    diagnostics["status"] = "selected"
    diagnostics["reason"] = f"extrema_relation_proof_{branch}"
    return {
        "answer_ids": list(selected.answers),
        "answers": [
            {"id": value, "label": labels.get(value, value)}
            for value in selected.answers
        ],
        "selected_graph": {
            "graph_id": selected.graph.graph_id,
            "triples": deepcopy(selected.graph.triples),
            "answer_var": selected.graph.answer_var,
            "operators": deepcopy(selected.graph.operators),
            "score": selected.graph.score,
            "provenance": deepcopy(selected.graph.provenance),
        },
        "diagnostics": diagnostics,
    }
