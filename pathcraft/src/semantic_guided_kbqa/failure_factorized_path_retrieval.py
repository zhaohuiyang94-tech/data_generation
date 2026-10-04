"""Bounded final-empty fallback over cached CWQ TRAIN relation paths.

This module is intentionally a terminal failure lane.  Its caller must invoke
it only after earlier failure retrieval lanes have all returned no answer.  It
uses the current question, saved decomposition/semantic trace, configured
entity links, local ontology, and a read-only cache built from CWQ TRAIN Gold
SPARQL.  It makes no LLM or embedding call and executes at most three final
SELECT queries.

The frozen policy independently retrieves a constant-to-answer path for every
linked anchor, joins two or more paths at their common answer terminal, and
uses answer-set consensus followed by retrieval score for runtime selection.
Raw queries are never emitted in diagnostics; only SHA-256 hashes are kept.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import heapq
from itertools import product
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Sequence

from .failure_gold_sparql_retrieval import (
    _decompositions,
    _entities,
    _trace_surfaces,
)
from .ontology import FreebaseOntology, relation_id_from_label
from .pipeline import _answer_values
from .template_path_retrieval import mask_entities


VERSION = "failure-factorized-source-train-path-production-v1-frozen"
CACHE_VERSION = 2
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
FULL_DATE_RE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
YEAR_RE = re.compile(r"(?<![\d.])(?:1[0-9]{3}|20[0-9]{2})(?![\d.])")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")
ENTITY_ID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")


def _normalized(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _tokens(value: Any) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in TOKEN_RE.finditer(str(value)))


def _normalize_scalars(value: Any) -> str:
    text = FULL_DATE_RE.sub(" date ", str(value))
    text = YEAR_RE.sub(" year ", text)
    text = NUMBER_RE.sub(" number ", text)
    return " ".join(text.split())


def _masked(value: Any, surfaces: Sequence[str]) -> str:
    return mask_entities(_normalize_scalars(str(value)), surfaces)


def _relation_text(relations: Sequence[str]) -> str:
    return " ".join(
        segment.replace("_", " ")
        for relation in relations
        for segment in str(relation).split(".")
    )


def _token_f1(left: Any, right: Any) -> float:
    one, two = set(_tokens(left)), set(_tokens(right))
    if not one or not two:
        return 0.0
    overlap = len(one & two)
    return 2.0 * overlap / (len(one) + len(two))


def _set_f1(left: Sequence[Any], right: Sequence[Any]) -> float:
    one, two = set(left), set(right)
    if not one or not two:
        return float(one == two)
    overlap = len(one & two)
    return 2.0 * overlap / (len(one) + len(two))


@dataclass(frozen=True, slots=True)
class GoldPath:
    relations: tuple[str, ...]
    directions: tuple[str, ...]

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.relations, self.directions))


@dataclass(frozen=True, slots=True)
class _PathDocument:
    source_index: int
    question: str
    goal: str
    path: GoldPath
    semantic_path: GoldPath
    operator_signature: tuple[str, ...]
    tokens: tuple[str, ...]
    alignment: tuple[float, ...]
    source_kind: str


class TrainGoldPathIndex:
    """Read-only sparse BM25 index over the prebuilt TRAIN path cache."""

    def __init__(self, cache_path: str | Path) -> None:
        started = time.perf_counter()
        source = Path(cache_path).expanduser()
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != CACHE_VERSION
            or not isinstance(payload.get("documents"), list)
        ):
            raise ValueError(f"incompatible TRAIN Gold path cache: {source}")
        self.documents = [
            _PathDocument(
                source_index=int(item["source_index"]),
                question=str(item.get("question", "")),
                goal=str(item.get("goal", "")),
                path=GoldPath(
                    tuple(map(str, item.get("relations", []))),
                    tuple(map(str, item.get("directions", []))),
                ),
                semantic_path=GoldPath(
                    tuple(map(str, item.get("semantic_relations", []))),
                    tuple(map(str, item.get("semantic_directions", []))),
                ),
                operator_signature=tuple(
                    map(str, item.get("operator_signature", []))
                ),
                tokens=tuple(map(str, item.get("tokens", []))),
                alignment=tuple(map(float, item.get("alignment", []))),
                source_kind=str(item.get("source_kind", "semantic_aligned")),
            )
            for item in payload["documents"]
            if isinstance(item, dict)
            and item.get("relations")
            and len(item.get("relations", [])) == len(item.get("directions", []))
        ]
        self.frequencies = [Counter(document.tokens) for document in self.documents]
        self.average_length = sum(
            len(document.tokens) for document in self.documents
        ) / max(1, len(self.documents))
        document_frequency: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, frequency in enumerate(self.frequencies):
            document_frequency.update(set(frequency))
            for token, count in frequency.items():
                self.postings[token].append((index, count))
        count = len(self.documents)
        self.idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.load_seconds = time.perf_counter() - started

    def retrieve(self, query: str, *, top_k: int = 64) -> list[tuple[float, int, _PathDocument]]:
        query_frequency = Counter(_tokens(query))
        selected_tokens = sorted(
            query_frequency,
            key=lambda token: (
                self.idf.get(token, 0.0),
                query_frequency[token],
                token,
            ),
            reverse=True,
        )[:28]
        sparse_scores: dict[int, float] = defaultdict(float)
        for token in selected_tokens:
            inverse = self.idf.get(token, 0.0)
            for index, frequency in self.postings.get(token, ()):
                normalization = (
                    0.25
                    + 0.75
                    * len(self.documents[index].tokens)
                    / max(1e-9, self.average_length)
                )
                sparse_scores[index] += inverse * (
                    frequency * 2.2 / (frequency + 1.2 * normalization)
                ) * min(2, query_frequency[token])
        preselected = heapq.nlargest(
            max(256, top_k * 4),
            sparse_scores.items(),
            key=lambda item: (item[1], -item[0]),
        )
        ranked: list[tuple[float, int, _PathDocument]] = []
        for index, lexical_score in preselected:
            document = self.documents[index]
            alignment = document.alignment
            score = float(lexical_score)
            score += 0.35 * float(alignment[1] if len(alignment) > 1 else 0.0)
            score += 0.20 * float(alignment[2] if len(alignment) > 2 else 0.0)
            ranked.append((score, index, document))
        ranked.sort(
            key=lambda value: (-value[0], value[2].source_index, value[1])
        )
        return ranked[: max(1, int(top_k))]


@dataclass(frozen=True, slots=True)
class _SemanticIntent:
    surface: str
    goal: str
    path: GoldPath


def _semantic_intents(context: Any) -> list[_SemanticIntent]:
    traces = (context.trace_bundle.get("decomposition", {}) or {}).get("traces", [])
    output: list[_SemanticIntent] = []
    seen: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
    for event in traces if isinstance(traces, list) else []:
        if not isinstance(event, dict) or event.get("stage") != "semantic_path_generation":
            continue
        candidates = event.get("output", [])
        for candidate in candidates if isinstance(candidates, list) else []:
            if not isinstance(candidate, dict):
                continue
            payload = candidate.get("output") or {}
            anchors = {
                str(anchor.get("id", "")): str(anchor.get("surface", ""))
                for anchor in payload.get("anchors", [])
                if isinstance(anchor, dict)
            }
            for path in payload.get("semantic_paths", []):
                if not isinstance(path, dict):
                    continue
                relations: list[str] = []
                directions: list[str] = []
                for step in path.get("steps", []):
                    if not isinstance(step, dict):
                        continue
                    relation = relation_id_from_label(step.get("relation_label", []))
                    if not relation:
                        continue
                    direction = str(step.get("direction", "forward")).casefold()
                    relations.append(relation)
                    directions.append(
                        direction if direction in {"forward", "backward"} else "forward"
                    )
                if not relations:
                    continue
                surface = anchors.get(str(path.get("anchor_ref", "")), "")
                goal = str(path.get("goal", ""))
                key = (surface, goal, tuple(relations), tuple(directions))
                if key in seen:
                    continue
                seen.add(key)
                output.append(
                    _SemanticIntent(
                        surface,
                        goal,
                        GoldPath(tuple(relations), tuple(directions)),
                    )
                )
    return output


def _intent_entity_score(
    intent: _SemanticIntent, label: str, question: str
) -> float:
    one, two = _normalized(intent.surface), _normalized(label)
    exact = float(bool(one and two and one == two))
    contained = float(bool(one and two and (one in two or two in one)))
    mention = float(bool(two and two in _normalized(intent.goal)))
    question_mention = float(bool(two and two in _normalized(question)))
    return (
        5.0 * exact
        + 1.5 * contained
        + mention
        + 0.1 * question_mention
        + _token_f1(one, two)
    )


def _assign_intents(
    intents: Sequence[_SemanticIntent],
    entities: Sequence[tuple[str, str]],
    question: str,
) -> dict[str, list[_SemanticIntent]]:
    output: dict[str, list[_SemanticIntent]] = {
        entity_id: [] for entity_id, _ in entities
    }
    for intent in intents:
        ranked = sorted(
            (
                (_intent_entity_score(intent, label, question), entity_id)
                for entity_id, label in entities
            ),
            key=lambda value: (-value[0], value[1]),
        )
        if ranked and ranked[0][0] >= 1.0:
            output[ranked[0][1]].append(intent)
    return output


def _local_clauses(
    label: str,
    decomposition: Sequence[str],
    intents: Sequence[_SemanticIntent],
) -> list[str]:
    needle = _normalized(label)
    explicit = [
        str(value)
        for value in decomposition
        if needle and needle in _normalized(value)
    ]
    goals = [intent.goal for intent in intents if intent.goal]
    return _unique([*explicit, *goals]) or list(map(str, decomposition))


def _path_similarity(
    left: GoldPath, right: GoldPath
) -> tuple[float, float, float, float]:
    return (
        float(left.pairs == right.pairs),
        _set_f1(left.pairs, right.pairs),
        _set_f1(left.relations, right.relations),
        min(len(left.relations), len(right.relations))
        / max(1, max(len(left.relations), len(right.relations))),
    )


def _path_start_type(path: GoldPath, ontology: FreebaseOntology) -> str:
    if not path.relations:
        return ""
    relation, direction = path.relations[0], path.directions[0]
    return (
        ontology.domain_for_relation(relation)
        if direction == "forward"
        else ontology.range_for_relation(relation)
    )


def _path_terminal_type(path: GoldPath, ontology: FreebaseOntology) -> str:
    if not path.relations:
        return ""
    relation, direction = path.relations[-1], path.directions[-1]
    return (
        ontology.range_for_relation(relation)
        if direction == "forward"
        else ontology.domain_for_relation(relation)
    )


def _type_compatibility(
    left: str, right: str, ontology: FreebaseOntology
) -> float:
    if not left or not right:
        return 0.25
    if left == right:
        return 1.0
    left_supers = set(ontology.supertypes(left))
    right_supers = set(ontology.supertypes(right))
    if left in right_supers or right in left_supers:
        return 0.9
    overlap = (left_supers & right_supers) - {"common.topic", "type.object"}
    return 0.65 if overlap else 0.0


@dataclass(slots=True)
class _RetrievedPath:
    entity_id: str
    entity_label: str
    path: GoldPath
    score: float
    retrieval_rank: int
    source_index: int
    source_kind: str
    support: int

    @property
    def signature(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self.path.relations, self.path.directions


def _retrieve_for_entity(
    index: TrainGoldPathIndex,
    ontology: FreebaseOntology,
    *,
    question: str,
    decomposition: Sequence[str],
    surfaces: Sequence[str],
    entity_id: str,
    entity_label: str,
    intents: Sequence[_SemanticIntent],
    top_k: int,
) -> list[_RetrievedPath]:
    clauses = _local_clauses(entity_label, decomposition, intents)
    semantic_relations = [
        relation for intent in intents for relation in intent.path.relations
    ]
    query = " ".join(
        [
            _masked(question, surfaces),
            _masked(question, surfaces),
            *(_masked(clause, surfaces) for clause in clauses for _ in range(3)),
            _relation_text(semantic_relations),
        ]
    )
    retrieved = index.retrieve(query, top_k=max(64, top_k * 8))
    support: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter(
        (document.path.relations, document.path.directions)
        for _, _, document in retrieved[:64]
    )
    ranked: list[_RetrievedPath] = []
    for retrieval_rank, (bm25, _, document) in enumerate(retrieved, start=1):
        similarities = [
            _path_similarity(intent.path, document.path) for intent in intents
        ]
        best = max(similarities, default=(0.0, 0.0, 0.0, 0.0))
        start_compatibility = max(
            (
                _type_compatibility(
                    _path_start_type(intent.path, ontology),
                    _path_start_type(document.path, ontology),
                    ontology,
                )
                for intent in intents
            ),
            default=0.25,
        )
        signature = (document.path.relations, document.path.directions)
        score = (
            float(bm25)
            + 2.4 * best[0]
            + 1.4 * best[1]
            + 0.7 * best[2]
            + 0.6 * start_compatibility
            + 0.25 * math.log1p(support[signature])
            + 0.25 * float(document.alignment[1] if len(document.alignment) > 1 else 0.0)
        )
        ranked.append(
            _RetrievedPath(
                entity_id=entity_id,
                entity_label=entity_label,
                path=document.path,
                score=score,
                retrieval_rank=retrieval_rank,
                source_index=document.source_index,
                source_kind=document.source_kind,
                support=support[signature],
            )
        )
    ranked.sort(
        key=lambda value: (
            -value.score,
            value.retrieval_rank,
            value.source_index,
            value.signature,
        )
    )
    output: list[_RetrievedPath] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for value in ranked:
        if value.signature in seen:
            continue
        seen.add(value.signature)
        output.append(value)
        if len(output) >= max(1, int(top_k)):
            break
    return output


def _render_path(path: _RetrievedPath, path_index: int) -> list[str]:
    current = f"ns:{path.entity_id}"
    triples: list[str] = []
    for step_index, (relation, direction) in enumerate(path.path.pairs):
        terminal = step_index == len(path.path.relations) - 1
        next_node = "?x" if terminal else f"?p{path_index}_{step_index}"
        if direction == "forward":
            triples.append(f"{current} ns:{relation} {next_node} .")
        else:
            triples.append(f"{next_node} ns:{relation} {current} .")
        current = next_node
    return triples


def _candidate_query(paths: Sequence[_RetrievedPath]) -> str:
    body = "\n".join(
        triple
        for path_index, path in enumerate(paths)
        for triple in _render_path(path, path_index)
    )
    return (
        "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
        "SELECT DISTINCT ?x\nWHERE {\n"
        f"{body}\n"
        "FILTER (!isLiteral(?x) || lang(?x) = '' || langMatches(lang(?x), 'en'))\n"
        "}\nLIMIT 256"
    )


def _build_candidates(
    entity_paths: dict[str, list[_RetrievedPath]],
    ontology: FreebaseOntology,
    *,
    combination_beam: int,
) -> list[dict[str, Any]]:
    active = [(entity_id, paths) for entity_id, paths in entity_paths.items() if paths]
    if len(active) < 2:
        return []
    combinations: list[tuple[float, tuple[_RetrievedPath, ...]]] = []
    for values in product(
        *(paths[: max(1, int(combination_beam))] for _, paths in active)
    ):
        terminal_types = [
            _path_terminal_type(value.path, ontology) for value in values
        ]
        compatibilities = [
            _type_compatibility(
                terminal_types[left], terminal_types[right], ontology
            )
            for left in range(len(values))
            for right in range(left + 1, len(values))
        ]
        if compatibilities and min(compatibilities) <= 0.0:
            continue
        score = sum(value.score for value in values)
        score += 1.2 * sum(compatibilities) / max(1, len(compatibilities))
        score += 0.15 * len({value.source_index for value in values})
        combinations.append((score, tuple(values)))
    combinations.sort(
        key=lambda value: (
            -value[0],
            tuple(path.retrieval_rank for path in value[1]),
            tuple(path.signature for path in value[1]),
        )
    )
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, paths in combinations:
        query = _candidate_query(paths)
        compact = " ".join(query.split())
        if compact in seen:
            continue
        seen.add(compact)
        output.append(
            {
                "key": hashlib.sha256(query.encode()).hexdigest()[:20],
                "query": query,
                "runtime_score": score,
                "answer_var": "x",
                "paths": [
                    {
                        "entity_id": path.entity_id,
                        "entity_label": path.entity_label,
                        "relations": list(path.path.relations),
                        "directions": list(path.path.directions),
                        "retrieval_rank": path.retrieval_rank,
                        "source_index": path.source_index,
                        "source_kind": path.source_kind,
                        "support_top64": path.support,
                    }
                    for path in paths
                ],
            }
        )
        if len(output) >= 32:
            break
    return output


def _choose_runtime(candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    nonempty = [candidate for candidate in candidates if candidate.get("answers")]
    if not nonempty:
        return None
    support = Counter(
        tuple(sorted(map(str, candidate["answers"]))) for candidate in nonempty
    )
    return max(
        nonempty,
        key=lambda candidate: (
            support[tuple(sorted(map(str, candidate["answers"])))],
            float(candidate.get("runtime_score", 0.0)),
            str(candidate.get("key", "")),
        ),
    )


def _index(pipeline: Any) -> TrainGoldPathIndex:
    cached = getattr(pipeline, "_failure_factorized_path_index", None)
    if cached is not None:
        return cached
    lock = getattr(pipeline, "_failure_factorized_path_index_lock", None)
    if lock is None:
        cached = TrainGoldPathIndex(pipeline.failure_factorized_path_cache_path)
        pipeline._failure_factorized_path_index = cached
        return cached
    with lock:
        cached = getattr(pipeline, "_failure_factorized_path_index", None)
        if cached is None:
            cached = TrainGoldPathIndex(pipeline.failure_factorized_path_cache_path)
            pipeline._failure_factorized_path_index = cached
    return cached


def retrieve(context: Any, endpoint: Any) -> dict[str, Any]:
    """Run the frozen terminal fallback and return a pipeline-style outcome."""
    started = time.perf_counter()
    pipeline = context.pipeline
    diagnostics: dict[str, Any] = {
        "version": VERSION,
        "status": "abstained",
        "trigger": "final_empty_after_temporal_direct_and_adaptive_lanes",
        "source": "CWQ_TRAIN_gold_sparql_path_cache_only",
        "uses_evaluation_gold": False,
        "uses_test_sparql_or_reasoning_information": False,
        "llm_calls": 0,
        "embedding_calls": 0,
        "entity_or_relation_whitelist": False,
        "candidate_query_limit": 3,
        "raw_queries_exposed": False,
    }
    entities = _entities(context)
    if len(entities) < 2:
        diagnostics["reason"] = "requires_at_least_two_linked_entities"
        diagnostics["elapsed_seconds"] = time.perf_counter() - started
        return {"answer_ids": [], "answers": [], "selected_graph": None, "diagnostics": diagnostics}
    decomposition = _decompositions(context)
    intents = _semantic_intents(context)
    assigned = _assign_intents(intents, entities, context.question)
    ontology = getattr(pipeline, "ontology", None)
    if ontology is None:
        diagnostics["reason"] = "ontology_unavailable"
        diagnostics["elapsed_seconds"] = time.perf_counter() - started
        return {"answer_ids": [], "answers": [], "selected_graph": None, "diagnostics": diagnostics}
    index = _index(pipeline)
    surfaces = _unique(
        [label for _, label in entities] + _trace_surfaces(context)
    )
    entity_paths: dict[str, list[_RetrievedPath]] = {}
    top_k = max(1, int(getattr(pipeline, "failure_factorized_path_top_k", 6)))
    for entity_id, entity_label in entities:
        entity_paths[entity_id] = _retrieve_for_entity(
            index,
            ontology,
            question=context.question,
            decomposition=decomposition,
            surfaces=surfaces,
            entity_id=entity_id,
            entity_label=entity_label,
            intents=assigned.get(entity_id, []),
            top_k=top_k,
        )
    candidates = _build_candidates(
        entity_paths,
        ontology,
        combination_beam=max(
            1, int(getattr(pipeline, "failure_factorized_path_combination_beam", 5))
        ),
    )
    limit = min(
        3,
        max(1, int(getattr(pipeline, "failure_factorized_path_candidate_limit", 3))),
    )
    selected_candidates = candidates[:limit]
    execution_queries = 0
    for candidate in selected_candidates:
        query_started = time.perf_counter()
        try:
            rows = endpoint.execute(candidate["query"])
            answers = _unique(_answer_values(rows, candidate["answer_var"]))
            error = ""
        except Exception as exc:
            answers = []
            error = f"{type(exc).__name__}:{exc}"
        execution_queries += 1
        candidate["answers"] = answers
        candidate["error"] = error
        candidate["elapsed_seconds"] = time.perf_counter() - query_started
    selected = _choose_runtime(selected_candidates)
    answer_ids = list(selected.get("answers", [])) if selected else []

    label_rows: list[dict[str, str]] = []
    label_error = ""
    entity_answer_ids = [
        answer_id for answer_id in answer_ids if ENTITY_ID_RE.fullmatch(answer_id)
    ]
    if entity_answer_ids:
        try:
            label_rows = list(pipeline.kg.labels(entity_answer_ids))
        except Exception as exc:
            label_error = f"{type(exc).__name__}:{exc}"
    labels_by_id = {
        str(item.get("id", "")): str(item.get("label", ""))
        for item in label_rows
        if isinstance(item, dict) and str(item.get("id", ""))
    }
    candidate_traces: list[dict[str, Any]] = []
    for rank, candidate in enumerate(selected_candidates, start=1):
        query = str(candidate.pop("query"))
        candidate_traces.append(
            {
                "rank": rank,
                "key": str(candidate["key"]),
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "runtime_score": float(candidate["runtime_score"]),
                "answer_count": len(candidate["answers"]),
                "elapsed_seconds": float(candidate["elapsed_seconds"]),
                "error": str(candidate["error"]),
                "paths": candidate["paths"],
            }
        )
    diagnostics.update(
        {
            "status": "selected" if selected else "all_candidates_empty",
            "reason": (
                "factorized_train_path_answer_consensus"
                if selected
                else "all_factorized_queries_empty"
            ),
            "index_size": len(index.documents),
            "index_load_seconds": index.load_seconds,
            "linked_entity_count": len(entities),
            "retrieved_path_counts": {
                entity_id: len(paths) for entity_id, paths in entity_paths.items()
            },
            "generated_candidate_count": len(candidates),
            "endpoint_execution_queries": execution_queries,
            "answer_count": len(answer_ids),
            "selected_candidate_key": str(selected.get("key", "")) if selected else "",
            "candidate_traces": candidate_traces,
            "selected_label_queries": int(bool(entity_answer_ids)),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    if label_error:
        diagnostics["selected_label_error"] = label_error
    return {
        "answer_ids": answer_ids,
        "answers": [
            {"id": value, "label": labels_by_id.get(value, value)}
            for value in answer_ids
        ],
        "selected_graph": None,
        "diagnostics": diagnostics,
    }


__all__ = ["TrainGoldPathIndex", "retrieve"]
