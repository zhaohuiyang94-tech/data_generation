"""Source-verified local train-template consensus post-selection gate.

The selector is intentionally independent of the prediction models.  It uses
only already executed graph candidates plus an index built from aligned CWQ
Semantic/Compose/Operator training contracts whose Compose relation multiset
is verified against source-train SPARQL.  Runtime retrieval is sparse BM25 over
an entity-masked question and decomposition; it performs no LLM, embedding, KG
or evaluation-label call.

Thresholds are frozen from the v12 train/calibration audit.  The additional
operator/cardinality guards are generic, but the plural/disjoint guard was
formulated after inspecting a held-out audit error.  That provenance is
preserved in every decision trace and must not be hidden when reporting the
feature as independently validated.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import gzip
import heapq
import json
import math
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Iterable, Sequence

from .contracts import ExecutedGraph
from .ontology import relation_id_from_label
from .template_path_retrieval import mask_entities


INDEX_VERSION = "source-verified-template-index-v1"
SELECTOR_VERSION = "source-verified-template-consensus-production-v1"
EPSILON = 1e-12
TOP_K = 48

# Frozen v12 train/calibration lanes.  Values are structural confidence
# thresholds, never entity/relation/question allow-lists.
SUBSET_SCORE_GAP = 0.07
SUBSET_VOTE_COUNT = 40
DISJOINT_SCORE_GAP = 0.20
DISJOINT_VOTE = 0.20
DISJOINT_RELATION_SUPPORT = 0.30

_TOKEN_RE = re.compile(r"<entity>|[a-z0-9]+", re.I)
_VARIABLE_RE = re.compile(r"^(?:V\d+|P\d+\.V\d+)$", re.I)
_SPARQL_TRIPLE_RE = re.compile(
    r"(?:\?[A-Za-z_][A-Za-z0-9_]*|ns:[mg]\.[A-Za-z0-9_]+)\s+"
    r"ns:([A-Za-z0-9_.]+)\s+"
)
_INDEXES: dict[str, "VerifiedTemplateIndex"] = {}
_INDEXES_LOCK = Lock()


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text))]


def _parse_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_array(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"template consensus source must be a JSON array: {path}")
    return [row for row in value if isinstance(row, dict)]


def _operator_signature(value: Any) -> Counter[str]:
    if isinstance(value, dict) and "output" in value:
        value = _parse_object(value.get("output"))
    operators = value.get("operators", []) if isinstance(value, dict) else []
    return Counter(
        str(operator.get("type", "")).upper()
        for operator in operators
        if isinstance(operator, dict) and str(operator.get("type", "")).strip()
    )


def _multiset_f1(left: Counter[str], right: Counter[str]) -> float:
    left_size, right_size = sum(left.values()), sum(right.values())
    if not left_size or not right_size:
        return float(left_size == right_size)
    return (2.0 * sum((left & right).values())) / (left_size + right_size)


@dataclass(frozen=True, slots=True)
class GraphFeatures:
    triples: tuple[tuple[str, str, str], ...]
    answer_var: str
    relations: Counter[str]
    answer_incidence: Counter[str]
    paths: tuple[tuple[tuple[str, str], ...], ...]
    abstract_paths: tuple[tuple[str, ...], ...]


def _is_variable(value: Any) -> bool:
    return bool(_VARIABLE_RE.fullmatch(str(value)))


def _graph_features(
    triples: Iterable[Sequence[Any]], answer_var: Any
) -> GraphFeatures:
    normalized = tuple(
        (str(triple[0]), str(triple[1]), str(triple[2]))
        for triple in triples
        if isinstance(triple, (list, tuple))
        and len(triple) == 3
        and str(triple[1])
    )
    answer = str(answer_var)
    relations = Counter(relation for _, relation, _ in normalized)
    answer_incidence = Counter(
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
        found: tuple[tuple[str, str], ...] | None = None
        while queue:
            node, path = queue.popleft()
            if node == answer:
                found = path
                break
            for neighbour, relation, direction in sorted(adjacency.get(node, [])):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                queue.append((neighbour, (*path, (relation, direction))))
        if found is not None:
            paths.append(found)
    ordered_paths = tuple(sorted(paths))
    return GraphFeatures(
        triples=normalized,
        answer_var=answer,
        relations=relations,
        answer_incidence=answer_incidence,
        paths=ordered_paths,
        abstract_paths=tuple(
            sorted(tuple(direction for _, direction in path) for path in paths)
        ),
    )


def _path_similarity(
    left: tuple[tuple[str, str], ...], right: tuple[tuple[str, str], ...]
) -> float:
    if not left or not right:
        return float(left == right)
    aligned = sum(a == b for a, b in zip(left, right)) / max(len(left), len(right))
    relation = _set_f1((value[0] for value in left), (value[0] for value in right))
    direction = sum(a[1] == b[1] for a, b in zip(left, right)) / max(
        len(left), len(right)
    )
    return 0.55 * aligned + 0.25 * relation + 0.20 * direction


def _set_f1(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return float(not a and not b)
    return (2.0 * len(a & b)) / (len(a) + len(b))


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


def _direction_path_similarity(
    left: tuple[tuple[str, ...], ...], right: tuple[tuple[str, ...], ...]
) -> float:
    if not left or not right:
        return float(left == right)

    def one(source: Sequence[str], target: Sequence[str]) -> float:
        if not source or not target:
            return float(not source and not target)
        return sum(a == b for a, b in zip(source, target)) / max(
            len(source), len(target)
        )

    def directed(source: Any, target: Any) -> float:
        return sum(max(one(path, other) for other in target) for path in source) / len(
            source
        )

    return 0.5 * (directed(left, right) + directed(right, left))


@dataclass(slots=True)
class TemplateDocument:
    source_index: int
    question: str
    tokens: list[str]
    graph: GraphFeatures
    operators: Counter[str]


def _source_manifest(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        stat = path.stat()
        manifest.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return manifest


def _compose_triples(output: dict[str, Any]) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for triple in output.get("triples", []):
        if not isinstance(triple, dict):
            continue
        relation = relation_id_from_label(triple.get("relation_label", []))
        if relation:
            triples.append(
                (
                    str(triple.get("subject", "")),
                    relation,
                    str(triple.get("object", "")),
                )
            )
    return triples


def build_verified_template_cache(
    *,
    semantic_train: str | Path,
    compose_train: str | Path,
    operator_train: str | Path,
    source_train: str | Path,
    cache_path: str | Path,
) -> dict[str, Any]:
    """Build and atomically persist the source-verified local index."""
    paths = (semantic_train, compose_train, operator_train, source_train)
    semantic_rows = _load_array(semantic_train)
    compose_rows = _load_array(compose_train)
    operator_rows = _load_array(operator_train)
    source_rows = _load_array(source_train)
    source_graphs: dict[str, list[Counter[str]]] = defaultdict(list)
    for row in source_rows:
        source_graphs[str(row.get("question", ""))].append(
            Counter(_SPARQL_TRIPLE_RE.findall(str(row.get("sparql", ""))))
        )
    documents: list[dict[str, Any]] = []
    audit: Counter[str] = Counter()
    for source_index, (semantic, compose, operator) in enumerate(
        zip(semantic_rows, compose_rows, operator_rows)
    ):
        semantic_input = _parse_object(semantic.get("input"))
        semantic_output = _parse_object(semantic.get("output"))
        compose_input = _parse_object(compose.get("input"))
        operator_input = _parse_object(operator.get("input"))
        question = str(semantic_input.get("question", ""))
        if not question or question != str(compose_input.get("question", "")):
            audit["semantic_compose_alignment_rejected"] += 1
            continue
        if question != str(operator_input.get("question", "")):
            audit["operator_alignment_rejected"] += 1
            continue
        compose_output = _parse_object(compose.get("output"))
        triples = _compose_triples(compose_output)
        relations = Counter(relation for _, relation, _ in triples)
        sources = source_graphs.get(question, [])
        if not sources:
            audit["source_question_missing"] += 1
            continue
        if not any(not (relations - source) for source in sources):
            audit["source_relation_verification_rejected"] += 1
            continue
        if not triples:
            audit["empty_compose_graph_rejected"] += 1
            continue
        decomposition = [
            str(value)
            for value in semantic_input.get("decomposition", [])
            if str(value).strip()
        ]
        surfaces = [
            str(anchor.get("surface", ""))
            for anchor in semantic_output.get("anchors", [])
            if isinstance(anchor, dict) and str(anchor.get("surface", "")).strip()
        ]
        text = " ".join(
            [
                mask_entities(question, surfaces),
                *(mask_entities(value, surfaces) for value in decomposition),
            ]
        )
        tokens = _tokens(text)
        if not tokens:
            audit["empty_document_rejected"] += 1
            continue
        documents.append(
            {
                "source_index": source_index,
                "question": question,
                "tokens": tokens,
                "triples": triples,
                "answer_var": str(compose_output.get("answer_var", "")),
                "operators": list(
                    sorted(_operator_signature(operator).elements())
                ),
            }
        )
        audit["source_verified_documents"] += 1
    payload = {
        "version": INDEX_VERSION,
        "source_manifest": _source_manifest(paths),
        "audit": dict(audit),
        "documents": documents,
    }
    target = Path(cache_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(target)
    return payload


class VerifiedTemplateIndex:
    def __init__(self, payload: dict[str, Any], *, cache_hit: bool) -> None:
        self.cache_hit = bool(cache_hit)
        self.audit = dict(payload.get("audit", {}))
        self.documents: list[TemplateDocument] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        for raw in payload.get("documents", []):
            if not isinstance(raw, dict):
                continue
            tokens = [str(value) for value in raw.get("tokens", [])]
            graph = _graph_features(
                raw.get("triples", []), raw.get("answer_var", "")
            )
            if not tokens or not graph.triples:
                continue
            document_index = len(self.documents)
            document = TemplateDocument(
                source_index=int(raw.get("source_index", -1)),
                question=str(raw.get("question", "")),
                tokens=tokens,
                graph=graph,
                operators=Counter(map(str, raw.get("operators", []))),
            )
            self.documents.append(document)
            frequencies = Counter(tokens)
            document_frequency.update(frequencies)
            for token, frequency in frequencies.items():
                self.postings[token].append((document_index, frequency))
        count = max(1, len(self.documents))
        self.average_length = sum(len(item.tokens) for item in self.documents) / count
        self.idf = {
            token: math.log(1.0 + ((count - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in document_frequency.items()
        }
        self.max_posting_length = max(64, int(len(self.documents) * 0.18))

    @classmethod
    def load_or_build(
        cls,
        *,
        semantic_train: str | Path,
        compose_train: str | Path,
        operator_train: str | Path,
        source_train: str | Path,
        cache_path: str | Path,
    ) -> "VerifiedTemplateIndex":
        target = Path(cache_path).expanduser().resolve()
        key = str(target)
        with _INDEXES_LOCK:
            cached_index = _INDEXES.get(key)
            if cached_index is not None:
                return cached_index
            expected_manifest = _source_manifest(
                (semantic_train, compose_train, operator_train, source_train)
            )
            payload: dict[str, Any] | None = None
            if target.is_file():
                try:
                    with gzip.open(target, "rt", encoding="utf-8") as handle:
                        candidate = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    candidate = None
                if (
                    isinstance(candidate, dict)
                    and candidate.get("version") == INDEX_VERSION
                    and candidate.get("source_manifest") == expected_manifest
                ):
                    payload = candidate
            cache_hit = payload is not None
            if payload is None:
                payload = build_verified_template_cache(
                    semantic_train=semantic_train,
                    compose_train=compose_train,
                    operator_train=operator_train,
                    source_train=source_train,
                    cache_path=target,
                )
            index = cls(payload, cache_hit=cache_hit)
            _INDEXES[key] = index
            return index

    def retrieve(
        self,
        question: str,
        decompositions: Sequence[str],
        surfaces: Sequence[str],
        *,
        top_k: int = TOP_K,
    ) -> list[tuple[float, TemplateDocument]]:
        text = " ".join(
            [
                mask_entities(question, surfaces),
                *(mask_entities(value, surfaces) for value in decompositions),
            ]
        )
        query = Counter(_tokens(text))
        informative = sorted(
            (
                (self.idf.get(token, 0.0), token, frequency)
                for token, frequency in query.items()
                if len(self.postings.get(token, ())) <= self.max_posting_length
            ),
            reverse=True,
        )[:28]
        scores: dict[int, float] = defaultdict(float)
        for _, token, query_frequency in informative:
            inverse_frequency = self.idf.get(token, 0.0)
            for index, frequency in self.postings.get(token, ()):
                document = self.documents[index]
                normalization = 0.25 + 0.75 * len(document.tokens) / max(
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
                max(1, int(top_k)),
                ((score, index) for index, score in scores.items()),
            )
        ]


@dataclass(slots=True)
class _Candidate:
    execution: ExecutedGraph
    graph: GraphFeatures
    operators: Counter[str]


def _component_similarity(
    candidate: _Candidate, document: TemplateDocument
) -> tuple[float, float, float, float, float, float]:
    relation = _multiset_f1(candidate.graph.relations, document.graph.relations)
    incidence = _multiset_f1(
        candidate.graph.answer_incidence, document.graph.answer_incidence
    )
    path = _path_set_similarity(candidate.graph.paths, document.graph.paths)
    direction = _direction_path_similarity(
        candidate.graph.abstract_paths, document.graph.abstract_paths
    )
    operators = _multiset_f1(candidate.operators, document.operators)
    total = (
        0.34 * relation
        + 0.24 * incidence
        + 0.22 * path
        + 0.14 * direction
        + 0.06 * operators
    )
    return total, relation, incidence, path, direction, operators


def _answer_relation(
    challenger: Sequence[str], incumbent: Sequence[str]
) -> str:
    left, right = set(map(str, challenger)), set(map(str, incumbent))
    if left == right:
        return "equal"
    if left < right:
        return "subset"
    if left > right:
        return "superset"
    if left & right:
        return "overlap"
    return "disjoint"


def _morphological_plural_request(question: str) -> bool:
    tokens = re.findall(r"[a-z]+", str(question).casefold())
    try:
        start = next(
            index for index, token in enumerate(tokens[:5]) if token in {"which", "what"}
        )
    except StopIteration:
        return False
    return any(
        len(token) > 3 and token.endswith("s") and not token.endswith("ss")
        for token in tokens[start + 1 : start + 3]
    )


def _hard_gate_already_selected(decision: dict[str, Any]) -> bool:
    return any(
        str(reason).startswith("hard_")
        or str(reason) == "path_alignment_gate"
        for reason in decision.get("reason_codes", [])
    )


class SourceVerifiedTemplateConsensusSelector:
    """Frozen, Gold-free KNN consensus over already executed candidates."""

    def __init__(self, index: VerifiedTemplateIndex, *, top_k: int = TOP_K) -> None:
        self.index = index
        self.top_k = max(1, min(TOP_K, int(top_k)))

    @classmethod
    def load_or_build(
        cls,
        *,
        semantic_train: str | Path,
        compose_train: str | Path,
        operator_train: str | Path,
        source_train: str | Path,
        cache_path: str | Path,
        top_k: int = TOP_K,
    ) -> "SourceVerifiedTemplateConsensusSelector":
        return cls(
            VerifiedTemplateIndex.load_or_build(
                semantic_train=semantic_train,
                compose_train=compose_train,
                operator_train=operator_train,
                source_train=source_train,
                cache_path=cache_path,
            ),
            top_k=top_k,
        )

    def challenge(
        self,
        *,
        question: str,
        decompositions: Sequence[str],
        anchor_surfaces: Sequence[str],
        selected: ExecutedGraph,
        executions: Sequence[ExecutedGraph],
        prior_decision: dict[str, Any],
    ) -> tuple[ExecutedGraph, dict[str, Any]]:
        started = time.perf_counter()
        evidence: dict[str, Any] = {
            "version": SELECTOR_VERSION,
            "status": "not_applicable",
            "model_calls": 0,
            "embedding_calls": 0,
            "endpoint_queries": 0,
            "uses_evaluation_gold": False,
            "entity_or_relation_whitelist": False,
            "source_verified_template_count": len(self.index.documents),
            "index_cache_hit": self.index.cache_hit,
            "posthoc_risk": (
                "plural_disjoint_cardinality_collapse was formulated after held-out "
                "audit diagnosis and requires fresh shadow validation"
            ),
        }
        if _hard_gate_already_selected(prior_decision):
            evidence.update(
                {
                    "status": "skipped_higher_priority_postselection_gate",
                    "prior_reason_codes": list(prior_decision.get("reason_codes", [])),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        unique: list[ExecutedGraph] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for execution in [*executions, selected]:
            if not execution.answer_ids:
                continue
            key = (
                str(execution.graph.graph_id),
                tuple(sorted(set(map(str, execution.answer_ids)))),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(execution)
        if len(unique) < 2:
            evidence.update(
                {
                    "status": "insufficient_executed_candidates",
                    "candidate_count": len(unique),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        neighbours = self.index.retrieve(
            question,
            decompositions,
            anchor_surfaces,
            top_k=self.top_k,
        )
        if not neighbours:
            evidence.update(
                {
                    "status": "no_verified_template_neighbours",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        candidates = [
            _Candidate(
                execution=execution,
                graph=_graph_features(
                    execution.graph.triples, execution.graph.answer_var
                ),
                operators=_operator_signature(
                    {"operators": execution.graph.operators}
                ),
            )
            for execution in unique
        ]
        top_bm25 = max(score for score, _ in neighbours)
        rows: list[tuple[float, TemplateDocument, list[tuple[float, ...]]]] = []
        for rank, (bm25, document) in enumerate(neighbours):
            weight = (bm25 / max(top_bm25, 1e-9)) / math.log2(rank + 2.0)
            rows.append(
                (
                    weight,
                    document,
                    [
                        _component_similarity(candidate, document)
                        for candidate in candidates
                    ],
                )
            )
        denominator = sum(weight for weight, _, _ in rows) or 1.0
        scored: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            component_means = [
                sum(
                    weight * similarities[candidate_index][component]
                    for weight, _, similarities in rows
                )
                / denominator
                for component in range(6)
            ]
            mean = component_means[0]
            vote = 0.0
            vote_count = 0
            relation_support = 0.0
            exact_support = 0.0
            for weight, document, similarities in rows:
                ranked = sorted(
                    (
                        (item[0], index)
                        for index, item in enumerate(similarities)
                    ),
                    reverse=True,
                )
                winner_score = ranked[0][0]
                winners = [
                    index
                    for score, index in ranked
                    if abs(score - winner_score) <= EPSILON
                ]
                if candidate_index in winners:
                    vote += weight / len(winners)
                    vote_count += 1
                relation_support += weight * float(
                    candidate.graph.relations == document.graph.relations
                )
                exact_support += weight * float(
                    candidate.graph.relations == document.graph.relations
                    and candidate.graph.answer_incidence
                    == document.graph.answer_incidence
                    and candidate.graph.abstract_paths
                    == document.graph.abstract_paths
                    and candidate.operators == document.operators
                )
            scored.append(
                {
                    "candidate": candidate,
                    "score": mean,
                    "vote": vote / denominator,
                    "vote_count": vote_count,
                    "relation_support": relation_support / denominator,
                    "exact_support": exact_support / denominator,
                    "component_means": component_means,
                }
            )
        ranking = sorted(
            scored,
            key=lambda item: (
                float(item["score"]),
                float(item["vote"]),
                float(item["exact_support"]),
                float(item["candidate"].execution.graph.score),
                str(item["candidate"].execution.graph.graph_id),
            ),
            reverse=True,
        )
        proposal = ranking[0]
        runner = ranking[1] if len(ranking) > 1 else None
        selected_key = (
            str(selected.graph.graph_id),
            frozenset(map(str, selected.answer_ids)),
        )
        incumbent = next(
            (
                item
                for item in scored
                if (
                    str(item["candidate"].execution.graph.graph_id),
                    frozenset(map(str, item["candidate"].execution.answer_ids)),
                )
                == selected_key
            ),
            None,
        )
        if incumbent is None:
            evidence.update(
                {
                    "status": "incumbent_feature_missing",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        challenger = proposal["candidate"].execution
        relation = _answer_relation(challenger.answer_ids, selected.answer_ids)
        score_gap = float(proposal["score"]) - float(incumbent["score"])
        lane = ""
        if (
            relation == "subset"
            and score_gap + EPSILON >= SUBSET_SCORE_GAP
            and int(proposal["vote_count"]) >= SUBSET_VOTE_COUNT
        ):
            lane = "subset"
        elif (
            relation == "disjoint"
            and score_gap + EPSILON >= DISJOINT_SCORE_GAP
            and float(proposal["vote"]) + EPSILON >= DISJOINT_VOTE
            and float(proposal["relation_support"]) + EPSILON
            >= DISJOINT_RELATION_SUPPORT
        ):
            lane = "disjoint"
        evidence.update(
            {
                "candidate_count": len(candidates),
                "neighbour_count": len(neighbours),
                "source_graph_id": selected.graph.graph_id,
                "proposed_graph_id": challenger.graph.graph_id,
                "proposed_answer_ids": list(map(str, challenger.answer_ids)),
                "answer_relation": relation,
                "source_answer_count": len(set(map(str, selected.answer_ids))),
                "proposed_answer_count": len(set(map(str, challenger.answer_ids))),
                "source_score": float(incumbent["score"]),
                "proposed_score": float(proposal["score"]),
                "score_gap": score_gap,
                "candidate_margin": (
                    float(proposal["score"])
                    - (float(runner["score"]) if runner is not None else 0.0)
                ),
                "vote": float(proposal["vote"]),
                "vote_count": int(proposal["vote_count"]),
                "relation_support": float(proposal["relation_support"]),
                "exact_support": float(proposal["exact_support"]),
                "component_means": {
                    name: float(value)
                    for name, value in zip(
                        (
                            "total",
                            "relation",
                            "answer_incidence",
                            "anchor_answer_path",
                            "anchor_answer_direction",
                            "operator_signature",
                        ),
                        proposal["component_means"],
                    )
                },
                "source_operator_signature": list(
                    sorted(
                        _operator_signature(
                            {"operators": selected.graph.operators}
                        ).elements()
                    )
                ),
                "proposed_operator_signature": list(
                    sorted(
                        _operator_signature(
                            {"operators": challenger.graph.operators}
                        ).elements()
                    )
                ),
                "morphological_plural_request": _morphological_plural_request(
                    question
                ),
                "lane": lane,
                "top_source_template_indexes": [
                    document.source_index for _, document in neighbours[:8]
                ],
            }
        )
        if not lane or challenger is selected:
            evidence.update(
                {
                    "status": "confidence_gate_rejected",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        source_operators = _operator_signature(
            {"operators": selected.graph.operators}
        )
        challenger_operators = _operator_signature(
            {"operators": challenger.graph.operators}
        )
        if lane == "subset" and source_operators != challenger_operators:
            evidence.update(
                {
                    "status": "structural_guard_rejected",
                    "guard_reason": "subset_operator_signature_changed",
                    "source_operator_signature": list(
                        sorted(source_operators.elements())
                    ),
                    "proposed_operator_signature": list(
                        sorted(challenger_operators.elements())
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        if (
            lane == "disjoint"
            and len(set(map(str, challenger.answer_ids)))
            < len(set(map(str, selected.answer_ids)))
            and _morphological_plural_request(question)
        ):
            evidence.update(
                {
                    "status": "structural_guard_rejected",
                    "guard_reason": "plural_disjoint_cardinality_collapse",
                    "morphological_plural_request": True,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        evidence.update(
            {
                "status": "selected_template_consensus_challenger",
                "selected_graph_id": challenger.graph.graph_id,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return challenger, evidence


__all__ = [
    "SourceVerifiedTemplateConsensusSelector",
    "VerifiedTemplateIndex",
    "build_verified_template_cache",
]
