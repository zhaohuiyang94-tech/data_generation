"""Frozen Gold-blind path-alignment post-selection gate.

The gate compares each saved Semantic path intent with natural constant-to-
answer paths in already executed candidate graphs.  It deliberately removes
all V0/V1-like variable spellings before local BGE scoring.  No LLM or KG
endpoint is called, and an earlier ``hard_*`` decision always has first
refusal.

Model coefficients and thresholds are frozen from the v12 train/calibration
audit in ``resources/path_alignment_v12_model.json``.  Gold answers are never
accepted by this module and are not part of its API.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from threading import RLock
import time
from typing import Any, Iterable, Protocol, Sequence

from .contracts import ExecutedGraph
from .ontology import relation_id_from_label, relation_label_from_id


SELECTOR_VERSION = "path-alignment-production-v1"
REASON_CODE = "path_alignment_gate"
EPSILON = 1e-12
_VARIABLE_RE = re.compile(r"^\??(?:V|P\d+\.V|v|x|var)[A-Za-z0-9_.-]*$", re.I)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_MODEL_PATH = Path(__file__).with_name("resources") / "path_alignment_v12_model.json"


class _EmbeddingClient(Protocol):
    def embed(self, texts: list[str], *, input_type: str) -> list[list[float]]: ...


def _load_model() -> dict[str, Any]:
    value = json.loads(_MODEL_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != "path-alignment-v12-frozen-model-v1":
        raise ValueError(f"invalid frozen path-alignment model: {_MODEL_PATH}")
    names = value.get("pairwise_feature_names", [])
    if not isinstance(names, list) or not names:
        raise ValueError("path-alignment model has no feature names")
    for key in ("means", "scales", "weights"):
        if not isinstance(value.get(key), list) or len(value[key]) != len(names):
            raise ValueError(f"path-alignment model {key} has wrong length")
    return value


@dataclass(frozen=True, slots=True)
class SemanticIntent:
    path_id: str
    goal: str
    anchor_surface: str
    anchor_id: str
    relation_ids: tuple[str, ...]
    directions: tuple[str, ...]
    grounded_relations: tuple[str, ...]
    grounded_directions: tuple[str, ...]

    @property
    def query_document(self) -> str:
        relation_path = "; then ".join(
            _direction_phrase(direction, relation)
            for direction, relation in zip(self.directions, self.relation_ids)
        )
        return (
            f"Goal: {self.goal}. Start with the real entity "
            f"{self.anchor_surface or 'the linked entity'}. Intended relationship "
            f"path: {relation_path}. Return the path target."
        )

    @property
    def relation_query_document(self) -> str:
        relation_path = "; then ".join(
            _direction_phrase(direction, relation)
            for direction, relation in zip(self.directions, self.relation_ids)
        )
        return f"Relationship intent for the requested target: {relation_path}."


@dataclass(frozen=True, slots=True)
class NaturalPath:
    start_id: str
    start_label: str
    relations: tuple[str, ...]
    directions: tuple[str, ...]
    terminal_type: str
    continuity: float
    known_type_coverage: float

    @property
    def document(self) -> str:
        relation_path = "; then ".join(
            _direction_phrase(direction, relation)
            for direction, relation in zip(self.directions, self.relations)
        )
        terminal = self.terminal_type.replace(".", " ").replace("_", " ")
        suffix = f" The returned target has type {terminal}." if terminal else ""
        return (
            f"Start with the real entity or literal {self.start_label or self.start_id}. "
            f"Natural relation path: {relation_path}. Return the target.{suffix}"
        )

    @property
    def relation_document(self) -> str:
        relation_path = "; then ".join(
            _direction_phrase(direction, relation)
            for direction, relation in zip(self.directions, self.relations)
        )
        return f"Concrete relationship path to the answer: {relation_path}."


@dataclass(slots=True)
class _Candidate:
    execution: ExecutedGraph
    intents: list[SemanticIntent]
    paths: list[NaturalPath]
    features: list[float] | None = None


class ThreadSafeEmbeddingCache:
    """Bounded single-flight cache around the shared local embedding client."""

    def __init__(self, client: _EmbeddingClient, *, maximum_entries: int = 32768) -> None:
        self.client = client
        self.maximum_entries = max(256, int(maximum_entries))
        self._values: OrderedDict[tuple[str, str], tuple[float, ...]] = OrderedDict()
        self._lock = RLock()

    def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
    ) -> tuple[dict[str, tuple[float, ...]], dict[str, int]]:
        unique = list(dict.fromkeys(str(text) for text in texts if str(text)))
        with self._lock:
            hits = 0
            missing: list[str] = []
            for text in unique:
                key = (input_type, text)
                if key in self._values:
                    hits += 1
                    self._values.move_to_end(key)
                else:
                    missing.append(text)
            calls = 0
            if missing:
                vectors = self.client.embed(missing, input_type=input_type)
                if len(vectors) != len(missing):
                    raise RuntimeError("path-alignment embedding response has wrong length")
                calls = 1
                for text, vector in zip(missing, vectors):
                    self._values[(input_type, text)] = tuple(float(value) for value in vector)
                    self._values.move_to_end((input_type, text))
                while len(self._values) > self.maximum_entries:
                    self._values.popitem(last=False)
            return (
                {
                    text: self._values[(input_type, text)]
                    for text in unique
                    if (input_type, text) in self._values
                },
                {"hits": hits, "misses": len(missing), "calls": calls},
            )


def _tokens(value: Any) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(str(value)))


def _set_f1(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(map(str, left)), set(map(str, right))
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _is_variable(value: Any) -> bool:
    return bool(_VARIABLE_RE.fullmatch(str(value).strip()))


def _human_relation(relation_id: str) -> str:
    parts = relation_label_from_id(str(relation_id))
    if len(parts) <= 1:
        return parts[0] if parts else str(relation_id)
    return f"{parts[-1]} ({' / '.join(parts[:-1])})"


def _direction_phrase(direction: str, relation_id: str) -> str:
    relation = _human_relation(relation_id)
    if str(direction).casefold() == "backward":
        return f"find the subject whose {relation} points to the current item"
    return f"follow {relation} from subject to object"


def _relation_leaf_tokens(relations: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        token
        for relation in relations
        for token in _tokens(str(relation).split(".")[-1].replace("_", " "))
    )


def _operator_signature(execution: ExecutedGraph) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(item.get("type", "")).upper()
            for item in execution.graph.operators
            if isinstance(item, dict) and str(item.get("type", "")).strip()
        )
    )


def _question_operator_flags(question: str) -> tuple[int, int, int, int, int]:
    text = " ".join(str(question).casefold().split())
    return (
        int(bool(re.search(r"\b(latest|last|most recent|highest|largest|longest|newest|maximum)\b", text))),
        int(bool(re.search(r"\b(earliest|first|lowest|smallest|shortest|oldest|youngest|minimum)\b", text))),
        int(bool(re.search(r"\b(at least|more than|greater than|over|above|after|later than)\b", text))),
        int(bool(re.search(r"\b(at most|less than|fewer than|under|below|before|prior to)\b", text))),
        int(bool(re.search(r"\b(equal|exactly|same as)\b", text))),
    )


def _operator_compatibility(question: str, execution: ExecutedGraph) -> float:
    max_q, min_q, high_q, low_q, equal_q = _question_operator_flags(question)
    values = set(_operator_signature(execution))
    requirements = (
        (max_q, "ARGMAX" in values),
        (min_q, "ARGMIN" in values),
        (high_q, "GREATER_THAN" in values),
        (low_q, "LESS_THAN" in values),
        (equal_q, "EQUAL" in values),
    )
    active = [float(present) for requested, present in requirements if requested]
    if active:
        return sum(active) / len(active)
    substantive = values & {"ARGMAX", "ARGMIN", "GREATER_THAN", "LESS_THAN", "EQUAL"}
    return 1.0 if not substantive else 0.0


def _binding_value(binding: Any, field: str) -> str:
    if isinstance(binding, dict):
        return str(binding.get(field, "")).strip()
    return str(getattr(binding, field, "")).strip()


def _grounded_hops(execution: ExecutedGraph, path_index: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    grounded = execution.graph.provenance.get("grounded_semantic", {})
    hops = grounded.get("hops", []) if isinstance(grounded, dict) else []
    path_hops = hops[path_index] if path_index < len(hops) and isinstance(hops[path_index], list) else []
    relations: list[str] = []
    directions: list[str] = []
    for hop in path_hops:
        if not isinstance(hop, dict):
            continue
        relation = str(hop.get("relation_id", ""))
        direction = str(hop.get("direction", "")).casefold()
        if relation:
            relations.append(relation)
            directions.append(direction if direction in {"forward", "backward"} else "forward")
        second = str(hop.get("second_relation_id", ""))
        if second and second not in relations:
            relations.append(second)
            direction = str(hop.get("second_direction", "")).casefold()
            directions.append(direction if direction in {"forward", "backward"} else "forward")
    return tuple(relations), tuple(directions)


def _semantic_intents(
    execution: ExecutedGraph,
    semantic_graphs: Sequence[dict[str, Any]],
) -> list[SemanticIntent]:
    grounded = execution.graph.provenance.get("grounded_semantic", {})
    semantic_index = grounded.get("semantic_graph_index") if isinstance(grounded, dict) else None
    if not isinstance(semantic_index, int) or not (0 <= semantic_index < len(semantic_graphs)):
        return []
    semantic = semantic_graphs[semantic_index]
    anchors = {
        str(item.get("id", "")): item
        for item in semantic.get("anchors", [])
        if isinstance(item, dict)
    }
    bindings = execution.graph.provenance.get("anchor_bindings", {})
    result: list[SemanticIntent] = []
    for path_index, path in enumerate(semantic.get("semantic_paths", [])):
        if not isinstance(path, dict):
            continue
        anchor_ref = str(path.get("anchor_ref", ""))
        anchor = anchors.get(anchor_ref, {})
        binding = bindings.get(anchor_ref, {}) if isinstance(bindings, dict) else {}
        relation_ids: list[str] = []
        directions: list[str] = []
        for step in path.get("steps", []):
            if not isinstance(step, dict):
                continue
            relation = relation_id_from_label(step.get("relation_label", []))
            if not relation:
                continue
            relation_ids.append(relation)
            direction = str(step.get("direction", "forward")).casefold()
            directions.append(direction if direction in {"forward", "backward"} else "forward")
        if not relation_ids:
            continue
        grounded_relations, grounded_directions = _grounded_hops(execution, path_index)
        result.append(
            SemanticIntent(
                path_id=str(path.get("id", f"P{path_index}")),
                goal=str(path.get("goal", "")).strip(),
                anchor_surface=str(anchor.get("surface") or _binding_value(binding, "label")).strip(),
                anchor_id=_binding_value(binding, "id"),
                relation_ids=tuple(relation_ids),
                directions=tuple(directions),
                grounded_relations=grounded_relations,
                grounded_directions=grounded_directions,
            )
        )
    return result


def _type_compatible(left: str, right: str, ontology: Any) -> bool | None:
    if not left or not right:
        return None
    if left == right:
        return True
    return bool(set(ontology.supertypes(left)) & set(ontology.supertypes(right)))


def _path_types(
    relations: Sequence[str], directions: Sequence[str], ontology: Any
) -> tuple[str, float, float]:
    inputs: list[str] = []
    outputs: list[str] = []
    known_steps = 0
    for relation, direction in zip(relations, directions):
        domain = str(ontology.domain_for_relation(relation))
        range_ = str(ontology.range_for_relation(relation))
        input_type, output_type = (range_, domain) if direction == "backward" else (domain, range_)
        inputs.append(input_type)
        outputs.append(output_type)
        known_steps += int(bool(input_type and output_type))
    comparisons = [
        _type_compatible(outputs[index], inputs[index + 1], ontology)
        for index in range(max(0, len(outputs) - 1))
    ]
    known = [value for value in comparisons if value is not None]
    continuity = sum(bool(value) for value in known) / len(known) if known else 1.0
    return outputs[-1] if outputs else "", continuity, known_steps / max(1, len(relations))


def _natural_paths(
    execution: ExecutedGraph,
    ontology: Any,
    *,
    maximum_hops: int = 7,
) -> list[NaturalPath]:
    triples = [tuple(map(str, triple)) for triple in execution.graph.triples if len(triple) == 3]
    answer = str(execution.graph.answer_var)
    if not triples or not answer:
        return []
    adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    nodes: set[str] = set()
    for subject, relation, object_ in triples:
        nodes.update((subject, object_))
        adjacency[subject].append((object_, relation, "forward"))
        adjacency[object_].append((subject, relation, "backward"))
    labels: dict[str, str] = {}
    bindings = execution.graph.provenance.get("anchor_bindings", {})
    if isinstance(bindings, dict):
        for binding in bindings.values():
            entity_id = _binding_value(binding, "id")
            label = _binding_value(binding, "label")
            if entity_id and label:
                labels[entity_id] = label
    result: list[NaturalPath] = []
    signatures: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    for start in sorted(node for node in nodes if not _is_variable(node)):
        queue: deque[tuple[str, tuple[str, ...], tuple[str, ...], frozenset[str]]] = deque(
            [(start, (), (), frozenset({start}))]
        )
        best_depth: int | None = None
        found = 0
        while queue and found < 4:
            node, relations, directions, visited = queue.popleft()
            if best_depth is not None and len(relations) > best_depth + 1:
                break
            if node == answer and relations:
                best_depth = len(relations) if best_depth is None else best_depth
                signature = (start, relations, directions)
                if signature in signatures:
                    continue
                signatures.add(signature)
                terminal, continuity, coverage = _path_types(relations, directions, ontology)
                result.append(
                    NaturalPath(
                        start_id=start,
                        start_label=labels.get(start, start),
                        relations=relations,
                        directions=directions,
                        terminal_type=terminal,
                        continuity=continuity,
                        known_type_coverage=coverage,
                    )
                )
                found += 1
                continue
            if len(relations) >= maximum_hops:
                continue
            for neighbour, relation, direction in adjacency.get(node, []):
                if neighbour in visited:
                    continue
                queue.append((neighbour, (*relations, relation), (*directions, direction), visited | {neighbour}))
    return result


def _expected_terminal_type(intent: SemanticIntent, ontology: Any) -> str:
    if not intent.relation_ids:
        return ""
    relation, direction = intent.relation_ids[-1], intent.directions[-1]
    return str(
        ontology.domain_for_relation(relation)
        if direction == "backward"
        else ontology.range_for_relation(relation)
    )


def _pair_features(
    intent: SemanticIntent,
    path: NaturalPath,
    queries: dict[str, tuple[float, ...]],
    documents: dict[str, tuple[float, ...]],
    ontology: Any,
) -> tuple[float, ...]:
    semantic_pairs = tuple(zip(intent.relation_ids, intent.directions))
    natural_pairs = tuple(zip(path.relations, path.directions))
    grounded_pairs = tuple(zip(intent.grounded_relations, intent.grounded_directions))
    compatible = _type_compatible(_expected_terminal_type(intent, ontology), path.terminal_type, ontology)
    return (
        _cosine(queries.get(intent.query_document, ()), documents.get(path.document, ())),
        _cosine(queries.get(intent.relation_query_document, ()), documents.get(path.relation_document, ())),
        _set_f1(_relation_leaf_tokens(intent.relation_ids), _relation_leaf_tokens(path.relations)),
        _set_f1(intent.relation_ids, path.relations),
        _set_f1(semantic_pairs, natural_pairs),
        _set_f1(intent.grounded_relations, path.relations),
        _set_f1(grounded_pairs, natural_pairs),
        float(bool(intent.anchor_id) and intent.anchor_id == path.start_id),
        _set_f1(_tokens(intent.anchor_surface), _tokens(path.start_label)),
        0.5 if compatible is None else float(compatible),
        path.continuity,
        path.known_type_coverage,
        min(len(intent.relation_ids), len(path.relations))
        / max(1, max(len(intent.relation_ids), len(path.relations))),
    )


def _pair_alignment(values: Sequence[float]) -> float:
    weights = (0.40, 0.19, 0.09, 0.05, 0.04, 0.04, 0.03, 0.05, 0.03, 0.04, 0.02, 0.01, 0.01)
    return sum(weight * value for weight, value in zip(weights, values))


def _optimistic_structural_profile(
    candidate: _Candidate,
    ontology: Any,
) -> tuple[float, float, float, float, float]:
    """BGE-free upper profile used only as a necessary-condition filter.

    Each intent may choose its best natural path independently for each
    structural signal.  This is intentionally optimistic: rejecting a row
    means even that upper profile cannot clear the frozen structural margin.
    """
    if not candidate.intents or len(candidate.paths) < len(candidate.intents):
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    feature_indexes = (2, 3, 4, 9, 12)
    totals = [0.0] * len(feature_indexes)
    for intent in candidate.intents:
        pairs = [
            _pair_features(intent, path, {}, {}, ontology)
            for path in candidate.paths
        ]
        for output_index, feature_index in enumerate(feature_indexes):
            totals[output_index] += max(values[feature_index] for values in pairs)
    return tuple(value / len(candidate.intents) for value in totals)


def _cheap_structural_prefilter(
    candidates: Sequence[_Candidate],
    incumbent_index: int,
    eligible: Sequence[int],
    ontology: Any,
    *,
    rule_score_floor: float,
    structural_gain: float = 0.20,
) -> tuple[bool, dict[str, Any]]:
    """Reject obvious non-contenders before any embedding request."""
    incumbent = candidates[incumbent_index]
    incumbent_profile = _optimistic_structural_profile(incumbent, ontology)
    contenders: list[dict[str, Any]] = []
    for index in eligible:
        candidate = candidates[index]
        rule_delta = (
            float(candidate.execution.graph.score)
            - float(incumbent.execution.graph.score)
        )
        if rule_delta + EPSILON < rule_score_floor:
            continue
        profile = _optimistic_structural_profile(candidate, ontology)
        maximum_gain = max(
            (left - right for left, right in zip(profile, incumbent_profile)),
            default=0.0,
        )
        if maximum_gain + EPSILON < structural_gain:
            continue
        contenders.append(
            {
                "graph_id": candidate.execution.graph.graph_id,
                "rule_score_delta": rule_delta,
                "maximum_structural_gain": maximum_gain,
            }
        )
    return bool(contenders), {
        "version": "path-alignment-cheap-prefilter-v1",
        "rule_score_floor": rule_score_floor,
        "minimum_optimistic_structural_gain": structural_gain,
        "signals": [
            "semantic_relation_leaf",
            "semantic_exact_relation",
            "semantic_direction",
            "terminal_ontology_type",
            "hop_length",
        ],
        "eligible_candidate_count": len(eligible),
        "contender_count": len(contenders),
        "contenders": contenders,
    }


def _maximum_matching(matrix: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    if not matrix or not matrix[0]:
        return []
    rows, columns = len(matrix), len(matrix[0])
    if columns > 14:
        retained = sorted(
            range(columns),
            key=lambda column: max(matrix[row][column] for row in range(rows)),
            reverse=True,
        )[:14]
        reduced = [[row[column] for column in retained] for row in matrix]
        return [(row, retained[column]) for row, column in _maximum_matching(reduced)]
    states: dict[int, tuple[float, tuple[tuple[int, int], ...]]] = {0: (0.0, ())}
    for row in range(rows):
        next_states = dict(states)
        for mask, (score, assignments) in states.items():
            for column in range(columns):
                bit = 1 << column
                if mask & bit:
                    continue
                proposed = score + matrix[row][column]
                previous = next_states.get(mask | bit)
                if previous is None or proposed > previous[0] + EPSILON:
                    next_states[mask | bit] = (proposed, (*assignments, (row, column)))
        states = next_states
    return list(max(states.values(), key=lambda item: (item[0], len(item[1])))[1])


def _candidate_features(
    question: str,
    candidate: _Candidate,
    queries: dict[str, tuple[float, ...]],
    documents: dict[str, tuple[float, ...]],
    ontology: Any,
) -> list[float]:
    pairs = [
        [_pair_features(intent, path, queries, documents, ontology) for path in candidate.paths]
        for intent in candidate.intents
    ]
    alignment_matrix = [[_pair_alignment(values) for values in row] for row in pairs]
    matching = _maximum_matching(alignment_matrix)
    matched = [pairs[row][column] for row, column in matching]
    alignments = [alignment_matrix[row][column] for row, column in matching]
    intent_count = len(candidate.intents)

    def mean(index: int) -> float:
        return sum(values[index] for values in matched) / max(1, intent_count)

    execution = candidate.execution
    return [
        sum(alignments) / max(1, intent_count),
        min(alignments) if len(alignments) == intent_count and alignments else 0.0,
        max(alignments, default=0.0),
        *(mean(index) for index in range(13)),
        len(matching) / max(1, intent_count),
        sum(score >= 0.65 for score in alignments) / max(1, intent_count),
        sum(score >= 0.75 for score in alignments) / max(1, intent_count),
        sum(score >= 0.82 for score in alignments) / max(1, intent_count),
        min(len(candidate.intents), len(candidate.paths))
        / max(1, max(len(candidate.intents), len(candidate.paths))),
        float(execution.graph.score),
        _operator_compatibility(question, execution),
        math.log1p(len(set(map(str, execution.answer_ids)))),
        math.log1p(len(execution.graph.triples)),
    ]


def _pairwise_features(
    candidates: Sequence[_Candidate], incumbent_index: int, candidate_index: int
) -> list[float]:
    candidate = candidates[candidate_index].features or []
    incumbent = candidates[incumbent_index].features or []
    top_alignment = max((item.features or [0.0])[0] for item in candidates)
    top_rule = max((item.features or [0.0] * 22)[21] for item in candidates)
    return [
        *candidate,
        *(left - right for left, right in zip(candidate, incumbent)),
        candidate[0] - top_alignment,
        candidate[21] - top_rule,
        math.log1p(len(candidates)),
    ]


def _probability(values: Sequence[float], model: dict[str, Any]) -> float:
    normalized = [
        (value - float(mean)) / float(scale)
        for value, mean, scale in zip(values, model["means"], model["scales"])
    ]
    raw = max(
        -30.0,
        min(
            30.0,
            float(model["bias"])
            + sum(float(weight) * value for weight, value in zip(model["weights"], normalized)),
        ),
    )
    return 1.0 / (1.0 + math.exp(-raw))


def _has_hard_reason(decision: dict[str, Any]) -> bool:
    return any(str(reason).startswith("hard_") for reason in decision.get("reason_codes", []))


class PathAlignmentSelector:
    """Frozen, local-BGE challenger over already executed graph candidates."""

    def __init__(
        self,
        embedding_client: _EmbeddingClient,
        ontology: Any,
        *,
        complete_answer_limit: int = 100,
        embedding_cache_size: int = 1024,
    ) -> None:
        self.ontology = ontology
        self.complete_answer_limit = max(1, int(complete_answer_limit))
        self.model = _load_model()
        self.cache = ThreadSafeEmbeddingCache(
            embedding_client,
            maximum_entries=embedding_cache_size,
        )

    def challenge(
        self,
        *,
        question: str,
        semantic_graphs: Sequence[dict[str, Any]],
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
            "entity_or_relation_whitelist": False,
            "variable_names_sent_to_embedding": False,
            "frozen_model_version": self.model["version"],
            "frozen_gate": dict(self.model["gate"]),
        }
        if _has_hard_reason(prior_decision):
            evidence.update(
                {
                    "status": "skipped_existing_hard_gate",
                    "prior_reason_codes": list(prior_decision.get("reason_codes", [])),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        if self.ontology is None or not semantic_graphs:
            evidence.update(
                {
                    "status": "missing_ontology_or_semantic_graphs",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        unique: list[ExecutedGraph] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for execution in executions:
            if not execution.answer_ids or len(execution.answer_ids) > self.complete_answer_limit:
                continue
            key = (
                str(execution.graph.graph_id),
                tuple(str(answer_id) for answer_id in execution.answer_ids),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(execution)
        selected_answers = set(map(str, selected.answer_ids))
        incumbent_indexes = [
            index
            for index, execution in enumerate(unique)
            if set(map(str, execution.answer_ids)) == selected_answers
        ]
        if not incumbent_indexes:
            evidence.update(
                {
                    "status": "incumbent_not_materialized_as_complete_candidate",
                    "candidate_count": len(unique),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        incumbent_index = next(
            (
                index
                for index in incumbent_indexes
                if unique[index].graph.graph_id == selected.graph.graph_id
            ),
            incumbent_indexes[0],
        )
        candidates = [
            _Candidate(
                execution=execution,
                intents=_semantic_intents(execution, semantic_graphs),
                paths=_natural_paths(execution, self.ontology),
            )
            for execution in unique
        ]
        eligible = [
            index
            for index, candidate in enumerate(candidates)
            if index != incumbent_index
            and set(map(str, candidate.execution.answer_ids)) != selected_answers
            and candidate.intents
            and candidate.paths
        ]
        if not eligible:
            evidence.update(
                {
                    "status": "no_distinct_path_aligned_challenger",
                    "candidate_count": len(candidates),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        prefilter_passed, prefilter = _cheap_structural_prefilter(
            candidates,
            incumbent_index,
            eligible,
            self.ontology,
            rule_score_floor=float(
                self.model["gate"]["minimum_rule_score_delta"]
            ),
        )
        evidence["cheap_prefilter"] = prefilter
        if not prefilter_passed:
            evidence.update(
                {
                    "status": "cheap_structural_prefilter_rejected",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        query_texts = [
            text
            for candidate in candidates
            for intent in candidate.intents
            for text in (intent.query_document, intent.relation_query_document)
        ]
        document_texts = [
            text
            for candidate in candidates
            for path in candidate.paths
            for text in (path.document, path.relation_document)
        ]
        try:
            queries, query_stats = self.cache.embed(query_texts, input_type="query")
            documents, document_stats = self.cache.embed(document_texts, input_type="document")
        except Exception as exc:
            evidence.update(
                {
                    "status": "embedding_unavailable",
                    "embedding_error": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        evidence.update(
            {
                "embedding_calls": query_stats["calls"] + document_stats["calls"],
                "embedding_cache_hits": query_stats["hits"] + document_stats["hits"],
                "embedding_cache_misses": query_stats["misses"] + document_stats["misses"],
            }
        )
        for candidate in candidates:
            candidate.features = _candidate_features(
                question,
                candidate,
                queries,
                documents,
                self.ontology,
            )
        probabilities = {
            index: _probability(
                _pairwise_features(candidates, incumbent_index, index),
                self.model,
            )
            for index in eligible
        }
        order = sorted(
            eligible,
            key=lambda index: (
                -probabilities[index],
                -(candidates[index].features or [0.0])[0],
                -float(candidates[index].execution.graph.score),
                str(candidates[index].execution.graph.graph_id),
            ),
        )
        chosen = order[0]
        chosen_features = candidates[chosen].features or []
        incumbent_features = candidates[incumbent_index].features or []
        alignment_values = sorted(
            (candidates[index].features or [0.0])[0]
            for index in [*eligible, incumbent_index]
        )
        top_alignment = alignment_values[-1]
        second_alignment = alignment_values[-2] if len(alignment_values) > 1 else 0.0
        alignment_gap = (
            chosen_features[0] - second_alignment
            if chosen_features[0] >= top_alignment - EPSILON
            else chosen_features[0] - top_alignment
        )
        proposal = {
            "graph_id": candidates[chosen].execution.graph.graph_id,
            "source_graph_id": selected.graph.graph_id,
            "probability": probabilities[chosen],
            "probability_margin": probabilities[chosen]
            - (probabilities[order[1]] if len(order) > 1 else 0.0),
            "alignment_delta": chosen_features[0] - incumbent_features[0],
            "alignment_gap": alignment_gap,
            "coverage": chosen_features[16],
            "rule_score_delta": chosen_features[21] - incumbent_features[21],
            "candidate_count": len(candidates),
            "challenger_answer_count": len(candidates[chosen].execution.answer_ids),
            "incumbent_answer_count": len(selected.answer_ids),
        }
        gate = self.model["gate"]
        passed = bool(
            proposal["probability"] + EPSILON >= float(gate["probability"])
            and proposal["probability_margin"] + EPSILON >= float(gate["probability_margin"])
            and proposal["alignment_delta"] + EPSILON >= float(gate["alignment_delta"])
            and proposal["alignment_gap"] + EPSILON >= float(gate["alignment_gap"])
            and proposal["coverage"] + EPSILON >= float(gate["required_coverage"])
            and proposal["rule_score_delta"] + EPSILON
            >= float(gate["minimum_rule_score_delta"])
        )
        evidence.update(
            {
                "status": "applied" if passed else "confidence_gate_rejected",
                "reason": "frozen_path_alignment_confidence"
                if passed
                else "frozen_thresholds_not_met",
                "proposal": proposal,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return (candidates[chosen].execution if passed else selected), evidence


__all__ = [
    "PathAlignmentSelector",
    "REASON_CODE",
    "SELECTOR_VERSION",
    "ThreadSafeEmbeddingCache",
]
