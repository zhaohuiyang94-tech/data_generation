"""Frozen TRAIN-Gold path gate for explicit unique extrema questions.

This selector is deliberately narrow.  It runs after all existing hard,
path-alignment and source-template gates, and immediately abstains when one of
those gates already changed the incumbent.  A challenger must preserve the
complete singleton/operator structure and gain exact top-1 relation-path
support from a local index parsed from *TRAIN* Gold SPARQL.

There are no model, embedding, endpoint, entity-list or relation-list calls.
The evaluation split's Gold answers are not accepted by the API.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import heapq
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Sequence

from .contracts import ExecutedGraph
from .path_alignment_selector import (
    NaturalPath,
    SemanticIntent,
    _natural_paths,
    _operator_signature,
    _semantic_intents,
)
from .template_path_retrieval import mask_entities


SELECTOR_VERSION = "train-gold-path-unique-extrema-production-v1"
INDEX_VERSION = 2
REASON_CODE = "train_gold_path_unique_extrema_gate"
TOP_K = 48
EPSILON = 1e-12

# Generic high-confidence threshold frozen for the focused lane.  Use a round
# value rather than copying any individual evaluation sample's score.
MIN_ONTOLOGY_VOTE_DELTA = 0.95
MIN_GOLD_SCORE_DELTA = 0.5
MIN_RULE_SCORE_DELTA = -0.02
RETRIEVAL_RULE_SCORE_FLOOR = -0.12

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_MAXIMUM_RE = re.compile(
    r"\b(?:latest|last|most recent|biggest|largest|highest|maximum|newest|longest)\b",
    re.I,
)
_MINIMUM_RE = re.compile(
    r"\b(?:earliest|least|smallest|lowest|minimum|first|oldest|youngest|shortest)\b",
    re.I,
)
_NON_EXTREMA_NAME_RE = re.compile(r"\b(?:first|last)\s+name\b", re.I)
_QUOTED_RE = re.compile(r'["“][^"”]*["”]')
_FULL_DATE_RE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_YEAR_RE = re.compile(r"(?<![\d.])(?:1[0-9]{3}|20[0-9]{2})(?![\d.])")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")


def _tokens(value: Any) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(str(value)))


def _set_f1(left: Iterable[Any], right: Iterable[Any]) -> float:
    one, two = set(left), set(right)
    if not one or not two:
        return 0.0
    return 2.0 * len(one & two) / (len(one) + len(two))


def _relation_leaf_tokens(relations: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        token
        for relation in relations
        for token in _tokens(str(relation).split(".")[-1].replace("_", " "))
    )


def _normalize_scalars(value: Any) -> str:
    text = _FULL_DATE_RE.sub(" date ", str(value))
    text = _YEAR_RE.sub(" year ", text)
    text = _NUMBER_RE.sub(" number ", text)
    return " ".join(text.split())


def _relation_text(relations: Sequence[str]) -> str:
    return " ".join(
        segment.replace("_", " ")
        for relation in relations
        for segment in str(relation).split(".")
    )


def _unique_extrema_operator(question: str) -> str:
    text = _QUOTED_RE.sub(" ", str(question))
    text = _NON_EXTREMA_NAME_RE.sub(" ", text)
    maximum = bool(_MAXIMUM_RE.search(text))
    minimum = bool(_MINIMUM_RE.search(text))
    if maximum == minimum:
        return ""
    return "ARGMAX" if maximum else "ARGMIN"


def _higher_priority_selected(decision: dict[str, Any]) -> bool:
    return any(
        str(reason).startswith("hard_")
        or str(reason) in {
            "path_alignment_gate",
            "source_verified_template_consensus_gate",
        }
        for reason in decision.get("reason_codes", [])
    )


@dataclass(frozen=True, slots=True)
class GoldPathDocument:
    source_index: int
    relations: tuple[str, ...]
    directions: tuple[str, ...]
    operator_signature: tuple[str, ...]
    tokens: tuple[str, ...]
    alignment: tuple[float, ...]


class TrainGoldPathIndex:
    """Read-only sparse index produced by the offline TRAIN parser."""

    def __init__(self, documents: Sequence[GoldPathDocument], *, cache_hit: bool) -> None:
        self.documents = list(documents)
        self.cache_hit = bool(cache_hit)
        self.frequencies = [Counter(document.tokens) for document in self.documents]
        self.average_length = sum(len(document.tokens) for document in self.documents) / max(
            1, len(self.documents)
        )
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(set(document.tokens))
        count = len(self.documents)
        self.idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, frequencies in enumerate(self.frequencies):
            for token, frequency in frequencies.items():
                self.inverted[token].append((index, frequency))

    @classmethod
    def load(cls, path: str | Path) -> "TrainGoldPathIndex":
        cache_path = Path(path).expanduser().resolve()
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("version") != INDEX_VERSION:
            raise ValueError(f"invalid TRAIN Gold path cache: {cache_path}")
        documents = []
        for value in payload.get("documents", []):
            if not isinstance(value, dict):
                continue
            relations = tuple(map(str, value.get("relations", [])))
            directions = tuple(map(str, value.get("directions", [])))
            tokens = tuple(map(str, value.get("tokens", [])))
            if not relations or len(relations) != len(directions) or not tokens:
                continue
            documents.append(
                GoldPathDocument(
                    source_index=int(value.get("source_index", -1)),
                    relations=relations,
                    directions=directions,
                    operator_signature=tuple(map(str, value.get("operator_signature", []))),
                    tokens=tokens,
                    alignment=tuple(float(item) for item in value.get("alignment", [])),
                )
            )
        if not documents:
            raise ValueError(f"TRAIN Gold path cache has no documents: {cache_path}")
        return cls(documents, cache_hit=True)

    def retrieve(
        self,
        query: str,
        *,
        operator_signature: tuple[str, ...],
        top_k: int = TOP_K,
    ) -> list[tuple[float, GoldPathDocument]]:
        query_frequencies = Counter(_tokens(query))
        selected_tokens = sorted(
            query_frequencies,
            key=lambda token: (self.idf.get(token, 0.0), query_frequencies[token], token),
            reverse=True,
        )[:28]
        sparse_scores: dict[int, float] = defaultdict(float)
        for token in selected_tokens:
            query_frequency = query_frequencies[token]
            token_idf = self.idf.get(token, 0.0)
            for index, frequency in self.inverted.get(token, ()):
                length_norm = 0.25 + (
                    0.75 * len(self.documents[index].tokens) / max(1e-9, self.average_length)
                )
                sparse_scores[index] += token_idf * (
                    frequency * 2.2 / (frequency + 1.2 * length_norm)
                ) * min(2, query_frequency)
        preselected = heapq.nlargest(
            max(256, int(top_k) * 4),
            sparse_scores.items(),
            key=lambda item: (item[1], -item[0]),
        )
        target_operators = set(operator_signature) - {"NO_EQUAL", "AND"}
        result: list[tuple[float, int, GoldPathDocument]] = []
        for index, lexical_score in preselected:
            document = self.documents[index]
            score = lexical_score
            source_operators = set(document.operator_signature) - {"NO_EQUAL", "AND"}
            if target_operators and source_operators == target_operators:
                score += 1.25
            elif target_operators and source_operators and source_operators != target_operators:
                score -= 2.0
            alignment = document.alignment
            score += 0.35 * (alignment[1] if len(alignment) > 1 else 0.0)
            score += 0.20 * (alignment[2] if len(alignment) > 2 else 0.0)
            result.append((score, index, document))
        result.sort(key=lambda item: (-item[0], item[2].source_index, item[1]))
        return [(score, document) for score, _, document in result[: max(1, int(top_k))]]


def _query_document(question: str, intents: Sequence[SemanticIntent]) -> str:
    surfaces = [intent.anchor_surface for intent in intents if intent.anchor_surface]
    masked_question = mask_entities(_normalize_scalars(question), surfaces)
    goals = [
        mask_entities(_normalize_scalars(intent.goal), surfaces)
        for intent in intents
        if intent.goal
    ]
    relations = [relation for intent in intents for relation in intent.relation_ids]
    return " ".join(
        [masked_question, masked_question, *(goal for goal in goals for _ in range(2)), _relation_text(relations)]
    )


def _reverse_equivalent(
    left: tuple[str, str], right: tuple[str, str], ontology: Any
) -> bool:
    return (
        left[1] != right[1]
        and right[0] in set(ontology.reverse_for_relation(left[0]))
    )


def _path_match(path: NaturalPath, document: GoldPathDocument, ontology: Any) -> dict[str, float]:
    candidate_pairs = tuple(zip(path.relations, path.directions))
    gold_pairs = tuple(zip(document.relations, document.directions))
    ontology_exact = len(candidate_pairs) == len(gold_pairs) and all(
        left == right or _reverse_equivalent(left, right, ontology)
        for left, right in zip(candidate_pairs, gold_pairs)
    )
    return {
        "exact": float(candidate_pairs == gold_pairs),
        "ontology_exact": float(ontology_exact),
        "pair_f1": _set_f1(candidate_pairs, gold_pairs),
        "relation_f1": _set_f1(path.relations, document.relations),
        "leaf_f1": _set_f1(
            _relation_leaf_tokens(path.relations),
            _relation_leaf_tokens(document.relations),
        ),
        "length": min(len(path.relations), len(document.relations))
        / max(1, max(len(path.relations), len(document.relations))),
    }


def _candidate_features(
    paths: Sequence[NaturalPath],
    intents: Sequence[SemanticIntent],
    retrieved: Sequence[tuple[float, GoldPathDocument]],
    ontology: Any,
) -> dict[str, Any]:
    if not paths or not retrieved:
        return {
            "score": 0.0,
            "exact_vote": 0.0,
            "ontology_vote": 0.0,
            "soft_vote": 0.0,
            "top1_ontology_exact": 0.0,
            "best_match": {},
        }
    anchor_ids = {intent.anchor_id for intent in intents if intent.anchor_id}
    anchored_paths = [path for path in paths if not anchor_ids or path.start_id in anchor_ids]
    compared_paths = anchored_paths or list(paths)
    top_score = retrieved[0][0]
    weights = [
        math.exp(min(0.0, (score - top_score) / 2.5)) / math.log2(rank + 2)
        for rank, (score, _) in enumerate(retrieved)
    ]
    exact_vote = ontology_vote = soft_vote = top1 = 0.0
    best_value = -1.0
    best_match: dict[str, Any] = {}
    for rank, ((retrieval_score, document), weight) in enumerate(zip(retrieved, weights), start=1):
        match, path = max(
            ((_path_match(path, document, ontology), path) for path in compared_paths),
            key=lambda item: (
                item[0]["ontology_exact"],
                item[0]["exact"],
                item[0]["pair_f1"],
                item[0]["relation_f1"],
                item[0]["leaf_f1"],
                item[0]["length"],
            ),
        )
        exact_vote += weight * match["exact"]
        ontology_vote += weight * match["ontology_exact"]
        soft = (
            0.42 * match["pair_f1"]
            + 0.27 * match["relation_f1"]
            + 0.18 * match["leaf_f1"]
            + 0.13 * match["length"]
        )
        soft_vote += weight * soft
        if rank == 1:
            top1 = match["ontology_exact"]
        value = weight * (0.62 * match["ontology_exact"] + 0.38 * soft)
        if value > best_value:
            best_value = value
            best_match = {
                "rank": rank,
                "retrieval_score": retrieval_score,
                "source_index": document.source_index,
                "gold_relations": list(document.relations),
                "gold_directions": list(document.directions),
                "candidate_relations": list(path.relations),
                "candidate_directions": list(path.directions),
                "match": match,
            }
    denominator = sum(weights) or 1.0
    exact_vote /= denominator
    ontology_vote /= denominator
    soft_vote /= denominator
    return {
        "score": 0.54 * ontology_vote + 0.18 * exact_vote + 0.28 * soft_vote,
        "exact_vote": exact_vote,
        "ontology_vote": ontology_vote,
        "soft_vote": soft_vote,
        "top1_ontology_exact": top1,
        "best_match": best_match,
    }


@dataclass(slots=True)
class _Candidate:
    execution: ExecutedGraph
    intents: list[SemanticIntent]
    paths: list[NaturalPath]
    features: dict[str, Any] | None = None


class TrainGoldPathExtremaSelector:
    """Apply the frozen unique-extrema TRAIN-path challenger."""

    def __init__(self, index: TrainGoldPathIndex, ontology: Any, *, top_k: int = TOP_K) -> None:
        self.index = index
        self.ontology = ontology
        self.top_k = max(1, min(TOP_K, int(top_k)))

    @classmethod
    def load(
        cls,
        cache_path: str | Path,
        ontology: Any,
        *,
        top_k: int = TOP_K,
    ) -> "TrainGoldPathExtremaSelector":
        return cls(TrainGoldPathIndex.load(cache_path), ontology, top_k=top_k)

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
        wanted = _unique_extrema_operator(question)
        evidence: dict[str, Any] = {
            "version": SELECTOR_VERSION,
            "status": "not_applicable",
            "model_calls": 0,
            "embedding_calls": 0,
            "endpoint_queries": 0,
            "uses_evaluation_gold": False,
            "uses_train_gold_sparql_paths": True,
            "entity_or_relation_whitelist": False,
            "index_document_count": len(self.index.documents),
            "index_cache_hit": self.index.cache_hit,
            "frozen_gate": {
                "minimum_ontology_vote_delta": MIN_ONTOLOGY_VOTE_DELTA,
                "minimum_gold_score_delta": MIN_GOLD_SCORE_DELTA,
                "minimum_rule_score_delta": MIN_RULE_SCORE_DELTA,
                "retrieval_rule_score_floor": RETRIEVAL_RULE_SCORE_FLOOR,
            },
        }
        if _higher_priority_selected(prior_decision):
            evidence.update(
                {
                    "status": "skipped_higher_priority_postselection_gate",
                    "prior_reason_codes": list(prior_decision.get("reason_codes", [])),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        if not wanted or self.ontology is None or not semantic_graphs:
            evidence.update(
                {
                    "status": "question_or_schema_not_applicable",
                    "extrema_operator": wanted,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        selected_answers = set(map(str, selected.answer_ids))
        selected_signature = _operator_signature(selected)
        if len(selected_answers) != 1 or wanted not in selected_signature:
            evidence.update(
                {
                    "status": "incumbent_singleton_or_operator_guard_rejected",
                    "extrema_operator": wanted,
                    "incumbent_answer_count": len(selected_answers),
                    "incumbent_operator_signature": list(selected_signature),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence

        unique: list[ExecutedGraph] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for execution in [*executions, selected]:
            answers = tuple(sorted(set(map(str, execution.answer_ids))))
            if not answers:
                continue
            key = (str(execution.graph.graph_id), answers)
            if key in seen:
                continue
            seen.add(key)
            unique.append(execution)
        incumbent_index = next(
            (
                index
                for index, execution in enumerate(unique)
                if execution.graph.graph_id == selected.graph.graph_id
                and set(map(str, execution.answer_ids)) == selected_answers
            ),
            None,
        )
        if incumbent_index is None:
            evidence.update(
                {
                    "status": "incumbent_not_materialized_as_complete_candidate",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        candidates = [
            _Candidate(
                execution=execution,
                intents=_semantic_intents(execution, semantic_graphs),
                paths=_natural_paths(execution, self.ontology),
            )
            for execution in unique
        ]
        incumbent_rule_score = float(selected.graph.score)
        eligible = [
            index
            for index, candidate in enumerate(candidates)
            if index != incumbent_index
            and len(set(map(str, candidate.execution.answer_ids))) == 1
            and set(map(str, candidate.execution.answer_ids)) != selected_answers
            and _operator_signature(candidate.execution) == selected_signature
            and float(candidate.execution.graph.score) + EPSILON
            >= incumbent_rule_score + RETRIEVAL_RULE_SCORE_FLOOR
            and candidate.intents
            and candidate.paths
        ]
        incumbent = candidates[incumbent_index]
        if not eligible or not incumbent.intents or not incumbent.paths:
            evidence.update(
                {
                    "status": "no_singleton_operator_matched_challenger",
                    "candidate_count": len(candidates),
                    "eligible_candidate_count": len(eligible),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        retrieved = self.index.retrieve(
            _query_document(question, incumbent.intents),
            operator_signature=selected_signature,
            top_k=self.top_k,
        )
        if not retrieved:
            evidence.update(
                {
                    "status": "no_train_gold_path_neighbours",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return selected, evidence
        for index in [incumbent_index, *eligible]:
            candidate = candidates[index]
            candidate.features = _candidate_features(
                candidate.paths,
                candidate.intents,
                retrieved,
                self.ontology,
            )
        chosen_index = max(
            eligible,
            key=lambda index: (
                float((candidates[index].features or {}).get("score", 0.0)),
                float((candidates[index].features or {}).get("ontology_vote", 0.0)),
                float((candidates[index].features or {}).get("soft_vote", 0.0)),
                float(candidates[index].execution.graph.score),
                str(candidates[index].execution.graph.graph_id),
            ),
        )
        challenger = candidates[chosen_index]
        source_features = incumbent.features or {}
        proposed_features = challenger.features or {}
        proposal = {
            "source_graph_id": selected.graph.graph_id,
            "proposed_graph_id": challenger.execution.graph.graph_id,
            "extrema_operator": wanted,
            "source_answer_count": len(selected_answers),
            "proposed_answer_count": len(set(map(str, challenger.execution.answer_ids))),
            "operator_signature": list(selected_signature),
            "source_top1_ontology_exact": float(source_features.get("top1_ontology_exact", 0.0)),
            "proposed_top1_ontology_exact": float(proposed_features.get("top1_ontology_exact", 0.0)),
            "ontology_vote_delta": float(proposed_features.get("ontology_vote", 0.0))
            - float(source_features.get("ontology_vote", 0.0)),
            "gold_score_delta": float(proposed_features.get("score", 0.0))
            - float(source_features.get("score", 0.0)),
            "rule_score_delta": float(challenger.execution.graph.score) - incumbent_rule_score,
            "source_best_match": source_features.get("best_match", {}),
            "proposed_best_match": proposed_features.get("best_match", {}),
            "neighbour_count": len(retrieved),
            "top_source_index": retrieved[0][1].source_index,
        }
        passed = bool(
            proposal["source_top1_ontology_exact"] == 0.0
            and proposal["proposed_top1_ontology_exact"] == 1.0
            and proposal["ontology_vote_delta"] + EPSILON >= MIN_ONTOLOGY_VOTE_DELTA
            and proposal["gold_score_delta"] + EPSILON >= MIN_GOLD_SCORE_DELTA
            and proposal["rule_score_delta"] + EPSILON >= MIN_RULE_SCORE_DELTA
        )
        evidence.update(
            {
                "status": "selected_train_gold_path_extrema_challenger"
                if passed
                else "confidence_gate_rejected",
                "candidate_count": len(candidates),
                "eligible_candidate_count": len(eligible),
                "proposal": proposal,
                "selected_graph_id": challenger.execution.graph.graph_id if passed else selected.graph.graph_id,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return (challenger.execution if passed else selected), evidence


__all__ = [
    "GoldPathDocument",
    "INDEX_VERSION",
    "MIN_GOLD_SCORE_DELTA",
    "MIN_ONTOLOGY_VOTE_DELTA",
    "MIN_RULE_SCORE_DELTA",
    "REASON_CODE",
    "SELECTOR_VERSION",
    "TrainGoldPathExtremaSelector",
    "TrainGoldPathIndex",
    "_unique_extrema_operator",
]
