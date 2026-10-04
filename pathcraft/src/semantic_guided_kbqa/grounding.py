from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import re
from threading import local
import unicodedata
from typing import Any, Iterator

from .clients import EmbeddingRanker, KnowledgeGraph
from .contracts import EntityCandidate, GroundedSemanticCandidate
from .ontology import FreebaseOntology, relation_id_from_label


@dataclass(slots=True)
class _PathVariant:
    path: dict[str, Any]
    relation_bindings: dict[str, str]
    entity: EntityCandidate
    score: float
    hops: list[dict[str, Any]]


@dataclass(slots=True)
class _StepwiseState:
    """One partial path in the recursive (WebQSP-style) beam."""

    node_id: str
    node_type: str
    node_datatype: str
    node_lang: str
    relation_ids: list[str]
    relation_bindings: dict[str, str]
    hops: list[dict[str, Any]]
    score_sum: float


def _answer_type_challenger_frontier(
    states: list[_StepwiseState],
    limit: int,
    *,
    answer_type_hint: str,
    ontology: FreebaseOntology | None,
    ranker: EmbeddingRanker,
    score_cache: dict[tuple[str, str], float] | None = None,
    minimum_gain: float = 0.05,
    minimum_score: float = 0.50,
) -> list[_StepwiseState]:
    """Reserve one bounded beam slot for an answer-type challenger.

    The caller has already sorted ``states`` by its ordinary accumulated
    relation score.  Preserve the first ``limit - 1`` states verbatim and use
    only the last slot for the best ontology endpoint type outside that head.
    Type similarity therefore controls survival, never the score or final
    ordering of a candidate.
    """

    bounded = max(1, int(limit))
    normalized_hint = " ".join(str(answer_type_hint).split())
    if (
        len(states) <= bounded
        or bounded < 2
        or not normalized_hint
        or ontology is None
    ):
        return states[:bounded]

    head = states[: bounded - 1]
    challengers = states[bounded - 1 :]
    typed_challengers: list[tuple[_StepwiseState, str]] = []
    for state in challengers:
        hop = state.hops[-1] if state.hops else {}
        relation_id = str(hop.get("relation_id", "")).strip()
        direction = str(hop.get("direction", "")).strip().casefold()
        if not relation_id or direction not in {"forward", "backward"}:
            continue
        endpoint_type = (
            ontology.range_for_relation(relation_id)
            if direction == "forward"
            else ontology.domain_for_relation(relation_id)
        )
        endpoint_text = _relation_text(str(endpoint_type))
        if endpoint_text:
            typed_challengers.append((state, endpoint_text))
    if not typed_challengers:
        return states[:bounded]

    unique_type_texts = list(
        dict.fromkeys(text for _, text in typed_challengers)
    )
    cache = score_cache if score_cache is not None else {}
    missing_type_texts = [
        text
        for text in unique_type_texts
        if (normalized_hint, text) not in cache
    ]
    if missing_type_texts:
        missing_scores = ranker.score(normalized_hint, missing_type_texts)
        cache.update(
            {
                (normalized_hint, text): float(score)
                for text, score in zip(missing_type_texts, missing_scores)
            }
        )
    score_by_text = {
        text: float(cache.get((normalized_hint, text), 0.0))
        for text in unique_type_texts
    }
    baseline_entry = next(
        (
            item
            for item in typed_challengers
            if item[0] is states[bounded - 1]
        ),
        None,
    )
    if baseline_entry is None:
        return states[:bounded]
    challenger, challenger_type = max(
        typed_challengers,
        key=lambda item: float(score_by_text.get(item[1], 0.0)),
    )
    baseline_score = float(score_by_text.get(baseline_entry[1], 0.0))
    challenger_score = float(score_by_text.get(challenger_type, 0.0))
    if challenger_score < float(minimum_score):
        return states[:bounded]
    if (
        challenger is not baseline_entry[0]
        and challenger_score < baseline_score + max(0.0, float(minimum_gain))
    ):
        return states[:bounded]
    return [*head, challenger]


class SemanticGrounder:
    """Ground semantic paths and expand only real local KG edges."""

    def __init__(
        self,
        kg: KnowledgeGraph,
        ranker: EmbeddingRanker,
        *,
        entity_top_k: int = 3,
        first_hop_top_k: int = 8,
        second_hop_top_k: int = 8,
        hop_top_k: int = 8,
        path_beam: int = 200,
        hop_query_limit: int = 1024,
        hop_values_per_relation: int = 2,
        long_path_strategy: str = "path_sequence",
        avoid_immediate_backtrack: bool = False,
        candidate_limit: int = 200,
        semantic_beam: int = 12,
        gold_entities: dict[str, dict[str, str]] | None = None,
        ontology: FreebaseOntology | None = None,
        ontology_relation_top_k: int = 8,
        retrieval_challenger_enabled: bool = True,
        ontology_expansion_total_budget: int = 24,
    ) -> None:
        self._limit_overrides = local()
        self.kg = kg
        self.ranker = ranker
        self.entity_top_k = max(1, entity_top_k)
        self.first_hop_top_k = max(1, first_hop_top_k)
        self.second_hop_top_k = max(1, second_hop_top_k)
        self.hop_top_k = max(1, hop_top_k)
        self.path_beam = max(1, path_beam)
        self.hop_query_limit = max(1, hop_query_limit)
        self.hop_values_per_relation = max(1, hop_values_per_relation)
        self.avoid_immediate_backtrack = bool(avoid_immediate_backtrack)
        strategy = str(long_path_strategy).strip().casefold()
        if strategy not in {"path_sequence", "stepwise"}:
            raise ValueError(
                "long_path_strategy must be path_sequence or stepwise"
            )
        self.long_path_strategy = strategy
        self.candidate_limit = max(1, candidate_limit)
        self.semantic_beam = max(1, semantic_beam)
        self.gold_entities = gold_entities or {}
        self.ontology = ontology
        self.ontology_relation_top_k = max(1, ontology_relation_top_k)
        self.retrieval_challenger_enabled = bool(retrieval_challenger_enabled)
        self.ontology_expansion_total_budget = max(
            3,
            int(ontology_expansion_total_budget),
        )
        # Failure-only answer-type comparisons are local BGE calls. Cache a
        # hint/type pair so duplicate structural graphs do not add inference
        # work; ordinary grounding never supplies a hint.
        self._answer_type_similarity_cache: dict[tuple[str, str], float] = {}
        # Neighbor queries are identical whenever two partial paths reach the
        # same node with the same direction.  Cache them for the duration of
        # the grounder; this is important for graphs with converging paths.
        self._hop_cache: dict[tuple[str, str, str, str, str, int], list[dict[str, Any]]] = {}
        self._ontology_similarity_cache: dict[tuple[str, ...], tuple[str, ...]] = {}

    @property
    def path_beam(self) -> int:
        return int(
            getattr(
                self._limit_overrides,
                "path_beam",
                self._base_path_beam,
            )
        )

    @path_beam.setter
    def path_beam(self, value: int) -> None:
        self._base_path_beam = max(1, int(value))

    @property
    def candidate_limit(self) -> int:
        return int(
            getattr(
                self._limit_overrides,
                "candidate_limit",
                self._base_candidate_limit,
            )
        )

    @candidate_limit.setter
    def candidate_limit(self, value: int) -> None:
        self._base_candidate_limit = max(1, int(value))

    @contextmanager
    def scoped_retrieval_limits(
        self,
        *,
        path_beam_cap: int | None = None,
        candidate_limit_cap: int | None = None,
    ) -> Iterator[None]:
        """Apply per-thread retrieval caps without mutating shared config."""

        marker = object()
        previous_path_beam = getattr(
            self._limit_overrides,
            "path_beam",
            marker,
        )
        previous_candidate_limit = getattr(
            self._limit_overrides,
            "candidate_limit",
            marker,
        )
        current_path_beam = self.path_beam
        current_candidate_limit = self.candidate_limit
        if path_beam_cap is not None:
            self._limit_overrides.path_beam = min(
                current_path_beam,
                max(1, int(path_beam_cap)),
            )
        if candidate_limit_cap is not None:
            self._limit_overrides.candidate_limit = min(
                current_candidate_limit,
                max(1, int(candidate_limit_cap)),
            )
        try:
            yield
        finally:
            if previous_path_beam is marker:
                if hasattr(self._limit_overrides, "path_beam"):
                    del self._limit_overrides.path_beam
            else:
                self._limit_overrides.path_beam = previous_path_beam
            if previous_candidate_limit is marker:
                if hasattr(self._limit_overrides, "candidate_limit"):
                    del self._limit_overrides.candidate_limit
            else:
                self._limit_overrides.candidate_limit = previous_candidate_limit

    def retrieve(
        self,
        *,
        question: str,
        decompositions: list[str],
        semantic_graphs: list[dict[str, Any]],
        prefer_ontology_paths: bool = False,
        relax_relation_constraints: bool = False,
        constrained_paths_only: bool = False,
        stepwise_answer_type_hints: dict[int, str] | None = None,
    ) -> tuple[list[GroundedSemanticCandidate], dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "entity_top_k": self.entity_top_k,
            "first_hop_top_k": self.first_hop_top_k,
            "second_hop_top_k": self.second_hop_top_k,
            "hop_top_k": self.hop_top_k,
            "path_beam": self.path_beam,
            "hop_query_limit": self.hop_query_limit,
            "hop_values_per_relation": self.hop_values_per_relation,
            "long_path_strategy": self.long_path_strategy,
            "avoid_immediate_backtrack": self.avoid_immediate_backtrack,
            "relax_relation_constraints": bool(relax_relation_constraints),
            "ontology": (
                self.ontology.diagnostics()
                if self.ontology is not None
                else {"enabled": False}
            ),
            "semantic_graphs": [],
        }
        all_candidates: list[GroundedSemanticCandidate] = []
        per_graph_candidates: list[list[GroundedSemanticCandidate]] = []
        context_text = " ".join([question, *decompositions])
        for graph_index, graph in enumerate(semantic_graphs):
            answer_type_hint = str(
                (stepwise_answer_type_hints or {}).get(graph_index, "")
            ).strip()
            graph_candidates, graph_diag = self._retrieve_graph(
                question=question,
                context_text=context_text,
                graph=graph,
                graph_index=graph_index,
                prefer_ontology_paths=prefer_ontology_paths,
                relax_relation_constraints=relax_relation_constraints,
                constrained_paths_only=constrained_paths_only,
                answer_type_hint=answer_type_hint,
            )
            all_candidates.extend(graph_candidates)
            per_graph_candidates.append(graph_candidates)
            diagnostics["semantic_graphs"].append(graph_diag)
        strict_candidate_exists = any(
            not _candidate_uses_retrieval_repair(candidate)
            for candidate in all_candidates
        )
        strict_candidate_keys = {
            _candidate_key(candidate)
            for candidate in all_candidates
            if not _candidate_uses_retrieval_repair(candidate)
        }
        beam_limit = (
            min(self.semantic_beam, len(strict_candidate_keys))
            if strict_candidate_keys
            else self.semantic_beam
        )
        if not strict_candidate_exists:
            for candidate in all_candidates:
                if not _candidate_uses_retrieval_repair(candidate):
                    continue
                candidate.provenance["grounding_repair"] = {
                    "trigger": "STRICT_RETRIEVAL_EMPTY",
                    "strategy": "ontology_relation_expansion",
                    "compose_mode": "deterministic",
                    "plan": {
                        "operator_mode": "deterministic_empty",
                    },
                    "model_used": False,
                }
        all_candidates.sort(
            key=lambda item: (-item.score, _candidate_key(item)),
        )
        deduped: list[GroundedSemanticCandidate] = []
        seen: set[str] = set()
        # Reserve one slot for every grounded Semantic hypothesis before
        # globally filling the beam. This keeps an original and a reviewed
        # decomposition comparable without changing relation scores or using
        # question-specific routing. Challenger-only candidates are not
        # allowed to take these protected head slots; they still compete for
        # the remaining beam by their penalized score.
        for graph_candidates in per_graph_candidates:
            if not graph_candidates:
                continue
            candidate = next(
                (
                    item
                    for item in graph_candidates
                    if not _candidate_uses_retrieval_repair(item)
                ),
                None,
            )
            if candidate is None:
                continue
            key = _candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
            if len(deduped) >= beam_limit:
                break
        for candidate in all_candidates:
            if len(deduped) >= beam_limit:
                break
            key = _candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        diagnostics["candidate_count"] = len(deduped)
        diagnostics["code"] = "OK" if deduped else "NO_GROUNDED_SEMANTIC_CANDIDATE"
        return deduped, diagnostics

    def _retrieve_graph(
        self,
        *,
        question: str,
        context_text: str,
        graph: dict[str, Any],
        graph_index: int,
        prefer_ontology_paths: bool = False,
        relax_relation_constraints: bool = False,
        constrained_paths_only: bool = False,
        answer_type_hint: str = "",
    ) -> tuple[list[GroundedSemanticCandidate], dict[str, Any]]:
        anchors = {
            str(anchor["id"]): anchor
            for anchor in graph.get("anchors", [])
            if isinstance(anchor, dict)
        }
        paths = [path for path in graph.get("semantic_paths", []) if isinstance(path, dict)]
        diagnostics: dict[str, Any] = {
            "graph_index": graph_index,
            "path_count": len(paths),
            "anchors": [],
            "paths": [],
        }
        if answer_type_hint:
            diagnostics["answer_type_challenger_hint"] = answer_type_hint
        entity_candidates: dict[str, list[EntityCandidate]] = {}
        primary_anchor_id = (
            str(paths[0].get("anchor_ref", ""))
            if paths
            else ""
        )
        path_variants: list[list[_PathVariant]] = []
        gold_entities = self.gold_entities.get(_question_key(question), {})
        for anchor_id, anchor in anchors.items():
            surface = str(anchor.get("surface", "")).strip()
            candidates = _gold_entity_candidates(
                surface,
                gold_entities,
                limit=self.entity_top_k,
            )
            if not candidates and gold_entities and anchor_id == primary_anchor_id:
                candidates = _all_gold_entity_candidates(
                    gold_entities,
                    limit=self.entity_top_k,
                )
            if not candidates and not gold_entities:
                candidates = self.kg.search_entities(surface, limit=self.entity_top_k)
            entity_candidates[anchor_id] = candidates[: self.entity_top_k]
            diagnostics["anchors"].append(
                {
                    "anchor_id": anchor_id,
                    "surface": surface,
                    "candidates": [
                        {"id": item.entity_id, "label": item.label, "score": item.score}
                        for item in candidates[: self.entity_top_k]
                    ],
                }
            )
        grounded_paths: list[dict[str, Any]] = []
        dropped_paths: list[dict[str, str]] = []
        primary_path_failed = False
        for path_index, path in enumerate(paths):
            path_id = str(path.get("id", f"P{path_index}"))
            anchor_id = str(path.get("anchor_ref", ""))
            steps = [step for step in path.get("steps", []) if isinstance(step, dict)]
            variants: list[_PathVariant] = []
            for entity in entity_candidates.get(anchor_id, []):
                variants.extend(
                    self._expand_path(
                        question=question,
                        context_text=context_text,
                        path=path,
                        steps=steps,
                        entity=entity,
                        prefer_ontology_paths=prefer_ontology_paths,
                        relax_relation_constraints=relax_relation_constraints,
                        constrained_paths_only=constrained_paths_only,
                        answer_type_hint=answer_type_hint,
                    )
                )
            variants.sort(key=lambda item: (-item.score, _variant_key(item)))
            variants = variants[: self.candidate_limit]
            if variants:
                grounded_paths.append(path)
                path_variants.append(variants)
            else:
                dropped_paths.append(
                    {
                        "path_id": path_id,
                        "anchor_id": anchor_id,
                        "reason": "NO_GROUNDED_VARIANTS",
                    }
                )
                if path_index == 0:
                    primary_path_failed = True
            diagnostics["paths"].append(
                {
                    "path_id": path_id,
                    "anchor_id": anchor_id,
                    "hop_count": len(steps),
                    "expansion_strategy": (
                        "path_sequence"
                        if len(steps) > 2
                        and self.long_path_strategy == "path_sequence"
                        and callable(getattr(self.kg, "path_hops", None))
                        else (
                            "stepwise"
                            if len(steps) > 2
                            and self.long_path_strategy == "stepwise"
                            else "legacy"
                        )
                    ),
                    "first_hop_candidates": sum(len(item.hops) >= 1 for item in variants),
                    "second_hop_candidates": sum(len(item.hops) >= 2 for item in variants),
                    "retained": len(variants),
                    "dropped": not variants,
                }
            )
        diagnostics["dropped_paths"] = dropped_paths
        if primary_path_failed or not path_variants:
            diagnostics["code"] = "PRIMARY_PATH_GROUNDING_FAILED"
            return [], diagnostics

        combinations = _consistent_combinations(
            grounded_paths,
            path_variants,
            self.candidate_limit,
        )
        candidates: list[GroundedSemanticCandidate] = []
        for rank, combination in enumerate(combinations, start=1):
            grounded_path_variants: list[dict[str, Any]] = []
            anchor_bindings: dict[str, EntityCandidate] = {}
            relation_bindings: dict[str, str] = {}
            for path, variant in zip(grounded_paths, combination):
                grounded_path_variants.append(variant.path)
                anchor_id = str(path.get("anchor_ref", ""))
                anchor_bindings[anchor_id] = variant.entity
                relation_bindings.update(variant.relation_bindings)
            retained_anchor_ids = {
                str(path.get("anchor_ref", ""))
                for path in grounded_paths
            }
            compose_anchors: list[dict[str, Any]] = []
            for anchor in graph.get("anchors", []):
                if not isinstance(anchor, dict):
                    continue
                anchor_id = str(anchor.get("id", ""))
                if anchor_id not in retained_anchor_ids:
                    continue
                canonical_anchor = deepcopy(anchor)
                binding = anchor_bindings.get(anchor_id)
                if binding is not None and str(binding.label).strip():
                    canonical_anchor["surface"] = str(binding.label).strip()
                compose_anchors.append(canonical_anchor)
            compose_input = {
                "question": question,
                "anchors": compose_anchors,
                "semantic_paths": grounded_path_variants,
            }
            candidates.append(
                GroundedSemanticCandidate(
                    compose_input=compose_input,
                    anchor_bindings=anchor_bindings,
                    relation_bindings=relation_bindings,
                    score=(
                        sum(item.score for item in combination) / len(combination)
                        - (0.15 * len(dropped_paths))
                    ),
                    provenance={
                        "semantic_graph_index": graph_index,
                        "rank": rank,
                        "hops": [item.hops for item in combination],
                        "dropped_paths": deepcopy(dropped_paths),
                    },
                )
            )
        if candidates and dropped_paths:
            diagnostics["code"] = "OK_WITH_DROPPED_PATHS"
        else:
            diagnostics["code"] = "OK" if candidates else "NO_CONSISTENT_COMBINATION"
        return candidates, diagnostics

    def _expand_path(
        self,
        *,
        question: str,
        context_text: str,
        path: dict[str, Any],
        steps: list[dict[str, Any]],
        entity: EntityCandidate,
        prefer_ontology_paths: bool = False,
        relax_relation_constraints: bool = False,
        constrained_paths_only: bool = False,
        answer_type_hint: str = "",
    ) -> list[_PathVariant]:
        if not steps:
            return []
        if relax_relation_constraints:
            repair_path = deepcopy(path)
            repair_path["goal"] = question
            repair_steps = deepcopy(steps)
            for step in repair_steps:
                step["relation_label"] = []
            variants = self._expand_stepwise_path(
                question=question,
                context_text="",
                path=repair_path,
                steps=repair_steps,
                entity=entity,
                answer_type_hint=answer_type_hint,
            )
            return variants
        # In path-sequence mode every semantic depth follows the same
        # retrieval contract: issue one connected SPARQL path query, return an
        # ordered relation sequence, and rank the complete sequence. Keep the
        # legacy one/two-hop branch below only for older KnowledgeGraph clients
        # and test doubles that do not implement ``path_hops``.
        if (
            len(steps) > 2
            and self.long_path_strategy == "path_sequence"
            and callable(
            getattr(self.kg, "path_hops", None)
            )
        ):
            return self._expand_long_path(
                question=question,
                context_text=context_text,
                path=path,
                steps=steps,
                entity=entity,
            )
        if prefer_ontology_paths and self.ontology is not None and len(steps) >= 2:
            constrained = self._expand_long_path(
                question=question,
                context_text=context_text,
                path=path,
                steps=steps,
                entity=entity,
            )
            if constrained or constrained_paths_only:
                return constrained
        if len(steps) > 2 and self.long_path_strategy == "stepwise":
            return self._expand_stepwise_path(
                question=question,
                context_text=context_text,
                path=path,
                steps=steps,
                entity=entity,
                answer_type_hint=answer_type_hint,
            )
        if len(steps) > 2:
            return self._expand_long_path(
                question=question,
                context_text=context_text,
                path=path,
                steps=steps,
                entity=entity,
            )
        first_step = steps[0]
        raw_first_rows = self.kg.first_hop(
            entity.entity_id,
            limit=max(self.first_hop_top_k * 32, 256),
        )
        first_rows = [
            row
            for row in raw_first_rows
            if str(row.get("direction", "")) == str(first_step.get("direction", ""))
        ]
        first_rows.extend(
            _ontology_direction_rescue_rows(
                raw_first_rows,
                relation_key="relation_id",
                direction_key="direction",
                expected_relation=relation_id_from_label(
                    first_step.get("relation_label", [])
                ),
                required_direction=str(first_step.get("direction", "")),
                ontology=(self.ontology if self.retrieval_challenger_enabled else None),
            )
        )
        first_ranked = self._rank_rows(
            query=_step_query(question, context_text, path, first_step),
            rows=first_rows,
            relation_key="relation_id",
        )[: self.first_hop_top_k]
        annotate_cvt = getattr(self.kg, "annotate_cvt_metadata", None)
        if callable(annotate_cvt):
            first_ranked = annotate_cvt(first_ranked)
        variants: list[_PathVariant] = []
        for first in first_ranked:
            first_id = str(first["relation_id"])
            if len(steps) == 1:
                if _row_reaches_cvt(first):
                    auto_second = self._expand_cvt_second_hop(
                        question=question,
                        context_text=context_text,
                        path=path,
                        first_step=first_step,
                        first=first,
                        entity=entity,
                    )
                    if auto_second:
                        variants.extend(auto_second)
                        continue
                grounded = _grounded_path(
                    path,
                    [first_id],
                    directions=[str(first.get("direction", first_step.get("direction", "")))],
                )
                variants.append(
                    _PathVariant(
                        grounded,
                        {str(first_step["id"]): first_id},
                        entity,
                        _path_score(entity, [float(first["score"])]),
                        [{"hop": 1, **first}],
                    )
                )
                continue
            second_step = steps[1]
            raw_second = self.kg.second_hop(
                entity.entity_id,
                first,
                limit=max(self.second_hop_top_k * 8, self.second_hop_top_k),
            )
            second_rows = [
                row
                for row in raw_second
                if str(row.get("second_direction", "")) == str(second_step.get("direction", ""))
            ]
            second_rows.extend(
                _ontology_direction_rescue_rows(
                    raw_second,
                    relation_key="second_relation_id",
                    direction_key="second_direction",
                    expected_relation=relation_id_from_label(
                        second_step.get("relation_label", [])
                    ),
                    required_direction=str(second_step.get("direction", "")),
                    ontology=(self.ontology if self.retrieval_challenger_enabled else None),
                )
            )
            second_ranked = self._rank_rows(
                query=_step_query(question, context_text, path, second_step),
                rows=second_rows,
                relation_key="second_relation_id",
            )[: self.second_hop_top_k]
            for second in second_ranked:
                second_id = str(second["second_relation_id"])
                grounded = _grounded_path(
                    path,
                    [first_id, second_id],
                    directions=[
                        str(first.get("direction", first_step.get("direction", ""))),
                        str(second.get("second_direction", second_step.get("direction", ""))),
                    ],
                )
                variants.append(
                    _PathVariant(
                        grounded,
                        {
                            str(first_step["id"]): first_id,
                            str(second_step["id"]): second_id,
                        },
                        entity,
                        _path_score(entity, [float(first["score"]), float(second["score"])]),
                        [{"hop": 1, **first}, {"hop": 2, **second}],
                    )
                )
        variants.sort(key=lambda item: (-item.score, _variant_key(item)))
        if len(steps) == 2:
            return variants[: self.second_hop_top_k]
        return variants[: self.first_hop_top_k]

    def _expand_stepwise_path(
        self,
        *,
        question: str,
        context_text: str,
        path: dict[str, Any],
        steps: list[dict[str, Any]],
        entity: EntityCandidate,
        answer_type_hint: str = "",
    ) -> list[_PathVariant]:
        """Expand a long path one edge at a time with a bounded beam.

        This is the WebQSP-style alternative to ``path_hops``.  Every
        frontier item carries its actual current Freebase node, so the next
        SPARQL request is anchored to the endpoint reached by the previous
        hop.  Relation candidates are pruned per parent (``hop_top_k``) and
        then globally (``path_beam``) after each depth.  The resulting object
        deliberately has the same shape as the legacy path-sequence variant,
        so Compose and SPARQL lowering do not need a second contract.
        """
        expand_hop = getattr(self.kg, "expand_hop", None)
        if not callable(expand_hop):
            # Keep old KnowledgeGraph test doubles and deployments usable.  A
            # missing endpoint-aware API falls back to the original complete
            # sequence query when available.
            return self._expand_long_path(
                question=question,
                context_text=context_text,
                path=path,
                steps=steps,
                entity=entity,
            )
        directions = [str(step.get("direction", "")).strip().casefold() for step in steps]
        if any(direction not in {"forward", "backward"} for direction in directions):
            return []

        raw_limit = max(self.hop_query_limit, self.hop_top_k * 32, 256)
        frontier: list[_StepwiseState] = [
            _StepwiseState(
                node_id=entity.entity_id,
                node_type="uri",
                node_datatype="",
                node_lang="",
                relation_ids=[],
                relation_bindings={},
                hops=[],
                score_sum=0.0,
            )
        ]
        for step_index, (step, direction) in enumerate(zip(steps, directions)):
            is_final = step_index == len(steps) - 1
            expanded: list[_StepwiseState] = []
            for state in frontier:
                cache_key = (
                    state.node_type,
                    state.node_id,
                    state.node_datatype,
                    state.node_lang,
                    direction,
                    raw_limit,
                )
                if cache_key not in self._hop_cache:
                    try:
                        node_arg: Any = (
                            state.node_id
                            if state.node_type == "uri"
                            else {
                                "value": state.node_id,
                                "type": state.node_type,
                                "datatype": state.node_datatype,
                                "lang": state.node_lang,
                            }
                        )
                        rows = expand_hop(
                            node_arg,
                            direction,
                            limit=raw_limit,
                        )
                    except (TypeError, ValueError, RuntimeError):
                        rows = []
                    self._hop_cache[cache_key] = [
                        dict(row) for row in rows if isinstance(row, dict)
                    ]
                rows = [dict(row) for row in self._hop_cache[cache_key]]
                rows = [
                    row
                    for row in rows
                    if str(row.get("relation_id", "")).strip()
                    and str(row.get("next_node_id", "")).strip()
                    and (
                        is_final
                        or _can_continue_from_row(
                            row,
                            next_direction=(
                                directions[step_index + 1]
                                if not is_final
                                else ""
                            ),
                        )
                    )
                ]
                ranked = self._rank_edges(
                    query=_stepwise_step_query(
                        question,
                        context_text,
                        path,
                        step,
                        step_index,
                    ),
                    rows=rows,
                    relation_limit=self.hop_top_k,
                    values_per_relation=self.hop_values_per_relation,
                )
                for row in ranked:
                    relation_id = str(row.get("relation_id", "")).strip()
                    next_node_id = str(row.get("next_node_id", "")).strip()
                    if not relation_id or not next_node_id:
                        continue
                    next_node_type = _row_next_type(row)
                    next_node_datatype = str(row.get("next_node_datatype", ""))
                    next_node_lang = str(row.get("next_node_lang", ""))
                    if self.avoid_immediate_backtrack and _is_immediate_backtrack(
                        state,
                        relation_id=relation_id,
                        direction=direction,
                        next_node_id=next_node_id,
                        next_node_type=next_node_type,
                    ):
                        continue
                    bindings = dict(state.relation_bindings)
                    bindings[str(step.get("id", f"S{step_index}"))] = relation_id
                    hop = dict(row)
                    hop["hop"] = step_index + 1
                    hop["current_node_id"] = state.node_id
                    hop["next_node_id"] = next_node_id
                    expanded.append(
                        _StepwiseState(
                            node_id=next_node_id,
                            node_type=next_node_type,
                            node_datatype=next_node_datatype,
                            node_lang=next_node_lang,
                            relation_ids=[*state.relation_ids, relation_id],
                            relation_bindings=bindings,
                            hops=[*state.hops, hop],
                            score_sum=state.score_sum + float(row.get("score", 0.0)),
                        )
                    )
            if not expanded:
                return []

            # Multiple edges can converge on the same endpoint with an
            # identical relation prefix.  Keep the best-scoring representative
            # before applying the global beam so duplicate branches do not
            # consume all slots.
            best_by_key: dict[tuple[str, tuple[str, ...]], _StepwiseState] = {}
            for state in expanded:
                key = (state.node_id, tuple(state.relation_ids))
                previous = best_by_key.get(key)
                if previous is None or state.score_sum > previous.score_sum:
                    best_by_key[key] = state
            expanded = list(best_by_key.values())
            expanded.sort(
                key=lambda state: (
                    -(state.score_sum / max(1, len(state.relation_ids))),
                    tuple(state.relation_ids),
                    state.node_id,
                )
            )
            # ``path_beam`` is the explicit global width of the recursive
            # search.  Do not silently raise it to ``candidate_limit``: the
            # caller may intentionally choose a very small beam for a smoke
            # run or latency-constrained deployment.
            if answer_type_hint:
                frontier = _answer_type_challenger_frontier(
                    expanded,
                    self.path_beam,
                    answer_type_hint=answer_type_hint,
                    ontology=self.ontology,
                    ranker=self.ranker,
                    score_cache=self._answer_type_similarity_cache,
                )
            else:
                frontier = expanded[: self.path_beam]

        variants: list[_PathVariant] = []
        for state in frontier:
            if len(state.relation_ids) != len(steps):
                continue
            variants.append(
                _PathVariant(
                    _grounded_path(path, state.relation_ids),
                    state.relation_bindings,
                    entity,
                    _path_score(
                        entity,
                        [float(item.get("score", 0.0)) for item in state.hops],
                    ),
                    state.hops,
                )
            )
        variants.sort(key=lambda item: (-item.score, _variant_key(item)))
        return variants[: self.candidate_limit]

    def _expand_long_path(
        self,
        *,
        question: str,
        context_text: str,
        path: dict[str, Any],
        steps: list[dict[str, Any]],
        entity: EntityCandidate,
    ) -> list[_PathVariant]:
        """Ground a path of any positive depth using a bounded KG sequence query.

        Older ``KnowledgeGraph`` implementations only expose ``first_hop`` and
        ``second_hop``; long paths are dropped for such clients, while one/two
        hops can still use the compatibility branch in ``_expand_path``.
        ``SparqlKnowledgeGraph.path_hops`` supplies ordered
        relation-id sequences while keeping intermediate nodes existential.  A
        sequence is then ranked independently against each natural-language
        step and lowered only after every step has a real relation binding.
        """
        path_hops = getattr(self.kg, "path_hops", None)
        if not callable(path_hops):
            return []
        directions = [str(step.get("direction", "")).strip() for step in steps]
        if any(direction not in {"forward", "backward"} for direction in directions):
            return []
        relation_candidates: list[list[str]] | None = None
        if self.ontology is not None:
            candidate_sets = self.ontology.candidates_for_steps(
                steps,
                limit=self.ontology_relation_top_k,
            )
            if all(candidate_sets):
                relation_candidates = candidate_sets
        # An unconstrained existential join grows exponentially with path
        # depth and can consume the whole question deadline.  Long paths with
        # incomplete ontology candidates are deferred to the existing bounded
        # structural/stepwise failure lane instead of issuing that query.
        if len(steps) >= 3 and relation_candidates is None:
            return []

        query_limit = max(
            self.candidate_limit * 8,
            self.first_hop_top_k * self.second_hop_top_k,
        )

        def query_path(
            actual_directions: list[str],
            candidates: list[list[str]] | None,
            *,
            limit: int = query_limit,
        ) -> list[dict[str, Any]]:
            kwargs: dict[str, Any] = {"limit": max(1, int(limit))}
            if candidates is not None:
                kwargs["relation_candidates"] = candidates
            try:
                try:
                    value = path_hops(
                        entity.entity_id,
                        actual_directions,
                        **kwargs,
                    )
                except TypeError:
                    # Preserve compatibility with older KnowledgeGraph test
                    # doubles. Production clients support constrained paths.
                    if candidates is None:
                        raise
                    value = path_hops(
                        entity.entity_id,
                        actual_directions,
                        limit=max(1, int(limit)),
                    )
            except (TypeError, ValueError, RuntimeError):
                return []
            return value if isinstance(value, list) else []

        def parse_sequences(
            raw_sequences: list[dict[str, Any]],
            actual_directions: list[str],
            *,
            retrieval_mode: str,
            ontology_constrained: bool,
        ) -> list[tuple[list[str], list[str], list[dict[str, Any]]]]:
            parsed: list[tuple[list[str], list[str], list[dict[str, Any]]]] = []
            seen: set[tuple[str, ...]] = set()
            for raw in raw_sequences:
                if not isinstance(raw, dict):
                    continue
                relation_ids = raw.get("relation_ids", raw.get("relations", []))
                if isinstance(relation_ids, tuple):
                    relation_ids = list(relation_ids)
                if not isinstance(relation_ids, list) or len(relation_ids) != len(steps):
                    continue
                relation_ids = [str(value).strip() for value in relation_ids]
                if not all(_is_relation_id(value) for value in relation_ids):
                    continue
                key = tuple(relation_ids)
                if key in seen:
                    continue
                seen.add(key)
                hops: list[dict[str, Any]] = []
                for index in range(len(steps)):
                    hop = {
                        "hop": index + 1,
                        "relation_id": relation_ids[index],
                        "direction": actual_directions[index],
                        "ontology_constrained": ontology_constrained,
                    }
                    if retrieval_mode != "exact":
                        hop["retrieval_mode"] = retrieval_mode
                    if actual_directions[index] != directions[index]:
                        hop["direction_repaired"] = True
                        hop["predicted_direction"] = directions[index]
                    hops.append(hop)
                parsed.append((relation_ids, list(actual_directions), hops))
            return parsed

        retrieval_mode = "exact"
        raw_sequences = query_path(directions, relation_candidates)
        sequences = parse_sequences(
            raw_sequences,
            directions,
            retrieval_mode=retrieval_mode,
            ontology_constrained=relation_candidates is not None,
        )

        # Exact ontology labels are intentionally the unchanged fast path.
        # Only an empty verified path activates local schema-neighbor search.
        if (
            not sequences
            and self.ontology is not None
            and self.retrieval_challenger_enabled
        ):
            expanded_candidates: list[list[str]] = []
            expansion_limit = min(
                self.ontology_relation_top_k,
                max(
                    3,
                    self.ontology_expansion_total_budget
                    // max(1, len(steps)),
                ),
            )
            for step in steps:
                label_key = tuple(str(value) for value in step.get("relation_label", []))
                cached = self._ontology_similarity_cache.get(label_key)
                if cached is None:
                    cached = tuple(
                        self.ontology.similar_candidates_for_label(
                            step.get("relation_label", []),
                            limit=expansion_limit,
                        )
                    )
                    self._ontology_similarity_cache[label_key] = cached
                expanded_candidates.append(list(cached))
            if all(expanded_candidates) and expanded_candidates != relation_candidates:
                retrieval_mode = "ontology_expansion"
                raw_sequences = query_path(
                    directions,
                    expanded_candidates,
                    limit=max(
                        self.candidate_limit * 2,
                        self.first_hop_top_k * self.second_hop_top_k,
                    ),
                )
                sequences = parse_sequences(
                    raw_sequences,
                    directions,
                    retrieval_mode=retrieval_mode,
                    ontology_constrained=True,
                )

        if not sequences:
            return []

        # Rank each hop's relation candidates in one embedding batch.  The
        # combined score is the mean of the per-step scores, preserving the
        # same 0.7 embedding / 0.3 lexical weighting as the two-hop path.
        step_scores: list[list[float]] = []
        for step_index, step in enumerate(steps):
            if retrieval_mode == "exact":
                query = _step_query(question, context_text, path, step)
                relation_texts = [
                    _relation_text(relation_ids[step_index])
                    for relation_ids, _, _ in sequences
                ]
            else:
                query = _focused_step_query(question, path, step, step_index)
                relation_texts = [
                    _relation_schema_text(
                        relation_ids[step_index],
                        self.ontology,
                    )
                    for relation_ids, _, _ in sequences
                ]
            embeddings = self.ranker.score(query, relation_texts)
            step_scores.append(
                [
                    (0.7 * float(embedding))
                    + (0.3 * _lexical_coverage(query, relation_text))
                    for embedding, relation_text in zip(embeddings, relation_texts)
                ]
            )

        variants: list[_PathVariant] = []
        score_penalty = {
            "exact": 0.0,
            "ontology_expansion": 0.04,
        }.get(retrieval_mode, 0.06)
        for sequence_index, (relation_ids, actual_directions, hops) in enumerate(sequences):
            scores = [
                values[sequence_index]
                for values in step_scores
                if sequence_index < len(values)
            ]
            if len(scores) != len(steps):
                continue
            variants.append(
                _PathVariant(
                    _grounded_path(
                        path,
                        relation_ids,
                        directions=actual_directions,
                    ),
                    {
                        str(step["id"]): relation_ids[index]
                        for index, step in enumerate(steps)
                    },
                    entity,
                    _path_score(entity, scores)
                    + _ontology_path_compatibility_bonus(
                        relation_ids,
                        actual_directions,
                        self.ontology,
                    )
                    - score_penalty,
                    hops,
                )
            )
        variants.sort(key=lambda item: (-item.score, _variant_key(item)))
        return variants[: self.candidate_limit]

    def _expand_cvt_second_hop(
        self,
        *,
        question: str,
        context_text: str,
        path: dict[str, Any],
        first_step: dict[str, Any],
        first: dict[str, Any],
        entity: EntityCandidate,
    ) -> list[_PathVariant]:
        """Add a second step when a one-hop relation targets a CVT type."""
        first_id = str(first["relation_id"])
        raw_second = self.kg.second_hop(
            entity.entity_id,
            first,
            limit=max(self.second_hop_top_k * 8, self.second_hop_top_k),
        )
        backtrack_direction = (
            "backward" if str(first.get("direction", "forward")) == "forward" else "forward"
        )
        second_rows = [
            row
            for row in raw_second
            if str(row.get("second_relation_id", "")) != first_id
            or str(row.get("second_direction", "")) != backtrack_direction
        ]
        second_ranked = self._rank_rows(
            query=_cvt_step_query(question, context_text, path),
            rows=second_rows,
            relation_key="second_relation_id",
        )[: self.second_hop_top_k]
        if not second_ranked:
            return []
        variants: list[_PathVariant] = []
        path_id = str(path.get("id", "P0"))
        second_step_id = f"{path_id}.S1"
        second_from = str(first_step.get("to", f"{path_id}.V0"))
        second_to = f"{path_id}.V1"
        for second in second_ranked:
            second_id = str(second["second_relation_id"])
            second_direction = str(second.get("second_direction", "forward"))
            grounded = _grounded_cvt_path(
                path,
                first_id=first_id,
                first_direction=str(
                    first.get("direction", first_step.get("direction", ""))
                ),
                second_id=second_id,
                second_direction=second_direction,
                second_step_id=second_step_id,
                second_from=second_from,
                second_to=second_to,
            )
            variants.append(
                _PathVariant(
                    grounded,
                    {
                        str(first_step["id"]): first_id,
                        second_step_id: second_id,
                    },
                    entity,
                    _path_score(entity, [float(first["score"]), float(second["score"])]),
                    [
                        {"hop": 1, **first},
                        {"hop": 2, "auto_expanded_cvt": True, **second},
                    ],
                )
            )
        return variants

    def _rank_rows(
        self,
        *,
        query: str,
        rows: list[dict[str, Any]],
        relation_key: str,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        relation_texts = [_relation_text(str(row.get(relation_key, ""))) for row in rows]
        embedding_scores = self.ranker.score(query, relation_texts)
        scores = [
            (0.7 * embedding_score) + (0.3 * _lexical_coverage(query, relation_text))
            for embedding_score, relation_text in zip(embedding_scores, relation_texts)
        ]
        ranked: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row, score in zip(rows, scores):
            relation_id = str(row.get(relation_key, ""))
            direction = str(row.get("direction", row.get("second_direction", "")))
            key = (relation_id, direction)
            if not relation_id or key in seen:
                continue
            seen.add(key)
            adjusted_score = float(score) - (
                0.06 if bool(row.get("direction_repaired")) else 0.0
            )
            ranked.append({**row, "score": adjusted_score})
        ranked.sort(key=lambda item: (-float(item["score"]), str(item.get(relation_key, ""))))
        return ranked

    def _rank_edges(
        self,
        *,
        query: str,
        rows: list[dict[str, Any]],
        relation_limit: int | None = None,
        values_per_relation: int = 1,
    ) -> list[dict[str, Any]]:
        """Rank endpoint-aware edges without collapsing distinct neighbors."""
        if not rows:
            return []
        relation_texts = [
            _relation_text(str(row.get("relation_id", "")))
            for row in rows
        ]
        # All endpoint values of one relation share the same text.  Score
        # each distinct relation once instead of sending thousands of
        # duplicate documents to the embedding service for high-degree nodes.
        unique_relation_texts = list(dict.fromkeys(relation_texts))
        unique_embedding_scores = self.ranker.score(query, unique_relation_texts)
        embedding_by_text = dict(zip(unique_relation_texts, unique_embedding_scores))
        embedding_scores = [
            float(embedding_by_text.get(text, 0.0)) for text in relation_texts
        ]
        base_scores = [
            (0.7 * float(embedding_score))
            + (0.3 * _lexical_coverage(query, relation_text))
            for embedding_score, relation_text in zip(embedding_scores, relation_texts)
        ]
        scores = base_scores
        ranked: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for row, score in zip(rows, scores):
            relation_id = str(row.get("relation_id", "")).strip()
            direction = str(row.get("direction", "")).strip()
            next_node_id = str(row.get("next_node_id", "")).strip()
            key = (relation_id, direction, next_node_id)
            if not relation_id or not next_node_id or key in seen:
                continue
            seen.add(key)
            ranked.append({**row, "score": float(score)})
        ranked.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item.get("relation_id", "")),
                str(item.get("next_node_id", "")),
            )
        )
        if relation_limit is None:
            return ranked

        # Select relations by their best endpoint score, then preserve a
        # small number of endpoint values for each selected relation.  A raw
        # edge sort alone lets a high-cardinality relation consume the whole
        # hop beam and hides other semantically relevant predicates.
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in ranked:
            key = (
                str(row.get("relation_id", "")),
                str(row.get("direction", "")),
            )
            grouped.setdefault(key, []).append(row)
        relation_groups = sorted(
            grouped.items(),
            key=lambda item: (
                -float(item[1][0].get("score", 0.0)),
                item[0],
            ),
        )[: max(1, int(relation_limit))]
        selected: list[dict[str, Any]] = []
        for _, values in relation_groups:
            selected.extend(values[: max(1, int(values_per_relation))])
        selected.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("relation_id", "")),
                str(item.get("next_node_id", "")),
            )
        )
        return selected


def _grounded_path(
    path: dict[str, Any],
    relation_ids: list[str],
    *,
    directions: list[str] | None = None,
) -> dict[str, Any]:
    result = deepcopy(path)
    for index, (step, relation_id) in enumerate(zip(result["steps"], relation_ids)):
        step["relation_label"] = _relation_label(relation_id)
        if directions is not None and index < len(directions):
            step["direction"] = str(directions[index])
    return result


def _grounded_cvt_path(
    path: dict[str, Any],
    *,
    first_id: str,
    first_direction: str,
    second_id: str,
    second_direction: str,
    second_step_id: str,
    second_from: str,
    second_to: str,
) -> dict[str, Any]:
    result = deepcopy(path)
    steps = result["steps"]
    steps[0]["relation_label"] = _relation_label(first_id)
    steps[0]["direction"] = str(first_direction)
    steps.append(
        {
            "id": second_step_id,
            "relation_label": _relation_label(second_id),
            "direction": second_direction,
            "from": second_from,
            "to": second_to,
        }
    )
    result["path_output_var"] = second_to
    return result


def _relation_label(relation_id: str) -> list[str]:
    return [part.replace("_", " ") for part in relation_id.split(".") if part]


def _is_relation_id(value: str) -> bool:
    """Validate a compact dotted Freebase relation identifier."""
    return bool(re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", str(value)))


def _is_entity_id(value: str) -> bool:
    """Return whether a hop endpoint is a Freebase entity/CVT identifier."""
    return bool(re.fullmatch(r"(?:m|g)\.[A-Za-z0-9_]+", str(value)))


def _relation_text(relation_id: str) -> str:
    return " ".join(_relation_label(relation_id))


def _ontology_direction_rescue_rows(
    rows: list[dict[str, Any]],
    *,
    relation_key: str,
    direction_key: str,
    expected_relation: str,
    required_direction: str,
    ontology: FreebaseOntology | None,
) -> list[dict[str, Any]]:
    """Return verified opposite-direction edges only when strict exact is absent."""
    if (
        ontology is None
        or not expected_relation
        or required_direction not in {"forward", "backward"}
    ):
        return []
    equivalent_relations = {
        expected_relation,
        *ontology.reverse_for_relation(expected_relation),
    }
    strict_has_equivalent = any(
        str(row.get(direction_key, "")) == required_direction
        and str(row.get(relation_key, "")) in equivalent_relations
        for row in rows
    )
    if strict_has_equivalent:
        return []
    repaired: list[dict[str, Any]] = []
    for row in rows:
        actual_direction = str(row.get(direction_key, ""))
        relation_id = str(row.get(relation_key, ""))
        if (
            actual_direction not in {"forward", "backward"}
            or actual_direction == required_direction
            or relation_id not in equivalent_relations
        ):
            continue
        repaired.append(
            {
                **row,
                "direction_repaired": True,
                "predicted_direction": required_direction,
            }
        )
        if len(repaired) >= 2:
            break
    return repaired


def _relation_schema_text(
    relation_id: str,
    ontology: FreebaseOntology | None,
) -> str:
    labels = _relation_label(relation_id)
    leaf = labels[-1] if labels else ""
    domain = ontology.domain_for_relation(relation_id) if ontology is not None else ""
    range_id = ontology.range_for_relation(relation_id) if ontology is not None else ""
    return " | ".join(
        value
        for value in (
            leaf,
            " ".join(labels),
            _relation_text(domain) if domain else "",
            _relation_text(range_id) if range_id else "",
        )
        if value
    )


def _ontology_path_compatibility_bonus(
    relation_ids: list[str],
    directions: list[str],
    ontology: FreebaseOntology | None,
) -> float:
    """Small positive-only type continuity bonus for recovery paths."""
    if ontology is None or len(relation_ids) < 2:
        return 0.0
    compatible = 0.0
    known_pairs = 0
    for index in range(len(relation_ids) - 1):
        left_relation, right_relation = relation_ids[index : index + 2]
        left_direction, right_direction = directions[index : index + 2]
        left_output = (
            ontology.range_for_relation(left_relation)
            if left_direction == "forward"
            else ontology.domain_for_relation(left_relation)
        )
        right_input = (
            ontology.domain_for_relation(right_relation)
            if right_direction == "forward"
            else ontology.range_for_relation(right_relation)
        )
        if not left_output or not right_input:
            continue
        known_pairs += 1
        if left_output == right_input:
            compatible += 1.0
            continue
        left_types = set(ontology.supertypes(left_output))
        right_types = set(ontology.supertypes(right_input))
        if left_types & right_types:
            compatible += 0.5
    return 0.0 if not known_pairs else 0.03 * (compatible / known_pairs)


def _lexical_coverage(query: str, candidate: str) -> float:
    query_tokens = set(re.findall(r"[a-z0-9]+", query.casefold()))
    candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate.casefold()))
    if not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(candidate_tokens)


def _gold_entity_candidates(
    surface: str,
    entities: dict[str, str],
    *,
    limit: int,
) -> list[EntityCandidate]:
    def normalized_name(value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", str(value).casefold())
        accentless = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        return " ".join(accentless.split())

    surface_norm = normalized_name(surface)
    candidates: list[EntityCandidate] = []
    for entity_id, label in entities.items():
        label_norm = normalized_name(label)
        if not surface_norm or not label_norm:
            continue
        if surface_norm == label_norm:
            score = 1.05
        elif surface_norm in label_norm or label_norm in surface_norm:
            score = 0.9
        else:
            continue
        candidates.append(EntityCandidate(str(entity_id), str(label), score, "golden"))
    candidates.sort(key=lambda item: (-item.score, item.label, item.entity_id))
    return candidates[: max(1, limit)]


def _all_gold_entity_candidates(
    entities: dict[str, str],
    *,
    limit: int,
) -> list[EntityCandidate]:
    """Use the dataset topic entities when the model's surface is a variant."""
    candidates = [
        EntityCandidate(str(entity_id), str(label), 0.85, "golden")
        for entity_id, label in entities.items()
        if str(entity_id).strip() and str(label).strip()
    ]
    candidates.sort(key=lambda item: (item.label, item.entity_id))
    return candidates[: max(1, limit)]


def _question_key(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _step_query(question: str, context: str, path: dict[str, Any], step: dict[str, Any]) -> str:
    label = " ".join(str(value) for value in step.get("relation_label", []))
    return " ".join(value for value in (question, context, str(path.get("goal", "")), label) if value)


def _focused_step_query(
    question: str,
    path: dict[str, Any],
    step: dict[str, Any],
    step_index: int,
) -> str:
    """Question/goal query without feeding the predicted predicate back in.

    Used only by a failed-exact retrieval challenger. The legacy success lane
    retains ``_step_query`` and therefore keeps its historical ranking.
    """
    goal = str(path.get("goal", "")).strip()
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", goal)
        if part.strip()
    ]
    local_goal = sentences[step_index] if step_index < len(sentences) else goal
    return " ".join(
        dict.fromkeys(
            value
            for value in (str(question).strip(), local_goal)
            if value
        )
    )


def _stepwise_step_query(
    question: str,
    context: str,
    path: dict[str, Any],
    step: dict[str, Any],
    step_index: int,
) -> str:
    """Build a focused ranking query for one recursive hop.

    The legacy scorer intentionally concatenates every decomposition.  That
    is useful for a two-hop global ranking but can make a three-hop query's
    first-hop words dominate all later hops.  Long-path goals are normally
    sentence-separated (one sentence/question per hop), so select the
    corresponding sentence and fall back to the complete goal when a model
    emits an unsplit form.
    """
    goal = str(path.get("goal", "")).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", goal) if part.strip()]
    local_goal = sentences[step_index] if step_index < len(sentences) else goal
    label = " ".join(str(value) for value in step.get("relation_label", []))
    # Keep the original question only when no usable hop-specific sentence
    # exists; otherwise it often repeats the anchor entity at every depth.
    parts = (local_goal or context or question, label)
    return " ".join(value for value in parts if value)


def _cvt_step_query(question: str, context: str, path: dict[str, Any]) -> str:
    return " ".join(value for value in (question, context, str(path.get("goal", ""))) if value)


def _row_reaches_cvt(row: dict[str, Any]) -> bool:
    value = row.get("reaches_cvt", False)
    return value is True or str(value).casefold() in {"1", "true", "yes"}


def _row_next_is_entity(row: dict[str, Any]) -> bool:
    """Use explicit RDF type metadata when available, with a test-double fallback."""
    if "next_node_is_entity" in row:
        value = row.get("next_node_is_entity")
        return value is True or str(value).casefold() in {"1", "true", "yes", "uri"}
    return _is_entity_id(str(row.get("next_node_id", "")).strip())


def _row_next_type(row: dict[str, Any]) -> str:
    """Normalize endpoint type metadata from ``expand_hop`` responses."""
    value = str(
        row.get("next_node_type", row.get("neighbor_kind", ""))
    ).strip().casefold()
    if value in {"uri", "iri", "entity"}:
        return "uri"
    if value in {"bnode", "blank", "blank_node"}:
        return "bnode"
    if value in {"literal", "lit"}:
        return "literal"
    if _row_next_is_entity(row):
        return "uri"
    return "literal"


def _can_continue_from_row(row: dict[str, Any], *, next_direction: str) -> bool:
    """Whether an endpoint can legally be used as the next hop's subject/object."""
    endpoint_type = _row_next_type(row)
    if endpoint_type in {"uri", "bnode"}:
        return True
    # RDF literals cannot be subjects, but a backward hop can still use one as
    # the object of a triple (this occurs in numeric/CVT paths).
    return str(next_direction).casefold() == "backward"


def _is_immediate_backtrack(
    state: _StepwiseState,
    *,
    relation_id: str,
    direction: str,
    next_node_id: str,
    next_node_type: str,
) -> bool:
    """Detect an exact URI edge reversal without banning legitimate CVT loops."""
    if not state.hops or state.node_type not in {"uri", "bnode"}:
        return False
    if next_node_type not in {"uri", "bnode"}:
        return False
    previous = state.hops[-1]
    previous_relation = str(previous.get("relation_id", "")).strip()
    previous_direction = str(previous.get("direction", "")).strip().casefold()
    previous_node = str(previous.get("current_node_id", "")).strip()
    return (
        relation_id == previous_relation
        and direction in {"forward", "backward"}
        and previous_direction in {"forward", "backward"}
        and direction != previous_direction
        and next_node_id == previous_node
    )


def _path_score(entity: EntityCandidate, relation_scores: list[float]) -> float:
    relation_score = sum(relation_scores) / max(1, len(relation_scores))
    return (0.2 * min(1.0, max(0.0, entity.score))) + (0.8 * relation_score)


def _consistent_combinations(
    paths: list[dict[str, Any]],
    variants: list[list[_PathVariant]],
    limit: int,
) -> list[tuple[_PathVariant, ...]]:
    current: list[tuple[_PathVariant, ...]] = [tuple()]
    for path, path_variants in zip(paths, variants):
        expanded: list[tuple[_PathVariant, ...]] = []
        anchor_id = str(path.get("anchor_ref", ""))
        for prefix, candidate in product(current, path_variants):
            prior = next(
                (item.entity.entity_id for old_path, item in zip(paths, prefix) if str(old_path.get("anchor_ref", "")) == anchor_id),
                None,
            )
            if prior is not None and prior != candidate.entity.entity_id:
                continue
            expanded.append((*prefix, candidate))
        expanded.sort(key=lambda combo: (-sum(item.score for item in combo), _variant_tuple_key(combo)))
        current = expanded[: max(limit, limit * 2)]
    return current[: max(1, limit)]


def _candidate_key(candidate: GroundedSemanticCandidate) -> str:
    entities = tuple(sorted((key, value.entity_id) for key, value in candidate.anchor_bindings.items()))
    return repr((entities, candidate.compose_input.get("semantic_paths", [])))


def _candidate_uses_retrieval_repair(
    candidate: GroundedSemanticCandidate,
) -> bool:
    return any(
        str(hop.get("retrieval_mode", "exact")) != "exact"
        or bool(hop.get("direction_repaired"))
        for path_hops in candidate.provenance.get("hops", [])
        for hop in path_hops
        if isinstance(hop, dict)
    )


def _variant_key(variant: _PathVariant) -> str:
    return repr((variant.entity.entity_id, variant.relation_bindings))


def _variant_tuple_key(variants: tuple[_PathVariant, ...]) -> str:
    return repr(tuple(_variant_key(item) for item in variants))
