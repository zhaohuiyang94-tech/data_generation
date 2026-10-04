"""Bounded endpoint path retrieval for an already-failed pipeline row.

This module is deliberately not wired into :mod:`pipeline`.  A caller may use
it after normal grounding/execution has failed, without making another chat
model call.  It enumerates real one- and two-hop paths from the supplied
anchor bindings, rejects schema-inconsistent joins, and retains a small ranked
beam.

There is intentionally no dataset index, Gold answer, question pattern, or
relation allow-list in this implementation.  The only persistent state is an
endpoint-result cache keyed by ``(entity_id, directions)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading
import time
from typing import Any, Mapping, Sequence

from .clients import EmbeddingRanker, KnowledgeGraph
from .contracts import EntityCandidate
from .ontology import FreebaseOntology


__all__ = ["FailurePathCandidate", "FailurePathRetriever"]


_RELATION_ID_RE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DIRECTION_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("forward",),
    ("backward",),
    ("forward", "forward"),
    ("forward", "backward"),
    ("backward", "forward"),
    ("backward", "backward"),
)
@dataclass(frozen=True, slots=True)
class FailurePathCandidate:
    """One endpoint-verified relation sequence for a bound anchor."""

    anchor_ref: str
    entity_id: str
    relation_ids: tuple[str, ...]
    directions: tuple[str, ...]
    score: float
    schema_status: str
    intermediate_type: str = ""
    cvt_continuity: bool = False

    @property
    def depth(self) -> int:
        return len(self.relation_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_ref": self.anchor_ref,
            "entity_id": self.entity_id,
            "relation_ids": list(self.relation_ids),
            "directions": list(self.directions),
            "depth": self.depth,
            "score": self.score,
            "schema_status": self.schema_status,
            "intermediate_type": self.intermediate_type,
            "cvt_continuity": self.cvt_continuity,
        }


@dataclass(frozen=True, slots=True)
class _SchemaCheck:
    accepted: bool
    status: str
    intermediate_type: str = ""
    cvt_continuity: bool = False
    score_adjustment: float = 0.0


@dataclass(frozen=True, slots=True)
class _CachedPathRows:
    rows: tuple[tuple[str, ...], ...]
    error: str = ""


@dataclass(frozen=True, slots=True)
class _UnrankedPath:
    anchor_ref: str
    entity_id: str
    relation_ids: tuple[str, ...]
    directions: tuple[str, ...]
    schema: _SchemaCheck


class FailurePathRetriever:
    """Retrieve a bounded local path beam after normal retrieval has failed.

    ``retrieve`` explores all direction patterns through ``max_depth`` (one
    or two).  The endpoint is therefore the authority on path existence;
    ontology metadata is used only for type/CVT continuity and ranking.  At
    most ``top_k`` paths across the enabled depths are retained for each
    anchor.  ``top_k`` is capped at 24 so accidental integration cannot
    silently create an unbounded graph beam.

    The class performs no failure detection itself.  Keeping that policy at
    the caller makes it impossible for this isolated module to alter a normal
    successful lane before it is explicitly integrated.
    """

    def __init__(
        self,
        *,
        ontology: FreebaseOntology,
        kg: KnowledgeGraph,
        ranker: EmbeddingRanker,
        top_k: int = 24,
        query_limit: int = 10_000,
        max_depth: int = 2,
    ) -> None:
        self.ontology = ontology
        self.kg = kg
        self.ranker = ranker
        self.top_k = min(24, max(1, int(top_k)))
        self.query_limit = min(10_000, max(self.top_k, int(query_limit)))
        self.max_depth = min(2, max(1, int(max_depth)))
        self.direction_patterns = tuple(
            pattern
            for pattern in _DIRECTION_PATTERNS
            if len(pattern) <= self.max_depth
        )
        self._cache: dict[tuple[str, tuple[str, ...]], _CachedPathRows] = {}
        self._cache_lock = threading.Lock()
        self._inflight: dict[
            tuple[str, tuple[str, ...]],
            threading.Event,
        ] = {}

    def clear_cache(self) -> None:
        """Discard endpoint path rows cached by this retriever instance."""

        with self._cache_lock:
            self._cache.clear()

    def retrieve(
        self,
        *,
        question: str,
        decompositions: Sequence[str],
        anchor_bindings: Mapping[str, Any],
    ) -> tuple[list[FailurePathCandidate], dict[str, Any]]:
        started = time.monotonic()
        context = " ".join(
            value
            for value in (
                str(question).strip(),
                *(str(item).strip() for item in decompositions),
            )
            if value
        )
        anchors = _normalize_anchor_bindings(anchor_bindings)
        diagnostics: dict[str, Any] = {
            "strategy": "failure_endpoint_path_retrieval",
            "model_calls": 0,
            "llm_calls": 0,
            "ranker_call_count": 0,
            "top_k": self.top_k,
            "query_limit": self.query_limit,
            "anchor_count": len(anchors),
            "max_depth": self.max_depth,
            "direction_pattern_count": len(self.direction_patterns),
            "query_count": 0,
            "cache_hits": 0,
            "query_elapsed_seconds": 0.0,
            "ranking_elapsed_seconds": 0.0,
            "raw_candidate_count": 0,
            "deduplicated_candidate_count": 0,
            "schema_rejected_count": 0,
            "retained_candidate_count": 0,
            "errors": [],
            "anchors": [],
        }
        if not anchors:
            diagnostics["code"] = "NO_ANCHOR_BINDINGS"
            diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
            return [], diagnostics

        unranked: list[_UnrankedPath] = []
        seen: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
        for anchor_ref, entity_id in anchors:
            anchor_diag: dict[str, Any] = {
                "anchor_ref": anchor_ref,
                "entity_id": entity_id,
                "queries": [],
            }
            for directions in self.direction_patterns:
                rows, query_diag = self._path_rows(entity_id, directions)
                diagnostics["query_count"] += int(not query_diag["cache_hit"])
                diagnostics["cache_hits"] += int(query_diag["cache_hit"])
                diagnostics["query_elapsed_seconds"] += float(
                    query_diag["elapsed_seconds"]
                )
                diagnostics["raw_candidate_count"] += len(rows)
                anchor_diag["queries"].append(query_diag)
                if query_diag.get("error"):
                    diagnostics["errors"].append(
                        {
                            "anchor_ref": anchor_ref,
                            "entity_id": entity_id,
                            "directions": list(directions),
                            "error": query_diag["error"],
                        }
                    )
                for relation_ids in rows:
                    key = (anchor_ref, entity_id, directions, relation_ids)
                    if key in seen:
                        continue
                    seen.add(key)
                    schema = self._schema_check(relation_ids, directions)
                    if not schema.accepted:
                        diagnostics["schema_rejected_count"] += 1
                        continue
                    unranked.append(
                        _UnrankedPath(
                            anchor_ref=anchor_ref,
                            entity_id=entity_id,
                            relation_ids=relation_ids,
                            directions=directions,
                            schema=schema,
                        )
                    )
            diagnostics["anchors"].append(anchor_diag)

        diagnostics["deduplicated_candidate_count"] = len(unranked)
        ranking_started = time.monotonic()
        diagnostics["ranker_call_count"] = int(bool(unranked))
        candidates, ranker_error = self._rank(context, unranked)
        diagnostics["ranking_elapsed_seconds"] = round(
            time.monotonic() - ranking_started,
            6,
        )
        if ranker_error:
            diagnostics["errors"].append(
                {"stage": "embedding_ranking", "error": ranker_error}
            )

        # Bound the complete one/two-hop beam per anchor.  Anchors remain
        # separate because starving a secondary constraint anchor would make
        # a later deterministic intersection impossible.
        retained: list[FailurePathCandidate] = []
        grouped: dict[str, list[FailurePathCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.anchor_ref, []).append(candidate)
        for key in sorted(grouped):
            values = sorted(grouped[key], key=_candidate_sort_key)
            retained.extend(values[: self.top_k])
        retained.sort(key=_candidate_sort_key)

        diagnostics["retained_candidate_count"] = len(retained)
        diagnostics["retained_by_depth"] = {
            str(depth): sum(candidate.depth == depth for candidate in retained)
            for depth in (1, 2)
        }
        diagnostics["query_elapsed_seconds"] = round(
            float(diagnostics["query_elapsed_seconds"]),
            6,
        )
        diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 6)
        diagnostics["code"] = "OK" if retained else "NO_SCHEMA_VALID_PATH"
        return retained, diagnostics

    def _path_rows(
        self,
        entity_id: str,
        directions: tuple[str, ...],
    ) -> tuple[tuple[tuple[str, ...], ...], dict[str, Any]]:
        key = (entity_id, directions)
        while True:
            owner = False
            with self._cache_lock:
                cached = self._cache.get(key)
                if cached is not None:
                    return cached.rows, {
                        "directions": list(directions),
                        "cache_hit": True,
                        "elapsed_seconds": 0.0,
                        "row_count": len(cached.rows),
                        "error": "",
                    }
                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    owner = True
            if owner:
                break
            event.wait()

        started = time.monotonic()
        try:
            error = ""
            raw_rows: Any = []
            try:
                raw_rows = self.kg.path_hops(
                    entity_id,
                    list(directions),
                    limit=self.query_limit,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            rows: list[tuple[str, ...]] = []
            seen: set[tuple[str, ...]] = set()
            for raw in raw_rows if isinstance(raw_rows, list) else []:
                if not isinstance(raw, dict):
                    continue
                values = raw.get("relation_ids", raw.get("relations", []))
                if not isinstance(values, (list, tuple)) or len(values) != len(directions):
                    continue
                relation_ids = tuple(str(value).strip() for value in values)
                if not all(_RELATION_ID_RE.fullmatch(value) for value in relation_ids):
                    continue
                if relation_ids in seen:
                    continue
                seen.add(relation_ids)
                rows.append(relation_ids)
            rows.sort()
            result = _CachedPathRows(tuple(rows), "")
            # Endpoint failures are transient and must never become a durable
            # empty-path fact for later questions or worker threads.
            if not error:
                with self._cache_lock:
                    self._cache[key] = result
            elapsed = time.monotonic() - started
            return result.rows, {
                "directions": list(directions),
                "cache_hit": False,
                "elapsed_seconds": round(elapsed, 6),
                "row_count": len(result.rows),
                "error": error,
            }
        finally:
            with self._cache_lock:
                event = self._inflight.pop(key, None)
                if event is not None:
                    event.set()

    def _schema_check(
        self,
        relation_ids: tuple[str, ...],
        directions: tuple[str, ...],
    ) -> _SchemaCheck:
        if len(relation_ids) == 1:
            relation_id = relation_ids[0]
            known = bool(
                self.ontology.domain_for_relation(relation_id)
                or self.ontology.range_for_relation(relation_id)
            )
            return _SchemaCheck(
                True,
                "known_one_hop" if known else "unknown_one_hop",
                score_adjustment=0.01 if known else -0.04,
            )

        left_output = _output_type(
            self.ontology,
            relation_ids[0],
            directions[0],
        )
        right_input = _input_type(
            self.ontology,
            relation_ids[1],
            directions[1],
        )
        if not left_output or not right_input:
            return _SchemaCheck(
                True,
                "unknown_bridge",
                intermediate_type=left_output or right_input,
                score_adjustment=-0.06,
            )

        cvt_like = _is_cvt_like(self.ontology, left_output) or _is_cvt_like(
            self.ontology,
            right_input,
        )
        if left_output == right_input:
            return _SchemaCheck(
                True,
                "cvt_exact" if cvt_like else "exact_bridge",
                intermediate_type=left_output,
                cvt_continuity=cvt_like,
                score_adjustment=0.05 if cvt_like else 0.03,
            )

        # A mediator/CVT must join on the exact anonymous-record type.  Using
        # a broad ancestor here creates the familiar parallel-CVT false join.
        if cvt_like:
            return _SchemaCheck(
                False,
                "cvt_type_mismatch",
                intermediate_type=f"{left_output}!={right_input}",
            )

        if _types_compatible(self.ontology, left_output, right_input):
            return _SchemaCheck(
                True,
                "subtype_bridge",
                intermediate_type=f"{left_output}~{right_input}",
                score_adjustment=0.015,
            )
        return _SchemaCheck(
            False,
            "type_mismatch",
            intermediate_type=f"{left_output}!={right_input}",
        )

    def _rank(
        self,
        context: str,
        paths: list[_UnrankedPath],
    ) -> tuple[list[FailurePathCandidate], str]:
        if not paths:
            return [], ""
        relation_ids = sorted(
            {relation_id for path in paths for relation_id in path.relation_ids}
        )
        relation_texts = [_relation_text(value) for value in relation_ids]
        ranker_error = ""
        try:
            embedding_scores = self.ranker.score(context, relation_texts)
            if len(embedding_scores) != len(relation_ids):
                raise ValueError("ranker returned a score vector with the wrong length")
            embedding_by_relation = {
                relation_id: float(score)
                for relation_id, score in zip(relation_ids, embedding_scores)
            }
        except (RuntimeError, TypeError, ValueError) as exc:
            ranker_error = f"{type(exc).__name__}: {exc}"
            embedding_by_relation = {relation_id: 0.0 for relation_id in relation_ids}

        lexical_by_relation = {
            relation_id: _lexical_coverage(context, _relation_text(relation_id))
            for relation_id in relation_ids
        }
        output: list[FailurePathCandidate] = []
        for path in paths:
            hop_scores = [
                (0.7 * embedding_by_relation.get(relation_id, 0.0))
                + (0.3 * lexical_by_relation.get(relation_id, 0.0))
                for relation_id in path.relation_ids
            ]
            score = (
                sum(hop_scores) / max(1, len(hop_scores))
                + path.schema.score_adjustment
            )
            output.append(
                FailurePathCandidate(
                    anchor_ref=path.anchor_ref,
                    entity_id=path.entity_id,
                    relation_ids=path.relation_ids,
                    directions=path.directions,
                    score=score,
                    schema_status=path.schema.status,
                    intermediate_type=path.schema.intermediate_type,
                    cvt_continuity=path.schema.cvt_continuity,
                )
            )
        return output, ranker_error


def _normalize_anchor_bindings(
    values: Mapping[str, Any],
) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_anchor_ref, raw_value in values.items():
        anchor_ref = str(raw_anchor_ref).strip()
        entity_id = ""
        if isinstance(raw_value, EntityCandidate):
            entity_id = str(raw_value.entity_id).strip()
        elif isinstance(raw_value, Mapping):
            entity_id = str(
                raw_value.get(
                    "entity_id",
                    raw_value.get("id", raw_value.get("value", "")),
                )
            ).strip()
        else:
            entity_id = str(raw_value).strip()
        key = (anchor_ref, entity_id)
        if not anchor_ref or not entity_id or key in seen:
            continue
        seen.add(key)
        output.append(key)
    output.sort()
    return output


def _input_type(
    ontology: FreebaseOntology,
    relation_id: str,
    direction: str,
) -> str:
    if direction == "forward":
        return str(ontology.domain_for_relation(relation_id))
    return str(ontology.range_for_relation(relation_id))


def _output_type(
    ontology: FreebaseOntology,
    relation_id: str,
    direction: str,
) -> str:
    if direction == "forward":
        return str(ontology.range_for_relation(relation_id))
    return str(ontology.domain_for_relation(relation_id))


def _types_compatible(
    ontology: FreebaseOntology,
    left: str,
    right: str,
) -> bool:
    if left == right:
        return True
    left_parents = set(ontology.supertypes(left))
    right_parents = set(ontology.supertypes(right))
    return right in left_parents or left in right_parents


def _is_cvt_like(ontology: FreebaseOntology, type_id: str) -> bool:
    """Infer mediator-like schema types without a relation/type allow-list."""

    explicit = getattr(ontology, "is_cvt_type", None)
    if callable(explicit):
        return bool(explicit(type_id))
    # Primitive/value types use Freebase's ``type.*`` schema namespace.  This
    # namespace-level rule avoids maintaining a hand-authored type list.
    if not type_id or type_id.startswith("type."):
        return False
    # Ordinary named entities inherit ``common.topic`` in fb_types; anonymous
    # mediator/value records generally do not.  This is intentionally a
    # schema-derived fallback rather than a namespace list.
    return "common.topic" not in set(ontology.supertypes(type_id))


def _relation_text(relation_id: str) -> str:
    return " ".join(
        segment.replace("_", " ")
        for segment in str(relation_id).split(".")
        if segment
    )


def _lexical_coverage(query: str, candidate: str) -> float:
    query_tokens = set(_TOKEN_RE.findall(str(query).casefold()))
    candidate_tokens = set(_TOKEN_RE.findall(str(candidate).casefold()))
    if not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(candidate_tokens)


def _candidate_sort_key(candidate: FailurePathCandidate) -> tuple[Any, ...]:
    return (
        -candidate.score,
        candidate.anchor_ref,
        candidate.depth,
        candidate.directions,
        candidate.relation_ids,
        candidate.entity_id,
    )
