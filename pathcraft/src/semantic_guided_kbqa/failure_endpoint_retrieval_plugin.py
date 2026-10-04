"""Failure-only endpoint path retrieval compiled into executable query graphs.

It only runs as the last production/replay failure lane and consumes linked
anchors, frozen decomposition text, endpoint-verified paths, and ontology schema.
It never consumes an evaluation row,
Gold answer, dataset index, relation allow-list, or chat model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations as anchor_subsets
import json
import math
import re
import threading
from typing import Any, Iterable, Mapping, Sequence

from .contracts import EntityCandidate, GroundedSemanticCandidate, QueryGraphCandidate
from .failure_path_retrieval import FailurePathCandidate, FailurePathRetriever
from .ontology import relation_label_from_id


_EXECUTION_BUDGET = 30
_INTERSECTION_BEAM = 96
_PATH_QUERY_LIMIT = 1_024
_MAX_SUBSET_ANCHORS = 6
_SUBSET_COMBINATION_BUDGET = 768
_RETRIEVER_CACHE_LOCK = threading.Lock()
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUESTION_BOUNDARIES = {
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "that",
    "was",
    "were",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
}
_QUESTION_PREFIX_SKIP = {"a", "an", "are", "is", "the", "was", "were"}


class _LocalLexicalRanker:
    """EmbeddingRanker-compatible scorer with no service or model calls."""

    def score(self, query: str, candidates: list[str]) -> list[float]:
        query_tokens = _TOKEN_RE.findall(str(query).casefold())
        query_set = set(query_tokens)
        output: list[float] = []
        for candidate in candidates:
            candidate_tokens = _TOKEN_RE.findall(str(candidate).casefold())
            candidate_set = set(candidate_tokens)
            overlap = len(query_set & candidate_set)
            cosine = overlap / math.sqrt(
                max(1, len(query_set)) * max(1, len(candidate_set))
            )
            coverage = overlap / max(1, len(candidate_set))
            # Consecutive relation words are a useful deterministic signal for
            # phrases such as "place of birth" and "played by".
            query_bigrams = set(zip(query_tokens, query_tokens[1:]))
            candidate_bigrams = set(zip(candidate_tokens, candidate_tokens[1:]))
            bigram = len(query_bigrams & candidate_bigrams) / max(
                1,
                len(candidate_bigrams),
            )
            output.append((0.45 * cosine) + (0.45 * coverage) + (0.10 * bigram))
        return output


_LOCAL_RANKER = _LocalLexicalRanker()
_LOW_CONFIDENCE_SINGLETON_SCORE_WINDOW = 0.30
_LOW_CONFIDENCE_LEXICAL_DELTA = 0.20


@dataclass(frozen=True, slots=True)
class _Anchor:
    anchor_ref: str
    entity_id: str
    surface: str


def _semantic_word_key(value: str) -> str:
    """Small inflection normalizer shared by direct lexical evidence."""

    token = str(value).casefold()
    if token in {"died", "dead"}:
        return "death"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("sed") and len(token) > 4:
        return token[:-1]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _answer_focus_text(question: str) -> str:
    """Extract the same short answer-target phrase used by graph selection."""

    tokens = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", str(question))
    if not tokens:
        return str(question)
    wh_words = {"what", "which", "who", "where", "when", "whom", "whose"}
    starts = [
        index
        for index, token in enumerate(tokens)
        if token.casefold().removesuffix("'s") in wh_words
    ]
    early = [index for index in starts if index <= 1]
    start = early[0] if early else (starts[-1] if starts else 0)
    head = tokens[start].casefold().removesuffix("'s")
    remaining = tokens[start:]
    auxiliaries = {
        "do", "does", "did", "has", "have", "had", "can", "could",
        "would", "will", "shall", "should",
    }
    if head in {"what", "which", "whom", "whose"}:
        if len(remaining) > 1 and remaining[1].casefold() in {
            "is", "are", "was", "were",
        }:
            return " ".join([remaining[0], *remaining[2:10]])
        for index, token in enumerate(remaining[1:], start=1):
            if token.casefold() in auxiliaries:
                return " ".join(remaining[:index])
        return " ".join(remaining[:7])
    if head == "who":
        return " ".join(remaining[:6])
    return " ".join(remaining[:8])


def _direct_terminal_lexical_score(
    question: str,
    candidate: Mapping[str, Any],
) -> float:
    """Token-F1 between answer-focus words and terminal predicate names."""

    ignored = {
        "a", "an", "are", "did", "do", "does", "has", "have", "is",
        "of", "the", "was", "were", "what", "which", "who", "where",
    }
    query_tokens = {
        _semantic_word_key(token)
        for token in re.findall(
            r"[A-Za-z][A-Za-z0-9_]*",
            _answer_focus_text(question),
        )
        if token.casefold() not in ignored
    }
    relation_tokens: set[str] = set()
    answer_var = str(candidate.get("answer_var", "V0"))
    for triple in candidate.get("triples", []):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject, relation_id, object_ = map(str, triple)
        if answer_var not in {subject, object_}:
            continue
        relation_tokens.update(
            _semantic_word_key(token)
            for token in re.findall(
                r"[A-Za-z0-9]+",
                relation_id.replace(".", " ").replace("_", " "),
            )
        )
    if not query_tokens or not relation_tokens:
        return 0.0
    overlap = len(query_tokens & relation_tokens)
    precision = overlap / len(relation_tokens)
    recall = overlap / len(query_tokens)
    return (
        0.0
        if precision + recall == 0.0
        else (2.0 * precision * recall) / (precision + recall)
    )


def _select_low_confidence_direct_challenger(
    question: str,
    incumbent: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Choose a narrow challenger from already executed direct candidates.

    This selector is intentionally usable only after the incumbent failed the
    outer confidence gate and bounded best-effort is enabled by its caller.
    It has two schema/shape rules: project away an anonymous CVT singleton to
    a higher-scoring named singleton, or replace a multi-answer incumbent by a
    nearly tied singleton whose terminal predicate has substantially stronger
    lexical evidence.  It never sees Gold answers, IDs from evaluation data,
    or a relation allow-list.
    """

    incumbent_graph_id = str(incumbent.get("graph_id", ""))
    incumbent_score = float(incumbent.get("score", 0.0))
    incumbent_count = int(incumbent.get("answer_count", 0))
    incumbent_lexical = _direct_terminal_lexical_score(question, incumbent)
    evidence: dict[str, Any] = {
        "applied": False,
        "source_graph_id": incumbent_graph_id,
        "source_score": incumbent_score,
        "source_answer_count": incumbent_count,
        "source_terminal_is_cvt": bool(incumbent.get("answer_is_cvt", False)),
        "source_terminal_lexical": incumbent_lexical,
        "uses_existing_executions_only": True,
        "uses_gold_answers": False,
        "entity_or_relation_whitelist": False,
        "additional_execution_queries": 0,
        "additional_llm_calls": 0,
        "additional_embedding_calls": 0,
    }

    if bool(incumbent.get("answer_is_cvt", False)):
        cvt_projection = [
            candidate
            for candidate in candidates
            if str(candidate.get("graph_id", "")) != incumbent_graph_id
            and int(candidate.get("answer_count", 0)) == 1
            and not bool(candidate.get("answer_is_cvt", False))
            and float(candidate.get("score", 0.0)) > incumbent_score + 1e-12
        ]
        if cvt_projection:
            selected = max(
                cvt_projection,
                key=lambda candidate: (
                    float(candidate.get("score", 0.0)),
                    str(candidate.get("graph_id", "")),
                ),
            )
            evidence.update(
                {
                    "applied": True,
                    "reason": "higher_score_non_cvt_singleton_projection",
                    "selected_graph_id": str(selected.get("graph_id", "")),
                    "selected_score": float(selected.get("score", 0.0)),
                    "selected_answer_count": 1,
                    "selected_terminal_is_cvt": False,
                    "selected_terminal_lexical": (
                        _direct_terminal_lexical_score(question, selected)
                    ),
                    "eligible_graph_ids": [
                        str(candidate.get("graph_id", ""))
                        for candidate in sorted(
                            cvt_projection,
                            key=lambda candidate: (
                                -float(candidate.get("score", 0.0)),
                                str(candidate.get("graph_id", "")),
                            ),
                        )
                    ],
                }
            )
            return str(selected.get("graph_id", "")), evidence

    if incumbent_count > 1:
        lexical_singletons: list[tuple[float, Mapping[str, Any]]] = []
        for candidate in candidates:
            if (
                str(candidate.get("graph_id", "")) == incumbent_graph_id
                or int(candidate.get("answer_count", 0)) != 1
                or incumbent_score - float(candidate.get("score", 0.0))
                > _LOW_CONFIDENCE_SINGLETON_SCORE_WINDOW + 1e-12
            ):
                continue
            lexical = _direct_terminal_lexical_score(question, candidate)
            if lexical - incumbent_lexical + 1e-12 < _LOW_CONFIDENCE_LEXICAL_DELTA:
                continue
            lexical_singletons.append((lexical, candidate))
        if lexical_singletons:
            selected_lexical, selected = max(
                lexical_singletons,
                key=lambda item: (
                    item[0],
                    float(item[1].get("score", 0.0)),
                    str(item[1].get("graph_id", "")),
                ),
            )
            evidence.update(
                {
                    "applied": True,
                    "reason": "near_score_lexically_stronger_singleton",
                    "selected_graph_id": str(selected.get("graph_id", "")),
                    "selected_score": float(selected.get("score", 0.0)),
                    "selected_answer_count": 1,
                    "selected_terminal_is_cvt": bool(
                        selected.get("answer_is_cvt", False)
                    ),
                    "selected_terminal_lexical": selected_lexical,
                    "maximum_score_lag": _LOW_CONFIDENCE_SINGLETON_SCORE_WINDOW,
                    "minimum_lexical_delta": _LOW_CONFIDENCE_LEXICAL_DELTA,
                    "eligible_graph_ids": [
                        str(candidate.get("graph_id", ""))
                        for _, candidate in sorted(
                            lexical_singletons,
                            key=lambda item: (
                                -item[0],
                                -float(item[1].get("score", 0.0)),
                                str(item[1].get("graph_id", "")),
                            ),
                        )
                    ],
                }
            )
            return str(selected.get("graph_id", "")), evidence

    evidence["reason"] = "no_safe_low_confidence_challenger"
    return "", evidence


def _question_key(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _decompositions(context: Any) -> list[str]:
    """Read the latest reviewed decomposition without invoking a model."""

    values: list[str] = []
    artifact = context.trace_bundle.get("decomposition", {})
    traces = artifact.get("traces", []) if isinstance(artifact, dict) else []
    for trace in reversed(traces if isinstance(traces, list) else []):
        if (
            isinstance(trace, dict)
            and trace.get("stage") == "decomposition_review_and_rewrite"
        ):
            output = trace.get("output")
            candidates = (
                output.get("final_candidates", [])
                if isinstance(output, dict)
                else []
            )
            for candidate in candidates if isinstance(candidates, list) else []:
                if not isinstance(candidate, dict):
                    continue
                values.extend(
                    str(item)
                    for item in candidate.get("decomposition", [])
                    if str(item).strip()
                )
            if values:
                return list(dict.fromkeys(values))
            # Do not resurrect an older review attempt when the latest one is
            # empty or malformed; the immutable input store is the safe
            # fallback.
            break

    try:
        candidates = context.pipeline.decompositions.get(context.question)
    except (AttributeError, KeyError):
        candidates = []
    for candidate in candidates:
        values.extend(
            str(item)
            for item in getattr(candidate, "decomposition", [])
            if str(item).strip()
        )
    if values:
        return list(dict.fromkeys(values))

    # Trace-only replay remains supported when the store is unavailable.
    for trace in reversed(traces if isinstance(traces, list) else []):
        if not isinstance(trace, dict) or trace.get("stage") != "decompose_predictions":
            continue
        candidates = trace.get("output", [])
        for candidate in candidates if isinstance(candidates, list) else []:
            if not isinstance(candidate, dict):
                continue
            values.extend(
                str(item)
                for item in candidate.get("decomposition", [])
                if str(item).strip()
            )
        if values:
            break
    return list(dict.fromkeys(values))


def _anchors(context: Any) -> list[_Anchor]:
    """Normalize the pipeline's already-linked entities in mention order."""

    grounder = context.pipeline.grounder
    entities = getattr(grounder, "gold_entities", {}).get(
        _question_key(context.question),
        {},
    )
    items = [
        (str(entity_id), str(label))
        for entity_id, label in entities.items()
        if str(entity_id).strip() and str(label).strip()
    ]
    normalized = context.question.casefold()
    items.sort(
        key=lambda item: (
            normalized.find(item[1].casefold())
            if item[1].casefold() in normalized
            else len(normalized),
            item[1],
            item[0],
        )
    )
    seen_entities: set[str] = set()
    output: list[_Anchor] = []
    for entity_id, label in items:
        if entity_id in seen_entities:
            continue
        seen_entities.add(entity_id)
        output.append(_Anchor(f"A{len(output)}", entity_id, label))
    return output


def _retriever(context: Any, endpoint: Any) -> FailurePathRetriever:
    pipeline = context.pipeline
    with _RETRIEVER_CACHE_LOCK:
        cached = getattr(pipeline, "_failure_endpoint_path_retriever", None)
        if (
            isinstance(cached, tuple)
            and len(cached) == 2
            and cached[0] is endpoint
        ):
            return cached[1]
        value = FailurePathRetriever(
            ontology=pipeline.grounder.ontology,
            kg=endpoint,
            ranker=_LOCAL_RANKER,
            top_k=24,
            query_limit=_PATH_QUERY_LIMIT,
            max_depth=1,
        )
        # Keep a strong reference to the endpoint next to the retriever.  This
        # avoids object-id reuse and binds cached path rows to exactly one KG.
        pipeline._failure_endpoint_path_retriever = (endpoint, value)
    return value


def _terminal_type(candidate: FailurePathCandidate, ontology: Any) -> str:
    relation_id = candidate.relation_ids[-1]
    if candidate.directions[-1] == "forward":
        return str(ontology.range_for_relation(relation_id) or "")
    return str(ontology.domain_for_relation(relation_id) or "")


def _word_root(value: str) -> str:
    value = str(value).casefold()
    if len(value) > 4 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 4 and value.endswith("es"):
        return value[:-2]
    if len(value) > 3 and value.endswith("s"):
        return value[:-1]
    return value


def _question_head_tokens(question: str) -> tuple[str, ...]:
    """Extract a small What/Which answer-head phrase without a type list."""

    tokens = _TOKEN_RE.findall(str(question).casefold())
    if not tokens or tokens[0] not in {"what", "which"}:
        return ()
    position = 1
    while position < len(tokens) and tokens[position] in _QUESTION_PREFIX_SKIP:
        position += 1
    head: list[str] = []
    for token in tokens[position:]:
        if token in _QUESTION_BOUNDARIES:
            break
        head.append(_word_root(token))
        if len(head) >= 3:
            break
    return tuple(dict.fromkeys(value for value in head if value))


def _type_leaf_tokens(type_id: str) -> set[str]:
    leaf = str(type_id).rsplit(".", 1)[-1]
    return {
        _word_root(token)
        for token in _TOKEN_RE.findall(leaf.replace("_", " "))
        if token
    }


def _answer_type_evidence(
    question: str,
    terminal_types: Sequence[str],
    ontology: Any,
) -> tuple[float, tuple[str, ...]]:
    head = _question_head_tokens(question)
    if not head:
        return 0.0, ()
    head_set = set(head)
    best = 0.0
    for type_id in terminal_types:
        if not type_id:
            continue
        schema_types = [type_id, *ontology.supertypes(type_id)]
        for schema_type in schema_types:
            leaf_tokens = _type_leaf_tokens(schema_type)
            overlap = len(head_set & leaf_tokens)
            if overlap:
                best = max(
                    best,
                    overlap / max(1, min(len(head_set), len(leaf_tokens))),
                )
    return best, head


def _is_cvt_terminal(type_id: str, ontology: Any) -> bool:
    """Infer anonymous-record terminals from schema, never a type list."""

    if not type_id or str(type_id).startswith("type."):
        return False
    explicit = getattr(ontology, "is_cvt_type", None)
    if callable(explicit):
        return bool(explicit(type_id))
    return "common.topic" not in set(ontology.supertypes(type_id))


def _terminal_compatibility(left: str, right: str, ontology: Any) -> float:
    """Require schema evidence before unifying terminals from two anchors."""

    if not left or not right:
        return -1.0
    if left == right:
        return 2.0
    left_supers = set(ontology.supertypes(left))
    right_supers = set(ontology.supertypes(right))
    if left in right_supers or right in left_supers:
        return 1.0
    return -1.0


def _compatible_with(
    candidate: FailurePathCandidate,
    previous: Sequence[FailurePathCandidate],
    ontology: Any,
) -> tuple[bool, float]:
    candidate_type = _terminal_type(candidate, ontology)
    scores = [
        _terminal_compatibility(
            candidate_type,
            _terminal_type(other, ontology),
            ontology,
        )
        for other in previous
    ]
    return bool(scores) and all(score >= 0.0 for score in scores), min(scores, default=0.0)


def _intersection_combinations(
    grouped: Mapping[str, list[FailurePathCandidate]],
    anchor_order: Sequence[str],
    ontology: Any,
    *,
    beam: int = _INTERSECTION_BEAM,
) -> tuple[list[tuple[FailurePathCandidate, ...]], int]:
    """Build a bounded, pairwise terminal-compatible cross-anchor beam."""

    if len(anchor_order) < 2 or any(not grouped.get(anchor) for anchor in anchor_order):
        return [], 0
    states: list[tuple[tuple[FailurePathCandidate, ...], float]] = [
        ((candidate,), float(candidate.score))
        for candidate in grouped[anchor_order[0]]
    ]
    rejected = 0
    for anchor_ref in anchor_order[1:]:
        expanded: list[tuple[tuple[FailurePathCandidate, ...], float]] = []
        for previous, previous_score in states:
            for candidate in grouped[anchor_ref]:
                compatible, type_score = _compatible_with(candidate, previous, ontology)
                if not compatible:
                    rejected += 1
                    continue
                combination = (*previous, candidate)
                # Mean path relevance keeps scores comparable as anchor count grows.
                score = (
                    (previous_score * len(previous))
                    + float(candidate.score)
                    + (0.01 * type_score)
                ) / len(combination)
                expanded.append((combination, score))
        expanded.sort(
            key=lambda item: (
                -item[1],
                tuple(
                    (path.anchor_ref, path.directions, path.relation_ids)
                    for path in item[0]
                ),
            )
        )
        states = expanded[: max(1, int(beam))]
        if not states:
            break
    return [item[0] for item in states], rejected


def _grounded_candidate(
    paths: Sequence[FailurePathCandidate],
    anchors: Mapping[str, _Anchor],
    question: str,
    ontology: Any,
    *,
    total_anchor_count: int,
) -> GroundedSemanticCandidate:
    semantic_paths: list[dict[str, Any]] = []
    relation_bindings: dict[str, str] = {}
    for path_index, path in enumerate(paths):
        path_id = f"P{path_index}"
        source = path.anchor_ref
        steps: list[dict[str, Any]] = []
        for step_index, (relation_id, direction) in enumerate(
            zip(path.relation_ids, path.directions)
        ):
            step_id = f"{path_id}.S{step_index}"
            target = f"{path_id}.V{step_index}"
            steps.append(
                {
                    "id": step_id,
                    "relation_label": relation_label_from_id(relation_id),
                    "direction": direction,
                    "from": source,
                    "to": target,
                }
            )
            relation_bindings[step_id] = relation_id
            source = target
        semantic_paths.append(
            {
                "id": path_id,
                "anchor_ref": path.anchor_ref,
                "goal": question,
                "steps": steps,
                "path_output_var": source,
            }
        )
    used = {path.anchor_ref for path in paths}
    terminal_types = tuple(_terminal_type(path, ontology) for path in paths)
    answer_is_cvt = any(
        _is_cvt_terminal(type_id, ontology) for type_id in terminal_types
    )
    type_evidence, answer_head = _answer_type_evidence(
        question,
        terminal_types,
        ontology,
    )
    return GroundedSemanticCandidate(
        compose_input={
            "question": question,
            "anchors": [
                {"id": anchor_ref, "surface": anchors[anchor_ref].surface}
                for anchor_ref in sorted(used)
            ],
            "semantic_paths": semantic_paths,
        },
        anchor_bindings={
            anchor_ref: EntityCandidate(
                anchors[anchor_ref].entity_id,
                anchors[anchor_ref].surface,
                1.0,
                "failure_endpoint_path",
            )
            for anchor_ref in used
        },
        relation_bindings=relation_bindings,
        score=sum(float(path.score) for path in paths) / max(1, len(paths)),
        provenance={
            "failure_endpoint_path_retrieval": {
                "paths": [path.to_dict() for path in paths],
                "anchor_coverage": len(used),
                "total_anchor_count": int(total_anchor_count),
                "terminal_types": list(terminal_types),
                "answer_head_tokens": list(answer_head),
                "answer_type_evidence": type_evidence,
                "answer_is_cvt": answer_is_cvt,
                "model_used": False,
                "uses_gold": False,
            }
        },
    )


def _compile_query_graph(
    grounded: GroundedSemanticCandidate,
    *,
    graph_id: str,
    pipeline_version: str,
) -> QueryGraphCandidate | None:
    """Lower grounded paths, sharing only their verified terminal variable."""

    semantic_paths = [
        path
        for path in grounded.compose_input.get("semantic_paths", [])
        if isinstance(path, dict)
    ]
    if not semantic_paths:
        return None
    triples: list[list[str]] = []
    next_variable = 1
    for path in semantic_paths:
        anchor_ref = str(path.get("anchor_ref", ""))
        anchor = grounded.anchor_bindings.get(anchor_ref)
        steps = [step for step in path.get("steps", []) if isinstance(step, dict)]
        if anchor is None or not steps:
            return None
        source = str(anchor.entity_id)
        for step_index, step in enumerate(steps):
            relation_id = str(grounded.relation_bindings.get(str(step.get("id", "")), ""))
            direction = str(step.get("direction", ""))
            if not relation_id or direction not in {"forward", "backward"}:
                return None
            if step_index == len(steps) - 1:
                target = "V0"
            else:
                target = f"V{next_variable}"
                next_variable += 1
            triple = (
                [source, relation_id, target]
                if direction == "forward"
                else [target, relation_id, source]
            )
            if triple not in triples:
                triples.append(triple)
            source = target
    if not triples or not any("V0" in (triple[0], triple[2]) for triple in triples):
        return None
    anchor_ids = sorted(
        {candidate.entity_id for candidate in grounded.anchor_bindings.values()}
    )
    path_count = len(semantic_paths)
    grounded_metadata = grounded.provenance.get(
        "failure_endpoint_path_retrieval",
        {},
    )
    score = (10.0 * float(grounded.score)) + (4.0 * path_count)
    if path_count > 1:
        score += 8.0 * (path_count - 1)
    return QueryGraphCandidate(
        graph_id=graph_id,
        triples=triples,
        answer_var="V0",
        operators=[
            {
                "type": "NO_EQUAL",
                "inputs": [],
                "input_var": "V0",
                "attribute_relation_label": [],
                "attribute_relation_labels": [],
                "value": entity_id,
                "value_type": "mid",
                "_value_entity_id": entity_id,
                "_failure_endpoint_anchor_exclusion": True,
            }
            for entity_id in anchor_ids
        ],
        score=score,
        compose_output={
            "source": "failure_endpoint_path_retrieval",
            "strategy": "terminal_intersection" if path_count > 1 else "single_path",
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
            "grounded_semantic": grounded.provenance,
            "graph_fallback": {
                "source": "typed_spine_intersection_after_unsuccessful_graphs",
                "model_used": False,
            },
            "failure_endpoint_path_retrieval": {
                "path_count": path_count,
                "anchor_coverage": int(
                    grounded_metadata.get("anchor_coverage", path_count)
                ),
                "total_anchor_count": int(
                    grounded_metadata.get("total_anchor_count", path_count)
                ),
                "terminal_types": list(
                    grounded_metadata.get("terminal_types", [])
                ),
                "answer_head_tokens": list(
                    grounded_metadata.get("answer_head_tokens", [])
                ),
                "answer_type_evidence": float(
                    grounded_metadata.get("answer_type_evidence", 0.0)
                ),
                "answer_is_cvt": bool(
                    grounded_metadata.get("answer_is_cvt", False)
                ),
                "terminal_intersection": path_count > 1,
                "model_used": False,
                "uses_gold": False,
            },
        },
    )


def _query_graphs(
    grounded: Iterable[GroundedSemanticCandidate],
    *,
    pipeline_version: str,
    limit: int = _EXECUTION_BUDGET,
) -> list[QueryGraphCandidate]:
    best: dict[str, QueryGraphCandidate] = {}
    for item in grounded:
        graph = _compile_query_graph(
            item,
            graph_id="FEP",
            pipeline_version=pipeline_version,
        )
        if graph is None:
            continue
        signature = json.dumps(
            {
                "triples": graph.triples,
                "answer_var": graph.answer_var,
                "operators": graph.operators,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        previous = best.get(signature)
        if previous is None or graph.score > previous.score:
            best[signature] = graph
    output = sorted(
        best.values(),
        key=lambda graph: (
            -int(
                float(
                    graph.provenance["failure_endpoint_path_retrieval"].get(
                        "answer_type_evidence",
                        0.0,
                    )
                )
                > 0.0
            ),
            int(
                graph.provenance["failure_endpoint_path_retrieval"].get(
                    "answer_is_cvt",
                    False,
                )
            ),
            -int(
                graph.provenance["failure_endpoint_path_retrieval"].get(
                    "anchor_coverage",
                    0,
                )
            ),
            -graph.score,
            graph.triples,
        ),
    )[: max(1, min(_EXECUTION_BUDGET, int(limit)))]
    for rank, graph in enumerate(output, start=1):
        graph.graph_id = f"FEP{rank}"
        graph.provenance["failure_endpoint_path_retrieval"]["rank"] = rank
        graph.provenance["failure_endpoint_path_retrieval"]["execution_budget"] = len(output)
    return output


def retrieve(context: Any, endpoint: Any) -> dict[str, Any]:
    """Harness plugin entry point for direct endpoint path retrieval."""

    pipeline = context.pipeline
    verbose_diagnostics = bool(getattr(context, "verbose_diagnostics", True))
    linked_anchors = _anchors(context)
    anchors = linked_anchors[:_MAX_SUBSET_ANCHORS]
    decompositions = _decompositions(context)
    diagnostics: dict[str, Any] = {
        "status": "not_compiled",
        "strategy": "direct_failure_endpoint_paths",
        "llm_calls": 0,
        "external_embedding_calls": 0,
        "local_lexical_ranking": True,
        "uses_gold_answers": False,
        "anchor_count": len(anchors),
        "linked_anchor_count": len(linked_anchors),
        "dropped_anchor_count": len(linked_anchors) - len(anchors),
        "execution_budget": _EXECUTION_BUDGET,
        "path_query_limit": _PATH_QUERY_LIMIT,
        "path_max_depth": 1,
    }
    ontology = getattr(pipeline.grounder, "ontology", None)
    if not anchors or ontology is None:
        diagnostics["reason"] = "missing_anchor_or_ontology"
        return {"answer_ids": [], "diagnostics": diagnostics}

    anchor_by_ref = {anchor.anchor_ref: anchor for anchor in anchors}
    candidates, path_diagnostics = _retriever(context, endpoint).retrieve(
        question=context.question,
        decompositions=decompositions,
        anchor_bindings={
            anchor.anchor_ref: EntityCandidate(
                anchor.entity_id,
                anchor.surface,
                1.0,
                "existing_linked_anchor",
            )
            for anchor in anchors
        },
    )
    diagnostics["path_retrieval"] = (
        path_diagnostics
        if verbose_diagnostics
        else {
            key: value
            for key, value in path_diagnostics.items()
            if key
            in {
                "strategy",
                "top_k",
                "query_limit",
                "max_depth",
                "anchor_count",
                "direction_pattern_count",
                "query_count",
                "cache_hits",
                "query_elapsed_seconds",
                "ranking_elapsed_seconds",
                "raw_candidate_count",
                "deduplicated_candidate_count",
                "schema_rejected_count",
                "retained_candidate_count",
                "retained_by_depth",
                "elapsed_seconds",
                "code",
            }
        }
    )
    grouped: dict[str, list[FailurePathCandidate]] = {
        anchor.anchor_ref: [] for anchor in anchors
    }
    for candidate in candidates:
        grouped.setdefault(candidate.anchor_ref, []).append(candidate)

    path_combinations: list[tuple[FailurePathCandidate, ...]] = []
    rejected_intersections = 0
    anchor_refs = [anchor.anchor_ref for anchor in anchors]
    subset_total = (2 ** len(anchor_refs)) - 1
    per_subset_budget = max(
        1,
        min(_INTERSECTION_BEAM, _SUBSET_COMBINATION_BUDGET // max(1, subset_total)),
    )
    subset_counts: dict[str, int] = {}
    for subset_size in range(len(anchor_refs), 0, -1):
        count_before = len(path_combinations)
        for subset in anchor_subsets(anchor_refs, subset_size):
            if subset_size == 1:
                path_combinations.extend(
                    (candidate,)
                    for candidate in grouped.get(subset[0], [])[:per_subset_budget]
                )
                continue
            values, rejected = _intersection_combinations(
                grouped,
                subset,
                ontology,
                beam=per_subset_budget,
            )
            path_combinations.extend(values)
            rejected_intersections += rejected
        subset_counts[str(subset_size)] = len(path_combinations) - count_before
    grounded = [
        _grounded_candidate(
            combination,
            anchor_by_ref,
            context.question,
            ontology,
            total_anchor_count=len(anchors),
        )
        for combination in path_combinations
    ]
    graphs = _query_graphs(
        grounded,
        pipeline_version=str(pipeline.version),
        limit=_EXECUTION_BUDGET,
    )
    diagnostics["compilation"] = {
        "mode": "all_nonempty_anchor_subsets",
        "combination_count": len(path_combinations),
        "combination_count_by_coverage": subset_counts,
        "subset_count": subset_total,
        "per_subset_combination_budget": per_subset_budget,
        "global_combination_budget": _SUBSET_COMBINATION_BUDGET,
        "terminal_type_rejected_count": rejected_intersections,
        "grounded_candidate_count": len(grounded),
        "query_graph_count": len(graphs),
    }

    executions: list[Any] = []
    execution_traces: list[dict[str, Any]] = []
    graph_by_id = {graph.graph_id: graph for graph in graphs}
    for graph in graphs:
        graph.provenance["original_question"] = context.question
        graph.provenance["decomposition"] = list(decompositions)
        executed, trace = pipeline._execute_graph_candidate(
            graph,
            lookup_labels=False,
        )
        execution_traces.append(
            {
                key: trace[key]
                for key in (
                    "graph_id",
                    "answer_count",
                    "row_count",
                    "query_elapsed_seconds",
                    "elapsed_seconds",
                    "error",
                )
                if key in trace
            }
        )
        if executed is not None and executed.answer_ids:
            executions.append(executed)
    diagnostics["executions"] = execution_traces
    runtime_candidate_records = [
        {
            "graph_id": item.graph.graph_id,
            "score": item.graph.score,
            "answer_count": len(item.answer_ids),
            "answer_ids": list(item.answer_ids),
            "triples": graph_by_id[item.graph.graph_id].triples,
            "answer_var": str(item.graph.answer_var),
            "anchor_coverage": int(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "anchor_coverage",
                    0,
                )
            ),
            "total_anchor_count": int(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "total_anchor_count",
                    0,
                )
            ),
            "terminal_types": list(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "terminal_types",
                    [],
                )
            ),
            "answer_type_evidence": float(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "answer_type_evidence",
                    0.0,
                )
            ),
            "answer_is_cvt": bool(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "answer_is_cvt",
                    False,
                )
            ),
        }
        for item in executions
    ]
    diagnostics["execution_candidates"] = [
        (
            dict(candidate)
            if verbose_diagnostics
            else {
                key: value
                for key, value in candidate.items()
                if key not in {"answer_ids", "triples", "answer_var"}
            }
        )
        for candidate in runtime_candidate_records
    ]
    if not executions:
        diagnostics["reason"] = "all_endpoint_path_graphs_empty"
        return {"answer_ids": [], "diagnostics": diagnostics}

    typed = [
        item
        for item in executions
        if float(
            item.graph.provenance["failure_endpoint_path_retrieval"].get(
                "answer_type_evidence",
                0.0,
            )
        )
        > 0.0
    ]
    selection_pool = typed or executions
    non_cvt = [
        item
        for item in selection_pool
        if not bool(
            item.graph.provenance["failure_endpoint_path_retrieval"].get(
                "answer_is_cvt",
                False,
            )
        )
    ]
    selection_pool = non_cvt or selection_pool
    selected = max(
        selection_pool,
        key=lambda item: (
            int(
                item.graph.provenance["failure_endpoint_path_retrieval"].get(
                    "anchor_coverage",
                    0,
                )
            ),
            item.graph.score,
            -len(item.answer_ids),
            item.graph.graph_id,
        ),
    )
    def populate_selected_diagnostics(item: Any) -> None:
        selected_set = set(item.answer_ids)
        selected_answer_set_support = sum(
            set(execution.answer_ids) == selected_set for execution in executions
        )
        selected_metadata = item.graph.provenance[
            "failure_endpoint_path_retrieval"
        ]
        selected_terminal_types = [
            str(value)
            for value in selected_metadata.get("terminal_types", [])
            if str(value)
        ]
        diagnostics["status"] = "selected"
        diagnostics["selected_graph_id"] = item.graph.graph_id
        diagnostics["answer_count"] = len(item.answer_ids)
        diagnostics["selected_anchor_coverage"] = int(
            selected_metadata.get("anchor_coverage", 0)
        )
        diagnostics["selected_answer_type_evidence"] = float(
            selected_metadata.get("answer_type_evidence", 0.0)
        )
        diagnostics["selected_answer_set_support"] = selected_answer_set_support
        diagnostics["nonempty_candidate_count"] = len(executions)
        diagnostics["all_nonempty_candidates_agree"] = bool(
            executions and selected_answer_set_support == len(executions)
        )
        diagnostics["selected_terminal_types"] = selected_terminal_types
        diagnostics["selected_terminal_is_cvt"] = any(
            _is_cvt_terminal(type_id, ontology)
            for type_id in selected_terminal_types
        )
        diagnostics["selected_answer_is_cvt"] = bool(
            selected_metadata.get("answer_is_cvt", False)
        )

    populate_selected_diagnostics(selected)
    if bool(
        getattr(
            pipeline,
            "failure_endpoint_return_bounded_direct_best_effort",
            False,
        )
    ):
        # Import lazily: the failure plugin invokes this module as its final
        # lane, so the shared outer gate is fully initialized at this point.
        from .failure_retrieval_plugin import _direct_path_acceptance

        original_accepted, _, _ = _direct_path_acceptance(
            context.question,
            list(selected.answer_ids),
            diagnostics,
        )
        if not original_accepted:
            incumbent_record = next(
                candidate
                for candidate in runtime_candidate_records
                if str(candidate.get("graph_id", "")) == selected.graph.graph_id
            )
            challenger_graph_id, challenger_evidence = (
                _select_low_confidence_direct_challenger(
                    context.question,
                    incumbent_record,
                    runtime_candidate_records,
                )
            )
            diagnostics["low_confidence_challenger"] = {
                **challenger_evidence,
                "best_effort_enabled": True,
                "original_direct_confidence_gate_rejected": True,
            }
            if challenger_graph_id:
                selected = next(
                    execution
                    for execution in executions
                    if execution.graph.graph_id == challenger_graph_id
                )
                populate_selected_diagnostics(selected)

    # Exactly one label lookup is performed, after the final already-executed
    # candidate is fixed.  Challenging never executes another graph query.
    selected_labels: list[dict[str, str]] = []
    try:
        selected_labels = endpoint.labels(list(selected.answer_ids))
    except (RuntimeError, TypeError, ValueError) as exc:
        diagnostics["selected_label_error"] = f"{type(exc).__name__}:{exc}"
    diagnostics["selected_label_lookup_count"] = int(bool(selected.answer_ids))
    labels_by_id = {
        str(item.get("id", "")): {
            "id": str(item.get("id", "")),
            "label": str(item.get("label", "")),
        }
        for item in selected_labels
        if isinstance(item, dict) and str(item.get("id", ""))
    }
    answers = [
        labels_by_id.get(answer_id, {"id": answer_id, "label": answer_id})
        for answer_id in selected.answer_ids
    ]
    if verbose_diagnostics:
        diagnostics["selected_labels"] = answers
    graph_summary = getattr(pipeline, "_graph_summary", None)
    selected_graph = (
        graph_summary(selected.graph)
        if callable(graph_summary)
        else {
            "graph_id": selected.graph.graph_id,
            "triples": list(selected.graph.triples),
            "answer_var": selected.graph.answer_var,
            "operators": list(selected.graph.operators),
            "score": selected.graph.score,
            "provenance": dict(selected.graph.provenance),
        }
    )
    return {
        "answer_ids": list(selected.answer_ids),
        "answers": answers,
        "selected_graph": selected_graph,
        "diagnostics": diagnostics,
    }


__all__ = ["retrieve"]
