"""Fixed-budget recovery for entity questions with scalar/schema graph slots.

This module is deliberately independent from :mod:`pipeline` so both the
legacy and v2 execution loops can integrate it without a circular import.  A
route is planned before endpoint execution from only:

* the question wh-head;
* ontology types incident on graph answer variables;
* existing semantic/decomposition traces; and
* a cache built from CWQ TRAIN Gold-SPARQL templates.

Prepared raw-SPARQL templates replace existing execution slots.  They never
increase the endpoint-query budget and are excluded from the model selector.
After ordinary selection, a prepared result may replace the incumbent only
when the incumbent answers are all non-entity values and the prepared answer
set is entirely Freebase entities.  Existing entity selections are therefore
a hard non-regression interlock.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ExecutedGraph, QueryGraphCandidate
from .ontology import relation_id_from_label


VERSION = "entity-kind-fixed-beam-train-template-v1"
ENTITY_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
NS_VALUE_RE = re.compile(r"ns:([A-Za-z0-9_.]+)")
DOUBLE_LITERAL_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
SCALAR_OR_SCHEMA_TYPES = frozenset(
    {
        "type.boolean",
        "type.datetime",
        "type.enumeration",
        "type.float",
        "type.int",
        "type.property",
        "type.rawstring",
        "type.schema",
        "type.text",
        "type.type",
    }
)
SCALAR_WH_HEADS = frozenset(
    {
        "age", "amount", "area", "capacity", "code", "coordinate",
        "cost", "date", "distance", "duration", "height", "id",
        "identifier", "latitude", "length", "longitude", "number",
        "percent", "percentage", "population", "price", "rank", "rate",
        "score", "temperature", "time", "weight", "width", "year",
    }
)
WH_SKIP = frozenset(
    {
        "a", "an", "are", "be", "been", "being", "can", "could",
        "did", "do", "does", "had", "has", "have", "is", "should",
        "the", "was", "were", "will", "would",
    }
)
TOKEN_STOP = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by",
        "did", "do", "does", "for", "from", "had", "has", "have",
        "her", "his", "in", "is", "it", "its", "of", "on", "that",
        "the", "their", "to", "was", "were", "what", "when", "where",
        "which", "who", "with",
    }
)


@dataclass(slots=True)
class PreparedEntityTemplateQuery:
    graph_id: str
    query: str
    answer_var: str
    score: float
    source_index: int
    source_template_rank: int
    mapping_rank: int
    source_question: str
    rank_evidence: dict[str, float]

    def trace_summary(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "query_sha256": hashlib.sha256(self.query.encode()).hexdigest(),
            "answer_var": self.answer_var,
            "fixed_beam_score": self.score,
            "source_index": self.source_index,
            "source_template_rank": self.source_template_rank,
            "mapping_rank": self.mapping_rank,
            "rank_evidence": dict(self.rank_evidence),
        }


@dataclass(slots=True)
class EntityKindFixedBeamPlan:
    ordinary_graphs: list[QueryGraphCandidate]
    template_queries: list[PreparedEntityTemplateQuery]
    diagnostics: dict[str, Any]


def _stem(token: str) -> str:
    value = str(token).casefold()
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and len(value) > 4:
        return value[:-1]
    return value


def _words(value: str, *, stop: bool = True) -> list[str]:
    result = [_stem(token) for token in re.findall(r"[a-z]+", str(value).casefold())]
    return [token for token in result if not stop or token not in TOKEN_STOP]


def _dice(left: set[Any], right: set[Any]) -> float:
    return 2.0 * len(left & right) / (len(left) + len(right)) if left and right else 0.0


def question_expects_entity(question: str) -> bool:
    """Recognize entity wh-heads while excluding clear scalar/time heads."""
    text = " ".join(str(question).casefold().split())
    if re.match(r"^(?:when|how\s+(?:many|much|old|long))\b", text):
        return False
    tokens = re.findall(r"[a-z]+", text)
    if not tokens:
        return False
    if tokens[0] in {"who", "where"}:
        return True
    if tokens[0] not in {"what", "which"}:
        return False
    position = 1
    while position < len(tokens) and tokens[position] in WH_SKIP:
        position += 1
    if position >= len(tokens):
        return False
    head = _stem(tokens[position])
    if head in SCALAR_WH_HEADS:
        return False
    if head == "iso":
        return "country" in tokens[position + 1 : position + 7]
    return True


def answer_kind(values: Sequence[Any]) -> str:
    materialized = [str(value) for value in values]
    if not materialized:
        return "empty"
    flags = [bool(ENTITY_RE.fullmatch(value)) for value in materialized]
    if all(flags):
        return "entity"
    if not any(flags):
        return "nonentity"
    return "mixed"


def terminal_types(graph: QueryGraphCandidate, ontology: Any) -> frozenset[str]:
    answer = str(graph.answer_var)
    output: set[str] = set()
    for subject, relation, object_ in graph.triples:
        if str(object_) == answer:
            value = str(ontology.range_for_relation(str(relation)))
            if value:
                output.add(value)
        if str(subject) == answer:
            value = str(ontology.domain_for_relation(str(relation)))
            if value:
                output.add(value)
    return frozenset(output)


def is_scalar_schema_answer_slot(graph: QueryGraphCandidate, ontology: Any) -> bool:
    values = terminal_types(graph, ontology)
    return bool(values and values <= SCALAR_OR_SCHEMA_TYPES)


def _slot_plan(
    graphs: Sequence[QueryGraphCandidate], ontology: Any, maximum: int
) -> tuple[list[int], dict[str, Any]]:
    scalar_indexes = [
        index
        for index, graph in enumerate(graphs)
        if is_scalar_schema_answer_slot(graph, ontology)
    ]
    strongest_index = max(
        scalar_indexes,
        key=lambda index: (float(graphs[index].score), str(graphs[index].graph_id)),
        default=-1,
    )
    replaceable = [index for index in scalar_indexes if index != strongest_index]
    policy = "replace_lower_scored_scalar_schema_slots"
    if not replaceable and strongest_index >= 0 and len(graphs) >= 2:
        top_index = max(
            range(len(graphs)),
            key=lambda index: (float(graphs[index].score), str(graphs[index].graph_id)),
        )
        if top_index == strongest_index:
            replaceable = [
                min(
                    (index for index in range(len(graphs)) if index != strongest_index),
                    key=lambda index: (
                        float(graphs[index].score),
                        str(graphs[index].graph_id),
                    ),
                )
            ]
            policy = "singleton_top_scalar_preserve_sentinel_replace_lowest_other"
    replaceable.sort(
        key=lambda index: (float(graphs[index].score), str(graphs[index].graph_id))
    )
    selected = replaceable[: min(3, max(0, int(maximum)))]
    return selected, {
        "scalar_slot_count": len(scalar_indexes),
        "sentinel_graph_id": (
            str(graphs[strongest_index].graph_id) if strongest_index >= 0 else ""
        ),
        "replacement_slot_count": len(selected),
        "replaced_graph_ids": [str(graphs[index].graph_id) for index in selected],
        "slot_policy": policy if selected else "no_fixed_slot_available",
        "types_by_graph": {
            str(graphs[index].graph_id): sorted(terminal_types(graphs[index], ontology))
            for index in scalar_indexes
        },
    }


def semantic_relation_unions(semantic_graphs: Sequence[Mapping[str, Any]]) -> list[frozenset[str]]:
    output: list[frozenset[str]] = []
    for raw in semantic_graphs:
        graph: Any = raw.get("output", raw) if isinstance(raw, Mapping) else {}
        if not isinstance(graph, Mapping):
            continue
        relation_ids: set[str] = set()
        for path in graph.get("semantic_paths", []):
            if not isinstance(path, Mapping):
                continue
            for step in path.get("steps", []):
                if not isinstance(step, Mapping):
                    continue
                relation_id = relation_id_from_label(step.get("relation_label", []))
                if relation_id:
                    relation_ids.add(relation_id)
        frozen = frozenset(relation_ids)
        if frozen and frozen not in output:
            output.append(frozen)
    return output


def _canonical_relation_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _query_relations(query: str) -> frozenset[str]:
    return frozenset(
        value
        for value in NS_VALUE_RE.findall(str(query))
        if not ENTITY_RE.fullmatch(value)
    )


def _query_operator_signature(query: str) -> frozenset[str]:
    text = " ".join(str(query).casefold().split())
    output: set[str] = set()
    if "order by asc" in text:
        output.add("argmin")
    if "order by desc" in text:
        output.add("argmax")
    if re.search(r"filter\s*\([^)]*?\s<\s", text):
        output.add("lt")
    if re.search(r"filter\s*\([^)]*?\s>\s", text):
        output.add("gt")
    if "count(" in text:
        output.add("count")
    return frozenset(output)


def _incompatible_copied_string(query: str, target_text: str) -> bool:
    target_tokens = set(_words(target_text))
    for literal in DOUBLE_LITERAL_RE.findall(query):
        value = bytes(literal, "utf-8").decode("unicode_escape")
        if not value or re.fullmatch(
            r"[-+]?\d+(?:\.\d+)?(?:-\d{2}(?:-\d{2})?)?", value
        ):
            continue
        literal_tokens = set(_words(value))
        if literal_tokens and not literal_tokens <= target_tokens:
            return True
    return False


def _rank_template(
    candidate: dict[str, Any],
    *,
    question: str,
    target_text: str,
    semantic_unions: Sequence[frozenset[str]],
    target_operator: set[str],
) -> tuple[float, dict[str, float]] | None:
    query = str(candidate["query"])
    if _incompatible_copied_string(query, target_text):
        return None
    relations = {
        _canonical_relation_id(value) for value in _query_relations(query)
    }
    relation_alignment = max(
        (
            _dice(
                relations,
                {_canonical_relation_id(item) for item in semantic_union},
            )
            for semantic_union in semantic_unions
        ),
        default=0.0,
    )
    source_operator = set(candidate.get("source_operator_signature", ()))
    if target_operator:
        source_operator.update(_query_operator_signature(query))
    operator_alignment = (
        1.0
        if source_operator == target_operator
        else len(source_operator & target_operator) / len(source_operator | target_operator)
        if source_operator | target_operator
        else 1.0
    )
    source_question = str(candidate.get("source_question", ""))
    token_alignment = _dice(set(_words(source_question)), set(_words(question)))
    source_raw = _words(source_question, stop=False)
    target_raw = _words(question, stop=False)
    bigram_alignment = _dice(
        set(zip(source_raw, source_raw[1:])),
        set(zip(target_raw, target_raw[1:])),
    )
    score = (
        10.0 * relation_alignment
        + 4.0 * operator_alignment
        + 3.0 * token_alignment
        + 2.0 * bigram_alignment
        - 0.01 * int(candidate.get("source_template_rank", 0))
        + 0.001 * float(candidate.get("runtime_score", 0.0))
    )
    return score, {
        "relation_alignment": relation_alignment,
        "operator_alignment": operator_alignment,
        "token_alignment": token_alignment,
        "bigram_alignment": bigram_alignment,
    }


def _answer_values(rows: Sequence[Mapping[str, Any]], answer_var: str) -> list[str]:
    key = str(answer_var).lstrip("?")
    values: list[str] = []
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
        text = str(value).strip()
        prefix = "http://rdf.freebase.com/ns/"
        if text.startswith(prefix):
            text = text[len(prefix):]
        if text and text not in values:
            values.append(text)
    return values


def _answer_labels(answer_ids: Sequence[str], labels: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    by_id = {
        str(item.get("id", "")): {
            "id": str(item.get("id", "")),
            "label": str(item.get("label", "")),
        }
        for item in labels
        if str(item.get("id", ""))
    }
    return [by_id.get(value, {"id": value, "label": value}) for value in answer_ids]


class EntityKindFixedBeamPlanner:
    """Lazy TRAIN-template planner shared by both pipeline execution modes."""

    def __init__(self, cache_path: str | Path, *, top_k: int = 32) -> None:
        self.cache_path = str(Path(cache_path).expanduser())
        self.top_k = min(64, max(3, int(top_k)))
        self._index_value: Any = None
        self._index_lock = Lock()

    def _index(self) -> Any:
        value = self._index_value
        if value is not None:
            return value
        # Lazy import avoids failure_gold_sparql_retrieval -> pipeline import
        # cycles while reusing the already-audited TRAIN-only cache reader.
        from .failure_gold_sparql_retrieval import SourceGoldSparqlIndex

        with self._index_lock:
            if self._index_value is None:
                self._index_value = SourceGoldSparqlIndex(self.cache_path)
            return self._index_value

    def _templates(
        self,
        *,
        question: str,
        decomposition: Sequence[str],
        entities: Sequence[tuple[str, str]],
        semantic_unions: Sequence[frozenset[str]],
        limit: int,
    ) -> list[PreparedEntityTemplateQuery]:
        from .failure_gold_sparql_retrieval import (
            _effective_operator_signature,
            _instantiate,
            _operator_signature,
        )

        index = self._index()
        matches = index.retrieve(
            question=question,
            decomposition=decomposition,
            surfaces=list(dict.fromkeys(label for _, label in entities)),
            entity_count=len(entities),
            top_k=self.top_k,
        )
        values: list[dict[str, Any]] = []
        for match in matches:
            document = match["document"]
            for candidate in _instantiate(match, entities, question, decomposition):
                candidate["source_question"] = document.question
                candidate["source_operator_signature"] = list(
                    _effective_operator_signature(document)
                )
                values.append(candidate)
        values.sort(
            key=lambda item: (
                -float(item["runtime_score"]),
                int(item["source_template_rank"]),
                int(item["mapping_rank"]),
                str(item["key"]),
            )
        )
        pool: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in values:
            signature = " ".join(str(candidate["query"]).split())
            if signature in seen:
                continue
            seen.add(signature)
            pool.append(candidate)
            if len(pool) >= 20:
                break
        ranked: list[tuple[float, dict[str, float], dict[str, Any]]] = []
        target_operator = set(_operator_signature(question))
        target_text = " ".join([question, *map(str, decomposition)])
        for candidate in pool:
            result = _rank_template(
                candidate,
                question=question,
                target_text=target_text,
                semantic_unions=semantic_unions,
                target_operator=target_operator,
            )
            if result is None:
                continue
            score, evidence = result
            ranked.append((score, evidence, candidate))
        ranked.sort(
            key=lambda item: (
                -item[0],
                int(item[2]["source_template_rank"]),
                int(item[2]["mapping_rank"]),
                str(item[2]["key"]),
            )
        )
        return [
            PreparedEntityTemplateQuery(
                graph_id=f"EK{offset}",
                query=str(candidate["query"]),
                answer_var=str(candidate["answer_var"]),
                score=float(score),
                source_index=int(candidate["source_index"]),
                source_template_rank=int(candidate["source_template_rank"]),
                mapping_rank=int(candidate["mapping_rank"]),
                source_question=str(candidate["source_question"]),
                rank_evidence=dict(evidence),
            )
            for offset, (score, evidence, candidate) in enumerate(ranked[: max(0, int(limit))])
        ]

    def prepare(
        self,
        *,
        question: str,
        graphs: Sequence[QueryGraphCandidate],
        decomposition: Sequence[str],
        semantic_graphs: Sequence[Mapping[str, Any]],
        entities: Sequence[tuple[str, str]],
        ontology: Any,
        execution_budget: int,
        max_replacements: int = 3,
    ) -> EntityKindFixedBeamPlan:
        budget = min(len(graphs), max(0, int(execution_budget)))
        active = list(graphs[:budget])
        diagnostics: dict[str, Any] = {
            "version": VERSION,
            "status": "not_applied",
            "reason": "",
            "preexecution_trigger": False,
            "query_count_before": budget,
            "query_count_after": budget,
            "additional_endpoint_queries": 0,
            "additional_model_calls": 0,
            "uses_evaluation_gold": False,
            "uses_test_sparql_or_reasoning_information": False,
            "source": "CWQ_TRAIN_gold_sparql_only",
            "entity_or_relation_whitelist": False,
        }
        if ontology is None or budget < 2:
            diagnostics["reason"] = "missing_ontology_or_execution_slots"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        if not question_expects_entity(question):
            diagnostics["reason"] = "question_not_entity_wh_head"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        evicted, slot_diagnostics = _slot_plan(active, ontology, max_replacements)
        diagnostics.update(slot_diagnostics)
        if not evicted:
            diagnostics["reason"] = "no_fixed_replacement_slot"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        if not entities:
            diagnostics["reason"] = "no_linked_entities"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        unions = semantic_relation_unions(semantic_graphs)
        if not unions:
            diagnostics["reason"] = "no_semantic_relation_union"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        started = time.perf_counter()
        templates = self._templates(
            question=question,
            decomposition=decomposition,
            entities=entities,
            semantic_unions=unions,
            limit=len(evicted),
        )
        if not templates:
            diagnostics["reason"] = "no_compatible_train_template"
            return EntityKindFixedBeamPlan(active, [], diagnostics)
        # If unmatched-string filtering yields fewer templates than slots,
        # evict exactly the number actually occupied.
        evicted = evicted[: len(templates)]
        evicted_set = set(evicted)
        ordinary = [graph for index, graph in enumerate(active) if index not in evicted_set]
        diagnostics.update(
            {
                "status": "prepared",
                "reason": "entity_wh_head_with_scalar_schema_answer_slot",
                "preexecution_trigger": True,
                "replacement_query_count": len(templates),
                "replaced_graph_ids": [active[index].graph_id for index in evicted],
                "template_graph_ids": [item.graph_id for item in templates],
                "query_count_after": len(ordinary) + len(templates),
                "semantic_relation_unions": [sorted(value) for value in unions],
                "planning_elapsed_seconds": time.perf_counter() - started,
            }
        )
        return EntityKindFixedBeamPlan(ordinary, templates, diagnostics)


def execute_prepared_query(
    prepared: PreparedEntityTemplateQuery,
    endpoint: Any,
    *,
    lookup_labels: bool = True,
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Execute one already budgeted raw template query."""
    started = time.perf_counter()
    trace: dict[str, Any] = {
        **prepared.trace_summary(),
        "fixed_execution_slot": True,
        "additional_endpoint_queries": 0,
        "additional_model_calls": 0,
    }
    try:
        rows = endpoint.execute(prepared.query)
        answer_ids = _answer_values(rows, prepared.answer_var)
    except Exception as exc:
        trace.update(
            {
                "error": f"{type(exc).__name__}:{exc}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return None, trace
    raw_labels: list[dict[str, str]] = []
    if lookup_labels and answer_ids:
        entity_ids = [value for value in answer_ids if ENTITY_RE.fullmatch(value)]
        if entity_ids:
            try:
                raw_labels = list(endpoint.labels(entity_ids))
            except Exception as exc:
                trace["label_error"] = f"{type(exc).__name__}:{exc}"
    graph = QueryGraphCandidate(
        graph_id=prepared.graph_id,
        triples=[],
        answer_var=prepared.answer_var,
        operators=[],
        score=prepared.score,
        compose_output={},
        provenance={
            "entity_kind_fixed_beam": {
                **prepared.trace_summary(),
                "source_question": prepared.source_question,
                "additional_endpoint_queries": 0,
                "additional_model_calls": 0,
            }
        },
        sparql=prepared.query,
    )
    trace.update(
        {
            "row_count": len(rows),
            "answer_count": len(answer_ids),
            "answer_kind": answer_kind(answer_ids),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return (
        ExecutedGraph(
            graph=graph,
            answer_ids=answer_ids,
            answers=_answer_labels(answer_ids, raw_labels),
            row_count=len(rows),
        ),
        trace,
    )


def is_entity_kind_variant(execution: ExecutedGraph) -> bool:
    return isinstance(execution.graph.provenance.get("entity_kind_fixed_beam"), dict)


def postselect_entity_kind_execution(
    selected: ExecutedGraph,
    variants: Sequence[ExecutedGraph],
) -> tuple[ExecutedGraph, dict[str, Any]]:
    """Switch only a pure non-entity incumbent to a pure entity variant."""
    diagnostics: dict[str, Any] = {
        "version": VERSION,
        "status": "not_applied",
        "source_graph_id": selected.graph.graph_id,
        "source_answer_kind": answer_kind(selected.answer_ids),
        "additional_endpoint_queries": 0,
        "additional_model_calls": 0,
        "entity_or_relation_whitelist": False,
    }
    if diagnostics["source_answer_kind"] != "nonentity":
        diagnostics["reason"] = "existing_answer_is_not_pure_nonentity"
        return selected, diagnostics
    eligible = [
        execution
        for execution in variants
        if execution.answer_ids and answer_kind(execution.answer_ids) == "entity"
    ]
    eligible.sort(
        key=lambda execution: (
            -float(execution.graph.score),
            str(execution.graph.graph_id),
        )
    )
    if not eligible:
        diagnostics["reason"] = "no_nonempty_entity_template"
        return selected, diagnostics
    challenger = eligible[0]
    diagnostics.update(
        {
            "status": "applied",
            "reason": "entity_template_replaces_nonentity_answer",
            "selected_graph_id": challenger.graph.graph_id,
            "selected_answer_count": len(challenger.answer_ids),
            "eligible_graph_ids": [item.graph.graph_id for item in eligible],
        }
    )
    return challenger, diagnostics


def select_empty_entity_kind_execution(
    question: str,
    variants: Sequence[ExecutedGraph],
) -> tuple[ExecutedGraph | None, dict[str, Any]]:
    """Commit an already-budgeted entity variant only under answer consensus."""

    diagnostics: dict[str, Any] = {
        "version": VERSION,
        "status": "not_applied",
        "source_answer_kind": "empty",
        "additional_endpoint_queries": 0,
        "additional_model_calls": 0,
        "entity_or_relation_whitelist": False,
        "ordinary_nonempty_count": 0,
    }
    if not question_expects_entity(question):
        diagnostics["reason"] = "question_does_not_request_entity"
        return None, diagnostics
    eligible = [
        execution
        for execution in variants
        if execution.answer_ids and answer_kind(execution.answer_ids) == "entity"
    ]
    diagnostics["entity_variant_nonempty_count"] = len(eligible)
    if not eligible:
        diagnostics["reason"] = "no_nonempty_entity_template"
        return None, diagnostics
    answer_sets = {
        tuple(sorted(set(map(str, execution.answer_ids))))
        for execution in eligible
    }
    diagnostics["entity_variant_answer_set_support"] = {
        " | ".join(values): sum(
            tuple(sorted(set(map(str, item.answer_ids)))) == values
            for item in eligible
        )
        for values in sorted(answer_sets)
    }
    if len(answer_sets) != 1:
        diagnostics["reason"] = "divergent_fixed_variant_answers"
        return None, diagnostics
    selected = max(
        eligible,
        key=lambda execution: (
            float(execution.graph.score),
            str(execution.graph.graph_id),
        ),
    )
    diagnostics.update(
        {
            "status": "applied",
            "reason": "ordinary_empty_consistent_entity_fixed_variant",
            "selected_graph_id": selected.graph.graph_id,
            "selected_answer_count": len(selected.answer_ids),
            "eligible_graph_ids": [item.graph.graph_id for item in eligible],
            "fixed_execution_slot": True,
        }
    )
    return selected, diagnostics


def linked_entities_for_question(
    gold_entities: Mapping[str, Mapping[str, str]], question: str
) -> list[tuple[str, str]]:
    """Project configured topic links in their linker-provenance order.

    The source order commonly reflects path/constraint roles.  Re-sorting by
    surface position loses that signal for multi-anchor questions; template
    instantiation already evaluates alternative permutations explicitly.
    """
    normalized = " ".join(str(question).casefold().split())
    mapping = gold_entities.get(normalized, {})
    if not isinstance(mapping, Mapping):
        return []
    return [
        (str(entity_id), str(label))
        for entity_id, label in mapping.items()
    ]


__all__ = [
    "EntityKindFixedBeamPlan",
    "EntityKindFixedBeamPlanner",
    "PreparedEntityTemplateQuery",
    "answer_kind",
    "execute_prepared_query",
    "is_entity_kind_variant",
    "is_scalar_schema_answer_slot",
    "linked_entities_for_question",
    "postselect_entity_kind_execution",
    "select_empty_entity_kind_execution",
    "question_expects_entity",
    "semantic_relation_unions",
    "terminal_types",
]
