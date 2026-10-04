"""Bounded, model-free query-graph reconstruction for failed executions.

The compiler only rearranges already-grounded semantic paths.  It never reads
Gold answers, invents a relation, or performs a KG request.  Callers execute at
most the returned ``limit`` graphs inside their existing failure budget.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import islice, product
import json
import re
from typing import Any, Iterable

from .contracts import EntityCandidate, GroundedSemanticCandidate, QueryGraphCandidate


_ENTITY_ID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
_FALLBACK_SOURCE = "typed_spine_intersection_after_unsuccessful_graphs"
_SCALAR_TYPES = {
    "type.boolean",
    "type.datetime",
    "type.enumeration",
    "type.float",
    "type.int",
    "type.key",
    "type.rawstring",
    "type.text",
}
_TYPE_WORD_RE = re.compile(r"[a-z0-9]+")
_AUXILIARY_ANSWER_HEADS = {
    "a",
    "an",
    "are",
    "did",
    "do",
    "does",
    "is",
    "of",
    "the",
    "was",
    "were",
}
_LITERAL_ANSWER_HEADS = {
    "age",
    "amount",
    "date",
    "id",
    "identifier",
    "many",
    "number",
    "percentage",
    "time",
    "year",
}


@dataclass(frozen=True, slots=True)
class _PathView:
    path_id: str
    anchor: str
    variables: tuple[str, ...]
    terminal: str
    triples: tuple[tuple[str, str, str], ...]
    node_types: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _CompiledGraph:
    key: str
    graph: QueryGraphCandidate


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def is_graph_reconstruction_graph(graph: QueryGraphCandidate) -> bool:
    fallback = graph.provenance.get("graph_fallback", {})
    return bool(
        isinstance(fallback, dict)
        and fallback.get("source") == _FALLBACK_SOURCE
    )


def _singular_word(value: Any) -> str:
    """Apply morphology-only normalization to one schema/question word."""
    token = str(value).strip().casefold()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _type_leaf_head(type_id: str) -> str:
    words = _TYPE_WORD_RE.findall(
        str(type_id).rsplit(".", 1)[-1].replace("_", " ").casefold()
    )
    return _singular_word(words[-1]) if words else ""


def _type_closure(type_id: str, ontology: Any) -> set[str]:
    try:
        inherited = ontology.supertypes(str(type_id))
    except (AttributeError, TypeError, ValueError):
        inherited = ()
    return {str(type_id), *(str(value) for value in inherited if str(value))}


def _variable_schema_types(
    graph: QueryGraphCandidate,
    variable: str,
    ontology: Any,
) -> set[str]:
    """Return the most-specific incident schema types for one graph variable."""
    values: set[str] = set()
    for triple in graph.triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject, relation_id, object_ = (str(value) for value in triple)
        if subject == variable:
            value = str(ontology.domain_for_relation(relation_id))
            if value:
                values.add(value)
        if object_ == variable:
            value = str(ontology.range_for_relation(relation_id))
            if value:
                values.add(value)
    # An incident variable can receive both a leaf and one of its ancestors.
    # Retaining only leaves prevents a broad root from masking a role error.
    return {
        type_id
        for type_id in values
        if not any(
            type_id != other and type_id in _type_closure(other, ontology)
            for other in values
        )
    }


def _question_answer_head(question: str) -> str:
    """Extract only a direct What/Which nominal head.

    The returned word is surface-derived and morphology-normalized.  It is not
    a vocabulary or dataset whitelist; the ontology supplies all type evidence
    used by the caller.
    """
    match = re.search(r"\b(?:what|which)\s+([A-Za-z][A-Za-z-]*)\b", question, re.I)
    if match is None:
        return ""
    head = _singular_word(match.group(1))
    if head in _AUXILIARY_ANSWER_HEADS or head in _LITERAL_ANSWER_HEADS:
        return ""
    return head


def _question_requests_entity(question: str, answer_head: str) -> bool:
    text = " ".join(str(question).strip().casefold().split())
    if re.match(r"^(?:who|where)\b", text):
        return True
    if re.search(r"\bwhich\b", text):
        return True
    return bool(answer_head)


def graph_reconstruction_answer_role_gate(
    question: str,
    graph: QueryGraphCandidate,
    ontology: Any,
    *,
    answer_count: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Reject only a schema-provable answer-role error on a recovery graph.

    Graph reconstruction is allowed to merge grounded paths, but it also lets
    an intermediate path variable become the answer.  A non-empty result can
    therefore be a film, CVT, owner, or event even though the question asks for
    an actor, station, team, or person.  Such a result prevents the ordinary
    final-empty recovery stack from running.

    This gate is deliberately asymmetric: missing/ambiguous ontology evidence
    always keeps the graph.  Rejection requires one of three generic proofs:

    * a subject ``Who`` (or ``name ... who``) projects no agent type;
    * an entity question projects only mediator/CVT types; or
    * the direct What/Which head matches an adjacent variable's schema role but
      not the selected answer variable's role.

    It performs no endpoint, embedding, or model call and contains no entity,
    relation, question, or dataset-item routing list.
    """
    diagnostics: dict[str, Any] = {
        "status": "accepted",
        "reason": "not_applicable",
        "model_calls": 0,
        "endpoint_queries": 0,
        "embedding_calls": 0,
        "uses_gold": False,
    }
    if not is_graph_reconstruction_graph(graph):
        return True, diagnostics
    if ontology is None:
        diagnostics["reason"] = "ontology_unavailable"
        return True, diagnostics

    answer_var = str(graph.answer_var)
    actual_types = _variable_schema_types(graph, answer_var, ontology)
    diagnostics.update(
        {
            "reason": "insufficient_hard_type_evidence",
            "answer_var": answer_var,
            "answer_types": sorted(actual_types),
        }
    )
    if not actual_types:
        return True, diagnostics

    closures = {
        type_id: sorted(_type_closure(type_id, ontology))
        for type_id in actual_types
    }
    diagnostics["answer_type_closures"] = closures
    closure_union = {
        inherited for values in closures.values() for inherited in values
    }
    text = " ".join(str(question).strip().casefold().split())
    answer_head = _question_answer_head(question)
    # An embedded Who is the main request in common fronted constructions
    # ("In the film ..., who played ...?").  Do not apply that reading when a
    # direct What/Which nominal head already identifies the answer role; there
    # Who is normally only a relative-clause constraint on another entity.
    subject_who = bool(
        (
            re.match(r"^who\b", text)
            and not re.match(r"^who\s+(?:did|do|does)\b", text)
        )
        or (
            not re.match(r"^who\b", text)
            and not answer_head
            and re.search(r"\bwho\b", text)
        )
    )
    named_who_role = bool(
        re.search(r"\bname\s+of\b[^?]*\bwho\b", text)
    )
    diagnostics["subject_who"] = subject_who
    diagnostics["named_who_role"] = named_who_role

    # A reconstructed graph is only a last-resort recovery.  If a question
    # asks for one definite named role (``the president``, ``the architect``)
    # but the recovery expands to hundreds of entities, the reconstruction
    # has demonstrably dropped the role constraint.  Keep the threshold high
    # and limit this to singular/definite requests: open-ended plural Who
    # questions are intentionally unaffected.  Rejecting here lets the
    # existing EMPTY_RESULT recovery path try its bounded ontology retrieval.
    definite_role = bool(
        named_who_role
        or re.match(
            r"^who\s+(?:was|is|were|are)\s+(?:the|a|an)\s+"
            r"[a-z][a-z-]*(?:\s+[a-z][a-z-]*)?\b",
            text,
        )
    )
    diagnostics["definite_singular_role"] = definite_role
    diagnostics["answer_count"] = answer_count
    if definite_role and answer_count is not None and answer_count > 256:
        diagnostics.update(
            {
                "status": "rejected",
                "reason": "definite_role_constraint_was_dropped",
            }
        )
        return False, diagnostics

    if (subject_who or named_who_role) and not (
        {"people.person", "base.type_ontology.agent"} & closure_union
    ):
        diagnostics.update(
            {
                "status": "rejected",
                "reason": "requested_agent_but_answer_role_is_non_agent",
            }
        )
        return False, diagnostics

    diagnostics["answer_head"] = answer_head
    non_scalar_types = {
        type_id for type_id in actual_types if not type_id.startswith("type.")
    }
    mediator_only = bool(non_scalar_types) and all(
        "common.topic" not in _type_closure(type_id, ontology)
        for type_id in non_scalar_types
    )
    diagnostics["mediator_only"] = mediator_only
    if mediator_only and _question_requests_entity(question, answer_head):
        diagnostics.update(
            {
                "status": "rejected",
                "reason": "entity_question_answered_by_mediator_role",
            }
        )
        return False, diagnostics

    if answer_head:
        actual_heads = {
            head
            for type_id in actual_types
            if (head := _type_leaf_head(type_id))
        }
        diagnostics["answer_type_heads"] = sorted(actual_heads)
        if answer_head not in actual_heads:
            neighbor_evidence: dict[str, list[str]] = {}
            for triple in graph.triples:
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    continue
                subject, _, object_ = (str(value) for value in triple)
                neighbor = (
                    object_ if subject == answer_var else
                    subject if object_ == answer_var else
                    ""
                )
                if not neighbor.startswith("V"):
                    continue
                neighbor_types = _variable_schema_types(graph, neighbor, ontology)
                matching = sorted(
                    type_id
                    for type_id in neighbor_types
                    if _type_leaf_head(type_id) == answer_head
                )
                if matching:
                    neighbor_evidence[neighbor] = matching
            diagnostics["matching_neighbor_roles"] = neighbor_evidence
            if neighbor_evidence:
                diagnostics.update(
                    {
                        "status": "rejected",
                        "reason": "requested_role_belongs_to_adjacent_variable",
                    }
                )
                return False, diagnostics

    diagnostics["reason"] = "no_hard_answer_role_conflict"
    return True, diagnostics


def _type_compatibility(ontology: Any, left: Iterable[str], right: Iterable[str]) -> float:
    left_values = {value for value in left if value}
    right_values = {value for value in right if value}
    if not left_values or not right_values:
        return 0.25
    best = -1.0
    for left_type in left_values:
        for right_type in right_values:
            if left_type == right_type:
                best = max(best, 4.0)
                continue
            left_supers = set(ontology.supertypes(left_type))
            right_supers = set(ontology.supertypes(right_type))
            if left_type in right_supers or right_type in left_supers:
                best = max(best, 3.0)
            # Sibling types sharing only a broad ancestor are not proof that
            # two query variables denote the same node.  Accept only exact or
            # direct/transitive parent-child compatibility; this prevents a
            # performance record from being merged with its actor merely
            # because both ultimately inherit a generic Freebase root.
    return best


def _path_views(grounded: GroundedSemanticCandidate, ontology: Any) -> list[_PathView]:
    result: list[_PathView] = []
    paths = grounded.compose_input.get("semantic_paths", [])
    for path in paths:
        if not isinstance(path, dict):
            continue
        path_id = str(path.get("id", ""))
        anchor_ref = str(path.get("anchor_ref", ""))
        anchor_binding = grounded.anchor_bindings.get(anchor_ref)
        anchor = str(anchor_binding.entity_id) if anchor_binding is not None else ""
        if not path_id or not _ENTITY_ID_RE.fullmatch(anchor):
            continue
        variables: list[str] = []
        triples: list[tuple[str, str, str]] = []
        node_types: dict[str, set[str]] = defaultdict(set)
        valid = True
        for step in path.get("steps", []):
            if not isinstance(step, dict):
                valid = False
                break
            step_id = str(step.get("id", ""))
            relation_id = str(grounded.relation_bindings.get(step_id, ""))
            source_ref = str(step.get("from", ""))
            target_ref = str(step.get("to", ""))
            source = anchor if source_ref == anchor_ref else source_ref
            target = target_ref
            if not relation_id or not source or not target:
                valid = False
                break
            if target not in variables:
                variables.append(target)
            if str(step.get("direction", "")) == "forward":
                subject, object_ = source, target
            elif str(step.get("direction", "")) == "backward":
                subject, object_ = target, source
            else:
                valid = False
                break
            triples.append((subject, relation_id, object_))
            if subject.startswith("P"):
                node_types[subject].add(str(ontology.domain_for_relation(relation_id)))
            if object_.startswith("P"):
                node_types[object_].add(str(ontology.range_for_relation(relation_id)))
        terminal = str(path.get("path_output_var", ""))
        if not valid or not triples or terminal not in variables:
            continue
        result.append(
            _PathView(
                path_id=path_id,
                anchor=anchor,
                variables=tuple(variables),
                terminal=terminal,
                triples=tuple(triples),
                node_types={
                    key: tuple(sorted(value for value in values if value))
                    for key, values in node_types.items()
                },
            )
        )
    return result


def _is_entity_question(question: str) -> bool:
    return bool(re.match(r"\s*(?:who|which|what)\b", question, re.I)) and not bool(
        re.match(r"\s*(?:when|how many|what (?:year|date|time))\b", question, re.I)
    )


def _compile_graph(
    *,
    grounded: GroundedSemanticCandidate,
    paths: list[_PathView],
    merges: list[tuple[str, str]],
    answer: str,
    strategy: str,
    base_score: float,
    join_score: float,
    trim_at_join: bool,
    pipeline_version: str,
) -> _CompiledGraph | None:
    variables = {value for path in paths for value in path.variables}
    if answer not in variables:
        return None
    union = _UnionFind(variables)
    for left, right in merges:
        if left not in variables or right not in variables:
            return None
        union.union(left, right)

    join_nodes = {value for pair in merges for value in pair}
    retained: list[tuple[str, str, str]] = []
    for path in paths:
        triples = list(path.triples)
        if trim_at_join:
            relevant = [
                path.variables.index(value)
                for value in path.variables
                if value in join_nodes or value == answer
            ]
            if relevant:
                triples = triples[: max(relevant) + 1]
        retained.extend(triples)

    root_names: dict[str, str] = {}
    lowered: list[list[str]] = []
    for subject, relation, object_ in retained:
        nodes: list[str] = []
        for node in (subject, object_):
            if node in variables:
                root = union.find(node)
                if root not in root_names:
                    root_names[root] = f"V{len(root_names)}"
                nodes.append(root_names[root])
            else:
                nodes.append(node)
        if nodes[0] == nodes[1]:
            return None
        triple = [nodes[0], relation, nodes[1]]
        if triple not in lowered:
            lowered.append(triple)

    answer_var = root_names.get(union.find(answer), "")
    if not answer_var or not any(answer_var in (triple[0], triple[2]) for triple in lowered):
        return None
    anchors = sorted(
        {
            node
            for triple in lowered
            for node in (triple[0], triple[2])
            if _ENTITY_ID_RE.fullmatch(node)
        }
    )
    terminal_support = sum(
        union.find(path.terminal) == union.find(answer) for path in paths
    )
    score = base_score + join_score + (4.0 * terminal_support)
    if trim_at_join:
        score -= 0.5
    operators = [
        {
            "type": "NO_EQUAL",
            "inputs": [],
            "input_var": answer_var,
            "attribute_relation_label": [],
            "attribute_relation_labels": [],
            "value": anchor,
            "value_type": "mid",
            "_value_entity_id": anchor,
            "_graph_reconstruction_anchor_exclusion": True,
        }
        for anchor in anchors
    ]
    provenance = {
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
            "source": _FALLBACK_SOURCE,
            "model_used": False,
        },
        "graph_reconstruction": {
            "strategy": strategy,
            "path_count": len(paths),
            "edge_count": len(lowered),
            "terminal_support": terminal_support,
            "trimmed": trim_at_join,
            "merge_count": len(merges),
            "uses_gold": False,
            "model_used": False,
        },
    }
    graph = QueryGraphCandidate(
        graph_id="GR",
        triples=lowered,
        answer_var=answer_var,
        operators=operators,
        score=score,
        compose_output={
            "source": "grounded_semantic_graph_reconstruction",
            "strategy": strategy,
        },
        provenance=provenance,
    )
    key = json.dumps(
        {
            "triples": graph.triples,
            "answer_var": graph.answer_var,
            "operators": graph.operators,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _CompiledGraph(key=key, graph=graph)


def _candidate_graphs(
    grounded: GroundedSemanticCandidate,
    *,
    ontology: Any,
    pipeline_version: str,
    per_candidate_limit: int,
    allow_single_path: bool,
) -> list[_CompiledGraph]:
    paths = _path_views(grounded, ontology)
    if not paths:
        return []
    question = str(grounded.compose_input.get("question", ""))
    base_score = 10.0 * float(grounded.score)
    generated: list[_CompiledGraph] = []

    if len(paths) == 1:
        if not allow_single_path:
            return []
        compiled = _compile_graph(
            grounded=grounded,
            paths=paths,
            merges=[],
            answer=paths[0].terminal,
            strategy="single_grounded_path",
            base_score=base_score,
            join_score=0.0,
            trim_at_join=False,
            pipeline_version=pipeline_version,
        )
        return [compiled] if compiled is not None else []

    terminal_merges = [(paths[0].terminal, path.terminal) for path in paths[1:]]
    terminals_compatible = all(
        _type_compatibility(
            ontology,
            paths[0].node_types.get(paths[0].terminal, ()),
            path.node_types.get(path.terminal, ()),
        )
        >= 0.0
        for path in paths[1:]
    )
    if terminals_compatible:
        for trimmed in (False, True):
            compiled = _compile_graph(
                grounded=grounded,
                paths=paths,
                merges=terminal_merges,
                answer=paths[0].terminal,
                strategy="terminal_intersection",
                base_score=base_score,
                join_score=8.0 * (len(paths) - 1),
                trim_at_join=trimmed,
                pipeline_version=pipeline_version,
            )
            if compiled is not None:
                generated.append(compiled)

    for base_index, base in enumerate(paths):
        other_paths = [path for index, path in enumerate(paths) if index != base_index]
        attachment_options: list[list[tuple[float, str, str]]] = []
        for other in other_paths:
            options: list[tuple[float, str, str]] = []
            for other_node in other.variables:
                for base_node in base.variables:
                    compatibility = _type_compatibility(
                        ontology,
                        other.node_types.get(other_node, ()),
                        base.node_types.get(base_node, ()),
                    )
                    if compatibility < 0.0:
                        continue
                    terminal_bonus = 2.0 * (other_node == other.terminal)
                    terminal_bonus += 2.0 * (base_node == base.terminal)
                    options.append((compatibility + terminal_bonus, other_node, base_node))
            options.sort(key=lambda value: (-value[0], value[1], value[2]))
            attachment_options.append(options[:6])
        if any(not options for options in attachment_options):
            continue
        for selected in islice(product(*attachment_options), 216):
            merges = [(other_node, base_node) for _, other_node, base_node in selected]
            join_score = sum(value for value, _, _ in selected)
            for answer in base.variables:
                answer_types = set(base.node_types.get(answer, ()))
                if _is_entity_question(question) and answer_types and answer_types <= _SCALAR_TYPES:
                    continue
                answer_bonus = 5.0 if answer == base.terminal else 0.0
                for trimmed in (False, True):
                    compiled = _compile_graph(
                        grounded=grounded,
                        paths=paths,
                        merges=merges,
                        answer=answer,
                        strategy="typed_spine_attachment",
                        base_score=base_score + answer_bonus,
                        join_score=join_score,
                        trim_at_join=trimmed,
                        pipeline_version=pipeline_version,
                    )
                    if compiled is not None:
                        generated.append(compiled)

    best: dict[str, _CompiledGraph] = {}
    for compiled in generated:
        previous = best.get(compiled.key)
        if previous is None or compiled.graph.score > previous.graph.score:
            best[compiled.key] = compiled
    return sorted(
        best.values(),
        key=lambda item: (-item.graph.score, item.key),
    )[: max(1, int(per_candidate_limit))]


def _hybrid_candidates(
    grounded_candidates: list[GroundedSemanticCandidate],
) -> list[GroundedSemanticCandidate]:
    groups: dict[tuple[Any, ...], list[GroundedSemanticCandidate]] = defaultdict(list)
    for grounded in grounded_candidates:
        paths = [
            path
            for path in grounded.compose_input.get("semantic_paths", [])
            if isinstance(path, dict)
        ]
        key = (
            tuple((str(path.get("id", "")), len(path.get("steps", []))) for path in paths),
            tuple(
                sorted(
                    (anchor_id, candidate.entity_id)
                    for anchor_id, candidate in grounded.anchor_bindings.items()
                )
            ),
        )
        if paths:
            groups[key].append(grounded)

    result: list[GroundedSemanticCandidate] = []
    seen: set[str] = set()
    for values in groups.values():
        template = values[0]
        template_paths = template.compose_input.get("semantic_paths", [])
        choices: list[list[tuple[dict[str, Any], dict[str, str], float]]] = []
        for template_path in template_paths:
            path_id = str(template_path.get("id", ""))
            unique: dict[tuple[str, ...], tuple[dict[str, Any], dict[str, str], float]] = {}
            for grounded in values:
                candidate_path = next(
                    (
                        path
                        for path in grounded.compose_input.get("semantic_paths", [])
                        if isinstance(path, dict) and str(path.get("id", "")) == path_id
                    ),
                    None,
                )
                if candidate_path is None:
                    continue
                step_ids = [
                    str(step.get("id", ""))
                    for step in candidate_path.get("steps", [])
                    if isinstance(step, dict)
                ]
                bindings = {
                    step_id: str(grounded.relation_bindings.get(step_id, ""))
                    for step_id in step_ids
                }
                signature = tuple(bindings.get(step_id, "") for step_id in step_ids)
                if not signature or not all(signature):
                    continue
                item = (candidate_path, bindings, float(grounded.score))
                if signature not in unique or item[2] > unique[signature][2]:
                    unique[signature] = item
            choices.append(sorted(unique.values(), key=lambda item: -item[2])[:4])
        if any(not choice for choice in choices):
            continue
        for combination in islice(product(*choices), 128):
            compose_input = dict(template.compose_input)
            compose_input["semantic_paths"] = [item[0] for item in combination]
            relation_bindings: dict[str, str] = {}
            for _, bindings, _ in combination:
                relation_bindings.update(bindings)
            signature = json.dumps(
                {
                    "anchors": sorted(
                        (key, value.entity_id)
                        for key, value in template.anchor_bindings.items()
                    ),
                    "relations": sorted(relation_bindings.items()),
                },
                sort_keys=True,
            )
            if signature in seen:
                continue
            seen.add(signature)
            result.append(
                GroundedSemanticCandidate(
                    compose_input=compose_input,
                    anchor_bindings=dict(template.anchor_bindings),
                    relation_bindings=relation_bindings,
                    score=sum(item[2] for item in combination) / len(combination),
                    provenance={
                        **dict(template.provenance),
                        "graph_reconstruction_hybrid": {
                            "model_used": False,
                            "uses_gold": False,
                        },
                    },
                )
            )
    return result


def reconstruct_failure_query_graphs(
    grounded_candidates: list[GroundedSemanticCandidate],
    *,
    ontology: Any,
    pipeline_version: str,
    limit: int = 3,
    per_candidate_limit: int = 8,
    pool_limit: int = 80,
    allow_single_path: bool = False,
    allow_hybrid: bool = True,
    max_execution_limit: int = 3,
) -> tuple[list[QueryGraphCandidate], dict[str, Any]]:
    """Return at most ``limit`` Gold-blind graphs for an already-failed row."""
    budget = max(1, min(max(1, int(max_execution_limit)), int(limit)))
    if ontology is None or not grounded_candidates:
        return [], {
            "status": "not_compiled",
            "reason": "missing_ontology_or_grounded_candidates",
            "model_used": False,
            "uses_gold": False,
        }
    hybrids = (
        _hybrid_candidates(grounded_candidates)
        if allow_hybrid
        else []
    )
    candidates = [*grounded_candidates, *hybrids]
    best: dict[str, _CompiledGraph] = {}
    for grounded in candidates:
        for compiled in _candidate_graphs(
            grounded,
            ontology=ontology,
            pipeline_version=pipeline_version,
            per_candidate_limit=per_candidate_limit,
            allow_single_path=allow_single_path,
        ):
            previous = best.get(compiled.key)
            if previous is None or compiled.graph.score > previous.graph.score:
                best[compiled.key] = compiled
    ranked = sorted(
        best.values(),
        key=lambda item: (-item.graph.score, item.key),
    )[: max(budget, int(pool_limit))]
    graphs = [item.graph for item in ranked[:budget]]
    for rank, graph in enumerate(graphs, start=1):
        graph.graph_id = f"GR{rank}"
        graph.provenance["graph_reconstruction"]["rank"] = rank
        graph.provenance["graph_reconstruction"]["query_budget"] = budget
    return graphs, {
        "status": "compiled" if graphs else "not_compiled",
        "candidate_count": len(best),
        "bounded_pool_count": min(len(ranked), max(budget, int(pool_limit))),
        "returned_count": len(graphs),
        "execution_budget": budget,
        "grounded_candidate_count": len(grounded_candidates),
        "hybrid_candidate_count": len(candidates) - len(grounded_candidates),
        "model_used": False,
        "uses_gold": False,
    }
