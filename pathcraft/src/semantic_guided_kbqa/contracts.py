from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import re
from typing import Any


_ANCHOR_RE = re.compile(r"^A\d+$")
_PATH_RE = re.compile(r"^P\d+$")
_PATH_VAR_RE = re.compile(r"^P\d+\.V\d+$")
_GLOBAL_ENTITY_RE = re.compile(r"^E\d+$")
_GLOBAL_VAR_RE = re.compile(r"^V\d+$")


class ContractError(ValueError):
    pass


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    text = str(value).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ContractError("model output must be one JSON object")
    return parsed


def validate_semantic_graph(value: Any, *, max_hops: int = 2) -> dict[str, Any]:
    graph = parse_json_object(value)
    if set(graph) != {"anchors", "semantic_paths"}:
        raise ContractError("semantic output must contain only anchors and semantic_paths")
    anchors = graph.get("anchors")
    paths = graph.get("semantic_paths")
    if not isinstance(anchors, list) or not anchors:
        raise ContractError("anchors must be a non-empty array")
    if not isinstance(paths, list) or not paths:
        raise ContractError("semantic_paths must be a non-empty array")
    anchor_ids: set[str] = set()
    for anchor in anchors:
        if not isinstance(anchor, dict) or set(anchor) != {"id", "surface"}:
            raise ContractError("each anchor must contain only id and surface")
        anchor_id = str(anchor["id"])
        if not _ANCHOR_RE.fullmatch(anchor_id) or anchor_id in anchor_ids:
            raise ContractError(f"invalid or duplicate anchor id: {anchor_id}")
        if not str(anchor["surface"]).strip():
            raise ContractError(f"anchor {anchor_id} has an empty surface")
        anchor_ids.add(anchor_id)
    path_ids: set[str] = set()
    for path in paths:
        required = {"id", "anchor_ref", "goal", "steps", "path_output_var"}
        if not isinstance(path, dict) or set(path) != required:
            raise ContractError(f"semantic path must contain exactly {sorted(required)}")
        path_id = str(path["id"])
        anchor_ref = str(path["anchor_ref"])
        if not _PATH_RE.fullmatch(path_id) or path_id in path_ids:
            raise ContractError(f"invalid or duplicate path id: {path_id}")
        if anchor_ref not in anchor_ids:
            raise ContractError(f"path {path_id} uses unknown anchor {anchor_ref}")
        steps = path["steps"]
        if not isinstance(steps, list) or not 1 <= len(steps) <= max_hops:
            raise ContractError(f"path {path_id} must have 1..{max_hops} steps")
        expected_from = anchor_ref
        for index, step in enumerate(steps):
            required_step = {"id", "relation_label", "direction", "from", "to"}
            if not isinstance(step, dict) or set(step) != required_step:
                raise ContractError(f"path {path_id} step must contain exactly {sorted(required_step)}")
            if str(step["id"]) != f"{path_id}.S{index}":
                raise ContractError(f"path {path_id} has a non-canonical step id")
            if str(step["from"]) != expected_from:
                raise ContractError(f"path {path_id} is disconnected")
            if str(step["direction"]) not in {"forward", "backward"}:
                raise ContractError(f"path {path_id} has an invalid direction")
            label = step["relation_label"]
            if not isinstance(label, list) or not label or not all(str(x).strip() for x in label):
                raise ContractError(f"path {path_id} has an invalid relation_label")
            to_ref = str(step["to"])
            if not _PATH_VAR_RE.fullmatch(to_ref):
                raise ContractError(f"path {path_id} has an invalid local variable")
            expected_from = to_ref
        if str(path["path_output_var"]) != expected_from:
            raise ContractError(f"path {path_id} output is not its final variable")
        path_ids.add(path_id)
    return graph


def validate_compose_output(value: Any, semantic_graph: dict[str, Any]) -> dict[str, Any]:
    output = parse_json_object(value)
    required = {"selected_paths", "variable_equalities", "operators", "answer_var"}
    if set(output) != required:
        raise ContractError(f"compose output must contain exactly {sorted(required)}")
    known_paths = {str(path["id"]) for path in semantic_graph["semantic_paths"]}
    known_vars = {
        str(step["to"])
        for path in semantic_graph["semantic_paths"]
        for step in path["steps"]
    }
    selected = output["selected_paths"]
    if not isinstance(selected, list) or not selected:
        raise ContractError("selected_paths must be a non-empty array")
    if any(str(path_id) not in known_paths for path_id in selected):
        raise ContractError("selected_paths contains an unknown path")
    equalities = output["variable_equalities"]
    if not isinstance(equalities, list):
        raise ContractError("variable_equalities must be an array")
    for equality in equalities:
        if not isinstance(equality, dict) or set(equality) != {"left", "right"}:
            raise ContractError("each variable equality must contain left and right")
        if str(equality["left"]) not in known_vars or str(equality["right"]) not in known_vars:
            raise ContractError("variable equality references an unknown local variable")
    answer_var = str(output["answer_var"])
    if answer_var not in known_vars:
        raise ContractError("answer_var references an unknown local variable")
    if not isinstance(output["operators"], list):
        raise ContractError("operators must be an array")
    for operator in output["operators"]:
        if not isinstance(operator, dict) or not str(operator.get("type", "")):
            raise ContractError("each operator must contain a type")
    return output


def validate_compose_graph_output(value: Any) -> dict[str, Any]:
    """Validate the 0.2.0 COMPOSE contract using global E/V identifiers."""
    output = parse_json_object(value)
    required = {"entities", "triples", "answer_var"}
    if set(output) != required:
        raise ContractError(f"compose output must contain exactly {sorted(required)}")

    entities = output["entities"]
    triples = output["triples"]
    if not isinstance(entities, list) or not entities:
        raise ContractError("entities must be a non-empty array")
    if not isinstance(triples, list) or not triples:
        raise ContractError("triples must be a non-empty array")

    entity_ids: set[str] = set()
    normalized_entities: list[dict[str, str]] = []
    for entity in entities:
        if not isinstance(entity, dict) or set(entity) != {"id", "surface"}:
            raise ContractError("each entity must contain only id and surface")
        entity_id = str(entity["id"])
        surface = str(entity["surface"]).strip()
        if not _GLOBAL_ENTITY_RE.fullmatch(entity_id) or entity_id in entity_ids:
            raise ContractError(f"invalid or duplicate entity id: {entity_id}")
        if not surface:
            raise ContractError(f"entity {entity_id} has an empty surface")
        entity_ids.add(entity_id)
        normalized_entities.append({"id": entity_id, "surface": surface})

    variables: set[str] = set()
    normalized_triples: list[dict[str, Any]] = []
    for triple in triples:
        required_triple = {"subject", "relation_label", "object"}
        if not isinstance(triple, dict) or set(triple) != required_triple:
            raise ContractError(
                f"each triple must contain exactly {sorted(required_triple)}"
            )
        subject = str(triple["subject"])
        object_ = str(triple["object"])
        for ref in (subject, object_):
            if _GLOBAL_VAR_RE.fullmatch(ref):
                variables.add(ref)
            elif ref not in entity_ids:
                raise ContractError(f"triple references an unknown node: {ref}")
        label = triple["relation_label"]
        if not isinstance(label, list) or not label or not all(str(x).strip() for x in label):
            raise ContractError("triple relation_label must be a non-empty string array")
        normalized_triples.append(
            {
                "subject": subject,
                "relation_label": [str(part).strip() for part in label],
                "object": object_,
            }
        )

    answer_var = str(output["answer_var"])
    if answer_var not in variables:
        raise ContractError("answer_var must reference a variable used by a triple")
    _validate_connected_graph(normalized_triples)
    return {
        "entities": normalized_entities,
        "triples": normalized_triples,
        "answer_var": answer_var,
    }


_OPERATOR_TYPES = {
    "COUNT",
    "ARGMAX",
    "ARGMIN",
    "GREATER_THAN",
    "GREATER_OR_EQUAL",
    "LESS_THAN",
    "LESS_OR_EQUAL",
    "EQUAL",
    "TC",
    "NO_EQUAL",
}


def validate_operator_output(
    value: Any,
    compose_output: dict[str, Any],
) -> dict[str, Any]:
    """Validate the 0.2.0 OPERATOR contract against global compose variables."""
    output = parse_json_object(value)
    if set(output) != {"operators"}:
        raise ContractError("operator output must contain exactly operators")
    operators = output["operators"]
    if not isinstance(operators, list):
        raise ContractError("operators must be an array")
    known_vars = {
        str(triple[field])
        for triple in compose_output.get("triples", [])
        if isinstance(triple, dict)
        for field in ("subject", "object")
        if _GLOBAL_VAR_RE.fullmatch(str(triple.get(field, "")))
    }
    required_fields = {
        "type",
        "inputs",
        "input_var",
        "attribute_relation_label",
        "attribute_relation_labels",
        "value",
        "value_type",
    }
    normalized: list[dict[str, Any]] = []
    for operator in operators:
        if not isinstance(operator, dict) or set(operator) != required_fields:
            raise ContractError(
                f"each operator must contain exactly {sorted(required_fields)}"
            )
        operator_type = str(operator["type"]).upper()
        if operator_type not in _OPERATOR_TYPES:
            raise ContractError(f"unsupported operator type: {operator_type}")
        input_var = str(operator["input_var"])
        if input_var not in known_vars:
            raise ContractError(f"operator references an unknown variable: {input_var}")
        inputs = operator["inputs"]
        label = operator["attribute_relation_label"]
        labels = operator["attribute_relation_labels"]
        if not isinstance(inputs, list):
            raise ContractError("operator inputs must be an array")
        if not isinstance(label, list) or not isinstance(labels, list):
            raise ContractError("operator relation labels must be arrays")
        normalized.append(
            {
                "type": operator_type,
                "inputs": [str(item) for item in inputs],
                "input_var": input_var,
                "attribute_relation_label": [str(item) for item in label],
                "attribute_relation_labels": [str(item) for item in labels],
                "value": str(operator["value"]),
                "value_type": str(operator["value_type"]),
            }
        )
    return {"operators": normalized}


def _validate_connected_graph(triples: list[dict[str, Any]]) -> None:
    adjacency: dict[str, set[str]] = {}
    for triple in triples:
        subject = str(triple["subject"])
        object_ = str(triple["object"])
        adjacency.setdefault(subject, set()).add(object_)
        adjacency.setdefault(object_, set()).add(subject)
    pending = [next(iter(adjacency))]
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(adjacency[node] - visited)
    if visited != set(adjacency):
        raise ContractError("compose triples must form one connected graph")


@dataclass(slots=True)
class DecompositionCandidate:
    decomposition: list[str]
    score: float = 0.0
    source: str = "input"


@dataclass(slots=True)
class EntityCandidate:
    entity_id: str
    label: str
    score: float
    source: str = "freebase"


@dataclass(slots=True)
class GroundedSemanticCandidate:
    compose_input: dict[str, Any]
    anchor_bindings: dict[str, EntityCandidate]
    relation_bindings: dict[str, str]
    score: float
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryGraphCandidate:
    graph_id: str
    triples: list[list[str]]
    answer_var: str
    operators: list[dict[str, Any]]
    score: float
    compose_output: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    sparql: str = ""


@dataclass(slots=True)
class ExecutedGraph:
    graph: QueryGraphCandidate
    answer_ids: list[str]
    answers: list[dict[str, str]]
    row_count: int
