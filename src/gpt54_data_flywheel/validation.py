from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
import shlex
import subprocess
from typing import Any

from .schemas import EXECUTABLE_OPERATOR_TYPES, OPERATOR_TYPES


_ANCHOR_RE = re.compile(r"^A\d+$")
_PATH_RE = re.compile(r"^P\d+$")
_PATH_VAR_RE = re.compile(r"^P\d+\.V\d+$")
_PROGRAM_VAR_RE = re.compile(r"^V\d+$")
_ENTITY_RE = re.compile(r"^E\d+$")
_DOTTED_ID_RE = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+){2,}$")
_ATTRIBUTE_OPERATORS = {
    "ARGMAX",
    "ARGMIN",
    "TC",
}
_VALUE_OPERATORS = {
    "TC",
    "GREATER_THAN",
    "GREATER_OR_EQUAL",
    "LESS_THAN",
    "LESS_OR_EQUAL",
}


def normalize_operator(operator: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": str(operator.get("type", "")).upper(),
        "inputs": [str(value) for value in operator.get("inputs", []) or []],
        "input_var": str(operator.get("input_var", "")),
        "attribute_relation_label": _label(operator.get("attribute_relation_label", [])),
        "attribute_relation_labels": [
            _label(value)
            for value in operator.get("attribute_relation_labels", []) or []
            if _label(value)
        ],
        "value": str(operator.get("value", "")),
        "value_type": str(operator.get("value_type", "")),
    }


def expected_operators(source_compose: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in source_compose.get("operators", []) or []:
        if not isinstance(raw, dict):
            continue
        operator = normalize_operator(raw)
        if operator["type"] in EXECUTABLE_OPERATOR_TYPES:
            output.append(operator)
    return output


def operator_signature(operator: dict[str, Any]) -> tuple[Any, ...]:
    normalized = normalize_operator(operator)
    return (
        normalized["type"],
        tuple(normalized["attribute_relation_label"]),
        tuple(tuple(label) for label in normalized["attribute_relation_labels"]),
        normalized["value"],
        normalized["value_type"],
    )


def validate_semantic_graph(
    graph: Any,
    *,
    expected: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(graph, dict):
        return ["semantic_graph must be an object"]
    if set(graph) != {"anchors", "semantic_paths", "operators"}:
        errors.append("semantic_graph must contain exactly anchors, semantic_paths, and operators")

    raw_anchors = graph.get("anchors", [])
    raw_paths = graph.get("semantic_paths", [])
    raw_operators = graph.get("operators", [])
    if not isinstance(raw_anchors, list):
        errors.append("anchors must be an array")
        raw_anchors = []
    if not isinstance(raw_paths, list) or not raw_paths:
        errors.append("semantic_paths must be a non-empty array")
        raw_paths = []
    if not isinstance(raw_operators, list):
        errors.append("operators must be an array")
        raw_operators = []

    anchors: set[str] = set()
    for index, anchor in enumerate(raw_anchors):
        if not isinstance(anchor, dict):
            errors.append(f"anchors[{index}] must be an object")
            continue
        anchor_id = str(anchor.get("id", ""))
        surface = str(anchor.get("surface", "")).strip()
        if not _ANCHOR_RE.fullmatch(anchor_id) or anchor_id in anchors:
            errors.append(f"anchors[{index}] has invalid or duplicate id {anchor_id!r}")
        if not surface:
            errors.append(f"anchors[{index}] has an empty surface")
        if _DOTTED_ID_RE.fullmatch(surface):
            errors.append(f"anchors[{index}] leaks a dotted KB id")
        anchors.add(anchor_id)

    path_ids: set[str] = set()
    path_variables: set[str] = set()
    for path_index, path in enumerate(raw_paths):
        if not isinstance(path, dict):
            errors.append(f"semantic_paths[{path_index}] must be an object")
            continue
        path_id = str(path.get("id", ""))
        anchor_ref = str(path.get("anchor_ref", ""))
        if not _PATH_RE.fullmatch(path_id) or path_id in path_ids:
            errors.append(f"semantic_paths[{path_index}] has invalid or duplicate id {path_id!r}")
        path_ids.add(path_id)
        if anchor_ref not in anchors:
            errors.append(f"semantic_paths[{path_index}] refers to unknown anchor {anchor_ref!r}")
        if not str(path.get("goal", "")).strip():
            errors.append(f"semantic_paths[{path_index}] has an empty goal")
        steps = path.get("steps", [])
        if not isinstance(steps, list) or not steps:
            errors.append(f"semantic_paths[{path_index}].steps must be non-empty")
            continue
        current_ref = anchor_ref
        local_variables: set[str] = set()
        for step_index, step in enumerate(steps):
            location = f"semantic_paths[{path_index}].steps[{step_index}]"
            if not isinstance(step, dict):
                errors.append(f"{location} must be an object")
                continue
            expected_step_id = f"{path_id}.S{step_index}"
            if str(step.get("id", "")) != expected_step_id:
                errors.append(f"{location}.id must be {expected_step_id!r}")
            if str(step.get("direction", "")) not in {"forward", "backward"}:
                errors.append(f"{location}.direction is invalid")
            source_ref = str(step.get("from", ""))
            target_ref = str(step.get("to", ""))
            if source_ref != current_ref:
                errors.append(f"{location}.from breaks path connectivity")
            if not _PATH_VAR_RE.fullmatch(target_ref) or not target_ref.startswith(f"{path_id}."):
                errors.append(f"{location}.to must be a variable local to {path_id}")
            _validate_relation_label(step.get("relation_label"), f"{location}.relation_label", errors)
            local_variables.add(target_ref)
            path_variables.add(target_ref)
            current_ref = target_ref
        output_var = str(path.get("path_output_var", ""))
        if output_var != current_ref or output_var not in local_variables:
            errors.append(f"semantic_paths[{path_index}].path_output_var must be the final path variable")

    normalized_operators: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_operators):
        if not isinstance(raw, dict):
            errors.append(f"operators[{index}] must be an object")
            continue
        operator = normalize_operator(raw)
        normalized_operators.append(operator)
        _validate_operator(
            operator,
            path_variables,
            f"operators[{index}]",
            errors,
            allowed_types=OPERATOR_TYPES,
        )
        if operator["type"] == "AND":
            if any(not _PATH_RE.fullmatch(value) for value in operator["inputs"]):
                errors.append(f"operators[{index}].inputs must use semantic path ids")
        elif operator["inputs"]:
            errors.append(f"operators[{index}].inputs must be empty for unary operators")

    _validate_semantic_and_scope(path_ids, normalized_operators, errors)
    _compare_operator_requirements(
        [value for value in normalized_operators if value["type"] != "AND"],
        expected,
        "semantic_graph",
        errors,
    )
    return errors


def validate_compose_program(
    program: Any,
    *,
    semantic_graph: dict[str, Any],
    expected: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(program, dict):
        return ["compose_program must be an object"]
    if set(program) != {"entities", "triples", "operators", "answer_var"}:
        errors.append("compose_program must contain exactly entities, triples, operators, and answer_var")

    raw_entities = program.get("entities", [])
    raw_triples = program.get("triples", [])
    raw_operators = program.get("operators", [])
    if not isinstance(raw_entities, list):
        errors.append("entities must be an array")
        raw_entities = []
    if not isinstance(raw_triples, list) or not raw_triples:
        errors.append("triples must be a non-empty array")
        raw_triples = []
    if not isinstance(raw_operators, list):
        errors.append("operators must be an array")
        raw_operators = []

    anchor_surfaces = {
        str(anchor.get("surface", "")).strip()
        for anchor in semantic_graph.get("anchors", []) or []
        if isinstance(anchor, dict)
    }
    entity_ids: set[str] = set()
    for index, entity in enumerate(raw_entities):
        if not isinstance(entity, dict):
            errors.append(f"entities[{index}] must be an object")
            continue
        entity_id = str(entity.get("id", ""))
        surface = str(entity.get("surface", "")).strip()
        if not _ENTITY_RE.fullmatch(entity_id) or entity_id in entity_ids:
            errors.append(f"entities[{index}] has invalid or duplicate id {entity_id!r}")
        if not surface:
            errors.append(f"entities[{index}] has an empty surface")
        elif surface not in anchor_surfaces:
            errors.append(f"entities[{index}] surface {surface!r} is not a semantic anchor")
        if _DOTTED_ID_RE.fullmatch(surface):
            errors.append(f"entities[{index}] leaks a dotted KB id")
        entity_ids.add(entity_id)

    variables: set[str] = set()
    adjacency: dict[str, set[str]] = {}
    for index, triple in enumerate(raw_triples):
        location = f"triples[{index}]"
        if not isinstance(triple, dict):
            errors.append(f"{location} must be an object")
            continue
        subject = str(triple.get("subject", ""))
        obj = str(triple.get("object", ""))
        for field, term in (("subject", subject), ("object", obj)):
            if _PROGRAM_VAR_RE.fullmatch(term):
                variables.add(term)
            elif _ENTITY_RE.fullmatch(term):
                if term not in entity_ids:
                    errors.append(f"{location}.{field} refers to unknown entity {term!r}")
            elif _quoted_literal(term):
                pass
            else:
                errors.append(f"{location}.{field} has invalid reference {term!r}")
        _validate_relation_label(triple.get("relation_label"), f"{location}.relation_label", errors)
        adjacency.setdefault(subject, set()).add(obj)
        adjacency.setdefault(obj, set()).add(subject)

    answer_var = str(program.get("answer_var", ""))
    if not _PROGRAM_VAR_RE.fullmatch(answer_var) or answer_var not in variables:
        errors.append("answer_var must be a variable present in triples")
    if adjacency:
        visited: set[str] = set()
        pending = [next(iter(adjacency))]
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            pending.extend(adjacency.get(node, set()) - visited)
        if visited != set(adjacency):
            errors.append("compose triples are disconnected")

    normalized_operators: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_operators):
        if not isinstance(raw, dict):
            errors.append(f"operators[{index}] must be an object")
            continue
        operator = normalize_operator(raw)
        normalized_operators.append(operator)
        _validate_operator(
            operator,
            variables,
            f"operators[{index}]",
            errors,
            allowed_types=EXECUTABLE_OPERATOR_TYPES,
        )
        if operator["inputs"]:
            errors.append(f"operators[{index}].inputs must be empty for unary operators")

    semantic_operators = [
        normalize_operator(value)
        for value in semantic_graph.get("operators", []) or []
        if isinstance(value, dict) and str(value.get("type", "")).upper() != "AND"
    ]
    _compare_operator_requirements(normalized_operators, semantic_operators, "compose_program", errors)
    _compare_operator_requirements(normalized_operators, expected, "compose_program", errors)
    mappings = _semantic_program_mappings(semantic_graph, program)
    if not mappings:
        errors.append("semantic paths cannot be mapped consistently onto compose triples")
    elif not any(
        _operators_follow_mapping(semantic_operators, normalized_operators, mapping)
        for mapping in mappings
    ):
        errors.append("semantic operator variables are not preserved by the path-to-program mapping")
    return errors


def run_external_validator(
    command: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> list[str]:
    if not command.strip():
        return []
    try:
        completed = subprocess.run(
            shlex.split(command),
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"external validator could not run: {exc}"]
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return [f"external validator exited {completed.returncode}: {detail[:1000]}"]
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return [f"external validator returned invalid JSON: {exc}"]
    if not isinstance(result, dict) or result.get("valid") is not True:
        raw_errors = result.get("errors", []) if isinstance(result, dict) else []
        if not isinstance(raw_errors, list):
            raw_errors = [str(raw_errors)]
        return [str(value) for value in raw_errors] or ["external validator rejected candidate"]
    return []


def _validate_operator(
    operator: dict[str, Any],
    variables: set[str],
    location: str,
    errors: list[str],
    *,
    allowed_types: tuple[str, ...],
) -> None:
    operator_type = operator["type"]
    if operator_type not in allowed_types:
        errors.append(f"{location} has unsupported type {operator_type!r}")
    if operator_type == "AND":
        if operator["input_var"]:
            errors.append(f"{location}.input_var must be empty for AND")
        if len(operator["inputs"]) < 2 or len(set(operator["inputs"])) != len(operator["inputs"]):
            errors.append(f"{location}.inputs must contain at least two unique path ids")
        if _operator_has_attribute(operator):
            errors.append(f"{location} must not define attribute relations for AND")
        if operator["value"] or operator["value_type"]:
            errors.append(f"{location} must not define value fields for AND")
        return

    input_var = operator["input_var"]
    if not input_var or input_var not in variables:
        errors.append(f"{location}.input_var must refer to a graph variable")
    for value in operator["inputs"]:
        if value not in variables:
            errors.append(f"{location}.inputs refers to unknown variable {value!r}")
    labels = [
        operator["attribute_relation_label"],
        *operator["attribute_relation_labels"],
    ]
    if operator_type in _ATTRIBUTE_OPERATORS and not any(len(label) >= 2 for label in labels):
        errors.append(f"{location} requires a canonical attribute relation label")
    for label_index, label in enumerate(labels):
        if label:
            _validate_relation_label(
                label,
                f"{location}.attribute_labels[{label_index}]",
                errors,
                min_segments=2,
            )
    if operator_type in _VALUE_OPERATORS and not operator["value"]:
        errors.append(f"{location} requires a value")
    if operator_type in {"COUNT", "ARGMAX", "ARGMIN"} and (
        operator["value"] or operator["value_type"]
    ):
        errors.append(f"{location} must not define value fields for {operator_type}")
    if operator_type == "COUNT" and _operator_has_attribute(operator):
        errors.append(f"{location} must not define attribute relations for COUNT")


def _validate_semantic_and_scope(
    path_ids: set[str],
    operators: list[dict[str, Any]],
    errors: list[str],
) -> None:
    and_operators = [value for value in operators if value["type"] == "AND"]
    if len(path_ids) <= 1:
        if and_operators:
            errors.append("semantic_graph must not emit AND for fewer than two semantic paths")
        return
    if len(and_operators) != 1:
        errors.append("semantic_graph with multiple paths must contain exactly one normalized AND")
        return
    if set(and_operators[0]["inputs"]) != path_ids:
        errors.append("semantic AND inputs must cover every semantic path exactly once")


def _operator_has_attribute(operator: dict[str, Any]) -> bool:
    return bool(operator["attribute_relation_label"] or operator["attribute_relation_labels"])


def _semantic_program_mappings(
    semantic_graph: dict[str, Any],
    program: dict[str, Any],
) -> list[dict[str, str]]:
    anchor_targets: dict[str, set[str]] = defaultdict(set)
    surfaces = {
        str(anchor.get("id", "")): str(anchor.get("surface", "")).strip()
        for anchor in semantic_graph.get("anchors", []) or []
        if isinstance(anchor, dict)
    }
    for entity in program.get("entities", []) or []:
        if not isinstance(entity, dict):
            continue
        surface = str(entity.get("surface", "")).strip()
        entity_id = str(entity.get("id", ""))
        for anchor_id, anchor_surface in surfaces.items():
            if surface == anchor_surface:
                anchor_targets[anchor_id].add(entity_id)
    for triple in program.get("triples", []) or []:
        if not isinstance(triple, dict):
            continue
        for term in (str(triple.get("subject", "")), str(triple.get("object", ""))):
            literal = _quoted_literal(term)
            if literal is None:
                continue
            for anchor_id, anchor_surface in surfaces.items():
                if literal == anchor_surface:
                    anchor_targets[anchor_id].add(term)

    triples = [triple for triple in program.get("triples", []) or [] if isinstance(triple, dict)]
    edges: list[tuple[str, tuple[str, ...], str]] = []
    for path in semantic_graph.get("semantic_paths", []) or []:
        if not isinstance(path, dict):
            continue
        for step in path.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            source_ref = str(step.get("from", ""))
            target_ref = str(step.get("to", ""))
            if str(step.get("direction", "")) == "backward":
                subject, obj = target_ref, source_ref
            else:
                subject, obj = source_ref, target_ref
            edges.append((subject, tuple(_label(step.get("relation_label", []))), obj))

    candidates: list[list[tuple[str, str]]] = []
    for subject, relation, obj in edges:
        matches: list[tuple[str, str]] = []
        for triple in triples:
            if tuple(_label(triple.get("relation_label", []))) != relation:
                continue
            candidate_subject = str(triple.get("subject", ""))
            candidate_object = str(triple.get("object", ""))
            if _reference_can_map(subject, candidate_subject, anchor_targets) and _reference_can_map(
                obj, candidate_object, anchor_targets
            ):
                matches.append((candidate_subject, candidate_object))
        if not matches:
            return []
        candidates.append(matches)

    order = sorted(range(len(edges)), key=lambda index: len(candidates[index]))
    mappings: list[dict[str, str]] = []

    def visit(position: int, mapping: dict[str, str]) -> None:
        if len(mappings) >= 128:
            return
        if position >= len(order):
            mappings.append(dict(mapping))
            return
        edge_index = order[position]
        subject, _, obj = edges[edge_index]
        for candidate_subject, candidate_object in candidates[edge_index]:
            additions: list[str] = []
            valid = True
            for semantic_ref, program_ref in (
                (subject, candidate_subject),
                (obj, candidate_object),
            ):
                existing = mapping.get(semantic_ref)
                if existing is not None and existing != program_ref:
                    valid = False
                    break
                if existing is None:
                    mapping[semantic_ref] = program_ref
                    additions.append(semantic_ref)
            if valid:
                visit(position + 1, mapping)
            for semantic_ref in additions:
                mapping.pop(semantic_ref, None)

    visit(0, {})
    return mappings


def _reference_can_map(
    semantic_ref: str,
    program_ref: str,
    anchor_targets: dict[str, set[str]],
) -> bool:
    if _ANCHOR_RE.fullmatch(semantic_ref):
        return program_ref in anchor_targets.get(semantic_ref, set())
    if _PATH_VAR_RE.fullmatch(semantic_ref):
        return bool(_PROGRAM_VAR_RE.fullmatch(program_ref))
    return False


def _operators_follow_mapping(
    semantic_operators: list[dict[str, Any]],
    program_operators: list[dict[str, Any]],
    mapping: dict[str, str],
) -> bool:
    remaining = list(program_operators)
    for semantic_operator in semantic_operators:
        signature = operator_signature(semantic_operator)
        mapped_input = mapping.get(semantic_operator["input_var"], "")
        mapped_inputs = [mapping.get(value, "") for value in semantic_operator["inputs"]]
        match_index = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if operator_signature(candidate) == signature
                and candidate["input_var"] == mapped_input
                and candidate["inputs"] == mapped_inputs
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


def _compare_operator_requirements(
    actual: list[dict[str, Any]],
    required: list[dict[str, Any]],
    location: str,
    errors: list[str],
) -> None:
    actual_signatures = Counter(operator_signature(value) for value in actual)
    required_signatures = Counter(operator_signature(value) for value in required)
    if actual_signatures == required_signatures:
        return
    missing = list((required_signatures - actual_signatures).elements())
    extra = list((actual_signatures - required_signatures).elements())
    if missing:
        errors.append(f"{location} is missing required operator signatures: {missing!r}")
    if extra:
        errors.append(f"{location} has unsupported extra operator signatures: {extra!r}")


def _validate_relation_label(
    value: Any,
    location: str,
    errors: list[str],
    *,
    min_segments: int = 3,
) -> None:
    if not isinstance(value, list) or len(value) < min_segments:
        errors.append(f"{location} must contain at least {min_segments} canonical segments")
        return
    for segment in value:
        text = str(segment).strip()
        if not text:
            errors.append(f"{location} contains an empty segment")
        if "." in text:
            errors.append(f"{location} contains a dotted KB id")


def _label(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(segment) for segment in value]


def _quoted_literal(value: str) -> str | None:
    if len(value) < 2 or not value.startswith('"') or not value.endswith('"'):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, str) and decoded else None
