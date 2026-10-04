from __future__ import annotations

from typing import Any


RELATION_LABEL: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 3,
}

OPTIONAL_RELATION_LABEL: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string"},
}

EXECUTABLE_OPERATOR_TYPES = (
    "COUNT",
    "ARGMAX",
    "ARGMIN",
    "TC",
    "GREATER_THAN",
    "GREATER_OR_EQUAL",
    "LESS_THAN",
    "LESS_OR_EQUAL",
)

OPERATOR_TYPES = ("AND", *EXECUTABLE_OPERATOR_TYPES)


def _operator_schema(types: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "string", "enum": list(types)},
            "inputs": {"type": "array", "items": {"type": "string"}},
            "input_var": {"type": "string"},
            "attribute_relation_label": OPTIONAL_RELATION_LABEL,
            "attribute_relation_labels": {
                "type": "array",
                "items": OPTIONAL_RELATION_LABEL,
            },
            "value": {"type": "string"},
            "value_type": {"type": "string"},
        },
        "required": [
            "type",
            "inputs",
            "input_var",
            "attribute_relation_label",
            "attribute_relation_labels",
            "value",
            "value_type",
        ],
    }


SEMANTIC_OPERATOR_SCHEMA = _operator_schema(OPERATOR_TYPES)
COMPOSE_OPERATOR_SCHEMA = _operator_schema(EXECUTABLE_OPERATOR_TYPES)

SEMANTIC_GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "anchors": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "surface": {"type": "string"},
                },
                "required": ["id", "surface"],
            },
        },
        "semantic_paths": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "anchor_ref": {"type": "string"},
                    "goal": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "string"},
                                "relation_label": RELATION_LABEL,
                                "direction": {
                                    "type": "string",
                                    "enum": ["forward", "backward"],
                                },
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                            },
                            "required": [
                                "id",
                                "relation_label",
                                "direction",
                                "from",
                                "to",
                            ],
                        },
                    },
                    "path_output_var": {"type": "string"},
                },
                "required": [
                    "id",
                    "anchor_ref",
                    "goal",
                    "steps",
                    "path_output_var",
                ],
            },
        },
        "operators": {"type": "array", "items": SEMANTIC_OPERATOR_SCHEMA},
    },
    "required": ["anchors", "semantic_paths", "operators"],
}

COMPOSE_PROGRAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "surface": {"type": "string"},
                },
                "required": ["id", "surface"],
            },
        },
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string"},
                    "relation_label": RELATION_LABEL,
                    "object": {"type": "string"},
                },
                "required": ["subject", "relation_label", "object"],
            },
        },
        "operators": {"type": "array", "items": COMPOSE_OPERATOR_SCHEMA},
        "answer_var": {"type": "string"},
    },
    "required": ["entities", "triples", "operators", "answer_var"],
}

VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["pass", "corrected", "reject"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "semantic_graph": SEMANTIC_GRAPH_SCHEMA,
        "compose_program": COMPOSE_PROGRAM_SCHEMA,
    },
    "required": ["status", "issues", "semantic_graph", "compose_program"],
}


LIGHT_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["pass", "reject"]},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "issues"],
}
