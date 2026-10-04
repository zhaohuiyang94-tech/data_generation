"""Full-pipeline-verified CWQ decomposition review demonstrations."""
from __future__ import annotations

import json
from pathlib import Path


EXAMPLE_PATH = Path(__file__).with_name("resources") / "glm_cwq_full_pipeline_examples.json"
PRESERVE_PATH = Path(__file__).with_name("resources") / "glm_cwq_global_preserve_examples.json"
ENTITY_CONTEXT_PATH = Path(__file__).with_name("resources") / "glm_cwq_example_entity_context.json"
_BASE_REPAIR_SOURCE_INDEXES = {18, 67, 120, 201, 769, 1000, 1277, 1367, 9590}
_ENTITY_CONTEXT_REPAIR_SOURCE_INDEXES = (
    _BASE_REPAIR_SOURCE_INDEXES | {1007, 1105}
)
_ENTITY_CONTEXT_PRESERVE_SOURCE_INDEXES = {10}


def _load_entity_contexts() -> dict[str, list[dict]]:
    data = json.loads(ENTITY_CONTEXT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("CWQ example entity context resource must be an object")
    return data


def _example_entity_context(question: str) -> list[dict]:
    context = _load_entity_contexts().get(question, [])
    if not isinstance(context, list) or not context:
        raise ValueError(f"CWQ example lacks entity context: {question}")
    return context


def _load_examples() -> list[dict]:
    data = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("CWQ full-pipeline example resource is empty")
    for item in examples:
        verification = item.get("verification", {})
        if (verification.get("full_pipeline_verified") is not True
                or verification.get("candidate_f1") != 1.0
                or verification.get("candidate_exact") is not True):
            raise ValueError(f"Unverified CWQ example: {item.get('source_index')}")
        if not item.get("original_decomposition") or not item.get("corrected_decomposition"):
            raise ValueError(f"CWQ example lacks before/after decomposition: {item.get('source_index')}")
    return examples


def _load_preserve_examples() -> list[dict]:
    data = json.loads(PRESERVE_PATH.read_text(encoding="utf-8"))
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("CWQ global preserve examples are empty")
    for item in examples:
        verification = item.get("verification", {})
        if (verification.get("full_pipeline_verified") is not True
                or verification.get("candidate_f1") != 1.0
                or verification.get("candidate_exact") is not True):
            raise ValueError(f"Unverified CWQ preserve example: {item.get('source_index')}")
        if not item.get("question") or not item.get("decomposition"):
            raise ValueError(f"Incomplete CWQ preserve example: {item.get('source_index')}")
    return examples


def _selected_repair_examples(*, include_entity_context: bool = False) -> list[dict]:
    source_indexes = (
        _ENTITY_CONTEXT_REPAIR_SOURCE_INDEXES
        if include_entity_context
        else _BASE_REPAIR_SOURCE_INDEXES
    )
    return [
        item for item in _load_examples()
        if int(item.get("source_index", -1)) in source_indexes
    ]


def cwq_examples(action: str, *, include_entity_context: bool = False) -> str:
    if action not in {"review", "rewrite"}:
        raise ValueError("CWQ examples support review and rewrite only")
    blocks = [
        "CWQ examples below are sampled across the full dataset, not from the first-20 regression slice. Preserve examples are complete and must not be rewritten. Repair examples are verified before/after cases whose corrected plans achieved full-pipeline F1=1. Use them for executable path structure and constraint ownership, never for facts or answer memorization.",
    ]
    if action == "review":
        preserve_examples = _load_preserve_examples()
        selected_preserve_examples = preserve_examples[:5]
        if include_entity_context:
            selected_preserve_examples += [
                item for item in preserve_examples[5:]
                if int(item.get("source_index", -1))
                in _ENTITY_CONTEXT_PRESERVE_SOURCE_INDEXES
            ]
        for item in selected_preserve_examples:
            request = {
                "question": item["question"],
                "decomposition": item["decomposition"],
            }
            if include_entity_context:
                request["entity_context"] = _example_entity_context(item["question"])
            response = {
                "is_reasonable": True,
                "issues": [],
                "reason": "The plan is executable in the trained path format, reaches the requested terminal, and keeps every explicit condition on its owner.",
            }
            blocks.extend([
                "CWQ preserve request: " + json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                "CWQ preserve response: " + json.dumps(response, ensure_ascii=False, separators=(",", ":")),
            ])
    for item in _selected_repair_examples(
        include_entity_context=include_entity_context,
    ):
        request = {
            "question": item["question"],
            "decomposition": item["original_decomposition"],
        }
        if include_entity_context:
            request["entity_context"] = _example_entity_context(item["question"])
        if action == "review":
            response = {
                "is_reasonable": False,
                "issues": [
                    item["error_type"].replace("_", " ")
                    + "; restore the requested terminal target and keep every condition attached to its owning path."
                ],
                "reason": "The original path does not express the complete CWQ answer path.",
            }
        else:
            request["issues"] = [
                item["error_type"].replace("_", " ")
                + "; repair only this confirmed path defect."
            ]
            response = {"decomposition": item["corrected_decomposition"]}
        blocks.extend([
            "CWQ example request: " + json.dumps(request, ensure_ascii=False, separators=(",", ":")),
            "CWQ example response: " + json.dumps(response, ensure_ascii=False, separators=(",", ":")),
        ])
    return "\n".join(blocks)
