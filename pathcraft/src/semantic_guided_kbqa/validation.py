from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .clients import ChatClient


StageValidator = Callable[[Any], dict[str, Any]]


SEMANTIC_VALIDATION_INSTRUCTION = """
You are the semantic-output verifier and corrector for a Freebase KBQA pipeline.
Inspect the question, decomposition, local validation errors, and candidate output.
Return exactly one corrected semantic JSON object and no explanation or wrapper.
The object must contain only anchors and semantic_paths. Anchor ids are A0, A1, ...;
path ids are P0, P1, ...; every step must be connected, use the canonical step id,
use direction forward or backward, and use a relation_label array in
[domain, type, property] form. Every path must start at its anchor_ref and may use
only its own Pn.Vm variables; never switch to another anchor or another path's
variables inside a path. Represent constraints from different anchors as separate
entity-centric paths. Do not merely patch the reported structural error: verify that
every relation meaning, direction, path output, and anchor role matches the question
and decomposition. Keep the intended question semantics, do not invent entities, and
do not add metadata. If the candidate is already valid, return it unchanged.
""".strip()


COMPOSE_VALIDATION_INSTRUCTION = """
You are the GLM-5.2 Compose-output verifier and corrector for a Freebase KBQA pipeline.
Inspect the question, grounded semantic input, local validation errors, and candidate graph.
Return exactly one corrected Compose JSON object and no explanation or wrapper.
The object must contain only entities, triples, and answer_var. Entity entries must be
objects with id and surface, with ids E0, E1, ...; triple nodes must be E* or V*;
relation_label must be a non-empty array; answer_var must be a variable used by a triple;
and triples must form one connected graph. Use only anchors and relation labels supplied by
the grounded semantic input. Copy anchor surfaces exactly when an entity is an anchor; do
not use aliases or invent relation names. If the candidate is already valid, return it unchanged.
""".strip()


OPERATOR_VALIDATION_INSTRUCTION = """
You are the GLM-5.2 Operator-output verifier and corrector for a Freebase KBQA pipeline.
Inspect the question, Compose graph, local validation errors, and candidate operator program.
Return exactly one corrected JSON object and no explanation or wrapper.
The object must contain only operators. Every operator must use the exact required fields,
reference a variable present in the Compose triples, and use a supported type. Do not add
EQUAL, NO_EQUAL, TC, ARGMIN, or ARGMAX unless the question and graph provide evidence.
Remove duplicate or conflicting constraints. Temporal constraints must use a temporal
relation and an explicit value. If no reliable operator is justified, return {"operators":[]}.
If the candidate is already valid, return it unchanged.
""".strip()


@dataclass(slots=True)
class CorrectionResult:
    value: dict[str, Any] | None
    trace: dict[str, Any]


class GLMOutputCorrector:
    """Run deterministic validation and one optional GLM-5.2 correction pass.

    The model is advisory: a correction is accepted only after the same local validator
    accepts it. A valid original is retained when the validator service fails or returns
    another invalid object.
    """

    def __init__(
        self,
        model: ChatClient | None,
        *,
        validate_valid: bool = True,
        max_attempts: int = 1,
    ) -> None:
        self.model = model
        self.validate_valid = bool(validate_valid)
        self.max_attempts = max(0, int(max_attempts))

    def run(
        self,
        *,
        stage: str,
        input_payload: dict[str, Any],
        candidate: Any,
        validator: StageValidator,
        schema: dict[str, Any] | None = None,
    ) -> CorrectionResult:
        original: dict[str, Any] | None = None
        original_error = ""
        try:
            original = validator(candidate)
        except Exception as exc:  # validator errors are returned as data for tracing
            original_error = str(exc)

        trace: dict[str, Any] = {
            "stage": stage,
            "local_valid": original is not None,
        }
        if original_error:
            trace["local_error"] = original_error

        if self.model is None or self.max_attempts == 0 or (
            original is not None and not self.validate_valid
        ):
            trace["status"] = "accepted_local" if original is not None else "rejected_local"
            return CorrectionResult(original, trace)

        instruction = {
            "semantic": SEMANTIC_VALIDATION_INSTRUCTION,
            "compose": COMPOSE_VALIDATION_INSTRUCTION,
            "operator": OPERATOR_VALIDATION_INSTRUCTION,
        }.get(stage, "Return exactly one corrected JSON object and no explanation.")
        repair_payload = {
            "question": input_payload.get("question", ""),
            "input": input_payload,
            "candidate_output": candidate,
            "local_validation": {
                "valid": original is not None,
                "error": original_error,
            },
        }
        trace["validator_requested"] = True
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                outputs = self.model.generate_json(
                    instruction=instruction,
                    payload=repair_payload,
                    schema=schema,
                    count=1,
                )
                raw_corrected = outputs[0] if outputs else {}
                corrected = _unwrap_corrected_output(raw_corrected)
                trace.setdefault("validator_outputs", []).append(corrected)
                try:
                    validated = validator(corrected)
                except Exception as exc:
                    last_error = str(exc)
                    repair_payload["local_validation"] = {
                        "valid": False,
                        "error": last_error,
                    }
                    continue
                trace["status"] = "corrected" if original is None else "verified_or_corrected"
                trace["attempts"] = attempt
                return CorrectionResult(validated, trace)
            except Exception as exc:
                last_error = str(exc)
                break

        if last_error:
            trace["validator_error"] = last_error
        trace["attempts"] = self.max_attempts
        if original is not None:
            trace["status"] = "kept_original_after_validator_failure"
            return CorrectionResult(original, trace)
        trace["status"] = "rejected_after_correction"
        return CorrectionResult(None, trace)


def semantic_output_schema() -> dict[str, Any]:
    """A permissive guided schema; the local contract remains authoritative."""
    step = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^P[0-9]+\\.S[0-9]+$"},
            "relation_label": {"type": "array", "items": {"type": "string"}},
            "direction": {"type": "string", "enum": ["forward", "backward"]},
            "from": {"type": "string"},
            "to": {"type": "string", "pattern": "^P[0-9]+\\.V[0-9]+$"},
        },
        "required": ["id", "relation_label", "direction", "from", "to"],
        "additionalProperties": False,
    }
    path = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^P[0-9]+$"},
            "anchor_ref": {"type": "string", "pattern": "^A[0-9]+$"},
            "goal": {"type": "string"},
            "steps": {"type": "array", "items": step, "minItems": 1},
            "path_output_var": {"type": "string", "pattern": "^P[0-9]+\\.V[0-9]+$"},
        },
        "required": ["id", "anchor_ref", "goal", "steps", "path_output_var"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "anchors": {"type": "array", "minItems": 1},
            "semantic_paths": {"type": "array", "minItems": 1, "items": path},
        },
        "required": ["anchors", "semantic_paths"],
        "additionalProperties": False,
    }


def _unwrap_corrected_output(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("corrected_output", "corrected", "output"):
        nested = value.get(key)
        if isinstance(nested, dict):
            return nested
    return value
