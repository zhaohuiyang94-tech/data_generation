from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from .clients import ChatClient
from .contracts import ContractError, parse_json_object


def select_candidates(
    model: ChatClient,
    *,
    instruction: str,
    question: str,
    decompositions: list[str],
    candidates: list[dict[str, Any]],
    max_prompt_bytes: int = 0,
    rounds: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select with bounded prompts; compare batch winners until one remains.

    UTF-8 byte length conservatively bounds Llama's byte-level BPE token count.
    The configured budget leaves space for chat template tokens and generation.
    Graph structure and operators are never truncated; answer previews may shrink.
    """
    remaining = deepcopy(candidates)
    if not remaining:
        raise ValueError("selector requires at least one executed candidate")

    def payload_for(items):
        return {"question": question, "decompositions": decompositions, "candidates": items}

    def prompt_size(items):
        text = instruction + json.dumps(payload_for(items), ensure_ascii=False, separators=(",", ":"))
        return len(text.encode("utf-8"))

    def pack(items):
        batches: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for item in items:
            if max_prompt_bytes and prompt_size([item]) > max_prompt_bytes:
                return None
            if current and max_prompt_bytes and prompt_size([*current, item]) > max_prompt_bytes:
                batches.append(current)
                current = []
            current.append(item)
        if current:
            batches.append(current)
        return batches

    round_number = 0
    while True:
        batches = pack(remaining)
        # A round consisting entirely of singleton batches cannot reduce the
        # candidate set. Shrink answer previews and try again, with a hard stop.
        while batches is None or (len(remaining) > 1 and len(batches) == len(remaining)):
            changed = False
            for item in remaining:
                for field in ("answers", "answer_ids"):
                    values = item.get(field, [])
                    if len(values) > 1:
                        item[field] = values[:max(1, len(values) // 2)]
                        item["answers_truncated"] = True
                        changed = True
            if not changed:
                raise RuntimeError("SELECTOR_PROMPT_TOO_LARGE: graph structure exceeds prompt byte budget")
            batches = pack(remaining)
        round_number += 1
        winners = []
        for batch in batches:
            if len(batch) == 1 and len(batches) > 1:
                winners.append(batch[0])
                continue
            ids = [item["graph_id"] for item in batch]
            schema = {
                "type": "object",
                "properties": {
                    "selected_graph_id": {"type": "string", "enum": ids},
                    "reason_codes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["selected_graph_id", "reason_codes"],
                "additionalProperties": False,
            }
            payload = payload_for(batch)
            event = {"round": round_number, "candidate_ids": ids,
                     "prompt_bytes": prompt_size(batch), "input": deepcopy(payload)}
            rounds.append(event)
            try:
                outputs = model.generate_json(
                    instruction=instruction, payload=payload, schema=schema, count=1
                )
                decision = parse_json_object(outputs[0] if outputs else {})
                event["model_output"] = deepcopy(decision)
                selected_id = decision.get("selected_graph_id")
                if selected_id not in ids:
                    raise ContractError("selector must choose an executed graph in its current batch")
                reasons = decision.get("reason_codes")
                if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
                    raise ContractError("selector reason_codes must be an array of strings")
                event["status"] = "selected"
            except Exception as exc:
                event.update({"status": "error", "error": str(exc)})
                raise
            if len(batches) == 1:
                return {"selected_graph_id": selected_id, "reason_codes": reasons}
            winners.append(next(item for item in batch if item["graph_id"] == selected_id))
        remaining = winners
