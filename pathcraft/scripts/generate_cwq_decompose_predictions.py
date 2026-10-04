#!/usr/bin/env python3
"""Generate CWQ decomposition (subquestion) predictions from a local API.

The decomposition stage is the service mapped to port 18001.  Port 18002 is
the semantic-path stage and expects a decomposition as input; it cannot be
used to generate subquestions directly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib import error, request


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT / "data/v0.3.1_operator_glm_cwq/cwq_test_source.json"
DEFAULT_OUTPUT = PROJECT / "data/v0.3.1_operator_glm_cwq/decompose_test_pred_cwq.json"


INSTRUCTION = (
    "[TASK=DECOMPOSE] Use the KaeDe path-level decomposition format for CWQ. "
    "Return one item per complete entity-centric KG path. Each item may contain "
    "multiple hop-wise questions or statements for that same path; do not split "
    "those hop expressions into separate items. Preserve path order, explicit "
    "entities, dates, constraints, and the source wording. Do not invent KB IDs "
    "or facts. Return only one valid JSON object."
)
_KB_ID_RE = re.compile(r"(?<![A-Za-z0-9_])(?:m|g)\.[A-Za-z0-9_]+")


def _http_json(url: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    # A shell-wide proxy can point at a different container.  Local forwarded
    # ports must be reached directly.
    opener = request.build_opener(request.ProxyHandler({}))
    response = opener.open(request.Request(url, data=data, headers=headers), timeout=timeout)
    return json.loads(response.read().decode("utf-8"))


def _model_id(base_url: str, configured: str, timeout: float) -> str:
    if configured:
        return configured
    raw = _http_json(base_url.rstrip("/") + "/models", None, timeout)
    models = raw.get("data", []) if isinstance(raw, dict) else []
    if not models or not isinstance(models[0], dict) or not models[0].get("id"):
        raise RuntimeError(f"{base_url}/models returned no model id: {raw!r}")
    return str(models[0]["id"])


def _extract_decomposition(content: str) -> tuple[list[str], bool, str | None]:
    raw = content.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        # Tolerate a short markdown fence while still recording invalid JSON.
        if "```" in raw:
            candidate = raw.replace("```json", "").replace("```", "").strip()
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError as exc:
                return [], False, str(exc)
        else:
            return [], False, "model output is not valid JSON"
    if isinstance(value, dict):
        value = value.get("decomposition", value.get("prediction", []))
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return [], True, "decomposition must be a list"
    items = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
    return items, True, None


def _call(base_url: str, model: str, question: str, timeout: float, retries: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": INSTRUCTION + "\n" + json.dumps({"question": question}, ensure_ascii=False)}],
        "temperature": 0,
        "max_tokens": 1024,
        "stream": False,
    }
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = _http_json(base_url.rstrip("/") + "/chat/completions", body, timeout)
            choices = response.get("choices", []) if isinstance(response, dict) else []
            content = ""
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                content = str(message.get("content", "")) if isinstance(message, dict) else ""
            if not content:
                raise RuntimeError(f"chat completion returned no content: {response!r}")
            prediction, valid_json, parse_error = _extract_decomposition(content)
            return {
                "prediction": prediction,
                "raw_prediction": content,
                "validation": {
                    "json_valid": valid_json,
                    "schema_valid": parse_error is None,
                    "non_empty": bool(prediction),
                    "kb_id_leakage": any(
                        _KB_ID_RE.search(item) or "ns:" in item
                        for item in prediction
                    ),
                    **({"parse_error": parse_error} if parse_error else {}),
                },
            }
        except (OSError, ValueError, RuntimeError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(min(2.0 * (attempt + 1), 10.0))
    assert last is not None
    raise last


def _reference_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep exactly the seven columns used by the existing prediction file."""
    return {
        "instruction": INSTRUCTION,
        "input": row.get("input", ""),
        # This is an inference snapshot, not a supervised training array.
        # Match the existing reference file, whose output column is empty.
        "output": "",
        "history": row.get("history", []),
        "prediction": row.get("prediction", []),
        "raw_prediction": row.get("raw_prediction", ""),
        "validation": row.get("validation", {}),
    }


def _usable_prediction(row: dict[str, Any]) -> bool:
    """Only resume rows that contain a successful, non-empty prediction."""
    prediction = row.get("prediction")
    if isinstance(prediction, str):
        try:
            prediction = json.loads(prediction)
        except (TypeError, ValueError, json.JSONDecodeError):
            prediction = []
    validation = row.get("validation", {})
    return bool(prediction) and isinstance(validation, dict) and bool(
        validation.get("json_valid") and validation.get("schema_valid")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default="http://127.0.0.1:18001/v1")
    parser.add_argument("--model", default="", help="API model id; omitted means use the first /v1/models id")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining rows")
    parser.add_argument("--resume", action="store_true", help="reuse existing rows by source ID")
    parser.add_argument(
        "--resume-from",
        type=Path,
        action="append",
        default=[],
        help="additional prediction checkpoint(s) to merge when resuming; output takes precedence",
    )
    parser.add_argument("--workers", type=int, default=8, help="parallel API requests (default: 8)")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("CWQ input must be a JSON array")
    end = len(source) if args.limit <= 0 else min(len(source), args.start + args.limit)
    model = _model_id(args.base_url, args.model, args.timeout)
    # Build a stable source-key index before loading checkpoints.  Prediction
    # rows intentionally keep the seven reference columns and therefore do
    # not carry their source index; mapping by ID/question lets a resumed
    # range preserve rows generated by earlier ranges.
    source_key_to_index: dict[str, int] = {}
    for source_index, item in enumerate(source):
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("ID", source_index)).strip()
        question = str(item.get("question", "")).strip()
        for key in (source_id, question):
            if key:
                source_key_to_index.setdefault(key, source_index)
    existing: dict[str, dict[str, Any]] = {}

    def load_checkpoint(path: Path, *, overwrite: bool) -> None:
        if not path.is_file():
            return
        old = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(old, list):
            return
        for row in old:
            if not isinstance(row, dict):
                continue
            # The reference format has no metadata column.  Resume by the
            # question embedded in its JSON input (newer files may also
            # contain source_id, which is accepted for compatibility).
            key = str(row.get("source_id", ""))
            if not key:
                try:
                    key = str(json.loads(str(row.get("input", "{}"))).get("question", ""))
                except (TypeError, ValueError, json.JSONDecodeError):
                    key = ""
            if key and _usable_prediction(row) and (overwrite or key not in existing):
                existing[key] = _reference_row(row)

    if args.resume:
        # The output checkpoint wins over imported checkpoints, so a newer
        # target-directory result is never replaced by a legacy copy.
        for checkpoint in args.resume_from:
            load_checkpoint(checkpoint.expanduser().resolve(), overwrite=False)
        load_checkpoint(args.output, overwrite=True)

    rows_by_index: dict[int, dict[str, Any]] = {}
    if args.resume:
        # Keep successful checkpoint rows outside the requested range.  The
        # previous implementation only copied rows from ``args.start`` onward
        # and consequently truncated the output whenever a later range was
        # resumed.
        for key, row in existing.items():
            source_index = source_key_to_index.get(key)
            if source_index is not None:
                rows_by_index[source_index] = row
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(source[args.start:end], start=args.start):
        question = str(item.get("question", "")).strip()
        source_id = str(item.get("ID", index))
        if index not in rows_by_index:
            pending.append((index, item))

    def make_row(index: int, item: dict[str, Any]) -> dict[str, Any]:
        question = str(item.get("question", "")).strip()
        source_id = str(item.get("ID", index))
        try:
            result = _call(args.base_url, model, question, args.timeout, args.retries)
            return {
                "index": index,
                "instruction": INSTRUCTION,
                "input": json.dumps({"question": question}, ensure_ascii=False),
                # Prediction snapshots use ``prediction`` as the model result;
                # keep ``output`` empty exactly like the reference
                # decompose_test_pred.json (the training gold output field).
                "output": "",
                "history": [], "prediction": result["prediction"],
                "raw_prediction": result["raw_prediction"],
                "validation": {**result["validation"], "exact_match": None, "format_passed": result["validation"]["json_valid"] and result["validation"]["schema_valid"]},
            }
        except Exception as exc:  # preserve failed rows for resumable batches
            return {
                "index": index,
                "instruction": INSTRUCTION,
                "input": json.dumps({"question": question}, ensure_ascii=False),
                "output": "", "history": [], "prediction": [], "raw_prediction": "",
                "validation": {"json_valid": False, "schema_valid": False, "non_empty": False, "kb_id_leakage": False, "exact_match": None, "format_passed": False},
            }

    target_count = max(0, end - args.start)
    processed_target = sum(
        args.start <= index < end for index in rows_by_index
    )
    processed_existing = processed_target
    print(
        json.dumps(
            {
                "resume": bool(args.resume),
                "existing_successful_rows": len(rows_by_index),
                "existing_in_requested_range": processed_target,
                "requested_range": [args.start, end],
                "pending_requests": len(pending),
                "total_source_rows": len(source),
                "resume_from": [str(path.expanduser().resolve()) for path in args.resume_from],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(make_row, index, item) for index, item in pending]
        for future in as_completed(futures):
            row = future.result()
            rows_by_index[int(row["index"])] = row
            processed_target += 1
            if processed_target % max(1, args.checkpoint_every) == 0 or processed_target == target_count:
                checkpoint_rows = [rows_by_index[i] for i in sorted(rows_by_index)]
                checkpoint_rows = [_reference_row(row) for row in checkpoint_rows]
                args.output.parent.mkdir(parents=True, exist_ok=True)
                tmp = args.output.with_suffix(args.output.suffix + ".tmp")
                tmp.write_text(json.dumps(checkpoint_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                os.replace(tmp, args.output)
                completed_new = max(0, processed_target - processed_existing)
                print(
                    json.dumps(
                        {
                            "processed": processed_target,
                            "total": target_count,
                            "pending_requests": max(0, len(futures) - completed_new),
                            "last_index": row.get("index"),
                            "question": str(row.get("input", ""))[:160],
                            "valid": bool(row.get("validation", {}).get("format_passed", False)),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    rows = [_reference_row(rows_by_index[i]) for i in sorted(rows_by_index)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    valid = sum(bool(row.get("validation", {}).get("json_valid")) for row in rows)
    print(json.dumps({"dataset": "cwq", "rows": len(rows), "valid_json": valid, "output": str(args.output), "model": model}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
