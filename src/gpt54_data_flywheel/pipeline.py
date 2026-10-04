from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Protocol

from .openai_client import Generation
from .prompts import (
    COMPOSE_INSTRUCTIONS,
    COMPOSE_SFT_INSTRUCTION,
    GATEWAY_COMPOSE_INSTRUCTIONS,
    GATEWAY_SEMANTIC_INSTRUCTIONS,
    GATEWAY_VERIFY_INSTRUCTIONS,
    LEAN_COMPOSE_INSTRUCTIONS,
    LEAN_SEMANTIC_INSTRUCTIONS,
    LEAN_VERIFY_INSTRUCTIONS,
    MICRO_COMPOSE_INSTRUCTIONS,
    MICRO_SEMANTIC_INSTRUCTIONS,
    MICRO_VERIFY_INSTRUCTIONS,
    SEMANTIC_INSTRUCTIONS,
    SEMANTIC_SFT_INSTRUCTION,
    VERIFY_INSTRUCTIONS,
)
from .schemas import (
    COMPOSE_PROGRAM_SCHEMA,
    LIGHT_VERIFICATION_SCHEMA,
    SEMANTIC_GRAPH_SCHEMA,
    VERIFICATION_SCHEMA,
)
from .validation import (
    expected_operators,
    run_external_validator,
    validate_compose_program,
    validate_semantic_graph,
)


class JsonGenerator(Protocol):
    model: str

    def generate_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        instructions: str,
        payload: dict[str, Any],
        reasoning_effort: str | None = None,
        gateway_instructions: str | None = None,
        output_tokens: int | None = None,
    ) -> Generation: ...


@dataclass(frozen=True)
class SourcePair:
    source_id: str
    index: int
    semantic_index: int
    compose_index: int
    question: str
    semantic_input: dict[str, Any]
    semantic_output: dict[str, Any]
    compose_input: dict[str, Any]
    compose_output: dict[str, Any]

    @property
    def expected_operators(self) -> list[dict[str, Any]]:
        return expected_operators(self.compose_output)

    @property
    def has_structural_and(self) -> bool:
        paths = self.semantic_output.get("semantic_paths", [])
        return isinstance(paths, list) and len(paths) > 1

    @property
    def has_semantic_operator(self) -> bool:
        return self.has_structural_and or bool(self.expected_operators)


@dataclass(frozen=True)
class FlywheelConfig:
    output_dir: Path
    split_name: str = "train"
    max_attempts: int = 5
    checkpoint_every: int = 20
    verifier_reasoning_effort: str = "high"
    external_validator_command: str = ""
    external_validator_timeout: float = 120.0
    pipeline_profile: str = "standard"
    show_progress: bool = False


class _ProgressReporter:
    def __init__(self, *, total: int, enabled: bool) -> None:
        self.total = max(1, total)
        self.enabled = enabled
        self.is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.started = time.monotonic()
        self.accepted = 0
        self.rejected = 0
        self.skipped = 0
        self._last_width = 0
        self._open_line = False

    def stage(
        self,
        *,
        position: int,
        attempt: int,
        max_attempts: int,
        stage: str,
        question: str,
    ) -> None:
        if not self.enabled:
            return
        self._write(
            completed=position - 1,
            position=position,
            status=f"{stage} | attempt {attempt}/{max_attempts}",
            question=question,
            final=False,
        )

    def finish(self, *, position: int, status: str, question: str) -> None:
        if status == "accepted":
            self.accepted += 1
        elif status == "rejected":
            self.rejected += 1
        elif status == "skipped":
            self.skipped += 1
        if not self.enabled:
            return
        self._write(
            completed=position,
            position=position,
            status=status,
            question=question,
            final=True,
        )

    def close(self) -> None:
        if self.enabled and self.is_tty and self._open_line:
            print(file=sys.stderr, flush=True)
            self._open_line = False

    def _write(
        self,
        *,
        completed: int,
        position: int,
        status: str,
        question: str,
        final: bool,
    ) -> None:
        width = 24
        ratio = min(1.0, max(0.0, completed / self.total))
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self.started
        eta = "--:--"
        if completed > 0 and completed < self.total:
            eta = _format_duration((elapsed / completed) * (self.total - completed))
        elif completed >= self.total:
            eta = "00:00"
        counts = f"A/R/S {self.accepted}/{self.rejected}/{self.skipped}"
        current = f"row {min(position, self.total)}/{self.total}"
        text = (
            f"[{bar}] {completed}/{self.total} {ratio * 100:5.1f}% | {current} | "
            f"{status} | {counts} | elapsed {_format_duration(elapsed)} eta {eta} | "
            f"{_short_question(question)}"
        )
        if self.is_tty:
            padding = " " * max(0, self._last_width - len(text))
            print(f"\r{text}{padding}", end="\n" if final else "", file=sys.stderr, flush=True)
            self._last_width = 0 if final else len(text)
            self._open_line = not final
        else:
            print(text, file=sys.stderr, flush=True)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _short_question(question: str, limit: int = 64) -> str:
    compact = " ".join(question.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def load_source_pairs(semantic_path: Path, compose_path: Path) -> list[SourcePair]:
    semantic_rows = _load_rows(semantic_path)
    compose_rows = _load_rows(compose_path)
    compose_by_question: dict[str, deque[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]]] = (
        defaultdict(deque)
    )
    for index, row in enumerate(compose_rows):
        compose_input = _row_json(row, "input", compose_path, index)
        compose_output = _row_json(row, "output", compose_path, index)
        question = str(compose_input.get("question", "")).strip()
        if not question:
            raise ValueError(f"{compose_path}: row {index} has no question")
        compose_by_question[question].append((index, row, compose_input, compose_output))

    pairs: list[SourcePair] = []
    for semantic_index, semantic_row in enumerate(semantic_rows):
        semantic_input = _row_json(semantic_row, "input", semantic_path, semantic_index)
        semantic_output = _row_json(semantic_row, "output", semantic_path, semantic_index)
        question = str(semantic_input.get("question", "")).strip()
        if not question:
            raise ValueError(f"{semantic_path}: row {semantic_index} has no question")
        if not compose_by_question[question]:
            raise ValueError(
                f"no compose row matches semantic row {semantic_index} question {question!r}"
            )
        compose_index, _, compose_input, compose_output = compose_by_question[question].popleft()
        identity = {
            "semantic_input": semantic_input,
            "semantic_output": semantic_output,
            "compose_output": compose_output,
        }
        digest = hashlib.sha256(_json_text(identity).encode("utf-8")).hexdigest()[:20]
        pairs.append(
            SourcePair(
                source_id=digest,
                index=len(pairs),
                semantic_index=semantic_index,
                compose_index=compose_index,
                question=question,
                semantic_input=semantic_input,
                semantic_output=semantic_output,
                compose_input=compose_input,
                compose_output=compose_output,
            )
        )

    leftovers = sum(len(queue) for queue in compose_by_question.values())
    if leftovers:
        raise ValueError(f"{compose_path} contains {leftovers} unmatched compose rows")
    return pairs


def source_summary(pairs: list[SourcePair]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    operator_rows = 0
    executable_operator_rows = 0
    for pair in pairs:
        requirements = pair.expected_operators
        if pair.has_semantic_operator:
            operator_rows += 1
        if requirements:
            executable_operator_rows += 1
        if pair.has_structural_and:
            counts["AND"] += 1
        counts.update(str(operator.get("type", "")) for operator in requirements)
    return {
        "pairs": len(pairs),
        "rows_with_operators": operator_rows,
        "rows_with_executable_operators": executable_operator_rows,
        "operator_counts": dict(sorted(counts.items())),
    }


class Flywheel:
    def __init__(
        self,
        *,
        client: JsonGenerator,
        config: FlywheelConfig,
        semantic_source: Path,
        compose_source: Path,
    ) -> None:
        if config.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if config.pipeline_profile not in {"standard", "lean", "micro"}:
            raise ValueError("pipeline_profile must be standard, lean, or micro")
        self.client = client
        self.config = config
        self.semantic_source = semantic_source.resolve()
        self.compose_source = compose_source.resolve()
        self.output_dir = config.output_dir.resolve()
        self.accepted_path = self.output_dir / "accepted.jsonl"
        self.rejected_path = self.output_dir / "rejected.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_manifest()

    def run(self, pairs: list[SourcePair]) -> dict[str, int]:
        accepted_ids = set(_accepted_records(self.accepted_path))
        accepted_now = 0
        rejected_now = 0
        skipped = 0
        progress = _ProgressReporter(total=len(pairs), enabled=self.config.show_progress)
        try:
            for position, pair in enumerate(pairs, start=1):
                if pair.source_id in accepted_ids:
                    skipped += 1
                    progress.finish(position=position, status="skipped", question=pair.question)
                    continue
                record = self._process_pair(
                    pair,
                    progress=progress,
                    position=position,
                )
                if record["status"] == "accepted":
                    _append_jsonl(self.accepted_path, record)
                    accepted_ids.add(pair.source_id)
                    accepted_now += 1
                    progress.finish(position=position, status="accepted", question=pair.question)
                    if accepted_now % max(1, self.config.checkpoint_every) == 0:
                        materialize(self.output_dir, split_name=self.config.split_name)
                else:
                    _append_jsonl(self.rejected_path, record)
                    rejected_now += 1
                    progress.finish(position=position, status="rejected", question=pair.question)
        finally:
            progress.close()
            materialize(self.output_dir, split_name=self.config.split_name)
        return {"accepted": accepted_now, "rejected": rejected_now, "skipped": skipped}

    def _process_pair(
        self,
        pair: SourcePair,
        *,
        progress: _ProgressReporter,
        position: int,
    ) -> dict[str, Any]:
        lean = self.config.pipeline_profile == "lean"
        micro = self.config.pipeline_profile == "micro"
        constrained = lean or micro
        requirements = pair.expected_operators
        feedback: list[str] = []
        trace: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        last_semantic: dict[str, Any] = {}
        last_compose: dict[str, Any] = {}

        for attempt in range(1, self.config.max_attempts + 1):
            progress.stage(
                position=position,
                attempt=attempt,
                max_attempts=self.config.max_attempts,
                stage="semantic 1/3: waiting for model",
                question=pair.question,
            )
            semantic_payload: dict[str, Any] = {
                "question": pair.question,
                "source_semantic_graph": pair.semantic_output,
                "required_operators": requirements,
                "validation_feedback": feedback[-2:] if micro else (feedback[-4:] if lean else feedback[-12:]),
            }
            if not constrained:
                semantic_payload["decomposition"] = pair.semantic_input.get("decomposition", [])
                semantic_payload["source_compose_program"] = pair.compose_output
            if last_semantic and not constrained:
                semantic_payload["previous_semantic_candidate"] = last_semantic
            semantic_generation = self.client.generate_json(
                schema_name="rewritten_semantic_graph",
                schema=SEMANTIC_GRAPH_SCHEMA,
                instructions=SEMANTIC_INSTRUCTIONS,
                payload=semantic_payload,
                gateway_instructions=(
                    MICRO_SEMANTIC_INSTRUCTIONS
                    if micro
                    else (LEAN_SEMANTIC_INSTRUCTIONS if lean else GATEWAY_SEMANTIC_INSTRUCTIONS)
                ),
                output_tokens=384 if micro else (1024 if lean else None),
            )
            _merge_usage(usage, semantic_generation.usage)
            last_semantic = semantic_generation.value
            semantic_errors = validate_semantic_graph(last_semantic, expected=requirements)
            attempt_trace: dict[str, Any] = {
                "attempt": attempt,
                "semantic_response_id": semantic_generation.response_id,
                "semantic_transport_mode": semantic_generation.transport_mode,
                "semantic_errors": semantic_errors,
            }
            if semantic_errors:
                feedback = semantic_errors
                trace.append(attempt_trace)
                continue

            progress.stage(
                position=position,
                attempt=attempt,
                max_attempts=self.config.max_attempts,
                stage="compose 2/3: waiting for model",
                question=pair.question,
            )
            compose_payload: dict[str, Any] = {
                "question": pair.question,
                "semantic_graph": last_semantic,
                "source_compose_program": pair.compose_output,
                "validation_feedback": feedback[-2:] if micro else (feedback[-4:] if lean else feedback[-12:]),
            }
            if last_compose and not constrained:
                compose_payload["previous_compose_candidate"] = last_compose
            compose_generation = self.client.generate_json(
                schema_name="rebuilt_compose_program",
                schema=COMPOSE_PROGRAM_SCHEMA,
                instructions=COMPOSE_INSTRUCTIONS,
                payload=compose_payload,
                gateway_instructions=(
                    MICRO_COMPOSE_INSTRUCTIONS
                    if micro
                    else (LEAN_COMPOSE_INSTRUCTIONS if lean else GATEWAY_COMPOSE_INSTRUCTIONS)
                ),
                output_tokens=384 if micro else (1024 if lean else None),
            )
            _merge_usage(usage, compose_generation.usage)
            last_compose = compose_generation.value
            compose_errors = validate_compose_program(
                last_compose,
                semantic_graph=last_semantic,
                expected=requirements,
            )
            attempt_trace["compose_response_id"] = compose_generation.response_id
            attempt_trace["compose_transport_mode"] = compose_generation.transport_mode
            attempt_trace["compose_errors"] = compose_errors
            if compose_errors:
                feedback = compose_errors
                trace.append(attempt_trace)
                continue

            external_errors = self._external_errors(pair, last_semantic, last_compose, attempt)
            attempt_trace["external_errors"] = external_errors
            if external_errors:
                feedback = external_errors
                trace.append(attempt_trace)
                continue

            progress.stage(
                position=position,
                attempt=attempt,
                max_attempts=self.config.max_attempts,
                stage="verifier 3/3: waiting for model",
                question=pair.question,
            )
            verify_generation = self.client.generate_json(
                schema_name=(
                    "flywheel_verification_light" if constrained else "flywheel_verification"
                ),
                schema=(LIGHT_VERIFICATION_SCHEMA if constrained else VERIFICATION_SCHEMA),
                instructions=VERIFY_INSTRUCTIONS,
                payload={
                    "question": pair.question,
                    "expected_operators": requirements,
                    "semantic_candidate": last_semantic,
                    "compose_candidate": last_compose,
                },
                reasoning_effort=self.config.verifier_reasoning_effort,
                gateway_instructions=(
                    MICRO_VERIFY_INSTRUCTIONS
                    if micro
                    else (LEAN_VERIFY_INSTRUCTIONS if lean else GATEWAY_VERIFY_INSTRUCTIONS)
                ),
                output_tokens=96 if micro else (256 if lean else None),
            )
            _merge_usage(usage, verify_generation.usage)
            verdict = verify_generation.value
            status = str(verdict.get("status", "reject"))
            issues = [str(value) for value in verdict.get("issues", []) or []]
            attempt_trace.update(
                {
                    "verify_response_id": verify_generation.response_id,
                    "verify_transport_mode": verify_generation.transport_mode,
                    "verify_status": status,
                    "verify_issues": issues,
                }
            )

            if status == "pass":
                if not constrained and (
                    verdict.get("semantic_graph") != last_semantic
                    or verdict.get("compose_program") != last_compose
                ):
                    feedback = ["verifier changed a candidate while claiming status=pass"]
                    attempt_trace["post_verify_errors"] = feedback
                    trace.append(attempt_trace)
                    continue
                trace.append(attempt_trace)
                return self._accepted_record(
                    pair,
                    attempt,
                    last_semantic,
                    last_compose,
                    issues,
                    trace,
                    usage,
                )

            if status == "corrected" and not constrained:
                corrected_semantic = verdict.get("semantic_graph")
                corrected_compose = verdict.get("compose_program")
                post_errors = validate_semantic_graph(corrected_semantic, expected=requirements)
                if isinstance(corrected_semantic, dict):
                    post_errors.extend(
                        validate_compose_program(
                            corrected_compose,
                            semantic_graph=corrected_semantic,
                            expected=requirements,
                        )
                    )
                else:
                    post_errors.append("verifier correction has no semantic graph")
                if not post_errors and isinstance(corrected_semantic, dict) and isinstance(corrected_compose, dict):
                    post_errors.extend(
                        self._external_errors(
                            pair,
                            corrected_semantic,
                            corrected_compose,
                            attempt,
                        )
                    )
                attempt_trace["post_verify_errors"] = post_errors
                trace.append(attempt_trace)
                if not post_errors and isinstance(corrected_semantic, dict) and isinstance(corrected_compose, dict):
                    return self._accepted_record(
                        pair,
                        attempt,
                        corrected_semantic,
                        corrected_compose,
                        issues,
                        trace,
                        usage,
                    )
                feedback = [*issues, *post_errors]
                last_semantic = corrected_semantic if isinstance(corrected_semantic, dict) else last_semantic
                last_compose = corrected_compose if isinstance(corrected_compose, dict) else last_compose
                continue

            feedback = issues or ["independent verifier rejected the candidate"]
            trace.append(attempt_trace)

        return {
            "status": "rejected",
            "source_id": pair.source_id,
            "source_index": pair.index,
            "semantic_source_index": pair.semantic_index,
            "compose_source_index": pair.compose_index,
            "question": pair.question,
            "attempts": self.config.max_attempts,
            "errors": feedback,
            "last_semantic_candidate": last_semantic,
            "last_compose_candidate": last_compose,
            "trace": trace,
            "usage": usage,
            "created_at": _utc_now(),
        }

    def _accepted_record(
        self,
        pair: SourcePair,
        attempt: int,
        semantic_graph: dict[str, Any],
        compose_program: dict[str, Any],
        issues: list[str],
        trace: list[dict[str, Any]],
        usage: dict[str, int],
    ) -> dict[str, Any]:
        compose_input = {
            "question": pair.question,
            "anchors": semantic_graph["anchors"],
            "semantic_paths": semantic_graph["semantic_paths"],
            "operators": semantic_graph["operators"],
        }
        return {
            "status": "accepted",
            "source_id": pair.source_id,
            "source_index": pair.index,
            "semantic_source_index": pair.semantic_index,
            "compose_source_index": pair.compose_index,
            "question": pair.question,
            "attempts": attempt,
            "verifier_issues": issues,
            "semantic_row": {
                "instruction": SEMANTIC_SFT_INSTRUCTION,
                "input": _json_text(pair.semantic_input),
                "output": _json_text(semantic_graph),
                "history": [],
            },
            "compose_row": {
                "instruction": COMPOSE_SFT_INSTRUCTION,
                "input": _json_text(compose_input),
                "output": _json_text(compose_program),
                "history": [],
            },
            "trace": trace,
            "usage": usage,
            "created_at": _utc_now(),
        }

    def _external_errors(
        self,
        pair: SourcePair,
        semantic_graph: dict[str, Any],
        compose_program: dict[str, Any],
        attempt: int,
    ) -> list[str]:
        return run_external_validator(
            self.config.external_validator_command,
            {
                "source_id": pair.source_id,
                "source_index": pair.index,
                "attempt": attempt,
                "question": pair.question,
                "semantic_graph": semantic_graph,
                "compose_program": compose_program,
                "source_semantic_graph": pair.semantic_output,
                "source_compose_program": pair.compose_output,
            },
            timeout=self.config.external_validator_timeout,
        )

    def _ensure_manifest(self) -> None:
        configuration = {
            "contract_version": "operator-v2-fewshot",
            "semantic_source": str(self.semantic_source),
            "semantic_source_sha256": _file_sha256(self.semantic_source),
            "compose_source": str(self.compose_source),
            "compose_source_sha256": _file_sha256(self.compose_source),
            "model": self.client.model,
            "api_mode": str(getattr(self.client, "api_mode", "unspecified")),
            "api_url": str(getattr(self.client, "api_url", "unspecified")),
            "json_mode": str(getattr(self.client, "json_mode", "unspecified")),
            "responses_url": str(getattr(self.client, "responses_url", "unspecified")),
            "generator_reasoning_effort": str(
                getattr(self.client, "reasoning_effort", "unspecified")
            ),
            "verifier_reasoning_effort": self.config.verifier_reasoning_effort,
            "external_validator_command": self.config.external_validator_command,
            "pipeline_profile": self.config.pipeline_profile,
            "split_name": self.config.split_name,
        }
        if self.manifest_path.exists():
            current = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            existing = current.get("configuration", {})
            if existing != configuration:
                counts = current.get("counts", {})
                has_records = any(
                    path.exists() and path.stat().st_size > 0
                    for path in (self.accepted_path, self.rejected_path)
                )
                if (
                    not has_records
                    and int(counts.get("accepted", 0) or 0) == 0
                    and int(counts.get("rejected_events", 0) or 0) == 0
                ):
                    current["configuration"] = configuration
                    current["updated_at"] = _utc_now()
                    _atomic_write_json(self.manifest_path, current)
                    return
                raise ValueError(
                    "output directory belongs to a different source/model configuration; "
                    "choose another --output directory"
                )
            return
        _atomic_write_json(
            self.manifest_path,
            {
                "format_version": 1,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "configuration": configuration,
                "counts": {"accepted": 0, "rejected_events": 0},
            },
        )


def materialize(output_dir: Path, *, split_name: str = "train") -> dict[str, Any]:
    root = output_dir.resolve()
    accepted = list(_accepted_records(root / "accepted.jsonl").values())
    accepted.sort(key=lambda record: int(record.get("source_index", 0)))
    semantic_rows = [record["semantic_row"] for record in accepted]
    compose_rows = [record["compose_row"] for record in accepted]
    semantic_name = f"semantic_path_{split_name}.json"
    compose_name = f"compose_{split_name}.json"
    _atomic_write_json(root / semantic_name, semantic_rows)
    _atomic_write_json(root / compose_name, compose_rows)
    _atomic_write_json(
        root / "dataset_info.json",
        {
            f"semantic_path_{split_name}": _dataset_entry(semantic_name),
            f"compose_{split_name}": _dataset_entry(compose_name),
        },
    )

    manifest_path = root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"format_version": 1, "created_at": _utc_now(), "configuration": {}}
    )
    operator_counts: Counter[str] = Counter()
    for row in semantic_rows:
        output = json.loads(row["output"])
        operator_counts.update(str(value.get("type", "")) for value in output.get("operators", []))
    manifest.update(
        {
            "updated_at": _utc_now(),
            "counts": {
                "accepted": len(accepted),
                "rejected_events": _line_count(root / "rejected.jsonl"),
                "operators": dict(sorted(operator_counts.items())),
            },
            "outputs": {
                "semantic_path": semantic_name,
                "compose": compose_name,
                "dataset_info": "dataset_info.json",
                "accepted_checkpoint": "accepted.jsonl",
                "rejected_audit": "rejected.jsonl",
            },
        }
    )
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain one JSON array")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} contains a non-object row")
    return value


def _row_json(row: dict[str, Any], field: str, path: Path, index: int) -> dict[str, Any]:
    raw = row.get(field)
    if not isinstance(raw, str):
        raise ValueError(f"{path}: row {index} field {field!r} must be a JSON string")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: row {index} field {field!r} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: row {index} field {field!r} must decode to an object")
    return value


def _accepted_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
            source_id = str(record.get("source_id", ""))
            if source_id and record.get("status") == "accepted":
                records[source_id] = record
    return records


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_text(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history",
        },
    }


def _merge_usage(target: dict[str, int], usage: dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
