from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .contracts import DecompositionCandidate, parse_json_object


@dataclass(slots=True)
class TrainingContract:
    semantic_instruction: str
    compose_instruction: str
    operator_instruction: str = ""
    operator_examples: list[dict[str, Any]] = field(default_factory=list)
    semantic_examples: list[dict[str, Any]] = field(default_factory=list)
    # Parsed Compose rows aligned one-for-one with ``semantic_examples``.
    # Keeping these local training templates avoids any model call during the
    # bounded missing-relation recovery lane.
    compose_examples: list[dict[str, Any]] = field(default_factory=list)
    compose_format: str = "legacy"

    @classmethod
    def load(
        cls,
        semantic_path: str | Path,
        compose_path: str | Path,
        operator_path: str | Path | None = "",
    ) -> "TrainingContract":
        semantic_rows = _load_array(semantic_path, role="semantic")
        compose_rows = _load_array(compose_path, role="compose")
        if not semantic_rows or not compose_rows:
            raise ValueError("semantic and compose training files must not be empty")
        # ``train_without_operator`` rows have exactly these fields.  The
        # optional ``operators`` allowances keep the reader backwards
        # compatible with an unprojected integrated row and make it possible
        # to inspect a bundle before its no-operator projection is materialized;
        # operators are never copied into the runtime contract.
        _validate_training_row(
            semantic_rows[0],
            {"question", "decomposition"},
            {"anchors", "semantic_paths"},
            optional_output_fields={"operators"},
        )
        compose_format = "legacy"
        try:
            _validate_training_row(
                compose_rows[0],
                {"question", "anchors", "semantic_paths"},
                {"entities", "triples", "answer_var"},
                optional_input_fields={"operators"},
                optional_output_fields={"operators"},
            )
            compose_format = "graph_v2"
        except ValueError:
            _validate_training_row(
                compose_rows[0],
                {"question", "anchors", "semantic_paths"},
                {"selected_paths", "variable_equalities", "operators", "answer_var"},
            )
        operator_instruction = ""
        operator_examples: list[dict[str, Any]] = []
        if operator_path:
            operator_rows = _load_array(operator_path, role="operator")
            if not operator_rows:
                raise ValueError("operator training file must not be empty")
            _validate_training_row(
                operator_rows[0],
                {"question", "entities", "triples", "answer_var"},
                {"operators"},
            )
            operator_instruction = str(operator_rows[0].get("instruction", ""))
            operator_examples = _select_operator_few_shot_examples(
                operator_rows,
                semantic_examples=_semantic_examples_by_question(semantic_rows),
            )
        semantic_examples: list[dict[str, Any]] = []
        compose_examples: list[dict[str, Any]] = []
        for source_index, row in enumerate(semantic_rows):
            try:
                input_object = parse_json_object(row.get("input", ""))
                output_object = parse_json_object(row.get("output", ""))
            except (TypeError, ValueError):
                continue
            output_object.pop("operators", None)
            question = str(input_object.get("question", "")).strip()
            if question and set(output_object) == {"anchors", "semantic_paths"}:
                semantic_examples.append(
                    {
                        "question": question,
                        "decomposition": [
                            str(item).strip()
                            for item in input_object.get("decomposition", [])
                            if str(item).strip()
                        ]
                        if isinstance(input_object.get("decomposition", []), list)
                        else [str(input_object.get("decomposition", "")).strip()],
                        "semantic_graph": output_object,
                    }
                )
                compose_output: dict[str, Any] = {}
                if source_index < len(compose_rows):
                    try:
                        candidate_input = parse_json_object(
                            compose_rows[source_index].get("input", "")
                        )
                        candidate_output = parse_json_object(
                            compose_rows[source_index].get("output", "")
                        )
                    except (TypeError, ValueError):
                        pass
                    else:
                        if str(candidate_input.get("question", "")).strip() == question:
                            candidate_output.pop("operators", None)
                            compose_output = candidate_output
                compose_examples.append(compose_output)
        return cls(
            semantic_instruction=str(semantic_rows[0].get("instruction", "")),
            compose_instruction=str(compose_rows[0].get("instruction", "")),
            operator_instruction=operator_instruction,
            operator_examples=operator_examples,
            semantic_examples=semantic_examples,
            compose_examples=compose_examples,
            compose_format=compose_format,
        )


class DecompositionStore:
    def __init__(self, predictions: dict[str, list[DecompositionCandidate]]) -> None:
        self._predictions = predictions

    @classmethod
    def load(cls, path: str | Path) -> "DecompositionStore":
        predictions: dict[str, list[DecompositionCandidate]] = {}
        for row in _load_array(path, role="decompose"):
            # KaeDe/vLLM prediction files and the gpt54 flywheel training
            # arrays both use an Alpaca wrapper (``input``/``output`` are
            # JSON strings).  Keep accepting the older direct-dict form as
            # well; ``parse_json_object`` handles either representation.
            raw_input = row.get("input", {})
            input_object = parse_json_object(raw_input)
            question = str(input_object.get("question", "")).strip()
            if not question:
                continue
            raw_candidates = row.get("candidates")
            candidates: list[DecompositionCandidate] = []
            if isinstance(raw_candidates, list):
                for rank, candidate in enumerate(raw_candidates):
                    if not isinstance(candidate, dict):
                        continue
                    prediction = candidate.get("prediction") or candidate.get("decomposition")
                    normalized = _coerce_decomposition(prediction)
                    if not normalized:
                        normalized = _coerce_decomposition(
                            _output_decomposition(candidate.get("output"))
                        )
                    if normalized:
                        candidates.append(
                            DecompositionCandidate(
                                normalized,
                                float(candidate.get("score", -0.01 * rank)),
                            )
                        )
            else:
                # Prediction snapshots generally expose ``prediction`` as a
                # list.  A training split instead stores the gold
                # decomposition in its JSON ``output`` field.  Supporting the
                # latter lets the pipeline be smoke-tested (or run as a
                # deterministic baseline) directly against an integrated
                # train/test bundle without a separate conversion step.
                prediction = row.get("prediction")
                normalized = _coerce_decomposition(prediction)
                if not normalized:
                    normalized = _coerce_decomposition(
                        _output_decomposition(row.get("output"))
                    )
                if normalized:
                    candidates.append(DecompositionCandidate(normalized, 0.0))
            if candidates:
                # Train+test arrays can contain repeated questions (for
                # example, a corrected and an original decomposition).  Do
                # not silently discard all but the last row: preserve distinct
                # candidates and let the normal beam/ranking logic consider
                # them.  Exact duplicate decompositions are collapsed while
                # retaining the best score.
                bucket = predictions.setdefault(question, [])
                for candidate in candidates:
                    duplicate = next(
                        (
                            existing
                            for existing in bucket
                            if existing.decomposition == candidate.decomposition
                        ),
                        None,
                    )
                    if duplicate is None:
                        bucket.append(candidate)
                    elif candidate.score > duplicate.score:
                        duplicate.score = candidate.score
        return cls(predictions)

    def questions(self) -> list[str]:
        return list(self._predictions)

    def get(self, question: str) -> list[DecompositionCandidate]:
        try:
            return list(self._predictions[question])
        except KeyError as exc:
            raise KeyError(f"question not found in decomposition predictions: {question}") from exc


@dataclass(slots=True)
class GoldEntityStore:
    """Question-indexed topic entities and answer labels from a processed KBQA split."""

    _entities: dict[str, dict[str, str]]
    _records: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, *, member: str = "") -> "GoldEntityStore":
        source = Path(path).expanduser()
        if source.suffix.casefold() == ".zip":
            with ZipFile(source) as archive:
                if not member:
                    raise ValueError("gold entity zip member is required")
                try:
                    payload = json.loads(archive.read(member))
                except KeyError as exc:
                    raise FileNotFoundError(f"gold entity member does not exist: {member}") from exc
        else:
            payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("gold entity data must be a JSON array")
        entities: dict[str, dict[str, str]] = {}
        records: dict[str, dict[str, Any]] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            question = _question_key(row.get("question", ""))
            if not question:
                continue
            records[question] = dict(row)
            values = row.get("golden_entities") or row.get("gold_entities") or {}
            normalized = _gold_entity_mapping(values)
            if normalized:
                entities[question] = normalized
        return cls(entities, records)

    def get(self, question: str) -> dict[str, str]:
        return dict(self._entities.get(_question_key(question), {}))

    def mapping(self) -> dict[str, dict[str, str]]:
        return {question: dict(entities) for question, entities in self._entities.items()}

    def record(self, question: str) -> dict[str, Any]:
        return dict(self._records.get(_question_key(question), {}))


_DATASET_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "semantic": (
        "semantic_path_train_test.json",
        "semantic_path_train_no_operator.json",
        "semantic_path_train.json",
    ),
    "compose": (
        "compose_train_test.json",
        "compose_train_no_operator.json",
        "compose_train.json",
    ),
    "operator": ("operator_train.json",),
    "decompose": (
        "decompose_train_test.json",
        "decompose_train.json",
        "decompose_test_pred.json",
    ),
}


def _load_array(path: str | Path, *, role: str = "") -> list[dict[str, Any]]:
    """Load a JSON-array (or JSONL) dataset, accepting a dataset directory.

    Configurations historically pointed at individual files.  The integrated
    flywheel bundle is organized as a ``train_without_operator`` directory,
    so accepting that directory here makes the data contract less brittle and
    keeps old file paths fully compatible.  A deterministic candidate order is
    used whenever more than one compatible filename is present.
    """
    source = Path(path).expanduser()
    if source.is_dir():
        names = _DATASET_FILE_CANDIDATES.get(str(role).casefold(), ())
        candidates = [source / name for name in names if (source / name).is_file()]
        if not candidates:
            expected = ", ".join(names) if names else "a JSON dataset file"
            raise FileNotFoundError(
                f"no {role or 'dataset'} file found in {source}; expected {expected}"
            )
        source = candidates[0]
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Some export tools emit one Alpaca record per line.  Fall back to
        # JSONL only when the complete document is not valid JSON.
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"expected a JSON array/object or JSONL at {source}; "
                    f"invalid line {line_number}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"expected JSONL objects at {source}; line {line_number} "
                    f"contains {type(item).__name__}"
                )
            rows.append(item)
        value = rows
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON array of objects: {source}")
    return value


def _validate_training_row(
    row: dict[str, Any],
    input_fields: set[str],
    output_fields: set[str],
    *,
    optional_input_fields: set[str] | None = None,
    optional_output_fields: set[str] | None = None,
) -> None:
    """Validate the JSON objects embedded in an Alpaca training row.

    The outer ``history``/``instruction`` metadata is intentionally ignored;
    only the serialized input/output objects form a model contract.  By
    default the check remains exact (the behavior used by the legacy
    contract).  A small, explicit optional-field set is useful for reading
    integrated rows that still carry an ``operators`` key before projection.
    """
    input_object = parse_json_object(row.get("input", ""))
    output_object = parse_json_object(row.get("output", ""))
    allowed_input = set(input_fields) | set(optional_input_fields or ())
    allowed_output = set(output_fields) | set(optional_output_fields or ())
    actual_input = set(input_object)
    actual_output = set(output_object)
    if not input_fields.issubset(actual_input) or not actual_input.issubset(allowed_input):
        if not optional_input_fields:
            raise ValueError(
                f"training input fields are {sorted(actual_input)}, "
                f"expected {sorted(input_fields)}"
            )
        raise ValueError(
            "training input fields are "
            f"{sorted(actual_input)}, expected required {sorted(input_fields)} "
            f"(optional {sorted(set(optional_input_fields or ()))})"
        )
    if not output_fields.issubset(actual_output) or not actual_output.issubset(allowed_output):
        if not optional_output_fields:
            raise ValueError(
                f"training output fields are {sorted(actual_output)}, "
                f"expected {sorted(output_fields)}"
            )
        raise ValueError(
            "training output fields are "
            f"{sorted(actual_output)}, expected required {sorted(output_fields)} "
            f"(optional {sorted(set(optional_output_fields or ()))})"
        )


_OPERATOR_FEW_SHOT_TYPES = (
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
)


def _select_operator_few_shot_examples(
    rows: list[dict[str, Any]],
    *,
    semantic_examples: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    semantic_examples = semantic_examples or {}
    parsed: list[tuple[dict[str, Any], dict[str, Any], list[str]]] = []
    for row in rows:
        try:
            input_object = parse_json_object(row.get("input", ""))
            output_object = parse_json_object(row.get("output", ""))
        except (TypeError, ValueError):
            continue
        operators = output_object.get("operators")
        if not isinstance(operators, list):
            continue
        operator_types = [
            str(operator.get("type", "")).upper()
            for operator in operators
            if isinstance(operator, dict)
        ]
        parsed.append((input_object, output_object, operator_types))

    def example_size(
        item: tuple[dict[str, Any], dict[str, Any], list[str]],
    ) -> int:
        return len(json.dumps(item[0], ensure_ascii=False)) + len(
            json.dumps(item[1], ensure_ascii=False)
        )

    selected: list[dict[str, Any]] = []
    empty_candidates = [item for item in parsed if not item[2]]
    if empty_candidates:
        input_object, output_object, _ = min(
            empty_candidates,
            key=example_size,
        )
        selected.append(
            _operator_few_shot_example(
                "NONE",
                input_object,
                output_object,
                semantic_examples,
            )
        )

    for operator_type in _OPERATOR_FEW_SHOT_TYPES:
        candidates = [item for item in parsed if operator_type in item[2]]
        if not candidates:
            continue
        single_operator = [
            item
            for item in candidates
            if len(item[2]) == 1 and item[2][0] == operator_type
        ]
        input_object, output_object, _ = min(
            single_operator or candidates,
            key=example_size,
        )
        selected.append(
            _operator_few_shot_example(
                operator_type,
                input_object,
                output_object,
                semantic_examples,
            )
        )

    marriage_candidates = [
        item for item in parsed if _is_marriage_self_exclusion_example(item)
    ]
    if marriage_candidates:
        input_object, output_object, _ = min(
            marriage_candidates,
            key=example_size,
        )
        selected.append(
            _operator_few_shot_example(
                "NO_EQUAL",
                input_object,
                output_object,
                semantic_examples,
                example_role="MARRIAGE_SELF_EXCLUSION",
            )
        )
    return selected


def _is_marriage_self_exclusion_example(
    item: tuple[dict[str, Any], dict[str, Any], list[str]],
) -> bool:
    input_object, output_object, _ = item
    triples = input_object.get("triples", [])
    relations = {
        tuple(str(part) for part in triple.get("relation_label", []))
        for triple in triples
        if isinstance(triple, dict)
    }
    if not {
        ("people", "person", "spouse s"),
        ("people", "marriage", "spouse"),
    }.issubset(relations):
        return False

    topic_entity_ids = {
        str(triple.get("subject", ""))
        for triple in triples
        if isinstance(triple, dict)
        and tuple(triple.get("relation_label", []))
        == ("people", "person", "spouse s")
    }
    topic_surfaces = {
        _question_key(entity.get("surface", ""))
        for entity in input_object.get("entities", [])
        if isinstance(entity, dict)
        and str(entity.get("id", "")) in topic_entity_ids
    }
    question = _question_key(input_object.get("question", ""))
    topic_surfaces = {
        surface for surface in topic_surfaces if surface and surface in question
    }
    if not topic_surfaces:
        return False
    answer_var = str(input_object.get("answer_var", ""))
    return any(
        isinstance(operator, dict)
        and str(operator.get("type", "")).upper() == "NO_EQUAL"
        and str(operator.get("input_var", "")) == answer_var
        and _question_key(operator.get("value", "")) in topic_surfaces
        for operator in output_object.get("operators", [])
    )


def _semantic_examples_by_question(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            semantic_input = parse_json_object(row.get("input", ""))
            semantic_output = parse_json_object(row.get("output", ""))
        except (TypeError, ValueError):
            continue
        question = str(semantic_input.get("question", "")).strip()
        if not question:
            continue
        examples[_question_key(question)] = {
            "semantic_input": semantic_input,
            "semantic_graph": semantic_output,
        }
    return examples


def _operator_few_shot_example(
    operator_type: str,
    input_object: dict[str, Any],
    output_object: dict[str, Any],
    semantic_examples: dict[str, dict[str, Any]],
    *,
    example_role: str = "",
) -> dict[str, Any]:
    question = str(input_object.get("question", "")).strip()
    semantic = semantic_examples.get(_question_key(question), {})
    return {
        "operator_type": operator_type,
        "example_role": example_role,
        "question": question,
        "semantic_input": semantic.get("semantic_input", {}),
        "semantic_graph": semantic.get("semantic_graph", {}),
        "input": input_object,
        "output": output_object,
    }


def _decomposition_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_decomposition(value: Any) -> list[str]:
    """Normalize list/string/dict decomposition representations."""
    if isinstance(value, dict):
        value = value.get("decomposition", value.get("prediction"))
    # Some prediction writers serialize the object one level deeper.
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = value
        if isinstance(parsed, dict):
            value = parsed.get("decomposition", parsed.get("prediction"))
        elif isinstance(parsed, list):
            value = parsed
    return _decomposition_list(value)


def _output_decomposition(value: Any) -> Any:
    """Extract a decomposition from an Alpaca training ``output`` field.

    The integrated flywheel files encode the response as a JSON string such
    as ``{"decomposition": [...]}``, while a few local tools write the
    object directly.  Returning the raw value (rather than normalizing here)
    keeps the existing ``_decomposition_list`` behavior for lists and strings.
    Invalid/empty outputs simply produce ``None`` so one malformed row does
    not prevent the rest of a prediction file from loading.
    """
    if value is None or value == "":
        return None
    try:
        parsed = parse_json_object(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed.get("decomposition", parsed.get("prediction"))


def _question_key(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _gold_entity_mapping(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            str(entity_id).strip(): str(label or entity_id).strip()
            for entity_id, label in value.items()
            if str(entity_id).strip()
        }
    if not isinstance(value, list):
        return {}
    result: dict[str, str] = {}
    for item in value:
        if isinstance(item, dict):
            entity_id = str(item.get("id") or item.get("kb_id") or item.get("mid") or "").strip()
            label = str(item.get("label") or item.get("name") or entity_id).strip()
        else:
            entity_id = str(item).strip()
            label = entity_id
        if entity_id:
            result[entity_id] = label
    return result
