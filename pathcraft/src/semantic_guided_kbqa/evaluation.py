from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable


EVALUATION_VERSION = "webqsp-id-aware-v2"
# Evaluate every WebQSP item from its configured Gold record; no question-level
# perfect-score exceptions are injected by the runtime.
_EMPTY_GOLD_PERFECT_QUESTIONS = frozenset()
_FREEBASE_NS_PREFIXES = (
    "http://rdf.freebase.com/ns/",
    "https://rdf.freebase.com/ns/",
)
_SLASH_MID_RE = re.compile(r"^/?([mg])/([A-Za-z0-9_]+)$")
_TYPED_LITERAL_RE = re.compile(
    r'^(?P<quoted>"(?:[^"\\]|\\.)*")(?:@[A-Za-z0-9-]+|\^\^\S+)?$'
)
_GYEAR_WITH_TIMEZONE_RE = re.compile(r"^(?P<year>[12][0-9]{3})[-+][0-9]{2}:[0-9]{2}$")
_DATE_WITH_TIMEZONE_RE = re.compile(
    r"^(?P<date>[12][0-9]{3}-[0-9]{2}-[0-9]{2})[-+][0-9]{2}:[0-9]{2}$"
)
_WEBQSP_PUNCT_RE = re.compile(r"[\s\-_.,;:!?/\\()（）【】\[\]{}<>《》、，。；：！？\"']+")


def canonicalize_answer_id(value: Any) -> str:
    """Normalize equivalent Freebase IDs and serialized literal forms."""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    for prefix in _FREEBASE_NS_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    if text.startswith("ns:"):
        return text[3:]
    slash_mid = _SLASH_MID_RE.fullmatch(text)
    if slash_mid:
        return f"{slash_mid.group(1)}.{slash_mid.group(2)}"
    typed_literal = _TYPED_LITERAL_RE.fullmatch(text)
    if typed_literal:
        try:
            text = str(json.loads(typed_literal.group("quoted"))).strip()
        except json.JSONDecodeError:
            pass
    date = _DATE_WITH_TIMEZONE_RE.fullmatch(text)
    if date:
        return date.group("date")
    gyear = _GYEAR_WITH_TIMEZONE_RE.fullmatch(text)
    if gyear:
        return gyear.group("year")
    return text


def canonicalize_answer_ids(values: Iterable[Any]) -> set[str]:
    return {
        canonical
        for value in values
        if (canonical := canonicalize_answer_id(value))
    }


def canonicalize_answer_id_list(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        canonical = canonicalize_answer_id(value)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        output.append(canonical)
    return output


def normalize_answer_label(value: Any) -> str:
    text = unicodedata.normalize("NFKC", canonicalize_answer_id(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def normalize_webqsp_text(value: Any) -> str:
    """Normalize WebQSP answer text after RDF literal canonicalization."""
    text = canonicalize_answer_id(value).strip()
    for prefix in _FREEBASE_NS_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return _WEBQSP_PUNCT_RE.sub("", text).strip().lower()


def webqsp_unique_answers(values: Iterable[Any]) -> list[str]:
    """Deduplicate answer strings with WebQSP's punctuation normalization."""
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = normalize_webqsp_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def dedupe_answer_labels(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = unicodedata.normalize("NFKC", str(value)).strip()
        normalized = normalize_answer_label(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(text)
    return output


def gold_answer_ids(record: dict[str, Any]) -> set[str]:
    answers = record.get("answer_entities", {})
    if isinstance(answers, dict):
        values = (
            value if str(key).casefold() == "value" else key
            for key, value in answers.items()
        )
        return canonicalize_answer_ids(values)
    if isinstance(answers, list):
        return canonicalize_answer_ids(answers)
    return set()


def gold_answer_labels(record: dict[str, Any]) -> list[str]:
    answers = record.get("answer_entities", {})
    if isinstance(answers, dict):
        return webqsp_unique_answers(answers.values())
    return []


def score_answer_ids(gold: Iterable[Any], predicted: Iterable[Any]) -> dict[str, Any]:
    gold_set = canonicalize_answer_ids(gold)
    predicted_set = canonicalize_answer_ids(predicted)
    if not gold_set:
        return {"exact": None, "precision": None, "recall": None, "f1": None}
    if not predicted_set:
        return {"exact": False, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = gold_set & predicted_set
    precision = len(overlap) / len(predicted_set)
    recall = len(overlap) / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "exact": predicted_set == gold_set,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_webqsp_answers(
    gold_answers: Iterable[Any],
    predicted_answers: Iterable[Any],
) -> dict[str, Any]:
    """Compute the same text-set metrics as ``webqsp_mas.metrics``."""
    gold_values = webqsp_unique_answers(gold_answers)
    predicted_values = webqsp_unique_answers(predicted_answers)
    gold_keys = {normalize_webqsp_text(value) for value in gold_values}
    predicted_keys = [normalize_webqsp_text(value) for value in predicted_values]
    predicted_key_set = set(predicted_keys)
    if not gold_keys:
        return {
            "labeled": False,
            "precision": None,
            "recall": None,
            "f1": None,
            "hits_at_1": None,
            "exact_match": None,
            "exact": None,
        }
    true_positive = len(gold_keys & predicted_key_set)
    precision = true_positive / len(predicted_key_set) if predicted_key_set else 0.0
    recall = true_positive / len(gold_keys)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact_match = 1.0 if predicted_key_set == gold_keys else 0.0
    first = predicted_keys[0] if predicted_keys else ""
    return {
        "labeled": True,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hits_at_1": 1.0 if first and first in gold_keys else 0.0,
        "exact_match": exact_match,
        # Preserve the historical field as a boolean compatibility alias.
        "exact": bool(exact_match),
    }


def score_normalized_labels(gold: Iterable[Any], predicted: Iterable[Any]) -> dict[str, Any]:
    gold_set = {
        normalized
        for value in gold
        if (normalized := normalize_answer_label(value))
    }
    predicted_set = {
        normalized
        for value in predicted
        if (normalized := normalize_answer_label(value))
    }
    if not gold_set:
        return {"precision": None, "recall": None, "f1": None, "hit": None}
    overlap = gold_set & predicted_set
    precision = len(overlap) / len(predicted_set) if predicted_set else 0.0
    recall = len(overlap) / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": 1 if overlap else 0,
    }


def score_kaede_labels(gold: Iterable[Any], predicted: Iterable[Any]) -> dict[str, Any]:
    """Retain KaeDe substring matching while removing duplicate labels."""
    gold_values = dedupe_answer_labels(gold)
    predicted_values = dedupe_answer_labels(predicted)
    if not gold_values:
        return {"precision": None, "recall": None, "f1": None, "hit": None}
    true_positive = sum(_find_substring_match(item, predicted_values) for item in gold_values)
    false_negative = len(gold_values) - true_positive
    false_positive = sum(
        not _find_substring_match(item, gold_values)
        for item in predicted_values
    )
    precision = (
        0.0
        if true_positive + false_positive == 0
        else true_positive / (true_positive + false_positive)
    )
    recall = (
        0.0
        if true_positive + false_negative == 0
        else true_positive / (true_positive + false_negative)
    )
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": 1 if true_positive > 0 else 0,
    }


def predicted_answer_labels(row: dict[str, Any]) -> list[str]:
    labels_by_id = _predicted_answer_label_map(row)
    return dedupe_answer_labels(
        labels_by_id.get(canonicalize_answer_id(answer_id), answer_id)
        for answer_id in row.get("prediction", []) or []
    )


def score_result_row(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    scored = dict(row)
    for key in ("evaluation_prediction", "normalized_label_score", "kaede_label_score"):
        scored.pop(key, None)
    gold_ids = sorted(gold_answer_ids(record))
    raw_gold_labels = gold_answer_labels(record)
    gold_labels = webqsp_unique_answers(raw_gold_labels)
    compatibility_gold_labels = dedupe_answer_labels(raw_gold_labels)
    predicted_ids = row.get("prediction", row.get("answer_ids", [])) or []
    scored["prediction"] = canonicalize_answer_id_list(predicted_ids)
    scored["answer_ids"] = list(scored["prediction"])
    raw_predicted_labels = predicted_answer_labels(scored)
    predicted_labels = webqsp_unique_answers(raw_predicted_labels)
    compatibility_predicted_labels = dedupe_answer_labels(raw_predicted_labels)
    scored["gold_answers"] = gold_ids
    scored["gold_answer_labels"] = gold_labels
    scored["predicted_answers"] = predicted_labels
    text_score = score_webqsp_answers(gold_labels, predicted_labels)
    scored["raw_text_score"] = text_score
    # Entity identity is independent of label availability, aliases and preview
    # limits. Keep the original text protocol for literal/text predictions.
    entity_ids = bool(gold_ids) and all(_is_freebase_id(value) for value in gold_ids)
    entity_predictions = all(_is_freebase_id(value) for value in scored["prediction"])
    if entity_ids and entity_predictions:
        id_score = score_answer_ids(gold_ids, scored["prediction"])
        scored["score"] = {
            "labeled": True,
            **id_score,
            "hits_at_1": float(bool(scored["prediction"]) and scored["prediction"][0] in gold_ids),
            "exact_match": float(id_score["exact"]),
        }
        scored["score_basis"] = "answer_ids"
    else:
        scored["score"] = text_score
        scored["score_basis"] = "answer_text"
    # Keep the two historical label metrics as additional compatibility
    # indicators. They use the pre-existing normalization and KaeDe substring
    # rules. These diagnostics can remain imperfect when entity IDs are correct.
    scored["normalized_label_score"] = score_normalized_labels(
        compatibility_gold_labels,
        compatibility_predicted_labels,
    )
    scored["kaede_label_score"] = score_kaede_labels(
        compatibility_gold_labels,
        compatibility_predicted_labels,
    )
    return scored


def summarize_result_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        row["score"]
        for row in rows
        if is_metric_result_row(row)
    ]
    if not scored:
        return {"count": 0, "scored": 0}
    precision = sum(float(item["precision"] or 0.0) for item in scored) / len(scored)
    recall = sum(float(item["recall"] or 0.0) for item in scored) / len(scored)
    f1 = sum(float(item["f1"] or 0.0) for item in scored) / len(scored)
    exact_match = sum(float(item["exact_match"] or 0.0) for item in scored) / len(scored)
    hits_at_1 = sum(float(item["hits_at_1"] or 0.0) for item in scored) / len(scored)
    summary: dict[str, Any] = {
        "total": len(rows),
        "labeled": len(scored),
        "unlabeled": len(rows) - len(scored),
        "count": len(scored),
        "scored": len(scored),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "hits_at_1": hits_at_1,
        "exact_match": exact_match,
        # Keep the previous summary names as numerical aliases.
        "exact": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    for row_key, summary_key in (
        ("normalized_label_score", "normalized_label"),
        ("kaede_label_score", "kaede_label"),
    ):
        compatibility = [
            row[row_key]
            for row in rows
            if is_metric_result_row(row)
            and isinstance(row.get(row_key), dict)
            and row[row_key].get("f1") is not None
        ]
        if compatibility:
            summary[summary_key] = {
                "scored": len(compatibility),
                "precision": sum(float(item["precision"] or 0.0) for item in compatibility)
                / len(compatibility),
                "recall": sum(float(item["recall"] or 0.0) for item in compatibility)
                / len(compatibility),
                "f1": sum(float(item["f1"] or 0.0) for item in compatibility)
                / len(compatibility),
                "hit": sum(float(item["hit"] or 0.0) for item in compatibility)
                / len(compatibility),
            }
    return summary


def is_error_result_row(row: dict[str, Any]) -> bool:
    """Return whether pipeline execution failed for a result row."""
    return bool(row.get("failure"))


def is_metric_result_row(row: dict[str, Any]) -> bool:
    """Return whether a row has a computable WebQSP-compatible score.

    Pipeline failures are stored separately and are excluded from the metric
    denominator, matching this project's error-output policy.
    """
    score = row.get("score", {})
    return (
        not is_error_result_row(row)
        and isinstance(score, dict)
        and score.get("labeled") is True
        and score.get("f1") is not None
    )


def row_metrics_are_all_one(row: dict[str, Any]) -> bool:
    score = row.get("score")
    return isinstance(score, dict) and _metric_value_is_one(score.get("f1")) and _metric_value_is_one(
        score.get("recall")
    )


def row_id_f1_recall_are_all_one(row: dict[str, Any]) -> bool:
    """Return whether primary F1 and recall are both perfect."""
    score = row.get("score")
    return (
        isinstance(score, dict)
        and _metric_value_is_one(score.get("f1"))
        and _metric_value_is_one(score.get("recall"))
    )


def evaluation_metadata(source: str) -> dict[str, Any]:
    return {
        "version": EVALUATION_VERSION,
        "source": source,
        "matching": "canonical ID sets when Gold and predictions are Freebase entities; otherwise WebQSP-compatible normalized answer text sets",
        "normalization": "strip Freebase namespace, remove WebQSP punctuation/separators, lowercase, deduplicate",
        "gold_source": "answer_entities IDs/labels from the configured Gold record",
        "protocol_change": "entity identity takes precedence over labels; this primary metric differs from the historical text-only protocol",
        "aggregation": "macro precision/recall/F1, hits_at_1 and exact_match over successful labeled rows; failure rows are stored separately",
        "additional_metrics": {
            "raw_text_score": "unchanged historical WebQSP text-set score, including label lookup limitations",
            "normalized_label": "historical NFKC/casefold exact-label metric",
            "kaede_label": "historical KaeDe substring-label metric",
        },
    }


def _bridge_predictions_to_gold_ids(
    record: dict[str, Any],
    row: dict[str, Any],
) -> list[str]:
    """Bridge literal answers to gold MIDs only through unique exact labels."""
    answers = record.get("answer_entities", {})
    predicted_ids = row.get("prediction", []) or []
    if not isinstance(answers, dict):
        return list(predicted_ids)

    label_to_ids: dict[str, set[str]] = {}
    for raw_id, raw_label in answers.items():
        if str(raw_id).casefold() == "value":
            continue
        gold_id = canonicalize_answer_id(raw_id)
        label = normalize_answer_label(raw_label)
        if gold_id and label:
            label_to_ids.setdefault(label, set()).add(gold_id)

    bridged: list[str] = []
    seen: set[str] = set()
    gold_ids = gold_answer_ids(record)
    labels_by_id = _predicted_answer_label_map(row)
    for predicted_id in predicted_ids:
        label = labels_by_id.get(predicted_id, predicted_id)
        matching_ids = label_to_ids.get(normalize_answer_label(label), set())
        evaluation_id = (
            next(iter(matching_ids))
            if not _is_freebase_id(predicted_id)
            and predicted_id not in gold_ids
            and len(matching_ids) == 1
            else predicted_id
        )
        if evaluation_id and evaluation_id not in seen:
            seen.add(evaluation_id)
            bridged.append(evaluation_id)
    return bridged


def _predicted_answer_label_map(row: dict[str, Any]) -> dict[str, str]:
    return {
        canonicalize_answer_id(item.get("id", "")): str(
            item.get("label", item.get("id", ""))
        )
        for item in row.get("answers", []) or []
        if isinstance(item, dict) and canonicalize_answer_id(item.get("id", ""))
    }


def _is_freebase_id(value: str) -> bool:
    return re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", value) is not None


def _metric_value_is_one(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return abs(float(value) - 1.0) <= 1e-12
    return False


def _metric_block_is_all_one(block: Any) -> bool:
    if not isinstance(block, dict) or not block:
        return False
    return all(_metric_value_is_one(value) for value in block.values())


def _find_substring_match(entry: str, values: list[str]) -> bool:
    entry_norm = normalize_answer_label(entry)
    if not entry_norm:
        return False
    return any(entry_norm in normalize_answer_label(value) for value in values)
