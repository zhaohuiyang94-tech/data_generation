"""Deterministic lexical retrieval of semantic path templates.

The index is built exclusively from ``TrainingContract.semantic_examples``.
It intentionally has no access to evaluation answers, gold query graphs, an
endpoint, or a language model.  Entity surfaces are masked before indexing so
that retrieval is driven by relational intent rather than memorized entities.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Sequence


_INDEX_VERSION = 1
_TOKEN_RE = re.compile(r"<entity>|[a-z0-9]+", re.IGNORECASE)


def _surface(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("surface") or value.get("label") or value.get("name") or "").strip()
    return str(getattr(value, "label", value) or "").strip()


def _ordered_anchors(value: Any, text: str = "") -> list[tuple[str, str]]:
    """Normalize supported anchor containers and order anchors by mention."""
    anchors: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict) and item.get("id"):
                anchor_id = str(item["id"])
            else:
                anchor_id = str(key)
            anchors.append((anchor_id, _surface(item)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            if isinstance(item, dict):
                anchor_id = str(item.get("id") or f"A{index}")
            else:
                anchor_id = f"A{index}"
            anchors.append((anchor_id, _surface(item)))
    normalized_text = text.casefold()
    original_order = {anchor_id: index for index, (anchor_id, _) in enumerate(anchors)}
    return sorted(
        anchors,
        key=lambda item: (
            normalized_text.find(item[1].casefold())
            if item[1] and item[1].casefold() in normalized_text
            else len(normalized_text) + original_order[item[0]],
            original_order[item[0]],
        ),
    )


def mask_entities(text: str, surfaces: Iterable[str]) -> str:
    """Replace entity mentions, longest first, without a name whitelist."""
    result = " ".join(str(text).split())
    values = sorted(
        {" ".join(str(value).split()) for value in surfaces if str(value).strip()},
        key=lambda value: (-len(value), value.casefold()),
    )
    for value in values:
        result = re.sub(
            rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])",
            "<entity>",
            result,
            flags=re.IGNORECASE,
        )
    return " ".join(result.split())


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def _relation_text(graph: dict[str, Any]) -> str:
    parts: list[str] = []
    for path in graph.get("semantic_paths", []):
        if not isinstance(path, dict):
            continue
        for step in path.get("steps", []):
            if not isinstance(step, dict):
                continue
            label = step.get("relation_label", [])
            if isinstance(label, str):
                parts.append(label.replace(".", " ").replace("_", " "))
            elif isinstance(label, list):
                parts.extend(str(item).replace("_", " ") for item in label)
    return " ".join(parts)


def _extract_paths(graph: dict[str, Any], anchor_slots: dict[str, int]) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    for path in graph.get("semantic_paths", []):
        if not isinstance(path, dict):
            continue
        anchor_ref = str(path.get("anchor_ref", ""))
        if anchor_ref not in anchor_slots:
            continue
        steps: list[dict[str, Any]] = []
        for step in path.get("steps", []):
            if not isinstance(step, dict):
                continue
            raw_label = step.get("relation_label", [])
            if isinstance(raw_label, str):
                label = [part for part in raw_label.split(".") if part]
            elif isinstance(raw_label, list):
                label = [str(part).strip() for part in raw_label if str(part).strip()]
            else:
                label = []
            direction = str(step.get("direction", "")).casefold()
            if not label or direction not in {"forward", "backward"}:
                steps = []
                break
            steps.append({"relation_label": label, "direction": direction})
        if steps:
            paths.append(
                {
                    "anchor_slot": anchor_slots[anchor_ref],
                    "steps": steps,
                    "hop_count": len(steps),
                }
            )
    return paths


@dataclass(slots=True)
class _Document:
    example_index: int
    question: str
    decomposition: list[str]
    anchor_count: int
    paths: list[dict[str, Any]]
    tokens: list[str]

    def dump(self) -> dict[str, Any]:
        return {
            "example_index": self.example_index,
            "question": self.question,
            "decomposition": self.decomposition,
            "anchor_count": self.anchor_count,
            "paths": self.paths,
            "tokens": self.tokens,
        }


class TemplatePathRetriever:
    """A cached BM25/IDF index over accepted training semantic graphs."""

    def __init__(
        self,
        semantic_examples: list[dict[str, Any]],
        *,
        cache_path: str | Path | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.cache_path = Path(cache_path) if cache_path else None
        self.fingerprint = self._fingerprint(semantic_examples)
        cached = self._load_cache()
        if cached is None:
            self.documents = self._build_documents(semantic_examples)
            self._write_cache()
            self.cache_hit = False
        else:
            try:
                self.documents = [_Document(**item) for item in cached]
            except (TypeError, ValueError):
                self.documents = self._build_documents(semantic_examples)
                self._write_cache()
                self.cache_hit = False
            else:
                self.cache_hit = True
        self._term_frequencies = [Counter(document.tokens) for document in self.documents]
        self._average_length = (
            sum(len(document.tokens) for document in self.documents) / len(self.documents)
            if self.documents
            else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(set(document.tokens))
        count = len(self.documents)
        self._idf = {
            token: math.log(1.0 + ((count - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in document_frequency.items()
        }

    @classmethod
    def from_contract(
        cls,
        contract: Any,
        *,
        cache_path: str | Path | None = None,
    ) -> "TemplatePathRetriever":
        return cls(list(contract.semantic_examples), cache_path=cache_path)

    @staticmethod
    def _fingerprint(examples: list[dict[str, Any]]) -> str:
        payload = json.dumps(examples, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_cache(self) -> list[dict[str, Any]] | None:
        if not self.cache_path or not self.cache_path.is_file():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _INDEX_VERSION
            or payload.get("fingerprint") != self.fingerprint
            or not isinstance(payload.get("documents"), list)
        ):
            return None
        return payload["documents"]

    def _write_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_VERSION,
            "fingerprint": self.fingerprint,
            "documents": [document.dump() for document in self.documents],
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.cache_path.parent,
            prefix=self.cache_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            temporary = Path(handle.name)
        temporary.replace(self.cache_path)

    @staticmethod
    def _build_documents(examples: list[dict[str, Any]]) -> list[_Document]:
        documents: list[_Document] = []
        for example_index, example in enumerate(examples):
            graph = example.get("semantic_graph", {})
            if not isinstance(graph, dict):
                continue
            anchors = [item for item in graph.get("anchors", []) if isinstance(item, dict)]
            anchor_text = " ".join(
                [str(example.get("question", ""))]
                + [str(item) for item in example.get("decomposition", [])]
            )
            ordered = _ordered_anchors(anchors, anchor_text)
            slots = {anchor_id: slot for slot, (anchor_id, _) in enumerate(ordered)}
            paths = _extract_paths(graph, slots)
            if not paths:
                continue
            surfaces = [surface for _, surface in ordered]
            question = mask_entities(str(example.get("question", "")), surfaces)
            decomposition = [
                mask_entities(str(item), surfaces)
                for item in example.get("decomposition", [])
                if str(item).strip()
            ]
            # Relation-label words make paraphrased questions retrievable while
            # remaining entirely within the accepted training contract.
            document_text = " ".join([question, *decomposition, _relation_text(graph)])
            documents.append(
                _Document(
                    example_index=example_index,
                    question=question,
                    decomposition=decomposition,
                    anchor_count=len(ordered),
                    paths=paths,
                    tokens=_tokens(document_text),
                )
            )
        return documents

    def _score(self, query_tokens: list[str], index: int) -> float:
        if not query_tokens or not self.documents:
            return 0.0
        frequencies = self._term_frequencies[index]
        length = len(self.documents[index].tokens)
        normalization = 1.0 - self.b
        if self._average_length:
            normalization += self.b * length / self._average_length
        score = 0.0
        for token, query_frequency in Counter(query_tokens).items():
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            numerator = frequency * (self.k1 + 1.0)
            denominator = frequency + (self.k1 * normalization)
            score += self._idf.get(token, 0.0) * (numerator / denominator) * min(query_frequency, 2)
        return score

    def retrieve(
        self,
        *,
        question: str,
        decomposition: Sequence[str] | None,
        current_anchors: Any,
        top_k: int = 12,
        preselect: int = 128,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return ranked templates instantiated with the current anchor IDs."""
        if isinstance(decomposition, str):
            decomposition = [decomposition]
        decomposition = [str(item) for item in (decomposition or ()) if str(item).strip()]
        current = _ordered_anchors(current_anchors, " ".join([question, *decomposition]))
        diagnostics: dict[str, Any] = {
            "strategy": "masked_training_bm25",
            "index_size": len(self.documents),
            "index_fingerprint": self.fingerprint,
            "cache_hit": self.cache_hit,
            "current_anchor_count": len(current),
            "query_tokens": [],
            "considered": 0,
            "rejected_anchor_count": 0,
            "top_k": [],
        }
        if not current or not self.documents or top_k <= 0:
            return [], diagnostics
        surfaces = [surface for _, surface in current]
        masked_question = mask_entities(question, surfaces)
        masked_decomposition = [mask_entities(item, surfaces) for item in decomposition]
        query_tokens = _tokens(" ".join([masked_question, *masked_decomposition]))
        diagnostics["query_tokens"] = query_tokens

        ranked: list[tuple[float, int, _Document]] = []
        for index, document in enumerate(self.documents):
            if document.anchor_count > len(current):
                diagnostics["rejected_anchor_count"] += 1
                continue
            bm25 = self._score(query_tokens, index)
            # Prefer structurally exact anchor counts only as a deterministic
            # tie/near-tie signal; lexical intent remains the main ranker.
            compatibility = 0.2 if document.anchor_count == len(current) else 0.0
            ranked.append((bm25 + compatibility, index, document))
        ranked.sort(key=lambda item: (-item[0], item[2].example_index))
        ranked = ranked[: max(int(top_k), int(preselect), 1)]
        diagnostics["considered"] = len(ranked)

        matches: list[dict[str, Any]] = []
        seen: set[str] = set()
        for score, index, document in ranked:
            anchor_mapping = {
                slot: current[slot][0]
                for slot in range(document.anchor_count)
            }
            paths = [
                {
                    "anchor_ref": anchor_mapping[path["anchor_slot"]],
                    "anchor_slot": path["anchor_slot"],
                    "steps": path["steps"],
                    "hop_count": path["hop_count"],
                }
                for path in document.paths
            ]
            signature = json.dumps(paths, sort_keys=True, separators=(",", ":"))
            if signature in seen:
                continue
            seen.add(signature)
            match = {
                "rank": len(matches) + 1,
                "score": score,
                "bm25_score": self._score(query_tokens, index),
                "source_example_index": document.example_index,
                "source_question_masked": document.question,
                "anchor_mapping": anchor_mapping,
                "paths": paths,
            }
            matches.append(match)
            diagnostics["top_k"].append(
                {
                    "rank": match["rank"],
                    "score": score,
                    "bm25_score": match["bm25_score"],
                    "source_example_index": document.example_index,
                    "anchor_count": document.anchor_count,
                    "path_count": len(paths),
                    "hop_counts": [path["hop_count"] for path in paths],
                    "paths": paths,
                }
            )
            if len(matches) >= top_k:
                break
        return matches, diagnostics


__all__ = ["TemplatePathRetriever", "mask_entities"]
