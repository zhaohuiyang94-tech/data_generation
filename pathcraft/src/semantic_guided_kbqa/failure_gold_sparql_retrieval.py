"""Bounded final-empty retrieval over cached CWQ TRAIN Gold SPARQL.

This module is deliberately isolated from evaluation data.  Runtime inputs are
the current question, reviewed decomposition trace, configured entity links,
and a cache built solely from CWQ TRAIN question/Gold-SPARQL pairs.  It performs
no generative-model call.  The ordinary lane executes at most three final
SELECT queries; only when all three are empty may the optional local-BGE lane
execute at most three different queries (six total for that failure only).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Sequence

from .pipeline import _answer_values
from .template_path_retrieval import mask_entities


VERSION = "failure-source-train-gold-sparql-production-v4-adaptive-role-quota"
CACHE_VERSION = 3
TOKEN_RE = re.compile(r"<entity>|<date>|<year>|<number>|[a-z0-9]+", re.I)
FULL_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
YEAR_RE = re.compile(r"(?<![\d.])(?:1[0-9]{3}|20[0-9]{2})(?![\d.])")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")
SELECT_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", re.I)
ENTITY_ID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
RELATION_RE = re.compile(r"\bns:([A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)\b")
PLURAL_ANSWER_RE = re.compile(
    r"\b(?:(?:which|what|name|list)\s+(?:are|were)|names|people|actors|"
    r"characters|countries|colleges|movies|films|who\s+(?:are|were))\b",
    re.I,
)
SINGULAR_ANSWER_RE = re.compile(
    r"\b(?:who\s+(?:is|was)|what\s+is\s+the\s+name|"
    r"which\s+(?:person|country|college|movie|film|actor|president|"
    r"prime\s+minister|character)|where\s+did|from\s+which)\b",
    re.I,
)


def _normalized(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(value)]


def _scalar_slots(text: str) -> list[tuple[str, str]]:
    occupied: list[tuple[int, int]] = []
    output: list[tuple[int, str, str]] = []
    for match in FULL_DATE_RE.finditer(text):
        occupied.append(match.span())
        output.append((match.start(), "date", match.group(0)))
    for match in YEAR_RE.finditer(text):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        occupied.append(match.span())
        output.append((match.start(), "year", match.group(0)))
    for match in NUMBER_RE.finditer(text):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        output.append((match.start(), "number", match.group(0)))
    return [(kind, value) for _, kind, value in sorted(output)]


def _normalize_scalars(text: str) -> str:
    value = FULL_DATE_RE.sub(" <date> ", str(text))
    value = YEAR_RE.sub(" <year> ", value)
    value = NUMBER_RE.sub(" <number> ", value)
    return " ".join(value.split())


def _masked_text(
    question: str,
    decomposition: Sequence[str],
    surfaces: Sequence[str],
) -> str:
    masked_question = mask_entities(_normalize_scalars(question), surfaces)
    return " ".join(
        [
            masked_question,
            masked_question,
            *(
                mask_entities(_normalize_scalars(value), surfaces)
                for value in decomposition
                if str(value).strip()
            ),
        ]
    )


def _operator_signature(question: str) -> tuple[str, ...]:
    text = _normalized(question)
    result: list[str] = []
    if re.search(r"\b(?:largest|highest|latest|last|most recent|maximum|greatest)\b", text):
        result.append("argmax")
    if re.search(r"\b(?:smallest|lowest|earliest|first|minimum|least)\b", text):
        result.append("argmin")
    if re.search(r"\b(?:before|earlier than|less than|under)\b", text):
        result.append("lt")
    if re.search(r"\b(?:after|later than|greater than|over|more than)\b", text):
        result.append("gt")
    if re.search(r"\b(?:how many|number of)\b", text):
        result.append("count")
    if (
        _scalar_slots(text)
        and not set(result) & {"argmax", "argmin", "count", "gt", "lt"}
        and re.search(
            r"\b(?:at|during|equal(?:s|\s+to)?|exactly|has|had|having|in|on|with)\b",
            text,
        )
    ):
        result.append("eq")
    return tuple(sorted(set(result)))


def _effective_operator_signature(document: "_Document") -> tuple[str, ...]:
    """Recover scalar equality intent for caches built before ``eq`` existed."""

    return tuple(
        sorted(
            set(map(str, document.operator_signature))
            | set(_operator_signature(document.question))
        )
    )


@dataclass(frozen=True, slots=True)
class _Document:
    source_index: int
    question: str
    decomposition: tuple[str, ...]
    source_surfaces: tuple[str, ...]
    constants: tuple[str, ...]
    constant_roles: tuple[str, ...]
    sparql: str
    answer_var: str
    signature: str
    operator_signature: tuple[str, ...]
    tokens: tuple[str, ...]


def _operator_conflict(source: set[str], target: set[str]) -> bool:
    source_comparison = source & {"eq", "gt", "lt"}
    target_comparison = target & {"eq", "gt", "lt"}
    if (
        source_comparison
        and target_comparison
        and source_comparison != target_comparison
    ):
        return True
    for pair in ({"lt", "gt"}, {"argmin", "argmax"}):
        source_side, target_side = source & pair, target & pair
        if source_side and target_side and source_side != target_side:
            return True
    return False


def _relation_text(sparql: str) -> str:
    relations = _unique(
        relation
        for relation in RELATION_RE.findall(sparql)
        if not relation.startswith(("m.", "g."))
    )
    return " ; ".join(
        " ".join(relation.replace("_", " ").split("."))
        for relation in relations
    )


def _semantic_document(document: "_Document") -> str:
    """Gold-blind BGE document with entity symbols removed.

    The source question and pseudo decomposition provide paraphrase evidence;
    real relation/constant-role names preserve graph distinctions that vanish
    if every template is represented only by V0/V1-like placeholders.
    """

    return " ".join(
        (
            "passage:",
            _masked_text(
                document.question,
                document.decomposition,
                document.source_surfaces,
            ),
            "relation path:",
            _relation_text(document.sparql),
            "constant roles:",
            " ; ".join(document.constant_roles),
        )
    )


def _cosine_vectors(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(
        sum(b * b for b in right)
    )
    return numerator / denominator if denominator else 0.0


class SourceGoldSparqlIndex:
    """Read-only cached index with sparse BM25 postings."""

    def __init__(self, cache_path: str | Path) -> None:
        started = time.perf_counter()
        source = Path(cache_path).expanduser()
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != CACHE_VERSION
            or not isinstance(payload.get("documents"), list)
        ):
            raise ValueError(f"incompatible source Gold SPARQL cache: {source}")
        self.documents = [
            _Document(
                source_index=int(item["source_index"]),
                question=str(item["question"]),
                decomposition=tuple(map(str, item.get("decomposition", []))),
                source_surfaces=tuple(map(str, item.get("source_surfaces", []))),
                constants=tuple(map(str, item.get("constants", []))),
                constant_roles=tuple(map(str, item.get("constant_roles", []))),
                sparql=str(item["sparql"]),
                answer_var=str(item.get("answer_var", "x")),
                signature=str(item.get("signature", "")),
                operator_signature=tuple(map(str, item.get("operator_signature", []))),
                tokens=tuple(map(str, item.get("tokens", []))),
            )
            for item in payload["documents"]
            if isinstance(item, dict)
        ]
        self.frequencies: list[Counter[str]] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for index, document in enumerate(self.documents):
            frequency = Counter(document.tokens)
            self.frequencies.append(frequency)
            total_length += len(document.tokens)
            document_frequency.update(set(frequency))
            for token, count in frequency.items():
                self.postings[token].append((index, count))
        count = max(1, len(self.documents))
        self.average_length = total_length / count
        self.idf = {
            token: math.log(1.0 + ((count - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in document_frequency.items()
        }
        self.load_seconds = time.perf_counter() - started

    def retrieve(
        self,
        *,
        question: str,
        decomposition: Sequence[str],
        surfaces: Sequence[str],
        entity_count: int,
        top_k: int = 32,
    ) -> list[dict[str, Any]]:
        query = Counter(_tokens(_masked_text(question, decomposition, surfaces)))
        scores: dict[int, float] = defaultdict(float)
        for token, query_frequency in query.items():
            inverse = self.idf.get(token, 0.0)
            for index, frequency in self.postings.get(token, ()):
                document = self.documents[index]
                if len(document.constants) != entity_count:
                    continue
                normalization = 0.25 + 0.75 * len(document.tokens) / self.average_length
                scores[index] += inverse * (
                    frequency * 2.2 / (frequency + 1.2 * normalization)
                ) * min(query_frequency, 2)
        target_operator = set(_operator_signature(question))
        ranked: list[tuple[float, int]] = []
        for index, score in scores.items():
            document = self.documents[index]
            source_operator = set(_effective_operator_signature(document))
            if _operator_conflict(source_operator, target_operator):
                continue
            if target_operator and source_operator == target_operator:
                score += 1.0
            elif target_operator and source_operator and source_operator != target_operator:
                score -= 2.0
            ranked.append((score, index))
        ranked.sort(key=lambda item: (-item[0], self.documents[item[1]].source_index))
        support = Counter(
            self.documents[index].signature for _, index in ranked[:64]
        )
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank, (score, index) in enumerate(ranked[:128], start=1):
            document = self.documents[index]
            if document.signature in seen:
                continue
            seen.add(document.signature)
            output.append(
                {
                    "rank": rank,
                    "bm25": score,
                    "support": support[document.signature],
                    "document": document,
                }
            )
            if len(output) >= max(3, min(64, int(top_k))):
                break
        return output


def _token_similarity(left: str, right: str) -> float:
    one = set(_tokens(left)) - {"entity", "date", "year", "number"}
    two = set(_tokens(right)) - {"entity", "date", "year", "number"}
    return len(one & two) / len(one | two) if one and two else 0.0


def _local_context(label: str, text: str, radius: int = 14) -> str:
    words = list(re.finditer(r"[a-z0-9]+", _normalized(text)))
    label_tokens = _tokens(label)
    if not words or not label_tokens:
        return text
    normalized = _normalized(text)
    position = normalized.find(_normalized(label))
    if position < 0:
        return " ".join(label_tokens)
    center = next((i for i, word in enumerate(words) if word.start() >= position), 0)
    return " ".join(
        word.group(0)
        for word in words[
            max(0, center - radius) : min(
                len(words), center + len(label_tokens) + radius
            )
        ]
    )


def _semantic_context(
    label: str,
    question: str,
    decomposition: Sequence[str],
) -> str:
    needle = _normalized(label)
    clauses = [
        str(value)
        for value in decomposition
        if needle and needle in _normalized(value)
    ]
    return " ".join(clauses) if clauses else _local_context(label, question)


def _mapping_candidates(
    document: _Document,
    entities: Sequence[tuple[str, str]],
    question: str,
    decomposition: Sequence[str],
) -> list[tuple[float, tuple[tuple[str, str], ...]]]:
    surfaces = list(document.source_surfaces[: len(document.constants)])
    surfaces.extend([""] * (len(document.constants) - len(surfaces)))
    target = list(entities)
    output: list[tuple[float, tuple[tuple[str, str], ...]]] = []
    for assigned in permutations(target, len(document.constants)):
        score = 0.0
        for position, ((_, label), source_surface) in enumerate(zip(assigned, surfaces)):
            if assigned[position] == target[position]:
                score += 0.10
            score += 0.20 * _token_similarity(source_surface, label)
            role = document.constant_roles[position] if position < len(document.constant_roles) else ""
            score += 2.0 * _token_similarity(
                role, _semantic_context(label, question, decomposition)
            )
        output.append((score, tuple(assigned)))
    output.sort(key=lambda item: (-item[0], tuple(value[0] for value in item[1])))
    return output


def _replace_scalars(
    sparql: str,
    source_question: str,
    target_question: str,
) -> tuple[str, bool]:
    source, target = _scalar_slots(source_question), _scalar_slots(target_question)
    if Counter(kind for kind, _ in source) != Counter(kind for kind, _ in target):
        return sparql, not source
    source_by_kind: dict[str, list[str]] = defaultdict(list)
    target_by_kind: dict[str, list[str]] = defaultdict(list)
    for kind, value in source:
        source_by_kind[kind].append(value)
    for kind, value in target:
        target_by_kind[kind].append(value)
    result = sparql
    for kind, source_values in source_by_kind.items():
        for old, new in zip(source_values, target_by_kind[kind]):
            if kind == "year":
                result = result.replace(f'"{old}-01-01"', f'"{new}-01-01"')
                result = result.replace(f'"{old}-12-31"', f'"{new}-12-31"')
            result = re.sub(rf"(?<![\d.]){re.escape(old)}(?![\d.])", new, result)
    return result, True


def _rewrite_numeric_equality_literal(
    query: str,
    *,
    question: str,
    source_signature: Sequence[str],
) -> tuple[str, int]:
    """Lower one quoted numeric equality like the main SPARQL compiler."""

    target_signature = set(_operator_signature(question))
    if target_signature != {"eq"} or set(source_signature) != {"eq"}:
        return query, 0
    numeric_slots = [
        value for kind, value in _scalar_slots(question) if kind == "number"
    ]
    if len(numeric_slots) != 1:
        return query, 0
    raw_value = numeric_slots[0]
    try:
        numeric_value = float(raw_value)
    except ValueError:
        return query, 0
    variable = "?numeric_equal_0"
    if variable in query:
        return query, 0
    literal_pattern = re.compile(
        rf'(?P<prefix>(?:\?[A-Za-z_][A-Za-z0-9_]*|ns:[A-Za-z0-9_.]+)\s+'
        rf'ns:[A-Za-z][A-Za-z0-9_.]+\s+)"{re.escape(raw_value)}"'
        rf'(?:\^\^[^\s.]+)?\s*\.',
        re.I,
    )
    tolerance = max(0.000001, abs(numeric_value) * 0.000001)
    cast = "bif:atoi" if re.fullmatch(r"[-+]?\d+", raw_value) else "bif:atof"
    replacement = (
        rf"\g<prefix>{variable} .\n"
        f"FILTER (ABS({cast}(STR({variable})) - {numeric_value:.15g}) "
        f"< {tolerance:.15g}) ."
    )
    rewritten, count = literal_pattern.subn(replacement, query)
    return (rewritten, count) if count == 1 else (query, 0)


def _instantiate(
    match: dict[str, Any],
    entities: Sequence[tuple[str, str]],
    question: str,
    decomposition: Sequence[str],
) -> list[dict[str, Any]]:
    document: _Document = match["document"]
    source_signature = _effective_operator_signature(document)
    target_signature = _operator_signature(question)
    if _operator_conflict(set(source_signature), set(target_signature)):
        return []
    output: list[dict[str, Any]] = []
    for mapping_rank, (mapping_score, assigned) in enumerate(
        _mapping_candidates(document, entities, question, decomposition), start=1
    ):
        query, compatible = _replace_scalars(document.sparql, document.question, question)
        if not compatible:
            continue
        query, numeric_rewrite_count = _rewrite_numeric_equality_literal(
            query,
            question=question,
            source_signature=source_signature,
        )
        mapping: dict[str, str] = {}
        for source_id, (target_id, _) in zip(document.constants, assigned):
            mapping[source_id] = target_id
            query = re.sub(
                rf"(?<![A-Za-z0-9_])ns:{re.escape(source_id)}(?![A-Za-z0-9_])",
                f"ns:{target_id}",
                query,
            )
        if not SELECT_LIMIT_RE.search(query):
            query = f"{query.strip()}\nLIMIT 256"
        output.append(
            {
                "key": hashlib.sha256(query.encode()).hexdigest()[:20],
                "query": query,
                "source_index": document.source_index,
                "source_template_rank": int(match["rank"]),
                "template_signature": document.signature,
                "mapping_rank": mapping_rank,
                "mapping_score": mapping_score,
                "answer_var": document.answer_var,
                "runtime_score": (
                    float(match["bm25"])
                    + 0.30 * math.log1p(int(match["support"]))
                    + 0.35 * mapping_score
                    - 0.04 * (mapping_rank - 1)
                ),
                "entity_mapping": mapping,
                "source_effective_constraint_signature": list(source_signature),
                "target_constraint_signature": list(target_signature),
                "constraint_signature_compatible": True,
                "numeric_equality_cast_rewrite_count": numeric_rewrite_count,
            }
        )
    return output


def _semantic_fallback_candidates(
    *,
    matches: Sequence[dict[str, Any]],
    entities: Sequence[tuple[str, str]],
    question: str,
    decomposition: Sequence[str],
    ranker: Any,
    executed_queries: set[str],
    limit: int,
) -> tuple[list[dict[str, Any]], str, int, int]:
    """Rerank unexecuted TRAIN templates after the ordinary cap three is empty.

    This helper never executes a query.  It uses the already configured local
    embedding ranker, removes entity identifiers/surfaces from similarity
    text, and reuses the same mapping-aware allocator as the ordinary lane.
    Returning an empty list is a safe abstention when embeddings are absent or
    malformed.
    """

    query_text = "query: " + _masked_text(
        question,
        decomposition,
        [label for _, label in entities],
    )
    documents = [_semantic_document(match["document"]) for match in matches]
    scores = list(ranker.score(query_text, documents))
    if len(scores) != len(matches):
        raise ValueError("failure semantic reranker returned the wrong score count")
    semantic_matches: list[dict[str, Any]] = []
    for match, score in zip(matches, scores):
        candidate_match = dict(match)
        candidate_match["embedding_score"] = float(score)
        # Preserve the existing support/mapping tie breakers while making BGE
        # the primary order.  This exact fixed scale is frozen by the full-80
        # production-context audit.
        candidate_match["bm25"] = 100.0 * float(score)
        semantic_matches.append(candidate_match)
    semantic_matches.sort(
        key=lambda item: (
            -float(item["embedding_score"]),
            int(item["rank"]),
            int(item["document"].source_index),
        )
    )
    embedding_calls = 2
    embedded_text_count = len(documents) + 1
    role_vectors: dict[str, list[float]] = {}
    entity_context_vectors: list[list[float]] = []
    embedding_client = getattr(ranker, "client", None)
    if callable(getattr(embedding_client, "embed", None)):
        entity_contexts = [
            "entity semantic context: "
            + _semantic_context(label, question, decomposition)
            for _, label in entities
        ]
        role_texts = _unique(
            "entity relation role: " + role
            for match in semantic_matches
            for role in match["document"].constant_roles
        )
        entity_context_vectors = list(
            embedding_client.embed(entity_contexts, input_type="query")
        )
        role_embeddings = list(
            embedding_client.embed(role_texts, input_type="document")
        )
        if len(entity_context_vectors) != len(entity_contexts) or len(
            role_embeddings
        ) != len(role_texts):
            raise ValueError("failure relation-role embeddings have wrong shape")
        role_vectors = dict(zip(role_texts, role_embeddings))
        embedding_calls += 2
        embedded_text_count += len(entity_contexts) + len(role_texts)
    candidates: list[dict[str, Any]] = []
    for match in semantic_matches:
        instantiated = _instantiate(match, entities, question, decomposition)
        for candidate in instantiated:
            candidate["embedding_score"] = float(match["embedding_score"])
            if role_vectors:
                document = match["document"]
                entity_positions = {
                    entity_id: position
                    for position, (entity_id, _) in enumerate(entities)
                }
                alignments: list[float] = []
                for position, source_id in enumerate(document.constants):
                    target_id = candidate["entity_mapping"].get(source_id, "")
                    target_position = entity_positions.get(target_id)
                    role = (
                        document.constant_roles[position]
                        if position < len(document.constant_roles)
                        else ""
                    )
                    vector = role_vectors.get("entity relation role: " + role)
                    if target_position is not None and vector is not None:
                        alignments.append(
                            _cosine_vectors(
                                entity_context_vectors[target_position], vector
                            )
                        )
                candidate["role_score"] = (
                    sum(alignments) / len(alignments) if alignments else 0.0
                )
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            -float(item["runtime_score"]),
            int(item["source_template_rank"]),
            int(item["mapping_rank"]),
            str(item["key"]),
        )
    )
    deduplicated: list[dict[str, Any]] = []
    seen = set(executed_queries)
    for candidate in candidates:
        normalized_query = " ".join(str(candidate["query"]).split())
        if normalized_query in seen:
            continue
        seen.add(normalized_query)
        deduplicated.append(candidate)
    query_limit = min(3, max(1, int(limit)))
    if role_vectors and query_limit >= 3:
        allocated = list(deduplicated[:2])
        allocated_keys = {str(candidate["key"]) for candidate in allocated}
        role_candidates = sorted(
            (
                candidate
                for candidate in deduplicated
                if str(candidate["key"]) not in allocated_keys
            ),
            key=lambda item: (
                -float(item.get("role_score", 0.0)),
                -float(item["runtime_score"]),
                int(item["source_template_rank"]),
                int(item["mapping_rank"]),
                str(item["key"]),
            ),
        )
        if role_candidates:
            allocated.append(role_candidates[0])
        allocation_reason = "bge_top2_plus_relation_role_top1"
    else:
        allocated, allocation_reason = _allocate_cap_three(
            deduplicated,
            entity_count=len(entities),
            limit=query_limit,
        )
    return (
        allocated,
        allocation_reason,
        embedded_text_count,
        embedding_calls,
    )


def _allocate_cap_three(
    candidates: Sequence[dict[str, Any]],
    *,
    entity_count: int,
    limit: int = 3,
) -> tuple[list[dict[str, Any]], str]:
    """Allocate the fixed SELECT budget without evaluation-time evidence.

    A confident slot mapping should not let its alternative permutations fill
    the whole budget.  Conversely, nearly tied mappings are real ambiguity and
    retain the original score order.  The thresholds are absolute because the
    mapping score itself is a bounded sum of fixed lexical/role similarities.
    """

    query_limit = min(3, max(1, int(limit)))
    if not candidates:
        return [], "no_candidates"
    first_signature = str(candidates[0].get("template_signature", ""))
    first_template = [
        candidate
        for candidate in candidates
        if str(candidate.get("template_signature", "")) == first_signature
    ]
    if entity_count >= 3 and len(first_template) >= 3:
        first_to_third_gap = float(first_template[0].get("mapping_score", 0.0)) - float(
            first_template[2].get("mapping_score", 0.0)
        )
        if first_to_third_gap <= 0.20:
            return list(candidates[:query_limit]), "flat_multi_entity_mapping"
    if entity_count > 1 and len(first_template) >= 2:
        mapping_margin = float(first_template[0].get("mapping_score", 0.0)) - float(
            first_template[1].get("mapping_score", 0.0)
        )
        if mapping_margin < 0.50:
            return list(candidates[:query_limit]), "ambiguous_entity_mapping"
    selected: list[dict[str, Any]] = []
    seen_templates: set[str] = set()
    for candidate in candidates:
        signature = str(candidate.get("template_signature", ""))
        if signature in seen_templates:
            continue
        seen_templates.add(signature)
        selected.append(candidate)
        if len(selected) >= query_limit:
            break
    return selected, "confident_mapping_template_diversity"


def _expects_single_answer(question: str) -> bool:
    return not PLURAL_ANSWER_RE.search(question) and bool(
        SINGULAR_ANSWER_RE.search(question)
    )


def _choose_runtime(
    candidates: Sequence[dict[str, Any]], question: str
) -> dict[str, Any] | None:
    nonempty = [candidate for candidate in candidates if candidate.get("answers")]
    if not nonempty:
        return None
    support = Counter(tuple(sorted(candidate["answers"])) for candidate in nonempty)
    singular = _expects_single_answer(question)
    return max(
        nonempty,
        key=lambda item: (
            support[tuple(sorted(item["answers"]))],
            int(singular and len(item["answers"]) == 1),
            -abs(len(item["answers"]) - 1) if singular else 0,
            float(item["runtime_score"]),
            -int(item["source_template_rank"]),
            str(item["key"]),
        ),
    )


def _decompositions(context: Any) -> list[str]:
    traces = (context.trace_bundle.get("decomposition", {}) or {}).get("traces", [])
    for trace in reversed(traces if isinstance(traces, list) else []):
        if isinstance(trace, dict) and trace.get("stage") == "decomposition_review_and_rewrite":
            output = trace.get("output", {})
            values = [
                str(item)
                for candidate in output.get("final_candidates", []) if isinstance(output, dict) and isinstance(candidate, dict)
                for item in candidate.get("decomposition", [])
                if str(item).strip()
            ]
            if values:
                return _unique(values)
    for trace in reversed(traces if isinstance(traces, list) else []):
        if isinstance(trace, dict) and trace.get("stage") == "semantic_path_generation":
            output = trace.get("output", [])
            values = [
                str(item)
                for candidate in output if isinstance(output, list) and isinstance(candidate, dict)
                for item in candidate.get("decomposition", [])
                if str(item).strip()
            ]
            if values:
                return _unique(values)
    for trace in reversed(traces if isinstance(traces, list) else []):
        if isinstance(trace, dict) and trace.get("stage") == "final_graph_beam":
            output = trace.get("output", [])
            values = [
                str(item)
                for graph in output if isinstance(output, list) and isinstance(graph, dict)
                for item in (graph.get("provenance") or {}).get("decomposition", [])
                if str(item).strip()
            ]
            if values:
                return _unique(values)
    # Strict trace replay can contain the already-reviewed hypotheses under
    # ``decompose_predictions`` without a separate review stage.  Prefer that
    # executed evidence over an absent/empty prediction store.
    for trace in reversed(traces if isinstance(traces, list) else []):
        if isinstance(trace, dict) and trace.get("stage") == "decompose_predictions":
            output = trace.get("output", [])
            values = [
                str(item)
                for candidate in output if isinstance(output, list) and isinstance(candidate, dict)
                for item in candidate.get("decomposition", [])
                if str(item).strip()
            ]
            if values:
                return _unique(values)
    try:
        candidates = context.pipeline.decompositions.get(context.question)
    except (AttributeError, KeyError):
        candidates = []
    return _unique(
        str(item)
        for candidate in candidates
        for item in getattr(candidate, "decomposition", [])
        if str(item).strip()
    )


def _trace_surfaces(context: Any) -> list[str]:
    traces = (context.trace_bundle.get("decomposition", {}) or {}).get("traces", [])
    surfaces: list[str] = []
    for trace in traces if isinstance(traces, list) else []:
        if not isinstance(trace, dict) or trace.get("stage") != "semantic_path_generation":
            continue
        output = trace.get("output", [])
        current = [
            str(anchor.get("surface", ""))
            for candidate in output if isinstance(output, list) and isinstance(candidate, dict)
            for anchor in (candidate.get("output") or {}).get("anchors", [])
            if isinstance(anchor, dict) and str(anchor.get("surface", "")).strip()
        ]
        if current:
            surfaces = _unique(current)
    return surfaces


def _entities(context: Any) -> list[tuple[str, str]]:
    mapping = getattr(context.pipeline.grounder, "gold_entities", {}).get(
        _normalized(context.question), {}
    )
    items = [(str(entity_id), str(label)) for entity_id, label in mapping.items()]
    normalized = _normalized(context.question)
    positions = {entity_id: index for index, (entity_id, _) in enumerate(items)}
    return sorted(
        items,
        key=lambda item: (
            normalized.find(item[1].casefold())
            if item[1].casefold() in normalized
            else len(normalized) + positions[item[0]],
            positions[item[0]],
        ),
    )


def _index(pipeline: Any) -> SourceGoldSparqlIndex:
    value = getattr(pipeline, "_failure_gold_sparql_index", None)
    if value is not None:
        return value
    lock = getattr(pipeline, "_failure_gold_sparql_index_lock", None)
    if lock is None:
        value = SourceGoldSparqlIndex(pipeline.failure_gold_sparql_cache_path)
        pipeline._failure_gold_sparql_index = value
        return value
    with lock:
        value = getattr(pipeline, "_failure_gold_sparql_index", None)
        if value is None:
            value = SourceGoldSparqlIndex(pipeline.failure_gold_sparql_cache_path)
            pipeline._failure_gold_sparql_index = value
    return value


def retrieve(context: Any, endpoint: Any) -> dict[str, Any]:
    started = time.perf_counter()
    pipeline = context.pipeline
    diagnostics: dict[str, Any] = {
        "version": VERSION,
        "status": "abstained",
        "trigger": "final_empty_after_existing_failure_lanes",
        "uses_evaluation_gold": False,
        "uses_test_sparql_or_reasoning_information": False,
        "source": "CWQ_TRAIN_gold_sparql_only",
        "llm_calls": 0,
        "embedding_calls": 0,
        "candidate_query_limit": 3,
        "entity_or_relation_whitelist": False,
    }
    entities = _entities(context)
    decomposition = _decompositions(context)
    if not entities:
        diagnostics["reason"] = "no_linked_entities"
        return {"answer_ids": [], "diagnostics": diagnostics}
    index = _index(pipeline)
    matches = index.retrieve(
        question=context.question,
        decomposition=decomposition,
        # Only grounded entities are masks.  Semantic traces can also expose
        # type/category anchors (for example "Country") which are useful
        # retrieval words, not interchangeable entity slots; masking them
        # erases precisely the template distinction this lane needs.
        surfaces=_unique(label for _, label in entities),
        entity_count=len(entities),
        top_k=int(getattr(pipeline, "failure_gold_sparql_top_k", 32)),
    )
    candidates: list[dict[str, Any]] = []
    for match in matches:
        candidates.extend(_instantiate(match, entities, context.question, decomposition))
    candidates.sort(
        key=lambda item: (
            -float(item["runtime_score"]),
            int(item["source_template_rank"]),
            int(item["mapping_rank"]),
            str(item["key"]),
        )
    )
    deduplicated_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized_query = " ".join(candidate["query"].split())
        if normalized_query in seen:
            continue
        seen.add(normalized_query)
        deduplicated_candidates.append(candidate)
    selected_candidates, allocation_reason = _allocate_cap_three(
        deduplicated_candidates,
        entity_count=len(entities),
        limit=int(getattr(pipeline, "failure_gold_sparql_candidate_limit", 3)),
    )
    base_execution_queries = 0
    for candidate in selected_candidates:
        query_started = time.perf_counter()
        try:
            rows = endpoint.execute(candidate["query"])
            answers = _unique(_answer_values(rows, candidate["answer_var"]))
            error = ""
        except Exception as exc:
            answers = []
            error = f"{type(exc).__name__}:{exc}"
        base_execution_queries += 1
        candidate["answers"] = answers
        candidate["error"] = error
        candidate["elapsed_seconds"] = time.perf_counter() - query_started
    selected = _choose_runtime(selected_candidates, context.question)
    adaptive_candidates: list[dict[str, Any]] = []
    adaptive_allocation_reason = "not_triggered"
    adaptive_execution_queries = 0
    adaptive_embedding_calls = 0
    adaptive_embedded_texts = 0
    adaptive_error = ""
    if (
        selected is None
        and bool(
            getattr(
                pipeline,
                "failure_gold_sparql_adaptive_semantic_enabled",
                False,
            )
        )
    ):
        ranker = getattr(getattr(pipeline, "grounder", None), "ranker", None)
        if ranker is None or not callable(getattr(ranker, "score", None)):
            adaptive_error = "embedding_ranker_unavailable"
        else:
            executed_queries = {
                " ".join(str(candidate["query"]).split())
                for candidate in selected_candidates
            }
            try:
                (
                    adaptive_candidates,
                    adaptive_allocation_reason,
                    adaptive_embedded_texts,
                    adaptive_embedding_calls,
                ) = _semantic_fallback_candidates(
                    matches=matches,
                    entities=entities,
                    question=context.question,
                    decomposition=decomposition,
                    ranker=ranker,
                    executed_queries=executed_queries,
                    limit=int(
                        getattr(
                            pipeline,
                            "failure_gold_sparql_adaptive_candidate_limit",
                            3,
                        )
                    ),
                )
                for candidate in adaptive_candidates:
                    query_started = time.perf_counter()
                    try:
                        rows = endpoint.execute(candidate["query"])
                        answers = _unique(
                            _answer_values(rows, candidate["answer_var"])
                        )
                        error = ""
                    except Exception as exc:
                        answers = []
                        error = f"{type(exc).__name__}:{exc}"
                    adaptive_execution_queries += 1
                    candidate["answers"] = answers
                    candidate["error"] = error
                    candidate["elapsed_seconds"] = (
                        time.perf_counter() - query_started
                    )
                selected = _choose_runtime(adaptive_candidates, context.question)
            except Exception as exc:
                adaptive_error = f"{type(exc).__name__}:{exc}"

    execution_queries = base_execution_queries + adaptive_execution_queries
    adaptive_triggered = adaptive_embedding_calls > 0
    selected_from_adaptive = bool(
        selected is not None and selected in adaptive_candidates
    )

    def candidate_trace(candidate: dict[str, Any], lane: str) -> dict[str, Any]:
        trace = {
            "lane": lane,
            "key": candidate["key"],
            "query_sha256": hashlib.sha256(candidate["query"].encode()).hexdigest(),
            "source_index": int(candidate["source_index"]),
            "source_template_rank": int(candidate["source_template_rank"]),
            "mapping_rank": int(candidate["mapping_rank"]),
            "answer_count": len(candidate["answers"]),
            "elapsed_seconds": candidate["elapsed_seconds"],
            "error": candidate["error"],
        }
        if "embedding_score" in candidate:
            trace["embedding_score"] = float(candidate["embedding_score"])
        if "role_score" in candidate:
            trace["role_score"] = float(candidate["role_score"])
        return trace

    diagnostics.update(
        {
            "status": (
                "selected_adaptive_semantic"
                if selected_from_adaptive
                else ("selected" if selected else "all_candidates_empty")
            ),
            "reason": (
                "source_train_gold_sparql_adaptive_semantic_answer_consensus"
                if selected_from_adaptive
                else (
                    "source_train_gold_sparql_answer_consensus"
                    if selected
                    else "all_instantiated_queries_empty"
                )
            ),
            "embedding_calls": adaptive_embedding_calls,
            "embedded_text_count": adaptive_embedded_texts,
            "adaptive_semantic_triggered": adaptive_triggered,
            "adaptive_semantic_error": adaptive_error,
            "adaptive_candidate_query_limit": 3,
            "total_candidate_query_limit": (
                6
                if bool(
                    getattr(
                        pipeline,
                        "failure_gold_sparql_adaptive_semantic_enabled",
                        False,
                    )
                )
                else 3
            ),
            "base_endpoint_execution_queries": base_execution_queries,
            "adaptive_endpoint_execution_queries": adaptive_execution_queries,
            "adaptive_candidate_allocation": adaptive_allocation_reason,
            "adaptive_answer_count": (
                len(selected["answers"]) if selected_from_adaptive else 0
            ),
            "selection_lane": (
                "adaptive_semantic" if selected_from_adaptive else "ordinary"
            ),
            "index_size": len(index.documents),
            "index_load_seconds": index.load_seconds,
            "retrieved_template_count": len(matches),
            "candidate_allocation": allocation_reason,
            "endpoint_execution_queries": execution_queries,
            "answer_count": len(selected["answers"]) if selected else 0,
            "selected_source_index": int(selected["source_index"]) if selected else -1,
            "selected_source_template_rank": int(selected["source_template_rank"]) if selected else -1,
            "selected_mapping_rank": int(selected["mapping_rank"]) if selected else -1,
            "candidate_traces": [
                *(
                    candidate_trace(candidate, "ordinary")
                    for candidate in selected_candidates
                ),
                *(
                    candidate_trace(candidate, "adaptive_semantic")
                    for candidate in adaptive_candidates
                ),
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    answer_ids = list(selected["answers"]) if selected else []
    label_rows: list[dict[str, str]] = []
    label_error = ""
    entity_answer_ids = [
        answer_id for answer_id in answer_ids if ENTITY_ID_RE.fullmatch(answer_id)
    ]
    if entity_answer_ids:
        try:
            label_rows = list(pipeline.kg.labels(entity_answer_ids))
        except Exception as exc:
            label_error = f"{type(exc).__name__}:{exc}"
    labels_by_id = {
        str(item.get("id", "")): str(item.get("label", ""))
        for item in label_rows
        if isinstance(item, dict) and str(item.get("id", ""))
    }
    diagnostics["selected_label_queries"] = int(bool(entity_answer_ids))
    if label_error:
        diagnostics["selected_label_error"] = label_error
    return {
        "answer_ids": answer_ids,
        "answers": [
            {"id": value, "label": labels_by_id.get(value, value)}
            for value in answer_ids
        ],
        "selected_graph": None,
        "diagnostics": diagnostics,
    }


__all__ = ["SourceGoldSparqlIndex", "retrieve"]
