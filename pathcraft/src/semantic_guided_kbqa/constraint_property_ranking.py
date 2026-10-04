"""Conservative evidence for ranking a late scalar constraint property.

This module is intentionally independent from :mod:`pipeline`.  It contains
only question/schema scoring and answer-set safety checks, so an offline
replay can evaluate a new ranker without changing production behaviour.

No entity, relation, question, or sample whitelist is used.  The only lexical
constants are language stop words and generic scalar words; dates and numbers
remain handled by the caller's existing normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "did", "do", "does", "for", "had", "has", "have", "he",
    "her", "his", "how", "i", "in", "is", "it", "its", "name",
    "of", "on", "or", "our", "she", "that", "the", "their", "them", "to",
    "there", "these", "they", "this", "those", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "with",
}

_GENERIC_SCALAR_WORDS = {
    "amount", "date", "datetime", "float", "id", "identifier", "integer",
    "number", "rate", "time", "unit", "value", "year",
}

_EXTREMA_WORDS = {
    "earliest", "first", "highest", "largest", "last", "latest", "least",
    "lowest", "maximum", "minimum", "most", "recent", "smallest",
}

_COMPARISON_WORDS = {
    "above", "after", "before", "below", "fewer", "greater", "higher",
    "larger", "later", "less", "lower", "more", "over", "prior",
    "smaller", "under",
}

_TEMPORAL_WORDS = {
    "begin", "birth", "date", "death", "die", "end", "found", "from",
    "open", "release", "start", "time", "to", "year",
}

_NUMERIC_WORDS = {
    "amount", "area", "army", "casualty", "code", "count", "height",
    "id", "length", "number", "population", "postgraduate", "rate",
    "soldier", "total", "undergraduate",
}

_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}


def _word_key(value: str) -> str:
    """Apply the same deliberately small morphology used by late repair."""
    token = str(value).casefold()
    if token in {"died", "dead"}:
        return "death"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("sed") and len(token) > 4:
        return token[:-1]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def content_tokens(value: str, *, drop_scalar_words: bool = False) -> set[str]:
    """Return meaningful normalized English tokens from surface/schema text."""
    tokens = {
        _word_key(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(value))
    }
    tokens = {token for token in tokens if len(token) > 1}
    tokens.difference_update(_STOP_WORDS)
    if drop_scalar_words:
        tokens.difference_update(_GENERIC_SCALAR_WORDS)
    return tokens


def relation_tail_text(relation: str) -> str:
    """Convert a relation id to its terminal human-readable property name."""
    return " ".join(
        re.findall(
            r"[A-Za-z0-9]+",
            str(relation).rsplit(".", 1)[-1].replace("_", " "),
        )
    )


def constraint_focus_phrase(focus: str, *, window: int = 7) -> str:
    """Keep the local attribute phrase instead of unrelated question nouns.

    Full-question similarity overweights words describing the source graph
    (for example ``country`` in a question whose requested attribute is
    ``army``).  A bounded window around the explicit comparison/extremum is a
    more faithful embedding query and does not require a domain lexicon.
    """
    tokens = re.findall(r"[A-Za-z0-9]+(?:[.,/-][A-Za-z0-9]+)*", str(focus))
    if not tokens:
        return str(focus)
    normalized = [_word_key(token) for token in tokens]
    cue_indexes = [
        index
        for index, token in enumerate(normalized)
        if token in _EXTREMA_WORDS or token in _COMPARISON_WORDS
    ]
    numeric_indexes = [
        index for index, token in enumerate(tokens) if re.search(r"\d", token)
    ]
    if cue_indexes:
        pivot = cue_indexes[-1]
        # In an extrema phrase the attribute normally follows the cue
        # (``smallest army``, ``earliest released film``).  Starting at the
        # cue prevents the answer target immediately before it (``country``)
        # from looking like property evidence.  If no meaningful word follows
        # the cue, fall back to the left context for forms such as
        # ``number of undergraduates the largest`` or ``position from latest``.
        if normalized[pivot] in _EXTREMA_WORDS:
            right_end = min(len(tokens), pivot + max(3, int(window)))
            right = tokens[pivot:right_end]
            right_content = content_tokens(" ".join(tokens[pivot + 1:right_end]))
            if right_content:
                return " ".join(right)
            return " ".join(tokens[max(0, pivot - 5): pivot + 1])
    elif numeric_indexes:
        pivot = numeric_indexes[-1]
    else:
        return " ".join(tokens[-max(3, int(window)):])
    # Two words before a comparison normally cover its attribute phrase
    # (``calling codes over`` / ``released after``) without pulling in the
    # answer target (``countries`` / ``movies``).
    start = max(0, pivot - 2)
    # Equality phrases often insert light verbs immediately before the value
    # (``population was once 10,004,081``). Walk a few tokens farther left to
    # retain the nearest substantive property word without using a domain
    # vocabulary.
    light_tokens = {
        "a", "an", "at", "been", "being", "had", "has", "have", "is",
        "of", "once", "the", "to", "was", "were", "with",
    }
    if all(
        token in light_tokens
        for token in normalized[max(0, pivot - 2):pivot]
    ):
        for index in range(pivot - 3, max(-1, pivot - 7), -1):
            if index < 0:
                break
            if normalized[index] not in light_tokens:
                start = index
                break
    # A written calendar date occupies two surface tokens (``December
    # 25,2003``), so the usual two-token look-behind stops at the preposition
    # and loses the temporal predicate.  For this time-normalization form only,
    # retain the nearest local temporal verb (released/opened/founded/etc.).
    # Numeric constraints keep the original tight window.
    if any(
        token in _MONTH_WORDS
        for token in normalized[max(0, pivot - 2): pivot + 1]
    ):
        temporal_indexes = [
            index
            for index in range(max(0, pivot - 5), pivot)
            if normalized[index] in _TEMPORAL_WORDS
        ]
        if temporal_indexes:
            start = temporal_indexes[-1]
    end = min(len(tokens), pivot + max(3, int(window)))
    return " ".join(tokens[start:end])


def requested_family(spec: Mapping[str, Any]) -> str:
    """Return ``temporal``, ``numeric`` or an empty unknown family."""
    value_type = str(spec.get("value_type", "")).casefold()
    if value_type == "datetime":
        return "temporal"
    if value_type == "number":
        return "numeric"
    tokens = content_tokens(str(spec.get("focus", "")))
    if tokens & _TEMPORAL_WORDS:
        return "temporal"
    if tokens & _NUMERIC_WORDS:
        return "numeric"
    return ""


def property_family(
    first_relation: str,
    second_relation: str,
    terminal_range: str,
) -> str:
    """Infer a scalar property's family from ontology range and leaf words."""
    terminal = second_relation or first_relation
    tokens = content_tokens(relation_tail_text(terminal))
    if str(terminal_range) == "type.datetime" or tokens & _TEMPORAL_WORDS:
        return "temporal"
    if str(terminal_range) in {"type.float", "type.int", "type.enumeration"}:
        return "numeric"
    if tokens & _NUMERIC_WORDS:
        return "numeric"
    return ""


@dataclass(frozen=True, slots=True)
class PropertyRankEvidence:
    """Inspectable, sortable evidence for one schema property path."""

    score: float
    family_score: float
    lexical_recall: float
    lexical_precision: float
    semantic_similarity: float
    directness: float
    matched_tokens: tuple[str, ...]
    first_matched_tokens: tuple[str, ...]
    query_tokens: tuple[str, ...]
    property_tokens: tuple[str, ...]
    terminal_relation: str

    @property
    def rank_key(self) -> tuple[float, float, float, float, float]:
        return (
            self.score,
            self.family_score,
            self.lexical_recall,
            self.semantic_similarity,
            self.lexical_precision,
        )


def property_rank_evidence(
    *,
    first_relation: str,
    second_relation: str,
    spec: Mapping[str, Any],
    terminal_range: str,
    semantic_similarity: float,
    graph_score: float = 0.0,
) -> PropertyRankEvidence:
    """Score one property without variables, answer ids, or answer values.

    A relation token is counted at most once across a two-hop path.  This is
    the critical difference from adding independent first/second-hop overlap:
    ``release_dates -> release_date`` must not receive two votes for the same
    word and outrank the canonical direct ``initial_release_date`` property.
    """
    local_focus = constraint_focus_phrase(str(spec.get("focus", "")))
    query_tokens = content_tokens(local_focus, drop_scalar_words=True)
    first_tokens = content_tokens(
        relation_tail_text(first_relation), drop_scalar_words=True
    )
    second_tokens = content_tokens(
        relation_tail_text(second_relation), drop_scalar_words=True
    )
    path_tokens = first_tokens | second_tokens
    matched = query_tokens & path_tokens
    lexical_recall = len(matched) / max(1, len(query_tokens))
    lexical_precision = len(matched) / max(1, len(path_tokens))

    wanted_family = requested_family(spec)
    actual_family = property_family(
        first_relation,
        second_relation,
        terminal_range,
    )
    family_score = (
        1.0
        if wanted_family and actual_family == wanted_family
        else -1.0
        if wanted_family and actual_family and actual_family != wanted_family
        else 0.0
    )

    if not second_relation:
        directness = 0.10
    elif query_tokens & first_tokens:
        directness = 0.05
    elif float(semantic_similarity) >= 0.62:
        # A strong local-focus paraphrase is sufficient for a normal Freebase
        # measurement CVT (army -> size_of_armed_forces -> number).
        directness = 0.025
    else:
        directness = -0.10

    # Scores are deliberately continuous: BGE can bridge a genuine lexical
    # paraphrase (army -> armed forces), while lexical precision keeps an
    # unrelated long relation from winning on one accidental word.
    score = (
        1.40 * lexical_recall
        # Precision is a modest tie-breaker. A canonical property can carry a
        # useful modifier not repeated verbatim in the question (``initial``
        # release date), so unmatched schema words must not dominate recall
        # and local semantic similarity.
        + 0.20 * lexical_precision
        + 1.25 * float(semantic_similarity)
        + 0.55 * family_score
        + directness
        + 0.02 * float(graph_score)
    )
    return PropertyRankEvidence(
        score=score,
        family_score=family_score,
        lexical_recall=lexical_recall,
        lexical_precision=lexical_precision,
        semantic_similarity=float(semantic_similarity),
        directness=directness,
        matched_tokens=tuple(sorted(matched)),
        first_matched_tokens=tuple(sorted(query_tokens & first_tokens)),
        query_tokens=tuple(sorted(query_tokens)),
        property_tokens=tuple(sorted(path_tokens)),
        terminal_relation=str(second_relation or first_relation),
    )


def semantic_only_is_confident(
    best: PropertyRankEvidence,
    alternatives: Iterable[PropertyRankEvidence],
    *,
    minimum_similarity: float = 0.62,
    minimum_margin: float = 0.05,
) -> bool:
    """Gate a property supported by BGE but by no literal surface token.

    Lexical matches are already auditable evidence.  A pure paraphrase is
    accepted only when its local-focus similarity is both strong and clearly
    separated from other property paths.  This admits ``army`` -> ``size of
    armed forces`` while abstaining on an underspecified phrase such as
    ``smallest on the program``.
    """
    if any(len(token) >= 5 for token in best.matched_tokens):
        return True
    distinct = sorted(
        (
            item.semantic_similarity
            for item in alternatives
            if item.terminal_relation != best.terminal_relation
            and item.family_score == best.family_score
        ),
        reverse=True,
    )
    runner_up = distinct[0] if distinct else -1.0
    return bool(
        best.semantic_similarity >= float(minimum_similarity)
        and best.semantic_similarity - runner_up >= float(minimum_margin)
    )


def answer_set_safety_reason(
    source_ids: Sequence[str],
    repaired_ids: Sequence[str],
    *,
    operator_type: str,
    source_answers: Sequence[Mapping[str, Any]] = (),
    repaired_answers: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Return an answer-evidence-independent reason to abstain, or ``""``.

    The checks use only cardinality and labels already fetched for execution.
    They never inspect gold.  They target two observed failure modes: applying
    an extremum to a very broad/incomplete source graph, and choosing a nested
    child label when a discarded source answer is its named parent.
    """
    source = set(map(str, source_ids))
    repaired = set(map(str, repaired_ids))
    if not repaired or not repaired < source:
        return "not_strict_subset"
    if (
        str(operator_type).upper() in {"ARGMIN", "ARGMAX"}
        and len(source) > 32
        and len(repaired) == 1
        and len(repaired) * 20 < len(source)
    ):
        return "broad_source_extrema_collapse"

    if (
        str(operator_type).upper()
        in {"GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL"}
        and len(repaired) == 1
        and source_answers
        and repaired_answers
    ):
        if nested_label_parent_ids(
            source_ids,
            repaired_ids,
            source_answers=source_answers,
            repaired_answers=repaired_answers,
        ):
            return "nested_child_label_collapse"
    return ""


def nested_label_parent_ids(
    source_ids: Sequence[str],
    repaired_ids: Sequence[str],
    *,
    source_answers: Sequence[Mapping[str, Any]] = (),
    repaired_answers: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    """Return source answers that are lexical parents of one repaired child.

    This is deliberately label-structural: a parent must lose at least two
    content tokens relative to the child, and no entity or relation identity
    is consulted.  Source order is preserved so a unique parent can be kept
    alongside a numeric-boundary child without restoring unrelated answers.
    """

    repaired = set(map(str, repaired_ids))
    if len(repaired) != 1:
        return []
    repaired_id = next(iter(repaired))
    source_labels = {
        str(item.get("id", "")): content_tokens(str(item.get("label", "")))
        for item in source_answers
        if isinstance(item, Mapping)
    }
    repaired_labels = {
        str(item.get("id", "")): content_tokens(str(item.get("label", "")))
        for item in repaired_answers
        if isinstance(item, Mapping)
    }
    child_tokens = repaired_labels.get(repaired_id) or source_labels.get(
        repaired_id, set()
    )
    if not child_tokens:
        return []
    return [
        answer_id
        for answer_id in map(str, source_ids)
        if answer_id not in repaired
        and len(source_labels.get(answer_id, set())) >= 2
        and source_labels.get(answer_id, set()) < child_tokens
        and len(child_tokens - source_labels.get(answer_id, set())) >= 2
    ]
