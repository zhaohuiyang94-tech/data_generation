"""Bounded failure retrieval for event/office interval-overlap questions.

The lane is an exclusive router in front of broad source-SPARQL retrieval.  It
indexes only the interval-overlap templates already stored in the CWQ TRAIN
cache, keeps one structurally best entity assignment per source template, and
executes at most three SELECT queries.  It never reads evaluation answers,
sample indexes, or an entity/relation allow-list, and performs no model call.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from itertools import permutations
import math
import re
import time
from typing import Any, Iterable, Sequence

from .failure_gold_sparql_retrieval import (
    _decompositions,
    _entities,
    _index,
    _operator_conflict,
    _token_similarity,
    _tokens,
    _unique,
)
from .pipeline import _answer_values
from .template_path_retrieval import mask_entities


VERSION = "failure-temporal-interval-retrieval-production-v1"
FULL_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
US_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})-(\d{1,2})-(\d{4})(?!\d)")
MONTH_DATE_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    r"(?P<month_name>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?P<month_day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<month_year>\d{4})"
    r"|(?P<day_day>\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(?P<day_month>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"[,]?\s+(?P<day_year>\d{4}))",
    re.I,
)
YEAR_RE = re.compile(r"(?<![\d.])(?:1[0-9]{3}|20[0-9]{2})(?![\d.])")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")
ENTITY_ID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
SELECT_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", re.I)
RELATION_RE = re.compile(r"\bns:([A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)\b")
MONTHS = {
    name.casefold(): position
    for position, name in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ),
        start=1,
    )
}

TERM_PATTERN = r'(?:\?[A-Za-z_][A-Za-z0-9_]*|ns:[A-Za-z0-9_.]+|"(?:[^"\\]|\\.)*"(?:\^\^[A-Za-z0-9_:.-]+)?|[-+]?\d+(?:\.\d+)?)'
REL_PATTERN = r"ns:([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+)"
TRIPLE_RE = re.compile(
    rf"(?P<s>{TERM_PATTERN})\s+{REL_PATTERN}\s+(?P<o>{TERM_PATTERN})",
    re.I,
)
LESS_FILTER_RE = re.compile(
    r"FILTER\s*\(\s*(?P<left>\?[A-Za-z_][A-Za-z0-9_]*)\s*<\s*(?P<right>\?[A-Za-z_][A-Za-z0-9_]*)\s*\)",
    re.I,
)
GREATER_FILTER_RE = re.compile(
    r"FILTER\s*\(\s*(?P<left>\?[A-Za-z_][A-Za-z0-9_]*)\s*>\s*(?P<right>\?[A-Za-z_][A-Za-z0-9_]*)\s*\)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class _IntervalSchema:
    event_subject: str
    event_start_relation: str
    event_end_relation: str
    tenure_subject: str
    answer_to_tenure_relation: str
    office_title_relation: str
    office_constant: str
    tenure_from_relation: str
    tenure_to_relation: str


def _normalized(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _traces(context: Any) -> list[dict[str, Any]]:
    bundle = getattr(context, "trace_bundle", {}) or {}
    values = (bundle.get("decomposition", {}) or {}).get("traces", [])
    return [value for value in values if isinstance(value, dict)]


def _type_hints(context: Any) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for trace in _traces(context):
        if trace.get("stage") != "decomposition_review_and_rewrite":
            continue
        value = trace.get("output", {})
        for candidate in value.get("candidates", []) if isinstance(value, dict) else []:
            for event in candidate.get("events", []) if isinstance(candidate, dict) else []:
                item = event.get("input", {}) if isinstance(event, dict) else {}
                for entity in item.get("entity_context", []) if isinstance(item, dict) else []:
                    if not isinstance(entity, dict):
                        continue
                    label = str(entity.get("name", ""))
                    if label:
                        output[label] = [str(value) for value in entity.get("types", [])]
    return output


def _eligibility_evidence(context: Any) -> tuple[bool, dict[str, bool]]:
    question = str(context.question)
    decomposition = _decompositions(context)
    type_hints = _type_hints(context)
    text = _normalized(" ".join([question, *decomposition]))
    question_text = _normalized(question)
    office_language = bool(
        re.search(
            r"\b(?:president|governor|prime minister|government(?:al)? position|office holder|tenure|leader)\b",
            text,
        )
    )
    temporal_join_language = bool(
        re.search(r"\b(?:during|while|at the same time|time period|tenure)\b", text)
    )
    # A legislative-session year window uses a different schema and remains
    # in the broad direct lane.
    event_language = bool(
        re.search(r"\b(?:war|battle|conflict|event|time period)\b", text)
    )
    explicit_bounds = bool(
        ("start and end date" in text or "start and end dates" in text)
        or re.search(r"\bbegan before\b[^.?!]*\bended after\b", text)
        or re.search(r"\bstarted before\b[^.?!]*\bended after\b", text)
    )
    typed_event = any(
        any(token in _normalized(" ".join(types)) for token in ("event", "war", "conflict", "battle"))
        for types in type_hints.values()
    )
    typed_office = any(
        any(token in _normalized(" ".join(types)) for token in ("government office", "appointed role"))
        for types in type_hints.values()
    )
    surface_join = bool(
        re.search(r"\b(?:during|while|at the same time|time period)\b", question_text)
        and event_language
    )
    evidence = {
        "office_language": office_language,
        "temporal_join_language": temporal_join_language,
        "event_language": event_language,
        "explicit_bounds": explicit_bounds,
        "typed_event": typed_event,
        "typed_office": typed_office,
        "surface_join": surface_join,
    }
    return (
        office_language
        and temporal_join_language
        and (explicit_bounds or (typed_event and typed_office) or surface_join),
        evidence,
    )


def eligible(context: Any) -> bool:
    """Return whether this row exclusively belongs to the interval lane."""

    return _eligibility_evidence(context)[0]


def _parse_triples(sparql: str) -> list[tuple[str, str, str]]:
    text = "\n".join(line.split("#", 1)[0] for line in str(sparql).splitlines())
    triples: list[tuple[str, str, str]] = []
    for match in TRIPLE_RE.finditer(text):
        subject, relation, object_ = match.group("s"), match.group(2), match.group("o")
        triples.append((subject, relation, object_))
        cursor = match.end()
        current_relation = relation
        while True:
            separator = re.match(r"\s*([;,])", text[cursor:])
            if not separator:
                break
            cursor += separator.end()
            if separator.group(1) == ";":
                continuation = re.match(
                    rf"\s*{REL_PATTERN}\s+(?P<o>{TERM_PATTERN})",
                    text[cursor:],
                    re.I,
                )
                if not continuation:
                    break
                current_relation = continuation.group(1)
                object_ = continuation.group("o")
            else:
                continuation = re.match(
                    rf"\s*(?P<o>{TERM_PATTERN})", text[cursor:], re.I
                )
                if not continuation:
                    break
                object_ = continuation.group("o")
            cursor += continuation.end()
            triples.append((subject, current_relation, object_))
    return list(dict.fromkeys(triples))


def _answer_term(sparql: str) -> str:
    match = re.search(
        r"\bSELECT\s+(?:DISTINCT\s+)?(?P<answer>\?[A-Za-z_][A-Za-z0-9_]*)",
        sparql,
        re.I,
    )
    return match.group("answer") if match else "?x"


def _derive_interval_schema(sparql: str) -> _IntervalSchema | None:
    """Discover the overlap topology without comparing any relation ID."""

    triples = _parse_triples(sparql)
    answer = _answer_term(sparql)
    by_object: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    by_subject: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for triple in triples:
        by_subject[triple[0]].append(triple)
        by_object[triple[2]].append(triple)
    for less in LESS_FILTER_RE.finditer(sparql):
        from_var, end_var = less.group("left"), less.group("right")
        for greater in GREATER_FILTER_RE.finditer(sparql):
            to_var, start_var = greater.group("left"), greater.group("right")
            for from_triple in by_object.get(from_var, []):
                tenure = from_triple[0]
                to_triple = next(
                    (value for value in by_object.get(to_var, []) if value[0] == tenure),
                    None,
                )
                if to_triple is None:
                    continue
                for start_triple in by_object.get(start_var, []):
                    event = start_triple[0]
                    end_triple = next(
                        (value for value in by_object.get(end_var, []) if value[0] == event),
                        None,
                    )
                    if end_triple is None:
                        continue
                    answer_edge = next(
                        (value for value in by_subject.get(answer, []) if value[2] == tenure),
                        None,
                    )
                    if answer_edge is None:
                        continue
                    title_edge = next(
                        (
                            value
                            for value in by_subject.get(tenure, [])
                            if value[2].startswith(("ns:m.", "ns:g."))
                            and value[1] not in {from_triple[1], to_triple[1]}
                        ),
                        None,
                    )
                    if title_edge is None:
                        continue
                    return _IntervalSchema(
                        event_subject=event,
                        event_start_relation=start_triple[1],
                        event_end_relation=end_triple[1],
                        tenure_subject=tenure,
                        answer_to_tenure_relation=answer_edge[1],
                        office_title_relation=title_edge[1],
                        office_constant=title_edge[2].removeprefix("ns:"),
                        tenure_from_relation=from_triple[1],
                        tenure_to_relation=to_triple[1],
                    )
    return None


def _is_interval_template(sparql: str) -> bool:
    return _derive_interval_schema(sparql) is not None


def _source_role(schema: _IntervalSchema, constant: str) -> str:
    if schema.event_subject == f"ns:{constant}":
        return "event"
    if schema.office_constant == constant:
        return "office"
    return "constraint"


def _relation_text(sparql: str) -> str:
    output: list[str] = []
    for relation in RELATION_RE.findall(sparql):
        if relation.startswith(("m.", "g.")):
            continue
        output.extend(part.replace("_", " ") for part in relation.split("."))
    return " ".join(output)


def _interval_operator_signature(question: str, sparql: str = "") -> set[str]:
    text = _normalized(question)
    output: set[str] = set()
    if re.search(r"\b(?:last|latest|most recent|highest|largest|greatest)\b", text):
        output.add("argmax")
    if re.search(r"\b(?:first|earliest|lowest|smallest|least)\b", text):
        output.add("argmin")
    if re.search(r"\b(?:after|later than|greater than|more than|over)\b", text):
        output.add("gt")
    if re.search(r"\b(?:before|earlier than|less than|under|prior to)\b", text):
        output.add("lt")
    upper = sparql.upper()
    if "ORDER BY DESC" in upper:
        output.add("argmax")
    elif "ORDER BY" in upper:
        output.add("argmin")
    filtered = re.sub(r"\?from\s*<\s*\?end|\?to\s*>\s*\?start", "", sparql)
    if re.search(r"FILTER\s*\([^)]*\?num\s*>", filtered, re.I):
        output.add("gt")
    if re.search(r"FILTER\s*\([^)]*\?num\s*<", filtered, re.I):
        output.add("lt")
    return output


def _scalar_slots(text: str) -> list[tuple[str, str]]:
    occupied: list[tuple[int, int]] = []
    output: list[tuple[int, str, str]] = []
    for pattern in (FULL_DATE_RE, US_DATE_RE):
        for match in pattern.finditer(text):
            if any(left <= match.start() < right for left, right in occupied):
                continue
            occupied.append(match.span())
            value = match.group(0)
            if pattern is US_DATE_RE:
                value = f"{int(match.group(3)):04d}-{int(match.group(1)):02d}-{int(match.group(2)):02d}"
            output.append((match.start(), "date", value))
    for match in MONTH_DATE_RE.finditer(text):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        occupied.append(match.span())
        month = match.group("month_name") or match.group("day_month")
        day = match.group("month_day") or match.group("day_day")
        year = match.group("month_year") or match.group("day_year")
        output.append(
            (
                match.start(),
                "date",
                f"{int(year):04d}-{MONTHS[month.casefold()]:02d}-{int(day):02d}",
            )
        )
    for match in YEAR_RE.finditer(text):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        occupied.append(match.span())
        output.append((match.start(), "year", match.group(0)))
    for match in NUMBER_RE.finditer(text):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        if re.search(r"world\s+war\s*$", text[max(0, match.start() - 16):match.start()], re.I):
            continue
        output.append((match.start(), "number", match.group(0)))
    return [(kind, value) for _, kind, value in sorted(output)]


def _normalize_scalars(text: str) -> str:
    value = FULL_DATE_RE.sub(" <date> ", str(text))
    value = US_DATE_RE.sub(" <date> ", value)
    value = MONTH_DATE_RE.sub(" <date> ", value)
    value = YEAR_RE.sub(" <year> ", value)
    output: list[str] = []
    cursor = 0
    for match in NUMBER_RE.finditer(value):
        output.append(value[cursor:match.start()])
        before = value[max(0, match.start() - 16):match.start()]
        output.append(match.group(0) if re.search(r"world\s+war\s*$", before, re.I) else " <number> ")
        cursor = match.end()
    output.append(value[cursor:])
    return " ".join("".join(output).split())


def _replace_scalars(query: str, source_question: str, target_question: str) -> tuple[str, bool]:
    source, target = _scalar_slots(source_question), _scalar_slots(target_question)
    if Counter(kind for kind, _ in source) != Counter(kind for kind, _ in target):
        return query, not source and not target
    source_by_kind: dict[str, list[str]] = defaultdict(list)
    target_by_kind: dict[str, list[str]] = defaultdict(list)
    for kind, value in source:
        source_by_kind[kind].append(value)
    for kind, value in target:
        target_by_kind[kind].append(value)
    for kind, values in source_by_kind.items():
        for old, new in zip(values, target_by_kind[kind]):
            if kind == "year":
                query = query.replace(f'"{old}-01-01"', f'"{new}-01-01"')
                query = query.replace(f'"{old}-12-31"', f'"{new}-12-31"')
            query = re.sub(rf"(?<![\d.]){re.escape(old)}(?![\d.])", new, query)
    return query, True


def _target_role_score(role: str, label: str, types: Sequence[str], context: str) -> float:
    type_text, label_text, context_text = (
        _normalized(" ".join(types)),
        _normalized(label),
        _normalized(context),
    )
    score = 0.0
    if role == "event":
        if any(value in type_text for value in ("event", "war", "conflict", "battle")):
            score += 3.0
        if re.search(rf"\b(?:during|while|period|war|battle)\b[^.?!]*\b{re.escape(label_text)}\b", context_text):
            score += 0.8
    elif role == "office":
        if any(value in type_text for value in ("government office", "appointed role")):
            score += 3.0
        if any(value in label_text for value in ("president", "governor", "minister", "secretary", "office", "title")):
            score += 0.8
    return score


def _mapping(
    document: Any,
    schema: _IntervalSchema,
    entities: Sequence[tuple[str, str]],
    type_hints: dict[str, list[str]],
    context: str,
) -> tuple[float, tuple[tuple[str, str], ...]] | None:
    output: list[tuple[float, tuple[tuple[str, str], ...]]] = []
    roles = [_source_role(schema, value) for value in document.constants]
    for assigned in permutations(entities, len(document.constants)):
        score = 0.0
        for position, ((target_id, label), source_id) in enumerate(zip(assigned, document.constants)):
            if target_id == source_id:
                score += 4.0
            score += _target_role_score(roles[position], label, type_hints.get(label, []), context)
            role_text = document.constant_roles[position] if position < len(document.constant_roles) else ""
            score += 2.0 * _token_similarity(role_text, context)
        output.append((score, tuple(assigned)))
    return max(output, default=None, key=lambda item: (item[0], tuple(value[0] for value in item[1])))


def _interval_documents(pipeline: Any, index: Any) -> tuple[tuple[Any, _IntervalSchema], ...]:
    cached = getattr(pipeline, "_temporal_interval_documents", None)
    if cached is not None:
        return cached
    documents = tuple(
        (document, schema)
        for document in index.documents
        if (schema := _derive_interval_schema(document.sparql)) is not None
    )
    pipeline._temporal_interval_documents = documents
    return documents


def _retrieve_documents(
    context: Any,
    interval_documents: Sequence[tuple[Any, _IntervalSchema]],
    entities: Sequence[tuple[str, str]],
) -> list[tuple[float, Any, _IntervalSchema]]:
    decomposition = _decompositions(context)
    surfaces = [label for _, label in entities]
    masked_question = mask_entities(_normalize_scalars(context.question), surfaces)
    query_tokens = Counter(
        _tokens(
            " ".join(
                [masked_question, masked_question]
                + [mask_entities(_normalize_scalars(value), surfaces) for value in decomposition]
            )
        )
    )
    compatible = [
        (document, schema)
        for document, schema in interval_documents
        if len(document.constants) == len(entities)
    ]
    document_tokens = [
        _tokens(" ".join([_normalize_scalars(document.question), _relation_text(document.sparql)]))
        for document, _ in compatible
    ]
    frequencies = [Counter(values) for values in document_tokens]
    average_length = (
        sum(len(values) for values in document_tokens) / len(document_tokens)
        if document_tokens else 0.0
    )
    document_frequency: Counter[str] = Counter()
    for values in document_tokens:
        document_frequency.update(set(values))
    count = len(document_tokens)
    idf = {
        token: math.log(1.0 + ((count - frequency + 0.5) / (frequency + 0.5)))
        for token, frequency in document_frequency.items()
    }
    target_operator = _interval_operator_signature(context.question)
    ranked: list[tuple[float, Any, _IntervalSchema]] = []
    for position, (document, schema) in enumerate(compatible):
        source_operator = _interval_operator_signature(
            document.question, document.sparql
        )
        if _operator_conflict(source_operator, target_operator):
            continue
        frequency = frequencies[position]
        normalization = 0.25 + (
            0.75 * len(document_tokens[position]) / average_length
            if average_length else 0.0
        )
        score = sum(
            idf.get(token, 0.0)
            * (count * 2.2 / (count + 1.2 * normalization))
            * min(query_count, 2)
            for token, query_count in query_tokens.items()
            if (count := frequency.get(token, 0))
        )
        score += 12.0 * _token_similarity(str(context.question), document.question)
        if target_operator == source_operator:
            score += 1.0
        ranked.append((score, document, schema))
    ranked.sort(key=lambda item: (-item[0], item[1].source_index))
    return ranked[:32]


def _instantiate_templates(
    context: Any,
    entities: Sequence[tuple[str, str]],
    interval_documents: Sequence[tuple[Any, _IntervalSchema]],
) -> list[dict[str, Any]]:
    decomposition = _decompositions(context)
    type_hints = _type_hints(context)
    text = " ".join([str(context.question), *decomposition])
    output: list[dict[str, Any]] = []
    for rank, (score, document, schema) in enumerate(
        _retrieve_documents(context, interval_documents, entities), start=1
    ):
        assigned = _mapping(document, schema, entities, type_hints, text)
        if assigned is None:
            continue
        mapping_score, values = assigned
        query, compatible = _replace_scalars(document.sparql, document.question, context.question)
        if not compatible:
            continue
        mapping: dict[str, str] = {}
        for source_id, (target_id, _) in zip(document.constants, values):
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
                "source_rank": rank,
                "mapping_score": mapping_score,
                "answer_var": document.answer_var,
                "runtime_score": score + 0.8 * mapping_score,
                "mode": "train_interval_template",
            }
        )
    return output


def _relation_id(value: Any) -> str:
    if isinstance(value, list):
        return ".".join(str(part).strip().replace(" ", "_") for part in value)
    return str(value).strip()


def _event_clause(
    context: Any,
    entities: Sequence[tuple[str, str]],
    type_hints: dict[str, list[str]],
    office_id: str,
) -> tuple[list[str], str, dict[str, Any]] | None:
    for entity_id, label in entities:
        if entity_id == office_id:
            continue
        type_text = _normalized(" ".join(type_hints.get(label, [])))
        if any(value in type_text for value in ("event", "war", "conflict", "battle")):
            return [], f"ns:{entity_id}", {"mode": "direct_typed_event"}
    # Type hints can be absent after a guarded decomposition rollback even
    # though entity linking still gives an unambiguous event/office pair.  A
    # single remaining entity whose surface itself denotes the event is safer
    # than splicing an unrelated semantic path.  This is surface grammar, not
    # an entity or relation allow-list.
    remaining = [
        (entity_id, label)
        for entity_id, label in entities
        if entity_id != office_id
    ]
    if len(remaining) == 1:
        entity_id, label = remaining[0]
        label_text = _normalized(label)
        question_text = _normalized(str(context.question))
        if (
            label_text
            and label_text in question_text
            and re.search(r"\b(?:event|war|battle|conflict)\b", label_text)
            and re.search(r"\b(?:during|while|at the same time)\b", question_text)
        ):
            return [], f"ns:{entity_id}", {"mode": "direct_surface_event"}
    entity_by_label = {
        _normalized(label): entity_id for entity_id, label in entities if entity_id != office_id
    }
    for trace in _traces(context):
        if trace.get("stage") != "semantic_path_generation":
            continue
        for candidate in trace.get("output", []) if isinstance(trace.get("output"), list) else []:
            value = candidate.get("output", {}) if isinstance(candidate, dict) else {}
            anchors = {
                str(item.get("id", "")): str(item.get("surface", ""))
                for item in value.get("anchors", []) if isinstance(item, dict)
            }
            for path in value.get("semantic_paths", []):
                if not isinstance(path, dict):
                    continue
                anchor_ref = str(path.get("anchor_ref", ""))
                entity_id = entity_by_label.get(_normalized(anchors.get(anchor_ref, "")))
                steps = [item for item in path.get("steps", []) if isinstance(item, dict)]
                if not entity_id or not steps:
                    continue
                # The reviewed decomposition names the event-producing path;
                # its first projected node is the event.  This avoids a
                # relation-name allow-list when locating the splice point.
                event_position = 1
                triples: list[str] = []
                current, event_term = f"ns:{entity_id}", ""
                for position, step in enumerate(steps[:event_position], start=1):
                    relation = _relation_id(step.get("relation_label", ""))
                    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+", relation):
                        triples = []
                        break
                    target = f"?event_path_{position}"
                    triples.append(
                        f"{target} ns:{relation} {current} ."
                        if str(step.get("direction", "")).casefold() == "backward"
                        else f"{current} ns:{relation} {target} ."
                    )
                    current, event_term = target, target
                if triples:
                    return triples, event_term, {
                        "mode": "frozen_semantic_event_path",
                        "hop_count": len(triples),
                    }
    return None


def _canonical_candidate(
    context: Any,
    entities: Sequence[tuple[str, str]],
    schema: _IntervalSchema | None,
) -> dict[str, Any] | None:
    if schema is None:
        return None
    decomposition, type_hints = _decompositions(context), _type_hints(context)
    text = " ".join([str(context.question), *decomposition])
    office_id, office_label = max(
        entities,
        key=lambda item: _target_role_score("office", item[1], type_hints.get(item[1], []), text),
    )
    if _target_role_score("office", office_label, type_hints.get(office_label, []), text) <= 0.0:
        return None
    event = _event_clause(context, entities, type_hints, office_id)
    if event is None:
        return None
    event_triples, event_term, event_evidence = event
    question_text = _normalized(context.question)
    slots = _scalar_slots(context.question)
    argmax = bool(re.search(r"\b(?:last|latest|most recent)\b", question_text))
    argmin = bool(re.search(r"\b(?:first|earliest)\b", question_text))
    if not slots and not argmax and not argmin and not event_triples:
        return None
    extra: list[str] = []
    ordering = "LIMIT 256"
    scalar_evidence: dict[str, Any] = {}
    if slots:
        dates = [value for kind, value in slots if kind in {"date", "year"}]
        if len(dates) != 1 or not re.search(r"\b(?:left|ended|end|terminated|until)\b", question_text):
            return None
        value = dates[0] if len(dates[0]) > 4 else f"{dates[0]}-01-01"
        if re.search(r"\b(?:after|later than)\b", question_text):
            comparison = ">"
        elif re.search(r"\b(?:before|prior to|earlier than)\b", question_text):
            comparison = "<"
        else:
            return None
        extra.extend(
            [
                f"?tenure ns:{schema.tenure_to_relation} ?constraint_date .",
                f'FILTER (xsd:dateTime(?constraint_date) {comparison} "{value}T00:00:00"^^xsd:dateTime)',
            ]
        )
        scalar_evidence = {
            "predicate_role": "tenure_end_from_train_schema",
            "comparison": comparison,
            "normalized_value": value,
        }
    elif argmax or argmin:
        ordering = f"ORDER BY {'DESC' if argmax else 'ASC'}(?to) LIMIT 1"
    lines = [
        "PREFIX ns: <http://rdf.freebase.com/ns/>",
        "SELECT DISTINCT ?x",
        "WHERE {",
        *(f"  {value}" for value in event_triples),
        f"  {event_term} ns:{schema.event_start_relation} ?start ;",
        f"             ns:{schema.event_end_relation} ?end .",
        f"  ?x ns:{schema.answer_to_tenure_relation} ?tenure .",
        f"  ?tenure ns:{schema.office_title_relation} ns:{office_id} ;",
        f"           ns:{schema.tenure_from_relation} ?from ;",
        f"           ns:{schema.tenure_to_relation} ?to .",
        "  FILTER (?from < ?end)",
        "  FILTER (?to > ?start)",
        *(f"  {value}" for value in extra),
        "}",
        ordering,
    ]
    query = "\n".join(lines)
    return {
        "key": hashlib.sha256(query.encode()).hexdigest()[:20],
        "query": query,
        "source_index": -1,
        "source_rank": 0,
        "mapping_score": 0.0,
        "answer_var": "x",
        "runtime_score": 1_000.0,
        "mode": "canonical_interval_splice",
        "canonical_evidence": {
            "event": event_evidence,
            "scalar": scalar_evidence,
            "operator": "argmax" if argmax else ("argmin" if argmin else "none"),
        },
    }


def retrieve(context: Any, endpoint: Any) -> dict[str, Any]:
    started = time.perf_counter()
    is_eligible, eligibility = _eligibility_evidence(context)
    diagnostics: dict[str, Any] = {
        "version": VERSION,
        "status": "abstained",
        "reason": "not_explicit_event_office_interval",
        "trigger": "final_empty_exclusive_interval_route",
        "eligible": is_eligible,
        "eligibility_evidence": eligibility,
        "claimed_query_budget": is_eligible,
        "uses_evaluation_gold": False,
        "uses_test_sparql_or_reasoning_information": False,
        "source": "CWQ_TRAIN_interval_gold_sparql_only",
        "llm_calls": 0,
        "embedding_calls": 0,
        "candidate_query_limit": 3,
        "entity_or_relation_whitelist": False,
    }
    if not is_eligible:
        return {"answer_ids": [], "diagnostics": diagnostics}
    entities = _entities(context)
    if len(entities) < 2:
        diagnostics["reason"] = "fewer_than_two_linked_entities"
        return {"answer_ids": [], "diagnostics": diagnostics}
    index = _index(context.pipeline)
    interval_documents = _interval_documents(context.pipeline, index)
    candidates = _instantiate_templates(context, entities, interval_documents)
    schema = interval_documents[0][1] if interval_documents else None
    canonical = _canonical_candidate(context, entities, schema)
    if canonical is not None:
        candidates.append(canonical)
    candidates.sort(
        key=lambda item: (
            -float(item["runtime_score"]),
            int(item["source_rank"]),
            str(item["key"]),
        )
    )
    selected_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        compact = " ".join(candidate["query"].split())
        if compact in seen:
            continue
        seen.add(compact)
        selected_candidates.append(candidate)
        if len(selected_candidates) >= min(
            3,
            max(1, int(getattr(context.pipeline, "temporal_interval_candidate_limit", 3))),
        ):
            break
    for candidate in selected_candidates:
        query_started = time.perf_counter()
        try:
            rows = endpoint.execute(candidate["query"])
            candidate["answers"] = _unique(_answer_values(rows, candidate["answer_var"]))
            candidate["error"] = ""
        except Exception as exc:
            candidate["answers"] = []
            candidate["error"] = f"{type(exc).__name__}:{exc}"
        candidate["elapsed_seconds"] = time.perf_counter() - query_started
    nonempty = [value for value in selected_candidates if value["answers"]]
    support = Counter(tuple(sorted(value["answers"])) for value in nonempty)
    selected = max(
        nonempty,
        key=lambda item: (
            support[tuple(sorted(item["answers"]))],
            float(item["runtime_score"]),
            -int(item["source_rank"]),
            str(item["key"]),
        ),
        default=None,
    )
    diagnostics.update(
        {
            "status": "selected" if selected else "all_candidates_empty",
            "reason": "temporal_interval_answer_consensus" if selected else "all_interval_queries_empty",
            "interval_template_count": len(interval_documents),
            "endpoint_execution_queries": len(selected_candidates),
            "answer_count": len(selected["answers"]) if selected else 0,
            "selected_source_index": int(selected["source_index"]) if selected else -1,
            "selected_source_rank": int(selected["source_rank"]) if selected else -1,
            "selected_mode": str(selected["mode"]) if selected else "",
            "candidate_traces": [
                {
                    "key": candidate["key"],
                    "query_sha256": hashlib.sha256(candidate.pop("query").encode()).hexdigest(),
                    "source_index": int(candidate["source_index"]),
                    "source_rank": int(candidate["source_rank"]),
                    "mode": str(candidate["mode"]),
                    "answer_count": len(candidate["answers"]),
                    "elapsed_seconds": candidate["elapsed_seconds"],
                    "error": candidate["error"],
                    **(
                        {"canonical_evidence": candidate["canonical_evidence"]}
                        if "canonical_evidence" in candidate else {}
                    ),
                }
                for candidate in selected_candidates
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    answer_ids = list(selected["answers"]) if selected else []
    entity_ids = [value for value in answer_ids if ENTITY_ID_RE.fullmatch(value)]
    labels: list[dict[str, str]] = []
    label_error = ""
    if entity_ids:
        try:
            labels = list(context.pipeline.kg.labels(entity_ids))
        except Exception as exc:
            label_error = f"{type(exc).__name__}:{exc}"
    label_by_id = {
        str(item.get("id", "")): str(item.get("label", ""))
        for item in labels if isinstance(item, dict) and str(item.get("id", ""))
    }
    diagnostics["selected_label_queries"] = int(bool(entity_ids))
    if label_error:
        diagnostics["selected_label_error"] = label_error
    return {
        "answer_ids": answer_ids,
        "answers": [{"id": value, "label": label_by_id.get(value, value)} for value in answer_ids],
        "selected_graph": None,
        "diagnostics": diagnostics,
    }


__all__ = ["eligible", "retrieve"]
