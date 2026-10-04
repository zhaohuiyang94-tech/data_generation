"""Task-general temporal/numeric rules; no test questions, dates or answer examples."""

TEMPORAL_NUMERIC_POLICY = """
TEMPORAL AND NUMERIC SEMANTICS (WebQSP and CWQ)
Preserve each explicit date/year, comparator, unit, time reference, ordinal,
cardinality and superlative. Keep them attached to the entity or intermediate
record that owns the attribute. A path may return an entity while its roster,
tenure, education or performance record owns the filtered date/number.
Before/after are boundary comparisons, not ARGMIN/ARGMAX by themselves. Distinguish
tenure start, tenure end, birth, death, publication, release, election, appointment
and event dates. An election year is not automatically an office-start year.
During/in a period requires interval overlap, not merely a start date inside it.
Current membership requires a tenure covering NOW, including open-ended intervals;
latest historic tenure is not automatically current. A named event's interval
must come from KG evidence; never invent calendar dates from memory.
First/earliest/latest/last select extrema of the relevant date or attribute; keep
the attribute available in the path. A reference such as a prior office/event may
require a separate connected path to obtain its cutoff. Do not fabricate a literal
cutoff, an ordering column or a tie-breaker. A textual first-name field is not an
earliest-record request. Numbers inside titles/entity names are not thresholds.
How-many can request a stored numerical attribute or COUNT of distinct results:
choose the requested semantics, not COUNT just because a number appears. A fixed
number of named results is an enumeration, not a scalar count. Greater/less use
strict comparisons; at-least/at-most use inclusive comparisons. Preserve negatives,
decimals, units and range bounds. Rank numerically/datetimely, never lexically.
For population/size/amount extrema return the entity associated with the extreme,
unless the question asks for the numerical value itself. Scope extrema/counts to
the constrained candidate set; do not rank globally and filter afterward.
Use only operators and variables supported by the supplied contract. Do not invent
an OFFSET, arithmetic, unit conversion, date projection or aggregate operator when
the backend cannot represent it. Keep unsupported requirements visible in the
decomposition rather than silently changing the answer to fit a familiar template.
Date spelling may normalize when unambiguous; ambiguous date ordering must not be
guessed. Preserve the source temporal requirement even when emitting ISO values.
""".strip()

OPERATOR_BINDING_POLICY = """
OPERATOR CALIBRATION
Only infer operations required by the original question or explicit path goal.
Do not emit a boilerplate NO_EQUAL instead of required temporal/numeric operators.
Bind the operation to the global Vn that owns its attribute, often a CVT rather
than answer_var. If the compared date/number is already a graph variable, use that
variable and an empty attribute_relation_label. Otherwise name the owner's real
attribute relation using the supplied human-readable Freebase components.
Keep COUNT's counted variable distinct from a stored-number attribute request.
ARGMIN/ARGMAX select the requested entity/event, using a relevant sorting attribute.
TC represents a year/current-tenure restriction; boundary comparators represent
before/after of the explicitly referenced date. Retain event/office/type filters
through the composed graph. Do not guess dates for named historical intervals.
Output only the operators object with the exact eight required fields per operator.
Return an empty array if a needed operation cannot bind to any supplied variable;
never hallucinate variables, relation IDs, date values or a different answer target.
""".strip()

SELECTOR_CONSTRAINT_POLICY = """
Verify candidates in this order: requested answer role/type, named entities and
relationships, explicit type/intersection/exclusion restrictions, time/numeric
operations, then supporting rule_score and answer_count. A relation/constraint
mention in the decomposition is not proof that a candidate graph implemented it.
Before/after require the correct boundary attribute/operator, not automatic extrema.
During/current require interval coverage. First/latest require relevant sorting;
first-name strings and title digits are not rankings. Counting requires the right
counted set; stored numeric attributes and fixed-number enumerations are different.
Numeric extrema return the associated entity unless the question requests the value.
Reject answer-type mismatch even if a preview label overlaps a question word.
Do not prefer a smaller set solely for its size or invent a new graph/answer.
If all candidates are imperfect, select the closest graph actually supported by
the question, with an honest reason code; do not pretend missing constraints exist.
""".strip()

SEMANTIC_TARGET_POLICY = """
Generate only the trained anchors/semantic_paths JSON, never answers or operators.
Read every complete path and constraint item jointly with the original question.
The final path output must have the requested answer role; do not stop at an
intermediate bridge or substitute a familiar unrelated property. An office/title
is not a profession, an election date is not automatically an appointment date,
an event is not its type/season, and an artwork is not its genre/style.
Use realistic human-readable Freebase relation components and preserve direction.
Keep the real entity names and the owner of each attribute. Do not change a topic
to another source merely to reuse a known path. Explicit type/conjunction/entity
constraints must be represented by connected paths to the same variable; don't
drop them. Leave temporal/count/extrema operations to Operator while retaining the
required attribute owner, dependencies and explicit conditions in path goals.
Do not fabricate a date/number, KB ID, generic wildcard relation or disconnected
extra path to make a sparse query appear complete. Preserve the provided contract.
""".strip()


import re
def question_requirements(question):
    text = question.casefold()
    kind = ("country entity" if re.search(r"\bcountry|\bcountries", text) else
            "year" if "what year" in text or "which year" in text else
            "location" if re.search(r"\bwhere\b", text) else
            "number: distinguish stored attribute from distinct count" if "how many" in text else
            "person or named entity" if re.search(r"\bwho\b", text) else "infer from question")
    terms = [m.group() for m in re.finditer(r"\b(?:before|after|during|between|now|current|first|last|latest|earliest|largest|smallest|most|least|official(?:ly)?|college|university)\b|\b\d+(?:[./-]\d+)*\b", text)]
    return {"answer_kind_hint": kind, "explicit_terms": terms,
            "hint_scope": "Lexical hints only; first-name fields and entity-title digits are not ranking/numeric filters."}
