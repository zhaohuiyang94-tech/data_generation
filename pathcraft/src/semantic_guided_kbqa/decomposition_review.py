from __future__ import annotations
from copy import deepcopy
import re
import unicodedata
from typing import Any
from .clients import ChatClient
from .contracts import ContractError, parse_json_object
from .prompt_principles import TEMPORAL_NUMERIC_POLICY, question_requirements
from .train_fewshots import decomposition_examples
from .glm_webqsp_fewshots import glm_webqsp_examples
from .cwq_fewshots import cwq_examples



DECOMPOSITION_POLICY = """
Judge or repair path-level decomposition of the original KG question, not the
original question itself. The question is authoritative. One array item is one
complete entity-centric path; multiple connected questions in the SAME string
are valid. Read all items jointly with attached constraints. Awkward grammar,
synonyms and useful source-domain annotations are not defects by themselves.
Read What is RELATION of TYPE ENTITY as a relation query. Keep bridge objects
and dependencies. Inspect the terminal requested result, not just the first hop.
Only concrete wrong targets/entities/directions, missing terminal hops or required
constraints, disconnected references and unsupported additions justify rewriting.
Retain correct paths and entity spellings, fix wrong prefixes when necessary,
and attach constraints to the object that owns them. Do not introduce speculative
relations merely to hide a missing subtype. Do not guess answers, entity identities,
dates or more-specific intentions for ambiguous questions. Demonyms in country-language questions can refer to a country's inhabitants, and faith adjectives can refer to adherents; do not invent a fictitious person or demand an unnecessary population node. No Gold answers or previous predicted answers are supplied. Optional entity_context contains only names/types/event dates from existing topic linking and read-only KG evidence. A linked entity can be a constraint, not the main subject. Use its real type to disambiguate the source and repair a mismatched annotation; do not preserve an irrelevant region/person prefix. Use supplied event dates for time constraints, never guess dates. If no certain defect exists, preserve
all original items. Changes should be minimal and remain in the trained path format.
""".strip()

DECOMPOSITION_POLICY += """
For a vague standalone question about what a person did, a profession query is a
reasonable interpretation unless the question specifies an event/action/time.
Do not replace it with a verbatim vague question. For where/from location questions,
do not impose a country-only answer unless the question explicitly requires country
or that constraint was already present. A result count is not the number of hops.
Never duplicate a path/constraint or invent repeated rank queries to manufacture a
requested cardinality. Keep a requested sorting measure explicit and supported.
"""


def rewrite_preservation_issues(question, original, final):
    q = question.casefold().strip().rstrip('?')
    before = ' '.join(original).casefold()
    after = ' '.join(final).casefold()
    issues = []
    normalized = [re.sub(r'\s+', ' ', x).strip().casefold() for x in final]
    if len(normalized) != len(set(normalized)):
        issues.append('Duplicate path/constraint items are not a semantic repair.')
    if (re.fullmatch(r'what (?:did|does) .+ do', q) and 'profession' in before
            and normalized != [re.sub(r'\s+', ' ', x).strip().casefold() for x in original]):
        issues.append('An unconstrained vague person-action question permits the original profession interpretation.')
    if not re.search(r'\bcountry\b|\bcountries\b', q) and 'country' not in before and re.search(r'\bcountry\b|\bcountries\b', after):
        issues.append('The repair adds a country-only answer level not requested or already supplied.')
    return issues

_REVIEW_OUTPUT_RULE = """
Return exactly ONE JSON object with keys is_reasonable (boolean), issues (array
of descriptive strings), reason (string). Reasonable means issues=[]. Every defect
must quote the actual missing/incorrect relation or requirement; a bare code is
insufficient. Do not copy question/decomposition/review_issues/review_reason or
output_error to output keys. No issue objects, prose, fences or second JSON object.
""".strip()

DECOMPOSITION_REVIEW_INSTRUCTION = "\n\n".join([DECOMPOSITION_POLICY,TEMPORAL_NUMERIC_POLICY,decomposition_examples('review'),
"Judge the ACTUAL current request using the lexical question_requirements as hints, not extra facts. Compare requested terminal type/relation to the full current path. The positive train examples are not exact-match templates. A valid equivalent country-language interpretation remains valid. Do not rewrite during judgement.",_REVIEW_OUTPUT_RULE])
DECOMPOSITION_CONFIRM_INSTRUCTION = "\n\n".join([DECOMPOSITION_POLICY,TEMPORAL_NUMERIC_POLICY,
"Challenge the first review against the whole current list. Preserve a valid bridge, annotation or constraint already expressed elsewhere. Do not merely echo the first review or output a proposed rewrite. Give a concrete comparison of the requested target and current terminal target; do not invent a missing constraint absent from the question.",_REVIEW_OUTPUT_RULE])
DECOMPOSITION_REWRITE_INSTRUCTION = "\n\n".join([DECOMPOSITION_POLICY,TEMPORAL_NUMERIC_POLICY,decomposition_examples('rewrite'),
"Repair only confirmed current defects. Keep connected hop questions in one item and restore missing constraints as attached statements. Use the train examples for structure, not their entity names or facts. Return exactly ONE JSON object with EXACTLY ONE key decomposition, a non-empty array of non-empty strings. Never copy input metadata or output facts, guessed dates, KB IDs, STEP placeholders, explanations or a second object. Stop after the closing brace."])
DECOMPOSITION_RECHECK_INSTRUCTION = "\n\n".join([DECOMPOSITION_POLICY,TEMPORAL_NUMERIC_POLICY,
"This is a POST-REPAIR verification of ONLY the current candidate. There are no examples in this verifier. Independently identify the question's requested terminal object, then read the current final main question and attached statements. A path that now explicitly asks for the requested object is not missing that same hop. Never repeat an old issue by copying a phrase. Constraints already supplied are satisfied. Religious-adherent questions may query the corresponding faith's beliefs, not a population percentage. Reject genuinely wrong roots, disconnected steps, wrong directions, missing explicit constraints or unsupported facts. If the current path is a faithful coherent interpretation, accept it. Do not demand exact training-example wording.",_REVIEW_OUTPUT_RULE])

GLM_WEBQSP_TASK_CONTEXT = """
WEBQSP TASK
Review English path-level decompositions for WebQSP Freebase KGQA. The original
question is authoritative; retrieve its full KG answer set, not an answer from memory.
Decomposition is a natural-language query plan: Semantic generates relation paths,
Grounding binds entities/relations, Compose merges shared nodes and the answer variable,
Operator adds conditions, and SPARQL executes the graph. Low F1, label or execution
errors alone do not prove that this plan is wrong.

PATH FORMAT
One array item is one complete entity-centric path and may contain multiple connected
questions. Separate constraint statements are valid when their owner is clear. Read
all items jointly. Keep correct anchors and bridge/CVT records (performance, roster,
education, tenure); each next hop must start from its stated source. Actor-to-film
phrasing may first retrieve performances. Never add a hop by repeating a query about
the original subject. Grammar, synonyms and useful domain annotations are not defects.

REPAIR RULES
Rewrite only certain wrong entities/directions/terminal relations, missing required
hops or conditions, disconnected dependencies or unsupported additions. Repair minimally;
preserve correct paths. Distinguish actor/character, killer/weapon, school/degree/field,
work/genre/material, birthplace/country, and entity/date. Attach all film/actor, team/member,
type and time conditions to the same intermediate record that owns them.
Keep explicit country/college/type restrictions; do not invent country-only or official-
language limits. Preserve valid book-edition paths. Bare other does not imply a guessed
excluded work. Vague standalone person-action questions may retain profession; explicit
study/design/play/produce/event wording requires its actual target. Demonyms and faith
adjectives may refer to a country or religion, not a fictitious person/population node.
Do not guess entity identities, title details, answers, dates or unsupported relations.
Inputs are only the original question and original decomposition; the repair call
also receives issues from the first review. No downstream results or entity context
are available. Do not infer facts from absent pipeline information.

TIME AND NUMBERS
Dates used to filter jobs/teams must not replace the requested job/team answer.
Before/after compare the correct boundary, not extrema. During requires interval overlap;
current requires tenure covering NOW, including open-ended records. Latest historic
tenure is not current.
First/latest need the relevant sorting attribute and constrained candidate set. Keep
reference tenure/event paths for relative cutoffs; never guess their calendar dates.
Distinguish election, appointment, start/end, birth/death and release dates; preserve
requested year/date granularity. First name and title digits are not ranking filters.
Distinguish stored numeric attributes, distinct COUNT and fixed-number enumeration.
Preserve comparators, inclusive/strict bounds, dates, units and numbers from the question;
keep unsupported requirements visible instead of inventing operators or duplicate paths.
For a first/latest request over reified candidate records, determine whether the
requested answer is the record/entity or the scalar used to order it. When the
record/entity is requested, keep it as the terminal answer and describe its date or
numeric property only as the ordering attribute. Bind subtype, event-category and
other filters to that same candidate record. State the owner and semantic role of
the ordering property; do not depend on a memorized wording template.
If no concrete semantic defect is certain, preserve the whole input.
""".strip()

GLM_DECOMPOSITION_STYLE_POLICY = """
TRAINED DECOMPOSITION STYLE (general patterns, not facts about this question)
Use "What/Who is the RELATION of SOURCE_TYPE ENTITY?" followed by questions about
the intermediate record. SOURCE_TYPE is the schema role of that relation, not a
generic biological class: do not use person for every human-related relation.
Preserve familiar property wording/plurals; do not invent prose-like relation names.
Reusable trained phrases (fill names only from the current input):
- Employment/office: government positions held of politician NAME -> office position
  or title of the government position held. Basic title denotes an office category;
  do not substitute it for a requested specific office. Dates use from/to of the record.
- Sports membership: teams of pro athlete NAME -> team of the sports team roster;
  membership dates belong to that roster, not to the team entity.
- Education: education of person NAME -> institution / degree / major field of study
  of the education; choose the attribute actually requested.
- Film roles: film of actor NAME -> actor / character / film of the performance.
- TV roles: regular cast of tv program NAME -> actor / character of the regular tv
  appearance. An actor condition and character answer refer to the same appearance.
These are format patterns, not an exhaustive whitelist or reasons to alter correct paths.

CONSTRAINT STYLE
Use "VALUE is the PROPERTY of the RECORD" for a bound entity/type condition.
For relative time, name both owning records and their actual date attributes, e.g.
the end/from date of the answer tenure versus the from date of the reference tenure.
Identify which tenure/event is the reference. Write the time comparison as a separate
constraint statement, not another main query or an invented reference-title path.
Before a tenure compares its start; after a completed tenure uses its end. An adjacent
successor may start on the previous end date; a literal before/after date is strict.
If a single next/previous holder is requested, select the earliest/latest qualifying
tenure; do not add extrema to an all-positions question. Never invent a calendar cutoff.
Never write "RECORD is before ENTITY" as a standalone relation query or turn
before/during/first into a relation name. Keep the main path
ending at the requested object; auxiliary filter/date paths are not the answer.
Do not output NAME/RECORD placeholders. Internally check each relation/source-type
pair, bridge dependency, terminal target and explicit condition before responding.
For numeric IDs/counts/measures name the exact scalar property and its owner, preserve
the supplied threshold and unit, and distinguish numeric comparison from string order.
""".strip()

DECOMPOSITION_TWO_CALL_REVIEW_INSTRUCTION = "\n\n".join([
    GLM_WEBQSP_TASK_CONTEXT, GLM_DECOMPOSITION_STYLE_POLICY, glm_webqsp_examples('review'),
    'FIRST call: compare the requested terminal answer and conditions against the full plan. Reasonable means no rewrite. Otherwise identify only concrete mismatches, naming the current target and the required target or missing condition and its owner. Do not output a repaired decomposition.',
    'Return only {"is_reasonable": boolean, "issues": [descriptive strings], "reason": string}. If reasonable, issues=[]. No extra keys, issue objects, fences or prose.',
])
DECOMPOSITION_TWO_CALL_REWRITE_INSTRUCTION = "\n\n".join([
    GLM_WEBQSP_TASK_CONTEXT, GLM_DECOMPOSITION_STYLE_POLICY, glm_webqsp_examples('rewrite'),
    'SECOND and FINAL call: check the supplied issues against the question, then repair only supported defects. Internally verify source -> bridge -> requested terminal and all conditions on their correct owner. Preserve correct items; do not repeat the original subject question or copy an example blindly. There is no further model review.',
    'Return only {"decomposition": [non-empty English path strings]}. Preserve explicit dates/numbers; no guessed facts, KB IDs, SQL, operator JSON, STEP placeholders, metadata, explanations or fences.',
])

CWQ_TASK_CONTEXT = """
CWQ TASK
Review English path-level decompositions for ComplexWebQuestions over the Freebase KG.
The original question is authoritative. The decomposition is a natural-language query
plan consumed by Semantic path generation, Grounding, Compose, Operator and SPARQL
execution. A corrected decomposition must preserve one complete entity-centric path,
the requested terminal answer, every explicit date/numeric/type condition, and all
shared intermediate variables needed by joins.

CWQ PATH RULES
Keep education, government_position_held, film.performance, sports_team_roster,
ownership, relationship and event records as bridge/CVT nodes. Conditions such as
mascot, owner, character, actor, newspaper, currency, population, enrollment and
award must be attached to the same candidate entity or record as the main answer.
For intersections, state that the conditions refer to the same film, team, country,
institution, person or tenure record. For ARGMAX/ARGMIN, name the candidate set and
the scalar property being compared. For dates, preserve the requested before/after,
during or overlap meaning and keep the date on the record that owns it.
Decomposition supplies entity/relation paths; Operator separately reads the original
question and adds extrema, time, numeric, count and exclusion operations. Do not demand
that a decomposition restate an operator when its path already exposes the correct
candidate entity or owning CVT. Rewrite only when the candidate set, owner, bridge,
shared entity or requested terminal itself is absent or wrong.

Do not return an intermediate entity when the question asks for its next-hop answer.
Do not replace an answer with its type, label, source person, event or bridge record.
Do not invent a KB ID, answer, date, country layer or UNION branch. Do not repeat a
valid path merely to express a constraint. If no concrete defect is certain, preserve
the complete input decomposition.
""".strip()

CWQ_ENTITY_CONTEXT_POLICY = """
GOLD TOPIC ENTITY CONTEXT
entity_context contains only pre-linked Gold topic/constraint entity IDs, names, KG
types and event dates. It never contains Gold answers. Use an entity only when its name,
ID or semantic role corresponds to an explicit mention or constraint in the original
question. The source annotation can contain inherited or ambiguous candidates, so do
not force every supplied entity into the decomposition and never add an entity that the
question does not require. For each explicit question mention that does match a supplied
entity, preserve its identity as the main anchor or attach it to the candidate/bridge it
constrains; do not silently replace it with a broader, narrower or guessed entity. Treat
spelling variants as the same entity when the supplied ID agrees. Entity context
validates anchors; it is not permission to shorten or replace an already executable
relation chain.

Before rejecting a decomposition for returning an intermediate object, trace every hop
and connected question to the end of the item. A path that mentions a CVT and then asks
for the requested terminal after that CVT does not return the CVT. Preserve such trained
chains even when a shorter fluent paraphrase seems possible.
""".strip()

CWQ_DECOMPOSITION_STYLE = """
CWQ STYLE
Use short connected questions or statements in the KaeDe path-level format. One array
item may contain multiple connected hops. A separate item is appropriate for a clear
constraint, but it must identify the same candidate set or bridge record. Prefer
"Which X has Y? Which attribute does that same X have?" over disconnected prose.
The corrected path must be usable by the existing CWQ Semantic, Compose, Operator and
execution models; do not output SPARQL, relation IDs, graph JSON or operator JSON.

EXECUTABLE PATH OWNERSHIP
Every array item must be groundable from an explicit named anchor in that item or be
an established constraint statement whose owner is unambiguous in another item.
Do not create a new pronoun-only item such as "which of those" for a filter, extrema,
or terminal hop. When a condition has no independent named anchor, keep it attached
to the same entity-centric path and name the bridge/record that owns it. Split paths
only when each side has a real anchor and Compose can unify their terminal candidate.
Preserve terse trained KaeDe relation wording when it is semantically complete; do
not replace it with fluent prose that the downstream Semantic model was not trained on.

REWRITE DIALECT
A rewrite must remain in the same terse KaeDe dialect as the input: short
"ENTITY is the RELATION of the TYPE" statements and connected "What/Who is the
RELATION of the TYPE?" questions. Do not output explanatory graph prose such as
"lists ... in its field", "follow this route", "matching performance", or
"which of those". Such wording can be semantically clear but is not a trained
relation label. Preserve every correct input item verbatim and replace only the
incorrect anchor, relation phrase, missing bridge hop, terminal question or owner.

Use the exact supplied entity_context name for an entity mention, including spelling
and punctuation. Do not substitute a related person, parent, league, region or alias
whose ID differs. When a relation is reified, retain each trained CVT hop rather than
collapsing it into a fluent plural relation. For an intersection, keep independently
anchored constraints as separate KaeDe items and let Compose unify their candidate
terminal; prose saying "the same entity" cannot replace a groundable path.
""".strip()

CWQ_TEMPORAL_NUMERIC_BOUNDARY_POLICY = """
CWQ OPERATOR BOUNDARY FOR TIME AND NUMBERS
First identify the requested answer variable independently of every value used to
filter or rank it. A date, year, percentage, count or population value is not the
answer merely because the question compares it. Keep the candidate entity/event/person
as the terminal answer; date and numeric properties belong to that candidate or its
owning CVT and serve only as equality filters, strict/inclusive bounds, or ordering
keys.

Operator reads the original question after decomposition. Therefore a decomposition
that already reaches the correct candidate set and owner is not defective merely
because it does not repeat a numeric/date literal or ARGMAX/ARGMIN phrase. Preserve
such a trained path. Rewrite only when the current path has the wrong anchor, relation,
owner, candidate set or terminal projection, or when making the property owner explicit
is necessary to prevent the constraint from attaching to the wrong node.

Operator instructions are not KG path steps. Remove meta-questions such as asking which
result "satisfies the time condition", symbolic NOW checks, or generic filter/result
sentences when they have been appended as if they were relations. Keep the real
entity-to-candidate/CVT path and its type constraint; Operator will apply current,
before/after, extrema and numeric semantics from the original question. For a current
office-holder question, the decomposition should reach the office holder through the
government-position record and bind its requested title; it must not query a generic
result node for NOW.

If an otherwise complete trained chain is followed only by such an operator meta-clause,
the repair is deletion-only: remove that clause and preserve every preceding path item
and its terse relation wording verbatim. Do not paraphrase the chain, change its bridge
direction, split it into pronoun-only items, or invent a new jurisdiction/tenure path.
If a bound office title/type condition was fused after the terminal question, keep the
main entity-centric chain ending at the requested office holder and move only that bound
condition into its own array item, naming the same government-position record. Do not
append the title after the answer hop in a way that turns the title or an unconstrained
office-holder set into the terminal answer.

When a rewrite is necessary, preserve the exact signed decimal/date and comparator,
name the property's owner, and explicitly return the candidate rather than the scalar.
Examples of semantic roles: an event is filtered by its end date; a film is ordered by
initial release date; a championship is ordered by start date; a location is filtered
by its population record; a country is filtered by its dated percentage record; a
current office holder is selected through a tenure interval covering NOW. Before/prior
and after/later remain strict unless the question says inclusive. Do not interpret a
year or number inside an entity title as a constraint.

When the existing candidate path is correct and only a filter, ordering role or return
clarification is missing, copy that path text verbatim and only append the missing
clause. Do not make it more fluent. In particular, do not change spacing, plurality,
direction or source-type wording inside a trained relation label; even a cosmetically
equivalent paraphrase can ground to a different Freebase relation.

Before accepting a rewrite, perform an answer-projection check: removing every filter
or sorting clause from the rewritten wording must still leave a path whose terminal is
the object requested by the original question. If it instead leaves a date, rate,
population, count or CVT record, repair the projection before responding.
A rewrite has exactly one terminal target. Once it says to return a candidate entity,
event or person, it must not append a second request to return that candidate's date,
rate, count, population or ordering value. Those scalars may be named only in their
filter/sort role. Follow this rule even when a surface question uses "when" but the
trained candidate path and supplied example establish an event/entity terminal.
""".strip()

CWQ_ENTITY_CONTEXT_REVIEW_INSTRUCTION = "\n\n".join([
    CWQ_TASK_CONTEXT, CWQ_ENTITY_CONTEXT_POLICY, DECOMPOSITION_POLICY,
    TEMPORAL_NUMERIC_POLICY, CWQ_TEMPORAL_NUMERIC_BOUNDARY_POLICY,
    CWQ_DECOMPOSITION_STYLE,
    cwq_examples("review", include_entity_context=True),
    "FIRST call: compare the question's requested terminal object, bridges and constraints with the whole current list. Identify only concrete defects. Do not output a repair.",
    _REVIEW_OUTPUT_RULE,
])

CWQ_ENTITY_CONTEXT_REWRITE_INSTRUCTION = "\n\n".join([
    CWQ_TASK_CONTEXT, CWQ_ENTITY_CONTEXT_POLICY, DECOMPOSITION_POLICY,
    TEMPORAL_NUMERIC_POLICY, CWQ_TEMPORAL_NUMERIC_BOUNDARY_POLICY,
    CWQ_DECOMPOSITION_STYLE,
    cwq_examples("rewrite", include_entity_context=True),
    "SECOND and FINAL call: check the supplied issues against the original question and entity_context. Repair only supported defects. Preserve correct KaeDe paths and return only {\"decomposition\": [non-empty English path strings]}; no metadata, facts, KB IDs, SPARQL, operators, explanations or fences.",
])

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "is_reasonable": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["is_reasonable", "issues", "reason"],
    "additionalProperties": False,
}
REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "decomposition": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        }
    },
    "required": ["decomposition"],
    "additionalProperties": False,
}


def _validate_review(raw: Any) -> dict[str, Any]:
    value = parse_json_object(raw)
    if set(value) not in (
        {"is_reasonable", "issues", "reason"},
        {"is_reasonable", "issues"},
    ):
        raise ContractError("review must contain exactly is_reasonable, issues and reason")
    if not isinstance(value["is_reasonable"], bool):
        raise ContractError("is_reasonable must be a JSON boolean")
    issues = value["issues"]
    if not isinstance(issues, list) or not all(isinstance(item, str) for item in issues):
        raise ContractError("issues must be an array of strings")
    if value["is_reasonable"] and issues:
        raise ContractError("a reasonable decomposition must have no unresolved issues")
    if not value["is_reasonable"] and not any(item.strip() for item in issues):
        raise ContractError("an unreasonable decomposition must identify concrete issues")
    reason = value.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ContractError("reason must be a string")
    if reason is None:
        # GLM-5.3 low occasionally omits the explanatory field while still
        # returning a valid boolean decision and concrete issues.  Recover the
        # redundant explanation locally instead of discarding the review or
        # spending another model call.
        reason = "; ".join(item.strip() for item in issues if item.strip())
        if value["is_reasonable"]:
            reason = "No unresolved decomposition issues."
        value["reason"] = reason
    return value


def _validate_rewrite(raw: Any) -> dict[str, Any]:
    value = parse_json_object(raw)
    if set(value) != {"decomposition"}:
        raise ContractError("rewrite must contain only decomposition")
    paths = value["decomposition"]
    if not isinstance(paths, list) or not paths:
        raise ContractError("decomposition must be a non-empty array")
    if not all(isinstance(item, str) and item.strip() for item in paths):
        raise ContractError("decomposition items must be non-empty strings")
    return {"decomposition": [item.strip() for item in paths]}


def _profile_bool(value: Any) -> bool:
    return value is True or str(value).strip().casefold() in {"1", "true", "yes", "on"}


def _name_key(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    accentless = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", "", accentless)


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def _canonicalize_context_entity_names(question, decomposition, entity_context):
    """Replace a near-exact question spelling with its linked canonical name."""
    words = list(re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)?", question))
    output = list(decomposition)
    changes = []
    for entity in entity_context or []:
        canonical = str(entity.get("name", "")).strip() if isinstance(entity, dict) else ""
        canonical_key = _name_key(canonical)
        token_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", canonical))
        if len(canonical_key) < 4 or token_count < 1 or len(words) < token_count:
            continue
        best = None
        for start in range(len(words) - token_count + 1):
            end = start + token_count - 1
            surface = question[words[start].start():words[end].end()]
            surface_key = _name_key(surface)
            distance = _edit_distance(surface_key, canonical_key)
            threshold = max(1, int(len(canonical_key) * 0.08))
            if distance <= threshold and (best is None or distance < best[0]):
                best = (distance, surface)
        if best is None:
            continue
        surface = best[1]
        changed = False
        for index, item in enumerate(output):
            replaced = re.sub(re.escape(surface), canonical, item, flags=re.I)
            if replaced != item:
                output[index] = replaced
                changed = True
        if changed and surface != canonical:
            changes.append({"from": surface, "to": canonical})
    return output, changes


def _merge_anchorless_rewrite_items(original, rewritten, entity_context):
    """Merge only newly generated, non-question constraints with no real anchor."""
    original_keys = {" ".join(str(item).casefold().split()) for item in original}
    anchor_names = [
        str(entity.get("name", "")).strip()
        for entity in entity_context or []
        if isinstance(entity, dict) and str(entity.get("name", "")).strip()
    ]
    merged = []
    changes = []
    for item in rewritten:
        normalized = " ".join(str(item).casefold().split())
        is_original = normalized in original_keys
        has_anchor = any(
            _name_key(name) and _name_key(name) in _name_key(item)
            for name in anchor_names
        )
        starts_query = bool(re.match(r"\s*(?:what|who|which|where|when|how)\b", item, re.I))
        should_merge = (
            bool(merged)
            and not is_original
            and not has_anchor
            and (
                not starts_query
                or bool(re.match(r"\s*(?:return|select|the\s+(?:earliest|latest))\b", item, re.I))
            )
        )
        if should_merge:
            target_index = next(
                (
                    index
                    for index in range(len(merged) - 1, -1, -1)
                    if re.match(
                        r"\s*(?:what|who|which|where|when|how)\b",
                        merged[index],
                        re.I,
                    )
                ),
                len(merged) - 1,
            )
            merged[target_index] = (
                f"{merged[target_index].rstrip()} {str(item).strip()}"
            )
            changes.append(str(item))
        else:
            merged.append(str(item).strip())
    return merged, changes


class DecompositionReviewer:
    """Review with either a strict two-call budget or the original verified workflow."""

    def __init__(self, model: ChatClient, *, max_rewrites: int = 2, output_attempts: int = 2,
                 confirm_before_rewrite: bool = False, workflow: str = "verified",
                 prompt_family: str = "webqsp",
                 prompt_profile: dict[str, str] | None = None):
        if workflow not in {"verified", "two_call"}:
            raise ValueError("decomposition reviewer workflow must be verified or two_call")
        if prompt_family not in {"webqsp", "cwq"}:
            raise ValueError("decomposition reviewer prompt_family must be webqsp or cwq")
        self.model = model
        self.workflow = workflow
        self.prompt_family = prompt_family
        self.prompt_profile = dict(prompt_profile or {})
        self.max_rewrites = 1 if workflow == "two_call" else max(0, int(max_rewrites))
        self.output_attempts = 1 if workflow == "two_call" else max(1, int(output_attempts))
        self.confirm_before_rewrite = False if workflow == "two_call" else bool(confirm_before_rewrite)
        if workflow == "two_call" and getattr(getattr(model, "config", None), "retries", 0):
            raise ValueError("two_call reviewer requires HTTP retries=0")

    def _profiled_instruction(self, instruction: str, action: str) -> str:
        suffix = str(self.prompt_profile.get(f"{action}_append", "")).strip()
        return "\n\n".join([instruction, suffix]) if suffix else instruction

    def _generate(self, *, action, instruction, payload, schema, validator, events):
        last_error = ""
        for attempt in range(1, self.output_attempts + 1):
            request = deepcopy(payload)
            if last_error:
                request["output_error"] = last_error
            event = {"action": action, "attempt": attempt, "input": request}
            events.append(event)
            try:
                outputs = self.model.generate_json(
                    instruction=instruction, payload=request, schema=deepcopy(schema), count=1
                )
                raw = outputs[0] if outputs else {}
                event["model_output"] = deepcopy(raw)
                value = validator(raw)
                event["status"] = "valid"
                return value
            except Exception as exc:
                last_error = str(exc)
                event.update({"status": "error", "error": last_error})
        raise RuntimeError(f"decomposition {action} failed: {last_error}")

    def run(self, question: str, decomposition: list[str], *, entity_context=None) -> tuple[list[str] | None, dict[str, Any]]:
        if self.workflow == "two_call":
            return self._run_two_call(question, decomposition, entity_context=entity_context)
        current = list(decomposition)
        events: list[dict[str, Any]] = []
        trace: dict[str, Any] = {"original_decomposition": list(current), "events": events}
        rewrites = 0
        context = {"entity_context": entity_context} if entity_context else {}
        context["question_requirements"] = question_requirements(question)
        try:
            while True:
                review = self._generate(
                    action="review",
                    instruction=self._profiled_instruction(
                        DECOMPOSITION_RECHECK_INSTRUCTION if rewrites else DECOMPOSITION_REVIEW_INSTRUCTION,
                        "review",
                    ),
                    payload={"question": question, "decomposition": current, **context},
                    schema=REVIEW_SCHEMA,
                    validator=_validate_review,
                    events=events,
                )
                if review["is_reasonable"]:
                    trace.update({
                        "status": "rewritten" if rewrites else "accepted",
                        "rewrite_count": rewrites,
                        "final_decomposition": list(current),
                    })
                    return current, trace
                if rewrites == 0 and self.confirm_before_rewrite and self.max_rewrites > 0:
                    review = self._generate(
                        action="confirm_defect", instruction=self._profiled_instruction(
                            DECOMPOSITION_CONFIRM_INSTRUCTION,
                            "review",
                        ),
                        payload={"question": question, "decomposition": current,
                                 "review_issues": review["issues"], "review_reason": review["reason"], **context},
                        schema=REVIEW_SCHEMA, validator=_validate_review, events=events,
                    )
                    if review["is_reasonable"]:
                        trace.update({"status": "accepted_after_confirmation", "rewrite_count": 0,
                                      "final_decomposition": list(current)})
                        return current, trace
                if rewrites >= self.max_rewrites:
                    trace.update({"status": "rejected", "rewrite_count": rewrites,
                                  "last_decomposition": list(current)})
                    return None, trace
                rewritten = self._generate(
                    action="rewrite",
                    instruction=self._profiled_instruction(
                        DECOMPOSITION_REWRITE_INSTRUCTION,
                        "rewrite",
                    ),
                    payload={"question": question, "decomposition": current,
                             "issues": review["issues"], "review_reason": review["reason"], **context},
                    schema=REWRITE_SCHEMA,
                    validator=_validate_rewrite,
                    events=events,
                )
                current = rewritten["decomposition"]
                guard_issues = rewrite_preservation_issues(question, decomposition, current)
                if guard_issues:
                    events.append({"action": "rewrite_preservation_guard", "status": "rejected", "issues": guard_issues})
                    trace.update({"status": "rejected_guard", "rewrite_count": rewrites + 1,
                                  "last_decomposition": list(current), "guard_issues": guard_issues})
                    return None, trace
                rewrites += 1
        except Exception as exc:
            trace.update({"status": "error", "error": str(exc), "rewrite_count": rewrites})
            return None, trace

    def _run_two_call(self, question, decomposition, *, entity_context=None):
        events = []
        original_decomposition = list(decomposition)
        current_decomposition = list(decomposition)
        if _profile_bool(self.prompt_profile.get("canonicalize_context_names")):
            current_decomposition, name_changes = _canonicalize_context_entity_names(
                question,
                current_decomposition,
                entity_context,
            )
            if name_changes:
                events.append({
                    "action": "context_name_canonicalization",
                    "status": "applied",
                    "changes": name_changes,
                })
        trace = {"workflow": "two_call", "original_decomposition": original_decomposition,
                 "events": events, "rewrite_count": 0, "model_call_count": 0,
                 "model": getattr(getattr(self.model, "config", None), "model", ""),
                 "prompt_family": self.prompt_family,
                 "cwq_entity_context_prompt": self.prompt_family == "cwq",
                 "prompt_profile_version": str(self.prompt_profile.get("version", "")),
                 "prompt_profile_source": str(self.prompt_profile.get("source", ""))}
        use_cwq_entity_context = self.prompt_family == "cwq"
        context = (
            {"entity_context": deepcopy(entity_context)}
            if use_cwq_entity_context and entity_context else {}
        )
        try:
            if self.prompt_family == "cwq":
                review_instruction = CWQ_ENTITY_CONTEXT_REVIEW_INSTRUCTION
                rewrite_instruction = CWQ_ENTITY_CONTEXT_REWRITE_INSTRUCTION
            else:
                review_instruction = DECOMPOSITION_TWO_CALL_REVIEW_INSTRUCTION
                rewrite_instruction = DECOMPOSITION_TWO_CALL_REWRITE_INSTRUCTION
            review_instruction = self._profiled_instruction(
                review_instruction,
                "review",
            )
            rewrite_instruction = self._profiled_instruction(
                rewrite_instruction,
                "rewrite",
            )
            review = self._generate(
                action="review", instruction=review_instruction,
                payload={"question": question, "decomposition": list(current_decomposition), **context},
                schema=REVIEW_SCHEMA, validator=_validate_review, events=events,
            )
            if review["is_reasonable"]:
                changed = current_decomposition != original_decomposition
                trace.update(
                    status="rewritten" if changed else "accepted",
                    final_decomposition=list(current_decomposition),
                    post_rewrite_validation=(
                        "deterministic_context_name_canonicalization"
                        if changed else "not_applicable"
                    ),
                )
                return list(current_decomposition), trace
            rewritten = self._generate(
                action="rewrite", instruction=rewrite_instruction,
                payload={"question": question, "decomposition": list(current_decomposition),
                         "issues": review["issues"], **context},
                schema=REWRITE_SCHEMA, validator=_validate_rewrite, events=events,
            )["decomposition"]
            if _profile_bool(self.prompt_profile.get("canonicalize_context_names")):
                rewritten, name_changes = _canonicalize_context_entity_names(
                    question,
                    rewritten,
                    entity_context,
                )
                if name_changes:
                    events.append({
                        "action": "rewrite_context_name_canonicalization",
                        "status": "applied",
                        "changes": name_changes,
                    })
            if _profile_bool(self.prompt_profile.get("merge_anchorless_rewrite_items")):
                rewritten, merged_items = _merge_anchorless_rewrite_items(
                    current_decomposition,
                    rewritten,
                    entity_context,
                )
                if merged_items:
                    events.append({
                        "action": "anchorless_rewrite_merge",
                        "status": "applied",
                        "merged_items": merged_items,
                    })
            trace["rewrite_count"] = 1
            guard_issues = rewrite_preservation_issues(
                question,
                original_decomposition,
                rewritten,
            )
            if guard_issues:
                events.append({"action": "rewrite_preservation_guard", "status": "rejected", "issues": guard_issues})
                trace.update(status="rejected_guard", last_decomposition=rewritten, guard_issues=guard_issues)
                return None, trace
            trace.update(status="rewritten", final_decomposition=rewritten,
                         post_rewrite_validation="local_only")
            return rewritten, trace
        except Exception as exc:
            trace.update(status="error", error=str(exc))
            return None, trace
        finally:
            trace["model_call_count"] = sum(event["action"] in {"review", "rewrite"} for event in events)
