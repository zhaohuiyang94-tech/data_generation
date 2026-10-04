from __future__ import annotations


OPERATOR_CONTRACT = r"""
<operator_contract>
Allowed semantic-stage operators are exactly:
AND, COUNT, ARGMAX, ARGMIN, GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN,
LESS_OR_EQUAL, and TC. The formal short names map to JSON types as follows:
GT -> GREATER_THAN, GE -> GREATER_OR_EQUAL, LT -> LESS_THAN, and
LE -> LESS_OR_EQUAL. Never output GT, GE, LT, or LE as the JSON type.
The formal names GREATERTHAN, GREATEREQUAL, LESS THAN, and LESS EQUAL are also
normalized to those same four JSON types; they are not additional operator types.

Let E, E1, and E2 denote entity or literal sets, r a canonical relation, and i a literal:
- AND(E1, E2) is the intersection of E1 and E2. Normalize an n-way intersection to
  one AND whose inputs contain every semantic path id exactly once. AND is present iff
  there are at least two semantic paths. Each path id denotes the set produced by that
  path. In semantic output use inputs=["P0","P1",...], input_var="", empty attribute
  fields, value="", and value_type="". In the complete compose program, lower AND to
  shared global variables and connected triples; do not emit AND in compose operators.
- COUNT(E) returns the cardinality of E. Put E's path-local variable in input_var.
  Use inputs=[] and leave both attribute fields, value, and value_type empty.
- ARGMAX(E,r) selects members of E with the maximum literal in {JOIN(r,e) | e in E}.
- ARGMIN(E,r) selects members of E with the minimum literal in {JOIN(r,e) | e in E}.
  For ARGMAX/ARGMIN put E in input_var, r in attribute_relation_label (or use
  attribute_relation_labels for a relation chain), inputs=[], value="", value_type="".
- GT(E,i), GE(E,i), LT(E,i), and LE(E,i) return the subset of E whose literal is >, >=,
  <, or <= i. E is the literal-valued path variable in input_var and i is value. The
  projection relation normally appears as a semantic-path triple, so attribute fields
  may be empty; when the source label explicitly supplies an attribute relation, preserve
  it exactly. inputs=[] and value must be non-empty.
- TC(E,i) returns the subset of E constrained by temporal literal i. Put the temporal
  record/entity set in input_var, the canonical time relation in attribute_relation_label,
  i in value (including NOW), and use inputs=[].
- If no operation is expressed, operators must be exactly []. Do not infer an operator
  from generic words. For example, "first name" is not ARGMIN and "how old" is not COUNT.

Every operator object must always contain exactly these fields, even when unused:
type, inputs, input_var, attribute_relation_label, attribute_relation_labels, value,
and value_type. Unused strings are "" and unused arrays are []. All non-AND operators
are unary and therefore have inputs=[]. Semantic input_var values use Pn.Vm; compose
input_var values use Vn. Compose must preserve type, attribute fields, value, and
value_type exactly while changing only the variable namespace.
</operator_contract>
"""


OPERATOR_FEW_SHOTS = r"""
<operator_few_shot_examples>
The examples below show only the operator-related slice of each input/output. The actual
response must still contain the complete schema requested by the task. Do not copy labels
such as "Input context", "Semantic output", "Compose output", or "lowering" into JSON.

Example 1 - NO_OPERATOR
Input context: question="Where was Ada Lovelace born?"; P0.V0 is the place of birth.
Semantic output: {"operators":[]}
Compose output: {"operators":[]}

Example 2 - AND
Input context: question="Which actors appeared in both Film A and Film B?";
P0 and P1 each denote one actor set.
Semantic output:
{"operators":[{"type":"AND","inputs":["P0","P1"],"input_var":"",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"","value_type":""}]}
Compose output:
{"operators":[]}
Lowering note: map the intersected path variables to the same global V0.

Example 3 - COUNT
Input context: question="How many children does Barack Obama have?"; P0.V0 is the child set.
Semantic output:
{"operators":[{"type":"COUNT","inputs":[],"input_var":"P0.V0",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"","value_type":""}]}
Compose output:
{"operators":[{"type":"COUNT","inputs":[],"input_var":"V0",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"","value_type":""}]}

Example 4 - ARGMAX
Input context: question="Which city in France has the largest population?";
P0.V0 is the city set and population is the ordering attribute.
Semantic output:
{"operators":[{"type":"ARGMAX","inputs":[],"input_var":"P0.V0",
"attribute_relation_label":["location","statistical region","population"],
"attribute_relation_labels":[],"value":"","value_type":""}]}
Compose output:
{"operators":[{"type":"ARGMAX","inputs":[],"input_var":"V0",
"attribute_relation_label":["location","statistical region","population"],
"attribute_relation_labels":[],"value":"","value_type":""}]}

Example 5 - ARGMIN
Input context: question="What is the first Harry Potter novel?";
P0.V0 is the novel set and publication date is the ordering attribute.
Semantic output:
{"operators":[{"type":"ARGMIN","inputs":[],"input_var":"P0.V0",
"attribute_relation_label":["book","written work","date of first publication"],
"attribute_relation_labels":[],"value":"","value_type":""}]}
Compose output:
{"operators":[{"type":"ARGMIN","inputs":[],"input_var":"V0",
"attribute_relation_label":["book","written work","date of first publication"],
"attribute_relation_labels":[],"value":"","value_type":""}]}

Example 6 - GT / GREATER_THAN
Input context: question="Which mountains are higher than 8000 metres?";
P0.V0 is the mountain answer set and P0.V1 is its elevation literal set; the elevation
relation is already a semantic-path step.
Semantic output:
{"operators":[{"type":"GREATER_THAN","inputs":[],"input_var":"P0.V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"8000","value_type":"number"}]}
Compose output:
{"operators":[{"type":"GREATER_THAN","inputs":[],"input_var":"V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"8000","value_type":"number"}]}

Example 7 - GE / GREATER_OR_EQUAL
Input context: question="Which films run for at least 180 minutes?";
P0.V1 is the runtime literal set.
Semantic output:
{"operators":[{"type":"GREATER_OR_EQUAL","inputs":[],"input_var":"P0.V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"180","value_type":"number"}]}
Compose output:
{"operators":[{"type":"GREATER_OR_EQUAL","inputs":[],"input_var":"V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"180","value_type":"number"}]}

Example 8 - LT / LESS_THAN
Input context: question="Which buildings are lower than 50 metres?";
P0.V1 is the height literal set.
Semantic output:
{"operators":[{"type":"LESS_THAN","inputs":[],"input_var":"P0.V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"50","value_type":"number"}]}
Compose output:
{"operators":[{"type":"LESS_THAN","inputs":[],"input_var":"V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"50","value_type":"number"}]}

Example 9 - LE / LESS_OR_EQUAL
Input context: question="Which rivers are no longer than 100 kilometres?";
P0.V1 is the length literal set.
Semantic output:
{"operators":[{"type":"LESS_OR_EQUAL","inputs":[],"input_var":"P0.V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"100","value_type":"number"}]}
Compose output:
{"operators":[{"type":"LESS_OR_EQUAL","inputs":[],"input_var":"V1",
"attribute_relation_label":[],"attribute_relation_labels":[],"value":"100","value_type":"number"}]}

Example 10 - TC
Input context: question="Who played for the Warriors in 2015?";
P0.V0 is the roster record set and P0.V1 is the player answer set.
Semantic output:
{"operators":[{"type":"TC","inputs":[],"input_var":"P0.V0",
"attribute_relation_label":["sports","sports team roster","from"],
"attribute_relation_labels":[],"value":"2015","value_type":"year"}]}
Compose output:
{"operators":[{"type":"TC","inputs":[],"input_var":"V0",
"attribute_relation_label":["sports","sports team roster","from"],
"attribute_relation_labels":[],"value":"2015","value_type":"year"}]}
</operator_few_shot_examples>
"""


GATEWAY_OPERATOR_FEW_SHOTS = r"""Operator examples use all required fields. Semantic
variables are path-local (Pn.Vm); compose maps them to global Vn and otherwise preserves
the object. NO_OPERATOR / NONE: {"operators":[]}.
AND semantic: {"type":"AND","inputs":["P0","P1"],"input_var":"","attribute_relation_label":[],"attribute_relation_labels":[],"value":"","value_type":""}; compose lowers it to shared variables and operators=[].
COUNT: {"type":"COUNT","inputs":[],"input_var":"P0.V0","attribute_relation_label":[],"attribute_relation_labels":[],"value":"","value_type":""}.
ARGMAX: {"type":"ARGMAX","inputs":[],"input_var":"P0.V0","attribute_relation_label":["location","statistical region","population"],"attribute_relation_labels":[],"value":"","value_type":""}.
ARGMIN: {"type":"ARGMIN","inputs":[],"input_var":"P0.V0","attribute_relation_label":["book","written work","date of first publication"],"attribute_relation_labels":[],"value":"","value_type":""}.
GT / GREATER_THAN: {"type":"GREATER_THAN","inputs":[],"input_var":"P0.V1","attribute_relation_label":[],"attribute_relation_labels":[],"value":"8000","value_type":"number"}.
GE / GREATER_OR_EQUAL: {"type":"GREATER_OR_EQUAL","inputs":[],"input_var":"P0.V1","attribute_relation_label":[],"attribute_relation_labels":[],"value":"180","value_type":"number"}.
LT / LESS_THAN: {"type":"LESS_THAN","inputs":[],"input_var":"P0.V1","attribute_relation_label":[],"attribute_relation_labels":[],"value":"50","value_type":"number"}.
LE / LESS_OR_EQUAL: {"type":"LESS_OR_EQUAL","inputs":[],"input_var":"P0.V1","attribute_relation_label":[],"attribute_relation_labels":[],"value":"100","value_type":"number"}.
TC: {"type":"TC","inputs":[],"input_var":"P0.V0","attribute_relation_label":["sports","sports team roster","from"],"attribute_relation_labels":[],"value":"2015","value_type":"year"}.
For compose, map P0.V0->V0 and P0.V1->V1 in these unary examples."""


GATEWAY_OPERATOR_RULES = f"""Allowed types: AND, COUNT, ARGMAX, ARGMIN,
GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL, TC. Never emit short
GT/GE/LT/LE as JSON types. Every operator has exactly: type, inputs, input_var,
attribute_relation_label, attribute_relation_labels, value, value_type. Unused strings
are "" and arrays are []. AND exists iff there are multiple semantic paths, uses every
path id in inputs, and is lowered to shared compose variables. Unary operators use
inputs=[]. Preserve source non-AND type, relation fields, value, and value_type exactly.
{GATEWAY_OPERATOR_FEW_SHOTS}"""


GATEWAY_SEMANTIC_INSTRUCTIONS = f"""Rewrite one KAeDe-VQG semantic graph without changing
the question meaning. Return only JSON with anchors, semantic_paths, operators. Each
anchor has id,surface. Each path has id,anchor_ref,goal,steps,path_output_var; each step
has id,relation_label,direction,from,to. Use reversible relation-label arrays and Pn.Vm
variables. Preserve expected_operators exactly and derive AND from path count. No KB IDs
or SPARQL.
{GATEWAY_OPERATOR_RULES}"""


GATEWAY_COMPOSE_INSTRUCTIONS = f"""Build one complete KAeDe-VQG compose JSON object from
the verified semantic graph. Return only entities,triples,operators,answer_var. Entities
have id,surface; triples have subject,relation_label,object. Use E0/E1 entities and V0/V1
global variables. Lower AND by sharing variables; map every unary operator input_var from
Pn.Vm to Vn and preserve all other fields. No KB IDs or SPARQL.
{GATEWAY_OPERATOR_RULES}"""


GATEWAY_VERIFY_INSTRUCTIONS = f"""Strictly verify one rewritten KAeDe-VQG training pair
against its question, decomposition, source graphs, joins, directions, answer projection,
and operators. Return only JSON with status,issues,semantic_graph,compose_program. status
is pass only if candidates are unchanged; corrected only with complete reliable corrected
objects; otherwise reject and repeat the candidates. Never add outside facts or KB IDs.
{GATEWAY_OPERATOR_RULES}"""


LEAN_OPERATOR_EXAMPLES = r"""O(t,ins,v,a,chain,x,xt) means the required JSON object
{"type":t,"inputs":ins,"input_var":v,"attribute_relation_label":a,
"attribute_relation_labels":chain,"value":x,"value_type":xt}. Examples:
NO_OPERATOR=[]; AND=O("AND",["P0","P1"],"",[],[],"","");
COUNT=O("COUNT",[],"P0.V0",[],[],"","");
ARGMAX=O("ARGMAX",[],"P0.V0",["location","statistical region","population"],[],"","");
ARGMIN=O("ARGMIN",[],"P0.V0",["book","written work","date of first publication"],[],"","");
GT / GREATER_THAN=O("GREATER_THAN",[],"P0.V1",[],[],"8000","number");
GE / GREATER_OR_EQUAL=O("GREATER_OR_EQUAL",[],"P0.V1",[],[],"180","number");
LT / LESS_THAN=O("LESS_THAN",[],"P0.V1",[],[],"50","number");
LE / LESS_OR_EQUAL=O("LESS_OR_EQUAL",[],"P0.V1",[],[],"100","number");
TC=O("TC",[],"P0.V0",["sports","sports team roster","from"],[],"2015","year").
Compose maps Pn.Vm to Vn; AND becomes shared variables and is omitted from compose operators."""


LEAN_SEMANTIC_INSTRUCTIONS = f"""Return only one complete semantic graph JSON with keys
anchors,semantic_paths,operators. Preserve the source meaning; copy valid content and only
fix clear defects. Anchors have id,surface. Paths have id,anchor_ref,goal,steps,
path_output_var. Steps have id,relation_label,direction,from,to. Use Pn.Vm variables and
never output KB IDs or SPARQL. required_operators must be preserved exactly. AND exists
iff multiple paths and contains all path ids. {LEAN_OPERATOR_EXAMPLES}"""


LEAN_COMPOSE_INSTRUCTIONS = f"""Return only one complete compose JSON with keys entities,
triples,operators,answer_var. Build it from semantic_graph while preserving source meaning.
Entities have id,surface. Triples have subject,relation_label,object. Use En and Vn. Lower
AND to shared variables. Map unary operator Pn.Vm to Vn without changing other fields.
Never output KB IDs or SPARQL. {LEAN_OPERATOR_EXAMPLES}"""


LEAN_VERIFY_INSTRUCTIONS = """Return only JSON with status and issues. status is pass or
reject. Compare the candidates with the question and source graphs. Reject wrong meaning,
paths, joins, directions, answer projection, relation labels, variables, or operators.
Use pass only when both candidates are answer-equivalent and internally consistent."""


MICRO_SEMANTIC_INSTRUCTIONS = """Return only a complete semantic graph JSON. Preserve
source meaning and valid content. Use keys anchors,semantic_paths,operators and Pn.Vm
variables. required_ops is exact: copy every non-AND object; [] means none. Use AND with
all path ids only when multiple paths. Every operator needs type,inputs,input_var,
attribute_relation_label,attribute_relation_labels,value,value_type. No KB IDs or SPARQL."""


MICRO_COMPOSE_INSTRUCTIONS = """Return only complete compose JSON with entities,triples,
operators,answer_var. Preserve source meaning. Convert semantic paths to En/Vn triples,
lower AND to shared variables, and map unary operator Pn.Vm to Vn without other changes.
No KB IDs or SPARQL."""


MICRO_VERIFY_INSTRUCTIONS = """Return only {"status":"pass|reject","issues":[]}.
Pass only if semantic and compose agree with the question, joins, answer variable and
operators. Otherwise reject with short concrete issues."""


SEMANTIC_INSTRUCTIONS = f"""You rewrite supervised Freebase KBQA data for KAeDe-VQG.
Return one semantic graph, not an answer to the question. Preserve the source example's
meaning while correcting only clear structural defects. Use canonical reversible
relation_label segments and path-local variables P0.V0, P0.V1, etc. Anchors must be
grounded in the question or source example. Move every operator into the semantic
graph's top-level operators field. The expected_operators list is mandatory for all
non-AND operators: preserve its type, value, value_type, and attribute relation labels
exactly. Derive normalized AND from the final semantic path count as specified below.
Do not emit dotted Freebase IDs or SPARQL.
{OPERATOR_CONTRACT}
{OPERATOR_FEW_SHOTS}
Now return only the complete semantic graph for the supplied request."""


COMPOSE_INSTRUCTIONS = f"""You rebuild a complete JSON logical query program from a verified
semantic graph. Compose paths by sharing global variables V0, V1, etc. Use E0, E1, etc.
for entity references. Lower semantic AND into shared variables. For every other semantic
operator, change only its path-local input_var to the corresponding global variable and
preserve all remaining fields exactly. Return only entities, triples, operators, and
answer_var. Do not emit dotted Freebase IDs or SPARQL.
{OPERATOR_CONTRACT}
{OPERATOR_FEW_SHOTS}
Now return only the complete compose program for the supplied request."""


VERIFY_INSTRUCTIONS = f"""Act as an independent, strict reviewer of one rewritten KAeDe-VQG
training pair. Check the original question, decomposition, source labels, rewritten
semantic paths, answer projection, joins, relation directions, entity surfaces, and every
operator. The rewritten pair must be answer-equivalent to the source. Check AND lowering
and require every non-AND semantic operator to be preserved in compose_program after
path-local variables are mapped to global variables. Never add outside facts or KB IDs.
Return status=pass only when no change is needed and repeat both candidates exactly.
Return status=corrected only for a minimal reliable correction and include both complete
corrected objects. Otherwise return status=reject, list concrete issues, and repeat the
uncorrected candidates in the object fields.
{OPERATOR_CONTRACT}"""


_SFT_OPERATOR_TYPES = (
    "Allowed semantic operators: AND, COUNT, ARGMAX, ARGMIN, GREATER_THAN, "
    "GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL, TC, or no operator. GT/GE/LT/LE "
    "must be emitted using their long canonical JSON names. AND inputs are path ids "
    "and compose lowers AND to shared variables."
)


SEMANTIC_SFT_INSTRUCTION = (
    "[TASK=SEMANTIC_PATH] Convert the decomposition into canonical Freebase semantic "
    "paths and extract operators at this stage. Operators must use path-local variables "
    "and canonical attribute relation labels. "
    f"{_SFT_OPERATOR_TYPES}\n{OPERATOR_CONTRACT}\n{OPERATOR_FEW_SHOTS}\n"
    "Return only one JSON object with anchors, semantic_paths, and operators; do not "
    "emit KB IDs."
)


COMPOSE_SFT_INSTRUCTION = (
    "[TASK=COMPOSE] Build the complete JSON logical query program from the supplied "
    "semantic graph. Map path-local variables to shared global variables and preserve "
    "every non-AND operator exactly. "
    f"{_SFT_OPERATOR_TYPES}\n{OPERATOR_CONTRACT}\n{OPERATOR_FEW_SHOTS}\n"
    "Return only entities, triples, operators, and answer_var; do not emit KB IDs."
)
