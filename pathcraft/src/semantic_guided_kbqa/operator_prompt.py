from __future__ import annotations


OPERATOR_INSTRUCTION = r"""
# Task

Given the original question, anchors, and semantic paths, predict the required logical operators.

Ignore any operators predicted by the old COMPOSE model. Infer operators only from the question and semantic paths.

Return only one valid JSON object:

{
  "and_groups": [],
  "operators": []
}

# Rules

* `AND`: intersection of two or more semantic paths that independently produce complete final answer sets. Do not use AND for ordinary path connection, shared variables, JOIN, or intermediate-node constraints. Put AND only in `and_groups`, never in `operators`.
* `COUNT`: return the number of elements in a result set.
* `ARGMAX`: select the result with the maximum value of an attribute, e.g. latest, last, largest.
* `ARGMIN`: select the result with the minimum value of an attribute, e.g. earliest, first, smallest.
* `GREATER_THAN`: keep values strictly greater than the given value.
* `GREATER_OR_EQUAL`: keep values greater than or equal to the given value.
* `LESS_THAN`: keep values strictly less than the given value.
* `LESS_OR_EQUAL`: keep values less than or equal to the given value.
* `TC`: apply a temporal constraint such as a year, date, `NOW`, or current membership.
* `EQUAL`: keep entities whose specified attribute equals the given value. Do not use it for variable equality between paths.
* `NO_EQUAL`: exclude a specified anchor entity from the candidate answer set.

Every execution operator must contain these fields:

{
  "type": "",
  "inputs": [],
  "input_var": "",
  "attribute_relation_label": [],
  "attribute_relation_labels": [],
  "value": "",
  "value_type": ""
}

Use path-local variables such as `P0.V0`, `P0.V1`, and `P1.V0`.
Do not invent variables, values, relations, or constraints.
Use `NOW` for current or present-tense membership questions.
If no operator is needed, return empty arrays.

# Examples

For a question whose semantic paths directly identify the requested answer,
return no operator:

```json
{
  "and_groups": [],
  "operators": []
}
```

For an intersection of two independent paths plus exclusion of the topic entity:

```json
{
  "and_groups": [["P0", "P1"]],
  "operators": [
    {
      "type": "NO_EQUAL",
      "inputs": [],
      "input_var": "P0.V1",
      "attribute_relation_label": [],
      "attribute_relation_labels": [],
      "value": "Justin Bieber",
      "value_type": "entity"
    }
  ]
}
```

For a present-tense team membership:

```json
{
  "and_groups": [],
  "operators": [
    {
      "type": "TC",
      "inputs": [],
      "input_var": "P0.V0",
      "attribute_relation_label": ["sports", "sports team roster", "from"],
      "attribute_relation_labels": [],
      "value": "NOW",
      "value_type": ""
    }
  ]
}
```

For `last`, `latest`, or largest, use `ARGMAX` with the relevant attribute;
for `first`, `earliest`, or smallest, use `ARGMIN`:

```json
{
  "and_groups": [],
  "operators": [
    {
      "type": "ARGMAX",
      "inputs": [],
      "input_var": "P0.V0",
      "attribute_relation_label": ["time", "event", "end date"],
      "attribute_relation_labels": [],
      "value": "",
      "value_type": ""
    }
  ]
}
```

For a type restriction and topic exclusion, use `EQUAL` and `NO_EQUAL`:

```json
{
  "and_groups": [],
  "operators": [
    {
      "type": "EQUAL",
      "inputs": [],
      "input_var": "P0.V0",
      "attribute_relation_label": ["base", "biblioness", "bibs location", "loc type"],
      "attribute_relation_labels": [],
      "value": "Country",
      "value_type": "string"
    },
    {
      "type": "NO_EQUAL",
      "inputs": [],
      "input_var": "P0.V0",
      "attribute_relation_label": [],
      "attribute_relation_labels": [],
      "value": "Balkans",
      "value_type": "entity"
    }
  ]
}
```

# Output

Return only:

{
  "and_groups": [],
  "operators": []
}

The user message contains the JSON object with `question`, `anchors`, and `semantic_paths`.
""".strip()


OPERATOR_INSTRUCTION_V2 = r"""
# Task=OPERATOR

Given the original question and the composed logical query graph, predict only
the execution operators required to answer the question.

The input contains:
- question
- entities
- triples
- answer_var

The compose graph already resolves path intersection and variable unification.
Do not emit AND, path ids, path-local variables, selected_paths, equalities,
Freebase IDs, or SPARQL.

Allowed operator types:
COUNT, ARGMAX, ARGMIN, GREATER_THAN, GREATER_OR_EQUAL,
LESS_THAN, LESS_OR_EQUAL, EQUAL, TC, NO_EQUAL.

Return exactly:
{"operators": []}

Each operator must contain exactly these fields:
{
  "type": "",
  "inputs": [],
  "input_var": "V0",
  "attribute_relation_label": [],
  "attribute_relation_labels": [],
  "value": "",
  "value_type": ""
}

Use only global variables Vn from the composed triples. Use NOW for current or
present-tense membership. Use TC for temporal/current constraints, ARGMAX or
ARGMIN for latest/earliest/max/min, COUNT for counting, and NO_EQUAL to exclude
the topic entity when required. If no execution operator is needed, return an
empty operators array.

Return JSON only.
""".strip()


OPERATOR_INSTRUCTION_GLM = r"""
# Task=OPERATOR

Generate the execution operators for the original question and composed
logical query graph. Infer the minimum necessary operator set directly from
the question, graph, and training examples appended to this instruction.

The input contains:
- question: the original natural-language question
- semantic_graph: the grounded Semantic result with anchors and semantic_paths
- entities: grounded topic entities with ids such as E0
- triples: the composed graph; every triple has subject, relation_label,
  and object
- answer_var: the global variable returned as the answer

The Compose graph already represents joins and intersections through shared
global variables. Never emit AND, path ids, path-local variables, equalities
between variables, selected paths, Freebase ids, or SPARQL.

Return exactly one JSON object:

{
  "operators": []
}

Allowed operator types:
- COUNT: return the number of distinct results.
- ARGMAX: select the result associated with the maximum attribute value;
  use for latest, last, newest, largest, highest, or longest.
- ARGMIN: select the result associated with the minimum attribute value;
  use for first, earliest, oldest, smallest, lowest, or shortest.
- GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL: apply an explicit
  numeric or temporal comparison.
- EQUAL: require an attribute to equal a literal value. Do not use it for graph
  joins or for restating the topic entity.
- TC: apply an explicit year/date constraint or current membership. Use the
  literal NOW for current, present, now, or present-tense membership.
- NO_EQUAL: exclude a named topic entity from the candidate answer set.

Every operator must contain exactly these fields:

{
  "type": "",
  "inputs": [],
  "input_var": "V0",
  "attribute_relation_label": [],
  "attribute_relation_labels": [],
  "value": "",
  "value_type": ""
}

Rules:
1. Use only global variables Vn that occur in the composed triples.
2. input_var is the graph node that owns the filtered or ordered attribute.
   It can differ from answer_var. For example, a roster CVT can be V0 while its
   team answer is V1.
3. Express an attribute relation as human-readable Freebase components, for
   example ["sports", "sports team roster", "from"].
4. For COUNT, put the counted result variable in input_var and leave relation
   and value fields empty.
5. For ARGMAX and ARGMIN, leave value and value_type empty.
6. For TC with current membership, set value to NOW and leave value_type empty.
   For a stated year or date, copy that value exactly from the question.
7. Use the singular attribute_relation_label for one relation. Keep
   attribute_relation_labels empty unless the constraint explicitly needs
   multiple alternative relations.
8. Do not add an operator merely because the graph contains a date, number, or
   CVT. The wording of the question must require the operation.
9. If the question asks for a direct relation without counting, comparison,
   extrema, temporal restriction, equality filter, or exclusion, return an
   empty operators array.
10. If the graph has no variable that can support a required operator, return
    an empty array rather than inventing a variable.
11. The words before and after are temporal boundaries, not extrema by
    themselves. Never emit ARGMAX or ARGMIN merely because a question contains
    before or after. Use an extrema operator only when the question explicitly
    asks for first, last, earliest, latest, or another superlative.
12. Use semantic_graph to understand path intent and path-local constraints,
    but use only global Vn variables from the composed triples in the output.
13. A `people.person.spouse_s` to `people.marriage.spouse` path returns both
    participants in the marriage. Always emit NO_EQUAL on the answer variable
    with the topic person's surface so the person is not returned as their own
    spouse.
14. More generally, when the answer variable can return a topic entity itself
    but the question asks for the counterpart or another entity related to that
    topic, emit NO_EQUAL on the answer variable with that topic entity's exact
    surface. Do not exclude a topic entity when it is a valid requested answer.

Return JSON only. Do not explain the answer.
""".strip()
