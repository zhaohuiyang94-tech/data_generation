"""Bind relative episode dates from KG evidence, without Gold answers or LLM calls."""
from copy import deepcopy
import json
import re

from .evaluation import canonicalize_answer_id
from .ontology import relation_label_from_id


def relative_reference(question):
    q = question.strip().rstrip(" ?.")
    match = re.search(r"\b(before|after)\s+(?:(?:he|she|they)\s+(was|became|joined)\s+)?(.+)$", q, re.I)
    if not match:
        return None
    direction, verb, name = match.groups()
    name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.I).strip()
    if not name or re.search(r"\d|\b(?:how|many|died|death|born|named)\b", name, re.I):
        return None
    return {"direction": direction.lower(), "verb": (verb or "episode").lower(), "name": name}


def semantic_projection_items(question, decomposition):
    """Defer time-only declarations; retain relation questions and bound entity conditions."""
    relative = relative_reference(question)
    original_years = set(re.findall(r"\b[12]\d{3}\b", question))
    kept, deferred = [], []
    for item in decomposition:
        is_question = re.match(r"\s*(?:what|who|when|where|which|how)\b", item, re.I)
        comparison = relative and re.search(r"\b(before|after)\b", item, re.I) and re.search(r"\b(from|to|start|end)(?:\s+date)?\s+of\b", item, re.I)
        item_years = set(re.findall(r"\b[12]\d{3}\b", item))
        explicit_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", question))
        item_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", item))
        literal_comparison = (item_numbers and item_numbers <= explicit_numbers
                              and re.search(r"\b(before|after|less than|more than|greater than|at least|at most)\b", item, re.I))
        year_condition = (item_years and item_years <= original_years
                          and re.search(r"\b(covers|during|in year)\b", item, re.I))
        current_condition = (
            re.search(r"\b(?:covers|during|current)\b", item, re.I)
            and re.search(r"\b(?:now|current)\b", item, re.I)
        )
        ordinal_condition = (re.search(r"\b(first|earliest)\b", question, re.I)
                             and re.search(r"\bearliest\b", item, re.I) and re.search(r"\b(from|start)\b", item, re.I))
        if not is_question and (
            comparison
            or year_condition
            or current_condition
            or literal_comparison
            or ordinal_condition
        ):
            deferred.append(item)
        else:
            kept.append(item)
    return (kept, deferred) if kept else (list(decomposition), [])


class RelativeTemporalBinder:
    def __init__(self, kg, ontology):
        self.kg = kg
        self.ontology = ontology
        self.cache = {}

    def bind(self, question, graph):
        type_binding = self.bind_declared_types(graph)
        ordinal = self.bind_first(question, graph)
        ordinal["declared_type_constraints"] = type_binding
        literal = self.bind_literal(question, graph)
        if literal["status"] != "not_applicable":
            return literal
        intent = relative_reference(question)
        if not intent or self.ontology is None:
            return ordinal
        for owner, role_relation, target in graph.triples:
            if target != graph.answer_var or not owner.startswith("V"):
                continue
            role_range = self.ontology.range_for_relation(role_relation)
            if role_range.startswith("type."):
                continue
            domain = self.ontology.domain_for_relation(role_relation)
            dates = {r.rsplit(".", 1)[-1]: r for r in self.ontology.relations_for_domain(domain)
                     if self.ontology.range_for_relation(r) == "type.datetime"}
            start = dates.get("from") or dates.get("start_date")
            end = dates.get("to") or dates.get("end_date")
            if not start:
                continue
            incoming = next(((s, r) for s, r, o in graph.triples
                             if o == owner and re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", s)), None)
            if not incoming:
                continue
            subject, incoming_relation = incoming
            key = (subject, incoming_relation, role_relation, start, end, intent["name"].casefold())
            if key not in self.cache:
                optional_end = f"OPTIONAL {{ ?record <http://rdf.freebase.com/ns/{end}> ?end . }}" if end else ""
                prefix = json.dumps(intent["name"].casefold())
                query = f'''SELECT DISTINCT ?record ?reference ?start ?end WHERE {{
 <http://rdf.freebase.com/ns/{subject}> <http://rdf.freebase.com/ns/{incoming_relation}> ?record .
 ?record <http://rdf.freebase.com/ns/{role_relation}> ?reference .
 ?reference <http://rdf.freebase.com/ns/type.object.name> ?name .
 FILTER(LANGMATCHES(LANG(?name), "EN"))
 FILTER(STRSTARTS(LCASE(STR(?name)), {prefix}))
 FILTER(STRLEN(STR(?name)) = {len(intent['name'])} || SUBSTR(STR(?name), {len(intent['name']) + 1}, 1) IN (" ", ".", "-"))
 ?record <http://rdf.freebase.com/ns/{start}> ?start .
 {optional_end}
}}'''
                self.cache[key] = (query, self.kg.execute(query))
            query, evidence = self.cache[key]
            cutoff_field = "start" if intent["direction"] == "before" or intent["verb"] in {"became", "joined"} else "end"
            values = [canonicalize_answer_id(r[cutoff_field]) for r in evidence if r.get(cutoff_field)]
            values = [v for v in values if re.fullmatch(r"[12]\d{3}(?:-\d{2}-\d{2})?", v)]
            if not values:
                return {"status": "unresolved", "reason": "reference_episode_date_missing", "query": query}
            cutoff = min(values) if intent["direction"] == "before" else max(values)
            successor = role_relation.rsplit(".", 1)[-1] in {"office_holder", "coach"} and re.match(r"who\b", question.strip(), re.I)
            after_end = intent["direction"] == "after" and cutoff_field == "end"
            operator = {"type": "LESS_THAN" if intent["direction"] == "before" else "GREATER_OR_EQUAL" if after_end else "GREATER_THAN",
                        "inputs": [], "input_var": owner, "attribute_relation_label": relation_label_from_id(start),
                        "attribute_relation_labels": [], "value": cutoff,
                        "value_type": "year" if len(cutoff) == 4 else "date"}
            if len(cutoff) == 4:
                operator["_comparison_date_precision"] = "year"
            else:
                operator["_calendar_day_boundary"] = True
            # Only supersede a comparative on this same episode date. Preserve
            # every unrelated type/entity/time/numeric restriction.
            graph.operators = [op for op in graph.operators if not (
                (op.get("type") in {"LESS_THAN", "GREATER_THAN", "LESS_OR_EQUAL", "GREATER_OR_EQUAL"}
                and op.get("input_var") == owner
                and op.get("attribute_relation_label") == operator["attribute_relation_label"])
                or (op.get("type") == "TC" and op.get("input_var") == owner
                    and not re.search(r"\b[12]\d{3}\b", question)))]
            graph.operators.append(operator)
            if successor:
                graph.operators.append({"type": "ARGMAX" if intent["direction"] == "before" else "ARGMIN",
                    "inputs": [], "input_var": owner, "attribute_relation_label": relation_label_from_id(start),
                    "attribute_relation_labels": [], "value": "", "value_type": ""})
            return {"status": "bound", "intent": intent, "owner": owner, "reference_date": cutoff,
                    "reference_date_field": cutoff_field, "query": query, "evidence": deepcopy(evidence),
                    "operator": deepcopy(operator), "uses_gold_answers": False, "llm_calls": 0}
        return ordinal

    def bind_declared_types(self, graph):
        if self.ontology is None:
            return []
        added = []
        for item in graph.provenance.get("decomposition", []):
            match = re.fullmatch(r"\s*(.+?) is the type of the (.+?)\.\s*", item, re.I)
            if not match:
                continue
            type_name, role_name = [s.casefold().strip() for s in match.groups()]
            types = [d for d in self.ontology.domain_to_relation_ids
                     if d.rsplit(".", 1)[-1].replace("_", " ").casefold() == type_name]
            if len(types) != 1:
                continue
            target = next((o for _, r, o in graph.triples
                           if r.rsplit(".", 1)[-1].replace("_", " ").casefold() == role_name), None)
            if target:
                triple = [target, "type.object.type", types[0]]
                if triple not in graph.triples:
                    graph.triples.append(triple)
                    added.append(triple)
        return added

    def bind_first(self, question, graph):
        if self.ontology is None or re.search(r"first name|first language", question, re.I):
            return {"status": "not_applicable"}
        masked = question
        for entity in graph.provenance.get("anchor_bindings", {}).values():
            if entity.get("label"):
                masked = re.sub(re.escape(entity["label"]), "", masked, flags=re.I)
        if not re.search(r"\b(first|earliest)\b", masked, re.I):
            return {"status": "not_applicable"}
        for owner, role, target in graph.triples:
            if target != graph.answer_var or not owner.startswith("V"):
                continue
            domain = self.ontology.domain_for_relation(role)
            relation = next((r for r in self.ontology.relations_for_domain(domain)
                             if r.rsplit(".", 1)[-1] in {"from", "start_date"}
                             and self.ontology.range_for_relation(r) == "type.datetime"), None)
            if relation:
                graph.operators = [op for op in graph.operators if not (
                    op.get("type") in {"ARGMIN", "ARGMAX"} and op.get("input_var") == owner)]
                op = {"type": "ARGMIN", "inputs": [], "input_var": owner,
                      "attribute_relation_label": relation_label_from_id(relation),
                      "attribute_relation_labels": [], "value": "", "value_type": "",
                      "_order_temporal_lexical": True}
                graph.operators.append(op)
                return {"status": "bound", "intent": "first_episode", "operator": deepcopy(op),
                        "uses_gold_answers": False, "llm_calls": 0}
        return {"status": "not_applicable"}

    @staticmethod
    def words(text):
        aliases = {"died": "death", "dead": "death", "born": "birth", "released": "release"}
        result = set()
        for word in re.findall(r"[a-z]+", text.casefold()):
            word = aliases.get(word, word)
            if word.endswith("ing") and len(word) > 5:
                word = word[:-3]
                if len(word) > 2 and word[-1] == word[-2]:
                    word = word[:-1]
            elif word.endswith("s") and len(word) > 3:
                word = word[:-1]
            result.add(word)
        return result

    def bind_literal(self, question, graph):
        if self.ontology is None:
            return {"status": "not_applicable"}
        decomposition_text = " ".join(
            str(item) for item in graph.provenance.get("decomposition", [])
        )
        decomposition_years = [
            match.group(1)
            for match in re.finditer(
                r"\b(?:covers|during|in year)\s+([12]\d{3})\b",
                decomposition_text,
                re.I,
            )
        ]
        date_pattern = r"\b(before|after|in)\s+(?:the year\s+)?([12]\d{3}-\d{2}-\d{2}|[12]\d{3})(?![-\d])"
        dates = list(re.finditer(date_pattern, question, re.I))
        seen_dates = {
            (match.group(1).casefold(), match.group(2)) for match in dates
        }
        for match in re.finditer(date_pattern, decomposition_text, re.I):
            key = (match.group(1).casefold(), match.group(2))
            if key not in seen_dates:
                dates.append(match)
                seen_dates.add(key)
        numeric = re.search(r"\b(less than|more than|greater than|at least|at most)\s+(-?\d+(?:\.\d+)?)", question, re.I)
        numeric_equality = self.explicit_numeric_equality(question, graph)
        current = bool(
            re.search(r"\b(?:current|now)\b", question, re.I)
            or re.search(r"\b(?:covers|during)\s+(?:the\s+)?(?:current|now)\b", decomposition_text, re.I)
        )
        masked_question = question
        for entity in graph.provenance.get("anchor_bindings", {}).values():
            label = entity.get("label", "")
            if label and re.search(r"\d", label):
                masked_question = re.sub(re.escape(label), "", masked_question, flags=re.I)
        standalone_years = re.findall(r"\b[12]\d{3}\b", masked_question)
        standalone_year = standalone_years[0] if not dates and not numeric and len(standalone_years) == 1 else None
        if (
            not dates
            and not numeric
            and numeric_equality is None
            and not standalone_year
            and not decomposition_years
            and not current
        ):
            return {"status": "not_applicable"}
        owners = []
        for subject, relation, target in graph.triples:
            if target != graph.answer_var:
                continue
            target_type = self.ontology.range_for_relation(relation)
            if target_type.startswith("type."):
                continue
            domain = self.ontology.domain_for_relation(relation)
            if subject.startswith("V"):
                owners.append((subject, domain))
            owners.append((target, target_type))
        matches = []
        qwords = self.words(question)
        for owner, domain in owners:
            for relation in self.ontology.relations_for_domain(domain):
                value_type = self.ontology.range_for_relation(relation)
                name = relation.rsplit(".", 1)[-1].replace("_", " ")
                overlap = self.words(name) & qwords
                if (
                    dates
                    or (standalone_year and owner.startswith("V") and name in {"from", "start date"})
                    or (decomposition_years and owner.startswith("V") and name in {"from", "start date"})
                    or (current and owner.startswith("V") and name in {"from", "start date"})
                ) and value_type == "type.datetime":
                    priority = 2 if name in {"from", "start date"} else 1 if "release" in name else 0
                    matches.append((len(overlap) * 3 + priority, owner, relation, value_type))
                elif numeric and value_type in {"type.int", "type.float", "type.rawstring", "type.text", "type.enumeration"} and overlap - {"id", "number"}:
                    matches.append((len(overlap), owner, relation, value_type))
        if not matches:
            measurement_year = (
                standalone_year
                if standalone_year
                and re.search(
                    r"\b(?:rate|percentage|percent|value|amount|population|gdp|cpi|"
                    r"inflation|emission|emissions|labor|labour|metric ton|metric tons)\b",
                    question,
                    re.I,
                )
                else None
            )
            if (numeric_equality is None and measurement_year is None) or not owners:
                return {"status": "not_applicable"}
            # Semantic can omit a scalar attribute even though the original
            # question contains an explicit value constraint.  Preserve that
            # constraint as a deliberately unresolved operator.  The bounded
            # ontology/embedding repair in the pipeline resolves its direct or
            # CVT attribute path before an answer is accepted.  This uses no
            # entity/value whitelist and does not add a model call.
            owner = next(
                (candidate for candidate, _ in owners if candidate == graph.answer_var),
                owners[0][0],
            )
            if numeric_equality is not None:
                value, attribute_words = numeric_equality
                value_type = "number"
                intent = "explicit_numeric_equality_coverage"
            else:
                value = str(measurement_year)
                attribute_words = " ".join(
                    word
                    for word in re.findall(r"[A-Za-z]+", question.casefold())
                    if word not in {
                        "a", "an", "the", "that", "which", "what", "where",
                        "who", "whose", "of", "in", "on", "once", "with",
                        "and", "or", "had", "has", "have", "was", "were",
                        "is", "are", "country", "countries", "location",
                    }
                )[-160:]
                value_type = "year"
                intent = "explicit_measurement_year_coverage"
            graph.operators = [
                old
                for old in graph.operators
                if not (
                    str(old.get("type", "")).upper() in {"ARGMIN", "ARGMAX"}
                    and not re.search(
                        r"\b(?:largest|smallest|highest|lowest|biggest|latest|earliest|most recent|first|last)\b",
                        question,
                        re.I,
                    )
                )
            ]
            if measurement_year is not None:
                existing = next(
                    (
                        old
                        for old in graph.operators
                        if str(old.get("type", "")).upper()
                        in {
                            "EQUAL",
                            "GREATER_THAN",
                            "GREATER_OR_EQUAL",
                            "LESS_THAN",
                            "LESS_OR_EQUAL",
                        }
                        and str(old.get("value", "")).strip() == str(measurement_year)
                    ),
                    None,
                )
                if existing is not None:
                    existing["type"] = "EQUAL"
                    existing["value_type"] = "year"
                    existing["_constraint_coverage_repair"] = True
                    existing.pop("_numeric_comparison_cast", None)
                    existing.pop("_numeric_cast_compare", None)
                    graph.provenance["explicit_numeric_constraint_repair"] = {
                        "value": str(measurement_year),
                        "attribute_words": attribute_words,
                        "owner": str(existing.get("input_var", owner)),
                        "reused_existing_operator": True,
                        "uses_gold_answers": False,
                        "llm_calls": 0,
                    }
                    return {
                        "status": "bound",
                        "intent": intent,
                        "operator": deepcopy(existing),
                        "uses_gold_answers": False,
                        "llm_calls": 0,
                    }
            operator = {
                "type": "EQUAL",
                "inputs": [],
                "input_var": owner,
                "attribute_relation_label": ["explicit numeric constraint", attribute_words],
                "attribute_relation_labels": [],
                "value": value,
                "value_type": value_type,
                "_numeric_cast_compare": True,
                "_constraint_coverage_repair": True,
            }
            graph.operators.append(operator)
            graph.provenance["explicit_numeric_constraint_repair"] = {
                "value": value,
                "attribute_words": attribute_words,
                "owner": owner,
                "uses_gold_answers": False,
                "llm_calls": 0,
            }
            return {
                "status": "bound",
                "intent": intent,
                "operator": deepcopy(operator),
                "uses_gold_answers": False,
                "llm_calls": 0,
            }
        _, owner, relation, value_type = max(matches)
        bound = []
        if value_type == "type.datetime":
            constraints = (
                [(match.group(1).lower(), match.group(2)) for match in dates]
                or ([("in", standalone_year)] if standalone_year else [])
            )
            for year in decomposition_years:
                if ("in", year) not in constraints:
                    constraints.append(("in", year))
            if current:
                constraints.append(("in", "NOW"))
        else:
            constraints = [(numeric.group(1).lower(), numeric.group(2))]
        types = {"before": "LESS_THAN", "after": "GREATER_THAN", "in": "TC", "less than": "LESS_THAN",
                 "more than": "GREATER_THAN", "greater than": "GREATER_THAN", "at least": "GREATER_OR_EQUAL", "at most": "LESS_OR_EQUAL"}
        for direction, value in constraints:
            op = {"type": types[direction], "inputs": [], "input_var": owner,
                  "attribute_relation_label": relation_label_from_id(relation), "attribute_relation_labels": [],
                  "value": value, "value_type": "year" if value_type == "type.datetime" and len(value) == 4 else "date" if value_type == "type.datetime" else "number"}
            if value_type == "type.datetime" and op["type"] != "TC":
                op["_comparison_date_precision"] = "year" if len(value) == 4 else "day"
                op["_calendar_day_boundary"] = len(value) == 10
            elif value_type in {"type.rawstring", "type.text", "type.enumeration"}:
                op["_numeric_comparison_cast"] = True
            graph.operators = [old for old in graph.operators if not (
                old.get("input_var") == owner
                and old.get("attribute_relation_label") == op["attribute_relation_label"]
                and old.get("type") == op["type"]
                and (
                    old.get("type") != "TC"
                    or str(old.get("value")) == value
                )
            )]
            graph.operators.append(op)
            bound.append(deepcopy(op))
        return {"status": "bound", "intent": "explicit_date_or_number", "operators": bound, "uses_gold_answers": False, "llm_calls": 0}

    @staticmethod
    def explicit_numeric_equality(question, graph):
        """Return an uncovered scalar equality expressed by the question.

        Four-digit values are normally years and are left to temporal binding.
        They count as scalar values only when an adjacent measurement unit makes
        that reading explicit.  Numbers inside already-linked entity labels are
        masked before matching, preventing event titles such as ``2010 FIFA
        World Cup`` from becoming accidental numeric filters.
        """
        masked = str(question)
        for entity in graph.provenance.get("anchor_bindings", {}).values():
            label = str(entity.get("label", "")).strip()
            if label and re.search(r"\d", label):
                masked = re.sub(re.escape(label), " ", masked, flags=re.I)

        # Comparison operators and dates already have dedicated semantics.
        if re.search(
            r"\b(?:less than|more than|greater than|at least|at most|before|after)\s+-?\d",
            masked,
            re.I,
        ):
            return None
        masked = re.sub(r"\b[12]\d{3}[-/]\d{1,2}[-/]\d{1,2}\b", " ", masked)
        matches = list(re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", masked))
        if not matches:
            return None

        scalar_cues = re.compile(
            r"\b(?:rate|percentage|percent|value|amount|number|count|population|"
            r"episode|episodes|gdp|cpi|inflation|emission|emissions|labor|labour|"
            r"metric ton|metric tons|ton|tons|kilomet(?:er|re)|mile|dollar)\b",
            re.I,
        )
        candidates = []
        for match in matches:
            value = match.group(0)
            start, end = match.span()
            context = masked[max(0, start - 72):min(len(masked), end + 32)]
            if not scalar_cues.search(context):
                continue
            # A four-digit scalar in CWQ's dated measurement templates is a
            # year even when the unit follows it (for example, ``2009 metric
            # ton``).  It is handled by measurement-year coverage above.
            if re.fullmatch(r"[12]\d{3}", value):
                continue
            candidates.append((match, value, context))
        if len(candidates) != 1:
            return None

        _, value, context = candidates[0]
        normalized_value = value.lstrip("+")
        for operator in graph.operators:
            old_value = str(operator.get("value", "")).strip().replace(",", "")
            if old_value == normalized_value and str(operator.get("type", "")).upper() in {
                "EQUAL",
                "GREATER_THAN",
                "GREATER_OR_EQUAL",
                "LESS_THAN",
                "LESS_OR_EQUAL",
            }:
                return None

        words = re.findall(r"[A-Za-z]+", context.casefold())
        stop = {
            "a", "an", "the", "that", "which", "what", "where", "who",
            "whose", "do", "does", "did", "had", "has", "have", "was",
            "were", "is", "are", "at", "of", "in", "on", "once", "with",
            "and", "or", "people", "country", "countries", "location",
        }
        attribute_words = " ".join(word for word in words if word not in stop)[-160:]
        if not attribute_words:
            return None
        return normalized_value, attribute_words
