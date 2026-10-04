from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from .contracts import GroundedSemanticCandidate, QueryGraphCandidate
from .ontology import relation_id_from_label


class LoweringError(ValueError):
    pass


def build_query_graph_v2(
    grounded: GroundedSemanticCandidate,
    compose_output: dict[str, Any],
    *,
    graph_id: str,
    pipeline_version: str = "0.2.0",
) -> QueryGraphCandidate:
    """Lower the 0.2.0 global COMPOSE graph while preserving grounding scores."""
    entities = compose_output.get("entities", [])
    bindings = _entity_bindings_v2(grounded, entities)
    relation_ids = _relation_bindings_v2(grounded)
    consumed_relations: set[int] = set()
    answer_var = str(compose_output.get("answer_var", ""))
    if not answer_var.startswith("V"):
        raise LoweringError("answer_var must be a global Vn variable")
    terminal_goal = _terminal_goal_clause(grounded.compose_input)
    relation_repairs: list[dict[str, str]] = []
    triples: list[list[str]] = []
    for triple in compose_output.get("triples", []):
        raw_subject = str(triple["subject"])
        raw_object = str(triple["object"])
        subject = _node_v2(raw_subject, bindings)
        object_ = _node_v2(raw_object, bindings)
        relation_id, repair = _resolve_relation_v2(
            triple.get("relation_label", []),
            relation_ids,
            consumed_relations,
            answer_edge=answer_var in {raw_subject, raw_object},
            terminal_goal=terminal_goal,
        )
        if not relation_id:
            raise LoweringError(
                f"unable to transfer grounded relation for {triple.get('relation_label')}"
            )
        if repair is not None:
            relation_repairs.append(repair)
        triples.append([subject, relation_id, object_])
    mediator_repairs = _repair_parallel_mediator_anchor(
        triples,
        answer_var=answer_var,
        terminal_goal=terminal_goal,
        question=str(grounded.compose_input.get("question", "")),
    )
    # A terminal predicate correction is intentionally coupled to the
    # parallel-mediator co-reference proof below.  Without that proof it would
    # merely replace one grounded beam alternative with an unverified model
    # preference and could perturb already-correct candidate ranking.
    if relation_repairs and not mediator_repairs:
        for repair in relation_repairs:
            for triple in triples:
                if (
                    triple[1] == repair["to"]
                    and answer_var in {triple[0], triple[2]}
                ):
                    triple[1] = repair["from"]
                    break
        relation_repairs = []
    if len(consumed_relations) != len(relation_ids):
        raise LoweringError(
            "compose output omitted one or more grounded semantic steps"
        )

    if not any(answer_var in (triple[0], triple[2]) for triple in triples):
        raise LoweringError("answer_var does not occur in a triple")

    variables = {
        value
        for triple in triples
        for value in (triple[0], triple[2])
        if value.startswith("V")
    }
    parent = {value: value for value in variables}
    global_names = {value: value for value in variables}
    operators = _lower_operators(
        compose_output.get("operators", []),
        answer_var=answer_var,
        triples=triples,
        parent=parent,
        global_names=global_names,
        anchor_values=_anchor_values(grounded),
    )
    provenance = {
        "version": str(pipeline_version),
        "anchor_bindings": {
            key: {
                "id": value.entity_id,
                "label": value.label,
                "score": value.score,
            }
            for key, value in grounded.anchor_bindings.items()
        },
        "relation_bindings": dict(grounded.relation_bindings),
        "grounded_semantic": grounded.provenance,
    }
    if relation_repairs:
        provenance["compose_terminal_relation_repairs"] = relation_repairs
    if mediator_repairs:
        provenance["parallel_mediator_anchor_repairs"] = mediator_repairs
    return QueryGraphCandidate(
        graph_id=graph_id,
        triples=triples,
        answer_var=answer_var,
        operators=operators,
        score=float(grounded.score),
        compose_output=deepcopy(compose_output),
        provenance=provenance,
    )


def _entity_bindings_v2(
    grounded: GroundedSemanticCandidate,
    entities: list[dict[str, Any]],
) -> dict[str, str]:
    bindings: dict[str, str] = {}
    available = list(grounded.anchor_bindings.values())
    for entity in entities:
        entity_id = str(entity.get("id", ""))
        surface = _surface_key(entity.get("surface", ""))
        match = next(
            (
                candidate
                for candidate in available
                if _surface_key(candidate.label) == surface
            ),
            None,
        )
        if match is None:
            match = next(
                (
                    candidate
                    for candidate in available
                    if surface and surface in _surface_key(candidate.label)
                ),
                None,
            )
        if match is None:
            raise LoweringError(f"unable to ground compose entity {entity.get('surface')!r}")
        bindings[entity_id] = match.entity_id
    return bindings


def _relation_bindings_v2(grounded: GroundedSemanticCandidate) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    paths = grounded.compose_input.get("semantic_paths", [])
    for path in paths:
        if not isinstance(path, dict):
            continue
        for step in path.get("steps", []):
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("id", ""))
            relation_id = str(grounded.relation_bindings.get(step_id, ""))
            if relation_id:
                result.append((_label_key(step.get("relation_label", [])), relation_id))
    return result


def _resolve_relation_v2(
    label: Any,
    candidates: list[tuple[str, str]],
    consumed: set[int],
    *,
    answer_edge: bool = False,
    terminal_goal: str = "",
) -> tuple[str, dict[str, str] | None]:
    key = _label_key(label)
    for index, (candidate_key, relation_id) in enumerate(candidates):
        if index in consumed:
            continue
        if candidate_key == key:
            consumed.add(index)
            return relation_id, None
    # A Graph-v2 Compose model can repair the terminal predicate while keeping
    # the same Freebase role owner (for example performance.character ->
    # performance.actor).  The old positional fallback silently discarded that
    # correction.  Trust it only on the answer edge, only within the same role
    # namespace, and only when the final natural-language goal explicitly names
    # the Compose property but not the stale grounded property.  This is a
    # constant-time structural check and does not add a model/KG request.
    next_unused = next(
        (
            (index, relation_id)
            for index, (_, relation_id) in enumerate(candidates)
            if index not in consumed
        ),
        None,
    )
    direct = relation_id_from_label(label)
    if (
        next_unused is not None
        and answer_edge
        and _terminal_relation_correction_is_safe(
            direct,
            next_unused[1],
            terminal_goal,
        )
    ):
        consumed.add(next_unused[0])
        return direct, {
            "strategy": "compose_terminal_predicate",
            "from": next_unused[1],
            "to": direct,
        }
    # Compose may normalize labels differently; retain a deterministic fallback
    # to the next unused relation from the same grounded semantic candidate.
    for index, (_, relation_id) in enumerate(candidates):
        if index not in consumed:
            consumed.add(index)
            return relation_id, None
    return "", None


def _terminal_goal_clause(compose_input: dict[str, Any]) -> str:
    clauses: list[str] = []
    for path in compose_input.get("semantic_paths", []):
        if not isinstance(path, dict):
            continue
        parts = [
            part.strip()
            for part in re.split(r"[.!?]+", str(path.get("goal", "")))
            if part.strip()
        ]
        if parts:
            clauses.append(parts[-1])
    return " ".join(clauses).casefold()


def _relation_property_tokens(relation_id: str) -> set[str]:
    if not relation_id or "." not in relation_id:
        return set()
    return {
        token
        for token in re.split(r"[^a-z0-9]+", relation_id.rsplit(".", 1)[-1].casefold())
        if token
    }


def _terminal_relation_correction_is_safe(
    compose_relation: str,
    grounded_relation: str,
    terminal_goal: str,
) -> bool:
    compose_parts = compose_relation.split(".")
    grounded_parts = grounded_relation.split(".")
    if (
        len(compose_parts) < 3
        or len(compose_parts) != len(grounded_parts)
        or compose_parts[:-1] != grounded_parts[:-1]
        or compose_relation == grounded_relation
    ):
        return False
    goal_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", str(terminal_goal).casefold())
        if token
    }
    compose_tokens = _relation_property_tokens(compose_relation)
    grounded_tokens = _relation_property_tokens(grounded_relation)
    return bool(compose_tokens & goal_tokens) and not bool(grounded_tokens & goal_tokens)


def _repair_parallel_mediator_anchor(
    triples: list[list[str]],
    *,
    answer_var: str,
    terminal_goal: str,
    question: str,
) -> list[dict[str, str]]:
    """Carry an anchor constraint across two parallel mediator occurrences.

    A model can express ``anchored mediator -> parent -> mediator -> answer``
    with two different variables even though the final definite noun phrase
    (``the performance``, ``the position held``, ...) refers back to the
    anchored mediator.  When both mediator variables share the same parent
    edge and role namespace, copy the existing constant constraint onto the
    answer-side mediator.  This tightens the current query in place; it adds no
    model request, KG expansion, or extra candidate execution.
    """
    repairs: list[dict[str, str]] = []
    goal = " ".join(str(terminal_goal).casefold().split())
    for terminal_subject, terminal_relation, terminal_object in list(triples):
        if terminal_object != answer_var or not terminal_subject.startswith("V"):
            continue
        terminal_parts = terminal_relation.split(".")
        if len(terminal_parts) < 3:
            continue
        role = terminal_parts[-2].replace("_", " ").casefold()
        if f"the {role}" not in goal:
            continue
        namespace = terminal_parts[:-1]
        for anchor_subject, anchor_relation, anchor_object in list(triples):
            if (
                anchor_subject == terminal_subject
                or not anchor_subject.startswith("V")
                or anchor_object.startswith("V")
                or anchor_relation.split(".")[:-1] != namespace
            ):
                continue
            shared_parent = next(
                (
                    (left, relation)
                    for left, relation, right in triples
                    if right == anchor_subject and left.startswith("V")
                    and any(
                        other_left == left
                        and other_relation == relation
                        and other_right == terminal_subject
                        for other_left, other_relation, other_right in triples
                    )
                ),
                None,
            )
            if shared_parent is None:
                continue
            parent_parts = shared_parent[1].split(".")
            parent_role = (
                parent_parts[-2].replace("_", " ").casefold()
                if len(parent_parts) >= 3
                else ""
            )
            question_tokens = {
                token
                for token in re.split(r"[^a-z0-9]+", str(question).casefold())
                if token
            }
            parent_role_tokens = {
                token for token in parent_role.split() if token
            }
            # If the question explicitly names the shared parent (for example
            # "the person who attended X, what other colleges..."), the two
            # mediator occurrences intentionally represent siblings and must
            # remain distinct.  Co-reference is safe only for an implicit
            # bridge introduced by path construction.
            if parent_role_tokens & question_tokens:
                continue
            added = [terminal_subject, anchor_relation, anchor_object]
            if added in triples:
                continue
            triples.append(added)
            repairs.append({
                "strategy": "parallel_mediator_anchor",
                "parent": shared_parent[0],
                "parent_relation": shared_parent[1],
                "from_mediator": anchor_subject,
                "to_mediator": terminal_subject,
                "constraint_relation": anchor_relation,
                "constraint_entity": anchor_object,
            })
    return repairs


def _node_v2(value: str, entity_bindings: dict[str, str]) -> str:
    if value.startswith("E"):
        try:
            return entity_bindings[value]
        except KeyError as exc:
            raise LoweringError(f"unknown compose entity: {value}") from exc
    if value.startswith("V"):
        return value
    raise LoweringError(f"invalid compose node: {value}")


def _surface_key(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _label_key(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(
            part
            for item in value
            for part in str(item).casefold().replace("_", " ").split()
        )
    return " ".join(str(value).casefold().replace("_", " ").split())


def build_query_graph(
    grounded: GroundedSemanticCandidate,
    compose_output: dict[str, Any],
    *,
    graph_id: str,
    pipeline_version: str = "0.2.0",
) -> QueryGraphCandidate:
    paths = {
        str(path["id"]): path
        for path in grounded.compose_input["semantic_paths"]
        if isinstance(path, dict)
    }
    selected_ids = [str(value) for value in compose_output["selected_paths"]]
    if any(path_id not in paths for path_id in selected_ids):
        raise LoweringError("compose selected an unknown path")
    selected_paths = [paths[path_id] for path_id in selected_ids]
    local_variables = _ordered_variables(selected_paths)
    parent = {value: value for value in local_variables}
    for equality in compose_output.get("variable_equalities", []):
        left, right = str(equality["left"]), str(equality["right"])
        if left in parent and right in parent:
            _union(parent, left, right)
    global_names: dict[str, str] = {}
    for local in local_variables:
        root = _find(parent, local)
        global_names.setdefault(root, f"V{len(global_names)}")
    anchor_ids = {
        str(path["anchor_ref"]): grounded.anchor_bindings[str(path["anchor_ref"])].entity_id
        for path in selected_paths
    }
    triples: list[list[str]] = []
    for path in selected_paths:
        for step in path["steps"]:
            relation_id = grounded.relation_bindings.get(str(step["id"]))
            if not relation_id:
                raise LoweringError(f"missing grounded relation for {step['id']}")
            left = _resolve_ref(str(step["from"]), anchor_ids, parent, global_names)
            right = _resolve_ref(str(step["to"]), anchor_ids, parent, global_names)
            if str(step["direction"]) == "forward":
                triples.append([left, relation_id, right])
            else:
                triples.append([right, relation_id, left])
    answer_local = str(compose_output["answer_var"])
    answer_var = _resolve_ref(answer_local, {}, parent, global_names)
    if not answer_var.startswith("V"):
        raise LoweringError("answer_var must lower to a variable")
    operators = _lower_operators(
        compose_output.get("operators", []),
        answer_var=answer_var,
        triples=triples,
        parent=parent,
        global_names=global_names,
        anchor_values=_anchor_values(grounded),
    )
    if not any(answer_var in (triple[0], triple[2]) for triple in triples):
        raise LoweringError("answer_var does not occur in a triple")
    return QueryGraphCandidate(
        graph_id=graph_id,
        triples=triples,
        answer_var=answer_var,
        operators=operators,
        score=grounded.score,
        compose_output=deepcopy(compose_output),
        provenance={
            "version": str(pipeline_version),
            "anchor_bindings": {
                key: {
                    "id": value.entity_id,
                    "label": value.label,
                    "score": value.score,
                }
                for key, value in grounded.anchor_bindings.items()
            },
            "relation_bindings": dict(grounded.relation_bindings),
            "grounded_semantic": grounded.provenance,
        },
    )


def lower_sparql(graph: QueryGraphCandidate, *, limit: int | None = None) -> str:
    variables = {value for triple in graph.triples for value in (triple[0], triple[2]) if value.startswith("V")}
    answer = _var(graph.answer_var)
    lines: list[str] = []
    select = f"SELECT DISTINCT {answer}"
    for operator in graph.operators:
        operator_type = str(operator.get("type", "")).upper()
        if operator_type == "COUNT":
            input_var = _var(str(operator.get("input_var") or graph.answer_var))
            select = f"SELECT (COUNT(DISTINCT {input_var}) AS {answer})"
        elif operator_type in {"ARGMAX", "ARGMIN"}:
            select = f"SELECT DISTINCT {answer}"
    lines.append(select)
    lines.append("WHERE {")
    for subject, relation, object_ in graph.triples:
        lines.append(f"  {_term(subject)} <{_relation_uri(relation)}> {_term(object_)} .")
    filter_index = 0
    order_clause = ""
    projected_order_values: list[str] = []
    for operator in graph.operators:
        operator_type = str(operator.get("type", "")).upper()
        input_var = str(operator.get("input_var") or graph.answer_var)
        relation_id = _operator_relation_id(operator)
        if operator_type in {"ARGMAX", "ARGMIN"} and relation_id:
            order_var = f"?order{filter_index}"
            if operator.get("_project_order_value"):
                projected_order_values.append(order_var)
            lines.append(f"  {_var(input_var)} <{_relation_uri(relation_id)}> {order_var} .")
            if operator.get("_numeric_order_cast"):
                cast = (
                    "bif:atof"
                    if operator.get("_numeric_order_cast") == "float"
                    else "bif:atoi"
                )
                expression = f"{cast}(STR({order_var}))"
            else:
                expression = (
                    f"STR({order_var})"
                    if operator.get("_order_temporal_lexical")
                    else order_var
                )
            order_clause = f"ORDER BY {'DESC' if operator_type == 'ARGMAX' else 'ASC'}({expression})"
            filter_index += 1
        elif (
            operator_type in {"ARGMAX", "ARGMIN"}
            and operator.get("_order_bound_value")
        ):
            order_var = _var(input_var)
            expression = (
                (
                    f"bif:atof(STR({order_var}))"
                    if operator.get("_numeric_order_cast") == "float"
                    else f"bif:atoi(STR({order_var}))"
                )
                if operator.get("_numeric_order_cast")
                else order_var
            )
            order_clause = f"ORDER BY {'DESC' if operator_type == 'ARGMAX' else 'ASC'}({expression})"
        elif operator_type == "TC" and relation_id:
            time_var = f"?time{filter_index}"
            value = str(operator.get("value", "")).strip()
            counterpart = _temporal_counterpart(relation_id)
            counterpart_var = f"?time_end{filter_index}"
            if operator.get("_temporal_exists_semantics"):
                lines.extend(
                    _temporal_exists_filters(
                        input_var=input_var,
                        relation_id=relation_id,
                        counterpart=counterpart,
                        time_var=time_var,
                        counterpart_var=counterpart_var,
                        value=value,
                    )
                )
            elif counterpart:
                lines.append(
                    f"  OPTIONAL {{ {_var(input_var)} <{_relation_uri(relation_id)}> {time_var} . }}"
                )
                lines.append(
                    f"  OPTIONAL {{ {_var(input_var)} <{_relation_uri(counterpart)}> {counterpart_var} . }}"
                )
            else:
                lines.append(f"  {_var(input_var)} <{_relation_uri(relation_id)}> {time_var} .")
            if not operator.get("_temporal_exists_semantics"):
                temporal_filter = _temporal_filter(
                    relation_id,
                    time_var,
                    counterpart_var if counterpart else "",
                    value,
                )
                if temporal_filter:
                    lines.append(f"  FILTER({temporal_filter})")
            filter_index += 1
        elif operator_type in {"GREATER_THAN", "LESS_THAN", "GREATER_OR_EQUAL", "LESS_OR_EQUAL"}:
            op = {
                "GREATER_THAN": ">",
                "LESS_THAN": "<",
                "GREATER_OR_EQUAL": ">=",
                "LESS_OR_EQUAL": "<=",
            }[operator_type]
            comparison_var = f"?comparison{filter_index}"
            if relation_id:
                lines.append(
                    f"  {_var(input_var)} <{_relation_uri(relation_id)}> {comparison_var} ."
                )
                comparison_target = comparison_var
            else:
                comparison_target = _var(input_var)
            calendar_day = (
                _calendar_date(str(operator.get("value", "")))
                if operator.get("_calendar_day_boundary")
                else ""
            )
            if operator.get("_numeric_comparison_cast"):
                value = str(operator.get("value", ""))
                if not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
                    raise LoweringError("Numeric comparison requires a numeric value")
                cast = (
                    "bif:atof"
                    if operator.get("_numeric_cast_kind") == "float"
                    else "bif:atoi"
                )
                lines.append(f"  FILTER({cast}(STR({comparison_target})) {op} {value})")
            elif operator.get("_comparison_date_precision") == "year":
                year = str(operator.get("value", ""))
                if not re.fullmatch(r"[12]\d{3}", year):
                    raise LoweringError("Date year comparison requires a four-digit year")
                lines.append(f"  FILTER(<http://www.w3.org/2001/XMLSchema#integer>(SUBSTR(STR({comparison_target}), 1, 4)) {op} {year})")
            elif operator.get("_comparison_date_precision") == "month":
                month = str(operator.get("value", ""))
                lines.append(
                    f"  FILTER(SUBSTR(STR({comparison_target}), 1, 7) {op} {_literal(month, '')})"
                )
            elif calendar_day:
                day_value = _literal(calendar_day, "")
                lines.append(
                    f"  FILTER(SUBSTR(STR({comparison_target}), 1, 10) {op} {day_value})"
                )
            else:
                value = _operator_value_term(operator)
                lines.append(f"  FILTER({comparison_target} {op} {value})")
            filter_index += 1
        elif operator_type == "EQUAL":
            equality_target = _operator_value_term(operator)
            date_precision = str(operator.get("_comparison_date_precision", ""))
            if relation_id:
                equality_var = f"?equal{filter_index}"
                lines.append(
                    f"  {_var(input_var)} <{_relation_uri(relation_id)}> {equality_var} ."
                )
                if date_precision in {"year", "month", "day"}:
                    width = {"year": 4, "month": 7, "day": 10}[date_precision]
                    equality_target = (
                        f"SUBSTR(STR({equality_var}), 1, {width}) = "
                        f"{_literal(str(operator.get('value', '')), '')}"
                    )
                elif operator.get("_numeric_cast_compare"):
                    numeric_value = float(str(operator.get("value", "0")))
                    tolerance = max(0.000001, abs(numeric_value) * 0.000001)
                    equality_target = (
                        f"ABS({('bif:atof' if operator.get('_numeric_cast_kind') == 'float' else 'bif:atoi')}(STR({equality_var})) - {numeric_value:.15g}) "
                        f"< {tolerance:.15g}"
                    )
                else:
                    equality_target = f"{equality_var} = {equality_target}"
            elif date_precision in {"year", "month", "day"}:
                width = {"year": 4, "month": 7, "day": 10}[date_precision]
                equality_target = (
                    f"SUBSTR(STR({_var(input_var)}), 1, {width}) = "
                    f"{_literal(str(operator.get('value', '')), '')}"
                )
            elif operator.get("_numeric_cast_compare"):
                numeric_value = float(str(operator.get("value", "0")))
                tolerance = max(0.000001, abs(numeric_value) * 0.000001)
                equality_target = (
                    f"ABS({('bif:atof' if operator.get('_numeric_cast_kind') == 'float' else 'bif:atoi')}(STR({_var(input_var)})) - {numeric_value:.15g}) "
                    f"< {tolerance:.15g}"
                )
            else:
                equality_target = f"{_var(input_var)} = {equality_target}"
            lines.append(f"  FILTER({equality_target})")
            filter_index += 1
        elif operator_type == "NO_EQUAL":
            entity_id = str(operator.get("_value_entity_id", "")).strip()
            value_type = str(operator.get("value_type", "")).casefold()
            value = str(operator.get("value", "")).strip()
            if entity_id or (value_type in {"mid", "id"} and re.fullmatch(r"m\.[A-Za-z0-9_]+", value)):
                lines.append(f"  FILTER({_var(input_var)} != {_operator_value_term(operator)})")
            elif value_type == "entity" and value:
                label_var = f"?excluded_label{filter_index}"
                lines.append(
                    "  FILTER NOT EXISTS { "
                    f"{_var(input_var)} <http://rdf.freebase.com/ns/type.object.name> {label_var} . "
                    f"FILTER(LCASE(STR({label_var})) = LCASE({_literal(value, '')})) "
                    "}"
                )
            else:
                lines.append(f"  FILTER({_var(input_var)} != {_operator_value_term(operator)})")
            filter_index += 1
    if projected_order_values:
        lines[0] = " ".join(
            [select, *dict.fromkeys(projected_order_values)]
        )
    lines.append("}")
    if order_clause:
        lines.append(order_clause)
        lines.append("LIMIT 1")
    elif limit is not None and int(limit) > 0:
        lines.append(f"LIMIT {int(limit)}")
    return "\n".join(lines)


def _lower_operators(
    operators: list[dict[str, Any]],
    *,
    answer_var: str,
    triples: list[list[str]],
    parent: dict[str, str],
    global_names: dict[str, str],
    anchor_values: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    anchor_values = anchor_values or {}
    lowered: list[dict[str, Any]] = []
    for raw in operators:
        if not isinstance(raw, dict):
            continue
        value = deepcopy(raw)
        operator_type = str(value.get("type", "")).upper()
        if operator_type == "AND":
            continue
        raw_input = str(value.get("input_var", ""))
        relation_id = _operator_relation_id(value)
        relation_id = _ground_operator_relation(
            relation_id,
            raw_input=raw_input,
            answer_var=answer_var,
            triples=triples,
            parent=parent,
            global_names=global_names,
        )
        if relation_id:
            value["attribute_relation_id"] = relation_id
        if raw_input in parent:
            value["input_var"] = global_names[_find(parent, raw_input)]
        else:
            value["input_var"] = _infer_operator_input(
                relation_id,
                triples,
                answer_var,
            )
        # Operator may point at a scalar variable that Semantic/Compose has
        # already reached through the same attribute relation.  Re-applying
        # that relation would create an impossible scalar -> attribute hop.
        # Keep the model contract unchanged and lower the comparison directly
        # on the already-bound value instead.
        lowered_input = str(value.get("input_var", ""))
        if relation_id and any(
            relation == relation_id and object_ == lowered_input
            for _, relation, object_ in triples
        ):
            value["attribute_relation_id"] = ""
            value["attribute_relation_label"] = []
            value["attribute_relation_labels"] = []
            value["_attribute_value_already_bound"] = True
            try:
                float(str(value.get("value", "")).strip())
            except (TypeError, ValueError):
                pass
            else:
                value["_numeric_cast_compare"] = True
        operator_value = str(value.get("value", "")).strip().casefold()
        if operator_value and operator_value in anchor_values:
            value["_value_entity_id"] = anchor_values[operator_value]
        value["type"] = operator_type
        if (
            operator_type
            in {"GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL"}
            and str(value.get("value_type", "")).casefold() == "year"
        ):
            value["_comparison_date_precision"] = "year"
        if operator_type == "TC" and relation_id:
            # TC represents Gold-style permissive temporal bounds: a missing
            # boundary is allowed, while an existing boundary must satisfy the
            # requested interval. Keep this internal so the training JSON
            # contract remains unchanged.
            value["_temporal_exists_semantics"] = True
        lowered.append(value)
    return lowered


def _operator_relation_id(operator: dict[str, Any]) -> str:
    relation_id = str(operator.get("attribute_relation_id", ""))
    if relation_id:
        return relation_id
    label = operator.get("attribute_relation_label", [])
    if not label:
        label = operator.get("attribute_relation_labels", [])
    return _label_to_relation(label) if label else ""


def _ground_operator_relation(
    relation_id: str,
    *,
    raw_input: str,
    answer_var: str,
    triples: list[list[str]],
    parent: dict[str, str],
    global_names: dict[str, str],
) -> str:
    """Map generic temporal labels onto the grounded relation namespace.

    The operator model sees human-readable semantic paths and may call a
    position's start field ``time.event.start_date``. The grounded path already
    identifies the owning Freebase type, so preserve that prefix and only use
    the temporal suffix predicted by the model.
    """
    if not relation_id or raw_input not in parent:
        return relation_id
    generic_temporal = {
        "start": "from",
        "start_date": "from",
        "begin": "from",
        "begin_date": "from",
        "end": "to",
        "end_date": "to",
        "finish": "to",
        "finish_date": "to",
    }
    suffix = generic_temporal.get(relation_id.rsplit(".", 1)[-1].casefold())
    if suffix is None or not relation_id.startswith("time."):
        return relation_id
    input_var = global_names[_find(parent, raw_input)]
    # A generic event date on the answer entity is already a complete
    # Freebase predicate (for example time.event.end_date). Namespace
    # transfer is only for intermediate tenure/roster records whose concrete
    # temporal properties are represented as from/to.
    if input_var == answer_var:
        return relation_id
    for subject, grounded_relation, _ in triples:
        if subject != input_var:
            continue
        parts = grounded_relation.split(".")
        if len(parts) >= 3:
            return ".".join(parts[:2] + [suffix])
    return relation_id


def _anchor_values(grounded: GroundedSemanticCandidate) -> dict[str, str]:
    values: dict[str, str] = {}
    for anchor_id, entity in grounded.anchor_bindings.items():
        values[str(anchor_id).casefold()] = entity.entity_id
        values[str(entity.label).strip().casefold()] = entity.entity_id
    for anchor in grounded.compose_input.get("anchors", []):
        if not isinstance(anchor, dict):
            continue
        anchor_id = str(anchor.get("id", ""))
        surface = str(anchor.get("surface", "")).strip()
        entity = grounded.anchor_bindings.get(anchor_id)
        if entity is not None and surface:
            values[surface.casefold()] = entity.entity_id
    return values


def _operator_value_term(operator: dict[str, Any]) -> str:
    entity_id = str(operator.get("_value_entity_id", "")).strip()
    value = str(operator.get("value", "")).strip()
    value_type = str(operator.get("value_type", ""))
    if value_type.casefold() == "variable" and re.fullmatch(r"V\d+", value):
        return _var(value)
    if value_type.casefold() == "uri" and re.fullmatch(r"https?://[^\s<>]+", value):
        return f"<{value}>"
    if not entity_id and value_type.casefold() in {"entity", "mid", "id"}:
        if re.fullmatch(r"m\.[A-Za-z0-9_]+", value):
            entity_id = value
    if entity_id:
        return _term(entity_id)
    return _literal(value, value_type)


def _ordered_variables(paths: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for path in paths:
        for step in path["steps"]:
            value = str(step["to"])
            if value not in values:
                values.append(value)
    return values


def _resolve_ref(value: str, anchors: dict[str, str], parent: dict[str, str], global_names: dict[str, str]) -> str:
    if value in anchors:
        return anchors[value]
    if value not in parent:
        raise LoweringError(f"unknown graph reference: {value}")
    return global_names[_find(parent, value)]


def _infer_operator_input(
    attribute_relation: str,
    triples: list[list[str]],
    answer_var: str,
) -> str:
    prefix = ".".join(attribute_relation.split(".")[:2])
    if prefix:
        for subject, relation, _ in triples:
            if ".".join(relation.split(".")[:2]) == prefix and subject.startswith("V"):
                return subject
    return answer_var


def _find(parent: dict[str, str], value: str) -> str:
    if parent[value] != value:
        parent[value] = _find(parent, parent[value])
    return parent[value]


def _union(parent: dict[str, str], left: str, right: str) -> None:
    left_root, right_root = _find(parent, left), _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _term(value: str) -> str:
    return _var(value) if value.startswith("V") else f"<http://rdf.freebase.com/ns/{value}>"


def _var(value: str) -> str:
    return value if value.startswith("?") else f"?{value}"


def _relation_uri(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", value):
        raise LoweringError(f"unsafe relation id: {value}")
    return f"http://rdf.freebase.com/ns/{value}"


def _label_to_relation(label: Any) -> str:
    if not isinstance(label, list) or not label:
        return ""
    parts = [re.sub(r"[^A-Za-z0-9]+", "_", str(item).strip()).strip("_") for item in label]
    return ".".join(part for part in parts if part)


def _literal(value: str, value_type: str) -> str:
    if value_type.casefold() in {"date", "datetime", "date_time"}:
        normalized = value.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
            normalized = f"{normalized}T00:00:00Z"
        escaped = normalized.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"^^<http://www.w3.org/2001/XMLSchema#dateTime>'
    if value_type.casefold() in {"year", "integer", "number", "float"} and re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _year_literal(value: str) -> str:
    match = re.search(r"\b(\d{4})\b", value)
    return match.group(1) if match else "0"


def _calendar_date(value: str) -> str:
    match = re.search(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", str(value))
    return match.group(1) if match else ""


def _temporal_counterpart(relation_id: str) -> str:
    replacements = (
        (".from_date", ".to_date"),
        (".to_date", ".from_date"),
        (".start_date", ".end_date"),
        (".end_date", ".start_date"),
        (".from", ".to"),
        (".to", ".from"),
        (".start", ".end"),
        (".end", ".start"),
    )
    for source, target in replacements:
        if relation_id.endswith(source):
            return f"{relation_id[: -len(source)]}{target}"
    return ""


def _temporal_filter(
    relation_id: str,
    time_var: str,
    counterpart_var: str,
    value: str,
) -> str:
    is_end = relation_id.endswith((".to", ".to_date", ".end", ".end_date"))
    interval_parts = str(value).split("/", 1)
    if counterpart_var and len(interval_parts) == 2:
        interval_start = _calendar_date(interval_parts[0])
        interval_end = _calendar_date(interval_parts[1])
        if interval_start and interval_end:
            item_start = counterpart_var if is_end else time_var
            item_end = time_var if is_end else counterpart_var
            start_value = _literal(interval_start, "")
            end_value = _literal(interval_end, "")
            return (
                f"(!BOUND({item_start}) || "
                f"SUBSTR(STR({item_start}), 1, 10) <= {end_value}) && "
                f"(!BOUND({item_end}) || "
                f"SUBSTR(STR({item_end}), 1, 10) >= {start_value})"
            )
    if value.upper() == "NOW":
        current = "NOW()"
        if is_end:
            return f"{time_var} >= {current}" if not counterpart_var else f"(!BOUND({time_var}) || {time_var} >= {current}) && (!BOUND({counterpart_var}) || {counterpart_var} <= {current})"
        return f"{time_var} <= {current}" if not counterpart_var else f"(!BOUND({time_var}) || {time_var} <= {current}) && (!BOUND({counterpart_var}) || {counterpart_var} >= {current})"
    year = _year_literal(value)
    if year == "0":
        return ""
    if is_end:
        return f"{_lexical_year(time_var)} >= {year}" if not counterpart_var else f"(!BOUND({time_var}) || {_lexical_year(time_var)} >= {year}) && (!BOUND({counterpart_var}) || {_lexical_year(counterpart_var)} <= {year})"
    return f"{_lexical_year(time_var)} <= {year}" if not counterpart_var else f"(!BOUND({time_var}) || {_lexical_year(time_var)} <= {year}) && (!BOUND({counterpart_var}) || {_lexical_year(counterpart_var)} >= {year})"


def _temporal_exists_filters(
    *,
    input_var: str,
    relation_id: str,
    counterpart: str,
    time_var: str,
    counterpart_var: str,
    value: str,
) -> list[str]:
    """Emit exact Gold-style NOT EXISTS/EXISTS temporal boundary filters."""
    owner = _var(input_var)
    is_end = relation_id.endswith((".to", ".to_date", ".end", ".end_date"))
    interval = value.split("/", 1)

    def boundary_filter(relation: str, variable: str, op: str, boundary: str) -> str:
        date = _calendar_date(boundary)
        if date:
            condition = f"SUBSTR(STR({variable}), 1, 10) {op} {_literal(date, '')}"
        else:
            year = _year_literal(boundary)
            condition = f"{_lexical_year(variable)} {op} {year}"
        missing = f"?missing_{variable.lstrip('?')}"
        return (
            "  FILTER(NOT EXISTS { "
            f"{owner} <{_relation_uri(relation)}> {missing} . "
            "} || EXISTS { "
            f"{owner} <{_relation_uri(relation)}> {variable} . "
            f"FILTER({condition}) "
            "})"
        )

    if counterpart and len(interval) == 2:
        start_value, end_value = interval
        start_relation = counterpart if is_end else relation_id
        end_relation = relation_id if is_end else counterpart
        start_var = counterpart_var if is_end else time_var
        end_var = time_var if is_end else counterpart_var
        return [
            boundary_filter(start_relation, start_var, "<=", end_value),
            boundary_filter(end_relation, end_var, ">=", start_value),
        ]
    op = ">=" if is_end else "<="
    return [boundary_filter(relation_id, time_var, op, value)]


def _lexical_year(variable):
    return f"<http://www.w3.org/2001/XMLSchema#integer>(SUBSTR(STR({variable}), 1, 4))"
