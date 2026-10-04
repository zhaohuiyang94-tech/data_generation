from __future__ import annotations

from copy import deepcopy
from itertools import permutations, product
import json
import re
from typing import Any


def _goal_clauses(goal: str) -> list[str]:
    """Split a path goal at natural-language sentence boundaries."""
    text = " ".join(str(goal).split())
    if not text:
        return []
    clauses: list[str] = []
    start = 0
    for index, character in enumerate(text):
        if character not in ".?!":
            continue
        if (
            character == "."
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        ):
            continue
        clause = text[start : index + 1].strip()
        if clause:
            clauses.append(clause)
        start = index + 1
    tail = text[start:].strip()
    if tail:
        clauses.append(tail)
    return clauses


def _truncate_path(
    path: dict[str, Any],
    retained_hops: int,
) -> dict[str, Any]:
    """Truncate one path and keep Compose's goal consistent with its steps."""
    steps = path.get("steps", [])
    original_hops = len(steps) if isinstance(steps, list) else 0
    retained_hops = max(1, min(int(retained_hops), original_hops))
    clauses = _goal_clauses(str(path.get("goal", "")))
    retained_goal = " ".join(clauses[:retained_hops]).strip()
    removed_goal = " ".join(clauses[retained_hops:]).strip()
    if retained_goal:
        path["goal"] = retained_goal
    path["steps"] = steps[:retained_hops]
    path["path_output_var"] = str(path["steps"][-1]["to"])
    return {
        "path_id": str(path.get("id", "")),
        "original_hops": original_hops,
        "repaired_hops": retained_hops,
        "retained_goal": str(path.get("goal", "")),
        "removed_goal": removed_goal,
    }


def _prune_unused_anchors(graph: dict[str, Any]) -> None:
    used_anchors = {
        str(path.get("anchor_ref", ""))
        for path in graph.get("semantic_paths", [])
        if isinstance(path, dict)
    }
    graph["anchors"] = [
        anchor
        for anchor in graph.get("anchors", [])
        if isinstance(anchor, dict) and str(anchor.get("id", "")) in used_anchors
    ]


def build_prefix_repair_graphs(
    semantic_graphs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create structurally simpler path variants without calling a model.

    The repair is deliberately independent of question entities and domains.
    It only removes an ungroundable suffix; it never invents relations.
    """
    repaired: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    seen: set[str] = set()
    for graph_index, graph in enumerate(semantic_graphs):
        paths = graph.get("semantic_paths", [])
        if not isinstance(paths, list) or not paths:
            continue
        pivot_candidate = deepcopy(graph)
        pivot_changes: list[dict[str, int | str]] = []
        for path in pivot_candidate.get("semantic_paths", []):
            steps = path.get("steps", [])
            if not isinstance(steps, list) or len(steps) < 2:
                continue
            directions = [str(step.get("direction", "")).casefold() for step in steps]
            pivot = next(
                (
                    index
                    for index in range(1, len(directions))
                    if directions[index] != directions[index - 1]
                ),
                0,
            )
            if pivot <= 0:
                continue
            pivot_changes.append(_truncate_path(path, pivot))
        if pivot_changes:
            _prune_unused_anchors(pivot_candidate)
            key = json.dumps(pivot_candidate, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                repaired.append(pivot_candidate)
                plans.append(
                    {
                        "source_graph_index": graph_index,
                        "strategy": "first_direction_change",
                        "changed_paths": pivot_changes,
                    }
                )
            continue
        longest = max(
            (len(path.get("steps", [])) for path in paths if isinstance(path, dict)),
            default=0,
        )
        for prefix_length in range(longest - 1, 0, -1):
            candidate = deepcopy(graph)
            changed_paths: list[dict[str, int | str]] = []
            for path in candidate.get("semantic_paths", []):
                steps = path.get("steps", [])
                if not isinstance(steps, list) or len(steps) <= prefix_length:
                    continue
                changed_paths.append(_truncate_path(path, prefix_length))
            if not changed_paths:
                continue
            _prune_unused_anchors(candidate)
            key = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            repaired.append(candidate)
            plans.append(
                {
                    "source_graph_index": graph_index,
                    "prefix_length": prefix_length,
                    "changed_paths": changed_paths,
                }
            )
    return repaired, plans


def _anchor_assignment(
    template_surfaces: list[str],
    current_surfaces: list[str],
    ranker: Any,
) -> list[str] | None:
    if not template_surfaces or len(template_surfaces) > len(current_surfaces):
        return None
    bounded_surfaces = current_surfaces
    scores: list[list[float]] = []
    for template_surface in template_surfaces:
        scores.append(ranker.score(template_surface, bounded_surfaces))
    best: tuple[float, tuple[int, ...]] | None = None
    for indexes in permutations(
        range(len(bounded_surfaces)),
        len(template_surfaces),
    ):
        score = sum(scores[row][column] for row, column in enumerate(indexes))
        candidate = (score, indexes)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    return [bounded_surfaces[index] for index in best[1]]


def build_semantic_template_repair_graphs(
    *,
    question: str,
    decompositions: list[str] | None = None,
    semantic_graphs: list[dict[str, Any]] | None = None,
    linked_entities: dict[str, str],
    semantic_examples: list[dict[str, Any]],
    ranker: Any,
    preselect: int = 128,
    limit: int = 24,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Instantiate semantically nearest accepted training structures.

    The repair path deliberately uses the embedding model for both retrieval
    and anchor assignment.  It does not mask entity strings, tokenize text,
    or use lexical overlap.  Retrieval compares the entity-independent
    relation/direction description emitted by the existing Semantic call;
    reviewed decomposition text is only a last-resort query when that
    description is unavailable.
    """
    current_surfaces = sorted(
        {
            str(value).strip()
            for value in linked_entities.values()
            if str(value).strip()
        }
    )
    if not current_surfaces or not semantic_examples:
        return [], []
    query_decomposition = [
        str(item).strip()
        for item in (decompositions or [])
        if str(item).strip()
    ]

    def graph_description(graph: dict[str, Any]) -> str:
        relation_steps = []
        for path in graph.get("semantic_paths", []):
            if not isinstance(path, dict):
                continue
            for step in path.get("steps", []):
                if not isinstance(step, dict):
                    continue
                label = " ".join(str(part) for part in step.get("relation_label", []))
                direction = str(step.get("direction", "")).strip()
                if label:
                    relation_steps.append(" ".join(value for value in (direction, label) if value))
        return " ".join(relation_steps).strip()

    query_relation_description = " ".join(
        graph_description(graph)
        for graph in (semantic_graphs or [])
        if isinstance(graph, dict)
    ).strip()
    semantic_documents: list[tuple[int, str, str, dict[str, Any]]] = []
    for index, example in enumerate(semantic_examples):
        graph = example.get("semantic_graph", {})
        if not isinstance(graph, dict):
            continue
        example_question = str(example.get("question", "")).strip()
        if not example_question:
            example_question = " ".join(
                str(item).strip()
                for item in example.get("decomposition", [])
                if str(item).strip()
            )
        semantic_documents.append(
            (index, example_question, graph_description(graph), graph)
        )
    if not semantic_documents:
        return [], []

    question_scores = ranker.score(
        question or " ".join(query_decomposition),
        [item[1] for item in semantic_documents],
    )
    relation_scores = (
        ranker.score(
            query_relation_description,
            [item[2] for item in semantic_documents],
        )
        if query_relation_description
        else [0.0] * len(semantic_documents)
    )
    ranked = sorted(
        (
            (
                (0.8 * float(question_score)) + (0.2 * float(relation_score)),
                item[0],
                item[3],
            )
            for question_score, relation_score, item in zip(
                question_scores,
                relation_scores,
                semantic_documents,
            )
        ),
        key=lambda item: (-item[0], item[1]),
    )
    ranked = ranked[: max(1, int(preselect))]

    graphs: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    seen: set[str] = set()
    for template_score, example_index, source_graph in ranked:
        graph = deepcopy(source_graph)
        anchors = [
            anchor
            for anchor in graph.get("anchors", [])
            if isinstance(anchor, dict)
        ]
        assignment = _anchor_assignment(
            [str(anchor.get("surface", "")) for anchor in anchors],
            current_surfaces,
            ranker,
        )
        if assignment is None:
            continue
        for anchor, surface in zip(anchors, assignment):
            anchor["surface"] = surface
        for path in graph.get("semantic_paths", []):
            if isinstance(path, dict):
                path["goal"] = question
        key = json.dumps(graph, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        graphs.append(graph)
        plans.append(
            {
                "source_graph_index": example_index,
                "strategy": "accepted_semantic_template",
                "template_score": template_score,
                "changed_paths": [],
                "compose_mode": "model",
            }
        )
        if len(graphs) >= max(1, int(limit)):
            break
    return graphs, plans


def build_structural_extension_graphs(
    semantic_graphs: list[dict[str, Any]],
    *,
    max_extra_hops: int = 2,
    max_total_hops: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enumerate bounded direction-only continuations without schema guesses."""
    graphs: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    seen: set[str] = set()
    for graph_index, source_graph in enumerate(semantic_graphs):
        source_paths = source_graph.get("semantic_paths", [])
        for path_index, source_path in enumerate(source_paths):
            if not isinstance(source_path, dict):
                continue
            source_steps = [
                step
                for step in source_path.get("steps", [])
                if isinstance(step, dict)
            ]
            if not source_steps:
                continue
            available = max_total_hops - len(source_steps)
            for extra_hops in range(1, min(max_extra_hops, available) + 1):
                for directions in product(
                    ("forward", "backward"),
                    repeat=extra_hops,
                ):
                    graph = deepcopy(source_graph)
                    path = graph["semantic_paths"][path_index]
                    steps = path["steps"]
                    relation_label = deepcopy(
                        source_steps[-1].get("relation_label", [])
                    )
                    for offset, direction in enumerate(directions):
                        step_index = len(source_steps) + offset
                        previous = str(steps[-1]["to"])
                        target = f"{path['id']}.V{step_index}"
                        steps.append(
                            {
                                "id": f"{path['id']}.S{step_index}",
                                "relation_label": deepcopy(relation_label),
                                "direction": direction,
                                "from": previous,
                                "to": target,
                            }
                        )
                    path["path_output_var"] = str(steps[-1]["to"])
                    key = json.dumps(graph, ensure_ascii=False, sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key)
                    graphs.append(graph)
                    plans.append(
                        {
                            "source_graph_index": graph_index,
                            "strategy": "bounded_structural_extension",
                            "path_id": str(path.get("id", "")),
                            "extra_hops": extra_hops,
                            "directions": list(directions),
                            "changed_paths": [],
                        }
                    )
    return graphs, plans


_TEMPORAL_SUFFIX_PAIRS = (
    ("from", "to"),
    ("start_date", "end_date"),
    ("start", "end"),
    ("begin_date", "end_date"),
    ("begin", "end"),
)


def _calendar_date(value: Any) -> str:
    match = re.search(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", str(value))
    return match.group(1) if match else ""


def discover_temporal_entity_intervals(
    kg: Any,
    entities: dict[str, str],
) -> list[dict[str, str]]:
    """Read start/end pairs for already linked entities from the KG.

    This is only graph inspection. It neither calls a language model nor
    assumes a particular entity type or relation namespace.
    """
    intervals: list[dict[str, str]] = []
    for entity_id, label in sorted(entities.items()):
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", str(entity_id)):
            continue
        suffix_filters = " || ".join(
            f'STRENDS(STR(?relation), ".{suffix}")'
            for pair in _TEMPORAL_SUFFIX_PAIRS
            for suffix in pair
        )
        query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation ?value WHERE {{
  ns:{entity_id} ?relation ?value .
  FILTER(isLiteral(?value))
  FILTER({suffix_filters})
}}
""".strip()
        try:
            rows = kg.execute(query)
        except (RuntimeError, ValueError):
            continue
        by_relation = {
            str(row.get("relation", "")).replace(
                "http://rdf.freebase.com/ns/", "", 1
            ): _calendar_date(row.get("value", ""))
            for row in rows
            if isinstance(row, dict) and _calendar_date(row.get("value", ""))
        }
        for start_suffix, end_suffix in _TEMPORAL_SUFFIX_PAIRS:
            found = False
            for start_relation, start in sorted(by_relation.items()):
                marker = f".{start_suffix}"
                if not start_relation.endswith(marker):
                    continue
                prefix = start_relation[: -len(marker)]
                end_relation = f"{prefix}.{end_suffix}"
                end = by_relation.get(end_relation, "")
                if not end:
                    continue
                intervals.append(
                    {
                        "entity_id": str(entity_id),
                        "surface": str(label),
                        "start": start,
                        "end": end,
                        "start_relation_id": start_relation,
                        "end_relation_id": end_relation,
                    }
                )
                found = True
                break
            if found:
                break
    return intervals


def _relation_id(label: Any) -> str:
    if not isinstance(label, list):
        return ""
    parts = [
        re.sub(r"[^A-Za-z0-9]+", "_", str(item).strip()).strip("_")
        for item in label
    ]
    return ".".join(part for part in parts if part)


def _relation_label(relation_id: str) -> list[str]:
    return [part.replace("_", " ") for part in relation_id.split(".")]


def _temporal_owner_relations(
    compose_output: dict[str, Any],
    ontology: Any,
) -> dict[str, tuple[str, str]]:
    relation_ids = set(getattr(ontology, "relation_ids", ()))
    owners: dict[str, tuple[str, str]] = {}
    for triple in compose_output.get("triples", []):
        if not isinstance(triple, dict):
            continue
        subject = str(triple.get("subject", ""))
        relation_id = _relation_id(triple.get("relation_label", []))
        if not subject.startswith("V") or "." not in relation_id:
            continue
        prefix = relation_id.rsplit(".", 1)[0]
        for start_suffix, end_suffix in _TEMPORAL_SUFFIX_PAIRS:
            start_relation = f"{prefix}.{start_suffix}"
            end_relation = f"{prefix}.{end_suffix}"
            if start_relation in relation_ids and end_relation in relation_ids:
                owners.setdefault(subject, (start_relation, end_relation))
                break
    return owners


def repair_temporal_operator_output(
    *,
    question: str,
    compose_output: dict[str, Any],
    operator_output: dict[str, Any],
    repair_context: dict[str, Any],
    ontology: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Correct temporal operator placement using graph schema and KG intervals.

    The function runs only on candidates produced by the no-grounding repair
    branch. It is generic over entity and relation domains.
    """
    output = deepcopy(operator_output)
    operators = [
        item
        for item in output.get("operators", [])
        if isinstance(item, dict)
    ]
    intervals = [
        item
        for item in repair_context.get("temporal_intervals", [])
        if isinstance(item, dict)
        and _calendar_date(item.get("start", ""))
        and _calendar_date(item.get("end", ""))
    ]
    owners = _temporal_owner_relations(compose_output, ontology)
    trace: dict[str, Any] = {
        "status": "not_applicable",
        "temporal_owner_vars": sorted(owners),
        "temporal_interval_count": len(intervals),
        "changes": [],
    }
    if not owners:
        output["operators"] = operators
        return output, trace

    owner_var = next(
        (
            str(item.get("input_var", ""))
            for item in operators
            if str(item.get("input_var", "")) in owners
            and str(item.get("type", "")).upper() == "TC"
        ),
        next(iter(owners)),
    )
    start_relation, _ = owners[owner_var]
    start_label = _relation_label(start_relation)

    for operator in operators:
        operator_type = str(operator.get("type", "")).upper()
        value_type = str(operator.get("value_type", "")).casefold()
        value = str(operator.get("value", ""))
        is_date_comparison = (
            operator_type
            in {"GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL"}
            and (
                value_type in {"date", "datetime", "date_time"}
                or bool(_calendar_date(value))
            )
        )
        if is_date_comparison and (
            str(operator.get("input_var", "")) not in owners
            or not operator.get("attribute_relation_label")
        ):
            before = {
                "input_var": str(operator.get("input_var", "")),
                "attribute_relation_label": deepcopy(
                    operator.get("attribute_relation_label", [])
                ),
            }
            operator["input_var"] = owner_var
            operator["attribute_relation_label"] = deepcopy(start_label)
            operator["_calendar_day_boundary"] = True
            trace["changes"].append(
                {
                    "kind": "place_date_comparison_on_temporal_owner",
                    "before": before,
                    "after": {
                        "input_var": owner_var,
                        "attribute_relation_label": start_label,
                    },
                }
            )

    interval = intervals[0] if intervals else None
    interval_value = (
        f"{_calendar_date(interval['start'])}/{_calendar_date(interval['end'])}"
        if interval is not None
        else ""
    )
    temporal_cue = bool(
        re.search(
            r"\b(during|while|when|at\s+the\s+time\s+of|concurrent(?:ly)?|overlap(?:ping)?)\b",
            question,
            flags=re.IGNORECASE,
        )
    )
    tc_operator = next(
        (
            item
            for item in operators
            if str(item.get("type", "")).upper() == "TC"
            and str(item.get("input_var", "")) in owners
        ),
        None,
    )
    if interval_value and (tc_operator is not None or temporal_cue):
        replacement = {
            "type": "TC",
            "inputs": [],
            "input_var": owner_var,
            "attribute_relation_label": deepcopy(start_label),
            "attribute_relation_labels": [],
            "value": interval_value,
            "value_type": "date_interval",
            "_calendar_day_boundary": True,
        }
        if tc_operator is None:
            operators.append(replacement)
            trace["changes"].append(
                {
                    "kind": "add_linked_event_interval",
                    "entity_id": str(interval.get("entity_id", "")),
                    "input_var": owner_var,
                    "value": interval_value,
                }
            )
        else:
            index = operators.index(tc_operator)
            operators[index] = replacement
            trace["changes"].append(
                {
                    "kind": "replace_point_tc_with_linked_event_interval",
                    "entity_id": str(interval.get("entity_id", "")),
                    "input_var": owner_var,
                    "value": interval_value,
                }
            )

    if trace["changes"]:
        trace["status"] = "changed"
    output["operators"] = operators
    return output, trace


def _words(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value).casefold())
        if token
        not in {
            "a",
            "an",
            "and",
            "is",
            "it",
            "of",
            "the",
            "to",
            "what",
            "which",
            "who",
        }
    }


def repair_terminal_cvt_operator_output(
    *,
    compose_output: dict[str, Any],
    operator_output: dict[str, Any],
    repair_context: dict[str, Any],
    ontology: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover a trimmed literal/CVT property from the local ontology."""
    output = deepcopy(operator_output)
    operators = [
        item
        for item in output.get("operators", [])
        if isinstance(item, dict)
    ]
    trace: dict[str, Any] = {
        "status": "not_applicable",
        "changes": [],
        "removed_invalid_operators": [],
    }
    if not all(
        callable(getattr(ontology, method, None))
        for method in (
            "domain_for_relation",
            "range_for_relation",
            "relations_for_domain",
        )
    ):
        output["operators"] = operators
        return output, trace
    rewrites = [
        item
        for item in repair_context.get("answer_var_rewrites", [])
        if isinstance(item, dict)
    ]
    plan_changes = {
        str(item.get("path_id", "")): item
        for item in repair_context.get("plan", {}).get("changed_paths", [])
        if isinstance(item, dict)
    }

    variable_types: dict[str, set[str]] = {}
    triple_by_relation: dict[str, dict[str, Any]] = {}
    for triple in compose_output.get("triples", []):
        if not isinstance(triple, dict):
            continue
        relation_id = _relation_id(triple.get("relation_label", []))
        if not relation_id:
            continue
        triple_by_relation.setdefault(relation_id, triple)
        subject = str(triple.get("subject", ""))
        object_ = str(triple.get("object", ""))
        domain = str(ontology.domain_for_relation(relation_id))
        range_id = str(ontology.range_for_relation(relation_id))
        if subject.startswith("V") and domain:
            variable_types.setdefault(subject, set()).add(domain)
        if object_.startswith("V") and range_id:
            variable_types.setdefault(object_, set()).add(range_id)

    for rewrite in rewrites:
        terminal_relation = str(rewrite.get("terminal_relation_id", ""))
        terminal_triple = triple_by_relation.get(terminal_relation)
        range_id = str(ontology.range_for_relation(terminal_relation))
        if terminal_triple is None or not range_id:
            continue
        cvt_var = str(terminal_triple.get("object", ""))
        if not cvt_var.startswith("V"):
            continue
        removed_goal = str(
            plan_changes.get(str(rewrite.get("path_id", "")), {}).get(
                "removed_goal",
                "",
            )
        )
        goal_words = _words(removed_goal)
        ranked_properties: list[tuple[float, str]] = []
        normalized_goal = " ".join(str(removed_goal).casefold().split())
        for relation_id in ontology.relations_for_domain(range_id):
            leaf = relation_id.rsplit(".", 1)[-1].replace("_", " ")
            leaf_words = _words(leaf)
            overlap = len(goal_words & leaf_words)
            if not overlap:
                continue
            score = overlap / max(1, len(leaf_words))
            if leaf in normalized_goal:
                score += 1.0
            ranked_properties.append((score, relation_id))
        if not ranked_properties:
            continue
        ranked_properties.sort(key=lambda item: (-item[0], item[1]))
        attribute_relation = ranked_properties[0][1]
        attribute_label = _relation_label(attribute_relation)
        relevant = [
            item
            for item in operators
            if (
                str(item.get("type", "")).upper()
                in {
                    "ARGMAX",
                    "ARGMIN",
                    "GREATER_THAN",
                    "GREATER_OR_EQUAL",
                    "LESS_THAN",
                    "LESS_OR_EQUAL",
                }
                or (
                    str(item.get("type", "")).upper() == "EQUAL"
                    and str(item.get("value_type", "")).casefold()
                    in {"number", "float", "integer", "date", "datetime", "date_time", "year"}
                )
            )
        ]
        for operator in relevant:
            before = {
                "input_var": str(operator.get("input_var", "")),
                "attribute_relation_label": deepcopy(
                    operator.get("attribute_relation_label", [])
                ),
            }
            operator["input_var"] = cvt_var
            operator["attribute_relation_label"] = deepcopy(attribute_label)
            if (
                str(operator.get("type", "")).upper() == "EQUAL"
                and str(operator.get("value_type", "")).casefold()
                in {"number", "float", "integer"}
            ):
                operator["_numeric_cast_compare"] = True
            trace["changes"].append(
                {
                    "kind": "restore_trimmed_cvt_attribute",
                    "before": before,
                    "after": {
                        "input_var": cvt_var,
                        "attribute_relation_label": attribute_label,
                    },
                }
            )

    retained: list[dict[str, Any]] = []
    ontology_relations = set(getattr(ontology, "relation_ids", ()))
    for operator in operators:
        label = operator.get("attribute_relation_label", [])
        relation_id = _relation_id(label) if label else ""
        input_var = str(operator.get("input_var", ""))
        if relation_id:
            domain = str(ontology.domain_for_relation(relation_id))
            known_types = variable_types.get(input_var, set())
            invalid = (
                relation_id not in ontology_relations
                or (domain and known_types and domain not in known_types)
            )
            if invalid:
                trace["removed_invalid_operators"].append(
                    {
                        "type": str(operator.get("type", "")),
                        "input_var": input_var,
                        "attribute_relation_label": deepcopy(label),
                    }
                )
                # Keep the constraint visible. Execution may fail, but the
                # pipeline can then repair the attribute against the ontology;
                # silently dropping it would execute a broader query.
        retained.append(operator)
    output["operators"] = retained
    if trace["changes"] or trace["removed_invalid_operators"]:
        trace["status"] = "changed"
    return output, trace
