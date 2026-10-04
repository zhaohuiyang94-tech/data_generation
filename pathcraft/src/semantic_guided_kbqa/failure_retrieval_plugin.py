"""Gold-blind failure-only bridge from training templates to endpoint paths."""

from __future__ import annotations

from copy import deepcopy
from itertools import islice, permutations, product
import json
import re
from typing import Any

from .contracts import EntityCandidate, GroundedSemanticCandidate
from .graph_reconstruction import reconstruct_failure_query_graphs
from .ontology import relation_id_from_label, relation_label_from_id
from .template_path_retrieval import TemplatePathRetriever


_ANCHOR_COVERAGE_SCORE_FLOOR = 0.5
_SINGLETON_REFINEMENT_SCORE_FLOOR = 0.8
_SINGLETON_REFINEMENT_MIN_BROAD_ANSWERS = 32
_DIVERSITY_TEMPLATE_TOP_K = 24
_DIVERSITY_PER_CANDIDATE_LIMIT = 1
_CONTINUATION_DIVERSITY_SCORE_FLOOR = 40.0
_CONTINUATION_DIVERSITY_GRAPH_BUDGET = 8
_NOUN_QUESTION_RE = re.compile(r"\b(?:what|which)\s+(.+)", re.IGNORECASE)
_TYPE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ENTITY_ID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
_EXPLICIT_EXTREMA_RE = re.compile(
    r"\b(?:latest|last|earliest|first|largest|smallest|highest|lowest|"
    r"maximum|minimum|most recent)\b",
    re.IGNORECASE,
)


def _question_key(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _expects_entity_answer(question: str) -> bool:
    """Conservatively identify interrogatives that cannot return schema IDs."""

    text = " ".join(str(question).strip().casefold().split())
    if re.match(r"^(?:who|where)\b", text):
        return True
    return bool(
        re.match(r"^which\b", text)
        and not re.match(
            r"^which\s+(?:year|date|number|id|identifier|percentage|age|time)\b",
            text,
        )
    )


def _direct_path_acceptance(
    question: str,
    answer_ids: list[str],
    diagnostics: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """Apply the outer confidence gate to one already executed direct path.

    Keeping this calculation in one helper lets the endpoint path retriever
    tell whether its *original* incumbent would be rejected before considering
    a low-confidence best-effort challenger.  The helper uses only runtime
    shape/schema evidence; it never reads Gold answers or dataset identity.
    """

    direct_ids = [str(value) for value in answer_ids]
    full_anchor_consensus = bool(
        int(diagnostics.get("selected_anchor_coverage", -1))
        == int(diagnostics.get("anchor_count", -2))
        and diagnostics.get("all_nonempty_candidates_agree")
    )
    typed_non_cvt_consensus = bool(
        not diagnostics.get("selected_terminal_is_cvt", True)
        and float(diagnostics.get("selected_answer_type_evidence", 0.0)) > 0.0
        and int(diagnostics.get("selected_answer_set_support", 0)) >= 2
    )
    entity_answer_shape_compatible = bool(
        not _expects_entity_answer(question)
        or all(_ENTITY_ID_RE.fullmatch(value) for value in direct_ids)
    )
    accepted = bool(
        len(direct_ids) == 1
        and (full_anchor_consensus or typed_non_cvt_consensus)
        # Direct endpoint paths contain relations only; they cannot implement
        # an explicit ordering program.
        and not _EXPLICIT_EXTREMA_RE.search(question)
        and entity_answer_shape_compatible
    )
    reason = (
        (
            "direct_full_anchor_consensus_singleton"
            if full_anchor_consensus
            else "direct_typed_non_cvt_consensus_singleton"
        )
        if accepted
        else "low_confidence_abstention"
    )
    return accepted, reason, {
        "answer_count": len(direct_ids),
        "full_anchor_consensus": full_anchor_consensus,
        "typed_non_cvt_consensus": typed_non_cvt_consensus,
        "anchor_coverage": diagnostics.get("selected_anchor_coverage"),
        "anchor_count": diagnostics.get("anchor_count"),
        "answer_set_support": diagnostics.get("selected_answer_set_support"),
        "nonempty_candidate_count": diagnostics.get("nonempty_candidate_count"),
        "answer_type_evidence": diagnostics.get("selected_answer_type_evidence"),
        "terminal_is_cvt": diagnostics.get("selected_terminal_is_cvt"),
        "entity_answer_shape_compatible": entity_answer_shape_compatible,
    }


def _decompositions(context: Any) -> list[str]:
    """Use the last reviewed runtime decomposition, then frozen inputs.

    ``pipeline.decompositions`` is the immutable prediction store.  When the
    reviewer rewrites a candidate, the value actually used by the failed run
    exists only in its trace.  Reading the store first silently discarded that
    rewrite during failure recovery.  A trace bundle can contain more than one
    attempt, so only the latest review is authoritative.
    """

    artifact = context.trace_bundle.get("decomposition", {})
    traces = artifact.get("traces", []) if isinstance(artifact, dict) else []
    for trace in reversed(traces if isinstance(traces, list) else []):
        if (
            isinstance(trace, dict)
            and trace.get("stage") == "decomposition_review_and_rewrite"
        ):
            output = trace.get("output")
            candidates = (
                output.get("final_candidates", [])
                if isinstance(output, dict)
                else []
            )
            values = [
                str(item)
                for candidate in candidates
                if isinstance(candidate, dict)
                for item in candidate.get("decomposition", [])
                if str(item).strip()
            ]
            if values:
                return list(dict.fromkeys(values))
            # The latest review is authoritative even when it produced no
            # usable candidate.  Fall back to the immutable store, not an
            # older review attempt.
            break

    pipeline = context.pipeline
    try:
        candidates = pipeline.decompositions.get(context.question)
    except (AttributeError, KeyError):
        candidates = []
    values = [
        str(item)
        for candidate in candidates
        for item in getattr(candidate, "decomposition", [])
        if str(item).strip()
    ]
    if values:
        return list(dict.fromkeys(values))
    # Retain trace-only replay support when no decomposition store is loaded.
    for trace in reversed(traces if isinstance(traces, list) else []):
        if not isinstance(trace, dict) or trace.get("stage") != "decompose_predictions":
            continue
        candidates = trace.get("output", [])
        for candidate in candidates if isinstance(candidates, list) else []:
            if isinstance(candidate, dict):
                values.extend(
                    str(item) for item in candidate.get("decomposition", [])
                )
        if values:
            break
    return list(dict.fromkeys(value for value in values if value.strip()))


def _current_anchors(context: Any) -> list[dict[str, str]]:
    grounder = context.pipeline.grounder
    entities = getattr(grounder, "gold_entities", {}).get(
        _question_key(context.question),
        {},
    )
    items = [
        (str(entity_id), str(label))
        for entity_id, label in entities.items()
        if str(entity_id).strip() and str(label).strip()
    ]
    normalized = context.question.casefold()
    items.sort(
        key=lambda item: (
            normalized.find(item[1].casefold())
            if item[1].casefold() in normalized
            else len(normalized),
            item[1],
            item[0],
        )
    )
    return [
        {"id": f"A{index}", "surface": label, "entity_id": entity_id}
        for index, (entity_id, label) in enumerate(items)
    ]


def _execution_rank(executed: Any) -> tuple[float, int, str]:
    return (
        float(executed.graph.score),
        -len(executed.answer_ids),
        str(executed.graph.graph_id),
    )


def _anchor_entity_ids(anchors: list[dict[str, str]]) -> set[str]:
    return {
        str(anchor.get("entity_id", ""))
        for anchor in anchors
        if isinstance(anchor, dict) and str(anchor.get("entity_id", "")).strip()
    }


def _anchor_coverage(executed: Any, anchor_ids: set[str]) -> int:
    if not anchor_ids:
        return 0
    graph_nodes = {
        str(node)
        for triple in getattr(executed.graph, "triples", [])
        if isinstance(triple, (list, tuple)) and len(triple) == 3
        for node in (triple[0], triple[2])
    }
    return len(graph_nodes & anchor_ids)


def _type_tokens(type_id: str) -> tuple[str, ...]:
    leaf = str(type_id).rsplit(".", 1)[-1].replace("_", " ").casefold()
    return tuple(_TYPE_TOKEN_RE.findall(leaf))


def _ontology_type_ids(ontology: Any) -> set[str]:
    values: set[str] = set()
    for attribute in ("relation_domains", "relation_ranges"):
        mapping = getattr(ontology, attribute, {})
        if isinstance(mapping, dict):
            values.update(str(value) for value in mapping.values() if str(value))
    parents = getattr(ontology, "type_parents", {})
    if isinstance(parents, dict):
        values.update(str(value) for value in parents if str(value))
        values.update(
            str(value)
            for parent_values in parents.values()
            for value in parent_values
            if str(value)
        )
    return values


def _expected_answer_types(question: str, ontology: Any) -> set[str]:
    """Infer a conservative answer type from interrogative grammar and schema.

    This is deliberately exact and schema-driven: noun phrases must match the
    leading tokens of an ontology type's leaf name.  It therefore cannot route
    on a dataset item, entity, or relation whitelist.
    """

    normalized = " ".join(str(question).casefold().split())
    available = _ontology_type_ids(ontology)
    match = _NOUN_QUESTION_RE.search(normalized)
    if match is None:
        return set()
    question_tokens = tuple(_TYPE_TOKEN_RE.findall(match.group(1)))
    if not question_tokens:
        return set()
    prefix_matches: list[tuple[int, str]] = []
    head_matches: set[str] = set()
    for type_id in available:
        tokens = _type_tokens(type_id)
        if tokens and question_tokens[: len(tokens)] == tokens:
            prefix_matches.append((len(tokens), type_id))
        elif tokens and question_tokens[0] == tokens[-1]:
            # The first post-interrogative noun is commonly the head while a
            # Freebase leaf may qualify it (``sports_team``, ``us_president``).
            # This is derived from ontology spelling, not a type dictionary.
            head_matches.add(type_id)
    if prefix_matches:
        longest = max(length for length, _ in prefix_matches)
        return {
            type_id
            for length, type_id in prefix_matches
            if length == longest
        }
    return head_matches


def _projected_answer_types(executed: Any, ontology: Any) -> set[str]:
    answer_var = str(getattr(executed.graph, "answer_var", ""))
    values: set[str] = set()
    for triple in getattr(executed.graph, "triples", []):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject, relation_id, object_ = (str(value) for value in triple)
        if subject == answer_var:
            value = str(ontology.domain_for_relation(relation_id))
            if value:
                values.add(value)
        if object_ == answer_var:
            value = str(ontology.range_for_relation(relation_id))
            if value:
                values.add(value)
    # Keep ontology leaves without naming any special root type.  A broad
    # ancestor (for example the root inherited by most entities) must not make
    # two otherwise different answer projections look compatible.
    return {
        type_id
        for type_id in values
        if not any(
            type_id != other
            and type_id in set(ontology.supertypes(other))
            for other in values
        )
    }


def _projected_types_are_cvt(types: set[str], ontology: Any) -> bool:
    non_scalar = {type_id for type_id in types if not type_id.startswith("type.")}
    return bool(non_scalar) and all(
        "common.topic" not in set(ontology.supertypes(type_id))
        for type_id in non_scalar
    )


def _project_cvt_answer_neighbor(
    pipeline: Any,
    selected: Any,
    ontology: Any,
) -> tuple[Any, dict[str, Any]]:
    """Project an anonymous mediator answer to one unique entity neighbor."""

    if ontology is None:
        return selected, {"status": "disabled_no_ontology"}
    selected_types = _projected_answer_types(selected, ontology)
    if not _projected_types_are_cvt(selected_types, ontology):
        return selected, {"status": "not_applicable"}
    answer_var = str(selected.graph.answer_var)
    neighbor_vars = {
        str(object_ if str(subject) == answer_var else subject)
        for subject, _, object_ in selected.graph.triples
        if answer_var in {str(subject), str(object_)}
        and str(object_ if str(subject) == answer_var else subject).startswith("V")
    }
    candidates: list[tuple[str, set[str]]] = []
    for neighbor in sorted(neighbor_vars):
        probe = deepcopy(selected)
        probe.graph.answer_var = neighbor
        types = _projected_answer_types(probe, ontology)
        if types and not _projected_types_are_cvt(types, ontology):
            candidates.append((neighbor, types))
    if len(candidates) != 1:
        return selected, {
            "status": "ambiguous_neighbor",
            "source_types": sorted(selected_types),
            "candidate_count": len(candidates),
        }
    neighbor, neighbor_types = candidates[0]
    graph = deepcopy(selected.graph)
    graph.graph_id = f"{graph.graph_id}P"
    graph.answer_var = neighbor
    graph.sparql = ""
    execution, trace = pipeline._execute_graph_candidate(
        graph,
        lookup_labels=False,
    )
    if execution is None or not execution.answer_ids:
        return selected, {
            "status": "projection_empty",
            "source_types": sorted(selected_types),
            "neighbor_types": sorted(neighbor_types),
            "error": trace.get("error", ""),
        }
    return execution, {
        "status": "projected",
        "source_answer_var": answer_var,
        "answer_var": neighbor,
        "source_types": sorted(selected_types),
        "answer_types": sorted(neighbor_types),
        "answer_count": len(execution.answer_ids),
        "model_used": False,
    }


def _types_compatible(expected: set[str], actual: set[str], ontology: Any) -> bool:
    for expected_type in expected:
        parent_mapping = getattr(ontology, "type_parents", {})
        direct_expected_parents = (
            set(parent_mapping.get(expected_type, ()))
            if isinstance(parent_mapping, dict)
            else set()
        )
        if not direct_expected_parents:
            try:
                direct_expected_parents = set(
                    ontology.supertypes(expected_type, max_depth=1)
                ) - {expected_type}
            except TypeError:
                direct_expected_parents = set()
        for actual_type in actual:
            if (
                expected_type == actual_type
                # A direct parent is a useful schema approximation, while a
                # remote root ancestor is too broad to identify an answer.
                or actual_type in direct_expected_parents
                # A projected subtype is always at least as specific as the
                # requested ontology type.
                or expected_type in set(ontology.supertypes(actual_type))
            ):
                return True
    return False


def _has_explicit_singular_type(expected_types: set[str]) -> bool:
    """Return whether ontology-backed target wording has a singular head."""

    return any(
        tokens and not tokens[-1].endswith("s")
        for type_id in expected_types
        if (tokens := _type_tokens(type_id))
    )


def _triple_count(executed: Any) -> int:
    return sum(
        isinstance(triple, (list, tuple)) and len(triple) == 3
        for triple in getattr(executed.graph, "triples", [])
    )


def _refine_broad_selection_to_singleton(
    selected: Any,
    executions: list[Any],
    *,
    anchor_ids: set[str],
    expected_types: set[str],
    ontology: Any,
) -> Any:
    """Narrow a pathological broad result only with convergent local evidence.

    The singleton must already occur inside the selected answer set, preserve
    anchor coverage and projected ontology type, use a strictly smaller graph,
    and retain most of the selected graph's score.  Competing singleton values
    make the evidence ambiguous and therefore disable the refinement.
    """

    if (
        len(selected.answer_ids) < _SINGLETON_REFINEMENT_MIN_BROAD_ANSWERS
        or not _has_explicit_singular_type(expected_types)
    ):
        return selected
    selected_answers = set(selected.answer_ids)
    selected_coverage = _anchor_coverage(selected, anchor_ids)
    selected_types = _projected_answer_types(selected, ontology)
    selected_triples = _triple_count(selected)
    if not selected_types or selected_triples < 2:
        return selected
    candidates = [
        executed
        for executed in executions
        if len(executed.answer_ids) == 1
        and executed.answer_ids[0] in selected_answers
        and float(executed.graph.score)
        >= _SINGLETON_REFINEMENT_SCORE_FLOOR * float(selected.graph.score)
        and _anchor_coverage(executed, anchor_ids) >= selected_coverage
        and _triple_count(executed) < selected_triples
        and bool(
            _projected_answer_types(executed, ontology) & selected_types
        )
    ]
    if len({executed.answer_ids[0] for executed in candidates}) != 1:
        return selected
    return max(candidates, key=_execution_rank)


def select_failure_execution(
    question: str,
    executions: list[Any],
    anchors: list[dict[str, str]],
    ontology: Any,
) -> Any | None:
    """Select an executed failure candidate without Gold data or KG/model calls.

    The ordinary score winner remains the default.  A challenger is considered
    only when it covers more already-linked anchors and retains at least half
    of the winner's graph score.  Among maximum-coverage challengers, a
    confidently inferred ontology answer type is used when available; all
    missing or inconclusive type evidence falls back to the original ranking.
    """

    if not executions:
        return None
    baseline = max(executions, key=_execution_rank)
    anchor_ids = _anchor_entity_ids(anchors)
    baseline_coverage = _anchor_coverage(baseline, anchor_ids)
    minimum_score = _ANCHOR_COVERAGE_SCORE_FLOOR * float(baseline.graph.score)
    challengers = [
        executed
        for executed in executions
        if _anchor_coverage(executed, anchor_ids) > baseline_coverage
        and float(executed.graph.score) >= minimum_score
    ]
    if challengers:
        maximum_coverage = max(
            _anchor_coverage(executed, anchor_ids) for executed in challengers
        )
        pool = [
            executed
            for executed in challengers
            if _anchor_coverage(executed, anchor_ids) == maximum_coverage
        ]
    else:
        # A confidently inferred ontology projection may distinguish two
        # same-coverage graphs.  Keep the same 0.5 score floor so a weak type
        # coincidence cannot displace the ordinary score winner.
        pool = [
            executed
            for executed in executions
            if _anchor_coverage(executed, anchor_ids) == baseline_coverage
            and float(executed.graph.score) >= minimum_score
        ]
    expected_types = _expected_answer_types(question, ontology)
    selected = None
    if expected_types:
        compatible = [
            executed
            for executed in pool
            if _types_compatible(
                expected_types,
                _projected_answer_types(executed, ontology),
                ontology,
            )
        ]
        if compatible:
            selected = max(compatible, key=_execution_rank)
    if selected is None:
        selected = max(pool, key=_execution_rank) if challengers else baseline
    return _refine_broad_selection_to_singleton(
        selected,
        executions,
        anchor_ids=anchor_ids,
        expected_types=expected_types,
        ontology=ontology,
    )


def _semantic_graph(match: dict[str, Any], anchors: list[dict[str, str]], question: str) -> dict[str, Any] | None:
    by_id = {item["id"]: item for item in anchors}
    paths: list[dict[str, Any]] = []
    used: set[str] = set()
    for path_index, raw in enumerate(match.get("paths", [])):
        anchor_ref = str(raw.get("anchor_ref", ""))
        if anchor_ref not in by_id:
            return None
        used.add(anchor_ref)
        path_id = f"P{path_index}"
        steps: list[dict[str, Any]] = []
        source = anchor_ref
        for step_index, raw_step in enumerate(raw.get("steps", [])):
            direction = str(raw_step.get("direction", "")).casefold()
            label = [str(value) for value in raw_step.get("relation_label", []) if str(value)]
            if direction not in {"forward", "backward"} or not label:
                return None
            target = f"{path_id}.V{step_index}"
            steps.append(
                {
                    "id": f"{path_id}.S{step_index}",
                    "relation_label": label,
                    "direction": direction,
                    "from": source,
                    "to": target,
                }
            )
            source = target
        if not steps or len(steps) > 8:
            return None
        paths.append(
            {
                "id": path_id,
                "anchor_ref": anchor_ref,
                "goal": str(question),
                "steps": steps,
                "path_output_var": source,
            }
        )
    if not paths:
        return None
    return {
        "anchors": [
            {"id": item["id"], "surface": item["surface"]}
            for item in anchors
            if item["id"] in used
        ],
        "semantic_paths": paths,
    }


def _template_retriever(pipeline: Any) -> TemplatePathRetriever:
    value = getattr(pipeline, "_failure_template_retriever", None)
    if value is not None:
        return value
    lock = getattr(pipeline, "_failure_template_retriever_lock", None)
    if lock is None:
        value = TemplatePathRetriever.from_contract(pipeline.contract)
        pipeline._failure_template_retriever = value
        return value
    with lock:
        value = getattr(pipeline, "_failure_template_retriever", None)
        if value is None:
            value = TemplatePathRetriever.from_contract(pipeline.contract)
            pipeline._failure_template_retriever = value
    return value


def _compact_execution_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Keep recovery traces useful without persisting every full SPARQL body."""

    return {
        key: trace[key]
        for key in (
            "graph_id",
            "answer_count",
            "row_count",
            "query_elapsed_seconds",
            "label_elapsed_seconds",
            "elapsed_seconds",
            "error",
        )
        if key in trace
    }


def _graph_signature(graph: Any) -> str:
    return json.dumps(
        {
            "triples": getattr(graph, "triples", []),
            "answer_var": getattr(graph, "answer_var", ""),
            "operators": getattr(graph, "operators", []),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _answer_relation_is_explicit(question: str, executed: Any) -> bool:
    """Whether an answer-edge relation is stated lexically in the question."""

    question_tokens = set(_TYPE_TOKEN_RE.findall(str(question).casefold()))
    answer_var = str(getattr(executed.graph, "answer_var", ""))
    for triple in getattr(executed.graph, "triples", []):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        subject, relation_id, object_ = (str(value) for value in triple)
        if answer_var not in {subject, object_}:
            continue
        leaf = relation_id.rsplit(".", 1)[-1].replace("_", " ")
        relation_tokens = set(_TYPE_TOKEN_RE.findall(leaf.casefold()))
        if relation_tokens and relation_tokens <= question_tokens:
            return True
    return False


def _template_acceptance_reason(
    *,
    question: str,
    selected: Any,
    executions: list[Any],
    lane: str,
    anchors: list[dict[str, str]],
    ontology: Any,
) -> tuple[str, dict[str, Any]]:
    """Return a Gold-blind high-confidence reason or abstain with ``""``.

    Failure rows are excluded from this project's CLI macro denominator.  A
    low-confidence non-empty answer must therefore remain a failure instead of
    silently enlarging that denominator with mostly-wrong predictions.
    """

    answer_count = len(selected.answer_ids)
    selected_set = set(selected.answer_ids)
    support = sum(set(item.answer_ids) == selected_set for item in executions)
    anchor_ids = _anchor_entity_ids(anchors)
    coverage = _anchor_coverage(selected, anchor_ids)
    score = float(selected.graph.score)
    evidence: dict[str, Any] = {
        "lane": lane,
        "answer_count": answer_count,
        "answer_set_support": support,
        "nonempty_candidate_count": len(executions),
        "graph_score": score,
        "anchor_coverage": coverage,
        "anchor_count": len(anchor_ids),
    }
    projected_types = (
        _projected_answer_types(selected, ontology)
        if ontology is not None
        else set()
    )
    projected_non_scalar = {
        type_id for type_id in projected_types if not type_id.startswith("type.")
    }
    projected_answer_is_cvt = bool(projected_non_scalar) and all(
        "common.topic" not in set(ontology.supertypes(type_id))
        for type_id in projected_non_scalar
    )
    evidence["projected_answer_types"] = sorted(projected_types)
    evidence["projected_answer_is_cvt"] = projected_answer_is_cvt
    if projected_answer_is_cvt:
        # Anonymous mediator records are implementation details, not final
        # answers.  Their labeled entity projection must be recovered by a
        # graph rather than accepted as a high-score singleton.
        return "", evidence

    if lane == "diversity_template":
        if answer_count != 1:
            return "", evidence
        if score >= 20.0:
            evidence["nested_singleton_refinement"] = False
            return "diversity_singleton_score", evidence
        nested = False
        selected_types = (
            _projected_answer_types(selected, ontology)
            if ontology is not None
            else set()
        )
        selected_triples = _triple_count(selected)
        if selected_types:
            for broad in executions:
                if (
                    broad is not selected
                    and len(broad.answer_ids)
                    >= _SINGLETON_REFINEMENT_MIN_BROAD_ANSWERS
                    and selected.answer_ids[0] in set(broad.answer_ids)
                    and float(broad.graph.score) > score
                    and score
                    >= _SINGLETON_REFINEMENT_SCORE_FLOOR
                    * float(broad.graph.score)
                    and coverage >= _anchor_coverage(broad, anchor_ids)
                    and selected_triples < _triple_count(broad)
                    and bool(
                        selected_types
                        & _projected_answer_types(broad, ontology)
                    )
                ):
                    nested = True
                    break
        evidence["nested_singleton_refinement"] = nested
        if nested:
            return "diversity_nested_singleton", evidence
        return "", evidence

    if lane != "ordinary_template":
        return "", evidence
    coverage_ok = bool(anchor_ids) and coverage * 3 >= len(anchor_ids) * 2
    evidence["two_thirds_anchor_coverage"] = coverage_ok
    if score < 20.0 or not coverage_ok:
        return "", evidence
    if answer_count == 1:
        return "ordinary_singleton", evidence
    if answer_count <= 5 and support >= 5:
        return "ordinary_answer_set_consensus", evidence
    if (
        2 <= answer_count <= 10
        and len(executions) == 1
        and re.match(r"\s*(?:what|which)\b", str(question), re.IGNORECASE)
        and _answer_relation_is_explicit(question, selected)
    ):
        return "ordinary_explicit_answer_relation", evidence
    return "", evidence


def _execute_failure_graphs(
    *,
    pipeline: Any,
    graphs: list[Any],
    question: str,
    decompositions: list[str],
    graph_budget: int,
    seen_signatures: set[str],
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Execute a score-ordered lane with deduplication and a safe early stop."""

    executions: list[Any] = []
    traces: list[dict[str, Any]] = []
    first_nonempty_score: float | None = None
    stopped_below_score_floor = False
    duplicates_skipped = 0
    for graph in graphs:
        signature = _graph_signature(graph)
        if signature in seen_signatures:
            duplicates_skipped += 1
            continue
        if (
            first_nonempty_score is not None
            and first_nonempty_score > 0.0
            and float(graph.score)
            < _ANCHOR_COVERAGE_SCORE_FLOOR * first_nonempty_score
        ):
            stopped_below_score_floor = True
            break
        seen_signatures.add(signature)
        graph.provenance["original_question"] = question
        graph.provenance["decomposition"] = list(decompositions)
        executed, trace = pipeline._execute_graph_candidate(
            graph,
            lookup_labels=False,
        )
        traces.append(_compact_execution_trace(trace))
        if executed is not None and executed.answer_ids:
            if first_nonempty_score is None:
                first_nonempty_score = float(executed.graph.score)
            executions.append(executed)
    return executions, traces, {
        "maximum_graph_budget": graph_budget,
        "executed_graph_count": len(traces),
        "duplicate_graphs_skipped": duplicates_skipped,
        "score_floor_ratio": _ANCHOR_COVERAGE_SCORE_FLOOR,
        "stopped_below_score_floor": stopped_below_score_floor,
        "candidate_label_queries": 0,
    }


def _ground_templates(
    *,
    matches: list[dict[str, Any]],
    anchors: list[dict[str, str]],
    question: str,
    ontology: Any,
    endpoint: Any,
) -> tuple[list[GroundedSemanticCandidate], dict[str, Any]]:
    """Verify all template paths with one VALUES query per direction group."""
    anchors_by_ref = {item["id"]: item for item in anchors}
    prepared: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    groups: dict[tuple[str, tuple[str, ...]], list[set[str]]] = {}
    for match in matches:
        raw_paths = [path for path in match.get("paths", []) if isinstance(path, dict)]
        slots = sorted({int(path.get("anchor_slot", 0)) for path in raw_paths})
        if not raw_paths or len(slots) > len(anchors):
            continue
        for assigned in permutations(sorted(anchors_by_ref), len(slots)):
            slot_mapping = dict(zip(slots, assigned))
            paths: list[dict[str, Any]] = []
            valid = True
            for raw_path in raw_paths:
                anchor_ref = slot_mapping[int(raw_path.get("anchor_slot", 0))]
                directions: list[str] = []
                allowed: list[set[str]] = []
                for raw_step in raw_path.get("steps", []):
                    direction = str(raw_step.get("direction", "")).casefold()
                    label = raw_step.get("relation_label", [])
                    if direction not in {"forward", "backward"}:
                        valid = False
                        break
                    candidates = set(
                        ontology.candidates_for_label(label, limit=4)
                    )
                    direct = relation_id_from_label(label)
                    if direct:
                        candidates.add(direct)
                    if not candidates:
                        valid = False
                        break
                    directions.append(direction)
                    allowed.append(candidates)
                if not valid or not directions or len(directions) > 8:
                    valid = False
                    break
                key = (anchor_ref, tuple(directions))
                union_sets = groups.setdefault(
                    key,
                    [set() for _ in directions],
                )
                for index, values in enumerate(allowed):
                    union_sets[index].update(values)
                paths.append(
                    {
                        "anchor_ref": anchor_ref,
                        "directions": tuple(directions),
                        "allowed": tuple(frozenset(values) for values in allowed),
                    }
                )
            if valid and paths:
                prepared.append((match, paths))

    verified: dict[tuple[str, tuple[str, ...]], list[tuple[str, ...]]] = {}
    query_diagnostics: list[dict[str, Any]] = []
    for (anchor_ref, directions), candidate_sets in sorted(groups.items()):
        anchor = anchors_by_ref[anchor_ref]
        error = ""
        rows: list[dict[str, Any]] = []
        try:
            rows = endpoint.path_hops(
                anchor["entity_id"],
                list(directions),
                limit=10_000,
                relation_candidates=[sorted(values) for values in candidate_sets],
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}:{exc}"
        sequences = sorted(
            {
                tuple(str(value) for value in row.get("relation_ids", []))
                for row in rows
                if isinstance(row, dict)
                and len(row.get("relation_ids", [])) == len(directions)
            }
        )
        verified[(anchor_ref, directions)] = sequences
        query_diagnostics.append(
            {
                "anchor_ref": anchor_ref,
                "directions": list(directions),
                "candidate_widths": [len(values) for values in candidate_sets],
                "sequence_count": len(sequences),
                "error": error,
            }
        )

    grounded: list[GroundedSemanticCandidate] = []
    seen: set[str] = set()
    for match, paths in prepared:
        choices: list[list[tuple[str, ...]]] = []
        for path in paths:
            sequences = verified.get(
                (path["anchor_ref"], path["directions"]),
                [],
            )
            matching = [
                sequence
                for sequence in sequences
                if all(
                    relation_id in path["allowed"][index]
                    for index, relation_id in enumerate(sequence)
                )
            ]
            choices.append(matching[:2])
        if any(not values for values in choices):
            continue
        for combination in islice(product(*choices), 16):
            semantic_paths: list[dict[str, Any]] = []
            relation_bindings: dict[str, str] = {}
            used_anchors: set[str] = set()
            for path_index, (path, sequence) in enumerate(zip(paths, combination)):
                path_id = f"P{path_index}"
                anchor_ref = path["anchor_ref"]
                used_anchors.add(anchor_ref)
                source = anchor_ref
                steps: list[dict[str, Any]] = []
                for step_index, (relation_id, direction) in enumerate(
                    zip(sequence, path["directions"])
                ):
                    step_id = f"{path_id}.S{step_index}"
                    target = f"{path_id}.V{step_index}"
                    steps.append(
                        {
                            "id": step_id,
                            "relation_label": relation_label_from_id(relation_id),
                            "direction": direction,
                            "from": source,
                            "to": target,
                        }
                    )
                    relation_bindings[step_id] = relation_id
                    source = target
                semantic_paths.append(
                    {
                        "id": path_id,
                        "anchor_ref": anchor_ref,
                        "goal": question,
                        "steps": steps,
                        "path_output_var": source,
                    }
                )
            signature = json.dumps(
                {
                    "anchors": sorted(used_anchors),
                    "bindings": sorted(relation_bindings.items()),
                },
                sort_keys=True,
            )
            if signature in seen:
                continue
            seen.add(signature)
            anchor_bindings = {
                anchor_ref: EntityCandidate(
                    anchors_by_ref[anchor_ref]["entity_id"],
                    anchors_by_ref[anchor_ref]["surface"],
                    1.0,
                    "failure_template",
                )
                for anchor_ref in used_anchors
            }
            grounded.append(
                GroundedSemanticCandidate(
                    compose_input={
                        "question": question,
                        "anchors": [
                            {
                                "id": anchor_ref,
                                "surface": anchors_by_ref[anchor_ref]["surface"],
                            }
                            for anchor_ref in sorted(used_anchors)
                        ],
                        "semantic_paths": semantic_paths,
                    },
                    anchor_bindings=anchor_bindings,
                    relation_bindings=relation_bindings,
                    score=1.0 / max(1, int(match.get("rank", 1))),
                    provenance={
                        "failure_template_retrieval": {
                            "template_rank": int(match.get("rank", 0)),
                            "source_example_index": int(
                                match.get("source_example_index", -1)
                            ),
                            "model_used": False,
                            "uses_gold": False,
                        }
                    },
                )
            )
    grounded.sort(key=lambda item: -item.score)
    return grounded[:80], {
        "group_query_count": len(query_diagnostics),
        "groups": query_diagnostics,
        "prepared_template_count": len(prepared),
        "grounded_candidate_count": len(grounded),
    }


def _direct_template_candidates(
    *,
    matches: list[dict[str, Any]],
    anchors: list[dict[str, str]],
    question: str,
) -> tuple[list[GroundedSemanticCandidate], dict[str, Any]]:
    """Compile training templates directly; final SPARQL verifies existence."""
    anchors_by_ref = {item["id"]: item for item in anchors}
    output: list[GroundedSemanticCandidate] = []
    seen: set[str] = set()
    for match in matches:
        raw_paths = [path for path in match.get("paths", []) if isinstance(path, dict)]
        slots = sorted({int(path.get("anchor_slot", 0)) for path in raw_paths})
        if not raw_paths or len(slots) > len(anchors):
            continue
        expected_mapping = {
            int(slot): str(anchor_ref)
            for slot, anchor_ref in (match.get("anchor_mapping") or {}).items()
        }
        for assigned in permutations(sorted(anchors_by_ref), len(slots)):
            slot_mapping = dict(zip(slots, assigned))
            semantic_paths: list[dict[str, Any]] = []
            relation_bindings: dict[str, str] = {}
            valid = True
            for path_index, raw_path in enumerate(raw_paths):
                path_id = f"P{path_index}"
                anchor_ref = slot_mapping[int(raw_path.get("anchor_slot", 0))]
                source = anchor_ref
                steps: list[dict[str, Any]] = []
                for step_index, raw_step in enumerate(raw_path.get("steps", [])):
                    direction = str(raw_step.get("direction", "")).casefold()
                    relation_id = relation_id_from_label(
                        raw_step.get("relation_label", [])
                    )
                    if direction not in {"forward", "backward"} or not relation_id:
                        valid = False
                        break
                    step_id = f"{path_id}.S{step_index}"
                    target = f"{path_id}.V{step_index}"
                    steps.append(
                        {
                            "id": step_id,
                            "relation_label": relation_label_from_id(relation_id),
                            "direction": direction,
                            "from": source,
                            "to": target,
                        }
                    )
                    relation_bindings[step_id] = relation_id
                    source = target
                if not valid or not steps or len(steps) > 8:
                    valid = False
                    break
                semantic_paths.append(
                    {
                        "id": path_id,
                        "anchor_ref": anchor_ref,
                        "goal": question,
                        "steps": steps,
                        "path_output_var": source,
                    }
                )
            if not valid or not semantic_paths:
                continue
            signature = json.dumps(
                {
                    "mapping": slot_mapping,
                    "bindings": sorted(relation_bindings.items()),
                },
                sort_keys=True,
            )
            if signature in seen:
                continue
            seen.add(signature)
            used = {path["anchor_ref"] for path in semantic_paths}
            mapping_bonus = 0.1 * sum(
                expected_mapping.get(slot) == anchor_ref
                for slot, anchor_ref in slot_mapping.items()
            )
            output.append(
                GroundedSemanticCandidate(
                    compose_input={
                        "question": question,
                        "anchors": [
                            {
                                "id": anchor_ref,
                                "surface": anchors_by_ref[anchor_ref]["surface"],
                            }
                            for anchor_ref in sorted(used)
                        ],
                        "semantic_paths": semantic_paths,
                    },
                    anchor_bindings={
                        anchor_ref: EntityCandidate(
                            anchors_by_ref[anchor_ref]["entity_id"],
                            anchors_by_ref[anchor_ref]["surface"],
                            1.0,
                            "failure_template",
                        )
                        for anchor_ref in used
                    },
                    relation_bindings=relation_bindings,
                    # Keep BM25 template order dominant over the many graph
                    # merge variants produced downstream. Endpoint execution
                    # remains the factual verifier.
                    score=(10.0 / max(1, int(match.get("rank", 1)))) + mapping_bonus,
                    provenance={
                        "failure_template_retrieval": {
                            "template_rank": int(match.get("rank", 0)),
                            "source_example_index": int(
                                match.get("source_example_index", -1)
                            ),
                            "direct_template": True,
                            "model_used": False,
                            "uses_gold": False,
                        }
                    },
                )
            )
    output.sort(key=lambda item: -item.score)
    return output[:80], {
        "prepared_template_count": len(matches),
        "grounded_candidate_count": min(80, len(output)),
        "endpoint_path_queries": 0,
        "verification": "final_sparql_only",
    }


def retrieve(context: Any, endpoint: Any) -> dict[str, Any]:
    """Retriever plugin consumed by ``replay_failure_path_retrieval.py``."""
    pipeline = context.pipeline
    verbose_diagnostics = bool(getattr(context, "verbose_diagnostics", True))
    decompositions = _decompositions(context)
    anchors = _current_anchors(context)
    diagnostics: dict[str, Any] = {
        "status": "not_compiled",
        "strategy": "failure_only_template_then_endpoint_path",
        "llm_calls": 0,
        "uses_gold_answers": False,
        "anchor_count": len(anchors),
    }
    if not anchors:
        diagnostics["reason"] = "no_linked_anchors"
        return {"answer_ids": [], "diagnostics": diagnostics}

    matches, template_diagnostics = _template_retriever(pipeline).retrieve(
        question=context.question,
        decomposition=decompositions,
        current_anchors=anchors,
        top_k=int(getattr(pipeline, "failure_endpoint_template_top_k", 24)),
        preselect=128,
    )
    diagnostics["template"] = (
        template_diagnostics
        if verbose_diagnostics
        else {
            key: value
            for key, value in template_diagnostics.items()
            if key
            in {
                "strategy",
                "index_size",
                "index_fingerprint",
                "cache_hit",
                "current_anchor_count",
                "considered",
                "rejected_anchor_count",
            }
        }
    )
    grounded, grounding_diagnostics = _direct_template_candidates(
        matches=matches,
        anchors=anchors,
        question=context.question,
    )
    diagnostics["grounding"] = grounding_diagnostics
    diagnostics["grounded_candidate_count"] = len(grounded)
    graph_budget = int(getattr(pipeline, "failure_endpoint_graph_budget", 80))
    graphs, reconstruction_diagnostics = reconstruct_failure_query_graphs(
        grounded,
        ontology=getattr(pipeline.grounder, "ontology", None),
        pipeline_version=str(pipeline.version),
        limit=graph_budget,
        allow_single_path=True,
        allow_hybrid=False,
        max_execution_limit=graph_budget,
        pool_limit=graph_budget,
    )
    diagnostics["reconstruction"] = reconstruction_diagnostics
    seen_signatures: set[str] = set()
    executions, execution_diagnostics, execution_policy = (
        _execute_failure_graphs(
            pipeline=pipeline,
            graphs=graphs,
            question=context.question,
            decompositions=decompositions,
            graph_budget=graph_budget,
            seen_signatures=seen_signatures,
        )
    )
    diagnostics["execution_policy"] = execution_policy
    diagnostics["executions"] = execution_diagnostics

    ordinary_selected = select_failure_execution(
        context.question,
        executions,
        anchors,
        getattr(pipeline.grounder, "ontology", None),
    )
    ordinary_acceptance_reason = ""
    ordinary_acceptance_evidence: dict[str, Any] = {}
    if ordinary_selected is not None:
        projected, projection_diagnostics = _project_cvt_answer_neighbor(
            pipeline,
            ordinary_selected,
            getattr(pipeline.grounder, "ontology", None),
        )
        if projection_diagnostics.get("status") != "not_applicable":
            diagnostics["ordinary_cvt_answer_projection"] = (
                projection_diagnostics
            )
        if projected is not ordinary_selected:
            ordinary_selected = projected
            executions = [projected]
        (
            ordinary_acceptance_reason,
            ordinary_acceptance_evidence,
        ) = _template_acceptance_reason(
            question=context.question,
            selected=ordinary_selected,
            executions=executions,
            lane="ordinary_template",
            anchors=anchors,
            ontology=getattr(pipeline.grounder, "ontology", None),
        )
    ordinary_low_confidence = bool(
        executions and not ordinary_acceptance_reason
    )

    # A score-sorted reconstruction pool can be saturated by several merge
    # variants of the same high-ranked template.  After every ordinary graph
    # is empty *or its selected answer fails the confidence gate*, compile one
    # graph per candidate from the first 24 templates.  Previously executed
    # graph structures are skipped exactly.  Accepted ordinary answers retain
    # the original immediate-return path and pay no additional endpoint cost.
    if not executions or ordinary_low_confidence:
        diagnostics["ordinary_lane"] = {
            "grounding": grounding_diagnostics,
            "reconstruction": reconstruction_diagnostics,
            "execution_policy": execution_policy,
            "executions": execution_diagnostics,
        }
        if ordinary_low_confidence:
            diagnostics["ordinary_lane"]["acceptance"] = {
                "accepted": False,
                "reason": "low_confidence_abstention",
                "uses_gold_answers": False,
                "evidence": ordinary_acceptance_evidence,
            }
        diversity_grounded, diversity_grounding = _direct_template_candidates(
            matches=matches[:_DIVERSITY_TEMPLATE_TOP_K],
            anchors=anchors,
            question=context.question,
        )
        diversity_graph_budget = (
            min(graph_budget, _CONTINUATION_DIVERSITY_GRAPH_BUDGET)
            if ordinary_low_confidence
            else graph_budget
        )
        diversity_graphs, diversity_reconstruction = (
            reconstruct_failure_query_graphs(
                diversity_grounded,
                ontology=getattr(pipeline.grounder, "ontology", None),
                pipeline_version=str(pipeline.version),
                limit=diversity_graph_budget,
                per_candidate_limit=_DIVERSITY_PER_CANDIDATE_LIMIT,
                allow_single_path=True,
                allow_hybrid=False,
                max_execution_limit=diversity_graph_budget,
                pool_limit=diversity_graph_budget,
            )
        )
        (
            executions,
            diversity_execution_diagnostics,
            diversity_execution_policy,
        ) = _execute_failure_graphs(
            pipeline=pipeline,
            graphs=diversity_graphs,
            question=context.question,
            decompositions=decompositions,
            graph_budget=diversity_graph_budget,
            seen_signatures=seen_signatures,
        )
        diagnostics["diversity_fallback"] = {
            "triggered": True,
            "reason": (
                "ordinary_lane_low_confidence"
                if ordinary_low_confidence
                else "ordinary_lane_all_executions_empty"
            ),
            "template_top_k": min(_DIVERSITY_TEMPLATE_TOP_K, len(matches)),
            "per_candidate_limit": _DIVERSITY_PER_CANDIDATE_LIMIT,
            "graph_budget": diversity_graph_budget,
            "grounding": diversity_grounding,
            "reconstruction": diversity_reconstruction,
            "execution_policy": diversity_execution_policy,
            "executions": diversity_execution_diagnostics,
        }
        # Top-level fields always describe the active template lane so trace
        # counters and the eventual selected graph use one consistent scope.
        diagnostics["grounding"] = diversity_grounding
        diagnostics["grounded_candidate_count"] = len(diversity_grounded)
        diagnostics["reconstruction"] = diversity_reconstruction
        diagnostics["execution_policy"] = diversity_execution_policy
        diagnostics["executions"] = diversity_execution_diagnostics
    else:
        diagnostics["diversity_fallback"] = {
            "triggered": False,
            "reason": "ordinary_lane_nonempty",
        }

    active_lane = (
        "diversity_template"
        if diagnostics["diversity_fallback"].get("triggered")
        else "ordinary_template"
    )
    active_selected = select_failure_execution(
        context.question,
        executions,
        anchors,
        getattr(pipeline.grounder, "ontology", None),
    )
    active_acceptance_reason = ""
    active_acceptance_evidence: dict[str, Any] = {}
    if active_selected is not None:
        projected, projection_diagnostics = _project_cvt_answer_neighbor(
            pipeline,
            active_selected,
            getattr(pipeline.grounder, "ontology", None),
        )
        if projection_diagnostics.get("status") != "not_applicable":
            diagnostics["cvt_answer_projection"] = projection_diagnostics
        if projected is not active_selected:
            active_selected = projected
            # The mediator and its entity projection are not competing answer
            # hypotheses.  Keep only the projected execution so the ordinary
            # selector cannot switch back to the anonymous record later.
            executions = [projected]
        (
            active_acceptance_reason,
            active_acceptance_evidence,
        ) = _template_acceptance_reason(
            question=context.question,
            selected=active_selected,
            executions=executions,
            lane=active_lane,
            anchors=anchors,
            ontology=getattr(pipeline.grounder, "ontology", None),
        )
    if (
        ordinary_low_confidence
        and active_acceptance_reason == "diversity_singleton_score"
        and float(active_acceptance_evidence.get("graph_score", 0.0))
        < _CONTINUATION_DIVERSITY_SCORE_FLOOR
    ):
        # A diversity singleton with the ordinary score floor is useful when
        # the ordinary lane was completely empty.  It is weaker evidence when
        # it contradicts an existing non-empty ordinary answer, so the
        # continuation path requires a doubled confidence floor.
        active_acceptance_evidence["continuation_score_floor"] = (
            _CONTINUATION_DIVERSITY_SCORE_FLOOR
        )
        active_acceptance_reason = ""
    if executions and not active_acceptance_reason:
        diagnostics["diversity_fallback"]["acceptance"] = {
            "accepted": False,
            "reason": "low_confidence_abstention",
            "uses_gold_answers": False,
            "evidence": active_acceptance_evidence,
        }
    diagnostics["execution_candidates"] = [
        {
            "graph_id": item.graph.graph_id,
            "score": item.graph.score,
            "answer_count": len(item.answer_ids),
            **(
                {
                    "answer_ids": list(item.answer_ids),
                    "triples": list(item.graph.triples),
                    "answer_var": str(item.graph.answer_var),
                }
                if verbose_diagnostics
                else {}
            ),
        }
        for item in executions
    ]
    if not executions or not active_acceptance_reason:
        # The last lane enumerates only real one-hop endpoint paths.  It runs
        # after both template lanes are empty or rejected by their confidence
        # gate, so it cannot replace an accepted template answer.
        from .failure_endpoint_retrieval_plugin import retrieve as retrieve_paths

        direct = retrieve_paths(context, endpoint)
        diagnostics["direct_endpoint_fallback"] = direct.get(
            "diagnostics",
            {},
        )
        direct_ids = [str(value) for value in direct.get("answer_ids", [])]
        if direct_ids:
            diagnostics["status"] = "selected_direct_endpoint_path"
            diagnostics["selected_lane"] = "direct_endpoint_path"
            diagnostics["selected_graph_id"] = (
                diagnostics["direct_endpoint_fallback"].get(
                    "selected_graph_id",
                    "",
                )
            )
            diagnostics["answer_count"] = len(direct_ids)
            direct_diagnostics = diagnostics["direct_endpoint_fallback"]
            accepted, acceptance_reason, acceptance_evidence = (
                _direct_path_acceptance(
                    context.question,
                    direct_ids,
                    direct_diagnostics,
                )
            )
            challenger_evidence = direct_diagnostics.get(
                "low_confidence_challenger",
                {},
            )
            if (
                isinstance(challenger_evidence, dict)
                and challenger_evidence.get("applied") is True
            ):
                # The endpoint helper is allowed to replace the *rejected*
                # incumbent only for the bounded low-confidence lane.  Never
                # promote that alternative to high confidence after the fact;
                # this also preserves missing-relation first refusal.
                accepted = False
                acceptance_reason = "low_confidence_abstention"
                acceptance_evidence = {
                    **acceptance_evidence,
                    "original_direct_confidence_gate_rejected": True,
                    "low_confidence_challenger_applied": True,
                }
            diagnostics["acceptance"] = {
                "accepted": accepted,
                "reason": acceptance_reason,
                "uses_gold_answers": False,
                "evidence": acceptance_evidence,
            }
            if not accepted:
                best_effort_limit = int(
                    getattr(
                        pipeline,
                        "failure_endpoint_best_effort_answer_limit",
                        64,
                    )
                )
                return_best_effort = bool(
                    getattr(
                        pipeline,
                        "failure_endpoint_return_bounded_direct_best_effort",
                        False,
                    )
                    and 0 < len(direct_ids) <= best_effort_limit
                )
                if return_best_effort:
                    diagnostics["status"] = "selected_bounded_direct_best_effort"
                    diagnostics["acceptance"] = {
                        **diagnostics["acceptance"],
                        "accepted": True,
                        "reason": "bounded_direct_best_effort_after_final_empty",
                        "confidence": "low",
                        "evidence": {
                            **diagnostics["acceptance"]["evidence"],
                            "best_effort_answer_limit": best_effort_limit,
                            "normal_pipeline_was_final_empty": True,
                            "additional_execution_queries": 0,
                            "additional_label_queries": 0,
                        },
                    }
                    return {
                        "answer_ids": direct_ids,
                        "answers": list(direct.get("answers", [])),
                        "selected_graph": direct.get("selected_graph"),
                        "diagnostics": diagnostics,
                    }
                diagnostics["status"] = "abstained_low_confidence"
                diagnostics["reason"] = "direct_path_confidence_gate"
                return {"answer_ids": [], "diagnostics": diagnostics}
            return {
                "answer_ids": direct_ids,
                "answers": list(direct.get("answers", [])),
                "selected_graph": direct.get("selected_graph"),
                "diagnostics": diagnostics,
            }
        if executions:
            diagnostics["status"] = "abstained_low_confidence"
            diagnostics["selected_lane"] = active_lane
            diagnostics["selected_graph_id"] = (
                active_selected.graph.graph_id if active_selected is not None else ""
            )
            diagnostics["answer_count"] = (
                len(active_selected.answer_ids)
                if active_selected is not None
                else 0
            )
            diagnostics["acceptance"] = {
                "accepted": False,
                "reason": "low_confidence_abstention",
                "uses_gold_answers": False,
                "evidence": active_acceptance_evidence,
            }
            diagnostics["reason"] = "template_confidence_gate"
        else:
            diagnostics["reason"] = "all_failure_retrieval_lanes_empty"
        return {"answer_ids": [], "diagnostics": diagnostics}
    selected = select_failure_execution(
        context.question,
        executions,
        anchors,
        getattr(pipeline.grounder, "ontology", None),
    )
    if selected is None:
        diagnostics["reason"] = "selection_failed"
        return {"answer_ids": [], "diagnostics": diagnostics}
    diagnostics["status"] = "selected"
    selected_lane = (
        "diversity_template"
        if diagnostics["diversity_fallback"].get("triggered")
        else "ordinary_template"
    )
    diagnostics["selected_lane"] = selected_lane
    diagnostics["selected_graph_id"] = selected.graph.graph_id
    diagnostics["answer_count"] = len(selected.answer_ids)
    acceptance_reason, acceptance_evidence = _template_acceptance_reason(
        question=context.question,
        selected=selected,
        executions=executions,
        lane=selected_lane,
        anchors=anchors,
        ontology=getattr(pipeline.grounder, "ontology", None),
    )
    diagnostics["acceptance"] = {
        "accepted": bool(acceptance_reason),
        "reason": acceptance_reason or "low_confidence_abstention",
        "uses_gold_answers": False,
        "evidence": acceptance_evidence,
    }
    if not acceptance_reason:
        diagnostics["status"] = "abstained_low_confidence"
        diagnostics["reason"] = "template_confidence_gate"
        return {"answer_ids": [], "diagnostics": diagnostics}
    raw_labels: list[dict[str, str]] = []
    try:
        raw_labels = pipeline.kg.labels(list(selected.answer_ids))
    except Exception as exc:
        diagnostics["selected_label_error"] = f"{type(exc).__name__}: {exc}"
    labels_by_id = {
        str(item.get("id", "")): {
            "id": str(item.get("id", "")),
            "label": str(item.get("label", "")),
        }
        for item in raw_labels
        if isinstance(item, dict) and str(item.get("id", ""))
    }
    selected.answers = [
        labels_by_id.get(answer_id, {"id": answer_id, "label": answer_id})
        for answer_id in selected.answer_ids
    ]
    diagnostics["execution_policy"]["selected_label_queries"] = int(
        bool(selected.answer_ids)
    )
    return {
        "answer_ids": list(selected.answer_ids),
        "answers": list(selected.answers),
        "selected_graph": pipeline._graph_summary(selected.graph),
        "diagnostics": diagnostics,
    }


__all__ = ["retrieve", "select_failure_execution"]
