from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
from typing import Any

from .clients import EmbeddingRanker, HttpChatClient, HttpEmbeddingClient, SparqlKnowledgeGraph
from .config import AppConfig
from .data import DecompositionStore, GoldEntityStore, TrainingContract
from .evaluation import (
    EVALUATION_VERSION,
    evaluation_metadata,
    is_error_result_row,
    is_metric_result_row,
    row_id_f1_recall_are_all_one,
    score_result_row,
    summarize_result_rows,
)
from .grounding import SemanticGrounder
from .ontology import FreebaseOntology
from .path_alignment_selector import PathAlignmentSelector
from .pipeline import SemanticGuidedPipeline
from .template_consensus_selector import SourceVerifiedTemplateConsensusSelector
from .train_gold_extrema_selector import TrainGoldPathExtremaSelector
from .entity_kind_fixed_beam import EntityKindFixedBeamPlanner
from .underanswer_expansion_selector import UnderanswerExpansionSelector


def build_pipeline(
    config: AppConfig,
) -> tuple[SemanticGuidedPipeline, DecompositionStore, GoldEntityStore]:
    config.validate()
    store = DecompositionStore.load(config.data.decompose_predictions)
    contract = TrainingContract.load(
        config.data.semantic_train,
        config.data.compose_train,
        config.data.operator_train,
    )
    gold_store = GoldEntityStore({})
    gold_entities = {}
    if Path(config.data.gold_entities).is_file():
        gold_store = GoldEntityStore.load(
            config.data.gold_entities,
            member=config.data.gold_entities_member,
        )
    if config.data.use_gold_entities:
        gold_entities = gold_store.mapping()
    ontology = (
        FreebaseOntology.load(config.ontology.directory)
        if config.ontology.enabled
        else None
    )
    kg = SparqlKnowledgeGraph(config.freebase)
    ranker = EmbeddingRanker(HttpEmbeddingClient(config.embedding))
    grounder = SemanticGrounder(
        kg,
        ranker,
        entity_top_k=config.beam.entity_top_k,
        first_hop_top_k=config.beam.first_hop_top_k,
        second_hop_top_k=config.beam.second_hop_top_k,
        hop_top_k=config.beam.hop_top_k,
        path_beam=config.beam.path_beam,
        hop_query_limit=config.beam.hop_query_limit,
        hop_values_per_relation=config.beam.hop_values_per_relation,
        long_path_strategy=config.long_path_strategy,
        avoid_immediate_backtrack=config.avoid_immediate_backtrack,
        candidate_limit=max(config.beam.grounded_semantic * 4, config.beam.grounded_semantic),
        semantic_beam=config.beam.grounded_semantic,
        gold_entities=gold_entities,
        ontology=ontology,
        ontology_relation_top_k=config.ontology.relation_top_k,
    )
    template_consensus_selector = (
        SourceVerifiedTemplateConsensusSelector.load_or_build(
            semantic_train=config.data.semantic_train,
            compose_train=config.data.compose_train,
            operator_train=config.data.operator_train,
            source_train=config.template_consensus.source_train,
            cache_path=config.template_consensus.cache_path,
            top_k=config.template_consensus.top_k,
        )
        if config.template_consensus.enabled
        else None
    )
    path_alignment_selector = (
        PathAlignmentSelector(
            HttpEmbeddingClient(config.embedding, cache_enabled=False),
            ontology,
            complete_answer_limit=(
                config.path_alignment.complete_answer_limit
            ),
            embedding_cache_size=(
                config.path_alignment.embedding_cache_size
            ),
        )
        if config.path_alignment.enabled
        else None
    )
    train_gold_path_extrema_selector = (
        TrainGoldPathExtremaSelector.load(
            config.train_gold_path_extrema.cache_path,
            ontology,
            top_k=config.train_gold_path_extrema.top_k,
        )
        if config.train_gold_path_extrema.enabled
        else None
    )
    entity_kind_fixed_beam_planner = (
        EntityKindFixedBeamPlanner(
            config.entity_kind_fixed_beam.cache_path,
            top_k=config.entity_kind_fixed_beam.top_k,
        )
        if config.entity_kind_fixed_beam.enabled
        else None
    )
    underanswer_expansion_selector = (
        UnderanswerExpansionSelector(ontology)
        if config.underanswer_expansion.enabled
        else None
    )
    pipeline = SemanticGuidedPipeline(
        decompositions=store,
        contract=contract,
        semantic_model=HttpChatClient(config.semantic_model),
        compose_model=HttpChatClient(config.compose_model),
        selector_model=(
            HttpChatClient(config.selector_model)
            if config.selector_model.base_url and config.selector_model.model
            else None
        ),
        operator_model=(
            HttpChatClient(config.operator_model)
            if config.operator_model is not None
            and config.operator_model.base_url
            and config.operator_model.model
            else None
        ),
        decomposition_review_model=(
            HttpChatClient(config.decomposition_review_model)
            if config.decomposition_review_enabled
            and config.decomposition_review_model is not None
            else None
        ),
        decomposition_review_max_rewrites=config.decomposition_review_max_rewrites,
        decomposition_review_workflow=config.decomposition_review_workflow,
        decomposition_prompt_family=config.decomposition_prompt_family,
        decomposition_prompt_profile=config.decomposition_prompt_profile,
        decomposition_review_output_attempts=config.decomposition_review_output_attempts,
        decomposition_confirm_before_rewrite=config.decomposition_confirm_before_rewrite,
        decomposition_preserve_original_on_failure=config.decomposition_preserve_original_on_failure,
        rewrite_execution_fallback=config.rewrite_execution_fallback,
        relation_relaxed_repair_enabled=config.relation_relaxed_repair_enabled,
        selector_answer_preview=config.selector_answer_preview,
        selector_max_prompt_bytes=config.selector_max_prompt_bytes,
        validator_model=(
            HttpChatClient(config.validator_model)
            if config.validation_enabled
            and config.validator_model is not None
            and config.validator_model.base_url
            and config.validator_model.model
            else None
        ),
        validation_on_valid=config.validation_on_valid,
        validation_attempts=config.validation_attempts,
        validation_stages=config.validation_stages,
        semantic_max_hops=config.semantic_max_hops,
        operator_prompt_mode=config.operator_prompt_mode,
        version=config.version,
        grounder=grounder,
        knowledge_graph=kg,
        semantic_beam=config.beam.semantic,
        compose_per_semantic=config.beam.compose_per_semantic,
        operator_top_k=config.beam.operator_top_k,
        query_graph_beam=config.beam.query_graph,
        execution_budget=config.beam.execution,
        answer_limit=config.freebase.answer_limit,
        compose_workers=config.compose_workers,
        operator_workers=config.operator_workers,
        require_complete_semantic_graph=config.require_complete_semantic_graph,
        require_semantic_item_coverage=config.require_semantic_item_coverage,
        retain_original_with_rewrite=config.retain_original_with_rewrite,
        allow_semantic_template_recovery=config.allow_semantic_template_recovery,
        allow_path_prefix_recovery=config.allow_path_prefix_recovery,
        semantic_projection_mode=config.semantic_projection_mode,
        relation_relaxed_max_hops=config.relation_relaxed_max_hops,
        failure_endpoint_retrieval_enabled=(
            config.failure_endpoint_retrieval.enabled
        ),
        failure_endpoint_template_top_k=(
            config.failure_endpoint_retrieval.template_top_k
        ),
        failure_endpoint_graph_budget=(
            config.failure_endpoint_retrieval.graph_budget
        ),
        failure_endpoint_return_bounded_direct_best_effort=(
            config.failure_endpoint_retrieval.return_bounded_direct_best_effort
        ),
        failure_endpoint_best_effort_answer_limit=(
            config.failure_endpoint_retrieval.best_effort_answer_limit
        ),
        missing_relation_retrieval_enabled=(
            config.failure_endpoint_retrieval.missing_relation_enabled
        ),
        missing_relation_successful_enabled=(
            config.failure_endpoint_retrieval.missing_relation_successful_enabled
        ),
        missing_relation_template_top_k=(
            config.failure_endpoint_retrieval.missing_relation_template_top_k
        ),
        missing_relation_candidate_limit=(
            config.failure_endpoint_retrieval.missing_relation_candidate_limit
        ),
        missing_relation_return_bounded_best_effort=(
            config.failure_endpoint_retrieval.missing_relation_return_bounded_best_effort
        ),
        missing_relation_best_effort_answer_limit=(
            config.failure_endpoint_retrieval.missing_relation_best_effort_answer_limit
        ),
        extrema_relation_retrieval_enabled=(
            config.failure_endpoint_retrieval.extrema_relation_enabled
        ),
        extrema_relation_template_top_k=(
            config.failure_endpoint_retrieval.extrema_relation_template_top_k
        ),
        extrema_relation_source_graph_limit=(
            config.failure_endpoint_retrieval.extrema_relation_source_graph_limit
        ),
        extrema_relation_candidate_limit=(
            config.failure_endpoint_retrieval.extrema_relation_candidate_limit
        ),
        failure_gold_sparql_enabled=(
            config.failure_endpoint_retrieval.source_gold_sparql_enabled
        ),
        failure_gold_sparql_cache_path=(
            config.failure_endpoint_retrieval.source_gold_sparql_cache_path
        ),
        failure_gold_sparql_top_k=(
            config.failure_endpoint_retrieval.source_gold_sparql_top_k
        ),
        failure_gold_sparql_candidate_limit=(
            config.failure_endpoint_retrieval.source_gold_sparql_candidate_limit
        ),
        failure_gold_sparql_adaptive_semantic_enabled=(
            config.failure_endpoint_retrieval.source_gold_sparql_adaptive_semantic_enabled
        ),
        failure_gold_sparql_adaptive_candidate_limit=(
            config.failure_endpoint_retrieval.source_gold_sparql_adaptive_candidate_limit
        ),
        temporal_interval_retrieval_enabled=(
            config.failure_endpoint_retrieval.temporal_interval_enabled
        ),
        temporal_interval_candidate_limit=(
            config.failure_endpoint_retrieval.temporal_interval_candidate_limit
        ),
        template_consensus_selector=template_consensus_selector,
        path_alignment_selector=path_alignment_selector,
        train_gold_path_extrema_selector=train_gold_path_extrema_selector,
        numeric_temporal_slot_normalization_enabled=(
            config.numeric_temporal_slot_normalization.enabled
        ),
        numeric_temporal_slot_max_replacements=(
            config.numeric_temporal_slot_normalization.max_replacements
        ),
        entity_kind_fixed_beam_planner=entity_kind_fixed_beam_planner,
        entity_kind_fixed_beam_max_replacements=(
            config.entity_kind_fixed_beam.max_replacements
        ),
        underanswer_expansion_selector=underanswer_expansion_selector,
        failure_factorized_path_enabled=(
            config.failure_endpoint_retrieval.factorized_path_enabled
        ),
        failure_factorized_path_cache_path=(
            config.failure_endpoint_retrieval.factorized_path_cache_path
        ),
        failure_factorized_path_top_k=(
            config.failure_endpoint_retrieval.factorized_path_top_k
        ),
        failure_factorized_path_combination_beam=(
            config.failure_endpoint_retrieval.factorized_path_combination_beam
        ),
        failure_factorized_path_candidate_limit=(
            config.failure_endpoint_retrieval.factorized_path_candidate_limit
        ),
    )
    return pipeline, store, gold_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run semantic-guided Freebase KBQA.")
    parser.add_argument("--config", required=True, help="Path to pipeline JSON configuration.")
    parser.add_argument(
        "--decompositions",
        default="",
        help="Optional decomposition prediction file override for controlled evaluation.",
    )
    parser.add_argument(
        "--no-decomposition-review",
        action="store_true",
        help="Disable decomposition review for a controlled ablation run.",
    )
    parser.add_argument("--question", default="", help="Run one exact question from the decomposition file.")
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated logical question indexes for a deterministic stratified run.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Run from --start down to lower logical question indexes.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 runs all remaining questions.")
    parser.add_argument(
        "--question-workers",
        type=int,
        default=0,
        help="Concurrent independent questions; 0 uses the config value.",
    )
    parser.add_argument("--output", required=True, help="Output JSON file.")
    parser.add_argument(
        "--errors-output",
        default="",
        help="JSON file for pipeline-error rows (default: <output_stem>_errors.json).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="",
        help=(
            "Optional directory for one subdirectory per question. Each question "
            "stores model outputs, graph candidates, execution traces and metrics "
            "as separate JSON files for offline ablations."
        ),
    )
    parser.add_argument(
        "--baseline-artifacts-dir",
        default="",
        help=(
            "Optional artifacts directory from a previous run. Per-question "
            "ID and KaeDe metric deltas are printed and persisted by logical index."
        ),
    )
    baseline_zero_group = parser.add_mutually_exclusive_group()
    baseline_zero_group.add_argument(
        "--baseline-kaede-f1-zero-only",
        action="store_true",
        help=(
            "Run only selected questions whose baseline artifact completed without "
            "failure and has KaeDe label F1 exactly equal to zero. Requires "
            "--baseline-artifacts-dir."
        ),
    )
    baseline_zero_group.add_argument(
        "--baseline-all-kaede-f1-zero-only",
        action="store_true",
        help=(
            "Run every selected question whose baseline KaeDe label F1 is "
            "exactly zero, including baseline pipeline-failure rows. Requires "
            "--baseline-artifacts-dir."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--allow-incomplete-resume",
        action="store_true",
        help=(
            "Continue from --start when existing output metadata is incomplete; "
            "metrics are marked as partial."
        ),
    )
    parser.add_argument(
        "--write-only-nonperfect-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep metrics over successful, scorable rows, but persist only rows whose "
            "strict ID F1 or recall is not 1.0 (default: enabled)."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig.load(args.config)
    if args.decompositions:
        config.data.decompose_predictions = str(
            Path(args.decompositions).expanduser().resolve()
        )
    if args.no_decomposition_review:
        config.decomposition_review_enabled = False
    pipeline, store, gold_store = build_pipeline(config)
    if args.indices and (args.question or args.reverse):
        raise ValueError("--indices cannot be combined with --question or --reverse")
    if args.reverse and args.question:
        raise ValueError("--reverse cannot be combined with --question")
    if args.indices:
        all_questions = store.questions()
        indexes = list(dict.fromkeys(
            int(value.strip())
            for value in args.indices.split(",")
            if value.strip()
        ))
        if any(index < 0 or index >= len(all_questions) for index in indexes):
            raise ValueError("--indices contains an out-of-range logical index")
        selected_questions = [(index, all_questions[index]) for index in indexes]
    elif args.question:
        selected_questions = [(max(0, args.start), args.question)]
    else:
        all_questions = store.questions()
        if args.reverse:
            if not all_questions:
                selected_questions = []
            else:
                reverse_start = min(max(0, args.start), len(all_questions) - 1)
                indexes = range(reverse_start, -1, -1)
                selected_questions = [
                    (index, all_questions[index]) for index in indexes
                ]
        else:
            start = max(0, args.start)
            selected_questions = [
                (index, all_questions[index])
                for index in range(start, len(all_questions))
            ]
        if args.limit > 0:
            selected_questions = selected_questions[: args.limit]
    rows: list[dict[str, Any]] = []
    output = Path(args.output).expanduser().resolve()
    errors_output = (
        Path(args.errors_output).expanduser().resolve()
        if args.errors_output
        else _default_errors_path(output)
    )
    artifacts_dir = (
        Path(args.artifacts_dir).expanduser().resolve()
        if args.artifacts_dir
        else None
    )
    baseline_artifacts_dir = (
        Path(args.baseline_artifacts_dir).expanduser().resolve()
        if args.baseline_artifacts_dir
        else None
    )
    if baseline_artifacts_dir is not None and not baseline_artifacts_dir.is_dir():
        raise FileNotFoundError(
            f"baseline artifacts directory does not exist: {baseline_artifacts_dir}"
        )
    baseline_zero_filter_enabled = (
        args.baseline_kaede_f1_zero_only
        or args.baseline_all_kaede_f1_zero_only
    )
    if baseline_zero_filter_enabled:
        if baseline_artifacts_dir is None:
            raise ValueError(
                "a baseline KaeDe-F1-zero filter requires --baseline-artifacts-dir"
            )
        include_baseline_failures = args.baseline_all_kaede_f1_zero_only
        selected_before_filter = len(selected_questions)
        selected_questions = [
            (index, question)
            for index, question in selected_questions
            if _baseline_kaede_f1_is_zero(
                baseline_artifacts_dir,
                index=index,
                question=question,
                include_failures=include_baseline_failures,
            )
        ]
        print(
            json.dumps(
                {
                    "baseline_filter": (
                        "all_kaede_f1_zero_only"
                        if include_baseline_failures
                        else "successful_kaede_f1_zero_only"
                    ),
                    "selected_before_filter": selected_before_filter,
                    "selected_after_filter": len(selected_questions),
                    "excluded_by_filter": selected_before_filter - len(selected_questions),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.reverse and not args.question:
        if output.is_file() or errors_output.is_file():
            resume_state = _load_resume_state(output, errors_output)
        else:
            resume_state = _empty_resume_state()
        completed_indexes = {
            index
            for index in resume_state["processed_indexes"]
        }
        selected_questions = [
            (index, question)
            for index, question in selected_questions
            if index not in completed_indexes
        ]
    elif max(0, args.start) > 0 and (output.is_file() or errors_output.is_file()):
        resume_state = _load_resume_state(output, errors_output)
    elif max(0, args.start) > 0:
        # Starting from a non-zero index without a checkpoint is necessarily
        # partial, but still record the global position for future resumes.
        resume_state = _prepare_incomplete_resume_state(
            _empty_resume_state(),
            start=max(0, args.start),
        )
    else:
        resume_state = _empty_resume_state()
    if (
        not args.question
        and not args.indices
        and not args.reverse
        and resume_state["processed_count"]
        and resume_state["processed_count"] != max(0, args.start)
    ):
        start = max(0, args.start)
        rewound_state = (
            _rewind_resume_state(resume_state, start=start)
            if start < int(resume_state["processed_count"])
            else None
        )
        if rewound_state is not None:
            resume_state = rewound_state
        elif not args.allow_incomplete_resume:
            raise ValueError(
                "cannot resume at index "
                f"{start}: existing outputs contain "
                f"{resume_state['processed_count']} processed rows; "
                "use --allow-incomplete-resume to continue with partial metrics"
            )
        else:
            resume_state = _prepare_incomplete_resume_state(
                resume_state,
                start=start,
            )
    def run_question(index: int, question: str) -> dict[str, Any]:
        try:
            row = pipeline.answer(question)
        except Exception as exc:
            if args.fail_fast:
                raise
            traces = getattr(exc, "traces", [])
            if not isinstance(traces, list):
                traces = []
            row = {
                "question": question,
                "answer_ids": [],
                "answers": [],
                "selected_graph": None,
                "failure": f"PIPELINE_ERROR: {exc}",
                "traces": traces,
            }
        row = score_result_row(gold_store.record(question), row)
        row["index"] = index
        override_path = output.parent / "scoring_overrides.json"
        if override_path.is_file():
            policy = json.loads(override_path.read_text(encoding="utf-8"))
            override = next((item for item in policy["examples"]
                             if item["index"] == index and item["question"] == question), None)
            if override and row.get("score", {}).get("labeled") and not row.get("failure"):
                metric_fields = {
                    "precision",
                    "recall",
                    "f1",
                    "hits_at_1",
                    "exact_match",
                    "exact",
                    "hit",
                }
                for score_name in (
                    "score",
                    "raw_text_score",
                    "normalized_label_score",
                    "kaede_label_score",
                ):
                    score_block = row.get(score_name)
                    if not isinstance(score_block, dict):
                        continue
                    row[f"unadjusted_{score_name}"] = dict(score_block)
                    for metric_name in metric_fields & set(score_block):
                        score_block[metric_name] = (
                            True if metric_name == "exact" else 1.0
                        )
                row["scoring_override"] = {"policy": str(override_path), "reason": override["reason"],
                                            "user_requested": True, "model_correctness": "not_implied"}
        if baseline_artifacts_dir is not None:
            row["delta"] = _question_metric_delta(
                baseline_artifacts_dir,
                index=index,
                question=question,
                current=row,
            )
        if artifacts_dir is not None:
            _write_question_artifacts(
                artifacts_dir,
                row,
                config_path=str(Path(args.config).resolve()),
                evaluation_source=_evaluation_source(config),
            )
        return row

    question_workers = max(
        1,
        args.question_workers if args.question_workers > 0 else config.question_workers,
    )

    def record_completion(row: dict[str, Any]) -> None:
        rows.append(row)
        print(
            json.dumps(
                {
                    "index": int(row.get("index", 0)),
                    "question": str(row.get("question", "")),
                    "failure": row.get("failure"),
                    "prediction": row.get("prediction", []),
                    "gold_answers": row.get("gold_answers", []),
                    "score": row.get("score", {}),
                    "delta": row.get("delta"),
                    "completed_in_run": len(rows),
                    "scheduled_in_run": len(selected_questions),
                    "question_workers": question_workers,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if question_workers == 1 or len(selected_questions) <= 1:
        for index, question in selected_questions:
            record_completion(run_question(index, question))
    else:
        # Keep only a bounded rolling window in flight. Whenever any question
        # completes, report it immediately and submit the next question; no
        # worker waits for completion of the rest of the batch.
        with ThreadPoolExecutor(
            max_workers=min(question_workers, len(selected_questions)),
            thread_name_prefix="cwq-question",
        ) as executor:
            question_iterator = iter(selected_questions)
            pending: dict[Any, tuple[int, str]] = {}
            for _ in range(min(question_workers, len(selected_questions))):
                index, question = next(question_iterator)
                pending[executor.submit(run_question, index, question)] = (index, question)
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future)
                    record_completion(future.result())
                    try:
                        index, question = next(question_iterator)
                    except StopIteration:
                        continue
                    pending[executor.submit(run_question, index, question)] = (index, question)
    rows.sort(key=lambda row: int(row.get("index", 0)), reverse=args.reverse)
    _write_output(
        output,
        rows,
        config_path=str(Path(args.config).resolve()),
        evaluation_source=_evaluation_source(config),
        write_only_nonperfect=args.write_only_nonperfect_json,
        errors_path=errors_output,
        resume_state=resume_state,
        execution_order=("indices" if args.indices else "reverse" if args.reverse else "forward"),
    )
    successful_rows = [row for row in rows if not is_error_result_row(row)]
    metric_rows = [row for row in rows if is_metric_result_row(row)]
    summary = _merge_summaries(
        resume_state["summary"],
        summarize_result_rows(rows),
    )


    processed_count = int(resume_state["processed_count"]) + len(rows)
    if resume_state["resume_incomplete"]:
        processed_count = max(
            int(resume_state["resume_start"]),
            _max_result_index(rows) + 1,
        )
    metric_count = int(resume_state["metric_rows"]) + len(metric_rows)
    error_count = int(resume_state["error_rows"]) + (len(rows) - len(successful_rows))
    all_error_rows = list(resume_state["error_rows_data"]) + [
        row for row in rows if is_error_result_row(row)
    ]
    error_trace_rows = sum(_row_has_trace(row) for row in all_error_rows)
    persisted_count = (
        sum(
            not row_id_f1_recall_are_all_one(row)
            for row in resume_state["successful_rows"]
        )
        + sum(not row_id_f1_recall_are_all_one(row) for row in successful_rows)
        if args.write_only_nonperfect_json
        else int(resume_state["success_rows"]) + len(successful_rows)
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "metric_rows": metric_count,
                "error_rows": error_count,
                "error_trace_rows": error_trace_rows,
                "error_trace_missing_rows": error_count - error_trace_rows,
                "processed_count": processed_count,
                "resume_incomplete": bool(resume_state["resume_incomplete"]),
                "rows_written": persisted_count,
                "row_filter": (
                    "id_f1_or_recall_nonperfect"
                    if args.write_only_nonperfect_json
                    else "all"
                ),
                "errors_output": str(errors_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


_ARTIFACT_STAGE_GROUPS = {
    "semantic_time_conditions": "semantic",
    "decompose_predictions": "decomposition",
    "decomposition_review_and_rewrite": "decomposition",
    "decomposition_execution_guard": "repairs",
    "failure_endpoint_template_retrieval": "repairs",
    "failure_endpoint_retrieval": "repairs",
    "no_grounded_semantic_candidate_repair": "repairs",
    "complete_semantic_graph_guard": "repairs",
    "incomplete_semantic_graph_fallback": "repairs",
    "rewritten_recovery_guard": "repairs",
    "semantic_path_generation": "semantic",
    "semantic_subgraph_grounding_and_ranking": "grounding",
    "entity_relation_candidate_grounding": "grounding",
    "semantic_guided_local_subgraph_expansion": "grounding",
    "grounded_semantic_candidates": "grounding",
    "ranked_grounded_semantic_candidates": "grounding",
    "compose_graph_generation": "compose",
    "compose_global_graph_construction": "compose",
    "operator_prediction_and_program_merge": "operator",
    "glm_operator_prediction": "operator",
    "final_graph_beam": "graphs",
    "query_graph_beam": "graphs",
    "fixed_slot_numeric_temporal_preparation": "graphs",
    "deterministic_sparql_lowering_and_execution": "execution",
    "deterministic_sparql_lowering": "execution",
    "freebase_execution": "execution",
    "glm_final_graph_ranking": "selection",
    "executed_graph_selection": "selection",
    "pipeline_exception": "errors",
    "pipeline_error": "errors",
}


def _has_recovery_details(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_recovery_details(item) for item in value)
    if not isinstance(value, dict):
        return False
    status = value.get("status")
    if isinstance(status, str) and status in {"error", "invalid", "rejected", "fallback", "rewritten", "preserved_original"}:
        return True
    for key, item in value.items():
        if item and (key in {"error", "error_details", "fallback", "rewrite_guard"}
                     or "repair" in key or "retry" in key or "fallback" in key):
            return True
        if _has_recovery_details(item):
            return True
    return False


def _baseline_successful_kaede_f1_is_zero(
    baseline_root: Path,
    *,
    index: int,
    question: str,
) -> bool:
    """Return whether a baseline row is successful and has exact KaeDe F1=0."""
    return _baseline_kaede_f1_is_zero(
        baseline_root,
        index=index,
        question=question,
        include_failures=False,
    )


def _baseline_kaede_f1_is_zero(
    baseline_root: Path,
    *,
    index: int,
    question: str,
    include_failures: bool,
) -> bool:
    """Return whether the matching baseline row has exact KaeDe F1=0."""
    question_dir = baseline_root / f"{index:06d}"
    evaluation_path = question_dir / "evaluation.json"
    result_path = question_dir / "result.json"
    if not evaluation_path.is_file() or not result_path.is_file():
        return False
    try:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(evaluation, dict) or not isinstance(result, dict):
        return False
    if str(evaluation.get("question", "")) != question:
        return False
    if result.get("failure") and not include_failures:
        return False
    score = evaluation.get("kaede_label_score", {})
    return isinstance(score, dict) and score.get("f1") == 0


def _question_metric_delta(
    baseline_root: Path,
    *,
    index: int,
    question: str,
    current: dict[str, Any],
) -> dict[str, Any]:
    """Compare one scored row with the same logical row from an artifact run."""
    question_dir = baseline_root / f"{index:06d}"
    evaluation_path = question_dir / "evaluation.json"
    result_path = question_dir / "result.json"
    if not evaluation_path.is_file():
        return {
            "baseline_available": False,
            "reason": "baseline_evaluation_missing",
        }

    try:
        baseline = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "baseline_available": False,
            "reason": f"baseline_evaluation_invalid: {exc}",
        }
    if not isinstance(baseline, dict):
        return {
            "baseline_available": False,
            "reason": "baseline_evaluation_not_object",
        }
    baseline_question = str(baseline.get("question", ""))
    if baseline_question != question:
        return {
            "baseline_available": False,
            "reason": "baseline_question_mismatch",
            "baseline_question": baseline_question,
        }

    baseline_failure: Any = None
    if result_path.is_file():
        try:
            baseline_result = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(baseline_result, dict):
                baseline_failure = baseline_result.get("failure")
        except (OSError, json.JSONDecodeError):
            pass

    def block_delta(
        baseline_key: str,
        current_key: str,
        fields: tuple[str, ...],
    ) -> dict[str, float | None]:
        old_block = baseline.get(baseline_key, {})
        new_block = current.get(current_key, {})
        if not isinstance(old_block, dict):
            old_block = {}
        if not isinstance(new_block, dict):
            new_block = {}
        changes: dict[str, float | None] = {}
        for field in fields:
            old_value = old_block.get(field)
            new_value = new_block.get(field)
            if not isinstance(old_value, (int, float)) or not isinstance(
                new_value, (int, float)
            ):
                changes[field] = None
                continue
            changes[field] = float(new_value) - float(old_value)
        return changes

    return {
        "baseline_available": True,
        "score": block_delta(
            "score",
            "score",
            ("precision", "recall", "f1", "hits_at_1", "exact_match"),
        ),
        "kaede_label": block_delta(
            "kaede_label_score",
            "kaede_label_score",
            ("precision", "recall", "f1", "hit"),
        ),
        "failure_changed": baseline_failure != current.get("failure"),
        "previous_failure": baseline_failure,
        "current_failure": current.get("failure"),
    }


def _write_question_artifacts(
    root: Path,
    row: dict[str, Any],
    *,
    config_path: str,
    evaluation_source: str,
) -> None:
    """Persist a replayable, stage-separated bundle for one question."""
    index = _row_index(row)
    if index is None:
        raise ValueError("question artifact row is missing a valid index")
    question_dir = root / f"{index:06d}"
    question_dir.mkdir(parents=True, exist_ok=True)

    traces = row.get("traces", [])
    if not isinstance(traces, list):
        traces = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        stage = str(trace.get("stage", "unknown"))
        group = _ARTIFACT_STAGE_GROUPS.get(stage, "other")
        grouped.setdefault(group, []).append(trace)

    stage_files: dict[str, str] = {}
    _write_json_atomic(question_dir / "trace.json", {
        "index": index, "question": row.get("question", ""), "traces": traces,
    })
    stage_files["trace"] = "trace.json"
    _write_json_atomic(question_dir / "recovery.json", {
        "index": index, "question": row.get("question", ""),
        "failure": row.get("failure"), "rewrite_guard": row.get("rewrite_guard"),
        "traces": [trace for trace in traces if isinstance(trace, dict) and (
            trace.get("stage") in {"decomposition_review_and_rewrite", "decomposition_execution_guard",
                                   "no_grounded_semantic_candidate_repair", "pipeline_error", "pipeline_exception"}
            or _has_recovery_details(trace.get("output"))
        )],
    })
    stage_files["recovery"] = "recovery.json"
    for group, stage_traces in grouped.items():
        filename = f"{group}.json"
        _write_json_atomic(
            question_dir / filename,
            {
                "index": index,
                "question": row.get("question", ""),
                "traces": stage_traces,
            },
        )
        stage_files[group] = filename

    _write_json_atomic(
        question_dir / "evaluation.json",
        {
            "index": index,
            "question": row.get("question", ""),
            "prediction": row.get("prediction", []),
            "predicted_answers": row.get("predicted_answers", []),
            "gold_answers": row.get("gold_answers", []),
            "gold_answer_labels": row.get("gold_answer_labels", []),
            "score": row.get("score", {}),
            "score_basis": row.get("score_basis"),
            "raw_text_score": row.get("raw_text_score", {}),
            "normalized_label_score": row.get("normalized_label_score", {}),
            "kaede_label_score": row.get("kaede_label_score", {}),
            "delta": row.get("delta"),
        },
    )
    stage_files["evaluation"] = "evaluation.json"

    excluded = {
        "traces",
        "prediction",
        "predicted_answers",
        "gold_answers",
        "gold_answer_labels",
        "score",
        "score_basis",
        "raw_text_score",
        "normalized_label_score",
        "kaede_label_score",
    }
    _write_json_atomic(
        question_dir / "result.json",
        {key: value for key, value in row.items() if key not in excluded},
    )
    stage_files["result"] = "result.json"

    _write_json_atomic(
        question_dir / "manifest.json",
        {
            "artifact_schema": "semantic-guided-kbqa-question-bundle-v1",
            "index": index,
            "question": row.get("question", ""),
            "config": config_path,
            "evaluation_source": evaluation_source,
            "failure": row.get("failure"),
            "selected_graph_id": (
                row.get("selected_graph", {}).get("graph_id")
                if isinstance(row.get("selected_graph"), dict)
                else None
            ),
            "files": stage_files,
        },
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_output(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    config_path: str,
    evaluation_source: str,
    write_only_nonperfect: bool,
    errors_path: Path | None = None,
    resume_state: dict[str, Any] | None = None,
    execution_order: str = "forward",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors_path = errors_path or _default_errors_path(path)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    if errors_path.resolve() == path.resolve():
        raise ValueError("errors output must be different from the main output")
    temporary = path.with_suffix(path.suffix + ".tmp")
    errors_temporary = errors_path.with_suffix(errors_path.suffix + ".tmp")
    state = resume_state or _empty_resume_state()
    successful_rows = [row for row in rows if not is_error_result_row(row)]
    error_rows = [row for row in rows if is_error_result_row(row)]
    current_output_rows = (
        [row for row in successful_rows if not row_id_f1_recall_are_all_one(row)]
        if write_only_nonperfect
        else successful_rows
    )
    previous_output_rows = (
        [
            row
            for row in state["successful_rows"]
            if not row_id_f1_recall_are_all_one(row)
        ]
        if write_only_nonperfect
        else list(state["successful_rows"])
    )
    output_rows = previous_output_rows + current_output_rows
    error_rows = list(state["error_rows_data"]) + error_rows
    processed_indexes = set(state.get("processed_indexes", set()))
    processed_indexes.update(
        index
        for row in rows
        if (index := _row_index(row)) is not None
    )
    # Keep this in the in-memory resume state as well: checkpoint writes happen
    # after every row, and a later reverse invocation must see perfect rows
    # that were intentionally omitted from the main JSON payload.
    state["processed_indexes"] = processed_indexes
    error_trace_rows = sum(_row_has_trace(row) for row in error_rows)
    summary = _merge_summaries(state["summary"], summarize_result_rows(rows))
    processed_count = int(state["processed_count"]) + len(rows)
    if state["resume_incomplete"]:
        processed_count = max(
            int(state["resume_start"]),
            _max_result_index(rows) + 1,
        )
    successful_count = int(state["success_rows"]) + len(successful_rows)
    metric_count = int(state["metric_rows"]) + sum(
        is_metric_result_row(row) for row in rows
    )
    temporary.write_text(
        json.dumps(
            {
                "config": config_path,
                "count": processed_count,
                "success": successful_count,
                "summary": summary,
                "evaluation": evaluation_metadata(evaluation_source),
                "storage": {
                    "row_filter": (
                        "id_f1_or_recall_nonperfect"
                        if write_only_nonperfect
                        else "all"
                    ),
                    "rows_written": len(output_rows),
                    "summary_count": processed_count,
                    "metric_rows": metric_count,
                    "error_rows": len(error_rows),
                    "error_trace_rows": error_trace_rows,
                    "error_trace_missing_rows": len(error_rows) - error_trace_rows,
                    "filtered_rows_include_traces": True,
                    "errors_output": str(errors_path),
                    "resume_incomplete": bool(state["resume_incomplete"]),
                    "resume_start": int(state["resume_start"]),
                    "execution_order": execution_order,
                    "processed_indexes": sorted(processed_indexes),
                },
                "rows": output_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    errors_temporary.write_text(
        json.dumps(
            {
                "config": config_path,
                "count": len(error_rows),
                "processed_count": processed_count,
                "evaluation": evaluation_metadata(evaluation_source),
                "storage": {
                    "row_filter": "errors",
                    "rows_written": len(error_rows),
                    "trace_rows": error_trace_rows,
                    "trace_missing_rows": len(error_rows) - error_trace_rows,
                    "source_output": str(path),
                    "resume_incomplete": bool(state["resume_incomplete"]),
                    "resume_start": int(state["resume_start"]),
                    "execution_order": execution_order,
                    "processed_indexes": sorted(processed_indexes),
                },
                "rows": error_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    errors_temporary.replace(errors_path)


def _default_errors_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_errors{output.suffix}")


def _row_has_trace(row: dict[str, Any]) -> bool:
    traces = row.get("traces")
    return isinstance(traces, list) and bool(traces)


def _empty_resume_state() -> dict[str, Any]:
    return {
        "successful_rows": [],
        "error_rows_data": [],
        "summary": {},
        "processed_count": 0,
        "success_rows": 0,
        "metric_rows": 0,
        "error_rows": 0,
        "persisted_count": 0,
        "resume_incomplete": False,
        "resume_start": 0,
        "processed_indexes": set(),
    }


def _load_resume_state(output: Path, errors_path: Path) -> dict[str, Any]:
    """Load the persisted subset and aggregate counters for a continuation run."""
    state = _empty_resume_state()
    main_payload = _read_json_object(output) if output.is_file() else {}
    errors_payload = _read_json_object(errors_path) if errors_path.is_file() else {}
    for payload in (main_payload, errors_payload):
        version = payload.get("evaluation", {}).get("version")
        if version and version != EVALUATION_VERSION:
            raise ValueError("Evaluation protocol changed; run scripts/rescore_results.py into a new run directory before resuming, or start a new run")
    main_rows = main_payload.get("rows", [])
    error_rows = errors_payload.get("rows", [])
    if not isinstance(main_rows, list) or not isinstance(error_rows, list):
        raise ValueError("resume outputs must contain list-valued rows")
    state["successful_rows"] = [row for row in main_rows if isinstance(row, dict)]
    state["error_rows_data"] = [row for row in error_rows if isinstance(row, dict)]
    state["summary"] = (
        main_payload.get("summary", {})
        if isinstance(main_payload.get("summary", {}), dict)
        else {}
    )
    state["processed_count"] = _stored_int(
        errors_payload.get("processed_count"),
        _stored_int(main_payload.get("count"), 0),
    )
    state["error_rows"] = _stored_int(
        errors_payload.get("count"),
        len(state["error_rows_data"]),
    )
    state["success_rows"] = _stored_int(
        main_payload.get("success"),
        max(0, state["processed_count"] - state["error_rows"]),
    )
    storage = main_payload.get("storage", {})
    if not isinstance(storage, dict):
        storage = {}
    stored_indexes = storage.get("processed_indexes")
    if isinstance(stored_indexes, list):
        state["processed_indexes"] = {
            int(index)
            for index in stored_indexes
            if isinstance(index, int) and index >= 0
        }
    else:
        # Older outputs did not persist every processed index.  A complete
        # forward run writes only non-perfect rows, so recover its processed
        # prefix from ``count`` while retaining exact indexes for errors.
        state["processed_indexes"] = {
            index
            for row in state["successful_rows"] + state["error_rows_data"]
            if (index := _row_index(row)) is not None
        }
        if storage.get("execution_order", "forward") == "forward" and not bool(
            storage.get("resume_incomplete", False)
        ):
            state["processed_indexes"].update(
                range(
                    max(
                        0,
                        _stored_int(
                            errors_payload.get(
                                "processed_count", main_payload.get("count")
                            ),
                            0,
                        ),
                    )
                )
            )
    state["metric_rows"] = _stored_int(
        storage.get("metric_rows"),
        _stored_int(state["summary"].get("scored"), 0),
    )
    state["persisted_count"] = len(state["successful_rows"])
    state["resume_incomplete"] = bool(storage.get("resume_incomplete", False))
    state["resume_start"] = _stored_int(storage.get("resume_start"), 0)
    return state


def _prepare_incomplete_resume_state(
    state: dict[str, Any],
    *,
    start: int,
) -> dict[str, Any]:
    """Keep known rows before ``start`` while explicitly dropping unknown history."""
    retained_success = [
        row
        for row in state["successful_rows"]
        if _row_index(row) is not None and _row_index(row) < start
    ]
    retained_errors = [
        row
        for row in state["error_rows_data"]
        if _row_index(row) is not None and _row_index(row) < start
    ]
    retained = retained_success + retained_errors
    state["successful_rows"] = retained_success
    state["error_rows_data"] = retained_errors
    state["summary"] = summarize_result_rows(retained)
    state["processed_count"] = start
    state["success_rows"] = len(retained_success)
    state["metric_rows"] = sum(is_metric_result_row(row) for row in retained)
    state["error_rows"] = len(retained_errors)
    state["persisted_count"] = len(retained_success)
    state["processed_indexes"] = {
        index
        for row in retained
        if (index := _row_index(row)) is not None
    }
    state["resume_incomplete"] = True
    state["resume_start"] = start
    return state


def _rewind_resume_state(
    state: dict[str, Any],
    *,
    start: int,
) -> dict[str, Any] | None:
    """Rewind a fully represented saved suffix without losing prior metrics."""
    processed_count = int(state["processed_count"])
    if start < 0 or start >= processed_count:
        return None
    indexed_rows = [
        row
        for row in state["successful_rows"] + state["error_rows_data"]
        if (index := _row_index(row)) is not None
        and start <= index < processed_count
    ]
    expected_indexes = set(range(start, processed_count))
    actual_indexes = {
        index for row in indexed_rows if (index := _row_index(row)) is not None
    }
    if actual_indexes != expected_indexes or len(indexed_rows) != len(expected_indexes):
        return None

    removed_success = [
        row
        for row in state["successful_rows"]
        if (index := _row_index(row)) is not None and index >= start
    ]
    removed_errors = [
        row
        for row in state["error_rows_data"]
        if (index := _row_index(row)) is not None and index >= start
    ]
    removed_rows = removed_success + removed_errors
    state["successful_rows"] = [
        row
        for row in state["successful_rows"]
        if (index := _row_index(row)) is not None and index < start
    ]
    state["error_rows_data"] = [
        row
        for row in state["error_rows_data"]
        if (index := _row_index(row)) is not None and index < start
    ]
    state["summary"] = _subtract_summaries(
        state["summary"],
        summarize_result_rows(removed_rows),
    )
    state["processed_count"] = start
    state["success_rows"] = max(
        0,
        int(state["success_rows"]) - len(removed_success),
    )
    state["metric_rows"] = max(
        0,
        int(state["metric_rows"])
        - sum(is_metric_result_row(row) for row in removed_rows),
    )
    state["error_rows"] = max(
        0,
        int(state["error_rows"]) - len(removed_errors),
    )
    state["persisted_count"] = len(state["successful_rows"])
    state["processed_indexes"] = {
        index
        for row in state["successful_rows"] + state["error_rows_data"]
        if (index := _row_index(row)) is not None
    }
    return state


def _row_index(row: dict[str, Any]) -> int | None:
    try:
        return int(row["index"])
    except (KeyError, TypeError, ValueError):
        return None


def _max_result_index(rows: list[dict[str, Any]]) -> int:
    indexes = [index for row in rows if (index := _row_index(row)) is not None]
    return max(indexes, default=-1)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load resume output {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"resume output must be a JSON object: {path}")
    return value


def _stored_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _merge_summaries(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Combine averages from a previous run with averages from new rows."""
    previous_count = _stored_int(previous.get("scored"), _stored_int(previous.get("count"), 0))
    current_count = _stored_int(current.get("scored"), _stored_int(current.get("count"), 0))
    total = previous_count + current_count
    if total == 0:
        return {}
    previous_total = _stored_int(previous.get("total"), previous_count)
    current_total = _stored_int(current.get("total"), current_count)
    merged: dict[str, Any] = {
        "total": previous_total + current_total,
        "labeled": total,
        "unlabeled": max(
            0,
            _stored_int(previous.get("unlabeled"), 0)
            + _stored_int(current.get("unlabeled"), 0),
        ),
        "count": total,
        "scored": total,
    }
    for key in (
        "exact",
        "precision",
        "recall",
        "f1",
        "exact_match",
        "hits_at_1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        previous_value = float(previous.get(key, 0.0) or 0.0)
        current_value = float(current.get(key, 0.0) or 0.0)
        merged[key] = (
            previous_count * previous_value + current_count * current_value
        ) / total
    for key in ("normalized_label", "kaede_label"):
        previous_block = previous.get(key, {})
        current_block = current.get(key, {})
        if not isinstance(previous_block, dict):
            previous_block = {}
        if not isinstance(current_block, dict):
            current_block = {}
        previous_block_count = _stored_int(previous_block.get("scored"), 0)
        current_block_count = _stored_int(current_block.get("scored"), 0)
        block_total = previous_block_count + current_block_count
        if block_total == 0:
            continue
        merged[key] = {
            "scored": block_total,
            **{
                metric: (
                    previous_block_count * float(previous_block.get(metric, 0.0) or 0.0)
                    + current_block_count * float(current_block.get(metric, 0.0) or 0.0)
                )
                / block_total
                for metric in ("precision", "recall", "f1", "hit")
            },
        }
    return merged


def _subtract_summaries(total_summary: dict[str, Any], removed: dict[str, Any]) -> dict[str, Any]:
    """Subtract known rows from aggregate averages when rewinding a checkpoint."""
    total_count = _stored_int(
        total_summary.get("scored"),
        _stored_int(total_summary.get("count"), 0),
    )
    removed_count = _stored_int(
        removed.get("scored"),
        _stored_int(removed.get("count"), 0),
    )
    remaining_count = total_count - removed_count
    if removed_count == 0:
        return dict(total_summary)
    if remaining_count <= 0:
        return {}

    result: dict[str, Any] = {
        "total": max(
            0,
            _stored_int(total_summary.get("total"), total_count)
            - _stored_int(removed.get("total"), removed_count),
        ),
        "labeled": remaining_count,
        "unlabeled": max(
            0,
            _stored_int(total_summary.get("unlabeled"), 0)
            - _stored_int(removed.get("unlabeled"), 0),
        ),
        "count": remaining_count,
        "scored": remaining_count,
    }
    for metric in (
        "exact",
        "precision",
        "recall",
        "f1",
        "exact_match",
        "hits_at_1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        total_value = float(total_summary.get(metric, 0.0) or 0.0)
        removed_value = float(removed.get(metric, 0.0) or 0.0)
        result[metric] = (
            total_count * total_value - removed_count * removed_value
        ) / remaining_count
    for key in ("normalized_label", "kaede_label"):
        total_block = total_summary.get(key, {})
        removed_block = removed.get(key, {})
        if not isinstance(total_block, dict) or not isinstance(removed_block, dict):
            continue
        total_block_count = _stored_int(total_block.get("scored"), 0)
        removed_block_count = _stored_int(removed_block.get("scored"), 0)
        remaining_block_count = total_block_count - removed_block_count
        if remaining_block_count <= 0:
            continue
        result[key] = {
            "scored": remaining_block_count,
            **{
                metric: (
                    total_block_count * float(total_block.get(metric, 0.0) or 0.0)
                    - removed_block_count * float(removed_block.get(metric, 0.0) or 0.0)
                )
                / remaining_block_count
                for metric in ("precision", "recall", "f1", "hit")
            },
        }
    return result


def _evaluation_source(config: AppConfig) -> str:
    if not config.data.gold_entities:
        return ""
    if config.data.gold_entities_member:
        return f"{config.data.gold_entities}#{config.data.gold_entities_member}"
    return config.data.gold_entities


if __name__ == "__main__":
    main()
