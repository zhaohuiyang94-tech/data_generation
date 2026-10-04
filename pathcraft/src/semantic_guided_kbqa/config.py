from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any


_DATASET_DIRECTORY_FILES: dict[str, tuple[str, ...]] = {
    "decompose_predictions": (
        "decompose_train_test.json",
        "decompose_train_no_operator.json",
        "decompose_train.json",
        "decompose_test_pred.json",
    ),
    "semantic_train": (
        "semantic_path_train_test.json",
        "semantic_path_train_no_operator.json",
        "semantic_path_train.json",
    ),
    "compose_train": (
        "compose_train_test.json",
        "compose_train_no_operator.json",
        "compose_train.json",
    ),
    "operator_train": ("operator_train.json",),
}


def _data_source_exists(path: str, field_name: str) -> bool:
    """Return whether a configured dataset file or supported directory exists."""
    source = Path(path)
    if source.is_file():
        return True
    if not source.is_dir():
        return False
    return any((source / name).is_file() for name in _DATASET_DIRECTORY_FILES.get(field_name, ()))


def _secret(value: str) -> str:
    if value.startswith("env:"):
        return os.environ.get(value[4:], "")
    return value


def _bool_value(value: Any, default: bool = False) -> bool:
    """Parse JSON booleans and human-readable string overrides safely."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off", ""}:
        return False
    return default


@dataclass(slots=True)
class ChatConfig:
    base_url: str
    model: str
    api: str = "chat_completions"
    api_key: str = ""
    timeout: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 2048
    retries: int = 0
    retry_delay: float = 1.0
    thinking: str = ""
    reasoning_effort: str = ""
    prompt_mode: str = "system_user"
    # `prompt` uses instruction + local validation for the stock Factory API.
    schema_mode: str = "guided_json"
    # None preserves provider defaults; set false for providers that require it.
    parallel_tool_calls: bool | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChatConfig":
        thinking = value.get("thinking", "")
        if isinstance(thinking, dict):
            thinking = thinking.get("type", "")
        return cls(
            base_url=str(value.get("base_url", "")).rstrip("/"),
            model=str(value.get("model", "")),
            api=str(value.get("api", "chat_completions")).casefold(),
            prompt_mode=str(value.get("prompt_mode", "system_user")).casefold(),
            schema_mode=str(value.get("schema_mode", "guided_json")).casefold(),
            api_key=_secret(str(value.get("api_key", ""))),
            timeout=float(value.get("timeout", 120.0)),
            temperature=float(value.get("temperature", 0.0)),
            max_tokens=int(value.get("max_tokens", 2048)),
            retries=max(0, int(value.get("retries", 0))),
            retry_delay=max(0.0, float(value.get("retry_delay", 1.0))),
            thinking=str(thinking).casefold(),
            reasoning_effort=str(value.get("reasoning_effort", "")).casefold(),
            parallel_tool_calls=(
                bool(value["parallel_tool_calls"])
                if "parallel_tool_calls" in value
                else None
            ),
        )


@dataclass(slots=True)
class EmbeddingConfig:
    url: str
    model: str
    api_key: str = ""
    timeout: float = 120.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EmbeddingConfig":
        return cls(
            url=str(value.get("url", "")),
            model=str(value.get("model", "")),
            api_key=_secret(str(value.get("api_key", ""))),
            timeout=float(value.get("timeout", 120.0)),
        )


@dataclass(slots=True)
class FreebaseConfig:
    endpoint: str
    method: str = "GET"
    timeout: float = 60.0
    answer_limit: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FreebaseConfig":
        return cls(
            endpoint=str(value.get("endpoint", "")),
            method=str(value.get("method", "GET")).upper(),
            timeout=float(value.get("timeout", 60.0)),
            answer_limit=int(value.get("answer_limit", 0)),
        )


@dataclass(slots=True)
class OntologyConfig:
    enabled: bool = False
    directory: str = ""
    relation_top_k: int = 8

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OntologyConfig":
        return cls(
            enabled=_bool_value(value.get("enabled", False), default=False),
            directory=str(value.get("directory", "")).strip(),
            relation_top_k=max(1, int(value.get("relation_top_k", 8))),
        )


@dataclass(slots=True)
class FailureEndpointRetrievalConfig:
    """Deterministic retrieval used only after the normal pipeline is empty."""

    enabled: bool = False
    template_top_k: int = 24
    graph_budget: int = 80
    return_bounded_direct_best_effort: bool = False
    best_effort_answer_limit: int = 64
    missing_relation_enabled: bool = False
    missing_relation_successful_enabled: bool = False
    missing_relation_template_top_k: int = 64
    missing_relation_candidate_limit: int = 3
    missing_relation_return_bounded_best_effort: bool = False
    missing_relation_best_effort_answer_limit: int = 64
    extrema_relation_enabled: bool = False
    extrema_relation_template_top_k: int = 64
    extrema_relation_source_graph_limit: int = 3
    extrema_relation_candidate_limit: int = 3
    source_gold_sparql_enabled: bool = False
    source_gold_sparql_cache_path: str = ""
    source_gold_sparql_top_k: int = 32
    source_gold_sparql_candidate_limit: int = 3
    source_gold_sparql_adaptive_semantic_enabled: bool = False
    source_gold_sparql_adaptive_candidate_limit: int = 3
    temporal_interval_enabled: bool = False
    temporal_interval_candidate_limit: int = 3
    factorized_path_enabled: bool = False
    factorized_path_cache_path: str = ""
    factorized_path_top_k: int = 6
    factorized_path_combination_beam: int = 5
    factorized_path_candidate_limit: int = 3

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FailureEndpointRetrievalConfig":
        return cls(
            enabled=_bool_value(value.get("enabled", False), default=False),
            template_top_k=min(64, max(1, int(value.get("template_top_k", 24)))),
            graph_budget=min(80, max(1, int(value.get("graph_budget", 80)))),
            return_bounded_direct_best_effort=_bool_value(
                value.get("return_bounded_direct_best_effort", False),
                default=False,
            ),
            best_effort_answer_limit=min(
                512,
                max(1, int(value.get("best_effort_answer_limit", 64))),
            ),
            missing_relation_enabled=_bool_value(
                value.get("missing_relation_enabled", False),
                default=False,
            ),
            missing_relation_successful_enabled=_bool_value(
                value.get("missing_relation_successful_enabled", False),
                default=False,
            ),
            missing_relation_template_top_k=min(
                64,
                max(3, int(value.get("missing_relation_template_top_k", 64))),
            ),
            missing_relation_candidate_limit=min(
                3,
                max(1, int(value.get("missing_relation_candidate_limit", 3))),
            ),
            missing_relation_return_bounded_best_effort=_bool_value(
                value.get("missing_relation_return_bounded_best_effort", False),
                default=False,
            ),
            missing_relation_best_effort_answer_limit=min(
                512,
                max(
                    1,
                    int(value.get("missing_relation_best_effort_answer_limit", 64)),
                ),
            ),
            extrema_relation_enabled=_bool_value(
                value.get("extrema_relation_enabled", False),
                default=False,
            ),
            extrema_relation_template_top_k=min(
                64,
                max(3, int(value.get("extrema_relation_template_top_k", 64))),
            ),
            extrema_relation_source_graph_limit=min(
                3,
                max(1, int(value.get("extrema_relation_source_graph_limit", 3))),
            ),
            extrema_relation_candidate_limit=min(
                3,
                max(1, int(value.get("extrema_relation_candidate_limit", 3))),
            ),
            source_gold_sparql_enabled=_bool_value(
                value.get("source_gold_sparql_enabled", False),
                default=False,
            ),
            source_gold_sparql_cache_path=str(
                value.get("source_gold_sparql_cache_path", "")
            ).strip(),
            source_gold_sparql_top_k=min(
                64,
                max(3, int(value.get("source_gold_sparql_top_k", 32))),
            ),
            source_gold_sparql_candidate_limit=min(
                3,
                max(1, int(value.get("source_gold_sparql_candidate_limit", 3))),
            ),
            source_gold_sparql_adaptive_semantic_enabled=_bool_value(
                value.get("source_gold_sparql_adaptive_semantic_enabled", False),
                default=False,
            ),
            source_gold_sparql_adaptive_candidate_limit=min(
                3,
                max(
                    1,
                    int(
                        value.get(
                            "source_gold_sparql_adaptive_candidate_limit", 3
                        )
                    ),
                ),
            ),
            temporal_interval_enabled=_bool_value(
                value.get("temporal_interval_enabled", False),
                default=False,
            ),
            temporal_interval_candidate_limit=min(
                3,
                max(1, int(value.get("temporal_interval_candidate_limit", 3))),
            ),
            factorized_path_enabled=_bool_value(
                value.get("factorized_path_enabled", False),
                default=False,
            ),
            factorized_path_cache_path=str(
                value.get("factorized_path_cache_path", "")
            ).strip(),
            factorized_path_top_k=min(
                16,
                max(1, int(value.get("factorized_path_top_k", 6))),
            ),
            factorized_path_combination_beam=min(
                8,
                max(1, int(value.get("factorized_path_combination_beam", 5))),
            ),
            factorized_path_candidate_limit=min(
                3,
                max(1, int(value.get("factorized_path_candidate_limit", 3))),
            ),
        )


@dataclass(slots=True)
class TemplateConsensusConfig:
    """Gold-free source-verified train-template post-selection gate."""

    enabled: bool = False
    source_train: str = ""
    cache_path: str = ""
    top_k: int = 48

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TemplateConsensusConfig":
        return cls(
            enabled=_bool_value(value.get("enabled", False), default=False),
            source_train=str(value.get("source_train", "")).strip(),
            cache_path=str(value.get("cache_path", "")).strip(),
            top_k=min(48, max(1, int(value.get("top_k", 48)))),
        )


@dataclass(slots=True)
class PathAlignmentConfig:
    """Frozen local-BGE post-selection path alignment."""

    enabled: bool = False
    complete_answer_limit: int = 100
    embedding_cache_size: int = 1024

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PathAlignmentConfig":
        return cls(
            enabled=_bool_value(value.get("enabled", False), default=False),
            complete_answer_limit=min(
                100,
                max(1, int(value.get("complete_answer_limit", 100))),
            ),
            embedding_cache_size=max(
                256,
                int(value.get("embedding_cache_size", 1024)),
            ),
        )


@dataclass(slots=True)
class TrainGoldPathExtremaConfig:
    """Frozen local TRAIN-Gold path gate for unique extrema questions."""

    enabled: bool = False
    cache_path: str = ""
    top_k: int = 48

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainGoldPathExtremaConfig":
        return cls(
            enabled=_bool_value(value.get("enabled", False), default=False),
            cache_path=str(value.get("cache_path", "")).strip(),
            top_k=min(48, max(1, int(value.get("top_k", 48)))),
        )


@dataclass(slots=True)
class NumericTemporalSlotNormalizationConfig:
    """Ontology-only scalar repair within the existing execution beam."""

    enabled: bool = False
    max_replacements: int = 3

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> "NumericTemporalSlotNormalizationConfig":
        return cls(
            enabled=_bool_value(value.get("enabled", False), default=False),
            max_replacements=min(
                3,
                max(1, int(value.get("max_replacements", 3))),
            ),
        )


@dataclass(slots=True)
class EntityKindFixedBeamConfig:
    """TRAIN-template entity repair sharing the existing execution slots."""

    enabled: bool = False
    cache_path: str = ""
    top_k: int = 32
    max_replacements: int = 3

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EntityKindFixedBeamConfig":
        return cls(
            enabled=_bool_value(value.get("enabled", False), default=False),
            cache_path=str(value.get("cache_path", "")).strip(),
            top_k=min(64, max(3, int(value.get("top_k", 32)))),
            max_replacements=min(
                3,
                max(1, int(value.get("max_replacements", 3))),
            ),
        )


@dataclass(slots=True)
class UnderanswerExpansionConfig:
    """Model-free final post-selector over already executed candidates."""

    enabled: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UnderanswerExpansionConfig":
        return cls(
            enabled=_bool_value(value.get("enabled", False), default=False),
        )


@dataclass(slots=True)
class BeamConfig:
    semantic: int = 2
    entity_top_k: int = 3
    first_hop_top_k: int = 8
    second_hop_top_k: int = 8
    # Per-parent beam used by the WebQSP-style recursive expansion.  The
    # legacy full-path strategy continues to use first/second_hop_top_k.
    hop_top_k: int = 8
    # Global number of partial paths retained after every recursive hop.
    path_beam: int = 200
    # Raw neighbor rows fetched per recursive state before semantic ranking.
    # A relation can have many values, so this is intentionally larger than
    # ``hop_top_k``; it can be lowered for a small local endpoint.
    hop_query_limit: int = 1024
    # Number of endpoint values retained for each selected relation.  Keeping
    # more than one value is important when a relation fans out through CVTs.
    hop_values_per_relation: int = 2
    grounded_semantic: int = 12
    compose_per_semantic: int = 2
    operator_top_k: int = 0
    query_graph: int = 20
    execution: int = 12

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BeamConfig":
        defaults = cls()
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            minimum = 0 if field_name == "operator_top_k" else 1
            kwargs[field_name] = max(
                minimum,
                int(value.get(field_name, getattr(defaults, field_name))),
            )
        return cls(**kwargs)


@dataclass(slots=True)
class DataConfig:
    decompose_predictions: str
    semantic_train: str
    compose_train: str
    operator_train: str = ""
    use_gold_entities: bool = False
    gold_entities: str = ""
    gold_entities_member: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DataConfig":
        return cls(
            decompose_predictions=str(value.get("decompose_predictions", "")),
            semantic_train=str(value.get("semantic_train", "")),
            compose_train=str(value.get("compose_train", "")),
            operator_train=str(value.get("operator_train", "")),
            use_gold_entities=bool(value.get("use_gold_entities", False)),
            gold_entities=str(value.get("gold_entities", "")),
            gold_entities_member=str(
                value.get(
                    "gold_entities_member",
                    "",
                )
            ),
        )


@dataclass(slots=True)
class AppConfig:
    data: DataConfig
    semantic_model: ChatConfig
    compose_model: ChatConfig
    selector_model: ChatConfig
    embedding: EmbeddingConfig
    freebase: FreebaseConfig
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    failure_endpoint_retrieval: FailureEndpointRetrievalConfig = field(
        default_factory=FailureEndpointRetrievalConfig
    )
    template_consensus: TemplateConsensusConfig = field(
        default_factory=TemplateConsensusConfig
    )
    path_alignment: PathAlignmentConfig = field(
        default_factory=PathAlignmentConfig
    )
    train_gold_path_extrema: TrainGoldPathExtremaConfig = field(
        default_factory=TrainGoldPathExtremaConfig
    )
    numeric_temporal_slot_normalization: NumericTemporalSlotNormalizationConfig = field(
        default_factory=NumericTemporalSlotNormalizationConfig
    )
    entity_kind_fixed_beam: EntityKindFixedBeamConfig = field(
        default_factory=EntityKindFixedBeamConfig
    )
    underanswer_expansion: UnderanswerExpansionConfig = field(
        default_factory=UnderanswerExpansionConfig
    )
    beam: BeamConfig = field(default_factory=BeamConfig)
    operator_model: ChatConfig | None = None
    validator_model: ChatConfig | None = None
    decomposition_review_model: ChatConfig | None = None
    decomposition_review_enabled: bool = False
    decomposition_review_workflow: str = "verified"
    decomposition_prompt_family: str = "webqsp"
    decomposition_prompt_profile: dict[str, str] = field(default_factory=dict)
    decomposition_review_max_rewrites: int = 2
    decomposition_review_output_attempts: int = 2
    decomposition_confirm_before_rewrite: bool = False
    decomposition_preserve_original_on_failure: bool = False
    rewrite_execution_fallback: bool = False
    # Relation-relaxed grounding falls back to stepwise expansion even when
    # the primary strategy is path_sequence.  Keep it configurable because a
    # wide path_beam can make this recovery path substantially more expensive.
    relation_relaxed_repair_enabled: bool = True
    selector_answer_preview: int = 100
    selector_max_prompt_bytes: int = 0
    validation_enabled: bool = False
    validation_on_valid: bool = True
    validation_attempts: int = 1
    validation_stages: tuple[str, ...] = ("semantic", "compose", "operator")
    # The configured dataset may use paths longer than the WebQSP majority.
    semantic_max_hops: int = 2
    # ``path_sequence`` uses one query for a
    # complete long relation sequence); ``stepwise`` expands one edge at a
    # time, like the WebQSP pipeline.  Keeping this as a config switch makes
    # the two implementations directly comparable.
    long_path_strategy: str = "path_sequence"
    # Suppress an immediate URI edge reversal in recursive search.  Literal
    # backtracking is retained because numeric/CVT paths legitimately use
    # it.
    avoid_immediate_backtrack: bool = False
    operator_prompt_mode: str = "training"
    # Maximum number of Compose requests submitted concurrently.  Keep at 1
    # for the 0.3.4 serial behavior; larger values let vLLM batch candidates.
    compose_workers: int = 1
    # Maximum number of Operator requests in flight after each Compose result.
    # A value greater than one overlaps Operator with remaining Compose work.
    operator_workers: int = 1
    # Independent questions can overlap while preserving per-question traces.
    question_workers: int = 1
    # Reject a candidate when Semantic emitted an explicit path that Grounding
    # could not realize. This is a structural invariant, not a question rule.
    require_complete_semantic_graph: bool = False
    require_semantic_item_coverage: bool = False
    retain_original_with_rewrite: bool = False
    allow_semantic_template_recovery: bool = True
    allow_path_prefix_recovery: bool = True
    semantic_projection_mode: str = "legacy"
    relation_relaxed_max_hops: int = 2
    version: str = "0.1.0"

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration must be a JSON object")
        review = raw.get("decomposition_review", {})
        selection = raw.get("selection", {})
        grounding_repair = raw.get("grounding_repair", {})
        failure_endpoint_retrieval = raw.get("failure_endpoint_retrieval", {})
        template_consensus = raw.get("template_consensus", {})
        path_alignment = raw.get("path_alignment", {})
        train_gold_path_extrema = raw.get("train_gold_path_extrema", {})
        numeric_temporal_slot_normalization = raw.get(
            "numeric_temporal_slot_normalization",
            {},
        )
        entity_kind_fixed_beam = raw.get("entity_kind_fixed_beam", {})
        underanswer_expansion = raw.get("underanswer_expansion", {})
        if (
            not isinstance(review, dict)
            or not isinstance(selection, dict)
            or not isinstance(grounding_repair, dict)
            or not isinstance(failure_endpoint_retrieval, dict)
            or not isinstance(template_consensus, dict)
            or not isinstance(path_alignment, dict)
            or not isinstance(train_gold_path_extrema, dict)
            or not isinstance(numeric_temporal_slot_normalization, dict)
            or not isinstance(entity_kind_fixed_beam, dict)
            or not isinstance(underanswer_expansion, dict)
        ):
            raise ValueError(
                "decomposition_review, selection, grounding_repair and "
                "failure_endpoint_retrieval, template_consensus and "
                "path_alignment, train_gold_path_extrema and "
                "numeric_temporal_slot_normalization, entity_kind_fixed_beam "
                "and underanswer_expansion must be JSON objects"
            )
        validation = raw.get("validation", {})
        if not isinstance(validation, dict):
            validation = {}
        raw_validation_stages = validation.get(
            "stages",
            ["semantic", "compose", "operator"],
        )
        if isinstance(raw_validation_stages, str):
            raw_validation_stages = [raw_validation_stages]
        if not isinstance(raw_validation_stages, list):
            raise ValueError("validation.stages must be an array of stage names")
        validation_stages = tuple(
            dict.fromkeys(str(stage).strip().casefold() for stage in raw_validation_stages)
        )
        prompt_profile: dict[str, str] = {}
        prompt_profile_file = str(review.get("prompt_profile_file", "")).strip()
        if prompt_profile_file:
            profile_path = Path(prompt_profile_file).expanduser()
            if not profile_path.is_absolute():
                profile_path = (config_path.parent / profile_path).resolve()
            profile_value = json.loads(profile_path.read_text(encoding="utf-8"))
            if not isinstance(profile_value, dict):
                raise ValueError("decomposition_review.prompt_profile_file must contain a JSON object")
            prompt_profile = {
                str(key): str(value)
                for key, value in profile_value.items()
                if value is not None
            }
            prompt_profile["source"] = str(profile_path)
        data = DataConfig.from_dict(raw.get("data", {}))
        ontology = OntologyConfig.from_dict(raw.get("ontology", {}))
        base = config_path.parent
        if ontology.enabled and ontology.directory:
            ontology_path = Path(ontology.directory).expanduser()
            if not ontology_path.is_absolute():
                ontology_path = (base / ontology_path).resolve()
            ontology.directory = str(ontology_path)
        consensus = TemplateConsensusConfig.from_dict(template_consensus)
        for name in ("source_train", "cache_path"):
            raw_value = str(getattr(consensus, name)).strip()
            if not raw_value:
                continue
            value = Path(raw_value).expanduser()
            if not value.is_absolute():
                value = (base / value).resolve()
            setattr(consensus, name, str(value))
        extrema_path = TrainGoldPathExtremaConfig.from_dict(
            train_gold_path_extrema
        )
        if extrema_path.cache_path:
            value = Path(extrema_path.cache_path).expanduser()
            if not value.is_absolute():
                value = (base / value).resolve()
            extrema_path.cache_path = str(value)
        failure_retrieval = FailureEndpointRetrievalConfig.from_dict(
            failure_endpoint_retrieval
        )
        if failure_retrieval.source_gold_sparql_cache_path:
            cache_path = Path(
                failure_retrieval.source_gold_sparql_cache_path
            ).expanduser()
            if not cache_path.is_absolute():
                cache_path = (base / cache_path).resolve()
            failure_retrieval.source_gold_sparql_cache_path = str(cache_path)
        if failure_retrieval.factorized_path_cache_path:
            cache_path = Path(
                failure_retrieval.factorized_path_cache_path
            ).expanduser()
            if not cache_path.is_absolute():
                cache_path = (base / cache_path).resolve()
            failure_retrieval.factorized_path_cache_path = str(cache_path)
        entity_kind = EntityKindFixedBeamConfig.from_dict(
            entity_kind_fixed_beam
        )
        if entity_kind.cache_path:
            cache_path = Path(entity_kind.cache_path).expanduser()
            if not cache_path.is_absolute():
                cache_path = (base / cache_path).resolve()
            entity_kind.cache_path = str(cache_path)
        for name in (
            "decompose_predictions",
            "semantic_train",
            "compose_train",
            "operator_train",
            "gold_entities",
        ):
            raw_value = str(getattr(data, name)).strip()
            # Empty optional paths (notably ``operator_train`` for the
            # no-operator bundle) must remain empty.  ``Path("")``
            # resolves to the config directory, which previously made an
            # omitted operator dataset fail validation as "directory is not a
            # file".
            if not raw_value:
                setattr(data, name, "")
                continue
            value = Path(raw_value).expanduser()
            if not value.is_absolute():
                value = (base / value).resolve()
            setattr(data, name, str(value))
        return cls(
            data=data,
            semantic_model=ChatConfig.from_dict(raw.get("semantic_model", {})),
            compose_model=ChatConfig.from_dict(raw.get("compose_model", {})),
            selector_model=ChatConfig.from_dict(raw.get("selector_model", {})),
            embedding=EmbeddingConfig.from_dict(raw.get("embedding", {})),
            freebase=FreebaseConfig.from_dict(raw.get("freebase", {})),
            ontology=ontology,
            failure_endpoint_retrieval=failure_retrieval,
            template_consensus=consensus,
            path_alignment=PathAlignmentConfig.from_dict(path_alignment),
            train_gold_path_extrema=extrema_path,
            numeric_temporal_slot_normalization=(
                NumericTemporalSlotNormalizationConfig.from_dict(
                    numeric_temporal_slot_normalization
                )
            ),
            entity_kind_fixed_beam=entity_kind,
            underanswer_expansion=UnderanswerExpansionConfig.from_dict(
                underanswer_expansion
            ),
            beam=BeamConfig.from_dict(raw.get("beam", {})),
            operator_model=(
                ChatConfig.from_dict(raw["operator_model"])
                if isinstance(raw.get("operator_model"), dict)
                else None
            ),
            validator_model=(
                ChatConfig.from_dict(raw["validator_model"])
                if isinstance(raw.get("validator_model"), dict)
                else None
            ),
            decomposition_review_model=(
                ChatConfig.from_dict(raw["decomposition_review_model"])
                if isinstance(raw.get("decomposition_review_model"), dict)
                else None
            ),
            decomposition_review_enabled=_bool_value(review.get("enabled", False)),
            decomposition_review_workflow=str(review.get("workflow", "verified")),
            decomposition_prompt_family=str(review.get("prompt_family", "webqsp")).casefold(),
            decomposition_prompt_profile=prompt_profile,
            decomposition_review_max_rewrites=int(review.get("max_rewrites", 2)),
            decomposition_review_output_attempts=int(review.get("output_attempts", 2)),
            decomposition_confirm_before_rewrite=_bool_value(review.get("confirm_before_rewrite", False)),
            decomposition_preserve_original_on_failure=_bool_value(review.get("preserve_original_on_failure", False)),
            rewrite_execution_fallback=_bool_value(review.get("execution_fallback", False)),
            relation_relaxed_repair_enabled=_bool_value(
                grounding_repair.get("relation_relaxed_enabled", True)
            ),
            selector_answer_preview=int(selection.get("answer_preview", 100)),
            selector_max_prompt_bytes=int(selection.get("max_prompt_bytes", 0)),
            validation_enabled=bool(validation.get("enabled", False)),
            validation_on_valid=bool(validation.get("on_valid", True)),
            validation_attempts=max(
                0,
                int(validation.get("attempts", 1)),
            ),
            validation_stages=validation_stages,
            semantic_max_hops=max(1, int(raw.get("semantic_max_hops", 2))),
            long_path_strategy=str(
                raw.get("long_path_strategy", "path_sequence")
            ).casefold(),
            avoid_immediate_backtrack=_bool_value(
                raw.get("avoid_immediate_backtrack", False),
                default=False,
            ),
            operator_prompt_mode=str(
                raw.get("operator_prompt_mode", "training")
            ).casefold(),
            compose_workers=max(1, int(raw.get("compose_workers", 1))),
            operator_workers=max(1, int(raw.get("operator_workers", 1))),
            question_workers=max(1, int(raw.get("question_workers", 1))),
            require_complete_semantic_graph=_bool_value(
                raw.get("require_complete_semantic_graph", False),
                default=False,
            ),
            require_semantic_item_coverage=_bool_value(
                raw.get("require_semantic_item_coverage", False),
                default=False,
            ),
            retain_original_with_rewrite=_bool_value(
                raw.get("retain_original_with_rewrite", False),
                default=False,
            ),
            allow_semantic_template_recovery=_bool_value(
                raw.get("allow_semantic_template_recovery", True),
                default=True,
            ),
            allow_path_prefix_recovery=_bool_value(
                raw.get("allow_path_prefix_recovery", True),
                default=True,
            ),
            semantic_projection_mode=str(
                raw.get("semantic_projection_mode", "legacy")
            ).strip().casefold(),
            relation_relaxed_max_hops=max(
                1,
                int(raw.get("relation_relaxed_max_hops", 2)),
            ),
            version=str(raw.get("version", "0.1.0")),
        )

    def validate(self) -> None:
        if self.decomposition_review_workflow not in {"verified", "two_call"}:
            raise ValueError("decomposition_review.workflow must be verified or two_call")
        if self.decomposition_prompt_family not in {"webqsp", "cwq"}:
            raise ValueError("decomposition_review.prompt_family must be webqsp or cwq")
        allowed_prompt_profile_keys = {
            "version",
            "source",
            "review_append",
            "rewrite_append",
            "canonicalize_context_names",
            "merge_anchorless_rewrite_items",
        }
        unknown_prompt_profile_keys = (
            set(self.decomposition_prompt_profile) - allowed_prompt_profile_keys
        )
        if unknown_prompt_profile_keys:
            raise ValueError(
                "decomposition prompt profile contains unsupported keys: "
                + ", ".join(sorted(unknown_prompt_profile_keys))
            )
        if self.decomposition_review_enabled and self.decomposition_review_workflow == "two_call":
            if (self.decomposition_review_max_rewrites != 1 or self.decomposition_review_output_attempts != 1
                    or self.decomposition_confirm_before_rewrite):
                raise ValueError("two_call requires max_rewrites=1, output_attempts=1, confirm_before_rewrite=false")
            if self.decomposition_review_model is not None and self.decomposition_review_model.retries:
                raise ValueError("two_call requires decomposition_review_model.retries=0")
        if self.decomposition_review_max_rewrites < 0:
            raise ValueError("decomposition_review.max_rewrites must be nonnegative")
        if self.decomposition_review_output_attempts < 1:
            raise ValueError("decomposition_review.output_attempts must be at least 1")
        if self.selector_answer_preview < 1 or self.selector_max_prompt_bytes < 0:
            raise ValueError("selection.answer_preview must be positive and max_prompt_bytes nonnegative")
        if self.decomposition_review_enabled and self.decomposition_review_model is None:
            raise ValueError("decomposition_review_model is required when decomposition review is enabled")
        if self.semantic_max_hops < 1:
            raise ValueError("semantic_max_hops must be at least 1")
        if self.long_path_strategy not in {"path_sequence", "stepwise"}:
            raise ValueError(
                "long_path_strategy must be path_sequence or stepwise"
            )
        if self.semantic_projection_mode not in {"legacy", "full"}:
            raise ValueError("semantic_projection_mode must be legacy or full")
        if self.ontology.enabled:
            ontology_dir = Path(self.ontology.directory)
            if not ontology_dir.is_dir():
                raise ValueError(
                    f"ontology directory does not exist: {self.ontology.directory}"
                )
            if not (ontology_dir / "fb_roles").is_file():
                raise ValueError(
                    f"ontology fb_roles does not exist: {ontology_dir / 'fb_roles'}"
                )
        for field_name in (
            "decompose_predictions",
            "semantic_train",
            "compose_train",
        ):
            path = str(getattr(self.data, field_name))
            if not _data_source_exists(path, field_name):
                raise ValueError(f"data file or dataset directory does not exist: {path}")
        if self.data.operator_train and not _data_source_exists(
            self.data.operator_train,
            "operator_train",
        ):
            raise ValueError(
                "data file or dataset directory does not exist: "
                f"{self.data.operator_train}"
            )
        if self.template_consensus.enabled:
            if not self.data.operator_train:
                raise ValueError(
                    "data.operator_train is required when template_consensus is enabled"
                )
            if not self.template_consensus.source_train:
                raise ValueError(
                    "template_consensus.source_train is required when enabled"
                )
            if not Path(self.template_consensus.source_train).is_file():
                raise ValueError(
                    "template consensus source train does not exist: "
                    f"{self.template_consensus.source_train}"
                )
            if not self.template_consensus.cache_path:
                raise ValueError(
                    "template_consensus.cache_path is required when enabled"
                )
        if (
            self.failure_endpoint_retrieval.source_gold_sparql_enabled
            or self.failure_endpoint_retrieval.temporal_interval_enabled
        ):
            cache_path = self.failure_endpoint_retrieval.source_gold_sparql_cache_path
            if not cache_path:
                raise ValueError(
                    "failure_endpoint_retrieval.source_gold_sparql_cache_path "
                    "is required when source_gold_sparql_enabled"
                )
            if not Path(cache_path).is_file():
                raise ValueError(
                    "source Gold SPARQL cache does not exist: "
                    f"{cache_path}"
                )
        if self.path_alignment.enabled and not self.ontology.enabled:
            raise ValueError(
                "ontology must be enabled when path_alignment is enabled"
            )
        if (
            self.numeric_temporal_slot_normalization.enabled
            and not self.ontology.enabled
        ):
            raise ValueError(
                "ontology must be enabled when "
                "numeric_temporal_slot_normalization is enabled"
            )
        if self.entity_kind_fixed_beam.enabled:
            if not self.ontology.enabled:
                raise ValueError(
                    "ontology must be enabled when entity_kind_fixed_beam is enabled"
                )
            if not self.entity_kind_fixed_beam.cache_path:
                raise ValueError(
                    "entity_kind_fixed_beam.cache_path is required when enabled"
                )
            if not Path(self.entity_kind_fixed_beam.cache_path).is_file():
                raise ValueError(
                    "entity-kind TRAIN template cache does not exist: "
                    f"{self.entity_kind_fixed_beam.cache_path}"
                )
        if self.underanswer_expansion.enabled and not self.ontology.enabled:
            raise ValueError(
                "ontology must be enabled when underanswer_expansion is enabled"
            )
        if self.failure_endpoint_retrieval.factorized_path_enabled:
            cache_path = self.failure_endpoint_retrieval.factorized_path_cache_path
            if not self.ontology.enabled:
                raise ValueError(
                    "ontology must be enabled when failure factorized path is enabled"
                )
            if not cache_path:
                raise ValueError(
                    "failure_endpoint_retrieval.factorized_path_cache_path "
                    "is required when factorized_path_enabled"
                )
            if not Path(cache_path).is_file():
                raise ValueError(
                    "factorized TRAIN path cache does not exist: "
                    f"{cache_path}"
                )
        if self.train_gold_path_extrema.enabled:
            if not self.ontology.enabled:
                raise ValueError(
                    "ontology must be enabled when train_gold_path_extrema is enabled"
                )
            if not self.train_gold_path_extrema.cache_path:
                raise ValueError(
                    "train_gold_path_extrema.cache_path is required when enabled"
                )
            if not Path(self.train_gold_path_extrema.cache_path).is_file():
                raise ValueError(
                    "TRAIN Gold path cache does not exist: "
                    f"{self.train_gold_path_extrema.cache_path}"
                )
        if self.data.use_gold_entities and not self.data.gold_entities:
            raise ValueError("data.gold_entities is required when data.use_gold_entities is true")
        if self.data.use_gold_entities and not Path(self.data.gold_entities).is_file():
            raise ValueError(f"data.gold_entities does not exist: {self.data.gold_entities}")
        required_models = [
            ("semantic_model", self.semantic_model),
            ("compose_model", self.compose_model),
            ("selector_model", self.selector_model),
        ]
        if self.decomposition_review_enabled:
            required_models.append(("decomposition_review_model", self.decomposition_review_model))
        for name, model in required_models:
            if not model.base_url or not model.model:
                raise ValueError(f"{name}.base_url and {name}.model are required")
            if model.api not in {"chat_completions", "chat", "responses", "response"}:
                raise ValueError(f"{name}.api must be chat_completions or responses")
            if model.prompt_mode not in {"system_user", "alpaca"}:
                raise ValueError(f"{name}.prompt_mode must be system_user or alpaca")
            if model.schema_mode not in {"guided_json", "json_schema", "structured_outputs", "prompt"}:
                raise ValueError(f"{name}.schema_mode is unsupported")
        if self.operator_model is not None:
            if not self.operator_model.base_url or not self.operator_model.model:
                raise ValueError("operator_model.base_url and operator_model.model are required")
            if self.operator_model.api not in {"chat_completions", "chat", "responses", "response"}:
                raise ValueError("operator_model.api must be chat_completions or responses")
            if self.operator_model.prompt_mode not in {"system_user", "alpaca"}:
                raise ValueError("operator_model.prompt_mode must be system_user or alpaca")
        if self.operator_prompt_mode not in {
            "training",
            "glm_zero_shot",
            "glm_few_shot",
        }:
            raise ValueError(
                "operator_prompt_mode must be training, glm_zero_shot, or glm_few_shot"
            )
        if self.validation_enabled:
            allowed_validation_stages = {"semantic", "compose", "operator"}
            if not self.validation_stages:
                raise ValueError("validation.stages must not be empty when validation is enabled")
            unknown_validation_stages = set(self.validation_stages) - allowed_validation_stages
            if unknown_validation_stages:
                raise ValueError(
                    "validation.stages contains unsupported stages: "
                    + ", ".join(sorted(unknown_validation_stages))
                )
            if self.validator_model is None:
                raise ValueError("validator_model is required when validation.enabled is true")
            if not self.validator_model.base_url or not self.validator_model.model:
                raise ValueError("validator_model.base_url and validator_model.model are required")
            if self.validator_model.api not in {"chat_completions", "chat", "responses", "response"}:
                raise ValueError("validator_model.api must be chat_completions or responses")
            if self.validator_model.prompt_mode not in {"system_user", "alpaca"}:
                raise ValueError("validator_model.prompt_mode must be system_user or alpaca")
        if not self.embedding.url or not self.embedding.model:
            raise ValueError("embedding.url and embedding.model are required")
        if not self.freebase.endpoint:
            raise ValueError("freebase.endpoint is required")
