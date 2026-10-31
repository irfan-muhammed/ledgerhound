"""
sentinel.pipeline
=================

The fraud-model data pipeline (v4 Stage 1): construct D, then certify how
it was constructed. See ``docs/v4.0/next_steps_v4.md``.
"""

from sentinel.pipeline.aggregate import (
    CHUNK_COMPOSABLE,
    MODE_CHUNKED,
    MODE_PUSHDOWN,
    NEEDS_FULL_GROUP,
    AggregationPlan,
    AggSpec,
    NotChunkComposable,
    aggregate_chunked,
    build_aggregation_sql,
    plan_for_grain,
)
from sentinel.pipeline.appropriateness import (
    AppropriatenessReport,
    Finding,
    check_sample,
)
from sentinel.pipeline.certificate import (
    DEFAULT_REQUIRED,
    CertificateBuilder,
    CertificateSection,
    ConstructionCertificate,
    UncertifiedSampleError,
)
from sentinel.pipeline.cluster_psi import (
    ClusterAwarePSI,
    ClusterDiagnostics,
    ClusterPSIError,
    ICCEstimate,
    assess_psi,
    cluster_diagnostics,
    design_effect,
    entity_level_frame,
    estimate_icc,
    psi_critical_value,
)
from sentinel.pipeline.history_fetch import (
    EntityKeySet,
    HistoryFetchConfig,
    HistoryFetchPlan,
    UnsafeSQLValue,
    build_history_sql,
    extract_entity_keys,
    plan_from_story,
    sql_literal,
)
from sentinel.pipeline.labeller import (
    LABEL_COLUMN,
    LabellingConfig,
    LabellingError,
    LabellingResult,
    apply_labels,
)
from sentinel.pipeline.model_story import (
    GRAIN_ACCOUNT,
    GRAIN_TRANSACTION,
    SCHEMA,
    LabelRule,
    ModelStory,
)
from sentinel.pipeline.prevalence import (
    PrevalenceCheck,
    PrevalenceError,
    PrevalenceMeasurement,
    alert_volume,
    build_population_correction,
    check_prevalence,
    corrected_precision,
    measure_prevalence,
    precision_shift,
    prior_correction_offset,
)
from sentinel.pipeline.pu import (
    NEGATIVE_NAME,
    POSITIVE_NAME,
    UNLABELLED_NAME,
    ContaminationScenario,
    PUError,
    SpikeInResult,
    assert_honest_naming,
    contamination_scenarios,
    correct_for_contamination,
    honest_class_name,
    spike_in_sensitivity,
)
from sentinel.pipeline.splits import (
    MODE_ENTITY,
    MODE_GROUPED_TEMPORAL,
    MODE_TEMPORAL,
    SplitError,
    SplitResult,
    SplitSpec,
    make_split,
    verify_split,
)
from sentinel.pipeline.typology import (
    SAFEST_SCOPE,
    TypologyCoverage,
    TypologyError,
    TypologyMap,
    TypologyReconciliation,
    default_rule,
    reconcile_typologies,
)

__all__ = [
    "ModelStory", "LabelRule", "SCHEMA",
    "GRAIN_ACCOUNT", "GRAIN_TRANSACTION",
    "EntityKeySet", "HistoryFetchConfig", "HistoryFetchPlan", "UnsafeSQLValue",
    "build_history_sql", "extract_entity_keys", "plan_from_story", "sql_literal",
    "AggSpec", "AggregationPlan", "NotChunkComposable",
    "aggregate_chunked", "build_aggregation_sql", "plan_for_grain",
    "CHUNK_COMPOSABLE", "NEEDS_FULL_GROUP", "MODE_CHUNKED", "MODE_PUSHDOWN",
    "apply_labels", "LabellingConfig", "LabellingResult", "LabellingError",
    "LABEL_COLUMN",
    "measure_prevalence", "check_prevalence", "PrevalenceMeasurement",
    "PrevalenceCheck", "PrevalenceError", "corrected_precision",
    "precision_shift", "alert_volume", "prior_correction_offset",
    "build_population_correction",
    "check_sample", "AppropriatenessReport", "Finding",
    "make_split", "verify_split", "SplitSpec", "SplitResult", "SplitError",
    "MODE_TEMPORAL", "MODE_ENTITY", "MODE_GROUPED_TEMPORAL",
    "assess_psi", "estimate_icc", "design_effect", "cluster_diagnostics",
    "psi_critical_value", "entity_level_frame", "ClusterAwarePSI",
    "ClusterDiagnostics", "ICCEstimate", "ClusterPSIError",
    "honest_class_name", "assert_honest_naming", "contamination_scenarios",
    "correct_for_contamination", "spike_in_sensitivity",
    "ContaminationScenario", "SpikeInResult", "PUError",
    "POSITIVE_NAME", "NEGATIVE_NAME", "UNLABELLED_NAME",
    "reconcile_typologies", "TypologyMap", "TypologyReconciliation",
    "TypologyCoverage", "TypologyError", "default_rule", "SAFEST_SCOPE",
    "ConstructionCertificate", "CertificateBuilder", "CertificateSection",
    "UncertifiedSampleError", "DEFAULT_REQUIRED",
]
