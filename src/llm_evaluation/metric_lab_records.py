"""Immutable evidence records for the known-truth metric laboratory."""

from __future__ import annotations

import json
import math
import platform
import re
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from canonical_sync.codec import canonical_record_hash
from llm_evaluation.artifact_io import publish_immutable, reject_symlink_components
from llm_evaluation.metric_lab_design import (
    FINAL_REPLICATES_PER_CELL,
    FrozenMetricLabDesignRecord,
    MetricLabDesign,
    MetricLabPhase,
)
from llm_evaluation.metric_lab_runner import (
    AmbiguityCellSummary,
    BaseCellSummary,
    MetricMonteCarloSummary,
    run_auxiliary_deterministic_controls,
)
from llm_evaluation.metric_lab_targets import SamplingUnit, TargetKind
from llm_evaluation.records import software_identity

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class _StrictFrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def _exact_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an exact integer")
        return value


def _finite(value: float | None, *, label: str) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


class MetricSummaryRecord(_StrictFrozenRecord):
    record_type: Literal["metric_lab_metric_summary"] = "metric_lab_metric_summary"
    schema_version: Literal[1] = 1
    true_value: float
    mean_estimate: float
    bias: float
    rmse: float
    mc_se: float
    interval_coverage: float | None = None
    mean_interval_width: float | None = None

    @field_validator(
        "true_value",
        "mean_estimate",
        "bias",
        "rmse",
        "mc_se",
        "interval_coverage",
        "mean_interval_width",
    )
    @classmethod
    def _finite_metrics(cls, value: float | None) -> float | None:
        return _finite(value, label="metric summary value")

    @model_validator(mode="after")
    def _validate_nonnegative_fields(self) -> Self:
        if self.rmse < 0.0 or self.mc_se < 0.0:
            raise ValueError("rmse and mc_se must be non-negative")
        if (
            self.interval_coverage is not None
            and not 0.0 <= self.interval_coverage <= 1.0
        ):
            raise ValueError("interval_coverage must lie in [0, 1]")
        if self.mean_interval_width is not None and self.mean_interval_width < 0.0:
            raise ValueError("mean_interval_width must be non-negative")
        return self


class TargetSummaryRecord(_StrictFrozenRecord):
    record_type: Literal["metric_lab_target_summary"] = "metric_lab_target_summary"
    schema_version: Literal[1] = 1
    target: TargetKind
    expected_brier: float
    expected_log_score: float
    fixed_bin_calibration: float
    equal_mass_calibration: float
    l2_calibration: float
    classification_error: float
    aurc: float
    metrics: dict[str, MetricSummaryRecord]

    @field_validator(
        "expected_brier",
        "expected_log_score",
        "fixed_bin_calibration",
        "equal_mass_calibration",
        "l2_calibration",
        "classification_error",
        "aurc",
    )
    @classmethod
    def _finite_truth(cls, value: float) -> float:
        result = _finite(value, label="truth value")
        assert result is not None
        return result

    @field_validator("metrics")
    @classmethod
    def _nonempty_metrics(
        cls, value: dict[str, MetricSummaryRecord]
    ) -> dict[str, MetricSummaryRecord]:
        if not value or any(not key or key != key.strip() for key in value):
            raise ValueError("metrics must be a non-empty mapping with trimmed names")
        return value


class CellSummaryRecord(_StrictFrozenRecord):
    record_type: Literal["metric_lab_cell_summary"] = "metric_lab_cell_summary"
    schema_version: Literal[2] = 2
    cell_kind: Literal["base", "ambiguity"]
    cell_key: str
    sampling_unit: SamplingUnit | None = None
    replicates_count: int
    budget_unit: Literal["rows", "ratings"]
    realized_budget: int
    targets: tuple[TargetSummaryRecord, ...]

    @field_validator("cell_key")
    @classmethod
    def _cell_key(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("cell_key must be non-empty and trimmed")
        return value

    @field_validator("replicates_count", "realized_budget")
    @classmethod
    def _positive_counts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("cell counts must be positive")
        return value

    @field_validator("targets")
    @classmethod
    def _targets(
        cls, value: tuple[TargetSummaryRecord, ...]
    ) -> tuple[TargetSummaryRecord, ...]:
        if not value:
            raise ValueError("cell must contain target summaries")
        target_names = tuple(item.target for item in value)
        if len(target_names) != len(set(target_names)):
            raise ValueError("cell target summaries must be unique")
        return tuple(sorted(value, key=lambda item: item.target.value))

    @model_validator(mode="after")
    def _cell_semantics(self) -> Self:
        if self.cell_kind == "base":
            if self.sampling_unit is None or self.budget_unit != "rows":
                raise ValueError("base cells require a sampling_unit and row budget")
        elif self.sampling_unit is not None or self.budget_unit != "ratings":
            raise ValueError(
                "ambiguity cells require no sampling_unit and a rating budget"
            )
        return self


class PrevalenceControlRecord(_StrictFrozenRecord):
    record_type: Literal["metric_lab_prevalence_control"] = (
        "metric_lab_prevalence_control"
    )
    schema_version: Literal[1] = 1
    cell_key: str
    prevalence: float
    tpr: float
    fpr: float
    precision: float

    @field_validator("prevalence", "tpr", "fpr", "precision")
    @classmethod
    def _finite_values(cls, value: float) -> float:
        result = _finite(value, label="prevalence control value")
        assert result is not None
        return result


class AuxiliaryControlsRecord(_StrictFrozenRecord):
    record_type: Literal["metric_lab_auxiliary_controls"] = (
        "metric_lab_auxiliary_controls"
    )
    schema_version: Literal[2] = 2
    prevalence: tuple[PrevalenceControlRecord, ...]
    subgroup_pooled_ece: float
    subgroup_ece: dict[str, float]
    subgroup_max_ece: float
    aurc_upstream_negative_control: float
    aurc_prefix_oracle: float
    aurc_defect_detected: bool
    positive_class_correct_brier: float
    positive_class_inverted_brier: float
    positive_class_inversion_detected: bool
    label_noise_observed_reference_probability: float
    label_noise_candidate_clean_probabilities: tuple[float, float]
    label_noise_candidate_flip_rates: tuple[float, float]
    label_noise_nonidentifiable: bool

    @field_validator(
        "subgroup_pooled_ece",
        "subgroup_max_ece",
        "aurc_upstream_negative_control",
        "aurc_prefix_oracle",
        "positive_class_correct_brier",
        "positive_class_inverted_brier",
        "label_noise_observed_reference_probability",
    )
    @classmethod
    def _finite_controls(cls, value: float) -> float:
        result = _finite(value, label="auxiliary control value")
        assert result is not None
        return result

    @field_validator(
        "label_noise_candidate_clean_probabilities",
        "label_noise_candidate_flip_rates",
    )
    @classmethod
    def _finite_probability_pairs(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        for item in value:
            if _finite(item, label="label-noise control probability") is None:
                raise ValueError("invalid label-noise control probability")
            if not 0.0 <= item <= 1.0:
                raise ValueError("label-noise control probabilities must lie in [0, 1]")
        return value

    @field_validator("subgroup_ece")
    @classmethod
    def _subgroup_values(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("subgroup_ece must be non-empty")
        for key, metric in value.items():
            if not key or _finite(metric, label="subgroup ece") is None:
                raise ValueError("invalid subgroup ece")
        return value


class MetricLabRunRecord(_StrictFrozenRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.metric-lab-run"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["known_truth_metric_lab_run"] = "known_truth_metric_lab_run"
    schema_version: Literal[3] = 3
    phase: MetricLabPhase
    design_sha256: str
    frozen_design_record_sha256: str | None = None
    replicates_per_cell: int
    cell_count: int
    realized_budget_by_unit: dict[str, int]
    cells: tuple[CellSummaryRecord, ...]
    auxiliary_controls: AuxiliaryControlsRecord
    software_commit: str | None = None
    package_source_sha256: str | None = None
    python_version: str

    @field_validator("design_sha256", "frozen_design_record_sha256")
    @classmethod
    def _hashes(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_RE.fullmatch(value) is None:
            raise ValueError("design identities must be lowercase SHA-256 digests")
        return value

    @field_validator("software_commit")
    @classmethod
    def _commit(cls, value: str | None) -> str | None:
        if value is not None and _GIT_OID_RE.fullmatch(value) is None:
            raise ValueError("software_commit must be a lowercase Git object ID")
        return value

    @field_validator("package_source_sha256")
    @classmethod
    def _package_hash(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_RE.fullmatch(value) is None:
            raise ValueError("package_source_sha256 must be a lowercase SHA-256 digest")
        return value

    @field_validator("python_version")
    @classmethod
    def _python_version(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("python_version must be non-empty and trimmed")
        return value

    @field_validator("replicates_per_cell", "cell_count")
    @classmethod
    def _positive_run_counts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("run counts must be positive")
        return value

    @field_validator("realized_budget_by_unit")
    @classmethod
    def _budget_map(cls, value: dict[str, int]) -> dict[str, int]:
        if not value or any(key not in {"rows", "ratings"} for key in value):
            raise ValueError("realized budget units must be rows and/or ratings")
        if any(type(count) is not int or count < 1 for count in value.values()):
            raise ValueError("realized budgets must be positive exact integers")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def _validate_run(self) -> Self:
        if len(self.cells) != self.cell_count:
            raise ValueError("cell_count must equal the exact cell inventory")
        identities = {
            (cell.cell_kind, cell.cell_key, cell.sampling_unit) for cell in self.cells
        }
        if len(identities) != len(self.cells):
            raise ValueError("run cells must have unique kind/key/sampling identities")
        if any(
            cell.replicates_count != self.replicates_per_cell for cell in self.cells
        ):
            raise ValueError("all cells must use the declared replicates_per_cell")
        budget: dict[str, int] = defaultdict_int()
        for cell in self.cells:
            budget[cell.budget_unit] = (
                budget.get(cell.budget_unit, 0) + cell.realized_budget
            )
        if dict(sorted(budget.items())) != self.realized_budget_by_unit:
            raise ValueError(
                "realized_budget_by_unit must equal the exact cell budget sums"
            )
        if (self.software_commit is None) != (self.package_source_sha256 is None):
            raise ValueError(
                "software commit and package hash must be present together"
            )
        if self.phase is MetricLabPhase.FINAL:
            if self.frozen_design_record_sha256 is None:
                raise ValueError("final run requires a frozen design record identity")
            if self.replicates_per_cell != FINAL_REPLICATES_PER_CELL:
                raise ValueError("final run requires exactly 2000 replicates per cell")
        elif self.frozen_design_record_sha256 is not None:
            raise ValueError("development run must not claim a frozen final design")
        return self


def defaultdict_int() -> dict[str, int]:
    """Small typed helper used by strict record validators/builders."""
    return {}


def _metric_summary_record(summary: MetricMonteCarloSummary) -> MetricSummaryRecord:
    return MetricSummaryRecord(
        true_value=summary.true_value,
        mean_estimate=summary.mean_estimate,
        bias=summary.bias,
        rmse=summary.rmse,
        mc_se=summary.mc_se,
        interval_coverage=summary.interval_coverage,
        mean_interval_width=summary.mean_interval_width,
    )


def _base_target_records(summary: BaseCellSummary) -> tuple[TargetSummaryRecord, ...]:
    return tuple(
        TargetSummaryRecord(
            target=target,
            expected_brier=target_summary.truth.expected_brier,
            expected_log_score=target_summary.truth.expected_log_score,
            fixed_bin_calibration=target_summary.truth.fixed_bin_calibration,
            equal_mass_calibration=target_summary.truth.equal_mass_calibration,
            l2_calibration=target_summary.truth.l2_calibration,
            classification_error=target_summary.truth.classification_error,
            aurc=target_summary.truth.aurc,
            metrics={
                name: _metric_summary_record(metric)
                for name, metric in sorted(target_summary.metrics.items())
            },
        )
        for target, target_summary in sorted(
            summary.targets.items(), key=lambda item: item[0].value
        )
    )


def _ambiguity_target_records(
    summary: AmbiguityCellSummary,
) -> tuple[TargetSummaryRecord, ...]:
    return tuple(
        TargetSummaryRecord(
            target=target,
            expected_brier=target_summary.truth.expected_brier,
            expected_log_score=target_summary.truth.expected_log_score,
            fixed_bin_calibration=target_summary.truth.fixed_bin_calibration,
            equal_mass_calibration=target_summary.truth.equal_mass_calibration,
            l2_calibration=target_summary.truth.l2_calibration,
            classification_error=target_summary.truth.classification_error,
            aurc=target_summary.truth.aurc,
            metrics={
                name: _metric_summary_record(metric)
                for name, metric in sorted(target_summary.metrics.items())
            },
        )
        for target, target_summary in sorted(
            summary.targets.items(), key=lambda item: item[0].value
        )
    )


def _cell_record(summary: BaseCellSummary | AmbiguityCellSummary) -> CellSummaryRecord:
    if isinstance(summary, BaseCellSummary):
        return CellSummaryRecord(
            cell_kind="base",
            cell_key=summary.cell_key,
            sampling_unit=summary.sampling_unit,
            replicates_count=summary.replicates_count,
            budget_unit="rows",
            realized_budget=summary.realized_budget_rows,
            targets=_base_target_records(summary),
        )
    if isinstance(summary, AmbiguityCellSummary):
        return CellSummaryRecord(
            cell_kind="ambiguity",
            cell_key=summary.cell_key,
            sampling_unit=None,
            replicates_count=summary.replicates_count,
            budget_unit="ratings",
            realized_budget=summary.realized_budget_ratings,
            targets=_ambiguity_target_records(summary),
        )
    raise TypeError("unsupported metric-lab cell summary type")


def _auxiliary_record(design: MetricLabDesign) -> AuxiliaryControlsRecord:
    controls = run_auxiliary_deterministic_controls(design)
    return AuxiliaryControlsRecord(
        prevalence=tuple(
            PrevalenceControlRecord(
                cell_key=item.cell_key,
                prevalence=item.prevalence,
                tpr=item.tpr,
                fpr=item.fpr,
                precision=item.precision,
            )
            for item in controls.prevalence
        ),
        subgroup_pooled_ece=controls.subgroup.pooled_ece,
        subgroup_ece=controls.subgroup.subgroup_ece,
        subgroup_max_ece=controls.subgroup.max_subgroup_ece,
        aurc_upstream_negative_control=controls.aurc_upstream_negative_control,
        aurc_prefix_oracle=controls.aurc_prefix_oracle,
        aurc_defect_detected=controls.aurc_defect_detected,
        positive_class_correct_brier=controls.positive_class_inversion.correct_brier,
        positive_class_inverted_brier=controls.positive_class_inversion.inverted_brier,
        positive_class_inversion_detected=controls.positive_class_inversion.inversion_detected,
        label_noise_observed_reference_probability=(
            controls.label_noise_nonidentifiability.observed_reference_probability
        ),
        label_noise_candidate_clean_probabilities=(
            controls.label_noise_nonidentifiability.candidate_clean_probabilities
        ),
        label_noise_candidate_flip_rates=(
            controls.label_noise_nonidentifiability.candidate_flip_rates
        ),
        label_noise_nonidentifiable=(
            controls.label_noise_nonidentifiability.distinct_parameterizations_match_observed
        ),
    )


def validate_final_execution_gate(
    *,
    design: MetricLabDesign,
    frozen_design: FrozenMetricLabDesignRecord,
    replicates_per_cell: int,
) -> None:
    if not isinstance(design, MetricLabDesign):
        raise TypeError("design must be a MetricLabDesign")
    if not isinstance(frozen_design, FrozenMetricLabDesignRecord):
        raise TypeError("frozen_design must be a FrozenMetricLabDesignRecord")
    if replicates_per_cell != FINAL_REPLICATES_PER_CELL:
        raise ValueError("final execution requires exactly 2000 replicates per cell")
    if (
        frozen_design.design != design
        or frozen_design.design_sha256 != canonical_record_hash(design)
    ):
        raise ValueError("final execution design differs from the exact frozen design")


def preview_metric_lab_run_record(
    *,
    design: MetricLabDesign,
    phase: MetricLabPhase,
    cell_summaries: tuple[BaseCellSummary | AmbiguityCellSummary, ...],
    frozen_design: FrozenMetricLabDesignRecord | None = None,
) -> MetricLabRunRecord:
    if not cell_summaries:
        raise ValueError("cell_summaries must be non-empty")
    design_hash = canonical_record_hash(design)
    if any(summary.design_sha256 != design_hash for summary in cell_summaries):
        raise ValueError("cell summary design identity differs from requested design")
    if any(summary.phase is not phase for summary in cell_summaries):
        raise ValueError("cell summary phase differs from requested run phase")
    replicate_counts = {summary.replicates_count for summary in cell_summaries}
    if len(replicate_counts) != 1:
        raise ValueError("all cell summaries must use the same replicate count")
    replicates_per_cell = next(iter(replicate_counts))

    frozen_hash: str | None = None
    if phase is MetricLabPhase.FINAL:
        if frozen_design is None:
            raise ValueError("final preview requires the exact frozen design")
        validate_final_execution_gate(
            design=design,
            frozen_design=frozen_design,
            replicates_per_cell=replicates_per_cell,
        )
        frozen_hash = canonical_record_hash(frozen_design)
    elif frozen_design is not None:
        raise ValueError("development preview must not supply a frozen final design")

    cells = tuple(_cell_record(summary) for summary in cell_summaries)
    budget: dict[str, int] = defaultdict_int()
    for cell in cells:
        budget[cell.budget_unit] = (
            budget.get(cell.budget_unit, 0) + cell.realized_budget
        )
    return MetricLabRunRecord(
        phase=phase,
        design_sha256=design_hash,
        frozen_design_record_sha256=frozen_hash,
        replicates_per_cell=replicates_per_cell,
        cell_count=len(cells),
        realized_budget_by_unit=budget,
        cells=cells,
        auxiliary_controls=_auxiliary_record(design),
        software_commit=None,
        package_source_sha256=None,
        python_version=platform.python_version(),
    )


def metric_lab_record_bytes(record: MetricLabRunRecord) -> bytes:
    if not isinstance(record, MetricLabRunRecord):
        raise TypeError("record must be a MetricLabRunRecord")
    payload = json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (payload + "\n").encode("utf-8")


def finalize_metric_lab_run(
    path: Path, preview: MetricLabRunRecord
) -> MetricLabRunRecord:
    if preview.software_commit is not None or preview.package_source_sha256 is not None:
        raise ValueError("finalization requires an unfinalized preview record")
    commit, package_hash = software_identity()
    finalized = preview.model_copy(
        update={
            "software_commit": commit,
            "package_source_sha256": package_hash,
            "python_version": platform.python_version(),
        }
    )
    finalized = MetricLabRunRecord.model_validate(finalized.model_dump(), strict=True)
    publish_immutable(path, metric_lab_record_bytes(finalized))
    readback = read_metric_lab_run(path)
    if readback != finalized:
        raise ValueError("metric-lab artifact readback differs from finalized record")
    return finalized


def read_metric_lab_run(path: Path) -> MetricLabRunRecord:
    reject_symlink_components(path)
    if not path.is_file():
        raise ValueError("metric-lab artifact is not an ordinary file")
    data = path.read_bytes()
    try:
        record = MetricLabRunRecord.model_validate_json(data, strict=True)
    except Exception as exc:
        raise ValueError(f"invalid metric-lab artifact: {exc}") from exc
    if record.software_commit is None or record.package_source_sha256 is None:
        raise ValueError("finalized metric-lab artifact is missing software identity")
    if metric_lab_record_bytes(record) != data:
        raise ValueError("metric-lab artifact is not canonical JSON")
    return record
