"""Immutable evidence records for E-004 drift-surveillance Phase A."""

from __future__ import annotations

import platform
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from canonical_sync.codec import canonical_record_bytes, canonical_record_hash, parse_canonical_record
from llm_evaluation.artifact_io import publish_immutable, reject_symlink_components
from llm_evaluation.drift_lab_design import DriftLabDesign, FrozenDriftLabDesignRecord
from llm_evaluation.records import software_identity

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
PRIMARY_STATES = ("stable/stable", "distribution_only", "process_only", "both")
MATCHED_POLICIES = (
    "psi_only",
    "pearson_only",
    "spc_only_ewma",
    "spc_only_cusum",
    "layered_raw",
    "layered_adjusted_ewma",
    "layered_adjusted_cusum",
)
REPORTING_SCENARIOS = (
    *PRIMARY_STATES,
    "gradual_process",
    "temporary_process",
    "localized_b_process",
)


class _StrictFrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def _exact_schema(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an exact integer")
        return value


def _decimal_text(value: str, *, label: str, probability: bool = False) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty trimmed decimal text")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be decimal text") from exc
    if not parsed.is_finite():
        raise ValueError(f"{label} must be finite")
    if probability and not Decimal("0") <= parsed <= Decimal("1"):
        raise ValueError(f"{label} must lie in [0, 1]")
    return value


def _decimal_map(value: dict[str, str], *, label: str, probability: bool = False) -> dict[str, str]:
    if not value:
        raise ValueError(f"{label} must be non-empty")
    if any(not key or key != key.strip() for key in value):
        raise ValueError(f"{label} keys must be non-empty and trimmed")
    return {
        key: _decimal_text(item, label=label, probability=probability)
        for key, item in sorted(value.items())
    }


class DriftOutcomeSummaryRecord(_StrictFrozenRecord):
    record_type: Literal["e004_drift_outcome_summary"] = "e004_drift_outcome_summary"
    schema_version: Literal[1] = 1
    matched_policy_false_alarm: dict[str, str]
    psi_heuristic_null_false_alarm: dict[str, str]
    psi_calibrated_null_false_alarm: dict[str, str]
    distribution_only_adjusted_process_alarm: str
    distribution_only_raw_process_alarm: str
    adjusted_typed_accuracy: dict[str, str]
    raw_typed_accuracy: dict[str, str]
    variable_denominator_fixed_n_shewhart_false_alarm: str
    variable_denominator_adjusted_shewhart_false_alarm: str
    overdispersion_adjusted_ewma_false_alarm: str
    iid_adjusted_ewma_false_alarm: str
    localized_aggregate_detection: str
    localized_b_detection: str

    @field_validator(
        "matched_policy_false_alarm",
        "psi_heuristic_null_false_alarm",
        "psi_calibrated_null_false_alarm",
        "adjusted_typed_accuracy",
        "raw_typed_accuracy",
    )
    @classmethod
    def _maps(cls, value: dict[str, str], info: object) -> dict[str, str]:
        return _decimal_map(
            value,
            label=str(getattr(info, "field_name", "probability map")),
            probability=True,
        )

    @field_validator(
        "distribution_only_adjusted_process_alarm",
        "distribution_only_raw_process_alarm",
        "variable_denominator_fixed_n_shewhart_false_alarm",
        "variable_denominator_adjusted_shewhart_false_alarm",
        "overdispersion_adjusted_ewma_false_alarm",
        "iid_adjusted_ewma_false_alarm",
        "localized_aggregate_detection",
        "localized_b_detection",
    )
    @classmethod
    def _probabilities(cls, value: str, info: object) -> str:
        return _decimal_text(
            value,
            label=str(getattr(info, "field_name", "probability")),
            probability=True,
        )

    @model_validator(mode="after")
    def _primary_state_inventory(self) -> Self:
        required = set(PRIMARY_STATES)
        if set(self.matched_policy_false_alarm) != set(MATCHED_POLICIES):
            raise ValueError("matched policy false-alarm map must contain the exact frozen policy inventory")
        if set(self.adjusted_typed_accuracy) != required or set(self.raw_typed_accuracy) != required:
            raise ValueError("typed accuracy maps must contain exactly the four primary truth states")
        if set(self.psi_heuristic_null_false_alarm) != {"100", "200", "1000"}:
            raise ValueError("heuristic PSI null map must contain n=100,200,1000")
        if set(self.psi_calibrated_null_false_alarm) != {"100", "200", "1000"}:
            raise ValueError("calibrated PSI null map must contain n=100,200,1000")
        return self


class ScenarioPolicyOutcomeRecord(_StrictFrozenRecord):
    record_type: Literal["e004_scenario_policy_outcome"] = "e004_scenario_policy_outcome"
    schema_version: Literal[1] = 1
    scenario: str
    policy: str
    horizon_alarm_successes: int
    trials: int
    horizon_alarm_probability: str
    horizon_alarm_wilson_lower: str
    horizon_alarm_wilson_upper: str
    prechange_alarm_probability: str
    prechange_alarm_wilson_lower: str
    prechange_alarm_wilson_upper: str
    postchange_detection_probability: str | None
    postchange_detection_wilson_lower: str | None
    postchange_detection_wilson_upper: str | None
    missed_detection_probability: str | None
    median_first_alarm_week: str | None
    median_postchange_delay_weeks: str | None
    mean_alarm_weeks: str

    @field_validator("scenario", "policy")
    @classmethod
    def _names(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("scenario/policy names must be non-empty and trimmed")
        return value

    @field_validator("horizon_alarm_successes", "trials")
    @classmethod
    def _counts(cls, value: int, info: object) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{getattr(info, 'field_name', 'count')} must be a non-negative exact integer")
        return value

    @field_validator(
        "horizon_alarm_probability",
        "horizon_alarm_wilson_lower",
        "horizon_alarm_wilson_upper",
        "prechange_alarm_probability",
        "prechange_alarm_wilson_lower",
        "prechange_alarm_wilson_upper",
        "postchange_detection_probability",
        "postchange_detection_wilson_lower",
        "postchange_detection_wilson_upper",
        "missed_detection_probability",
    )
    @classmethod
    def _probability_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _decimal_text(
            value,
            label=str(getattr(info, "field_name", "probability")),
            probability=True,
        )

    @field_validator("median_first_alarm_week", "median_postchange_delay_weeks", "mean_alarm_weeks")
    @classmethod
    def _nonnegative_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        parsed = _decimal_text(value, label=str(getattr(info, "field_name", "summary")))
        if Decimal(parsed) < 0:
            raise ValueError("timing/alarm-burden summaries must be non-negative")
        return parsed

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.trials < 1 or self.horizon_alarm_successes > self.trials:
            raise ValueError("scenario-policy outcome requires 0 <= successes <= trials")
        if Decimal(self.horizon_alarm_wilson_lower) > Decimal(self.horizon_alarm_probability):
            raise ValueError("horizon Wilson lower bound exceeds estimate")
        if Decimal(self.horizon_alarm_wilson_upper) < Decimal(self.horizon_alarm_probability):
            raise ValueError("horizon Wilson upper bound is below estimate")
        optional_post_probabilities = (
            self.postchange_detection_probability,
            self.postchange_detection_wilson_lower,
            self.postchange_detection_wilson_upper,
            self.missed_detection_probability,
        )
        populated = tuple(item is not None for item in optional_post_probabilities)
        if any(populated) and not all(populated):
            raise ValueError("post-change probability fields must be all present or all absent")
        if self.postchange_detection_probability is not None:
            assert self.missed_detection_probability is not None
            detection = Decimal(self.postchange_detection_probability)
            if abs(detection + Decimal(self.missed_detection_probability) - Decimal("1")) > Decimal("1e-12"):
                raise ValueError("post-change detection and missed-detection probabilities must sum to one")
            if detection > 0 and self.median_postchange_delay_weeks is None:
                raise ValueError("positive post-change detection requires a conditional median delay")
            if detection == 0 and self.median_postchange_delay_weeks is not None:
                raise ValueError("zero post-change detection must not declare a conditional median delay")
        elif self.median_postchange_delay_weeks is not None:
            raise ValueError("no-change outcome must not declare a post-change delay")
        if self.scenario == "stable/stable":
            if self.postchange_detection_probability is not None:
                raise ValueError("stable/stable must not declare post-change detection")
        elif self.scenario in REPORTING_SCENARIOS:
            if self.postchange_detection_probability is None:
                raise ValueError("changed reporting scenario requires post-change detection fields")
        else:
            raise ValueError("scenario is outside the frozen E-004 reporting inventory")
        return self


class DriftCalibrationRecord(_StrictFrozenRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.e004-drift-calibration"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["e004_drift_calibration"] = "e004_drift_calibration"
    schema_version: Literal[1] = 1
    design_sha256: str
    frozen_design_sha256: str
    development_paths: Literal[20000] = 20000
    thresholds: dict[str, str]
    layered_component_tail_probability: dict[str, str]
    development_family_false_alarm: dict[str, str]
    software_commit: str | None
    package_source_sha256: str | None
    python_version: str

    @field_validator("design_sha256", "frozen_design_sha256", "package_source_sha256")
    @classmethod
    def _hashes(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_RE.fullmatch(value) is None:
            raise ValueError("SHA-256 fields must be lowercase digests")
        return value

    @field_validator("software_commit")
    @classmethod
    def _commit(cls, value: str | None) -> str | None:
        if value is not None and _GIT_OID_RE.fullmatch(value) is None:
            raise ValueError("software_commit must be a lowercase Git object ID")
        return value

    @field_validator("thresholds")
    @classmethod
    def _thresholds(cls, value: dict[str, str]) -> dict[str, str]:
        return _decimal_map(value, label="thresholds")

    @field_validator("layered_component_tail_probability", "development_family_false_alarm")
    @classmethod
    def _rates(cls, value: dict[str, str], info: object) -> dict[str, str]:
        return _decimal_map(
            value,
            label=str(getattr(info, "field_name", "rate map")),
            probability=True,
        )

    @field_validator("python_version")
    @classmethod
    def _python_version(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("python_version must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def _software_pair(self) -> Self:
        if (self.software_commit is None) != (self.package_source_sha256 is None):
            raise ValueError("software commit and package digest must be both present or both absent")
        return self


class DriftFinalRunRecord(_StrictFrozenRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.e004-drift-final-run"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["e004_drift_final_run"] = "e004_drift_final_run"
    schema_version: Literal[1] = 1
    design_sha256: str
    frozen_design_sha256: str
    calibration_record_sha256: str
    final_paths_per_scenario: Literal[20000] = 20000
    outcomes: DriftOutcomeSummaryRecord
    scenario_policy_outcomes: tuple[ScenarioPolicyOutcomeRecord, ...]
    realized_cases: int
    runtime_seconds: str
    software_commit: str | None
    package_source_sha256: str | None
    python_version: str

    @field_validator(
        "design_sha256", "frozen_design_sha256", "calibration_record_sha256", "package_source_sha256"
    )
    @classmethod
    def _hashes(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_RE.fullmatch(value) is None:
            raise ValueError("SHA-256 fields must be lowercase digests")
        return value

    @field_validator("software_commit")
    @classmethod
    def _commit(cls, value: str | None) -> str | None:
        if value is not None and _GIT_OID_RE.fullmatch(value) is None:
            raise ValueError("software_commit must be a lowercase Git object ID")
        return value

    @field_validator("realized_cases")
    @classmethod
    def _cases(cls, value: int) -> int:
        if type(value) is not int or value < 1:
            raise ValueError("realized_cases must be a positive exact integer")
        return value

    @field_validator("runtime_seconds")
    @classmethod
    def _runtime(cls, value: str) -> str:
        parsed = _decimal_text(value, label="runtime_seconds")
        if Decimal(parsed) < 0:
            raise ValueError("runtime_seconds must be non-negative")
        return parsed

    @field_validator("python_version")
    @classmethod
    def _python_version(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("python_version must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def _software_pair(self) -> Self:
        if (self.software_commit is None) != (self.package_source_sha256 is None):
            raise ValueError("software commit and package digest must be both present or both absent")
        if not self.scenario_policy_outcomes:
            raise ValueError("FINAL record requires scenario-policy operating characteristics")
        keys = tuple((item.scenario, item.policy) for item in self.scenario_policy_outcomes)
        if len(keys) != len(set(keys)):
            raise ValueError("scenario-policy operating-characteristic keys must be unique")
        expected = {(scenario, policy) for scenario in REPORTING_SCENARIOS for policy in MATCHED_POLICIES}
        expected.add(("localized_b_process", "b_stratum_ewma"))
        if set(keys) != expected:
            raise ValueError("scenario-policy operating-characteristic inventory does not match frozen E-004 design")
        return self


def _finalize(path: Path, preview: DriftCalibrationRecord | DriftFinalRunRecord):
    if preview.software_commit is not None or preview.package_source_sha256 is not None:
        raise ValueError("finalization requires an unfinalized preview record")
    commit, digest = software_identity()
    finalized = preview.model_copy(
        update={
            "software_commit": commit,
            "package_source_sha256": digest,
            "python_version": platform.python_version(),
        }
    )
    finalized = type(preview).model_validate(finalized.model_dump(), strict=True)
    publish_immutable(path, canonical_record_bytes(finalized))
    return finalized


def finalize_drift_calibration(path: Path, preview: DriftCalibrationRecord) -> DriftCalibrationRecord:
    finalized = _finalize(path, preview)
    assert isinstance(finalized, DriftCalibrationRecord)
    if read_drift_calibration(path) != finalized:
        raise ValueError("calibration readback differs from finalized record")
    return finalized


def finalize_drift_final_run(path: Path, preview: DriftFinalRunRecord) -> DriftFinalRunRecord:
    finalized = _finalize(path, preview)
    assert isinstance(finalized, DriftFinalRunRecord)
    if read_drift_final_run(path) != finalized:
        raise ValueError("FINAL readback differs from finalized record")
    return finalized


def _read(path: Path, record_type: type[DriftCalibrationRecord] | type[DriftFinalRunRecord]):
    reject_symlink_components(path)
    if not path.is_file():
        raise ValueError("drift-lab artifact is not an ordinary file")
    payload = path.read_bytes()
    record = parse_canonical_record(payload, record_type)
    if canonical_record_bytes(record) != payload:
        raise ValueError("drift-lab artifact is not canonical JSON")
    return record


def read_drift_calibration(path: Path) -> DriftCalibrationRecord:
    record = _read(path, DriftCalibrationRecord)
    assert isinstance(record, DriftCalibrationRecord)
    return record


def read_drift_final_run(path: Path) -> DriftFinalRunRecord:
    record = _read(path, DriftFinalRunRecord)
    assert isinstance(record, DriftFinalRunRecord)
    return record


def _validate_frozen_design(
    design: DriftLabDesign, frozen_design: FrozenDriftLabDesignRecord
) -> tuple[str, str]:
    if not isinstance(design, DriftLabDesign):
        raise TypeError("design must be a DriftLabDesign")
    if not isinstance(frozen_design, FrozenDriftLabDesignRecord):
        raise TypeError("frozen_design must be a FrozenDriftLabDesignRecord")
    design_hash = canonical_record_hash(design)
    if frozen_design.design != design or frozen_design.design_sha256 != design_hash:
        raise ValueError("execution design differs from the exact frozen E-004 design")
    return design_hash, canonical_record_hash(frozen_design)


def preview_drift_calibration(
    *,
    design: DriftLabDesign,
    frozen_design: FrozenDriftLabDesignRecord,
    development_paths: int,
    thresholds: dict[str, str],
    layered_component_tail_probability: dict[str, str],
    development_family_false_alarm: dict[str, str],
) -> DriftCalibrationRecord:
    design_hash, frozen_hash = _validate_frozen_design(design, frozen_design)
    if development_paths != design.development_paths or development_paths != 20_000:
        raise ValueError("E-004 DEVELOPMENT requires exactly 20000 paths per calibration scenario")
    return DriftCalibrationRecord(
        design_sha256=design_hash,
        frozen_design_sha256=frozen_hash,
        development_paths=20_000,
        thresholds=thresholds,
        layered_component_tail_probability=layered_component_tail_probability,
        development_family_false_alarm=development_family_false_alarm,
        software_commit=None,
        package_source_sha256=None,
        python_version=platform.python_version(),
    )


def preview_drift_final_run(
    *,
    design: DriftLabDesign,
    frozen_design: FrozenDriftLabDesignRecord,
    calibration: DriftCalibrationRecord,
    final_paths_per_scenario: int,
    outcomes: DriftOutcomeSummaryRecord,
    scenario_policy_outcomes: tuple[ScenarioPolicyOutcomeRecord, ...],
    realized_cases: int,
    runtime_seconds: str,
) -> DriftFinalRunRecord:
    design_hash, frozen_hash = _validate_frozen_design(design, frozen_design)
    if final_paths_per_scenario != design.final_paths_per_scenario or final_paths_per_scenario != 20_000:
        raise ValueError("E-004 FINAL requires exactly 20000 paths per frozen scenario")
    if calibration.software_commit is None or calibration.package_source_sha256 is None:
        raise ValueError("E-004 FINAL requires a finalized DEVELOPMENT calibration record")
    if calibration.design_sha256 != design_hash or calibration.frozen_design_sha256 != frozen_hash:
        raise ValueError("calibration record does not bind the exact frozen E-004 design")
    return DriftFinalRunRecord(
        design_sha256=design_hash,
        frozen_design_sha256=frozen_hash,
        calibration_record_sha256=canonical_record_hash(calibration),
        final_paths_per_scenario=20_000,
        outcomes=outcomes,
        scenario_policy_outcomes=scenario_policy_outcomes,
        realized_cases=realized_cases,
        runtime_seconds=runtime_seconds,
        software_commit=None,
        package_source_sha256=None,
        python_version=platform.python_version(),
    )
