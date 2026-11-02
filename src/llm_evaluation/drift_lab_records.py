"""Immutable evidence records for E-004 drift-surveillance Phase A."""

from __future__ import annotations

import platform
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from canonical_sync.codec import canonical_record_bytes, parse_canonical_record
from llm_evaluation.artifact_io import publish_immutable, reject_symlink_components
from llm_evaluation.records import software_identity

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
PRIMARY_STATES = ("stable/stable", "distribution_only", "process_only", "both")


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
        if set(self.adjusted_typed_accuracy) != required or set(self.raw_typed_accuracy) != required:
            raise ValueError("typed accuracy maps must contain exactly the four primary truth states")
        if set(self.psi_heuristic_null_false_alarm) != {"100", "200", "1000"}:
            raise ValueError("heuristic PSI null map must contain n=100,200,1000")
        if set(self.psi_calibrated_null_false_alarm) != {"100", "200", "1000"}:
            raise ValueError("calibrated PSI null map must contain n=100,200,1000")
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
