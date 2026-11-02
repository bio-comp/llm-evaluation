"""Versioned design and deterministic seed contracts for E-004 Phase A."""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from canonical_sync.codec import canonical_record_hash

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PrimaryTruthState = Literal["stable/stable", "distribution_only", "process_only", "both"]


class DriftLabPhase(StrEnum):
    DEVELOPMENT = "development"
    FINAL = "final"


class _StrictFrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def _exact_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an exact integer")
        return value


class DriftLabAcceptanceCriteria(_StrictFrozenRecord):
    record_type: Literal["e004_drift_lab_acceptance_criteria"] = (
        "e004_drift_lab_acceptance_criteria"
    )
    schema_version: Literal[1] = 1
    matched_family_false_alarm_min: str = "0.04"
    matched_family_false_alarm_max: str = "0.06"
    psi_heuristic_null_spread_min: str = "0.20"
    distribution_only_adjusted_process_alarm_max: str = "0.10"
    distribution_only_raw_process_alarm_min: str = "0.80"
    typed_accuracy_stable_min: str = "0.90"
    typed_accuracy_distribution_min: str = "0.90"
    typed_accuracy_process_min: str = "0.80"
    typed_accuracy_both_min: str = "0.80"
    typed_macro_advantage_min: str = "0.15"
    fixed_n_shewhart_false_alarm_gap_min: str = "0.10"
    overdispersion_false_alarm_inflation_min: str = "0.05"

    @field_validator("*", mode="before")
    @classmethod
    def _validate_decimal_fields(cls, value: object, info: object) -> object:
        # record_type/schema_version are validated by their own literal/exact-int contracts.
        field_name = getattr(info, "field_name", "")
        if field_name in {"record_type", "schema_version"}:
            return value
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("acceptance criteria must be trimmed decimal text")
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError("acceptance criteria must be decimal values") from exc
        if not parsed.is_finite() or not Decimal("0") <= parsed <= Decimal("1"):
            raise ValueError("acceptance criteria probabilities must lie in [0, 1]")
        return value


class DriftLabDesign(_StrictFrozenRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.e004-drift-lab-design"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["e004_drift_surveillance_phase_a_design"] = (
        "e004_drift_surveillance_phase_a_design"
    )
    schema_version: Literal[1] = 1
    acceptance: DriftLabAcceptanceCriteria = Field(default_factory=DriftLabAcceptanceCriteria)

    looks: Literal[52] = 52
    baseline_cases: Literal[4000] = 4000
    core_weekly_cases: Literal[200] = 200
    change_week: Literal[13] = 13
    baseline_b_probability: Literal["0.20"] = "0.20"
    shifted_b_probability: Literal["0.60"] = "0.60"
    baseline_a_error: Literal["0.05"] = "0.05"
    baseline_b_error: Literal["0.20"] = "0.20"
    process_error_delta: Literal["0.05"] = "0.05"
    temporary_error_delta: Literal["0.10"] = "0.10"
    gradual_change_weeks: Literal[20] = 20
    temporary_change_weeks: Literal[4] = 4
    ewma_lambda: Literal["0.20"] = "0.20"
    cusum_reference_value: Literal["0.5"] = "0.5"
    psi_pseudocount: Literal["0.5"] = "0.5"
    psi_heuristic_threshold: Literal["0.10"] = "0.10"
    target_family_false_alarm: Literal["0.05"] = "0.05"
    development_paths: Literal[20000] = 20000
    final_paths_per_scenario: Literal[20000] = 20000
    psi_null_weekly_cases: tuple[Literal[100, 200, 1000], ...] = (100, 200, 1000)
    variable_denominators: tuple[Literal[50, 100, 200, 400], ...] = (50, 100, 200, 400)
    overdispersion_icc: Literal["0.05"] = "0.05"
    component_tail_grid: tuple[str, ...] = tuple(f"{i / 1000:.3f}" for i in range(1, 51))
    primary_truth_states: tuple[PrimaryTruthState, ...] = (
        "stable/stable",
        "distribution_only",
        "process_only",
        "both",
    )
    seed_namespace_development: Literal["E004-DEVELOPMENT"] = "E004-DEVELOPMENT"
    seed_namespace_final: Literal["E004-FINAL"] = "E004-FINAL"
    alarm_comparator: Literal["strict-greater-than"] = "strict-greater-than"
    coverage_interval: Literal["wilson95-v1"] = "wilson95-v1"

    @field_validator("component_tail_grid")
    @classmethod
    def _component_grid(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("component_tail_grid must be non-empty and unique")
        parsed: list[Decimal] = []
        for item in value:
            try:
                number = Decimal(item)
            except InvalidOperation as exc:
                raise ValueError("component_tail_grid must contain decimal text") from exc
            if not Decimal("0") < number <= Decimal("1"):
                raise ValueError("component_tail_grid values must lie in (0, 1]")
            parsed.append(number)
        if tuple(sorted(parsed)) != tuple(parsed):
            raise ValueError("component_tail_grid must be increasing")
        return value

    @model_validator(mode="after")
    def _fixed_contract(self) -> Self:
        if self.primary_truth_states != (
            "stable/stable",
            "distribution_only",
            "process_only",
            "both",
        ):
            raise ValueError("primary_truth_states must preserve the frozen four-state order")
        return self


class FrozenDriftLabDesignRecord(_StrictFrozenRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.frozen-e004-drift-lab-design"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["frozen_e004_drift_surveillance_phase_a_design"] = (
        "frozen_e004_drift_surveillance_phase_a_design"
    )
    schema_version: Literal[1] = 1
    design: DriftLabDesign
    design_sha256: str
    development_paths: Literal[20000] = 20000
    final_paths_per_scenario: Literal[20000] = 20000

    @field_validator("design_sha256")
    @classmethod
    def _hash(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("design_sha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def _bind(self) -> Self:
        if self.design_sha256 != canonical_record_hash(self.design):
            raise ValueError("design_sha256 does not match the exact frozen E-004 design")
        if self.development_paths != self.design.development_paths:
            raise ValueError("development path budget does not match design")
        if self.final_paths_per_scenario != self.design.final_paths_per_scenario:
            raise ValueError("FINAL path budget does not match design")
        return self


def derive_drift_lab_seed(
    design: DriftLabDesign,
    *,
    phase: DriftLabPhase,
    scenario: str,
    replicate_index: int,
    purpose: str = "path",
) -> int:
    if not isinstance(design, DriftLabDesign):
        raise TypeError("design must be a DriftLabDesign")
    if not isinstance(phase, DriftLabPhase):
        raise TypeError("phase must be a DriftLabPhase")
    if type(replicate_index) is not int or replicate_index < 0:
        raise ValueError("replicate_index must be a non-negative exact integer")
    if not scenario or scenario != scenario.strip():
        raise ValueError("scenario must be non-empty and trimmed")
    if not purpose or purpose != purpose.strip():
        raise ValueError("purpose must be non-empty and trimmed")
    namespace = (
        design.seed_namespace_development
        if phase is DriftLabPhase.DEVELOPMENT
        else design.seed_namespace_final
    )
    payload = (
        "llm-evaluation.e004-drift-seed.v1\0"
        f"{canonical_record_hash(design)}\0{namespace}\0{scenario}\0{replicate_index}\0{purpose}"
    ).encode("utf-8")
    low_63 = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
    if phase is DriftLabPhase.FINAL:
        return low_63 | (1 << 63)
    return low_63


def freeze_drift_lab_design(design: DriftLabDesign) -> FrozenDriftLabDesignRecord:
    if not isinstance(design, DriftLabDesign):
        raise TypeError("design must be a DriftLabDesign")
    return FrozenDriftLabDesignRecord(
        design=design,
        design_sha256=canonical_record_hash(design),
        development_paths=design.development_paths,
        final_paths_per_scenario=design.final_paths_per_scenario,
    )
