"""Versioned design and deterministic seed contracts for the known-truth metric lab."""

from __future__ import annotations

import hashlib
import random
import re
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from canonical_sync.codec import canonical_record_hash

FINAL_REPLICATES_PER_CELL: Literal[2000] = 2000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ForecastKind = Literal["oracle", "constant", "overconfident", "underconfident"]


class MetricLabPhase(StrEnum):
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


class MetricLabDesign(_StrictFrozenRecord):
    """Candidate/frozen scientific design parameters that affect interpretation."""

    canonical_identity_domain: ClassVar[str] = "llm-evaluation.metric-lab-design"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["known_truth_metric_lab_design"] = (
        "known_truth_metric_lab_design"
    )
    schema_version: Literal[1] = 1

    p_values: tuple[str, ...] = ("0.01", "0.1", "0.3", "0.5", "0.7", "0.9", "0.99")
    forecast_kinds: tuple[ForecastKind, ...] = (
        "oracle",
        "constant",
        "overconfident",
        "underconfident",
    )
    n_base_items: tuple[int, ...] = (50, 200, 1000)
    k_within_item: tuple[int, ...] = (1, 4, 16)
    shared_within_item: tuple[bool, ...] = (False, True)
    reference_flip_etas: tuple[str, ...] = ("0", "0.1", "0.2")

    fixed_bin_count: int = Field(default=10, ge=1)
    equal_mass_bin_count: int = Field(default=10, ge=1)
    decision_threshold: str = "0.5"
    decision_tie_break: Literal["positive", "negative"] = "positive"
    aurc_tie_convention: Literal["average", "optimistic", "pessimistic"] = "average"
    l1_debias_resamples: int = Field(default=1000, ge=1)

    ambiguity_theta_values: tuple[str, ...] = (
        "0.01",
        "0.1",
        "0.3",
        "0.5",
        "0.7",
        "0.9",
        "0.99",
    )
    ambiguity_n_items: tuple[int, ...] = (50, 200, 1000)
    ambiguity_raters_per_item: tuple[int, ...] = (3, 9, 25)
    prevalence_values: tuple[str, ...] = ("0.01", "0.1", "0.5")
    prevalence_tpr: str = "0.9"
    prevalence_fpr: str = "0.01"
    operational_coverages: tuple[str, ...] = ("0.1", "0.5", "0.9", "1")
    sampling_units: tuple[Literal["fixed_case", "question_population"], ...] = (
        "fixed_case",
        "question_population",
    )
    brier_interval_methods: tuple[
        Literal[
            "normal95-row-v1",
            "normal95-item-cluster-v1",
            "hoeffding95-item-cluster-v1",
        ],
        ...,
    ] = (
        "normal95-row-v1",
        "normal95-item-cluster-v1",
        "hoeffding95-item-cluster-v1",
    )
    calibration_independent_unit_policy: Literal["first-row-per-item-v1"] = (
        "first-row-per-item-v1"
    )
    l2_debias_method: Literal["kumar-unbiased-l2-v1"] = "kumar-unbiased-l2-v1"
    l1_debias_method: Literal["kumar-normal-resampling-l1-v1"] = (
        "kumar-normal-resampling-l1-v1"
    )
    aurc_reference_negative_control: Literal["han678-get_AURC-fa445bcd"] = (
        "han678-get_AURC-fa445bcd"
    )
    subgroup_control_version: Literal["opposing-gap-cancellation-v1"] = (
        "opposing-gap-cancellation-v1"
    )
    question_population_stress_policy: Literal[
        "k1-independent-reference-eta-endpoints-all-forecast-n-v1"
    ] = "k1-independent-reference-eta-endpoints-all-forecast-n-v1"
    l2_debias_primary_sampling_target: Literal["iid-question-population-v1"] = (
        "iid-question-population-v1"
    )
    fixed_case_l2_debias_role: Literal["stress-diagnostic-not-unbiased-v1"] = (
        "stress-diagnostic-not-unbiased-v1"
    )
    soft_label_l2_debias_policy: Literal["inapplicable-bernoulli-correction-v1"] = (
        "inapplicable-bernoulli-correction-v1"
    )

    @staticmethod
    def _validate_probability_text(value: str, *, label: str) -> str:
        if not value or value != value.strip():
            raise ValueError(f"{label} values must be non-empty and trimmed")
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"{label} values must be decimal probabilities") from exc
        if not Decimal("0") <= parsed <= Decimal("1"):
            raise ValueError(f"{label} values must lie in [0, 1]")
        # Exact textual identity is intentional. Scientific design records never
        # depend on binary floating-point JSON serialization.
        return value

    @field_validator("p_values")
    @classmethod
    def _validate_p_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("p_values must be non-empty and unique")
        return tuple(
            cls._validate_probability_text(item, label="p_values") for item in value
        )

    @field_validator("ambiguity_theta_values", "prevalence_values")
    @classmethod
    def _validate_probability_grids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("probability grids must be non-empty and unique")
        return tuple(
            cls._validate_probability_text(item, label="probability grid")
            for item in value
        )

    @field_validator("prevalence_tpr", "prevalence_fpr")
    @classmethod
    def _validate_rate_text(cls, value: str) -> str:
        return cls._validate_probability_text(value, label="rate")

    @field_validator("operational_coverages")
    @classmethod
    def _validate_operational_coverages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("operational_coverages must be non-empty and unique")
        normalized = tuple(
            cls._validate_probability_text(item, label="operational_coverages")
            for item in value
        )
        parsed = tuple(Decimal(item) for item in normalized)
        if tuple(sorted(parsed)) != parsed or parsed[-1] != Decimal("1"):
            raise ValueError(
                "operational_coverages must be increasing and terminate at 1"
            )
        return normalized

    @field_validator("forecast_kinds")
    @classmethod
    def _validate_forecast_kinds(
        cls, value: tuple[ForecastKind, ...]
    ) -> tuple[ForecastKind, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("forecast_kinds must be non-empty and unique")
        return value

    @field_validator(
        "n_base_items",
        "k_within_item",
        "ambiguity_n_items",
        "ambiguity_raters_per_item",
    )
    @classmethod
    def _validate_positive_unique_ints(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("grid integer values must be non-empty and unique")
        if any(type(item) is not int or item < 1 for item in value):
            raise ValueError("grid integer values must be positive exact integers")
        return value

    @model_validator(mode="after")
    def _odd_rater_counts(self) -> Self:
        if any(count % 2 == 0 for count in self.ambiguity_raters_per_item):
            raise ValueError("ambiguity_raters_per_item must all be odd")
        if self.sampling_units != ("fixed_case", "question_population"):
            raise ValueError(
                "sampling_units must explicitly contain fixed_case and question_population"
            )
        return self

    @field_validator("shared_within_item")
    @classmethod
    def _validate_dependence_arms(cls, value: tuple[bool, ...]) -> tuple[bool, ...]:
        if value != (False, True):
            raise ValueError(
                "shared_within_item must contain both independent and shared arms"
            )
        return value

    @field_validator("reference_flip_etas")
    @classmethod
    def _validate_reference_flip_etas(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("reference_flip_etas must be non-empty and unique")
        normalized = tuple(
            cls._validate_probability_text(item, label="reference_flip_etas")
            for item in value
        )
        if any(Decimal(item) > Decimal("0.5") for item in normalized):
            raise ValueError("reference_flip_etas must lie in [0, 0.5]")
        return normalized

    @field_validator("decision_threshold")
    @classmethod
    def _validate_decision_threshold(cls, value: str) -> str:
        return cls._validate_probability_text(value, label="decision_threshold")

    @property
    def p_value_floats(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.p_values)

    @property
    def reference_flip_eta_floats(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.reference_flip_etas)

    @property
    def decision_threshold_float(self) -> float:
        return float(self.decision_threshold)

    @property
    def ambiguity_theta_floats(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.ambiguity_theta_values)

    @property
    def prevalence_floats(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.prevalence_values)

    @property
    def prevalence_tpr_float(self) -> float:
        return float(self.prevalence_tpr)

    @property
    def prevalence_fpr_float(self) -> float:
        return float(self.prevalence_fpr)

    @property
    def operational_coverage_floats(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.operational_coverages)


class BaseCellSpec(_StrictFrozenRecord):
    record_type: Literal["known_truth_metric_lab_base_cell"] = (
        "known_truth_metric_lab_base_cell"
    )
    schema_version: Literal[1] = 1
    forecast_kind: ForecastKind
    n_base_items: int = Field(ge=1)
    k_within_item: int = Field(ge=1)
    shared_within_item: bool
    reference_flip_eta: float = Field(ge=0.0, le=0.5)

    @property
    def cell_key(self) -> str:
        dependence = "shared" if self.shared_within_item else "independent"
        return (
            f"forecast={self.forecast_kind}|n={self.n_base_items}|"
            f"k={self.k_within_item}|dependence={dependence}|"
            f"eta={self.reference_flip_eta:g}"
        )


class AmbiguityCellSpec(_StrictFrozenRecord):
    record_type: Literal["known_truth_metric_lab_ambiguity_cell"] = (
        "known_truth_metric_lab_ambiguity_cell"
    )
    schema_version: Literal[1] = 1
    forecast_kind: ForecastKind
    n_items: int = Field(ge=1)
    raters_per_item: int = Field(ge=1)

    @model_validator(mode="after")
    def _odd_raters(self) -> Self:
        if self.raters_per_item % 2 == 0:
            raise ValueError("raters_per_item must be odd")
        return self

    @property
    def cell_key(self) -> str:
        return (
            f"ambiguity|forecast={self.forecast_kind}|n={self.n_items}|"
            f"raters={self.raters_per_item}"
        )


class PrevalenceCellSpec(_StrictFrozenRecord):
    record_type: Literal["known_truth_metric_lab_prevalence_cell"] = (
        "known_truth_metric_lab_prevalence_cell"
    )
    schema_version: Literal[1] = 1
    prevalence: float = Field(ge=0.0, le=1.0)
    tpr: float = Field(ge=0.0, le=1.0)
    fpr: float = Field(ge=0.0, le=1.0)

    @property
    def cell_key(self) -> str:
        return f"prevalence|pi={self.prevalence:g}|tpr={self.tpr:g}|fpr={self.fpr:g}"


class FrozenMetricLabDesignRecord(_StrictFrozenRecord):
    """Exact identity required before any final-grid execution is permitted."""

    canonical_identity_domain: ClassVar[str] = "llm-evaluation.frozen-metric-lab-design"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["frozen_known_truth_metric_lab_design"] = (
        "frozen_known_truth_metric_lab_design"
    )
    schema_version: Literal[1] = 1
    design: MetricLabDesign
    design_sha256: str
    replicates_per_cell: Literal[2000] = FINAL_REPLICATES_PER_CELL

    @field_validator("design_sha256")
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("design_sha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def _bind_design_hash(self) -> Self:
        if canonical_record_hash(self.design) != self.design_sha256:
            raise ValueError("design_sha256 does not match the exact frozen design")
        return self


def base_grid_cells(design: MetricLabDesign) -> tuple[BaseCellSpec, ...]:
    """Enumerate the exact base cross-product in deterministic order."""
    cells: list[BaseCellSpec] = []
    for forecast_kind in design.forecast_kinds:
        for n_base_items in design.n_base_items:
            for k_within_item in design.k_within_item:
                for shared_within_item in design.shared_within_item:
                    for reference_flip_eta in design.reference_flip_eta_floats:
                        cells.append(
                            BaseCellSpec(
                                forecast_kind=forecast_kind,
                                n_base_items=n_base_items,
                                k_within_item=k_within_item,
                                shared_within_item=shared_within_item,
                                reference_flip_eta=reference_flip_eta,
                            )
                        )
    return tuple(cells)


def question_population_stress_cells(
    design: MetricLabDesign,
) -> tuple[BaseCellSpec, ...]:
    """Exact population-sampling stress inventory encoded by the design contract."""
    if design.question_population_stress_policy != (
        "k1-independent-reference-eta-endpoints-all-forecast-n-v1"
    ):
        raise ValueError("unsupported question-population stress policy")
    etas = design.reference_flip_eta_floats
    endpoint_etas = {min(etas), max(etas)}
    return tuple(
        cell
        for cell in base_grid_cells(design)
        if cell.k_within_item == 1
        and not cell.shared_within_item
        and cell.reference_flip_eta in endpoint_etas
    )


def ambiguity_grid_cells(design: MetricLabDesign) -> tuple[AmbiguityCellSpec, ...]:
    cells: list[AmbiguityCellSpec] = []
    for forecast_kind in design.forecast_kinds:
        for n_items in design.ambiguity_n_items:
            for raters_per_item in design.ambiguity_raters_per_item:
                cells.append(
                    AmbiguityCellSpec(
                        forecast_kind=forecast_kind,
                        n_items=n_items,
                        raters_per_item=raters_per_item,
                    )
                )
    return tuple(cells)


def prevalence_grid_cells(design: MetricLabDesign) -> tuple[PrevalenceCellSpec, ...]:
    return tuple(
        PrevalenceCellSpec(
            prevalence=prevalence,
            tpr=design.prevalence_tpr_float,
            fpr=design.prevalence_fpr_float,
        )
        for prevalence in design.prevalence_floats
    )


def balanced_probability_schedule(
    p_values: tuple[float, ...], n_items: int, *, seed: int
) -> tuple[float, ...]:
    """Build a reproducible near-balanced population without fixed remainder bias."""
    if type(n_items) is not int or n_items < 1:
        raise ValueError("n_items must be a positive exact integer")
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if not p_values or len(p_values) != len(set(p_values)):
        raise ValueError("p_values must be non-empty and unique")
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p_values must lie in [0, 1]")

    quotient, remainder = divmod(n_items, len(p_values))
    counts = [quotient] * len(p_values)
    start = seed % len(p_values)
    for offset in range(remainder):
        counts[(start + offset) % len(p_values)] += 1

    schedule = [
        value
        for value, count in zip(p_values, counts, strict=True)
        for _ in range(count)
    ]
    random.Random(seed).shuffle(schedule)
    return tuple(schedule)


def derive_metric_lab_seed(
    *,
    design_hash: str,
    phase: MetricLabPhase,
    cell_key: str,
    replicate_index: int,
    purpose: str,
) -> int:
    """Derive stable seeds with mechanically disjoint DEV/FINAL namespaces."""
    if _SHA256_RE.fullmatch(design_hash) is None:
        raise ValueError("design_hash must be a lowercase SHA-256 digest")
    if not isinstance(phase, MetricLabPhase):
        raise TypeError("phase must be a MetricLabPhase")
    if type(replicate_index) is not int or replicate_index < 0:
        raise ValueError("replicate_index must be a non-negative exact integer")
    if not cell_key or cell_key != cell_key.strip():
        raise ValueError("cell_key must be non-empty and trimmed")
    if not purpose or purpose != purpose.strip():
        raise ValueError("purpose must be non-empty and trimmed")

    payload = (
        "llm-evaluation.metric-lab-seed.v1\0"
        f"{design_hash}\0{phase.value}\0{cell_key}\0{replicate_index}\0{purpose}"
    ).encode("utf-8")
    low_63 = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    if phase is MetricLabPhase.FINAL:
        return low_63 | (1 << 63)
    return low_63


def freeze_metric_lab_design(design: MetricLabDesign) -> FrozenMetricLabDesignRecord:
    if not isinstance(design, MetricLabDesign):
        raise TypeError("design must be a MetricLabDesign")
    return FrozenMetricLabDesignRecord(
        design=design,
        design_sha256=canonical_record_hash(design),
        replicates_per_cell=FINAL_REPLICATES_PER_CELL,
    )
