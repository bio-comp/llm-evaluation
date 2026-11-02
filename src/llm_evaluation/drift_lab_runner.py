"""Known-truth generator primitives for E-004 Phase A."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from llm_evaluation.drift_lab_design import DriftLabDesign, DriftLabPhase, derive_drift_lab_seed
from llm_evaluation.drift_lab_detectors import (
    adjusted_standardized_residual,
    max_ewma,
    max_shewhart,
    max_upper_cusum,
    pearson_two_sample_2x2,
    population_stability_index,
    raw_aggregate_standardized_residual,
)


@dataclass(frozen=True, slots=True)
class StratumCounts:
    a_cases: int
    b_cases: int
    a_errors: int
    b_errors: int

    @property
    def total_cases(self) -> int:
        return self.a_cases + self.b_cases

    @property
    def total_errors(self) -> int:
        return self.a_errors + self.b_errors


@dataclass(frozen=True, slots=True)
class BaselineEstimate:
    b_probability: float
    a_error_rate: float
    b_error_rate: float
    aggregate_error_rate: float


@dataclass(frozen=True, slots=True)
class WeekObservation(StratumCounts):
    week: int
    truth_b_probability: float
    truth_a_error: float
    truth_b_error: float


@dataclass(frozen=True, slots=True)
class GeneratedPath:
    phase: DriftLabPhase
    scenario: str
    replicate_index: int
    seed: int
    baseline: StratumCounts
    weeks: tuple[WeekObservation, ...]


@dataclass(frozen=True, slots=True)
class PathStatistics:
    max_psi: float
    max_pearson: float
    max_raw_aggregate_ewma: float
    max_adjusted_ewma: float
    max_adjusted_cusum: float
    max_adjusted_shewhart: float
    max_fixed_average_n_shewhart: float
    max_b_stratum_ewma: float


def beta_parameters_from_mean_icc(*, mean: float, icc: float) -> tuple[float, float]:
    mean = float(mean)
    icc = float(icc)
    if not 0.0 < mean < 1.0:
        raise ValueError("mean must lie in (0, 1)")
    if not 0.0 < icc < 1.0:
        raise ValueError("icc must lie in (0, 1)")
    concentration = (1.0 - icc) / icc
    return mean * concentration, (1.0 - mean) * concentration


def variable_denominator_for_week(design: DriftLabDesign, *, week: int) -> int:
    if type(week) is not int or week < 1:
        raise ValueError("week must be a positive exact integer")
    return design.variable_denominators[(week - 1) % len(design.variable_denominators)]


def scenario_parameters(
    design: DriftLabDesign, scenario: str, *, week: int
) -> tuple[float, float, float]:
    if type(week) is not int or not 1 <= week <= design.looks:
        raise ValueError("week must be within the monitoring horizon")
    if scenario not in design.primary_truth_states:
        raise ValueError("scenario is not a frozen primary truth state")
    b_probability = float(design.baseline_b_probability)
    a_error = float(design.baseline_a_error)
    b_error = float(design.baseline_b_error)
    if week >= design.change_week:
        if scenario in {"distribution_only", "both"}:
            b_probability = float(design.shifted_b_probability)
        if scenario in {"process_only", "both"}:
            delta = float(design.process_error_delta)
            a_error += delta
            b_error += delta
    return b_probability, a_error, b_error


def _sample_binomial(rng: random.Random, n: int, p: float) -> int:
    """Exact binomial draw via geometric waiting times, using symmetry above 0.5."""
    if type(n) is not int or n < 0:
        raise ValueError("binomial n must be a non-negative exact integer")
    if not 0.0 <= p <= 1.0 or not math.isfinite(p):
        raise ValueError("binomial probability must lie in [0, 1]")
    if n == 0 or p == 0.0:
        return 0
    if p == 1.0:
        return n
    if p > 0.5:
        return n - _sample_binomial(rng, n, 1.0 - p)
    log_failure = math.log1p(-p)
    successes = 0
    position = 0
    while True:
        gap = int(math.log1p(-rng.random()) / log_failure) + 1
        position += gap
        if position > n:
            return successes
        successes += 1




def estimate_baseline(counts: StratumCounts) -> BaselineEstimate:
    if counts.a_cases <= 0 or counts.b_cases <= 0:
        raise ValueError("baseline estimate cannot use an empty stratum")
    if not 0 <= counts.a_errors <= counts.a_cases or not 0 <= counts.b_errors <= counts.b_cases:
        raise ValueError("baseline error counts must lie within stratum counts")
    total = counts.total_cases
    return BaselineEstimate(
        b_probability=counts.b_cases / total,
        a_error_rate=counts.a_errors / counts.a_cases,
        b_error_rate=counts.b_errors / counts.b_cases,
        aggregate_error_rate=counts.total_errors / total,
    )


def stress_scenario_parameters(
    design: DriftLabDesign, scenario: str, *, week: int
) -> tuple[float, float, float]:
    if type(week) is not int or not 1 <= week <= design.looks:
        raise ValueError("week must be within the monitoring horizon")
    b_probability = float(design.baseline_b_probability)
    a_error = float(design.baseline_a_error)
    b_error = float(design.baseline_b_error)
    if scenario in {"variable_denominator_stable", "overdispersion_stable"}:
        return b_probability, a_error, b_error
    if scenario == "gradual_process":
        if week >= design.change_week:
            step = min(week - design.change_week + 1, design.gradual_change_weeks)
            delta = float(design.process_error_delta) * step / design.gradual_change_weeks
            a_error += delta
            b_error += delta
        return b_probability, a_error, b_error
    if scenario == "temporary_process":
        if design.change_week <= week < design.change_week + design.temporary_change_weeks:
            delta = float(design.temporary_error_delta)
            a_error += delta
            b_error += delta
        return b_probability, a_error, b_error
    if scenario == "localized_b_process":
        if week >= design.change_week:
            b_error += float(design.process_error_delta)
        return b_probability, a_error, b_error
    raise ValueError("unknown frozen stress scenario")


def _sample_stratified_counts(
    rng: random.Random,
    *,
    total_cases: int,
    b_probability: float,
    a_error: float,
    b_error: float,
) -> StratumCounts:
    b_cases = _sample_binomial(rng, total_cases, b_probability)
    a_cases = total_cases - b_cases
    a_errors = _sample_binomial(rng, a_cases, a_error)
    b_errors = _sample_binomial(rng, b_cases, b_error)
    return StratumCounts(
        a_cases=a_cases,
        b_cases=b_cases,
        a_errors=a_errors,
        b_errors=b_errors,
    )


def generate_path(
    design: DriftLabDesign,
    *,
    phase: DriftLabPhase,
    scenario: str,
    replicate_index: int,
) -> GeneratedPath:
    stress_scenarios = {
        "gradual_process",
        "temporary_process",
        "variable_denominator_stable",
        "overdispersion_stable",
        "localized_b_process",
    }
    if scenario not in design.primary_truth_states and scenario not in stress_scenarios:
        raise ValueError("scenario is not part of the frozen Phase-A design")
    seed = derive_drift_lab_seed(
        design,
        phase=phase,
        scenario=scenario,
        replicate_index=replicate_index,
        purpose="path",
    )
    rng = random.Random(seed)
    baseline = _sample_stratified_counts(
        rng,
        total_cases=design.baseline_cases,
        b_probability=float(design.baseline_b_probability),
        a_error=float(design.baseline_a_error),
        b_error=float(design.baseline_b_error),
    )
    weeks: list[WeekObservation] = []
    for week in range(1, design.looks + 1):
        if scenario in design.primary_truth_states:
            b_probability, a_error, b_error = scenario_parameters(design, scenario, week=week)
        else:
            b_probability, a_error, b_error = stress_scenario_parameters(
                design, scenario, week=week
            )
        total_cases = (
            variable_denominator_for_week(design, week=week)
            if scenario == "variable_denominator_stable"
            else design.core_weekly_cases
        )
        if scenario == "overdispersion_stable":
            icc = float(design.overdispersion_icc)
            a_alpha, a_beta = beta_parameters_from_mean_icc(mean=a_error, icc=icc)
            b_alpha, b_beta = beta_parameters_from_mean_icc(mean=b_error, icc=icc)
            a_error = rng.betavariate(a_alpha, a_beta)
            b_error = rng.betavariate(b_alpha, b_beta)
        counts = _sample_stratified_counts(
            rng,
            total_cases=total_cases,
            b_probability=b_probability,
            a_error=a_error,
            b_error=b_error,
        )
        weeks.append(
            WeekObservation(
                a_cases=counts.a_cases,
                b_cases=counts.b_cases,
                a_errors=counts.a_errors,
                b_errors=counts.b_errors,
                week=week,
                truth_b_probability=b_probability,
                truth_a_error=a_error,
                truth_b_error=b_error,
            )
        )
    return GeneratedPath(
        phase=phase,
        scenario=scenario,
        replicate_index=replicate_index,
        seed=seed,
        baseline=baseline,
        weeks=tuple(weeks),
    )

def compute_path_statistics(design: DriftLabDesign, path: GeneratedPath) -> PathStatistics:
    if not isinstance(design, DriftLabDesign) or not isinstance(path, GeneratedPath):
        raise TypeError("compute_path_statistics requires a DriftLabDesign and GeneratedPath")
    baseline = estimate_baseline(path.baseline)
    baseline_counts = (path.baseline.a_cases, path.baseline.b_cases)
    psi_values: list[float] = []
    pearson_values: list[float] = []
    raw_residuals: list[float] = []
    adjusted_residuals: list[float] = []
    fixed_n_residuals: list[float] = []
    b_residuals: list[float] = []
    for week in path.weeks:
        current_counts = (week.a_cases, week.b_cases)
        psi_values.append(
            population_stability_index(
                baseline_counts,
                current_counts,
                pseudocount=float(design.psi_pseudocount),
            )
        )
        pearson_values.append(pearson_two_sample_2x2(baseline_counts, current_counts))
        raw_residuals.append(
            raw_aggregate_standardized_residual(
                observed_errors=week.total_errors,
                denominator=week.total_cases,
                baseline_error_rate=baseline.aggregate_error_rate,
            )
        )
        adjusted_residuals.append(
            adjusted_standardized_residual(
                observed_errors=week.total_errors,
                stratum_counts=current_counts,
                baseline_stratum_error_rates=(baseline.a_error_rate, baseline.b_error_rate),
            )
        )
        standard_error = math.sqrt(
            baseline.aggregate_error_rate
            * (1.0 - baseline.aggregate_error_rate)
            / design.core_weekly_cases
        )
        fixed_n_residuals.append(
            (week.total_errors / week.total_cases - baseline.aggregate_error_rate) / standard_error
        )
        if week.b_cases == 0:
            b_residuals.append(0.0)
        else:
            expected_b = week.b_cases * baseline.b_error_rate
            variance_b = week.b_cases * baseline.b_error_rate * (1.0 - baseline.b_error_rate)
            b_residuals.append((week.b_errors - expected_b) / math.sqrt(variance_b))
    smoothing = float(design.ewma_lambda)
    return PathStatistics(
        max_psi=max(psi_values),
        max_pearson=max(pearson_values),
        max_raw_aggregate_ewma=max_ewma(raw_residuals, smoothing=smoothing),
        max_adjusted_ewma=max_ewma(adjusted_residuals, smoothing=smoothing),
        max_adjusted_cusum=max_upper_cusum(
            adjusted_residuals, reference_value=float(design.cusum_reference_value)
        ),
        max_adjusted_shewhart=max_shewhart(adjusted_residuals),
        max_fixed_average_n_shewhart=max_shewhart(fixed_n_residuals),
        max_b_stratum_ewma=max_ewma(b_residuals, smoothing=smoothing),
    )

