"""Known-truth generator primitives for E-004 Phase A."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

from llm_evaluation.drift_lab_design import DriftLabDesign, DriftLabPhase, derive_drift_lab_seed
from llm_evaluation.drift_lab_detectors import (
    adjusted_standardized_residual,
    calibrate_exceedance_threshold,
    calibrate_layered_equal_tail,
    ewma_path,
    upper_cusum_path,
    pearson_two_sample_2x2,
    population_stability_index,
    raw_aggregate_standardized_residual,
    typed_alarm,
)
from llm_evaluation.drift_lab_records import (
    DriftOutcomeSummaryRecord,
    ScenarioPolicyOutcomeRecord,
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
class PathMonitoringSeries:
    psi: tuple[float, ...]
    pearson: tuple[float, ...]
    raw_aggregate_ewma: tuple[float, ...]
    adjusted_ewma: tuple[float, ...]
    adjusted_cusum: tuple[float, ...]
    adjusted_shewhart: tuple[float, ...]
    fixed_average_n_shewhart: tuple[float, ...]
    b_stratum_ewma: tuple[float, ...]


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

def compute_path_monitoring_series(
    design: DriftLabDesign, path: GeneratedPath
) -> PathMonitoringSeries:
    if not isinstance(design, DriftLabDesign) or not isinstance(path, GeneratedPath):
        raise TypeError("compute_path_monitoring_series requires a DriftLabDesign and GeneratedPath")
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
        expected_rate = (
            week.a_cases * baseline.a_error_rate + week.b_cases * baseline.b_error_rate
        ) / week.total_cases
        variance_per_case = (
            (week.a_cases / week.total_cases)
            * baseline.a_error_rate
            * (1.0 - baseline.a_error_rate)
            + (week.b_cases / week.total_cases)
            * baseline.b_error_rate
            * (1.0 - baseline.b_error_rate)
        )
        fixed_n_standard_error = math.sqrt(variance_per_case / design.core_weekly_cases)
        fixed_n_residuals.append(
            (week.total_errors / week.total_cases - expected_rate) / fixed_n_standard_error
        )
        if week.b_cases == 0:
            b_residuals.append(0.0)
        else:
            expected_b = week.b_cases * baseline.b_error_rate
            variance_b = week.b_cases * baseline.b_error_rate * (1.0 - baseline.b_error_rate)
            b_residuals.append((week.b_errors - expected_b) / math.sqrt(variance_b))
    smoothing = float(design.ewma_lambda)
    return PathMonitoringSeries(
        psi=tuple(psi_values),
        pearson=tuple(pearson_values),
        raw_aggregate_ewma=ewma_path(raw_residuals, smoothing=smoothing),
        adjusted_ewma=ewma_path(adjusted_residuals, smoothing=smoothing),
        adjusted_cusum=upper_cusum_path(
            adjusted_residuals, reference_value=float(design.cusum_reference_value)
        ),
        adjusted_shewhart=tuple(adjusted_residuals),
        fixed_average_n_shewhart=tuple(fixed_n_residuals),
        b_stratum_ewma=ewma_path(b_residuals, smoothing=smoothing),
    )


def compute_path_statistics(design: DriftLabDesign, path: GeneratedPath) -> PathStatistics:
    series = compute_path_monitoring_series(design, path)
    return PathStatistics(
        max_psi=max(series.psi),
        max_pearson=max(series.pearson),
        max_raw_aggregate_ewma=max(series.raw_aggregate_ewma),
        max_adjusted_ewma=max(series.adjusted_ewma),
        max_adjusted_cusum=max(series.adjusted_cusum),
        max_adjusted_shewhart=max(series.adjusted_shewhart),
        max_fixed_average_n_shewhart=max(series.fixed_average_n_shewhart),
        max_b_stratum_ewma=max(series.b_stratum_ewma),
    )


@dataclass(frozen=True, slots=True)
class DevelopmentCalibrationSummary:
    path_count: int
    thresholds: dict[str, float]
    layered_component_tail_probability: dict[str, float]
    development_family_false_alarm: dict[str, float]
    psi_null_thresholds: dict[int, float]


def _generate_stable_with_weekly_cases(
    design: DriftLabDesign,
    *,
    phase: DriftLabPhase,
    replicate_index: int,
    weekly_cases: int,
) -> GeneratedPath:
    if type(weekly_cases) is not int or weekly_cases < 1:
        raise ValueError("weekly_cases must be a positive exact integer")
    scenario = f"psi_null_n{weekly_cases}"
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
        counts = _sample_stratified_counts(
            rng,
            total_cases=weekly_cases,
            b_probability=float(design.baseline_b_probability),
            a_error=float(design.baseline_a_error),
            b_error=float(design.baseline_b_error),
        )
        weeks.append(
            WeekObservation(
                a_cases=counts.a_cases,
                b_cases=counts.b_cases,
                a_errors=counts.a_errors,
                b_errors=counts.b_errors,
                week=week,
                truth_b_probability=float(design.baseline_b_probability),
                truth_a_error=float(design.baseline_a_error),
                truth_b_error=float(design.baseline_b_error),
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


def _rate(flags: list[bool]) -> float:
    if not flags:
        raise ValueError("probability estimate requires at least one path")
    return sum(flags) / len(flags)


def _decimal_probability(value: float) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    return format(value, ".17g")


def _single_family_rate(values: list[float], threshold: float) -> float:
    return _rate([value > threshold for value in values])


def run_development_calibration(
    design: DriftLabDesign, *, path_count: int
) -> DevelopmentCalibrationSummary:
    if not isinstance(design, DriftLabDesign):
        raise TypeError("design must be a DriftLabDesign")
    if type(path_count) is not int or path_count < 1:
        raise ValueError("path_count must be a positive exact integer")
    target = float(design.target_family_false_alarm)
    stable_stats = [
        compute_path_statistics(
            design,
            generate_path(
                design,
                phase=DriftLabPhase.DEVELOPMENT,
                scenario="stable/stable",
                replicate_index=index,
            ),
        )
        for index in range(path_count)
    ]
    channels = {
        "psi": [item.max_psi for item in stable_stats],
        "pearson": [item.max_pearson for item in stable_stats],
        "adjusted_ewma": [item.max_adjusted_ewma for item in stable_stats],
        "adjusted_cusum": [item.max_adjusted_cusum for item in stable_stats],
        "adjusted_shewhart": [item.max_adjusted_shewhart for item in stable_stats],
        "fixed_average_n_shewhart": [item.max_fixed_average_n_shewhart for item in stable_stats],
        "b_stratum_ewma": [item.max_b_stratum_ewma for item in stable_stats],
        "raw_aggregate_ewma": [item.max_raw_aggregate_ewma for item in stable_stats],
    }
    thresholds = {
        name: calibrate_exceedance_threshold(values, target_tail_probability=target).threshold
        for name, values in channels.items()
        if name not in {"raw_aggregate_ewma"}
    }
    development_rates: dict[str, float] = {
        "psi_only": _single_family_rate(channels["psi"], thresholds["psi"]),
        "pearson_only": _single_family_rate(channels["pearson"], thresholds["pearson"]),
        "spc_only_ewma": _single_family_rate(
            channels["adjusted_ewma"], thresholds["adjusted_ewma"]
        ),
        "spc_only_cusum": _single_family_rate(
            channels["adjusted_cusum"], thresholds["adjusted_cusum"]
        ),
    }
    component_grid = tuple(float(item) for item in design.component_tail_grid)
    layered_specs = {
        "layered_raw": channels["raw_aggregate_ewma"],
        "layered_adjusted_ewma": channels["adjusted_ewma"],
        "layered_adjusted_cusum": channels["adjusted_cusum"],
    }
    layered_tail: dict[str, float] = {}
    for policy, process_values in layered_specs.items():
        result = calibrate_layered_equal_tail(
            channels["psi"],
            process_values,
            target_family_probability=target,
            component_tail_grid=component_grid,
        )
        thresholds[f"{policy}.distribution"] = result.distribution_threshold
        thresholds[f"{policy}.process"] = result.process_threshold
        layered_tail[policy] = result.component_tail_probability
        development_rates[policy] = result.family_exceedance_probability

    psi_null_thresholds: dict[int, float] = {}
    for weekly_cases in design.psi_null_weekly_cases:
        values = [
            compute_path_statistics(
                design,
                _generate_stable_with_weekly_cases(
                    design,
                    phase=DriftLabPhase.DEVELOPMENT,
                    replicate_index=index,
                    weekly_cases=weekly_cases,
                ),
            ).max_psi
            for index in range(path_count)
        ]
        psi_null_thresholds[int(weekly_cases)] = calibrate_exceedance_threshold(
            values, target_tail_probability=target
        ).threshold

    return DevelopmentCalibrationSummary(
        path_count=path_count,
        thresholds=thresholds,
        layered_component_tail_probability=layered_tail,
        development_family_false_alarm=development_rates,
        psi_null_thresholds=psi_null_thresholds,
    )


def _policy_alarm(stats: PathStatistics, calibration: DevelopmentCalibrationSummary, policy: str) -> bool:
    t = calibration.thresholds
    if policy == "psi_only":
        return stats.max_psi > t["psi"]
    if policy == "pearson_only":
        return stats.max_pearson > t["pearson"]
    if policy == "spc_only_ewma":
        return stats.max_adjusted_ewma > t["adjusted_ewma"]
    if policy == "spc_only_cusum":
        return stats.max_adjusted_cusum > t["adjusted_cusum"]
    process_attr = {
        "layered_raw": "max_raw_aggregate_ewma",
        "layered_adjusted_ewma": "max_adjusted_ewma",
        "layered_adjusted_cusum": "max_adjusted_cusum",
    }.get(policy)
    if process_attr is None:
        raise ValueError("unknown E-004 policy")
    return (
        stats.max_psi > t[f"{policy}.distribution"]
        or getattr(stats, process_attr) > t[f"{policy}.process"]
    )


def _typed_policy_alarm(
    stats: PathStatistics, calibration: DevelopmentCalibrationSummary, *, adjusted: bool
) -> str:
    policy = "layered_adjusted_ewma" if adjusted else "layered_raw"
    process_value = stats.max_adjusted_ewma if adjusted else stats.max_raw_aggregate_ewma
    return typed_alarm(
        stats.max_psi > calibration.thresholds[f"{policy}.distribution"],
        process_value > calibration.thresholds[f"{policy}.process"],
    )


def run_final_summary(
    design: DriftLabDesign,
    *,
    calibration: DevelopmentCalibrationSummary,
    path_count: int,
) -> DriftOutcomeSummaryRecord:
    if type(path_count) is not int or path_count < 1:
        raise ValueError("path_count must be a positive exact integer")
    policies = (
        "psi_only",
        "pearson_only",
        "spc_only_ewma",
        "spc_only_cusum",
        "layered_raw",
        "layered_adjusted_ewma",
        "layered_adjusted_cusum",
    )
    primary_stats: dict[str, list[PathStatistics]] = {}
    for scenario in design.primary_truth_states:
        primary_stats[scenario] = [
            compute_path_statistics(
                design,
                generate_path(
                    design,
                    phase=DriftLabPhase.FINAL,
                    scenario=scenario,
                    replicate_index=index,
                ),
            )
            for index in range(path_count)
        ]
    stable = primary_stats["stable/stable"]
    matched = {
        policy: _rate([_policy_alarm(stats, calibration, policy) for stats in stable])
        for policy in policies
    }

    psi_heuristic: dict[str, float] = {}
    psi_calibrated: dict[str, float] = {}
    for weekly_cases in design.psi_null_weekly_cases:
        stats = [
            compute_path_statistics(
                design,
                _generate_stable_with_weekly_cases(
                    design,
                    phase=DriftLabPhase.FINAL,
                    replicate_index=index,
                    weekly_cases=weekly_cases,
                ),
            )
            for index in range(path_count)
        ]
        key = str(weekly_cases)
        psi_heuristic[key] = _rate(
            [item.max_psi > float(design.psi_heuristic_threshold) for item in stats]
        )
        psi_calibrated[key] = _rate(
            [item.max_psi > calibration.psi_null_thresholds[int(weekly_cases)] for item in stats]
        )

    truth_alarm = {
        "stable/stable": "none",
        "distribution_only": "distribution",
        "process_only": "process",
        "both": "both",
    }
    adjusted_accuracy: dict[str, float] = {}
    raw_accuracy: dict[str, float] = {}
    for scenario_name, stats in primary_stats.items():
        expected = truth_alarm[scenario_name]
        adjusted_accuracy[scenario_name] = _rate(
            [_typed_policy_alarm(item, calibration, adjusted=True) == expected for item in stats]
        )
        raw_accuracy[scenario_name] = _rate(
            [_typed_policy_alarm(item, calibration, adjusted=False) == expected for item in stats]
        )

    distribution_stats = primary_stats["distribution_only"]
    adjusted_process_alarm = _rate(
        [
            item.max_adjusted_ewma
            > calibration.thresholds["layered_adjusted_ewma.process"]
            for item in distribution_stats
        ]
    )
    raw_process_alarm = _rate(
        [
            item.max_raw_aggregate_ewma > calibration.thresholds["layered_raw.process"]
            for item in distribution_stats
        ]
    )
    process_stats = primary_stats["process_only"]
    adjusted_distribution_alarm = _rate(
        [
            item.max_psi > calibration.thresholds["layered_adjusted_ewma.distribution"]
            for item in process_stats
        ]
    )
    raw_distribution_alarm = _rate(
        [
            item.max_psi > calibration.thresholds["layered_raw.distribution"]
            for item in process_stats
        ]
    )

    variable = [
        compute_path_statistics(
            design,
            generate_path(
                design,
                phase=DriftLabPhase.FINAL,
                scenario="variable_denominator_stable",
                replicate_index=index,
            ),
        )
        for index in range(path_count)
    ]
    fixed_n_fa = _rate(
        [
            item.max_fixed_average_n_shewhart
            > calibration.thresholds["fixed_average_n_shewhart"]
            for item in variable
        ]
    )
    adjusted_shewhart_fa = _rate(
        [
            item.max_adjusted_shewhart > calibration.thresholds["adjusted_shewhart"]
            for item in variable
        ]
    )

    overdispersed = [
        compute_path_statistics(
            design,
            generate_path(
                design,
                phase=DriftLabPhase.FINAL,
                scenario="overdispersion_stable",
                replicate_index=index,
            ),
        )
        for index in range(path_count)
    ]
    overdispersion_fa = _rate(
        [item.max_adjusted_ewma > calibration.thresholds["adjusted_ewma"] for item in overdispersed]
    )
    iid_ewma_fa = matched["spc_only_ewma"]

    localized = [
        compute_path_statistics(
            design,
            generate_path(
                design,
                phase=DriftLabPhase.FINAL,
                scenario="localized_b_process",
                replicate_index=index,
            ),
        )
        for index in range(path_count)
    ]
    localized_aggregate = _rate(
        [item.max_adjusted_ewma > calibration.thresholds["adjusted_ewma"] for item in localized]
    )
    localized_b = _rate(
        [item.max_b_stratum_ewma > calibration.thresholds["b_stratum_ewma"] for item in localized]
    )

    return DriftOutcomeSummaryRecord(
        matched_policy_false_alarm={key: _decimal_probability(value) for key, value in matched.items()},
        psi_heuristic_null_false_alarm={
            key: _decimal_probability(value) for key, value in psi_heuristic.items()
        },
        psi_calibrated_null_false_alarm={
            key: _decimal_probability(value) for key, value in psi_calibrated.items()
        },
        distribution_only_adjusted_process_alarm=_decimal_probability(adjusted_process_alarm),
        distribution_only_raw_process_alarm=_decimal_probability(raw_process_alarm),
        process_only_adjusted_distribution_alarm=_decimal_probability(
            adjusted_distribution_alarm
        ),
        process_only_raw_distribution_alarm=_decimal_probability(raw_distribution_alarm),
        adjusted_typed_accuracy={
            key: _decimal_probability(value) for key, value in adjusted_accuracy.items()
        },
        raw_typed_accuracy={key: _decimal_probability(value) for key, value in raw_accuracy.items()},
        variable_denominator_fixed_n_shewhart_false_alarm=_decimal_probability(fixed_n_fa),
        variable_denominator_adjusted_shewhart_false_alarm=_decimal_probability(
            adjusted_shewhart_fa
        ),
        overdispersion_adjusted_ewma_false_alarm=_decimal_probability(overdispersion_fa),
        iid_adjusted_ewma_false_alarm=_decimal_probability(iid_ewma_fa),
        localized_aggregate_detection=_decimal_probability(localized_aggregate),
        localized_b_detection=_decimal_probability(localized_b),
    )

_NORMAL_95_Z = 1.959963984540054


def _wilson_probability(successes: int, trials: int) -> tuple[float, float, float]:
    if type(successes) is not int or type(trials) is not int or trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires exact counts with 0 <= successes <= trials")
    estimate = successes / trials
    z2 = _NORMAL_95_Z**2
    denominator = 1.0 + z2 / trials
    center = (estimate + z2 / (2.0 * trials)) / denominator
    half = (
        _NORMAL_95_Z
        * math.sqrt(estimate * (1.0 - estimate) / trials + z2 / (4.0 * trials**2))
        / denominator
    )
    return (
        estimate,
        min(estimate, max(0.0, center - half)),
        max(estimate, min(1.0, center + half)),
    )


def _decimal_nonnegative(value: float) -> str:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("summary value must be finite and non-negative")
    return format(value, ".17g")


def _policy_alarm_trace(
    series: PathMonitoringSeries,
    calibration: DevelopmentCalibrationSummary,
    policy: str,
) -> tuple[bool, ...]:
    t = calibration.thresholds
    if policy == "psi_only":
        return tuple(value > t["psi"] for value in series.psi)
    if policy == "pearson_only":
        return tuple(value > t["pearson"] for value in series.pearson)
    if policy == "spc_only_ewma":
        return tuple(value > t["adjusted_ewma"] for value in series.adjusted_ewma)
    if policy == "spc_only_cusum":
        return tuple(value > t["adjusted_cusum"] for value in series.adjusted_cusum)
    if policy == "b_stratum_ewma":
        return tuple(value > t["b_stratum_ewma"] for value in series.b_stratum_ewma)
    if policy in {"layered_raw", "layered_adjusted_ewma", "layered_adjusted_cusum"}:
        distribution_trace = tuple(
            value > t[f"{policy}.distribution"] for value in series.psi
        )
        process_values = {
            "layered_raw": series.raw_aggregate_ewma,
            "layered_adjusted_ewma": series.adjusted_ewma,
            "layered_adjusted_cusum": series.adjusted_cusum,
        }[policy]
        return tuple(
            distribution_alarm or process_value > t[f"{policy}.process"]
            for distribution_alarm, process_value in zip(
                distribution_trace, process_values, strict=True
            )
        )
    raise ValueError("unknown E-004 reporting policy")


def _scenario_policy_outcome(
    *,
    scenario: str,
    policy: str,
    traces: list[tuple[bool, ...]],
    change_week: int | None,
) -> ScenarioPolicyOutcomeRecord:
    trials = len(traces)
    if trials < 1 or any(not trace for trace in traces):
        raise ValueError("scenario-policy reporting requires non-empty traces")
    horizon = [any(trace) for trace in traces]
    horizon_successes = sum(horizon)
    horizon_est, horizon_low, horizon_high = _wilson_probability(horizon_successes, trials)
    first_weeks = [
        next(index + 1 for index, alarm in enumerate(trace) if alarm)
        for trace in traces
        if any(trace)
    ]
    mean_alarm_weeks = sum(sum(trace) for trace in traces) / trials

    if change_week is None:
        prechange = horizon
        post_est = post_low = post_high = missed = None
        median_delay = None
    else:
        split = change_week - 1
        prechange = [any(trace[:split]) for trace in traces]
        postchange = [any(trace[split:]) for trace in traces]
        post_successes = sum(postchange)
        post_est, post_low, post_high = _wilson_probability(post_successes, trials)
        missed = 1.0 - post_est
        delays = [
            next(index + 1 for index, alarm in enumerate(trace[split:], start=split) if alarm)
            - change_week
            for trace in traces
            if any(trace[split:])
        ]
        median_delay = statistics.median(delays) if delays else None
    pre_successes = sum(prechange)
    pre_est, pre_low, pre_high = _wilson_probability(pre_successes, trials)

    return ScenarioPolicyOutcomeRecord(
        scenario=scenario,
        policy=policy,
        horizon_alarm_successes=horizon_successes,
        trials=trials,
        horizon_alarm_probability=_decimal_probability(horizon_est),
        horizon_alarm_wilson_lower=_decimal_probability(horizon_low),
        horizon_alarm_wilson_upper=_decimal_probability(horizon_high),
        prechange_alarm_probability=_decimal_probability(pre_est),
        prechange_alarm_wilson_lower=_decimal_probability(pre_low),
        prechange_alarm_wilson_upper=_decimal_probability(pre_high),
        postchange_detection_probability=(
            _decimal_probability(post_est) if post_est is not None else None
        ),
        postchange_detection_wilson_lower=(
            _decimal_probability(post_low) if post_low is not None else None
        ),
        postchange_detection_wilson_upper=(
            _decimal_probability(post_high) if post_high is not None else None
        ),
        missed_detection_probability=(
            _decimal_probability(missed) if missed is not None else None
        ),
        median_first_alarm_week=(
            _decimal_nonnegative(float(statistics.median(first_weeks)))
            if first_weeks
            else None
        ),
        median_postchange_delay_weeks=(
            _decimal_nonnegative(float(median_delay)) if median_delay is not None else None
        ),
        mean_alarm_weeks=_decimal_nonnegative(mean_alarm_weeks),
    )


def run_scenario_policy_reporting(
    design: DriftLabDesign,
    *,
    calibration: DevelopmentCalibrationSummary,
    path_count: int,
) -> tuple[ScenarioPolicyOutcomeRecord, ...]:
    if type(path_count) is not int or path_count < 1:
        raise ValueError("path_count must be a positive exact integer")
    policies = (
        "psi_only",
        "pearson_only",
        "spc_only_ewma",
        "spc_only_cusum",
        "layered_raw",
        "layered_adjusted_ewma",
        "layered_adjusted_cusum",
    )
    scenarios = (
        *design.primary_truth_states,
        "gradual_process",
        "temporary_process",
        "localized_b_process",
    )
    results: list[ScenarioPolicyOutcomeRecord] = []
    for scenario in scenarios:
        series_by_path = [
            compute_path_monitoring_series(
                design,
                generate_path(
                    design,
                    phase=DriftLabPhase.FINAL,
                    scenario=scenario,
                    replicate_index=index,
                ),
            )
            for index in range(path_count)
        ]
        change_week = None if scenario == "stable/stable" else design.change_week
        scenario_policies = policies + (("b_stratum_ewma",) if scenario == "localized_b_process" else ())
        for policy in scenario_policies:
            traces = [_policy_alarm_trace(series, calibration, policy) for series in series_by_path]
            results.append(
                _scenario_policy_outcome(
                    scenario=scenario,
                    policy=policy,
                    traces=traces,
                    change_week=change_week,
                )
            )
    return tuple(results)

