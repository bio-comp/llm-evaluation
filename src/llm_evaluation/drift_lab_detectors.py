from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Literal, Sequence

TypedAlarm = Literal["none", "distribution", "process", "both"]


@dataclass(frozen=True, slots=True)
class CalibratedThreshold:
    threshold: float
    exceedance_probability: float


@dataclass(frozen=True, slots=True)
class LayeredCalibration:
    distribution_threshold: float
    process_threshold: float
    component_tail_probability: float
    family_exceedance_probability: float


def _validate_counts(counts: Sequence[int], *, label: str) -> tuple[int, ...]:
    values = tuple(counts)
    if len(values) != 2:
        raise ValueError(f"{label} must contain exactly two strata")
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError(f"{label} must contain non-negative integer counts")
    if sum(values) <= 0:
        raise ValueError(f"{label} must have positive total count")
    return values


def _validate_probability(value: float, *, label: str, open_interval: bool = False) -> float:
    value = float(value)
    valid = 0.0 < value < 1.0 if open_interval else 0.0 <= value <= 1.0
    if not math.isfinite(value) or not valid:
        bounds = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{label} must be finite and in {bounds}")
    return value


def population_stability_index(
    baseline_counts: Sequence[int],
    current_counts: Sequence[int],
    *,
    pseudocount: float = 0.5,
) -> float:
    baseline = _validate_counts(baseline_counts, label="baseline_counts")
    current = _validate_counts(current_counts, label="current_counts")
    pseudocount = float(pseudocount)
    if not math.isfinite(pseudocount) or pseudocount <= 0.0:
        raise ValueError("pseudocount must be finite and positive")
    baseline_total = sum(baseline) + 2.0 * pseudocount
    current_total = sum(current) + 2.0 * pseudocount
    baseline_prop = tuple((value + pseudocount) / baseline_total for value in baseline)
    current_prop = tuple((value + pseudocount) / current_total for value in current)
    return sum(
        (current_value - baseline_value) * math.log(current_value / baseline_value)
        for baseline_value, current_value in zip(baseline_prop, current_prop, strict=True)
    )


def pearson_two_sample_2x2(
    baseline_counts: Sequence[int], current_counts: Sequence[int]
) -> float:
    baseline = _validate_counts(baseline_counts, label="baseline_counts")
    current = _validate_counts(current_counts, label="current_counts")
    rows = (float(sum(baseline)), float(sum(current)))
    columns = (float(baseline[0] + current[0]), float(baseline[1] + current[1]))
    total = rows[0] + rows[1]
    observed = ((float(baseline[0]), float(baseline[1])), (float(current[0]), float(current[1])))
    statistic = 0.0
    for row_index in range(2):
        for column_index in range(2):
            expected = rows[row_index] * columns[column_index] / total
            if expected == 0.0:
                raise ValueError("Pearson statistic is undefined with an empty pooled stratum")
            statistic += (observed[row_index][column_index] - expected) ** 2 / expected
    return statistic


def raw_aggregate_standardized_residual(
    *, observed_errors: int, denominator: int, baseline_error_rate: float
) -> float:
    if type(denominator) is not int or denominator <= 0:
        raise ValueError("denominator must be a positive integer")
    if type(observed_errors) is not int or not 0 <= observed_errors <= denominator:
        raise ValueError("observed_errors must be an integer in [0, denominator]")
    rate = _validate_probability(baseline_error_rate, label="baseline_error_rate", open_interval=True)
    expected = denominator * rate
    variance = denominator * rate * (1.0 - rate)
    return (observed_errors - expected) / math.sqrt(variance)


def adjusted_standardized_residual(
    *,
    observed_errors: int,
    stratum_counts: Sequence[int],
    baseline_stratum_error_rates: Sequence[float],
) -> float:
    counts = _validate_counts(stratum_counts, label="stratum_counts")
    rates = tuple(float(value) for value in baseline_stratum_error_rates)
    if len(rates) != 2:
        raise ValueError("baseline_stratum_error_rates must contain exactly two rates")
    rates = tuple(
        _validate_probability(value, label="baseline stratum error rate", open_interval=True)
        for value in rates
    )
    denominator = sum(counts)
    if type(observed_errors) is not int or not 0 <= observed_errors <= denominator:
        raise ValueError("observed_errors must be an integer in [0, denominator]")
    expected = sum(count * rate for count, rate in zip(counts, rates, strict=True))
    variance = sum(count * rate * (1.0 - rate) for count, rate in zip(counts, rates, strict=True))
    if variance <= 0.0:
        raise ValueError("adjusted residual variance must be positive")
    return (observed_errors - expected) / math.sqrt(variance)


def ewma_path(values: Sequence[float], *, smoothing: float = 0.2) -> tuple[float, ...]:
    smoothing = _validate_probability(smoothing, label="smoothing", open_interval=True)
    result: list[float] = []
    current = 0.0
    for raw in values:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("EWMA values must be finite")
        current = smoothing * value + (1.0 - smoothing) * current
        result.append(current)
    if not result:
        raise ValueError("EWMA requires at least one value")
    return tuple(result)


def max_ewma(values: Sequence[float], *, smoothing: float = 0.2) -> float:
    return max(ewma_path(values, smoothing=smoothing))


def upper_cusum_path(
    values: Sequence[float], *, reference_value: float = 0.5
) -> tuple[float, ...]:
    reference_value = float(reference_value)
    if not math.isfinite(reference_value) or reference_value < 0.0:
        raise ValueError("reference_value must be finite and non-negative")
    result: list[float] = []
    current = 0.0
    for raw in values:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("CUSUM values must be finite")
        current = max(0.0, current + value - reference_value)
        result.append(current)
    if not result:
        raise ValueError("CUSUM requires at least one value")
    return tuple(result)


def max_upper_cusum(values: Sequence[float], *, reference_value: float = 0.5) -> float:
    return max(upper_cusum_path(values, reference_value=reference_value))


def max_shewhart(values: Sequence[float]) -> float:
    data = tuple(float(value) for value in values)
    if not data or any(not math.isfinite(value) for value in data):
        raise ValueError("Shewhart requires at least one finite value")
    return max(data)


def _calibrate_sorted_exceedance(
    sorted_values: Sequence[float],
    cutoffs: Sequence[float],
    target_tail_probability: float,
) -> CalibratedThreshold:
    """Calibrate from a pre-sorted sample and its sorted distinct cutoffs."""
    n = len(sorted_values)
    # cutoffs is sorted increasingly and the exceedance rate is non-increasing
    # in the threshold, so the feasibility predicate is monotone.
    lo = 0
    hi = len(cutoffs)
    while lo < hi:
        mid = (lo + hi) // 2
        threshold = cutoffs[mid]
        exceedances = n - bisect.bisect_right(sorted_values, threshold)
        if exceedances / n <= target_tail_probability:
            hi = mid
        else:
            lo = mid + 1
    threshold = cutoffs[lo]
    exceedances = n - bisect.bisect_right(sorted_values, threshold)
    return CalibratedThreshold(
        threshold=threshold,
        exceedance_probability=exceedances / n,
    )


def calibrate_exceedance_threshold(
    values: Sequence[float], *, target_tail_probability: float
) -> CalibratedThreshold:
    data = tuple(float(value) for value in values)
    if not data or any(not math.isfinite(value) for value in data):
        raise ValueError("calibration requires at least one finite statistic")
    target = _validate_probability(
        target_tail_probability, label="target_tail_probability", open_interval=False
    )
    sorted_data = sorted(data)
    cutoffs = sorted(set(data))
    return _calibrate_sorted_exceedance(sorted_data, cutoffs, target)


def calibrate_layered_equal_tail(
    distribution_statistics: Sequence[float],
    process_statistics: Sequence[float],
    *,
    target_family_probability: float,
    component_tail_grid: Sequence[float],
) -> LayeredCalibration:
    distribution = tuple(float(value) for value in distribution_statistics)
    process = tuple(float(value) for value in process_statistics)
    if len(distribution) != len(process) or not distribution:
        raise ValueError("layered calibration requires equally sized non-empty paired paths")
    if any(not math.isfinite(value) for value in distribution) or any(
        not math.isfinite(value) for value in process
    ):
        raise ValueError("calibration requires at least one finite statistic")
    target_family = _validate_probability(
        target_family_probability, label="target_family_probability", open_interval=False
    )
    grid = tuple(float(value) for value in component_tail_grid)
    if not grid or tuple(sorted(set(grid))) != grid:
        raise ValueError("component_tail_grid must be non-empty, unique, and increasing")

    distribution_sorted = sorted(distribution)
    process_sorted = sorted(process)
    distribution_cutoffs = sorted(set(distribution))
    process_cutoffs = sorted(set(process))

    feasible: LayeredCalibration | None = None
    for component_tail in grid:
        validated_component_tail = _validate_probability(
            component_tail, label="component tail probability", open_interval=False
        )
        distribution_threshold = _calibrate_sorted_exceedance(
            distribution_sorted, distribution_cutoffs, validated_component_tail
        )
        process_threshold = _calibrate_sorted_exceedance(
            process_sorted, process_cutoffs, validated_component_tail
        )
        family_probability = sum(
            distribution_value > distribution_threshold.threshold
            or process_value > process_threshold.threshold
            for distribution_value, process_value in zip(distribution, process, strict=True)
        ) / len(distribution)
        if family_probability <= target_family:
            feasible = LayeredCalibration(
                distribution_threshold=distribution_threshold.threshold,
                process_threshold=process_threshold.threshold,
                component_tail_probability=component_tail,
                family_exceedance_probability=family_probability,
            )
    if feasible is None:
        raise ValueError("no component tail allocation satisfies the family alert budget")
    return feasible


def typed_alarm(distribution_alarm: bool, process_alarm: bool) -> TypedAlarm:
    if distribution_alarm and process_alarm:
        return "both"
    if distribution_alarm:
        return "distribution"
    if process_alarm:
        return "process"
    return "none"
