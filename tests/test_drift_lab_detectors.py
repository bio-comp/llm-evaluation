from __future__ import annotations

import itertools
import math

import pytest

from llm_evaluation import drift_lab_detectors as d


def _brute_exceedance_threshold(data, target):
    for threshold in sorted(set(data)):
        probability = sum(value > threshold for value in data) / len(data)
        if probability <= target:
            return d.CalibratedThreshold(
                threshold=threshold, exceedance_probability=probability
            )
    raise AssertionError("maximum observed statistic must always be feasible")


def _brute_layered(distribution, process, target_family, grid):
    feasible = None
    for component_tail in grid:
        distribution_result = _brute_exceedance_threshold(distribution, component_tail)
        process_result = _brute_exceedance_threshold(process, component_tail)
        family_probability = sum(
            distribution_value > distribution_result.threshold
            or process_value > process_result.threshold
            for distribution_value, process_value in zip(
                distribution, process, strict=True
            )
        ) / len(distribution)
        if family_probability <= target_family:
            feasible = (
                distribution_result.threshold,
                process_result.threshold,
                component_tail,
                family_probability,
            )
    if feasible is None:
        raise ValueError("no component tail allocation satisfies the family alert budget")
    return feasible


def test_two_bin_psi_matches_manual_formula_and_is_symmetric() -> None:
    baseline = (80, 20)
    current = (40, 60)
    observed = d.population_stability_index(baseline, current, pseudocount=0.5)
    b = tuple((x + 0.5) / 101.0 for x in baseline)
    c = tuple((x + 0.5) / 101.0 for x in current)
    expected = sum((ci - bi) * math.log(ci / bi) for bi, ci in zip(b, c, strict=True))
    assert observed == pytest.approx(expected, abs=1e-15)
    assert d.population_stability_index(current, baseline, pseudocount=0.5) == pytest.approx(
        observed, abs=1e-15
    )
    assert d.population_stability_index((0, 100), (0, 100), pseudocount=0.5) == 0.0


def test_pearson_two_sample_2x2_matches_manual_contingency_statistic() -> None:
    # rows are baseline/current; columns are A/B
    observed = d.pearson_two_sample_2x2((80, 20), (40, 60))
    table = ((80.0, 20.0), (40.0, 60.0))
    row = (100.0, 100.0)
    col = (120.0, 80.0)
    total = 200.0
    expected = 0.0
    for i in range(2):
        for j in range(2):
            exp = row[i] * col[j] / total
            expected += (table[i][j] - exp) ** 2 / exp
    assert observed == pytest.approx(expected, abs=1e-15)
    assert d.pearson_two_sample_2x2((80, 20), (80, 20)) == 0.0


def test_raw_aggregate_residual_is_case_mix_confounded_but_adjusted_residual_is_centered() -> None:
    # Baseline: 80% A at 5% error, 20% B at 20% error => aggregate 8%.
    # Pure case-mix shift: 40% A / 60% B, conditional rates unchanged => 14% aggregate.
    raw = d.raw_aggregate_standardized_residual(
        observed_errors=28,
        denominator=200,
        baseline_error_rate=0.08,
    )
    adjusted = d.adjusted_standardized_residual(
        observed_errors=28,
        stratum_counts=(80, 120),
        baseline_stratum_error_rates=(0.05, 0.20),
    )
    assert raw > 3.0
    assert adjusted == pytest.approx(0.0, abs=1e-15)


def test_ewma_cusum_and_shewhart_recursions_are_exact() -> None:
    values = (0.0, 1.0, 2.0, -1.0)
    ewma = d.ewma_path(values, smoothing=0.2)
    assert ewma == pytest.approx((0.0, 0.2, 0.56, 0.248), abs=1e-15)
    assert d.max_ewma(values, smoothing=0.2) == pytest.approx(0.56)

    cusum = d.upper_cusum_path(values, reference_value=0.5)
    assert cusum == pytest.approx((0.0, 0.5, 2.0, 0.5), abs=1e-15)
    assert d.max_upper_cusum(values, reference_value=0.5) == pytest.approx(2.0)
    assert d.max_shewhart(values) == 2.0


def test_calibration_uses_strict_exceedance_and_tie_safe_smallest_threshold() -> None:
    values = (1.0, 2.0, 2.0, 3.0)
    calibrated = d.calibrate_exceedance_threshold(values, target_tail_probability=0.25)
    assert calibrated.threshold == 2.0
    assert calibrated.exceedance_probability == 0.25
    assert tuple(x > calibrated.threshold for x in values) == (False, False, False, True)

    tighter = d.calibrate_exceedance_threshold(values, target_tail_probability=0.20)
    assert tighter.threshold == 3.0
    assert tighter.exceedance_probability == 0.0


def test_layered_equal_tail_calibration_selects_largest_feasible_component_tail() -> None:
    distribution = tuple(float(i) for i in range(20))
    process = tuple(float(19 - i) for i in range(20))
    result = d.calibrate_layered_equal_tail(
        distribution,
        process,
        target_family_probability=0.20,
        component_tail_grid=(0.05, 0.10, 0.15, 0.20),
    )
    # With strict `>` semantics alpha=.10 yields two upper-tail alarms per channel,
    # disjoint here, so union family alarm is 4/20=.20. Alpha=.15 yields 6/20=.30.
    assert result.component_tail_probability == 0.10
    assert result.family_exceedance_probability == 0.20
    assert result.distribution_threshold == 17.0
    assert result.process_threshold == 17.0


def test_typed_alarm_preserves_distribution_and_process_channels() -> None:
    assert d.typed_alarm(False, False) == "none"
    assert d.typed_alarm(True, False) == "distribution"
    assert d.typed_alarm(False, True) == "process"
    assert d.typed_alarm(True, True) == "both"


def test_calibration_matches_brute_force_exhaustive() -> None:
    pool = (-1.0, 0.0, 1.0)
    targets = (0.0, 0.05, 0.2, 0.5, 1.0)
    for n in range(1, 6):
        for sequence in itertools.product(pool, repeat=n):
            for target in targets:
                expected = _brute_exceedance_threshold(sequence, target)
                actual = d.calibrate_exceedance_threshold(
                    sequence, target_tail_probability=target
                )
                assert actual.threshold == expected.threshold
                assert actual.exceedance_probability == expected.exceedance_probability


def test_calibration_signed_zero_case() -> None:
    data = (-0.0, 0.0, 1.0)
    targets = (0.0, 0.05, 0.2, 0.5, 1.0)
    for target in targets:
        expected = _brute_exceedance_threshold(data, target)
        actual = d.calibrate_exceedance_threshold(data, target_tail_probability=target)
        assert actual.threshold == expected.threshold
        assert actual.exceedance_probability == expected.exceedance_probability


def test_calibration_mostly_unique_127() -> None:
    values = [float(value) for value in range(127, 0, -1)]
    target = 0.05
    expected = _brute_exceedance_threshold(values, target)
    actual = d.calibrate_exceedance_threshold(values, target_tail_probability=target)
    assert actual.threshold == expected.threshold
    assert actual.exceedance_probability == expected.exceedance_probability


def test_layered_calibration_matches_naive_reference() -> None:
    # Small paired tied fixture.
    distribution = (1.0, 2.0, 2.0, 3.0, 4.0)
    process = (4.0, 3.0, 2.0, 2.0, 1.0)
    grid = (0.2, 0.4, 0.6)
    target_family = 0.6
    expected = _brute_layered(distribution, process, target_family, grid)
    actual = d.calibrate_layered_equal_tail(
        distribution,
        process,
        target_family_probability=target_family,
        component_tail_grid=grid,
    )
    assert actual.distribution_threshold == expected[0]
    assert actual.process_threshold == expected[1]
    assert actual.component_tail_probability == expected[2]
    assert actual.family_exceedance_probability == expected[3]

    # Anti-correlated fixture.
    distribution = (0.0, 0.0, 10.0, 10.0)
    process = (10.0, 10.0, 0.0, 0.0)
    grid = (0.25, 0.5, 0.75)
    target_family = 0.5
    expected = _brute_layered(distribution, process, target_family, grid)
    actual = d.calibrate_layered_equal_tail(
        distribution,
        process,
        target_family_probability=target_family,
        component_tail_grid=grid,
    )
    assert actual.distribution_threshold == expected[0]
    assert actual.process_threshold == expected[1]
    assert actual.component_tail_probability == expected[2]
    assert actual.family_exceedance_probability == expected[3]


def test_layered_calibration_largest_feasible_not_first() -> None:
    distribution = tuple(float(i) for i in range(20))
    process = tuple(float(19 - i) for i in range(20))
    grid = (0.05, 0.10, 0.15, 0.20)
    target_family = 0.20
    actual = d.calibrate_layered_equal_tail(
        distribution,
        process,
        target_family_probability=target_family,
        component_tail_grid=grid,
    )
    assert actual.component_tail_probability == 0.10


def test_layered_calibration_value_errors() -> None:
    with pytest.raises(ValueError):
        d.calibrate_layered_equal_tail(
            (1.0, float("nan")),
            (1.0, 2.0),
            target_family_probability=0.2,
            component_tail_grid=(0.1, 0.2),
        )
    with pytest.raises(ValueError):
        d.calibrate_layered_equal_tail(
            (1.0, 2.0),
            (1.0, float("inf")),
            target_family_probability=0.2,
            component_tail_grid=(0.1, 0.2),
        )
    with pytest.raises(ValueError):
        d.calibrate_layered_equal_tail((), (1.0,), target_family_probability=0.2, component_tail_grid=(0.1,))
    with pytest.raises(ValueError):
        d.calibrate_layered_equal_tail((1.0, 2.0), (1.0,), target_family_probability=0.2, component_tail_grid=(0.1,))
    with pytest.raises(ValueError):
        d.calibrate_layered_equal_tail(
            (1.0, 2.0), (1.0, 2.0), target_family_probability=1.5, component_tail_grid=(0.1, 0.2)
        )
    with pytest.raises(ValueError):
        d.calibrate_layered_equal_tail(
            (1.0, 2.0), (1.0, 2.0), target_family_probability=0.2, component_tail_grid=(0.2, 0.1)
        )

    with pytest.raises(ValueError):
        d.calibrate_exceedance_threshold((), target_tail_probability=0.1)
    with pytest.raises(ValueError):
        d.calibrate_exceedance_threshold((1.0, float("nan")), target_tail_probability=0.1)
    with pytest.raises(ValueError):
        d.calibrate_exceedance_threshold((1.0, 2.0), target_tail_probability=1.5)
