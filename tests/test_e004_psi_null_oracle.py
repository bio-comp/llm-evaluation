"""Exact, fixed-known-baseline PSI null oracle for E-004 Phase A.

This diagnostic deliberately conditions on 800 B cases among 4,000 baseline cases.
The FINAL simulator instead draws a finite baseline once per path.
"""

from __future__ import annotations

import math

import pytest

from llm_evaluation import drift_lab_runner as runner  # type: ignore[import-untyped]
from llm_evaluation.drift_lab_design import DriftLabDesign  # type: ignore[import-untyped]
from llm_evaluation.drift_lab_detectors import population_stability_index  # type: ignore[import-untyped]
from llm_evaluation.drift_lab_runner import _single_family_rate  # type: ignore[import-untyped]

_BASELINE_CASES = 4000
_BASELINE_B_CASES = 800
_B_PROBABILITY = 0.20
_PSEUDOCOUNT = 0.5
_THRESHOLD = 0.10
_LOOKS = 52


def _binomial_probabilities(n: int) -> tuple[float, ...]:
    """Enumerate P(K=k) for all k, with K ~ Binomial(n, 0.20)."""
    probabilities = [(1.0 - _B_PROBABILITY) ** n]
    for k in range(n):
        probabilities.append(
            probabilities[-1]
            * (n - k)
            / (k + 1)
            * _B_PROBABILITY
            / (1.0 - _B_PROBABILITY)
        )
    return tuple(probabilities)


def _psi_for_b_cases(n: int, k: int) -> float:
    """Independent algebraic two-bin PSI; never delegates to production."""
    b = (_BASELINE_B_CASES + _PSEUDOCOUNT) / (_BASELINE_CASES + 2 * _PSEUDOCOUNT)
    c = (k + _PSEUDOCOUNT) / (n + 2 * _PSEUDOCOUNT)
    return (c - b) * math.log(c / b) + (b - c) * math.log((1.0 - c) / (1.0 - b))


def _weekly_false_alarm(n: int, *, threshold: float = _THRESHOLD) -> float:
    return math.fsum(
        probability
        for k, probability in enumerate(_binomial_probabilities(n))
        if _psi_for_b_cases(n, k) > threshold
    )


def _family_false_alarm(weekly_probability: float, *, looks: int = _LOOKS) -> float:
    return -math.expm1(looks * math.log1p(-weekly_probability))


@pytest.mark.parametrize("n", (100, 200, 1000))
def test_exact_binomial_enumeration_normalizes(n: int) -> None:
    probabilities = _binomial_probabilities(n)
    assert len(probabilities) == n + 1
    assert all(probability >= 0.0 for probability in probabilities)
    assert math.fsum(probabilities) == pytest.approx(1.0, rel=0.0, abs=2e-13)


@pytest.mark.parametrize(
    ("n", "k"),
    (
        (100, 0),
        (100, 10),
        (100, 20),
        (100, 25),
        (100, 35),
        (100, 100),
        (200, 0),
        (200, 20),
        (200, 40),
        (200, 80),
        (200, 200),
        (1000, 0),
        (1000, 150),
        (1000, 200),
        (1000, 250),
        (1000, 1000),
    ),
)
def test_independent_psi_matches_production(n: int, k: int) -> None:
    expected = _psi_for_b_cases(n, k)
    actual = population_stability_index(
        (_BASELINE_CASES - _BASELINE_B_CASES, _BASELINE_B_CASES),
        (n - k, k),
        pseudocount=_PSEUDOCOUNT,
    )
    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-14)


@pytest.mark.parametrize(
    ("n", "expected", "absolute_tolerance"),
    (
        (100, 0.0795242763, 1e-8),
        (200, 0.00049036385, 1e-9),
    ),
)
def test_exact_52_look_probabilities(
    n: int, expected: float, absolute_tolerance: float
) -> None:
    assert _family_false_alarm(_weekly_false_alarm(n)) == pytest.approx(
        expected, rel=0.0, abs=absolute_tolerance
    )


def test_large_sample_horizon_probability_matches_exact_reference() -> None:
    probability = _family_false_alarm(_weekly_false_alarm(1000))
    assert probability == pytest.approx(7.4340850775778672e-20, rel=1e-8, abs=0.0)


def test_strict_threshold_boundary_and_nextafter_mass_entry() -> None:
    n, k = 100, 15
    cutoff = _psi_for_b_cases(n, k)
    next_lower_cutoff = math.nextafter(cutoff, -math.inf)
    assert not (_psi_for_b_cases(n, k) > cutoff)
    assert _psi_for_b_cases(n, k) > next_lower_cutoff

    exact_alarm = _weekly_false_alarm(n, threshold=cutoff)
    lowered_alarm = _weekly_false_alarm(n, threshold=next_lower_cutoff)
    assert lowered_alarm > exact_alarm
    assert lowered_alarm - exact_alarm == pytest.approx(
        _binomial_probabilities(n)[k], rel=1e-12, abs=1e-14
    )

    # The standalone family-rate helper uses strict comparison too;
    # the actual FINAL heuristic comparator is exercised separately below.
    assert _single_family_rate([cutoff, cutoff, next_lower_cutoff], cutoff) == 0.0
    assert (
        _single_family_rate([cutoff, math.nextafter(cutoff, math.inf)], cutoff) == 0.5
    )


@pytest.mark.parametrize(
    ("psi", "expected_rate"),
    (
        (math.nextafter(_THRESHOLD, -math.inf), "0"),
        (_THRESHOLD, "0"),
        (math.nextafter(_THRESHOLD, math.inf), "1"),
    ),
)
def test_final_psi_heuristic_comparator_respects_threshold_ties(
    monkeypatch: pytest.MonkeyPatch, psi: float, expected_rate: str
) -> None:
    """The actual FINAL heuristic alarm must exclude statistics equal to 0.10."""
    thresholds = {
        key: 1.0
        for key in (
            "psi",
            "pearson",
            "adjusted_ewma",
            "adjusted_cusum",
            "adjusted_shewhart",
            "fixed_average_n_shewhart",
            "b_stratum_ewma",
            "layered_raw.distribution",
            "layered_raw.process",
            "layered_adjusted_ewma.distribution",
            "layered_adjusted_ewma.process",
            "layered_adjusted_cusum.distribution",
            "layered_adjusted_cusum.process",
        )
    }
    calibration = runner.DevelopmentCalibrationSummary(
        path_count=1,
        thresholds=thresholds,
        layered_component_tail_probability={},
        development_family_false_alarm={},
        psi_null_thresholds={n: 1.0 for n in (100, 200, 1000)},
    )
    stats = runner.PathStatistics(
        max_psi=psi,
        max_pearson=0.0,
        max_raw_aggregate_ewma=0.0,
        max_adjusted_ewma=0.0,
        max_adjusted_cusum=0.0,
        max_adjusted_shewhart=0.0,
        max_fixed_average_n_shewhart=0.0,
        max_b_stratum_ewma=0.0,
    )
    # Stub only the calculated statistics; run_final_summary's own
    # threshold selection, alarm comparison, and aggregation remain real.
    monkeypatch.setattr(runner, "compute_path_statistics", lambda _design, _path: stats)
    outcome = runner.run_final_summary(
        DriftLabDesign(), calibration=calibration, path_count=1
    )
    assert outcome.psi_heuristic_null_false_alarm == {
        str(n): expected_rate for n in (100, 200, 1000)
    }


def test_weekly_probability_is_not_horizon_family_probability() -> None:
    q = _weekly_false_alarm(100)
    family_probability = _family_false_alarm(q)
    assert 0.0 < q < family_probability < 1.0
    assert family_probability == pytest.approx(1.0 - (1.0 - q) ** _LOOKS, abs=1e-14)


def test_fixed_baseline_sample_size_order_for_specified_conditions() -> None:
    p100, p200, p1000 = (
        _family_false_alarm(_weekly_false_alarm(n)) for n in (100, 200, 1000)
    )
    assert p100 > p200 > p1000 > 0.0


def test_pseudocount_and_observed_proportion_negative_controls() -> None:
    # At exactly 20/100 B, the unsmoothed statistic would be zero. The
    # prescribed 0.5 pseudocounts make it strictly positive.
    smoothed = _psi_for_b_cases(100, 20)
    assert smoothed > 5e-5
    assert population_stability_index((3200, 800), (80, 20), pseudocount=0.5) == (
        pytest.approx(smoothed, rel=1e-13, abs=1e-14)
    )

    # Replacing the observed proportion with the baseline proportion produces
    # zero regardless of k; a genuinely shifted observation must exceed 0.10.
    shifted = _psi_for_b_cases(100, 35)
    assert shifted > _THRESHOLD
    assert population_stability_index((3200, 800), (65, 35), pseudocount=0.5) == (
        pytest.approx(shifted, rel=1e-13, abs=1e-14)
    )
