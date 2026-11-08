"""Retrospective deterministic finite-baseline PSI null oracle for E-004 Phase A.

This diagnostic integrates a Binomial(4000, 0.20) baseline drawn once per
path, then 52 conditionally independent Binomial(n, 0.20) weekly observations.
It does not revise the original held-out FINAL (FAILED) or its acceptance gate.
The reference calculation is independent of production detector functions.
"""

from __future__ import annotations

import math
from functools import lru_cache

import pytest

from llm_evaluation.drift_lab_detectors import population_stability_index  # type: ignore[import-untyped]

_BASELINE_N = 4000
_TRUE_B_PROBABILITY = 0.20
_PSEUDOCOUNT = 0.5
_THRESHOLD = 0.10
_LOOKS = 52
_WEEKLY_NS = (100, 200, 1000)
_REFERENCE_RATES = {
    100: 0.094307391306118704,
    200: 0.00091156689055408679,
    1000: 1.3871475758270881e-16,
}


@lru_cache(maxsize=None)
def _binomial_pmf(n: int, p: float) -> tuple[float, ...]:
    """Full mode-centered binomial PMF, normalized without an underflowed seed."""
    if n < 1 or not 0.0 < p < 1.0:
        raise ValueError("requires a positive sample size and 0 < p < 1")
    mode = math.floor((n + 1) * p)
    weights = [0.0] * (n + 1)
    weights[mode] = 1.0

    # Mode-centered recurrences avoid (1-p)**4000, which underflows.
    for k in range(mode, n):
        weights[k + 1] = weights[k] * (n - k) / (k + 1) * p / (1.0 - p)
    for k in range(mode, 0, -1):
        weights[k - 1] = weights[k] * k / (n - k + 1) * (1.0 - p) / p

    total_weight = math.fsum(weights)
    return tuple(weight / total_weight for weight in weights)


def _psi(b: int, k: int, n: int) -> float:
    """Independent two-stratum PSI with the prescribed 0.5 pseudocounts."""
    baseline_b = (b + _PSEUDOCOUNT) / (_BASELINE_N + 2.0 * _PSEUDOCOUNT)
    current_b = (k + _PSEUDOCOUNT) / (n + 2.0 * _PSEUDOCOUNT)
    return (current_b - baseline_b) * math.log(current_b / baseline_b) + (
        baseline_b - current_b
    ) * math.log((1.0 - current_b) / (1.0 - baseline_b))


def _conditional_weekly_alarm(
    b: int, n: int, *, threshold: float = _THRESHOLD
) -> float:
    return math.fsum(
        probability
        for k, probability in enumerate(_binomial_pmf(n, _TRUE_B_PROBABILITY))
        if _psi(b, k, n) > threshold
    )


def _horizon_probability(q: float) -> float:
    """Probability of >=1 alarm in 52 conditionally independent weeks."""
    if q == 1.0:
        return 1.0
    return -math.expm1(_LOOKS * math.log1p(-q))


@lru_cache(maxsize=None)
def _finite_baseline_aggregates(n: int) -> tuple[float, float]:
    """Return (E[h(q(B0))], E[q(B0)]), preserving the shared baseline."""
    baseline_pmf = _binomial_pmf(_BASELINE_N, _TRUE_B_PROBABILITY)
    weighted_horizons: list[float] = []
    weighted_weekly: list[float] = []
    for b, probability in enumerate(baseline_pmf):
        if probability == 0.0:  # Negligible tails may underflow.
            continue
        q = _conditional_weekly_alarm(b, n)
        weighted_horizons.append(probability * _horizon_probability(q))
        weighted_weekly.append(probability * q)
    return math.fsum(weighted_horizons), math.fsum(weighted_weekly)


@pytest.mark.parametrize("n", (_BASELINE_N, *_WEEKLY_NS))
def test_binomial_normalization_mean_variance_and_mode(n: int) -> None:
    pmf = _binomial_pmf(n, _TRUE_B_PROBABILITY)
    p = _TRUE_B_PROBABILITY
    mode = math.floor((n + 1) * p)

    assert len(pmf) == n + 1
    assert all(math.isfinite(weight) and weight >= 0.0 for weight in pmf)
    assert math.fsum(pmf) == pytest.approx(1.0, abs=2e-14, rel=0.0)
    assert pmf[mode] > 0.0

    mean = math.fsum(k * weight for k, weight in enumerate(pmf))
    variance = math.fsum((k - n * p) ** 2 * weight for k, weight in enumerate(pmf))
    assert mean == pytest.approx(n * p, rel=2e-13, abs=1e-11)
    assert variance == pytest.approx(n * p * (1.0 - p), rel=2e-13, abs=1e-11)


@pytest.mark.parametrize(
    ("n", "k"),
    (
        (4000, 760),
        (4000, 800),
        (4000, 850),
        (100, 0),
        (100, 20),
        (100, 100),
        (200, 40),
        (1000, 150),
        (1000, 200),
        (1000, 300),
    ),
)
def test_mode_centered_probabilities_match_independent_lgamma(n: int, k: int) -> None:
    p = _TRUE_B_PROBABILITY
    log_probability = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    reference = math.exp(log_probability)
    assert _binomial_pmf(n, p)[k] == pytest.approx(reference, rel=2e-10, abs=0.0)


@pytest.mark.parametrize(
    ("b", "n", "k"),
    (
        (600, 100, 0),
        (600, 100, 20),
        (600, 200, 200),
        (800, 100, 0),
        (800, 100, 35),
        (800, 1000, 1000),
        (1000, 100, 100),
        (1000, 200, 40),
        (1000, 1000, 0),
        (1000, 1000, 250),
    ),
)
def test_psi_matches_production_for_noncentral_baselines(
    b: int, n: int, k: int
) -> None:
    expected = _psi(b, k, n)
    actual = population_stability_index(
        (_BASELINE_N - b, b), (n - k, k), pseudocount=_PSEUDOCOUNT
    )
    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-14)


def test_conditional_horizon_and_endpoint_probabilities() -> None:
    q = _conditional_weekly_alarm(800, 100)
    assert 0.0 < q < _horizon_probability(q) < 1.0
    assert _horizon_probability(0.0) == 0.0
    assert _horizon_probability(1.0) == 1.0
    assert _horizon_probability(q) == pytest.approx(
        1.0 - (1.0 - q) ** _LOOKS, rel=0.0, abs=1e-14
    )


def test_strict_psi_threshold_tie_and_nextafter_mass_entry() -> None:
    b, n, k = 800, 100, 15
    tie = _psi(b, k, n)
    lower = math.nextafter(tie, -math.inf)
    assert not (_psi(b, k, n) > tie)
    assert _psi(b, k, n) > lower

    at_tie = _conditional_weekly_alarm(b, n, threshold=tie)
    just_below = _conditional_weekly_alarm(b, n, threshold=lower)
    assert just_below > at_tie
    assert just_below - at_tie == pytest.approx(
        _binomial_pmf(n, _TRUE_B_PROBABILITY)[k], rel=2e-12, abs=1e-14
    )


@pytest.mark.parametrize(
    ("n", "reference", "tolerance"),
    (
        (100, _REFERENCE_RATES[100], 1e-8),
        (200, _REFERENCE_RATES[200], 1e-8),
        (1000, _REFERENCE_RATES[1000], 1e-7),
    ),
)
def test_exact_finite_baseline_horizon_probabilities(
    n: int, reference: float, tolerance: float
) -> None:
    correct, _ = _finite_baseline_aggregates(n)
    assert correct > 0.0
    assert correct == pytest.approx(reference, rel=tolerance, abs=0.0)


def test_shared_baseline_negative_control_detects_wrong_aggregation() -> None:
    correct, average_weekly = _finite_baseline_aggregates(100)
    wrong = _horizon_probability(average_weekly)
    assert wrong > correct
    assert wrong != pytest.approx(correct, rel=1e-7, abs=0.0)
    assert wrong - correct == pytest.approx(0.0003868741, rel=1e-6, abs=0.0)


@pytest.mark.parametrize(
    ("n", "fixed_reference"),
    ((100, 0.079524276278048867), (200, 0.00049036384919115225)),
)
def test_finite_exceeds_known_fixed_baseline_for_this_design(
    n: int, fixed_reference: float
) -> None:
    finite, _ = _finite_baseline_aggregates(n)
    fixed = _horizon_probability(_conditional_weekly_alarm(800, n))
    assert fixed == pytest.approx(fixed_reference, rel=1e-8, abs=0.0)
    assert finite > fixed


def test_retrospective_spread_remains_below_frozen_minimum() -> None:
    # Diagnostic only: not an E-004 FINAL pass/fail evaluator.
    rates = [_finite_baseline_aggregates(n)[0] for n in _WEEKLY_NS]
    spread = max(rates) - min(rates)
    assert spread == pytest.approx(
        _REFERENCE_RATES[100] - _REFERENCE_RATES[1000], rel=1e-8, abs=0.0
    )
    assert spread < 0.20  # Original preregistered minimum, not revised here.
