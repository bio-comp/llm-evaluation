"""Independent exact-oracle tests for the D-019 fitted-monitor demonstration.

All probabilities are synthetic, with a reference estimate frozen before six
conditionally independent monitoring looks. This is not a production gate.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import pytest

from llm_evaluation.fitted_monitor_diagnostic import (  # type: ignore[import-untyped]
    SelectionStatus,
    _binomial_pmf,
    _central_reference_bounds,
    _horizon_false_alarm,
    _strict_alarm,
    evaluate_margin,
    select_margin,
)

_MARGINS = (10, 12, 14, 16)
_REFERENCES = (
    (10, 0.057370598826341816, 0.4196338080102942),
    (12, 0.018558957621669151, 0.17800707473481228),
    (14, 0.005020808112737250, 0.058415919652308544),
    (16, 0.001148765579797580, 0.015468316212299695),
)


def _rational_binomial_pmf(n: int) -> tuple[Fraction, ...]:
    """Independent exact binomial PMF for p=1/5, no production calls."""
    return tuple(Fraction(math.comb(n, k) * 4 ** (n - k), 5**n) for k in range(n + 1))


@lru_cache(maxsize=None)
def _rational_marginal_and_conditionals(
    margin_40ths: int,
) -> tuple[Fraction, tuple[Fraction, ...], Fraction]:
    reference = _rational_binomial_pmf(40)
    weekly = _rational_binomial_pmf(20)
    weekly_rates = tuple(
        sum(
            (mass for k, mass in enumerate(weekly) if 2 * k > b + margin_40ths),
            Fraction(),
        )
        for b in range(41)
    )
    family = tuple(1 - (1 - q) ** 6 for q in weekly_rates)
    marginal = sum(
        (weight * conditional for weight, conditional in zip(reference, family)),
        Fraction(),
    )
    average_weekly = sum(
        (weight * q for weight, q in zip(reference, weekly_rates)), Fraction()
    )
    wrong = 1 - (1 - average_weekly) ** 6
    return marginal, family, wrong


@pytest.mark.parametrize("n", (20, 40))
def test_binomial_normalization_reference_distribution(n: int) -> None:
    calculated = _binomial_pmf(n)
    independent = _rational_binomial_pmf(n)
    assert len(calculated) == n + 1
    assert all(math.isfinite(value) and value >= 0.0 for value in calculated)
    assert math.fsum(calculated) == pytest.approx(1.0, rel=0.0, abs=1e-14)
    for actual, expected in zip(calculated, independent):
        assert actual == pytest.approx(float(expected), rel=1e-12, abs=1e-16)

    assert math.fsum(k * probability for k, probability in enumerate(calculated)) == (
        pytest.approx(n * 0.2, rel=1e-12)
    )


def test_reference_fit_is_frozen_and_integer_rule_matches_rational_rates() -> None:
    for b, weekly_errors, margin in (
        (0, 0, 10),
        (3, 6, 10),
        (4, 7, 10),
        (8, 10, 12),
        (13, 20, 14),
        (40, 20, 16),
    ):
        p_hat = Fraction(b, 40)
        fitted_limit = p_hat + Fraction(margin, 40)
        exact_decision = Fraction(weekly_errors, 20) > fitted_limit
        assert _strict_alarm(b, weekly_errors, margin) is exact_decision


def test_strict_integer_comparison_excludes_exact_threshold_ties() -> None:
    # 2*K = B + margin: equality must not alarm.
    assert 2 * 7 == 4 + 10
    assert _strict_alarm(4, 7, 10) is False
    assert _strict_alarm(4, 8, 10) is True
    assert _strict_alarm(4, 6, 10) is False


@pytest.mark.parametrize("margin", _MARGINS)
def test_conditional_six_look_probability_matches_exact_rational_oracle(
    margin: int,
) -> None:
    result = evaluate_margin(margin)
    _, exact_by_b, _ = _rational_marginal_and_conditionals(margin)
    assert len(result.conditional_family_false_alarm_by_reference_count) == 41
    for b, reference in enumerate(exact_by_b):
        assert result.conditional_family_false_alarm_by_reference_count[b] == (
            pytest.approx(float(reference), rel=1e-10, abs=1e-15)
        )
    assert result.max_protected_conditional_family_false_alarm == (
        pytest.approx(float(max(exact_by_b[3:14])), rel=1e-10, abs=0.0)
    )


def test_horizon_probability_handles_endpoints_and_nontrivial_conditional_rate() -> (
    None
):
    assert _horizon_false_alarm(0.0) == 0.0
    assert _horizon_false_alarm(1.0) == 1.0
    q = sum(
        (weight for k, weight in enumerate(_binomial_pmf(20)) if 2 * k > 3 + 12),
        0.0,
    )
    assert 0 < q < _horizon_false_alarm(q) < 1
    assert _horizon_false_alarm(q) == pytest.approx(1 - (1 - q) ** 6, abs=1e-14)


@pytest.mark.parametrize(("margin", "marginal", "conditional"), _REFERENCES)
def test_numerical_operating_characteristic_references(
    margin: int, marginal: float, conditional: float
) -> None:
    result = evaluate_margin(margin)
    assert result.margin_40ths == margin
    assert result.marginal_family_false_alarm == pytest.approx(
        marginal, rel=1e-10, abs=0.0
    )
    assert result.max_protected_conditional_family_false_alarm == pytest.approx(
        conditional, rel=1e-10, abs=0.0
    )


def test_central_reference_equal_tail_quantile_bounds() -> None:
    assert _central_reference_bounds(_binomial_pmf(40)) == (3, 13)
    assert evaluate_margin(12).protected_reference_count_bounds == (3, 13)

    pmf = _rational_binomial_pmf(40)
    cdf2, cdf3 = sum(pmf[:3]), sum(pmf[:4])
    cdf12, cdf13 = sum(pmf[:13]), sum(pmf[:14])
    assert cdf2 < Fraction(25, 1000) <= cdf3
    assert cdf12 < Fraction(975, 1000) <= cdf13


@pytest.mark.parametrize("margin", _MARGINS)
def test_marginal_reference_is_independently_baseline_averaged(margin: int) -> None:
    result = evaluate_margin(margin)
    rational_correct, _, _ = _rational_marginal_and_conditionals(margin)
    assert result.marginal_family_false_alarm == pytest.approx(
        float(rational_correct), rel=1e-10, abs=0.0
    )


def test_average_weekly_first_is_a_detectably_wrong_negative_control() -> None:
    result = evaluate_margin(10)
    rational_correct, _, rational_wrong = _rational_marginal_and_conditionals(10)
    wrong = float(rational_wrong)
    assert wrong - float(rational_correct) == pytest.approx(
        0.004137067935792291, rel=1e-10, abs=0.0
    )
    assert wrong > result.marginal_family_false_alarm
    assert result.marginal_family_false_alarm != pytest.approx(
        wrong, rel=1e-10, abs=0.0
    )


def test_marginal_selection_chooses_12() -> None:
    result = select_margin(_MARGINS, require_conditional=False)
    assert result.status is SelectionStatus.SELECTED
    assert result.selected_margin_40ths == 12
    assert tuple(item.margin_40ths for item in result.evaluations) == _MARGINS
    assert result.evaluations[0].marginal_family_false_alarm > 0.05
    assert result.evaluations[1].marginal_family_false_alarm <= 0.05
    assert result.evaluations[1].max_protected_conditional_family_false_alarm > 0.05


def test_conditional_selection_chooses_16() -> None:
    result = select_margin(_MARGINS, require_conditional=True)
    assert result.status is SelectionStatus.SELECTED
    assert result.selected_margin_40ths == 16
    assert all(
        item.max_protected_conditional_family_false_alarm > 0.05
        for item in result.evaluations[:-1]
    )
    assert result.evaluations[-1].marginal_family_false_alarm <= 0.05
    assert result.evaluations[-1].max_protected_conditional_family_false_alarm <= 0.05


def test_no_method_passes_is_explicit() -> None:
    result = select_margin((10, 12, 14), require_conditional=True)
    assert result.status is SelectionStatus.NO_METHOD_PASSES
    assert result.selected_margin_40ths is None
    assert tuple(item.margin_40ths for item in result.evaluations) == (10, 12, 14)


@pytest.mark.parametrize("invalid", (True, 10.0, "10", -1, None))
def test_evaluate_margin_rejects_invalid_input(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_margin(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid",
    (
        (),
        (10, 10),
        (12, 10),
        (10, -1),
        (10, True),
        (10, 12.0),
        ("10", 12),
        (None,),
    ),
)
def test_select_margin_rejects_malformed_candidates(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        select_margin(invalid, require_conditional=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", (0, 1, "true", None))
def test_select_margin_requires_boolean_conditional_policy(invalid: object) -> None:
    with pytest.raises(TypeError):
        select_margin(_MARGINS, require_conditional=invalid)  # type: ignore[arg-type]
