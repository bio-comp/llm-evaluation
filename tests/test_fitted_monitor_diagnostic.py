"""Independent exact-oracle tests for the D-019 fitted-monitor demonstration.

All probabilities are synthetic, with a reference estimate frozen before six
conditionally independent monitoring looks. This is not a production gate.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, dataclass, replace
from fractions import Fraction
from functools import lru_cache

import pytest

from llm_evaluation import fitted_monitor_diagnostic as policy_module  # type: ignore[import-untyped]
from llm_evaluation.fitted_monitor_diagnostic import (  # type: ignore[import-untyped]
    FrozenReferenceMarginSelection,
    FrozenReferencePolicyEvaluation,
    SelectionStatus,
    _binomial_pmf,
    _central_reference_bounds,
    _horizon_false_alarm,
    _strict_alarm,
    evaluate_detection_power,
    evaluate_frozen_reference_policy,
    evaluate_margin,
    select_margin,
    select_margin_for_frozen_reference,
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


def _rational_alternative_weekly_pmf() -> tuple[Fraction, ...]:
    """Independent Binomial(20, 3/10) oracle using integer arithmetic."""
    return tuple(
        Fraction(math.comb(20, k) * 3**k * 7 ** (20 - k), 10**20) for k in range(21)
    )


@lru_cache(maxsize=None)
def _rational_detection_power(
    margin_40ths: int,
) -> tuple[Fraction, tuple[Fraction, ...], Fraction]:
    """Enumerate over the same frozen reference for all six monitoring looks."""
    reference = _rational_binomial_pmf(40)
    weekly = _rational_alternative_weekly_pmf()
    weekly_rates = tuple(
        sum(
            (mass for k, mass in enumerate(weekly) if 2 * k > b + margin_40ths),
            Fraction(),
        )
        for b in range(41)
    )
    six_look_by_reference = tuple(1 - (1 - q) ** 6 for q in weekly_rates)
    marginal = sum(
        (reference[b] * six_look_by_reference[b] for b in range(41)),
        Fraction(),
    )
    # Deliberately incorrect: redraw or average the reference before
    # applying the nonlinear six-look transformation.
    averaged_weekly = sum(
        (reference[b] * weekly_rates[b] for b in range(41)), Fraction()
    )
    wrong_order = 1 - (1 - averaged_weekly) ** 6
    return marginal, six_look_by_reference, wrong_order


@pytest.mark.parametrize("margin", _MARGINS)
def test_detection_power_matches_independent_fraction_oracle(margin: int) -> None:
    result = evaluate_detection_power(margin)
    exact_marginal, exact_conditional, _ = _rational_detection_power(margin)
    assert result.margin_40ths == margin
    assert result.protected_reference_count_bounds == (3, 13)
    assert len(result.conditional_family_detection_probability_by_reference_count) == 41

    for b, expected in enumerate(exact_conditional):
        assert result.conditional_family_detection_probability_by_reference_count[
            b
        ] == pytest.approx(float(expected), rel=1e-12, abs=1e-15)
    assert result.marginal_family_detection_probability == pytest.approx(
        float(exact_marginal), rel=1e-12, abs=1e-15
    )
    assert result.min_protected_conditional_family_detection_probability == (
        pytest.approx(float(min(exact_conditional[3:14])), rel=1e-12, abs=1e-15)
    )
    assert result.max_protected_conditional_family_detection_probability == (
        pytest.approx(float(max(exact_conditional[3:14])), rel=1e-12, abs=1e-15)
    )


@pytest.mark.parametrize("margin", _MARGINS)
def test_alternative_strict_threshold_ties_do_not_alarm(margin: int) -> None:
    weekly = _rational_alternative_weekly_pmf()
    for b in range(41):
        candidate = b + margin
        if candidate % 2 or not 0 <= candidate // 2 <= 20:
            continue
        tie_k = candidate // 2
        assert _strict_alarm(b, tie_k, margin) is False
        if tie_k < 20:
            assert _strict_alarm(b, tie_k + 1, margin) is True
        rate_excluding_tie = sum((weekly[k] for k in range(tie_k + 1, 21)), Fraction())
        rate_including_tie = rate_excluding_tie + weekly[tie_k]
        assert rate_including_tie > rate_excluding_tie
        _, conditional, _ = _rational_detection_power(margin)
        assert conditional[b] == 1 - (1 - rate_excluding_tie) ** 6


def test_detection_power_is_nonincreasing_with_margin_for_each_frozen_fit() -> None:
    rows = [evaluate_detection_power(margin) for margin in _MARGINS]
    for low, high in zip(rows, rows[1:]):
        assert (
            low.marginal_family_detection_probability
            >= high.marginal_family_detection_probability
        )
        assert all(
            a >= b
            for a, b in zip(
                low.conditional_family_detection_probability_by_reference_count,
                high.conditional_family_detection_probability_by_reference_count,
            )
        )
        assert (
            low.min_protected_conditional_family_detection_probability
            >= high.min_protected_conditional_family_detection_probability
        )
        assert (
            low.max_protected_conditional_family_detection_probability
            >= high.max_protected_conditional_family_detection_probability
        )


@pytest.mark.parametrize("margin", _MARGINS)
def test_detection_power_wrong_order_negative_control(margin: int) -> None:
    result = evaluate_detection_power(margin)
    correct, _, wrong = _rational_detection_power(margin)
    assert wrong > correct
    assert result.marginal_family_detection_probability == pytest.approx(
        float(correct), rel=1e-12, abs=1e-15
    )
    # A mutant using 1-(1-E[q])**6 would fail this discrimination.
    assert result.marginal_family_detection_probability != pytest.approx(
        float(wrong), rel=1e-12, abs=1e-15
    )


def test_detection_power_is_immutable_and_does_not_change_null_selection() -> None:
    result = evaluate_detection_power(12)
    assert isinstance(
        result.conditional_family_detection_probability_by_reference_count, tuple
    )
    with pytest.raises(FrozenInstanceError):
        result.margin_40ths = 14  # type: ignore[misc]

    # Pre-extension null OC and both selector paths retain their exact reference rows.
    for margin, false_alarm, protected_max in _REFERENCES:
        original = evaluate_margin(margin)
        assert original.marginal_family_false_alarm == pytest.approx(
            false_alarm, rel=1e-10, abs=0.0
        )
        assert original.max_protected_conditional_family_false_alarm == pytest.approx(
            protected_max, rel=1e-10, abs=0.0
        )
    assert (
        select_margin(_MARGINS, require_conditional=False).selected_margin_40ths == 12
    )
    assert select_margin(_MARGINS, require_conditional=True).selected_margin_40ths == 16
    assert (
        select_margin((10, 12, 14), require_conditional=True).status
        is SelectionStatus.NO_METHOD_PASSES
    )


@pytest.mark.parametrize("invalid", (True, 10.0, "10", -1, None))
def test_detection_power_rejects_invalid_margin(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_detection_power(invalid)  # type: ignore[arg-type]


@dataclass(frozen=True)
class _ExactFrozenPolicy:
    """Independent rational oracle, never using production risk or selection."""

    margins_by_b: tuple[int | None, ...]
    conditional_null_by_b: tuple[Fraction | None, ...]
    conditional_alternative_by_b: tuple[Fraction | None, ...]
    mass_by_decision: tuple[Fraction, ...]
    activation: Fraction
    withheld: Fraction
    marginal_null: Fraction
    marginal_alternative: Fraction
    null_given_activation: Fraction | None
    alternative_given_activation: Fraction | None
    wrong_order_null: Fraction
    wrong_order_alternative: Fraction


@lru_cache(maxsize=None)
def _independent_frozen_policy_oracle(
    candidates: tuple[int, ...],
) -> _ExactFrozenPolicy:
    """Integrate exact six-look probabilities *after* choosing margin for each b."""
    reference = _rational_binomial_pmf(40)
    weekly_null = _rational_binomial_pmf(20)
    weekly_alternative = _rational_alternative_weekly_pmf()
    limit = Fraction(1, 20)
    selected: list[int | None] = []
    null_by_b: list[Fraction | None] = []
    alt_by_b: list[Fraction | None] = []
    weekly_null_by_b: list[Fraction] = []
    weekly_alt_by_b: list[Fraction] = []

    for b in range(41):
        chosen: int | None = None
        null_horizon: Fraction | None = None
        alternative_horizon: Fraction | None = None
        null_week = Fraction()
        alternative_week = Fraction()
        for margin in candidates:
            # Independent exact strict comparator: equality is NOT an alarm.
            q_null = sum(
                (weight for k, weight in enumerate(weekly_null) if 2 * k > b + margin),
                Fraction(),
            )
            h_null = 1 - (1 - q_null) ** 6
            if h_null <= limit:
                chosen = margin
                null_horizon = h_null
                null_week = q_null
                alternative_week = sum(
                    (
                        weight
                        for k, weight in enumerate(weekly_alternative)
                        if 2 * k > b + margin
                    ),
                    Fraction(),
                )
                alternative_horizon = 1 - (1 - alternative_week) ** 6
                break
        selected.append(chosen)
        null_by_b.append(null_horizon)
        alt_by_b.append(alternative_horizon)
        weekly_null_by_b.append(null_week)
        weekly_alt_by_b.append(alternative_week)

    margins_by_b = tuple(selected)
    mass_by_decision = tuple(
        sum(
            (reference[b] for b, margin in enumerate(selected) if margin == decision),
            Fraction(),
        )
        for decision in (None, *candidates)
    )
    withheld = mass_by_decision[0]
    activation = sum(mass_by_decision[1:], Fraction())
    marginal_null = sum(
        (
            reference[b] * value
            for b, value in enumerate(null_by_b)
            if value is not None
        ),
        Fraction(),
    )
    marginal_alternative = sum(
        (reference[b] * value for b, value in enumerate(alt_by_b) if value is not None),
        Fraction(),
    )
    # Deliberately WRONG: average per-week alarm probability over B first, as
    # if the same reference were not kept for all six looks.
    average_null = sum(
        (reference[b] * q for b, q in enumerate(weekly_null_by_b)), Fraction()
    )
    average_alternative = sum(
        (reference[b] * q for b, q in enumerate(weekly_alt_by_b)), Fraction()
    )
    return _ExactFrozenPolicy(
        margins_by_b=margins_by_b,
        conditional_null_by_b=tuple(null_by_b),
        conditional_alternative_by_b=tuple(alt_by_b),
        mass_by_decision=mass_by_decision,
        activation=activation,
        withheld=withheld,
        marginal_null=marginal_null,
        marginal_alternative=marginal_alternative,
        null_given_activation=marginal_null / activation if activation else None,
        alternative_given_activation=(
            marginal_alternative / activation if activation else None
        ),
        wrong_order_null=1 - (1 - average_null) ** 6,
        wrong_order_alternative=1 - (1 - average_alternative) ** 6,
    )


def _assert_policy_matches_independent_oracle(
    result: FrozenReferencePolicyEvaluation,
    exact: _ExactFrozenPolicy,
) -> None:
    """One comprehensive seam used for valid results and injected mutants."""
    assert result.candidate_margins_40ths == _MARGINS
    assert len(result.selections_by_reference_count) == 41
    for b, (decision, expected_margin, expected_null) in enumerate(
        zip(
            result.selections_by_reference_count,
            exact.margins_by_b,
            exact.conditional_null_by_b,
        )
    ):
        assert decision.reference_errors == b
        assert decision.selected_margin_40ths == expected_margin
        assert decision.status is (
            SelectionStatus.SELECTED
            if expected_margin is not None
            else SelectionStatus.NO_METHOD_PASSES
        )
        if expected_null is None:
            assert decision.conditional_family_false_alarm is None
        else:
            assert decision.conditional_family_false_alarm == pytest.approx(
                float(expected_null), rel=1e-12, abs=1e-15
            )
    assert len(result.decision_probability_masses) == len(_MARGINS) + 1
    for i, (decision_mass, exact_mass) in enumerate(
        zip(result.decision_probability_masses, exact.mass_by_decision)
    ):
        expected_margin = (None, *_MARGINS)[i]
        assert decision_mass.selected_margin_40ths == expected_margin
        assert decision_mass.status is (
            SelectionStatus.NO_METHOD_PASSES
            if expected_margin is None
            else SelectionStatus.SELECTED
        )
        assert decision_mass.probability_mass == pytest.approx(
            float(exact_mass), rel=1e-12, abs=1e-15
        )
    assert result.monitoring_activation_probability == pytest.approx(
        float(exact.activation), rel=1e-12, abs=1e-15
    )
    assert result.monitoring_withheld_probability == pytest.approx(
        float(exact.withheld), rel=1e-12, abs=1e-15
    )
    for actual, expected in (
        (result.marginal_family_false_alarm, exact.marginal_null),
        (result.marginal_family_detection_probability, exact.marginal_alternative),
    ):
        assert actual == pytest.approx(float(expected), rel=1e-12, abs=1e-15)
    assert exact.null_given_activation is not None
    assert exact.alternative_given_activation is not None
    assert result.conditional_family_false_alarm_given_activation == pytest.approx(
        float(exact.null_given_activation), rel=1e-12, abs=1e-15
    )
    assert result.conditional_family_detection_probability_given_activation == (
        pytest.approx(float(exact.alternative_given_activation), rel=1e-12, abs=1e-15)
    )


def test_frozen_policy_full_distribution_matches_exact_fraction_oracle() -> None:
    exact = _independent_frozen_policy_oracle(_MARGINS)
    result = evaluate_frozen_reference_policy(_MARGINS)
    _assert_policy_matches_independent_oracle(result, exact)
    assert result.monitoring_activation_probability == pytest.approx(
        0.99853784920463662, rel=1e-12, abs=1e-15
    )
    assert result.monitoring_withheld_probability == pytest.approx(
        0.00146215079536338, rel=1e-12, abs=1e-15
    )
    assert result.marginal_family_false_alarm == pytest.approx(
        0.011949878124584378, rel=1e-12, abs=1e-15
    )
    assert result.marginal_family_detection_probability == pytest.approx(
        0.20653292295559833, rel=1e-12, abs=1e-15
    )
    assert result.conditional_family_false_alarm_given_activation == pytest.approx(
        0.011967376233262255, rel=1e-12, abs=1e-15
    )
    assert result.conditional_family_detection_probability_given_activation == (
        pytest.approx(0.20683534742334264, rel=1e-12, abs=1e-15)
    )
    assert math.fsum(
        row.probability_mass for row in result.decision_probability_masses
    ) == pytest.approx(1.0, abs=1e-14)
    assert (
        result.monitoring_activation_probability
        + result.monitoring_withheld_probability
        == (pytest.approx(1.0, abs=1e-14))
    )
    # This mapping is independently confirmed for ALL b, not guessed by the implementation.
    assert exact.margins_by_b == (
        *(None for _ in range(2)),
        *(16 for _ in range(2)),
        *(14 for _ in range(2)),
        *(12 for _ in range(2)),
        *(10 for _ in range(33)),
    )


@pytest.mark.parametrize("b", range(41))
def test_frozen_reference_selection_is_minimal_and_conditionally_safe(b: int) -> None:
    exact = _independent_frozen_policy_oracle(_MARGINS)
    selection = select_margin_for_frozen_reference(b, _MARGINS)
    assert selection.selected_margin_40ths == exact.margins_by_b[b]
    assert selection.reference_errors == b
    if selection.status is SelectionStatus.NO_METHOD_PASSES:
        assert b in (0, 1)
        assert selection.selected_margin_40ths is None
        assert selection.conditional_family_false_alarm is None
        return
    margin = selection.selected_margin_40ths
    assert margin is not None
    null = _rational_binomial_pmf(20)
    q = sum((weight for k, weight in enumerate(null) if 2 * k > b + margin), Fraction())
    six_looks = 1 - (1 - q) ** 6
    assert six_looks <= Fraction(1, 20)
    assert selection.conditional_family_false_alarm == pytest.approx(
        float(six_looks), abs=1e-15
    )
    for earlier_margin in _MARGINS:
        if earlier_margin >= margin:
            break
        q_earlier = sum(
            (weight for k, weight in enumerate(null) if 2 * k > b + earlier_margin),
            Fraction(),
        )
        assert 1 - (1 - q_earlier) ** 6 > Fraction(1, 20)


def test_frozen_policy_reuses_one_reference_fit_for_all_six_looks() -> None:
    """The finite-reference draw is integrated outside the nonlinear horizon."""
    exact = _independent_frozen_policy_oracle(_MARGINS)
    observed = evaluate_frozen_reference_policy(_MARGINS)
    # The fitted margin is a single scalar chosen before any of the six weeks.
    for b in (2, 6, 8, 12, 40):
        decision = observed.selections_by_reference_count[b]
        assert decision.selected_margin_40ths == exact.margins_by_b[b]
        expected_null = exact.conditional_null_by_b[b]
        assert expected_null is not None
        assert decision.conditional_family_false_alarm == pytest.approx(
            float(expected_null), abs=1e-15
        )
    assert exact.wrong_order_null != exact.marginal_null
    assert exact.wrong_order_alternative != exact.marginal_alternative
    assert observed.marginal_family_false_alarm != pytest.approx(
        float(exact.wrong_order_null), abs=1e-12
    )
    assert observed.marginal_family_detection_probability != pytest.approx(
        float(exact.wrong_order_alternative), abs=1e-12
    )


def test_policy_withheld_paths_are_separate_from_enabled_detector_performance() -> None:
    result = evaluate_frozen_reference_policy(_MARGINS)
    assert all(
        result.selections_by_reference_count[b].status
        is SelectionStatus.NO_METHOD_PASSES
        for b in (0, 1)
    )
    assert result.monitoring_withheld_probability > 0.0
    assert result.marginal_family_false_alarm > 0.0
    assert result.marginal_family_detection_probability > 0.0
    assert result.conditional_family_false_alarm_given_activation is not None
    assert result.conditional_family_detection_probability_given_activation is not None
    assert result.conditional_family_false_alarm_given_activation == pytest.approx(
        result.marginal_family_false_alarm / result.monitoring_activation_probability
    )
    assert result.conditional_family_detection_probability_given_activation == (
        pytest.approx(
            result.marginal_family_detection_probability
            / result.monitoring_activation_probability
        )
    )


def test_policy_strict_alarm_ties_are_excluded() -> None:
    result = evaluate_frozen_reference_policy(_MARGINS)
    weekly_null = _rational_binomial_pmf(20)
    weekly_alternative = _rational_alternative_weekly_pmf()
    for b, selection in enumerate(result.selections_by_reference_count):
        margin = selection.selected_margin_40ths
        if margin is None or (b + margin) % 2:
            continue
        tie = (b + margin) // 2
        if not 0 <= tie <= 20:
            continue
        assert _strict_alarm(b, tie, margin) is False
        for weekly in (weekly_null, weekly_alternative):
            excluded = sum((weekly[k] for k in range(tie + 1, 21)), Fraction())
            included = excluded + weekly[tie]
            assert included > excluded
        assert selection.conditional_family_false_alarm == pytest.approx(
            float(
                1
                - (1 - sum((weekly_null[k] for k in range(tie + 1, 21)), Fraction()))
                ** 6
            ),
            abs=1e-15,
        )


@pytest.mark.parametrize("invalid", (-1, 41, True, 3.0, "3", None))
def test_frozen_reference_selection_rejects_invalid_reference(invalid: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        select_margin_for_frozen_reference(invalid, _MARGINS)  # type: ignore[arg-type]


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
        "10",
        b"10",
        None,
        iter((10, 12)),
    ),
)
def test_frozen_policy_rejects_invalid_candidate_sequences(invalid: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        select_margin_for_frozen_reference(7, invalid)  # type: ignore[arg-type]
    with pytest.raises((ValueError, TypeError)):
        evaluate_frozen_reference_policy(invalid)  # type: ignore[arg-type]


def test_frozen_reference_results_are_immutable() -> None:
    result = evaluate_frozen_reference_policy(_MARGINS)
    assert isinstance(result.selections_by_reference_count, tuple)
    assert isinstance(result.decision_probability_masses, tuple)
    with pytest.raises(FrozenInstanceError):
        result.monitoring_activation_probability = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.selections_by_reference_count[0].selected_margin_40ths = 10  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.decision_probability_masses[0].probability_mass = 0  # type: ignore[misc]


def test_frozen_policy_zero_activation_is_explicitly_not_estimable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Under the specified model B=40 always passes, so zero activation is
    # unreachable naturally; force it in memory to test the defensive branch.
    def withhold_all(
        b: int, candidates: tuple[int, ...]
    ) -> FrozenReferenceMarginSelection:
        return FrozenReferenceMarginSelection(
            b, SelectionStatus.NO_METHOD_PASSES, None, None
        )

    with monkeypatch.context() as context:
        context.setattr(
            policy_module, "select_margin_for_frozen_reference", withhold_all
        )
        result = evaluate_frozen_reference_policy(_MARGINS)
    assert result.monitoring_activation_probability == 0.0
    assert result.monitoring_withheld_probability == pytest.approx(1.0)
    assert result.marginal_family_false_alarm == 0.0
    assert result.marginal_family_detection_probability == 0.0
    assert result.conditional_family_false_alarm_given_activation is None
    assert result.conditional_family_detection_probability_given_activation is None
    assert all(
        decision.status is SelectionStatus.NO_METHOD_PASSES
        for decision in result.selections_by_reference_count
    )


def test_frozen_policy_in_memory_non_strict_comparator_mutant_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = _independent_frozen_policy_oracle(_MARGINS)
    # Clear existing memoized conditionals so the injected comparator is
    # actually exercised; do not edit/revert any repository source files.
    try:
        with monkeypatch.context() as context:
            context.setattr(
                policy_module,
                "_strict_alarm",
                lambda b, k, margin: 2 * k >= b + margin,
            )
            policy_module.evaluate_margin.cache_clear()
            policy_module.evaluate_detection_power.cache_clear()
            mutated = evaluate_frozen_reference_policy(_MARGINS)
            with pytest.raises(AssertionError):
                _assert_policy_matches_independent_oracle(mutated, exact)
    finally:
        policy_module.evaluate_margin.cache_clear()
        policy_module.evaluate_detection_power.cache_clear()


def test_frozen_policy_in_memory_global_margin_mutant_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = _independent_frozen_policy_oracle(_MARGINS)

    def globally_select_16(
        b: int, candidates: tuple[int, ...]
    ) -> FrozenReferenceMarginSelection:
        return FrozenReferenceMarginSelection(
            reference_errors=b,
            status=SelectionStatus.SELECTED,
            selected_margin_40ths=16,
            conditional_family_false_alarm=evaluate_margin(
                16
            ).conditional_family_false_alarm_by_reference_count[b],
        )

    with monkeypatch.context() as context:
        context.setattr(
            policy_module, "select_margin_for_frozen_reference", globally_select_16
        )
        mutated = evaluate_frozen_reference_policy(_MARGINS)
        with pytest.raises(AssertionError):
            _assert_policy_matches_independent_oracle(mutated, exact)


def test_frozen_policy_in_memory_averaging_first_mutant_rejected() -> None:
    correct = evaluate_frozen_reference_policy(_MARGINS)
    exact = _independent_frozen_policy_oracle(_MARGINS)
    assert abs(float(exact.wrong_order_null - exact.marginal_null)) > 1e-5
    assert abs(float(exact.wrong_order_alternative - exact.marginal_alternative)) > 1e-5
    mutant = replace(
        correct,
        marginal_family_false_alarm=float(exact.wrong_order_null),
        marginal_family_detection_probability=float(exact.wrong_order_alternative),
    )
    with pytest.raises(AssertionError):
        _assert_policy_matches_independent_oracle(mutant, exact)


def test_frozen_policy_in_memory_wrong_activation_normalization_mutant_rejected() -> (
    None
):
    correct = evaluate_frozen_reference_policy(_MARGINS)
    exact = _independent_frozen_policy_oracle(_MARGINS)
    # A common mutant reports unconditional probabilities as conditional.
    mutant = replace(
        correct,
        conditional_family_false_alarm_given_activation=correct.marginal_family_false_alarm,
        conditional_family_detection_probability_given_activation=(
            correct.marginal_family_detection_probability
        ),
    )
    with pytest.raises(AssertionError):
        _assert_policy_matches_independent_oracle(mutant, exact)
