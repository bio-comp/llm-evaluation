"""Exact-rational controls for D-019 plug-in versus known-null reference policies.

No test treats an LLM judge or the production floating-point implementation as
an oracle; every expected probability is independently enumerated with Fraction.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, dataclass, replace
from fractions import Fraction
from functools import lru_cache

import pytest

from llm_evaluation import fitted_monitor_estimation_sensitivity as sensitivity
from llm_evaluation.fitted_monitor_diagnostic import (
    SelectionStatus,
    evaluate_frozen_reference_policy,
)
from llm_evaluation.fitted_monitor_estimation_sensitivity import (
    EstimatedNullSensitivityEvaluation,
    evaluate_estimated_null_sensitivity,
)

_CANDIDATES = (10, 12, 14, 16)
_NULL = Fraction(1, 5)
_ALTERNATIVE = Fraction(3, 10)
_LIMIT = Fraction(1, 20)


@lru_cache(maxsize=None)
def _fraction_binomial(n: int, p: Fraction) -> tuple[Fraction, ...]:
    """Independent exact binomial distribution, including p=0 and p=1."""
    return tuple(
        Fraction(math.comb(n, k)) * p**k * (1 - p) ** (n - k) for k in range(n + 1)
    )


def _fraction_weekly_alarm(b: int, margin: int, p: Fraction) -> Fraction:
    return sum(
        (
            mass
            for k, mass in enumerate(_fraction_binomial(20, p))
            if 2 * k > b + margin
        ),
        Fraction(),
    )


def _fraction_family_alarm(b: int, margin: int, p: Fraction) -> Fraction:
    q = _fraction_weekly_alarm(b, margin, p)
    return 1 - (1 - q) ** 6


@dataclass(frozen=True)
class _ExactPolicy:
    selections: tuple[int | None, ...]
    estimated: tuple[Fraction | None, ...]
    actual_null: tuple[Fraction | None, ...]
    actual_alternative: tuple[Fraction | None, ...]
    violation: tuple[bool | None, ...]
    decision_masses: tuple[Fraction, ...]
    activation: Fraction
    withheld: Fraction
    marginal_null: Fraction
    marginal_alternative: Fraction
    violating_mass: Fraction
    violating_count: int
    wrong_averaging_first_null: Fraction
    wrong_averaging_first_alternative: Fraction


@lru_cache(maxsize=None)
def _exact_plug_in_oracle(candidates: tuple[int, ...]) -> _ExactPolicy:
    """Separate rational selection and realized-risk calculation for every b."""
    reference = _fraction_binomial(40, _NULL)
    chosen: list[int | None] = []
    estimated_risk: list[Fraction | None] = []
    actual_null: list[Fraction | None] = []
    alternative: list[Fraction | None] = []
    violated: list[bool | None] = []
    weekly_null = []
    weekly_alternative = []

    for b in range(41):
        selected: int | None = None
        estimated: Fraction | None = None
        for m in candidates:
            estimate = _fraction_family_alarm(b, m, Fraction(b, 40))
            if estimate <= _LIMIT:
                selected, estimated = m, estimate
                break
        chosen.append(selected)
        estimated_risk.append(estimated)
        if selected is None:
            actual_null.append(None)
            alternative.append(None)
            violated.append(None)
            weekly_null.append(Fraction())
            weekly_alternative.append(Fraction())
            continue
        true_weekly = _fraction_weekly_alarm(b, selected, _NULL)
        alternative_weekly = _fraction_weekly_alarm(b, selected, _ALTERNATIVE)
        actual_horizon = 1 - (1 - true_weekly) ** 6
        actual_null.append(actual_horizon)
        alternative.append(1 - (1 - alternative_weekly) ** 6)
        violated.append(actual_horizon > _LIMIT)
        weekly_null.append(true_weekly)
        weekly_alternative.append(alternative_weekly)

    masses = tuple(
        sum(
            (reference[b] for b in range(41) if chosen[b] == margin),
            Fraction(),
        )
        for margin in (None, *candidates)
    )
    marginal_null = sum(
        (reference[b] * val for b, val in enumerate(actual_null) if val is not None),
        Fraction(),
    )
    marginal_alt = sum(
        (reference[b] * val for b, val in enumerate(alternative) if val is not None),
        Fraction(),
    )
    mean_weekly_null = sum(
        (weight * q for weight, q in zip(reference, weekly_null)), Fraction()
    )
    mean_weekly_alt = sum(
        (weight * q for weight, q in zip(reference, weekly_alternative)), Fraction()
    )
    return _ExactPolicy(
        selections=tuple(chosen),
        estimated=tuple(estimated_risk),
        actual_null=tuple(actual_null),
        actual_alternative=tuple(alternative),
        violation=tuple(violated),
        decision_masses=masses,
        activation=sum(masses[1:], Fraction()),
        withheld=masses[0],
        marginal_null=marginal_null,
        marginal_alternative=marginal_alt,
        violating_mass=sum(
            (reference[b] for b, flag in enumerate(violated) if flag is True),
            Fraction(),
        ),
        violating_count=sum(flag is True for flag in violated),
        wrong_averaging_first_null=1 - (1 - mean_weekly_null) ** 6,
        wrong_averaging_first_alternative=1 - (1 - mean_weekly_alt) ** 6,
    )


def _assert_matches_oracle(
    observed: EstimatedNullSensitivityEvaluation,
    exact: _ExactPolicy,
    candidates: tuple[int, ...],
) -> None:
    """Comparison seam reused against fault-injected result records and code."""
    assert observed.candidate_margins_40ths == candidates
    assert len(observed.decisions_by_reference_count) == 41
    assert len(observed.decision_probability_masses) == len(candidates) + 1
    for b, decision in enumerate(observed.decisions_by_reference_count):
        assert decision.reference_errors == b
        assert decision.plug_in_null_probability == pytest.approx(
            float(Fraction(b, 40)), abs=1e-15
        )
        assert decision.selected_margin_40ths == exact.selections[b]
        assert decision.status is (
            SelectionStatus.SELECTED
            if exact.selections[b] is not None
            else SelectionStatus.NO_METHOD_PASSES
        )
        for field, expected in (
            ("estimated_conditional_null_family_alarm", exact.estimated[b]),
            ("actual_conditional_null_family_alarm", exact.actual_null[b]),
            ("actual_conditional_alternative_detection", exact.actual_alternative[b]),
        ):
            actual = getattr(decision, field)
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(float(expected), rel=1e-11, abs=1e-15)
        assert decision.actual_conditional_null_limit_violated is exact.violation[b]

    for i, (decision_mass, exact_mass) in enumerate(
        zip(observed.decision_probability_masses, exact.decision_masses)
    ):
        expected_margin = (None, *candidates)[i]
        assert decision_mass.selected_margin_40ths == expected_margin
        assert decision_mass.status is (
            SelectionStatus.SELECTED
            if expected_margin is not None
            else SelectionStatus.NO_METHOD_PASSES
        )
        assert decision_mass.probability_mass == pytest.approx(
            float(exact_mass), rel=1e-12, abs=1e-15
        )
    for field, expected in (
        ("monitoring_activation_probability", exact.activation),
        ("monitoring_withheld_probability", exact.withheld),
        ("marginal_actual_null_family_alarm", exact.marginal_null),
        ("marginal_actual_alternative_detection", exact.marginal_alternative),
        ("violating_reference_probability_mass", exact.violating_mass),
    ):
        assert getattr(observed, field) == pytest.approx(
            float(expected), rel=1e-12, abs=1e-15
        )
    assert observed.violating_reference_condition_count == exact.violating_count
    assert math.fsum(
        row.probability_mass for row in observed.decision_probability_masses
    ) == (pytest.approx(1.0, abs=1e-14))


def test_all_41_conditions_and_marginal_results_match_independent_fraction_oracle() -> (
    None
):
    exact = _exact_plug_in_oracle(_CANDIDATES)
    observed = evaluate_estimated_null_sensitivity(_CANDIDATES)
    _assert_matches_oracle(observed, exact, _CANDIDATES)
    assert exact.activation == 1
    assert exact.withheld == 0
    assert exact.violating_count == 8
    assert tuple(b for b, flag in enumerate(exact.violation) if flag) == tuple(range(8))
    assert exact.marginal_null > _LIMIT
    assert observed.marginal_actual_null_family_alarm == pytest.approx(
        0.057155206374, abs=1e-12
    )
    assert observed.marginal_actual_alternative_detection == pytest.approx(
        0.363258210700, abs=1e-12
    )
    assert observed.violating_reference_probability_mass == pytest.approx(
        0.437145897987, abs=1e-12
    )
    assert observed.monitoring_activation_probability == 1.0
    assert observed.monitoring_withheld_probability == 0.0


@pytest.mark.parametrize("b", range(41))
def test_reference_condition_selection_uses_own_frozen_plug_in_probability(
    b: int,
) -> None:
    exact = _exact_plug_in_oracle(_CANDIDATES)
    observed = evaluate_estimated_null_sensitivity(_CANDIDATES)
    row = observed.decisions_by_reference_count[b]
    assert row.selected_margin_40ths == exact.selections[b]
    assert row.plug_in_null_probability == pytest.approx(b / 40, abs=0.0)
    if row.selected_margin_40ths is None:
        assert row.status is SelectionStatus.NO_METHOD_PASSES
        return
    expected_estimated = exact.estimated[b]
    assert expected_estimated is not None
    assert row.estimated_conditional_null_family_alarm == pytest.approx(
        float(expected_estimated), abs=1e-15
    )
    assert expected_estimated <= _LIMIT
    for smaller_margin in _CANDIDATES:
        if smaller_margin >= row.selected_margin_40ths:
            break
        assert _fraction_family_alarm(b, smaller_margin, Fraction(b, 40)) > _LIMIT


def test_zero_fitted_rate_reports_zero_estimated_but_large_actual_null_risk() -> None:
    row = evaluate_estimated_null_sensitivity(_CANDIDATES).decisions_by_reference_count[
        0
    ]
    assert row.plug_in_null_probability == 0.0
    assert row.selected_margin_40ths == 10
    assert row.estimated_conditional_null_family_alarm == 0.0
    assert row.actual_conditional_null_family_alarm == pytest.approx(
        0.729473608923, abs=1e-12
    )
    assert row.actual_conditional_null_limit_violated is True


def test_strict_comparison_excludes_each_reachable_threshold_tie() -> None:
    rows = evaluate_estimated_null_sensitivity(_CANDIDATES).decisions_by_reference_count
    examined = 0
    for row in rows:
        margin = row.selected_margin_40ths
        if margin is None or (row.reference_errors + margin) % 2:
            continue
        k = (row.reference_errors + margin) // 2
        if not 0 <= k <= 20:
            continue
        examined += 1
        assert sensitivity._strict_alarm(row.reference_errors, k, margin) is False
        fitted = _fraction_binomial(20, Fraction(row.reference_errors, 40))
        excluded = sum((fitted[j] for j in range(k + 1, 21)), Fraction())
        included = excluded + fitted[k]
        if 0 < row.plug_in_null_probability < 1:
            assert included > excluded
        else:
            assert included >= excluded
        expected = 1 - (1 - excluded) ** 6
        assert row.estimated_conditional_null_family_alarm == pytest.approx(
            float(expected), abs=1e-15
        )
    assert examined >= 10


def test_frozen_reference_is_integrated_only_after_six_look_transformation() -> None:
    exact = _exact_plug_in_oracle(_CANDIDATES)
    observed = evaluate_estimated_null_sensitivity(_CANDIDATES)
    assert abs(float(exact.marginal_null - exact.wrong_averaging_first_null)) > 1e-5
    assert (
        abs(float(exact.marginal_alternative - exact.wrong_averaging_first_alternative))
        > 1e-5
    )
    assert observed.marginal_actual_null_family_alarm != pytest.approx(
        float(exact.wrong_averaging_first_null), abs=1e-12
    )
    assert observed.marginal_actual_alternative_detection != pytest.approx(
        float(exact.wrong_averaging_first_alternative), abs=1e-12
    )


def test_nonactivation_is_typed_and_accounted_for_on_constrained_grid() -> None:
    candidates = (0,)
    exact = _exact_plug_in_oracle(candidates)
    observed = evaluate_estimated_null_sensitivity(candidates)
    _assert_matches_oracle(observed, exact, candidates)
    assert 0 < observed.monitoring_withheld_probability < 1
    assert observed.monitoring_activation_probability < 1
    assert any(
        row.status is SelectionStatus.NO_METHOD_PASSES
        and row.selected_margin_40ths is None
        and row.estimated_conditional_null_family_alarm is None
        and row.actual_conditional_null_family_alarm is None
        and row.actual_conditional_alternative_detection is None
        and row.actual_conditional_null_limit_violated is None
        for row in observed.decisions_by_reference_count
    )


def test_original_known_null_policy_exact_references_unchanged() -> None:
    existing = evaluate_frozen_reference_policy(_CANDIDATES)
    assert existing.monitoring_activation_probability == pytest.approx(
        0.99853784920463662, abs=1e-13
    )
    assert existing.monitoring_withheld_probability == pytest.approx(
        0.00146215079536338, abs=1e-13
    )
    assert existing.marginal_family_false_alarm == pytest.approx(
        0.011949878124584378, abs=1e-13
    )
    assert existing.marginal_family_detection_probability == pytest.approx(
        0.20653292295559833, abs=1e-13
    )
    assert existing.selections_by_reference_count[0].status is (
        SelectionStatus.NO_METHOD_PASSES
    )
    assert existing.selections_by_reference_count[2].selected_margin_40ths == 16
    assert existing.selections_by_reference_count[8].selected_margin_40ths == 10


@pytest.mark.parametrize(
    "bad",
    (
        (),
        (10, 10),
        (12, 10),
        (10, -1),
        (True, 10),
        (10, 12.0),
        ("10", 12),
        "10",
        b"10",
        None,
        iter((10, 12)),
    ),
)
def test_invalid_candidate_grid_rejected(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_estimated_null_sensitivity(bad)  # type: ignore[arg-type]


def test_immutable_results() -> None:
    result = evaluate_estimated_null_sensitivity(_CANDIDATES)
    assert isinstance(result.decisions_by_reference_count, tuple)
    assert isinstance(result.decision_probability_masses, tuple)
    with pytest.raises(FrozenInstanceError):
        result.monitoring_activation_probability = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.decisions_by_reference_count[0].selected_margin_40ths = 12  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.decision_probability_masses[0].probability_mass = 0  # type: ignore[misc]


def test_mutant_selection_uses_true_null_instead_of_fitted_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = _exact_plug_in_oracle(_CANDIDATES)
    original = sensitivity._binomial_pmf_at_rate
    with monkeypatch.context() as context:
        context.setattr(
            sensitivity,
            "_binomial_pmf_at_rate",
            lambda n, p: original(n, 0.20),
        )
        mutated = evaluate_estimated_null_sensitivity(_CANDIDATES)
        with pytest.raises(AssertionError):
            _assert_matches_oracle(mutated, exact, _CANDIDATES)
    assert mutated.decisions_by_reference_count[0].selected_margin_40ths is None


def test_mutant_realized_null_uses_fitted_rate_instead_of_true_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = _exact_plug_in_oracle(_CANDIDATES)
    with monkeypatch.context() as context:
        context.setattr(
            sensitivity,
            "_actual_null_family_alarm",
            lambda b, m: sensitivity._conditional_six_look_event(
                b, m, sensitivity._binomial_pmf_at_rate(20, b / 40)
            ),
        )
        mutated = evaluate_estimated_null_sensitivity(_CANDIDATES)
        with pytest.raises(AssertionError):
            _assert_matches_oracle(mutated, exact, _CANDIDATES)
    assert (
        mutated.decisions_by_reference_count[0].actual_conditional_null_family_alarm
        == 0
    )


def test_mutant_averages_weekly_risks_before_six_look_transform() -> None:
    exact = _exact_plug_in_oracle(_CANDIDATES)
    correct = evaluate_estimated_null_sensitivity(_CANDIDATES)
    mutant = replace(
        correct,
        marginal_actual_null_family_alarm=float(exact.wrong_averaging_first_null),
        marginal_actual_alternative_detection=float(
            exact.wrong_averaging_first_alternative
        ),
    )
    with pytest.raises(AssertionError):
        _assert_matches_oracle(mutant, exact, _CANDIDATES)


def test_mutant_non_strict_greater_equal_comparator_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = _exact_plug_in_oracle(_CANDIDATES)
    with monkeypatch.context() as context:
        context.setattr(
            sensitivity,
            "_strict_alarm",
            lambda b, k, m: 2 * k >= b + m,
        )
        mutated = evaluate_estimated_null_sensitivity(_CANDIDATES)
        with pytest.raises(AssertionError):
            _assert_matches_oracle(mutated, exact, _CANDIDATES)
