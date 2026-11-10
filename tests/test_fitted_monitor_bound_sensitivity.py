"""Independent exact-rational tests of D-019 finite-reference bound selection.

The oracle computes grid bounds by exhaustive ascending-grid enumeration,
not the production binary-search inversion, and computes strict weekly
tail events by an integer threshold rather than calling production helpers.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, dataclass, replace
from fractions import Fraction
from functools import lru_cache

import pytest

from llm_evaluation import fitted_monitor_bound_sensitivity as bound
from llm_evaluation.fitted_monitor_bound_sensitivity import (
    FiniteReferenceBoundEvaluation,
    evaluate_finite_reference_bound_policy,
)
from llm_evaluation.fitted_monitor_diagnostic import (
    SelectionStatus,
    evaluate_frozen_reference_policy,
)
from llm_evaluation.fitted_monitor_estimation_sensitivity import (
    evaluate_estimated_null_sensitivity,
)

_CANDIDATES = (10, 12, 14, 16)
_NULL = Fraction(1, 5)
_ALTERNATIVE = Fraction(3, 10)
_LIMIT = Fraction(1, 20)
_GRID = 1000


@lru_cache(maxsize=512)
def _independent_binomial(n: int, p: Fraction) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(math.comb(n, k)) * p**k * (1 - p) ** (n - k) for k in range(n + 1)
    )


def _independent_cdf(b: int, p: Fraction) -> Fraction:
    return sum(_independent_binomial(40, p)[: b + 1], Fraction())


@lru_cache(maxsize=1)
def _oracle_bounds() -> tuple[Fraction, ...]:
    """Exhaustively scan grid values in ascending order for all b."""
    first: list[Fraction | None] = [None] * 40
    for index in range(_GRID + 1):
        p = Fraction(index, _GRID)
        cumulative = Fraction()
        for b, mass in enumerate(_independent_binomial(40, p)[:-1]):
            cumulative += mass
            if first[b] is None and cumulative <= _LIMIT:
                first[b] = p
        if all(value is not None for value in first):
            break
    assert all(value is not None for value in first)
    return (*tuple(value for value in first if value is not None), Fraction(1))


def _independent_weekly_event(b: int, margin: int, p: Fraction) -> Fraction:
    """Strict 2K>b+m is K>=floor((b+m)/2)+1; no production predicate."""
    first_alarm_k = (b + margin) // 2 + 1
    return sum(_independent_binomial(20, p)[first_alarm_k:], Fraction())


def _independent_family_event(b: int, margin: int, p: Fraction) -> Fraction:
    weekly_no_alarm = 1 - _independent_weekly_event(b, margin, p)
    return 1 - weekly_no_alarm**6


@dataclass(frozen=True)
class _Oracle:
    bounds: tuple[Fraction, ...]
    selections: tuple[int | None, ...]
    selected_bound_risks: tuple[Fraction | None, ...]
    actual_null_risks: tuple[Fraction | None, ...]
    actual_alternative_risks: tuple[Fraction | None, ...]
    violations: tuple[bool | None, ...]
    miscovered: tuple[bool, ...]
    masses: tuple[Fraction, ...]
    activated: Fraction
    withheld: Fraction
    marginal_null: Fraction
    marginal_alternative: Fraction
    miscoverage: Fraction
    violating_mass: Fraction
    violating_count: int
    wrong_average_first_null: Fraction


@lru_cache(maxsize=128)
def _independent_oracle(
    candidates: tuple[int, ...],
    p0: Fraction,
    p1: Fraction,
) -> _Oracle:
    reference = _independent_binomial(40, p0)
    bounds = _oracle_bounds()
    selections: list[int | None] = []
    selected_bound_risks: list[Fraction | None] = []
    actual_null_risks: list[Fraction | None] = []
    actual_alt_risks: list[Fraction | None] = []
    violations: list[bool | None] = []
    miscovered: list[bool] = []
    weekly_actual_null: list[Fraction] = []
    for b, upper in enumerate(bounds):
        margin = next(
            (
                candidate
                for candidate in candidates
                if _independent_family_event(b, candidate, upper) <= _LIMIT
            ),
            None,
        )
        selections.append(margin)
        miscovered.append(p0 > upper)
        if margin is None:
            selected_bound_risks.append(None)
            actual_null_risks.append(None)
            actual_alt_risks.append(None)
            violations.append(None)
            weekly_actual_null.append(Fraction())
            continue
        selected_bound_risks.append(_independent_family_event(b, margin, upper))
        actual_null = _independent_family_event(b, margin, p0)
        actual_null_risks.append(actual_null)
        actual_alt_risks.append(_independent_family_event(b, margin, p1))
        violations.append(actual_null > _LIMIT)
        weekly_actual_null.append(_independent_weekly_event(b, margin, p0))

    masses = tuple(
        sum(
            (reference[b] for b, selected in enumerate(selections) if selected == m),
            Fraction(),
        )
        for m in (None, *candidates)
    )
    marginal_null = sum(
        (
            reference[b] * risk
            for b, risk in enumerate(actual_null_risks)
            if risk is not None
        ),
        Fraction(),
    )
    marginal_alternative = sum(
        (
            reference[b] * risk
            for b, risk in enumerate(actual_alt_risks)
            if risk is not None
        ),
        Fraction(),
    )
    mean_weekly_null = sum(
        (weight * q for weight, q in zip(reference, weekly_actual_null, strict=True)),
        Fraction(),
    )
    return _Oracle(
        bounds=bounds,
        selections=tuple(selections),
        selected_bound_risks=tuple(selected_bound_risks),
        actual_null_risks=tuple(actual_null_risks),
        actual_alternative_risks=tuple(actual_alt_risks),
        violations=tuple(violations),
        miscovered=tuple(miscovered),
        masses=masses,
        activated=sum(masses[1:], Fraction()),
        withheld=masses[0],
        marginal_null=marginal_null,
        marginal_alternative=marginal_alternative,
        miscoverage=sum(
            (reference[b] for b, miss in enumerate(miscovered) if miss), Fraction()
        ),
        violating_mass=sum(
            (
                reference[b]
                for b, violation in enumerate(violations)
                if violation is True
            ),
            Fraction(),
        ),
        violating_count=sum(violation is True for violation in violations),
        wrong_average_first_null=1 - (1 - mean_weekly_null) ** 6,
    )


def _assert_matches_independent_oracle(
    result: FiniteReferenceBoundEvaluation, oracle: _Oracle, p0: Fraction, p1: Fraction
) -> None:
    """Single oracle-comparison seam exercised against injected wrong results."""
    assert result.candidate_margins_40ths == _CANDIDATES
    assert result.true_null_probability == p0
    assert result.alternative_probability == p1
    assert len(result.decisions_by_reference_count) == 41
    assert len(result.decision_probability_masses) == 5
    for b, row in enumerate(result.decisions_by_reference_count):
        assert row.reference_errors == b
        assert row.upper_null_probability == oracle.bounds[b]
        assert row.selected_margin_40ths == oracle.selections[b]
        assert row.status is (
            SelectionStatus.SELECTED
            if oracle.selections[b] is not None
            else SelectionStatus.NO_METHOD_PASSES
        )
        assert row.bound_conditional_family_alarm == oracle.selected_bound_risks[b]
        assert row.actual_conditional_null_family_alarm == oracle.actual_null_risks[b]
        assert (
            row.actual_conditional_alternative_detection
            == oracle.actual_alternative_risks[b]
        )
        assert row.actual_conditional_null_limit_violated is oracle.violations[b]
        assert row.bound_miscovered is oracle.miscovered[b]
        if row.status is SelectionStatus.SELECTED:
            assert row.bound_conditional_family_alarm is not None
            assert row.bound_conditional_family_alarm <= _LIMIT
            assert row.selected_margin_40ths is not None
            for earlier in _CANDIDATES:
                if earlier >= row.selected_margin_40ths:
                    break
                assert _independent_family_event(b, earlier, oracle.bounds[b]) > _LIMIT
        else:
            assert all(
                _independent_family_event(b, margin, oracle.bounds[b]) > _LIMIT
                for margin in _CANDIDATES
            )
    for mass, margin, expected in zip(
        result.decision_probability_masses,
        (None, *_CANDIDATES),
        oracle.masses,
        strict=True,
    ):
        assert mass.selected_margin_40ths == margin
        assert mass.status is (
            SelectionStatus.NO_METHOD_PASSES
            if margin is None
            else SelectionStatus.SELECTED
        )
        assert mass.probability_mass == expected
    assert result.monitoring_activation_probability == oracle.activated
    assert result.monitoring_withheld_probability == oracle.withheld
    assert result.marginal_actual_null_family_alarm == oracle.marginal_null
    assert result.marginal_actual_alternative_detection == oracle.marginal_alternative
    assert result.reference_bound_miscoverage_probability == oracle.miscoverage
    assert (
        result.activated_conditional_risk_violation_probability == oracle.violating_mass
    )
    assert result.violating_reference_condition_count == oracle.violating_count
    assert sum(oracle.masses, Fraction()) == 1
    assert (
        result.monitoring_activation_probability
        + result.monitoring_withheld_probability
        == 1
    )
    assert (
        sum(
            (mass.probability_mass for mass in result.decision_probability_masses),
            Fraction(),
        )
        == 1
    )


@pytest.mark.parametrize("b", range(41))
def test_upper_bound_inversion_is_exact_grid_minimum_for_every_b(b: int) -> None:
    bounds = _oracle_bounds()
    u = bound._upper_confidence_bound(b)
    assert u == bounds[b]
    assert 0 <= u <= 1
    assert u * _GRID == int(u * _GRID)
    if b == 40:
        assert u == 1
    else:
        assert _independent_cdf(b, u) <= _LIMIT
        assert u > 0
        assert _independent_cdf(b, u - Fraction(1, _GRID)) > _LIMIT
    if b:
        assert bounds[b - 1] <= u


@pytest.mark.parametrize(
    ("invalid", "exception_type"),
    (
        (True, TypeError),
        (False, TypeError),
        (0.5, TypeError),
        (-1, ValueError),
        (41, ValueError),
    ),
)
def test_upper_bound_rejects_invalid_reference_counts(
    invalid: object, exception_type: type[Exception]
) -> None:
    # Exercise valid cached integer keys before invalid bool/float inputs.
    assert 0 <= bound._upper_confidence_bound(0) <= 1
    assert 0 <= bound._upper_confidence_bound(1) <= 1
    with pytest.raises(exception_type):
        bound._upper_confidence_bound(invalid)


def test_production_binomial_cache_has_bounded_retention() -> None:
    bound._binomial_pmf.cache_clear()
    try:
        for numerator in range(600):
            p = Fraction(numerator, 601)
            assert bound._binomial_pmf(1, p) == (1 - p, p)
        info = bound._binomial_pmf.cache_info()
        assert info.maxsize == 512
        assert info.currsize == 512
        assert info.misses == 600
        # Oldest value was evicted; recomputation preserves its exact PMF.
        assert bound._binomial_pmf(1, Fraction(0)) == (Fraction(1), Fraction(0))
        assert bound._binomial_pmf.cache_info().misses == 601
    finally:
        bound._binomial_pmf.cache_clear()


def test_primary_condition_all_reference_rows_and_crosschecks() -> None:
    observed = evaluate_finite_reference_bound_policy(_CANDIDATES)
    oracle = _independent_oracle(_CANDIDATES, _NULL, _ALTERNATIVE)
    _assert_matches_independent_oracle(observed, oracle, _NULL, _ALTERNATIVE)
    for value, checkpoint in (
        (observed.monitoring_activation_probability, 0.897922177952824),
        (observed.monitoring_withheld_probability, 0.102077822047176),
        (observed.marginal_actual_null_family_alarm, 0.004225622617954),
        (observed.marginal_actual_alternative_detection, 0.048414169943690),
        (
            observed.activated_conditional_risk_violation_probability,
            0.028462094459745,
        ),
    ):
        assert float(value) == pytest.approx(checkpoint, abs=2e-15)
    assert [
        row.reference_errors
        for row in observed.decisions_by_reference_count
        if row.status is SelectionStatus.NO_METHOD_PASSES
    ] == [11, 13, 15]
    assert observed.activated_conditional_risk_violation_probability <= (
        observed.reference_bound_miscoverage_probability
    )


_CROSS_NULLS = (
    Fraction(0),
    Fraction(1, 20),
    Fraction(1, 10),
    Fraction(3, 20),
    Fraction(199, 1000),
    Fraction(1, 5),
    Fraction(201, 1000),
    Fraction(1, 4),
    Fraction(3, 10),
    Fraction(1, 3),
    Fraction(2, 5),
    Fraction(1, 2),
    Fraction(7, 10),
    Fraction(9, 10),
    Fraction(1),
)


@pytest.mark.parametrize("p0", _CROSS_NULLS)
def test_cross_null_reference_coverage_and_risk_violation_bound(p0: Fraction) -> None:
    observed = evaluate_finite_reference_bound_policy(
        _CANDIDATES, true_null_p=p0, alternative_p=_ALTERNATIVE
    )
    oracle = _independent_oracle(_CANDIDATES, p0, _ALTERNATIVE)
    _assert_matches_independent_oracle(observed, oracle, p0, _ALTERNATIVE)
    assert observed.activated_conditional_risk_violation_probability <= (
        observed.reference_bound_miscoverage_probability
    )
    assert observed.reference_bound_miscoverage_probability <= _LIMIT
    assert 0 <= observed.monitoring_activation_probability <= 1
    assert (
        0
        <= observed.marginal_actual_null_family_alarm
        <= (observed.monitoring_activation_probability)
    )
    for row in observed.decisions_by_reference_count:
        if row.status is SelectionStatus.SELECTED and not row.bound_miscovered:
            assert row.actual_conditional_null_limit_violated is False
            assert row.actual_conditional_null_family_alarm is not None
            assert row.bound_conditional_family_alarm is not None
            assert (
                row.actual_conditional_null_family_alarm
                <= row.bound_conditional_family_alarm
                <= _LIMIT
            )


def test_baseline_regressions_and_activation_power_risk_tradeoff() -> None:
    observed = evaluate_finite_reference_bound_policy(_CANDIDATES)
    plug_in = evaluate_estimated_null_sensitivity(_CANDIDATES)
    known = evaluate_frozen_reference_policy(_CANDIDATES)
    assert plug_in.monitoring_activation_probability == pytest.approx(1.0)
    assert plug_in.marginal_actual_null_family_alarm == pytest.approx(
        0.0571552063736197, abs=1e-14
    )
    assert plug_in.marginal_actual_alternative_detection == pytest.approx(
        0.363258210699692, abs=1e-14
    )
    assert plug_in.violating_reference_probability_mass == pytest.approx(
        0.437145897987410, abs=1e-14
    )
    assert known.monitoring_activation_probability == pytest.approx(
        0.99853784920463662, abs=1e-14
    )
    assert known.marginal_family_false_alarm == pytest.approx(
        0.011949878124584378, abs=1e-14
    )
    assert known.marginal_family_detection_probability == pytest.approx(
        0.20653292295559833, abs=1e-14
    )
    assert float(observed.monitoring_activation_probability) < (
        known.monitoring_activation_probability
    )
    assert float(observed.marginal_actual_null_family_alarm) < (
        known.marginal_family_false_alarm
    )
    assert float(observed.marginal_actual_alternative_detection) < (
        known.marginal_family_detection_probability
    )
    assert float(observed.marginal_actual_alternative_detection) < (
        plug_in.marginal_actual_alternative_detection
    )


def test_strict_alarm_threshold_tie_exclusion_and_six_look_order() -> None:
    oracle = _independent_oracle(_CANDIDATES, _NULL, _ALTERNATIVE)
    observed = evaluate_finite_reference_bound_policy(_CANDIDATES)
    tie_rows = [
        (b, margin)
        for b, margin in enumerate(oracle.selections)
        if margin is not None and (b + margin) % 2 == 0 and 0 <= (b + margin) // 2 <= 20
    ]
    assert tie_rows
    for b, margin in tie_rows:
        tie_k = (b + margin) // 2
        assert bound._strict_alarm(b, tie_k, margin) is False
        pmf = _independent_binomial(20, _NULL)
        strict_tail = sum(pmf[tie_k + 1 :], Fraction())
        non_strict_tail = sum(pmf[tie_k:], Fraction())
        assert non_strict_tail > strict_tail
        assert _independent_family_event(b, margin, _NULL) == (
            1 - (1 - strict_tail) ** 6
        )
    assert observed.marginal_actual_null_family_alarm == oracle.marginal_null
    assert oracle.wrong_average_first_null != oracle.marginal_null


def test_negative_control_naive_plugin_substitution_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _independent_oracle(_CANDIDATES, _NULL, _ALTERNATIVE)
    with monkeypatch.context() as context:
        context.setattr(bound, "_upper_confidence_bound", lambda b: Fraction(b, 40))
        mutant = evaluate_finite_reference_bound_policy(_CANDIDATES)
    with pytest.raises(AssertionError):
        _assert_matches_independent_oracle(mutant, oracle, _NULL, _ALTERNATIVE)


def test_negative_control_wrong_grid_direction_or_minimality_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for b in (0, 5, 17, 39):
        correct = _oracle_bounds()[b]
        index = int(correct * _GRID)
        assert index >= 1
        too_low = Fraction(index - 1, _GRID)
        assert _independent_cdf(b, too_low) > _LIMIT
        if index < _GRID:
            too_high = Fraction(index + 1, _GRID)
            assert _independent_cdf(b, too_high) <= _LIMIT
            assert _independent_cdf(b, too_high - Fraction(1, _GRID)) <= _LIMIT

    oracle = _independent_oracle(_CANDIDATES, _NULL, _ALTERNATIVE)
    correct_bounds = _oracle_bounds()
    for shift in (-1, 1):
        with monkeypatch.context() as context:
            context.setattr(
                bound,
                "_upper_confidence_bound",
                lambda b, delta=shift: (
                    correct_bounds[b] + Fraction(delta, _GRID)
                    if b == 5
                    else correct_bounds[b]
                ),
            )
            mutant = evaluate_finite_reference_bound_policy(_CANDIDATES)
        with pytest.raises(AssertionError):
            _assert_matches_independent_oracle(mutant, oracle, _NULL, _ALTERNATIVE)


def test_negative_control_averaging_weekly_risk_before_horizon_rejected() -> None:
    observed = evaluate_finite_reference_bound_policy(_CANDIDATES)
    oracle = _independent_oracle(_CANDIDATES, _NULL, _ALTERNATIVE)
    mutant = replace(
        observed, marginal_actual_null_family_alarm=oracle.wrong_average_first_null
    )
    with pytest.raises(AssertionError):
        _assert_matches_independent_oracle(mutant, oracle, _NULL, _ALTERNATIVE)


def test_negative_control_non_strict_alarm_at_reachable_tie_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _independent_oracle(_CANDIDATES, _NULL, _ALTERNATIVE)
    with monkeypatch.context() as context:
        context.setattr(
            bound, "_strict_alarm", lambda b, k, margin: 2 * k >= b + margin
        )
        mutant = evaluate_finite_reference_bound_policy(_CANDIDATES)
    with pytest.raises(AssertionError):
        _assert_matches_independent_oracle(mutant, oracle, _NULL, _ALTERNATIVE)


def test_negative_control_withheld_counts_as_detection_rejected() -> None:
    observed = evaluate_finite_reference_bound_policy(_CANDIDATES)
    oracle = _independent_oracle(_CANDIDATES, _NULL, _ALTERNATIVE)
    mutant = replace(
        observed,
        marginal_actual_alternative_detection=(
            observed.marginal_actual_alternative_detection
            + observed.monitoring_withheld_probability
        ),
    )
    with pytest.raises(AssertionError):
        _assert_matches_independent_oracle(mutant, oracle, _NULL, _ALTERNATIVE)


def test_restrictive_grid_nonactivation_stays_explicit() -> None:
    result = evaluate_finite_reference_bound_policy((10,))
    assert result.monitoring_withheld_probability > 0
    assert result.monitoring_activation_probability < 1
    assert (
        result.monitoring_activation_probability
        + result.monitoring_withheld_probability
        == 1
    )
    for decision in result.decisions_by_reference_count:
        if decision.status is SelectionStatus.NO_METHOD_PASSES:
            assert decision.selected_margin_40ths is None
            assert decision.actual_conditional_alternative_detection is None
            assert decision.actual_conditional_null_limit_violated is None


@pytest.mark.parametrize(
    "bad",
    ((), (10, 10), (12, 10), (-1, 10), (10, True), (10, 12.5), "10", None),
)
def test_invalid_candidate_grid_is_rejected(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_finite_reference_bound_policy(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", (0.2, 0, -1, Fraction(-1, 10), Fraction(11, 10)))
def test_invalid_probability_inputs_rejected(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_finite_reference_bound_policy(
            _CANDIDATES,
            true_null_p=bad,  # type: ignore[arg-type]
        )
    with pytest.raises((TypeError, ValueError)):
        evaluate_finite_reference_bound_policy(
            _CANDIDATES,
            alternative_p=bad,  # type: ignore[arg-type]
        )


def test_immutable_records_and_exact_probability_types() -> None:
    observed = evaluate_finite_reference_bound_policy(_CANDIDATES)
    assert isinstance(observed.monitoring_activation_probability, Fraction)
    assert isinstance(observed.decisions_by_reference_count, tuple)
    assert isinstance(observed.decision_probability_masses, tuple)
    with pytest.raises(FrozenInstanceError):
        observed.monitoring_activation_probability = Fraction(0)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        observed.decisions_by_reference_count[0].selected_margin_40ths = 10  # type: ignore[misc]
