"""Independent high-precision selection and Fraction realized-risk D-019 controls.

The oracle never calls the comparison implementation for expected decisions.
It independently inverts Binomial(40,p) CDF with Decimal, enumerates K by
integer binomial coefficients, and integrates six-look event probabilities
under one frozen B with exact rational arithmetic.
"""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import TypeAdapter, ValidationError

from llm_evaluation import fitted_monitor_policy_comparison as subject
from llm_evaluation.fitted_monitor_diagnostic import SelectionStatus
from llm_evaluation.fitted_monitor_estimation_sensitivity import (
    evaluate_estimated_null_sensitivity,
)
from llm_evaluation.fitted_monitor_policy_comparison import (
    PolicyComparison,
    PolicyName,
    evaluate_policy_comparison,
)

_MARGINS = (10, 12, 14, 16)
_P0 = Fraction(1, 5)
_P1 = Fraction(3, 10)
_LIMIT = Fraction(1, 20)
_LEVELS: dict[PolicyName, Decimal] = {
    PolicyName.UPPER_CP_90: Decimal("0.90"),
    PolicyName.UPPER_CP_95: Decimal("0.95"),
    PolicyName.UPPER_CP_99_ABSTAIN: Decimal("0.99"),
}


@lru_cache(maxsize=None)
def _rational_binomial(n: int, p: Fraction) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(math.comb(n, k)) * p**k * (1 - p) ** (n - k) for k in range(n + 1)
    )


def _exact_horizon(b: int, margin: int, p: Fraction) -> Fraction:
    weekly = sum(
        (
            weight
            for k, weight in enumerate(_rational_binomial(20, p))
            if 2 * k > b + margin
        ),
        Fraction(),
    )
    return 1 - (1 - weekly) ** 6


def _decimal_cdf(b: int, p: Decimal) -> Decimal:
    return sum(
        (
            Decimal(math.comb(40, k)) * p**k * (Decimal(1) - p) ** (40 - k)
            for k in range(b + 1)
        ),
        Decimal(0),
    )


@lru_cache(maxsize=None)
def _independent_upper(b: int, confidence: Decimal) -> Decimal:
    if b == 40:
        return Decimal(1)
    with localcontext() as context:
        context.prec = 62
        low, high = Decimal(0), Decimal(1)
        alpha = Decimal(1) - confidence
        for _ in range(190):
            mid = (low + high) / 2
            if _decimal_cdf(b, mid) > alpha:
                low = mid
            else:
                high = mid
        return +high


def _oracle_selection(b: int, name: PolicyName) -> tuple[int | None, float, float]:
    if name is PolicyName.KNOWN_NULL_ORACLE:
        p = _P0
        alternatives = tuple(
            (margin, _exact_horizon(b, margin, p)) for margin in _MARGINS
        )
        for margin, risk in alternatives:
            if risk <= _LIMIT:
                return margin, float(p), float(risk)
        return None, float(p), float(alternatives[-1][1])
    if name is PolicyName.NAIVE_PLUGIN:
        p = Fraction(b, 40)
        alternatives = tuple(
            (margin, _exact_horizon(b, margin, p)) for margin in _MARGINS
        )
        for margin, risk in alternatives:
            if risk <= _LIMIT:
                return margin, float(p), float(risk)
        return None, float(p), float(alternatives[-1][1])

    bound = _independent_upper(b, _LEVELS[name])
    with localcontext() as context:
        context.prec = 62
        alternative = tuple(
            (
                margin,
                Decimal(1)
                - (
                    Decimal(1)
                    - sum(
                        (
                            Decimal(math.comb(20, k))
                            * bound**k
                            * (Decimal(1) - bound) ** (20 - k)
                            for k in range(21)
                            if 2 * k > b + margin
                        ),
                        Decimal(0),
                    )
                )
                ** 6,
            )
            for margin in _MARGINS
        )
    for margin, risk in alternative:
        if risk <= Decimal("0.05"):
            return margin, float(bound), float(risk)
    return None, float(bound), float(alternative[-1][1])


def _check_exact_oracle(result: PolicyComparison) -> None:
    pmf = _rational_binomial(40, _P0)
    assert result.design_id == subject.POLICY_DESIGN_ID
    assert result.candidate_margins_40ths == _MARGINS
    assert tuple(p.name for p in result.policies) == tuple(PolicyName)
    for policy in result.policies:
        assert len(policy.decisions) == 41
        selected: list[int | None] = []
        realized_null: list[Fraction | None] = []
        realized_alt: list[Fraction | None] = []
        violations: list[int] = []
        undercoverage: list[int] = []
        for b, row in enumerate(policy.decisions):
            chosen, fitted, fitted_risk = _oracle_selection(b, policy.name)
            selected.append(chosen)
            assert row.reference_errors == b
            assert row.selected_margin_40ths == chosen
            assert row.selection_null_probability == pytest.approx(fitted, abs=3e-15)
            assert row.minimum_candidate_selection_risk == pytest.approx(
                float(
                    _exact_horizon(b, 16, Fraction(b, 40))
                    if policy.name is PolicyName.NAIVE_PLUGIN
                    else (
                        _exact_horizon(b, 16, _P0)
                        if policy.name is PolicyName.KNOWN_NULL_ORACLE
                        else _decimal_horizon(
                            b, 16, _independent_upper(b, _LEVELS[policy.name])
                        )
                    )
                ),
                rel=1e-11,
                abs=2e-14,
            )
            if chosen is None:
                assert row.status is SelectionStatus.NO_METHOD_PASSES
                assert row.selection_risk is None
                assert row.actual_null_six_look_risk is None
                assert row.actual_alternative_six_look_detection is None
                assert row.conditional_limit_violated is None
                realized_null.append(None)
                realized_alt.append(None)
            else:
                expected_null = _exact_horizon(b, chosen, _P0)
                expected_alt = _exact_horizon(b, chosen, _P1)
                assert row.status is SelectionStatus.SELECTED
                assert row.selection_risk == pytest.approx(fitted_risk, abs=2e-14)
                assert row.selection_risk <= 0.05
                assert row.actual_null_six_look_risk == pytest.approx(
                    float(expected_null), rel=1e-12, abs=1e-15
                )
                assert row.actual_alternative_six_look_detection == pytest.approx(
                    float(expected_alt), rel=1e-12, abs=1e-15
                )
                assert row.conditional_limit_violated is (expected_null > _LIMIT)
                if expected_null > _LIMIT:
                    violations.append(b)
                realized_null.append(expected_null)
                realized_alt.append(expected_alt)
            if policy.name in _LEVELS and _independent_upper(
                b, _LEVELS[policy.name]
            ) < Decimal("0.2"):
                undercoverage.append(b)

        active_mass = sum(
            (pmf[b] for b, margin in enumerate(selected) if margin is not None),
            Fraction(),
        )
        withheld = Fraction(1) - active_mass
        marginal_null = sum(
            (pmf[b] * risk for b, risk in enumerate(realized_null) if risk is not None),
            Fraction(),
        )
        marginal_alt = sum(
            (
                pmf[b] * power
                for b, power in enumerate(realized_alt)
                if power is not None
            ),
            Fraction(),
        )
        violation_mass = sum((pmf[b] for b in violations), Fraction())
        for field, value in (
            (policy.activation_probability, active_mass),
            (policy.withheld_probability, withheld),
            (policy.marginal_null_alarm, marginal_null),
            (policy.marginal_alternative_detection, marginal_alt),
            (policy.violating_reference_mass, violation_mass),
        ):
            assert field == pytest.approx(float(value), rel=1e-12, abs=1e-15)
        assert policy.violating_reference_count == len(violations)
        assert policy.margin_probability_masses == pytest.approx(
            tuple(
                float(
                    sum(
                        (
                            pmf[b]
                            for b, margin in enumerate(selected)
                            if margin == decision
                        ),
                        Fraction(),
                    )
                )
                for decision in (None, *_MARGINS)
            ),
            abs=1e-14,
        )
        if active_mass:
            assert policy.null_alarm_given_activation == pytest.approx(
                float(marginal_null / active_mass), abs=1e-14
            )
            assert policy.detection_given_activation == pytest.approx(
                float(marginal_alt / active_mass), abs=1e-14
            )
        else:
            assert policy.null_alarm_given_activation is None
            assert policy.detection_given_activation is None
        if policy.name in _LEVELS:
            undercoverage_mass = sum((pmf[b] for b in undercoverage), Fraction())
            assert policy.confidence_noncoverage_mass_at_true_null == pytest.approx(
                float(undercoverage_mass), abs=1e-14
            )
            assert undercoverage_mass <= 1 - Fraction(str(_LEVELS[policy.name]))
            assert set(violations).issubset(set(undercoverage))
        else:
            assert policy.confidence_noncoverage_mass_at_true_null is None


def _decimal_horizon(b: int, margin: int, p: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 62
        q = sum(
            (
                Decimal(math.comb(20, k)) * p**k * (Decimal(1) - p) ** (20 - k)
                for k in range(21)
                if 2 * k > b + margin
            ),
            Decimal(0),
        )
        return Decimal(1) - (Decimal(1) - q) ** 6


def test_41_reference_conditions_against_independent_selection_and_risk_oracle() -> (
    None
):
    _check_exact_oracle(evaluate_policy_comparison(_MARGINS))


def test_existing_known_null_and_naive_phase_results_are_unchanged() -> None:
    result = evaluate_policy_comparison(_MARGINS)
    oracle, naive = result.policies[:2]
    baseline = evaluate_estimated_null_sensitivity(_MARGINS)
    assert oracle.activation_probability == pytest.approx(0.99853784920463662)
    assert oracle.marginal_null_alarm == pytest.approx(0.011949878124584378)
    assert oracle.marginal_alternative_detection == pytest.approx(0.206532922955598)
    assert naive.activation_probability == 1.0
    assert naive.marginal_null_alarm == pytest.approx(
        baseline.marginal_actual_null_family_alarm
    )
    assert naive.marginal_alternative_detection == pytest.approx(
        baseline.marginal_actual_alternative_detection
    )
    assert naive.violating_reference_count == 8
    assert naive.violating_reference_mass == pytest.approx(
        baseline.violating_reference_probability_mass
    )
    assert [
        b for b, r in enumerate(naive.decisions) if r.conditional_limit_violated
    ] == list(range(8))
    assert naive.decisions[0].selection_risk == 0.0
    assert naive.decisions[0].actual_null_six_look_risk == pytest.approx(
        0.729473608923, abs=1e-12
    )


def test_nonoracle_selection_does_not_read_true_p0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = tuple(
        (p.name, tuple(d.selected_margin_40ths for d in p.decisions))
        for p in evaluate_policy_comparison(_MARGINS).policies
    )
    monkeypatch.setattr(subject._known, "_TRUE_ERROR_RATE", 0.4)
    revised = tuple(
        (p.name, tuple(d.selected_margin_40ths for d in p.decisions))
        for p in evaluate_policy_comparison(_MARGINS).policies
    )
    # Every nonoracle selector is a function of B, candidate margins and
    # prespecified confidence only; changed p0 must not change its choices.
    assert original[1:] == revised[1:]
    assert original[0] != revised[0]


def test_wrong_null_in_realized_risk_is_detected() -> None:
    result = evaluate_policy_comparison(_MARGINS)
    naive = result.policies[1]
    row0 = naive.decisions[0]
    corrupted = replace(
        row0,
        actual_null_six_look_risk=row0.selection_risk,
        conditional_limit_violated=False,
    )
    assert corrupted != row0
    assert corrupted.actual_null_six_look_risk == 0
    assert row0.actual_null_six_look_risk is not None
    assert row0.actual_null_six_look_risk > 0.7
    with pytest.raises(AssertionError):
        _check_exact_oracle(
            replace(
                result,
                policies=(
                    result.policies[0],
                    replace(naive, decisions=(corrupted, *naive.decisions[1:])),
                    *result.policies[2:],
                ),
            )
        )


def test_frozen_reference_and_six_look_order_negative_control() -> None:
    reference = _rational_binomial(40, _P0)
    result = evaluate_policy_comparison(_MARGINS)
    for policy in result.policies:
        expected_weekly = []
        for b, row in enumerate(policy.decisions):
            selected = row.selected_margin_40ths
            if selected is None:
                expected_weekly.append(Fraction())
            else:
                q = sum(
                    (
                        weight
                        for k, weight in enumerate(_rational_binomial(20, _P0))
                        if 2 * k > b + selected
                    ),
                    Fraction(),
                )
                expected_weekly.append(q)
        mean_week = sum(
            (reference[b] * q for b, q in enumerate(expected_weekly)), Fraction()
        )
        wrong_independently_resampled_b = 1 - (1 - mean_week) ** 6
        assert (
            abs(policy.marginal_null_alarm - float(wrong_independently_resampled_b))
            > 1e-5
        )


def test_tie_alarm_and_nonactivation_negative_controls() -> None:
    result = evaluate_policy_comparison(_MARGINS)
    for policy in result.policies:
        for row in policy.decisions:
            margin = row.selected_margin_40ths
            if margin is not None and (row.reference_errors + margin) % 2 == 0:
                k = (row.reference_errors + margin) // 2
                if 0 <= k <= 20:
                    assert not subject._strict_alarm(row.reference_errors, k, margin)
    # Abstention is not counted as zero observed error or positive detection.
    abstain = result.policies[-1]
    assert abstain.withheld_probability > 0
    withheld = next(d for d in abstain.decisions if d.selected_margin_40ths is None)
    assert withheld.actual_null_six_look_risk is None
    assert withheld.actual_alternative_six_look_detection is None
    assert abstain.marginal_alternative_detection <= abstain.activation_probability


def test_wrong_averaging_and_violation_averaging_can_not_pass_oracle() -> None:
    result = evaluate_policy_comparison(_MARGINS)
    naive = result.policies[1]
    wrong = replace(
        naive,
        violating_reference_count=0,
        violating_reference_mass=0.0,
    )
    with pytest.raises(AssertionError):
        _check_exact_oracle(
            replace(result, policies=(result.policies[0], wrong, *result.policies[2:]))
        )


def test_artifact_schema_roundtrip_and_required_fields() -> None:
    adapter = TypeAdapter(PolicyComparison)
    original = evaluate_policy_comparison(_MARGINS)
    payload = adapter.dump_json(original, indent=2)
    assert adapter.validate_json(payload) == original
    with pytest.raises(ValidationError):
        adapter.validate_json(b'{"design_id":"bad"}')
    for malformed in ("unexpected_field", "coerced_number", "nested_unknown"):
        content = json.loads(payload)
        if malformed == "unexpected_field":
            content["unexpected"] = "invalid"
        elif malformed == "coerced_number":
            content["reference_n"] = "40"
        else:
            content["policies"][0]["decisions"][0]["unexpected"] = True
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(content))
    with pytest.raises(FrozenInstanceError):
        original.design_id = "mutated"  # type: ignore[misc]


@pytest.mark.parametrize("bad", ((), (10, 10), (12, 10), (10, "12"), (10, True)))
def test_candidate_grid_validation(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_policy_comparison(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", (-1, 41, True, 1.0))
def test_cp_rejects_reference_outside_support(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        subject._cp_upper_bound(bad, 0.95)  # type: ignore[arg-type]


@given(st.integers(min_value=0, max_value=39))
@settings(max_examples=41, deadline=None)
def test_cp_upper_is_monotone_and_has_correct_cdf_inversion(b: int) -> None:
    for confidence in (0.90, 0.95, 0.99):
        upper = subject._cp_upper_bound(b, confidence)
        assert 0 < upper < 1
        assert subject._cp_upper_bound(b + 1, confidence) >= upper
        expected = float(_independent_upper(b, Decimal(str(confidence))))
        assert upper == pytest.approx(expected, abs=3e-15)
        assert float(_decimal_cdf(b, Decimal(str(upper)))) == pytest.approx(
            1 - confidence, rel=1e-10, abs=3e-14
        )


@given(st.integers(min_value=0, max_value=40), st.sampled_from(_MARGINS))
@settings(max_examples=80, deadline=None)
def test_six_look_alarm_probability_monotone_in_monitor_p(b: int, margin: int) -> None:
    assert subject._conditional_event(b, margin, 0.0) <= subject._conditional_event(
        b, margin, 0.20
    )
    assert subject._conditional_event(b, margin, 0.20) <= subject._conditional_event(
        b, margin, 0.30
    )
    assert subject._conditional_event(b, margin, 0.30) <= subject._conditional_event(
        b, margin, 1.0
    )


def test_inclusive_comparator_mutant_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Replacing strict > with >= changes selection and must fail the oracle.
    monkeypatch.setattr(
        subject,
        "_strict_alarm",
        lambda b, k, m: 2 * k >= b + m,
    )
    with pytest.raises(AssertionError):
        _check_exact_oracle(evaluate_policy_comparison(_MARGINS))


def test_weekly_instead_of_six_look_transform_mutant_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Treating weekly alarm as the horizon event is a plausible bad shortcut.
    monkeypatch.setattr(subject, "_horizon_false_alarm", lambda weekly: weekly)
    with pytest.raises(AssertionError):
        _check_exact_oracle(evaluate_policy_comparison(_MARGINS))


def test_omitting_abstention_from_marginal_outcomes_fails_oracle() -> None:
    result = evaluate_policy_comparison(_MARGINS)
    abstain = result.policies[-1]
    assert abstain.activation_probability > 0
    assert abstain.withheld_probability > 0
    # A conditional-on-activation detection probability is not a
    # marginal detection probability; the independent oracle rejects it.
    assert abstain.detection_given_activation is not None
    wrong = replace(
        abstain,
        marginal_alternative_detection=abstain.detection_given_activation,
    )
    with pytest.raises(AssertionError):
        _check_exact_oracle(replace(result, policies=(*result.policies[:-1], wrong)))


def test_persisted_development_fixture_matches_executable_result() -> None:
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "d019_estimated_null_policy_comparison_dev_v1.json"
    )
    payload = path.read_bytes()
    adapter = TypeAdapter(PolicyComparison)
    expected = evaluate_policy_comparison(_MARGINS)
    assert payload == adapter.dump_json(expected, indent=2) + b"\n"
    assert adapter.validate_json(payload) == expected
    assert sum(len(policy.decisions) for policy in expected.policies) == 205
