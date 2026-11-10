"""D-019 exact finite-reference upper-bound sensitivity diagnostic.

For B~Binomial(40,p0), invert a one-sided binomial CDF on a 0.001
probability grid. Select one margin using the upper bound, freeze the
selection across six monitoring looks, then evaluate at the true p0
and at a distinct alternative p1. This is synthetic, not a deployment gate.

Monotonicity: p <= U(B) implies R(B,m;p) <= R(B,m;U(B)).
Because the binomial lower-tail CDF decreases with p, the event
p > U(B) lies within the size-delta lower-tail rejection event, giving
P_p(p > U(B)) <= delta for every fixed p in [0,1]. A finite grid of
tested p values alone is not a proof of this coverage statement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Sequence

from .fitted_monitor_diagnostic import (
    SelectionStatus,
    _validated_frozen_policy_candidates,
)

_REFERENCE_N = 40
_WEEKLY_N = 20
_LOOKS = 6
_GRID_DENOMINATOR = 1000
_ALPHA = Fraction(1, 20)
_DELTA = Fraction(1, 20)


@dataclass(frozen=True, slots=True)
class BoundReferenceDecision:
    """The frozen decision at a single reference outcome, indexed by B."""

    reference_errors: int
    upper_null_probability: Fraction
    status: SelectionStatus
    selected_margin_40ths: int | None
    bound_conditional_family_alarm: Fraction | None
    actual_conditional_null_family_alarm: Fraction | None
    actual_conditional_alternative_detection: Fraction | None
    actual_conditional_null_limit_violated: bool | None
    bound_miscovered: bool


@dataclass(frozen=True, slots=True)
class BoundReferenceDecisionMass:
    """True reference-law probability of a selection or non-activation."""

    status: SelectionStatus
    selected_margin_40ths: int | None
    probability_mass: Fraction


@dataclass(frozen=True, slots=True)
class FiniteReferenceBoundEvaluation:
    """Exact joint reference/monitoring operating characteristics.

    Marginal alarm and detection mass includes only activated paths.
    Inactive outcomes are separately counted as withheld; they are not
    successful detections or observations of zero detector error.
    """

    candidate_margins_40ths: tuple[int, ...]
    true_null_probability: Fraction
    alternative_probability: Fraction
    decisions_by_reference_count: tuple[BoundReferenceDecision, ...]
    decision_probability_masses: tuple[BoundReferenceDecisionMass, ...]
    monitoring_activation_probability: Fraction
    monitoring_withheld_probability: Fraction
    marginal_actual_null_family_alarm: Fraction
    marginal_actual_alternative_detection: Fraction
    reference_bound_miscoverage_probability: Fraction
    activated_conditional_risk_violation_probability: Fraction
    violating_reference_condition_count: int


@lru_cache(maxsize=None)
def _binomial_pmf(n: int, probability: Fraction) -> tuple[Fraction, ...]:
    """Exact independent Bernoulli count probabilities, including endpoints."""
    return tuple(
        math.comb(n, k) * probability**k * (1 - probability) ** (n - k)
        for k in range(n + 1)
    )


def _lower_tail_cdf(b: int, probability: Fraction) -> Fraction:
    return sum(_binomial_pmf(_REFERENCE_N, probability)[: b + 1], Fraction())


@lru_cache(maxsize=None)
def _upper_confidence_bound(b: int) -> Fraction:
    """Smallest one-thousandth with P_j(X <= b) <= 0.05; b=40 -> 1.

    The lower-tail CDF is nonincreasing in j; binary search preserves
    a failing lower index and a passing upper index at each iteration.
    """
    if not 0 <= b <= _REFERENCE_N:
        raise ValueError("reference count must lie in [0, 40]")
    if b == _REFERENCE_N:
        return Fraction(1)

    lower, upper = 0, _GRID_DENOMINATOR
    while upper - lower > 1:
        midpoint = (lower + upper) // 2
        if _lower_tail_cdf(b, Fraction(midpoint, _GRID_DENOMINATOR)) <= _DELTA:
            upper = midpoint
        else:
            lower = midpoint
    return Fraction(upper, _GRID_DENOMINATOR)


def _strict_alarm(b: int, weekly_errors: int, margin: int) -> bool:
    return 2 * weekly_errors > b + margin


def _conditional_six_look_risk(b: int, margin: int, p: Fraction) -> Fraction:
    weekly_alarm = sum(
        (
            mass
            for k, mass in enumerate(_binomial_pmf(_WEEKLY_N, p))
            if _strict_alarm(b, k, margin)
        ),
        Fraction(),
    )
    return 1 - (1 - weekly_alarm) ** _LOOKS


def evaluate_finite_reference_bound_policy(
    candidates: Sequence[int],
    *,
    true_null_p: Fraction = Fraction(1, 5),
    alternative_p: Fraction = Fraction(3, 10),
) -> FiniteReferenceBoundEvaluation:
    """Select by a finite-reference upper null bound, then evaluate exactly.

    The reference count is sampled once and the chosen margin stays fixed
    for all six independent weeks. The reference distribution is integrated
    *after* the conditional family-risk transformation.
    """
    margins = _validated_frozen_policy_candidates(candidates)
    for name, p in (("true_null_p", true_null_p), ("alternative_p", alternative_p)):
        if type(p) is not Fraction:
            raise TypeError(f"{name} must be an exact Fraction")
        if not 0 <= p <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")

    reference = _binomial_pmf(_REFERENCE_N, true_null_p)
    decisions: list[BoundReferenceDecision] = []
    for b in range(_REFERENCE_N + 1):
        upper = _upper_confidence_bound(b)
        selected: int | None = None
        selected_bound_risk: Fraction | None = None
        for margin in margins:
            bound_risk = _conditional_six_look_risk(b, margin, upper)
            if bound_risk <= _ALPHA:
                selected = margin
                selected_bound_risk = bound_risk
                break
        if selected is None:
            decisions.append(
                BoundReferenceDecision(
                    reference_errors=b,
                    upper_null_probability=upper,
                    status=SelectionStatus.NO_METHOD_PASSES,
                    selected_margin_40ths=None,
                    bound_conditional_family_alarm=None,
                    actual_conditional_null_family_alarm=None,
                    actual_conditional_alternative_detection=None,
                    actual_conditional_null_limit_violated=None,
                    bound_miscovered=true_null_p > upper,
                )
            )
            continue

        actual_null = _conditional_six_look_risk(b, selected, true_null_p)
        actual_alternative = _conditional_six_look_risk(b, selected, alternative_p)
        decisions.append(
            BoundReferenceDecision(
                reference_errors=b,
                upper_null_probability=upper,
                status=SelectionStatus.SELECTED,
                selected_margin_40ths=selected,
                bound_conditional_family_alarm=selected_bound_risk,
                actual_conditional_null_family_alarm=actual_null,
                actual_conditional_alternative_detection=actual_alternative,
                actual_conditional_null_limit_violated=actual_null > _ALPHA,
                bound_miscovered=true_null_p > upper,
            )
        )

    rows = tuple(decisions)
    masses = tuple(
        BoundReferenceDecisionMass(
            status=(
                SelectionStatus.NO_METHOD_PASSES
                if margin is None
                else SelectionStatus.SELECTED
            ),
            selected_margin_40ths=margin,
            probability_mass=sum(
                (
                    reference[b]
                    for b, row in enumerate(rows)
                    if row.selected_margin_40ths == margin
                ),
                Fraction(),
            ),
        )
        for margin in (None, *margins)
    )
    withheld = masses[0].probability_mass
    activated = sum((mass.probability_mass for mass in masses[1:]), Fraction())
    marginal_null = sum(
        (
            reference[b] * row.actual_conditional_null_family_alarm
            for b, row in enumerate(rows)
            if row.actual_conditional_null_family_alarm is not None
        ),
        Fraction(),
    )
    marginal_alternative = sum(
        (
            reference[b] * row.actual_conditional_alternative_detection
            for b, row in enumerate(rows)
            if row.actual_conditional_alternative_detection is not None
        ),
        Fraction(),
    )
    miscoverage = sum(
        (reference[b] for b, row in enumerate(rows) if row.bound_miscovered),
        Fraction(),
    )
    violations = sum(
        (
            reference[b]
            for b, row in enumerate(rows)
            if row.actual_conditional_null_limit_violated is True
        ),
        Fraction(),
    )
    return FiniteReferenceBoundEvaluation(
        candidate_margins_40ths=margins,
        true_null_probability=true_null_p,
        alternative_probability=alternative_p,
        decisions_by_reference_count=rows,
        decision_probability_masses=masses,
        monitoring_activation_probability=activated,
        monitoring_withheld_probability=withheld,
        marginal_actual_null_family_alarm=marginal_null,
        marginal_actual_alternative_detection=marginal_alternative,
        reference_bound_miscoverage_probability=miscoverage,
        activated_conditional_risk_violation_probability=violations,
        violating_reference_condition_count=sum(
            row.actual_conditional_null_limit_violated is True for row in rows
        ),
    )
