"""Exact synthetic fitted-monitor operating-characteristic diagnostic (D-019).

A reference Binomial(40, 0.20) count is fitted once, then frozen before
six independent Binomial(20, 0.20) monitoring weeks for null false alarms.
A separate Binomial(20, 0.30) synthetic alternative measures detection power.
Neither is a production threshold, acceptance gate, or E-004 amendment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Sequence

_REFERENCE_N = 40
_WEEKLY_N = 20
_TRUE_ERROR_RATE = 0.20
_LOOKS = 6
_MARGINAL_LIMIT = 0.05
_CONDITIONAL_LIMIT = 0.05
_LOWER_TAIL = 0.025
_UPPER_TAIL = 0.975


class SelectionStatus(StrEnum):
    """A candidate was selected, or none met the declared constraints."""

    SELECTED = "SELECTED"
    NO_METHOD_PASSES = "NO_METHOD_PASSES"


@dataclass(frozen=True, slots=True)
class MarginEvaluation:
    """Inspectable marginal and frozen-reference conditional operating characteristics.

    The conditional tuple is indexed directly by reference error count b=0,...,40.
    The protected bounds include both endpoints.
    """

    margin_40ths: int
    marginal_family_false_alarm: float
    conditional_family_false_alarm_by_reference_count: tuple[float, ...]
    protected_reference_count_bounds: tuple[int, int]
    max_protected_conditional_family_false_alarm: float


@dataclass(frozen=True, slots=True)
class DetectionPowerEvaluation:
    """Detection under p=0.30 with one shared frozen reference across six looks.

    The conditional tuple is indexed by reference count b=0,...,40; the
    protected interval includes both endpoints. No acceptance gate is implied.
    """

    margin_40ths: int
    marginal_family_detection_probability: float
    conditional_family_detection_probability_by_reference_count: tuple[float, ...]
    protected_reference_count_bounds: tuple[int, int]
    min_protected_conditional_family_detection_probability: float
    max_protected_conditional_family_detection_probability: float


@dataclass(frozen=True, slots=True)
class MarginSelection:
    """Full candidate evaluation and explicit no-solution outcome."""

    status: SelectionStatus
    selected_margin_40ths: int | None
    evaluations: tuple[MarginEvaluation, ...]


@dataclass(frozen=True, slots=True)
class FrozenReferenceMarginSelection:
    """Decision made once for an observed reference count, before monitoring."""

    reference_errors: int
    status: SelectionStatus
    selected_margin_40ths: int | None
    conditional_family_false_alarm: float | None


@dataclass(frozen=True, slots=True)
class ReferencePolicyDecisionMass:
    """Probability of one reference-conditioned decision under the null reference law."""

    status: SelectionStatus
    selected_margin_40ths: int | None
    probability_mass: float


@dataclass(frozen=True, slots=True)
class FrozenReferencePolicyEvaluation:
    """Operating characteristics of the full select-once, freeze, monitor policy.

    Selection rows are indexed by reference errors b=0,...,40; each decision
    mass includes reference uncertainty. Non-activation has no monitoring
    event, not an observed detector error of zero. Conditional-on-enabled
    probabilities are None (NOT_ESTIMABLE) if activation mass is zero.
    """

    candidate_margins_40ths: tuple[int, ...]
    selections_by_reference_count: tuple[FrozenReferenceMarginSelection, ...]
    decision_probability_masses: tuple[ReferencePolicyDecisionMass, ...]
    monitoring_activation_probability: float
    monitoring_withheld_probability: float
    marginal_family_false_alarm: float
    marginal_family_detection_probability: float
    conditional_family_false_alarm_given_activation: float | None
    conditional_family_detection_probability_given_activation: float | None


@lru_cache(maxsize=None)
def _binomial_pmf(n: int) -> tuple[float, ...]:
    """Binomial(n, 0.20) probabilities by small-sample exact combinatorics."""
    if type(n) is not int or n < 0:
        raise ValueError("n must be a nonnegative exact integer")
    return tuple(
        math.comb(n, k) * _TRUE_ERROR_RATE**k * (1.0 - _TRUE_ERROR_RATE) ** (n - k)
        for k in range(n + 1)
    )


def _strict_alarm(reference_errors: int, weekly_errors: int, margin_40ths: int) -> bool:
    """Apply the exact integer rule; ties are not alarms."""
    return 2 * weekly_errors > reference_errors + margin_40ths


def _horizon_false_alarm(weekly_probability: float) -> float:
    """Six-look false-alarm probability conditional on a single frozen baseline."""
    if not math.isfinite(weekly_probability) or not 0.0 <= weekly_probability <= 1.0:
        raise ValueError("weekly probability must lie in [0, 1]")
    if weekly_probability == 1.0:
        return 1.0
    return -math.expm1(_LOOKS * math.log1p(-weekly_probability))


def _central_reference_bounds(reference_pmf: tuple[float, ...]) -> tuple[int, int]:
    """Inclusive central equal-tail region from the reference-count CDF."""
    lower = next(
        b
        for b in range(len(reference_pmf))
        if math.fsum(reference_pmf[: b + 1]) >= _LOWER_TAIL
    )
    upper = next(
        b
        for b in range(len(reference_pmf))
        if math.fsum(reference_pmf[: b + 1]) >= _UPPER_TAIL
    )
    return lower, upper


@lru_cache(maxsize=None)
def evaluate_margin(margin_40ths: int) -> MarginEvaluation:
    """Evaluate one margin over all 41 possible frozen reference estimates.

    Average the *conditional six-look* probabilities, never the weekly rates
    before the six-look transformation. The baseline is frozen across looks.
    """
    if type(margin_40ths) is not int:
        raise TypeError("margin_40ths must be an exact integer, not bool or float")
    if margin_40ths < 0:
        raise ValueError("margin_40ths must be nonnegative")

    reference = _binomial_pmf(_REFERENCE_N)
    weekly = _binomial_pmf(_WEEKLY_N)
    bounds = _central_reference_bounds(reference)
    conditional = tuple(
        _horizon_false_alarm(
            math.fsum(
                weight
                for k, weight in enumerate(weekly)
                if _strict_alarm(b, k, margin_40ths)
            )
        )
        for b in range(_REFERENCE_N + 1)
    )
    marginal = math.fsum(weight * conditional[b] for b, weight in enumerate(reference))
    protected_max = max(conditional[bounds[0] : bounds[1] + 1])
    return MarginEvaluation(
        margin_40ths=margin_40ths,
        marginal_family_false_alarm=marginal,
        conditional_family_false_alarm_by_reference_count=conditional,
        protected_reference_count_bounds=bounds,
        max_protected_conditional_family_false_alarm=protected_max,
    )


@lru_cache(maxsize=None)
def evaluate_detection_power(margin_40ths: int) -> DetectionPowerEvaluation:
    """Exact six-look detection probability for a synthetic p=0.30 alternative.

    The reference count is still drawn from Binomial(40, 0.20) and frozen
    for all six independently sampled Binomial(20, 0.30) monitoring weeks.
    This is exploratory power, not a validated degradation threshold.
    """
    # Reuse the existing validation and protected-region definition without
    # changing the null evaluation, thresholds, or selection semantics.
    bounds = evaluate_margin(margin_40ths).protected_reference_count_bounds
    reference = _binomial_pmf(_REFERENCE_N)
    alternative_rate = 0.30
    weekly = tuple(
        math.comb(_WEEKLY_N, k)
        * alternative_rate**k
        * (1.0 - alternative_rate) ** (_WEEKLY_N - k)
        for k in range(_WEEKLY_N + 1)
    )
    conditional = tuple(
        _horizon_false_alarm(
            math.fsum(
                weight
                for k, weight in enumerate(weekly)
                if _strict_alarm(b, k, margin_40ths)
            )
        )
        for b in range(_REFERENCE_N + 1)
    )
    marginal = math.fsum(weight * conditional[b] for b, weight in enumerate(reference))
    protected = conditional[bounds[0] : bounds[1] + 1]
    return DetectionPowerEvaluation(
        margin_40ths=margin_40ths,
        marginal_family_detection_probability=marginal,
        conditional_family_detection_probability_by_reference_count=conditional,
        protected_reference_count_bounds=bounds,
        min_protected_conditional_family_detection_probability=min(protected),
        max_protected_conditional_family_detection_probability=max(protected),
    )


def select_margin(
    candidates: Sequence[int], *, require_conditional: bool
) -> MarginSelection:
    """Select the smallest satisfying candidate or explicitly reject the grid."""
    if type(require_conditional) is not bool:
        raise TypeError("require_conditional must be a bool")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be an ascending integer sequence")

    margins = tuple(candidates)
    if not margins:
        raise ValueError("candidates must not be empty")
    if any(type(margin) is not int for margin in margins):
        raise TypeError("all margins must be exact integers, not bool or float")
    if any(margin < 0 for margin in margins):
        raise ValueError("margins must be nonnegative")
    if any(left >= right for left, right in zip(margins, margins[1:])):
        raise ValueError("candidate margins must be strictly increasing")

    evaluations = tuple(evaluate_margin(margin) for margin in margins)
    for evaluation in evaluations:
        if evaluation.marginal_family_false_alarm > _MARGINAL_LIMIT:
            continue
        if (
            require_conditional
            and evaluation.max_protected_conditional_family_false_alarm
            > _CONDITIONAL_LIMIT
        ):
            continue
        return MarginSelection(
            status=SelectionStatus.SELECTED,
            selected_margin_40ths=evaluation.margin_40ths,
            evaluations=evaluations,
        )
    return MarginSelection(
        status=SelectionStatus.NO_METHOD_PASSES,
        selected_margin_40ths=None,
        evaluations=evaluations,
    )


def _validated_frozen_policy_candidates(candidates: Sequence[int]) -> tuple[int, ...]:
    """Validate a nonempty strictly increasing sequence of exact nonnegative ints."""
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be an ascending integer sequence")
    margins = tuple(candidates)
    if not margins:
        raise ValueError("candidates must not be empty")
    if any(type(margin) is not int for margin in margins):
        raise TypeError("all margins must be exact integers, not bool or float")
    if any(margin < 0 for margin in margins):
        raise ValueError("margins must be nonnegative")
    if any(left >= right for left, right in zip(margins, margins[1:])):
        raise ValueError("candidate margins must be strictly increasing")
    return margins


def select_margin_for_frozen_reference(
    reference_errors: int, candidates: Sequence[int]
) -> FrozenReferenceMarginSelection:
    """Select once from conditional null risk after observing the reference.

    The Binomial(20, 0.20) null is known here solely as an oracle-informed
    synthetic experiment; this is not a deployable unknown-null rule.
    """
    if type(reference_errors) is not int:
        raise TypeError("reference_errors must be an exact integer, not bool or float")
    if not 0 <= reference_errors <= _REFERENCE_N:
        raise ValueError("reference_errors must be in [0, 40]")
    margins = _validated_frozen_policy_candidates(candidates)
    for margin in margins:
        conditional_null = evaluate_margin(
            margin
        ).conditional_family_false_alarm_by_reference_count[reference_errors]
        if conditional_null <= _CONDITIONAL_LIMIT:
            return FrozenReferenceMarginSelection(
                reference_errors=reference_errors,
                status=SelectionStatus.SELECTED,
                selected_margin_40ths=margin,
                conditional_family_false_alarm=conditional_null,
            )
    return FrozenReferenceMarginSelection(
        reference_errors=reference_errors,
        status=SelectionStatus.NO_METHOD_PASSES,
        selected_margin_40ths=None,
        conditional_family_false_alarm=None,
    )


def evaluate_frozen_reference_policy(
    candidates: Sequence[int],
) -> FrozenReferencePolicyEvaluation:
    """Integrate the reference-conditioned *selected* policy over all B=0,...,40.

    A reference count is drawn once, determines one frozen margin, and is held
    across all six weeks. Inactive paths contribute no monitoring event; they
    are separately accounted for as withheld, not as successful detections.
    """
    margins = _validated_frozen_policy_candidates(candidates)
    selections = tuple(
        select_margin_for_frozen_reference(b, margins) for b in range(_REFERENCE_N + 1)
    )
    reference = _binomial_pmf(_REFERENCE_N)
    decision_masses = (
        ReferencePolicyDecisionMass(
            status=SelectionStatus.NO_METHOD_PASSES,
            selected_margin_40ths=None,
            probability_mass=math.fsum(
                reference[b]
                for b, decision in enumerate(selections)
                if decision.status is SelectionStatus.NO_METHOD_PASSES
            ),
        ),
        *(
            ReferencePolicyDecisionMass(
                status=SelectionStatus.SELECTED,
                selected_margin_40ths=margin,
                probability_mass=math.fsum(
                    reference[b]
                    for b, decision in enumerate(selections)
                    if decision.selected_margin_40ths == margin
                ),
            )
            for margin in margins
        ),
    )
    withheld = decision_masses[0].probability_mass
    activated = math.fsum(mass.probability_mass for mass in decision_masses[1:])
    conditional_null = {
        margin: evaluate_margin(
            margin
        ).conditional_family_false_alarm_by_reference_count
        for margin in margins
    }
    conditional_alternative = {
        margin: evaluate_detection_power(
            margin
        ).conditional_family_detection_probability_by_reference_count
        for margin in margins
    }
    marginal_null = math.fsum(
        reference[b] * conditional_null[margin][b]
        for b, selection in enumerate(selections)
        if (margin := selection.selected_margin_40ths) is not None
    )
    marginal_alternative = math.fsum(
        reference[b] * conditional_alternative[margin][b]
        for b, selection in enumerate(selections)
        if (margin := selection.selected_margin_40ths) is not None
    )
    return FrozenReferencePolicyEvaluation(
        candidate_margins_40ths=margins,
        selections_by_reference_count=selections,
        decision_probability_masses=decision_masses,
        monitoring_activation_probability=activated,
        monitoring_withheld_probability=withheld,
        marginal_family_false_alarm=marginal_null,
        marginal_family_detection_probability=marginal_alternative,
        conditional_family_false_alarm_given_activation=(
            marginal_null / activated if activated > 0.0 else None
        ),
        conditional_family_detection_probability_given_activation=(
            marginal_alternative / activated if activated > 0.0 else None
        ),
    )
