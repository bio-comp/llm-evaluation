"""D-019 deterministic sensitivity of a frozen-reference policy to plug-in null risk.

The selection step replaces the *known* null p=0.20 with p_hat=B/40.
The reference B is drawn once from Binomial(40, 0.20), and the selected
margin stays fixed across six independent monitoring looks. Realized
null risk is still evaluated under p=0.20, and synthetic alternative
detection under p=0.30. No deployment calibration is implied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from . import fitted_monitor_diagnostic as _known
from .fitted_monitor_diagnostic import (
    ReferencePolicyDecisionMass,
    SelectionStatus,
    _horizon_false_alarm,
    _strict_alarm,
    _validated_frozen_policy_candidates,
)


@dataclass(frozen=True, slots=True)
class EstimatedNullReferenceDecision:
    """One reference-conditioned choice and its estimated versus actual risks."""

    reference_errors: int
    plug_in_null_probability: float
    status: SelectionStatus
    selected_margin_40ths: int | None
    estimated_conditional_null_family_alarm: float | None
    actual_conditional_null_family_alarm: float | None
    actual_conditional_alternative_detection: float | None
    actual_conditional_null_limit_violated: bool | None


@dataclass(frozen=True, slots=True)
class EstimatedNullSensitivityEvaluation:
    """Inspectable complete-policy sensitivity, including non-activation mass.

    The reference decisions are indexed by B=0,...,40. Marginal events include
    only active monitoring paths; withheld paths are tracked separately and
    are NOT observations of correct detection or zero detector error.
    """

    candidate_margins_40ths: tuple[int, ...]
    decisions_by_reference_count: tuple[EstimatedNullReferenceDecision, ...]
    decision_probability_masses: tuple[ReferencePolicyDecisionMass, ...]
    monitoring_activation_probability: float
    monitoring_withheld_probability: float
    marginal_actual_null_family_alarm: float
    marginal_actual_alternative_detection: float
    violating_reference_probability_mass: float
    violating_reference_condition_count: int


def _binomial_pmf_at_rate(n: int, probability: float) -> tuple[float, ...]:
    """Binomial probabilities at the frozen plug-in estimate, including 0 and 1."""
    return tuple(
        math.comb(n, k) * probability**k * (1.0 - probability) ** (n - k)
        for k in range(n + 1)
    )


def _conditional_six_look_event(
    reference_errors: int, margin_40ths: int, weekly_pmf: tuple[float, ...]
) -> float:
    """Apply one reference and one margin to the six-look family event."""
    weekly_alarm = math.fsum(
        mass
        for k, mass in enumerate(weekly_pmf)
        if _strict_alarm(reference_errors, k, margin_40ths)
    )
    return _horizon_false_alarm(weekly_alarm)


def _actual_null_family_alarm(reference_errors: int, margin_40ths: int) -> float:
    """Reuse the fixed true-null generator (p=0.20), not the fitted p_hat."""
    return _known.evaluate_margin(
        margin_40ths
    ).conditional_family_false_alarm_by_reference_count[reference_errors]


def _actual_alternative_detection(reference_errors: int, margin_40ths: int) -> float:
    """Reuse the fixed synthetic alternative generator (p=0.30)."""
    return _known.evaluate_detection_power(
        margin_40ths
    ).conditional_family_detection_probability_by_reference_count[reference_errors]


def evaluate_estimated_null_sensitivity(
    candidates: Sequence[int],
) -> EstimatedNullSensitivityEvaluation:
    """Evaluate the naïve plug-in selector against the *actual* monitoring law.

    For each observed B, choose the first margin whose six-look risk calculated
    under Binomial(20, B/40) is <=0.05. Freeze B and the margin across all six
    weeks, evaluate under true monitoring p=0.20 or alternative p=0.30, and only
    then integrate over B ~ Binomial(40, 0.20). This oracle-informed comparison
    is exploratory and does not define a deployable null estimator or gate.
    """
    margins = _validated_frozen_policy_candidates(candidates)
    decisions: list[EstimatedNullReferenceDecision] = []
    for b in range(_known._REFERENCE_N + 1):
        fitted_rate = b / _known._REFERENCE_N
        fitted_weekly_pmf = _binomial_pmf_at_rate(_known._WEEKLY_N, fitted_rate)
        selected: int | None = None
        estimated_risk: float | None = None
        for margin in margins:
            estimated = _conditional_six_look_event(b, margin, fitted_weekly_pmf)
            if estimated <= _known._CONDITIONAL_LIMIT:
                selected, estimated_risk = margin, estimated
                break
        if selected is None:
            decisions.append(
                EstimatedNullReferenceDecision(
                    reference_errors=b,
                    plug_in_null_probability=fitted_rate,
                    status=SelectionStatus.NO_METHOD_PASSES,
                    selected_margin_40ths=None,
                    estimated_conditional_null_family_alarm=None,
                    actual_conditional_null_family_alarm=None,
                    actual_conditional_alternative_detection=None,
                    actual_conditional_null_limit_violated=None,
                )
            )
            continue
        actual_null = _actual_null_family_alarm(b, selected)
        decisions.append(
            EstimatedNullReferenceDecision(
                reference_errors=b,
                plug_in_null_probability=fitted_rate,
                status=SelectionStatus.SELECTED,
                selected_margin_40ths=selected,
                estimated_conditional_null_family_alarm=estimated_risk,
                actual_conditional_null_family_alarm=actual_null,
                actual_conditional_alternative_detection=_actual_alternative_detection(
                    b, selected
                ),
                actual_conditional_null_limit_violated=(
                    actual_null > _known._CONDITIONAL_LIMIT
                ),
            )
        )

    # Normalize only to remove double-precision PMF summation error; it does
    # not change the reference distribution or condition on activation.
    original_reference = _known._binomial_pmf(_known._REFERENCE_N)
    normalization = math.fsum(original_reference)
    reference = tuple(mass / normalization for mass in original_reference)
    selection_rows = tuple(decisions)
    masses = (
        ReferencePolicyDecisionMass(
            status=SelectionStatus.NO_METHOD_PASSES,
            selected_margin_40ths=None,
            probability_mass=math.fsum(
                reference[b]
                for b, row in enumerate(selection_rows)
                if row.status is SelectionStatus.NO_METHOD_PASSES
            ),
        ),
        *(
            ReferencePolicyDecisionMass(
                status=SelectionStatus.SELECTED,
                selected_margin_40ths=margin,
                probability_mass=math.fsum(
                    reference[b]
                    for b, row in enumerate(selection_rows)
                    if row.selected_margin_40ths == margin
                ),
            )
            for margin in margins
        ),
    )
    withheld = masses[0].probability_mass
    activation = 1.0 - withheld
    marginal_null = math.fsum(
        reference[b] * row.actual_conditional_null_family_alarm
        for b, row in enumerate(selection_rows)
        if row.actual_conditional_null_family_alarm is not None
    )
    marginal_alternative = math.fsum(
        reference[b] * row.actual_conditional_alternative_detection
        for b, row in enumerate(selection_rows)
        if row.actual_conditional_alternative_detection is not None
    )
    violating_mass = math.fsum(
        reference[b]
        for b, row in enumerate(selection_rows)
        if row.actual_conditional_null_limit_violated is True
    )
    return EstimatedNullSensitivityEvaluation(
        candidate_margins_40ths=margins,
        decisions_by_reference_count=selection_rows,
        decision_probability_masses=masses,
        monitoring_activation_probability=activation,
        monitoring_withheld_probability=withheld,
        marginal_actual_null_family_alarm=marginal_null,
        marginal_actual_alternative_detection=marginal_alternative,
        violating_reference_probability_mass=violating_mass,
        violating_reference_condition_count=sum(
            row.actual_conditional_null_limit_violated is True for row in selection_rows
        ),
    )
