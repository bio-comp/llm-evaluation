"""D-019 development-only fitted-null policy comparison with frozen reference.

One B ~ Binomial(40, .20) is observed and frozen for six independent
K ~ Binomial(20, p) monitoring looks. Selection never resamples B.
Only the explicitly labelled oracle arm may use the true p0 to select.
The uncertainty-aware arms invert a one-sided exact binomial test: for
B < 40, upper U solves P_U(X <= B) = 1-confidence, X ~ Binomial(40, U).
The worst-case six-look null alarm over [0, U] occurs at U by binomial
stochastic monotonicity. Confidence coverage is marginal over the fitted
reference, NOT a guaranteed conditional risk bound at every realized B.

All reported null/detection events are conditional on activated monitoring;
an inactive arm has NO observed monitoring event and thus null-valued
conditional event risks. No FINAL experiment or deployment is authorized.
"""

from __future__ import annotations

import math
from enum import StrEnum
from functools import lru_cache
from typing import Sequence

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from . import fitted_monitor_diagnostic as _known
from .fitted_monitor_diagnostic import (
    SelectionStatus,
    _horizon_false_alarm,
    _strict_alarm,
    _validated_frozen_policy_candidates,
)

POLICY_DESIGN_ID = "d019-estimated-null-policy-comparison-dev-v1"
_CANDIDATES = (10, 12, 14, 16)
_CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)
_CP_BISECTION_ITERATIONS = 90


class PolicyName(StrEnum):
    """Prespecified arms; the final arm emphasizes abstention trade-offs."""

    KNOWN_NULL_ORACLE = "known-null-oracle"
    NAIVE_PLUGIN = "naive-plugin"
    UPPER_CP_90 = "upper-cp-90"
    UPPER_CP_95 = "upper-cp-95"
    UPPER_CP_99_ABSTAIN = "upper-cp-99-abstain"


@dataclass(config=ConfigDict(strict=True, extra="forbid", frozen=True), slots=True)
class PolicyDecision:
    """One fitted-reference condition, including a withheld-monitoring state."""

    reference_errors: int
    status: SelectionStatus
    selected_margin_40ths: int | None
    selection_null_probability: float
    selection_risk: float | None
    minimum_candidate_selection_risk: float
    actual_null_six_look_risk: float | None
    actual_alternative_six_look_detection: float | None
    conditional_limit_violated: bool | None


@dataclass(config=ConfigDict(strict=True, extra="forbid", frozen=True), slots=True)
class PolicyEvaluation:
    """Marginal outcomes integrate over B after conditional horizon events."""

    name: PolicyName
    confidence_level: float | None
    decisions: tuple[PolicyDecision, ...]
    activation_probability: float
    withheld_probability: float
    marginal_null_alarm: float
    marginal_alternative_detection: float
    null_alarm_given_activation: float | None
    detection_given_activation: float | None
    violating_reference_count: int
    violating_reference_mass: float
    confidence_noncoverage_mass_at_true_null: float | None
    margin_probability_masses: tuple[float, ...]


@dataclass(config=ConfigDict(strict=True, extra="forbid", frozen=True), slots=True)
class PolicyComparison:
    design_id: str
    candidate_margins_40ths: tuple[int, ...]
    reference_n: int
    reference_true_null_probability: float
    monitoring_n: int
    monitoring_true_null_probability: float
    monitoring_alternative_probability: float
    monitoring_looks: int
    conditional_null_limit: float
    cp_bisection_iterations: int
    policies: tuple[PolicyEvaluation, ...]


def _binomial_cdf(count: int, trials: int, probability: float) -> float:
    """CDF by nonnegative mass summation, including p=0 and p=1."""
    return math.fsum(
        math.comb(trials, k) * probability**k * (1.0 - probability) ** (trials - k)
        for k in range(count + 1)
    )


def _cp_upper_bound(reference_errors: int, confidence_level: float) -> float:
    """Validate *before* cached computation; bool and 1 share Python cache keys."""
    if type(reference_errors) is not int:
        raise TypeError("reference_errors must be an exact integer")
    if not 0 <= reference_errors <= _known._REFERENCE_N:
        raise ValueError("reference_errors outside frozen reference support")
    if type(confidence_level) is not float or confidence_level not in (
        *_CONFIDENCE_LEVELS,
    ):
        raise ValueError("confidence_level must be a prespecified exact float")
    return _cp_upper_bound_cached(reference_errors, confidence_level)


@lru_cache(maxsize=None)
def _cp_upper_bound_cached(reference_errors: int, confidence_level: float) -> float:
    """Invert an exact binomial test numerically, using 90 bisection updates.

    Finite binary64 precision eventually stops bracket refinement; the returned
    upper endpoint is a conservative numerical approximation to the CP root.
    """
    if reference_errors == _known._REFERENCE_N:
        return 1.0

    alpha = 1.0 - confidence_level
    low, high = 0.0, 1.0
    for _ in range(_CP_BISECTION_ITERATIONS):
        mid = (low + high) / 2.0
        if _binomial_cdf(reference_errors, _known._REFERENCE_N, mid) > alpha:
            low = mid
        else:
            high = mid
    # The upper endpoint is the conservative side of the monotone inversion.
    return high


def _conditional_event(
    reference_errors: int,
    margin_40ths: int,
    weekly_probability: float,
) -> float:
    masses = (
        math.comb(_known._WEEKLY_N, k)
        * weekly_probability**k
        * (1.0 - weekly_probability) ** (_known._WEEKLY_N - k)
        for k in range(_known._WEEKLY_N + 1)
        if _strict_alarm(reference_errors, k, margin_40ths)
    )
    return _horizon_false_alarm(math.fsum(masses))


def _selection_probability(b: int, name: PolicyName) -> float:
    if name is PolicyName.KNOWN_NULL_ORACLE:
        return _known._TRUE_ERROR_RATE
    if name is PolicyName.NAIVE_PLUGIN:
        return b / _known._REFERENCE_N
    return _cp_upper_bound(b, _confidence_level(name))


def _confidence_level(name: PolicyName) -> float:
    if name is PolicyName.UPPER_CP_90:
        return 0.90
    if name is PolicyName.UPPER_CP_95:
        return 0.95
    if name is PolicyName.UPPER_CP_99_ABSTAIN:
        return 0.99
    raise ValueError("only confidence-bound policies have confidence levels")


def _evaluate_policy(
    name: PolicyName,
    margins: tuple[int, ...],
    reference: tuple[float, ...],
) -> PolicyEvaluation:
    decisions: list[PolicyDecision] = []
    true_null_by_margin = {
        margin: _known.evaluate_margin(
            margin
        ).conditional_family_false_alarm_by_reference_count
        for margin in margins
    }
    alternative_by_margin = {
        margin: _known.evaluate_detection_power(
            margin
        ).conditional_family_detection_probability_by_reference_count
        for margin in margins
    }
    for b in range(_known._REFERENCE_N + 1):
        selection_p = _selection_probability(b, name)
        candidates = tuple(
            (margin, _conditional_event(b, margin, selection_p)) for margin in margins
        )
        chosen = next(
            (
                (margin, risk)
                for margin, risk in candidates
                if risk <= _known._CONDITIONAL_LIMIT
            ),
            None,
        )
        if chosen is None:
            decisions.append(
                PolicyDecision(
                    reference_errors=b,
                    status=SelectionStatus.NO_METHOD_PASSES,
                    selected_margin_40ths=None,
                    selection_null_probability=selection_p,
                    selection_risk=None,
                    minimum_candidate_selection_risk=candidates[-1][1],
                    actual_null_six_look_risk=None,
                    actual_alternative_six_look_detection=None,
                    conditional_limit_violated=None,
                )
            )
            continue
        margin, bound_risk = chosen
        realized_null = true_null_by_margin[margin][b]
        decisions.append(
            PolicyDecision(
                reference_errors=b,
                status=SelectionStatus.SELECTED,
                selected_margin_40ths=margin,
                selection_null_probability=selection_p,
                selection_risk=bound_risk,
                minimum_candidate_selection_risk=candidates[-1][1],
                actual_null_six_look_risk=realized_null,
                actual_alternative_six_look_detection=alternative_by_margin[margin][b],
                conditional_limit_violated=realized_null > _known._CONDITIONAL_LIMIT,
            )
        )
    rows = tuple(decisions)
    activation = math.fsum(
        reference[b]
        for b, row in enumerate(rows)
        if row.status is SelectionStatus.SELECTED
    )
    withheld = math.fsum(
        reference[b]
        for b, row in enumerate(rows)
        if row.status is SelectionStatus.NO_METHOD_PASSES
    )
    null = math.fsum(
        reference[b] * row.actual_null_six_look_risk
        for b, row in enumerate(rows)
        if row.actual_null_six_look_risk is not None
    )
    power = math.fsum(
        reference[b] * row.actual_alternative_six_look_detection
        for b, row in enumerate(rows)
        if row.actual_alternative_six_look_detection is not None
    )
    violations = tuple(
        b for b, row in enumerate(rows) if row.conditional_limit_violated is True
    )
    confidence = (
        None
        if name in (PolicyName.KNOWN_NULL_ORACLE, PolicyName.NAIVE_PLUGIN)
        else _confidence_level(name)
    )
    noncoverage = (
        None
        if confidence is None
        else math.fsum(
            reference[b]
            for b, row in enumerate(rows)
            if row.selection_null_probability < _known._TRUE_ERROR_RATE
        )
    )
    return PolicyEvaluation(
        name=name,
        confidence_level=confidence,
        decisions=rows,
        activation_probability=activation,
        withheld_probability=withheld,
        marginal_null_alarm=null,
        marginal_alternative_detection=power,
        null_alarm_given_activation=null / activation if activation else None,
        detection_given_activation=power / activation if activation else None,
        violating_reference_count=len(violations),
        violating_reference_mass=math.fsum(reference[b] for b in violations),
        confidence_noncoverage_mass_at_true_null=noncoverage,
        margin_probability_masses=tuple(
            math.fsum(
                reference[b]
                for b, row in enumerate(rows)
                if row.selected_margin_40ths == candidate
            )
            for candidate in (None, *margins)
        ),
    )


def evaluate_policy_comparison(
    candidates: Sequence[int],
) -> PolicyComparison:
    """Evaluate all prespecified arms on exactly the same fitted-reference law."""
    margins = _validated_frozen_policy_candidates(candidates)
    reference_raw = _known._binomial_pmf(_known._REFERENCE_N)
    normalizer = math.fsum(reference_raw)
    reference = tuple(weight / normalizer for weight in reference_raw)
    policies = tuple(_evaluate_policy(name, margins, reference) for name in PolicyName)
    return PolicyComparison(
        design_id=POLICY_DESIGN_ID,
        candidate_margins_40ths=margins,
        reference_n=_known._REFERENCE_N,
        reference_true_null_probability=_known._TRUE_ERROR_RATE,
        monitoring_n=_known._WEEKLY_N,
        monitoring_true_null_probability=_known._TRUE_ERROR_RATE,
        monitoring_alternative_probability=0.30,
        monitoring_looks=_known._LOOKS,
        conditional_null_limit=_known._CONDITIONAL_LIMIT,
        cp_bisection_iterations=_CP_BISECTION_ITERATIONS,
        policies=policies,
    )
