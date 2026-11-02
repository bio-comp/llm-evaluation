"""Executable frozen acceptance policy for E-004 Phase A."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from llm_evaluation.drift_lab_design import DriftLabDesign
from llm_evaluation.drift_lab_records import DriftOutcomeSummaryRecord, PRIMARY_STATES


@dataclass(frozen=True, slots=True)
class DriftAcceptanceResult:
    passed: bool
    failures: tuple[str, ...]
    diagnostics: dict[str, float]


def _f(value: str) -> float:
    return float(Decimal(value))


def evaluate_drift_lab_acceptance(
    *, summary: DriftOutcomeSummaryRecord, design: DriftLabDesign
) -> DriftAcceptanceResult:
    if not isinstance(summary, DriftOutcomeSummaryRecord):
        raise TypeError("summary must be a DriftOutcomeSummaryRecord")
    if not isinstance(design, DriftLabDesign):
        raise TypeError("design must be a DriftLabDesign")
    criteria = design.acceptance
    low = _f(criteria.matched_family_false_alarm_min)
    high = _f(criteria.matched_family_false_alarm_max)
    failures: list[str] = []

    for policy, text in summary.matched_policy_false_alarm.items():
        rate = _f(text)
        if not low <= rate <= high:
            failures.append(f"matched-budget policy {policy} false-alarm rate {rate:.6f} outside [{low}, {high}]")

    heuristic_values = tuple(_f(value) for value in summary.psi_heuristic_null_false_alarm.values())
    psi_spread = max(heuristic_values) - min(heuristic_values)
    if psi_spread < _f(criteria.psi_heuristic_null_spread_min):
        failures.append("PSI heuristic fixed-threshold null-size spread is too small")
    for n, text in summary.psi_calibrated_null_false_alarm.items():
        rate = _f(text)
        if not low <= rate <= high:
            failures.append(f"calibrated PSI n={n} false-alarm rate outside matched budget")

    if _f(summary.distribution_only_adjusted_process_alarm) > _f(
        criteria.distribution_only_adjusted_process_alarm_max
    ):
        failures.append("adjusted process alarm exceeds distribution-only upper bound")
    if _f(summary.distribution_only_raw_process_alarm) < _f(
        criteria.distribution_only_raw_process_alarm_min
    ):
        failures.append("raw process alarm does not expose distribution-only confounding")

    accuracy_minima = {
        "stable/stable": _f(criteria.typed_accuracy_stable_min),
        "distribution_only": _f(criteria.typed_accuracy_distribution_min),
        "process_only": _f(criteria.typed_accuracy_process_min),
        "both": _f(criteria.typed_accuracy_both_min),
    }
    for state in PRIMARY_STATES:
        if _f(summary.adjusted_typed_accuracy[state]) < accuracy_minima[state]:
            failures.append(f"typed accuracy for {state} is below the frozen minimum")
    adjusted_macro = sum(_f(summary.adjusted_typed_accuracy[state]) for state in PRIMARY_STATES) / 4.0
    raw_macro = sum(_f(summary.raw_typed_accuracy[state]) for state in PRIMARY_STATES) / 4.0
    macro_advantage = adjusted_macro - raw_macro
    if macro_advantage < _f(criteria.typed_macro_advantage_min):
        failures.append("typed macro advantage over layered_raw is below the frozen minimum")

    denominator_gap = _f(summary.variable_denominator_fixed_n_shewhart_false_alarm) - _f(
        summary.variable_denominator_adjusted_shewhart_false_alarm
    )
    if denominator_gap < _f(criteria.fixed_n_shewhart_false_alarm_gap_min):
        failures.append("variable-denominator fixed-n Shewhart falsifier did not separate")

    overdispersion_inflation = _f(summary.overdispersion_adjusted_ewma_false_alarm) - _f(
        summary.iid_adjusted_ewma_false_alarm
    )
    if overdispersion_inflation < _f(criteria.overdispersion_false_alarm_inflation_min):
        failures.append("overdispersion falsifier did not inflate iid-calibrated EWMA enough")

    localized_aggregate = _f(summary.localized_aggregate_detection)
    localized_b = _f(summary.localized_b_detection)
    if localized_b <= localized_aggregate:
        failures.append("localized B-stratum detector did not preserve a stronger localized signal")

    diagnostics = {
        "psi_heuristic_null_spread": psi_spread,
        "typed_adjusted_macro_accuracy": adjusted_macro,
        "typed_raw_macro_accuracy": raw_macro,
        "typed_macro_advantage": macro_advantage,
        "variable_denominator_false_alarm_gap": denominator_gap,
        "overdispersion_false_alarm_inflation": overdispersion_inflation,
        "localized_detection_advantage": localized_b - localized_aggregate,
    }
    return DriftAcceptanceResult(
        passed=not failures,
        failures=tuple(failures),
        diagnostics=diagnostics,
    )
