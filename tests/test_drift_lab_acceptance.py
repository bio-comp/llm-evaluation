from __future__ import annotations

import pytest

from llm_evaluation.drift_lab_acceptance import evaluate_drift_lab_acceptance
from llm_evaluation.drift_lab_design import DriftLabDesign
from llm_evaluation.drift_lab_records import DriftOutcomeSummaryRecord


def passing_summary() -> DriftOutcomeSummaryRecord:
    return DriftOutcomeSummaryRecord(
        matched_policy_false_alarm={
            "psi_only": "0.05",
            "pearson_only": "0.05",
            "spc_only_ewma": "0.05",
            "spc_only_cusum": "0.05",
            "layered_raw": "0.05",
            "layered_adjusted_ewma": "0.05",
            "layered_adjusted_cusum": "0.05",
        },
        psi_heuristic_null_false_alarm={"100": "0.50", "200": "0.25", "1000": "0.01"},
        psi_calibrated_null_false_alarm={"100": "0.05", "200": "0.05", "1000": "0.05"},
        distribution_only_adjusted_process_alarm="0.08",
        distribution_only_raw_process_alarm="0.90",
        process_only_adjusted_distribution_alarm="0.05",
        process_only_raw_distribution_alarm="0.05",
        adjusted_typed_accuracy={
            "stable/stable": "0.95",
            "distribution_only": "0.95",
            "process_only": "0.85",
            "both": "0.85",
        },
        raw_typed_accuracy={
            "stable/stable": "0.75",
            "distribution_only": "0.60",
            "process_only": "0.75",
            "both": "0.75",
        },
        variable_denominator_fixed_n_shewhart_false_alarm="0.25",
        variable_denominator_adjusted_shewhart_false_alarm="0.05",
        overdispersion_adjusted_ewma_false_alarm="0.15",
        iid_adjusted_ewma_false_alarm="0.05",
        localized_aggregate_detection="0.60",
        localized_b_detection="0.90",
    )


def test_all_frozen_acceptance_gates_pass_for_conforming_summary() -> None:
    result = evaluate_drift_lab_acceptance(summary=passing_summary(), design=DriftLabDesign())
    assert result.passed is True
    assert result.failures == ()
    assert result.diagnostics["typed_macro_advantage"] == pytest.approx(0.1875)
    assert result.diagnostics["psi_heuristic_null_spread"] == pytest.approx(0.49)
    assert result.diagnostics["variable_denominator_false_alarm_gap"] == pytest.approx(0.20)
    assert result.diagnostics["overdispersion_false_alarm_inflation"] == pytest.approx(0.10)


@pytest.mark.parametrize(
    "field,value,expected_fragment",
    [
        ("matched_policy_false_alarm", {"psi_only": "0.07"}, "matched-budget"),
        ("psi_heuristic_null_false_alarm", {"100": "0.10", "200": "0.10", "1000": "0.10"}, "PSI heuristic"),
        ("psi_calibrated_null_false_alarm", {"100": "0.07", "200": "0.05", "1000": "0.05"}, "calibrated PSI"),
        ("distribution_only_adjusted_process_alarm", "0.11", "adjusted process"),
        ("distribution_only_raw_process_alarm", "0.79", "raw process"),
        ("adjusted_typed_accuracy", {"stable/stable": "0.89", "distribution_only": "0.95", "process_only": "0.85", "both": "0.85"}, "typed accuracy"),
        ("raw_typed_accuracy", {"stable/stable": "0.88", "distribution_only": "0.88", "process_only": "0.80", "both": "0.80"}, "macro advantage"),
        ("variable_denominator_fixed_n_shewhart_false_alarm", "0.14", "variable-denominator"),
        ("overdispersion_adjusted_ewma_false_alarm", "0.09", "overdispersion"),
        ("localized_aggregate_detection", "0.95", "localized"),
    ],
)
def test_each_frozen_gate_can_fail_independently(
    field: str, value: object, expected_fragment: str
) -> None:
    base = passing_summary().model_dump()
    if isinstance(value, dict) and field in {
        "matched_policy_false_alarm",
        "psi_heuristic_null_false_alarm",
        "psi_calibrated_null_false_alarm",
    }:
        base[field].update(value)
    else:
        base[field] = value
    summary = DriftOutcomeSummaryRecord.model_validate(base, strict=True)
    result = evaluate_drift_lab_acceptance(summary=summary, design=DriftLabDesign())
    assert result.passed is False
    assert any(expected_fragment in failure for failure in result.failures)
