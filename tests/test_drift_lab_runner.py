from __future__ import annotations

import pytest

from llm_evaluation.drift_lab_design import DriftLabDesign, DriftLabPhase
from llm_evaluation.drift_lab_runner import (
    beta_parameters_from_mean_icc,
    generate_path,
    scenario_parameters,
    variable_denominator_for_week,
)


def test_primary_scenario_parameters_encode_exact_truth_states() -> None:
    design = DriftLabDesign()
    stable = scenario_parameters(design, "stable/stable", week=13)
    distribution = scenario_parameters(design, "distribution_only", week=13)
    process = scenario_parameters(design, "process_only", week=13)
    both = scenario_parameters(design, "both", week=13)

    assert stable == pytest.approx((0.20, 0.05, 0.20))
    assert distribution == pytest.approx((0.60, 0.05, 0.20))
    assert process == pytest.approx((0.20, 0.10, 0.25))
    assert both == pytest.approx((0.60, 0.10, 0.25))
    assert scenario_parameters(design, "both", week=12) == pytest.approx((0.20, 0.05, 0.20))


def test_variable_denominator_cycle_is_exact_and_one_indexed() -> None:
    design = DriftLabDesign()
    assert tuple(variable_denominator_for_week(design, week=i) for i in range(1, 9)) == (
        50,
        100,
        200,
        400,
        50,
        100,
        200,
        400,
    )


def test_beta_parameters_preserve_requested_mean_and_intraclass_correlation() -> None:
    alpha, beta = beta_parameters_from_mean_icc(mean=0.20, icc=0.05)
    recovered_mean = alpha / (alpha + beta)
    recovered_icc = 1.0 / (alpha + beta + 1.0)
    assert recovered_mean == pytest.approx(0.20, abs=1e-15)
    assert recovered_icc == pytest.approx(0.05, abs=1e-15)


def test_generated_path_is_reproducible_and_phase_separated() -> None:
    design = DriftLabDesign()
    first = generate_path(
        design,
        phase=DriftLabPhase.DEVELOPMENT,
        scenario="both",
        replicate_index=3,
    )
    second = generate_path(
        design,
        phase=DriftLabPhase.DEVELOPMENT,
        scenario="both",
        replicate_index=3,
    )
    final = generate_path(
        design,
        phase=DriftLabPhase.FINAL,
        scenario="both",
        replicate_index=3,
    )
    assert first == second
    assert first != final
    assert len(first.weeks) == 52
    assert first.baseline.total_cases == 4000
    assert all(week.total_cases == 200 for week in first.weeks)
    assert first.weeks[11].truth_b_probability == pytest.approx(0.20)
    assert first.weeks[12].truth_b_probability == pytest.approx(0.60)
    assert first.weeks[12].truth_a_error == pytest.approx(0.10)
    assert first.weeks[12].truth_b_error == pytest.approx(0.25)


def test_stress_scenario_timing_is_exact() -> None:
    design = DriftLabDesign()
    from llm_evaluation.drift_lab_runner import stress_scenario_parameters

    assert stress_scenario_parameters(design, "gradual_process", week=12) == pytest.approx(
        (0.20, 0.05, 0.20)
    )
    assert stress_scenario_parameters(design, "gradual_process", week=13) == pytest.approx(
        (0.20, 0.0525, 0.2025)
    )
    assert stress_scenario_parameters(design, "gradual_process", week=32) == pytest.approx(
        (0.20, 0.10, 0.25)
    )
    assert stress_scenario_parameters(design, "gradual_process", week=40) == pytest.approx(
        (0.20, 0.10, 0.25)
    )

    for week in (13, 14, 15, 16):
        assert stress_scenario_parameters(design, "temporary_process", week=week) == pytest.approx(
            (0.20, 0.15, 0.30)
        )
    assert stress_scenario_parameters(design, "temporary_process", week=17) == pytest.approx(
        (0.20, 0.05, 0.20)
    )

    assert stress_scenario_parameters(design, "localized_b_process", week=13) == pytest.approx(
        (0.20, 0.05, 0.25)
    )


def test_variable_denominator_and_overdispersion_paths_are_reproducible() -> None:
    design = DriftLabDesign()
    variable = generate_path(
        design,
        phase=DriftLabPhase.DEVELOPMENT,
        scenario="variable_denominator_stable",
        replicate_index=11,
    )
    assert tuple(week.total_cases for week in variable.weeks[:8]) == (
        50,
        100,
        200,
        400,
        50,
        100,
        200,
        400,
    )

    over1 = generate_path(
        design,
        phase=DriftLabPhase.DEVELOPMENT,
        scenario="overdispersion_stable",
        replicate_index=11,
    )
    over2 = generate_path(
        design,
        phase=DriftLabPhase.DEVELOPMENT,
        scenario="overdispersion_stable",
        replicate_index=11,
    )
    assert over1 == over2
    assert any(
        week.truth_a_error != pytest.approx(0.05) or week.truth_b_error != pytest.approx(0.20)
        for week in over1.weeks
    )


def test_finite_baseline_estimate_recovers_observed_rates_and_fails_on_empty_stratum() -> None:
    from llm_evaluation.drift_lab_runner import StratumCounts, estimate_baseline

    estimate = estimate_baseline(StratumCounts(a_cases=800, b_cases=200, a_errors=40, b_errors=40))
    assert estimate.b_probability == pytest.approx(0.20)
    assert estimate.a_error_rate == pytest.approx(0.05)
    assert estimate.b_error_rate == pytest.approx(0.20)
    assert estimate.aggregate_error_rate == pytest.approx(0.08)

    with pytest.raises(ValueError, match="empty stratum"):
        estimate_baseline(StratumCounts(a_cases=1000, b_cases=0, a_errors=50, b_errors=0))


def test_path_statistics_preserve_distribution_vs_adjusted_process_separation() -> None:
    from llm_evaluation.drift_lab_runner import (
        GeneratedPath,
        StratumCounts,
        WeekObservation,
        compute_path_statistics,
    )

    design = DriftLabDesign()
    baseline = StratumCounts(a_cases=3200, b_cases=800, a_errors=160, b_errors=160)
    weeks = []
    for week in range(1, 53):
        if week < 13:
            weeks.append(
                WeekObservation(
                    a_cases=160,
                    b_cases=40,
                    a_errors=8,
                    b_errors=8,
                    week=week,
                    truth_b_probability=0.20,
                    truth_a_error=0.05,
                    truth_b_error=0.20,
                )
            )
        else:
            weeks.append(
                WeekObservation(
                    a_cases=80,
                    b_cases=120,
                    a_errors=4,
                    b_errors=24,
                    week=week,
                    truth_b_probability=0.60,
                    truth_a_error=0.05,
                    truth_b_error=0.20,
                )
            )
    path = GeneratedPath(
        phase=DriftLabPhase.DEVELOPMENT,
        scenario="distribution_only",
        replicate_index=0,
        seed=1,
        baseline=baseline,
        weeks=tuple(weeks),
    )
    stats = compute_path_statistics(design, path)
    assert stats.max_psi > 0.10
    assert stats.max_raw_aggregate_ewma > 3.0
    assert stats.max_adjusted_ewma == pytest.approx(0.0, abs=1e-15)
    assert stats.max_adjusted_cusum == pytest.approx(0.0, abs=1e-15)
    assert stats.max_adjusted_shewhart == pytest.approx(0.0, abs=1e-15)
