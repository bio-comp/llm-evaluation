from llm_evaluation.metric_lab_acceptance import metric_bias_within_tolerance


def test_metric_bias_tolerance_uses_mcse_or_absolute_floor_whichever_is_larger() -> None:
    assert metric_bias_within_tolerance(
        bias=3.9,
        bias_mc_se=1.0,
        mcse_multiplier=4.0,
        absolute_floor=1e-10,
    )
    assert not metric_bias_within_tolerance(
        bias=4.1,
        bias_mc_se=1.0,
        mcse_multiplier=4.0,
        absolute_floor=1e-10,
    )
    assert metric_bias_within_tolerance(
        bias=5e-11,
        bias_mc_se=1e-16,
        mcse_multiplier=4.0,
        absolute_floor=1e-10,
    )
    assert not metric_bias_within_tolerance(
        bias=2e-10,
        bias_mc_se=1e-16,
        mcse_multiplier=4.0,
        absolute_floor=1e-10,
    )


def test_acceptance_rejects_development_run_before_numeric_checks() -> None:
    from llm_evaluation.metric_lab_acceptance import evaluate_metric_lab_acceptance
    from llm_evaluation.metric_lab_design import BaseCellSpec, MetricLabDesign, MetricLabPhase
    from llm_evaluation.metric_lab_records import preview_metric_lab_run_record
    from llm_evaluation.metric_lab_runner import summarize_base_cell
    from llm_evaluation.metric_lab_targets import SamplingUnit

    design = MetricLabDesign(
        p_values=("0.5",),
        forecast_kinds=("constant",),
        n_base_items=(10,),
        k_within_item=(1,),
        reference_flip_etas=("0",),
        ambiguity_theta_values=("0.5",),
        ambiguity_n_items=(10,),
        ambiguity_raters_per_item=(1,),
        prevalence_values=("0.5",),
        fixed_bin_count=2,
        equal_mass_bin_count=2,
    )
    cell = BaseCellSpec(
        forecast_kind="constant",
        n_base_items=10,
        k_within_item=1,
        shared_within_item=False,
        reference_flip_eta=0.0,
    )
    summary = summarize_base_cell(
        design=design,
        cell=cell,
        phase=MetricLabPhase.DEVELOPMENT,
        sampling_unit=SamplingUnit.FIXED_CASE,
        num_replicates=5,
    )
    run = preview_metric_lab_run_record(
        design=design,
        phase=MetricLabPhase.DEVELOPMENT,
        cell_summaries=(summary,),
    )
    result = evaluate_metric_lab_acceptance(run=run, design=design)
    assert result.passed is False
    assert "phase must be final" in result.failures


def test_acceptance_can_pass_complete_small_final_under_explicit_test_policy() -> None:
    from llm_evaluation.metric_lab_acceptance import evaluate_metric_lab_acceptance
    from llm_evaluation.metric_lab_design import (
        MetricLabAcceptanceCriteria,
        MetricLabDesign,
        MetricLabPhase,
        ambiguity_grid_cells,
        base_grid_cells,
        freeze_metric_lab_design,
        question_population_stress_cells,
    )
    from llm_evaluation.metric_lab_records import preview_metric_lab_run_record
    from llm_evaluation.metric_lab_runner import summarize_ambiguity_cell, summarize_base_cell
    from llm_evaluation.metric_lab_targets import SamplingUnit

    lenient = MetricLabAcceptanceCriteria(
        analytic_bias_mcse_multiplier="1000",
        analytic_bias_absolute_floor="1",
        population_squared_l2_bias_mcse_multiplier="1000",
        shared_cluster_coverage_mean_min="0",
        shared_cluster_coverage_mean_max="1",
        shared_cluster_coverage_cell_min="0",
        shared_naive_cluster_coverage_gap_min="0",
        independent_cluster_coverage_mean_min="0",
        independent_cluster_coverage_mean_max="1",
        question_population_cluster_coverage_mean_min="0",
        question_population_cluster_coverage_mean_max="1",
        hoeffding_coverage_mean_min="0",
        cluster_to_hoeffding_width_ratio_max="1",
    )
    design = MetricLabDesign(
        acceptance=lenient,
        p_values=("0.5",),
        forecast_kinds=("constant",),
        n_base_items=(10,),
        k_within_item=(1, 4),
        reference_flip_etas=("0",),
        ambiguity_theta_values=("0.5",),
        ambiguity_n_items=(10,),
        ambiguity_raters_per_item=(1,),
        prevalence_values=("0.5",),
        fixed_bin_count=2,
        equal_mass_bin_count=2,
    )
    summaries = []
    for cell in base_grid_cells(design):
        summaries.append(
            summarize_base_cell(
                design=design,
                cell=cell,
                phase=MetricLabPhase.FINAL,
                sampling_unit=SamplingUnit.FIXED_CASE,
                num_replicates=2000,
            )
        )
    for cell in question_population_stress_cells(design):
        summaries.append(
            summarize_base_cell(
                design=design,
                cell=cell,
                phase=MetricLabPhase.FINAL,
                sampling_unit=SamplingUnit.QUESTION_POPULATION,
                num_replicates=2000,
            )
        )
    for cell in ambiguity_grid_cells(design):
        summaries.append(
            summarize_ambiguity_cell(
                design=design,
                cell=cell,
                phase=MetricLabPhase.FINAL,
                num_replicates=2000,
            )
        )
    run = preview_metric_lab_run_record(
        design=design,
        phase=MetricLabPhase.FINAL,
        cell_summaries=tuple(summaries),
        frozen_design=freeze_metric_lab_design(design),
    ).model_copy(
        update={"software_commit": "a" * 40, "package_source_sha256": "b" * 64}
    )
    result = evaluate_metric_lab_acceptance(run=run, design=design)
    assert result.passed is True, result.failures
