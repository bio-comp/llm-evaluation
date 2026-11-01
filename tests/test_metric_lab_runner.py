import math

from canonical_sync.codec import canonical_record_hash
from llm_evaluation.metric_lab_design import (
    BaseCellSpec,
    MetricLabDesign,
    MetricLabPhase,
)
from llm_evaluation.metric_lab_runner import (
    base_target_truth,
    evaluate_base_replicate,
    mean_loss_intervals,
    official_get_aurc_reference_behavior,
    risk_coverage_tie_blocks,
    summarize_base_cell,
    summarize_metric_estimates,
)
from llm_evaluation.metric_lab_targets import SamplingUnit, TargetKind
from llm_evaluation.metric_laboratory import mean_prefix_risk


def _single_p_design(p: str = "0.9") -> MetricLabDesign:
    return MetricLabDesign(
        p_values=(p,),
        forecast_kinds=("oracle",),
        n_base_items=(10,),
        k_within_item=(1, 2, 4),
        shared_within_item=(False, True),
        reference_flip_etas=("0", "0.2"),
        fixed_bin_count=2,
        equal_mass_bin_count=2,
    )


def test_official_aurc_negative_control_reproduces_exact_upstream_defect() -> None:
    residuals = (0.0, 1.0)
    confidence = (0.1, 0.9)
    upstream_behavior = official_get_aurc_reference_behavior(residuals, confidence)
    exact_prefix_oracle = mean_prefix_risk(confidence, residuals)
    assert upstream_behavior == 0.5
    assert exact_prefix_oracle == 0.75
    assert upstream_behavior != exact_prefix_oracle


def test_base_truth_is_target_indexed_and_matches_analytic_brier() -> None:
    design = _single_p_design("0.9")
    cell = BaseCellSpec(
        forecast_kind="oracle",
        n_base_items=10,
        k_within_item=1,
        shared_within_item=False,
        reference_flip_eta=0.2,
    )
    clean = base_target_truth(
        design=design,
        cell=cell,
        target=TargetKind.CLEAN_TRUTH,
        sampling_unit=SamplingUnit.FIXED_CASE,
    )
    noisy = base_target_truth(
        design=design,
        cell=cell,
        target=TargetKind.NOISY_REFERENCE,
        sampling_unit=SamplingUnit.FIXED_CASE,
    )
    assert math.isclose(clean.expected_brier, 0.09, rel_tol=1e-12)
    assert math.isclose(noisy.expected_brier, 0.218, rel_tol=1e-12)
    assert math.isclose(clean.classification_error, 0.1, rel_tol=1e-12)
    assert math.isclose(noisy.classification_error, 0.26, rel_tol=1e-12)
    assert math.isclose(clean.fixed_bin_calibration, 0.0, abs_tol=1e-12)
    assert math.isclose(noisy.fixed_bin_calibration, 0.16, rel_tol=1e-12)


def test_base_replicate_preserves_target_identity_reliability_counts_and_budget() -> (
    None
):
    design = _single_p_design("0.9")
    cell = BaseCellSpec(
        forecast_kind="oracle",
        n_base_items=10,
        k_within_item=4,
        shared_within_item=True,
        reference_flip_eta=0.2,
    )
    result = evaluate_base_replicate(
        design=design,
        cell=cell,
        phase=MetricLabPhase.DEVELOPMENT,
        replicate_index=3,
        sampling_unit=SamplingUnit.FIXED_CASE,
    )
    assert result.design_sha256 == canonical_record_hash(design)
    assert result.item_count == 10
    assert result.row_count == 40
    assert result.realized_budget_rows == 40
    assert set(result.targets) == {TargetKind.CLEAN_TRUTH, TargetKind.NOISY_REFERENCE}
    for target_result in result.targets.values():
        assert sum(target_result.fixed_bin_counts) == 40
        assert sum(target_result.equal_mass_bin_counts) == 40
        assert target_result.risk_coverage[-1].coverage == 1.0
        assert target_result.risk_coverage[-1].selected_count == 40


def test_cluster_interval_is_wider_than_naive_for_perfectly_shared_repeats() -> None:
    values = tuple([0.0] * 25 + [0.0] * 25 + [1.0] * 25 + [1.0] * 25)
    item_ids = tuple([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
    intervals = mean_loss_intervals(values=values, item_ids=item_ids)
    assert intervals.cluster.se > intervals.naive.se
    assert intervals.cluster.width > intervals.naive.width
    assert math.isclose(intervals.cluster.estimate, 0.5, rel_tol=1e-12)
    assert math.isclose(intervals.naive.estimate, 0.5, rel_tol=1e-12)


def test_metric_summary_computes_bias_rmse_mcse_coverage_and_width_exactly() -> None:
    summary = summarize_metric_estimates(
        estimates=(1.0, 2.0, 3.0),
        true_value=2.0,
        intervals=((1.5, 2.5), (0.0, 1.0), (1.0, 3.0)),
    )
    assert summary.mean_estimate == 2.0
    assert summary.bias == 0.0
    assert math.isclose(summary.rmse, math.sqrt(2 / 3), rel_tol=1e-12)
    assert math.isclose(summary.interval_coverage, 2 / 3, rel_tol=1e-12)
    assert math.isclose(summary.mean_interval_width, 4 / 3, rel_tol=1e-12)
    assert summary.mc_se > 0.0


def test_risk_coverage_curve_never_splits_a_confidence_tie_block() -> None:
    points = risk_coverage_tie_blocks(
        confidences=(0.9, 0.9, 0.7),
        errors=(0.0, 1.0, 1.0),
    )
    assert [point.selected_count for point in points] == [2, 3]
    assert math.isclose(points[0].coverage, 2 / 3, rel_tol=1e-12)
    assert math.isclose(points[0].risk, 0.5, rel_tol=1e-12)
    assert points[-1].coverage == 1.0
    assert math.isclose(points[-1].risk, 2 / 3, rel_tol=1e-12)


def test_cell_summary_uses_known_truth_and_reports_realized_budget() -> None:
    # q=0.5 has exactly Brier=0.25 and log=ln(2) for every binary outcome,
    # making this a deterministic oracle for the aggregation machinery.
    design = MetricLabDesign(
        p_values=("0.5",),
        forecast_kinds=("constant",),
        n_base_items=(10,),
        k_within_item=(2,),
        shared_within_item=(False, True),
        reference_flip_etas=("0",),
        fixed_bin_count=2,
        equal_mass_bin_count=2,
    )
    cell = BaseCellSpec(
        forecast_kind="constant",
        n_base_items=10,
        k_within_item=2,
        shared_within_item=False,
        reference_flip_eta=0.0,
    )
    summary = summarize_base_cell(
        design=design,
        cell=cell,
        phase=MetricLabPhase.DEVELOPMENT,
        sampling_unit=SamplingUnit.FIXED_CASE,
        num_replicates=20,
    )
    clean = summary.targets[TargetKind.CLEAN_TRUTH]
    assert clean.metrics["brier"].true_value == 0.25
    assert clean.metrics["brier"].mean_estimate == 0.25
    assert clean.metrics["brier"].bias == 0.0
    assert clean.metrics["brier"].rmse == 0.0
    assert math.isclose(
        clean.metrics["log_score"].mean_estimate, math.log(2), rel_tol=1e-12
    )
    assert summary.replicates_count == 20
    assert summary.realized_budget_rows == 20 * 10 * 2


def test_population_truth_differs_from_small_fixed_case_composition_when_unbalanced() -> (
    None
):
    design = MetricLabDesign(
        p_values=("0.1", "0.9"),
        forecast_kinds=("constant",),
        n_base_items=(3,),
        k_within_item=(1,),
        shared_within_item=(False, True),
        reference_flip_etas=("0",),
        fixed_bin_count=1,
        equal_mass_bin_count=1,
    )
    cell = BaseCellSpec(
        forecast_kind="constant",
        n_base_items=3,
        k_within_item=1,
        shared_within_item=False,
        reference_flip_eta=0.0,
    )
    fixed = base_target_truth(
        design=design,
        cell=cell,
        target=TargetKind.CLEAN_TRUTH,
        sampling_unit=SamplingUnit.FIXED_CASE,
    )
    population = base_target_truth(
        design=design,
        cell=cell,
        target=TargetKind.CLEAN_TRUTH,
        sampling_unit=SamplingUnit.QUESTION_POPULATION,
    )
    assert fixed.case_probabilities != population.case_probabilities
    assert len(fixed.case_probabilities) == 3
    assert population.case_probabilities == (0.1, 0.9)


def test_ambiguity_truth_distinguishes_rater_distribution_soft_and_consensus() -> None:
    from llm_evaluation.metric_lab_design import AmbiguityCellSpec
    from llm_evaluation.metric_lab_runner import ambiguity_target_truths

    design = MetricLabDesign(
        p_values=("0.9",),
        forecast_kinds=("oracle",),
        n_base_items=(10,),
        k_within_item=(1,),
        shared_within_item=(False, True),
        reference_flip_etas=("0",),
        ambiguity_theta_values=("0.9",),
        ambiguity_n_items=(10,),
        ambiguity_raters_per_item=(3,),
        prevalence_values=("0.5",),
    )
    cell = AmbiguityCellSpec(forecast_kind="oracle", n_items=10, raters_per_item=3)
    truths = ambiguity_target_truths(design=design, cell=cell)
    rater = truths[TargetKind.RATER_DISTRIBUTION]
    soft = truths[TargetKind.SOFT_LABEL]
    consensus = truths[TargetKind.MAJORITY_CONSENSUS]

    assert math.isclose(rater.expected_brier, 0.09, rel_tol=1e-12)
    # q == theta, so squared loss against the sample mean is exactly its sampling variance.
    assert math.isclose(soft.expected_brier, 0.9 * 0.1 / 3, rel_tol=1e-12)
    majority_p = 3 * (0.9**2) * 0.1 + 0.9**3
    expected_consensus_brier = (0.9 - majority_p) ** 2 + majority_p * (1 - majority_p)
    assert math.isclose(
        consensus.expected_brier, expected_consensus_brier, rel_tol=1e-12
    )
    assert (
        len({rater.expected_brier, soft.expected_brier, consensus.expected_brier}) == 3
    )


def test_ambiguity_replicate_reports_target_specific_evaluation_counts_and_budget() -> (
    None
):
    from llm_evaluation.metric_lab_design import AmbiguityCellSpec
    from llm_evaluation.metric_lab_runner import evaluate_ambiguity_replicate

    design = MetricLabDesign(
        ambiguity_theta_values=("0.1", "0.5", "0.9"),
        ambiguity_n_items=(9,),
        ambiguity_raters_per_item=(3,),
    )
    cell = AmbiguityCellSpec(forecast_kind="oracle", n_items=9, raters_per_item=3)
    result = evaluate_ambiguity_replicate(
        design=design,
        cell=cell,
        phase=MetricLabPhase.DEVELOPMENT,
        replicate_index=4,
    )
    assert result.item_count == 9
    assert result.realized_budget_ratings == 27
    assert result.targets[TargetKind.RATER_DISTRIBUTION].evaluation_count == 27
    assert result.targets[TargetKind.SOFT_LABEL].evaluation_count == 9
    assert result.targets[TargetKind.MAJORITY_CONSENSUS].evaluation_count == 9
    assert all(
        target.risk_coverage[-1].coverage == 1.0 for target in result.targets.values()
    )


def test_ambiguity_summary_reports_known_target_truths_and_rating_budget() -> None:
    from llm_evaluation.metric_lab_design import AmbiguityCellSpec
    from llm_evaluation.metric_lab_runner import summarize_ambiguity_cell

    design = MetricLabDesign(
        ambiguity_theta_values=("0.5",),
        ambiguity_n_items=(8,),
        ambiguity_raters_per_item=(3,),
        fixed_bin_count=1,
        equal_mass_bin_count=1,
    )
    cell = AmbiguityCellSpec(forecast_kind="constant", n_items=8, raters_per_item=3)
    summary = summarize_ambiguity_cell(
        design=design,
        cell=cell,
        phase=MetricLabPhase.DEVELOPMENT,
        num_replicates=20,
    )
    assert summary.replicates_count == 20
    assert summary.realized_budget_ratings == 20 * 8 * 3
    rater = summary.targets[TargetKind.RATER_DISTRIBUTION]
    assert rater.metrics["brier"].true_value == 0.25
    assert rater.metrics["brier"].mean_estimate == 0.25
    assert rater.metrics["brier"].bias == 0.0
