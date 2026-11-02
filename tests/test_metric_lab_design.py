from collections import Counter

import pytest
from pydantic import ValidationError

from canonical_sync.codec import canonical_record_hash
from llm_evaluation.metric_lab_design import (
    FINAL_REPLICATES_PER_CELL,
    BaseCellSpec,
    FrozenMetricLabDesignRecord,
    MetricLabDesign,
    MetricLabPhase,
    balanced_probability_schedule,
    base_grid_cells,
    derive_metric_lab_seed,
    freeze_metric_lab_design,
)


def test_base_grid_is_exact_authorized_216_cell_cross_product() -> None:
    design = MetricLabDesign()
    cells = base_grid_cells(design)
    assert len(cells) == 4 * 3 * 3 * 2 * 3 == 216
    assert len({cell.cell_key for cell in cells}) == 216
    assert {cell.n_base_items for cell in cells} == {50, 200, 1000}
    assert {cell.k_within_item for cell in cells} == {1, 4, 16}
    assert {cell.shared_within_item for cell in cells} == {False, True}
    assert {cell.reference_flip_eta for cell in cells} == {0.0, 0.1, 0.2}
    assert {cell.forecast_kind for cell in cells} == {
        "oracle",
        "constant",
        "overconfident",
        "underconfident",
    }


def test_base_cell_key_is_stable_and_meaning_bearing() -> None:
    cell = BaseCellSpec(
        forecast_kind="oracle",
        n_base_items=50,
        k_within_item=4,
        shared_within_item=True,
        reference_flip_eta=0.1,
    )
    assert cell.cell_key == "forecast=oracle|n=50|k=4|dependence=shared|eta=0.1"


def test_balanced_probability_schedule_is_deterministic_and_rotates_remainder() -> None:
    p_values = (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    first = balanced_probability_schedule(p_values, 50, seed=0)
    repeated = balanced_probability_schedule(p_values, 50, seed=0)
    second = balanced_probability_schedule(p_values, 50, seed=1)
    assert first == repeated
    assert first != second

    first_counts = Counter(first)
    second_counts = Counter(second)
    assert set(first_counts.values()) == {7, 8}
    assert set(second_counts.values()) == {7, 8}
    assert max(first_counts.values()) - min(first_counts.values()) == 1
    # n=50 leaves one remainder after seven complete cycles. It must not always
    # overweight the same p value across replicates.
    first_extra = {p for p, count in first_counts.items() if count == 8}
    second_extra = {p for p, count in second_counts.items() if count == 8}
    assert first_extra != second_extra


def test_development_and_final_seed_spaces_are_mechanically_disjoint() -> None:
    design = MetricLabDesign()
    design_hash = canonical_record_hash(design)
    dev = {
        derive_metric_lab_seed(
            design_hash=design_hash,
            phase=MetricLabPhase.DEVELOPMENT,
            cell_key="cell-a",
            replicate_index=i,
            purpose="outcomes",
        )
        for i in range(50)
    }
    final = {
        derive_metric_lab_seed(
            design_hash=design_hash,
            phase=MetricLabPhase.FINAL,
            cell_key="cell-a",
            replicate_index=i,
            purpose="outcomes",
        )
        for i in range(50)
    }
    assert len(dev) == len(final) == 50
    assert dev.isdisjoint(final)
    assert all(seed < 2**63 for seed in dev)
    assert all(seed >= 2**63 for seed in final)
    assert (
        derive_metric_lab_seed(
            design_hash=design_hash,
            phase=MetricLabPhase.DEVELOPMENT,
            cell_key="cell-a",
            replicate_index=7,
            purpose="outcomes",
        )
        in dev
    )


def test_freeze_binds_exact_design_identity_and_final_replicate_count() -> None:
    design = MetricLabDesign()
    frozen = freeze_metric_lab_design(design)
    assert frozen.design == design
    assert frozen.design_sha256 == canonical_record_hash(design)
    assert frozen.replicates_per_cell == FINAL_REPLICATES_PER_CELL == 2000

    changed = design.model_copy(update={"decision_threshold": "0.6"})
    with pytest.raises(ValidationError, match="design_sha256"):
        FrozenMetricLabDesignRecord(
            design=changed,
            design_sha256=frozen.design_sha256,
            replicates_per_cell=2000,
        )


def test_design_is_strict_frozen_and_rejects_invalid_grid_values() -> None:
    design = MetricLabDesign()
    with pytest.raises(ValidationError):
        MetricLabDesign(n_base_items=(50, 0, 1000))
    with pytest.raises(ValidationError):
        MetricLabDesign(reference_flip_etas=("0", "0.6"))
    with pytest.raises(ValidationError):
        MetricLabDesign(decision_threshold="1.1")
    with pytest.raises(ValidationError):
        MetricLabDesign.model_validate(
            design.model_dump() | {"unknown_field": "not allowed"}
        )


def test_auxiliary_grid_parameters_are_identity_bearing_and_enumerable() -> None:
    from llm_evaluation.metric_lab_design import (
        ambiguity_grid_cells,
        prevalence_grid_cells,
    )

    design = MetricLabDesign()
    ambiguity = ambiguity_grid_cells(design)
    prevalence = prevalence_grid_cells(design)
    assert len(ambiguity) == 4 * 3 * 3 == 36
    assert len({cell.cell_key for cell in ambiguity}) == 36
    assert {cell.raters_per_item for cell in ambiguity} == {1, 3, 5}
    assert {cell.n_items for cell in ambiguity} == {50, 200, 1000}
    assert {cell.forecast_kind for cell in ambiguity} == set(design.forecast_kinds)
    assert [cell.prevalence for cell in prevalence] == [0.01, 0.1, 0.5]
    assert all(cell.tpr == 0.9 and cell.fpr == 0.01 for cell in prevalence)


def test_design_hash_changes_when_auxiliary_or_operational_policy_changes() -> None:
    design = MetricLabDesign()
    baseline = canonical_record_hash(design)
    changed_raters = design.model_copy(
        update={"ambiguity_raters_per_item": (3, 11, 25)}
    )
    changed_coverages = design.model_copy(
        update={"operational_coverages": ("0.25", "0.5", "0.9", "1")}
    )
    changed_interval = design.model_copy(
        update={"brier_interval_methods": ("normal95-row-v1",)}
    )
    assert canonical_record_hash(changed_raters) != baseline
    assert canonical_record_hash(changed_coverages) != baseline
    assert canonical_record_hash(changed_interval) != baseline


def test_candidate_operational_and_sampling_policies_are_explicit() -> None:
    design = MetricLabDesign()
    assert design.schema_version == 3
    assert design.l2_debias_method == "kumar-unbiased-squared-l2-v1"
    assert (
        design.l2_presentation_method
        == "sqrt-positive-part-of-debiased-squared-l2-v1"
    )
    assert design.operational_coverage_floats == (0.1, 0.5, 0.9, 1.0)
    assert design.sampling_units == ("fixed_case", "question_population")
    assert design.brier_interval_methods == (
        "normal95-row-v1",
        "normal95-item-cluster-v1",
        "hoeffding95-item-cluster-v1",
    )
    assert design.aurc_reference_negative_control == "han678-get_AURC-fa445bcd"


def test_uncertainty_and_independent_unit_policies_are_identity_bearing() -> None:
    design = MetricLabDesign()
    baseline = canonical_record_hash(design)
    assert design.brier_interval_methods == (
        "normal95-row-v1",
        "normal95-item-cluster-v1",
        "hoeffding95-item-cluster-v1",
    )
    assert design.calibration_independent_unit_policy == "first-row-per-item-v1"
    changed = design.model_copy(
        update={"calibration_independent_unit_policy": "different-policy"}
    )
    assert canonical_record_hash(changed) != baseline


def test_question_population_stress_inventory_is_exact_and_identity_bearing() -> None:
    from llm_evaluation.metric_lab_design import question_population_stress_cells

    design = MetricLabDesign()
    cells = question_population_stress_cells(design)
    assert len(cells) == 4 * 3 * 2 == 24
    assert len({cell.cell_key for cell in cells}) == 24
    assert all(cell.k_within_item == 1 for cell in cells)
    assert all(cell.shared_within_item is False for cell in cells)
    assert {cell.reference_flip_eta for cell in cells} == {0.0, 0.2}
    assert design.question_population_stress_policy == (
        "k1-independent-reference-eta-endpoints-all-forecast-n-v1"
    )
    assert design.l2_debias_primary_sampling_target == "iid-question-population-v1"
    assert design.fixed_case_l2_debias_role == "stress-diagnostic-not-unbiased-v1"

    baseline = canonical_record_hash(design)
    changed = design.model_copy(
        update={"question_population_stress_policy": "different-policy"}
    )
    assert canonical_record_hash(changed) != baseline


def test_soft_label_bernoulli_debias_inapplicability_is_identity_bearing() -> None:
    design = MetricLabDesign()
    assert design.soft_label_l2_debias_policy == "inapplicable-bernoulli-correction-v1"
    baseline = canonical_record_hash(design)
    changed = design.model_copy(
        update={"soft_label_l2_debias_policy": "different-policy"}
    )
    assert canonical_record_hash(changed) != baseline


def test_negative_control_versions_are_identity_bearing() -> None:
    design = MetricLabDesign()
    assert design.positive_class_inversion_control_version == "brier-label-inversion-v1"
    assert (
        design.label_noise_identifiability_control_version
        == "symmetric-flip-two-solution-v1"
    )
    baseline = canonical_record_hash(design)
    changed = design.model_copy(
        update={"positive_class_inversion_control_version": "different-control"}
    )
    assert canonical_record_hash(changed) != baseline


def test_acceptance_policy_is_explicit_and_identity_bearing() -> None:
    design = MetricLabDesign()
    criteria = design.acceptance
    assert design.schema_version == 3
    assert criteria.schema_version == 1
    assert criteria.analytic_bias_mcse_multiplier == "4"
    assert criteria.analytic_bias_absolute_floor == "1e-10"
    assert criteria.population_squared_l2_bias_mcse_multiplier == "4"
    assert criteria.shared_cluster_coverage_mean_min == "0.94"
    assert criteria.shared_cluster_coverage_mean_max == "0.98"
    assert criteria.shared_cluster_coverage_cell_min == "0.90"
    assert criteria.shared_naive_cluster_coverage_gap_min == "0.10"
    assert criteria.hoeffding_coverage_mean_min == "0.99"
    assert criteria.cluster_to_hoeffding_width_ratio_max == "0.50"
    baseline = canonical_record_hash(design)
    changed = design.model_copy(
        update={
            "acceptance": criteria.model_copy(
                update={"analytic_bias_mcse_multiplier": "5"}
            )
        }
    )
    assert canonical_record_hash(changed) != baseline
