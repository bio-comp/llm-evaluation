import math

from llm_evaluation.metric_lab_targets import (
    AmbiguityRaterItem,
    SamplingUnit,
    TargetKind,
    binary_reference_targets,
    generate_item_probabilities,
    generate_rater_ambiguity_items,
    prevalence_precision,
    subgroup_ece,
    subgroup_calibration_cancellation_control,
)


def test_binary_reference_targets_keep_clean_and_noisy_estimands_distinct() -> None:
    targets = binary_reference_targets(clean_probability=0.9, reference_flip_eta=0.2)
    assert targets[TargetKind.CLEAN_TRUTH] == 0.9
    assert math.isclose(targets[TargetKind.NOISY_REFERENCE], 0.74, rel_tol=1e-12)
    assert set(targets) == {TargetKind.CLEAN_TRUTH, TargetKind.NOISY_REFERENCE}


def test_ambiguity_rater_arm_exposes_distribution_soft_and_consensus_targets() -> None:
    items = generate_rater_ambiguity_items(
        theta_values=(0.1, 0.5, 0.9),
        n_items=9,
        raters_per_item=9,
        seed=1234,
    )
    assert len(items) == 9
    assert all(isinstance(item, AmbiguityRaterItem) for item in items)
    for item in items:
        assert item.arm_type == "ambiguity_rater"
        assert 0.0 <= item.rater_distribution_probability <= 1.0
        assert 0.0 <= item.soft_label_probability <= 1.0
        assert item.majority_consensus_label in {0, 1}
        assert len(item.ratings) == 9
        assert not hasattr(item, "reference_flip_eta")
        indexed = item.target_values
        assert set(indexed) == {
            TargetKind.RATER_DISTRIBUTION,
            TargetKind.SOFT_LABEL,
            TargetKind.MAJORITY_CONSENSUS,
        }
        assert (
            indexed[TargetKind.RATER_DISTRIBUTION]
            == item.rater_distribution_probability
        )
        assert indexed[TargetKind.SOFT_LABEL] == item.soft_label_probability
        assert indexed[TargetKind.MAJORITY_CONSENSUS] == float(
            item.majority_consensus_label
        )


def test_rater_ambiguity_generation_is_deterministic_and_rejects_tied_consensus_design() -> (
    None
):
    first = generate_rater_ambiguity_items(
        theta_values=(0.2, 0.8), n_items=8, raters_per_item=5, seed=7
    )
    repeated = generate_rater_ambiguity_items(
        theta_values=(0.2, 0.8), n_items=8, raters_per_item=5, seed=7
    )
    different = generate_rater_ambiguity_items(
        theta_values=(0.2, 0.8), n_items=8, raters_per_item=5, seed=8
    )
    assert first == repeated
    assert first != different

    import pytest

    with pytest.raises(ValueError, match="odd"):
        generate_rater_ambiguity_items(
            theta_values=(0.2, 0.8), n_items=8, raters_per_item=4, seed=7
        )


def test_prevalence_reweighting_changes_precision_with_fixed_tpr_fpr() -> None:
    at_half = prevalence_precision(tpr=0.9, fpr=0.01, prevalence=0.5)
    at_one_percent = prevalence_precision(tpr=0.9, fpr=0.01, prevalence=0.01)
    assert math.isclose(at_half, 90 / 91, rel_tol=1e-12)
    assert math.isclose(at_one_percent, 10 / 21, rel_tol=1e-12)
    assert at_half > at_one_percent


def test_subgroup_control_exposes_pooled_calibration_cancellation() -> None:
    control = subgroup_calibration_cancellation_control()
    assert math.isclose(control.pooled_ece, 0.0, abs_tol=1e-12)
    assert math.isclose(control.subgroup_ece["A"], 0.25, rel_tol=1e-12)
    assert math.isclose(control.subgroup_ece["B"], 0.25, rel_tol=1e-12)
    assert math.isclose(control.max_subgroup_ece, 0.25, rel_tol=1e-12)

    direct = subgroup_ece(
        forecasts=(0.25, 0.25, 0.75, 0.75),
        outcomes=(0, 0, 1, 1),
        groups=("A", "A", "B", "B"),
        num_bins=1,
    )
    assert direct == control.subgroup_ece


def test_fixed_case_and_population_sampling_are_different_uncertainty_targets() -> None:
    p_values = (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    fixed_a = generate_item_probabilities(
        p_values=p_values,
        n_items=50,
        sampling_unit=SamplingUnit.FIXED_CASE,
        seed=11,
    )
    fixed_b = generate_item_probabilities(
        p_values=p_values,
        n_items=50,
        sampling_unit=SamplingUnit.FIXED_CASE,
        seed=99,
    )
    population_a = generate_item_probabilities(
        p_values=p_values,
        n_items=50,
        sampling_unit=SamplingUnit.QUESTION_POPULATION,
        seed=11,
    )
    population_b = generate_item_probabilities(
        p_values=p_values,
        n_items=50,
        sampling_unit=SamplingUnit.QUESTION_POPULATION,
        seed=99,
    )
    assert fixed_a == fixed_b
    assert population_a != population_b
    assert set(population_a).issubset(set(p_values))
    assert set(population_b).issubset(set(p_values))
