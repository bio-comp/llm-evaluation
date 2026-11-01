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
