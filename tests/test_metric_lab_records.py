import json
from pathlib import Path

import pytest

from llm_evaluation.metric_lab_design import (
    BaseCellSpec,
    MetricLabDesign,
    MetricLabPhase,
    freeze_metric_lab_design,
)
from llm_evaluation.metric_lab_records import (
    finalize_metric_lab_run,
    metric_lab_record_bytes,
    preview_metric_lab_run_record,
    read_metric_lab_run,
    validate_final_execution_gate,
)
from llm_evaluation.metric_lab_runner import summarize_base_cell
from llm_evaluation.metric_lab_targets import SamplingUnit


def _small_design() -> tuple[MetricLabDesign, BaseCellSpec]:
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
    return design, cell


def test_development_run_finalizes_canonically_and_second_write_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llm_evaluation.metric_lab_records as records

    design, cell = _small_design()
    summary = summarize_base_cell(
        design=design,
        cell=cell,
        phase=MetricLabPhase.DEVELOPMENT,
        sampling_unit=SamplingUnit.FIXED_CASE,
        num_replicates=5,
    )
    preview = preview_metric_lab_run_record(
        design=design,
        phase=MetricLabPhase.DEVELOPMENT,
        cell_summaries=(summary,),
    )
    assert preview.software_commit is None
    assert preview.package_source_sha256 is None
    assert preview.frozen_design_record_sha256 is None

    monkeypatch.setattr(records, "software_identity", lambda: ("a" * 40, "b" * 64))
    target = tmp_path / "metric-lab-dev.json"
    finalized = finalize_metric_lab_run(target, preview)
    assert finalized.software_commit == "a" * 40
    assert finalized.package_source_sha256 == "b" * 64
    assert target.read_bytes() == metric_lab_record_bytes(finalized)
    assert read_metric_lab_run(target) == finalized

    with pytest.raises(FileExistsError):
        finalize_metric_lab_run(target, preview)


def test_final_execution_gate_requires_exact_frozen_design_and_2000_replicates() -> (
    None
):
    design, _ = _small_design()
    frozen = freeze_metric_lab_design(design)
    validate_final_execution_gate(
        design=design,
        frozen_design=frozen,
        replicates_per_cell=2000,
    )
    with pytest.raises(ValueError, match="2000"):
        validate_final_execution_gate(
            design=design,
            frozen_design=frozen,
            replicates_per_cell=1999,
        )
    changed = design.model_copy(update={"decision_threshold": "0.6"})
    with pytest.raises(ValueError, match="frozen|design"):
        validate_final_execution_gate(
            design=changed,
            frozen_design=frozen,
            replicates_per_cell=2000,
        )


def test_final_preview_rejects_development_sized_summary() -> None:
    design, cell = _small_design()
    frozen = freeze_metric_lab_design(design)
    summary = summarize_base_cell(
        design=design,
        cell=cell,
        phase=MetricLabPhase.FINAL,
        sampling_unit=SamplingUnit.FIXED_CASE,
        num_replicates=3,
    )
    with pytest.raises(ValueError, match="2000"):
        preview_metric_lab_run_record(
            design=design,
            phase=MetricLabPhase.FINAL,
            cell_summaries=(summary,),
            frozen_design=frozen,
        )


def test_read_rejects_noncanonical_json_even_when_semantically_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llm_evaluation.metric_lab_records as records

    design, cell = _small_design()
    summary = summarize_base_cell(
        design=design,
        cell=cell,
        phase=MetricLabPhase.DEVELOPMENT,
        sampling_unit=SamplingUnit.FIXED_CASE,
        num_replicates=2,
    )
    preview = preview_metric_lab_run_record(
        design=design,
        phase=MetricLabPhase.DEVELOPMENT,
        cell_summaries=(summary,),
    )
    monkeypatch.setattr(records, "software_identity", lambda: ("c" * 40, "d" * 64))
    canonical_path = tmp_path / "canonical.json"
    finalized = finalize_metric_lab_run(canonical_path, preview)

    noncanonical = tmp_path / "pretty.json"
    noncanonical.write_text(
        json.dumps(finalized.model_dump(mode="json"), indent=2) + "\n"
    )
    with pytest.raises(ValueError, match="canonical"):
        read_metric_lab_run(noncanonical)
