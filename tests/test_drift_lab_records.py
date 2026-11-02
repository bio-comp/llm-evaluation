from __future__ import annotations

from pathlib import Path

import pytest

from canonical_sync.codec import canonical_record_hash
from llm_evaluation.drift_lab_design import DriftLabDesign, freeze_drift_lab_design
from llm_evaluation.drift_lab_records import (
    DriftCalibrationRecord,
    DriftFinalRunRecord,
    DriftOutcomeSummaryRecord,
    finalize_drift_calibration,
    finalize_drift_final_run,
    read_drift_calibration,
    read_drift_final_run,
)


def _summary() -> DriftOutcomeSummaryRecord:
    return DriftOutcomeSummaryRecord(
        matched_policy_false_alarm={"psi_only": "0.05"},
        psi_heuristic_null_false_alarm={"100": "0.5", "200": "0.25", "1000": "0.01"},
        psi_calibrated_null_false_alarm={"100": "0.05", "200": "0.05", "1000": "0.05"},
        distribution_only_adjusted_process_alarm="0.05",
        distribution_only_raw_process_alarm="0.90",
        adjusted_typed_accuracy={"stable/stable": "0.95", "distribution_only": "0.95", "process_only": "0.85", "both": "0.85"},
        raw_typed_accuracy={"stable/stable": "0.70", "distribution_only": "0.60", "process_only": "0.70", "both": "0.70"},
        variable_denominator_fixed_n_shewhart_false_alarm="0.25",
        variable_denominator_adjusted_shewhart_false_alarm="0.05",
        overdispersion_adjusted_ewma_false_alarm="0.15",
        iid_adjusted_ewma_false_alarm="0.05",
        localized_aggregate_detection="0.60",
        localized_b_detection="0.90",
    )


def test_calibration_and_final_records_bind_exact_frozen_design_and_calibration() -> None:
    design = DriftLabDesign()
    frozen = freeze_drift_lab_design(design)
    calibration = DriftCalibrationRecord(
        design_sha256=canonical_record_hash(design),
        frozen_design_sha256=canonical_record_hash(frozen),
        development_paths=20_000,
        thresholds={"psi": "1.2", "adjusted_ewma": "0.8"},
        layered_component_tail_probability={"layered_adjusted_ewma": "0.025"},
        development_family_false_alarm={"psi_only": "0.05", "layered_adjusted_ewma": "0.05"},
        software_commit=None,
        package_source_sha256=None,
        python_version="3.11.13",
    )
    final = DriftFinalRunRecord(
        design_sha256=canonical_record_hash(design),
        frozen_design_sha256=canonical_record_hash(frozen),
        calibration_record_sha256=canonical_record_hash(calibration),
        final_paths_per_scenario=20_000,
        outcomes=_summary(),
        realized_cases=1_000_000,
        runtime_seconds="1.0",
        software_commit=None,
        package_source_sha256=None,
        python_version="3.11.13",
    )
    assert final.calibration_record_sha256 == canonical_record_hash(calibration)


def test_finalization_is_immutable_and_readback_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llm_evaluation.drift_lab_records as records

    monkeypatch.setattr(records, "software_identity", lambda: ("a" * 40, "b" * 64))
    monkeypatch.setattr(records.platform, "python_version", lambda: "3.11.13")
    design = DriftLabDesign()
    frozen = freeze_drift_lab_design(design)
    calibration_preview = DriftCalibrationRecord(
        design_sha256=canonical_record_hash(design),
        frozen_design_sha256=canonical_record_hash(frozen),
        development_paths=20_000,
        thresholds={"psi": "1.2"},
        layered_component_tail_probability={"layered_adjusted_ewma": "0.025"},
        development_family_false_alarm={"psi_only": "0.05"},
        software_commit=None,
        package_source_sha256=None,
        python_version="3.11.13",
    )
    calibration_path = tmp_path / "calibration.json"
    finalized_calibration = finalize_drift_calibration(calibration_path, calibration_preview)
    assert read_drift_calibration(calibration_path) == finalized_calibration
    with pytest.raises(FileExistsError):
        finalize_drift_calibration(calibration_path, calibration_preview)

    final_preview = DriftFinalRunRecord(
        design_sha256=canonical_record_hash(design),
        frozen_design_sha256=canonical_record_hash(frozen),
        calibration_record_sha256=canonical_record_hash(finalized_calibration),
        final_paths_per_scenario=20_000,
        outcomes=_summary(),
        realized_cases=1_000_000,
        runtime_seconds="1.0",
        software_commit=None,
        package_source_sha256=None,
        python_version="3.11.13",
    )
    final_path = tmp_path / "final.json"
    finalized = finalize_drift_final_run(final_path, final_preview)
    assert read_drift_final_run(final_path) == finalized
    with pytest.raises(FileExistsError):
        finalize_drift_final_run(final_path, final_preview)
