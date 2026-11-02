from __future__ import annotations

from pathlib import Path

import pytest

from canonical_sync.codec import canonical_record_hash
from llm_evaluation.drift_lab_design import DriftLabDesign, freeze_drift_lab_design
from llm_evaluation.drift_lab_records import (
    DriftCalibrationRecord,
    DriftFinalRunRecord,
    DriftOutcomeSummaryRecord,
    ScenarioPolicyOutcomeRecord,
    finalize_drift_calibration,
    finalize_drift_final_run,
    read_drift_calibration,
    read_drift_final_run,
)


def _summary() -> DriftOutcomeSummaryRecord:
    return DriftOutcomeSummaryRecord(
        matched_policy_false_alarm={
            "psi_only": "0.05", "pearson_only": "0.05", "spc_only_ewma": "0.05",
            "spc_only_cusum": "0.05", "layered_raw": "0.05",
            "layered_adjusted_ewma": "0.05", "layered_adjusted_cusum": "0.05",
        },
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


def _detail(
    scenario: str = "stable/stable", policy: str = "psi_only"
) -> ScenarioPolicyOutcomeRecord:
    changed = scenario != "stable/stable"
    return ScenarioPolicyOutcomeRecord(
        scenario=scenario, policy=policy, horizon_alarm_successes=5, trials=100,
        horizon_alarm_probability="0.05", horizon_alarm_wilson_lower="0.02",
        horizon_alarm_wilson_upper="0.11", prechange_alarm_probability="0.05",
        prechange_alarm_wilson_lower="0.02", prechange_alarm_wilson_upper="0.11",
        postchange_detection_probability="0.8" if changed else None,
        postchange_detection_wilson_lower="0.7" if changed else None,
        postchange_detection_wilson_upper="0.87" if changed else None,
        missed_detection_probability="0.2" if changed else None,
        median_first_alarm_week="20", median_postchange_delay_weeks="7" if changed else None,
        mean_alarm_weeks="0.07",
    )


def _details() -> tuple[ScenarioPolicyOutcomeRecord, ...]:
    scenarios = (
        "stable/stable", "distribution_only", "process_only", "both",
        "gradual_process", "temporary_process", "localized_b_process",
    )
    policies = (
        "psi_only", "pearson_only", "spc_only_ewma", "spc_only_cusum",
        "layered_raw", "layered_adjusted_ewma", "layered_adjusted_cusum",
    )
    items = [_detail(scenario, policy) for scenario in scenarios for policy in policies]
    items.append(_detail("localized_b_process", "b_stratum_ewma"))
    return tuple(items)


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
        scenario_policy_outcomes=_details(),
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
        scenario_policy_outcomes=_details(),
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


def test_scenario_policy_outcome_requires_consistent_probability_fields() -> None:
    from llm_evaluation.drift_lab_records import ScenarioPolicyOutcomeRecord

    outcome = ScenarioPolicyOutcomeRecord(
        scenario="gradual_process",
        policy="spc_only_ewma",
        horizon_alarm_successes=80,
        trials=100,
        horizon_alarm_probability="0.8",
        horizon_alarm_wilson_lower="0.7111708343297916",
        horizon_alarm_wilson_upper="0.8666330660571393",
        prechange_alarm_probability="0.04",
        prechange_alarm_wilson_lower="0.01",
        prechange_alarm_wilson_upper="0.10",
        postchange_detection_probability="0.79",
        postchange_detection_wilson_lower="0.70",
        postchange_detection_wilson_upper="0.86",
        missed_detection_probability="0.21",
        median_first_alarm_week="20",
        median_postchange_delay_weeks="7",
        mean_alarm_weeks="4.2",
    )
    assert outcome.horizon_alarm_successes == 80
    assert outcome.postchange_detection_probability == "0.79"


def test_preview_builders_enforce_exact_scientific_budgets_and_frozen_identity() -> None:
    from llm_evaluation.drift_lab_records import (
        preview_drift_calibration,
        preview_drift_final_run,
    )

    design = DriftLabDesign()
    frozen = freeze_drift_lab_design(design)
    with pytest.raises(ValueError, match="20000"):
        preview_drift_calibration(
            design=design,
            frozen_design=frozen,
            development_paths=10,
            thresholds={"psi": "1.0"},
            layered_component_tail_probability={"layered_raw": "0.02"},
            development_family_false_alarm={"psi_only": "0.05"},
        )
    calibration = preview_drift_calibration(
        design=design,
        frozen_design=frozen,
        development_paths=20_000,
        thresholds={"psi": "1.0"},
        layered_component_tail_probability={"layered_raw": "0.02"},
        development_family_false_alarm={"psi_only": "0.05"},
    )
    assert calibration.software_commit is None
    assert calibration.design_sha256 == canonical_record_hash(design)

    finalized_calibration = calibration.model_copy(
        update={"software_commit": "a" * 40, "package_source_sha256": "b" * 64}
    )
    final = preview_drift_final_run(
        design=design,
        frozen_design=frozen,
        calibration=DriftCalibrationRecord.model_validate(
            finalized_calibration.model_dump(), strict=True
        ),
        final_paths_per_scenario=20_000,
        outcomes=_summary(),
        scenario_policy_outcomes=_details(),
        realized_cases=100,
        runtime_seconds="1.0",
    )
    assert final.calibration_record_sha256 == canonical_record_hash(finalized_calibration)
    with pytest.raises(ValueError, match="20000"):
        preview_drift_final_run(
            design=design,
            frozen_design=frozen,
            calibration=DriftCalibrationRecord.model_validate(
                finalized_calibration.model_dump(), strict=True
            ),
            final_paths_per_scenario=10,
            outcomes=_summary(),
            scenario_policy_outcomes=_details(),
            realized_cases=100,
            runtime_seconds="1.0",
        )


def test_changed_scenario_outcome_requires_postchange_detection_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="post-change"):
        ScenarioPolicyOutcomeRecord(
            scenario="gradual_process",
            policy="psi_only",
            horizon_alarm_successes=5,
            trials=100,
            horizon_alarm_probability="0.05",
            horizon_alarm_wilson_lower="0.02",
            horizon_alarm_wilson_upper="0.11",
            prechange_alarm_probability="0.01",
            prechange_alarm_wilson_lower="0.001",
            prechange_alarm_wilson_upper="0.05",
            postchange_detection_probability=None,
            postchange_detection_wilson_lower=None,
            postchange_detection_wilson_upper=None,
            missed_detection_probability=None,
            median_first_alarm_week="20",
            median_postchange_delay_weeks=None,
            mean_alarm_weeks="0.1",
        )


def test_final_record_rejects_incomplete_scenario_policy_inventory() -> None:
    from pydantic import ValidationError

    design = DriftLabDesign()
    frozen = freeze_drift_lab_design(design)
    calibration = DriftCalibrationRecord(
        design_sha256=canonical_record_hash(design),
        frozen_design_sha256=canonical_record_hash(frozen),
        development_paths=20_000,
        thresholds={"psi": "1"},
        layered_component_tail_probability={"layered_raw": "0.02"},
        development_family_false_alarm={"psi_only": "0.05"},
        software_commit="a" * 40,
        package_source_sha256="b" * 64,
        python_version="3.11.13",
    )
    with pytest.raises(ValidationError, match="inventory"):
        DriftFinalRunRecord(
            design_sha256=canonical_record_hash(design),
            frozen_design_sha256=canonical_record_hash(frozen),
            calibration_record_sha256=canonical_record_hash(calibration),
            final_paths_per_scenario=20_000,
            outcomes=_summary(),
            scenario_policy_outcomes=(_detail(),),
            realized_cases=100,
            runtime_seconds="1",
            software_commit=None,
            package_source_sha256=None,
            python_version="3.11.13",
        )
