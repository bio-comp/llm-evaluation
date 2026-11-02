from __future__ import annotations

from pathlib import Path

from canonical_sync.codec import canonical_record_bytes, canonical_record_hash, parse_canonical_record
from llm_evaluation.drift_lab_design import (
    DriftLabDesign,
    DriftLabPhase,
    FrozenDriftLabDesignRecord,
    derive_drift_lab_seed,
    freeze_drift_lab_design,
)


def test_default_design_matches_frozen_phase_a_contract() -> None:
    design = DriftLabDesign()
    assert design.looks == 52
    assert design.baseline_cases == 4000
    assert design.core_weekly_cases == 200
    assert design.change_week == 13
    assert design.baseline_b_probability == "0.20"
    assert design.shifted_b_probability == "0.60"
    assert design.baseline_a_error == "0.05"
    assert design.baseline_b_error == "0.20"
    assert design.process_error_delta == "0.05"
    assert design.ewma_lambda == "0.20"
    assert design.cusum_reference_value == "0.5"
    assert design.target_family_false_alarm == "0.05"
    assert design.development_paths == 20_000
    assert design.final_paths_per_scenario == 20_000
    assert design.psi_null_weekly_cases == (100, 200, 1000)
    assert design.variable_denominators == (50, 100, 200, 400)
    assert design.overdispersion_icc == "0.05"
    assert design.primary_truth_states == (
        "stable/stable",
        "distribution_only",
        "process_only",
        "both",
    )


def test_seed_derivation_is_deterministic_and_phase_purpose_separated() -> None:
    design = DriftLabDesign()
    d1 = derive_drift_lab_seed(
        design, phase=DriftLabPhase.DEVELOPMENT, scenario="stable/stable", replicate_index=7
    )
    d2 = derive_drift_lab_seed(
        design, phase=DriftLabPhase.DEVELOPMENT, scenario="stable/stable", replicate_index=7
    )
    final = derive_drift_lab_seed(
        design, phase=DriftLabPhase.FINAL, scenario="stable/stable", replicate_index=7
    )
    other = derive_drift_lab_seed(
        design, phase=DriftLabPhase.DEVELOPMENT, scenario="process_only", replicate_index=7
    )
    assert d1 == d2
    assert len({d1, final, other}) == 3


def test_frozen_design_fixture_is_canonical_and_identity_bound() -> None:
    payload = Path("fixtures/e004_drift_surveillance_phase_a_design.json").read_bytes()
    frozen = parse_canonical_record(payload, FrozenDriftLabDesignRecord)
    assert frozen.design == DriftLabDesign()
    assert frozen.design_sha256 == canonical_record_hash(frozen.design)
    assert frozen.development_paths == 20_000
    assert frozen.final_paths_per_scenario == 20_000
    assert canonical_record_bytes(frozen) == payload
    assert freeze_drift_lab_design(frozen.design) == frozen
