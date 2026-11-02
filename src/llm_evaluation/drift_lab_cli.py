"""Executable DEVELOPMENT/FINAL runner for E-004 drift-surveillance Phase A."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from canonical_sync.codec import canonical_record_hash, parse_canonical_record
from llm_evaluation.drift_lab_acceptance import evaluate_drift_lab_acceptance
from llm_evaluation.drift_lab_design import FrozenDriftLabDesignRecord
from llm_evaluation.drift_lab_records import (
    finalize_drift_calibration,
    finalize_drift_final_run,
    preview_drift_calibration,
    preview_drift_final_run,
    read_drift_calibration,
)
from llm_evaluation.drift_lab_runner import (
    DevelopmentCalibrationSummary,
    run_development_calibration,
    run_final_summary,
    run_scenario_policy_reporting,
)
from llm_evaluation.records import software_identity

_DEFAULT_FROZEN_DESIGN = Path("fixtures/e004_drift_surveillance_phase_a_design.json")


def _decimal(value: float) -> str:
    return format(value, ".17g")


def _load_frozen_design(path: Path) -> FrozenDriftLabDesignRecord:
    if path.is_symlink() or not path.is_file():
        raise ValueError("frozen E-004 design path must be an ordinary file")
    return parse_canonical_record(path.read_bytes(), FrozenDriftLabDesignRecord)


def _realized_final_cases(frozen: FrozenDriftLabDesignRecord) -> int:
    design = frozen.design
    core = design.baseline_cases + design.looks * design.core_weekly_cases
    primary_and_core_stress = 8 * core  # 4 primary + gradual + temporary + overdispersion + localized
    variable = design.baseline_cases + (design.looks // 4) * sum(design.variable_denominators)
    psi_null = sum(
        design.baseline_cases + design.looks * weekly_cases
        for weekly_cases in design.psi_null_weekly_cases
    )
    return design.final_paths_per_scenario * (primary_and_core_stress + variable + psi_null)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "development", "final"), required=True)
    parser.add_argument("--frozen-design", type=Path, default=_DEFAULT_FROZEN_DESIGN)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-paths", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    frozen = _load_frozen_design(args.frozen_design)
    design = frozen.design

    if args.phase == "smoke":
        if args.output is not None or args.calibration is not None:
            raise SystemExit("smoke mode never publishes artifacts and does not accept --output/--calibration")
        if args.smoke_paths < 1:
            raise SystemExit("--smoke-paths must be positive")
        calibration = run_development_calibration(design, path_count=args.smoke_paths)
        summary = run_final_summary(design, calibration=calibration, path_count=args.smoke_paths)
        reporting = run_scenario_policy_reporting(
            design, calibration=calibration, path_count=args.smoke_paths
        )
        print(
            f"E-004 smoke: paths={args.smoke_paths} policies={len(summary.matched_policy_false_alarm)} "
            f"scenario_policy_outcomes={len(reporting)} published=false"
        )
        return 0

    if args.output is None:
        raise SystemExit("--output is required for scientific DEVELOPMENT/FINAL execution")

    if args.phase == "development":
        if args.calibration is not None:
            raise SystemExit("--calibration is not accepted for DEVELOPMENT")
        calibration = run_development_calibration(design, path_count=design.development_paths)
        calibration_preview = preview_drift_calibration(
            design=design,
            frozen_design=frozen,
            development_paths=calibration.path_count,
            thresholds={key: _decimal(value) for key, value in calibration.thresholds.items()}
            | {
                f"psi_null_n{n}": _decimal(value)
                for n, value in calibration.psi_null_thresholds.items()
            },
            layered_component_tail_probability={
                key: _decimal(value)
                for key, value in calibration.layered_component_tail_probability.items()
            },
            development_family_false_alarm={
                key: _decimal(value)
                for key, value in calibration.development_family_false_alarm.items()
            },
        )
        finalized_calibration = finalize_drift_calibration(args.output, calibration_preview)
        print(
            f"E-004 DEVELOPMENT: paths={finalized_calibration.development_paths} output={args.output}"
        )
        return 0

    if args.calibration is None:
        raise SystemExit("--calibration is required for FINAL")
    calibration_record = read_drift_calibration(args.calibration)
    if (
        calibration_record.design_sha256 != canonical_record_hash(design)
        or calibration_record.frozen_design_sha256 != canonical_record_hash(frozen)
    ):
        raise SystemExit("FINAL calibration does not bind the exact frozen E-004 design")
    commit, digest = software_identity()
    if (
        calibration_record.software_commit != commit
        or calibration_record.package_source_sha256 != digest
    ):
        raise SystemExit("FINAL requires calibration produced by the exact executing software identity")
    threshold_map = {
        key: float(value)
        for key, value in calibration_record.thresholds.items()
        if not key.startswith("psi_null_n")
    }
    psi_null = {
        int(key.removeprefix("psi_null_n")): float(value)
        for key, value in calibration_record.thresholds.items()
        if key.startswith("psi_null_n")
    }
    calibration = DevelopmentCalibrationSummary(
        path_count=calibration_record.development_paths,
        thresholds=threshold_map,
        layered_component_tail_probability={
            key: float(value)
            for key, value in calibration_record.layered_component_tail_probability.items()
        },
        development_family_false_alarm={
            key: float(value)
            for key, value in calibration_record.development_family_false_alarm.items()
        },
        psi_null_thresholds=psi_null,
    )
    started = time.perf_counter()
    summary = run_final_summary(
        design, calibration=calibration, path_count=design.final_paths_per_scenario
    )
    reporting = run_scenario_policy_reporting(
        design, calibration=calibration, path_count=design.final_paths_per_scenario
    )
    runtime = time.perf_counter() - started
    final_preview = preview_drift_final_run(
        design=design,
        frozen_design=frozen,
        calibration=calibration_record,
        final_paths_per_scenario=design.final_paths_per_scenario,
        outcomes=summary,
        scenario_policy_outcomes=reporting,
        realized_cases=_realized_final_cases(frozen),
        runtime_seconds=_decimal(runtime),
    )
    finalized_final = finalize_drift_final_run(args.output, final_preview)
    acceptance = evaluate_drift_lab_acceptance(summary=finalized_final.outcomes, design=design)
    print(
        f"E-004 FINAL: paths_per_scenario={finalized_final.final_paths_per_scenario} "
        f"acceptance={'PASS' if acceptance.passed else 'FAIL'} output={args.output}"
    )
    if not acceptance.passed:
        for failure in acceptance.failures:
            print(f"FAIL: {failure}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
