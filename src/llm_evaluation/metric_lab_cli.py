"""Executable full-grid runner for the known-truth metric/reference-error laboratory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from canonical_sync.codec import parse_canonical_record
from llm_evaluation.metric_lab_design import (
    FINAL_REPLICATES_PER_CELL,
    FrozenMetricLabDesignRecord,
    MetricLabDesign,
    MetricLabPhase,
    ambiguity_grid_cells,
    base_grid_cells,
    question_population_stress_cells,
)
from llm_evaluation.metric_lab_records import (
    MetricLabRunRecord,
    finalize_metric_lab_run,
    preview_metric_lab_run_record,
)
from llm_evaluation.metric_lab_runner import (
    AmbiguityCellSummary,
    BaseCellSummary,
    summarize_ambiguity_cell,
    summarize_base_cell,
)
from llm_evaluation.metric_lab_targets import SamplingUnit


def build_full_grid_preview(
    *,
    design: MetricLabDesign,
    phase: MetricLabPhase,
    replicates_per_cell: int,
    frozen_design: FrozenMetricLabDesignRecord | None = None,
) -> MetricLabRunRecord:
    """Execute the exact encoded grid and return an unfinalized run record."""
    if type(replicates_per_cell) is not int or replicates_per_cell < 1:
        raise ValueError("replicates_per_cell must be a positive exact integer")
    if phase is MetricLabPhase.FINAL:
        if frozen_design is None:
            raise ValueError("final execution requires the exact frozen design record")
        if replicates_per_cell != FINAL_REPLICATES_PER_CELL:
            raise ValueError("final execution requires exactly 2000 replicates per cell")
    elif frozen_design is not None:
        raise ValueError("development execution must not supply a frozen final design")

    summaries: list[BaseCellSummary | AmbiguityCellSummary] = []
    for base_cell in base_grid_cells(design):
        summaries.append(
            summarize_base_cell(
                design=design,
                cell=base_cell,
                phase=phase,
                sampling_unit=SamplingUnit.FIXED_CASE,
                num_replicates=replicates_per_cell,
            )
        )
    for population_cell in question_population_stress_cells(design):
        summaries.append(
            summarize_base_cell(
                design=design,
                cell=population_cell,
                phase=phase,
                sampling_unit=SamplingUnit.QUESTION_POPULATION,
                num_replicates=replicates_per_cell,
            )
        )
    for ambiguity_cell in ambiguity_grid_cells(design):
        summaries.append(
            summarize_ambiguity_cell(
                design=design,
                cell=ambiguity_cell,
                phase=phase,
                num_replicates=replicates_per_cell,
            )
        )
    return preview_metric_lab_run_record(
        design=design,
        phase=phase,
        cell_summaries=tuple(summaries),
        frozen_design=frozen_design,
    )


def _load_frozen_design(path: Path) -> FrozenMetricLabDesignRecord:
    if path.is_symlink() or not path.is_file():
        raise ValueError("frozen design path must be an ordinary file")
    return parse_canonical_record(path.read_bytes(), FrozenMetricLabDesignRecord)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(item.value for item in MetricLabPhase), required=True)
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--frozen-design", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    phase = MetricLabPhase(args.phase)
    frozen: FrozenMetricLabDesignRecord | None = None
    if phase is MetricLabPhase.FINAL:
        if args.frozen_design is None:
            raise SystemExit("--frozen-design is required for final execution")
        frozen = _load_frozen_design(args.frozen_design)
        design = frozen.design
        replicates = FINAL_REPLICATES_PER_CELL if args.replicates is None else args.replicates
    else:
        if args.frozen_design is not None:
            raise SystemExit("--frozen-design is not allowed for development execution")
        design = MetricLabDesign()
        replicates = 100 if args.replicates is None else args.replicates

    preview = build_full_grid_preview(
        design=design,
        phase=phase,
        replicates_per_cell=replicates,
        frozen_design=frozen,
    )
    finalized = finalize_metric_lab_run(args.output, preview)
    print(
        f"metric lab {finalized.phase.value}: cells={finalized.cell_count} "
        f"replicates_per_cell={finalized.replicates_per_cell} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
