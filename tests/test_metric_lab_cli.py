from __future__ import annotations

import tomllib
from pathlib import Path

from llm_evaluation.metric_lab_design import (
    FINAL_REPLICATES_PER_CELL,
    MetricLabDesign,
    MetricLabPhase,
    freeze_metric_lab_design,
)


def _tiny_design() -> MetricLabDesign:
    return MetricLabDesign(
        p_values=("0.5",),
        forecast_kinds=("constant",),
        n_base_items=(2,),
        k_within_item=(1,),
        shared_within_item=(False, True),
        reference_flip_etas=("0",),
        ambiguity_theta_values=("0.5",),
        ambiguity_n_items=(2,),
        ambiguity_raters_per_item=(3,),
        prevalence_values=("0.5",),
        fixed_bin_count=1,
        equal_mass_bin_count=1,
    )


def test_full_grid_preview_enumerates_fixed_population_stress_and_ambiguity() -> None:
    from llm_evaluation.metric_lab_cli import build_full_grid_preview

    design = _tiny_design()
    preview = build_full_grid_preview(
        design=design,
        phase=MetricLabPhase.DEVELOPMENT,
        replicates_per_cell=2,
    )
    assert preview.phase is MetricLabPhase.DEVELOPMENT
    assert preview.replicates_per_cell == 2
    assert preview.cell_count == 4
    assert [cell.cell_kind for cell in preview.cells].count("base") == 3
    assert [cell.cell_kind for cell in preview.cells].count("ambiguity") == 1
    assert {cell.sampling_unit for cell in preview.cells if cell.cell_kind == "base"} == {
        "fixed_case",
        "question_population",
    }


def test_full_grid_final_preview_requires_exact_frozen_design_and_2000_replicates() -> None:
    from llm_evaluation.metric_lab_cli import build_full_grid_preview

    design = _tiny_design()
    frozen = freeze_metric_lab_design(design)
    preview = build_full_grid_preview(
        design=design,
        phase=MetricLabPhase.FINAL,
        replicates_per_cell=FINAL_REPLICATES_PER_CELL,
        frozen_design=frozen,
    )
    assert preview.phase is MetricLabPhase.FINAL
    assert preview.replicates_per_cell == FINAL_REPLICATES_PER_CELL
    assert preview.frozen_design_record_sha256 is not None


def test_metric_lab_cli_is_packaged_as_an_installed_entrypoint() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert (
        project["project"]["scripts"]["llm-evaluation-metric-lab"]
        == "llm_evaluation.metric_lab_cli:main"
    )
