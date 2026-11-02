"""Predeclared acceptance checks for the known-truth metric laboratory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean

from canonical_sync.codec import canonical_record_hash
from llm_evaluation.metric_lab_design import (
    FINAL_REPLICATES_PER_CELL,
    MetricLabDesign,
    MetricLabPhase,
    ambiguity_grid_cells,
    base_grid_cells,
    freeze_metric_lab_design,
    question_population_stress_cells,
)
from llm_evaluation.metric_lab_records import MetricLabRunRecord, MetricSummaryRecord
from llm_evaluation.metric_lab_targets import SamplingUnit, TargetKind


@dataclass(frozen=True, slots=True)
class MetricLabAcceptanceResult:
    passed: bool
    failures: tuple[str, ...]
    diagnostics: tuple[tuple[str, float], ...]


def metric_bias_within_tolerance(
    *,
    bias: float,
    bias_mc_se: float,
    mcse_multiplier: float,
    absolute_floor: float,
) -> bool:
    """Return whether an analytic-value discrepancy clears the frozen MC guard."""
    values = (bias, bias_mc_se, mcse_multiplier, absolute_floor)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bias tolerance inputs must be finite")
    if bias_mc_se < 0 or mcse_multiplier < 0 or absolute_floor < 0:
        raise ValueError("bias tolerance scales must be non-negative")
    tolerance = max(mcse_multiplier * bias_mc_se, absolute_floor)
    return abs(bias) <= tolerance


def _float(value: str) -> float:
    return float(Decimal(value))


def _coverage_values(cells: list, metric_name: str) -> tuple[list[float], list[float]]:
    coverages: list[float] = []
    widths: list[float] = []
    for cell in cells:
        for target in cell.targets:
            metric = target.metrics.get(metric_name)
            if metric is None or metric.interval_coverage is None or metric.mean_interval_width is None:
                continue
            coverages.append(metric.interval_coverage)
            widths.append(metric.mean_interval_width)
    return coverages, widths


def _check_bias_family(
    *,
    summaries: list[MetricSummaryRecord],
    multiplier: float,
    absolute_floor: float,
) -> bool:
    return bool(summaries) and all(
        metric_bias_within_tolerance(
            bias=item.bias,
            bias_mc_se=item.bias_mc_se,
            mcse_multiplier=multiplier,
            absolute_floor=absolute_floor,
        )
        for item in summaries
    )


def evaluate_metric_lab_acceptance(
    *, run: MetricLabRunRecord, design: MetricLabDesign
) -> MetricLabAcceptanceResult:
    """Evaluate a finalized held-out run against the exact frozen promotion policy."""
    failures: list[str] = []
    diagnostics: dict[str, float] = {}
    criteria = design.acceptance

    if run.phase is not MetricLabPhase.FINAL:
        failures.append("phase must be final")
    if run.replicates_per_cell != FINAL_REPLICATES_PER_CELL:
        failures.append("final run must use exactly 2000 replicates per cell")
    design_hash = canonical_record_hash(design)
    if run.design_sha256 != design_hash:
        failures.append("run design hash does not match frozen design")
    frozen = freeze_metric_lab_design(design)
    if run.frozen_design_record_sha256 != canonical_record_hash(frozen):
        failures.append("run does not bind the exact frozen design record")
    if run.software_commit is None or run.package_source_sha256 is None:
        failures.append("final run must be finalized with software identity")

    expected_inventory: set[tuple[str, str, SamplingUnit | None]] = {
        ("base", cell.cell_key, SamplingUnit.FIXED_CASE)
        for cell in base_grid_cells(design)
    }
    expected_inventory |= {
        ("base", cell.cell_key, SamplingUnit.QUESTION_POPULATION)
        for cell in question_population_stress_cells(design)
    }
    expected_inventory |= {
        ("ambiguity", cell.cell_key, None) for cell in ambiguity_grid_cells(design)
    }
    actual_inventory = {
        (cell.cell_kind, cell.cell_key, cell.sampling_unit) for cell in run.cells
    }
    if actual_inventory != expected_inventory:
        failures.append("final run cell inventory does not match the frozen design")

    base_cells = [cell for cell in run.cells if cell.cell_kind == "base"]
    fixed_cells = [
        cell for cell in base_cells if cell.sampling_unit is SamplingUnit.FIXED_CASE
    ]
    population_cells = [
        cell
        for cell in base_cells
        if cell.sampling_unit is SamplingUnit.QUESTION_POPULATION
    ]
    ambiguity_cells = [cell for cell in run.cells if cell.cell_kind == "ambiguity"]

    multiplier = _float(criteria.analytic_bias_mcse_multiplier)
    absolute_floor = _float(criteria.analytic_bias_absolute_floor)
    for metric_name in ("brier", "log_score", "classification_error"):
        summaries = [
            target.metrics[metric_name]
            for cell in base_cells
            for target in cell.targets
            if metric_name in target.metrics
        ]
        diagnostics[f"analytic_{metric_name}_count"] = float(len(summaries))
        if not _check_bias_family(
            summaries=summaries,
            multiplier=multiplier,
            absolute_floor=absolute_floor,
        ):
            failures.append(f"analytic {metric_name} bias exceeds frozen tolerance")

    l2_summaries = [
        target.metrics["debiased_squared_l2_independent_unit"]
        for cell in population_cells
        for target in cell.targets
        if "debiased_squared_l2_independent_unit" in target.metrics
    ]
    l2_multiplier = _float(criteria.population_squared_l2_bias_mcse_multiplier)
    diagnostics["population_squared_l2_count"] = float(len(l2_summaries))
    if not _check_bias_family(
        summaries=l2_summaries,
        multiplier=l2_multiplier,
        absolute_floor=absolute_floor,
    ):
        failures.append("question-population squared-L2 bias exceeds frozen tolerance")

    shared_k_gt_1 = [
        cell
        for cell in fixed_cells
        if "dependence=shared" in cell.cell_key and "|k=1|" not in cell.cell_key
    ]
    independent = [
        cell for cell in fixed_cells if "dependence=independent" in cell.cell_key
    ]
    shared_cluster, shared_cluster_widths = _coverage_values(
        shared_k_gt_1, "brier_cluster_interval"
    )
    shared_naive, _ = _coverage_values(shared_k_gt_1, "brier_naive_interval")
    shared_hoeffding, shared_hoeffding_widths = _coverage_values(
        shared_k_gt_1, "brier_cluster_hoeffding_interval"
    )
    independent_cluster, _ = _coverage_values(independent, "brier_cluster_interval")
    population_cluster, _ = _coverage_values(
        population_cells, "brier_cluster_interval"
    )

    if not shared_cluster or not shared_naive or not shared_hoeffding:
        failures.append("shared-dependence interval diagnostics are incomplete")
    else:
        shared_cluster_mean = fmean(shared_cluster)
        shared_naive_mean = fmean(shared_naive)
        shared_hoeffding_mean = fmean(shared_hoeffding)
        width_ratio = fmean(shared_cluster_widths) / fmean(shared_hoeffding_widths)
        diagnostics.update(
            {
                "shared_cluster_coverage_mean": shared_cluster_mean,
                "shared_cluster_coverage_min": min(shared_cluster),
                "shared_naive_coverage_mean": shared_naive_mean,
                "shared_hoeffding_coverage_mean": shared_hoeffding_mean,
                "shared_cluster_minus_naive_coverage": shared_cluster_mean
                - shared_naive_mean,
                "shared_cluster_to_hoeffding_width_ratio": width_ratio,
            }
        )
        if not (
            _float(criteria.shared_cluster_coverage_mean_min)
            <= shared_cluster_mean
            <= _float(criteria.shared_cluster_coverage_mean_max)
        ):
            failures.append("shared cluster-normal mean coverage is outside frozen bounds")
        if min(shared_cluster) < _float(criteria.shared_cluster_coverage_cell_min):
            failures.append("a shared cluster-normal cell is below the frozen coverage floor")
        if (
            shared_cluster_mean - shared_naive_mean
            < _float(criteria.shared_naive_cluster_coverage_gap_min)
        ):
            failures.append("cluster-normal interval does not sufficiently beat naive row coverage")
        if shared_hoeffding_mean < _float(criteria.hoeffding_coverage_mean_min):
            failures.append("cluster Hoeffding sensitivity interval coverage is too low")
        if width_ratio > _float(criteria.cluster_to_hoeffding_width_ratio_max):
            failures.append("cluster-normal interval is not materially narrower than Hoeffding")

    for label, values, low_name, high_name in (
        (
            "independent fixed-case",
            independent_cluster,
            "independent_cluster_coverage_mean_min",
            "independent_cluster_coverage_mean_max",
        ),
        (
            "question-population",
            population_cluster,
            "question_population_cluster_coverage_mean_min",
            "question_population_cluster_coverage_mean_max",
        ),
    ):
        if not values:
            failures.append(f"{label} cluster-normal coverage diagnostics are incomplete")
            continue
        observed = fmean(values)
        diagnostics[f"{label.replace('-', '_').replace(' ', '_')}_cluster_coverage_mean"] = observed
        if not (_float(getattr(criteria, low_name)) <= observed <= _float(getattr(criteria, high_name))):
            failures.append(f"{label} cluster-normal mean coverage is outside frozen bounds")

    soft_targets = [
        target
        for cell in ambiguity_cells
        for target in cell.targets
        if target.target is TargetKind.SOFT_LABEL
    ]
    diagnostics["soft_label_target_count"] = float(len(soft_targets))
    if not soft_targets or any(
        any("debiased" in metric_name for metric_name in target.metrics)
        for target in soft_targets
    ):
        failures.append("soft-label targets must not apply Bernoulli debias corrections")

    constant_fixed_targets = [
        target
        for cell in fixed_cells
        if cell.cell_key.startswith("forecast=constant|")
        for target in cell.targets
    ]
    if not constant_fixed_targets or any(
        target.metrics["realized_coverage_at_request_0.1"].mean_estimate != 1.0
        for target in constant_fixed_targets
    ):
        failures.append("constant-score tie blocks must preserve unsplittable realized coverage")

    controls = run.auxiliary_controls
    if not controls.aurc_defect_detected:
        failures.append("AURC known-bad negative control did not fail")
    if not controls.positive_class_inversion_detected:
        failures.append("positive-class inversion control did not fail")
    if not controls.label_noise_nonidentifiable:
        failures.append("single-reference label-noise non-identifiability control did not hold")
    if controls.reference_ranking_reversal.get("reversal_detected") is not True:
        failures.append("wrong-reference ranking reversal control did not fire")
    if abs(controls.subgroup_pooled_ece) > 1e-12 or controls.subgroup_max_ece < 0.25:
        failures.append("subgroup cancellation control did not reproduce the expected failure")

    return MetricLabAcceptanceResult(
        passed=not failures,
        failures=tuple(failures),
        diagnostics=tuple(sorted(diagnostics.items())),
    )
