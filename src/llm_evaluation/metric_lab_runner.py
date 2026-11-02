"""Known-truth execution and Monte Carlo summaries for the metric laboratory."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from canonical_sync.codec import canonical_record_hash
from llm_evaluation.metric_lab_design import (
    AmbiguityCellSpec,
    BaseCellSpec,
    MetricLabDesign,
    MetricLabPhase,
    PrevalenceCellSpec,
    derive_metric_lab_seed,
    prevalence_grid_cells,
)
from llm_evaluation.metric_lab_targets import (
    LabelNoiseNonidentifiabilityControl,
    PositiveClassInversionControl,
    SamplingUnit,
    SubgroupCalibrationControl,
    TargetKind,
    generate_item_probabilities,
    generate_rater_ambiguity_items,
    label_noise_nonidentifiability_control,
    positive_class_inversion_control,
    prevalence_precision,
    subgroup_calibration_cancellation_control,
)
from llm_evaluation.metric_laboratory import (
    binary_decision,
    brier_score,
    compute_aurc,
    compute_forecast,
    debiased_squared_l2_calibration_error,
    equal_mass_ece,
    expected_brier,
    expected_log_score,
    fixed_bin_ece,
    log_score,
    noisy_target_probability,
    check_reference_ranking_reversal,
)

_NORMAL_95_Z = 1.959963984540054


@dataclass(frozen=True, slots=True)
class MeanInterval:
    estimate: float
    se: float | None
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class MeanLossIntervals:
    naive: MeanInterval
    cluster: MeanInterval
    cluster_hoeffding: MeanInterval


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    coverage: float
    risk: float
    selected_count: int
    confidence_floor: float


@dataclass(frozen=True, slots=True)
class OperationalRiskPoint:
    requested_coverage: float
    realized_coverage: float
    risk: float
    selected_count: int


@dataclass(frozen=True, slots=True)
class PrevalenceControlResult:
    cell_key: str
    prevalence: float
    tpr: float
    fpr: float
    precision: float


@dataclass(frozen=True, slots=True)
class AuxiliaryDeterministicControls:
    prevalence: tuple[PrevalenceControlResult, ...]
    subgroup: SubgroupCalibrationControl
    aurc_upstream_negative_control: float
    aurc_prefix_oracle: float
    aurc_defect_detected: bool
    positive_class_inversion: PositiveClassInversionControl
    label_noise_nonidentifiability: LabelNoiseNonidentifiabilityControl
    reference_ranking_reversal: dict[str, float | bool]


@dataclass(frozen=True, slots=True)
class BaseTargetTruth:
    target: TargetKind
    case_probabilities: tuple[float, ...]
    expected_brier: float
    expected_log_score: float
    fixed_bin_calibration: float
    equal_mass_calibration: float
    l2_calibration: float
    classification_error: float
    aurc: float
    operational_risk: tuple[OperationalRiskPoint, ...]


@dataclass(frozen=True, slots=True)
class BaseTargetReplicateResult:
    target: TargetKind
    brier: float
    log_score: float
    fixed_bin_ece: float
    equal_mass_ece: float
    debiased_squared_l2_row: float
    debiased_squared_l2_independent_unit: float
    debiased_l2_row: float
    debiased_l2_independent_unit: float
    classification_error: float
    aurc: float
    brier_intervals: MeanLossIntervals
    fixed_bin_counts: tuple[int, ...]
    equal_mass_bin_counts: tuple[int, ...]
    risk_coverage: tuple[RiskCoveragePoint, ...]
    operational_risk: tuple[OperationalRiskPoint, ...]


@dataclass(frozen=True, slots=True)
class BaseReplicateResult:
    design_sha256: str
    cell_key: str
    phase: MetricLabPhase
    replicate_index: int
    sampling_unit: SamplingUnit
    item_count: int
    row_count: int
    realized_budget_rows: int
    targets: dict[TargetKind, BaseTargetReplicateResult]


@dataclass(frozen=True, slots=True)
class MetricMonteCarloSummary:
    true_value: float
    mean_estimate: float
    bias: float
    rmse: float
    mc_se: float
    bias_mc_se: float
    rmse_mc_se: float
    interval_coverage: float | None
    coverage_mc_interval: MeanInterval | None
    mean_interval_width: float | None


@dataclass(frozen=True, slots=True)
class TargetCellSummary:
    target: TargetKind
    truth: BaseTargetTruth
    metrics: dict[str, MetricMonteCarloSummary]


@dataclass(frozen=True, slots=True)
class BaseCellSummary:
    design_sha256: str
    cell_key: str
    phase: MetricLabPhase
    sampling_unit: SamplingUnit
    replicates_count: int
    realized_budget_rows: int
    targets: dict[TargetKind, TargetCellSummary]


@dataclass(frozen=True, slots=True)
class _BaseRow:
    item_id: int
    forecast: float
    clean_probability: float
    clean_label: int
    noisy_label: int


def _sample_standard_error(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _wilson_binomial_interval(successes: int, trials: int) -> MeanInterval:
    if type(successes) is not int or type(trials) is not int:
        raise TypeError("binomial counts must be exact integers")
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("binomial counts require 0 <= successes <= trials and trials >= 1")
    estimate = successes / trials
    z2 = _NORMAL_95_Z**2
    denominator = 1.0 + z2 / trials
    center = (estimate + z2 / (2.0 * trials)) / denominator
    half_width = (
        _NORMAL_95_Z
        * math.sqrt(
            estimate * (1.0 - estimate) / trials + z2 / (4.0 * trials**2)
        )
        / denominator
    )
    return MeanInterval(
        estimate=estimate,
        se=None,
        lower=max(0.0, center - half_width),
        upper=min(1.0, center + half_width),
    )


def _normal_interval(estimate: float, se: float) -> MeanInterval:
    return MeanInterval(
        estimate=estimate,
        se=se,
        lower=estimate - _NORMAL_95_Z * se,
        upper=estimate + _NORMAL_95_Z * se,
    )


def mean_loss_intervals(
    *, values: Sequence[float], item_ids: Sequence[int]
) -> MeanLossIntervals:
    """Return row-naive and item-cluster normal intervals for a mean quantity."""
    if len(values) != len(item_ids) or not values:
        raise ValueError("values and item_ids must have equal non-zero length")
    estimate = sum(values) / len(values)
    naive = _normal_interval(estimate, _sample_standard_error(values))

    grouped: dict[int, list[float]] = defaultdict(list)
    for value, item_id in zip(values, item_ids, strict=True):
        grouped[item_id].append(float(value))
    item_means = tuple(sum(group) / len(group) for _, group in sorted(grouped.items()))
    cluster = _normal_interval(estimate, _sample_standard_error(item_means))
    half_width = math.sqrt(math.log(2.0 / 0.05) / (2.0 * len(item_means)))
    cluster_hoeffding = MeanInterval(
        estimate=estimate,
        se=None,
        lower=max(0.0, estimate - half_width),
        upper=min(1.0, estimate + half_width),
    )
    return MeanLossIntervals(
        naive=naive, cluster=cluster, cluster_hoeffding=cluster_hoeffding
    )


def independent_unit_debiased_squared_l2(
    *,
    forecasts: Sequence[float],
    outcomes: Sequence[int | float],
    item_ids: Sequence[int],
    num_bins: int,
) -> float:
    """Kumar squared-L2 estimator on one Bernoulli observation per item.

    Repeated rows are not independent when an item-level outcome is shared.  The
    independent-unit diagnostic therefore deterministically keeps the first row
    for each item and rejects within-item forecast disagreement.  This gives the
    same answer regardless of how many identical shared rows were duplicated.
    """
    if (
        len(forecasts) != len(outcomes)
        or len(forecasts) != len(item_ids)
        or not forecasts
    ):
        raise ValueError(
            "forecasts, outcomes, and item_ids must have equal non-zero length"
        )
    first: dict[int, tuple[float, int | float]] = {}
    for forecast, outcome, item_id in zip(forecasts, outcomes, item_ids, strict=True):
        existing = first.get(item_id)
        if existing is None:
            first[item_id] = (float(forecast), outcome)
        elif existing[0] != float(forecast):
            raise ValueError(
                "within-item forecasts must be identical for independent-unit reduction"
            )
    ordered = tuple(first[item_id] for item_id in sorted(first))
    return debiased_squared_l2_calibration_error(
        tuple(item[0] for item in ordered),
        tuple(item[1] for item in ordered),
        num_bins=num_bins,
    )


def risk_coverage_tie_blocks(
    *, confidences: Sequence[float], errors: Sequence[float]
) -> tuple[RiskCoveragePoint, ...]:
    """Risk-coverage points at complete confidence-tie block boundaries only."""
    if len(confidences) != len(errors) or not confidences:
        raise ValueError("confidences and errors must have equal non-zero length")
    blocks: dict[float, list[float]] = defaultdict(list)
    for confidence, error in zip(confidences, errors, strict=True):
        blocks[float(confidence)].append(float(error))
    n = len(confidences)
    selected = 0
    cumulative_error = 0.0
    points: list[RiskCoveragePoint] = []
    for confidence in sorted(blocks, reverse=True):
        block = blocks[confidence]
        selected += len(block)
        cumulative_error += sum(block)
        points.append(
            RiskCoveragePoint(
                coverage=selected / n,
                risk=cumulative_error / selected,
                selected_count=selected,
                confidence_floor=confidence,
            )
        )
    return tuple(points)


def operational_risk_points(
    curve: Sequence[RiskCoveragePoint],
    *,
    requested_coverages: Sequence[float],
) -> tuple[OperationalRiskPoint, ...]:
    """Match requested coverages to the first attainable complete tie-block coverage."""
    if not curve:
        raise ValueError("curve must be non-empty")
    if not requested_coverages:
        raise ValueError("requested_coverages must be non-empty")
    result: list[OperationalRiskPoint] = []
    for requested in requested_coverages:
        if not 0.0 < requested <= 1.0:
            raise ValueError("requested coverages must lie in (0, 1]")
        match = next((point for point in curve if point.coverage >= requested), None)
        if match is None:
            raise ValueError("risk-coverage curve does not reach requested coverage")
        result.append(
            OperationalRiskPoint(
                requested_coverage=float(requested),
                realized_coverage=match.coverage,
                risk=match.risk,
                selected_count=match.selected_count,
            )
        )
    return tuple(result)


def official_get_aurc_reference_behavior(
    residuals: Sequence[float], confidence: Sequence[float]
) -> float:
    """Reproduce han678/AsymptoticAURC ``get_AURC`` at fa445bcd.

    This function is intentionally a negative control.  The upstream loop divides
    the post-removal accumulated residual by ``m - i`` rather than the remaining
    count, which yields 0.5 instead of the exact 0.75 on the canonical two-item
    counterexample.  Do not use this as the laboratory oracle.
    """
    if len(residuals) != len(confidence) or not residuals:
        raise ValueError("residuals and confidence must have equal non-zero length")
    m = len(residuals)
    sorted_indices = sorted(range(m), key=lambda index: confidence[index])
    acc = sum(float(residuals[index]) for index in sorted_indices)
    risks = [acc / m]
    for i in range(0, len(sorted_indices) - 1):
        acc -= float(residuals[sorted_indices[i]])
        risks.append(acc / (m - i))
    return sum(risks) / len(risks)


def _population_l2_from_fixed_bins(
    forecasts: Sequence[float], target_probabilities: Sequence[float], num_bins: int
) -> float:
    _, details = fixed_bin_ece(forecasts, target_probabilities, num_bins=num_bins)
    n = len(forecasts)
    squared = sum((detail.count / n) * detail.gap**2 for detail in details)
    return math.sqrt(squared)


def _truth_case_probabilities(
    *, design: MetricLabDesign, cell: BaseCellSpec, sampling_unit: SamplingUnit
) -> tuple[float, ...]:
    p_values = design.p_value_floats
    if sampling_unit is SamplingUnit.FIXED_CASE:
        return generate_item_probabilities(
            p_values=p_values,
            n_items=cell.n_base_items,
            sampling_unit=SamplingUnit.FIXED_CASE,
            seed=0,
        )
    if sampling_unit is SamplingUnit.QUESTION_POPULATION:
        # The population estimand is the declared equal-weight support, not one
        # realized sample from it.
        return p_values
    raise TypeError("unsupported sampling_unit")


def base_target_truth(
    *,
    design: MetricLabDesign,
    cell: BaseCellSpec,
    target: TargetKind,
    sampling_unit: SamplingUnit,
) -> BaseTargetTruth:
    """Analytic/population truth for one target under a base-grid cell."""
    if target not in {TargetKind.CLEAN_TRUTH, TargetKind.NOISY_REFERENCE}:
        raise ValueError(
            "base-grid truth supports clean_truth and noisy_reference targets"
        )
    case_p = _truth_case_probabilities(
        design=design, cell=cell, sampling_unit=sampling_unit
    )
    forecasts = tuple(compute_forecast(p, cell.forecast_kind) for p in case_p)
    if target is TargetKind.CLEAN_TRUTH:
        target_p = case_p
    else:
        target_p = tuple(
            noisy_target_probability(p, cell.reference_flip_eta) for p in case_p
        )

    expected_b = sum(
        expected_brier(p, q) for p, q in zip(target_p, forecasts, strict=True)
    ) / len(forecasts)
    expected_log = sum(
        expected_log_score(p, q) for p, q in zip(target_p, forecasts, strict=True)
    ) / len(forecasts)
    fixed_ece = fixed_bin_ece(forecasts, target_p, num_bins=design.fixed_bin_count)[0]
    equal_ece = equal_mass_ece(
        forecasts, target_p, num_bins=design.equal_mass_bin_count
    )[0]
    l2 = _population_l2_from_fixed_bins(
        forecasts, target_p, num_bins=design.fixed_bin_count
    )
    predictions = tuple(
        binary_decision(
            q,
            threshold=design.decision_threshold_float,
            tie_break=design.decision_tie_break,
        )
        for q in forecasts
    )
    error_probabilities = tuple(
        (1.0 - p) if prediction == 1 else p
        for p, prediction in zip(target_p, predictions, strict=True)
    )
    classification_error = sum(error_probabilities) / len(error_probabilities)
    confidences = tuple(max(q, 1.0 - q) for q in forecasts)
    aurc = compute_aurc(
        confidences,
        error_probabilities,
        tie_convention=design.aurc_tie_convention,
    )
    risk_curve = risk_coverage_tie_blocks(
        confidences=confidences, errors=error_probabilities
    )
    operational = operational_risk_points(
        risk_curve, requested_coverages=design.operational_coverage_floats
    )
    return BaseTargetTruth(
        target=target,
        case_probabilities=case_p,
        expected_brier=expected_b,
        expected_log_score=expected_log,
        fixed_bin_calibration=fixed_ece,
        equal_mass_calibration=equal_ece,
        l2_calibration=l2,
        classification_error=classification_error,
        aurc=aurc,
        operational_risk=operational,
    )


def _generate_base_rows(
    *,
    design: MetricLabDesign,
    cell: BaseCellSpec,
    phase: MetricLabPhase,
    replicate_index: int,
    sampling_unit: SamplingUnit,
) -> tuple[_BaseRow, ...]:
    design_hash = canonical_record_hash(design)
    case_seed = derive_metric_lab_seed(
        design_hash=design_hash,
        phase=phase,
        cell_key=cell.cell_key,
        replicate_index=replicate_index,
        purpose="case-population",
    )
    outcome_seed = derive_metric_lab_seed(
        design_hash=design_hash,
        phase=phase,
        cell_key=cell.cell_key,
        replicate_index=replicate_index,
        purpose="outcomes",
    )
    case_probabilities = generate_item_probabilities(
        p_values=design.p_value_floats,
        n_items=cell.n_base_items,
        sampling_unit=sampling_unit,
        seed=case_seed,
    )
    rng = random.Random(outcome_seed)
    rows: list[_BaseRow] = []
    for item_id, p in enumerate(case_probabilities):
        forecast = compute_forecast(p, cell.forecast_kind)
        shared_clean = 1 if rng.random() < p else 0
        for _ in range(cell.k_within_item):
            clean = (
                shared_clean
                if cell.shared_within_item
                else (1 if rng.random() < p else 0)
            )
            noisy = clean
            if cell.reference_flip_eta > 0.0 and rng.random() < cell.reference_flip_eta:
                noisy = 1 - noisy
            rows.append(
                _BaseRow(
                    item_id=item_id,
                    forecast=forecast,
                    clean_probability=p,
                    clean_label=clean,
                    noisy_label=noisy,
                )
            )
    return tuple(rows)


def _evaluate_target_rows(
    *, design: MetricLabDesign, rows: tuple[_BaseRow, ...], target: TargetKind
) -> BaseTargetReplicateResult:
    if target is TargetKind.CLEAN_TRUTH:
        outcomes = tuple(row.clean_label for row in rows)
    elif target is TargetKind.NOISY_REFERENCE:
        outcomes = tuple(row.noisy_label for row in rows)
    else:
        raise ValueError("base-grid replicate supports clean_truth and noisy_reference")
    forecasts = tuple(row.forecast for row in rows)
    item_ids = tuple(row.item_id for row in rows)
    brier_losses = tuple(
        (forecast - outcome) ** 2
        for forecast, outcome in zip(forecasts, outcomes, strict=True)
    )
    brier = brier_score(forecasts, outcomes)
    log = log_score(forecasts, outcomes)
    fixed, fixed_details = fixed_bin_ece(
        forecasts, outcomes, num_bins=design.fixed_bin_count
    )
    equal, equal_details = equal_mass_ece(
        forecasts, outcomes, num_bins=design.equal_mass_bin_count
    )
    squared_l2_row = debiased_squared_l2_calibration_error(
        forecasts, outcomes, num_bins=design.fixed_bin_count
    )
    squared_l2_independent = independent_unit_debiased_squared_l2(
        forecasts=forecasts,
        outcomes=outcomes,
        item_ids=item_ids,
        num_bins=design.fixed_bin_count,
    )
    l2_row = math.sqrt(max(squared_l2_row, 0.0))
    l2_independent = math.sqrt(max(squared_l2_independent, 0.0))
    predictions = tuple(
        binary_decision(
            forecast,
            threshold=design.decision_threshold_float,
            tie_break=design.decision_tie_break,
        )
        for forecast in forecasts
    )
    errors = tuple(
        float(prediction != outcome)
        for prediction, outcome in zip(predictions, outcomes, strict=True)
    )
    confidences = tuple(max(forecast, 1.0 - forecast) for forecast in forecasts)
    risk_curve = risk_coverage_tie_blocks(confidences=confidences, errors=errors)
    return BaseTargetReplicateResult(
        target=target,
        brier=brier,
        log_score=log,
        fixed_bin_ece=fixed,
        equal_mass_ece=equal,
        debiased_squared_l2_row=squared_l2_row,
        debiased_squared_l2_independent_unit=squared_l2_independent,
        debiased_l2_row=l2_row,
        debiased_l2_independent_unit=l2_independent,
        classification_error=sum(errors) / len(errors),
        aurc=compute_aurc(
            confidences,
            errors,
            tie_convention=design.aurc_tie_convention,
        ),
        brier_intervals=mean_loss_intervals(values=brier_losses, item_ids=item_ids),
        fixed_bin_counts=tuple(detail.count for detail in fixed_details),
        equal_mass_bin_counts=tuple(detail.count for detail in equal_details),
        risk_coverage=risk_curve,
        operational_risk=operational_risk_points(
            risk_curve, requested_coverages=design.operational_coverage_floats
        ),
    )


def evaluate_base_replicate(
    *,
    design: MetricLabDesign,
    cell: BaseCellSpec,
    phase: MetricLabPhase,
    replicate_index: int,
    sampling_unit: SamplingUnit,
) -> BaseReplicateResult:
    if type(replicate_index) is not int or replicate_index < 0:
        raise ValueError("replicate_index must be a non-negative exact integer")
    rows = _generate_base_rows(
        design=design,
        cell=cell,
        phase=phase,
        replicate_index=replicate_index,
        sampling_unit=sampling_unit,
    )
    targets = {
        target: _evaluate_target_rows(design=design, rows=rows, target=target)
        for target in (TargetKind.CLEAN_TRUTH, TargetKind.NOISY_REFERENCE)
    }
    return BaseReplicateResult(
        design_sha256=canonical_record_hash(design),
        cell_key=cell.cell_key,
        phase=phase,
        replicate_index=replicate_index,
        sampling_unit=sampling_unit,
        item_count=cell.n_base_items,
        row_count=len(rows),
        realized_budget_rows=len(rows),
        targets=targets,
    )


def summarize_metric_estimates(
    *,
    estimates: Sequence[float],
    true_value: float,
    intervals: Sequence[tuple[float, float]] | None = None,
) -> MetricMonteCarloSummary:
    if not estimates:
        raise ValueError("estimates must be non-empty")
    mean_estimate = sum(estimates) / len(estimates)
    bias = mean_estimate - true_value
    rmse = math.sqrt(
        sum((estimate - true_value) ** 2 for estimate in estimates) / len(estimates)
    )
    mc_se = _sample_standard_error(estimates)
    squared_errors = tuple((estimate - true_value) ** 2 for estimate in estimates)
    mse_mc_se = _sample_standard_error(squared_errors)
    rmse_mc_se = 0.0 if rmse == 0.0 else mse_mc_se / (2.0 * rmse)
    coverage: float | None = None
    coverage_mc_interval: MeanInterval | None = None
    width: float | None = None
    if intervals is not None:
        if len(intervals) != len(estimates):
            raise ValueError("intervals length must match estimates")
        coverage_count = sum(
            lower <= true_value <= upper for lower, upper in intervals
        )
        coverage = coverage_count / len(intervals)
        coverage_mc_interval = _wilson_binomial_interval(
            coverage_count, len(intervals)
        )
        width = sum(upper - lower for lower, upper in intervals) / len(intervals)
    return MetricMonteCarloSummary(
        true_value=true_value,
        mean_estimate=mean_estimate,
        bias=bias,
        rmse=rmse,
        mc_se=mc_se,
        bias_mc_se=mc_se,
        rmse_mc_se=rmse_mc_se,
        interval_coverage=coverage,
        coverage_mc_interval=coverage_mc_interval,
        mean_interval_width=width,
    )


def summarize_base_cell(
    *,
    design: MetricLabDesign,
    cell: BaseCellSpec,
    phase: MetricLabPhase,
    sampling_unit: SamplingUnit,
    num_replicates: int,
) -> BaseCellSummary:
    if type(num_replicates) is not int or num_replicates < 1:
        raise ValueError("num_replicates must be a positive exact integer")
    replicates = tuple(
        evaluate_base_replicate(
            design=design,
            cell=cell,
            phase=phase,
            replicate_index=index,
            sampling_unit=sampling_unit,
        )
        for index in range(num_replicates)
    )
    target_summaries: dict[TargetKind, TargetCellSummary] = {}
    for target in (TargetKind.CLEAN_TRUTH, TargetKind.NOISY_REFERENCE):
        truth = base_target_truth(
            design=design,
            cell=cell,
            target=target,
            sampling_unit=sampling_unit,
        )
        values = tuple(rep.targets[target] for rep in replicates)
        metric_specs = {
            "brier": (
                tuple(value.brier for value in values),
                truth.expected_brier,
            ),
            "log_score": (
                tuple(value.log_score for value in values),
                truth.expected_log_score,
            ),
            "fixed_bin_ece": (
                tuple(value.fixed_bin_ece for value in values),
                truth.fixed_bin_calibration,
            ),
            "equal_mass_ece": (
                tuple(value.equal_mass_ece for value in values),
                truth.equal_mass_calibration,
            ),
            "debiased_squared_l2_row": (
                tuple(value.debiased_squared_l2_row for value in values),
                truth.l2_calibration**2,
            ),
            "debiased_squared_l2_independent_unit": (
                tuple(value.debiased_squared_l2_independent_unit for value in values),
                truth.l2_calibration**2,
            ),
            "debiased_l2_row": (
                tuple(value.debiased_l2_row for value in values),
                truth.l2_calibration,
            ),
            "debiased_l2_independent_unit": (
                tuple(value.debiased_l2_independent_unit for value in values),
                truth.l2_calibration,
            ),
            "classification_error": (
                tuple(value.classification_error for value in values),
                truth.classification_error,
            ),
            "aurc": (
                tuple(value.aurc for value in values),
                truth.aurc,
            ),
        }
        for index, requested in enumerate(design.operational_coverage_floats):
            metric_specs[f"risk_at_coverage_{requested:g}"] = (
                tuple(value.operational_risk[index].risk for value in values),
                truth.operational_risk[index].risk,
            )
            metric_specs[f"realized_coverage_at_request_{requested:g}"] = (
                tuple(
                    value.operational_risk[index].realized_coverage for value in values
                ),
                truth.operational_risk[index].realized_coverage,
            )
        summaries = {
            name: summarize_metric_estimates(estimates=estimates, true_value=true_value)
            for name, (estimates, true_value) in metric_specs.items()
        }
        brier_estimates = tuple(value.brier for value in values)
        summaries["brier_cluster_interval"] = summarize_metric_estimates(
            estimates=brier_estimates,
            true_value=truth.expected_brier,
            intervals=tuple(
                (
                    value.brier_intervals.cluster.lower,
                    value.brier_intervals.cluster.upper,
                )
                for value in values
            ),
        )
        summaries["brier_cluster_hoeffding_interval"] = summarize_metric_estimates(
            estimates=brier_estimates,
            true_value=truth.expected_brier,
            intervals=tuple(
                (
                    value.brier_intervals.cluster_hoeffding.lower,
                    value.brier_intervals.cluster_hoeffding.upper,
                )
                for value in values
            ),
        )
        summaries["brier_naive_interval"] = summarize_metric_estimates(
            estimates=brier_estimates,
            true_value=truth.expected_brier,
            intervals=tuple(
                (
                    value.brier_intervals.naive.lower,
                    value.brier_intervals.naive.upper,
                )
                for value in values
            ),
        )
        target_summaries[target] = TargetCellSummary(
            target=target,
            truth=truth,
            metrics=summaries,
        )

    return BaseCellSummary(
        design_sha256=canonical_record_hash(design),
        cell_key=cell.cell_key,
        phase=phase,
        sampling_unit=sampling_unit,
        replicates_count=num_replicates,
        realized_budget_rows=sum(rep.realized_budget_rows for rep in replicates),
        targets=target_summaries,
    )


def evaluate_prevalence_control(cell: PrevalenceCellSpec) -> PrevalenceControlResult:
    return PrevalenceControlResult(
        cell_key=cell.cell_key,
        prevalence=cell.prevalence,
        tpr=cell.tpr,
        fpr=cell.fpr,
        precision=prevalence_precision(
            tpr=cell.tpr, fpr=cell.fpr, prevalence=cell.prevalence
        ),
    )


def run_auxiliary_deterministic_controls(
    design: MetricLabDesign,
) -> AuxiliaryDeterministicControls:
    prevalence = tuple(
        evaluate_prevalence_control(cell) for cell in prevalence_grid_cells(design)
    )
    subgroup = subgroup_calibration_cancellation_control()
    upstream = official_get_aurc_reference_behavior((0.0, 1.0), (0.1, 0.9))
    oracle = 0.75
    return AuxiliaryDeterministicControls(
        prevalence=prevalence,
        subgroup=subgroup,
        aurc_upstream_negative_control=upstream,
        aurc_prefix_oracle=oracle,
        aurc_defect_detected=not math.isclose(
            upstream, oracle, rel_tol=0.0, abs_tol=0.0
        ),
        positive_class_inversion=positive_class_inversion_control(),
        label_noise_nonidentifiability=label_noise_nonidentifiability_control(),
        reference_ranking_reversal=check_reference_ranking_reversal(
            clean_p=0.9,
            q_clean=0.9,
            q_noisy=0.74,
            eta=0.2,
        ),
    )


@dataclass(frozen=True, slots=True)
class AmbiguityTargetTruth:
    target: TargetKind
    expected_brier: float
    expected_log_score: float
    fixed_bin_calibration: float
    equal_mass_calibration: float
    l2_calibration: float
    classification_error: float
    aurc: float
    operational_risk: tuple[OperationalRiskPoint, ...]


@dataclass(frozen=True, slots=True)
class AmbiguityTargetReplicateResult:
    target: TargetKind
    evaluation_count: int
    brier: float
    log_score: float
    fixed_bin_ece: float
    equal_mass_ece: float
    debiased_squared_l2: float | None
    debiased_l2: float | None
    classification_error: float
    aurc: float
    brier_intervals: MeanLossIntervals
    risk_coverage: tuple[RiskCoveragePoint, ...]
    operational_risk: tuple[OperationalRiskPoint, ...]


@dataclass(frozen=True, slots=True)
class AmbiguityReplicateResult:
    design_sha256: str
    cell_key: str
    phase: MetricLabPhase
    replicate_index: int
    item_count: int
    realized_budget_ratings: int
    targets: dict[TargetKind, AmbiguityTargetReplicateResult]


@dataclass(frozen=True, slots=True)
class AmbiguityTargetCellSummary:
    target: TargetKind
    truth: AmbiguityTargetTruth
    metrics: dict[str, MetricMonteCarloSummary]


@dataclass(frozen=True, slots=True)
class AmbiguityCellSummary:
    design_sha256: str
    cell_key: str
    phase: MetricLabPhase
    replicates_count: int
    realized_budget_ratings: int
    targets: dict[TargetKind, AmbiguityTargetCellSummary]


def _majority_positive_probability(theta: float, raters_per_item: int) -> float:
    threshold = raters_per_item // 2 + 1
    return sum(
        math.comb(raters_per_item, positives)
        * theta**positives
        * (1.0 - theta) ** (raters_per_item - positives)
        for positives in range(threshold, raters_per_item + 1)
    )


def _ambiguity_truth_from_probabilities(
    *,
    design: MetricLabDesign,
    cell: AmbiguityCellSpec,
    target: TargetKind,
    thetas: tuple[float, ...],
    forecasts: tuple[float, ...],
) -> AmbiguityTargetTruth:
    if target in {TargetKind.RATER_DISTRIBUTION, TargetKind.SOFT_LABEL}:
        target_probabilities = thetas
    elif target is TargetKind.MAJORITY_CONSENSUS:
        target_probabilities = tuple(
            _majority_positive_probability(theta, cell.raters_per_item)
            for theta in thetas
        )
    else:
        raise ValueError("unsupported ambiguity target")

    if target is TargetKind.SOFT_LABEL:
        expected_b = sum(
            (forecast - theta) ** 2 + theta * (1.0 - theta) / cell.raters_per_item
            for theta, forecast in zip(thetas, forecasts, strict=True)
        ) / len(thetas)
    else:
        expected_b = sum(
            expected_brier(probability, forecast)
            for probability, forecast in zip(
                target_probabilities, forecasts, strict=True
            )
        ) / len(forecasts)
    expected_log = sum(
        expected_log_score(probability, forecast)
        for probability, forecast in zip(target_probabilities, forecasts, strict=True)
    ) / len(forecasts)
    fixed_ece = fixed_bin_ece(
        forecasts, target_probabilities, num_bins=design.fixed_bin_count
    )[0]
    equal_ece = equal_mass_ece(
        forecasts, target_probabilities, num_bins=design.equal_mass_bin_count
    )[0]
    l2 = _population_l2_from_fixed_bins(
        forecasts, target_probabilities, num_bins=design.fixed_bin_count
    )
    predictions = tuple(
        binary_decision(
            forecast,
            threshold=design.decision_threshold_float,
            tie_break=design.decision_tie_break,
        )
        for forecast in forecasts
    )
    error_probabilities = tuple(
        (1.0 - probability) if prediction == 1 else probability
        for probability, prediction in zip(
            target_probabilities, predictions, strict=True
        )
    )
    classification_error = sum(error_probabilities) / len(error_probabilities)
    confidences = tuple(max(forecast, 1.0 - forecast) for forecast in forecasts)
    aurc = compute_aurc(
        confidences,
        error_probabilities,
        tie_convention=design.aurc_tie_convention,
    )
    risk_curve = risk_coverage_tie_blocks(
        confidences=confidences, errors=error_probabilities
    )
    operational = operational_risk_points(
        risk_curve, requested_coverages=design.operational_coverage_floats
    )
    return AmbiguityTargetTruth(
        target=target,
        expected_brier=expected_b,
        expected_log_score=expected_log,
        fixed_bin_calibration=fixed_ece,
        equal_mass_calibration=equal_ece,
        l2_calibration=l2,
        classification_error=classification_error,
        aurc=aurc,
        operational_risk=operational,
    )


def ambiguity_target_truths(
    *, design: MetricLabDesign, cell: AmbiguityCellSpec
) -> dict[TargetKind, AmbiguityTargetTruth]:
    thetas = generate_item_probabilities(
        p_values=design.ambiguity_theta_floats,
        n_items=cell.n_items,
        sampling_unit=SamplingUnit.FIXED_CASE,
        seed=0,
    )
    forecasts = tuple(compute_forecast(theta, cell.forecast_kind) for theta in thetas)
    return {
        target: _ambiguity_truth_from_probabilities(
            design=design,
            cell=cell,
            target=target,
            thetas=thetas,
            forecasts=forecasts,
        )
        for target in (
            TargetKind.RATER_DISTRIBUTION,
            TargetKind.SOFT_LABEL,
            TargetKind.MAJORITY_CONSENSUS,
        )
    }


def _evaluate_ambiguity_target(
    *,
    design: MetricLabDesign,
    target: TargetKind,
    forecasts: tuple[float, ...],
    outcomes: tuple[float, ...],
    item_ids: tuple[int, ...],
) -> AmbiguityTargetReplicateResult:
    predictions = tuple(
        binary_decision(
            forecast,
            threshold=design.decision_threshold_float,
            tie_break=design.decision_tie_break,
        )
        for forecast in forecasts
    )
    errors = tuple(
        (1.0 - outcome) if prediction == 1 else outcome
        for prediction, outcome in zip(predictions, outcomes, strict=True)
    )
    confidences = tuple(max(forecast, 1.0 - forecast) for forecast in forecasts)
    brier_losses = tuple(
        (forecast - outcome) ** 2
        for forecast, outcome in zip(forecasts, outcomes, strict=True)
    )
    fixed = fixed_bin_ece(forecasts, outcomes, num_bins=design.fixed_bin_count)[0]
    equal = equal_mass_ece(forecasts, outcomes, num_bins=design.equal_mass_bin_count)[0]
    risk_curve = risk_coverage_tie_blocks(confidences=confidences, errors=errors)
    squared_l2: float | None
    l2_presentation: float | None
    if target is TargetKind.SOFT_LABEL:
        squared_l2 = None
        l2_presentation = None
    else:
        squared_l2 = debiased_squared_l2_calibration_error(
            forecasts, outcomes, num_bins=design.fixed_bin_count
        )
        l2_presentation = math.sqrt(max(squared_l2, 0.0))
    return AmbiguityTargetReplicateResult(
        target=target,
        evaluation_count=len(outcomes),
        brier=brier_score(forecasts, outcomes),
        log_score=log_score(forecasts, outcomes),
        fixed_bin_ece=fixed,
        equal_mass_ece=equal,
        debiased_squared_l2=squared_l2,
        debiased_l2=l2_presentation,
        classification_error=sum(errors) / len(errors),
        aurc=compute_aurc(
            confidences, errors, tie_convention=design.aurc_tie_convention
        ),
        brier_intervals=mean_loss_intervals(values=brier_losses, item_ids=item_ids),
        risk_coverage=risk_curve,
        operational_risk=operational_risk_points(
            risk_curve, requested_coverages=design.operational_coverage_floats
        ),
    )


def evaluate_ambiguity_replicate(
    *,
    design: MetricLabDesign,
    cell: AmbiguityCellSpec,
    phase: MetricLabPhase,
    replicate_index: int,
) -> AmbiguityReplicateResult:
    design_hash = canonical_record_hash(design)
    seed = derive_metric_lab_seed(
        design_hash=design_hash,
        phase=phase,
        cell_key=cell.cell_key,
        replicate_index=replicate_index,
        purpose="rater-population",
    )
    items = generate_rater_ambiguity_items(
        theta_values=design.ambiguity_theta_floats,
        n_items=cell.n_items,
        raters_per_item=cell.raters_per_item,
        seed=seed,
    )
    item_forecasts = tuple(
        compute_forecast(item.rater_distribution_probability, cell.forecast_kind)
        for item in items
    )

    rater_forecasts: list[float] = []
    rater_outcomes: list[float] = []
    rater_item_ids: list[int] = []
    for item, forecast in zip(items, item_forecasts, strict=True):
        for rating in item.ratings:
            rater_forecasts.append(forecast)
            rater_outcomes.append(float(rating))
            rater_item_ids.append(item.item_id)

    targets = {
        TargetKind.RATER_DISTRIBUTION: _evaluate_ambiguity_target(
            design=design,
            target=TargetKind.RATER_DISTRIBUTION,
            forecasts=tuple(rater_forecasts),
            outcomes=tuple(rater_outcomes),
            item_ids=tuple(rater_item_ids),
        ),
        TargetKind.SOFT_LABEL: _evaluate_ambiguity_target(
            design=design,
            target=TargetKind.SOFT_LABEL,
            forecasts=item_forecasts,
            outcomes=tuple(item.soft_label_probability for item in items),
            item_ids=tuple(item.item_id for item in items),
        ),
        TargetKind.MAJORITY_CONSENSUS: _evaluate_ambiguity_target(
            design=design,
            target=TargetKind.MAJORITY_CONSENSUS,
            forecasts=item_forecasts,
            outcomes=tuple(float(item.majority_consensus_label) for item in items),
            item_ids=tuple(item.item_id for item in items),
        ),
    }
    return AmbiguityReplicateResult(
        design_sha256=design_hash,
        cell_key=cell.cell_key,
        phase=phase,
        replicate_index=replicate_index,
        item_count=cell.n_items,
        realized_budget_ratings=cell.n_items * cell.raters_per_item,
        targets=targets,
    )


def summarize_ambiguity_cell(
    *,
    design: MetricLabDesign,
    cell: AmbiguityCellSpec,
    phase: MetricLabPhase,
    num_replicates: int,
) -> AmbiguityCellSummary:
    if type(num_replicates) is not int or num_replicates < 1:
        raise ValueError("num_replicates must be a positive exact integer")
    replicates = tuple(
        evaluate_ambiguity_replicate(
            design=design,
            cell=cell,
            phase=phase,
            replicate_index=index,
        )
        for index in range(num_replicates)
    )
    truths = ambiguity_target_truths(design=design, cell=cell)
    target_summaries: dict[TargetKind, AmbiguityTargetCellSummary] = {}
    for target, truth in truths.items():
        values = tuple(rep.targets[target] for rep in replicates)
        metric_specs = {
            "brier": (tuple(value.brier for value in values), truth.expected_brier),
            "log_score": (
                tuple(value.log_score for value in values),
                truth.expected_log_score,
            ),
            "fixed_bin_ece": (
                tuple(value.fixed_bin_ece for value in values),
                truth.fixed_bin_calibration,
            ),
            "equal_mass_ece": (
                tuple(value.equal_mass_ece for value in values),
                truth.equal_mass_calibration,
            ),
            "classification_error": (
                tuple(value.classification_error for value in values),
                truth.classification_error,
            ),
            "aurc": (tuple(value.aurc for value in values), truth.aurc),
        }
        if target is not TargetKind.SOFT_LABEL:
            squared_values = tuple(value.debiased_squared_l2 for value in values)
            l2_values = tuple(value.debiased_l2 for value in values)
            squared_numeric = tuple(value for value in squared_values if value is not None)
            l2_numeric = tuple(value for value in l2_values if value is not None)
            if len(squared_numeric) != len(values) or len(l2_numeric) != len(values):
                raise AssertionError("Bernoulli L2 diagnostics unexpectedly missing")
            metric_specs["debiased_squared_l2"] = (
                squared_numeric,
                truth.l2_calibration**2,
            )
            metric_specs["debiased_l2"] = (
                l2_numeric,
                truth.l2_calibration,
            )
        for index, requested in enumerate(design.operational_coverage_floats):
            metric_specs[f"risk_at_coverage_{requested:g}"] = (
                tuple(value.operational_risk[index].risk for value in values),
                truth.operational_risk[index].risk,
            )
            metric_specs[f"realized_coverage_at_request_{requested:g}"] = (
                tuple(
                    value.operational_risk[index].realized_coverage for value in values
                ),
                truth.operational_risk[index].realized_coverage,
            )
        metrics = {
            name: summarize_metric_estimates(estimates=estimates, true_value=true_value)
            for name, (estimates, true_value) in metric_specs.items()
        }
        brier_estimates = tuple(value.brier for value in values)
        metrics["brier_cluster_interval"] = summarize_metric_estimates(
            estimates=brier_estimates,
            true_value=truth.expected_brier,
            intervals=tuple(
                (
                    value.brier_intervals.cluster.lower,
                    value.brier_intervals.cluster.upper,
                )
                for value in values
            ),
        )
        metrics["brier_cluster_hoeffding_interval"] = summarize_metric_estimates(
            estimates=brier_estimates,
            true_value=truth.expected_brier,
            intervals=tuple(
                (
                    value.brier_intervals.cluster_hoeffding.lower,
                    value.brier_intervals.cluster_hoeffding.upper,
                )
                for value in values
            ),
        )
        metrics["brier_naive_interval"] = summarize_metric_estimates(
            estimates=brier_estimates,
            true_value=truth.expected_brier,
            intervals=tuple(
                (
                    value.brier_intervals.naive.lower,
                    value.brier_intervals.naive.upper,
                )
                for value in values
            ),
        )
        target_summaries[target] = AmbiguityTargetCellSummary(
            target=target, truth=truth, metrics=metrics
        )
    return AmbiguityCellSummary(
        design_sha256=canonical_record_hash(design),
        cell_key=cell.cell_key,
        phase=phase,
        replicates_count=num_replicates,
        realized_budget_ratings=sum(rep.realized_budget_ratings for rep in replicates),
        targets=target_summaries,
    )
