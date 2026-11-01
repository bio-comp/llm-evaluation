"""Runtime target/reference arms for the known-truth metric laboratory."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from llm_evaluation.metric_lab_design import balanced_probability_schedule
from llm_evaluation.metric_laboratory import fixed_bin_ece, noisy_target_probability


class TargetKind(StrEnum):
    CLEAN_TRUTH = "clean_truth"
    NOISY_REFERENCE = "noisy_reference"
    RATER_DISTRIBUTION = "rater_distribution"
    SOFT_LABEL = "soft_label"
    MAJORITY_CONSENSUS = "majority_consensus"


class SamplingUnit(StrEnum):
    FIXED_CASE = "fixed_case"
    QUESTION_POPULATION = "question_population"


@dataclass(frozen=True, slots=True)
class AmbiguityRaterItem:
    """One item whose repeated labels represent genuine rater-population variation."""

    item_id: int
    arm_type: str
    rater_distribution_probability: float
    ratings: tuple[int, ...]
    soft_label_probability: float
    majority_consensus_label: int

    @property
    def target_values(self) -> dict[TargetKind, float]:
        return {
            TargetKind.RATER_DISTRIBUTION: self.rater_distribution_probability,
            TargetKind.SOFT_LABEL: self.soft_label_probability,
            TargetKind.MAJORITY_CONSENSUS: float(self.majority_consensus_label),
        }


@dataclass(frozen=True, slots=True)
class SubgroupCalibrationControl:
    pooled_ece: float
    subgroup_ece: dict[str, float]
    max_subgroup_ece: float


def binary_reference_targets(
    *, clean_probability: float, reference_flip_eta: float
) -> dict[TargetKind, float]:
    """Return explicit clean and symmetric-flip reference estimands."""
    if not 0.0 <= clean_probability <= 1.0:
        raise ValueError("clean_probability must lie in [0, 1]")
    return {
        TargetKind.CLEAN_TRUTH: clean_probability,
        TargetKind.NOISY_REFERENCE: noisy_target_probability(
            clean_probability, reference_flip_eta
        ),
    }


def generate_rater_ambiguity_items(
    *,
    theta_values: tuple[float, ...],
    n_items: int,
    raters_per_item: int,
    seed: int,
) -> tuple[AmbiguityRaterItem, ...]:
    """Generate repeated ratings from an explicit per-item rater distribution.

    This arm intentionally has no reference-noise parameter. Disagreement is the
    estimand-generating rater population, not corruption of a hidden hard label.
    An odd rater count makes the majority-consensus target deterministic without
    adding an arbitrary consensus tie-break rule.
    """
    if type(n_items) is not int or n_items < 1:
        raise ValueError("n_items must be a positive exact integer")
    if type(raters_per_item) is not int or raters_per_item < 1:
        raise ValueError("raters_per_item must be a positive exact integer")
    if raters_per_item % 2 == 0:
        raise ValueError("raters_per_item must be odd for a tie-free majority target")
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if not theta_values or len(theta_values) != len(set(theta_values)):
        raise ValueError("theta_values must be non-empty and unique")
    if any(not 0.0 <= theta <= 1.0 for theta in theta_values):
        raise ValueError("theta_values must lie in [0, 1]")

    thetas = balanced_probability_schedule(theta_values, n_items, seed=seed)
    rng = random.Random(seed ^ 0xA5A5_5A5A_1357_2468)
    items: list[AmbiguityRaterItem] = []
    for item_id, theta in enumerate(thetas):
        ratings = tuple(
            1 if rng.random() < theta else 0 for _ in range(raters_per_item)
        )
        positives = sum(ratings)
        soft = positives / raters_per_item
        majority = 1 if positives > raters_per_item / 2 else 0
        items.append(
            AmbiguityRaterItem(
                item_id=item_id,
                arm_type="ambiguity_rater",
                rater_distribution_probability=theta,
                ratings=ratings,
                soft_label_probability=soft,
                majority_consensus_label=majority,
            )
        )
    return tuple(items)


def prevalence_precision(*, tpr: float, fpr: float, prevalence: float) -> float:
    """Exact precision induced by prevalence under fixed TPR/FPR."""
    for value, label in ((tpr, "tpr"), (fpr, "fpr"), (prevalence, "prevalence")):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must lie in [0, 1]")
    numerator = tpr * prevalence
    denominator = numerator + fpr * (1.0 - prevalence)
    if denominator == 0.0:
        raise ValueError("precision is undefined when there are no predicted positives")
    return numerator / denominator


def subgroup_ece(
    *,
    forecasts: Sequence[float],
    outcomes: Sequence[int | float],
    groups: Sequence[str],
    num_bins: int,
) -> dict[str, float]:
    """Compute binned ECE separately by declared subgroup."""
    if (
        len(forecasts) != len(outcomes)
        or len(forecasts) != len(groups)
        or not forecasts
    ):
        raise ValueError(
            "forecasts, outcomes, and groups must have equal non-zero length"
        )
    by_group: dict[str, tuple[list[float], list[int | float]]] = {}
    for forecast, outcome, group in zip(forecasts, outcomes, groups, strict=True):
        if not group:
            raise ValueError("group labels must be non-empty")
        if group not in by_group:
            by_group[group] = ([], [])
        by_group[group][0].append(float(forecast))
        by_group[group][1].append(outcome)
    return {
        group: fixed_bin_ece(group_forecasts, group_outcomes, num_bins=num_bins)[0]
        for group, (group_forecasts, group_outcomes) in sorted(by_group.items())
    }


def subgroup_calibration_cancellation_control() -> SubgroupCalibrationControl:
    """Known-bad pooled calibration control with equal/opposite subgroup gaps."""
    forecasts = (0.25, 0.25, 0.75, 0.75)
    outcomes = (0, 0, 1, 1)
    groups = ("A", "A", "B", "B")
    pooled = fixed_bin_ece(forecasts, outcomes, num_bins=1)[0]
    subgroup = subgroup_ece(
        forecasts=forecasts, outcomes=outcomes, groups=groups, num_bins=1
    )
    return SubgroupCalibrationControl(
        pooled_ece=pooled,
        subgroup_ece=subgroup,
        max_subgroup_ece=max(subgroup.values()),
    )


def generate_item_probabilities(
    *,
    p_values: tuple[float, ...],
    n_items: int,
    sampling_unit: SamplingUnit,
    seed: int,
) -> tuple[float, ...]:
    """Generate the case population for fixed-case or question-population inference."""
    if not isinstance(sampling_unit, SamplingUnit):
        raise TypeError("sampling_unit must be a SamplingUnit")
    if type(n_items) is not int or n_items < 1:
        raise ValueError("n_items must be a positive exact integer")
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if not p_values or len(p_values) != len(set(p_values)):
        raise ValueError("p_values must be non-empty and unique")
    if any(not 0.0 <= probability <= 1.0 for probability in p_values):
        raise ValueError("p_values must lie in [0, 1]")

    if sampling_unit is SamplingUnit.FIXED_CASE:
        # The case set is fixed by design; replicate seeds affect outcomes, not which
        # cases exist.  Seed zero is a stable case-set construction namespace.
        return balanced_probability_schedule(p_values, n_items, seed=0)
    rng = random.Random(seed)
    return tuple(rng.choice(p_values) for _ in range(n_items))
