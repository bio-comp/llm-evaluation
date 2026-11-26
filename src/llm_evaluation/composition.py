"""Descriptive, composition-aware binary error rates.

Target strata, their identities, and target weights must be declared before
outcomes are observed. This routine assumes complete observed binary
error-count input for each declared stratum.

The reported marginal and standardized rates are descriptive measurements.
They neither establish causal attribution nor support calibration or alert
decisions. A zero observed count in a stratum carries no inference about
its underlying conditional error rate.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Literal, Sequence

CompositionStatus = Literal["ESTIMABLE", "NOT_ESTIMABLE"]


@dataclass(frozen=True, slots=True)
class DescriptiveCompositionRates:
    marginal_rate: float
    standardized_rate: float | None
    status: CompositionStatus
    observed_mix: tuple[float, ...]
    missing_target_strata: tuple[int, ...]


def descriptive_composition_rates(
    *,
    stratum_counts: Sequence[int],
    stratum_errors: Sequence[int],
    target_weights: Sequence[float],
) -> DescriptiveCompositionRates:
    counts_input = tuple(stratum_counts)
    errors_input = tuple(stratum_errors)
    weights_input = tuple(target_weights)

    if not counts_input or not errors_input or not weights_input:
        raise ValueError("stratum_counts, stratum_errors, and target_weights must be non-empty")
    if not (len(counts_input) == len(errors_input) == len(weights_input)):
        raise ValueError("stratum_counts, stratum_errors, and target_weights must have equal length")

    n = len(counts_input)

    counts: list[int] = []
    for value in counts_input:
        if type(value) is not int:
            raise ValueError("stratum_counts must contain exact integers")
        if value < 0:
            raise ValueError("stratum_counts must be non-negative")
        counts.append(value)

    errors: list[int] = []
    for value in errors_input:
        if type(value) is not int:
            raise ValueError("stratum_errors must contain exact integers")
        errors.append(value)

    for i in range(n):
        if errors[i] < 0 or errors[i] > counts[i]:
            raise ValueError("stratum_errors must satisfy 0 <= errors <= counts")

    total = sum(counts)
    if total <= 0:
        raise ValueError("total observed count must be positive")

    weights: list[float] = []
    for raw in weights_input:
        if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
            raise ValueError("target_weights must contain real numeric values")
        try:
            converted = float(raw)
        except OverflowError as exc:
            raise ValueError("target_weights must contain real numeric values") from exc
        if not math.isfinite(converted) or converted < 0.0:
            raise ValueError("target_weights must be finite and non-negative")
        weights.append(converted)

    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("target_weights must sum to 1")

    observed_mix = tuple(count / total for count in counts)
    marginal_rate = sum(errors) / total

    missing_target_strata = tuple(i for i in range(n) if counts[i] == 0 and weights[i] > 0.0)

    if missing_target_strata:
        return DescriptiveCompositionRates(
            marginal_rate=marginal_rate,
            standardized_rate=None,
            status="NOT_ESTIMABLE",
            observed_mix=observed_mix,
            missing_target_strata=missing_target_strata,
        )

    standardized_rate = 0.0
    for i in range(n):
        if counts[i] > 0 and weights[i] > 0.0:
            standardized_rate += weights[i] * (errors[i] / counts[i])

    return DescriptiveCompositionRates(
        marginal_rate=marginal_rate,
        standardized_rate=standardized_rate,
        status="ESTIMABLE",
        observed_mix=observed_mix,
        missing_target_strata=(),
    )
