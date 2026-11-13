"""Phase 0B1 covariance and known-conditional-null proxy controls.

Independent analytical oracles are deliberately separate from generator internals.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from fractions import Fraction
from typing import cast

import pytest

from llm_evaluation.trex_known_truth import (
    CovarianceFamily,
    LinearGaussianCondition,
    LinearGaussianDesign,
    aggregate_generated_datasets,
    generate_linear_dataset,
    score_generated_dataset,
)

_HISTORICAL_ID = (
    "trex0a:af481762ac77947358040fcdc3bb1a7dfdeba47bedc9671fe0389e30c75024fa"
)


def _design(
    family: CovarianceFamily,
    *,
    n: int = 80,
    p: int = 6,
    seed: int = 23,
    beta: tuple[float, ...] | None = None,
    rho: float | None = None,
    block_size: int | None = None,
    noise_scale: float = 0.7,
) -> LinearGaussianDesign:
    return LinearGaussianDesign(
        n=n,
        p=p,
        seed=seed,
        beta=beta if beta is not None else (2.0, *(0.0 for _ in range(p - 1))),
        covariance_family=family,
        rho=rho,
        block_size=block_size,
        noise_scale=noise_scale,
    )


def _covariance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Independent sample covariance with population divisor n."""
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    return sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True)) / len(a)


def _column(x: tuple[tuple[float, ...], ...], index: int) -> tuple[float, ...]:
    return tuple(row[index] for row in x)


def _expected_covariance(
    family: CovarianceFamily, j: int, k: int, rho: float, block_size: int
) -> float:
    """Population covariance from unit variances and specified block partitions."""
    if j == k:
        return 1.0
    if family == "independent_standard_gaussian":
        return 0.0
    if family == "equicorrelated_gaussian":
        return rho
    return rho if j // block_size == k // block_size else 0.0


@pytest.mark.parametrize(
    ("family", "rho", "block_size"),
    [
        ("independent_standard_gaussian", None, None),
        ("equicorrelated_gaussian", 0.0, None),
        ("equicorrelated_gaussian", 0.5, None),
        ("equicorrelated_gaussian", 0.9, None),
        ("block_correlated_gaussian", 0.0, 3),
        ("block_correlated_gaussian", 0.5, 3),
        ("block_correlated_gaussian", 0.9, 3),
    ],
)
def test_covariance_designs_reproduce_and_seeds_change(
    family: CovarianceFamily, rho: float | None, block_size: int | None
) -> None:
    design = _design(family, rho=rho, block_size=block_size)
    first = generate_linear_dataset(design)
    assert first == generate_linear_dataset(design)
    other = generate_linear_dataset(replace(design, seed=24))
    assert other != first
    assert other.x != first.x
    assert other.dataset_id != first.dataset_id
    assert first.true_support == frozenset({0})
    assert all(len(row) == design.p for row in first.x)
    assert len(first.sampling_ids) == len(set(first.sampling_ids)) == design.n
    assert len(first.group_ids) == len(set(first.group_ids)) == design.n
    prefix = "trex0a:" if rho is None else "trex0b1:"
    assert first.dataset_id.startswith(prefix)


@pytest.mark.parametrize(("n", "p"), [(80, 40), (80, 160), (80, 400), (200, 40)])
def test_correlated_high_and_low_dimensional_shapes(n: int, p: int) -> None:
    design = _design("block_correlated_gaussian", n=n, p=p, rho=0.9, block_size=7)
    result = generate_linear_dataset(design)
    assert len(result.x) == n
    assert all(len(row) == p for row in result.x)
    assert result.true_support == frozenset({0})
    assert score_generated_dataset(result, (0,)).metrics.fdp == 0


@pytest.mark.parametrize(
    "bad", [True, False, -0.1, 1.0, 4.0, math.nan, math.inf, -math.inf, "0.5"]
)
def test_reject_bad_equicorrelation(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _design("equicorrelated_gaussian", rho=cast(float, bad))


@pytest.mark.parametrize("bad", [True, False, 0, 1, -2, 7, 10, 2.5, "3"])
def test_reject_invalid_block_size(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _design("block_correlated_gaussian", rho=0.5, block_size=cast(int, bad))


@pytest.mark.parametrize(
    ("family", "rho", "block_size"),
    [
        ("independent_standard_gaussian", 0.0, None),
        ("independent_standard_gaussian", None, 2),
        ("equicorrelated_gaussian", None, None),
        ("equicorrelated_gaussian", 0.5, 2),
        ("block_correlated_gaussian", None, 3),
        ("block_correlated_gaussian", 0.5, None),
    ],
)
def test_reject_unused_or_missing_covariance_arguments(
    family: CovarianceFamily, rho: float | None, block_size: int | None
) -> None:
    with pytest.raises(ValueError):
        _design(family, rho=rho, block_size=block_size)


def test_unknown_family_and_block_bad_rho_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported covariance family"):
        _design(cast(CovarianceFamily, "toeplitz_gaussian"))
    with pytest.raises(ValueError, match="rho"):
        _design("block_correlated_gaussian", rho=-0.1, block_size=3)
    with pytest.raises(TypeError, match="rho"):
        _design("block_correlated_gaussian", rho=cast(float, True), block_size=3)


def test_valid_covariance_boundaries_and_remainder_blocks() -> None:
    low = _design("equicorrelated_gaussian", p=1, rho=0.0)
    assert len(generate_linear_dataset(low).x[0]) == 1
    near_one = _design("block_correlated_gaussian", p=7, rho=0.999, block_size=3)
    assert near_one.block_size == 3
    assert near_one.rho == 0.999
    assert len(generate_linear_dataset(near_one).x[0]) == 7
    # [0,1,2], [3,4,5], [6]; the final singleton has variance 1.


@pytest.mark.parametrize(
    ("family", "rho", "block_size"),
    [
        ("independent_standard_gaussian", None, None),
        ("equicorrelated_gaussian", 0.5, None),
        ("equicorrelated_gaussian", 0.9, None),
        ("block_correlated_gaussian", 0.5, 3),
        ("block_correlated_gaussian", 0.9, 3),
    ],
)
def test_empirical_population_covariances_and_independent_rows(
    family: CovarianceFamily, rho: float | None, block_size: int | None
) -> None:
    # Deterministic large sample; loose absolute error guards Monte Carlo variation.
    design = _design(family, n=7500, p=7, rho=rho, block_size=block_size)
    dataset = generate_linear_dataset(design)
    columns = [_column(dataset.x, j) for j in range(design.p)]
    for j in range(design.p):
        for k in range(j, design.p):
            expected = _expected_covariance(
                family, j, k, rho or 0.0, block_size or design.p
            )
            assert abs(_covariance(columns[j], columns[k]) - expected) < 0.065, (
                family,
                j,
                k,
                expected,
            )
    assert abs(_covariance(columns[0][:-1], columns[0][1:])) < 0.065


def test_cross_block_negative_control_catches_reused_common_factor() -> None:
    n = 5000
    actual = generate_linear_dataset(
        _design("block_correlated_gaussian", n=n, rho=0.9, block_size=3)
    )
    cross_block_cov = _covariance(_column(actual.x, 0), _column(actual.x, 3))
    assert abs(cross_block_cov) < 0.065
    # Known-bad counterfactual: a single latent factor incorrectly shared by blocks.
    rng = random.Random(123)
    broken_rows = tuple(
        tuple(
            math.sqrt(0.9) * shared + math.sqrt(0.1) * rng.gauss(0, 1) for _ in range(6)
        )
        for shared in (rng.gauss(0, 1) for _ in range(n))
    )
    broken_cov = _covariance(_column(broken_rows, 0), _column(broken_rows, 3))
    assert broken_cov > 0.75
    assert abs(broken_cov) >= 0.065  # independent-block oracle rejects it


def test_active_null_proxy_population_and_sample_controls() -> None:
    beta = (2.0, 0.0, 0.0)
    correlated = generate_linear_dataset(
        _design(
            "equicorrelated_gaussian",
            n=14000,
            p=3,
            beta=beta,
            rho=0.9,
            noise_scale=0.0,
        )
    )
    independent = generate_linear_dataset(
        _design(
            "independent_standard_gaussian",
            n=14000,
            p=3,
            beta=beta,
            noise_scale=0.0,
        )
    )
    # Population identity calculated without using empirical covariance estimates:
    # Cov(X1, y) = sum_k beta[k] Sigma[1,k] = 0.9 * 2 = 1.8.
    assert sum(
        value * _expected_covariance("equicorrelated_gaussian", 1, k, 0.9, 3)
        for k, value in enumerate(beta)
    ) == pytest.approx(1.8)
    assert (
        sum(
            value * _expected_covariance("independent_standard_gaussian", 1, k, 0.0, 3)
            for k, value in enumerate(beta)
        )
        == 0.0
    )
    sample_proxy = _covariance(_column(correlated.x, 1), correlated.y)
    sample_control = _covariance(_column(independent.x, 1), independent.y)
    assert sample_proxy == pytest.approx(1.8, abs=0.09)
    assert sample_control == pytest.approx(0.0, abs=0.09)
    assert correlated.true_support == independent.true_support == frozenset({0})
    proxy_only = score_generated_dataset(correlated, (1,))
    assert (proxy_only.metrics.tp, proxy_only.metrics.v, proxy_only.metrics.fdp) == (
        0,
        1,
        Fraction(1),
    )
    with_active = score_generated_dataset(correlated, (0, 1))
    assert (with_active.metrics.tp, with_active.metrics.v, with_active.metrics.fdp) == (
        1,
        1,
        Fraction(1, 2),
    )


def test_all_null_correlated_and_zero_noise_protections() -> None:
    design = _design(
        "block_correlated_gaussian",
        beta=(0.0,) * 6,
        rho=0.9,
        block_size=4,
        noise_scale=0.0,
    )
    dataset = generate_linear_dataset(design)
    assert dataset.true_support == frozenset()
    assert dataset.epsilon == (0.0,) * design.n
    assert dataset.y == (0.0,) * design.n
    selected = score_generated_dataset(dataset, (1,))
    empty = score_generated_dataset(
        generate_linear_dataset(replace(design, seed=24)), ()
    )
    summary = aggregate_generated_datasets((selected, empty))
    assert summary.empirical_fdr == Fraction(1, 2)
    assert summary.mean_applicable_tpr is None
    assert summary.tpr_nonapplicable_count == 2
    assert summary.empty_selection_rate == Fraction(1, 2)
    with pytest.raises(ValueError, match="zero-noise"):
        score_generated_dataset(
            replace(dataset, epsilon=(1e-30, *dataset.epsilon[1:])), ()
        )
    with pytest.raises(ValueError, match="dataset_id"):
        score_generated_dataset(replace(dataset, design=replace(design, rho=0.5)), ())


def test_covariance_conditions_not_aggregated_across_mixed_parameters() -> None:
    base = _design("block_correlated_gaussian", rho=0.5, block_size=3)
    a = score_generated_dataset(generate_linear_dataset(base), (0,))
    compatible = score_generated_dataset(
        generate_linear_dataset(replace(base, seed=24)), (0, 1)
    )
    summary = aggregate_generated_datasets((a, compatible))
    assert summary.independent_dataset_count == 2
    assert summary.condition.rho == 0.5
    alternatives = (
        _design("block_correlated_gaussian", rho=0.9, block_size=3),
        _design("block_correlated_gaussian", rho=0.5, block_size=2),
        _design("equicorrelated_gaussian", rho=0.5),
        _design("independent_standard_gaussian"),
    )
    for alternative in alternatives:
        b = score_generated_dataset(
            generate_linear_dataset(replace(alternative, seed=24)), (0,)
        )
        assert a.condition != b.condition
        with pytest.raises(ValueError, match="different generating conditions"):
            aggregate_generated_datasets((a, b))
    with pytest.raises(ValueError, match="block_size"):
        replace(LinearGaussianCondition.from_design(base), block_size=0)


def test_correlated_provenance_binds_covariance_parameters_even_with_same_data() -> (
    None
):
    # A row-independent zero-correlation draw can be copied into a different
    # model, but its old fingerprint must not validate under the new condition.
    original = generate_linear_dataset(_design("equicorrelated_gaussian", rho=0.0))
    assert original.dataset_id.startswith("trex0b1:")
    with pytest.raises(ValueError, match="dataset_id"):
        score_generated_dataset(
            replace(
                original,
                design=_design("equicorrelated_gaussian", rho=0.5),
            ),
            (),
        )
    # Repr intentionally excludes rho but condition equality includes it.
    first = _design("equicorrelated_gaussian", rho=0.5)
    second = _design("equicorrelated_gaussian", rho=0.9)
    assert repr(first) == repr(second)
    assert first != second


def test_legacy_phase_0a_bitwise_generation_and_fingerprint_regression() -> None:
    design = LinearGaussianDesign(
        n=9, p=4, beta=(0.0, 1.5, 0.0, -2.0), seed=23, noise_scale=0.7
    )
    dataset = generate_linear_dataset(design)
    rng = random.Random(23)
    reference_x = tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(4)) for _ in range(9))
    reference_error = tuple(rng.gauss(0.0, 0.7) for _ in range(9))
    assert dataset.x == reference_x
    assert dataset.epsilon == reference_error
    assert dataset.dataset_id == _HISTORICAL_ID
    assert design.rho is None and design.block_size is None
    assert "rho" not in repr(design) and "block_size" not in repr(design)
    assert score_generated_dataset(dataset, (1,)).metrics.fdp == 0


def test_private_rng_state_is_unchanged_by_correlated_generation() -> None:
    random.seed(9817)
    before = random.getstate()
    generate_linear_dataset(_design("equicorrelated_gaussian", rho=0.5))
    generate_linear_dataset(_design("block_correlated_gaussian", rho=0.9, block_size=3))
    assert random.getstate() == before
