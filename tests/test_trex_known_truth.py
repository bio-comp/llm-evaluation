"""Phase 0A deterministic known-truth and independent metric reference tests."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import product
from typing import cast

import pytest

from llm_evaluation.trex_known_truth import (
    CovarianceFamily,
    GeneratedSelectionMetrics,
    KnownTruthDataset,
    LinearGaussianCondition,
    LinearGaussianDesign,
    NoiseLaw,
    ResponseFamily,
    SelectionMetrics,
    aggregate_generated_datasets,
    aggregate_independent_datasets,
    generate_linear_dataset,
    score_generated_dataset,
    score_selection,
)


def _design(
    *,
    n: int = 9,
    p: int = 4,
    beta: tuple[float, ...] = (0.0, 1.5, 0.0, -2.0),
    seed: int = 23,
    noise_scale: float = 0.7,
) -> LinearGaussianDesign:
    return LinearGaussianDesign(n=n, p=p, beta=beta, seed=seed, noise_scale=noise_scale)


def _score(
    dataset_id: str,
    true_support: tuple[int, ...],
    selected_features: tuple[int, ...],
    *,
    n_features: int = 5,
) -> SelectionMetrics:
    return score_selection(
        dataset_id=dataset_id,
        n_features=n_features,
        true_support=true_support,
        selected_features=selected_features,
    )


def test_fixed_seed_reproducibility_and_distinct_seed() -> None:
    design = _design()
    first = generate_linear_dataset(design)
    again = generate_linear_dataset(_design())
    changed = generate_linear_dataset(replace(design, seed=24))
    assert first == again
    assert first.dataset_id == again.dataset_id
    assert changed.dataset_id != first.dataset_id
    assert changed.x != first.x
    assert first.python_version
    assert first.rng_engine == "python.random.Random.gauss"


def test_dimensions_true_support_group_ids_and_metadata() -> None:
    generated = generate_linear_dataset(_design())
    assert len(generated.x) == len(generated.y) == 9
    assert all(len(row) == 4 for row in generated.x)
    assert generated.design.beta == (0.0, 1.5, 0.0, -2.0)
    assert generated.true_support == frozenset({1, 3})
    assert generated.conditional_null_definition == (
        "beta_j == 0 in the declared linear generating model"
    )
    assert len(set(generated.group_ids)) == 9
    assert len(set(generated.sampling_ids)) == 9
    assert all(generated.dataset_id in label for label in generated.group_ids)
    assert all(generated.dataset_id in label for label in generated.sampling_ids)
    assert generated.design.covariance_family == "independent_standard_gaussian"
    assert generated.design.response_family == "continuous_linear"
    assert generated.design.noise_law == "gaussian"


@pytest.mark.parametrize(
    ("p", "beta"),
    [(1, (0.0,)), (2, (1.0, 0.0)), (6, (0.0, 3.0, -0.0, 0.0, 2.0, 0.0))],
)
def test_support_uses_declared_coefficients_not_sample_association(
    p: int, beta: tuple[float, ...]
) -> None:
    generated = generate_linear_dataset(_design(n=3, p=p, beta=beta))
    assert generated.true_support == frozenset(
        index for index, coefficient in enumerate(beta) if coefficient != 0
    )


def test_exact_zero_noise_equation_and_all_null() -> None:
    design = _design(noise_scale=0)
    generated = generate_linear_dataset(design)
    for row, actual, noise in zip(
        generated.x, generated.y, generated.epsilon, strict=True
    ):
        expected = math.fsum(
            x * coefficient for x, coefficient in zip(row, design.beta, strict=True)
        )
        assert noise == 0.0
        assert actual == expected
    all_null = generate_linear_dataset(_design(beta=(0, 0, 0, 0), noise_scale=0))
    assert all_null.true_support == frozenset()
    assert all(value == 0 for value in all_null.y)


def test_nonzero_noise_realization_exposes_independent_equation_check() -> None:
    generated = generate_linear_dataset(_design())
    assert len(generated.epsilon) == generated.design.n
    assert any(noise != 0.0 for noise in generated.epsilon)
    for row, actual, noise in zip(
        generated.x, generated.y, generated.epsilon, strict=True
    ):
        assert actual == math.fsum(
            (
                *(
                    feature * beta
                    for feature, beta in zip(row, generated.design.beta, strict=True)
                ),
                noise,
            )
        )


def test_immutable_realization_and_no_global_random_state_leak() -> None:
    state = random.getstate()
    generated = generate_linear_dataset(_design())
    assert random.getstate() == state
    with pytest.raises((FrozenInstanceError, AttributeError)):
        generated.y = (10.0,)  # type: ignore[misc]
    assert isinstance(generated.x, tuple)
    assert isinstance(generated.x[0], tuple)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"n": 0}, ValueError),
        ({"p": 0}, ValueError),
        ({"n": -1}, ValueError),
        ({"n": True}, TypeError),
        ({"p": 1.5}, TypeError),
        ({"p": False}, TypeError),
        ({"seed": -2}, ValueError),
        ({"seed": False}, TypeError),
        ({"beta": (1.0, 2.0)}, ValueError),
        ({"beta": [0.0, 1.5, 0.0, -2.0]}, ValueError),
        ({"beta": (0.0, True, 0.0, 1.0)}, TypeError),
        ({"beta": (0.0, float("nan"), 0.0, 1.0)}, ValueError),
        ({"beta": (0.0, float("inf"), 0.0, 1.0)}, ValueError),
        ({"noise_scale": -0.1}, ValueError),
        ({"noise_scale": float("nan")}, ValueError),
        ({"noise_scale": float("inf")}, ValueError),
        ({"noise_scale": True}, TypeError),
        ({"covariance_family": "block"}, ValueError),
        ({"noise_law": "student_t"}, ValueError),
        ({"response_family": "binary"}, ValueError),
    ],
)
def test_design_rejects_invalid_inputs(
    changes: dict[str, object], error: type[Exception]
) -> None:
    params: dict[str, object] = {
        "n": 9,
        "p": 4,
        "beta": (0.0, 1.5, 0.0, -2.0),
        "seed": 23,
        "noise_scale": 0.7,
        "covariance_family": "independent_standard_gaussian",
        "noise_law": "gaussian",
        "response_family": "continuous_linear",
    }
    params.update(changes)
    with pytest.raises(error):
        LinearGaussianDesign(
            n=cast(int, params["n"]),
            p=cast(int, params["p"]),
            beta=cast(tuple[float, ...], params["beta"]),
            seed=cast(int, params["seed"]),
            noise_scale=cast(float, params["noise_scale"]),
            covariance_family=cast(CovarianceFamily, params["covariance_family"]),
            noise_law=cast(NoiseLaw, params["noise_law"]),
            response_family=cast(ResponseFamily, params["response_family"]),
        )


def test_bad_generator_design_rejected() -> None:
    with pytest.raises(TypeError, match="LinearGaussianDesign"):
        generate_linear_dataset(cast(LinearGaussianDesign, "bad"))


def test_nonfinite_generated_response_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite response"):
        generate_linear_dataset(
            _design(beta=(1e308, 1e308, 1e308, 1e308), noise_scale=0)
        )


def test_toy_scoring_exact_false_discovery_and_true_positive_rates() -> None:
    result = _score("toy", (0, 2), (0, 1, 4))
    # True positives {0}; false discoveries {1,4}.
    assert (result.r, result.tp, result.v) == (3, 1, 2)
    assert result.fdp == Fraction(2, 3)
    assert result.tpr == Fraction(1, 2)
    assert result.true_support == frozenset({0, 2})
    assert result.selected_features == frozenset({0, 1, 4})
    # Explicitly detect wrong denominators p=5, |true|=2, and swapped roles.
    assert result.fdp != Fraction(2, 5)
    assert result.fdp != Fraction(2, 2)
    assert result.tpr != Fraction(1, 3)


def test_empty_all_selected_and_all_null_policies() -> None:
    empty = _score("empty", (0, 1), ())
    assert (empty.r, empty.tp, empty.v) == (0, 0, 0)
    assert empty.fdp == 0
    assert empty.tpr == Fraction(0, 1)
    all_selected = _score("all", (0, 3), (0, 1, 2, 3, 4))
    assert (all_selected.r, all_selected.tp, all_selected.v) == (5, 2, 3)
    assert all_selected.fdp == Fraction(3, 5)
    assert all_selected.tpr == 1
    all_null = _score("null", (), (1, 3))
    assert (all_null.r, all_null.tp, all_null.v) == (2, 0, 2)
    assert all_null.fdp == 1
    assert all_null.tpr is None
    null_empty = _score("null-empty", (), ())
    assert null_empty.fdp == 0
    assert null_empty.tpr is None


@pytest.mark.parametrize(
    ("truth", "selected", "error"),
    [
        ((0, 0), (), ValueError),
        ((), (1, 1), ValueError),
        ((-1,), (), ValueError),
        ((), (-1,), ValueError),
        ((5,), (), ValueError),
        ((), (5,), ValueError),
        ((True,), (), TypeError),
        ((), (False,), TypeError),
        ((1.0,), (), TypeError),
        ((), (1.0,), TypeError),
        ((), "12", TypeError),
    ],
)
def test_invalid_duplicate_and_out_of_range_feature_indices(
    truth: object, selected: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        score_selection(
            dataset_id="bad",
            n_features=5,
            true_support=cast(tuple[int, ...], truth),
            selected_features=cast(tuple[int, ...], selected),
        )


@pytest.mark.parametrize("p", [0, -1, True, 1.2])
def test_invalid_feature_universe(p: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        score_selection(
            dataset_id="bad",
            n_features=cast(int, p),
            true_support=(),
            selected_features=(),
        )


@pytest.mark.parametrize("dataset_id", ["", " ", " bad", "bad ", None, 3])
def test_invalid_dataset_identifier(dataset_id: object) -> None:
    with pytest.raises(ValueError):
        score_selection(
            dataset_id=cast(str, dataset_id),
            n_features=2,
            true_support=(),
            selected_features=(),
        )


def test_set_input_and_null_proxy_marginal_association_counterexample() -> None:
    # Structural y = x0, beta1 = 0. A proxy x1 strongly follows x0, but is null
    # under the declared multivariate coefficient null. No selector is fitted.
    x0 = (-2.0, -1.0, 1.0, 2.0)
    x1 = (-2.1, -0.8, 0.9, 2.2)
    assert sum(a * b for a, b in zip(x0, x1, strict=True)) > 10
    assert sum(a * a for a in x0) == 10
    result = score_selection(
        dataset_id="correlated-null-proxy",
        n_features=2,
        true_support=frozenset({0}),
        selected_features={1},
    )
    assert (result.r, result.tp, result.v) == (1, 0, 1)
    assert result.fdp == 1
    assert result.tpr == 0


def test_macro_fdr_differs_from_pooled_counts() -> None:
    a = _score("dataset-a", (0,), (0, 1, 2, 3))
    b = _score("dataset-b", (0,), (0,))
    summary = aggregate_independent_datasets((a, b))
    # Mean FDP = ((3/4)+0)/2 = 3/8. Pooled V/R = (3+0)/(4+1)=3/5.
    assert (a.fdp, b.fdp) == (Fraction(3, 4), Fraction(0))
    assert summary.independent_dataset_count == 2
    assert summary.empirical_fdr == Fraction(3, 8)
    assert summary.empirical_fdr != Fraction(3, 5)
    assert summary.mean_applicable_tpr == 1
    assert summary.tpr_applicable_count == 2
    assert summary.tpr_nonapplicable_count == 0
    assert summary.empty_selection_rate == 0
    assert summary.per_dataset == (a, b)


def test_all_null_tpr_count_and_empty_selection_rate() -> None:
    records = (
        _score("a", (0, 1), (0,)),
        _score("b", (), (3,)),
        _score("c", (), ()),
        _score("d", (2,), ()),
    )
    summary = aggregate_independent_datasets(records)
    assert summary.independent_dataset_count == 4
    assert summary.empirical_fdr == Fraction(1, 4)
    assert summary.mean_applicable_tpr == Fraction(1, 4)
    assert summary.tpr_applicable_count == 2
    assert summary.tpr_nonapplicable_count == 2
    assert summary.empty_selection_rate == Fraction(1, 2)
    assert summary.per_dataset == records
    null_only = aggregate_independent_datasets(records[1:3])
    assert null_only.mean_applicable_tpr is None
    assert null_only.tpr_applicable_count == 0
    assert null_only.tpr_nonapplicable_count == 2


def test_repeat_selector_randomization_not_independent_dataset() -> None:
    generated = generate_linear_dataset(_design())
    first = score_selection(
        dataset_id=generated.dataset_id,
        n_features=4,
        true_support=generated.true_support,
        selected_features=(1,),
    )
    # Different selector seed/K may alter its selection, NOT its dataset identity.
    second = score_selection(
        dataset_id=generated.dataset_id,
        n_features=4,
        true_support=generated.true_support,
        selected_features=(1, 2, 3),
    )
    with pytest.raises(ValueError, match="duplicate dataset_id"):
        aggregate_independent_datasets((first, second))
    with pytest.raises(ValueError, match="duplicate dataset_id"):
        aggregate_independent_datasets((first, first))
    independently_generated = generate_linear_dataset(replace(_design(), seed=24))
    third = score_selection(
        dataset_id=independently_generated.dataset_id,
        n_features=4,
        true_support=independently_generated.true_support,
        selected_features=(),
    )
    assert aggregate_independent_datasets((first, third)).independent_dataset_count == 2


def test_empty_and_invalid_aggregates() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_independent_datasets(())
    with pytest.raises(TypeError, match="sequence"):
        aggregate_independent_datasets(cast(tuple[SelectionMetrics, ...], "invalid"))
    with pytest.raises(TypeError, match="SelectionMetrics"):
        aggregate_independent_datasets(cast(tuple[SelectionMetrics, ...], [None]))


def test_metric_record_invariant_falsification_controls() -> None:
    record = _score("record", (1, 2), (1, 3))
    for corrupt in (
        {"r": 3},
        {"tp": 0},
        {"v": 0},
        {"fdp": Fraction(0)},
        {"tpr": Fraction(1)},
        {"true_support": frozenset({0})},
        {"selected_features": frozenset({0})},
    ):
        with pytest.raises(ValueError, match="disagrees"):
            replace(record, **corrupt)


def test_coefficient_too_large_to_convert_rejected() -> None:
    with pytest.raises(ValueError, match="outside finite float range"):
        _design(beta=(0.0, 10**400, 0.0, 0.0))


def test_corrupted_random_noise_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Injecting a nonfinite RNG result distinguishes finite-output checks
    # from mere validation of the caller's declared noise scale.
    def broken_gauss(rng: random.Random, mu: float, sigma: float) -> float:
        return float("inf") if sigma == 0.7 else 0.0

    monkeypatch.setattr(random.Random, "gauss", broken_gauss)
    with pytest.raises(ValueError, match="non-finite response"):
        generate_linear_dataset(_design(beta=(0.0, 0.0, 0.0, 0.0)))


def test_metric_record_rejects_bad_identity_and_support_types() -> None:
    record = _score("good", (1,), (1, 2))
    with pytest.raises(ValueError, match="invalid dataset_id"):
        replace(record, dataset_id=" ")
    with pytest.raises(TypeError, match="frozen sets"):
        replace(record, true_support=cast(frozenset[int], (1,)))
    with pytest.raises(TypeError, match="frozen sets"):
        replace(record, selected_features=cast(frozenset[int], (1,)))


def test_generated_condition_accepts_distinct_seed_replicates_and_macro_fdr() -> None:
    design = _design(beta=(1.0, 0.0, 0.0, 0.0))
    a = score_generated_dataset(generate_linear_dataset(design), (0, 1, 2, 3))
    b = score_generated_dataset(generate_linear_dataset(replace(design, seed=24)), (0,))
    result = aggregate_generated_datasets((a, b))
    assert isinstance(a, GeneratedSelectionMetrics)
    assert result.condition == LinearGaussianCondition.from_design(design)
    assert result.condition == LinearGaussianCondition.from_design(
        replace(design, seed=24)
    )
    assert (a.replicate_seed, b.replicate_seed) == (23, 24)
    assert a.dataset_id != b.dataset_id
    assert a.dataset_id == a.metrics.dataset_id
    assert (a.metrics.fdp, b.metrics.fdp) == (Fraction(3, 4), Fraction(0))
    assert (result.independent_dataset_count, result.empirical_fdr) == (
        2,
        Fraction(3, 8),
    )
    assert result.empirical_fdr != Fraction(3, 5)  # pooled V/R is wrong
    assert result.mean_applicable_tpr == Fraction(1)
    assert result.tpr_applicable_count == 2
    assert result.tpr_nonapplicable_count == 0
    assert result.empty_selection_rate == 0
    assert result.per_dataset == (a, b)


def test_director_beta_common_random_stream_reproduction_and_rejection() -> None:
    base = LinearGaussianDesign(n=20, p=3, beta=(1.0, 0.0, 0.0), seed=23)
    a = generate_linear_dataset(base)
    b = generate_linear_dataset(replace(base, beta=(0.0, 1.0, 0.0)))
    assert a.x == b.x
    assert a.epsilon == b.epsilon
    assert a.dataset_id != b.dataset_id
    assert score_generated_dataset(a, (0,)).metrics.fdp == 0
    assert score_generated_dataset(b, (0,)).metrics.fdp == 1
    # The lower-level arithmetic function is purposely *not* condition-aware:
    assert aggregate_independent_datasets(
        (
            score_generated_dataset(a, (0,)).metrics,
            score_generated_dataset(b, (0,)).metrics,
        )
    ).empirical_fdr == Fraction(1, 2)
    with pytest.raises(ValueError, match="different generating conditions"):
        aggregate_generated_datasets(
            (
                score_generated_dataset(a, (0,)),
                score_generated_dataset(b, (0,)),
            )
        )


@pytest.mark.parametrize(
    "b_design",
    [
        _design(seed=24, beta=(0.0, 1.5, 0.0, 0.0)),
        _design(seed=24, beta=(0.0, 1.6, 0.0, -2.0)),
        _design(seed=24, noise_scale=0.8),
        _design(seed=24, noise_scale=0.0),
        _design(seed=24, n=10),
        _design(seed=24, p=3, beta=(0.0, 1.5, 0.0)),
        _design(seed=24, p=5, beta=(0.0, 1.5, 0.0, -2.0, 0.0)),
    ],
)
def test_generated_aggregation_rejects_condition_mixing(
    b_design: LinearGaussianDesign,
) -> None:
    design = _design()
    a = score_generated_dataset(generate_linear_dataset(design), (1,))
    b = score_generated_dataset(generate_linear_dataset(b_design), (1,))
    assert a.dataset_id != b.dataset_id
    assert a.condition != b.condition
    with pytest.raises(ValueError, match="different generating conditions"):
        aggregate_generated_datasets((a, b))


def test_repeat_generated_dataset_and_selector_randomizations_are_not_replicates() -> (
    None
):
    dataset = generate_linear_dataset(_design())
    first = score_generated_dataset(dataset, (1,))
    repeated_selector = score_generated_dataset(dataset, (1, 2, 3))
    assert first.dataset_id == repeated_selector.dataset_id
    assert first.replicate_seed == repeated_selector.replicate_seed
    with pytest.raises(ValueError, match="duplicate replicate seed"):
        aggregate_generated_datasets((first, repeated_selector))
    with pytest.raises(ValueError, match="duplicate replicate seed"):
        aggregate_generated_datasets((first, first))
    with pytest.raises(ValueError, match="duplicate replicate seed"):
        aggregate_generated_datasets(
            (
                first,
                replace(first, metrics=replace(first.metrics, dataset_id="forged-id")),
            )
        )
    with pytest.raises(ValueError, match="duplicate dataset_id"):
        aggregate_generated_datasets((first, replace(first, replicate_seed=24)))


def test_all_null_generated_replicates_have_nonapplicable_tpr() -> None:
    design = _design(beta=(0.0, 0.0, 0.0, 0.0), noise_scale=0.0)
    a = score_generated_dataset(generate_linear_dataset(design), ())
    b = score_generated_dataset(generate_linear_dataset(replace(design, seed=24)), (1,))
    result = aggregate_generated_datasets((a, b))
    assert result.empirical_fdr == Fraction(1, 2)
    assert result.mean_applicable_tpr is None
    assert result.tpr_applicable_count == 0
    assert result.tpr_nonapplicable_count == 2
    assert result.empty_selection_rate == Fraction(1, 2)


def test_generated_scoring_rejects_forged_truth_but_pure_oracle_can_score_it() -> None:
    base = LinearGaussianDesign(n=20, p=3, beta=(1.0, 0.0, 0.0), seed=23)
    dataset = generate_linear_dataset(base)
    forged = replace(dataset, true_support=frozenset())
    assert score_generated_dataset(dataset, (0,)).metrics.fdp == 0
    with pytest.raises(ValueError, match="true_support"):
        score_generated_dataset(forged, (0,))
    # Deliberate corruption remains an independent *arithmetic* test.
    deliberate = score_selection(
        dataset_id=forged.dataset_id,
        n_features=forged.design.p,
        true_support=forged.true_support,
        selected_features=(0,),
    )
    assert deliberate.fdp == 1
    assert deliberate.tpr is None


@pytest.mark.parametrize(
    ("forge", "message"),
    [
        (lambda d: replace(d, dataset_id="forged"), "dataset_id"),
        (lambda d: replace(d, dataset_id=""), "dataset_id"),
        (lambda d: replace(d, true_support=frozenset()), "true_support"),
        (
            lambda d: replace(
                d, conditional_null_definition="marginal correlation null"
            ),
            "conditional null",
        ),
        (lambda d: replace(d, rng_engine="forged"), "RNG/runtime"),
        (lambda d: replace(d, python_version=""), "RNG/runtime"),
        (lambda d: replace(d, x=((1.0, 2.0),)), "dimensions"),
        (
            lambda d: replace(d, x=((1.0, 2.0, 3.0, 4.0),) * 9),
            "response disagrees",
        ),
        (lambda d: replace(d, y=(0.0,) * 9), "response disagrees"),
        (lambda d: replace(d, epsilon=(0.0,) * 9), "response disagrees"),
        (lambda d: replace(d, group_ids=("bad",) * 9), "observation/group IDs"),
        (lambda d: replace(d, sampling_ids=("bad",) * 9), "observation/group IDs"),
        (lambda d: replace(d, design=replace(_design(), seed=24)), "dataset_id"),
    ],
)
def test_generated_path_rejects_corrupted_realization_or_metadata(
    forge: Callable[[KnownTruthDataset], KnownTruthDataset], message: str
) -> None:
    dataset = generate_linear_dataset(_design())
    with pytest.raises(ValueError, match=message):
        score_generated_dataset(forge(dataset), (1,))


def test_generated_path_refuses_explicit_truth_argument_and_invalid_features() -> None:
    generated = generate_linear_dataset(_design())
    with pytest.raises(TypeError):
        score_generated_dataset(generated, (1,), true_support=frozenset())  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="duplicate"):
        score_generated_dataset(generated, (1, 1))
    with pytest.raises(ValueError, match="outside"):
        score_generated_dataset(generated, (4,))


def test_generated_record_self_validation_rejects_mislabeled_support() -> None:
    record = score_generated_dataset(generate_linear_dataset(_design()), (1, 3))
    assert record.condition.n == 9 and record.condition.p == 4
    assert record.condition.beta == (0.0, 1.5, 0.0, -2.0)
    assert record.replicate_seed == 23
    wrong = score_selection(
        dataset_id=record.dataset_id,
        n_features=4,
        true_support=(1,),
        selected_features=(1,),
    )
    with pytest.raises(ValueError, match="support/universe"):
        replace(record, metrics=wrong)
    with pytest.raises(ValueError, match="support/universe"):
        replace(
            record,
            condition=LinearGaussianCondition.from_design(
                _design(p=3, beta=(0.0, 1.5, 0.0))
            ),
        )


@pytest.mark.parametrize("invalid", [(), "bad", (None,)])
def test_generated_aggregation_rejects_invalid_input(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        aggregate_generated_datasets(
            cast(tuple[GeneratedSelectionMetrics, ...], invalid)
        )


def test_5460_independent_exact_truth_selection_combinations() -> None:
    # sum_{p=1}^6 2^p * 2^p = 5_460. Reference enumerates bit indices,
    # never calls the production scorer for expected counts or fractions.
    checked = 0
    for p in range(1, 7):
        for truth_mask, selected_mask in product(range(1 << p), repeat=2):
            truth = tuple(j for j in range(p) if truth_mask & (1 << j))
            selected = tuple(j for j in range(p) if selected_mask & (1 << j))
            expected_tp = sum(
                bool(truth_mask & selected_mask & (1 << j)) for j in range(p)
            )
            expected_r = selected_mask.bit_count()
            expected_v = expected_r - expected_tp
            expected_fdp = Fraction(expected_v, max(expected_r, 1))
            expected_tpr = (
                Fraction(expected_tp, truth_mask.bit_count()) if truth_mask else None
            )
            record = score_selection(
                dataset_id="enumerated-toy",
                n_features=p,
                true_support=truth,
                selected_features=selected,
            )
            assert (record.r, record.tp, record.v) == (
                expected_r,
                expected_tp,
                expected_v,
            )
            assert (record.fdp, record.tpr) == (expected_fdp, expected_tpr)
            checked += 1
    assert checked == 5_460


def test_reject_even_equal_set_value_when_generated_support_type_is_corrupted() -> None:
    # bool indices compare equal to integers; equality alone is insufficient.
    dataset = generate_linear_dataset(_design())
    assert dataset.true_support == frozenset({1, 3})
    with pytest.raises(ValueError, match="true_support"):
        score_generated_dataset(
            replace(dataset, true_support=frozenset({True, 3})), (1,)
        )
    with pytest.raises(ValueError, match="true_support"):
        score_generated_dataset(
            replace(dataset, true_support=cast(frozenset[int], {1, 3})), (1,)
        )


def test_generated_path_rejects_invalid_object_and_design_type() -> None:
    with pytest.raises(TypeError, match="KnownTruthDataset"):
        score_generated_dataset(cast(KnownTruthDataset, "invalid"), ())
    dataset = generate_linear_dataset(_design())
    with pytest.raises(TypeError, match="LinearGaussianDesign"):
        score_generated_dataset(
            replace(dataset, design=cast(LinearGaussianDesign, "invalid")), ()
        )


@pytest.mark.parametrize(
    ("forge", "message"),
    [
        (
            lambda d: replace(d, x=((1.0, 2.0), *d.x[1:])),
            "predictor dimensions",
        ),
        (
            lambda d: replace(d, x=((float("nan"), *d.x[0][1:]), *d.x[1:])),
            "finite floats",
        ),
        (
            lambda d: replace(d, y=cast(tuple[float, ...], ("bad", *d.y[1:]))),
            "finite floats",
        ),
        (
            lambda d: replace(
                d, epsilon=cast(tuple[float, ...], (True, *d.epsilon[1:]))
            ),
            "finite floats",
        ),
    ],
)
def test_generated_dataset_rejects_malformed_numeric_boundary(
    forge: Callable[[KnownTruthDataset], KnownTruthDataset], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        score_generated_dataset(forge(generate_linear_dataset(_design())), (1,))


def test_generated_response_verification_rejects_overflow() -> None:
    dataset = generate_linear_dataset(
        _design(beta=(1.0, 1.0, 1.0, 1.0), noise_scale=0.0)
    )
    corrupted = replace(
        dataset,
        x=((1e308, 1e308, 1e308, 1e308), *dataset.x[1:]),
    )
    with pytest.raises(ValueError, match="response equation overflowed"):
        score_generated_dataset(corrupted, (1,))


def test_generated_record_rejects_bad_types_and_invalid_seed() -> None:
    record = score_generated_dataset(generate_linear_dataset(_design()), (1,))
    with pytest.raises(TypeError, match="LinearGaussianCondition"):
        replace(record, condition=cast(LinearGaussianCondition, "invalid"))
    with pytest.raises(TypeError, match="SelectionMetrics"):
        replace(record, metrics=cast(SelectionMetrics, "invalid"))
    with pytest.raises(TypeError, match="exact integer"):
        replace(record, replicate_seed=cast(int, True))
    with pytest.raises(ValueError, match="nonnegative"):
        replace(record, replicate_seed=-1)


def test_generated_condition_validation_reuses_existing_design_rules() -> None:
    condition = LinearGaussianCondition.from_design(_design())
    with pytest.raises(ValueError, match="positive"):
        replace(condition, n=0)
    with pytest.raises(ValueError, match="unsupported noise law"):
        replace(condition, noise_law=cast(NoiseLaw, "student_t"))


@pytest.mark.parametrize("absorbed_error", (1.0, 1e-30))
def test_generated_zero_noise_rejects_absorbed_nonzero_errors(
    absorbed_error: float,
) -> None:
    design = LinearGaussianDesign(n=1, p=1, beta=(1e100,), noise_scale=0.0, seed=23)
    dataset = generate_linear_dataset(design)
    assert dataset.epsilon == (0.0,)
    assert score_generated_dataset(dataset, (0,)).metrics.fdp == 0
    corrupted = replace(dataset, epsilon=(absorbed_error,))
    assert corrupted.y == dataset.y
    assert corrupted.dataset_id == dataset.dataset_id
    # Even if floating-point rounding hides the error in y, zero-noise
    # provenance must be checked as an exact deterministic invariant.
    with pytest.raises(ValueError, match="zero-noise"):
        score_generated_dataset(corrupted, (0,))


def test_positive_noise_allows_realized_zero_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = random.Random.gauss

    def draw_with_zero_response_noise(
        rng: random.Random, mu: float, sigma: float
    ) -> float:
        if sigma == 0.7:
            return 0.0
        return original(rng, mu, sigma)

    monkeypatch.setattr(random.Random, "gauss", draw_with_zero_response_noise)
    dataset = generate_linear_dataset(_design())
    assert dataset.design.noise_scale == 0.7
    assert all(error == 0.0 for error in dataset.epsilon)
    assert score_generated_dataset(dataset, (1, 3)).metrics.fdp == 0
