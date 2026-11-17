"""Negative controls and reproducibility for Issue #47 development-only runner."""

from __future__ import annotations

import dataclasses
import subprocess
from fractions import Fraction

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError

from llm_evaluation.trex_finite_k import (
    Method,
    PilotConfig,
    RunRecord,
    Scenario,
    WorkerRequest,
    WorkerResult,
    _invoke_worker,
    _jaccard,
    bootstrap_ci,
    derived_seed,
    eligibility,
    execute,
)
from llm_evaluation.trex_known_truth import (
    LinearGaussianDesign,
    aggregate_independent_datasets,
    generate_linear_dataset,
    score_generated_dataset,
    score_selection,
)


def scenario(**updates: object) -> Scenario:
    fields: dict[str, object] = {
        "name": "case",
        "n": 40,
        "p": 8,
        "covariance": "independent_standard_gaussian",
        "active": (0, 7),
        "coefficient": 3.0,
    }
    fields.update(updates)
    return Scenario.model_validate(fields)


def pilot(
    *, method: Method, replicates: int = 2, chosen: Scenario | None = None
) -> PilotConfig:
    return PilotConfig(
        schema_version="trex47-config-v1",
        pilot_id="unit-test",
        development_namespace="issue47-development-v1",
        scenarios=(chosen or scenario(),),
        methods=(method,),
        independent_replicates=replicates,
        target_q=0.2,
        seed_base=1903,
        analysis_seed=55,
        bootstrap_draws=100,
        per_method_timeout_seconds=1,
        overall_budget_seconds=5,
    )


def test_original_empty_and_all_null_truth() -> None:
    ds = generate_linear_dataset(
        LinearGaussianDesign(n=20, p=5, beta=(0.0,) * 5, seed=8)
    )
    empty = score_generated_dataset(ds, ())
    assert empty.metrics.fdp == 0
    assert empty.metrics.tpr is None
    selected = score_generated_dataset(ds, (0, 4))
    assert selected.metrics.fdp == 1
    assert selected.metrics.v == 2
    assert selected.metrics.tpr is None


def test_correlated_null_proxy_is_false_discovery() -> None:
    ds = generate_linear_dataset(
        LinearGaussianDesign(
            n=30,
            p=6,
            beta=(2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            seed=4,
            covariance_family="block_correlated_gaussian",
            rho=0.9,
            block_size=3,
        )
    )
    scored = score_generated_dataset(ds, (0, 1))
    assert scored.metrics.true_support == frozenset({0})
    assert scored.metrics.v == 1
    assert scored.metrics.fdp == Fraction(1, 2)


def test_macro_fdr_is_not_pooled_v_over_r() -> None:
    few = score_selection(
        dataset_id="one", n_features=15, true_support=(), selected_features=(0,)
    )
    many = score_selection(
        dataset_id="two",
        n_features=15,
        true_support=tuple(range(1, 10)),
        selected_features=tuple(range(1, 10)),
    )
    actual = aggregate_independent_datasets((few, many))
    assert actual.empirical_fdr == Fraction(1, 2)
    assert Fraction(few.v + many.v, few.r + many.r) == Fraction(1, 10)
    assert actual.empirical_fdr != Fraction(few.v + many.v, few.r + many.r)


@pytest.mark.parametrize("bad", [(0, 0), (-1,), (8,), (True,), (0.5,)])
def test_bad_indices_rejected_by_existing_oracle(
    bad: tuple[int | float | bool, ...],
) -> None:
    ds = generate_linear_dataset(scenario().design(22))
    with pytest.raises((TypeError, ValueError)):
        score_generated_dataset(ds, bad)


def test_first_and_last_indices_preserved() -> None:
    ds = generate_linear_dataset(scenario().design(45))
    metrics = score_generated_dataset(ds, (0, 7)).metrics
    assert metrics.selected_features == frozenset({0, 7})
    assert metrics.tp == 2


def test_corrupt_truth_and_response_both_rejected() -> None:
    ds = generate_linear_dataset(scenario().design(46))
    with pytest.raises(ValueError, match="support"):
        score_generated_dataset(
            dataclasses.replace(ds, true_support=frozenset({1})), [0]
        )
    with pytest.raises(ValueError, match="fingerprint|realized|response"):
        score_generated_dataset(dataclasses.replace(ds, y=tuple(reversed(ds.y))), [0])


def test_duplicate_records_not_replicates() -> None:
    ds = generate_linear_dataset(scenario().design(46))
    scored = score_generated_dataset(ds, [0])
    from llm_evaluation.trex_known_truth import aggregate_generated_datasets

    with pytest.raises(ValueError, match="duplicate"):
        aggregate_generated_datasets([scored, scored])


def test_reject_duplicate_dataset_seeds_before_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = pilot(
        method=Method(name="bh", kind="bh_independent"),
        chosen=scenario(covariance="block_correlated_gaussian", rho=0.5, block_size=4),
    )
    monkeypatch.setattr(
        "llm_evaluation.trex_finite_k.derived_seed", lambda base, *args: 11
    )
    with pytest.raises(ValueError, match="seed collision"):
        execute(config)


def test_failed_method_preserves_non_success_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = pilot(method=Method(name="trex_k10", kind="trex", k=10), replicates=1)
    monkeypatch.setattr(
        "llm_evaluation.trex_finite_k._invoke_worker",
        lambda req, timeout: ("failed", None, "deliberate child failure", 0.1),
    )
    records, summaries = execute(config)
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].fdp is None and records[0].selected is None
    assert summaries[0].fdr is None
    assert summaries[0].failures == {"failed": 1}


def test_worker_timeout_kept_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="worker", timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)
    status, result, error, elapsed = _invoke_worker(
        WorkerRequest(
            kind="trex",
            k=10,
            da_method=None,
            seed=3,
            q=0.2,
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[1.0, 2.0],
            stability_subsamples=10,
        ),
        1,
    )
    assert (
        status == "timeout"
        and result is None
        and "deadline" in str(error)
        and elapsed >= 0
    )


def test_upstream_ignored_k_explicit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = subprocess.CompletedProcess(
        args=["worker"],
        returncode=0,
        stdout=WorkerResult(selected=(0,), effective_k=20).model_dump_json(),
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: fake)
    status, result, error, _ = _invoke_worker(
        WorkerRequest(
            kind="trex",
            k=10,
            da_method=None,
            seed=4,
            q=0.2,
            x=[[1.0, 2.0]],
            y=[1.0],
            stability_subsamples=10,
        ),
        1,
    )
    assert status == "failed" and result is None and "effective K=20" in str(error)


def test_non_success_with_fabricated_zero_fdp_rejected() -> None:
    data = dict(
        schema_version="trex47-record-v2",
        pilot_id="t",
        scenario="s",
        dataset_id="d",
        dataset_seed=1,
        method="m",
        method_kind="trex",
        requested_k=10,
        effective_k=None,
        da_method=None,
        selector_seed=1,
        rerun=0,
        status="failed",
        true_support=(),
        selected=(),
        r=0,
        tp=0,
        v=0,
        fdp=0.0,
        empty=True,
    )
    with pytest.raises(ValidationError):
        RunRecord.model_validate(data)


def test_selector_null_mismatch_is_explicit_unsupported() -> None:
    correlated = scenario(covariance="equicorrelated_gaussian", rho=0.5)
    assert "does not match" in str(
        eligibility(correlated, Method(name="bh", kind="bh_independent"))
    )
    assert "block-correlated" in str(
        eligibility(correlated, Method(name="da", kind="trex_da", k=20, da_method="BT"))
    )
    assert (
        eligibility(
            correlated, Method(name="da", kind="trex_da", k=20, da_method="EQUI")
        )
        is None
    )


@pytest.mark.parametrize(
    "wrong",
    [
        {"coefficient": 0},
        {"proxy_index": 0},
        {"rho": 0.6},
        {"active": [0, 0]},
        {"block_size": 4},
    ],
)
def test_scenario_validation_rejects_invalid_controls(wrong: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        scenario(**wrong)


def test_no_unknown_config_fields_and_k_method_coherence() -> None:
    with pytest.raises(ValidationError):
        Method.model_validate(
            {"name": "x", "kind": "trex", "k": 10, "surprise": "ignored"}
        )
    with pytest.raises(ValidationError):
        Method(name="x", kind="trex", k=None)
    with pytest.raises(ValidationError):
        Method(name="x", kind="lasso_cv", k=10)


def test_independent_selector_seeds_change_across_reruns() -> None:
    seeds = [derived_seed(5, "case", "0", "trex", str(i)) for i in range(20)]
    assert len(set(seeds)) == len(seeds)
    assert all(1 <= s < 2**31 for s in seeds)


def test_bootstrap_is_fixed_and_respects_data_range() -> None:
    values = [0.0, 1.0, 0.5]
    actual = bootstrap_ci(values, seed=99, draws=200)
    assert actual == bootstrap_ci(values, seed=99, draws=200)
    assert 0 <= actual[0] <= actual[1] <= 1


@given(
    st.lists(
        st.floats(min_value=0, max_value=1, allow_nan=False), min_size=1, max_size=12
    )
)
@settings(max_examples=30)
def test_bootstrap_respects_bounds(values: list[float]) -> None:
    lo, hi = bootstrap_ci(values, seed=13, draws=100)
    assert min(values) - 1e-12 <= lo <= hi <= max(values) + 1e-12


@given(
    st.sets(st.integers(min_value=0, max_value=9)),
    st.sets(st.integers(min_value=0, max_value=9)),
)
@settings(max_examples=40)
def test_jaccard_symmetric_and_bounded(a: set[int], b: set[int]) -> None:
    ab = _jaccard(tuple(a), tuple(b))
    assert 0 <= ab <= 1
    assert ab == _jaccard(tuple(b), tuple(a))


def test_native_integer_boundary_and_empty_pybind_shape() -> None:
    np = pytest.importorskip("numpy")
    from llm_evaluation.trex_finite_k_worker import _indices

    assert _indices(np.array([0, 7], dtype=np.int64), 8) == (0, 7)
    assert _indices(np.array([], dtype=float), 8) == ()
    for bad in (np.array([1.0]), np.array([-1]), np.array([8]), np.array([0, 0])):
        with pytest.raises(ValueError):
            _indices(bad, 8)


def test_worker_protects_source_request_from_native_normalization() -> None:
    pytest.importorskip("trexselector")
    from llm_evaluation.trex_finite_k_worker import run

    ds = generate_linear_dataset(scenario().design(23))
    x = [list(row) for row in ds.x]
    y = list(ds.y)
    original_x, original_y = [row.copy() for row in x], y.copy()
    req = WorkerRequest(
        kind="trex",
        k=10,
        da_method=None,
        seed=33,
        q=0.2,
        x=x,
        y=y,
        stability_subsamples=8,
    )
    first = run(req)
    second = run(req)
    assert req.x == original_x and req.y == original_y
    assert first.effective_k == second.effective_k == 10
    assert first.selected == second.selected


def test_success_record_integrity_and_calibration() -> None:
    """A persisted result independently rejects several plausible wrong metrics."""
    base = dict(
        schema_version="trex47-record-v2",
        pilot_id="v2",
        scenario="known",
        dataset_id="sha",
        dataset_seed=2,
        method="trex10",
        method_kind="trex",
        requested_k=10,
        effective_k=10,
        da_method=None,
        selector_seed=90,
        rerun=0,
        status="success",
        selected=(0, 7),
        true_support=(0, 7),
        r=2,
        tp=2,
        v=0,
        fdp=0.0,
        tpr=1.0,
        empty=False,
        selection_threshold=0.75,
        solver="TLARS",
        dummy_law="STANDARD iid N(0,1)",
        scaling="upstream L2",
    )
    good = RunRecord.model_validate(base)
    assert good.r == 2 and good.effective_k == 10
    for change in (
        {"r": 3},
        {"fdp": 0.5},
        {"tp": 1, "v": 1, "fdp": 0.5, "tpr": 0.5},
        {"tpr": None},
        {"selected": (0, 0)},
        {"effective_k": 20},
        {"selection_threshold": None},
        {"empty": True},
        {"true_support": ()},
    ):
        with pytest.raises(ValidationError):
            RunRecord.model_validate({**base, **change})


def test_persisted_all_null_success_requires_nonapplicable_tpr() -> None:
    base = dict(
        schema_version="trex47-record-v2",
        pilot_id="v2",
        scenario="null",
        dataset_id="sha",
        dataset_seed=3,
        method="bh",
        method_kind="bh_independent",
        requested_k=None,
        effective_k=None,
        da_method=None,
        selector_seed=91,
        rerun=0,
        status="success",
        selected=(3,),
        true_support=(),
        r=1,
        tp=0,
        v=1,
        fdp=1.0,
        tpr=None,
        empty=False,
        solver="Pearson marginal BH",
        selection_threshold=0.2,
        scaling="centered correlation",
    )
    rec = RunRecord.model_validate(base)
    assert rec.fdp == 1.0 and rec.tpr is None
    with pytest.raises(ValidationError):
        RunRecord.model_validate({**base, "tpr": 0.0})
    with pytest.raises(ValidationError):
        RunRecord.model_validate({**base, "fdp": 0.0})
