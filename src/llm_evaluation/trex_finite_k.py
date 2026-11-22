"""Development-only finite-K T-Rex experiment orchestration (Issue #47).

Raw conditional-coefficient truth and arithmetic remain in trex_known_truth.
This module does not establish finite-K FDR control or FINAL acceptance.
"""

from __future__ import annotations

import hashlib
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_evaluation.trex_known_truth import (
    GeneratedSelectionMetrics,
    LinearGaussianDesign,
    aggregate_generated_datasets,
    generate_linear_dataset,
    score_generated_dataset,
)

Kind = Literal["trex", "trex_da", "lasso_cv", "stability", "bh_independent"]
Status = Literal[
    "success",
    "failed",
    "timeout",
    "unavailable",
    "unqualified",
    "unsupported",
    "budget_exhausted",
]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Scenario(StrictRecord):
    name: str = Field(min_length=1)
    n: int = Field(ge=10)
    p: int = Field(ge=4)
    covariance: Literal[
        "independent_standard_gaussian",
        "block_correlated_gaussian",
        "equicorrelated_gaussian",
    ]
    rho: float | None = Field(default=None, ge=0, lt=1)
    block_size: int | None = Field(default=None, ge=2)
    active: tuple[int, ...] = ()
    coefficient: float = Field(default=0.0, ge=0)
    noise_scale: float = Field(default=1.0, gt=0)
    proxy_index: int | None = None

    @model_validator(mode="after")
    def validate_science(self) -> Scenario:
        if (self.covariance == "independent_standard_gaussian") != (self.rho is None):
            raise ValueError(
                "independent covariance requires no rho; correlated requires rho"
            )
        if (self.covariance == "block_correlated_gaussian") != (
            self.block_size is not None
        ):
            raise ValueError("block_size required only for blocks")
        if self.block_size is not None and self.block_size > self.p:
            raise ValueError("block_size exceeds p")
        if len(set(self.active)) != len(self.active) or any(
            type(i) is not int or not 0 <= i < self.p for i in self.active
        ):
            raise ValueError("invalid or duplicate active indices")
        if bool(self.active) != (self.coefficient > 0):
            raise ValueError("coefficient must be positive iff support is nonempty")
        if self.proxy_index is not None:
            if not 0 <= self.proxy_index < self.p or self.proxy_index in self.active:
                raise ValueError("proxy must be structurally null")
            if not self.active or self.covariance == "independent_standard_gaussian":
                raise ValueError("proxy requires active correlated predictor")
            if self.covariance == "block_correlated_gaussian":
                block = self.block_size
                assert block is not None
                if not any(
                    i // block == self.proxy_index // block for i in self.active
                ):
                    raise ValueError("proxy must share block with an active")
        return self

    def design(self, dataset_seed: int) -> LinearGaussianDesign:
        beta = [0.0] * self.p
        for index in self.active:
            beta[index] = self.coefficient
        return LinearGaussianDesign(
            n=self.n,
            p=self.p,
            beta=tuple(beta),
            seed=dataset_seed,
            covariance_family=self.covariance,
            rho=self.rho,
            block_size=self.block_size,
            noise_scale=self.noise_scale,
        )


class Method(StrictRecord):
    name: str = Field(min_length=1)
    kind: Kind
    k: int | None = Field(default=None, ge=2, le=1000)
    da_method: Literal["BT", "EQUI"] | None = None
    # Pairwise Jaccard is defined for exactly two completed selector executions.
    reruns: int = Field(default=1, ge=1, le=2)
    stability_subsamples: int = Field(default=20, ge=4, le=100)

    @model_validator(mode="after")
    def validate_method(self) -> Method:
        if (self.kind in ("trex", "trex_da")) != (self.k is not None):
            raise ValueError("K must be supplied only for T-Rex methods")
        if (self.kind == "trex_da") != (self.da_method is not None):
            raise ValueError("DA method required only for T-Rex+DA")
        return self


class PilotConfig(StrictRecord):
    schema_version: Literal["trex47-config-v1"]
    pilot_id: str = Field(min_length=1)
    development_namespace: Literal["issue47-development-v1"]
    scenarios: tuple[Scenario, ...] = Field(min_length=1)
    methods: tuple[Method, ...] = Field(min_length=1)
    independent_replicates: int = Field(ge=1, le=256)
    target_q: float = Field(gt=0, lt=1)
    seed_base: int = Field(ge=1, le=1_000_000_000)
    analysis_seed: int = Field(ge=1)
    bootstrap_draws: int = Field(default=500, ge=100, le=20000)
    per_method_timeout_seconds: float = Field(default=20.0, gt=0)
    overall_budget_seconds: float = Field(default=480.0, gt=0)

    @model_validator(mode="after")
    def unique_identifiers(self) -> PilotConfig:
        if len({s.name for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("duplicate scenario name")
        if len({m.name for m in self.methods}) != len(self.methods):
            raise ValueError("duplicate method name")
        return self


class WorkerRequest(StrictRecord):
    kind: Kind
    k: int | None
    da_method: str | None
    seed: int
    q: float
    x: list[list[float]]
    y: list[float]
    stability_subsamples: int


class WorkerResult(StrictRecord):
    selected: tuple[int, ...]
    effective_k: int | None
    dummy_multiplier_l: int | None = None
    num_dummies: int | None = None
    t_stop: int | None = None
    peak_rss_mib: float | None = None
    solver: str | None = None
    dummy_law: str | None = None
    scaling: str | None = None
    selection_threshold: float | None = None


class RunRecord(StrictRecord):
    schema_version: Literal["trex47-record-v2"] = "trex47-record-v2"
    pilot_id: str
    scenario: str
    dataset_id: str
    dataset_seed: int
    method: str
    method_kind: Kind
    requested_k: int | None
    effective_k: int | None = None
    da_method: str | None
    selector_seed: int
    rerun: int
    status: Status
    selected: tuple[int, ...] | None = None
    true_support: tuple[int, ...]
    r: int | None = None
    tp: int | None = None
    v: int | None = None
    fdp: float | None = None
    tpr: float | None = None
    proxy_selected: bool | None = None
    empty: bool | None = None
    elapsed_seconds: float | None = None
    peak_rss_mib: float | None = None
    dummy_multiplier_l: int | None = None
    num_dummies: int | None = None
    t_stop: int | None = None
    selection_threshold: float | None = None
    solver: str | None = None
    dummy_law: str | None = None
    scaling: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def complete_success(self) -> RunRecord:
        metrics = (self.selected, self.r, self.tp, self.v, self.fdp, self.empty)
        if self.status == "success":
            if any(value is None for value in metrics):
                raise ValueError("successful run must contain scored metrics")
            selected, r, tp, v, fdp, empty = (
                self.selected,
                self.r,
                self.tp,
                self.v,
                self.fdp,
                self.empty,
            )
            assert selected is not None and r is not None and tp is not None
            assert v is not None and fdp is not None and empty is not None
            if r != len(selected) or r != tp + v or tp < 0 or v < 0:
                raise ValueError("inconsistent R, TP, V and selected feature count")
            if len(set(selected)) != r or any(i < 0 for i in selected):
                raise ValueError("duplicated or negative selected feature")
            if (r == 0) != empty or not math.isclose(
                fdp, v / max(1, r), rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("incorrect empty-selection or FDP arithmetic")
            true_support = set(self.true_support)
            if len(true_support) != len(self.true_support) or any(
                i < 0 for i in true_support
            ):
                raise ValueError("invalid structural reference support")
            if tp != len(true_support.intersection(selected)):
                raise ValueError("TP inconsistent with structural reference support")
            if not true_support:
                if self.tpr is not None:
                    raise ValueError("all-null TPR must be N/A")
            elif self.tpr is None or not math.isclose(
                self.tpr, tp / len(true_support), rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("TPR inconsistent with structural reference support")
            if self.method_kind in ("trex", "trex_da"):
                if self.requested_k is None or self.effective_k != self.requested_k:
                    raise ValueError("effective K must match requested K")
                if any(
                    x is None
                    for x in (
                        self.selection_threshold,
                        self.solver,
                        self.dummy_law,
                        self.scaling,
                    )
                ):
                    raise ValueError("T-Rex calibration and settings must be retained")
            elif self.effective_k is not None:
                raise ValueError("non-T-Rex method cannot claim effective K")
        elif any(value is not None for value in metrics):
            raise ValueError("non-success may not fabricate selection/metrics")
        return self


class Summary(StrictRecord):
    schema_version: Literal["trex47-summary-v1"] = "trex47-summary-v1"
    scenario: str
    method: str
    requested_k: int | None
    intended_datasets: int
    successful_datasets: int
    failures: dict[str, int]
    fdr: float | None = None
    fdr_ci95: tuple[float, float] | None = None
    mean_tpr: float | None = None
    tpr_ci95: tuple[float, float] | None = None
    p_fdp_gt_q: float | None = None
    p_fdp_ge_half: float | None = None
    p_empty: float | None = None
    p_proxy: float | None = None
    mean_runtime_seconds: float | None = None
    max_peak_rss_mib: float | None = None
    mean_rerun_jaccard: float | None = None
    rerun_pairs_completed: int | None = None
    rerun_pairs_incomplete: int | None = None


def derived_seed(base: int, *parts: str) -> int:
    payload = "|".join((str(base), *parts)).encode("utf-8")
    return 1 + int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 2)


def eligibility(scenario: Scenario, method: Method) -> str | None:
    if (
        method.kind == "bh_independent"
        and scenario.covariance != "independent_standard_gaussian"
    ):
        return "BH marginal null does not match structural conditional null under correlation"
    if method.kind == "trex_da":
        if scenario.covariance == "independent_standard_gaussian":
            return "DA restricted to correlated scenarios"
        if (
            method.da_method == "EQUI"
            and scenario.covariance != "equicorrelated_gaussian"
        ):
            return "EQUI dependence model requires equicorrelated generator"
        if (
            method.da_method == "BT"
            and scenario.covariance != "block_correlated_gaussian"
        ):
            return "BT only evaluated as a heuristic on block-correlated generator"
    return None


def _invoke_worker(
    req: WorkerRequest, timeout: float
) -> tuple[Status, WorkerResult | None, str | None, float]:
    started = time.monotonic()
    try:
        call = subprocess.run(
            [sys.executable, "-m", "llm_evaluation.trex_finite_k_worker"],
            input=req.model_dump_json(),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            "timeout",
            None,
            f"worker exceeded {timeout:g}s hard subprocess deadline",
            time.monotonic() - started,
        )
    except OSError as exc:
        # A child that never starts still consumes one explicit method attempt.
        # Do not leak executable paths or operating-system error details.
        number = exc.errno if exc.errno is not None else "unknown"
        return (
            "failed",
            None,
            f"worker launch failed: OSError errno={number}",
            time.monotonic() - started,
        )
    elapsed = time.monotonic() - started
    if call.returncode == 42:
        return "unavailable", None, call.stderr[-600:], elapsed
    if call.returncode == 43:
        return "unqualified", None, call.stderr[-600:], elapsed
    if call.returncode:
        return "failed", None, f"exit={call.returncode}: {call.stderr[-600:]}", elapsed
    try:
        result = WorkerResult.model_validate_json(call.stdout)
    except ValueError as exc:
        return "failed", None, f"invalid worker output: {exc!s}"[:600], elapsed
    if req.k is not None and result.effective_k != req.k:
        return (
            "failed",
            None,
            f"upstream effective K={result.effective_k}, requested={req.k}",
            elapsed,
        )
    return "success", result, None, elapsed


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    t = probability * (len(ordered) - 1)
    low = math.floor(t)
    high = math.ceil(t)
    return ordered[low] + (ordered[high] - ordered[low]) * (t - low)


def bootstrap_ci(values: list[float], *, seed: int, draws: int) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap empty sequence")
    rng = random.Random(seed)
    means = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    ]
    return (_percentile(means, 0.025), _percentile(means, 0.975))


def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    first, second = set(a), set(b)
    return len(first & second) / len(first | second) if first or second else 1.0


def summarize(
    config: PilotConfig,
    records: list[RunRecord],
    verified: dict[tuple[str, str], list[GeneratedSelectionMetrics]],
) -> list[Summary]:
    grouped: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for rec in records:
        grouped[(rec.scenario, rec.method)].append(rec)
    summaries: list[Summary] = []
    for scenario in config.scenarios:
        for method in config.methods:
            group = grouped.get((scenario.name, method.name), [])
            primary = [r for r in group if r.rerun == 0]
            success = [r for r in primary if r.status == "success"]
            # Independent generated support/oracle aggregator is the canonical arithmetic.
            generated = verified.get((scenario.name, method.name), [])
            if len(generated) != len(success):
                raise ValueError(
                    "success count disagrees with independently validated oracle"
                )
            canonical = aggregate_generated_datasets(generated) if generated else None
            fdps = [r.fdp for r in success if r.fdp is not None]
            tprs = [r.tpr for r in success if r.tpr is not None]
            failures: dict[str, int] = {}
            for r in primary:
                if r.status != "success":
                    failures[r.status] = failures.get(r.status, 0) + 1
            pairs: list[float] = []
            incomplete_pairs = 0
            if method.reruns == 2:
                by_ds: dict[str, dict[int, RunRecord]] = defaultdict(dict)
                for r in group:
                    by_ds[r.dataset_id][r.rerun] = r
                for by_run in by_ds.values():
                    first, second = by_run.get(0), by_run.get(1)
                    if (
                        first is not None
                        and second is not None
                        and first.status == "success"
                        and second.status == "success"
                        and first.selected is not None
                        and second.selected is not None
                    ):
                        pairs.append(_jaccard(first.selected, second.selected))
                    else:
                        incomplete_pairs += 1
            if canonical is not None:
                if not math.isclose(
                    float(canonical.empirical_fdr),
                    statistics.fmean(fdps),
                    abs_tol=1e-12,
                ):
                    raise ValueError("FDR drift from established oracle")
                if tprs:
                    mean_tpr = canonical.mean_applicable_tpr
                    if mean_tpr is None or not math.isclose(
                        float(mean_tpr), statistics.fmean(tprs), abs_tol=1e-12
                    ):
                        raise ValueError("TPR drift from established oracle")
            summaries.append(
                Summary(
                    scenario=scenario.name,
                    method=method.name,
                    requested_k=method.k,
                    intended_datasets=config.independent_replicates,
                    successful_datasets=len(success),
                    failures=failures,
                    fdr=float(canonical.empirical_fdr) if canonical else None,
                    fdr_ci95=bootstrap_ci(
                        fdps,
                        seed=derived_seed(
                            config.analysis_seed, scenario.name, method.name, "fdp"
                        ),
                        draws=config.bootstrap_draws,
                    )
                    if fdps
                    else None,
                    mean_tpr=float(canonical.mean_applicable_tpr)
                    if canonical and canonical.mean_applicable_tpr is not None
                    else None,
                    tpr_ci95=bootstrap_ci(
                        tprs,
                        seed=derived_seed(
                            config.analysis_seed, scenario.name, method.name, "tpr"
                        ),
                        draws=config.bootstrap_draws,
                    )
                    if tprs
                    else None,
                    p_fdp_gt_q=statistics.fmean(
                        float(v > config.target_q) for v in fdps
                    )
                    if fdps
                    else None,
                    p_fdp_ge_half=statistics.fmean(float(v >= 0.5) for v in fdps)
                    if fdps
                    else None,
                    p_empty=statistics.fmean(float(bool(r.empty)) for r in success)
                    if success
                    else None,
                    p_proxy=statistics.fmean(
                        float(bool(r.proxy_selected)) for r in success
                    )
                    if scenario.proxy_index is not None and success
                    else None,
                    mean_runtime_seconds=statistics.fmean(
                        r.elapsed_seconds
                        for r in success
                        if r.elapsed_seconds is not None
                    )
                    if success
                    else None,
                    max_peak_rss_mib=max(
                        (r.peak_rss_mib for r in success if r.peak_rss_mib is not None),
                        default=None,
                    ),
                    mean_rerun_jaccard=statistics.fmean(pairs) if pairs else None,
                    rerun_pairs_completed=len(pairs) if method.reruns == 2 else None,
                    rerun_pairs_incomplete=incomplete_pairs
                    if method.reruns == 2
                    else None,
                )
            )
    return summaries


def execute(config: PilotConfig) -> tuple[list[RunRecord], list[Summary]]:
    started = time.monotonic()
    records: list[RunRecord] = []
    verified: dict[tuple[str, str], list[GeneratedSelectionMetrics]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    seen_seeds: dict[str, set[int]] = defaultdict(set)
    for scenario in config.scenarios:
        for index in range(config.independent_replicates):
            ds_seed = derived_seed(
                config.seed_base,
                config.development_namespace,
                scenario.name,
                str(index),
                "dataset",
            )
            if ds_seed in seen_seeds[scenario.name]:
                raise ValueError(
                    "dataset seed collision: independent replicate integrity failed"
                )
            seen_seeds[scenario.name].add(ds_seed)
            dataset = generate_linear_dataset(scenario.design(ds_seed))
            if dataset.dataset_id in seen[scenario.name]:
                raise ValueError("duplicate realized dataset fingerprint")
            seen[scenario.name].add(dataset.dataset_id)
            for method in config.methods:
                unsupported = eligibility(scenario, method)
                for rerun in range(method.reruns):
                    method_seed = derived_seed(
                        config.seed_base,
                        config.development_namespace,
                        scenario.name,
                        str(index),
                        method.name,
                        str(rerun),
                        "selector",
                    )
                    base = dict(
                        pilot_id=config.pilot_id,
                        scenario=scenario.name,
                        dataset_id=dataset.dataset_id,
                        dataset_seed=ds_seed,
                        method=method.name,
                        method_kind=method.kind,
                        requested_k=method.k,
                        da_method=method.da_method,
                        selector_seed=method_seed,
                        rerun=rerun,
                        true_support=tuple(sorted(dataset.true_support)),
                    )
                    if unsupported is not None:
                        records.append(
                            RunRecord.model_validate(
                                {**base, "status": "unsupported", "error": unsupported}
                            )
                        )
                        continue
                    if time.monotonic() - started >= config.overall_budget_seconds:
                        records.append(
                            RunRecord.model_validate(
                                {
                                    **base,
                                    "status": "budget_exhausted",
                                    "error": "predeclared global budget elapsed",
                                }
                            )
                        )
                        continue
                    request = WorkerRequest(
                        kind=method.kind,
                        k=method.k,
                        da_method=method.da_method,
                        seed=method_seed,
                        q=config.target_q,
                        x=[list(row) for row in dataset.x],
                        y=list(dataset.y),
                        stability_subsamples=method.stability_subsamples,
                    )
                    status, outcome, error, elapsed = _invoke_worker(
                        request, config.per_method_timeout_seconds
                    )
                    if status != "success" or outcome is None:
                        records.append(
                            RunRecord.model_validate(
                                {
                                    **base,
                                    "status": status,
                                    "error": error,
                                    "elapsed_seconds": elapsed,
                                }
                            )
                        )
                        continue
                    try:
                        scored = score_generated_dataset(dataset, outcome.selected)
                    except (TypeError, ValueError) as exc:
                        records.append(
                            RunRecord.model_validate(
                                {
                                    **base,
                                    "status": "failed",
                                    "elapsed_seconds": elapsed,
                                    "error": f"selection/scorer integrity: {exc!s}"[
                                        :600
                                    ],
                                }
                            )
                        )
                        continue
                    metrics = scored.metrics
                    if rerun == 0:
                        verified[(scenario.name, method.name)].append(scored)
                    records.append(
                        RunRecord.model_validate(
                            {
                                **base,
                                "status": "success",
                                "selected": tuple(sorted(metrics.selected_features)),
                                "effective_k": outcome.effective_k,
                                "r": metrics.r,
                                "tp": metrics.tp,
                                "v": metrics.v,
                                "fdp": float(metrics.fdp),
                                "tpr": float(metrics.tpr)
                                if metrics.tpr is not None
                                else None,
                                "proxy_selected": (
                                    scenario.proxy_index in metrics.selected_features
                                    if scenario.proxy_index is not None
                                    else None
                                ),
                                "empty": metrics.r == 0,
                                "elapsed_seconds": elapsed,
                                "peak_rss_mib": outcome.peak_rss_mib,
                                "dummy_multiplier_l": outcome.dummy_multiplier_l,
                                "num_dummies": outcome.num_dummies,
                                "t_stop": outcome.t_stop,
                                "selection_threshold": outcome.selection_threshold,
                                "solver": outcome.solver,
                                "dummy_law": outcome.dummy_law,
                                "scaling": outcome.scaling,
                            }
                        )
                    )
    return records, summarize(config, records, verified)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_provenance() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }
