"""One process per selector pipeline: isolate native crashes, timeout and memory.

Invoked exclusively through the development runner; no network/paid APIs.
"""

from __future__ import annotations

import resource
import sys
import traceback
from importlib import import_module

from llm_evaluation.trex_finite_k import WorkerRequest, WorkerResult
from llm_evaluation.trex_runtime_provenance import inspect_installed_trex


class UnqualifiedTRexRuntime(RuntimeError):
    """Native selector was not run because its installed artifact is unqualified."""


def _indices(values: object, p: int) -> tuple[int, ...]:
    """Convert *documented zero-based integer* native indices; never shift."""
    np = import_module("numpy")

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("selector indices must be one-dimensional native integers")
    # pybind11 returns an empty float64 ndarray for no selections;
    # no values are coerced when the sequence is empty.
    if arr.size == 0:
        return ()
    if arr.dtype.kind not in ("i", "u"):
        raise ValueError("selector indices must be one-dimensional native integers")
    indices = tuple(int(index) for index in arr.tolist())
    if len(set(indices)) != len(indices) or any(
        index < 0 or index >= p for index in indices
    ):
        raise ValueError("invalid, duplicated or out-of-range selector indices")
    return indices


def _rss_mib() -> float:
    # Darwin ru_maxrss is bytes; Linux reports KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024**2) if sys.platform == "darwin" else value / 1024)


def run(request: WorkerRequest) -> WorkerResult:
    if request.kind in ("trex", "trex_da"):
        evidence = inspect_installed_trex()
        if evidence.status == "unavailable":
            raise ModuleNotFoundError(
                "trexselector is not installed", name="trexselector"
            )
        if evidence.status != "qualified":
            raise UnqualifiedTRexRuntime(
                evidence.reason or "unverified trexselector runtime"
            )

    np = import_module("numpy")

    x = np.array(request.x, dtype=np.float64, order="F", copy=True)
    y = np.array(request.y, dtype=np.float64, copy=True)
    if (
        x.ndim != 2
        or y.shape != (x.shape[0],)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("invalid worker input dimensions/finite values")
    p = int(x.shape[1])
    if request.kind in ("trex", "trex_da"):
        # The upstream Python constructor sometimes returns the provided
        # Fortran-order buffer and C++ normalizes it IN PLACE before restoring.
        # Never supply the caller's dataset array to this boundary.
        original_x = x.copy(order="F")
        original_y = y.copy()
        trex = import_module("trexselector")
        ctrl = trex.TRexControlParameter()
        assert request.k is not None
        ctrl.K = request.k
        ctrl.solver_type = trex.SolverTypeForTRex.TLARS
        ctrl.lloop_strategy = trex.LLoopStrategy.STANDARD
        ctrl.max_outer_threads = 1
        ctrl.max_inner_threads = 1
        ctrl.parallel_rnd_experiments = False
        if request.kind == "trex":
            selector = trex.TRexSelector(
                x,
                y,
                tFDR=request.q,
                trex_control=ctrl,
                seed=request.seed,
                verbose=False,
            )
        else:
            da = trex.TRexDAControlParameter()
            if request.da_method == "EQUI":
                da.method = trex.DAMethod.EQUI
            elif request.da_method == "BT":
                da.method = trex.DAMethod.BT
                da.bt_selection_mode = trex.BTSelectionMode.FeasibleOnly
            else:
                raise ValueError("unqualified DA method")
            selector = trex.TRexDASelector(
                x,
                y,
                tFDR=request.q,
                trex_control=ctrl,
                da_control=da,
                seed=request.seed,
                verbose=False,
            )
        outcome = selector.select()
        selected = _indices(selector.selected_indices, p)
        # The native backend normalizes Fortran-order arrays in place and
        # restores them only to floating-point precision; the canonical
        # generated dataset is never passed to this working buffer.
        if not np.allclose(x, original_x, rtol=0, atol=1e-12) or not np.array_equal(
            y, original_y
        ):
            raise ValueError(
                "native selector failed approximate working-input restoration"
            )
        effective_k = int(selector.K)
        if effective_k != request.k or int(ctrl.K) != request.k:
            raise ValueError("upstream ignored/overrode K")
        return WorkerResult(
            selected=selected,
            effective_k=effective_k,
            dummy_multiplier_l=int(outcome.dummy_factor_L),
            num_dummies=int(outcome.num_dummies),
            t_stop=int(outcome.T_stop),
            peak_rss_mib=_rss_mib(),
            solver="TLARS",
            dummy_law="STANDARD iid N(0,1)",
            scaling="upstream L2",
            selection_threshold=float(outcome.v_thresh),
        )

    # Optional research packages are loaded only in the isolated worker;
    # base distribution and public CI do not require their imports.
    linear = import_module("sklearn.linear_model")
    KFold = import_module("sklearn.model_selection").KFold
    StandardScaler = import_module("sklearn.preprocessing").StandardScaler

    standardized = StandardScaler().fit_transform(x)
    folds = KFold(n_splits=3, shuffle=True, random_state=request.seed)
    if request.kind in ("lasso_cv", "stability"):
        cv = linear.LassoCV(
            alphas=50, cv=folds, random_state=request.seed, n_jobs=1, max_iter=5000
        )
        cv.fit(standardized, y)
        alpha = float(cv.alpha_)
        if request.kind == "lasso_cv":
            selected = _indices(np.flatnonzero(np.abs(cv.coef_) > 1e-8), p)
            return WorkerResult(
                selected=selected,
                effective_k=None,
                peak_rss_mib=_rss_mib(),
                solver="LassoCV",
                scaling="StandardScaler",
                selection_threshold=alpha,
            )
        rng = np.random.default_rng(request.seed)
        counts = np.zeros(p, dtype=np.int64)
        for _ in range(request.stability_subsamples):
            rows = rng.choice(x.shape[0], size=x.shape[0] // 2, replace=False)
            # The development comparator fixes full-data CV alpha, then
            # measures half-sample inclusion frequency (not an FDR guarantee).
            model = linear.Lasso(alpha=alpha, max_iter=5000)
            model.fit(standardized[rows], y[rows])
            counts += (np.abs(model.coef_) > 1e-8).astype(np.int64)
        selected = _indices(
            np.flatnonzero(counts >= 0.8 * request.stability_subsamples), p
        )
        return WorkerResult(
            selected=selected,
            effective_k=None,
            peak_rss_mib=_rss_mib(),
            solver="LassoCV-fixed-alpha subsample stability",
            scaling="StandardScaler",
            selection_threshold=0.8,
        )

    if request.kind == "bh_independent":
        student_t = import_module("scipy.stats").t

        centered = x - x.mean(axis=0)
        cy = y - y.mean()
        norms = np.sqrt(np.sum(centered * centered, axis=0) * np.sum(cy * cy))
        if not np.all(norms > 0):
            raise ValueError("degenerate design/response for Pearson tests")
        corr = np.clip(centered.T @ cy / norms, -1, 1)
        df = x.shape[0] - 2
        t_stat = corr * np.sqrt(df / np.maximum(1 - corr * corr, np.finfo(float).eps))
        p_values = 2 * student_t.sf(np.abs(t_stat), df=df)
        order = np.argsort(p_values, kind="stable")
        rank = np.arange(1, p + 1)
        accepted = np.flatnonzero(p_values[order] <= request.q * rank / p)
        count = int(accepted[-1]) + 1 if len(accepted) else 0
        selected = _indices(np.sort(order[:count]), p)
        return WorkerResult(
            selected=selected,
            effective_k=None,
            peak_rss_mib=_rss_mib(),
            solver="Pearson marginal BH",
            scaling="centered correlation",
            selection_threshold=request.q,
        )
    raise ValueError("unsupported selector kind")


def main() -> int:
    try:
        req = WorkerRequest.model_validate_json(sys.stdin.read())
        print(run(req).model_dump_json(), flush=True)
        return 0
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"optional research dependency unavailable: {exc}", file=sys.stderr)
        return 42
    except UnqualifiedTRexRuntime as exc:
        print(f"unqualified research runtime: {exc}", file=sys.stderr)
        return 43
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
