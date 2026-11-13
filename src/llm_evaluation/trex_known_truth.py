"""Phase 0A known-coefficient synthetic data and independent selection oracles.

No T-Rex dependency or FDR-control claim is implemented here.
"""

from __future__ import annotations

import hashlib
import math
import platform
import random
from collections.abc import Sequence, Set
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

CovarianceFamily = Literal[
    "independent_standard_gaussian",
    "equicorrelated_gaussian",
    "block_correlated_gaussian",
]
NoiseLaw = Literal["gaussian"]
ResponseFamily = Literal["continuous_linear"]

_CONDITIONAL_NULL = "beta_j == 0 in the declared linear generating model"
_RNG = "python.random.Random.gauss"


def _finite_real(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite real number (not bool)")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside finite float range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_nonnegative_int(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


@dataclass(frozen=True, slots=True)
class LinearGaussianDesign:
    """Validated, independent-row continuous-response generating condition."""

    n: int
    p: int
    beta: tuple[float, ...]
    seed: int
    covariance_family: CovarianceFamily = "independent_standard_gaussian"
    noise_law: NoiseLaw = "gaussian"
    noise_scale: float = 1.0
    response_family: ResponseFamily = "continuous_linear"
    # repr=False preserves the Phase 0A design representation and fingerprints.
    rho: float | None = field(default=None, repr=False)
    block_size: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.n, "n", positive=True)
        _exact_nonnegative_int(self.p, "p", positive=True)
        _exact_nonnegative_int(self.seed, "seed")
        if self.covariance_family == "independent_standard_gaussian":
            if self.rho is not None or self.block_size is not None:
                raise ValueError(
                    "independent covariance does not accept rho or block_size"
                )
        elif self.covariance_family in (
            "equicorrelated_gaussian",
            "block_correlated_gaussian",
        ):
            if self.rho is None:
                raise ValueError("correlated covariance requires rho")
            rho = _finite_real(self.rho, "rho")
            if not 0.0 <= rho < 1.0:
                raise ValueError("rho must satisfy 0 <= rho < 1")
            object.__setattr__(self, "rho", rho)
            if self.covariance_family == "equicorrelated_gaussian":
                if self.block_size is not None:
                    raise ValueError("equicorrelated covariance rejects block_size")
            else:
                if self.block_size is None:
                    raise ValueError("block covariance requires block_size")
                block_size = _exact_nonnegative_int(
                    self.block_size, "block_size", positive=True
                )
                if not 2 <= block_size <= self.p:
                    raise ValueError("block_size must satisfy 2 <= block_size <= p")
        else:
            raise ValueError("unsupported covariance family")
        if self.noise_law != "gaussian":
            raise ValueError("unsupported noise law")
        if self.response_family != "continuous_linear":
            raise ValueError("unsupported response family")
        if not isinstance(self.beta, tuple) or len(self.beta) != self.p:
            raise ValueError("beta must be a tuple of exactly p coefficients")
        coefficients = tuple(
            _finite_real(coefficient, f"beta[{index}]")
            for index, coefficient in enumerate(self.beta)
        )
        scale = _finite_real(self.noise_scale, "noise_scale")
        if scale < 0.0:
            raise ValueError("noise_scale must be nonnegative")
        object.__setattr__(self, "beta", coefficients)
        object.__setattr__(self, "noise_scale", scale)


@dataclass(frozen=True, slots=True)
class KnownTruthDataset:
    """Immutable realized observations with their declared structural truth."""

    design: LinearGaussianDesign
    x: tuple[tuple[float, ...], ...]
    y: tuple[float, ...]
    epsilon: tuple[float, ...]
    true_support: frozenset[int]
    conditional_null_definition: str
    dataset_id: str
    group_ids: tuple[str, ...]
    sampling_ids: tuple[str, ...]
    rng_engine: str
    python_version: str


def _conditional_support(design: LinearGaussianDesign) -> frozenset[int]:
    return frozenset(
        index for index, coefficient in enumerate(design.beta) if coefficient != 0.0
    )


def _dataset_fingerprint(
    design: LinearGaussianDesign,
    x: tuple[tuple[float, ...], ...],
    y: tuple[float, ...],
) -> str:
    """Existing in-runtime fingerprint, shared by generation and verification."""
    payload: tuple[object, ...]
    if design.covariance_family == "independent_standard_gaussian":
        # Historical serialization domain: do not change its bytes.
        payload = ("trex-known-truth-v1", design, x, y)
        prefix = "trex0a"
    else:
        # Explicitly bind the correlated parameters omitted from design repr.
        payload = (
            "trex-known-truth-correlated-v1",
            design,
            design.rho,
            design.block_size,
            x,
            y,
        )
        prefix = "trex0b1"
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _response_value(
    row: tuple[float, ...], beta: tuple[float, ...], error: float
) -> float:
    return math.fsum(
        (
            *(
                feature * coefficient
                for feature, coefficient in zip(row, beta, strict=True)
            ),
            error,
        )
    )


def _gaussian_row(
    design: LinearGaussianDesign, rng: random.Random
) -> tuple[float, ...]:
    """One independently drawn Gaussian row with the declared covariance."""
    if design.covariance_family == "independent_standard_gaussian":
        # Preserve the Phase 0A RNG call order exactly.
        return tuple(rng.gauss(0.0, 1.0) for _ in range(design.p))

    rho = design.rho
    assert rho is not None  # validated by LinearGaussianDesign
    block_size = (
        design.p
        if design.covariance_family == "equicorrelated_gaussian"
        else design.block_size
    )
    assert block_size is not None
    shared_weight = math.sqrt(rho)
    individual_weight = math.sqrt(1.0 - rho)
    row: list[float] = []
    for block_start in range(0, design.p, block_size):
        shared = rng.gauss(0.0, 1.0)
        count = min(block_size, design.p - block_start)
        row.extend(
            shared_weight * shared + individual_weight * rng.gauss(0.0, 1.0)
            for _ in range(count)
        )
    return tuple(row)


def generate_linear_dataset(design: LinearGaussianDesign) -> KnownTruthDataset:
    """Draw independent Gaussian rows and independent N(0,scale**2) noise.

    Every observation is its own independent sampling group; no cluster model
    is simulated. A dataset_id fingerprints the realized data and condition.
    """
    if not isinstance(design, LinearGaussianDesign):
        raise TypeError("design must be a LinearGaussianDesign")
    rng = random.Random(design.seed)
    x = tuple(_gaussian_row(design, rng) for _ in range(design.n))
    errors = (
        tuple(0.0 for _ in range(design.n))
        if design.noise_scale == 0.0
        else tuple(rng.gauss(0.0, design.noise_scale) for _ in range(design.n))
    )
    try:
        y = tuple(
            _response_value(row, design.beta, error)
            for row, error in zip(x, errors, strict=True)
        )
    except OverflowError as exc:
        raise ValueError("generating model produced a non-finite response") from exc
    if any(not math.isfinite(response) for response in y):
        raise ValueError("generating model produced a non-finite response")

    # The fingerprint binds the design and realized x/y; see the shared
    # verification helper below. This is not cross-runtime serialization.
    dataset_id = _dataset_fingerprint(design, x, y)
    return KnownTruthDataset(
        design=design,
        x=x,
        y=y,
        epsilon=errors,
        true_support=_conditional_support(design),
        conditional_null_definition=_CONDITIONAL_NULL,
        dataset_id=dataset_id,
        group_ids=tuple(f"{dataset_id}:group:{i}" for i in range(design.n)),
        sampling_ids=tuple(f"{dataset_id}:observation:{i}" for i in range(design.n)),
        rng_engine=_RNG,
        python_version=platform.python_version(),
    )


def _feature_set(
    features: Sequence[int] | Set[int], *, p: int, name: str
) -> frozenset[int]:
    if not isinstance(features, (Sequence, Set)) or isinstance(
        features, (str, bytes, bytearray)
    ):
        raise TypeError(f"{name} must be a sequence or set of feature indices")
    validated: set[int] = set()
    for feature in features:
        if type(feature) is not int:
            raise TypeError(f"{name} indices must be exact integers")
        if feature < 0 or feature >= p:
            raise ValueError(f"{name} index outside [0, p)")
        if feature in validated:
            raise ValueError(f"{name} contains duplicate indices")
        validated.add(feature)
    return frozenset(validated)


@dataclass(frozen=True, slots=True)
class SelectionMetrics:
    dataset_id: str
    n_features: int
    true_support: frozenset[int]
    selected_features: frozenset[int]
    r: int
    tp: int
    v: int
    fdp: Fraction
    tpr: Fraction | None

    def __post_init__(self) -> None:
        """Prevent fabricated counts/rates from entering dataset aggregation."""
        if (
            not isinstance(self.dataset_id, str)
            or not self.dataset_id
            or (self.dataset_id.strip() != self.dataset_id)
        ):
            raise ValueError("invalid dataset_id")
        p = _exact_nonnegative_int(self.n_features, "n_features", positive=True)
        if not isinstance(self.true_support, frozenset) or not isinstance(
            self.selected_features, frozenset
        ):
            raise TypeError("metric supports must be frozen sets")
        truth = _feature_set(self.true_support, p=p, name="true_support")
        selected = _feature_set(self.selected_features, p=p, name="selected_features")
        tp = len(truth & selected)
        r = len(selected)
        v = r - tp
        tpr = Fraction(tp, len(truth)) if truth else None
        if (
            type(self.r) is not int
            or type(self.tp) is not int
            or type(self.v) is not int
            or not isinstance(self.fdp, Fraction)
            or (self.tpr is not None and not isinstance(self.tpr, Fraction))
            or (self.r, self.tp, self.v, self.fdp, self.tpr)
            != (r, tp, v, Fraction(v, max(r, 1)), tpr)
        ):
            raise ValueError("selection metric record disagrees with its supports")


def score_selection(
    *,
    dataset_id: str,
    n_features: int,
    true_support: Sequence[int] | Set[int],
    selected_features: Sequence[int] | Set[int],
) -> SelectionMetrics:
    """Score a declared *conditional-coefficient* support, not marginal tests."""
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or dataset_id.strip() != dataset_id
    ):
        raise ValueError("dataset_id must be a nonempty trimmed string")
    p = _exact_nonnegative_int(n_features, "n_features", positive=True)
    truth = _feature_set(true_support, p=p, name="true_support")
    selected = _feature_set(selected_features, p=p, name="selected_features")
    tp = len(selected & truth)
    r = len(selected)
    return SelectionMetrics(
        dataset_id=dataset_id,
        n_features=p,
        true_support=truth,
        selected_features=selected,
        r=r,
        tp=tp,
        v=len(selected - truth),
        fdp=Fraction(len(selected - truth), max(r, 1)),
        tpr=Fraction(tp, len(truth)) if truth else None,
    )


@dataclass(frozen=True, slots=True)
class IndependentSelectionSummary:
    independent_dataset_count: int
    empirical_fdr: Fraction
    mean_applicable_tpr: Fraction | None
    tpr_applicable_count: int
    tpr_nonapplicable_count: int
    empty_selection_rate: Fraction
    per_dataset: tuple[SelectionMetrics, ...]


def aggregate_independent_datasets(
    records: Sequence[SelectionMetrics],
) -> IndependentSelectionSummary:
    """Pure arithmetic macro-average of dataset FDP, never pooled V/R.

    This function cannot validate fixed generating conditions or independent
    sampling. For generated experiments use aggregate_generated_datasets().
    """
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of SelectionMetrics")
    if not records:
        raise ValueError("at least one independent dataset is required")
    if any(not isinstance(record, SelectionMetrics) for record in records):
        raise TypeError("all records must be SelectionMetrics")
    ids = [record.dataset_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "duplicate dataset_id: selector reruns are not independent datasets"
        )
    applicable = [record.tpr for record in records if record.tpr is not None]
    count = len(records)
    return IndependentSelectionSummary(
        independent_dataset_count=count,
        empirical_fdr=sum((record.fdp for record in records), Fraction()) / count,
        mean_applicable_tpr=(
            sum((value for value in applicable if value is not None), Fraction())
            / len(applicable)
            if applicable
            else None
        ),
        tpr_applicable_count=len(applicable),
        tpr_nonapplicable_count=count - len(applicable),
        empty_selection_rate=Fraction(sum(record.r == 0 for record in records), count),
        per_dataset=tuple(records),
    )


@dataclass(frozen=True, slots=True)
class LinearGaussianCondition:
    """One fixed generating distribution, intentionally excluding replicate seed."""

    n: int
    p: int
    beta: tuple[float, ...]
    covariance_family: CovarianceFamily
    noise_law: NoiseLaw
    noise_scale: float
    response_family: ResponseFamily
    rho: float | None = field(default=None, repr=False)
    block_size: int | None = field(default=None, repr=False)

    @classmethod
    def from_design(cls, design: LinearGaussianDesign) -> LinearGaussianCondition:
        return cls(
            n=design.n,
            p=design.p,
            beta=design.beta,
            covariance_family=design.covariance_family,
            noise_law=design.noise_law,
            noise_scale=design.noise_scale,
            response_family=design.response_family,
            rho=design.rho,
            block_size=design.block_size,
        )

    def __post_init__(self) -> None:
        # Construct one existing validated design rather than maintaining
        # a second, potentially divergent set of scientific input rules.
        LinearGaussianDesign(
            n=self.n,
            p=self.p,
            beta=self.beta,
            seed=0,
            covariance_family=self.covariance_family,
            noise_law=self.noise_law,
            noise_scale=self.noise_scale,
            response_family=self.response_family,
            rho=self.rho,
            block_size=self.block_size,
        )


def _validate_generated_dataset(dataset: KnownTruthDataset) -> None:
    """Check declared truth and provenance against the *realized* data."""
    if not isinstance(dataset, KnownTruthDataset):
        raise TypeError("dataset must be a KnownTruthDataset")
    design = dataset.design
    if not isinstance(design, LinearGaussianDesign):
        raise TypeError("dataset design must be a LinearGaussianDesign")
    if (
        not isinstance(dataset.true_support, frozenset)
        or any(type(index) is not int for index in dataset.true_support)
        or dataset.true_support != _conditional_support(design)
    ):
        raise ValueError("generated true_support disagrees with design coefficients")
    if dataset.conditional_null_definition != _CONDITIONAL_NULL:
        raise ValueError("generated conditional null definition disagrees with design")
    if dataset.rng_engine != _RNG or (
        not isinstance(dataset.python_version, str) or not dataset.python_version
    ):
        raise ValueError("generated RNG/runtime provenance is invalid")
    if (
        not isinstance(dataset.x, tuple)
        or not isinstance(dataset.y, tuple)
        or not isinstance(dataset.epsilon, tuple)
        or len(dataset.x) != design.n
        or len(dataset.y) != design.n
        or len(dataset.epsilon) != design.n
    ):
        raise ValueError("generated observation dimensions disagree with design")
    for row, y_value, noise in zip(dataset.x, dataset.y, dataset.epsilon, strict=True):
        if not isinstance(row, tuple) or len(row) != design.p:
            raise ValueError("generated predictor dimensions disagree with design")
        if (
            any(
                type(x_value) is not float or not math.isfinite(x_value)
                for x_value in row
            )
            or type(y_value) is not float
            or not math.isfinite(y_value)
            or type(noise) is not float
            or not math.isfinite(noise)
        ):
            raise ValueError("generated observations must be finite floats")
        if design.noise_scale == 0.0 and noise != 0.0:
            raise ValueError(
                "generated zero-noise condition requires exactly zero errors"
            )
        try:
            expected_y = _response_value(row, design.beta, noise)
        except OverflowError as exc:
            raise ValueError("generated response equation overflowed") from exc
        if y_value != expected_y:
            raise ValueError("generated response disagrees with design and noise")
    if dataset.dataset_id != _dataset_fingerprint(design, dataset.x, dataset.y):
        raise ValueError("generated dataset_id disagrees with design and realization")
    if dataset.group_ids != tuple(
        f"{dataset.dataset_id}:group:{i}" for i in range(design.n)
    ) or dataset.sampling_ids != tuple(
        f"{dataset.dataset_id}:observation:{i}" for i in range(design.n)
    ):
        raise ValueError("generated observation/group IDs disagree with realization")


@dataclass(frozen=True, slots=True)
class GeneratedSelectionMetrics:
    """Lightweight scored replicate; no predictor matrix is retained."""

    condition: LinearGaussianCondition
    replicate_seed: int
    metrics: SelectionMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.condition, LinearGaussianCondition):
            raise TypeError("condition must be LinearGaussianCondition")
        _exact_nonnegative_int(self.replicate_seed, "replicate_seed")
        if not isinstance(self.metrics, SelectionMetrics):
            raise TypeError("metrics must be SelectionMetrics")
        if self.metrics.n_features != self.condition.p or (
            self.metrics.true_support
            != frozenset(
                i for i, value in enumerate(self.condition.beta) if value != 0.0
            )
        ):
            raise ValueError(
                "generated metric support/universe disagrees with condition"
            )

    @property
    def dataset_id(self) -> str:
        return self.metrics.dataset_id


def score_generated_dataset(
    dataset: KnownTruthDataset,
    selected_features: Sequence[int] | Set[int],
) -> GeneratedSelectionMetrics:
    """Validated generated-dataset path; never accepts an external truth label."""
    _validate_generated_dataset(dataset)
    design = dataset.design
    metrics = score_selection(
        dataset_id=dataset.dataset_id,
        n_features=design.p,
        true_support=_conditional_support(design),
        selected_features=selected_features,
    )
    return GeneratedSelectionMetrics(
        condition=LinearGaussianCondition.from_design(design),
        replicate_seed=design.seed,
        metrics=metrics,
    )


@dataclass(frozen=True, slots=True)
class GeneratedSelectionSummary:
    """Fixed-condition Monte Carlo arithmetic and per-replicate provenance."""

    condition: LinearGaussianCondition
    independent_dataset_count: int
    empirical_fdr: Fraction
    mean_applicable_tpr: Fraction | None
    tpr_applicable_count: int
    tpr_nonapplicable_count: int
    empty_selection_rate: Fraction
    per_dataset: tuple[GeneratedSelectionMetrics, ...]


def aggregate_generated_datasets(
    records: Sequence[GeneratedSelectionMetrics],
) -> GeneratedSelectionSummary:
    """Aggregate one generating condition across distinct replicate seeds.

    Seeds and realization hashes are necessary guards, not proof of statistical
    independence. Cross-condition mixtures require a separate declared estimand.
    """
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of GeneratedSelectionMetrics")
    if not records:
        raise ValueError("at least one generated replicate is required")
    if any(not isinstance(record, GeneratedSelectionMetrics) for record in records):
        raise TypeError("all records must be GeneratedSelectionMetrics")
    condition = records[0].condition
    if any(record.condition != condition for record in records):
        raise ValueError("cannot mix different generating conditions")
    seeds = [record.replicate_seed for record in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate replicate seed within generating condition")
    # The lower-level oracle additionally checks dataset identifiers and
    # preserves its original per-dataset FDP/TPR definitions exactly.
    arithmetic = aggregate_independent_datasets(
        tuple(record.metrics for record in records)
    )
    return GeneratedSelectionSummary(
        condition=condition,
        independent_dataset_count=arithmetic.independent_dataset_count,
        empirical_fdr=arithmetic.empirical_fdr,
        mean_applicable_tpr=arithmetic.mean_applicable_tpr,
        tpr_applicable_count=arithmetic.tpr_applicable_count,
        tpr_nonapplicable_count=arithmetic.tpr_nonapplicable_count,
        empty_selection_rate=arithmetic.empty_selection_rate,
        per_dataset=tuple(records),
    )
