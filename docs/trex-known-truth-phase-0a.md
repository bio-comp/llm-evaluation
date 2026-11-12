# T-Rex known-truth laboratory — Phase 0A foundation

**Status:** Local implementation for independent review; not a T-Rex validation, a
published experimental result, or an accepted FDR guarantee. Google Drive
`llm_evaluation/EXPERIMENTS.md` (T-Rex candidate and Phase 0A–0D scaffold)
owns the scientific design; AR-18 and LE-131/LE-132 own the question and
source applicability. This document describes only executable Phase 0A behavior.

## Supported generating condition

`LinearGaussianDesign(n, p, beta, seed, covariance_family, noise_law,
noise_scale, response_family)` declares

```text
X[i,j] ~ independent N(0,1), i=0..n-1, j=0..p-1
epsilon[i] ~ independent N(0,noise_scale^2), independent of X
y[i] = fsum_j(X[i,j] * beta[j]) + epsilon[i] (computed with math.fsum)
```

No intercept. Exactly supported families:
`independent_standard_gaussian`, `gaussian`, and `continuous_linear`.
`n,p` must be positive exact integers; `seed` a nonnegative exact integer;
`beta` an immutable tuple of exactly `p` finite real, non-boolean
coefficients; `noise_scale` finite and nonnegative. Unsupported families,
malformed input, or nonfinite generated responses fail explicitly.
`noise_scale=0` uses exact zero errors and consumes **no** random noise draws.
The validated generated-data scorer also **requires each recorded error to
equal zero exactly** for a zero-noise condition, even when a nonzero error
would be absorbed by floating-point rounding in the response. Positive-noise
conditions do not require every realized error to be nonzero.

`generate_linear_dataset` returns immutable nested tuples `x`, `y` and realized `epsilon`,
validated design metadata, `true_support={j: beta[j] != 0}`, dataset
fingerprint, RNG/Python identity, and distinct `group_ids` and
`sampling_ids` for each independent row. There is no simulated cluster or
repeated measurement: grouping is one singleton per observation.

**Null family:** `H0,j: beta[j] = 0` in the *declared multivariate linear
data-generating model*. This is conditional coefficient support. It is not the
null of zero marginal association, a univariate test, causal attribution, or
out-of-sample predictive usefulness. A correlated, structurally null proxy
can have strong marginal association with `y`; selecting it remains a false
discovery for this oracle (a hand-constructed counterexample is tested).
The initial independent-X generator does not yet simulate that covariance
departure. Identifiability under collinearity is a separate design issue.

## Reference selection oracle

All feature indices are **zero-based** in `[0,p)`. The scorer rejects
duplicates, non-integer (including boolean) indices and invalid ranges.
For declared true support `S` and selected set `S_hat`:

```text
R = |S_hat|
TP = |S_hat ∩ S|
V = |S_hat \\ S|
FDP = V / max(R,1)
TPR = TP / |S|     when |S| > 0
TPR = None         when |S| = 0 (NOT_APPLICABLE)
```

For `R=0`, `FDP=0`; when truth is nonempty and selection is empty,
`TPR=0`. A nonempty selection under all-null truth has `FDP=1`.
Counts are integers and ratios are exact `fractions.Fraction` instances.
Metric records check their count/rate consistency on construction.

The lower-level `aggregate_independent_datasets` is an **arithmetic-only**
oracle: it rejects duplicate `dataset_id` but **cannot validate** a fixed
generating condition or demonstrate independent sampling. Use it for
explicitly constructed toy cases, reference calculations and adversarial
null-label controls—not as the validated simulation aggregation entrypoint.

For generated experiments, `score_generated_dataset(dataset, selected_features)`
first checks the dataset's coefficient-derived true support, exact dataset
fingerprint, observation dimensions and generating equation (including
realized noise), sampling/group IDs, null-definition label and recorded
RNG metadata. It derives its truth labels directly from the **validated
design**; no caller-supplied ground truth is accepted. A forged dataset
identity, coefficients, observations or reference support causes a failure.
The same private fingerprint helper is used in generation and scoring;
no regenerated random sample is required.

The scorer returns a frozen `GeneratedSelectionMetrics` record with
`LinearGaussianCondition` (n, p, beta, covariance family, noise law,
noise scale, response family), original `replicate_seed`, `dataset_id`
and the unchanged `SelectionMetrics`. No X matrix is kept in that scored
record or the summary. `LinearGaussianCondition` excludes the replicate
seed **only**, preserving the full fixed generating distribution.
`aggregate_generated_datasets` requires all records to share exactly
that condition, have distinct replicate seeds and distinct dataset
identities. It returns `GeneratedSelectionSummary` with the generating
condition, independent-replicate count, per-replicate metrics and seeds,
and the same exact macro arithmetic as the reference oracle.

Within one correctly specified fixed condition, aggregation returns
per-dataset records, `M`, the **unweighted mean of individual FDPs** `FDR_hat=sum(FDP_m)/M`, mean applicable
TPR, applicable/nonapplicable TPR counts, and empty-selection rate. All-null
datasets count in the FDR denominator and are **excluded** from mean TPR;
if none have true positives available for definition, mean TPR is `None`.

As an independent arithmetic control, datasets with `(V,R)=(3,4)` and
`(0,1)` give `FDR_hat=(3/4+0)/2=3/8`, **not**
pooled `(3+0)/(4+1)=3/5`. Unit tests assert both formulas differ.

## Reproducibility and independence limits

Each generator invocation uses its **own** `random.Random(seed)` instance
with `gauss`; it never seeds module-global RNG. Identical conditions and
seeds reproduce identical arrays **under the same Python/RNG/runtime
implementation**. No cross-Python-version or cross-platform byte-for-byte
Gaussian RNG guarantee is asserted. A `trex0a:sha256` dataset ID hashes
`repr((domain, design, realized X, realized y))`; this is an in-runtime
realization fingerprint, not a portable canonical serialization. The
`python_version` and `rng_engine` accompany the dataset.

The generated-data aggregator rejects repeated seeds and/or dataset
identities, including repeated selector randomizations on a **fixed** X/y.
Distinct seeds and hashes are necessary safeguards, **not mathematical proof
of independent sampling**. Callers remain responsible for fresh independent
generation and preserving realization identity. Dummy experiments, solver
seeds, and repeated selection on fixed data are **not** new dataset
replicates. A pair of **different** generating conditions that intentionally
shares a random stream (common random numbers) may support paired
scenario comparisons, but **cannot** be pooled into a single fixed-condition
empirical FDR. A mixture-of-scenarios FDR requires an independently defined
target scenario distribution and weighting/uncertainty rule; this Phase 0A
API does not implement one. The generated record validates its incoming
dataset at scoring time; it is not a cryptographically authenticated artifact
or a guarantee against deliberate later fabrication of typed records.

The layer calculates descriptive finite-M empirical FDR, not
expectation-level FDR control, Monte Carlo uncertainty, selector
qualification or statistical causation.

## Implemented versus deferred

Implemented: independently seeded identity-covariance linear-Gaussian
realizations (including exact zero-noise), known conditional support,
dataset/sampling provenance, separately validated generated-dataset scoring,
condition-specific aggregation, exact toy set-based metrics and macro arithmetic,
all-null/empty behavior and discriminating toy/negative-control tests.

Deferred to Phase 0B–0D: block/equicorrelation, heavy tails, heteroskedastic,
nonlinear/binary/clustered response, noisy references and leakage;
released T-Rex/T-Rex+DA solver and license qualification; competitor matching,
finite-K variation, final-grid holdout, Monte Carlo intervals, predictive
accuracy, stability, runtime/memory measures and any operational adoption.
Nothing here estimates or asserts T-Rex's actual FDR, power, or causal validity.

Focused verification:

```bash
uv run --offline --frozen python -m pytest -q tests/test_trex_known_truth.py
```
