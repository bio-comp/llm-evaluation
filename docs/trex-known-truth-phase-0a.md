# T-Rex known-truth laboratory — Phase 0A foundation

**Status:** Phase 0A foundation integrated in GitHub `main` at
`a0c2a5569e36ffa4c815f4ef5129f37c5e60e286`. Phase 0B1 correlated
design generation and proxy controls are implemented as exploratory instrumentation.
Neither phase is a T-Rex validation, a scored selector experiment, or an accepted
FDR guarantee. Google Drive
`llm_evaluation/EXPERIMENTS.md` (T-Rex candidate and Phase 0A–0D scaffold)
owns the scientific design; AR-18 and LE-131/LE-132 own the question and
source applicability. Phase 0A contracts and the Phase 0B1 extension
are documented separately below.

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
Phase 0B1 implements that covariance departure separately (see addendum).
Identifiability under collinearity is a separate design issue.

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

Implemented in Phase 0B1: block/equicorrelation and conditional-null
active-proxy controls (see addendum). Deferred to later phases: heavy tails,
heteroskedastic, nonlinear/binary/clustered response, noisy references and leakage;
released T-Rex/T-Rex+DA solver and license qualification; competitor matching,
finite-K variation, final-grid holdout, Monte Carlo intervals, predictive
accuracy, stability, runtime/memory measures and any operational adoption.
Nothing here estimates or asserts T-Rex's actual FDR, power, or causal validity.

Focused verification:

```bash
uv run --offline --frozen python -m pytest -q tests/test_trex_known_truth.py
```


## Phase 0B1 — correlated Gaussian known-truth extension (exploratory implementation)

This extends the existing generator without running T-Rex, T-Rex+DA, or any
selector. The live Drive EXPERIMENTS.md Phase 0A–0D scaffold and AR-18 remain
scientific authority. Fixtures below are exploratory development conditions,
not completed experiments or an accepted final FDR grid.

### Covariance contract

`LinearGaussianDesign` retains the Phase 0A constructor and adds optional
`rho: float | None = None` and `block_size: int | None = None`.
Both have `repr=False`; independent Phase 0A repr/fingerprints are unchanged.
Families are validated with no ignored or inconsistent arguments:

- `independent_standard_gaussian`: Σ=I_p; `rho=None` and `block_size=None` required.
- `equicorrelated_gaussian`: `0 <= rho < 1` required, block size forbidden;
  Σ(j,j)=1, Σ(j,k)=rho for distinct predictors.
- `block_correlated_gaussian`: `0 <= rho < 1` and exact non-boolean integer
  `2 <= block_size <= p` required. Blocks are contiguous indices
  [b*block_size, min((b+1)*block_size,p)); last block can be a singleton.
  Σ(j,k)=rho when distinct predictors share a block, zero across blocks,
  and one on the diagonal.

Nonfinite, boolean, negative, or >=1 `rho`, unknown families, nonintegral
or invalid block sizes, missing and unused parameters are rejected. Negative
equicorrelation and general covariance matrices are unsupported.
A correlated family with rho=0 has identity population covariance but a
distinct condition identity and RNG consumption pattern.

For independent Gaussian Z[i,b] shared per observation/block, independent
U[i,j], and independent Gaussian error epsilon[i]:

```text
X[i,j] = sqrt(rho)*Z[i,block(j)] + sqrt(1-rho)*U[i,j]
Var(X[i,j]) = rho + (1-rho) = 1
Cov(X[i,j], X[i,k]) = rho (j != k, same block); else 0
y[i] = sum_j X[i,j]*beta[j] + epsilon[i] (math.fsum)
```

Equicorrelation uses one block spanning all predictors. Fresh latent draws
per row guarantee independent rows under the generator; dependent predictors
do **not** create clustered/repeated observations. Independent `group_ids` and
`sampling_ids` remain one per row, with Gaussian noise independent of X.
Zero-noise consumes no error draws and stores exact zero errors.

### Active-null proxy oracle

For a single active feature k and structural-null proxy j, with
beta[k] ≠ 0, beta[j] = 0 and independent noise:

```text
Cov(X[j], y) = sum_l beta[l]*Cov(X[j],X[l])
             = rho*beta[k] (correlated j and k)
             = 0           (independent control)
```

At rho=0.9 and beta=(2,0,0), null predictor 1 has population marginal
covariance 1.8 with y despite zero conditional coefficient. Tests
independently check population algebra and sampled covariances. Selecting
only the proxy scores TP=0, V=1, FDP=1; selecting {0,1} scores
TP=1, V=1, FDP=1/2. This is a **conditional coefficient** null, not a
univariate zero-marginal-association null. A marginal BH comparator and
a conditional-support selector require separate hypothesis definitions.

### Versioned fingerprints and fixed-condition aggregation

The Phase 0A independent payload is unchanged:

```text
repr(("trex-known-truth-v1", design, X, y))
dataset_id = "trex0a:" + sha256(UTF8(payload)).hexdigest()
```

Regression on Python 3.11.13 with n=9, p=4, beta=(0,1.5,0,-2),
seed=23, noise_scale=0.7:

```text
trex0a:af481762ac77947358040fcdc3bb1a7dfdeba47bedc9671fe0389e30c75024fa
```

New correlated dataset IDs use:

```text
payload = repr(("trex-known-truth-correlated-v1",
                design, rho, block_size, X, y))
dataset_id = "trex0b1:" + sha256(UTF8(payload)).hexdigest()
```

The correlated parameters bind the fingerprint despite their exclusion
from the legacy design repr. These are local runtime realization checks,
not cross-platform canonical serializations or authenticated proofs of
sampling. `LinearGaussianCondition` equality includes the covariance
family, rho and block size; it excludes replicate seed only. Mixed
covariance families, strengths, structures, beta, noise, n, or p cannot
be aggregated. Score arithmetic, validated generated-data scoring,
all-null TPR non-applicability and exact zero-noise checks remain unchanged.

### Development scenario matrix (specified, NOT RUN as a grid)

All cases use Gaussian noise scale 1, continuous linear response,
independent observations, and zero coefficients outside the listed
supports. Active coefficients equal the listed amplitude. The seed
column is an *initial development fixture seed*, not a final replicate
budget. A future development replicate could use seed=base_seed+r
(r>=0) with the realization hash recorded for each r. Holdout/final
evaluation must use disjoint, prespecified seeds and a separate
Monte Carlo plan. CORE-CANDIDATE means a baseline family for future
solver qualification, **not** proven theorem applicability;
DEPENDENCE-STRESS means exact T-Rex/T-Rex+DA applicability is unknown.

| Scenario | n | p | Covariance | rho / block | Active support | beta | Seed | Intended role |
|---|---:|---:|---|---|---|---:|---:|---|
| 0B1-I-01 | 80 | 40 | independent | none | empty | none | 1001 | CORE-CANDIDATE; all-null |
| 0B1-E0-02 | 80 | 160 | equicorrelated | 0 / none | 0,40,80 | 0.4 | 1002 | CORE-CANDIDATE; zero-correlation |
| 0B1-E5-03 | 200 | 40 | equicorrelated | 0.5 / none | 0,11,27 | 1.5 | 1003 | DEPENDENCE-STRESS; p<n |
| 0B1-E9-04 | 80 | 400 | equicorrelated | 0.9 / none | 0,80,160 | 2.0 | 1004 | DEPENDENCE-STRESS; p>n/proxy |
| 0B1-B0-05 | 200 | 160 | block | 0 / 8 | 0,8,16,24,32,40,48,56,64,72 | 0.4 | 1005 | CORE-CANDIDATE; block wiring |
| 0B1-B5-06 | 80 | 400 | block | 0.5 / 10 | 0,40,80,120,160,200,240,280,320,360 | 1.5 | 1006 | DEPENDENCE-STRESS; ten active |
| 0B1-B9-07 | 200 | 40 | block | 0.9 / 5 | 0,10,20 | 1.5 | 1007 | DEPENDENCE-STRESS; proxy at j=1 |
| 0B1-B9-08 | 200 | 400 | block | 0.9 / 7 | 0,40,80,120,160,200,240,280,320,360 | 0.4 | 1008 | DEPENDENCE-STRESS; final singleton |
| 0B1-I-09 | 200 | 40 | independent | none | 0,10,20 | 1.5 | 1009 | CORE-CANDIDATE; independent proxy control |

The unit suite separately exercises finite deterministic fixtures, large-n
sample covariance with absolute tolerance 0.065, independent
population-formula checks, across-block independence and a deliberately
incorrect shared-latent-across-blocks counterfactual. Fixed-seed sample
covariance tolerance is not a statistical confidence interval. These
tests do not constitute a measured selector-FDR or power experiment.

### Exclusions and next phase

Not implemented: T-Rex or T-Rex+DA installation/selection, competitors,
finite K, final MC confidence intervals, heavy tails, binary/nonlinear
responses, clustered data, heteroskedastic noise, leakage, reference error,
causal explanation or deployment. Primary method lineages are
https://arxiv.org/html/2110.06048 (T-Rex) and
https://arxiv.org/html/2401.15796 (T-Rex+DA). Neither establishes a
universal fixed-K FDR guarantee for these synthetic regimes. Exact
selector/solver version, assumptions, budget, held-out scenario grid
and predeclared FDR/power/uncertainty rules belong to later phases.

Focused verification:

```bash
uv run --offline --frozen python -m pytest -q tests/test_trex_known_truth.py tests/test_trex_correlated_gaussian.py
```
