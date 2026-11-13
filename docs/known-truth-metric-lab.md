# Known-truth metric and reference-error laboratory

This laboratory is a deterministic/synthetic validation surface for evaluation instruments. It does **not** sample subject LLMs, does not make E-002 RUNNING, and does not establish a universal calibration, ECE, AURC, or release threshold.

Google Drive owns the experiment definition and interpretation. The public repository owns the executable generator, metrics, tests, frozen public-safe design fixture, and deliberately publishable immutable run records. The frozen public design is `fixtures/known_truth_metric_lab_design.json`; its exact canonical bytes SHA-256 is `0d95648cfb4125cd09718e6092a225cc20e3913e65fd34005193783919c1d148`, its canonical design hash is `9e02a784d8f6e14e3e046d5729d5a7a60ce9e0bf76ce88e19d87226058f653a7`, and its frozen-record canonical hash is `acfd7fabe0cd03fb50f7f1a8145245e39b7afc211f68f05cc500b99190e550ba`.

## Frozen FINAL design

The FINAL inventory contains 288 cells: 216 fixed-case base cells, 24 question-population stress cells, and 48 ambiguity/rater-population cells. Base conditions cross the seven probability values, four forecast rules, n in `{50, 200, 1000}`, k in `{1, 4, 16}`, independent versus perfectly shared within-item outcomes, and reference-flip eta in `{0, 0.1, 0.2}`. The question-population stress arm uses the frozen k=1/independent/reference-eta-endpoint policy. The ambiguity arm uses the same seven probability/theta values, four forecast rules, n in `{50, 200, 1000}`, and raters/item in `{1, 3, 9, 25}`; r=1 is the explicit single-reference boundary.

FINAL is exactly 2,000 independent dataset replicates per frozen cell. Seed derivation is deterministic and phase-separated under `llm-evaluation.metric-lab-seed.v1`, binding the design hash, phase, cell key, replicate index, and purpose. DEV and FINAL seed namespaces are mechanically disjoint. There is no early-success stop and no post-hoc cell reduction. Runtime is recorded as operational evidence but is not currently an executable stopping criterion; an interrupted run is incomplete and must not be interpreted as the frozen FINAL experiment.

Coverage/error Monte Carlo uncertainty uses Wilson 95% intervals (`wilson95-v1`); bias uncertainty uses the sample SE of replicate estimates (`sample-se-replicate-estimates-v1`); RMSE uncertainty uses a delta-method transformation of MSE Monte Carlo uncertainty (`delta-mse-v1`). The Kumar correction is identified as an unbiased estimator of **squared L2 calibration error**; `sqrt(max(0, corrected_squared_L2))` is retained only as a nonnegative presentation transform and is not called unbiased L2. Deterministic floating-point controls use exact mathematical oracles where available; the frozen promotion policy uses an absolute analytic-bias floor of `1e-10` together with a 4-MCSE discrepancy guard so machine-precision cells cannot create meaningless standardized failures. Hashes, bytes, identities, discrete counts, and schema values are exact.

## Required controls and interpretation

The run carries deterministic or known-truth controls for constant-but-calibrated/uninformative predictions, coarse-bin ECE cancellation, clean-versus-noisy-reference ranking reversal, positive-class inversion, single-reference symmetric-label-noise non-identifiability, prevalence sensitivity, subgroup cancellation, AURC denominator defects, dependence/pseudoreplication, sparse/empty bins, and extreme log-score boundaries. Bernoulli finite-sample L2 debiasing is explicitly inapplicable to the soft-label ambiguity target.

Observed estimator bias, interval undercoverage, reference-induced ranking reversal, or sparse-bin behavior are scientific outcomes, not run-invalidating failures. The run is invalid only when an integrity or predeclared control oracle fails: wrong frozen bytes/design identity, wrong cell inventory or replicate count, non-reconstructible seeds, failed deterministic controls, software-identity disagreement, or failed immutable finalization/readback. No estimator/default is promoted merely because it looks better in the grid; replacement requires a separate candidate-specific predeclared decision rule/tolerance and held-out validation for the intended target/reference/population.

The frozen executable acceptance policy is part of the design identity. Primary analytic Brier/log/classification-error discrepancies and question-population squared-L2 bias must remain within `max(4 × bias_MCSE, 1e-10)`. Under shared within-item outcomes with `k>1`, cluster-normal Brier coverage must have mean in `[0.94, 0.98]`, every evaluated cell at least `0.90`, and exceed naive-row mean coverage by at least `0.10`; the cluster-Hoeffding sensitivity interval must average at least `0.99` coverage while cluster-normal mean width is at most half its width. Independent fixed-case and question-population cluster-normal mean coverage must each remain in `[0.94, 0.99]`. The exact negative controls, soft-label debias inapplicability, unsplittable constant-score tie behavior, frozen inventory, 2,000-replicate count, frozen design binding, and software identity are also executable gates. These criteria validate the laboratory instrument; they do not create a universal estimator-selection or release threshold.

## Execution

Development execution is allowed to use fewer replicates and is explicitly a tuning/falsification phase:

```sh
uv run llm-evaluation-metric-lab \
  --phase development \
  --replicates 100 \
  --output /private/tmp/metric-lab-development.json
```

Historical execution note: the original FINAL was contingent on a fixture/implementation reconciliation, which was closed before the recorded November 2026 FINAL run. That one-time prerequisite is **already satisfied**, not a standing gate on future research. To independently reproduce the recorded FINAL condition, use its exact design and a verified source checkout:

```sh
uv run llm-evaluation-metric-lab \
  --phase final \
  --frozen-design fixtures/known_truth_metric_lab_design.json \
  --replicates 2000 \
  --output /private/tmp/metric-lab-final.json
```

`--replicates 2000` is redundant in FINAL but intentionally explicit here; the CLI rejects any other FINAL replicate count. Output publication is immutable and read back through the run-record verifier. Do not retrospectively mutate this completed FINAL fixture or tune its original condition after seeing outcomes. Design improvements are welcome as new prospective experiment versions with distinct identities; they do not need another global project freeze.
