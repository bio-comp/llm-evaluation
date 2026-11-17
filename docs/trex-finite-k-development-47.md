# Issue #47 — Finite-K T-Rex development evaluation (NOT FINAL)

**Status:** executable offline development experiment; no accepted FDR guarantee, no production selector, no independent confirmation. Scientific authority remains the existing llm_evaluation/EXPERIMENTS.md (Google Drive ID 11aLrfuynQQfp6C4utseQHV-KTYl01kWfW-JAEMs2Ti4), AR-18, LE-131/LE-132, and D-020. Git is inspectable **development evidence**, not a replacement canonical.

## Question, null, and source applicability

The reference is **structural conditional-coefficient support** from the existing Phase 0A/0B1 LinearGaussianDesign; zero-coefficient predictors remain false discoveries even when correlated with active predictors. Selected features are not causal claims. Empirical FDR is the **mean of independent-dataset FDPs**, with FDP=0 for empty selection; it is not pooled V/R or a per-realized-run FDP cap. TPR is undefined for all-null cases.

- [Original T-Rex paper, arXiv:2110.06048v7](https://arxiv.org/html/2110.06048): Theorem 1 establishes control **as K approaches infinity** with calibrated voting/stopping and stochastic-dominance assumptions. Appendix F A-I requires mutually independent nulls and null independence from actives, **violated in correlated stress cases here**. Appendix F A-II stochastic domination was not directly tested.
- [Dependency-aware paper, arXiv:2401.15796v2](https://arxiv.org/html/2401.15796): Theorem 2 is likewise **as K approaches infinity**, subject to eligible group construction/calibration and additional conditions. The release's BT/EQUI settings were executed, but sufficiency of theorem-specific groups and dominance was **not independently established**.
- Author simulation claims and author-package claims are not independent experimental evidence. Python 1.0.0, R 2.0.1 and historical pure-R 1.0.0 are different released APIs; no scientific implementation equivalence was assumed.

## Exact software qualification and configuration

- Installed optional Python [PyPI trexselector 1.0.0](https://pypi.org/project/trexselector/1.0.0/), 2026-11-07; upstream Git tag v1.0.0 commit aa7e5cade4b7129e83417f19c437a811e828e8ac. GPL-3.0-or-later, isolated outside the base distribution and its lockfile.
- Directly downloaded and SHA-256 verified CPython 3.11 macOS arm64 wheel, 1,896,206 bytes, digest **82b645e5bb8b67547fb35f652a83d0a9ca3d801839961f8d07c407e26b161f2e**. Source archive, 681,488 bytes, digest **61f0a57b275cf4fc2a2f8c690f75a5ce7364951f3f27caa2729a7128991ab538**. Both matched PyPI.
- Host: macOS 26.6.2 arm64, Python 3.11.13, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, Pydantic 2.14.1. Linux and R 2.0.1 execution **not qualified**.
- Original: TRexSelector with tFDR=0.20, TLARS, LLoopStrategy.STANDARD, native L2 scaling, N(0,1) dummy distribution, max_dummy_multiplier=10, opt_threshold=0.75, use_max_T_stop=True, serial outer/inner threads. Request K=10/20/50/100; independently compare effective selector.K each run. Actual L, T_stop and dummy count preserved in run records.
- DA: TRexDASelector with same core, K20; DAMethod.BT with BTSelectionMode.FeasibleOnly on block-correlated conditions or DAMethod.EQUI on equicorrelated conditions. Native defaults rho_thr_DA=0.02, cor_coef=-2.0 (auto) and hc_grid_length=0. Automatic dependence-group eligibility remains **unverified**.
- Indices are native **zero-based**; first/last predictor indices independently smoke tested, duplicates/negative/out-of-range/non-integer rejected. The native binding may return an *empty float64 array* for no selection, accepted only when actually empty. A Fortran-contiguous working X may be normalized in place and restored to floating-point precision, so every child worker receives independent copies, never a shared canonical dataset. Python import sets KMP_DUPLICATE_LIB_OK=TRUE as an upstream side effect; workers execute in fresh isolated processes.

## Prospective design and executed development matrix

- Nine synthetic conditions, n80, p40 and p160 (both p<n and p>n), beta supports {0,3}, strong coefficient 2.5, weak 0.45, Gaussian noise scale 1, independent standard normal predictors, block correlation rho=0.5/0.9 and equicorrelation rho=0.5/0.9, active versus structural-null correlated proxy. All-null independent also present. Not tested: n200, heavy-tailed noise, binary response, noisy labels, clustered repeats or reference error.
- 16 **independent datasets per generating condition** from SHA-derived seeds, base seed 470016 and namespace issue47-development-v1; independent selector-specific seeds and deliberately defined K20 reruns. Analysis seed 470019; 500 percentile-bootstrap draws of **dataset FDP**, never of the K dummy experiments.
- Methods: four original T-Rex K values; DA BT/EQUI K20 only for eligible covariance structures; tuned LassoCV (3-fold, **non-FDR**); fixed-full-data-alpha half-sample stability heuristic (12 subsamples, 0.8 inclusion threshold, **not FDR-certified**); marginal Pearson+BH only on independent Gaussian X, where null equivalence can be defended. Correlated BH is **unsupported**, never mis-scored against a structural null. Different algorithms, tuning and wall times are not cost-matched.
- Budget: 480 s overall and hard 15 s per worker, read-only synthetic data, no paid API. Qualification 4 scenarios, 2 datasets each: 68 success, 28 expected unsupported, zero failures/timeouts. First full development run dev-001 used **321.66 s** and returned **1,728 records: 1,232 successful method calls, 496 expected unsupported and zero failures/timeouts/unavailable/budget exhaustion**. Among the successes, **1,008 primary dataset-method outputs and 224 deliberate selector reruns**; the 1,232 calls are **not 1,232 independent Monte Carlo datasets**. A second dev-002 run uses identical dataset/selector seeds to qualify integration corrections without touching a FINAL holdout.

## Development observations — first completed pilot, target q=0.20

Every FDR/TPR cell reflects **16 independent datasets**, not K dummy experiments. Bootstrap 95% interval is in brackets; all-null TPR is N/A. Full 81-cell matrix, FDP-tail rates, null proxy frequencies, empty rates, run times, memory and repeat Jaccard are in summary.json and records.jsonl.

| Scenario | Selector | Mean FDP/FDR [bootstrap 95%] | Mean TPR |
|---|---|---:|---:|
| independent all-null p40 | Original K20 | 0.000 [0.000, 0.000] | N/A |
| strong independent p40 | Original K20 | 0.072 [0.016, 0.138] | 1.000 |
| weak independent p40 | Original K20 | 0.000 [0.000, 0.000] | 0.542 |
| strong independent p160>n | Original K20 | 0.078 [0.031, 0.125] | 1.000 |
| strong block rho0.5 p40 | Original K20 | 0.191 [0.109, 0.278] | 1.000 |
| strong block rho0.5 p160>n | Original K20 | 0.251 [0.154, 0.350] | 1.000 |
| strong block rho0.9, designated null proxy | Original K20 | 0.472 [0.398, 0.543] | 1.000 |
| strong block rho0.9, designated null proxy | DA BT K20 | 0.271 [0.078, 0.479] | 0.333 |
| strong equicorrelation rho0.5 | Original K20 | 0.578 [0.501, 0.643] | 1.000 |
| strong equicorrelation rho0.5 | DA EQUI K20 | 0.000 [0.000, 0.000] | 0.000 |
| weak equicorrelation rho0.9 | Original K20 | 0.774 [0.689, 0.861] | 0.438 |
| weak equicorrelation rho0.9 | DA EQUI K20 | 0.000 [0.000, 0.000] | 0.000 |

**Finite-K sensitivity:** strong block rho0.9 original K10/K20/K50/K100 FDR roughly **0.502 / 0.472 / 0.466 / 0.479**; equicorrelated rho0.5 original K10/K20/K50/K100 roughly **0.574 / 0.578 / 0.576 / 0.565**. Increasing K did not materially remove these exploratory adverse outcomes. Original null independence was intentionally violated, so this **does not falsify the theorem under its assumptions**.

**Matched-dataset comparator example:** strong independent p40, LassoCV FDR=0.562 TPR=1.000; heuristic stability FDR=0.159 TPR=1.000; independent Pearson+BH FDR=0.159 TPR=1.000. Stability and Lasso were *never* positioned as FDR-controlled; Pearson+BH is not extended to correlated structural-support truth.

**Specific false selections:** strong block rho0.9 proxy index 1 has conditional coefficient 0, despite correlation with active index 0. Original K20 selected it on 5/16 datasets (0.3125), DA BT K20 on 1/16 (0.0625); other nulls also contribute to FDP. For weak equicorrelation rho0.9, original K20 selected the designated null proxy 4/16 times.

**Stability:** original K20 mean within-dataset rerun Jaccard ranged 0.920–1.000 across conditions; DA BT at high-correlation proxy condition 0.812 and at p160 block 0.809. This measures deliberately repeated selector randomness on the *same X,y*, not stability across newly sampled datasets.

**Resources:** across 144 primary datasets per original K value, K10 mean ~0.105s/51MiB high-water process RSS versus K100 ~0.188s/69MiB. DA BT K20 mean ~0.161s/60MiB (48 eligible primary datasets). LassoCV ~0.572s/126MiB, heuristic stability ~0.576s/126MiB. Fresh-process interpreter and imports contribute to process wall time/RSS; unmatched tuning makes resource comparisons descriptive, not equal-cost superiority claims.

### Monte Carlo uncertainty and adverse controls

- n16 per condition cannot certify near-q FDR. Dataset-bootstrap percentile CIs may be **degenerate [0,0]** for all-zero FDP samples. In all-null data 0/16 nonempty selections, a two-sided **exact binomial 95% upper bound is approximately 0.206** (1−0.025^(1/16)), not 0. A zero bootstrap interval is not a zero-risk guarantee. Report nontrivial uncertainty and independently size a FINAL test.
- Original correlated designs violate Appendix F A-I null independence; native selection theorems require additional calibration/dominance not measured here. For DA, actual automatic group construction against paper's sufficient group design remains unverified.
- Strong power alongside large structurally-null FDP is a substantive diagnostic, not to be dismissed as marginal correlation. Conversely DA EQUI all-empty FDR=0 has no observed selection power.
- No scenario pool, no K dummy replicate inflation, no method replacement for unsupported cases, and no retrospective K choice or held-out tuning. Apparent Python/R algorithm identity is NOT established.

## Artifact contracts, reproduction and checks

Research-only adapter, worker, and CLI live in src/llm_evaluation/trex_finite_k.py, trex_finite_k_worker.py, and trex_finite_k_cli.py. They reuse existing generate_linear_dataset, score_generated_dataset and aggregate_generated_datasets from trex_known_truth.py. Failures, timeouts, unavailable optional packages, unsupported methods and empty successful selections are distinct typed records.

The authoritative development output for the corrected implementation is research/trex47-dev-002-output/{configuration.json,records.jsonl,summary.json,manifest.json}. Historical dev-001 files remain earlier development evidence, not FINAL. Manifests include source module SHA-256, config/artifact SHA-256, PyPI hashes, exact runtime and package versions. They also report the checkout HEAD at execution **before Issue #47 files were committed**; this HEAD does *not* contain the new source. Treat the actual source-module digests as the execution-code identity. Cross-platform bitwise identity is not asserted.

A reproducible local optional environment can be created without changing base pyproject or uv.lock:

~~~sh
uv venv --python 3.11.13 .venv-trex47
uv pip install --python .venv-trex47/bin/python \
  'trexselector==1.0.0' 'numpy==2.4.6' 'scipy==1.17.1' \
  'scikit-learn==1.9.1' 'pydantic==2.14.1'
PYTHONPATH=src .venv-trex47/bin/python -m llm_evaluation.trex_finite_k_cli \
  --config research/trex47-dev-002.json --output-dir /tmp/trex47-unique-replay
~~~

The destination must not already exist; runs do not overwrite previous artifacts. Development tests independently reject all-null TPR=0, false zero-selection failures, incorrect pooled V/R, corrupted known support, shuffled response, duplicate dataset seeds, native off-by-one and invalid indices, K override, and source-input mutation. They use fixed numerical controls plus bounded Hypothesis invariants.

## Separate prospective FINAL holdout (NOT RUN)

A scientific director may later approve a **different protocol**, frozen before any holdout execution, with candidate namespace trex47-confirmatory-holdout-v1, distinct seed base 910047001 and analysis seed 910047019 (not used here), predeclared n/p/rho/noise/group/reference strata, justified power/sample size and FDR near-q interval/decision rule, matched comparator budgets, unsupported/failure policy and subgroup reporting. The holdout has **not been generated, run, inspected or tuned**. E-004 FINAL FAILED and D-019 remain unaffected.

## Development disposition

- **Reject universal original T-Rex structural-support FDR promotion.** Independent-Gaussian observations are promising but correlation-stress results are adverse and do not establish a theorem violation.
- **Do not promote DA EQUI for useful feature selection** on tested equicorrelated conditions: FDR=0 because power and selections were zero.
- **Retain DA BT only as a conditional candidate** requiring a validated dependency/group design, calibration, adequate dataset counts and prospective confirmatory testing. No production application or science-canonical promotion is authorized.
