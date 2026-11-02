# E-004 Drift-Surveillance Phase A

E-004 is a known-truth synthetic laboratory for a narrow drift-surveillance question: at a common 52-look family alert budget, can a typed layer of case-mix distribution monitoring plus case-mix-adjusted process monitoring distinguish composition shift from conditional performance degradation better than deliberately weaker alternatives?

It is an instrument-validation experiment, not a production monitoring policy. It does not define a universal PSI cutoff, a universal SPC chart, a release threshold, or a clinically/business-material drift level.

## Frozen design

The exact public design is `fixtures/e004_drift_surveillance_phase_a_design.json`. The implementation rejects design drift through canonical hashes and strict versioned records.

The core generator uses 52 weekly looks, an independently sampled labeled baseline of 4,000 cases, weekly n=200, baseline stratum-B prevalence 0.20, and baseline conditional error rates A=0.05 / B=0.20. The primary change point is week 13. Four primary truth states are retained separately: stable/stable, distribution-only (B prevalence 0.20 → 0.60), process-only (both conditional error rates +0.05), and both.

Stress controls cover gradual degradation, a four-week temporary excursion, denominator cycling 50/100/200/400, beta-binomial week-level overdispersion at ICC 0.05, a B-only localized degradation, and fixed-threshold PSI null-size behavior at weekly n=100/200/1000.

## Instruments and policies

Distribution channels are two-bin PSI with pseudocount 0.5 and a two-sample Pearson 2×2 statistic. PSI=0.10 is retained only as a fixed-threshold negative control; matched comparisons use simulation-calibrated thresholds.

Process channels use one-sided standardized worsening-error residuals. Raw aggregate EWMA deliberately remains vulnerable to case-mix confounding. Adjusted EWMA, adjusted CUSUM (k=0.5), and adjusted Shewhart use finite-baseline stratum-specific error estimates and current observed stratum membership. A fixed-average-n Shewhart chart is a denominator-misspecification negative control.

The seven matched-budget policies are PSI-only, Pearson-only, adjusted-EWMA-only, adjusted-CUSUM-only, PSI+raw-EWMA, PSI+adjusted-EWMA, and PSI+adjusted-CUSUM. Layered outputs preserve typed `none`, `distribution`, `process`, and `both` states; they do not collapse the two estimands into one score.

## Calibration and randomness

DEVELOPMENT uses exactly 20,000 stable/stable paths. FINAL uses exactly 20,000 paths per frozen scenario. DEVELOPMENT and FINAL have disjoint deterministic seed namespaces derived from the exact design identity, scenario, replicate index, and purpose.

All matched policies target a 0.05 family alert probability over the 52-look horizon. Single-channel thresholds are selected from DEVELOPMENT maximum statistics using strict `statistic > threshold` alarms and the smallest observed threshold whose exceedance fraction is at or below the target. Layered policies use deterministic equal-component-tail grid search and retain the largest common component tail allocation whose observed union family alert is at or below 0.05. FINAL never recalibrates.

## Evidence records

Calibration and FINAL evidence use versioned immutable records. Quantitative persisted values are decimal text rather than binary floating-point scalars so canonical record identities remain exact. Finalized records bind the exact design, frozen-design record, Git software identity, package-source digest, Python runtime, immutable readback, and—during FINAL—the exact finalized DEVELOPMENT calibration identity.

FINAL retains gate-level outcomes plus scenario×policy operating characteristics: horizon alarm probability with Wilson 95% interval, pre-change alarm probability, post-change detection probability with Wilson interval, missed detection, first-alarm timing, conditional post-change delay, and alarm burden.

## Executable gates

The acceptance evaluator enforces the frozen seven checks: held-out matched-budget family false alarms in [0.04, 0.06]; sample-size sensitivity of fixed PSI=0.10 while calibrated PSI remains budgeted; pure case-mix shift causing high raw-process but low adjusted-process alarms; adjusted typed-state accuracy and macro advantage over the raw layer; denominator-misspecification failure; overdispersion false-alarm inflation of an iid-calibrated EWMA; and stronger localized B-stratum evidence than the aggregate channel in the localized-degradation control.

These checks are falsification controls. A pass does not establish robustness to autocorrelation, seasonality, evaluator/reference changes, prompt/harness changes, multivariate joint-only shift, sparse high-cardinality strata, or production multiple-stream escalation.

## Commands

A smoke run is diagnostic only and never publishes an evidence artifact:

```bash
uv run --frozen --all-packages llm-evaluation-drift-lab --phase smoke --smoke-paths 20
```

After repository reconciliation, exact DEVELOPMENT is:

```bash
uv run --frozen --all-packages llm-evaluation-drift-lab \
  --phase development \
  --frozen-design fixtures/e004_drift_surveillance_phase_a_design.json \
  --output runs/e004-drift-development.json
```

FINAL consumes that immutable calibration directly; it does not rerun DEVELOPMENT or retune thresholds:

```bash
uv run --frozen --all-packages llm-evaluation-drift-lab \
  --phase final \
  --frozen-design fixtures/e004_drift_surveillance_phase_a_design.json \
  --calibration runs/e004-drift-development.json \
  --output runs/e004-drift-final.json
```

An acceptance failure is still an evidence result: FINAL is finalized first, the failures are printed, and the CLI exits nonzero. The failed artifact must not be replaced by a retuned run.
