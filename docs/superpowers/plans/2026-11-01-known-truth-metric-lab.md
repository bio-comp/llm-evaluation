# Known-Truth Metric / Reference-Error Laboratory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Drive-authorized known-truth metric/reference-error laboratory without subject-model calls, with exact estimator definitions, discriminating controls, auditable target/reference arms, development/final separation, and immutable result evidence.

**Architecture:** Preserve `metric_laboratory.py` as the mathematical primitive layer, correct its currently ambiguous or misnamed estimators, and add focused design/runner/record modules rather than expanding one monolith. A versioned design record defines phase, seed space, targets, arms, thresholds, tie policy, interval policy, and pass criteria; the runner evaluates known-truth cells and produces target-indexed summaries. DEV may change the candidate design. FINAL is accepted only from an exact frozen design identity and disjoint final seed space.

**Tech Stack:** Python 3.11.13, Pydantic v2, stdlib deterministic RNG/statistics where adequate, existing `llm_evaluation.records.software_identity`, pytest, Ruff, mypy, uv.

**Spec:** Live Google Drive `EXPERIMENTS.md` candidate known-truth metric/reference-error laboratory plus `Research/METRIC_VALIDITY_AND_MEASUREMENT.md`; Git foundation merged by PR #13 at `88d456ba47b554ec96b7318621209b1a31b49641`.

## Global Constraints

- No E-002 subject-model calls.
- Exact metric target, reference, independent unit, weighting, tie/threshold convention, uncertainty method, data-generation parameters, and seeds must remain explicit.
- 2,000 independent Monte Carlo replicates per frozen final cell; development runs may be smaller and cannot be promoted to final evidence.
- Development and final seed spaces must be disjoint and mechanically enforced.
- Reference noise and genuine ambiguity/rater disagreement are separate arms and separate estimands.
- Fixed-case and population-sampling uncertainty remain separate outputs.
- No universal metric/estimator superiority claim follows from the lab.
- Full final design is frozen in Drive before final-grid execution; no post-final tuning.

## Review Focus

1. **Calibration estimator identity:** the existing `debiased_ece()` is neither Kumar et al.'s exact L2 debiasing estimator nor their separate L1 normal-resampling correction; tests must reject conflation.
2. **Confidence ties and 0.5 decision ties:** equal-mass binning and classification thresholds must not depend accidentally on stable input order or Python bankers' rounding.
3. **Balanced probability population:** `n` not divisible by seven must not always overweight the same probability values; development/final seed policy must preserve auditable balance.
4. **Reference/target confusion:** clean truth, flipped noisy labels, rater-distribution targets, soft-label targets, and majority consensus must never share an unlabeled metric output.
5. **Leakage from DEV to FINAL:** final execution must reject an unfrozen design, changed design hash, reused development seeds, or post-freeze parameter mutation.

---

### Task 1: Correct estimator definitions and deterministic conventions

**Files:**
- Modify: `src/llm_evaluation/metric_laboratory.py`
- Modify: `tests/test_metric_laboratory.py`

**Interfaces:**
- Produces exact L2 squared/debiased calibration estimators, explicitly named candidate L1 debiasing if retained, tie-preserving equal-mass ECE, and explicit binary decision threshold behavior.

- [ ] Write failing tests against independent formulas and adversarial ties.
- [ ] Verify RED for current `debiased_ece`, equal-mass tied forecasts, and `q == 0.5` classification behavior.
- [ ] Implement minimal corrected estimators/conventions with precise docstrings and no hidden global RNG.
- [ ] Verify targeted tests and existing metric tests pass.
- [ ] Commit.

### Task 2: Versioned lab design, phases, and deterministic seed policy

**Files:**
- Create: `src/llm_evaluation/metric_lab_design.py`
- Create: `tests/test_metric_lab_design.py`

**Interfaces:**
- Produces `MetricLabDesign`, `MetricLabPhase`, arm configs, deterministic balanced probability schedule, `development_seed(...)`, `final_seed(...)`, canonical design hash, and freeze validation.

- [ ] Write failing tests for exact base-grid inventory, balance, seed determinism/disjointness, phase restrictions, and freeze mutation rejection.
- [ ] Verify RED.
- [ ] Implement minimal design records and seed derivation.
- [ ] Verify targeted tests.
- [ ] Commit.

### Task 3: Target/reference arms and known-truth oracles

**Files:**
- Create: `src/llm_evaluation/metric_lab_targets.py`
- Create: `tests/test_metric_lab_targets.py`

**Interfaces:**
- Produces target-indexed clean/noisy-reference, ambiguity/rater, prevalence-reweighting, subgroup-gap, and fixed-case/population datasets plus exact analytic target quantities where available.

- [ ] Write failing tests for clean-vs-noisy ranking reversal, rater disagreement not being labeled flip noise, consensus-vs-rater-distribution target differences, prevalence shift, subgroup cancellation, and fixed-vs-population sampling units.
- [ ] Verify RED.
- [ ] Implement target generators/oracles with explicit estimand labels.
- [ ] Verify targeted tests.
- [ ] Commit.

### Task 4: Replicate evaluation, uncertainty, risk-coverage, and Monte Carlo summaries

**Files:**
- Create: `src/llm_evaluation/metric_lab_runner.py`
- Create: `tests/test_metric_lab_runner.py`
- Modify: `src/llm_evaluation/metric_laboratory.py` only for reusable primitives if required by tests.

**Interfaces:**
- Produces per-replicate target-indexed results and cell summaries with bias, RMSE, MC SE, interval coverage/width, reliability counts, risk-coverage operational points, AURC controls, clustered-vs-naive inference, ranking reversals, decision errors, subgroup failures, and realized budget.

- [ ] Write failing tests on analytically known cells and known-bad/counterfactual controls.
- [ ] Verify RED.
- [ ] Implement minimal runner and summary calculations.
- [ ] Mechanically compare prefix-risk AURC with official-code negative control on the known two-item counterexample.
- [ ] Verify targeted tests.
- [ ] Commit.

### Task 5: Immutable lab records and DEV/FINAL execution gate

**Files:**
- Create: `src/llm_evaluation/metric_lab_records.py`
- Create: `src/llm_evaluation/metric_lab_cli.py`
- Create: `tests/test_metric_lab_records.py`
- Create: `tests/test_metric_lab_cli.py`
- Modify: `pyproject.toml` only if a CLI entry point is warranted by the tests/design.

**Interfaces:**
- Produces versioned immutable diagnostic records bound to design hash, phase, seeds, software identity, package-source digest, config versions, counts/budgets, and exact results; second write rejects.

- [ ] Write failing tests for unfrozen-final rejection, design-hash mismatch, DEV seed reuse, software identity, canonical readback, and atomic second-write rejection.
- [ ] Verify RED.
- [ ] Implement records/finalization/CLI.
- [ ] Verify targeted tests and installed-entrypoint behavior if added.
- [ ] Commit.

### Task 6: Run and assess a bounded DEV grid

**Files:**
- No canonical mutation yet.
- DEV artifact under `/private/tmp` or another non-symlink local working path; do not commit generated results unless a deliberately public-safe compact fixture is justified.

**Interfaces:**
- Consumes candidate design and outputs development diagnostics only.

- [ ] Benchmark representative cells and choose a bounded DEV subset/repetition count sufficient to falsify obvious estimator/coverage defects.
- [ ] Run deterministic DEV grid and save immutable DEV artifact.
- [ ] Check mathematical truth controls, estimator bias/RMSE, uncertainty coverage, target/reference reversals, subgroup cancellation, clustering effect, and budget accounting.
- [ ] Correct defects using TDD; rerun DEV after each scientifically material change.
- [ ] Decide exact candidate formulas/intervals/ties/arm parameters/final seed policy and pass criteria from the development evidence.
- [ ] Commit any justified code changes.

### Task 7: Freeze the final design in Drive and reconcile the public implementation

**Files:**
- Live Drive `EXPERIMENTS.md` authoritative object.
- Potential Git public-safe design fixture/doc only if the frozen design adds public executable requirements not already represented.

**Interfaces:**
- Produces an exact Drive-frozen final design identity and, if required, a matching Git reconciliation obligation.

- [ ] Reread live `EXPERIMENTS.md` and relevant research note immediately before mutation.
- [ ] Update the same `EXPERIMENTS.md` object with exact frozen estimator versions, arms, thresholds/ties, interval method, seed derivation/spaces, final cell inventory, 2,000 replicate rule, pass/fail criteria, and explicit no-E-002 boundary.
- [ ] Reread and verify persistence.
- [ ] Run D-016 classification for the changed Drive source state; if public/executable implications exist, reconcile them through Git before final execution rather than silently treating Drive and code as equivalent.
- [ ] Persist/read back reconciliation state under the applicable disposition contract.

### Task 8: Execute untouched FINAL grid and interpret results

**Files:**
- Immutable finalized result artifact outside Git working tree unless a compact public-safe artifact is explicitly justified.

**Interfaces:**
- Consumes only the exact frozen design; produces final known-truth lab evidence.

- [ ] Verify exact frozen Drive design identity and matching Git implementation identity.
- [ ] Run the full final grid at 2,000 independent replicates/cell with final-only seeds.
- [ ] Finalize atomically and reread/verify artifact identity and software/design bindings.
- [ ] Evaluate pre-frozen pass/fail criteria without changing formulas or exclusions.
- [ ] Record limitations and failures; do not promote an estimator merely because it performs well on this grid.
- [ ] Update Drive experiment status/results only to the level supported by the final evidence, reread, and reconcile resulting public implications if any.

### Task 9: Full verification, review, and integration

**Files:** all changed code/tests/docs.

- [ ] Run `uv lock --check`, Ruff, strict mypy, complete credential-free pytest/branch-coverage gate, package build/install gates, and artifact finalization/readback controls.
- [ ] Scan public diff for credentials/private paths/data and verify no generated private artifacts are tracked.
- [ ] Push branch and open a PR with exact-head evidence.
- [ ] Require exact-head `Public CI / verify` success, merge with expected-head lease, and require exact post-merge `push/main` `Public CI / verify` success.
- [ ] Advance local `main`, remove merged feature branch only after clean/ancestor checks, and leave E-002 subject-model sampling untouched.
