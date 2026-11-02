# E-004 Drift-Surveillance Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the frozen E-004 Phase-A known-truth drift-surveillance laboratory exactly enough to support deterministic DEVELOPMENT calibration and later FINAL execution after repository reconciliation.

**Architecture:** Keep E-004 as an independent, standard-library project vertical slice parallel to the metric lab: frozen design/records, pure detector mathematics, deterministic generator/runner, acceptance evaluator, and CLI. Distribution and process detectors remain separate typed channels; layered policies combine alarm state but do not collapse their estimands. Reuse existing project software-identity/canonical-record mechanisms without changing `packages/canonical-sync`.

**Tech Stack:** Python 3.11.13, standard library, Pydantic/canonical-sync records already in the repo, pytest, Ruff, mypy.

**Spec:** Live canonical `EXPERIMENTS.md`, section `E-004 — AR-11 drift-surveillance Phase A implementation freeze (2026-11-01)`, verified before implementation.

## Global Constraints

- Branch starts from exact merged `main` `24f5b9f4cc824b1b35839bd087cf7d331bad1f70`.
- DEVELOPMENT and FINAL use disjoint purpose-separated deterministic seed namespaces.
- No E-004 DEVELOPMENT or FINAL scientific run before repository reconciliation completes; tests may use tiny deterministic grids.
- No subject-model/API calls, private production/customer data, `packages/canonical-sync`, ADC-helper, or Prompt Research changes.
- Matched policies target family false-alarm probability 0.05 over 52 looks; alarms use strict `statistic > threshold` semantics.
- Calibration thresholds and all scientific results are immutable/readback-verifiable and software-identity bound.

## Review Focus

- Threshold ties/order statistics must not accidentally exceed the declared family-alert budget.
- Pure case-mix shift must inflate raw aggregate process monitoring while adjusted residual monitoring remains centered under the oracle generator.
- Variable denominators and beta-binomial overdispersion must preserve their intended negative-control failure modes.
- Typed `none/distribution/process/both` outputs must preserve separate distribution/process estimands and truth-state attribution.
- FINAL records must reject design/software/threshold/budget mismatch and overwrite attempts.

---

### Task 1: Pure detector mathematics and calibration primitives

**Files:**
- Create: `src/llm_evaluation/drift_lab_detectors.py`
- Create: `tests/test_drift_lab_detectors.py`

**Interfaces:**
- Produces pure PSI, Pearson 2x2, raw/adjusted residual, EWMA, CUSUM, Shewhart, threshold calibration, layered threshold calibration, and typed-alarm helpers.

- [ ] Write RED tests for exact formulas, recursion, confounding counterexample, strict `>` threshold tie semantics, layered common-tail selection, and typed states.
- [ ] Run the focused test file and confirm failures are due to missing implementation.
- [ ] Implement the minimum pure functions required by those tests.
- [ ] Run focused tests to GREEN and refactor without changing behavior.
- [ ] Commit.

### Task 2: Frozen design and deterministic generator

**Files:**
- Create: `src/llm_evaluation/drift_lab_design.py`
- Create: `src/llm_evaluation/drift_lab_runner.py`
- Create: `fixtures/e004_drift_surveillance_phase_a_design.json`
- Create: `tests/test_drift_lab_design.py`
- Create: `tests/test_drift_lab_runner.py`

**Interfaces:**
- Consumes Task-1 detector primitives.
- Produces strict versioned frozen design, deterministic seed derivation, primary/stress scenario generator, baseline estimation, path statistics, and tiny-grid execution.

- [ ] Write RED design/seed/generator tests, including DEVELOPMENT/FINAL disjointness, exact primary truth states, denominator cycle, overdispersion mean/ICC construction, and deterministic reproducibility.
- [ ] Implement strict design models and canonical frozen fixture.
- [ ] Implement generator/runner sufficient for tiny deterministic simulation and calibration.
- [ ] Run focused tests to GREEN.
- [ ] Commit.

### Task 3: Durable run records, acceptance evaluator, and CLI

**Files:**
- Create: `src/llm_evaluation/drift_lab_records.py`
- Create: `src/llm_evaluation/drift_lab_acceptance.py`
- Create: `src/llm_evaluation/drift_lab_cli.py`
- Create: `tests/test_drift_lab_records.py`
- Create: `tests/test_drift_lab_acceptance.py`
- Create: `tests/test_drift_lab_cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes frozen design and runner outputs.
- Produces versioned immutable calibration/run records, software/design identity verification, executable gates 1–7, installed `llm-evaluation-drift-lab` CLI, and fail-closed readback.

- [ ] Write RED tests for schema/identity/readback/no-overwrite and one pass/fail fixture per acceptance gate.
- [ ] Implement minimal record/finalization/readback code and acceptance evaluator.
- [ ] Add CLI and tiny DEVELOPMENT/FINAL test mode without changing frozen scientific budgets.
- [ ] Run focused tests to GREEN.
- [ ] Commit.

### Task 4: Public documentation and repository verification

**Files:**
- Create: `docs/e004-drift-surveillance-phase-a.md`
- Modify/add tests only if verification exposes a real defect.

**Interfaces:**
- Documents exact frozen design, commands, outputs, negative controls, interpretation boundaries, and repo-reconciliation gate.

- [ ] Document the exact public-safe design and execution contract without copying private Drive narrative/IDs.
- [ ] Run lock/sync/Ruff/mypy/non-live full branch-coverage suite and >=80% coverage gate.
- [ ] Build both wheels in isolated environments and exercise the installed CLI.
- [ ] Run a tiny deterministic smoke run only; do not run the 20,000-path scientific DEVELOPMENT/FINAL.
- [ ] Verify `git diff --check`, repository cleanliness after commit, and no changes to `packages/canonical-sync`/ADC helper.
- [ ] Commit.

### Task 5: GitHub reconciliation and scientific execution gate

**Files:** no new implementation unless verification finds a defect.

- [ ] Push branch and open PR.
- [ ] Verify exact-head Public CI required job.
- [ ] Merge only after review/verification; verify post-merge push/main Public CI and authoritative Git readback of the approved E-004 projection.
- [ ] Persist/read back an E-004 partial completion receipt; mechanically prove global checkpoint bytes did not move.
- [ ] Update/reread Drive repository-reconciliation state only after exact closure evidence.
- [ ] Only then execute frozen 20,000-path DEVELOPMENT calibration and 20,000-path-per-scenario FINAL, immutably finalize/read back, evaluate gates, and persist scientific result/interpretation.
