# canonical-sync Standalone Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the verified `canonical-sync` package from `llm-evaluation` into a standalone public `bio-comp/canonical-sync` repository, add the already-identified Prompt Research raw-file and shared ADC-CLI capabilities, then migrate `llm-evaluation` and the duplicated Prompt Research ADC helper to the standalone release without changing project-specific authority semantics.

**Architecture:** Preserve the embedded `canonical_sync` API as the extraction baseline. The standalone package owns provider-neutral acquisition/snapshot/publication mechanics and operator ADC tooling only; consumer authority selection, reconciliation/scientific policy, mirror semantics, and publication authorization remain in their projects. Release `v0.1.0` as the extraction-equivalent baseline and `v0.2.0` after the additive raw-file representation and ADC CLI are verified.

**Tech Stack:** Python >=3.11, Pydantic 2.14.1, uv, hatchling, pytest, Google Drive API/google-auth optional dependency, Git/GitHub Actions.

**Spec:** `docs/design/2026-10-26-canonical-sync-design.md`; second-consumer requirements in `docs/canonical-sync-prlab-compatibility.md`; live Drive authority is `llm_evaluation` D-016 and Prompt Research D-019.

## Global Constraints

- `canonical-sync` owns verified mechanics only; no `llm_evaluation` or Prompt Research authority/reconciliation/scientific semantics enter the package.
- Preserve exact source/state/observation identity semantics and canonical serialization behavior.
- Public code must contain no Drive object IDs, private canonical bodies, credential paths/tokens, or private snapshots.
- The raw stored-file representation must never silently substitute export bytes for download bytes or vice versa.
- ADC tooling remains operator/deployment tooling; core acquisition consumes already-valid credentials.
- Consumer migrations must pin an immutable standalone release/tag and must not retain an active nested package copy.

## Review Focus

- Native Google Doc export and raw stored-file download remain distinct representation contracts.
- Provider checksum validation applies only when the stored representation exposes a meaningful checksum and must fail closed on mismatch.
- Existing `llm_evaluation` imports resolve to the external package after removal of the workspace member.
- The ADC CLI preserves exact-scope, atomic replacement, cross-platform path, and secret-sanitization behavior from the two verified duplicated scripts.
- No project-specific policy or private identifiers leak into the standalone repository.

---

### Task 1: Standalone extraction baseline

**Files:**
- Create standalone repo `/Users/mikeh/repos/canonical-sync/` from `packages/canonical-sync/{src,tests,pyproject.toml}`.
- Create standalone `README.md`, `.gitignore`, `.github/workflows/ci.yml`, and development/test configuration in `pyproject.toml`.
- Keep package source initially behavior-equivalent to embedded `0.1.0`.

**Interfaces:**
- Consumes: current embedded `canonical_sync` public API and its 159-test suite.
- Produces: installable standalone `canonical-sync==0.1.0` with the same import namespace/API and passing tests/build.

- [ ] Copy the package source/tests without semantic edits and establish standalone project metadata.
- [ ] Run the extracted test suite and verify the same 159 tests pass.
- [ ] Build wheel/sdist and install the wheel into a clean temporary environment; import `canonical_sync` and verify package metadata.
- [ ] Initialize Git history and commit the extraction baseline.

### Task 2: Add the second-consumer capabilities under TDD

**Files:**
- Modify: `src/canonical_sync/google_drive.py` and exports as required.
- Create: `src/canonical_sync/adc.py` (shared operator implementation) and console entry point.
- Move/adapt tests: raw-file Drive tests plus the existing ADC-helper regression matrix into standalone `tests/`.
- Modify: standalone `pyproject.toml` optional dependencies and script entry point; bump version to `0.2.0`.

**Interfaces:**
- Produces: explicit versioned raw stored-file representation and `canonical-sync refresh-google-drive-adc ...` operator command.

- [ ] Write failing tests for exact raw download bytes, checksum match/mismatch, unsupported/native-export separation, and response-type validation.
- [ ] Run the focused tests and verify RED for missing raw-file support.
- [ ] Implement the minimal explicit raw stored-file representation in `GoogleDriveTransport.read()` without fallback between raw and export representations.
- [ ] Run focused and full tests GREEN.
- [ ] Move the ADC helper regression tests first; verify RED because `canonical_sync.adc`/CLI do not exist.
- [ ] Move the verified helper implementation into `canonical_sync.adc`, add the console subcommand/entry point, and preserve secret-sanitized behavior.
- [ ] Run the full standalone suite, lint/type checks, build/install smoke, and commit `0.2.0`.

### Task 3: Publish the standalone repository and immutable release refs

**Files:** standalone Git repository only.

**Interfaces:**
- Produces: public `bio-comp/canonical-sync`, immutable tags `v0.1.0` and `v0.2.0`, and remote readback of exact commits.

- [ ] Create public `bio-comp/canonical-sync` from the verified local repository and push `main`.
- [ ] Tag the extraction baseline `v0.1.0` and additive shared-capability commit `v0.2.0`; push tags.
- [ ] Read back repository visibility/default branch/tag commit identities from GitHub.
- [ ] Verify CI for the published head if available; do not claim release verification without readback.

### Task 4: Migrate `llm-evaluation` from workspace package to standalone dependency

**Files:**
- Modify: root `pyproject.toml`, `uv.lock`, canonical-sync docs/runbook references and tests that invoked the repo-local ADC helper.
- Delete: `packages/canonical-sync/**`, `scripts/refresh_google_drive_adc.py`, its duplicate helper test after equivalent standalone coverage is bound.

**Interfaces:**
- Consumes: standalone `canonical-sync` `v0.2.0`.
- Produces: `llm-evaluation` imports external `canonical_sync`; no nested package/workspace source remains.

- [ ] Add a migration test/check that fails while the root still resolves `canonical-sync` from `packages/canonical-sync`.
- [ ] Replace uv workspace source with immutable Git/tag source, regenerate lockfile, remove nested package and repo-local helper.
- [ ] Update local-auth documentation to call the shared CLI.
- [ ] Run targeted canonical-sync/reconciliation tests, then full `llm-evaluation` test/lint/type/build gates.
- [ ] Commit the consumer migration and open a PR; do not merge without verified remote CI/readback.

### Task 5: Retire the Prompt Research duplicate ADC helper without prematurely replacing prlab acquisition semantics

**Files:**
- Prompt Research: modify `pyproject.toml`, `uv.lock`, local-auth runbook/test wiring; delete `scripts/refresh_google_drive_adc.py` after shared CLI parity is proven.

**Interfaces:**
- Consumes: standalone `canonical-sync` `v0.2.0` operator CLI.
- Produces: no duplicated ADC helper implementation; current `prlab` acquisition remains the behavioral comparator for a later full acquisition migration.

- [ ] Add dependency/CLI parity test that initially fails because Prompt Research still targets the repo-local helper.
- [ ] Pin `canonical-sync` `v0.2.0`, point runbook/verification at the shared CLI, and delete the duplicate helper/test implementation only after equivalent standalone tests are established.
- [ ] Run Prompt Research deterministic gates relevant to local auth plus its full suite if practical; record any unrelated failures explicitly.
- [ ] Commit and open a separate Prompt Research PR; do not claim full prlab acquisition migration yet.

### Task 6: Reconcile durable state only after implementation evidence exists

**Files:** live Drive `llm_evaluation` D-016 / README as appropriate and Prompt Research D-019 only after the corresponding repository states are verified.

**Interfaces:**
- Produces: durable state that distinguishes standalone extraction from consumer migration and does not claim completion before merges/readback.

- [ ] Verify standalone repo/tag identities and both consumer PR/CI states from authoritative GitHub/Git evidence.
- [ ] Update the owning canonical objects in place to record the extraction state and remaining Prompt Research full-acquisition migration obligation.
- [ ] Re-read the exact Drive objects and verify persistence before reporting durable completion.
