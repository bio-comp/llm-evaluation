# D-016 Closure Binding Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a newer Drive canonical snapshot from being marked reconciled by an older/partial Git transaction, preserve multiple public obligations explicitly, and recover the existing semantically invalid durable state without erasing history.

**Architecture:** Keep generic acquisition/Git verification in `canonical-sync`; repair only the `llm_evaluation` project reconciliation layer. A public-change review must provide the exact approved `PublicationTree` (including the canonical policy fixture and every approved implementation/documentation file) plus a non-empty set of public-safe obligation IDs. Operational closure verifies that exact tree at the merged commit and carries the obligation set into durable completion evidence. Legacy durable v1 state is fail-closed and may only be explicitly superseded into an immutable local history file before a corrected v2 state is written.

**Tech Stack:** Python 3.11.13, Pydantic v2, pytest, existing `canonical_sync` records/GitPublisher, GitHub closure adapter.

**Spec:** Live Drive `DECISIONS.md` D-016 plus `docs/canonical-sync.md` and `docs/design/2026-10-26-canonical-sync-design.md`.

## Global Constraints

- Google Drive remains the sole mutable methodological authority.
- `canonical-sync` owns verified mechanics only; project reconciliation meaning stays in `llm_evaluation`.
- No private Drive IDs/content enter Git.
- No E-002 model sampling.
- Preserve the existing legacy state bytes before removing them from the active state path.
- Do not change Prompt Research or its helper implementation in this repair.

## Review Focus

- Public-change review with only the generic policy fixture must fail when the obligation needs additional files.
- A publication tree may contain the policy fixture plus additional approved files and must be verified exactly at merged `main`.
- Multiple outstanding obligation IDs must survive REQUIRED -> COMPLETE -> durable persistence without being dropped or reordered ambiguously.
- Legacy durable v1 state must be rejected as current state and explicitly archived byte-for-byte before active-state removal.
- A stale/partial PR that lacks any approved-tree file must fail Git readback even if PR/CI metadata and the generic policy fixture are valid.

---

### Task 1: Bind REQUIRED reconciliation to exact approved projection and obligation set

**Files:**
- Modify: `src/llm_evaluation/canonical_reconciliation.py`
- Modify: `src/llm_evaluation/canonical_github.py`
- Modify: `tests/test_canonical_reconciliation.py`
- Modify: `tests/test_canonical_github_closure.py`

**Interfaces:**
- `ReviewedCanonicalImpact` gains `publication_tree: PublicationTree | None` and `obligation_ids: tuple[str, ...]`.
- `ReconciliationDispositionRecord` REQUIRED/COMPLETE carries normalized `obligation_ids`.
- `VerifiedPublicationReceiptRecord` carries the same obligation set.
- `complete_reconciliation_with_github()` requires an explicit `publication_tree`; `public_contract` may independently cross-check the fixture but may no longer synthesize the tree.

- [ ] Write failing tests proving public-change review without an explicit tree is rejected, an exact multi-file tree is accepted, obligation IDs are mandatory and preserved, and a mismatched receipt obligation set cannot complete.
- [ ] Run the targeted tests and verify RED failures are caused by the missing contract.
- [ ] Implement the minimal schema/validation/reconciliation changes; bump durable record schema versions that changed shape.
- [ ] Update GitHub projection resolution so the exact supplied tree must contain a canonical policy fixture matching the required contract hash, while allowing additional approved files.
- [ ] Run targeted tests to GREEN.
- [ ] Commit the task.

### Task 2: Add explicit safe recovery for legacy durable state

**Files:**
- Modify: `src/llm_evaluation/canonical_reconciliation.py`
- Modify: `tests/test_canonical_reconciliation.py`

**Interfaces:**
- `load_durable_reconciliation_state()` rejects legacy schema v1 as active authority with a specific recovery-required error.
- `supersede_legacy_reconciliation_state(*, expected_raw_sha256: str, path: Path | str | None = None) -> Path` archives exact legacy bytes under a non-symlink `reconciliation-history/` file named by raw SHA-256, verifies readback, then removes only the active legacy path.

- [ ] Write failing tests for legacy rejection, expected-hash mismatch, byte-for-byte archive/readback, symlink rejection, and no active-state deletion before archive verification.
- [ ] Run targeted tests and verify RED.
- [ ] Implement the minimal recovery path with fsync/readback and fail-closed ordering.
- [ ] Run targeted tests to GREEN.
- [ ] Commit the task.

### Task 3: Update the public executable contract and documentation

**Files:**
- Modify: `fixtures/reconciliation_policy_contract.json`
- Modify: `docs/canonical-sync.md`
- Modify: `docs/design/2026-10-26-canonical-sync-design.md`
- Modify: `docs/plans/2026-10-26-canonical-sync-incubation.md`
- Modify tests asserting schema/fixture versions as required.

**Interfaces:**
- Policy contract records exact-approved-publication-tree and obligation-set matching as closure requirements.
- Documentation states that the one-file policy fixture is a vertical-slice component, not a substitute for the exact project-approved implementation projection.

- [ ] Write/update the contract test first so the current fixture fails against the new executable contract.
- [ ] Update the contract schema/fixture and docs narrowly.
- [ ] Run contract/reconciliation tests to GREEN.
- [ ] Commit the task.

### Task 4: Full verification and operational D-016 recovery/closure

**Files:** no code changes unless a verified defect is found through RED->GREEN.

- [ ] Run lock, Ruff format/check, strict mypy, targeted reconciliation/GitHub tests, full pytest/coverage, build/install/finalization gates.
- [ ] Push `fix/d016-closure-binding`, open a PR, and verify PR CI.
- [ ] Merge only after required review/CI gates; verify post-merge `Public CI` `push/main` `verify` on the exact merge commit.
- [ ] On merged code, explicitly supersede the known legacy state using its previously observed raw SHA-256, verify immutable history bytes, reacquire the live four-object Drive snapshot, construct the exact approved PR #10+repair cumulative publication tree and obligation set, execute authoritative GitHub/Git closure, persist v2 state, reread, and mechanically compare snapshot/tree/obligation/PR/CI/Git identities.
- [ ] Update live Drive D-016 status to COMPLETE only if all operational checks pass; reread Drive to verify persistence.

### Task 5: Reconcile existing Task-1 + metric-lab branch

**Files:** existing `feat/integrate-task1-runtime` changes only; do not reimplement by default.

- [ ] Preserve historical `1e0fc73` test/runtime receipts as the before-baseline.
- [ ] Rebase/reconcile `feat/integrate-task1-runtime` onto corrected `main`; resolve conflicts semantically.
- [ ] Rerun deterministic positive runtime preflight and negative controls, metric-laboratory tests/microprobes, full Ruff/mypy/pytest/coverage/build/finalization gates.
- [ ] Compare new outputs with the historical `1e0fc73` baseline and investigate any material difference rather than normalizing it away.
- [ ] Push/open PR, verify CI, merge if all gates pass, and verify post-merge main.
- [ ] Do not start E-002 subject-model sampling.
