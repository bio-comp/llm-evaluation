# Local-first Python engineering gates (Issue #40)

## Scope and authority

This repository keeps GitHub Public CI as the independent required exact-merge gate. Local checks reduce avoidable CI failures but do not authorize a merge. The first implementation slice enforces the existing Python 3.11.13 main baseline; Issue #41 / PR #42 separately owns promotion to Python >=3.12 with development on 3.14.10. The gate reads the declared package minimum to select mypy's compatibility target.

No scientific-method, persisted-artifact, data schema, or historical Python 3.11 identity is changed by this slice. GitHub Issues #36–#39 remain the owners of historical assertions, formatting, strict typing, and Pydantic/settings work.

## Install and activate in a standalone clone

1. Install the project uv version and confirm the lock with: uv --version; uv lock --check; uv sync --locked --group dev --extra drive --extra local-auth.
2. Confirm the lock-managed hook binary with: uv run --locked prek --version (expected 0.5.10).
3. Install using: bash scripts/install_local_hooks.sh.
4. Inspect: git rev-parse --git-path hooks/pre-commit; git rev-parse --git-path hooks/pre-push. The native pre-push hook should match .githooks/pre-push.

The installer **refuses linked worktrees**, configured core.hooksPath, or existing pre-commit/pre-push hooks. This is deliberate: existing Git hooks can be shared by worktrees and must not be overwritten. Do not bypass the installer just to make the hook appear installed. The isolated Mac Issue #40 worktree is not used for shared hook installation; proof occurs in a disposable standalone clone.

## Enforced checks

- Pre-commit through pinned prek 0.5.10: non-mutating staged-file Ruff lint, changed-file Ruff formatting, staged whitespace check, and explicit validation of the actual .pre-commit-config.yaml file. Ordinary pre-commit execution proves activation; a YAML file alone is not an installed hook.
- Pre-push through native Git stdin ref inspection: **only one updated branch, exactly matching the checked-out clean HEAD, is supported**. Multi-ref pushes, other-branch pushes and branch deletion are rejected instead of being called qualified. This is an intentional fail-closed response to prek 0.5.10's first-ref/current-working-tree limitation.
- For the exact pushed SHA: uv lock --check, locked dev/drive/local-auth synchronization, validated hook configuration, Ruff across src/tests/scripts, Ruff formatting on changed Python files relative to main, applicable source mypy, strict mypy on the gate script, seeded full pytest (excluding only live_drive), nonvacuous JUnit test count, branch coverage >=80%, sdist and wheel builds, fresh installed-wheel CLI smoke and refusal of wheel finalization, and source E-002 no-overwrite verification. HEAD and clean-tree identity are checked before and after.
- Required stdout provenance includes outgoing ref, exact target HEAD/tree, Python minimum, executed/skipped tests, and exact passing SHA. A nonzero subprocess exit blocks the push.

## Transitional limitations and required end state

The 2026-11-15 main baseline has **56 unformatted Python files** in src/tests. Globally enabling Ruff format on every file would permanently break ordinary pushes without a separate audited formatting migration. The current gate validates all changed Python files (including new gate scripts); Issue #37 must eliminate the 56-file debt and replace this transition with repository-wide format enforcement. No blanket formatting exclusions, automatic fixes or hidden green checks are allowed.

Source mypy --strict has **15 pre-existing errors in six files**; Issue #38 owns the typed-contract corrections and strict source/test rollout. Current qualified mypy is still applied to all source modules. Ruff S110/S112 with check-typed-exception=true remains Issue #37 work; unsafe exception semantics must not be changed mechanically.

## Separate security-tool candidates

These are **not enforced** by this slice, and no scan result is claimed:

| Candidate | Present at baseline | Decision / required evidence |
| --- | --- | --- |
| Bandit | Not installed | Deferred pilot: pin version, measure incremental detection against Ruff S and benign cases, timing and false positives, then consider pre-push |
| Semgrep | Not installed | Conditional future pilot with reviewed local rules, offline/network/privacy controls and uniquely detected case |
| deptry | Not installed | Conditional dependency-change gate after verifying dev groups, extras and dynamic imports |
| pip-audit | Not installed | Conditional lock-change and scheduled fresh advisory scan after network/freshness policy and failure semantics are proven |
| Gitleaks | Not installed | Preferred secret-scanner pilot for staged content and outgoing commit history; prove harmless synthetic positives and benign controls |
| Betterleaks | Not installed | Deferred alternative to Gitleaks; do not stack two scanners without additional evidence |

No live credentials should be used in scanner experiments. A scanner executable not installed or advisory data unavailable is **UNVERIFIED**, not a clean result. These decisions and any future promotion must be documented under #40 with exact versions, detection controls, latency, privacy, and maintenance evidence.

## Limits and safe exceptions

Local hooks can be bypassed with Git --no-verify, modified hook paths, or manipulated commands. Their enforcement is a developer safeguard rather than a security boundary. An emergency bypass requires a recorded justification, exact-SHA replacement qualification, and independent required GitHub CI/review. Do not delete, reset, or rebase any unrelated worktree; do not weaken Public CI. PR #42 must update pyproject.toml and uv.lock for >=3.12 and requalify its own exact HEAD before migration; do not fold it into this branch.
