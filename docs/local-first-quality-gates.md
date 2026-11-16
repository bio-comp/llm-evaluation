# Local-first Python engineering gates (Issue #40)

## Scope and authority

This repository keeps GitHub Public CI as the independent required exact-merge gate. Local checks reduce avoidable CI failures but do not authorize a merge. The first implementation slice enforces the existing Python 3.11.13 main baseline; Issue #41 / PR #42 separately owns promotion to Python >=3.12 with development on 3.14.10. The gate reads the declared package minimum to select mypy's compatibility target.

No scientific-method, persisted-artifact, data schema, or historical Python 3.11 identity is changed by this slice. GitHub Issues #36–#39 remain the owners of historical assertions, formatting, strict typing, and Pydantic/settings work.

## Safe per-worktree activation

Each worktree uses its own hooks directory under its private Git directory,
selected through per-worktree core.hooksPath. The installer never writes into
the shared Git hooks directory and never implicitly activates this new gate
for older linked branches. Those branches remain independently usable, but
are NOT protected by the new gate; required GitHub CI remains independent.

1. Verify the locked environment and pinned prek 0.5.10.
2. In a standalone clone with no existing hook manager, run:
   bash scripts/install_local_hooks.sh
3. With multiple worktrees, inspect the existing Git configuration and all
   existing hooks first. An authorized operator explicitly opts into
   worktree-specific Git configuration by running:
   git config --local extensions.worktreeConfig true
   The installer will NEVER enable that shared extension automatically with
   multiple worktrees. Verify core.worktree migration requirements before
   enabling it.
4. Run the installer within each checkout that has the new gate script and
   is meant to be protected. Inspect the result with:
   git config --worktree --get core.hooksPath
5. Repeat installation is a no-op only when the original hooks exactly match.
   Existing other hook managers, core.hooksPath, or changed installed hooks
   require explicit reconciliation; no silent overwrite is performed.

The installer validates configuration, builds executable hooks in a private
staging directory, and switches worktree-specific core.hooksPath LAST. It
cleans up only its own staged files if activation fails. The tracked hook
source files remain mode 100644 to preserve the repository GitPublisher
contract. Worktree hook configuration is not shared with other developers.

## Enforced checks

- Pre-commit through pinned prek 0.5.10: non-mutating staged-file Ruff lint, changed-file Ruff formatting, staged whitespace check, and explicit validation of the actual .pre-commit-config.yaml file. Ordinary pre-commit execution proves activation; a YAML file alone is not an installed hook.
- Pre-push through native Git stdin ref inspection: **zero outgoing updates** are an explicitly reported successful no-op, with no revision published or qualified and no expensive quality gate. **Exactly one updated branch, matching the checked-out clean HEAD**, runs the complete gate. Multi-ref pushes, malformed records, other-branch pushes, branch deletion and unavailable/divergent remote history continue to fail closed.
- Existing-branch pushes use Git's remote-old SHA as the lower bound; initial publication requires a trustworthy ancestral main reference, including when that main HEAD equals the new branch HEAD. For a new remote ref at trusted main HEAD, the outgoing commit-diff set is explicitly empty, but the **full exact-HEAD gate still runs** before creating the new ref; this is not an empty pre-push stream. Every applicable outgoing commit is checked separately with git diff --check against its parent. This detects defects introduced and later removed in different commits. Clean worktree diff checks are not evidence of clean committed history.
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
