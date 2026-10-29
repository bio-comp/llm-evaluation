# canonical-sync incubation implementation plan

> **Spec:** `docs/design/2026-10-26-canonical-sync-design.md`
>
> **Execution:** inline in the existing isolated `feat/task2-drive-reconciliation` worktree. Preserve the two existing prototype commits as provenance; correct forward with small commits. No reset/clean of unrelated work, no Task-1/runtime implementation, no E-002 sampling, no Drive writes unless a final verified state change justifies one.

**Goal:** replace the project-specific Task-2 prototype with a reusable in-repository `canonical-sync` package, consume it through thin `llm_evaluation` adapters, prove the public/private reconciliation feedback path, and live-verify the read-only local OAuth boundary if existing credentials permit it.

> **Focused boundary correction — 2026-10-27:** the original Task 4/Task 6 wording below is retained as execution history but is superseded where it assigns project reconciliation states or public/private policy to `canonical_sync`. The implemented boundary is: `canonical_sync` owns verified acquisition/snapshot mechanics plus mechanical `GitPublisher` application/readback of caller-approved immutable path/byte projections over an exact verified base; omitted base paths are preserved and are not implicit deletion authorization. `llm_evaluation` owns `NONE` / `REQUIRED` / `BLOCKED` / `COMPLETE`, material-significance decisions, public-safe projection construction, and repository/ref/PR/CI completion binding to the exact merged commit. Authentication provisioning (interactive OAuth, GitHub OIDC, WIF/IAM) remains deployment/runbook work outside the package; both trusted auth paths converge through ADC.
> **Identity/serialization correction — 2026-10-27:** canonical persisted JSON and cryptographic identity are separate contracts. Identity uses an explicit versioned/domain-separated frame around canonical payload bytes. `source_state_id` is the domain-separated hash of `SnapshotStateRecord` (current storage spelling `snapshot_state_sha256`); `observation_id` is the hash of `SnapshotObservationRecord`; `receipt_id` is the hash of `AcquisitionReceiptRecord`. Provider versions, time, retries, runner/transport/auth provenance stay out of `source_state_id`. Runtime dataclasses remain non-persisted capability/value objects unless a separate versioned Pydantic record is introduced.

## Global constraints

- Drive remains sole mutable methodology authority; Git remains public executable implementation.
- `canonical_sync` contains mechanics only and must not import `llm_evaluation`, prlab, experiment IDs, or model/runtime policy.
- Pydantic v2 is used at config/wire/persisted/public boundaries; trusted internal state uses frozen/slotted dataclasses and Protocols.
- New `canonical-sync` tests use pytest. Existing root unittest tests remain intact unless a feature requires a focused change.
- Every production behavior is introduced after a test that was observed failing for the intended reason.
- Ordinary CI has no Drive credentials; `live_drive` is opt-in and excluded by default.
- Public Git never receives private Drive IDs/parents/revisions/canonical bodies, credentials, or local user paths.
- Hosted OIDC/WIF is not activated in this plan.
- The current native-Docs read-only transaction uses PRE/POST Drive observations plus deterministic Docs representation bytes; it does not claim raw native-Docs storage-byte identity.

## Task 1 — Workspace package scaffold and boundary codec

**Files:**
- modify `pyproject.toml`
- modify `uv.lock`
- add `packages/canonical-sync/pyproject.toml`
- add `packages/canonical-sync/src/canonical_sync/__init__.py`
- add `packages/canonical-sync/src/canonical_sync/errors.py`
- add `packages/canonical-sync/src/canonical_sync/records.py`
- add `packages/canonical-sync/src/canonical_sync/domain.py`
- add `packages/canonical-sync/src/canonical_sync/codec.py`
- add `packages/canonical-sync/tests/test_records_and_codec.py`

**Interfaces:**
- strict/frozen persisted Pydantic base record with explicit `record_type`/`schema_version` fields on durable records;
- `AuthorityObjectSpecRecord`, `ExactMembershipSpecRecord`, `AuthorityProfileRecord`;
- `SemanticEntryRecord`, `SnapshotStateRecord`, `SourceObservationRecord`, `SnapshotObservationRecord`;
- trusted dataclasses `AuthorityObjectSpec`, `ExactMembershipSpec`, `AuthorityProfile`, `SourceObservation`, `SourceContent`;
- `canonical_record_bytes(model)`, `canonical_record_hash(model)`, `parse_canonical_record(data, model_type)`.

**RED tests:**
- unknown fields and ambiguous/coerced types rejected;
- malformed/empty IDs, logical keys, SHA-256 values, unsafe logical paths, duplicate/case-fold-colliding keys rejected;
- persisted round-trip is byte-identical;
- noncanonical JSON representation is rejected even if semantically equivalent;
- sorted semantic entries produce identical state identity;
- one-byte content hash or authority-spec hash change changes semantic state identity;
- observation-only metadata/time change changes observation identity but not semantic state identity.

**GREEN:** implement the smallest records/domain/codec satisfying those tests; validate workspace resolution with installed uv 0.8.4.

**Verify:**
`uv run --all-packages pytest packages/canonical-sync/tests/test_records_and_codec.py -q`

**Commit:** `feat: scaffold canonical-sync boundary records`

## Task 2 — Transport protocol and stable multi-object acquisition

**Files:**
- add `packages/canonical-sync/src/canonical_sync/transport.py`
- add `packages/canonical-sync/src/canonical_sync/acquisition.py`
- add `packages/canonical-sync/tests/fakes.py`
- add `packages/canonical-sync/tests/test_acquisition.py`

**Interfaces:**
- `SourceTransport` Protocol: `observe(spec)`, `read(spec)`, `list_members(parent_id)`;
- typed transport exceptions: auth, permission, not-found, transient, permanent, unsupported-representation;
- `AcquisitionCandidate` trusted dataclass containing state, observation, acquired representations, and attempt count;
- `acquire_stable_candidate(transport, profile, *, max_movement_attempts=...)`.

**RED tests:**
- exact ID/name/MIME/parent/non-trashed identity enforced;
- wrong/missing required object fails before publication;
- stable two-object acquisition succeeds;
- identical semantic bytes with changed acquisition times/versions have same semantic state but distinct observation identity;
- one-byte source change changes semantic identity;
- source A changing while source B is read rejects the whole candidate;
- one movement retries the entire profile, not only the changed object;
- repeated movement raises typed exhaustion and returns no candidate;
- exact-membership profile rejects missing/extra/moved child membership;
- permanent/auth/identity failures do not get misclassified as source movement.

**GREEN:** implement whole-profile PRE/read/POST transaction and bounded movement retry using the in-memory transport first.

**Verify:**
`uv run --all-packages pytest packages/canonical-sync/tests/test_acquisition.py -q`

**Commit:** `feat: add stable multi-object acquisition`

## Task 3 — Verified private snapshot store and adversarial filesystem controls

**Files:**
- add `packages/canonical-sync/src/canonical_sync/store.py`
- add `packages/canonical-sync/tests/test_store.py`

**Interfaces:**
- `SnapshotStore(root)`;
- `stage(candidate) -> StagedAcquisition`;
- `publish(staged) -> PublicationResult`;
- `verify_completed(observation_hash) -> VerifiedSnapshot`;
- verified exact state/observation inventory and readback.

**RED tests:**
- traversal, absolute path, symlinked root/operational directory/file, and non-regular entry fail closed;
- staged content hash mismatch fails;
- publish semantic snapshot before observation; orphan semantic state is not completed;
- valid existing target is a true no-op after full verification;
- corrupt/conflicting existing target is never overwritten;
- unexpected files/directories fail verification;
- interrupted/partial staging is not treated as completed;
- two identical publishers serialize to publish + verified no-op on supported POSIX CI platforms.

**GREEN:** implement same-filesystem staging, fsync, exclusive lock, atomic rename, exact inventory verification, and observation-last completion.

**Verify:**
`uv run --all-packages pytest packages/canonical-sync/tests/test_store.py -q`

**Commit:** `feat: add verified canonical snapshot store`

## Task 4 — Reconciliation state and public projection boundary

**Files:**
- extend `packages/canonical-sync/src/canonical_sync/records.py`
- add `packages/canonical-sync/src/canonical_sync/reconciliation.py`
- add `packages/canonical-sync/src/canonical_sync/publication.py`
- add `packages/canonical-sync/tests/test_reconciliation.py`
- add `packages/canonical-sync/tests/test_publication.py`

**Interfaces:**
- enums `ReconciliationDisposition` = `NONE|REQUIRED|BLOCKED|COMPLETE` and `ProjectImpact` = no-public-impact|required|unknown;
- persisted `LastReconciledStateRecord`, `ReconciliationDispositionRecord`, `VerifiedPublicationReceiptRecord`;
- `classify_reconciliation(current, previous, impact)`;
- `complete_reconciliation(required, verified_receipt)`;
- `PublicProjection` boundary and `publish_public_projection(...)` that accepts only deliberate project projections, never `VerifiedSnapshot` or raw source bodies.

**RED tests:**
- all four dispositions remain distinct;
- unchanged state -> `NONE`; changed + no impact -> `NONE`; changed + public impact -> `REQUIRED`; changed + unknown -> `BLOCKED`;
- `COMPLETE` requires exact semantic identity + verified commit/readback binding;
- branch edit alone cannot complete;
- private Drive markers, exact raw canonical body reuse, credential/token shapes, traversal targets, and payload hash mismatch are rejected;
- narrow public-safe projection succeeds and is atomically read back.

**GREEN:** implement state machine and explicit projection/publication guard. Secret-pattern scanning is defense in depth, not the primary schema boundary.

**Verify:**
`uv run --all-packages pytest packages/canonical-sync/tests/test_reconciliation.py packages/canonical-sync/tests/test_publication.py -q`

**Commit:** `feat: add reconciliation and publication boundaries`

## Task 5 — Generic Google Drive adapter

**Files:**
- add `packages/canonical-sync/src/canonical_sync/google_drive.py`
- add `packages/canonical-sync/tests/test_google_drive.py`
- update `packages/canonical-sync/pyproject.toml` optional `drive` extra
- update root workspace dependency/extra wiring as needed
- update `uv.lock`

**Interfaces:**
- `GoogleDriveTransport` implementing `SourceTransport`;
- exact `drive.readonly` scope;
- bounded request-level transient retries with typed permanent failures;
- strict Pydantic validation of Drive metadata/list responses before trusted dataclasses;
- `google-drive-export-markdown-v1` exact `text/markdown` Drive-export representation for the current native-Docs profile;
- list-members support for exact direct-child membership when requested.

**RED tests:**
- exact OAuth scope and request fields/options;
- malformed metadata/docs response rejected;
- 401/403/404 classified distinctly; 429/5xx/specific rate-limit 403 retry boundedly; raw provider error/credential text never reaches durable error text;
- Markdown export uses exact returned bytes without normalization;
- unsupported representation schemas or non-byte exports fail closed;
- metadata-only Drive version change does not alter exported representation bytes;
- Google clients remain optional and are not installed by the base `canonical-sync` wheel.

**GREEN:** port only reusable provider mechanics from the existing prototype/prlab behavior; do not carry project names or IDs.

**Verify:**
`uv run --all-packages --extra drive pytest packages/canonical-sync/tests/test_google_drive.py -q`


**Live correction after initial Task-5 implementation:** local OAuth verification showed Drive metadata access succeeded but Docs `documents.get` returned `SERVICE_DISABLED`; the same `drive.readonly` credential successfully exported `text/markdown`. Repeated exports were byte-stable across all four registered objects while Drive versions were unchanged. The final adapter therefore uses Drive v3 metadata + `files.export` only and does not require enabling the Docs API.

**Commit:** `feat: add read-only Google canonical transport`

## Task 6 — llm_evaluation adapters and discriminating canonical-to-executable slice

**Files:**
- add `src/llm_evaluation/canonical_authority.py`
- add `src/llm_evaluation/canonical_reconciliation.py`
- add `tests/test_canonical_reconciliation.py`
- add a public-safe synthetic fixture only if needed under `fixtures/`
- remove superseded `src/llm_evaluation/authority_store/` after replacement tests are green
- replace/remove superseded `tests/test_authority_store.py`

**Interfaces:**
- strict private profile loader from a caller-provided TOML path (IDs never default in source);
- project-selected logical roles for D-016 transactions;
- public `ReconciliationPolicyContract` containing only the admitted executable facts needed by code;
- adapter that accepts verified snapshot identity + explicit reviewed project interpretation and returns generic impact/disposition/public projection.

**RED tests:**
- profile with extra/missing/unexpected project roles rejected;
- project adapter cannot serialize private authority config into public projection;
- current four-state policy contract matches executable implementation;
- counterfactual public-safe authoritative projection makes the stale executable contract test fail;
- after applying the matching reconciliation contract, the same test passes;
- private-only semantic change produces `NONE` and no public artifact;
- unknown/unsafe interpretation produces `BLOCKED`.

**GREEN:** implement the narrow adapter; do not parse arbitrary Drive narrative into executable code.

**Verify:**
`uv run --all-packages pytest tests/test_canonical_reconciliation.py -q`

**Commit:** `refactor: consume canonical-sync from llm-evaluation`

## Task 7 — Opt-in live Drive test and credential-free CI boundary

**Files:**
- add `tests/test_canonical_drive_live.py`
- modify `pyproject.toml` pytest marker config
- modify `.github/workflows/ci.yml`
- add/update public docs explaining local opt-in without IDs/credentials

**Interfaces:**
- marker `live_drive`;
- explicit `LLM_EVALUATION_LIVE_DRIVE=1` opt-in;
- private profile path environment variable; no default private path in public output;
- local ADC/OAuth constructor requests only `drive.readonly`;
- CI command excludes `live_drive` and does not install/use credentials unless explicitly testing package build extras.

**RED tests/checks:**
- default test run skips live Drive path;
- CI workflow contains no credential secret dependency for ordinary verification;
- a missing opt-in/profile gives a skip, not a network call;
- sanitized live failure cannot echo profile contents/tokens.

**GREEN:** wire the live test and CI exclusion.

**Verify deterministic boundary before network:**
`uv run --all-packages --extra drive pytest -m 'not live_drive' -q`

**Commit:** `test: add opt-in live canonical acquisition gate`

## Task 8 — Local OAuth live D-016 acquisition transaction

**Private working state only:** `.private-drive-cache/` (already gitignored).

**Steps:**
1. inspect existing ADC/OAuth availability without printing tokens or credential bodies;
2. construct a private `llm_evaluation` profile from the live registered authority already resolved from Drive, including only the transaction-relevant canonicals;
3. run the opt-in live test against a temporary/private snapshot store;
4. verify exact registered object access, stable multi-object acquisition, semantic/observation identities, store publication/readback, and controlled fail-closed identity mismatch;
5. record only public-safe outcome evidence in the ledger/commit notes; keep IDs/content/private paths out of Git;
6. classify the real D-016 transaction using the explicit project adapter. Do not call it `COMPLETE` unless the matching Git publication/readback/PR/CI evidence exists.

**Stop conditions:** do not launch an interactive consent flow, broaden scopes, alter Drive sharing, write Drive, or create hosted credentials without separate authorization. If ADC is absent/insufficient, record the blocker and continue deterministic verification.

**Commit:** none solely for private credentials/snapshot evidence. Public code fixes discovered by the live test follow RED→GREEN and get a focused commit.

## Task 9 — prlab second-consumer mapping and full verification

**Files:**
- add `docs/canonical-sync-prlab-compatibility.md` with public-safe behavior mapping only (no private canonical IDs/content)
- update design/README links if helpful, without changing Prompt Research code

**Map:**
- prlab `DriveTransport` -> generic transport behaviors;
- semantic `SnapshotState` vs observation manifest -> generic identities;
- source/membership movement -> acquisition contract;
- snapshot store integrity/publication -> generic store;
- prlab mirror publication -> generic exact-tree Git mechanics plus project-specific branch/state/scheduling policy;
- list any missing generic requirement as an extraction-gate item, not an immediate abstraction unless `llm_evaluation` also needs it.

**Fresh full verification:**
- `uv lock --check`
- `uv run --all-packages --extra drive ruff check src tests packages/canonical-sync/src packages/canonical-sync/tests`
- `uv run --all-packages --extra drive mypy --python-version 3.11 src packages/canonical-sync/src`
- `uv run --all-packages --extra drive pytest -m 'not live_drive' -q`
- branch coverage gate at the repository's current threshold, including workspace package sources
- build `canonical-sync` and `llm-evaluation` distributions
- install-smoke each wheel in fresh venvs; base installs must not pull Google clients unexpectedly
- inspect `git status`, `git diff --check`, tracked-file inventory, and grep for private IDs/credential/token/local-path sentinels
- verify runtime Task-1 worktree/branch remains untouched

**Final review:** perform the whole-branch review against the spec and current Drive D-016 boundaries. Critical/Important findings get one RED→GREEN fix pass; minor findings are explicitly deferred. No push/merge is performed without separate authorization.

**Commit:** `docs: record canonical-sync compatibility and verification`

## Completion evidence

Report, without private Drive material:

- branch base and final commits;
- package/files added/removed;
- exact test count/skips and live-test outcome;
- Ruff/mypy/lock/build/install-smoke results;
- mixed-state negative-control evidence;
- filesystem plus consumer-projection/publication-tree negative-control evidence;
- all four reconciliation dispositions mechanically exercised;
- canonical-to-executable stale-vs-reconciled counterfactual result;
- live OAuth authority profile scope and outcome at the level safe to publish;
- real D-016 disposition and why it is or is not `COMPLETE`;
- prlab API compatibility gaps/extraction gates;
- whether any Drive canonical required an update.

A Drive update is allowed only if implementation evidence changes durable project state. If needed, update the same authoritative object in place and reread/verify it; never create a replacement canonical.

## 2026-10-28 Plan amendment: Task 2 pre-merge review hardening and authoritative closure reconciliation

Following review of PR #9 and the 2026-10-28 D-016 clarification in Google Drive, the implementation plan incorporates the following pre-merge hardening steps:

1. **GitPublisher atomic CAS lease**: Replace unguarded remote pushes with `--force-with-lease=<ref>:<expected_base>` (or `--force-with-lease=<ref>:` for branch creation when `expected_base is None`), paired with mechanical verification that candidate commits declare the exact expected base as their parent. Negative tests verify that remote rewinds, deletions, and concurrent updates immediately before push raise `RemoteRaceError`.
2. **Authoritative GitHub closure acquisition**: Distinguish internal record validity from external evidence authenticity. Wire models validate external provider payloads strictly; project logic verifies PR merge state (`merged == True`, `merged_at`, `merge_commit_sha`) and CI post-merge push execution (`event == "push"`, `head_branch == "refs/heads/main"`, workflow path `.github/workflows/ci.yml`, required job `verify` with `conclusion == "success"`) before producing versioned durable evidence records.
3. **Public policy contract schema v2**: `fixtures/reconciliation_policy_contract.json` explicitly declares all authoritative closure requirements via dedicated boolean flags.
4. **SnapshotStore error sanitization**: Translate all filesystem `OSError` occurrences into `SnapshotIntegrityError` without leaking private host paths in message or cause (`from None`), and support platform root directory aliases (such as macOS `/tmp`).
5. **Base-tree path parity**: Reject case-folding and Unicode normalization (NFC/NFD) collisions in `_read_tree` to prevent platform-dependent tree desynchronization.
6. **Narrow D-016 root parent invariant**: Ensure all four required root authority roles share the same non-null Drive parent ID.
7. **Durable schema versioning**: Retain `GitPublicationReceiptRecord` v3 (restores declared guarantee); update `GitHubPullRequestMergeRecord` to v2; update `GitHubCIVerificationRecord` to v3; update `VerifiedPublicationReceiptRecord` to v4; update `ReconciliationDispositionRecord` to v3; update `ReconciliationPolicyContract` to v2.
