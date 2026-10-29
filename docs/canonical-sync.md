# Canonical synchronization boundary

`llm-evaluation` treats Google Drive as the sole mutable authority for methodology and reconciliation state. The public repository contains executable code, tests, schemas, public-safe contracts, and deliberately published artifacts; it does not mirror private canonical bodies or Drive metadata.

The reusable mechanics are incubated in the workspace package `packages/canonical-sync` (import namespace `canonical_sync`). The governing boundary is: `canonical_sync` owns verified acquisition/snapshot mechanics and mechanical Git application/readback of an already-approved path/byte projection over an exact verified base; omitted base paths are preserved and are not implicit deletion authorization. `llm_evaluation` owns authority-role interpretation, material significance, public-safe projection construction, repository/ref/review/CI completion semantics, and the `NONE` / `REQUIRED` / `BLOCKED` / `COMPLETE` governance state machine. A `VerifiedSnapshot` is evidence, not authorization to mutate Git.

## Identity and canonical serialization

The package keeps three authority/acquisition identities separate:

- `content_sha256`: exact bytes for one acquired source representation;
- `source_state_id`: `canonical_record_hash(SnapshotStateRecord)`, binding registered source identity + meaning-bearing logical mapping + exact content identities. Existing persisted names such as `snapshot_state_sha256` refer to this same identity;
- `observation_id` / `receipt_id`: hashes of the observation and acquisition-receipt records. Observations bind provider revision/version plus timing/attempt and safe runner/transport/auth provenance; both successful and failed acquisition receipts retain safe runner/transport/auth provenance, while failures retain only sanitized error codes rather than raw provider bodies or credentials.

Changing Drive object identity or logical mapping changes `source_state_id` even if bytes are identical. Reacquiring unchanged registered objects/mapping/bytes preserves `source_state_id` while new provider/runtime/auth observations can produce new observation/receipt identities.

Pydantic validates durable record structure, but hash identity is not `sha256(model_dump(...))`. Canonical persisted payloads use deterministic UTF-8 JSON with explicit scalar normalization and no trailing newline. Cryptographic identity wraps those bytes in `canonical-sync` identity protocol v1 with a declared record domain, domain version, record schema version, and length framing before SHA-256. Runtime-only frozen dataclasses such as verified/staged handles and approved `PublicationTree` values are not persistence formats.

## Local read-only verification

Live Drive acquisition is opt-in and is never part of ordinary PR CI. It uses Google Application Default Credentials with the single scope `https://www.googleapis.com/auth/drive.readonly`.

Keep the authority profile outside the repository. It is a private TOML file containing the exact registered object and parent identities for the D-016 roles `README.md`, `DECISIONS.md`, `EXPERIMENTS.md`, and `REVIEW_LEDGER.md`. A schematic profile entry is:

```toml
[profile]
profile_name = "llm-evaluation-d016"

[[profile.objects]]
logical_path = "README.md"
provider = "google_drive"
object_id = "<private registered Drive object ID>"
expected_name = "README.md"
expected_mime_type = "application/vnd.google-apps.document"
expected_parent_id = "<private registered parent ID>"
representation_schema = "google-drive-export-markdown-v1"
```

Repeat the object table for the other required roles. Do not commit the populated profile, OAuth client material, ADC files, tokens, private snapshots, or copied canonical bodies.

With already-authorized local ADC credentials and the private profile available, run:

```bash
export LLM_EVALUATION_LIVE_DRIVE=1
export LLM_EVALUATION_DRIVE_PROFILE=/absolute/private/path/llm-evaluation-d016.toml
uv run --all-packages --extra drive pytest -m live_drive tests/test_canonical_drive_live.py -q
```

Without both environment variables the live test skips before credential discovery or network access. The test performs only read operations, acquires a stable multi-object snapshot into a temporary private store, reopens and verifies it, and checks a controlled authority-mismatch failure.

## CI boundary

Ordinary public CI is credential-free. It may install the optional Google client libraries to type-check and unit-test the adapter, but it supplies no Google credentials and runs pytest with `-m 'not live_drive'`. Hosted GitHub OIDC → Google Workload Identity Federation is a later activation step and is not configured by this tranche.

## Public reconciliation contract

`fixtures/reconciliation_policy_contract.json` is deliberately public-safe. It records only the executable four-state reconciliation contract (`NONE`, `REQUIRED`, `BLOCKED`, `COMPLETE`) and the verified-publication conditions required for `COMPLETE` under schema version 2. Private Drive identities and canonical narrative are excluded by design.

Under the 2026-10-28 D-016 clarification, internal record validity is strictly distinguished from external evidence authenticity. Caller-constructed or synthetic Pydantic records can validate structurally, but do not constitute authoritative closure evidence. Project-owned closure acquisition in `llm_evaluation.canonical_github` strictly enforces:
1. Provider/wire validation of external responses (`_WirePullRequest`, `_WireWorkflowRun`, `_WireWorkflowJob`) using strict Pydantic parsing without network calls inside validators;
2. Verification against expected repository, PR number, reviewed head SHA, merge commit, workflow path (`.github/workflows/ci.yml`), push event to `refs/heads/main`, and `verify` job success;
3. Transformation into frozen trusted runtime facts (`VerifiedGitHubClosureFacts`);
4. Production of versioned durable Pydantic evidence records (`GitHubPullRequestMergeRecord` schema v2, `GitHubCIVerificationRecord` schema v3);
5. Binding into `VerifiedPublicationReceiptRecord` schema v4 and `ReconciliationDispositionRecord` schema v3 for the operational `COMPLETE` transition.

The policy contract schema v2 explicitly declares these invariants via dedicated boolean fields (`complete_requires_authoritative_github_evidence`, `complete_requires_merged_pull_request`, `complete_requires_post_merge_main_push_ci`, `complete_requires_exact_workflow_identity`, `complete_requires_exact_required_job_identity`, `complete_requires_snapshot_match`, `complete_requires_contract_match`, `complete_requires_readback`).

## Related design records

- [`canonical-sync` implementation design](design/2026-10-26-canonical-sync-design.md)
- [Prompt Research/prlab second-consumer compatibility check](canonical-sync-prlab-compatibility.md)
- [Hosted GitHub OIDC → Google WIF activation design](design/canonical-sync-oidc-wif-activation.md)

## Review hardening and historical records

### Atomic CAS Git publication lease

`GitPublisher.publish()` enforces a server-side atomic compare-and-swap (CAS) lease on the pushed ref. For an existing base, it pushes with `--force-with-lease=<ref>:<expected_base>`. For target creation when `expected_base is None`, it pushes with the explicit empty expectation `--force-with-lease=<ref>:`, ensuring the push fails if the ref was concurrently created. In addition, the publisher mechanically validates that candidate commits have the exact expected base as their single parent before pushing. A concurrent remote rewind, ref deletion, or concurrent branch creation raises `RemoteRaceError`, leaving the remote state untouched.

This CAS hardening restores the originally declared `fast_forward_only=True` guarantee in `GitPublicationReceiptRecord` schema v3; historical schema v3 records declared this guarantee, so schema v3 is retained as a bug fix rather than a semantic redefinition.

### Base-tree path and portable filesystem parity

Before creating commits or pushing, `GitPublisher._read_tree()` enforces portable path hygiene across the remote base tree and candidate publication tree. In addition to rejecting non-100644 mode entries and hierarchy collisions (where a path is both a file and a directory component), it rejects case-folding collisions (e.g. `foo` vs `FOO`) and Unicode normalization collisions (NFC vs NFD forms) to guarantee deterministic cross-platform compatibility on case-insensitive or Unicode-normalizing filesystems.

### SnapshotStore operational error boundary

`SnapshotStore` operations sanitize low-level filesystem `OSError` exceptions by translating them into `SnapshotIntegrityError` without leaking private local filesystem paths, store root paths, or internal diagnostics in `str(exc)` or `__cause__` (`raise SnapshotIntegrityError(...) from None`). Path traversal defenses reject symlinks across store operations while safely accommodating standard platform root directory aliases (such as macOS `/tmp` and `/var` owned by root).

### Narrow D-016 authority-profile parent invariant

The D-016 authority profile validator in `llm_evaluation.canonical_authority` enforces that all four required root authority roles (`README.md`, `DECISIONS.md`, `EXPERIMENTS.md`, `REVIEW_LEDGER.md`) share the exact same non-null Drive parent ID. This invariant reflects the single authoritative Drive governance folder for the project and prevents accidentally mixing objects across unrelated folders, while deliberately remaining narrow so secondary or nested canonical objects are not erroneously forced into the root folder.

### Durable schema versioning

- `GitPublicationReceiptRecord`: schema v3 retained (restores declared CAS guarantee).
- `GitHubPullRequestMergeRecord`: schema v2 (added `merged_at` ISO 8601 UTC timestamp, strict canonical `refs/heads/...` ref format).
- `GitHubCIVerificationRecord`: schema v3 (added `workflow_path`, `event="push"`, `head_branch="refs/heads/main"`, `run_conclusion="success"`, `job_id`, `job_name="verify"`, `job_conclusion="success"`).
- `VerifiedPublicationReceiptRecord`: schema v4 (binds PR v2 and CI v3).
- `ReconciliationDispositionRecord`: schema v3 (binds `VerifiedPublicationReceiptRecord` v4).
- `ReconciliationPolicyContract`: schema v2 (explicit booleans for all authoritative closure requirements).

Historical payloads remain untouched; current readers explicitly reject old schema versions and never infer missing provenance or completion evidence. No historical private receipt is deleted or migrated.

The live Drive authority requires project-owned last-reconciled state but does not yet designate an exact storage object/path for `LastReconciledStateRecord`. That home must be resolved before real closure; no public checkpoint is created by this implementation or this review.
