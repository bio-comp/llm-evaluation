# Canonical synchronization boundary

`llm-evaluation` treats Google Drive as the sole mutable authority for methodology and reconciliation state. The public repository contains executable code, tests, schemas, public-safe contracts, and deliberately published artifacts; it does not mirror private canonical bodies or Drive metadata.

The reusable mechanics are incubated in the workspace package `packages/canonical-sync` (import namespace `canonical_sync`). The governing boundary is: `canonical_sync` owns verified acquisition/snapshot mechanics and mechanical Git application/readback of an already-approved path/byte projection over an exact verified base; omitted base paths are preserved and are not implicit deletion authorization. `llm_evaluation` owns authority-role interpretation, material significance, public-safe projection construction, repository/ref/review/CI completion semantics, and the `NONE` / `REQUIRED` / `BLOCKED` / `COMPLETE` governance state machine. A `VerifiedSnapshot` is evidence, not authorization to mutate Git.

The acquisition vertical slice in `llm_evaluation` is intentionally narrow: it acquires the four configured root governance roles (`README.md`, `DECISIONS.md`, `EXPERIMENTS.md`, and `REVIEW_LEDGER.md`) rather than mirroring the entire evaluation methodology. The public policy fixture (`fixtures/reconciliation_policy_contract.json`) is a mandatory component of a public-change projection, but it is not a substitute for the exact project-reviewed implementation/documentation files required by a specific reconciliation obligation.

## Identity and canonical serialization

The package keeps three authority/acquisition identities separate:

- `content_sha256`: exact bytes for one acquired source representation;
- `source_state_id`: `canonical_record_hash(SnapshotStateRecord)`, binding registered source identity + meaning-bearing logical mapping + exact content identities. Existing persisted names such as `snapshot_state_sha256` refer to this same identity;
- `observation_id` / `receipt_id`: hashes of the observation and acquisition-receipt records. Observations bind provider revision/version plus timing/attempt and safe runner/transport/auth provenance; both successful and failed acquisition receipts retain safe runner/transport/auth provenance, while failures retain only sanitized error codes rather than raw provider bodies or credentials.

Changing Drive object identity or logical mapping changes `source_state_id` even if bytes are identical. Reacquiring unchanged registered objects/mapping/bytes preserves `source_state_id` while new provider/runtime/auth observations can produce new observation/receipt identities.

Pydantic validates durable record structure, but hash identity is not `sha256(model_dump(...))`. Canonical persisted payloads use deterministic UTF-8 JSON with explicit scalar normalization and no trailing newline. Cryptographic identity wraps those bytes in `canonical-sync` identity protocol v1 with a declared record domain, domain version, record schema version, and length framing before SHA-256. Runtime-only frozen dataclasses such as verified/staged handles and approved `PublicationTree` values are not persistence formats.

## Local read-only verification

Live Drive acquisition is opt-in and is never part of ordinary PR CI. It uses Google Application Default Credentials with the single scope `https://www.googleapis.com/auth/drive.readonly`. Operator provisioning of isolated local credentials across multiple consumers is detailed in [Local Google Authentication Runbook](canonical-sync-local-google-auth.md).

For the current `llm-evaluation` consumer, provision, renew, or validate the isolated authorized-user ADC with the tested operator helper rather than an ad-hoc serialization recipe:

```bash
uv run --all-packages --extra local-auth \
  python scripts/refresh_google_drive_adc.py \
  --client-secrets ~/.config/llm-evaluation/llm-evaluation-oauth.json \
  --cloud-sdk-config ~/.config/gcloud-llm-evaluation
```

Use `--check` for noninteractive validation and `--force` only for deliberate replacement of an existing destination. This repo-local helper is temporary migration debt pending separately authorized standalone `canonical-sync` extraction; it is not part of the acquisition runtime API.

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

`fixtures/reconciliation_policy_contract.json` is deliberately public-safe. It records only the executable four-state reconciliation contract (`NONE`, `REQUIRED`, `BLOCKED`, `COMPLETE`) and the verified-publication conditions required for `COMPLETE` under schema version 3. Private Drive identities and canonical narrative are excluded by design.

Under D-016, structural record validation is strictly distinguished from operational acquisition and readback. Pure structural helpers (`complete_reconciliation`, `VerifiedPublicationReceiptRecord`, `GitPublicationReceiptRecord`) validate persisted record invariants, but a caller-constructed or deserialized receipt never substitutes for live operational verification. Operational closure in `llm_evaluation.canonical_github.complete_reconciliation_with_github` executes the following read-only sequence:
1. Requires the exact project-reviewed `PublicationTree` and binds it against `required.publication_tree_sha256`; the tree must contain canonical bytes for the reviewed `ReconciliationPolicyContract`, and a contract alone can never synthesize implementation evidence. The matching non-empty reconciliation-obligation set is carried in the `REQUIRED` record;
2. Validates external GitHub responses (`_WirePullRequest`, `_WireWorkflowRun`, `_WireWorkflowJob`) using strict Pydantic wire models without network I/O inside validators;
3. Verifies exact bindings for `bio-comp/llm-evaluation`: pull request `pr_number` is merged (`merged == True`, explicit ISO-8601 UTC `merged_at`, `merge_commit_sha == M`, reviewed `head.sha == H` on a non-`main` branch where `H` and `M` may legitimately differ, and `base.ref == "refs/heads/main"`), and post-merge `push` workflow run `run_id` on `refs/heads/main` at `head_sha == M` succeeded for `.github/workflows/ci.yml` (`Public CI`) with unique required job `verify` (`conclusion == "success"`);
4. Performs actual read-only Git verification via `GitPublisher(remote, ref="refs/heads/main").verify(publication_tree, expected_commit=M)` to confirm `refs/heads/main` is at `M` without concurrent movement and contains the exact approved `PublicationTree` bytes;
5. Produces versioned durable evidence records (`GitHubPullRequestMergeRecord` schema v2, `GitHubCIVerificationRecord` schema v3, `GitPublicationReceiptRecord` schema v3) bound into `VerifiedPublicationReceiptRecord` schema v5 and `ReconciliationDispositionRecord` schema v4 (`COMPLETE`), preserving the exact obligation set and approved tree identity.

The policy contract schema v4 explicitly declares these invariants via dedicated boolean fields, including exact approved-publication-tree and obligation-set matching in addition to snapshot/contract/readback, authoritative GitHub evidence, merged PR, post-merge `push`/`main` CI, exact workflow identity, and exact required-job identity. It also declares that advancing the global durable checkpoint requires the completion evidence to match the caller-reviewed complete outstanding obligation set.

## Related design records

- [`canonical-sync` implementation design](design/2026-10-26-canonical-sync-design.md)
- [Prompt Research/prlab second-consumer compatibility check](canonical-sync-prlab-compatibility.md)
- [Hosted GitHub OIDC → Google WIF activation design](design/canonical-sync-oidc-wif-activation.md)
- [Local Google Authentication Runbook for canonical-sync consumers](canonical-sync-local-google-auth.md)

## Operational invariants and current status

### Atomic CAS Git publication lease

`GitPublisher.publish()` enforces a server-side atomic compare-and-swap (CAS) lease on the pushed ref. For an existing base, it pushes with `--force-with-lease=<ref>:<expected_base>`. For target creation when `expected_base is None`, it pushes with the explicit empty expectation `--force-with-lease=<ref>:`, ensuring the push fails if the ref was concurrently created. In addition, the publisher mechanically validates that candidate commits have the exact expected base as their single parent before pushing. A concurrent remote rewind, ref deletion, or concurrent branch creation raises `RemoteRaceError`, leaving the remote state untouched.

### Supported path/tree inventories, pre-mutation rejection, and omission-preservation

Path and inventory validators apply one consistent portable boundary across authority configurations (`AuthorityProfile`, `AuthorityProfileRecord`, `AuthoritySpecificationRecord`), snapshot records (`SnapshotStateRecord`, `SnapshotObservationRecord`), publication projections (`PublicationFile`, `PublicationTree`), remote-base inspection (`GitPublisher._read_tree()`), and the combined base-plus-projection tree in `GitPublisher.publish()`:
- **Single-path hygiene**: Every path must be a non-empty canonical relative POSIX path with at least one component. Literal `.`, `..`, empty segments, backslashes, ASCII control characters, Windows illegal characters/reserved stems, trailing dots/spaces, and Git metadata names/aliases under `core.protectHFS` and `core.protectNTFS` (including `.git`, `git~1`, and HFS-ignorable formatting characters `U+200C..U+200F`, `U+202A..U+202E`, `U+206A..U+206F`, `U+FEFF` in any directory or leaf component) are rejected.
- **Full-inventory ancestor hierarchy and directory-prefix consistency**: Across every inventory, portable file keys (NFC-normalized, HFS-stripped, case-folded) must be unique, every shared ancestor directory prefix must use one consistent exact spelling across all entries (rejecting case or NFC/NFD directory-prefix aliases such as `Foo/a.txt` + `foo/b.txt` or `Café/a.txt` + `Cafe\u0301/b.txt`), and no file path may collide with any ancestor directory prefix of another path regardless of lexical interlopers (such as `a`, `a-b`, `a/c`).
- **Supported Git tree entries and omission-preservation**: `GitPublisher` supports regular `100644` blobs and non-empty `040000` directories implied by those blobs. `_read_tree()` inspects `git ls-tree -r -t -z` and fails closed with `RemoteVerificationError` before commit creation or push (and on no-op or read-only `verify()`) if a remote commit contains executable/symlink/submodule modes, empty subtrees (`040000 tree` without descendant blobs), or path/prefix collisions. Valid base paths omitted from a `PublicationTree` projection are preserved byte-for-byte. Invalid authority configurations fail before any provider `observe`/`read` calls.

### SnapshotStore operational error boundary

`SnapshotStore` operations sanitize low-level filesystem `OSError` exceptions by translating them into `SnapshotIntegrityError` without leaking private local filesystem paths, store root paths, or internal diagnostics in `str(exc)` or `__cause__` (`raise SnapshotIntegrityError(...) from None`). Path traversal defenses reject symlinks across store operations while safely accommodating standard platform root directory aliases (such as macOS `/tmp` and `/var` owned by root).

### Narrow D-016 authority-profile parent invariant

The D-016 authority profile validator in `llm_evaluation.canonical_authority` enforces that all four required root authority roles (`README.md`, `DECISIONS.md`, `EXPERIMENTS.md`, `REVIEW_LEDGER.md`) share the exact same non-null Drive parent ID. This invariant reflects the single authoritative Drive governance folder for the project and prevents accidentally mixing objects across unrelated folders, while deliberately remaining narrow so secondary or nested canonical objects are not erroneously forced into the root folder.

### Durable schema versioning

- `GitPublicationReceiptRecord`: schema v3 retained (restores declared CAS and verified-readback guarantees without changing persisted fields).
- `GitHubPullRequestMergeRecord`: schema v2 (`merged_at` ISO 8601 UTC timestamp, canonical `refs/heads/...` ref format).
- `GitHubCIVerificationRecord`: schema v3 (`workflow_path`, `event="push"`, `head_branch="refs/heads/main"`, `run_conclusion="success"`, `job_id`, `job_name="verify"`, `job_conclusion="success"`).
- `VerifiedPublicationReceiptRecord`: schema v5 (binds the exact obligation set, PR v2, CI v3, and verified Git readback v3).
- `ReconciliationDispositionRecord`: schema v4 (binds the exact obligation set and `VerifiedPublicationReceiptRecord` v5).
- `ReconciliationPolicyContract`: schema v4 (adds the global-checkpoint rule that durable advance requires the complete outstanding obligation set, while retaining exact approved-tree and obligation matching).
- `DurableReconciliationStateRecord`: schema v2 (requires the new completion evidence shape).

Historical payloads are never silently reinterpreted. Durable schema v1 is fail-closed as active state and requires explicit supersession recovery: exact legacy bytes are archived under a hash-addressed local `reconciliation-history/` record with readback verification before the active legacy pathname is removed. The v2 writer also refuses to replace a detected v1 active file, so this recovery step cannot be bypassed by direct persistence. This preserves the historical receipt without allowing it to act as the current reconciliation baseline.

### Current operational boundary

- `canonical_sync` acquisition/store/Git mechanics and `llm_evaluation` GitHub/Git readback are implemented and deterministically tested. Dedicated local OAuth/ADC plus registered live Drive acquisition have been verified.
- PR #10 merged the tested portable local-auth helper/documentation path into `main` at `040715c91115a4adab29035bec4a38c76ac6846f`, with exact post-merge `Public CI` verification.
- A subsequent closure audit found that the existing durable v1 state bound the then-current Drive snapshot to older PR #11 evidence while a later helper obligation was still outstanding. The v1 state therefore cannot be treated as the authoritative current baseline. The repair requires an explicit exact publication tree, a non-empty obligation set, versioned v2 durable state, and explicit byte-preserving supersession of legacy v1 before corrected closure.
- The live `NONE` / `REQUIRED` / `BLOCKED` / `COMPLETE` disposition is Drive-owned governance state. Do not infer current completion merely from this repository document. Hosted GitHub OIDC -> Google WIF remains a later activation gate, and E-002 model sampling remains separately governed.
