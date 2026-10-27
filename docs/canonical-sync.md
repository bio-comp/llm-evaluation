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

`fixtures/reconciliation_policy_contract.json` is deliberately public-safe. It records only the executable four-state reconciliation contract (`NONE`, `REQUIRED`, `BLOCKED`, `COMPLETE`) and the verified-publication conditions required for `COMPLETE`. Private Drive identities and canonical narrative are excluded by design.

For this project, `COMPLETE` is stronger than a generic Git receipt or a Boolean CI assertion. Project-owned evidence must bind the public `bio-comp/llm-evaluation` repository, the reviewed pull request, the merged `main` commit, post-merge Git readback of that exact commit, and the required `verify` CI run on the same commit. A synthetic record can exercise those invariants in tests; it does not prove that the real D-016 closure transaction occurred.

## Related design records

- [`canonical-sync` implementation design](design/2026-10-26-canonical-sync-design.md)
- [Prompt Research/prlab second-consumer compatibility check](canonical-sync-prlab-compatibility.md)
- [Hosted GitHub OIDC → Google WIF activation design](design/canonical-sync-oidc-wif-activation.md)

## Review hardening and historical records

Post-merge closure must call `GitPublisher.verify(tree, expected_commit=M)` on
`refs/heads/main`. This method reads and compares remote state; a mismatch raises
without creating a commit or attempting a push. Never use `publish()` as a
post-merge repair operation. The reviewed head H may differ from merged commit M;
PR merge evidence, main readback and successful `Public CI` / `verify` evidence
must bind M. Real closure must obtain these facts from authoritative GitHub state,
not hand-constructed records.

Publication rejects file/directory hierarchy collisions. The current publisher
also rejects any base tree containing non-100644 entries before committing or
pushing; it does not normalize executable files, symlinks or gitlinks.

Git receipt v3 stores a normalized repository endpoint without userinfo, query
or fragment; unsupported/local locators become opaque SHA-256 identities. Project
completion still checks the exact authorized public endpoint. Acquisition receipt
v4 and observation v3 preserve recognized coarse transport/auth categories and
numeric versions; other transport metadata becomes opaque digests. Raw exception
bodies and paths do not enter failure receipts. Operational transport and staging
I/O failures are typed; programming errors still propagate.

CI evidence v2 requires the explicit `Public CI` workflow identity. Project
publication v3 and disposition v2 version the changed nested evidence contracts.
The prior Git receipt v1→v2, acquisition receipt v2→v3 and project publication
v1→v2 corrections were already versioned. Historical payloads remain untouched;
current readers explicitly reject old schema versions and never infer missing
provenance or completion evidence. No historical private receipt is deleted or
migrated. Use the original versioned implementation to inspect historical records.

The live Drive authority requires project-owned last-reconciled state but does
not yet designate an exact storage object/path for `LastReconciledStateRecord`.
That home must be resolved before real closure; no public checkpoint is created
by this implementation or this review.
