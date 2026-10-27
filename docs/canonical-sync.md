# Canonical synchronization boundary

`llm-evaluation` treats Google Drive as the sole mutable authority for methodology and reconciliation state. The public repository contains executable code, tests, schemas, public-safe contracts, and deliberately published artifacts; it does not mirror private canonical bodies or Drive metadata.

The reusable mechanics are incubated in the workspace package `packages/canonical-sync` (import namespace `canonical_sync`). The governing boundary is: `canonical_sync` owns verified acquisition/snapshot mechanics and mechanical Git publication/readback of an already-approved immutable tree; `llm_evaluation` owns authority-role interpretation, material significance, public-safe projection construction, and the `NONE` / `REQUIRED` / `BLOCKED` / `COMPLETE` governance state machine. A `VerifiedSnapshot` is evidence, not authorization to mutate Git.

## Identity and canonical serialization

The package keeps three authority/acquisition identities separate:

- `content_sha256`: exact bytes for one acquired source representation;
- `source_state_id`: `canonical_record_hash(SnapshotStateRecord)`, binding registered source identity + meaning-bearing logical mapping + exact content identities. Existing persisted names such as `snapshot_state_sha256` refer to this same identity;
- `observation_id` / `receipt_id`: hashes of the observation and acquisition-receipt records, including provider revision/version observations, acquisition timing/attempts, and safe runner/transport/authentication provenance.

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

## Related design records

- [`canonical-sync` implementation design](design/2026-10-26-canonical-sync-design.md)
- [Prompt Research/prlab second-consumer compatibility check](canonical-sync-prlab-compatibility.md)
- [Hosted GitHub OIDC → Google WIF activation design](design/canonical-sync-oidc-wif-activation.md)
