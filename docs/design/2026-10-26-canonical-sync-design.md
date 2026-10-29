# canonical-sync in-repository incubation design

**Status:** implementation design for D-016; Google Drive remains the sole mutable `llm_evaluation` methodological authority.

**Scope:** replace the branch's first project-specific `llm_evaluation.authority_store` prototype with a reusable in-repository package named `canonical-sync` / import namespace `canonical_sync`, then consume it through thin `llm_evaluation` adapters. Prompt Research/prlab is prior art and a future second-consumer check, not an implementation dependency or authority.

## 1. Evidence and constraints

Current live Drive authority establishes the following boundaries:

- Drive owns mutable methodology, evidence admission, experiment definitions/status, decisions, supersession, and interpretation.
- Git owns public executable implementation, tests, schemas, public-safe fixtures/docs, and deliberately publishable immutable artifacts.
- D-016 authorizes read-only Drive acquisition and explicit Drive-to-Git reconciliation but does not declare the path active until local OAuth acquisition, mixed-state/hash/publication-boundary checks, explicit last-reconciled state, and Git readback/CI evidence exist.
- Canonical changes are classified for repository reconciliation as `NONE`, `REQUIRED`, `BLOCKED`, or `COMPLETE`; this is distinct from evidence-candidate statuses.
- Ordinary PR CI remains credential-free. Hosted GitHub OIDC -> Google WIF/service identity is a later activation gate.
- D-017's local-model/runtime/storage architecture is separate and must not enter this package.

Current Git inspection adds two constraints:

- public `main` is clean at `2436e562ee6aa9f29e5c6fd9235cf0eab258eeff`;
- `feat/task1-runtime` remains a separate worktree/branch, while this work is already isolated on `feat/task2-drive-reconciliation`.

The existing Task-2 prototype proves useful behavior, but its `llm_evaluation.authority_store` namespace and single-document design are the wrong extraction seam for the newly authorized reusable package.

Prompt Research/prlab demonstrates reusable failure modes that should survive here: bounded request retries, whole-transaction retries after source movement, semantic-state identity separate from observation identity, exact inventory verification, symlink/path rejection, observation-last publication, idempotent verification of existing targets, and durable state that advances only after verified publication. Its Prompt Research authority parser, Research-tree policy, wholesale canonical mirror, and project-specific Git branch semantics do not transfer.

## 2. Selected architecture

Use a uv workspace with one root application and one reusable member package:

```text
llm-evaluation/
├── packages/
│   └── canonical-sync/
│       ├── pyproject.toml
│       ├── src/canonical_sync/
│       │   ├── __init__.py
│       │   ├── codec.py
│       │   ├── domain.py
│       │   ├── errors.py
│       │   ├── records.py
│       │   ├── transport.py
│       │   ├── acquisition.py
│       │   ├── store.py
│       │   ├── git_publication.py
│       │   └── google_drive.py
│       └── tests/
├── src/llm_evaluation/
│   ├── canonical_authority.py
│   └── canonical_reconciliation.py
├── tests/
│   └── test_canonical_reconciliation.py
├── pyproject.toml
└── uv.lock
```

The exact installed `uv 0.8.4` accepts `[tool.uv.workspace] members = ["packages/canonical-sync"]` plus a root `[tool.uv.sources] canonical-sync = { workspace = true }` dependency. No submodule or second repository is introduced.

Dependency direction is one-way:

```text
canonical_sync  <-  llm_evaluation.canonical_authority
                <-  llm_evaluation.canonical_reconciliation
```

`canonical_sync` must never import `llm_evaluation`, prlab, experiment IDs, model/runtime policy, or project-specific canonical names.

### Alternatives rejected

1. **Keep the current `llm_evaluation.authority_store` prototype.** Rejected because it bakes the first consumer into the reusable mechanics and makes later extraction/refactoring needlessly invasive.
2. **Copy prlab's snapshot/mirror modules.** Rejected because they contain Prompt Research authority semantics, blob-only assumptions, Research membership rules, and a wholesale mirror publication model that violates this public/private boundary.
3. **Create a standalone repository now.** Rejected by the extraction gate: the generic API must first survive real `llm_evaluation` use and a prlab second-consumer mapping.
4. **Automatically parse all canonical narrative into code.** Rejected because Drive narrative is authority for interpretation, not an implicit code generator. Project adapters must make public/executable implications explicit.

## 3. Boundary model rule

Use Pydantic v2 only at untrusted, configurable, serialized, or persisted boundaries. Boundary models are strict (`extra="forbid"`, `strict=True`), frozen where persisted, and explicitly versioned.

Examples:

- authority profile/config loaded from TOML/JSON/environment;
- normalized Google API wire responses;
- snapshot state and observation manifests;
- acquisition receipts;
- generic Git publication receipts;
- consumer-owned persisted reconciliation records in `llm_evaluation`.

After validation, trusted domain state is carried by `@dataclass(frozen=True, slots=True)` objects. Small `Protocol` interfaces define transport and project-decision dependencies.

No handwritten JSON Schema is maintained. Pydantic remains the schema authority for persisted records and can generate JSON Schema when needed.

## 4. Authority profile and exact identity

A generic authority profile declares a transaction's exact expected source set. Each object specification contains only mechanics:

- logical key;
- provider kind (`google_drive` for the first adapter);
- opaque object ID;
- expected name;
- expected MIME type;
- expected parent ID when location is authoritative;
- representation schema to acquire;
- required/optional flag (the first `llm_evaluation` profile uses required objects only).

Object IDs and parent IDs are private runtime configuration and are never committed to public Git.

The profile rejects duplicate logical keys, duplicate source identities where disallowed, unsafe logical paths, and unexpected fields. A deterministic `authority_spec_sha256` binds exact object/location expectations without requiring source IDs in public outputs.

An optional exact-membership specification is supported because prlab demonstrates that folder membership can itself be authority. When enabled, the transport must list direct children and the transaction rejects missing, extra, moved, or reparented members. `llm_evaluation` need not enable membership checks unless its registered profile requires them.

## 5. Transport protocol and Google adapter

Trusted protocol surface:

```text
observe(object_spec) -> SourceObservation
read(object_spec) -> SourceContent
list_members(parent_id) -> tuple[SourceObservation, ...]  # only when requested
```

`SourceObservation` is a frozen dataclass containing normalized identity/movement facts. `SourceContent` is a frozen dataclass containing deterministic representation bytes and the representation schema.

Transport exceptions are typed: authentication, permission, not-found, unsupported representation, permanent provider failure, and transient provider failure. Provider response bodies, credential paths, tokens, and raw exception text do not enter durable records.

### Google Drive adapter

The first Google adapter uses only `drive.readonly` and the official Drive v3 API.

For native Google Docs, the representation is `google-drive-export-markdown-v1`: the exact bytes returned by Drive `files.export` with MIME type `text/markdown`. The package does not normalize, reformat, or parse those bytes before hashing. These are deterministic export-representation bytes, not raw native-Docs storage bytes. Live Stage-A verification showed repeated Markdown exports were byte-stable for all four registered D-016 objects while their Drive versions were unchanged. The same local OAuth client had the Docs API disabled, so requiring Docs v1 would add an unnecessary second service dependency without improving the authority guarantee.

For movement identity, Drive `version` is used as a monotonic server-side change observation together with object ID, name, MIME type, modified time, parent IDs, trashed state, and relevant capabilities. PRE/POST equality is required. No Docs `revisionId` is required; semantic identity is the declared authority profile plus exact exported Markdown bytes, while provider observations remain distinct provenance.

Provider-level transient retries are bounded and distinct from whole-transaction source-movement retries. Permanent/auth/identity/representation/checksum failures are not retried as movement.

## 6. Deterministic serialization and identity

Persisted records use canonical UTF-8 JSON for storage/readback, while cryptographic identity uses a separate explicit framing protocol:

- Pydantic validates record structure; it does **not** define cryptographic provenance bytes;
- canonical payload JSON walks `model_dump(mode="python")` through an explicit allowed-scalar canonicalizer;
- timezone-aware datetimes normalize to UTC RFC 3339 with fixed microsecond precision and `Z`;
- floating-point identity scalars are rejected rather than inheriting runtime/Pydantic formatting;
- mapping keys are lexicographically sorted; set-like collections are normalized by record validators before serialization;
- compact separators `,` and `:` are used, Unicode is preserved, encoding is UTF-8, and canonical payloads have **no trailing newline**;
- canonical parser reserializes and rejects equivalent-but-noncanonical JSON;
- each identity-bearing durable record type declares a non-serialized `canonical_identity_domain` and positive `canonical_identity_version`;
- `canonical_record_hash()` hashes an unambiguous identity frame: fixed `canonical-sync` identity magic + identity-protocol version + length-framed domain + domain version + record schema version + length-framed canonical payload bytes. This prevents accidental cross-record collisions even when two Pydantic payload shapes serialize identically.

The normative identities are:

1. **`content_sha256`** — SHA-256 of the exact acquired representation bytes for one source.
2. **`source_state_id`** — `canonical_record_hash(SnapshotStateRecord)`. It binds `authority_spec_sha256` (registered source identities, locations/membership constraints, and meaning-bearing logical mapping) plus each logical representation schema and `content_sha256`, in deterministic order. Acquisition time, provider version, retries, runner, transport, and authentication provenance are excluded. The current persisted field/path spelling `snapshot_state_sha256` is the storage name for this same `source_state_id`; it is not a second identity.
3. **`observation_id`** — `canonical_record_hash(SnapshotObservationRecord)`. It binds the `source_state_id` to provider revision/version observations, acquisition time/attempt, and safe runner/Python/transport/authentication provenance. The current store/API spelling `observation_sha256` is this `observation_id`.
4. **`receipt_id`** — `canonical_record_hash(AcquisitionReceiptRecord)`, identifying one persisted acquisition receipt.

A one-byte representation change changes `content_sha256` and therefore `source_state_id`. A different registered Drive object with identical bytes, or a changed meaning-bearing logical mapping with the same object/bytes, also changes `source_state_id` through `authority_spec_sha256`. Reacquiring the same registered objects/mapping/bytes preserves `source_state_id` even when provider versions, time, runner, transport, or ADC credential category change; those changes instead affect `observation_id` and/or `receipt_id`. Renaming the private operator-facing profile label alone does not change the authority specification identity.

## 7. Stable multi-object acquisition

A transaction attempt is whole-profile, never file-by-file publication:

1. validate the complete authority profile;
2. PRE-observe every declared object and any exact membership set;
3. validate exact IDs/names/MIME/parents/non-trashed state;
4. acquire deterministic representation bytes for every object into private staging;
5. hash staged bytes and verify any provider checksum that is meaningful for that representation;
6. POST-observe every object and membership set;
7. compare PRE vs POST movement fingerprints and membership exactly;
8. on any movement, destroy the entire staged candidate and retry the whole transaction up to the configured bound;
9. on stability, construct semantic state and observation manifest;
10. verify state/manifest consistency before the store can publish them.

There is no claim that Google Drive supplies cross-object atomicity. The PRE/read/POST protocol is a fail-closed approximation that refuses a mixed observation.

Only `SourceMovementError` triggers transaction-level retry. Request-level transient errors are handled by the transport's smaller retry loop. Exhausted movement becomes a durable typed failure; no snapshot is published.

## 8. Snapshot store and filesystem safety

The snapshot store is private working evidence, not a competing canonical. It lives beneath a caller-supplied private root and has separate `staging/`, `snapshots/`, `observations/`, and `receipts/` directories plus an exclusive lock.

Safety invariants:

- reject symlinked store roots, operational directories, locks, files, and ancestors where relevant;
- reject traversal/absolute/empty logical paths, Windows-reserved/illegal materialization names, unsafe trailing/control characters, and portable Unicode-NFC/case-fold collisions;
- reject non-regular unexpected filesystem entries;
- stage inside the store only;
- fsync staged files before publication;
- publish semantic snapshot directory before observation directory;
- use same-filesystem atomic rename under an exclusive lock;
- fsync parent directories after rename;
- verify exact inventory and hashes after publication;
- an existing target is a no-op only after complete byte/inventory verification;
- an existing corrupt or conflicting target is a hard integrity failure and is never overwritten;
- an orphan semantic snapshot without its verified observation is not a completed acquisition;
- interrupted staging remains incomplete and cannot be mistaken for verified state.

Each acquisition run also persists a versioned `AcquisitionReceiptRecord` as auxiliary private provenance. Success and failure receipts share the same safe runner/Python/transport/authentication provenance contract. A successful receipt additionally binds the authority-profile identity, transaction-attempt count, exact `source_state_id`, and exact `observation_id`. A failed receipt records that safe provenance, the typed sanitized error code, and attempt count, but never raw provider exception text, credential bodies/paths, or successful snapshot identities. Receipt files are hash-addressed, canonically serialized, verified on readback, and subject to the same non-symlink/exact-inventory store boundary.

## 9. Consumer-owned reconciliation state

`canonical_sync` does **not** own project reconciliation meaning or the `NONE` / `REQUIRED` / `BLOCKED` / `COMPLETE` vocabulary. The reusable package ends at verified acquisition evidence plus mechanical Git publication/readback receipts.

`llm_evaluation.canonical_reconciliation` owns the persisted project records and rules that bind:

- current `source_state_id`;
- previous verified `source_state_id` when present;
- project decision (`no_public_impact`, `public_change_required`, or `unknown_or_unsafe`);
- resulting disposition `NONE`, `REQUIRED`, or `BLOCKED`;
- project-approved contract/projection identity plus project-owned PR-merge evidence, post-merge `main` Git readback for `bio-comp/llm-evaluation`, and the required CI check bound to that same merged commit for `COMPLETE`.

Project rules remain:

- same `source_state_id` -> `NONE`;
- changed + explicit no-public-impact -> `NONE`;
- changed + explicit public/executable impact -> `REQUIRED`;
- changed + unknown/unsafe impact -> `BLOCKED`;
- only matching project bindings around a mechanically verified Git receipt can transition a matching `REQUIRED` obligation to `COMPLETE`.

A changed file is therefore not synonymous with `REQUIRED`. A `VerifiedSnapshot` is evidence, not authorization, and a branch edit is not synonymous with `COMPLETE`.

## 10. Project adapter boundary

`llm_evaluation.canonical_authority` owns:

- loading a private, strict `llm_evaluation` authority profile;
- selecting which registered canonicals a transaction needs;
- constructing the generic acquisition request;
- exposing verified snapshot handles to project reconciliation without publishing private bodies.

`llm_evaluation.canonical_reconciliation` owns:

- human/project-reviewed interpretation of a verified snapshot;
- whether a semantic change has no public consequence, requires a public implementation change, or is unsafe/unclear;
- constructing a narrow public-safe contract/fixture/test/docs consequence;
- verifying the consequence against the intended executable implementation.

The adapter never treats arbitrary changed narrative as executable instructions.

## 11. Consumer authorization and generic Git publication boundary

Publication is deliberately split across the trust boundary:

1. **Consumer authorization:** project-specific logic interprets verified evidence and constructs the exact immutable path/byte projection approved for Git. `canonical_sync` never transforms `VerifiedSnapshot` content into a public artifact and never decides whether material is public-safe or significant.
2. **Generic Git mechanics:** `GitPublisher` accepts only the already-constructed `PublicationTree`, whose current contract is a caller-approved projection over the exact verified expected base. Omitted base paths are preserved and are never implicit deletion authorization. The publisher verifies the expected remote base, mechanically overlays only approved paths, enforces fast-forward-only publication, detects remote movement, and verifies the resulting commit/tree by remote readback. Explicit deletion is not part of the current projection API.

There is no generic project-policy callback and no generic secret/private-content classifier. Public-safety policy lives in the consumer's narrow projection schema and construction path. Generic path/inventory validation remains mechanical Git safety only.

No Drive IDs, parent IDs, modified times, tokens, local credential paths, or private canonical narrative are required in the `llm_evaluation` publication tree.

## 12. Canonical -> executable regression vertical slice

The first vertical slice is the public reconciliation-policy contract itself, not E-002 scientific content.

A public-safe project fixture represents only the admitted contract facts needed by code: the four reconciliation dispositions and the rule that `COMPLETE` requires a verified publication receipt bound to the exact semantic identity. The project adapter consumes a controlled verified-snapshot fixture and produces this narrow contract.

The test has a deliberate counterfactual: mutate the authoritative public-safe projection so the executable implementation is stale. The contract test must fail against the stale implementation, then pass after reconciliation. A separate unchanged/private-only change must produce `NONE` and no executable obligation.

This proves the feedback mechanism without copying Drive narrative into Git or redesigning E-002.

## 13. Local OAuth and live integration boundary

The live integration test is project-specific and opt-in:

- marker: `live_drive`;
- explicit environment opt-in required;
- private authority profile path supplied outside Git;
- credentials discovered through controlled local ADC/OAuth with exactly `drive.readonly` requested;
- test acquires the registered `llm_evaluation` profile, publishes to a temporary private store, reopens and verifies it, and checks expected fail-closed behavior without writing Drive;
- no token, credential body, Drive ID, canonical body, or private snapshot path is printed on success or failure.

Ordinary PR CI runs pytest with `live_drive` excluded and has no Google credentials. The Google client libraries remain an optional package extra used only for live Drive acquisition.

Hosted GitHub OIDC -> Google WIF/service identity is deliberately not implemented or activated in this tranche. After local OAuth is live-verified, a separate activation design can reuse the same transport/profile boundary.

## 14. CI and packaging

The root uv workspace owns one lockfile. `canonical-sync` is a real buildable package and the root `llm-evaluation` project depends on the workspace member rather than importing through path tricks.

The new package uses pytest. Existing root unittest tests are not rewritten merely for style. The combined CI can invoke pytest because pytest collects unittest tests as well; until the separate runtime branch is integrated, changes here must avoid silently recreating Task-1 runtime work.

Package checks include:

- `uv lock --check`;
- package-scoped pytest including adversarial tests;
- complete repository test collection with the live marker excluded;
- Ruff across root + workspace package;
- mypy across root + workspace package;
- build both distributions and install-smoke them in clean environments;
- verify base `llm-evaluation` install does not unexpectedly require Google client libraries unless the Drive extra is requested.

## 15. Extraction seam and gates

The in-repository package remains incubated until all extraction gates are met:

1. a real D-016 `llm_evaluation` acquisition/reconciliation transaction uses it;
2. local OAuth acquisition is live-verified;
3. all four reconciliation dispositions are mechanically exercised;
4. the canonical-to-executable regression slice is discriminating;
5. the API survives real `llm_evaluation` use;
6. prlab is mapped against the API and remaining generic requirements are understood.

Only then should a standalone `canonical-sync` repository be proposed. prlab migration happens after extraction and keeps its current implementation as a behavioral comparator until parity is demonstrated.

## 16. Design self-review

### Strongest falsifiers considered

- **Native Docs do not expose durable read-only `revisionId`.** The design does not depend on it; PRE/POST Drive observations plus deterministic representation hashing are the read-only stability check.
- **A provider version can move without visible semantic content changing.** That movement invalidates the current attempt, but after reacquisition the semantic snapshot can remain identical while observation identity changes. This preserves both safety and no-op semantics.
- **Cross-file atomicity is unavailable.** The design never claims it; any observed movement rejects the whole candidate.
- **A generic secret scanner cannot prove publication safety.** The consumer owns the narrow approved projection schema and constructs exact bytes; `canonical_sync` performs only mechanical Git/path/inventory verification and does not classify project content as public or private.
- **A new generic package could become a framework project.** The first API is constrained to behavior required by `llm_evaluation` plus concrete prlab prior art; no experiment/model/prompt semantics enter it.
- **A wholesale Drive mirror would simplify reconciliation.** It is rejected because this repository is public and D-016 explicitly forbids mirroring private canonical bodies/metadata merely for convenience.

### Corrections to the existing Task-2 prototype

- move reusable mechanics out of `llm_evaluation.authority_store` into `canonical_sync`;
- replace single-object acquisition with a stable multi-object transaction;
- separate strict persisted Pydantic records from trusted dataclass state;
- add a verified private snapshot store rather than only returning in-memory projection state;
- use pytest for the new package;
- make project semantic classification and public projection explicit adapters;
- add a discriminating canonical-to-executable regression slice;
- add an opt-in real `llm_evaluation` local OAuth test;
- keep Google clients optional and ordinary CI credential-free;
- do not implement or activate hosted OIDC/WIF until the local gate passes.

No Drive canonical change is required merely to record this design: it implements the architecture already authorized by D-016 rather than changing that authority.

## 17. 2026-10-28 Design amendment: authoritative GitHub closure and pre-merge hardening

Following the 2026-10-28 D-016 clarification in Google Drive and pre-merge review of PR #9, the following hardening and boundary amendments are incorporated:

### 17.1 Authoritative external evidence vs internal record validity

Syntactically valid or internally consistent Pydantic records do not prove external real-world events. In particular:
- An open PR can expose a synthetic test merge SHA, so `merge_commit_sha` is not evidence of actual merge; `merged == True` and explicit `merged_at` are required.
- PR-triggered CI runs execute untrusted branch code and cannot attest post-merge `main` state; evidence must bind an authoritative `push` event on `refs/heads/main` at `head_sha == merge_commit_sha`.
- Required job identity must bind exact workflow path (`.github/workflows/ci.yml`), workflow name `Public CI`, and job name `verify` with `conclusion == success`.

Project-owned closure acquisition in `llm_evaluation.canonical_github` implements this boundary:
`GitHub REST API` -> `strict Pydantic wire models` -> `project verification against expected state` -> `frozen trusted runtime facts` -> `durable Pydantic evidence records` (`GitHubPullRequestMergeRecord` v2, `GitHubCIVerificationRecord` v3) -> `VerifiedPublicationReceiptRecord` v4 -> `COMPLETE`.
Network operations remain in project transport adapters, not inside Pydantic validators.

### 17.2 Server-enforced atomic CAS Git publication lease

`GitPublisher.publish()` enforces atomic compare-and-swap on the remote ref using server-enforced leases:
- `--force-with-lease=<ref>:<expected_base>` when an expected base exists;
- `--force-with-lease=<ref>:` (empty expectation) for target branch creation (`expected_base is None`), failing if the ref was created concurrently.
In addition, the candidate publication commit must mechanically declare the exact expected base as its single parent, ensuring fast-forward semantics cannot be bypassed by an explicit lease. Competing remote rewinds, deletions, or concurrent updates raise `RemoteRaceError` and leave the remote untouched.

### 17.3 SnapshotStore operational error sanitization

Low-level filesystem `OSError` exceptions during store operations (directory creation, file read/write, fsync, rename, symlink checks) are sanitized into `SnapshotIntegrityError` without leaking private local filesystem paths, store root paths, or internal diagnostics in `str(exc)` or `__cause__` (`raise ... from None`). Path traversal defenses reject symlinks across store operations while safely accommodating standard platform root directory aliases (such as macOS `/tmp` and `/var` owned by root).

### 17.4 Base-tree path and portable filesystem parity

`GitPublisher._read_tree()` enforces portable path hygiene across the remote base tree and candidate publication tree by rejecting case-folding collisions and Unicode normalization collisions (NFC vs NFD forms), guaranteeing deterministic cross-platform compatibility on case-insensitive and Unicode-normalizing filesystems.

### 17.5 Narrow D-016 authority-profile parent invariant

The D-016 authority profile validator enforces that all four required root authority roles (`README.md`, `DECISIONS.md`, `EXPERIMENTS.md`, `REVIEW_LEDGER.md`) share the exact same non-null Drive parent ID, guaranteeing that root governance canonicals originate from the single authorized project folder without generalizing this invariant to secondary canonical objects.
