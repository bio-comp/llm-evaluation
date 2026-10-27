# canonical-sync second-consumer check: Prompt Research / prlab

## Purpose

This note checks the incubated `canonical_sync` API against the current Prompt Research/prlab acquisition and mirror implementation. Prompt Research remains read-only reference material here: this work does not modify prlab and does not make prlab an authority over `llm_evaluation`.

The question is narrow: which prlab behaviors are genuinely reusable mechanics, which are already represented by `canonical_sync`, and which requirements should remain project-specific or become extraction-gate follow-ups?

## Behavioral mapping

| prlab behavior | `canonical_sync` boundary | Result |
| --- | --- | --- |
| Read-only Drive transport with bounded transient retries and sanitized typed failures | `SourceTransport`, `GoogleDriveTransport` | Common mechanic represented. The incubated Google adapter currently uses Drive metadata plus exact native-Doc Markdown export bytes. |
| Declared source identity and parent/location checks | `AuthorityObjectSpecRecord`, `AuthorityProfileRecord`, acquisition validation | Common mechanic represented. |
| Exact folder membership where membership itself is authority | `ExactMembershipSpecRecord`, `list_members()`, PRE/POST membership comparison | Common mechanic represented without importing prlab's Research-tree semantics. |
| Whole-transaction restart when any source or membership moves | `acquire_stable_candidate()` | Common mechanic represented. Non-movement failures do not enter the movement retry loop. |
| Semantic state identity distinct from observation/provenance identity | `SnapshotStateRecord`, `SnapshotObservationRecord` | Common mechanic represented. Metadata-only churn can change observation identity without changing semantic identity. |
| Exact hashing and deterministic persisted records | canonical JSON codec plus exact representation SHA-256 | Common mechanic represented. |
| Portable path safety and materialization collision checks | strict logical-path validation with Windows-reserved/illegal-name checks plus Unicode NFC/casefold collision detection | Common mechanic represented after second-consumer review. |
| Private snapshot staging, exact inventory, symlink rejection, atomic publication, observation-last completion | `SnapshotStore` | Common mechanic represented. |
| Structured acquisition success/failure receipt | `AcquisitionReceiptRecord`, `acquire_and_publish()`, receipt store verification | Common mechanic represented after second-consumer review. Generic receipt excludes prlab-specific version/provider fields. |
| Durable reconciliation states that advance only after verified publication | generic reconciliation records/state machine | Common mechanic represented; project interpretation stays outside the package. |
| Prompt Research authority parser, scoped instruction extraction, and `Research/` membership semantics | prlab project layer | Intentionally **not** generic. These are Prompt Research semantics. |
| prlab Git remote/branch publication, remote-race checks, mirror-specific state, and branch workflow | prlab project layer | Intentionally **not** generic. `canonical_sync` stops at verified source/snapshot/reconciliation/publication primitives. |
| Prompt Research public mirror shape | prlab project layer | Intentionally **not** generic and is not copied into the public `llm-evaluation` design. |

## Important second-consumer gap: stored-file/raw-byte acquisition

prlab also acquires stored Drive blobs and verifies provider checksums where the provider exposes a meaningful checksum. The first `llm_evaluation` consumer currently needs native Google Docs only, and its proven representation is exact `text/markdown` bytes from Drive `files.export`.

Therefore the current generic package does **not** yet have a demonstrated raw stored-file/download representation adapter. This is not a defect in the `llm_evaluation` consumer and should not be added speculatively before extraction. It is, however, a concrete second-consumer requirement to implement during or immediately before prlab migration:

- add a versioned stored-file/raw-byte representation;
- use Drive download bytes exactly;
- verify provider checksum when that checksum applies to the stored representation;
- preserve the same PRE/read/POST and snapshot-store contracts;
- prove parity against prlab's existing adversarial tests before deleting prlab's implementation.

No provider fallback may silently substitute export bytes for raw stored bytes or vice versa.

## API pressure from the second consumer

The prlab comparison produced useful hardening without forcing project-specific abstractions into the package:

- stronger portable logical-path validation;
- exact membership/object-parent coherence validation;
- persisted acquisition receipts;
- Google identifier validation before query construction;
- structured, whitespace-independent provider error classification;
- receipt-directory and interrupted-staging adversarial checks.

Those changes improve the generic seam for both consumers and are now tested in `canonical-sync`.

## Extraction assessment

The API has now survived a real `llm_evaluation` local OAuth acquisition and a prlab second-consumer behavior mapping. The original technical extraction criteria are therefore substantially exercised locally: stable acquisition, all four reconciliation dispositions, a discriminating canonical-to-executable contract test, verified private publication, and second-consumer gaps are known.

**Recommended operational gate:** do not extract the package before this `llm-evaluation` branch has completed normal public Git review/CI and D-016 can be closed as `COMPLETE`. After that, propose a standalone `canonical-sync` repository/package, add the raw stored-file representation required by prlab under tests, and migrate prlab incrementally while retaining its current implementation as the behavioral comparator until parity is demonstrated.
