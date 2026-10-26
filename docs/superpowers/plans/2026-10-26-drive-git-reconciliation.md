# Task 2 — Verified Drive → Git Reconciliation Implementation Plan

> **Execution:** inline on `feat/task2-drive-reconciliation`, based on public `main`. Google Drive remains the sole mutable `llm_evaluation` methodology authority. This plan contains no Drive object IDs, revisions, canonical text, credentials, or private metadata.

**Goal:** implement the smallest read-only Google Drive acquisition and public-reconciliation foundation authorized by D-016, proving stable native-Docs semantic acquisition and explicit `NONE` / `REQUIRED` / `BLOCKED` / `COMPLETE` state without building a bidirectional sync, canonical writer, hosted WIF workflow, or generic multi-project framework.

**Architecture:** registered private identity -> Drive PRE observation -> native Docs structural read -> strict semantic projection/hash -> Drive POST observation -> bounded movement decision -> private receipt/checkpoint -> explicit project-supplied public-impact classification -> publication filter / verified Git race boundary. Runtime/model code and authority-store code remain siblings with no imports in either direction.

**Verified API facts used:** Drive `version` is movement/provenance and changes for server-side changes even when not user-visible; native Docs must not use Drive `headRevisionId` as document identity. `documents.get` supports `drive.readonly`, all-tab content, and `PREVIEW_WITHOUT_SUGGESTIONS`. Docs `revisionId` is edit-access-only, opaque, short-lived, and therefore not a read-only acquisition requirement. Future mutation would require a fresh mutation-capable read and `requiredRevisionId`, but no writer is implemented here.

**Observed current canonical structure:** the three task-relevant native Docs inspected for this task each currently have one root tab, no child tabs, one leading continuous section break, and ordinary paragraph bodies. The inspected documents exposed no tables, table-of-contents blocks, headers, footers, footnotes, inline/positioned objects, or named ranges. The first projector therefore supports only this actual structure and fails closed on richer semantically relevant structures. Unreferenced Docs list-definition registries are not projected; a paragraph `bullet` reference is unsupported and fails closed.

## Task 2.1 — Dependency and strict contract boundary

- Add exact current Pydantic v2 and official Google Python client/auth dependencies.
- Add strict/frozen Pydantic contracts for private registered identity, capability, Drive/source observation, semantic state, acquisition receipt, last-reconciled state, reconciliation obligation, and verified publication receipt.
- Keep read-only and future mutation capability distinct. A read-only source observation has no Docs revision field and cannot gain mutation authority by attaching one.
- RED tests: strict coercion/extra-field rejection, no `headRevisionId`/`revisionId` in read-only records, all four reconciliation statuses distinct.

## Task 2.2 — Narrow official Google transport

- Implement one Drive v3 + Docs v1 transport using ADC with exactly `drive.readonly`.
- Request only the metadata/Docs fields needed by the verified contract.
- Explicitly request `PREVIEW_WITHOUT_SUGGESTIONS` and all-tab content.
- Validate all external response dictionaries with Pydantic before project logic consumes them.
- Sanitize authentication/transport exceptions so credentials, token paths, or provider response bodies cannot enter durable errors.
- RED tests: exact request options/fields, malformed metadata/document response rejected.

## Task 2.3 — Deterministic native-Doc semantic projection

- Support exactly one root tab, no child tabs, leading `CONTINUOUS` section break, then `NORMAL_TEXT` paragraphs composed only of text runs without link targets.
- Project semantic identity from a versioned canonical JSON record containing only the ordered visible base text. Drive version/name/ID/parent and operational facts do not enter the semantic hash.
- Fail closed on multiple/nested tabs, table/TOC, additional section breaks, bullets, non-normal named styles, headers/footers/footnotes, inline/positioned/named-range content, non-text paragraph elements, or link targets.
- RED tests: deterministic hash across irrelevant dict ordering; text change changes hash; metadata/version-only change does not; each unsupported structure is rejected.

## Task 2.4 — PRE/Docs/POST acquisition and bounded movement handling

- Validate exact registered object ID/name/MIME/parent and non-trashed state at PRE and POST.
- Acquire Docs content between PRE and POST. No cross-file atomicity claim.
- If PRE/POST differ, discard the mixed candidate and retry the whole transaction once; repeated movement raises a typed `SourceMovementError` and yields no semantic acquisition.
- Return semantic state separately from private source observation/operational receipt.
- RED tests: wrong identity/type/parent rejected; stable acquisition succeeds; one movement reacquires; repeated movement fails; metadata movement with identical semantic text is still an acquisition movement, not silently accepted.

## Task 2.5 — Explicit last-reconciled state and four-way classification

- Store only the smallest private checkpoint needed to answer what semantic hash was last verified as reconciled; do not put Drive IDs, revisions, source text, or credentials in public Git.
- Classification rules: unchanged semantic hash -> `NONE`; changed + project-declared private/no-public impact -> `NONE`; changed + explicit public/executable impact -> `REQUIRED`; changed + unknown/unsafe impact -> `BLOCKED`; only a matching verified publication receipt may move a `REQUIRED` obligation to `COMPLETE`.
- The authority-store layer never decides scientific meaning; public impact is an explicit caller/project judgment.
- RED tests: statuses cannot collapse; mismatched receipt cannot complete; branch edit alone cannot complete.

## Task 2.6 — Publication and Git-race boundary

- Derived public publication accepts only deliberately constructed public bytes, never a raw `SemanticProjection` object.
- Reject exact private markers supplied from the private acquisition context plus known credential/token shapes; test both safe and known-bad controls.
- Use atomic local replacement for a derived public artifact and verify readback.
- Provide an exact expected-vs-observed remote-base guard; movement blocks publication. Do not implement push/PR or force semantics in this task.
- RED tests: fake Drive IDs/parents/revisions/tokens rejected, safe payload allowed, atomic readback exact, remote movement rejected.

## Task 2.7 — Deterministic verification, then bounded local live gate

- Fresh `uv lock --check`, Ruff, mypy, complete unittest suite, branch coverage >=80%, package build/install smoke, and tracked/private-state audit.
- Only after deterministic gates pass, inspect the existing operator OAuth/ADC prerequisite without printing tokens. If exact `drive.readonly` local ADC is already available and registered `llm_evaluation` identity can be supplied from private state, perform one bounded read-only PRE/Docs/POST live acquisition and retain private evidence outside Git.
- Do not create/broaden OAuth credentials, run an interactive consent flow, share Drive objects, write Drive, push Git, activate hosted OIDC/WIF, or manufacture a public reconciliation commit in this session.

## Completion boundary for this session

Task 2 may be reported as **implemented/verified** for its deterministic foundation only after fresh gates pass. `live verified` requires the bounded local OAuth gate. `COMPLETE` reconciliation requires a real separately authorized public transaction and verified readback; a synthetic test demonstrates the state machine but is not a completed reconciliation.
