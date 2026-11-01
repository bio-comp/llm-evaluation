# Hosted trusted acquisition: GitHub OIDC → Google WIF activation design

**Status:** design only. Local/manual Stage A is verified. This document does not create Google Cloud resources, alter Drive sharing, add credentials, or activate an unattended workflow.

**Package boundary:** everything in this document that provisions GitHub OIDC, WIF/IAM, a service account, or Drive sharing is deployment/runbook work outside `canonical_sync`. The package sees only standard `google-auth`/ADC credentials with `drive.readonly`; local authorized-user ADC and hosted WIF/service-account ADC converge on that same package-side credential interface.

## Decision

For the `llm_evaluation` Google Drive authority, use:

```text
trusted GitHub Actions workflow
        |
        | GitHub OIDC token
        v
Google Workload Identity Federation provider
        |
        | short-lived service-account impersonation
        v
dedicated acquisition service account
        |
        | OAuth access token restricted to drive.readonly
        v
Google Drive v3
        |
        +-- exact registered metadata observations
        +-- exact text/markdown exports
```

Do **not** use a static service-account key.

For this Google Workspace/Drive use case, prefer WIF **through a dedicated service account**, rather than direct WIF resource access. Direct WIF authorization is designed for Google Cloud resources that support IAM principals. Google Workspace assets use service-specific authorization; service accounts can authenticate to Workspace APIs and can be granted access to Workspace assets. The dedicated service account therefore provides the stable Drive sharing identity while WIF removes the long-lived key.

Sources used to finalize this design:

- Google Cloud: Workload Identity Federation with deployment pipelines — https://cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines
- Google Cloud: Workload Identity Federation — https://cloud.google.com/iam/docs/workload-identity-federation
- Google Cloud: Service accounts overview — https://cloud.google.com/iam/docs/service-account-overview
- GitHub: Configuring OIDC in Google Cloud Platform — https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-google-cloud-platform
- `google-github-actions/auth` — https://github.com/google-github-actions/auth

## Trust boundary

Ordinary `push`/`pull_request` CI remains unchanged and credential-free. It must not receive `id-token: write`, a workload-identity provider identifier, service-account identity, private authority profile, or Drive credentials merely to run tests.

Trusted acquisition is a **separate workflow/job boundary**. Initial activation should be `workflow_dispatch` only. Scheduling is a later explicit decision after trusted manual runs are demonstrated.

The trusted job gets only:

```yaml
permissions:
  contents: read
  id-token: write
```

`id-token: write` permits requesting an OIDC token; it does not itself grant repository or Google resource write permission.

## Google-side identity

Create one dedicated service account for this acquisition workload. Do not reuse a broad deployment, personal, or cross-project service account.

The WIF provider should map at minimum:

- `google.subject = assertion.sub`;
- repository owner;
- repository identity;
- workflow/ref claims needed by the admission condition.

The provider attribute condition must fail closed to the intended GitHub organization/repository and trusted execution context. At activation time, prefer an exact repository identity plus the specific trusted workflow/ref rather than only an organization-wide condition.

A conceptual condition is:

```text
repository owner is the expected organization
AND repository is the expected bio-comp/llm-evaluation repository
AND ref/workflow matches the explicitly trusted acquisition context
```

Use claims exactly as emitted by GitHub at activation time; do not copy a stale claim expression without inspecting a real token/debug-safe claim set.

Grant only the selected federated principal permission to impersonate the dedicated service account (`roles/iam.workloadIdentityUser`, or the minimum equivalent required by the chosen WIF/service-account flow). Do not grant the whole pool indiscriminately.

## Drive authorization

The service account must receive only enough Drive visibility to read the registered D-016 authority. The activation test must establish the narrowest workable sharing topology rather than assuming it:

1. first test sharing only the exact registered canonical objects with the dedicated service account;
2. verify that the API still returns the parent/location observations required by the authority profile;
3. if exact-object sharing cannot expose required parent identity, document that limitation before broadening access;
4. do not share an entire broader canonical tree merely for convenience.

Do not use domain-wide delegation for this read-only acquisition unless a later, separately justified requirement demonstrates that direct service-account sharing cannot satisfy the exact registered authority. Domain-wide delegation would materially broaden the trust boundary.

The OAuth/API scope remains exactly:

```text
https://www.googleapis.com/auth/drive.readonly
```

The hosted path uses the same `GoogleDriveTransport` and `google-drive-export-markdown-v1` representation already live-verified locally. It does not add the Google Docs API.

## Workflow behavior

The first trusted hosted workflow should:

1. check out the exact commit under test;
2. authenticate through GitHub OIDC and WIF/service-account impersonation using a SHA-pinned authentication action;
3. materialize the private authority profile from protected configuration without printing it;
4. run the opt-in live acquisition against a private ephemeral snapshot store;
5. verify stable PRE/read/POST acquisition, `source_state_id` / `observation_id`, store readback, `receipt_id`, and controlled negative failure behavior;
6. emit only a deliberately sanitized summary/artifact containing no Drive IDs, parent IDs, canonical bodies, OAuth material, credential-file contents, or user-specific paths;
7. destroy ephemeral credential files/private snapshot state at job completion.

The workflow must not write Drive, update Drive reconciliation state, open a PR, merge code, or schedule itself merely because authentication succeeds.

## Activation tests

Before considering hosted acquisition active, prove all of the following:

- intended manually dispatched trusted run can authenticate and read the exact registered authority;
- an untrusted pull-request/fork context cannot obtain the Google identity;
- a wrong repository/ref/workflow claim is rejected by the provider condition;
- service-account Drive visibility is no broader than the documented sharing grant;
- `drive.readonly` is the only requested Drive scope;
- private profile, IDs, canonical bodies, credentials, and snapshots do not appear in logs or uploaded public artifacts;
- hosted acquisition produces the same `source_state_id` as a temporally matched local acquisition when the authority has not changed;
- source movement and authority mismatch still fail closed;
- revoking the WIF/service-account binding or Drive sharing reliably removes access.

## Workflow pinning and supply-chain controls

When the workflow is implemented:

- pin `actions/checkout` and Google authentication actions to reviewed full commit SHAs;
- keep `persist-credentials: false` for checkout unless a later action requires otherwise;
- ignore any generated `gha-creds-*.json` files in Git/artifact packaging;
- do not cache credential files;
- avoid third-party actions in the trusted job unless independently justified and pinned;
- keep the job's repository permissions at `contents: read` unless another permission becomes demonstrably necessary.

## Scheduling gate

Do not add `schedule:` during initial activation. After multiple successful manual trusted runs, decide separately whether scheduled acquisition is useful. A scheduled read may detect authority movement, but it must not automatically turn Drive narrative into Git changes or mark reconciliation `COMPLETE`.

A future scheduled flow can create a durable **reconciliation observation/obligation**, but public code changes still require the explicit project adapter, branch/PR review, credential-free CI, and project-owned evidence binding the merged commit to post-merge Git readback and that same commit's CI before Drive reconciliation closure.

## Revocation and incident response

The hosted path must be easy to disable without touching the canonicals. Independent revocation points are:

- disable/delete the trusted GitHub workflow;
- disable/delete the WIF provider or tighten its attribute condition;
- remove the federated principal's service-account impersonation binding;
- remove the service account's Drive sharing access;
- disable the dedicated service account.

No long-lived private key exists to rotate.

## Activation gate

This design is ready for a separate activation task. That task must re-check current GitHub/Google documentation, current repository workflow state, and current Drive sharing requirements before making external configuration changes. It must not infer successful authorization from local OAuth evidence: hosted WIF/service-account access needs its own live verification.
