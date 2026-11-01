# Local Google Authentication Runbook for canonical-sync Consumers

This document specifies how an operator provisions local Google Drive Application Default Credentials (ADC) for consumer applications (such as `llm_evaluation` or Prompt Research) that use the reusable `canonical-sync` package.

`canonical-sync` provides verified Google Drive → Git mechanics across multiple independent consumers. To preserve that architectural boundary, **all credential provisioning, OAuth client creation, and interactive consent remain operator deployment responsibilities outside the `canonical-sync` package**.

---

## 1. Responsibility Boundary

The layers of the authentication architecture are strictly separated:

| Layer | Responsibility | What It Owns | What It Must NOT Own |
|---|---|---|---|
| **`canonical-sync`** | Reusable package mechanics | Standard `google-auth` / ADC consumption; requesting exact `drive.readonly` scope; sanitized transport error mapping; safe auth provenance classification (`adc:authorized-user`, `adc:workload-identity`, `adc:service-account`). | OAuth client IDs/secrets; interactive consent flows; token persistence; well-known path selection; consumer project naming or credential paths. |
| **Consumer Application** (e.g. `llm_evaluation`) | Project authority & policy | Defining registered canonical objects; enforcing D-016 authority profiles and invariants; public-safe projection rules; opt-in live test harness. | Hard-coded operator credentials; hosting shared OAuth clients across unrelated projects. |
| **Operator / Deployment** | Credential provisioning | Google Cloud project selection/creation; Drive API enablement; OAuth consent screen and test user management; Desktop OAuth client creation and secure local storage; interactive browser authorization; ADC configuration. | Committing credentials or tokens to Git; using broader scopes than required. |

---

## 2. Multi-Consumer Isolation Invariant

The core operational principle is:

```text
one canonical-sync package
  ≠ one Google Cloud project
  ≠ one OAuth application
  ≠ one Desktop OAuth client
  ≠ one local ADC credential file
  ≠ one authority profile
```

Multiple independent consumers invoke the same generic `canonical_sync.google_drive.GoogleDriveTransport` interface, each providing their own isolated credential context and consumer-specific authority profile:

```text
                         canonical-sync
                    generic google-auth / ADC
                            interface
                                |
             +------------------+------------------+
             |                                     |
     Prompt Research                        llm_evaluation
  consumer/deployment policy             consumer/deployment policy
  (historical first consumer)              (second consumer)
             |                                     |
~/.config/prlab/                       ~/.config/llm-evaluation/
prompt-research-prlab-oauth.json       llm-evaluation-oauth.json
             |                                     |
default gcloud location                isolated CLOUDSDK_CONFIG
~/.config/gcloud/                      ~/.config/gcloud-llm-evaluation/
application_default_credentials.json   application_default_credentials.json
             |                                     |
Prompt Research Profile                llm_evaluation D-016 Profile
```

The same human Google account may legitimately authorize both consumer applications. Isolation is enforced at the **client, configuration directory, and authority profile levels**, preventing accidental cross-talk or silent fallback between projects.

### Worked Multi-Consumer Reference

- **Consumer A (Prompt Research — historical first consumer)**:
  - Google Cloud Project: `prompt-research`
  - OAuth application: `Prompt Research prlab`
  - Desktop OAuth client: `Prompt Research prlab - Mac`
  - Client JSON: `~/.config/prlab/prompt-research-prlab-oauth.json`
  - ADC configuration: Default gcloud location (`~/.config/gcloud/application_default_credentials.json`). In historical single-consumer operation, Prompt Research used the default gcloud ADC path.
  - Authority Profile: Prompt Research registry / canonical profile
- **Consumer B (`llm_evaluation` — second consumer)**:
  - Google Cloud Project: `llm-evaluation` (e.g. project ID `llm-evaluation-513408`; project identity is an operator choice and not a package constant)
  - OAuth application: `LLM Evaluation`
  - Desktop OAuth client: `LLM Evaluation - Mac`
  - Client JSON: `~/.config/llm-evaluation/llm-evaluation-oauth.json`
  - ADC configuration: Dedicated isolated context (`CLOUDSDK_CONFIG="$HOME/.config/gcloud-llm-evaluation"`, generating `~/.config/gcloud-llm-evaluation/application_default_credentials.json`)
  - Authority Profile: `~/.config/llm-evaluation/llm-evaluation-d016.toml`
  - Scope: `https://www.googleapis.com/auth/drive.readonly`

### Recommended Future Multi-Consumer Pattern

```text
Prompt Research:
historical first-consumer setup used default ADC location (~/.config/gcloud/)

llm_evaluation:
second consumer deliberately used isolated CLOUDSDK_CONFIG (~/.config/gcloud-llm-evaluation/)

recommended future multi-consumer pattern:
prefer isolated ADC contexts (via CLOUDSDK_CONFIG) for all concurrent consumers
to prevent accidental credential cross-talk or silent fallback
```

Both consumers consume identical `canonical_sync` code without changes to package logic.

---

## 3. Reference Google Console / Google Auth Platform Setup

For each consumer application, perform this one-time setup in the Google Cloud Console:

1. **Select or Create a Google Cloud Project**:
   - Use a dedicated project for the consumer (e.g., `llm-evaluation`), or a deliberately designated shared infrastructure project.
2. **Enable the Google Drive API**:
   - Navigate to **APIs & Services** → **Library** → search for **Google Drive API** → click **Enable**.
3. **Configure Google Auth Platform / OAuth Consent Screen**:
   - Navigate to **Google Auth Platform** (or **APIs & Services** → **OAuth consent screen**).
   - **User Type / Audience**: Select **External** (for personal Google accounts).
   - **App Information**: Set App name (e.g. `LLM Evaluation`), User support email, and Developer contact information.
   - **Publishing Status**: Leave in **Testing** for bounded local development.
   - **Test Users**: Under Audience / Test users, click **+ Add Users** and add the operator's Google email address (e.g. `operator@example.com`). Save changes.
4. **Configure Scopes in Google Auth Platform**:
   - Navigate to **Google Auth Platform** → **Data Access** → click **Add or remove scopes**.
   - Filter or manually select the exact scope:
     ```text
     https://www.googleapis.com/auth/drive.readonly
     ```
   - Confirm it appears in the **Restricted scopes** summary and click **Update** / **Save**.
5. **Create OAuth 2.0 Client ID**:
   - Navigate to **Clients** (or **Credentials** → **Create Credentials** → **OAuth client ID**).
   - **Application Type**: Select **Desktop app**.
   - **Name**: Enter a clear consumer-specific display name (e.g. `LLM Evaluation - Mac`).
   - Leave any AI-agent or AI-powered-agent option unchecked unless specifically required.
   - Click **Create**.
6. **Download OAuth Client JSON**:
   - Download the client configuration JSON file immediately.
   - Move the file to a secure, private location outside the repository:
     - For `llm_evaluation`: `~/.config/llm-evaluation/llm-evaluation-oauth.json`
     - For Prompt Research: `~/.config/prlab/prompt-research-prlab-oauth.json`
   - Set owner-only file permissions (`chmod 0600 <path>`).

> [!CAUTION]
> **Never commit OAuth client JSON, client secrets, or ADC files to Git.** These files must remain outside all checkouts, artifacts, and public logs.

---

## 4. Local ADC Provisioning and Context Isolation

To prevent processes from accidentally reading credentials from other projects, isolate the Google Cloud SDK configuration directory via `CLOUDSDK_CONFIG`:

```bash
export CLOUDSDK_CONFIG="$HOME/.config/gcloud-<consumer-name>"
mkdir -p "$CLOUDSDK_CONFIG"
chmod 0700 "$CLOUDSDK_CONFIG"
```

### 4.1 Supported Repository Helper

During the current `canonical-sync` incubation period, the supported local provisioning and renewal path is the tested repository helper:

```bash
uv sync --frozen --all-packages --group dev --extra drive --extra local-auth

uv run --all-packages --extra local-auth \
  python scripts/refresh_google_drive_adc.py \
  --client-secrets ~/.config/llm-evaluation/llm-evaluation-oauth.json \
  --cloud-sdk-config ~/.config/gcloud-llm-evaluation
```

The helper exposes no scope override and requests exactly:

```text
https://www.googleapis.com/auth/drive.readonly
```

Its writable ADC destination resolves in this order:

1. explicit `--adc-path`;
2. explicit `--cloud-sdk-config/application_default_credentials.json`;
3. existing `CLOUDSDK_CONFIG/application_default_credentials.json`;
4. the platform-standard ADC location (`~/.config/gcloud/application_default_credentials.json` on macOS/Linux or `%APPDATA%\gcloud\application_default_credentials.json` on Windows).

`GOOGLE_APPLICATION_CREDENTIALS` is deliberately not a writable default because it can identify service-account, Workload Identity Federation, external-account, or other credential material that this operator helper must never overwrite.

Use `--check` to validate an existing ADC without launching consent. If an ADC already exists, deliberate reauthorization requires `--force`; `--force` acknowledges replacement but never bypasses client-file, credential-type, exact-scope, refresh-readiness, or loader validation.

### 4.2 gcloud Exact-Scope Compatibility Boundary

An operator may use the official `gcloud auth application-default login` path only if it completes with the exact Drive read-only scope. In the verified macOS condition with Google Cloud SDK 590.0.0, this command:

```bash
CLOUDSDK_CONFIG="$HOME/.config/gcloud-<consumer-name>" \
gcloud auth application-default login \
  --client-id-file="/path/to/private/oauth-client.json" \
  --scopes=https://www.googleapis.com/auth/drive.readonly
```

failed because that `gcloud` path demanded `cloud-platform`. **Do not broaden the credential merely to satisfy tooling.** Use the tested repository helper instead. This is an observed compatibility fact for that SDK condition, not a universal statement about every Cloud SDK release.

### 4.3 Valid ADC Representation and Loader Compatibility Check

> [!WARNING]
> Calling `credentials.to_json()` from `google.oauth2.credentials.Credentials` produces authorized-user fields but omits the top-level `"type": "authorized_user"` discriminator required by Application Default Credentials. Saving that payload directly as `application_default_credentials.json` causes the ADC loader to reject it.

The helper therefore writes an explicit refresh-capable authorized-user ADC document containing the required discriminator, client fields, refresh token, Google token endpoint, and exactly the Drive read-only scope. It does not persist a short-lived access token or expiry.

After structural validation, the helper deliberately calls `google.auth.load_credentials_from_file()` as a compatibility assertion: the artifact must be accepted by the same generic ADC loading path that exposed the original missing-`type` regression. Current `google-auth` emits a generic deprecation/security warning for that loader because arbitrary externally sourced credential configurations can be unsafe. The helper suppresses that warning only around this already-prevalidated compatibility check; it first requires `type == "authorized_user"`, exact required fields, and the exact scope, then verifies the returned credential class and fields again. This warning suppression is not a validation bypass.

### 4.4 Transactional Replacement and Headless Operation

Provisioning and renewal are fail-closed. The helper:

1. validates the installed/Desktop OAuth client file;
2. runs `InstalledAppFlow.run_local_server()` with exactly Drive read-only scope;
3. constructs an explicit `authorized_user` ADC candidate;
4. writes the candidate to a temporary file on the destination filesystem and `fsync`s it;
5. restricts permissions to `0600` on POSIX;
6. loads and refresh-validates the candidate before replacement;
7. atomically installs it with `os.replace()`; and
8. reloads the final destination to verify the persisted contract.

Any failure before replacement leaves an existing ADC byte-for-byte unchanged. Windows success does not pretend POSIX `chmod(0600)` is an equivalent Windows ACL guarantee.

`--no-open-browser` prevents automatic GUI browser launch but does not remove the localhost OAuth callback requirement. Use deliberate SSH/local-port forwarding or another supported topology for a remote/headless host; pass `--port PORT` when a stable forwarded port is required.

---

## 5. Testing-Mode Lifetime Warning

When a Google OAuth application is left in **Testing** publishing status with an **External** audience:
- Google issues refresh tokens that expire after **7 days** for sensitive or restricted scopes (including Google Drive scopes).
- This is appropriate and safe for bounded manual or development verification.
- It is **not** suitable for unattended continuous mirroring.
- If a token expires, `canonical-sync` will fail closed at a sanitized `TransportAuthError: Google credential refresh failed` boundary. It will **not** attempt to fallback to another project's credentials.

For production or automated environments, use Google Workload Identity Federation (WIF) or an approved service account rather than desktop testing tokens.

---

## 6. Verification and Negative Controls

Before treating the credential context as operational, verify it mechanically:

1. **Context Isolation**:
   Confirm that setting `CLOUDSDK_CONFIG="$HOME/.config/gcloud-<consumer>"` causes `google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])` to read from the consumer's directory, and that the Prompt Research and `llm_evaluation` directories are distinct.
2. **Negative Control (Uncredentialed)**:
   Point `CLOUDSDK_CONFIG` at an empty temporary directory. Confirm that `GoogleDriveTransport.from_application_default()` fails with `TransportAuthError: Google application-default read credentials are unavailable` without leaking paths.
3. **Negative Control (Authority Mismatch)**:
   Run acquisition with a valid credential but with an authority profile referencing mismatched names or parent IDs. Confirm that acquisition fails closed with `AuthorityMismatchError` before any state is published.
4. **Positive Live Acquisition**:
   Run the opt-in live test with the registered consumer profile:
   ```bash
   CLOUDSDK_CONFIG="$HOME/.config/gcloud-llm-evaluation" \
   LLM_EVALUATION_LIVE_DRIVE=1 \
   LLM_EVALUATION_DRIVE_PROFILE="$HOME/.config/llm-evaluation/llm-evaluation-d016.toml" \
   uv run --all-packages --extra drive pytest -m live_drive tests/test_canonical_drive_live.py -q
   ```
   Confirm that the test passes (`1 passed`), verifies the state and receipt hashes, and produces no Drive mutations.

---

## 7. Temporary Duplication Exit Condition

The repo-local helper is intentional temporary migration debt while `canonical-sync` remains incubating. Prompt Research carries the same behavioral implementation without a cross-repository runtime dependency. Keep both copies mechanically synchronized unless a divergence is explicitly justified.

Standalone extraction is complete only after a shared operator tool is separately authorized and validated, both consumers migrate their runbooks/invocations to it, both repo-local helper copies are deleted, consumer-specific authority semantics remain outside the shared tool, parity with this verified contract is demonstrated, and each consumer's relevant deterministic/live verification still passes. Do not retain the repo-local copies as fallback implementations without a new explicit architecture decision.
