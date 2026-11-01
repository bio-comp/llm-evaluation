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

### 4.1 Official gcloud CLI Flow (Try First)

Attempt the documented `gcloud` flow requesting only the exact read-only Drive scope:

```bash
CLOUDSDK_CONFIG="$HOME/.config/gcloud-<consumer-name>" \
gcloud auth application-default login \
  --client-id-file="/path/to/private/oauth-client.json" \
  --scopes=https://www.googleapis.com/auth/drive.readonly
```

Inspect the resulting authorization prompt and output:
- If `gcloud` successfully completes browser authentication with **only** `https://www.googleapis.com/auth/drive.readonly`, verify that `application_default_credentials.json` is generated inside `$CLOUDSDK_CONFIG` with permissions `0600`.

### 4.2 Historical gcloud Compatibility Note and Installed-App Fallback

In tested macOS environments with Google Cloud SDK 590.0.0, running `gcloud auth application-default login` with `--scopes=https://www.googleapis.com/auth/drive.readonly` failed with:

```text
ERROR: (gcloud.auth.application-default.login) Invalid value for [--scopes]:
https://www.googleapis.com/auth/cloud-platform scope is required but not requested.
Please include it in the --scopes flag.
```

**Do not add `cloud-platform` or broader scopes merely to satisfy tooling.** Least privilege requires restricting read access strictly to Google Drive.

If `gcloud` demands broader scopes, use the narrow installed-application flow using `google-auth-oauthlib`.

### 4.3 Valid ADC Document Schema and the `credentials.to_json()` Serialization Trap

> [!WARNING]
> Calling `credentials.to_json()` from `google.oauth2.credentials.Credentials` produces a JSON payload that lacks the top-level `"type": "authorized_user"` discriminator required by Application Default Credentials.
>
> If saved directly to `application_default_credentials.json`, `google.auth.default()` will fail with:
> ```text
> google.auth.exceptions.DefaultCredentialsError:
> The file .../application_default_credentials.json does not have a valid type.
> Type is None, expected one of ('authorized_user', 'service_account', ...)
> ```

The correct bootstrap pattern explicitly structures the authorized-user ADC document:

```python
import json
import os
from google_auth_oauthlib.flow import InstalledAppFlow

CLIENT_SECRETS_FILE = os.path.expanduser("~/.config/llm-evaluation/llm-evaluation-oauth.json")
CONFIG_DIR = os.path.expanduser("~/.config/gcloud-llm-evaluation")
ADC_FILE = os.path.join(CONFIG_DIR, "application_default_credentials.json")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

os.makedirs(CONFIG_DIR, exist_ok=True)
os.chmod(CONFIG_DIR, 0o700)

flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES)
creds = flow.run_local_server(port=0)

# Build valid ADC document with the required 'type' discriminator
payload = {
    "type": "authorized_user",
    "client_id": creds.client_id,
    "client_secret": creds.client_secret,
    "refresh_token": creds.refresh_token,
    "scopes": list(creds.scopes),
}

with open(ADC_FILE, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
os.chmod(ADC_FILE, 0o600)
print("Isolated authorized-user ADC generated successfully at:", ADC_FILE)
```

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
