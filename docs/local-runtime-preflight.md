# Local runtime/model preflight

This is a bounded infrastructure and model-characterization surface. It is **not E-002 evidence**, does not change E-002 status, and must not consume an E-002 protected holdout or reserve item.

## Frozen public task packet

`fixtures/local_runtime_preflight_v1.json` contains five synthetic, objectively scored probes:

- exact-format instruction control;
- a short factual constant;
- arithmetic;
- Python code comprehension;
- synthetic long-context retrieval.

The exact v1 bytes have SHA-256 `6f879e8ee1bca440306af91c6a6169c97be2c1473acffc0c1882cc20bf9a7bbc`. Tests pin the hash, task IDs, oracle type, and long-context target. Change the packet by creating a new version rather than silently editing v1.

Scoring is an exact match against **visible output with no normalization**. A blank answer, refusal, malformed answer, extra explanation, or output-limit exhaustion therefore fails the objective probe. Transport/provider failures remain distinct from task failures.

## Condition identity

A run is interpretable only when the condition records the exact model artifact and runtime state needed for the claim. At minimum retain:

- artifact SHA-256 when a physical artifact is available;
- model/weights identity, format and quantization;
- runtime and exact version/build;
- API path and provider-reported model ID;
- prompt/template identity;
- context and output limits;
- sampler and reasoning controls;
- seed semantics/status;
- concurrency and hardware/backend state.

A shared upstream model name is not artifact identity. Different quantizations, formats, templates or reasoning regimes are different conditions.

### Approved-model registry and onboarding

The reusable model-identity schema is owned here, while machine-specific paths stay outside Git in `~/.config/llm-runtime/model-registry.json`. Project consumers reference only a stable logical model ID. A registry entry records the absolute local artifact path, exact SHA-256, size, GGUF format, quantization, architecture, context capability, verification date, and separate qualification state/identity for llama.cpp and LM Studio. An empty registry is valid and means no standardized model has yet been approved.

Onboarding is deliberately fail-closed: locate the exact user-supplied artifact, measure size and SHA-256 independently, inspect GGUF metadata, assign a stable logical ID, qualify the exact bytes with the reference llama.cpp build, record context/runtime constraints, test only capabilities needed by consumers, optionally qualify the same bytes in LM Studio, then register the verified state. The tooling must not search for, download, convert, requantize or substitute a model when any step fails. Same bytes under llama.cpp and LM Studio are the same artifact but distinct runtime conditions.

## Comparison classes

Keep three questions separate:

1. **Same artifact, different runtime.** Require the same bytes (SHA-256), compatible prompt/template semantics and matched controls. A load failure is a valid runtime-compatibility result. Do not manufacture parity by converting, requantizing or silently changing templates.
2. **Different models, one runtime.** Use the same packet and matched controls where the models actually support them. Report correctness, latency/throughput and resource cost separately.
3. **Whole-condition comparison.** Use this label whenever artifact, format, quantization, template or reasoning regime differs. It is not runtime-only evidence.

The existing `model_runner` remains the portable OpenAI-compatible execution boundary. Runtime-native APIs may be used for capability or systems characterization when the portable API cannot expose the property under test, but those observations must be labeled runtime-native and must not be mixed into an allegedly matched API comparison.

## Measurement rules

Record cold and warm state separately. Preserve raw visible output in local/private run evidence, but do not commit raw model output to the public repository without an explicit publication and redistribution review. Record provider finish/termination state, usage counters when available, wall time, and failures without automatic retries.

For systems characterization, record non-invasive evidence such as model process RSS, memory pressure and swap state before/during/after a load. A model that eventually returns an answer under severe memory pressure is not equivalent to a comfortable fit. Runtime-reported evaluation token counts may include non-visible reasoning or control tokens; do not relabel them as visible-output throughput.

Repeated same-seed probes are reproducibility checks, not independent statistical observations. This packet is deliberately too small for model-success probability estimation or a universal model/runtime leaderboard.

## Artifact storage safety

Measure actual duplicate bytes before changing stores. Do not manually rewrite, symlink or hard-link inside runtime-managed stores unless that operation is explicitly supported and deletion/update behavior is understood. Prefer supported import/link mechanisms and verify identity mechanically. A zero-copy link proves storage identity, not runtime compatibility.


## Managed standalone infrastructure acceptance

The new `local_runtime_preflight` caller is separate from the frozen five-task
packet. It uses tiny infrastructure-only JSON requests and does not score model
quality. Its normal tests use a synthetic executable; the real model test is
skipped without explicit authorization and `LLM_EVAL_LLAMA_PREFLIGHT=1`.

The constrained profile is loopback-only, one `MTL0` slot, context 2048, prediction
limit 128, disabled web UI/agent/MCP proxy, no inherited LLAMA/tool/MCP/router/save
settings, disabled prompt-cache RAM and logging. The child receives a constructed
platform environment. Formal runs receive a fresh environment-only API key;
credentials are absent from identity, argv and retained records.

Shared/formal ownership uses the same persistent kernel-lock file. The guarantee
covers cooperating processes sharing the configured per-user lease namespace,
not arbitrary users/roots or unmanaged GPU workloads. The exec target inherits
the locked descriptor. Only owned processes are terminated, and shutdown waits
for termination. A timed-out request cannot release a still-running server's lock.
Known external-inference inspection reports conflict, no known conflict or unverified.
It recognizes standalone `llama-server`, `llama serve`, a `llama-cli` process with an
observable model binding, and the bounded MLX server forms already supported. LM Studio
is not rejected merely because its GUI is open: when the GUI process exists, the probe
uses `lms server status --json` to obtain the configured server port and requires that
exact listener to be owned by the LM Studio main PID before reporting conflict. Unknown,
malformed or mismatched process/listener state remains unverified. Absence of a known
competitor is not global GPU exclusivity evidence.

The live gate permits at most three server launches and six generation submissions
(including two rejected authorization controls), 45 seconds per startup, 30 seconds
per request and 240 seconds for the live checks including preceding identity work.
It independently measures the selected executable and artifact identity, verifies
the pinned build, model/alias and slots, checks shared/formal exclusion, runs
no-key/wrong-key controls, one tiny generation, one structured generation and two
concurrent requests, then checks actual-server lease retention after driver exit.
Polling records the nominal 20 ms interval, actual monotonic offsets, duration and
bounded slot samples. Together with verified one-slot launch and qualification,
these are sampled observations, not continuous proof. The generic OpenAI-compatible
transport leaves provider-specific `timings` untrusted. The exact llama.cpp preflight
layer alone projects directly reported `prompt_ms` and `predicted_ms` into seconds
after validation. Queue and server wait remain unknown unless reported directly.

The formal profile explicitly disables ordinary per-slot prompt-prefix reuse with
`--no-cache-prompt`; shared/manual launches are not changed for symmetry. The formal
`cache_prompt=false` policy is bound into qualified condition identity alongside the
one-slot/context/output profile. `--cache-ram 0` and `--no-cache-idle-slots` are
separate controls and are not treated as substitutes for disabling prefix reuse.

The managed condition explicitly launches with `--reasoning auto` or `off`;
formal generation rejects an omitted setting. Qualification hashes the exact value
reported by `/props.chat_template`, projects bounded Boolean template capabilities,
checks the build information and effective 2048 context, then binds that observed
runtime-reported template identity and reasoning profile to the qualified condition.
This does not independently verify source-artifact template identity or cross-runtime
template equivalence. A short `reasoning off` command line alone does not prove this
model stops reasoning; the technical structured response must validate.

In the version 2 preflight record, `tiny_generation=true` means only that a
technical HTTP generation completed with the expected provider-reported model
ID. It makes no visible-answer or quality claim. Safe generation summaries
retain visible byte counts, closed finish/termination categories, token counts
and reasoning byte counts without text. The structured gate requires visible
JSON and strict model validation. Historical unversioned records retain their
own reader and are not rewritten.

A separate `run_structured_diagnostic` entry point can isolate one formal
launch and one structured request under the same artifact/build, lease, auth,
context and output bounds. Its distinct immutable record retains failure stage,
safe code and cleanup outcome. A failed request is never silently retried.
The complete preflight additionally verifies the owned PID's listener is
loopback-only and records the disabled agent/tool launch controls.

Explicit use after authorization, with an existing compatible artifact:

```bash
LLM_EVAL_LLAMA_PREFLIGHT=1 \
LLM_EVAL_LLAMA_SERVER=/absolute/path/to/llama-server \
LLM_EVAL_LLAMA_GGUF_PATH=/absolute/path/to/model.gguf \
LLM_EVAL_LLAMA_GGUF_SHA256=<independently-verified-sha256> \
LLM_EVAL_LLAMA_RUNTIME_VERSION=1.0.0 \
LLM_EVAL_LLAMA_RUNTIME_BUILD=12359 \
LLM_EVAL_LLAMA_RUNTIME_COMMIT=b1b770d90 \
LLM_EVAL_LLAMA_RUNTIME_SHA256=780fa4764b08a1d9e0035a2c1245bb9bb9f4bd6838f2f2be09714af75ee2cae3 \
uv run --frozen pytest -m live_llama tests/test_llamacpp_live_preflight.py -vv
```

The historical default of `run_preflight()` and
`run_structured_diagnostic()` remains pinned to llama.cpp
`0.7.0 / build 11669 / 620d2225c` for existing fixtures and historical
interpretation. A newer runtime (including the locally verified Homebrew stable
`1.0.0 / build 12359 / b1b770d90` on 2026-11-17) must be passed explicitly
through `expected_executable=ExpectedExecutable(...)`. The expectation is
constructed from **independently frozen** version/build/commit and executable
SHA-256, never by treating the live probe as its own authority. Exact mismatch
fails before launching a model; both `run_preflight` and the one-generation
structured diagnostic bind the matched runtime version to the reported condition.
For example, after operator approval and external-resource clearance:

```python
from llm_evaluation.local_runtime_preflight import run_preflight
from llm_evaluation.runtime.llamacpp.qualification import ExpectedExecutable

runtime_pin = ExpectedExecutable(
    version="1.0.0",
    build=12359,
    commit="b1b770d90",
    sha256="780fa4764b08a1d9e0035a2c1245bb9bb9f4bd6838f2f2be09714af75ee2cae3",
)
# Pass expected_executable=runtime_pin to run_preflight along with the
# separately verified model artifact, output path and opt-in authorization.
```

This is a pinned technical qualification path, **not** model registration,
checkpoint approval, a new E-002 sample, a claim of v1 API compatibility,
or permission to interfere with LM Studio/Llama.app processes. Build-specific
capabilities, template/alias/context and actual request behavior still require
the live bounded gate with an approved checkpoint.

The caller publishes an immutable infrastructure record outside `runtime` and
verifies readback. Records contain safe gate outcomes and counts, not credentials,
raw environment or model text. Failure does not trigger model substitution,
download/conversion, generation retry or competitor termination. Keep local records
under ignored `runs/` when retaining them. A failed/unverified live gate means Task-1
live acceptance is incomplete, even when deterministic tests pass. E-002 stays PLANNED.

### Launch-byte qualification and bounded trust (R1 correction)

The executable and model digests in a qualification are measured from the
operator-supplied source paths. Path-based execution **after** hashing alone
would not bind those measurements to the bytes subsequently opened: a source
could change during external-conflict probing or before a later launch. Both
entry points now construct one private per-call launch snapshot **after** source
identity checks and conflict clearance. They independently verify the staged
executable's version/build/commit/SHA-256 and the staged model's size/SHA-256,
launch only those staged paths, and rehash both before a `passed` result. The
snapshot stays in scope through server lifecycle and is removed afterward;
there is no new model registry, persistent GGUF store, or change to the approved
Homebrew executable. An initial-to-staged mismatch fails closed before launch.

On macOS the model snapshot requires `cp -c` (`clonefile(2)`/copy-on-write)
on an appropriate volume, not a hard link; clone failure fails rather than
silently copying a large model or reading mutable source bytes at launch.
The staged executable is a private copy with owner-read/execute permissions.
For Homebrew's `@loader_path/../lib` relative linkage, the stage preserves
resolution of the installation's existing dynamic libraries, which are **not**
included in the executable's SHA-256 and are not claimed immutable. A portable
non-macOS path only supports synthetic files up to 1 MiB; larger model snapshots
fail without qualifying. Each operation is within the existing 240-second budget.

**Trust boundary:** this is local cooperative-process provenance, not a
same-UID adversarial security guarantee. A malicious process with the same
account privileges could change the stage's permissions, binaries, dependencies
or executable closure. The reported exact-byte qualification describes the
staged executable and GGUF content under that bounded assumption; it does not
attest dynamic-library closure, kernel execution identity or GGUF semantic
compatibility. If staging, reinspection, hashing or cleanup fails, no `passed`
qualification is published. The frozen E-002 packet, historical records and
scientific lifecycle remain unchanged.

### Quantum CUDA Q0 — explicit existing-GGUF qualification (Issue #50)

The opt-in `scripts/run_local_q0.py` adds a separate, DEVELOPMENT-only five-case
technical qualification of a previously available exact **Qwen3-8B Q4_K_M**
GGUF. It does **not** alter the historical macOS/Metal managed profile, approve
or register models, mutate E-002, launch E-002 repeated sampling, download
weights or use an unowned runtime. The pinned fixture remains byte-for-byte
`fixtures/local_runtime_preflight_v1.json`. A different checkpoint or runtime
requires an independently versioned explicit condition and contract; this
first profile is intentionally not a general inference framework.
The original 4096-token smoke context is insufficient: llama-server build 12602
reported 4723 input tokens for the frozen 256-record retrieval case. Q0 fixes
8192 context tokens so it can execute this unchanged packet, not truncate it.

Invoke `uv run --locked python scripts/run_local_q0.py --help` for all required
pins. Supply `--artifact`, `--artifact-size`, `--artifact-sha256`,
`--revision`, `--llama-server`, `--runtime-sha256`, `--runtime-version`,
`--runtime-build`, `--runtime-commit`, `--packet`, `--public-manifest`,
`--private-root`, `--alias`, and `--port` explicitly. The full pinned example
is in the script docstring. The program rejects a mismatched Hub snapshot,
GGUF size/hash, runtime version/build/hash, frozen packet hash, known conflicting
inference process, running LM Studio API, or occupied loopback listener before
launch. The fixed listener excludes port 8080; owned listener/lease and the
child process must be independently established before calling the five cases.

One formal CUDA0 `llama-server` child uses `--n-gpu-layers 99 --fit off`, one
slot, 8192 context tokens, at most 128 response tokens, `--reasoning off`,
Jinja with `enable_thinking=false`, `seed=42` and `temperature=0`. Effective
model alias, template SHA, build info and context are queried from the server;
actual offloaded layers require affirmative loader-log evidence. GPU memory is
host-total sampled by `nvidia-smi`, not model-exclusive allocation; loaded minus
baseline is a **derived delta**, and a sampled maximum is an **observed lower
bound on peak usage**. Host RAM available/swap and owned process RSS are
separate measured sources. No unavailable metric is fabricated.

The immutable public-safe `Q0Manifest` **schema version 2** is a strict Pydantic
record with status `COMPLETED`, `INCOMPLETE`, or `BLOCKED`, independently
recorded **expected and observed** model/runtime identity, actual executing
Python-client version/implementation/executable digest, fixture identity,
source re-verification, explicit configuration, cleanup result, and five
`Q0CaseResult` entries when completed. Historical v1 files are not rewritten;
their ambiguous identity fields and weaker validations are not retroactively
reinterpreted as v2 evidence. Each case records request-identity and
request/response SHA-256, finish reason, available usage/timing measurements,
reasoning/visible-byte counts, and separate `PASS`, `FAIL_ORACLE`,
`CENSORED_LENGTH`, `OVER_BUDGET`, `EMPTY_VISIBLE`, `MODEL_MISMATCH`,
`INVALID_RESPONSE`, `MISSING_MEASUREMENT`, `TIMEOUT`, `HTTP_FAILURE`. A valid wrong answer is
not a technical failure. Exact text equality is used **without normalization**.
Missing provider counters/timing cannot silently become a qualified case.

The CLI prints the absolute private evidence directory under the supplied
owner-only `--private-root`, named `q0-<run_id>`. Under this directory, each
case has an immutable `case-NN-request.json` and, when a credential-screened,
size-bounded HTTP 200 response was successfully decoded as JSON (even if the
response **schema is invalid**), `case-NN-response.json`;
`server.stderr.log` records loader diagnostics. Public manifest fields store only relative filenames and
hashes, not the prompts, responses, credentials or machine-local private root.
HTTP failures that occur before safe response evidence is available are
explicitly classified without inventing response bytes. Every `COMPLETED`
record is independently verified against the private request/response bytes,
the frozen packet, re-derived request identity and exact oracle, and the
observed local Python interpreter before publication. The verification entry
point `verify_q0_private_evidence(record, config, private_run_dir, packet)`
can also be used for separate readback and corruption controls. Verified
records do not imply reproducibility across a different Python executable;
the local interpreter check is intentionally host-bound. Only the child launched
by this invocation is stopped. LM Studio is recorded `NOT_TESTED` and is never
restarted. The manifest is not scientific validation or E-002 acceptance.

**Trust limitations:** on Linux the large GGUF is intentionally not duplicated
or automatically requantized; the source artifact and executable are hashed
before launch and rehashed after child shutdown. This detects ordinary changes,
but does not attest dynamic libraries or prevent an actively malicious same-UID
actor from altering and restoring bytes between inspections. Reproducibility
also requires the exact hardware/runtime condition and raw private evidence;
one Q0 observation does not establish model reliability or calibration.
