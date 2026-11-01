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
uv run --frozen pytest -m live_llama tests/test_llamacpp_live_preflight.py -vv
```

The caller publishes an immutable infrastructure record outside `runtime` and
verifies readback. Records contain safe gate outcomes and counts, not credentials,
raw environment or model text. Failure does not trigger model substitution,
download/conversion, generation retry or competitor termination. Keep local records
under ignored `runs/` when retaining them. A failed/unverified live gate means Task-1
live acceptance is incomplete, even when deterministic tests pass. E-002 stays PLANNED.
