# E-002 portable model-runner preflight

This document records a **preflight design**, not a completed E-002 model experiment. E-002 remains PLANNED. The intended architecture is local-first execution through a portable API contract, with hosted expansion only after an explicit scientific and expenditure decision.

OpenAI-compatible means **transport compatibility only**. Model weights, quantization, provider, inference backend, backend version, API mode, prompt/chat template, effective request, sampling controls, seed behavior, and local runtime remain separate experimental factors.

## Statistical decision carried forward

The current diagnostic still uses θ=1/2, p0=3/10, p1=7/10 and looks n={2,4,8,16} only as a method-development example. These numbers are not a deployment threshold.

For iid Bernoulli trials, the finite terminal accept event has been mechanically verified to be coordinatewise increasing and the reject event decreasing for the supported policies. With the common-uniform coupling X_i(p)=1{U_i≤p}, this establishes that the one-sided composite suprema occur at the nearest endpoints:

- sup_{p≤p0} P(false accept) = P_{p0}(accept);
- sup_{p≥p1} P(false reject) = P_{p1}(reject).

The check enumerates all 2^16 binary paths at the current cap. Regression tests fail if a future boundary loses the required monotonicity. These are **iid Bernoulli** statements only; dependence, drift and heterogeneous latent item probabilities require separate stress analysis.

Under the illustrative 5% false-accept and 5% false-reject endpoint constraints:

| Procedure | False accept at .3 | False reject at .7 | Endpoint inconclusive | E[N] |
| --- | ---: | ---: | ---: | ---: |
| Generic capped LR, reject ≤0.1 / accept ≥10 | 0.02088 | 0.02088 | 0.13305 | 10.29967 |
| Finite calibrated count boundary (accept counts 3,4,6,9 at n=2,4,8,16; symmetric reject boundary) | 0.03531 | 0.03531 | 0.04003 | 10.29967 |
| Fixed n=16 | 0.02567 | 0.02567 | 0.04868 | 16 |

The finite calibrated rule is not a universal winner. It spends more of the allowed error budget than the generic LR rule to reduce cap/inconclusive probability at the same endpoint expected sample count. Its selected finite count boundary also passes an independent exhaustive 2^16 terminal-event monotonicity check, so the same iid Bernoulli endpoint-supremum coupling argument applies. Fixed-16 has a different cost/error/abstention tradeoff. E-002 must keep false accept, false reject, inconclusive probability, sample count and stopping distribution separate unless a later utility rule is explicitly selected.

The small constant-LR grid also exposes discreteness: many nominal LR thresholds map to the same reachable finite-state boundary. Finite count-boundary calibration is therefore more informative for this cap than pretending continuous LR tuning always changes the policy.

## Independent reference sizing

`reference_sample_size()` exactly sizes a separate fixed Bernoulli reference arm for a threshold θ, distance Δ and maximum wrong-decision probability ε. Ties are explicit: `inconclusive`, independent fair `half` tie-break, or `wrong`.

For θ=.5, examples are:

| Δ | ε | Tie inconclusive | Tie half/wrong in these symmetric cases |
| ---: | ---: | ---: | ---: |
| .05 | .10 | 144 | 163 |
| .05 | .05 | 250 | 269 |
| .05 | .025 | 362 | 383 |
| .05 | .01 | 518 | 539 |
| .10 | .10 | 30 | 41 |
| .10 | .05 | 56 | 67 |
| .10 | .025 | 84 | 95 |
| .10 | .01 | 122 | 133 |

The equality of the half/wrong minima in these examples is not assumed in general. No real reference size is selected until the actual E-002 threshold, indifference zone and error target are frozen.

## Historical PR #4 grid

Schema-v1 `holdout_p` is historical and consumed: it has been evaluated, published and inspected. The schema and old output values are preserved unchanged. It must not be described or reused as an untouched future confirmation set. New diagnostics should use `evaluation_grid` unless a genuinely untouched holdout exists; for this finite Bernoulli problem, exact whole-region operating characteristics are preferred where monotonicity establishes them.

## Minimal generation boundary

`llm_evaluation.model_runner` contains only the transport-independent fields E-002 needs.

A `GenerationRequest` carries:

- ordered text messages;
- temperature, top-p and optional top-k;
- maximum output tokens;
- stop sequences;
- seed when the selected backend profile establishes support.

A `GenerationResult` retains:

- raw visible text;
- backend-reported model identifier;
- finish/status reason;
- prompt/input, completion/output and total token counts when available;
- provider request ID when available;
- the raw provider JSON metadata;
- elapsed client wall time;
- an `EffectiveRequest` containing API endpoint/mode, deterministic serialized request body, SHA-256 of that body, and SHA-256 of the normalized internal prompt packet.

Provider-specific metadata is not erased merely to create a uniform table.

`ModelCondition` separately identifies weights/checkpoint, artifact SHA-256 when locally inspectable, quantization, provider, backend and version, API mode, endpoint model ID, prompt/template identity, seed behavior and runtime environment. `BackendConfig` contains only provider ID, base URL, model ID, API mode, capability profile and an optional credential-environment-variable name. `BackendRegistry` is deliberately only an explicit lookup table; it is not a provider framework.

## Transport decision

**Chat Completions is the initial normative generation path.** The common implementation is direct HTTP rather than a provider SDK.

The official OpenAI Python SDK 3.37.0 was tested against a local fake OpenAI-compatible server. Configurable `base_url` and the required Chat Completions request fields worked, but the SDK adds SDK-specific wire headers and dependency behavior. Direct stdlib HTTP therefore gives the current laboratory a smaller dependency surface plus deterministic, auditable serialization of exactly what was transmitted. The SDK remains a tested interoperability option, not provider authority.

Responses parsing remains available behind an explicit capability profile, but it is not the E-002 common denominator: currently tested/documented compatible runtimes expose materially different subsets of Responses controls. No stateful conversation feature is required for E-002.

## Capability gate

Every requested scientific control is classified as one of:

`SUPPORTED`, `UNSUPPORTED`, `ACCEPTED_BUT_IGNORED`, `SEMANTICS_DIFFER`, or `UNKNOWN`.

A requested control is transmitted only when its profile says `SUPPORTED`; all other statuses fail before the request is sent. An unrequested parameter is omitted. This prevents an SDK or common schema from creating fake parity.

Before two conditions are compared scientifically, either restrict both to a demonstrated common control subset or preserve the control difference as part of the condition. In particular, seed, top-p, stop behavior, output-token semantics and prompt template must not be assumed identical across endpoints.

## Retry and failure semantics

The common transport has no automatic retry. Each technical call is represented by `GenerationAttempt` with an attempt index, optional `retry_of_attempt`, retry count, explicit status, and a Boolean saying whether a completed attempt contributes to the statistical sample.

Statuses distinguish transport failure, provider/runtime failure, completed model generation and oracle failure. A retry therefore cannot silently become another independent model sample. Oracle evaluation remains a separate stage.

## Local backend inventory and bounded probes

Checked 2026-10-25 on an Apple M4 MacBook Air with 24 GB unified memory and macOS 26.6.2. These are capability probes, not E-002 subject observations.

### Ollama

- Backend version: 0.40.8.
- Compatible base URL: `http://127.0.0.1:11434/v1`.
- `/v1/models`: works.
- `/v1/chat/completions`: live probe returned HTTP 200, model ID, finish reason and usage.
- `/v1/responses`: live stateless probe returned HTTP 200 and usage, but the documented portable Responses subset does not expose the same seed/stop surface as Chat Completions.
- Documented Chat subset includes temperature, top-p, seed, max tokens and stop; top-k is not assumed portable through this OpenAI-compatible surface.
- Two sequential short calls with seed 1701 produced the same message hash in this environment. Seed 1702 produced that same short truncated prefix too, so this probe establishes repeatability for that case but **not seed sensitivity or cross-runtime RNG equivalence**.
- A bounded stop probe returned `finish_reason="stop"` with the stop marker absent from retained text, supporting stop-sequence operation for the tested request.
- `qwen3.5:9b`: architecture qwen35, 9.7B parameters, Q4_K_M, 262144 context, renderer/parser qwen3.5. Its local GGUF blob has independently measured SHA-256 `dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c`.
- A 32-token thinking-model smoke call used the entire completion budget in reasoning and produced empty visible text with a length stop. That is a truncation/output-budget preflight result, not model quality evidence.

Ollama documents its OpenAI compatibility at <https://docs.ollama.com/openai>.

### LM Studio

- Application version: 0.4.26+4.
- Local server: `http://127.0.0.1:1234/v1`.
- `/v1/models`, Chat Completions and Responses are available.
- Chat Completions documents temperature, top-p, top-k, seed, max tokens and stop and applies the model prompt template automatically.
- Two sequential short calls using the same seed 1701 produced different message hashes. The endpoint accepts `seed`, but in this tested MLX setup it must **not** be treated as a reproducibility guarantee. A different seed also produced a different trace.
- A bounded stop probe returned `finish_reason="stop"` and omitted the marker; in that case the marker stopped generation before any visible answer.
- Available Qwen probe model: `qwen/qwen3-4b-thinking-2507`, 4B, 4-bit **safetensors/MLX**, not GGUF. Do not treat it as an artifact-equivalent control for the Ollama Qwen3.5 model.
- Its bounded Chat and Responses probes returned HTTP 200 with usage, but the short budget was again consumed by reasoning with no visible answer.

LM Studio documents its OpenAI-compatible endpoints at <https://lmstudio.ai/docs/developer/openai-compat>.

### standalone llama.cpp

- `llama-server` version 0.7.0, build 11669, commit `620d2225c` on Darwin arm64.
- The OpenAI-compatible Chat path is the intended portable surface when a model loads successfully.
- A same-artifact control was attempted by serving the exact Ollama Qwen3.5 GGUF. The installed standalone llama.cpp rejected the model metadata (`qwen35.rope.dimension_sections` shape mismatch), so this exact artifact is **not currently a valid Ollama↔standalone-llama.cpp parity condition**.
- No model was downloaded or converted to force a parity result.

This negative result matters: “both use GGUF/llama.cpp lineage” does not establish that the same current artifact can be served by both installed runtimes.

## Seed and concurrency discipline

For a real repeated-sampling run, freeze concurrency at 1 unless concurrency is intentionally a factor. If the backend supports a seed with verified semantics, preregister a deterministic sequence of **distinct** seeds derived from a public master seed and trial index; do not repeat one seed and call the outputs independent samples. The effective request records the transmitted seed. If a backend lacks or changes seed semantics, record that and preserve trial order/timestamps rather than inventing equivalence.

A capability probe should test repeated calls with the same seed and distinct seeds, but bitwise output parity is not required unless the backend documents that stronger guarantee. Prompt template changes, backend upgrades and concurrent RNG scheduling are abort/reclassification conditions.

## Resource accounting

Always record generation count, prompt/input tokens, completion/output tokens, total tokens and client wall time where available.

For local conditions additionally record hardware class, memory/VRAM or unified-memory constraints, backend/runtime version, offload/context settings when material, concurrency and derived completion tokens/second. For hosted conditions retain provider-reported usage, service tier/routing metadata and a cost computed from a dated frozen price table. Statistical sample efficiency, runtime throughput and monetary cost remain separate quantities.

## Hosted expansion candidates — design only, no calls performed

Checked against current official documentation on 2026-10-25. Prices below are per 1M text tokens and are **planning metadata only**; freeze exact prices again before any paid run.

| Candidate | Compatible path | Version/pinning note | Standard text price | Scientific caution |
| --- | --- | --- | --- | --- |
| OpenAI GPT-5.5 snapshot `gpt-5.5-2026-04-23` | Chat Completions + Responses | Dated snapshot is explicitly pin-able | $5 input / $30 output | Expensive, but much cleaner version identity than a moving alias |
| Groq `openai/gpt-oss-20b` | OpenAI-compatible Chat/Responses | Provider model ID; backend fingerprint should be recorded when returned | $0.075 input / $0.30 output | Compatibility differs: e.g. temperature 0 is translated; provider behavior is not local-weight equivalence |
| OpenRouter explicit model + explicit provider | OpenAI-compatible router | Pin explicit model/version and provider; disable fallback | underlying model/provider price | Default routing/fallback can silently change provider; unsuitable unless routing is frozen and recorded |

OpenAI pricing/model metadata: <https://developers.openai.com/api/docs/models/gpt-5.5> and <https://developers.openai.com/api/docs/pricing>. Groq model and compatibility documentation: <https://console.groq.com/docs/models> and <https://console.groq.com/docs/openai>. OpenRouter routing must be configured so provider fallback cannot silently change the condition.

No hosted request was made in this preflight. Credentials are not required merely to support configuration for a future endpoint.

## Candidate objective-oracle task strata

The pilot should contain at least two task families without an LLM judge.

### 1. Laboratory-generated executable arithmetic/string tasks

Preferred low-contamination stratum for repeated-sampling method validation. Generate instances from a frozen public algorithm version and deterministic task seed; require a narrow machine-checkable output such as one integer or exact token. The oracle is executable and the repository can own the generator rather than redistribute third-party questions.

Risk: an over-strict parser can falsely reject a substantively correct response. Freeze the output contract and include positive/negative parser tests. Keep development task seeds separate from final task seeds.

### 2. LiveCodeBench release_v6 coding tasks

Candidate executable coding stratum. The public project code is MIT-licensed and release_v6 contains 1,055 problems through April 2025. Execution/checker outcomes avoid an LLM judge, but timeout/runtime configuration is part of the oracle and can cause small score variation. Because problem material originates from external programming platforms, do not copy benchmark contents into this repository without a separate redistribution review; prefer stable IDs/source acquisition and hashes.

Release-date filtering reduces, but does not eliminate, contamination. For 2026-era models, an April-2025 cutoff may already be in training data, so release_v6 is better treated as a dated executable stress/development stratum unless a model-specific cutoff analysis justifies stronger claims.

GSM8K can remain a cheap pipeline-development fallback because its numeric oracle is simple and its repository is permissively licensed, but its age/saturation makes it a poor final contamination-sensitive stratum.

## Proposed initial local subject conditions

Keep the first real pilot small enough to interpret:

1. **Ollama Qwen3.5 9B Q4_K_M** — exact local artifact identity already inspectable; smaller/faster primary condition.
2. **Ollama GPT-OSS 20B** — stronger/larger local contrast on the same serving backend, subject to final exact artifact identity capture before use.
3. **Optional backend-parity control only after demonstrated artifact compatibility.** The attempted Qwen3.5 Ollama↔standalone llama.cpp control currently fails the capability gate and must not be represented as parity.

LM Studio Qwen3 4B Thinking is useful for transport/capability development, but mixing checkpoint and backend simultaneously makes it a poor same-model parity control.

## Portable E-002 preregistration packet — preflight draft

### Statistical decision

- Estimand: conditional per-generation binary task success probability under a frozen model condition and task/prompt packet.
- Threshold/indifference zone: **unresolved authorization/scientific choice**; current θ=.5 and .3/.7 are method-development values only.
- False-accept / false-reject targets: unresolved; characterize at least 10%, 5%, 2.5% and 1% diagnostically before selection.
- Adaptive candidates: generic capped LR and exact finite count boundary; carry both forward until a utility/constraint target selects among error, abstention and sample cost.
- Fixed comparator: smallest exact fixed n satisfying the selected matched realized-error constraints, with ties/inconclusive explicit.
- Reference arm: exact `reference_sample_size` calculation after θ, Δ, ε and tie semantics are frozen.

### Tasks

- Development: generated executable arithmetic/string tasks plus a small executable coding subset.
- Final: untouched generated task seeds and, only if source rights/cutoff analysis permit, a frozen executable coding subset.
- Oracle: objective executable checker; no LLM judge.
- Weighting: task-level weighting must be declared before final sampling; do not let easy task multiplication silently reweight the estimand.

### Subject condition

For every condition freeze checkpoint/weights identity, quantization, provider, backend/version, API mode, endpoint model ID, effective request controls, prompt/template, distinct seed schedule, concurrency and local hardware/runtime environment. Hosted family-name similarity is not evidence of weight identity.

### Runner protocol

- Direct OpenAI-compatible HTTP transport; Chat Completions normative initial mode.
- Capability gate before scientific comparison.
- Hash normalized prompt packet and exact serialized effective request.
- Retain raw provider metadata and request IDs when available.
- No automatic retries; every attempt/retry is visible and explicitly marked as contributing or not contributing to the statistical sample.
- Treat malformed/empty/truncated output as an explicit outcome class for preflight/abort handling, not silently as a transport retry.

### Resource envelope

The full run is not authorized here. Before authorization, compute the maximum generations from task count × per-task cap × subject conditions and estimate tokens from pilot measurements. Local runs report elapsed compute/tokens without converting them to a fictional dollar cost. Hosted expansion reports a dated price estimate and requires separate expenditure authorization.

### Abort/reclassification conditions

Abort or reclassify the affected condition if any of the following occurs: model artifact/version drift; endpoint/provider routing drift; unsupported/ignored critical sampler control; prompt/template drift; oracle defect or parser false rejection; severe task floor/ceiling effects; excessive truncation or empty visible output; runtime instability; changing concurrency/RNG semantics; malformed provider responses; or a retry policy that would alter the statistical sample.

## What remains unresolved

The runner and exact finite-state machinery can be tested without choosing the real scientific loss function. A full E-002 run still requires explicit selection of the actual threshold/indifference zone and error/loss targets, a frozen final task/oracle set, and a bounded local compute envelope. Hosted execution additionally requires explicit cost authorization and credentials. Until then E-002 remains **PLANNED**.
