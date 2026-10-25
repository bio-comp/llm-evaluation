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

Matched realized-error diagnostics at several symmetric endpoint targets expose the tradeoff rather than choosing a deployment policy. “Adaptive fastest” below means the lowest endpoint mean E[N] in the prespecified symmetric count-boundary family; “fixed fastest” allows threshold ties to remain inconclusive; “fixed zero-tie” is the smallest feasible odd n, which has no equality tie at θ=.5.

| Endpoint error target | Adaptive fastest boundary | FA=FR | Inconclusive | E[N] | Fixed fastest (n; inconclusive) | Fixed zero-tie n |
| ---: | --- | ---: | ---: | ---: | --- | ---: |
| .10 | accept 2,4,6,10 | .09710 | .07741 | 6.4315 | 2; .42000 | 9 |
| .05 | accept 3,4,6,9 | .03531 | .04003 | 10.2997 | 10; .10292 | 17 |
| .025 | accept 3,4,6,10 | .02088 | .13305 | 10.2997 | 18; .03862 | 23 |
| .01 | accept 3,4,7,11 | .00984 | .29704 | 12.2261 | 26; .01607 | 31 |

The corresponding symmetric reject count at look n is n minus the accept count. These rows are not a scalar ranking: at the 1% target, for example, a slower finite-count rule (accept 3,5,7,10; E[N]=13.9473) cuts inconclusive probability from .29704 to .16519 while keeping both realized endpoint errors at .00782. At the 10% target, fixed n=2 is technically feasible only because it is inconclusive with probability .42. Candidate classification therefore retains false accept and false reject separately; Pareto dominance also uses p0/p1 inconclusive probabilities and p0/p1 expected sample counts separately rather than averaging away endpoint tradeoffs. Stopping distributions remain reported, not collapsed into that frontier.

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

`ModelCondition` separately identifies weights/checkpoint, artifact SHA-256 when locally inspectable, quantization, provider, backend and version, API mode, endpoint model ID, prompt/template identity, seed behavior and runtime environment. `BackendConfig` contains provider ID, base URL, model ID, API mode, capability profile, an optional credential-environment-variable name, an optional configuration ID, and a tiny explicit internal→wire parameter-name map. The configuration ID is distinct from provider identity so one provider can serve multiple model/API configurations simultaneously. The wire map exists because compatible endpoints can name the same requested control differently—for example local Chat endpoints use `max_tokens`, while current OpenAI Chat prefers `max_completion_tokens`; the serialized translated body is what is hashed as the effective request. Base URLs with embedded credentials are rejected because endpoint identity is retained for provenance. `BackendRegistry` is deliberately only an explicit lookup table; it is not a provider framework.

## Transport decision

**Chat Completions is the initial normative generation path.** The common implementation is direct HTTP rather than a provider SDK.

The official OpenAI Python SDK 3.37.0 was tested against a local fake OpenAI-compatible server. Configurable `base_url` and the required Chat Completions request fields worked, but the SDK adds SDK-specific wire headers and dependency behavior. Direct stdlib HTTP therefore gives the current laboratory a smaller dependency surface plus deterministic, auditable serialization of exactly what was transmitted. The SDK remains a tested interoperability option, not provider authority.

Responses parsing remains available behind an explicit capability profile, but it is not the E-002 common denominator: currently tested/documented compatible runtimes expose materially different subsets of Responses controls. No stateful conversation feature is required for E-002.

## Capability gate

Every requested scientific control is classified as one of:

`SUPPORTED`, `UNSUPPORTED`, `ACCEPTED_BUT_IGNORED`, `SEMANTICS_DIFFER`, or `UNKNOWN`.

A requested control is transmitted only when its profile says `SUPPORTED`; all other statuses fail before the request is sent. An unrequested parameter is omitted. This prevents an SDK or common schema from creating fake parity.

Before two conditions are compared scientifically, either restrict both to a demonstrated common control subset or preserve the control difference as part of the condition. In particular, seed, top-p, stop behavior, output-token semantics and prompt template must not be assumed identical across endpoints. Protocol acceptance and scientific capability are intentionally distinct: a backend may accept a parameter while its reproducibility or semantic equivalence remains `UNKNOWN`.

The bounded local preflight establishes the following status for the subset E-002 may need. “Blocked for selected artifact” means the installed server may document the API feature in general, but the exact Qwen3.5 artifact does not load, so no scientific capability is asserted for that condition.

| Capability | Ollama 0.40.8 | LM Studio 0.4.26+4 | llama.cpp 0.7.0 + exact Qwen3.5 blob |
| --- | --- | --- | --- |
| model listing | SUPPORTED, observed | SUPPORTED, observed | blocked for selected artifact |
| Chat Completions text generation | SUPPORTED, observed | SUPPORTED, observed | blocked for selected artifact |
| Responses text generation | SUPPORTED transport, observed; not common control subset | SUPPORTED transport, observed; not common control subset | blocked for selected artifact |
| temperature / top-p | SUPPORTED, documented and transmitted | SUPPORTED, documented and transmitted | blocked for selected artifact |
| top-k through compatible Chat | UNKNOWN/not assumed portable | SUPPORTED/documented | blocked for selected artifact |
| seed as a reproducibility control | UNKNOWN despite documented/accepted field | UNKNOWN despite documented/accepted field | blocked for selected artifact |
| max output tokens | SUPPORTED, observed | SUPPORTED, observed | blocked for selected artifact |
| stop sequence | SUPPORTED/documented; visible-marker probe still confounded by reasoning | SUPPORTED/documented; visible-marker probe still confounded by reasoning | blocked for selected artifact |
| usage accounting / finish reason | SUPPORTED, observed | SUPPORTED, observed | blocked for selected artifact |
| structured output / streaming | not required; not promoted by this preflight | not required; not promoted by this preflight | not required |

No local probe established an `ACCEPTED_BUT_IGNORED` classification for a required field. Such a result must be recorded if a future differential probe demonstrates it; absence of evidence is not converted into `SUPPORTED`.

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
- The transport documents and accepts `seed`, but the bounded probe cannot promote its scientific semantics beyond **UNKNOWN**: seed 1701 twice and seed 1702 all exhausted the 64-token completion budget in hidden reasoning and returned empty visible text. Equal empty outputs do not demonstrate RNG equivalence or seed sensitivity.
- A bounded stop request returned `finish_reason="stop"` with empty visible text. Because the model was still using the reasoning surface, that establishes endpoint acceptance/termination for this request but is not a clean visible-marker truncation test.
- `qwen3.5:9b`: architecture qwen35, 9.7B parameters, Q4_K_M, 262144 context, renderer/parser qwen3.5. Its local GGUF blob has independently measured SHA-256 `dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c` over 6,594,462,816 bytes.
- A separate 32-token Chat smoke call returned `finish_reason="length"`, empty visible `content`, and a nonempty provider-specific `message.reasoning` field while all 32 completion tokens were consumed. Output-token budgets therefore need a visible-answer/truncation preflight for thinking models; token accounting alone is not enough.

Ollama documents its OpenAI compatibility at <https://docs.ollama.com/openai>.

### LM Studio

- Application version: 0.4.26+4.
- Local server: `http://127.0.0.1:1234/v1`.
- `/v1/models`, Chat Completions and Responses are available.
- Chat Completions documents temperature, top-p, top-k, seed, max tokens and stop and applies the model prompt template automatically.
- The endpoint documents and accepts `seed`, but reproducibility semantics remain **UNKNOWN** for scientific use. Two seed-1701 Chat calls both returned visible `PROBE`, yet provider-reported completion-token counts differed (57 versus 39); seed 1702 also returned `PROBE`. Visible equality on a trivial prompt is not evidence of identical stochastic traces.
- A bounded stop request returned `finish_reason="stop"` and empty visible content. As with Ollama, the thinking surface prevents this from serving as a clean visible-marker truncation proof.
- Available Qwen probe model: `qwen/qwen3-4b-thinking-2507`, selected variant `@4bit`, 4B, 4-bit **safetensors/MLX**, indexed size 2,279,153,047 bytes, not GGUF. Do not treat it as an artifact-equivalent control for the Ollama Qwen3.5 model.
- `lms ps` reported effective context length 8192 and server `parallel=4` during this preflight. Calls were issued sequentially, but this server-level configuration is another reason the seed probe is not evidence of single-worker deterministic RNG semantics; a real E-002 condition must freeze or explicitly vary it.
- A separate 32-token Chat smoke returned `finish_reason="length"`, empty visible `content`, provider-specific `reasoning_content`, and 30 reported reasoning tokens. The bounded Responses smoke likewise returned HTTP 200 and usage but no visible answer. Thinking/output-budget semantics therefore remain part of the condition, not a hidden transport detail.

LM Studio documents its OpenAI-compatible endpoints at <https://lmstudio.ai/docs/developer/openai-compat>.

### standalone llama.cpp

- `llama-server` version 0.7.0, build 11669, commit `620d2225c` on Darwin arm64.
- The OpenAI-compatible Chat path is the intended portable surface when a model loads successfully.
- A same-artifact control was attempted with the independently hashed Ollama Qwen3.5 blob `dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c`. Before inference, the installed standalone llama.cpp rejected it with `key qwen35.rope.dimension_sections has wrong array length; expected 4, got 3`, so this exact artifact is **not currently a valid Ollama↔standalone-llama.cpp parity condition**.
- No model was downloaded, converted, patched or re-quantized to force a parity result.

This negative result matters: “both use GGUF/llama.cpp lineage” does not establish that the same current artifact can be served by both installed runtimes.

## Seed and concurrency discipline

For a real repeated-sampling run, freeze concurrency at 1 unless concurrency is intentionally a factor. If the backend supports a seed with verified semantics, preregister a deterministic sequence of **distinct** seeds derived from a public master seed and trial index; do not repeat one seed and call the outputs independent samples. The effective request records the transmitted seed. If a backend lacks or changes seed semantics, record that and preserve trial order/timestamps rather than inventing equivalence.

A capability probe should test repeated calls with the same seed and distinct seeds, but bitwise output parity is not required unless the backend documents that stronger guarantee. Prompt template changes, backend upgrades and concurrent RNG scheduling are abort/reclassification conditions.

## Resource accounting

Always record generation count, prompt/input tokens, completion/output tokens, total tokens and client wall time where available.

For local conditions additionally record hardware class, memory/VRAM or unified-memory constraints, backend/runtime version, offload/context settings when material, concurrency and derived completion tokens/second. For hosted conditions retain provider-reported usage, service tier/routing metadata and a cost computed from a dated frozen price table. Statistical sample efficiency, runtime throughput and monetary cost remain separate quantities.

## Hosted expansion candidates — design only, no calls performed

Checked against current official documentation on 2026-10-25. Prices below are per 1M text tokens and are **planning metadata only**; freeze exact prices again before any paid run.

| Candidate | Compatible path | Version/pinning note | Required-control evidence | Current planning price / limits | Scientific caution |
| --- | --- | --- | --- | --- | --- |
| OpenAI GPT-5.5 snapshot `gpt-5.5-2026-04-23` | Chat Completions + Responses | Dated snapshot is explicitly pin-able; record returned model/system fingerprint and service tier | Chat API documents temperature/top-p/stop and best-effort seed reproducibility; exact snapshot must still be capability-probed before use | $5 input / $30 output per 1M; published limits are usage-tier dependent (e.g. Build 5K RPM/1M TPM, Launch 10K RPM/4M TPM) | Seed determinism is explicitly not guaranteed; service tier and backend fingerprint are condition metadata |
| Groq `openai/gpt-oss-20b` | OpenAI-compatible Chat + Responses | No dated weight snapshot was established in this preflight; freeze model ID, date, returned metadata and any revision/fingerprint exposed | Groq documents seed, temperature, top-p, stop and reasoning controls; temperature 0 is translated to 1e-8 | $0.075 input / $0.30 output per 1M; current public base Developer limits 30 RPM / 1K RPD / 8K TPM / 200K TPD (exact organization limits can differ); 131072 context, 65536 max completion | Provider-hosted `gpt-oss-20b` is not presumed weight-identical to the local MXFP4 artifact; compatibility has known semantic differences |
| OpenRouter explicit model + one explicit provider endpoint | Chat Completions portable path | Pin explicit model plus full provider slug where needed; set provider order/only and `allow_fallbacks:false`; record actual routed provider | Parameter support inherits the selected downstream provider; use `require_parameters` as a routing guard but still capability-probe the pinned route | Underlying model/provider token price; paid-model limits are generally provider-derived rather than plan-wide | Default routing and provider fallback are scientifically unsafe because the serving provider can change between trials |

OpenAI pricing/model metadata: <https://developers.openai.com/api/docs/models/gpt-5.5>, <https://developers.openai.com/api/docs/pricing>, and the Chat reproducibility guidance. Groq model, reasoning and compatibility documentation: <https://console.groq.com/docs/models>, <https://console.groq.com/docs/reasoning>, and <https://console.groq.com/docs/openai>. OpenRouter must be configured with a pinned provider and fallbacks disabled; its default is provider routing/fallback, which is useful operationally but changes the evaluation condition.

No hosted request was made in this preflight. Credentials are not required merely to support configuration for a future endpoint.

## Candidate objective-oracle task strata

The pilot should contain at least two task families without an LLM judge.

### 1. Laboratory-generated executable arithmetic/string tasks

Preferred low-contamination stratum for repeated-sampling method validation. Generate instances from a frozen public algorithm version and deterministic task seed; require a narrow machine-checkable output such as one integer or exact token. The oracle is executable and the repository can own the generator rather than redistribute third-party questions.

This is a **design candidate, not yet a versioned dataset**. It cannot enter a preregistered run until the generator/version, seed space, development/final split, parser and license are actually frozen. An over-strict parser can falsely reject a substantively correct response, so positive/negative parser controls are required.

### 2. LiveCodeBench `release_v6` coding tasks

Candidate external executable coding stratum. The inspected public repository revision is `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`; repository code is MIT-licensed and `release_v6` contains 1,055 problems released from May 2023 through April 2025. Execution/checker outcomes avoid an LLM judge, but the oracle is not infallible: the official `ERRATA.md` records multiple-output tasks that can mark valid answers incorrect, unsupported interactive problems, and erroneous test cases. Timeout/runtime/checker configuration is therefore part of the oracle condition.

Because problem material originates from external programming platforms, repository-code licensing does not by itself settle redistribution rights for every problem statement/test artifact. Do not copy benchmark contents into this repository without a separate redistribution review; prefer stable IDs/source acquisition and hashes. Release-date filtering reduces, but does not eliminate, contamination. For 2026-era models, an April-2025 cutoff may already be in training data, so `release_v6` is better treated as a dated executable stress/development stratum unless a model-specific cutoff analysis justifies stronger claims.

### 3. GSM8K original test split

The archived `openai/grade-school-math` repository at commit `3101c7d5072418e28b9008a6636bde82a006892c` is MIT-licensed and defines the original GSM8K train/test JSONL plus a reference `####` final-number extraction convention. This makes it a cheap objective-oracle pipeline-development candidate. Parser normalization still needs explicit tests (equivalent numeric formats, commas/signs/decimals), and the 2021 benchmark is old and heavily exposed, so it is a poor final contamination-sensitive stratum for 2026-era models.

## Proposed initial local subject conditions

Keep the first real pilot small enough to interpret:

1. **Ollama Qwen3.5 9B Q4_K_M** — exact local model blob SHA-256 `dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c` (6,594,462,816 bytes), qwen35 architecture, 9.7B parameters, 262144 context; smaller/faster primary condition.
2. **Ollama GPT-OSS 20B** — larger, distinct-model local contrast on the same serving backend: architecture `gptoss`, 20.9B parameters, MXFP4, 131072 context, exact local model blob SHA-256 `e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb` (13,793,422,144 bytes). No quality ordering versus Qwen3.5 is assumed before task-specific evidence.
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
