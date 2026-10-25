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
