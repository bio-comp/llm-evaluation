# E-002 Q0 multi-domain DEVELOPMENT packet — Issue #50

This public-safe, synthetic-only twelve-case extension is not the original frozen
five-case Qwen3-8B / llama-server technical preflight. It does not execute a
subject model, trusted or untrusted model-generated code, or any external tool.
It is neither a FINAL benchmark nor Q1 screening. E-002 remains PLANNED.

## Source identity and API

- Versioned source bytes: fixtures/q0_multidomain_packet_dev_v1.json,
  schema 1, ID e002-q0-multidomain-dev-v1, phase DEVELOPMENT.
- load_packet checks the exact raw source SHA-256 against the module's pinned
  version fingerprint. Each case's reference ID and canonical content SHA-256
  are compared to independent fixed version anchors; reference contradictions
  fail closed. A new reference requires a new reviewed packet version.
- Each result records packet ID/version/byte count/SHA-256, case ID/category,
  scoring method, reference ID/content SHA-256, synthetic answer SHA-256,
  disposition and a stable reason code. No private artifact is disclosed.
- Control fixture: fixtures/q0_multidomain_controls_dev_v1.json, with 41
  independently labeled positive, negative, alternative and deferred controls
  bound to this packet source hash. The reviewed v1 control source SHA-256 is
  2371f2e019f3b9862c7ec28a75c5486888efdf9b31c297348f5dd883db3b306a.
  The exact source digest and all 41 control identities/order are pinned; an
  altered, weakened, or substituting source cannot qualify. Reported source
  bytes are revalidated and compared with the evaluated typed controls before
  controls_qualified=true can be returned.
- Reproduce: uv run --locked python scripts/evaluate_q0_multidomain_dev.py
- Reusable typed API: load_packet, load_controls, evaluate_response,
  evaluate_all_controls, readiness_report and qualification_report.
  No GGUF downloads, GPU access, network calls or model runtime imports.

## Explicit case inventory

| Category | Cases | Method and scope |
| --- | --- | --- |
| Sanity | sanity-arithmetic, sanity-record-map, sanity-tag-extraction | Exact decimal integer, strict structured JSON map and ordered tag array |
| Coding | coding-median, coding-clamp-debug, coding-tool-call | First two SANDBOX_DEFERRED; third structurally exact JSON tool-call arguments |
| Scientific reasoning | science-pooled-rate, science-ci-decision, science-causal-claim | Reduced fraction 9/25 or exact equivalent terminating decimal, supplied CI rule, EXPERT_DEFERRED causal interpretation |
| General instruction following | general-exact-steps, general-faithful-summary, general-multistep-followup | Three-line EXACT_TEXT, then two EXPERT_DEFERRED tasks |

Seven objective, two qualified-sandbox-dependent, and three expert-dependent
cases remain separate in the readiness report; no single aggregate is made.
JSON key order and whitespace may vary, but duplicate keys, extra keys,
numeric-as-boolean substitutions, NaN/Infinity and wrong values/order fail.
Rationals are exact Fraction/Decimal comparisons, not approximate floats.
Fraction answers must be reduced (9/25 accepted; 18/50 or 36/100 rejected),
while 0.36 and 0.360 are both accepted terminating decimals.
Exact text rejects any extra punctuation or whitespace. The direct
response API rejects Python text containing unpaired Unicode surrogates with
a deterministic ValueError before scoring or answer hashing; it does not
claim a UTF-8 hash for text that cannot be encoded.

## Reference authority and negative controls

Tests independently calculate 17*23-19, parse the literal source records,
extract two tags, check a structured synthetic tool call, calculate 18/50=9/25
by Fraction, and apply the explicitly provided CI rule that includes zero.
The scorer does not generate its own expected values. Reference content is
also anchored independently of the file being validated. Controls include
plausible wrong intermediate arithmetic, malformed/duplicate-key JSON,
incorrect extraction, schema extras, numeric/boolean confusion, incorrect
pooled rate, unreduced fractions, and incorrectly rejected valid JSON or
exact terminating-decimal alternatives.

For coding, accepted alternative trusted implementations of median and
inclusive clamp are defined in tests/reference_q0_multidomain.py. That file
contains executable reference test_median_references and test_clamp_references,
called by tests/test_q0_multidomain_dev.py, with independently specified
boundary vectors and known-bad mutants. **None of this executes model-provided
Python source**. Any answer to coding-median or coding-clamp-debug is recorded
as NOT_AUTOMATICALLY_SCORABLE until an independently qualified sandbox is
implemented and validated.

The three qualitative tasks also return NOT_AUTOMATICALLY_SCORABLE for every
answer. For eventual expert review, proposed (not yet validated) rubric
criteria are:
- Causal claim: distinguish association from causation; explain missing
  randomization/confounder control; do not invent evidence.
- Faithful summary: preserve six respondents, four vs two, and non-random
  sampling; avoid unwarranted population generalization; permit paraphrases.
- Multi-step follow-up: produce a feasible two-step plan, surface deployment
  OS/security uncertainty, ask one material clarification, do not fabricate
  user authorization or an operating system.
Expert anchors, calibration, adjudication and disagreement protocol are not
yet qualified. Unsupported claims remain unscored, not silently certified.

## Q1 dependencies — BLOCKED FOR LIVE EXECUTION

A real model requires an exactly user-approved, SHA-256-verified GGUF in the
D-017 machine-local registry and a separately qualified inference runtime/host.
Q1 additionally requires a prospective frozen prompt/reference packet and
scoring contract, sampler/settings/seeds, randomization and bounded repeats,
technical censoring rules, memory/latency/tokens and costs, sandbox and expert
qualification, held-out cases and uncertainty protocol. The scientific
model-family evaluation and personal daily-driver selection remain distinct;
E-002 scientific repeated sampling is still PLANNED.
