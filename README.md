# llm-evaluation

A public, small Python laboratory for testing **evaluation methods**. The current E-002 diagnostic distinguishes marginal trial success, best-of-k capability, and all-k reliability using independently implemented finite-sample combinatorial estimators.

## Authority and publication

The private `llm_evaluation` Google Drive knowledge base owns current methodology, experiment definitions and status, source admission, decisions and interpretation. This Git repository owns executable code, tests, public synthetic fixtures, and publishable immutable records. Results here do not become accepted methodology without reconciliation into Drive. The repository runs offline and does not need or contain Drive credentials, file IDs, snapshots or private research material.

Local private acquisition, if later needed, must use a verified read-only path, private ignored staging, source-version checks around acquisition, bounded retries, and a deliberate publication review. It is not implemented by this bootstrap. No hosted credentialed workflow exists.

## Run the initial diagnostic

Python 3.11+ and an editable install:

```sh
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 -m llm_evaluation.cli fixtures/e002_synthetic.json --k 2
python3 -m llm_evaluation.cli fixtures/e002_synthetic.json --k 2 --output runs/local-e002.json
```

The final command creates a record once and fails if that path already exists. `runs/` is ignored: review public safety and provenance before deliberately promoting a compact artifact. The public fixture is synthetic. Its subject, prompt, harness, tool surface, runtime, sampling, retry policy, budget, benchmark, oracle, evaluator, and software identity are separate condition fields. The strict v1 record validator rejects unknown and missing fields; the resulting deterministic record captures the fixture SHA-256, k, equal task weighting, task counts and each metric. This is a diagnostic of estimators, **not** a completed E-002 model experiment or a measured production reliability claim.

For n observed binary trials, c successes and 1 <= k <= n, marginal success is c/n, best-of-k is 1 - C(n-c,k)/C(n,k), and all-k is C(c,k)/C(n,k). The latter two average events across size-k subsets of *observed* trials. Under independent identically distributed Bernoulli trials within a task, their sampling expectations are 1-(1-p)^k and p^k respectively. Correlation, drift, task selection, and deployment selection policies can break that future-run interpretation. The test suite enumerates subsets as an independent oracle, exhaustively enumerates a small Bernoulli sample space, rejects biased plug-in transforms, and includes dependent and nonstationary counterexamples. Report costs separately when real sampling is added.

Primary lineage: [HumanEval paper](https://arxiv.org/abs/2107.03374) and [official pass@k implementation](https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py); [tau-bench paper](https://arxiv.org/abs/2406.12045) and [official repeated-trial aggregation](https://github.com/sierra-research/tau-bench/blob/main/tau_bench/run.py). These links establish formula lineage, not independent validation of their benchmarks. No upstream code or benchmark data was copied.

## Public publication threat model

A public PR can execute attacker-controlled tests and build code, so CI has no Drive or model credentials, uses the ordinary `pull_request` event (never privileged `pull_request_target`), and requests only `contents: read`. Actions are pinned to reviewed commit identities. Private acquisition is deliberately absent. A future credentialed job would require a separate trusted ref/workflow and restricted identity, with untrusted PR code unable to execute in that job. Local staging, raw model output, personal identifiers, and private Drive metadata (including IDs, revisions, names and hashes) can reveal information; ignored paths are only an accident barrier, not a publication approval. Review the full staged diff and introduced history for secrets, proprietary content, redistribution rights and metadata before pushing. No service account, WIF provider, ACL change or paid infrastructure is provisioned here.

## Scope and security

Public CI runs on pull requests without secrets and has read-only repository permissions. No LLM judge, API vendor, model runner, benchmark corpus or private Drive material is a dependency. Future runners remain adapters: subject model, prompt/context, harness, tools, runtime, budget, oracle and evaluator must be identified separately. Do not commit local credentials, private Drive content, model outputs or benchmark material without explicit redistribution review.

No license has been selected. Public visibility alone does not grant reuse rights.
