"""Exact-artifact, development-only Quantum Q0 five-case runner.

Example:
  uv run --locked python scripts/run_local_q0.py \\
    --artifact /home/mikeh/Models/shared-gguf/hub/models--Qwen--Qwen3-8B-GGUF/snapshots/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf \\
    --artifact-size 5027783488 \\
    --artifact-sha256 d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785 \\
    --revision 7c41481f57cb95916b40956ab2f0b139b296d974 \\
    --llama-server /home/mikeh/.local/bin/llama-server \\
    --runtime-sha256 5c1bf7bb1b01cfce6525f05feac097a3ff5528c93a2d77faeb7550fd3df4e54e \\
    --runtime-version 1.0.0-dev --runtime-build 12602 \\
    --runtime-commit 418fe97b5 --packet fixtures/local_runtime_preflight_v1.json \\
    --public-manifest /tmp/quantum-q0-public-v1.json \\
    --private-root ~/.local/state/llm-evaluation/q0 --alias quantum-qwen3-8b-q4km \\
    --port 18081
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_evaluation.local_runtime_q0 import Q0Config, run_q0
from llm_evaluation.runtime.llamacpp.qualification import ExpectedExecutable


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--artifact-size", type=int, required=True)
    p.add_argument("--artifact-sha256", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--llama-server", type=Path, required=True)
    p.add_argument("--runtime-sha256", required=True)
    p.add_argument("--runtime-version", required=True)
    p.add_argument("--runtime-build", type=int, required=True)
    p.add_argument("--runtime-commit", required=True)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--public-manifest", type=Path, required=True)
    p.add_argument("--private-root", type=Path, required=True)
    p.add_argument("--alias", required=True)
    p.add_argument("--port", type=int, default=18081)
    args = p.parse_args()
    config = Q0Config(
        artifact=args.artifact,
        expected_size_bytes=args.artifact_size,
        expected_sha256=args.artifact_sha256,
        repository="Qwen/Qwen3-8B-GGUF",
        revision=args.revision,
        executable=args.llama_server,
        expected_runtime=ExpectedExecutable(
            version=args.runtime_version,
            build=args.runtime_build,
            commit=args.runtime_commit,
            sha256=args.runtime_sha256,
        ),
        packet=args.packet,
        public_manifest=args.public_manifest,
        private_root=args.private_root,
        alias=args.alias,
        port=args.port,
    )
    record = run_q0(config)
    print(
        f"{record.status}: {len(record.cases)}/5 cases, "
        f"failure={record.failure_stage}:{record.failure_code}, "
        f"manifest={config.public_manifest}, "
        f"private_evidence_dir={config.private_root / ('q0-' + record.run_id)}"
    )
    return 0 if record.status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
