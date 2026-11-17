"""Issue #50: fail-closed DEVELOPMENT preflight; never performs model sampling.

usage: uv run --locked python scripts/run_e002_pilot_preflight.py \
  --packet fixtures/e002_objective_packet_dev_v1.json \
  --llama-server /opt/homebrew/bin/llama-server \
  --output <new-path>.json

This command can only emit BLOCKED_NOT_EXECUTED when the authoritative
machine-local registry contains zero approved model entries. With any
registered model it refuses to claim readiness without separate qualification.
Output is exclusive-create, strict Pydantic JSON, no provider call.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_evaluation.e002_repeated_pilot import Preflight, preflight_blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight_blocked(args.packet.read_bytes(), args.llama_server)
    payload = result.model_dump_json(indent=2).encode("utf-8") + b"\n"
    if Preflight.model_validate_json(payload) != result:
        raise RuntimeError("preflight output fails typed self-verification")
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(f"{result.status}: {result.reason}; no model trials executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
