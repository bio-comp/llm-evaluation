"""Generate the public-safe D-019 deterministic DEVELOPMENT comparison artifact.

Example:
  uv run --locked python scripts/run_d019_policy_comparison.py \
    --output fixtures/d019_estimated_null_policy_comparison_dev_v1.json

Output is exclusive-create to avoid overwriting a prior experiment record.
This script performs no FINAL evaluation, API call, or Drive mutation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import TypeAdapter

from llm_evaluation.fitted_monitor_policy_comparison import (
    PolicyComparison,
    evaluate_policy_comparison,
)

_CANDIDATES = (10, 12, 14, 16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Exclusive-create JSON path; default is stdout",
    )
    args = parser.parse_args()
    result = evaluate_policy_comparison(_CANDIDATES)
    adapter = TypeAdapter(PolicyComparison)
    payload = adapter.dump_json(result, indent=2) + b"\n"
    assert adapter.validate_json(payload) == result

    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        with args.output.open("xb") as stream:
            stream.write(payload)
        print(f"DEVELOPMENT_ARTIFACT={args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
