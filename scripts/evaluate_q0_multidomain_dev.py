"""Offline deterministic Q0 multi-domain development packet qualification.

No inference, no model execution, no hosted provider or runtime launch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_evaluation.q0_multidomain_dev import (
    load_controls,
    load_packet,
    qualification_report,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT / "fixtures/q0_multidomain_packet_dev_v1.json",
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=ROOT / "fixtures/q0_multidomain_controls_dev_v1.json",
    )
    args = parser.parse_args()
    packet = load_packet(args.packet)
    control_bytes = args.controls.read_bytes()
    controls = load_controls(args.controls, packet, source_bytes=control_bytes)
    print(
        qualification_report(packet, controls, control_bytes).model_dump_json(indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
