"""Credential-free E-002 synthetic diagnostic."""

import argparse
import subprocess
from pathlib import Path

from .records import analyze, canonical_bytes, load_fixture, write_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", type=Path, help="new immutable run record; fails if present")
    args = parser.parse_args()
    result = analyze(load_fixture(args.fixture), args.k)
    if args.output:
        root = Path(__file__).resolve().parents[2]
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True)
        if status.stdout:
            parser.error("refusing to finalize against a dirty checkout")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        result["software_commit"] = commit.stdout.strip()
        write_immutable(args.output, result)
    else:
        print(canonical_bytes(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
