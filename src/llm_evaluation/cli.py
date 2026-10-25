"""Credential-free E-002 diagnostic with explicit preview/finalization."""

import argparse
from pathlib import Path

from .records import (analyze, canonical_bytes, current_git_commit, parse_fixture,
                      write_immutable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", type=Path, help="new immutable final record; fails if present")
    args = parser.parse_args()
    source_bytes = args.fixture.read_bytes()
    result = analyze(parse_fixture(source_bytes), args.k, source_bytes=source_bytes)
    if args.output:
        result["software_commit"] = current_git_commit()
        write_immutable(args.output, result, source_bytes)
    else:
        print(canonical_bytes(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
