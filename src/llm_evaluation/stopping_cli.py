"""Public synthetic operating-characteristic computation; no model calls."""
import argparse
import subprocess
from pathlib import Path

from .records import software_identity
from .stopping_records import analyze, canonical_bytes, finalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="exclusive finalized record from source checkout")
    args = parser.parse_args()
    record = analyze()
    if args.output is None:
        print(canonical_bytes(record).decode("utf-8"), end="")
        return
    try:
        record["software_commit"], record["package_source_sha256"] = software_identity()
    except subprocess.CalledProcessError:
        parser.error("finalization requires a source checkout; installed wheels support preview only")
    except ValueError as exc:
        parser.error(f"finalization rejected: {exc}")
    finalize(args.output, record)


if __name__ == "__main__":
    main()
