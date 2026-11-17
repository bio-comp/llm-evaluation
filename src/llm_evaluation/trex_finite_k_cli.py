"""Issue #47 executable development pilot, not a FINAL FDR certification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from llm_evaluation.trex_finite_k import (
    PilotConfig,
    execute,
    file_digest,
    runtime_provenance,
)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_head() -> str:
    run = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return run.stdout.strip() if run.returncode == 0 else "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = PilotConfig.model_validate_json(args.config.read_bytes())
    # Refuse an existing directory: no silent replacement of a prior run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records, summaries = execute(config)
    root = Path(__file__).resolve().parent

    records_path = args.output_dir / "records.jsonl"
    with records_path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.model_dump_json() + "\n")
    summaries_path = args.output_dir / "summary.json"
    summaries_path.write_text(
        json.dumps([summary.model_dump(mode="json") for summary in summaries], indent=2)
        + "\n",
        encoding="utf-8",
    )
    config_path = args.output_dir / "configuration.json"
    config_path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, str | int | dict[str, str]] = {
        "schema_version": "trex47-manifest-v1",
        "lifecycle": "DEVELOPMENT_ONLY_NOT_FINAL",
        "pilot_id": config.pilot_id,
        "repository_head_at_execution_before_research_commit": _git_head(),
        "canonical_drive_experiments_id": "11aLrfuynQQfp6C4utseQHV-KTYl01kWfW-JAEMs2Ti4",
        "canonical_ownership": "Drive controls scientific interpretation; Git artifacts are development evidence",
        "method_release_tag_commit": "trexselector/TRexSelector v1.0.0: aa7e5cade4b7129e83417f19c437a811e828e8ac",
        "pypi_macos_arm64_cp311_sha256": "82b645e5bb8b67547fb35f652a83d0a9ca3d801839961f8d07c407e26b161f2e",
        "pypi_sdist_sha256": "61f0a57b275cf4fc2a2f8c690f75a5ce7364951f3f27caa2729a7128991ab538",
        "source_digests_sha256": {
            name: file_digest(root / name)
            for name in (
                "trex_known_truth.py",
                "trex_finite_k.py",
                "trex_finite_k_worker.py",
                "trex_finite_k_cli.py",
            )
        },
        "artifacts_sha256": {
            name: file_digest(args.output_dir / name)
            for name in ("configuration.json", "records.jsonl", "summary.json")
        },
        "python_runtime": runtime_provenance(),
        "dependency_versions": {
            name: _version(name)
            for name in ("trexselector", "numpy", "scipy", "scikit-learn", "pydantic")
        },
        "planned_primary_runs": len(config.scenarios)
        * config.independent_replicates
        * len(config.methods),
        "executed_worker_successes": sum(
            record.status == "success" for record in records
        ),
        "record_count_including_reruns_and_explicit_exclusions": len(records),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pilot_id": config.pilot_id,
                "records": len(records),
                "successful": manifest["executed_worker_successes"],
                "outcome_counts": {
                    status: sum(record.status == status for record in records)
                    for status in (
                        "success",
                        "failed",
                        "timeout",
                        "unavailable",
                        "unsupported",
                        "budget_exhausted",
                    )
                },
                "output_dir": str(args.output_dir),
                "manifest_sha256": hashlib.sha256(
                    (args.output_dir / "manifest.json").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
