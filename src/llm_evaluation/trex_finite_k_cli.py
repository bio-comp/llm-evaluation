"""Issue #47 executable development pilot, not a FINAL FDR certification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from llm_evaluation.trex_finite_k import (
    PilotConfig,
    StrictRecord,
    execute,
    file_digest,
    runtime_provenance,
)
from llm_evaluation.trex_runtime_provenance import (
    InstalledArtifact,
    IntendedRelease,
    inspect_installed_trex,
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


class DevelopmentManifest(StrictRecord):
    """Typed v2 persistence boundary; historical v1 manifests remain immutable."""

    schema_version: Literal["trex47-manifest-v2"] = "trex47-manifest-v2"
    lifecycle: Literal["DEVELOPMENT_ONLY_NOT_FINAL"] = "DEVELOPMENT_ONLY_NOT_FINAL"
    pilot_id: str = Field(min_length=1)
    repository_head_at_execution: str
    canonical_drive_experiments_id: str
    canonical_ownership: str
    intended_upstream_release: IntendedRelease
    installed_upstream_artifact: InstalledArtifact
    source_digests_sha256: dict[str, str]
    artifacts_sha256: dict[str, str]
    python_runtime: dict[str, str]
    dependency_versions: dict[str, str]
    planned_primary_runs: int = Field(ge=1)
    executed_worker_successes: int = Field(ge=0)
    successful_native_trex_executions: int = Field(ge=0)
    record_count_including_reruns_and_explicit_exclusions: int = Field(ge=0)

    @model_validator(mode="after")
    def require_native_qualification(self) -> DevelopmentManifest:
        if (
            self.successful_native_trex_executions
            and self.installed_upstream_artifact.status != "qualified"
        ):
            raise ValueError(
                "T-Rex success may not claim an unqualified installed artifact"
            )
        if (
            self.executed_worker_successes
            > self.record_count_including_reruns_and_explicit_exclusions
        ):
            raise ValueError("successful outcomes exceed recorded attempts")
        return self


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

    # Installed runtime evidence is observed independently from the intended
    # upstream release. Only exact file-content matches qualify native success.
    observed = inspect_installed_trex()
    manifest = DevelopmentManifest(
        pilot_id=config.pilot_id,
        repository_head_at_execution=_git_head(),
        canonical_drive_experiments_id="11aLrfuynQQfp6C4utseQHV-KTYl01kWfW-JAEMs2Ti4",
        canonical_ownership="Drive controls scientific interpretation; Git artifacts are development evidence",
        intended_upstream_release=IntendedRelease(),
        installed_upstream_artifact=observed,
        source_digests_sha256={
            name: file_digest(root / name)
            for name in (
                "trex_known_truth.py",
                "trex_finite_k.py",
                "trex_finite_k_worker.py",
                "trex_finite_k_cli.py",
                "trex_runtime_provenance.py",
            )
        },
        artifacts_sha256={
            name: file_digest(args.output_dir / name)
            for name in ("configuration.json", "records.jsonl", "summary.json")
        },
        python_runtime=runtime_provenance(),
        dependency_versions={
            name: _version(name)
            for name in ("trexselector", "numpy", "scipy", "scikit-learn", "pydantic")
        },
        planned_primary_runs=len(config.scenarios)
        * config.independent_replicates
        * len(config.methods),
        executed_worker_successes=sum(record.status == "success" for record in records),
        successful_native_trex_executions=sum(
            record.status == "success" and record.method_kind in ("trex", "trex_da")
            for record in records
        ),
        record_count_including_reruns_and_explicit_exclusions=len(records),
    )
    (args.output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pilot_id": config.pilot_id,
                "records": len(records),
                "successful": manifest.executed_worker_successes,
                "outcome_counts": {
                    status: sum(record.status == status for record in records)
                    for status in (
                        "success",
                        "failed",
                        "timeout",
                        "unavailable",
                        "unqualified",
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
