"""Executable R1 counterexamples for runtime identity, spawn failures and reruns."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import llm_evaluation.trex_finite_k as lab
import llm_evaluation.trex_finite_k_cli as cli
import llm_evaluation.trex_finite_k_worker as worker
import llm_evaluation.trex_runtime_provenance as provenance
from llm_evaluation.trex_finite_k import (
    Method,
    PilotConfig,
    Scenario,
    WorkerRequest,
    WorkerResult,
)


def _config(*, method: Method, replicates: int = 1) -> PilotConfig:
    return PilotConfig(
        schema_version="trex47-config-v1",
        pilot_id="r1-fixture",
        development_namespace="issue47-development-v1",
        scenarios=(
            Scenario(
                name="test",
                n=40,
                p=8,
                covariance="independent_standard_gaussian",
                active=(0,),
                coefficient=2.0,
            ),
        ),
        methods=(method,),
        independent_replicates=replicates,
        target_q=0.2,
        seed_base=123,
        analysis_seed=456,
        bootstrap_draws=100,
        per_method_timeout_seconds=1.0,
        overall_budget_seconds=15.0,
    )


def _request() -> WorkerRequest:
    return WorkerRequest(
        kind="trex",
        k=10,
        da_method=None,
        seed=33,
        q=0.2,
        x=[[1.0, 2.0], [3.0, 4.0]],
        y=[1.0, 2.0],
        stability_subsamples=8,
    )


class _FakeDistribution:
    def __init__(self, root: Path, version: str, files: list[str] | None) -> None:
        self.root = root
        self.version = version
        self.files = files

    def locate_file(self, path: object) -> Path:
        return self.root / str(path)


def test_installed_runtime_missing_and_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(_: str) -> _FakeDistribution:
        raise importlib.metadata.PackageNotFoundError("trexselector")

    monkeypatch.setattr(provenance.importlib.metadata, "distribution", missing)
    absent = provenance.inspect_installed_trex()
    assert absent.status == "unavailable"
    assert absent.observed_package_payload_sha256 is None
    assert absent.installed_version is None

    fake = _FakeDistribution(tmp_path, "2.0.0", None)
    monkeypatch.setattr(provenance.importlib.metadata, "distribution", lambda _: fake)
    mismatched = provenance.inspect_installed_trex()
    assert mismatched.status == "unqualified"
    assert mismatched.installed_version == "2.0.0"
    assert mismatched.observed_package_payload_sha256 is None

    fake.version = "1.0.0"
    monkeypatch.setattr(provenance, "_supported_runtime", lambda: False)
    unsupported_abi = provenance.inspect_installed_trex()
    assert unsupported_abi.status == "unqualified"
    assert "CPython 3.11" in str(unsupported_abi.reason)
    assert unsupported_abi.observed_package_payload_sha256 is None


def test_qualified_package_requires_exact_wheel_payload_and_import_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "trexselector"
    package.mkdir()
    initializer = package / "__init__.py"
    binary = package / "_core.cpython-311-darwin.so"
    initializer.write_bytes(b"known-verified-python-package")
    binary.write_bytes(b"known-verified-native-extension")
    files = [
        ("trexselector/__init__.py", initializer),
        ("trexselector/_core.cpython-311-darwin.so", binary),
    ]
    expected = provenance._payload_digest(files)
    native_expected = hashlib.sha256(binary.read_bytes()).hexdigest()
    fake = _FakeDistribution(tmp_path, "1.0.0", [name for name, _ in files])
    monkeypatch.setattr(provenance.importlib.metadata, "distribution", lambda _: fake)
    monkeypatch.setattr(provenance, "_supported_runtime", lambda: True)
    monkeypatch.setattr(provenance, "EXPECTED_PACKAGE_FILES", 2)
    monkeypatch.setattr(provenance, "EXPECTED_PACKAGE_PAYLOAD_SHA256", expected)
    monkeypatch.setattr(provenance, "EXPECTED_NATIVE_SHA256", native_expected)
    monkeypatch.setattr(
        provenance.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(initializer)),
    )

    qualified = provenance.inspect_installed_trex()
    assert qualified.status == "qualified"
    assert qualified.installed_version == "1.0.0"
    assert qualified.observed_package_payload_sha256 == expected
    assert qualified.observed_native_sha256 == native_expected
    assert qualified.verified_file_count == 2

    # Known-bad counterexample: same version and metadata but tampered bytes.
    binary.write_bytes(b"tampered-native-extension")
    tampered = provenance.inspect_installed_trex()
    assert tampered.status == "unqualified"
    assert tampered.observed_package_payload_sha256 != expected
    assert tampered.observed_native_sha256 != native_expected

    binary.write_bytes(b"known-verified-native-extension")
    (package / "shadow.py").write_text("# unexpected code")
    assert provenance.inspect_installed_trex().status == "unqualified"
    (package / "shadow.py").unlink()
    monkeypatch.setattr(
        provenance.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(tmp_path / "other/__init__.py")),
    )
    assert provenance.inspect_installed_trex().status == "unqualified"


@pytest.mark.parametrize(
    ("status", "expected_exception"),
    [
        ("unavailable", ModuleNotFoundError),
        ("unqualified", worker.UnqualifiedTRexRuntime),
    ],
)
def test_native_worker_refuses_unverified_runtime_before_native_import(
    status: str, expected_exception: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        worker,
        "inspect_installed_trex",
        lambda: provenance.InstalledArtifact.model_validate(
            {"status": status, "reason": "rejected test package"}
        ),
    )
    with pytest.raises(expected_exception):
        worker.run(_request())


def test_unqualified_worker_exit_is_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = subprocess.CompletedProcess(
        args=["worker"], returncode=43, stdout="", stderr="unqualified research runtime"
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: failure)
    status, result, diagnostic, elapsed = lab._invoke_worker(_request(), 1)
    assert status == "unqualified" and result is None
    assert "unqualified research runtime" in str(diagnostic)
    assert elapsed >= 0


def test_spawn_failure_preserves_prior_and_following_scored_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    success = subprocess.CompletedProcess(
        args=["worker"],
        returncode=0,
        stdout=WorkerResult(selected=(0,), effective_k=None).model_dump_json(),
        stderr="",
    )

    def injected(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(24, "Too many open files", "/sensitive/private/path")
        return success

    monkeypatch.setattr(subprocess, "run", injected)
    config = _config(method=Method(name="bh", kind="bh_independent"), replicates=3)
    records, summaries = lab.execute(config)
    assert calls == 3
    assert [r.status for r in records] == ["success", "failed", "success"]
    assert records[1].selected is None and records[1].fdp is None
    assert records[1].elapsed_seconds is not None and records[1].elapsed_seconds >= 0
    assert "errno=24" in str(records[1].error)
    assert "/sensitive" not in str(records[1].error)
    assert summaries[0].successful_datasets == 2
    assert summaries[0].failures == {"failed": 1}
    assert summaries[0].fdr == 0.0


def test_three_rerun_known_bad_first_two_only_configuration_rejected() -> None:
    # Three selected sets {0}, {0}, {} have all-pair mean 1/3, not the
    # first-two-only value 1. The supported estimand intentionally uses two.
    assert sum(
        lab._jaccard(a, b) for a, b in [((0,), (0,)), ((0,), ()), ((0,), ())]
    ) / 3 == pytest.approx(1 / 3)
    for count in (3, 4):
        with pytest.raises(ValidationError):
            Method(name="trex", kind="trex", k=10, reruns=count)
    assert Method(name="trex", kind="trex", k=10, reruns=2).reruns == 2


def test_incomplete_second_rerun_not_reported_as_full_stability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def injected(
        req: WorkerRequest, timeout: float
    ) -> tuple[lab.Status, WorkerResult | None, str | None, float]:
        nonlocal calls
        calls += 1
        if calls == 4:
            return "failed", None, "deliberate incomplete rerun", 0.01
        return "success", WorkerResult(selected=(0,), effective_k=None), None, 0.01

    monkeypatch.setattr(lab, "_invoke_worker", injected)
    config = _config(
        method=Method(name="bh", kind="bh_independent", reruns=2), replicates=2
    )
    records, summaries = lab.execute(config)
    assert len(records) == 4 and calls == 4
    assert [r.status for r in records] == ["success", "success", "success", "failed"]
    summary = summaries[0]
    assert summary.mean_rerun_jaccard == 1.0
    assert summary.rerun_pairs_completed == 1
    assert summary.rerun_pairs_incomplete == 1
    assert summary.successful_datasets == 2
    assert summary.failures == {}


def test_manifest_boundary_rejects_unqualified_native_success() -> None:
    with pytest.raises(ValidationError, match="T-Rex success"):
        cli.DevelopmentManifest(
            pilot_id="test",
            repository_head_at_execution="a" * 40,
            canonical_drive_experiments_id="drive",
            canonical_ownership="research",
            intended_upstream_release=provenance.IntendedRelease(),
            installed_upstream_artifact=provenance.InstalledArtifact(
                status="unavailable"
            ),
            source_digests_sha256={"source.py": "abc"},
            artifacts_sha256={"records.jsonl": "def"},
            python_runtime={"python": "3.11"},
            dependency_versions={"trexselector": "not-installed"},
            planned_primary_runs=1,
            executed_worker_successes=1,
            successful_native_trex_executions=1,
            record_count_including_reruns_and_explicit_exclusions=1,
        )


@pytest.mark.parametrize(
    ("status", "version"),
    [("unavailable", None), ("unqualified", "9.0.0")],
)
def test_cli_manifest_separates_expected_from_unverified_installed_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    version: str | None,
) -> None:
    config_path = tmp_path / "config.json"
    config = _config(method=Method(name="trex", kind="trex", k=10))
    config_path.write_text(config.model_dump_json())
    output = tmp_path / status
    monkeypatch.setattr(
        cli,
        "inspect_installed_trex",
        lambda: provenance.InstalledArtifact.model_validate(
            {"status": status, "installed_version": version}
        ),
    )
    monkeypatch.setattr(
        lab,
        "_invoke_worker",
        lambda req, timeout: (status, None, "unverified test runtime", 0.01),
    )
    monkeypatch.setattr(
        sys, "argv", ["trex", "--config", str(config_path), "--output-dir", str(output)]
    )
    assert cli.main() == 0
    manifest = cli.DevelopmentManifest.model_validate_json(
        (output / "manifest.json").read_bytes()
    )
    assert manifest.schema_version == "trex47-manifest-v2"
    assert manifest.intended_upstream_release.version == "1.0.0"
    assert manifest.installed_upstream_artifact.status == status
    assert manifest.installed_upstream_artifact.installed_version == version
    assert manifest.installed_upstream_artifact.observed_package_payload_sha256 is None
    assert manifest.successful_native_trex_executions == 0
    rows = (output / "records.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["status"] == status
    assert json.loads(capsys.readouterr().out)["outcome_counts"][status] == 1


def test_historical_dev_manifests_are_preserved_and_artifacts_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    for i in (1, 2, 3):
        base = root / "research" / f"trex47-dev-{i:03d}-output"
        manifest = json.loads((base / "manifest.json").read_bytes())
        assert manifest["schema_version"] == "trex47-manifest-v1"
        assert manifest["lifecycle"] == "DEVELOPMENT_ONLY_NOT_FINAL"
        for name, expected_digest in manifest["artifacts_sha256"].items():
            assert (
                hashlib.sha256((base / name).read_bytes()).hexdigest()
                == expected_digest
            )
    docs = (root / "docs/trex-finite-k-development-47.md").read_text()
    assert "research/trex47-dev-003-output" in docs
    assert "dev-001/dev-002" in docs
