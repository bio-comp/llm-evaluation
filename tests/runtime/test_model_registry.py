import hashlib
import json
from pathlib import Path

import pytest


def _entry(model: Path) -> dict[str, object]:
    return {
        "logical_id": "approved-fixture",
        "name": "Approved fixture",
        "artifact_path": str(model.resolve()),
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "size_bytes": model.stat().st_size,
        "format": "gguf",
        "quantization": "Q4_K_M",
        "architecture": "fixture",
        "context_capability": 4096,
        "date_verified": "2026-10-26",
        "llamacpp": {
            "status": "qualified",
            "served_model_id": "approved-fixture",
            "runtime_identity": "llama.cpp 0.7.0 build 11669 commit 620d2225c",
            "qualified_on": "2026-10-26",
        },
        "lm_studio": {"status": "unqualified"},
    }


def _write(path: Path, entry: dict[str, object]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "models": [entry]}))


def test_registry_resolves_exact_approved_bytes_and_distinguishes_runtime_status(
    tmp_path: Path,
):
    from llm_evaluation.model_registry import resolve_approved_model

    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"GGUFfixture")
    registry = tmp_path / "models.json"
    _write(registry, _entry(model))

    approved = resolve_approved_model(registry, "approved-fixture", runtime="llamacpp")
    assert approved.logical_id == "approved-fixture"
    assert approved.sha256 == hashlib.sha256(model.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="not qualified"):
        resolve_approved_model(registry, "approved-fixture", runtime="lm_studio")


def test_registry_fails_closed_on_missing_wrong_or_unapproved_artifact(tmp_path: Path):
    from llm_evaluation.model_registry import resolve_approved_model

    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"GGUFfixture")
    registry = tmp_path / "models.json"
    entry = _entry(model)
    _write(registry, entry)

    with pytest.raises(KeyError):
        resolve_approved_model(registry, "not-approved", runtime="llamacpp")

    entry["sha256"] = "0" * 64
    _write(registry, entry)
    with pytest.raises(ValueError, match="digest"):
        resolve_approved_model(registry, "approved-fixture", runtime="llamacpp")

    entry = _entry(model)
    entry["artifact_path"] = str(tmp_path / "missing.gguf")
    _write(registry, entry)
    with pytest.raises(ValueError, match="unavailable"):
        resolve_approved_model(registry, "approved-fixture", runtime="llamacpp")


def test_registry_rejects_relative_paths_duplicate_ids_and_qualified_runtime_without_identity(
    tmp_path: Path,
):
    from pydantic import ValidationError

    from llm_evaluation.model_registry import ApprovedModelRegistry

    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"GGUFfixture")
    entry = _entry(model)

    relative = {**entry, "artifact_path": "fixture.gguf"}
    with pytest.raises(ValidationError, match="absolute"):
        ApprovedModelRegistry.model_validate(
            {"schema_version": 1, "models": [relative]}
        )

    with pytest.raises(ValidationError, match="duplicate"):
        ApprovedModelRegistry.model_validate(
            {"schema_version": 1, "models": [entry, entry]}
        )

    bad_runtime = {**entry, "llamacpp": {"status": "qualified"}}
    with pytest.raises(ValidationError, match="qualified runtime"):
        ApprovedModelRegistry.model_validate(
            {"schema_version": 1, "models": [bad_runtime]}
        )
