"""Independent R2 evidence-integrity counterexamples for Quantum Q0 (synthetic)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from llm_evaluation.local_runtime_q0 import (
    Q0Manifest,
    Q0Packet,
    Q0_CASE_CATEGORIES,
    Q0_CASE_IDS,
    Q0_PACKET_SHA256,
    run_q0,
    verify_q0_private_evidence,
)
from llm_evaluation.runtime.codec import sha256_hex
from test_local_runtime_q0 import PACKET, REVISION, config, patch_identifiers
from test_local_runtime_q0_r2 import complete_manifest, synthetic_completed_run


def test_r2_category_projection_matches_exact_frozen_packet() -> None:
    raw = PACKET.read_bytes()
    assert sha256_hex(raw) == Q0_PACKET_SHA256
    packet = Q0Packet.model_validate_json(raw)
    assert tuple(t.id for t in packet.tasks) == Q0_CASE_IDS
    assert tuple(t.category for t in packet.tasks) == Q0_CASE_CATEGORIES


def reader_config(tmp_path: Path):
    c = config(tmp_path / "reader")
    return c.model_copy(
        update={
            "artifact": (
                tmp_path
                / "models--Qwen--Qwen3-8B-GGUF"
                / "snapshots"
                / REVISION
                / "Qwen3-8B-Q4_K_M.gguf"
            )
        }
    )


def readback(record: Q0Manifest, c, private_dir: Path):
    return verify_q0_private_evidence(
        record, c, private_dir, Q0Packet.model_validate_json(PACKET.read_bytes())
    )


@pytest.mark.parametrize(
    "offloaded,total", [(0, 0), (0, 37), (37, 0), (None, 37), (37, None)]
)
def test_r2_completed_manifest_rejects_nonpositive_or_partial_offload(
    offloaded: int | None, total: int | None
) -> None:
    record = complete_manifest()
    changed = record.model_dump()
    changed["actual_gpu_offload_layers"] = offloaded
    changed["actual_gpu_total_layers"] = total
    with pytest.raises(ValidationError, match="COMPLETED"):
        Q0Manifest.model_validate(changed)


def test_r2_synthetic_full_gpu_offload_is_read_back_from_saved_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, evidence_dir = synthetic_completed_run(tmp_path, monkeypatch)
    assert record.status == "COMPLETED"
    assert (record.actual_gpu_offload_layers, record.actual_gpu_total_layers) == (
        37,
        37,
    )
    scope = readback(record, reader_config(tmp_path), evidence_dir)
    assert scope.gpu_offload == "digest_verified_server_log"


@pytest.mark.parametrize("values", [(0, 37), (0, 0)])
def test_r2_producer_rejects_false_full_offload_from_loader(
    values: tuple[int, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    count, total = values
    log = f"llama_model_load: offloaded {count}/{total} layers to GPU\n".encode()
    receipt, _ = synthetic_completed_run(
        tmp_path, monkeypatch, synthetic_offload_log=log
    )
    assert receipt.status == "INCOMPLETE"
    assert not receipt.cases


def test_r2_private_log_mismatch_rejected_even_when_hash_mutated_consistently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, evidence_dir = synthetic_completed_run(tmp_path, monkeypatch)
    log_path = evidence_dir / "server.stderr.log"
    bad = b"llama_model_load: offloaded 0/37 layers to GPU\n"
    log_path.write_bytes(bad)
    forged = Q0Manifest.model_validate(
        {**record.model_dump(), "server_log_sha256": sha256_hex(bad)}
    )
    with pytest.raises(ValueError, match="offload"):
        readback(forged, reader_config(tmp_path), evidence_dir)


def test_r2_public_offload_mutation_rejected_against_intact_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, evidence_dir = synthetic_completed_run(tmp_path, monkeypatch)
    forged = record.model_dump()
    forged["actual_gpu_offload_layers"] = 0
    with pytest.raises(ValidationError):
        Q0Manifest.model_validate(forged)
    # The private verifier must independently reject rather than assume callers validated.
    with pytest.raises((ValueError, ValidationError), match="offload|COMPLETED"):
        readback(
            record.model_copy(update={"actual_gpu_offload_layers": 0}),
            reader_config(tmp_path),
            evidence_dir,
        )


@pytest.mark.parametrize(
    "bad_log",
    [
        b"no observed GPU layer offload\n",
        b"llama_model_load: offloaded 0/0 layers to GPU\n",
        b"llama_model_load: offloaded 0/37 layers to GPU\n",
        (
            b"llama_model_load: offloaded 0/37 layers to GPU\n"
            b"llama_model_load: offloaded 37/37 layers to GPU\n"
        ),
    ],
)
def test_r2_missing_or_ambiguous_offload_evidence_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_log: bytes
) -> None:
    record, evidence_dir = synthetic_completed_run(tmp_path, monkeypatch)
    log_path = evidence_dir / "server.stderr.log"
    log_path.write_bytes(bad_log)
    forged = Q0Manifest.model_validate(
        {**record.model_dump(), "server_log_sha256": sha256_hex(bad_log)}
    )
    with pytest.raises(ValueError, match="offload"):
        readback(forged, reader_config(tmp_path), evidence_dir)


def test_r2_private_log_tamper_without_matching_digest_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, evidence_dir = synthetic_completed_run(tmp_path, monkeypatch)
    file = evidence_dir / "server.stderr.log"
    file.write_bytes(file.read_bytes() + b"alteration")
    with pytest.raises(ValueError, match="log corrupted"):
        readback(record, reader_config(tmp_path), evidence_dir)


@pytest.mark.parametrize("field", ["packet_sha256", "e002_packet_sha256"])
def test_r2_frozen_digest_cannot_be_falsified(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, evidence_dir = synthetic_completed_run(tmp_path, monkeypatch)
    data = record.model_dump()
    data[field] = "f" * 64
    with pytest.raises(ValidationError, match="packet"):
        Q0Manifest.model_validate(data)
    with pytest.raises((ValidationError, ValueError)):
        readback(
            record.model_copy(update={field: "f" * 64}),
            reader_config(tmp_path),
            evidence_dir,
        )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("software_commit", "f" * 40),
        ("package_source_sha256", "f" * 64),
    ],
)
def test_r2_same_checkout_software_identity_rejects_forged_receipt(
    field: str, bad: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, evidence_dir = synthetic_completed_run(tmp_path, monkeypatch)
    forged = Q0Manifest.model_validate({**record.model_dump(), field: bad})
    with pytest.raises(ValueError, match="software identity"):
        readback(forged, reader_config(tmp_path), evidence_dir)


@pytest.mark.parametrize("index", range(5))
def test_r2_case_category_is_bound_to_frozen_packet(
    index: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, evidence_dir = synthetic_completed_run(tmp_path, monkeypatch)
    wrong = record.model_dump()
    wrong["cases"] = list(wrong["cases"])
    wrong["cases"][index] = {**wrong["cases"][index], "category": "MISMATCHED"}
    with pytest.raises(ValidationError, match="category"):
        Q0Manifest.model_validate_json(json.dumps(wrong))
    with pytest.raises((ValidationError, ValueError), match="category"):
        readback(
            record.model_copy(
                update={
                    "cases": tuple(record.cases[:index])
                    + (
                        record.cases[index].model_copy(
                            update={"category": "MISMATCHED"}
                        ),
                    )
                    + tuple(record.cases[index + 1 :])
                }
            ),
            reader_config(tmp_path),
            evidence_dir,
        )


@pytest.mark.parametrize("field", ["chat_template_caps_sha256", "build_info_sha256"])
def test_r2_unretained_qualification_hashes_are_explicitly_not_readback_verified(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, evidence_dir = synthetic_completed_run(tmp_path, monkeypatch)
    forged = Q0Manifest.model_validate({**record.model_dump(), field: "f" * 64})
    scope = readback(forged, reader_config(tmp_path), evidence_dir)
    assert scope.qualification_metadata == "recorded_only_not_independently_verified"
    assert field in scope.unverified_fields


def test_r2_startup_after_launch_attempt_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    patch_identifiers(monkeypatch, c)
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.software_identity",
        lambda: ("a" * 40, "b" * 64),
    )

    def fail_attempt(config, artifact, runtime, packet, private_dir, state):
        state["failure_stage"] = "startup"
        raise RuntimeError("synthetic owned-child startup failure")

    monkeypatch.setattr("llm_evaluation.local_runtime_q0._collect", fail_attempt)
    # This mock isolates the run-stage transition, without starting a server.
    with patch(
        "llm_evaluation.local_runtime_q0.probe_known_external_conflicts",
        return_value=None,
    ):
        result = run_q0(c)
    assert result.status == "INCOMPLETE"
    assert result.failure_stage == "startup"
    assert result.failure_code == "RuntimeError"
    assert not result.cases


def test_r2_preflight_rejection_remains_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    c.artifact.unlink()
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.software_identity",
        lambda: ("a" * 40, "b" * 64),
    )
    receipt = run_q0(c)
    assert receipt.status == "BLOCKED"
    assert receipt.failure_stage == "identity"


@pytest.mark.parametrize("kind", ["request", "response"])
def test_r2_corrupt_case_bytes_still_rejected(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, evidence_dir = synthetic_completed_run(tmp_path, monkeypatch)
    case = record.cases[0]
    name = (
        case.private_request_file if kind == "request" else case.private_response_file
    )
    assert name is not None
    file = evidence_dir / name
    file.write_bytes(file.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="evidence|digest"):
        readback(record, reader_config(tmp_path), evidence_dir)
