"""Discriminating Quantum Q0 controls; no real server or model is launched."""

from __future__ import annotations

import asyncio
import socket
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from llm_evaluation.local_runtime_q0 import (
    Q0Config,
    Q0Manifest,
    Q0Packet,
    Q0Task,
    Q0Oracle,
    _collect,
    _gpu_used_mib,
    _offload_observation,
    _port_is_free,
    classify_case,
    run_q0,
    validate_identity,
)
from llm_evaluation.runtime.llamacpp.qualification import (
    ArtifactIdentity,
    ExecutableIdentity,
    ExpectedExecutable,
    ExternalConflictStatus,
)


PACKET = (
    Path(__file__).resolve().parents[1] / "fixtures/local_runtime_preflight_v1.json"
)
SHA = "a" * 64
REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"


def config(tmp_path: Path) -> Q0Config:
    model = (
        tmp_path
        / "models--Qwen--Qwen3-8B-GGUF"
        / "snapshots"
        / REVISION
        / "Qwen3-8B-Q4_K_M.gguf"
    )
    model.parent.mkdir(parents=True)
    model.write_bytes(b"GGUF")
    return Q0Config(
        artifact=model,
        expected_size_bytes=4,
        expected_sha256=SHA,
        repository="Qwen/Qwen3-8B-GGUF",
        revision=REVISION,
        executable=tmp_path / "llama-server",
        expected_runtime=ExpectedExecutable(
            version="1.0.0-dev",
            build=12602,
            commit="418fe97b5",
            sha256=SHA,
        ),
        packet=PACKET,
        public_manifest=tmp_path / "public.json",
        private_root=tmp_path / "private",
        alias="q0-test",
        port=18081,
    )


def dummy_runtime(config: Q0Config, sha: str = SHA) -> ExecutableIdentity:
    return ExecutableIdentity(
        path=str(config.executable),
        sha256=sha,
        version="1.0.0-dev",
        build=12602,
        commit="418fe97b5",
        platform="linux",
    )


def dummy_artifact(config: Q0Config, sha: str = SHA) -> ArtifactIdentity:
    return ArtifactIdentity(path=str(config.artifact), sha256=sha, size_bytes=4)


def patch_identifiers(
    monkeypatch: pytest.MonkeyPatch,
    c: Q0Config,
    *,
    model_sha: str = SHA,
    runtime_sha: str = SHA,
) -> None:
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.hash_model_artifact",
        lambda path: dummy_artifact(c, sha=model_sha),
    )
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.inspect_llama_executable",
        lambda path: dummy_runtime(c, sha=runtime_sha),
    )


def test_frozen_packet_validated_without_rewriting_bytes() -> None:
    b = PACKET.read_bytes()
    packet = Q0Packet.model_validate_json(b)
    assert len(packet.tasks) == 5
    assert packet.tasks[2].oracle.expected == "645"
    altered = bytearray(b)
    altered[-2] = ord(" ")
    assert bytes(altered) != b


def test_identity_accepts_only_correct_pinned_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    patch_identifiers(monkeypatch, c)
    artifact, runtime, packet = validate_identity(c)
    assert artifact.sha256 == SHA
    assert runtime.sha256 == SHA
    assert packet.tasks[0].id == "exact-format-control"


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "truncated",
        "wrong-size",
        "wrong-sha",
        "wrong-runtime",
        "wrong-revision",
        "wrong-packet",
    ],
)
def test_known_bad_identity_rejected_before_launch(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    patch_identifiers(
        monkeypatch,
        c,
        model_sha="b" * 64 if kind == "wrong-sha" else SHA,
        runtime_sha="b" * 64 if kind == "wrong-runtime" else SHA,
    )
    if kind == "missing":
        c.artifact.unlink()
    elif kind == "truncated":
        c.artifact.write_bytes(b"GG")
    elif kind == "wrong-size":
        c = c.model_copy(update={"expected_size_bytes": 20})
    elif kind == "wrong-revision":
        c = c.model_copy(update={"revision": "0" * 40})
    elif kind == "wrong-packet":
        p = tmp_path / "wrong-packet.json"
        p.write_bytes(PACKET.read_bytes() + b"\n")
        c = c.model_copy(update={"packet": p})
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_identity(c)


def test_missing_material_emits_blocked_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.software_identity",
        lambda: ("a" * 40, "b" * 64),
    )
    c = config(tmp_path)
    c.artifact.unlink()
    result = run_q0(c)
    assert result.status == "BLOCKED"
    assert not result.cases
    assert result.failure_stage == "identity"
    assert Q0Manifest.model_validate_json(c.public_manifest.read_bytes()) == result
    assert not c.private_root.exists()


def test_failed_manifest_cannot_claim_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.software_identity",
        lambda: ("a" * 40, "b" * 64),
    )
    c = config(tmp_path)
    result = run_q0(c.model_copy(update={"expected_size_bytes": 9}))
    invalid = result.model_dump()
    invalid.update(status="COMPLETED", failure_stage=None, failure_code=None)
    with pytest.raises(ValidationError, match="COMPLETED requires"):
        Q0Manifest.model_validate(invalid)


def test_port_protection_and_occupied_listener(tmp_path: Path) -> None:
    c = config(tmp_path)
    assert c.context_tokens == 8192
    with pytest.raises(ValidationError):
        Q0Config.model_validate({**c.model_dump(), "context_tokens": 4096})
    with pytest.raises(ValidationError, match="protected MCP"):
        Q0Config.model_validate({**c.model_dump(), "port": 8080})
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        assert not _port_is_free(port)
    assert _port_is_free(port)


@pytest.mark.parametrize(
    "status",
    [ExternalConflictStatus.CONFLICT_DETECTED, ExternalConflictStatus.UNVERIFIED],
)
def test_external_conflict_blocks_before_spawn(
    status: ExternalConflictStatus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.probe_known_external_conflicts", lambda: status
    )
    forbidden = Mock(side_effect=AssertionError("unexpected launch"))
    monkeypatch.setattr("llm_evaluation.local_runtime_q0.subprocess.Popen", forbidden)
    state = {}
    with pytest.raises(ValueError, match="external runtime conflict"):
        asyncio.run(
            _collect(
                c,
                dummy_artifact(c),
                dummy_runtime(c),
                Q0Packet.model_validate_json(PACKET.read_bytes()),
                tmp_path,
                state,
            )
        )
    forbidden.assert_not_called()


def test_occupied_port_blocks_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.probe_known_external_conflicts",
        lambda: ExternalConflictStatus.NO_KNOWN_CONFLICT,
    )
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0._lm_studio_api_stopped", lambda: True
    )
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0._port_is_free", lambda port: False
    )
    forbidden = Mock(side_effect=AssertionError("unexpected launch"))
    monkeypatch.setattr("llm_evaluation.local_runtime_q0.subprocess.Popen", forbidden)
    with pytest.raises(ValueError, match="port occupied"):
        asyncio.run(
            _collect(
                c,
                dummy_artifact(c),
                dummy_runtime(c),
                Q0Packet.model_validate_json(PACKET.read_bytes()),
                tmp_path,
                {},
            )
        )
    forbidden.assert_not_called()


def test_classify_case_differentiates_oracle_and_technical_failures() -> None:
    task = Q0Task(
        id="t",
        category="exact",
        prompt="answer",
        oracle=Q0Oracle(type="exact_text", expected="256"),
    )
    arguments = dict(
        task=task,
        visible="256",
        finish="stop",
        reported_model_match=True,
        reasoning_bytes=0,
        completion_tokens=2,
        limit=128,
    )
    assert classify_case(**arguments) == "PASS"
    assert classify_case(**(arguments | {"visible": "257"})) == "FAIL_ORACLE"
    assert classify_case(**(arguments | {"visible": " 256"})) == "FAIL_ORACLE"
    assert classify_case(**(arguments | {"visible": ""})) == "EMPTY_VISIBLE"
    assert classify_case(**(arguments | {"finish": "length"})) == "CENSORED_LENGTH"
    assert (
        classify_case(**(arguments | {"finish": "stop", "reasoning_bytes": 10}))
        == "INVALID_RESPONSE"
    )
    assert (
        classify_case(**(arguments | {"reported_model_match": False}))
        == "MODEL_MISMATCH"
    )
    assert (
        classify_case(**(arguments | {"reported_model_match": None}))
        == "MODEL_MISMATCH"
    )
    assert classify_case(**(arguments | {"finish": "unexpected"})) == "INVALID_RESPONSE"
    assert (
        classify_case(**(arguments | {"measurement_complete": False}))
        == "MISSING_MEASUREMENT"
    )


@given(visible=st.text(max_size=30).filter(lambda value: value != "256"))
def test_exact_oracle_never_accepts_nonmatching_visible_text(visible: str) -> None:
    task = Q0Task(
        id="t",
        category="exact",
        prompt="answer",
        oracle=Q0Oracle(type="exact_text", expected="256"),
    )
    assert (
        classify_case(
            task,
            visible=visible,
            finish="stop",
            reported_model_match=True,
            reasoning_bytes=0,
            completion_tokens=1,
            limit=128,
        )
        != "PASS"
    )


def test_offload_control_rejects_missing_or_partial_evidence() -> None:
    assert _offload_observation(b"CUDA0 buffer size 5 GiB") == (None, None)
    assert _offload_observation(b"offloaded 34/36 layers to GPU") == (34, 36)
    assert _offload_observation(b"offloaded 36/36 layers to GPU") == (36, 36)


def test_missing_gpu_instrumentation_is_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.subprocess.run",
        Mock(side_effect=FileNotFoundError),
    )
    assert _gpu_used_mib() is None


def test_startup_error_cleans_up_only_owned_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    patch_identifiers(monkeypatch, c)
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.probe_known_external_conflicts",
        lambda: ExternalConflictStatus.NO_KNOWN_CONFLICT,
    )
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0._lm_studio_api_stopped", lambda: True
    )
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0._port_is_free", lambda port: True
    )
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.acquire_resource_lease",
        lambda root, resource: nullcontext(),
    )
    monkeypatch.setattr("llm_evaluation.local_runtime_q0._gpu_used_mib", lambda: None)
    child = Mock(pid=999999)
    child.poll.return_value = None
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.subprocess.Popen", Mock(return_value=child)
    )
    stopped: list[int] = []
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.stop_managed_server",
        lambda handle: stopped.append(handle.process.pid),
    )

    class FailingTransport:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def request_json(self, method: str, path: str) -> None:
            raise ValueError("synthetic startup transport failure")

    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.AsyncOpenAICompatibleTransport",
        FailingTransport,
    )
    state = {}
    with pytest.raises(ValueError, match="synthetic startup"):
        asyncio.run(
            _collect(
                c,
                dummy_artifact(c),
                dummy_runtime(c),
                Q0Packet.model_validate_json(PACKET.read_bytes()),
                tmp_path,
                state,
            )
        )
    assert stopped == [999999]
    assert state["cleanup_verified"] is True


def test_no_scientific_status_field_in_public_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.software_identity",
        lambda: ("a" * 40, "b" * 64),
    )
    c = config(tmp_path)
    c.artifact.unlink()
    payload = run_q0(c).model_dump_json()
    assert "scientific_approval" not in payload
    assert '"lm_studio":"NOT_TESTED"' in payload
