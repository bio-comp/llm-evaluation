"""R1 discriminating controls for the existing Quantum Q0 branch (synthetic only)."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from llm_evaluation.local_runtime_q0 import (
    HostMemory,
    Q0CaseResult,
    Q0Manifest,
    Q0Packet,
    PythonClientIdentity,
    classify_case,
    observe_client_python,
    run_q0,
    verify_q0_private_evidence,
)
from llm_evaluation.runtime.codec import sha256_hex
from llm_evaluation.runtime.evidence import (
    GenerationExecution,
    GenerationObservation,
    RawResponseEvidence,
)
from llm_evaluation.runtime.failures import FailureKind, RuntimeFailure
from llm_evaluation.runtime.llamacpp.qualification import (
    ArtifactIdentity,
    ExternalConflictStatus,
    ListenerStatus,
    LlamaQualification,
)
from llm_evaluation.runtime.timing import RequestTiming

from test_local_runtime_q0 import (
    PACKET,
    REVISION,
    SHA,
    config,
    patch_identifiers,
)


def make_case(index: int, *, status: str = "PASS") -> dict[str, object]:
    task = Q0Packet.model_validate_json(PACKET.read_bytes()).tasks[index - 1]
    return {
        "task_id": task.id,
        "category": task.category,
        "status": status,
        "request_identity_sha256": "a" * 64,
        "request_body_sha256": "b" * 64,
        "response_body_sha256": "c" * 64,
        "private_request_file": f"case-{index:02d}-request.json",
        "private_response_file": f"case-{index:02d}-response.json",
        "finish_reason": "stop",
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
        "request_wall_seconds": 0.5,
        "prompt_processing_seconds": 0.1,
        "generation_seconds": 0.2,
        "visible_bytes": 2,
        "reasoning_bytes": 0,
    }


def complete_manifest() -> Q0Manifest:
    """Pure schema receipt; separate end-to-end test proves file-backed evidence."""
    return Q0Manifest(
        status="COMPLETED",
        failure_stage=None,
        failure_code=None,
        software_commit="a" * 40,
        package_source_sha256="b" * 64,
        run_id="20261122T140000Z-aabbccdd",
        repository="Qwen/Qwen3-8B-GGUF",
        revision=REVISION,
        artifact_filename="Qwen3-8B-Q4_K_M.gguf",
        expected_artifact_bytes=4,
        expected_artifact_sha256=SHA,
        artifact_bytes=4,
        artifact_sha256=SHA,
        expected_executable_sha256=SHA,
        expected_executable_version="1.0.0-dev",
        expected_executable_build=12602,
        expected_executable_commit="418fe97b5",
        executable_sha256=SHA,
        executable_version="1.0.0-dev",
        executable_build=12602,
        executable_commit="418fe97b5",
        client_python_version=observe_client_python().version,
        client_python_implementation=observe_client_python().implementation,
        client_python_executable_sha256=observe_client_python().executable_sha256,
        alias="q0-test",
        chat_template_sha256="c" * 64,
        chat_template_caps_sha256="d" * 64,
        build_info_sha256="e" * 64,
        actual_gpu_offload_layers=37,
        actual_gpu_total_layers=37,
        load_readiness_seconds=0.5,
        host_baseline=HostMemory(),
        owned_listener_verified=True,
        cleanup_verified=True,
        source_reverified=True,
        private_server_log="server.stderr.log",
        server_log_sha256="f" * 64,
        cases=tuple(Q0CaseResult.model_validate(make_case(i)) for i in range(1, 6)),
        wall_seconds=2,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["cases"][0].update(response_body_sha256=None),
        lambda d: d["cases"][0].update(private_response_file=None),
        lambda d: d["cases"][0].update(request_body_sha256="wrong"),
        lambda d: d["cases"][0].update(prompt_processing_seconds=None),
        lambda d: d["cases"][0].update(completion_tokens=129, total_tokens=137),
        lambda d: d["cases"][0].update(request_identity_sha256="q" * 64),
        lambda d: d["cases"][0].update(private_request_file="../outside.json"),
        lambda d: d["cases"][0].update(private_response_file="case-02-response.json"),
        lambda d: d.update(artifact_sha256="f" * 64),
        lambda d: d.update(executable_sha256=None),
        lambda d: d.update(source_reverified=False),
        lambda d: d.update(schema_version=1),
    ],
)
def test_completed_mutations_are_rejected(mutation) -> None:
    valid = complete_manifest()
    assert Q0Manifest.model_validate_json(valid.model_dump_json()) == valid
    data = json.loads(valid.model_dump_json())
    mutation(data)
    with pytest.raises(ValidationError):
        Q0Manifest.model_validate(data)


def test_valid_failure_case_without_received_response_is_not_completed() -> None:
    good = complete_manifest().model_dump()
    failed = make_case(3)
    failed.update(
        status="HTTP_FAILURE",
        response_body_sha256=None,
        private_response_file=None,
        finish_reason=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        request_wall_seconds=None,
        prompt_processing_seconds=None,
        generation_seconds=None,
        visible_bytes=None,
        reasoning_bytes=None,
        failure_kind="network_error",
    )
    good["status"] = "INCOMPLETE"
    good["failure_stage"] = "generation"
    good["failure_code"] = "ValueError"
    good["cases"] = (make_case(1), make_case(2), failed)
    record = Q0Manifest.model_validate(good)
    assert record.cases[-1].response_body_sha256 is None
    assert record.cases[-1].private_response_file is None
    good["status"] = "COMPLETED"
    good["failure_stage"] = None
    good["failure_code"] = None
    with pytest.raises(ValidationError):
        Q0Manifest.model_validate(good)


@given(above=st.integers(min_value=129, max_value=1_000_000))
def test_any_over_budget_report_is_not_valid_completion(above: int) -> None:
    task = Q0Packet.model_validate_json(PACKET.read_bytes()).tasks[0]
    assert (
        classify_case(
            task,
            visible=task.oracle.expected,
            finish="stop",
            reported_model_match=True,
            reasoning_bytes=0,
            completion_tokens=above,
            limit=128,
        )
        == "OVER_BUDGET"
    )


@pytest.mark.parametrize("condition", ["wrong-digest", "missing", "runtime-mismatch"])
def test_blocked_expected_vs_observed_identity(
    condition: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.software_identity",
        lambda: ("a" * 40, "b" * 64),
    )
    patch_identifiers(
        monkeypatch,
        c,
        model_sha="b" * 64 if condition == "wrong-digest" else SHA,
        runtime_sha="b" * 64 if condition == "runtime-mismatch" else SHA,
    )
    if condition == "missing":
        c.artifact.unlink()
    record = run_q0(c)
    assert record.status == "BLOCKED"
    assert record.expected_artifact_sha256 == SHA
    assert record.artifact_sha256 == (
        None
        if condition == "missing"
        else "b" * 64
        if condition == "wrong-digest"
        else SHA
    )
    if condition == "runtime-mismatch":
        assert record.expected_executable_sha256 == SHA
        assert record.executable_sha256 == "b" * 64
    with pytest.raises(ValidationError):
        Q0Manifest.model_validate(
            {
                **record.model_dump(),
                "status": "COMPLETED",
                "failure_stage": None,
                "failure_code": None,
            }
        )


def synthetic_completed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_at: int | None = None,
    fail_kind: FailureKind | None = None,
    cleanup_lease_fails: bool = False,
    corrupt_source_after_launch: bool = False,
) -> tuple[Q0Manifest, Path]:
    c = config(tmp_path)
    patch_identifiers(monkeypatch, c)
    if corrupt_source_after_launch:
        counts = 0
        from test_local_runtime_q0 import dummy_artifact

        def source_hash(path: Path | str) -> ArtifactIdentity:
            nonlocal counts
            counts += 1
            return dummy_artifact(c, "f" * 64 if counts >= 3 else SHA)

        monkeypatch.setattr(
            "llm_evaluation.local_runtime_q0.hash_model_artifact", source_hash
        )
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.software_identity",
        lambda: ("a" * 40, "b" * 64),
    )
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
    lease_calls = 0

    def fake_lease(root: Path | str, resource: str):
        nonlocal lease_calls
        lease_calls += 1
        if cleanup_lease_fails and lease_calls == 2:
            raise RuntimeError("synthetic lease reacquisition failure")
        return nullcontext()

    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.acquire_resource_lease", fake_lease
    )
    monkeypatch.setattr("llm_evaluation.local_runtime_q0._gpu_used_mib", lambda: None)
    monkeypatch.setattr(
        "llm_evaluation.runtime.llamacpp.qualification.observe_owned_listener",
        lambda pid, port: ListenerStatus.LOOPBACK_VERIFIED,
    )

    child = Mock(pid=991100)
    child.poll.return_value = None

    def spawn(*args: object, **kwargs: object) -> Mock:
        stream = kwargs["stdout"]
        stream.write(b"llama_model_load: offloaded 37/37 layers to GPU\n")
        stream.flush()
        return child

    def stop(handle) -> None:
        assert handle.process is child
        child.poll.return_value = 0

    monkeypatch.setattr("llm_evaluation.local_runtime_q0.subprocess.Popen", spawn)
    monkeypatch.setattr("llm_evaluation.local_runtime_q0.stop_managed_server", stop)

    async def qualified(*args: object, **kwargs: object) -> LlamaQualification:
        return LlamaQualification(
            total_slots=1,
            slot_count=1,
            model_path=str(c.artifact),
            alias=c.alias,
            chat_template_sha256="c" * 64,
            chat_template_caps=(("supports_tools", False),),
            chat_template_caps_sha256="d" * 64,
            build_info="b12602-418fe97b5",
            context_tokens=8192,
        )

    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.qualify_llama_server", qualified
    )
    packet = Q0Packet.model_validate_json(PACKET.read_bytes())
    answers = {task.prompt: task.oracle.expected for task in packet.tasks}

    class FakeBoundary:
        n = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def request_json(self, method: str, path: str):
            return {"status": "ok"}, None, 0.0

        async def generate(self, request):
            self.n += 1
            wire = json.loads(request.wire_bytes)
            prompt = wire["messages"][0]["content"]
            assert prompt in answers
            answer = answers[prompt]
            payload = {
                "model": c.alias,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 6,
                    "total_tokens": 26,
                },
                "timings": {"prompt_ms": 10.0, "predicted_ms": 20.0},
            }
            raw = json.dumps(payload, separators=(",", ":")).encode()
            evidence = RawResponseEvidence(body=raw, body_sha256=sha256_hex(raw))
            if fail_at == self.n and fail_kind is not None:
                raise RuntimeFailure(
                    fail_kind,
                    evidence=evidence
                    if fail_kind is FailureKind.INVALID_RESPONSE
                    else None,
                )
            return GenerationExecution(
                observation=GenerationObservation(
                    raw_output=answer,
                    backend_model_id=c.alias,
                    finish_reason="stop",
                    termination_reason="stop",
                    prompt_tokens=20,
                    completion_tokens=6,
                    total_tokens=26,
                    reported_model_matches_condition=True,
                ),
                evidence=evidence,
                effective_request=request,
                timing=RequestTiming(total_request_wall_seconds=0.5),
            )

    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.AsyncOpenAICompatibleTransport", FakeBoundary
    )
    record = run_q0(c)
    assert child.poll() == 0
    assert record.cleanup_verified is (not cleanup_lease_fails)
    assert record.source_reverified is (not corrupt_source_after_launch)
    assert record.private_server_log == "server.stderr.log"
    private_dir = c.private_root / ("q0-" + record.run_id)
    return record, private_dir


def test_complete_five_case_evidence_roundtrip_and_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, private_dir = synthetic_completed_run(tmp_path, monkeypatch)
    assert record.status == "COMPLETED"
    assert len(record.cases) == 5
    assert all(c.status == "PASS" for c in record.cases)
    c = config(tmp_path / "verification")
    c = c.model_copy(
        update={
            "artifact": tmp_path
            / "models--Qwen--Qwen3-8B-GGUF"
            / "snapshots"
            / REVISION
            / "Qwen3-8B-Q4_K_M.gguf"
        }
    )
    # Deserialization is independent of the producer object.
    reloaded = Q0Manifest.model_validate_json((tmp_path / "public.json").read_bytes())
    assert reloaded == record
    assert reloaded.client_python_version == observe_client_python().version
    verify_q0_private_evidence(
        reloaded, c, private_dir, Q0Packet.model_validate_json(PACKET.read_bytes())
    )
    bad = private_dir / record.cases[3].private_response_file
    bad.write_bytes(bad.read_bytes() + b"x")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_q0_private_evidence(
            reloaded, c, private_dir, Q0Packet.model_validate_json(PACKET.read_bytes())
        )


def test_bad_python_runtime_attestation_rejected_on_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, private_dir = synthetic_completed_run(tmp_path, monkeypatch)
    actual = observe_client_python()
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_q0.observe_client_python",
        lambda: PythonClientIdentity(
            version="3.12.0",
            implementation=actual.implementation,
            executable_sha256=actual.executable_sha256,
        ),
    )
    # Changing the observation used for readback cannot pass against historical receipt.
    c = config(tmp_path / "reader")
    c = c.model_copy(
        update={
            "artifact": tmp_path
            / "models--Qwen--Qwen3-8B-GGUF"
            / "snapshots"
            / REVISION
            / "Qwen3-8B-Q4_K_M.gguf"
        }
    )
    with pytest.raises(ValueError, match="Python client identity"):
        verify_q0_private_evidence(
            record, c, private_dir, Q0Packet.model_validate_json(PACKET.read_bytes())
        )


@pytest.mark.parametrize(
    "kind", [FailureKind.INVALID_RESPONSE, FailureKind.NETWORK_ERROR]
)
def test_invalid_or_missing_provider_response_preserves_only_received_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: FailureKind
) -> None:
    record, private_dir = synthetic_completed_run(
        tmp_path, monkeypatch, fail_at=3, fail_kind=kind
    )
    assert record.status == "INCOMPLETE"
    assert len(record.cases) == 3
    failed = record.cases[-1]
    assert failed.status == (
        "INVALID_RESPONSE" if kind is FailureKind.INVALID_RESPONSE else "HTTP_FAILURE"
    )
    if kind is FailureKind.INVALID_RESPONSE:
        assert failed.private_response_file is not None
        assert (private_dir / failed.private_response_file).is_file()
        assert (
            sha256_hex((private_dir / failed.private_response_file).read_bytes())
            == failed.response_body_sha256
        )
    else:
        assert failed.private_response_file is None
        assert failed.response_body_sha256 is None
    assert record.source_reverified is True


def test_cleanup_lease_failure_cannot_validate_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, _ = synthetic_completed_run(
        tmp_path,
        monkeypatch,
        cleanup_lease_fails=True,
    )
    assert record.status == "INCOMPLETE"
    assert record.cleanup_verified is False
    assert record.source_reverified is True
    assert record.failure_stage == "cleanup"
    assert (
        Q0Manifest.model_validate_json((tmp_path / "public.json").read_bytes())
        == record
    )


def test_generation_success_does_not_skip_mismatched_final_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, _ = synthetic_completed_run(
        tmp_path,
        monkeypatch,
        corrupt_source_after_launch=True,
    )
    assert len(record.cases) == 5
    assert record.status == "INCOMPLETE"
    assert record.source_reverified is False
    assert record.cleanup_verified is True
    assert record.failure_stage == "source_reverification"
    assert (
        Q0Manifest.model_validate_json((tmp_path / "public.json").read_bytes())
        == record
    )
