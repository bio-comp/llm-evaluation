import asyncio
import hashlib
import json
import pytest
from llm_evaluation.runtime.llamacpp.qualification import (
    qualify_llama_server,
    ExternalConflictStatus,
    classify_processes,
)


class Fake:
    def __init__(self, path, slots=1, model="alias"):
        self.path = path
        self.slots = slots
        self.model = model

    async def request_json(self, method, path):
        bodies = {
            "/health": {"status": "ok"},
            "/props": {
                "total_slots": self.slots,
                "model_path": self.path,
                "chat_template": "synthetic-template",
                "chat_template_caps": {"supports_tools": False},
                "build_info": "synthetic-build",
                "default_generation_settings": {"n_ctx": 2048},
                "verified_model_sha256": "fabricated",
            },
            "/slots": [{"id": 0, "is_processing": False}],
            "/v1/models": {"data": [{"id": self.model}]},
        }
        return bodies[path], None, 0


def test_qualification_validates_binding_slots_and_ignores_fake_digest(tmp_path):
    p = tmp_path / "model"
    p.write_bytes(b"exact")
    result = asyncio.run(qualify_llama_server(Fake(str(p)), artifact=p, alias="alias"))
    assert result.total_slots == 1
    assert "verified_model_sha256" not in result.model_dump()
    with pytest.raises(ValueError):
        asyncio.run(qualify_llama_server(Fake(str(p), 2), artifact=p, alias="alias"))
    with pytest.raises(ValueError):
        asyncio.run(
            qualify_llama_server(Fake(str(p), model="other"), artifact=p, alias="alias")
        )


def test_known_process_conflict_is_not_global_exclusivity():
    assert (
        classify_processes("123 /opt/bin/llama-server --model x", set())
        is ExternalConflictStatus.CONFLICT_DETECTED
    )
    assert (
        classify_processes("123 /opt/bin/llama-server --model x", {123})
        is ExternalConflictStatus.NO_KNOWN_CONFLICT
    )
    assert (
        classify_processes("123 /opt/homebrew/bin/llama serve --port 9931", set())
        is ExternalConflictStatus.CONFLICT_DETECTED
    )
    assert (
        classify_processes("111 /usr/bin/editor", set())
        is ExternalConflictStatus.NO_KNOWN_CONFLICT
    )


def test_retired_ollama_is_not_part_of_current_inference_conflict_vocabulary():
    assert (
        classify_processes("444 /opt/homebrew/bin/ollama runner --model historical", set())
        is ExternalConflictStatus.NO_KNOWN_CONFLICT
    )


def test_qualification_hashes_exact_template_and_checks_effective_context(tmp_path):
    p = tmp_path / "model"
    p.write_bytes(b"exact")
    probe = Fake(str(p))
    first = asyncio.run(qualify_llama_server(probe, artifact=p, alias="alias"))
    assert first.chat_template_sha256 == hashlib.sha256(b"synthetic-template").hexdigest()
    assert first.chat_template_caps == (("supports_tools", False),)
    assert first.build_info == "synthetic-build"
    assert first.context_tokens == 2048

    class Changed(Fake):
        async def request_json(self, method, path):
            body, evidence, elapsed = await super().request_json(method, path)
            if path == "/props":
                body["chat_template"] = "changed-template"
            return body, evidence, elapsed

    changed = asyncio.run(qualify_llama_server(Changed(str(p)), artifact=p, alias="alias"))
    assert changed.chat_template_sha256 != first.chat_template_sha256

    class Missing(Fake):
        async def request_json(self, method, path):
            body, evidence, elapsed = await super().request_json(method, path)
            if path == "/props":
                del body["chat_template"]
            return body, evidence, elapsed

    with pytest.raises(ValueError, match="template"):
        asyncio.run(qualify_llama_server(Missing(str(p)), artifact=p, alias="alias"))


def test_exact_llama_timing_projection_uses_only_direct_reported_fields():
    from llm_evaluation.runtime.llamacpp.qualification import project_exact_llama_timing
    from llm_evaluation.runtime.timing import RequestTiming

    generic = RequestTiming(total_request_wall_seconds=0.5)
    projected = project_exact_llama_timing(
        json.dumps({"timings": {"prompt_ms": 12.5, "predicted_ms": 20}}).encode(),
        generic,
    )
    assert generic.prompt_processing_seconds is None
    assert generic.generation_seconds is None
    assert projected.prompt_processing_seconds == 0.0125
    assert projected.generation_seconds == 0.02
    assert projected.queue_admission_wait_seconds is None
    assert projected.server_processing_wait_seconds is None
    assert projected.total_request_wall_seconds == 0.5


@pytest.mark.parametrize("bad", [-1, True, "2"])
def test_exact_llama_timing_projection_rejects_invalid_numbers(bad):
    from llm_evaluation.runtime.llamacpp.qualification import project_exact_llama_timing
    from llm_evaluation.runtime.timing import RequestTiming

    with pytest.raises(ValueError, match="timing"):
        project_exact_llama_timing(
            json.dumps({"timings": {"prompt_ms": bad, "predicted_ms": 1}}).encode(),
            RequestTiming(total_request_wall_seconds=0.5),
        )


def test_exact_llama_timing_projection_leaves_absent_fields_unknown():
    from llm_evaluation.runtime.llamacpp.qualification import project_exact_llama_timing
    from llm_evaluation.runtime.timing import RequestTiming

    original = RequestTiming(total_request_wall_seconds=0.5)
    assert project_exact_llama_timing(b"{}", original) == original


def test_ambiguous_process_line_stays_unverified():
    assert classify_processes("malformed process line", set()) is ExternalConflictStatus.UNVERIFIED


def test_owned_listener_probe_distinguishes_loopback_and_wildcard():
    import subprocess
    import sys
    from llm_evaluation.runtime.llamacpp.qualification import observe_owned_listener

    code = (
        "import socket,sys,time;"
        "s=socket.socket();s.bind((sys.argv[1],0));s.listen(1);"
        "print(s.getsockname()[1],flush=True);time.sleep(10)"
    )
    for host, expected in (("127.0.0.1", "loopback_verified"), ("0.0.0.0", "wildcard_detected")):
        child = subprocess.Popen(
            [sys.executable, "-c", code, host], stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout is not None
            port = int(child.stdout.readline().strip())
            assert observe_owned_listener(child.pid, port).value == expected
        finally:
            child.terminate()
            child.wait(timeout=3)


def test_llama_cli_loaded_model_is_known_conflict_even_with_zero_generation():
    assert (
        classify_processes(
            "321 /opt/homebrew/bin/llama-cli -m /models/qwen.gguf -n 0 -c 4096",
            set(),
        )
        is ExternalConflictStatus.CONFLICT_DETECTED
    )


def test_lm_studio_gui_without_server_is_not_automatically_a_conflict(monkeypatch):
    from llm_evaluation.runtime.llamacpp import qualification

    ps_text = "222 /Applications/LM Studio.app/Contents/MacOS/LM Studio\n"

    def fake_run(argv, **kwargs):
        if argv[:3] == ["ps", "-axo", "pid=,args="]:
            return subprocess.CompletedProcess(argv, 0, stdout=ps_text, stderr="")
        if argv == ["/fake/lms", "server", "status", "--json"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout='{"running":false}\n', stderr=""
            )
        raise AssertionError(f"unexpected command: {argv!r}")

    import subprocess
    import shutil

    monkeypatch.setattr(qualification.subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/lms" if name == "lms" else None)
    assert (
        qualification.probe_known_external_conflicts()
        is ExternalConflictStatus.NO_KNOWN_CONFLICT
    )


def test_lm_studio_owned_configured_listener_is_known_conflict(monkeypatch):
    from llm_evaluation.runtime.llamacpp import qualification
    import shutil
    import subprocess

    ps_text = "222 /Applications/LM Studio.app/Contents/MacOS/LM Studio\n"
    listener = (
        "p222\n"
        "cLM Studio\n"
        "f81\n"
        "n127.0.0.1:43210\n"
        "TST=LISTEN\n"
    )

    def fake_run(argv, **kwargs):
        if argv[:3] == ["ps", "-axo", "pid=,args="]:
            return subprocess.CompletedProcess(argv, 0, stdout=ps_text, stderr="")
        if argv == ["/fake/lms", "server", "status", "--json"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout='{"running":true,"port":43210}\n', stderr=""
            )
        if argv[0] == "lsof":
            assert "-p" in argv and argv[argv.index("-p") + 1] == "222"
            assert "-iTCP:43210" in argv
            return subprocess.CompletedProcess(argv, 0, stdout=listener, stderr="")
        raise AssertionError(f"unexpected command: {argv!r}")

    monkeypatch.setattr(qualification.subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/lms" if name == "lms" else None)
    assert (
        qualification.probe_known_external_conflicts()
        is ExternalConflictStatus.CONFLICT_DETECTED
    )


def test_malformed_listener_observation_is_unverified(monkeypatch):
    from llm_evaluation.runtime.llamacpp import qualification
    import shutil
    import subprocess

    ps_text = "222 /Applications/LM Studio.app/Contents/MacOS/LM Studio\n"

    def fake_run(argv, **kwargs):
        if argv[:3] == ["ps", "-axo", "pid=,args="]:
            return subprocess.CompletedProcess(argv, 0, stdout=ps_text, stderr="")
        if argv == ["/fake/lms", "server", "status", "--json"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout='{"running":true,"port":43210}\n', stderr=""
            )
        if argv[0] == "lsof":
            return subprocess.CompletedProcess(argv, 0, stdout="malformed\n", stderr="")
        raise AssertionError(f"unexpected command: {argv!r}")

    monkeypatch.setattr(qualification.subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/lms" if name == "lms" else None)
    assert (
        qualification.probe_known_external_conflicts()
        is ExternalConflictStatus.UNVERIFIED
    )


def test_llama_cli_without_observable_model_binding_is_unverified():
    assert (
        classify_processes("321 /opt/homebrew/bin/llama-cli --help", set())
        is ExternalConflictStatus.UNVERIFIED
    )


def test_qualification_requires_healthy_endpoint_and_unambiguous_model(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_bytes(b"exact")

    class Unhealthy(Fake):
        async def request_json(self, method, path):
            body, evidence, elapsed = await super().request_json(method, path)
            if path == "/health":
                body["status"] = "loading"
            return body, evidence, elapsed

    class DuplicateAlias(Fake):
        async def request_json(self, method, path):
            body, evidence, elapsed = await super().request_json(method, path)
            if path == "/v1/models":
                body["data"].append({"id": "alias"})
            return body, evidence, elapsed

    for invalid in (Unhealthy, DuplicateAlias):
        with pytest.raises(ValueError):
            asyncio.run(qualify_llama_server(invalid(str(p)), artifact=p, alias="alias"))



def test_formal_launch_pins_jinja_even_when_build_has_a_default(tmp_path):
    """A reasoning-relevant template engine setting must be explicit on the wire."""
    from llm_evaluation.runtime.llamacpp.lifecycle import (
        ManagedServerDeclaration, ManagedServerMode, ReasoningMode, build_launch,
    )

    executable = tmp_path / "fake-server"
    artifact = tmp_path / "model.gguf"
    executable.write_bytes(b"fixture executable")
    artifact.write_bytes(b"fixture weights")
    declaration = ManagedServerDeclaration(
        executable=str(executable), artifact=str(artifact),
        alias="fixture", mode=ManagedServerMode.FORMAL,
        reasoning_mode=ReasoningMode.OFF,
    )
    argv, _, _ = build_launch(declaration, 30099)
    assert argv.count("--jinja") == 1
    assert "--no-jinja" not in argv
