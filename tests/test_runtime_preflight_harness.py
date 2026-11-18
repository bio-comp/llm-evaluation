"""Deterministic end-to-end lifecycle controls using a tiny fake executable."""

import sys
import textwrap
import pytest
from llm_evaluation.local_runtime_preflight import run_preflight
from llm_evaluation.runtime.llamacpp.qualification import ExternalConflictStatus

FAKE = """
import os,sys,json,time
from http.server import BaseHTTPRequestHandler,HTTPServer
if '--version' in sys.argv:
 print('version: 0.7.0 (build 11669, commit 620d2225c)');print('synthetic-test-fixture');sys.exit(0)
args=sys.argv[1:]
def opt(name): return args[args.index(name)+1]
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def do_GET(self): self.respond()
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
  fmt=body.get('response_format')
  if fmt is not None:
   envelope=fmt.get('json_schema') if isinstance(fmt,dict) else None
   schema=envelope.get('schema') if isinstance(envelope,dict) else None
   if (not isinstance(fmt,dict) or fmt.get('type')!='json_schema'\n       or not isinstance(envelope,dict) or envelope.get('strict') is not True
       or envelope.get('name')!='output' or not isinstance(schema,dict)
       or schema.get('type')!='object' or schema.get('additionalProperties') is not False
       or schema.get('required')!=['value']
       or schema.get('properties',{}).get('value',{}).get('type')!='integer'):
    self.send_response(400);self.end_headers();return
  self.respond()
 def respond(self):
  key=os.environ.get('LLAMA_API_KEY')
  if self.path!='/health' and key and self.headers.get('Authorization')!='Bearer '+key:
   self.send_response(401);self.end_headers();return
  if self.path=='/': self.send_response(404);self.end_headers();return
  payload={'/health':{'status':'ok'},'/props':{'total_slots':1,'model_path':opt('--model'),\n      'chat_template':'synthetic-template','chat_template_caps':{'supports_tools':False},\n      'build_info':'synthetic-build',\n      'default_generation_settings':{'n_ctx':2048}},
    '/slots':[{'id':0,'is_processing':False}],'/v1/models':{'data':[{'id':opt('--alias')}]}}.get(self.path,
    {'model':opt('--alias'),'choices':[{'message':{'content':'{"value":1}','role':'assistant'},'finish_reason':'stop'}]})
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps(payload).encode())
HTTPServer(('127.0.0.1',int(opt('--port'))),Handler).serve_forever()
"""


@pytest.fixture
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "fake-llama-server"
    exe.write_text("#!" + sys.executable + "\n" + textwrap.dedent(FAKE))
    exe.chmod(0o700)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"synthetic-not-model-weights")
    monkeypatch.setenv("LLM_EVAL_LLAMA_PREFLIGHT", "1")
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_preflight.software_identity",
        lambda: ("a" * 40, "b" * 64),
    )
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_preflight.probe_known_external_conflicts",
        lambda: ExternalConflictStatus.NO_KNOWN_CONFLICT,
    )
    from llm_evaluation.runtime.llamacpp.qualification import hash_model_artifact

    return dict(
        executable=str(exe),
        artifact=str(model),
        expected_sha256=hash_model_artifact(model).sha256,
        output=tmp_path / "record.json",
        authorized=True,
        lease_root=tmp_path / "leases",
    )


@pytest.fixture
def pinned_current(fake):
    """Mock the independently pinned installed v1.0.0 executable condition."""
    from pathlib import Path
    from llm_evaluation.runtime.llamacpp.qualification import (
        ExpectedExecutable,
        hash_model_artifact,
    )

    exe = Path(fake["executable"])
    exe.write_text(
        exe.read_text().replace(
            "0.7.0 (build 11669, commit 620d2225c)",
            "1.0.0 (build 12359, commit b1b770d90)",
        )
    )
    pin = ExpectedExecutable(
        version="1.0.0",
        build=12359,
        commit="b1b770d90",
        sha256=hash_model_artifact(exe).sha256,
    )
    return fake | {"expected_executable": pin}


def test_current_runtime_exact_pin_passes_and_binds_backend(
    pinned_current, monkeypatch
):
    import llm_evaluation.local_runtime_preflight as preflight

    observed = []
    original = preflight.prepare_generation

    def capture(binding, request, structured=None):
        observed.append(binding.condition.backend_version)
        return original(binding, request, structured)

    monkeypatch.setattr(preflight, "prepare_generation", capture)
    record = run_preflight(**pinned_current)
    assert record.status == "passed"
    assert record.executable is not None
    assert record.executable.sha256 == pinned_current["expected_executable"].sha256
    assert observed
    assert set(observed) == {"1.0.0-12359-b1b770d90"}
    assert record.budget.server_launches == 3


def test_current_runtime_requires_explicit_pin_before_launch(pinned_current):
    args = {k: v for k, v in pinned_current.items() if k != "expected_executable"}
    result = run_preflight(**args)
    assert result.status == "failed" and result.failure_stage == "identity"
    assert result.budget.server_launches == result.budget.generation_submissions == 0


@pytest.mark.parametrize("change", ["sha256", "version", "commit", "build"])
def test_current_runtime_rejects_wrong_pin_without_launch(pinned_current, change):
    pin = pinned_current["expected_executable"]
    bad_value = {
        "sha256": "0" * 64,
        "version": "0.9.0",
        "commit": "deadbeef",
        "build": 12358,
    }[change]
    wrong = pin.model_copy(update={change: bad_value})
    result = run_preflight(**(pinned_current | {"expected_executable": wrong}))
    assert result.status == "failed" and result.failure_stage == "identity"
    assert result.budget.server_launches == result.budget.generation_submissions == 0


def test_current_runtime_rejects_binary_mutation_after_pin(pinned_current):
    from pathlib import Path

    exe = Path(pinned_current["executable"])
    exe.write_text(exe.read_text() + "\n# changed bytes\n")
    result = run_preflight(**pinned_current)
    assert result.status == "failed" and result.failure_stage == "identity"
    assert result.budget.server_launches == 0


def test_current_runtime_pin_cannot_bypass_validation(pinned_current):
    from pydantic import ValidationError

    pin = pinned_current["expected_executable"]
    with pytest.raises(ValidationError):
        run_preflight(
            **(
                pinned_current
                | {
                    "expected_executable": pin.model_copy(
                        update={"sha256": "not-a-sha"}
                    )
                }
            )
        )


def test_current_runtime_external_conflict_remains_hard_stop(
    pinned_current, monkeypatch
):
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_preflight.probe_known_external_conflicts",
        lambda: ExternalConflictStatus.CONFLICT_DETECTED,
    )
    result = run_preflight(**pinned_current)
    assert result.status == "incomplete"
    assert result.failure_stage == "external_conflict"
    assert result.budget.server_launches == result.budget.generation_submissions == 0


def test_current_runtime_structured_diagnostic_pin(pinned_current):
    from llm_evaluation.local_runtime_preflight import run_structured_diagnostic

    result = run_structured_diagnostic(**pinned_current, reasoning_mode="off")
    assert result.status == "passed"
    assert result.server_launches == 1
    assert result.generation_submissions == 1
    assert result.executable is not None
    assert result.executable.build == 12359


def test_full_synthetic_harness_and_readback(fake):
    record = run_preflight(**fake)
    assert record.status == "passed"
    assert record.budget.server_launches == 3
    assert record.budget.generation_submissions == 6
    assert record.gates.launcher_death_lease is True
    assert record.slot_poll_count >= 1
    assert len(record.request_timings) == 4
    assert "Bearer " not in fake["output"].read_text()
    with pytest.raises(FileExistsError):
        run_preflight(**fake)


def test_conflict_records_zero_launches(fake, monkeypatch):
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_preflight.probe_known_external_conflicts",
        lambda: ExternalConflictStatus.CONFLICT_DETECTED,
    )
    record = run_preflight(**fake)
    assert record.status == "incomplete" and record.budget.server_launches == 0
    assert record.failure_stage == "external_conflict"
    assert record.failure_code == "resource_busy"


def test_failed_start_preserves_attempt_count_and_failure(fake):
    path = __import__("pathlib").Path(fake["executable"])
    path.write_text(
        "#!"
        + sys.executable
        + '\nimport sys\nif "--version" in sys.argv:\n print("version: 0.7.0 (build 11669, commit 620d2225c)")\nelse: sys.exit(2)\n'
    )
    record = run_preflight(**fake)
    assert record.status == "failed" and record.budget.server_launches == 1
    assert record.budget.generation_submissions == 0


def test_authorization_and_identity_fail_before_launch(fake, monkeypatch):
    with pytest.raises(PermissionError):
        run_preflight(**(fake | {"authorized": False}))
    result = run_preflight(**(fake | {"expected_sha256": "0" * 64}))
    assert result.status == "failed" and result.failure_stage == "identity"
    assert result.budget.server_launches == 0


def test_failed_qualification_cleans_up_and_keeps_auth_counts(fake):
    path = __import__("pathlib").Path(fake["executable"])
    path.write_text(path.read_text().replace("'total_slots':1", "'total_slots':2"))
    record = run_preflight(**fake)
    assert record.status == "failed"
    assert record.budget.server_launches == 2
    assert record.budget.generation_submissions == 2
    from llm_evaluation.runtime.resources import acquire_resource_lease

    with acquire_resource_lease(fake["lease_root"], "llama-device:MTL0"):
        pass


def test_wrong_generation_model_cannot_pass_qualification(fake):
    path = __import__("pathlib").Path(fake["executable"])
    path.write_text(
        path.read_text().replace(
            "{'model':opt('--alias'),'choices'", "{'model':'wrong-model','choices'"
        )
    )
    record = run_preflight(**fake)
    assert record.status == "failed"
    assert record.budget.generation_submissions == 3


def test_failed_exclusion_aborts_and_counts_unexpected_launch(fake, monkeypatch):
    import llm_evaluation.local_runtime_preflight as preflight

    original = preflight.start_managed_server
    count = 0

    async def bypass_second_lease(declaration, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            declaration = declaration.model_copy(
                update={"lease_root": str(fake["lease_root"]) + "-broken"}
            )
        return await original(declaration, **kwargs)

    monkeypatch.setattr(preflight, "start_managed_server", bypass_second_lease)
    record = run_preflight(**fake)
    assert record.status == "failed"
    assert record.budget.server_launches == 2
    assert record.budget.generation_submissions == 0


def test_exhausted_identity_budget_cannot_start_live_checks(fake, monkeypatch):
    import llm_evaluation.local_runtime_preflight as preflight

    now = [0.0]
    monkeypatch.setattr(preflight.time, "monotonic", lambda: now[0])

    def slow_identity():
        now[0] = 241.0
        return "a" * 40, "b" * 64

    monkeypatch.setattr(preflight, "software_identity", slow_identity)

    def forbidden(*a, **k):
        raise AssertionError("must not start checks")

    monkeypatch.setattr(preflight, "_live_checks", forbidden)
    result = run_preflight(**fake)
    assert result.status == "failed" and result.failure_kind == "deadline"
    assert result.budget.server_launches == 0


def test_launcher_handshake_failure_cannot_start_model(fake, monkeypatch, tmp_path):
    import asyncio
    import time
    import llm_evaluation.local_runtime_preflight as preflight
    from llm_evaluation.runtime.llamacpp.lifecycle import (
        ManagedServerDeclaration,
        ManagedServerMode,
    )

    path = __import__("pathlib").Path(fake["executable"])
    marker = tmp_path / "model-started"
    path.write_text(
        path.read_text().replace(
            "args=sys.argv[1:]",
            "open(" + repr(str(marker)) + ",'w').write('started')\nargs=sys.argv[1:]",
        )
    )
    declaration = ManagedServerDeclaration(
        executable=fake["executable"],
        artifact=fake["artifact"],
        alias="preflight",
        mode=ManagedServerMode.SHARED,
        lease_root=str(fake["lease_root"]),
    )
    original_run = preflight.subprocess.run
    original_stamp = preflight._process_stamp
    owned = []

    def capture_driver(*args, **kwargs):
        result = original_run(*args, **kwargs)
        if len(args[0]) > 1 and args[0][1] == "-c":
            pid = int(result.stdout.strip())
            owned.append((pid, original_stamp(pid)))
        return result

    monkeypatch.setattr(preflight.subprocess, "run", capture_driver)

    def failure(pid):
        raise OSError("simulated ownership-query failure")

    monkeypatch.setattr(preflight, "_process_stamp", failure)
    try:
        with pytest.raises(OSError):
            asyncio.run(preflight._launcher_death_probe(declaration))
        deadline = time.monotonic() + 2
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not marker.exists()
    finally:
        monkeypatch.setattr(preflight, "_process_stamp", original_stamp)
        for pid, stamp in owned:
            if original_stamp(pid) == stamp:
                preflight._terminate_owned_orphan(pid, stamp)


def test_outer_preflight_deadline_publishes_failed_record(fake, monkeypatch):
    import llm_evaluation.local_runtime_preflight as preflight

    path = __import__("pathlib").Path(fake["executable"])
    path.write_text(
        path.read_text().replace(
            "  key=os.environ.get('LLAMA_API_KEY')",
            "  if self.path=='/props': time.sleep(1)\n  key=os.environ.get('LLAMA_API_KEY')",
        )
    )
    original = preflight._live_checks

    async def short(exe, artifact, root, sha, remaining, backend_version, progress):
        return await original(exe, artifact, root, sha, 0.4, backend_version, progress)

    monkeypatch.setattr(preflight, "_live_checks", short)
    record = run_preflight(**fake)
    assert record.status == "failed" and record.failure_kind == "deadline"
    assert fake["output"].exists()


@pytest.mark.parametrize("entrypoint", ["full", "structured"])
def test_r1_mutation_after_inspection_rejected(pinned_current, monkeypatch, entrypoint):
    """R1 known bad: source changes during conflict probe, before model launch."""
    from pathlib import Path
    import llm_evaluation.local_runtime_preflight as preflight

    source = Path(pinned_current["executable"])

    def mutate_after_identity():
        source.write_text(source.read_text() + "\n# changed after inspection\n")
        return ExternalConflictStatus.NO_KNOWN_CONFLICT

    monkeypatch.setattr(
        preflight, "probe_known_external_conflicts", mutate_after_identity
    )
    if entrypoint == "full":
        record = run_preflight(**pinned_current)
        assert record.status == "failed"
        assert record.failure_kind == "identity"
        assert record.budget.server_launches == 0
    else:
        record = preflight.run_structured_diagnostic(
            **pinned_current, reasoning_mode="off"
        )
        assert record.status == "failed"
        assert record.failure_stage == "identity"
        assert record.server_launches == 0


@pytest.mark.parametrize("entrypoint", ["full", "structured"])
def test_r1_model_mutation_after_inspection_rejected(fake, monkeypatch, entrypoint):
    """The GGUF supplied at launch must also match the measured artifact."""
    from pathlib import Path
    import llm_evaluation.local_runtime_preflight as preflight

    source = Path(fake["artifact"])

    def mutate_after_identity():
        source.write_bytes(source.read_bytes() + b"changed after initial hash")
        return ExternalConflictStatus.NO_KNOWN_CONFLICT

    monkeypatch.setattr(
        preflight, "probe_known_external_conflicts", mutate_after_identity
    )
    if entrypoint == "full":
        record = run_preflight(**fake)
        assert record.status == "failed"
        assert record.failure_kind == "identity"
        assert record.budget.server_launches == 0
    else:
        record = preflight.run_structured_diagnostic(**fake, reasoning_mode="off")
        assert record.status == "failed"
        assert record.failure_stage == "identity"
        assert record.server_launches == 0


@pytest.mark.parametrize(
    "change",
    [
        {"sha256": b"a" * 64},
        {"bogus": "unexpected"},
        {"sha256": "a" * 63},
        {"commit": "not-a-hex-commit"},
        {"version": ""},
        {"build": 0},
        {"build": True},
    ],
)
def test_r1_unvalidated_copied_pin_rejected(pinned_current, change):
    from pydantic import ValidationError

    pin = pinned_current["expected_executable"].model_copy(update=change)
    with pytest.raises(ValidationError):
        run_preflight(**(pinned_current | {"expected_executable": pin}))


def test_r1_constructed_invalid_pin_rejected(pinned_current):
    from pydantic import ValidationError
    from llm_evaluation.runtime.llamacpp.qualification import ExpectedExecutable

    pin = pinned_current["expected_executable"]
    constructed = ExpectedExecutable.model_construct(
        version=pin.version, build=-1, commit=pin.commit, sha256=pin.sha256
    )
    with pytest.raises(ValidationError):
        run_preflight(**(pinned_current | {"expected_executable": constructed}))


@pytest.mark.parametrize("entrypoint", ["full", "structured"])
def test_r1_staged_sources_survive_later_source_mutation(
    pinned_current, monkeypatch, entrypoint
):
    """Once snapshotted, source-path changes cannot change launched bytes."""
    from pathlib import Path
    import llm_evaluation.local_runtime_preflight as preflight
    from llm_evaluation.runtime.llamacpp.qualification import hash_model_artifact
    from llm_evaluation.runtime.resources import acquire_resource_lease

    exe = Path(pinned_current["executable"])
    model = Path(pinned_current["artifact"])
    initial_exe_hash = hash_model_artifact(exe).sha256
    initial_model_hash = hash_model_artifact(model).sha256
    launch_paths = []
    original = preflight.start_managed_server

    async def mutate_original_on_first_launch(declaration, **kwargs):
        launch_paths.append((declaration.executable, declaration.artifact))
        if len(launch_paths) == 1:
            exe.write_text(exe.read_text() + "\n# later executable source mutation\n")
            model.write_bytes(model.read_bytes() + b"later model source mutation")
        assert hash_model_artifact(declaration.executable).sha256 == initial_exe_hash
        assert hash_model_artifact(declaration.artifact).sha256 == initial_model_hash
        return await original(declaration, **kwargs)

    monkeypatch.setattr(
        preflight, "start_managed_server", mutate_original_on_first_launch
    )
    if entrypoint == "full":
        result = run_preflight(**pinned_current)
        assert result.status == "passed" and result.budget.server_launches == 3
    else:
        result = preflight.run_structured_diagnostic(
            **pinned_current, reasoning_mode="off"
        )
        assert result.status == "passed" and result.server_launches == 1
    assert launch_paths
    assert all(
        path != str(exe) and model_path != str(model)
        for path, model_path in launch_paths
    )
    assert hash_model_artifact(exe).sha256 != initial_exe_hash
    assert hash_model_artifact(model).sha256 != initial_model_hash
    # Temporary snapshots must be gone after lifecycle cleanup.
    assert all(
        not Path(path).exists() and not Path(model_path).exists()
        for path, model_path in launch_paths
    )
    with acquire_resource_lease(pinned_current["lease_root"], "llama-device:MTL0"):
        pass
