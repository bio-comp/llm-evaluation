"""Bounded opt-in infrastructure caller. No E-002 scoring or runtime persistence."""

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any, Literal, Callable
from .runtime.timing import RequestTiming
from pydantic import BaseModel, ConfigDict
from .artifact_io import publish_immutable
from .records import software_identity
from .runtime.contracts import (
    BackendConfig,
    CapabilityProfile,
    CapabilityStatus,
    GenerationRequest,
    HttpTimeouts,
    ModelCondition,
    OperationalEndpoint,
    RuntimeBinding,
    ScientificEndpointIdentity,
)
from .runtime.failures import RuntimeFailure, FailureKind
from .runtime.preparation import prepare_generation
from .runtime.resources import acquire_resource_lease, ResourceBusyError
from .runtime.structured import (
    build_structured_output_contract,
    parse_structured_output,
)
from .runtime.transport.base import BearerAuthorization
from .runtime.transport.openai_compatible import AsyncOpenAICompatibleTransport
from .runtime.llamacpp.lifecycle import (
    ManagedServerDeclaration,
    ManagedServerMode,
    ReasoningMode,
    FORMAL_CACHE_PROMPT,
    build_launch,
    start_managed_server,
    stop_managed_server,
)
from .runtime.llamacpp.qualification import (
    inspect_llama_executable,
    hash_model_artifact,
    probe_known_external_conflicts,
    ExternalConflictStatus,
    qualify_llama_server,
    project_exact_llama_timing,
    observe_owned_listener,
    ListenerStatus,
)
from .local_runtime_preflight_records import (
    PreflightRecord,
    PreflightBudget,
    GateResults,
    summarize_generation,
    StructuredDiagnosticRecord,
    SlotSample,
)


class ProbeOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: int


def _process_stamp(pid: int) -> str:
    p = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="],
        capture_output=True,
        text=True,
        timeout=2,
    )
    return p.stdout.strip() if p.returncode == 0 else ""


def _terminate_owned_orphan(pid: int, stamp: str) -> None:
    # PID came from our direct driver, not lease metadata. Match its birth stamp.
    if not stamp or _process_stamp(pid) != stamp:
        raise RuntimeError("owned server identity changed")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        if not state or state.startswith("Z"):
            return
        time.sleep(0.02)
    if _process_stamp(pid) != stamp:
        raise RuntimeError("owned server identity changed")
    os.kill(pid, signal.SIGKILL)


async def _launcher_death_probe(
    declaration: ManagedServerDeclaration,
    before_spawn: Callable[[], None] | None = None,
) -> bool:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args, env, _ = build_launch(declaration, port)
    control, child_control = socket.socketpair()
    # The detached child cannot acquire a lease or exec the model until its
    # observer has its PID/birth stamp and explicitly acknowledges ownership.
    gate_code = """import os,sys,json,select
fd=int(sys.argv[1]); command=json.loads(sys.argv[2])
ready=select.select([fd],[],[],5)[0]
if not ready or os.read(fd,1)!=b'G': sys.exit(75)
os.close(fd)
os.execve(command[0],command,dict(os.environ))
"""
    driver = """import subprocess,sys,json
fd=int(sys.argv[1]); command=json.loads(sys.argv[2]); gate=sys.argv[3]
p=subprocess.Popen([sys.executable,'-c',gate,str(fd),json.dumps(command)],pass_fds=(fd,),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print(p.pid,flush=True)
"""
    command = [
        sys.executable,
        "-m",
        "llm_evaluation.runtime.llamacpp.lease_exec",
        declaration.lease_root,
        declaration.resource,
        *args,
    ]
    pid = None
    stamp = ""
    acknowledged = False
    busy = False
    try:
        if before_spawn is not None:
            before_spawn()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                driver,
                str(child_control.fileno()),
                json.dumps(command),
                gate_code,
            ],
            env=env,
            pass_fds=(child_control.fileno(),),
            capture_output=True,
            text=True,
            timeout=5,
        )
        child_control.close()
        if result.returncode != 0:
            raise RuntimeError("launcher driver failed")
        pid = int(result.stdout.strip())
        stamp = _process_stamp(pid)
        if not stamp:
            raise RuntimeError("ownership handshake failed")
        control.sendall(b"G")
        acknowledged = True
        probe = AsyncOpenAICompatibleTransport(
            OperationalEndpoint(base_url=f"http://127.0.0.1:{port}"),
            timeouts=HttpTimeouts(
                connect=0.3, read=0.5, write=0.5, pool=0.5, total=1.0
            ),
        )
        async with asyncio.timeout(45):
            while True:
                try:
                    await probe.request_json("GET", "/health")
                    break
                except RuntimeFailure:
                    if _process_stamp(pid) != stamp:
                        raise RuntimeError("orphan server exited")
                await asyncio.sleep(0.05)
        try:
            with acquire_resource_lease(declaration.lease_root, declaration.resource):
                pass
        except ResourceBusyError:
            busy = True
    finally:
        control.close()
        child_control.close()
        if acknowledged and pid is not None:
            _terminate_owned_orphan(pid, stamp)
        # Without acknowledgement the child observes EOF or its five-second
        # handshake timeout and exits before acquiring any model resource.
    deadline = time.monotonic() + 5
    while True:
        try:
            with acquire_resource_lease(declaration.lease_root, declaration.resource):
                return busy
        except ResourceBusyError:
            if time.monotonic() > deadline:
                raise RuntimeError("server lease remains held")
            await asyncio.sleep(0.02)


async def _live_checks(
    executable: str,
    artifact: str,
    root: str,
    artifact_sha: str,
    remaining: float,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if remaining <= 0:
        raise TimeoutError("preflight deadline exhausted")
    live_deadline = time.monotonic() + remaining
    gates: dict[str, bool | None] = {}
    counts = {"server_launches": 0, "generation_submissions": 0}
    timings: list[RequestTiming] = []
    summaries: list[Any] = []
    samples: list[SlotSample] = []
    progress_handles: list[Any] = []
    poll_count = 0
    observation_seconds = 0.0
    if progress is None:
        progress = {}
    progress.update(
        gates=gates,
        counts=counts,
        request_timings=timings,
        generation_summaries=summaries,
        slot_samples=samples,
        owned_handles=progress_handles,
        failure_stage="shared_exclusion",
        failure_code="unknown_safe_failure",
        slot_poll_count=0,
        slot_observation_seconds=0.0,
    )
    contract = build_structured_output_contract(ProbeOutput)
    progress.update(
        structured_domain_sha256=contract.domain_schema_sha256,
        structured_effective_sha256=contract.effective_schema_sha256,
    )
    shared = ManagedServerDeclaration(
        executable=executable,
        artifact=artifact,
        alias="preflight",
        mode=ManagedServerMode.SHARED,
        reasoning_mode=ReasoningMode.AUTO,
        lease_root=root,
    )
    formal = shared.model_copy(update={"mode": ManagedServerMode.FORMAL})

    def before_spawn() -> None:
        if time.monotonic() >= live_deadline:
            raise TimeoutError("preflight deadline exhausted before launch")
        if counts["server_launches"] >= 3:
            raise RuntimeError("preflight launch budget exhausted")
        counts["server_launches"] += 1

    def before_generation(n: int = 1) -> None:
        if time.monotonic() >= live_deadline:
            raise TimeoutError("preflight deadline exhausted before generation")
        if counts["generation_submissions"] + n > 6:
            raise RuntimeError("preflight generation budget exhausted")
        counts["generation_submissions"] += n

    async def blocked(d: ManagedServerDeclaration) -> bool:
        try:
            unexpected = await start_managed_server(d, before_spawn=before_spawn)
        except ResourceBusyError:
            return True
        progress_handles.append(unexpected.handle)
        stop_managed_server(unexpected.handle)
        return False

    async with asyncio.timeout(remaining):
        launch = await start_managed_server(shared, before_spawn=before_spawn)
        progress_handles.append(launch.handle)
        try:
            gates["shared_blocks_formal"] = await blocked(formal)
            if not gates["shared_blocks_formal"]:
                raise RuntimeError("shared/formal exclusion failed")
        finally:
            stop_managed_server(launch.handle)
        progress["failure_stage"] = "formal_launch"
        launch = await start_managed_server(formal, before_spawn=before_spawn)
        progress_handles.append(launch.handle)
        try:
            progress["failure_stage"] = "formal_exclusion"
            gates["formal_blocks_shared"] = await blocked(shared)
            if not gates["formal_blocks_shared"]:
                raise RuntimeError("formal/shared exclusion failed")
            endpoint = launch.handle.endpoint
            profile = CapabilityProfile(
                statuses={
                    k: CapabilityStatus.SUPPORTED
                    for k in ("chat_completions", "max_output_tokens", "json_schema")
                }
            )
            condition = ModelCondition(
                condition_id="infrastructure-preflight",
                weights_id="exact-local-artifact",
                artifact_sha256=artifact_sha,
                provider_id="managed-llamacpp",
                backend_config_id="preflight",
                backend_name="llamacpp",
                backend_version="0.7.0-11669-620d2225c",
                api_mode="chat_completions",
                endpoint_model_id="preflight",
                prompt_template_id="runtime-default-recorded-build",
                seed_behavior="unrequested",
                runtime_environment={
                    "parallel": 1,
                    "ctx_size": 2048,
                    "output_limit": 128,
                    "jinja": True,
                },
            )
            binding = RuntimeBinding(
                backend=BackendConfig(
                    provider_id="managed-llamacpp",
                    model_id="preflight",
                    api_mode="chat_completions",
                    capability_profile=profile,
                    config_id="preflight",
                ),
                condition=condition,
                scientific_endpoint=ScientificEndpointIdentity(
                    value="managed-infrastructure-profile"
                ),
                timeouts=HttpTimeouts(
                    connect=2.0, read=30.0, write=2.0, pool=2.0, total=30.0
                ),
            )
            request = GenerationRequest(
                condition_id=condition.condition_id,
                messages=(("user", 'Return exactly {"value":1} as JSON.'),),
                max_output_tokens=128,
            )
            prepared = prepare_generation(binding, request)

            async def auth_rejected(authorization: BearerAuthorization | None) -> bool:
                client = AsyncOpenAICompatibleTransport(
                    endpoint, authorization=authorization, timeouts=binding.timeouts
                )
                before_generation()
                try:
                    await client.request_json(
                        "POST", "/v1" + prepared.path, prepared.wire_bytes
                    )
                except RuntimeFailure as e:
                    return e.kind is FailureKind.HTTP_STATUS and e.status_code in (
                        401,
                        403,
                    )
                return False

            progress["failure_stage"] = "authentication"
            gates["no_key_rejected"] = await auth_rejected(None)
            gates["wrong_key_rejected"] = await auth_rejected(
                BearerAuthorization("deliberately-wrong-key")
            )
            probe = AsyncOpenAICompatibleTransport(
                endpoint, authorization=launch.authorization, timeouts=binding.timeouts
            )
            progress["failure_stage"] = "qualification"
            qualification = await qualify_llama_server(probe, artifact=artifact, alias="preflight")
            progress["chat_template_sha256"] = qualification.chat_template_sha256
            progress["chat_template_caps"] = qualification.chat_template_caps
            progress["build_info_sha256"] = hashlib.sha256(qualification.build_info.encode()).hexdigest()
            progress["reasoning_profile"] = formal.reasoning_mode.value if formal.reasoning_mode else None
            condition = condition.model_copy(update={
                "prompt_template_id": "llamacpp-chat-template-sha256:" + qualification.chat_template_sha256,
                "runtime_environment": {
                    "parallel": 1, "ctx_size": qualification.context_tokens,
                    "output_limit": 128, "reasoning": progress["reasoning_profile"],
                    "jinja": True,
                    "cache_prompt": FORMAL_CACHE_PROMPT,
                },
            })
            binding = binding.model_copy(update={"condition": condition})
            prepared = prepare_generation(binding, request)
            progress["qualified_condition_sha256"] = prepared.condition_sha256
            gates["qualified_model_slot"] = True
            port = urlsplit(endpoint.base_url).port
            assert port is not None
            gates["listener_loopback"] = (
                observe_owned_listener(launch.handle.process.pid, port)
                is ListenerStatus.LOOPBACK_VERIFIED
            )
            if not gates["listener_loopback"]:
                raise ValueError("formal owned listener not verified as loopback")
            argv = launch.handle.process.args
            gates["agent_tools_disabled"] = (
                isinstance(argv, list)
                and all(flag in argv for flag in (
                    "--no-agent", "--no-ui-mcp-proxy", "--no-webui"
                ))
                and "--tools" not in argv
            )
            if not gates["agent_tools_disabled"]:
                raise ValueError("formal agent/tool launch controls not verified")
            try:
                await probe.request_json("GET", "/")
            except RuntimeFailure as e:
                gates["webui_disabled"] = (
                    e.kind is FailureKind.HTTP_STATUS and e.status_code == 404
                )
            else:
                gates["webui_disabled"] = False
            client = AsyncOpenAICompatibleTransport(
                OperationalEndpoint(base_url=endpoint.base_url + "/v1"),
                authorization=launch.authorization,
                timeouts=binding.timeouts,
            )
            progress["failure_stage"] = "tiny_generation"
            before_generation()
            tiny = await client.generate(prepared)
            if tiny.observation.reported_model_matches_condition is not True:
                gates["tiny_generation"] = False
                raise ValueError("generation model binding mismatch")
            summaries.append(summarize_generation(tiny.observation, tiny.evidence.body))
            timings.append(project_exact_llama_timing(tiny.evidence.body, tiny.timing))
            gates["tiny_generation"] = True
            progress["failure_stage"] = "structured_generation"
            gates["structured_generation"] = False
            before_generation()
            try:
                structured = await client.generate(
                    prepare_generation(binding, request, contract)
                )
            except RuntimeFailure as exc:
                progress["failure_code"] = (
                    exc.kind.value if exc.kind.value in {
                        "http_status", "invalid_json", "invalid_response"
                    } else "unknown_safe_failure"
                )
                raise
            summaries.append(summarize_generation(structured.observation, structured.evidence.body))
            if structured.observation.reported_model_matches_condition is not True:
                gates["structured_generation"] = False
                raise ValueError("structured generation model binding mismatch")
            if not structured.observation.raw_output:
                progress["failure_code"] = (
                    "output_exhausted"
                    if structured.observation.termination_reason == "length"
                    else "empty_visible_output"
                )
                raise ValueError("empty structured visible output")
            try:
                parse_structured_output(
                    contract, structured.observation.raw_output.encode()
                )
            except ValueError as exc:
                from pydantic import ValidationError

                progress["failure_code"] = (
                    "structured_validation"
                    if isinstance(exc, ValidationError)
                    else "structured_parse"
                )
                raise
            timings.append(
                project_exact_llama_timing(structured.evidence.body, structured.timing)
            )
            gates["structured_generation"] = True
            progress["failure_stage"] = "concurrency"
            before_generation(2)
            tasks = [asyncio.create_task(client.generate(prepared)) for _ in range(2)]
            observed = time.monotonic()
            try:
                while True:
                    slots, _, _ = await probe.request_json("GET", "/slots")
                    poll_count += 1
                    progress["slot_poll_count"] = poll_count
                    if (
                        not isinstance(slots, list)
                        or len(slots) != 1
                        or not isinstance(slots[0], dict)
                    ):
                        raise ValueError("slot count changed")
                    if any(type(s.get("is_processing")) is not bool for s in slots):
                        raise ValueError("invalid processing slot observation")
                    processing = sum(s["is_processing"] for s in slots)
                    if len(samples) < 4000:
                        samples.append(SlotSample(
                            monotonic_offset_seconds=time.monotonic() - observed,
                            slot_count=len(slots), processing_count=processing,
                        ))
                    if processing > 1:
                        raise ValueError("multiple processing slots")
                    if all(t.done() for t in tasks):
                        break
                    await asyncio.sleep(0.02)
                for execution in await asyncio.gather(*tasks):
                    if (
                        execution.observation.reported_model_matches_condition
                        is not True
                    ):
                        gates["concurrent_requests"] = False
                        raise ValueError("concurrent generation model binding mismatch")
                    summaries.append(summarize_generation(execution.observation, execution.evidence.body))
                    timings.append(
                        project_exact_llama_timing(
                            execution.evidence.body, execution.timing
                        )
                    )
                gates["concurrent_requests"] = True
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                observation_seconds = time.monotonic() - observed
                progress["slot_observation_seconds"] = observation_seconds
        finally:
            stop_managed_server(launch.handle)
        progress["failure_stage"] = "launcher_death"
        gates["launcher_death_lease"] = await _launcher_death_probe(
            shared, before_spawn
        )
        progress["failure_stage"] = None
        progress["failure_code"] = None
    return _snapshot(progress)


def _snapshot(progress: dict[str, Any]) -> dict[str, Any]:
    return dict(
        gates=GateResults(**progress.get("gates", {})),
        budget=PreflightBudget(**progress.get("counts", {})),
        request_timings=tuple(progress.get("request_timings", [])),
        generation_summaries=tuple(progress.get("generation_summaries", [])),
        slot_samples=tuple(progress.get("slot_samples", [])),
        slot_poll_count=progress.get("slot_poll_count", 0),
        slot_observation_seconds=progress.get("slot_observation_seconds", 0.0),
        structured_domain_sha256=progress.get("structured_domain_sha256"),
        structured_effective_sha256=progress.get("structured_effective_sha256"),
        chat_template_sha256=progress.get("chat_template_sha256"),
        chat_template_caps=progress.get("chat_template_caps"),
        build_info_sha256=progress.get("build_info_sha256"),
        qualified_condition_sha256=progress.get("qualified_condition_sha256"),
        reasoning_profile=progress.get("reasoning_profile"),
    )


def run_preflight(
    *,
    executable: str,
    artifact: str,
    expected_sha256: str,
    output: Path,
    authorized: bool = False,
    lease_root: Path | None = None,
) -> PreflightRecord:
    if not authorized or os.environ.get("LLM_EVAL_LLAMA_PREFLIGHT") != "1":
        raise PermissionError(
            "explicit bounded local-preflight authorization and opt-in required"
        )
    started = time.monotonic()
    deadline = started + 240
    commit, package = software_identity()
    exe = None
    model = None
    conflict = ExternalConflictStatus.UNVERIFIED
    progress: dict[str, Any] = {}
    status: Literal["passed", "failed", "incomplete"] = "incomplete"
    failure: Literal[
        "none", "conflict", "identity", "runtime", "deadline", "qualification"
    ] = "conflict"

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("preflight budget exhausted")
        return value

    try:
        remaining()
        exe = inspect_llama_executable(executable, deadline=deadline)
        model = hash_model_artifact(artifact, deadline=deadline)
        remaining()
        if (exe.version, exe.build, exe.commit) != (
            "0.7.0",
            11669,
            "620d2225c",
        ) or model.sha256 != expected_sha256:
            raise ValueError(
                "exact executable/artifact identity mismatch; no server launched"
            )
        conflict = probe_known_external_conflicts()
        remaining()
    except TimeoutError:
        status = "failed"
        failure = "deadline"
        progress["failure_stage"] = "deadline"
        progress["failure_code"] = "deadline"
    except ValueError as exc:
        if str(exc) != "exact executable/artifact identity mismatch; no server launched":
            raise
        status = "failed"
        failure = "identity"
        progress["failure_stage"] = "identity"
        progress["failure_code"] = "unknown_safe_failure"
    if status != "failed" and conflict is ExternalConflictStatus.NO_KNOWN_CONFLICT:
        assert exe is not None and model is not None
        root = lease_root or Path.home() / ".local/state/llm-runtime/leases"
        try:
            asyncio.run(
                _live_checks(
                    exe.path, model.path, str(root), model.sha256, remaining(), progress
                )
            )
            status = "passed"
            failure = "none"
        except TimeoutError:
            status = "failed"
            failure = "deadline"
            progress["failure_stage"] = "deadline"
            progress["failure_code"] = "deadline"
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError):
            status = "failed"
            failure = "runtime"
        finally:
            try:
                if any(h.process.poll() is None for h in progress.get("owned_handles", [])):
                    raise RuntimeError("owned server still alive")
                with acquire_resource_lease(root, "llama-device:MTL0"):
                    pass
            except (RuntimeError, OSError, ResourceBusyError):
                progress["cleanup_status"] = "failed"
                progress.setdefault("gates", {})["cleanup"] = False
                if status == "passed":
                    status = "failed"
                    failure = "runtime"
                    progress["failure_stage"] = "cleanup"
                    progress["failure_code"] = "unknown_safe_failure"
            else:
                progress["cleanup_status"] = "verified"
                progress.setdefault("gates", {})["cleanup"] = True
    elif status != "failed":
        progress["failure_stage"] = "external_conflict"
        progress["failure_code"] = "resource_busy"
    data = _snapshot(progress)
    if status == "passed" and not all(v is True for v in data["gates"].model_dump().values()):
        status = "failed"
        failure = "qualification"
    record = PreflightRecord(
        evidence_class="infrastructure_qualification_not_e002",
        software_commit=commit,
        package_source_sha256=package,
        executable=exe,
        artifact=model,
        external_conflict_status=conflict,
        status=status,
        failure_kind=failure,
        failure_stage=progress.get("failure_stage"),
        failure_code=progress.get("failure_code"),
        cleanup_status=progress.get("cleanup_status", "unverified"),
        wall_seconds=time.monotonic() - started,
        **data,
    )
    raw = record.model_dump_json().encode()
    publish_immutable(output, raw)
    verified = PreflightRecord.model_validate_json(output.read_bytes())
    if verified != record:
        raise RuntimeError("preflight publication verification failed")
    return verified


def run_structured_diagnostic(
    *,
    executable: str,
    artifact: str,
    expected_sha256: str,
    output: Path,
    reasoning_mode: Literal["auto", "off"],
    authorized: bool = False,
    lease_root: Path | None = None,
) -> StructuredDiagnosticRecord:
    """One immutable, non-evaluation technical probe: at most one launch/call."""

    if not authorized or os.environ.get("LLM_EVAL_LLAMA_PREFLIGHT") != "1":
        raise PermissionError("explicit bounded local diagnostic authorization required")
    if reasoning_mode not in ("auto", "off"):
        raise ValueError("explicit supported reasoning mode required")
    started = time.monotonic()
    deadline = started + 240
    commit, package = software_identity()
    root = lease_root or Path.home() / ".local/state/llm-runtime/leases"
    state: dict[str, Any] = {
        "executable": None, "artifact": None,
        "status": "incomplete", "failure_stage": "identity",
        "failure_code": "unknown_safe_failure", "cleanup_status": "unverified",
        "server_launches": 0, "generation_submissions": 0,
        "reasoning_profile": reasoning_mode, "structured_generation": None,
        "summary": None, "chat_template_sha256": None,
        "qualified_condition_sha256": None,
    }
    contract = build_structured_output_contract(ProbeOutput)
    state["structured_domain_sha256"] = contract.domain_schema_sha256
    state["structured_effective_sha256"] = contract.effective_schema_sha256
    owned: list[Any] = []

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("diagnostic budget exhausted")
        return left

    async def one_generation() -> None:
        declaration = ManagedServerDeclaration(
            executable=executable, artifact=artifact, alias="preflight",
            mode=ManagedServerMode.FORMAL, reasoning_mode=ReasoningMode(reasoning_mode),
            lease_root=str(root),
        )

        def admitted_launch() -> None:
            remaining()
            if state["server_launches"] != 0:
                raise RuntimeError("diagnostic launch budget exhausted")
            state["server_launches"] = 1

        state["failure_stage"] = "launch"
        launch = await start_managed_server(declaration, before_spawn=admitted_launch)
        owned.append(launch.handle)
        try:
            endpoint = launch.handle.endpoint
            probe = AsyncOpenAICompatibleTransport(
                endpoint, authorization=launch.authorization,
                timeouts=HttpTimeouts(connect=2.0, read=30.0, write=2.0, pool=2.0, total=30.0),
            )
            state["failure_stage"] = "qualification"
            qualification = await qualify_llama_server(
                probe, artifact=artifact, alias="preflight"
            )
            state["chat_template_sha256"] = qualification.chat_template_sha256
            profile = CapabilityProfile(statuses={
                key: CapabilityStatus.SUPPORTED for key in
                ("chat_completions", "max_output_tokens", "json_schema")
            })
            condition = ModelCondition(
                condition_id="infrastructure-preflight",
                weights_id="exact-local-artifact",
                artifact_sha256=state["artifact"].sha256,
                provider_id="managed-llamacpp",
                backend_config_id="preflight",
                backend_name="llamacpp",
                backend_version="0.7.0-11669-620d2225c",
                api_mode="chat_completions",
                endpoint_model_id="preflight",
                prompt_template_id="llamacpp-chat-template-sha256:" + qualification.chat_template_sha256,
                seed_behavior="unrequested",
                runtime_environment={
                    "parallel": 1, "ctx_size": qualification.context_tokens,
                    "output_limit": 128, "reasoning": reasoning_mode,
                    "jinja": True,
                    "cache_prompt": FORMAL_CACHE_PROMPT,
                },
            )
            binding = RuntimeBinding(
                backend=BackendConfig(
                    provider_id="managed-llamacpp", model_id="preflight",
                    api_mode="chat_completions", capability_profile=profile,
                    config_id="preflight",
                ),
                condition=condition,
                scientific_endpoint=ScientificEndpointIdentity(
                    value="managed-infrastructure-profile"
                ),
                timeouts=HttpTimeouts(connect=2.0, read=30.0, write=2.0, pool=2.0, total=30.0),
            )
            request = GenerationRequest(
                condition_id=condition.condition_id,
                messages=(("user", 'Return exactly {"value":1} as JSON.'),),
                max_output_tokens=128,
            )
            prepared = prepare_generation(binding, request, contract)
            state["qualified_condition_sha256"] = prepared.condition_sha256
            client = AsyncOpenAICompatibleTransport(
                OperationalEndpoint(base_url=endpoint.base_url + "/v1"),
                authorization=launch.authorization, timeouts=binding.timeouts,
            )
            state["failure_stage"] = "structured_generation"
            state["structured_generation"] = False
            remaining()
            state["generation_submissions"] = 1
            execution = await client.generate(prepared)
            execution = execution.model_copy(update={
                "timing": project_exact_llama_timing(
                    execution.evidence.body, execution.timing
                )
            })
            state["summary"] = summarize_generation(
                execution.observation, execution.evidence.body
            )
            if reasoning_mode == "off" and state["summary"].reasoning_bytes:
                # Keep the published schema-v1 failure vocabulary stable. The
                # retained summary distinguishes observed reasoning from other
                # invalid responses without inventing a new v1 enum member.
                state["failure_code"] = "invalid_response"
                raise ValueError("reasoning observed under disabled profile")
            if state["summary"].termination_reason != "stop":
                state["failure_code"] = (
                    "output_exhausted"
                    if state["summary"].termination_reason == "length"
                    else "invalid_response"
                )
                raise ValueError("structured output has unreliable termination")
            if not state["summary"].within_reported_budget(
                context_tokens=2048, output_tokens=128
            ):
                state["failure_code"] = "invalid_response"
                raise ValueError("reported token usage exceeds qualified budget")
            observation = execution.observation
            if observation.reported_model_matches_condition is not True:
                state["failure_code"] = "model_binding"
                raise ValueError("diagnostic model binding mismatch")
            if not observation.raw_output:
                state["failure_code"] = (
                    "output_exhausted" if observation.termination_reason == "length"
                    else "empty_visible_output"
                )
                raise ValueError("empty visible structured output")
            try:
                parse_structured_output(contract, observation.raw_output.encode())
            except ValueError as exc:
                from pydantic import ValidationError

                state["failure_code"] = (
                    "structured_validation" if isinstance(exc, ValidationError)
                    else "structured_parse"
                )
                raise
            state["structured_generation"] = True
            state["failure_stage"] = None
            state["failure_code"] = None
        finally:
            stop_managed_server(launch.handle)

    try:
        remaining()
        state["executable"] = inspect_llama_executable(executable, deadline=deadline)
        state["artifact"] = hash_model_artifact(artifact, deadline=deadline)
        if (
            state["executable"].version, state["executable"].build,
            state["executable"].commit, state["artifact"].sha256
        ) != ("0.7.0", 11669, "620d2225c", expected_sha256):
            state["failure_stage"] = "identity"
            raise ValueError("diagnostic identity mismatch")
        remaining()
        if probe_known_external_conflicts() is not ExternalConflictStatus.NO_KNOWN_CONFLICT:
            state["status"] = "incomplete"
            state["failure_stage"] = "external_conflict"
        else:
            asyncio.run(asyncio.wait_for(one_generation(), timeout=remaining()))
            state["status"] = "passed"
    except TimeoutError:
        state["status"] = "failed"
        state["failure_stage"] = "deadline"
        state["failure_code"] = "deadline"
        state["remote_completion_unknown"] = state["generation_submissions"] > 0
    except RuntimeFailure as exc:
        state["status"] = "failed"
        state["failure_code"] = (
            exc.kind.value if exc.kind.value in {
                "http_status", "invalid_json", "invalid_response"
            } else "unknown_safe_failure"
        )
        state["remote_completion_unknown"] = exc.remote_completion_unknown
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError):
        state["status"] = "failed"
    finally:
        if state["server_launches"]:
            try:
                if any(h.process.poll() is None for h in owned):
                    raise RuntimeError("owned server remains alive")
                with acquire_resource_lease(root, "llama-device:MTL0"):
                    pass
            except (RuntimeError, OSError, ResourceBusyError):
                state["cleanup_status"] = "failed"
                if state["status"] == "passed":
                    state["status"] = "failed"
                    state["failure_stage"] = "cleanup"
                    state["failure_code"] = "unknown_safe_failure"
            else:
                state["cleanup_status"] = "verified"
    state["wall_seconds"] = time.monotonic() - started
    record = StructuredDiagnosticRecord(
        software_commit=commit, package_source_sha256=package, **state
    )
    publish_immutable(output, record.model_dump_json().encode())
    verified = StructuredDiagnosticRecord.model_validate_json(output.read_bytes())
    if verified != record:
        raise RuntimeError("diagnostic publication verification failed")
    return verified
