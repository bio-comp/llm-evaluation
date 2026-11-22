"""Development-only Quantum CUDA Q0 qualification of an explicitly pinned GGUF.

Owns one child process and one CUDA resource lease. Does not register models,
change research status, reuse unowned servers, or execute the E-002 pilot.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import Field, model_validator

from .artifact_io import publish_immutable
from .records import software_identity
from .runtime.codec import sha256_hex
from .runtime.contracts import (
    BackendConfig,
    CapabilityProfile,
    CapabilityStatus,
    ClosedRecord,
    GenerationRequest,
    HttpTimeouts,
    ModelCondition,
    OperationalEndpoint,
    RuntimeBinding,
    ScientificEndpointIdentity,
)
from .runtime.failures import FailureKind, RuntimeFailure
from .runtime.llamacpp.lifecycle import (
    ManagedServerDeclaration,
    ManagedServerHandle,
    ManagedServerMode,
    ReasoningMode,
    stop_managed_server,
)
from .runtime.llamacpp.qualification import (
    ArtifactIdentity,
    ExecutableIdentity,
    ExpectedExecutable,
    ExternalConflictStatus,
    hash_model_artifact,
    inspect_llama_executable,
    probe_known_external_conflicts,
    project_exact_llama_timing,
    qualify_llama_server,
)
from .runtime.preparation import prepare_generation
from .runtime.resources import acquire_resource_lease
from .runtime.transport.base import BearerAuthorization
from .runtime.transport.openai_compatible import AsyncOpenAICompatibleTransport

Q0_PACKET_SHA256 = "6f879e8ee1bca440306af91c6a6169c97be2c1473acffc0c1882cc20bf9a7bbc"
E002_PACKET_SHA256 = "16547692cd5c4350790fb57cc7cf1936f706dc82af7d770d441bcc28d5a2ae26"
Q0_CASE_IDS = (
    "exact-format-control",
    "factual-constant",
    "arithmetic",
    "python-comprehension",
    "synthetic-long-context",
)
RESOURCE = "llama-device:CUDA0"


class Q0Oracle(ClosedRecord):
    type: Literal["exact_text"]
    expected: str = Field(min_length=1)


class Q0Task(ClosedRecord):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    oracle: Q0Oracle


class Q0Packet(ClosedRecord):
    packet_id: Literal["local-runtime-preflight-v1"]
    purpose: str
    sampling: str
    scoring: Literal["exact visible-text match after no normalization"]
    tasks: tuple[Q0Task, ...]

    @model_validator(mode="after")
    def frozen_case_order(self) -> Q0Packet:
        if tuple(task.id for task in self.tasks) != Q0_CASE_IDS:
            raise ValueError("frozen five-case packet or order mismatch")
        return self


class Q0Config(ClosedRecord):
    artifact: Path
    expected_size_bytes: int = Field(gt=0)
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository: Literal["Qwen/Qwen3-8B-GGUF"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    executable: Path
    expected_runtime: ExpectedExecutable
    packet: Path
    public_manifest: Path
    private_root: Path
    alias: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    port: int = Field(ge=1024, le=65535)
    context_tokens: Literal[8192] = 8192
    parallel: Literal[1] = 1
    gpu_layers: Literal[99] = 99
    max_output_tokens: Literal[128] = 128
    seed: Literal[42] = 42
    startup_seconds: int = Field(default=120, ge=5, le=180)
    request_seconds: int = Field(default=75, ge=5, le=180)

    @model_validator(mode="after")
    def protected_configuration(self) -> Q0Config:
        if self.port == 8080:
            raise ValueError("protected MCP tunnel port")
        if self.packet == self.public_manifest:
            raise ValueError("packet and manifest must differ")
        return self


class Q0CaseResult(ClosedRecord):
    task_id: str
    category: str
    status: Literal[
        "PASS",
        "FAIL_ORACLE",
        "CENSORED_LENGTH",
        "EMPTY_VISIBLE",
        "INVALID_RESPONSE",
        "MODEL_MISMATCH",
        "MISSING_MEASUREMENT",
        "HTTP_FAILURE",
        "TIMEOUT",
    ]
    request_identity_sha256: str
    request_body_sha256: str
    response_body_sha256: str | None
    private_request_file: str
    private_response_file: str | None
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    request_wall_seconds: float | None = Field(default=None, ge=0)
    prompt_processing_seconds: float | None = Field(default=None, ge=0)
    generation_seconds: float | None = Field(default=None, ge=0)
    visible_bytes: int | None = Field(default=None, ge=0)
    reasoning_bytes: int | None = Field(default=None, ge=0)
    failure_kind: str | None = None


class HostMemory(ClosedRecord):
    total_mib: int | None = Field(default=None, ge=0)
    available_mib: int | None = Field(default=None, ge=0)
    swap_used_mib: int | None = Field(default=None, ge=0)


class Q0Manifest(ClosedRecord):
    schema_version: Literal[1] = 1
    evidence_class: Literal["development_only_q0_not_e002"] = (
        "development_only_q0_not_e002"
    )
    status: Literal["COMPLETED", "INCOMPLETE", "BLOCKED"]
    failure_stage: str | None
    failure_code: str | None
    software_commit: str
    package_source_sha256: str
    run_id: str
    repository: str
    revision: str
    artifact_filename: str
    artifact_bytes: int
    artifact_sha256: str
    executable_sha256: str
    executable_version: str
    executable_build: int
    executable_commit: str
    packet_sha256: str
    e002_packet_sha256: str = E002_PACKET_SHA256
    context_tokens: Literal[8192] = 8192
    parallel: Literal[1] = 1
    gpu_layers_requested: Literal[99] = 99
    output_limit: Literal[128] = 128
    seed: Literal[42] = 42
    alias: str
    reasoning_mode: Literal["off"] = "off"
    chat_template_sha256: str | None = None
    chat_template_caps_sha256: str | None = None
    build_info_sha256: str | None = None
    actual_gpu_offload_layers: int | None = Field(default=None, ge=0)
    actual_gpu_total_layers: int | None = Field(default=None, ge=0)
    load_readiness_seconds: float | None = Field(default=None, ge=0)
    gpu_baseline_used_mib: int | None = Field(default=None, ge=0)
    gpu_loaded_used_mib: int | None = Field(default=None, ge=0)
    gpu_observed_peak_used_mib: int | None = Field(default=None, ge=0)
    host_baseline: HostMemory
    host_after_load: HostMemory | None = None
    host_observed_min_available_mib: int | None = Field(default=None, ge=0)
    child_observed_peak_rss_mib: int | None = Field(default=None, ge=0)
    owned_listener_verified: bool = False
    cleanup_verified: bool = False
    protected_port_8080_untouched: bool = True
    lm_studio: Literal["NOT_TESTED"] = "NOT_TESTED"
    private_server_log: str | None = None
    server_log_sha256: str | None = None
    control_token_128247_warning: bool | None = None
    cases: tuple[Q0CaseResult, ...] = ()
    wall_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def completion_is_not_scientific_approval(self) -> Q0Manifest:
        if self.status == "COMPLETED" and (
            len(self.cases) != 5
            or tuple(c.task_id for c in self.cases) != Q0_CASE_IDS
            or any(c.status not in ("PASS", "FAIL_ORACLE") for c in self.cases)
            or not self.cleanup_verified
            or not self.owned_listener_verified
            or self.chat_template_sha256 is None
            or self.chat_template_caps_sha256 is None
            or self.actual_gpu_offload_layers is None
            or self.actual_gpu_total_layers is None
            or self.failure_stage is not None
            or self.failure_code is not None
        ):
            raise ValueError(
                "COMPLETED requires five valid evaluations and verified cleanup"
            )
        return self


class Q0CaseCommon(TypedDict):
    task_id: str
    category: str
    request_identity_sha256: str
    request_body_sha256: str
    private_request_file: str


class Q0Progress(TypedDict, total=False):
    status: Literal["COMPLETED", "INCOMPLETE", "BLOCKED"]
    failure_stage: str | None
    failure_code: str | None
    cases: tuple[Q0CaseResult, ...]
    private_server_log: str
    server_log_sha256: str
    control_token_128247_warning: bool
    cleanup_verified: bool
    gpu_baseline_used_mib: int | None
    gpu_loaded_used_mib: int | None
    gpu_observed_peak_used_mib: int | None
    host_baseline: HostMemory
    host_after_load: HostMemory
    host_observed_min_available_mib: int
    child_observed_peak_rss_mib: int
    load_readiness_seconds: float
    owned_listener_verified: bool
    chat_template_sha256: str
    chat_template_caps_sha256: str
    build_info_sha256: str
    actual_gpu_offload_layers: int | None
    actual_gpu_total_layers: int | None


def validate_identity(
    config: Q0Config,
) -> tuple[ArtifactIdentity, ExecutableIdentity, Q0Packet]:
    """Verify bytes and the pinned Hub snapshot *before* process/resource use."""
    packet_bytes = config.packet.read_bytes()
    if sha256_hex(packet_bytes) != Q0_PACKET_SHA256:
        raise ValueError("frozen Q0 packet digest mismatch")
    packet = Q0Packet.model_validate_json(packet_bytes)
    input_path = config.artifact.expanduser()
    if input_path.name != "Qwen3-8B-Q4_K_M.gguf":
        raise ValueError("GGUF filename mismatch")
    snapshot = input_path.parent
    if snapshot.name != config.revision or snapshot.parent.name != "snapshots":
        raise ValueError("GGUF revision snapshot path mismatch")
    if snapshot.parent.parent.name != "models--Qwen--Qwen3-8B-GGUF":
        raise ValueError("GGUF repository snapshot path mismatch")
    if (
        not input_path.is_file()
        or input_path.stat().st_size != config.expected_size_bytes
    ):
        raise ValueError("GGUF missing or byte size mismatch")
    artifact = hash_model_artifact(input_path)
    if (
        artifact.sha256 != config.expected_sha256
        or artifact.size_bytes != config.expected_size_bytes
    ):
        raise ValueError("GGUF digest or size mismatch")
    runtime = inspect_llama_executable(config.executable)
    if not config.expected_runtime.matches(runtime):
        raise ValueError("llama-server version/build/digest mismatch")
    return artifact, runtime, packet


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _lm_studio_api_stopped() -> bool:
    executable = shutil.which("lms")
    if executable is None:
        return True
    try:
        completed = subprocess.run(
            [executable, "server", "status", "--json"],
            capture_output=True,
            timeout=5,
            check=True,
        )
        payload = json.loads(completed.stdout)
        return isinstance(payload, dict) and payload.get("running") is False
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _gpu_used_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return int(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _host_memory() -> HostMemory:
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            if name in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                fields[name] = int(value.strip().split()[0]) // 1024
        return HostMemory(
            total_mib=fields.get("MemTotal"),
            available_mib=fields.get("MemAvailable"),
            swap_used_mib=(
                fields["SwapTotal"] - fields["SwapFree"]
                if "SwapTotal" in fields and "SwapFree" in fields
                else None
            ),
        )
    except (OSError, ValueError, IndexError):
        return HostMemory()


def _child_rss_mib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _offload_observation(log_bytes: bytes) -> tuple[int | None, int | None]:
    matches = re.findall(
        rb"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", log_bytes
    )
    if not matches:
        return None, None
    layers, total = matches[-1]
    return int(layers), int(total)


def _request_binding(
    config: Q0Config,
    artifact: ArtifactIdentity,
    runtime: ExecutableIdentity,
    template_sha: str,
) -> RuntimeBinding:
    profile = CapabilityProfile(
        statuses={
            name: CapabilityStatus.SUPPORTED
            for name in ("chat_completions", "max_output_tokens", "temperature", "seed")
        }
    )
    condition = ModelCondition(
        condition_id="quantum-q0-five-case-v1",
        weights_id=config.repository + "@" + config.revision,
        artifact_sha256=artifact.sha256,
        quantization="Q4_K_M",
        provider_id="managed-llamacpp",
        backend_config_id="quantum-q0-cuda-v1",
        backend_name="llamacpp",
        backend_version=f"{runtime.version}-{runtime.build}-{runtime.commit}",
        api_mode="chat_completions",
        endpoint_model_id=config.alias,
        prompt_template_id="llamacpp-chat-template-sha256:" + template_sha,
        seed_behavior="seed=42 requested; effective stochasticity not independently verified",
        runtime_environment={
            "context_tokens": config.context_tokens,
            "parallel": config.parallel,
            "gpu_layers_requested": config.gpu_layers,
            "device": "CUDA0",
            "reasoning": "off",
            "chat_template_kwargs": {"enable_thinking": False},
            "output_limit": config.max_output_tokens,
            "fit": "off",
        },
        artifact_filename=config.artifact.name,
    )
    return RuntimeBinding(
        backend=BackendConfig(
            provider_id="managed-llamacpp",
            model_id=config.alias,
            api_mode="chat_completions",
            capability_profile=profile,
            config_id="quantum-q0-cuda-v1",
        ),
        condition=condition,
        scientific_endpoint=ScientificEndpointIdentity(value="quantum-q0-cuda-v1"),
        timeouts=HttpTimeouts(
            connect=3.0,
            read=float(config.request_seconds),
            write=3.0,
            pool=3.0,
            total=float(config.request_seconds),
        ),
    )


def classify_case(
    task: Q0Task,
    *,
    visible: str,
    finish: str | None,
    reported_model_match: bool | None,
    reasoning_bytes: int,
    completion_tokens: int | None,
    limit: int,
    measurement_complete: bool = True,
) -> Literal[
    "PASS",
    "FAIL_ORACLE",
    "CENSORED_LENGTH",
    "EMPTY_VISIBLE",
    "INVALID_RESPONSE",
    "MODEL_MISMATCH",
    "MISSING_MEASUREMENT",
]:
    if reported_model_match is not True:
        return "MODEL_MISMATCH"
    if finish == "length" or (
        completion_tokens is not None
        and completion_tokens >= limit
        and finish != "stop"
    ):
        return "CENSORED_LENGTH"
    if finish != "stop" or reasoning_bytes > 0:
        return "INVALID_RESPONSE"
    if not visible:
        return "EMPTY_VISIBLE"
    if not measurement_complete:
        return "MISSING_MEASUREMENT"
    return "PASS" if visible == task.oracle.expected else "FAIL_ORACLE"


async def _collect(
    config: Q0Config,
    artifact: ArtifactIdentity,
    runtime: ExecutableIdentity,
    packet: Q0Packet,
    run_dir: Path,
    state: Q0Progress,
) -> None:
    if probe_known_external_conflicts() is not ExternalConflictStatus.NO_KNOWN_CONFLICT:
        raise ValueError("external runtime conflict or unknown ownership")
    if not _lm_studio_api_stopped():
        raise ValueError("external LM Studio API running or state unknown")
    if not _port_is_free(config.port):
        raise ValueError("Q0 loopback port occupied")
    # Recheck source bytes at the launch boundary; the cache is not privately
    # snapshotted on this Linux profile. Later verification checks for changes
    # during the run. This is cooperative provenance, not malicious-UID security.
    before_launch = hash_model_artifact(artifact.path)
    if (before_launch.sha256, before_launch.size_bytes) != (
        artifact.sha256,
        artifact.size_bytes,
    ):
        raise ValueError("GGUF changed before launch")
    before_launch_runtime = inspect_llama_executable(runtime.path)
    if (
        before_launch_runtime.sha256 != runtime.sha256
        or before_launch_runtime.version != runtime.version
        or before_launch_runtime.build != runtime.build
        or before_launch_runtime.commit != runtime.commit
    ):
        raise ValueError("runtime changed before launch")
    lease_root = Path.home() / ".local/state/llm-runtime/leases"
    with acquire_resource_lease(lease_root, RESOURCE):
        pass
    args = [
        str(runtime.path),
        "--device",
        "CUDA0",
        "--n-gpu-layers",
        "99",
        "--fit",
        "off",
        "--parallel",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "--model",
        artifact.path,
        "--alias",
        config.alias,
        "--ctx-size",
        "8192",
        "--predict",
        "128",
        "--reasoning",
        "off",
        "--reasoning-format",
        "deepseek",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--jinja",
        "--verbosity",
        "5",
        "--no-webui",
        "--no-agent",
        "--no-ui-mcp-proxy",
        "--slots",
        "--cache-ram",
        "0",
        "--no-cache-idle-slots",
        "--no-cache-prompt",
        "--no-context-shift",
    ]
    env = {
        name: os.environ[name]
        for name in ("HOME", "PATH", "TMPDIR", "LANG", "LC_ALL")
        if name in os.environ
    }
    auth = BearerAuthorization(secrets.token_urlsafe(32))
    env["LLAMA_API_KEY"] = auth.reveal()
    server_log = run_dir / "server.stderr.log"
    endpoint = OperationalEndpoint(base_url=f"http://127.0.0.1:{config.port}")
    declaration = ManagedServerDeclaration(
        executable=runtime.path,
        artifact=artifact.path,
        alias=config.alias,
        mode=ManagedServerMode.FORMAL,
        reasoning_mode=ReasoningMode.OFF,
        resource=RESOURCE,
    )
    state["failure_stage"] = "startup"
    started = time.monotonic()
    with server_log.open("xb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "llm_evaluation.runtime.llamacpp.lease_exec",
                str(lease_root),
                RESOURCE,
                *args,
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    handle = ManagedServerHandle(
        process=process, endpoint=endpoint, declaration=declaration
    )
    state["private_server_log"] = server_log.name
    monitor_done = asyncio.Event()

    async def monitor() -> None:
        while not monitor_done.is_set():
            used = _gpu_used_mib()
            if used is not None:
                existing = state.get("gpu_observed_peak_used_mib")
                if not isinstance(existing, int) or used > existing:
                    state["gpu_observed_peak_used_mib"] = used
            host = _host_memory()
            available = host.available_mib
            previous = state.get("host_observed_min_available_mib")
            if available is not None and (
                not isinstance(previous, int) or available < previous
            ):
                state["host_observed_min_available_mib"] = available
            rss = _child_rss_mib(process.pid)
            peak_rss = state.get("child_observed_peak_rss_mib")
            if rss is not None and (not isinstance(peak_rss, int) or rss > peak_rss):
                state["child_observed_peak_rss_mib"] = rss
            await asyncio.sleep(0.35)

    task = asyncio.create_task(monitor())
    try:
        health_probe = AsyncOpenAICompatibleTransport(
            endpoint,
            authorization=auth,
            timeouts=HttpTimeouts(
                connect=0.5, read=1.0, write=1.0, pool=1.0, total=2.0
            ),
        )
        async with asyncio.timeout(config.startup_seconds):
            while True:
                if process.poll() is not None:
                    raise RuntimeError("owned llama-server exited before readiness")
                try:
                    health_status, _, _ = await health_probe.request_json(
                        "GET", "/health"
                    )
                    if (
                        isinstance(health_status, dict)
                        and health_status.get("status") == "ok"
                    ):
                        break
                except RuntimeFailure:
                    pass
                await asyncio.sleep(0.20)
        state["load_readiness_seconds"] = time.monotonic() - started
        state["gpu_loaded_used_mib"] = _gpu_used_mib()
        state["host_after_load"] = _host_memory()
        state["failure_stage"] = "qualification"
        from .runtime.llamacpp.qualification import (
            ListenerStatus,
            observe_owned_listener,
        )

        owned_listener = observe_owned_listener(process.pid, config.port)
        state["owned_listener_verified"] = (
            owned_listener is ListenerStatus.LOOPBACK_VERIFIED
        )
        if not state["owned_listener_verified"]:
            raise ValueError("Q0 listener ownership or loopback binding unverified")
        qualifier = AsyncOpenAICompatibleTransport(
            endpoint,
            authorization=auth,
            timeouts=HttpTimeouts(
                connect=3.0, read=8.0, write=3.0, pool=3.0, total=10.0
            ),
        )
        observation = await qualify_llama_server(
            qualifier,
            artifact=artifact.path,
            alias=config.alias,
            expected_context_tokens=config.context_tokens,
        )
        state["chat_template_sha256"] = observation.chat_template_sha256
        state["chat_template_caps_sha256"] = observation.chat_template_caps_sha256
        state["build_info_sha256"] = sha256_hex(observation.build_info.encode())
        if (
            str(runtime.build) not in observation.build_info
            or runtime.commit not in observation.build_info
        ):
            raise ValueError("server reported build_info mismatch")
        log_bytes = server_log.read_bytes()
        offloaded, total_layers = _offload_observation(log_bytes)
        state["actual_gpu_offload_layers"] = offloaded
        state["actual_gpu_total_layers"] = total_layers
        if offloaded is None or total_layers is None or offloaded < total_layers:
            raise ValueError("CUDA GPU-layer offload not fully observed")
        binding = _request_binding(
            config, artifact, runtime, observation.chat_template_sha256
        )
        transport = AsyncOpenAICompatibleTransport(
            OperationalEndpoint(base_url=endpoint.base_url + "/v1"),
            authorization=auth,
            timeouts=binding.timeouts,
        )
        cases: list[Q0CaseResult] = []
        state["failure_stage"] = "generation"
        for index, item in enumerate(packet.tasks, 1):
            if process.poll() is not None:
                raise RuntimeError("owned llama-server exited during Q0")
            prepared = prepare_generation(
                binding,
                GenerationRequest(
                    condition_id=binding.condition.condition_id,
                    messages=(("user", item.prompt),),
                    max_output_tokens=config.max_output_tokens,
                    temperature=0.0,
                    seed=config.seed,
                ),
            )
            request_name = f"case-{index:02d}-request.json"
            publish_immutable(run_dir / request_name, prepared.wire_bytes)
            common: Q0CaseCommon = dict(
                task_id=item.id,
                category=item.category,
                request_identity_sha256=prepared.request_identity_sha256,
                request_body_sha256=prepared.body_sha256,
                private_request_file=request_name,
            )
            try:
                execution = await transport.generate(prepared)
            except RuntimeFailure as exc:
                status: Literal["TIMEOUT", "HTTP_FAILURE", "INVALID_RESPONSE"] = (
                    "TIMEOUT"
                    if exc.kind
                    in (
                        FailureKind.TOTAL_TIMEOUT,
                        FailureKind.READ_TIMEOUT,
                        FailureKind.CONNECT_TIMEOUT,
                        FailureKind.WRITE_TIMEOUT,
                        FailureKind.POOL_TIMEOUT,
                    )
                    else "INVALID_RESPONSE"
                    if exc.kind
                    in (
                        FailureKind.INVALID_JSON,
                        FailureKind.INVALID_RESPONSE,
                        FailureKind.RESPONSE_TOO_LARGE,
                    )
                    else "HTTP_FAILURE"
                )
                cases.append(
                    Q0CaseResult(
                        **common,
                        status=status,
                        response_body_sha256=None,
                        private_response_file=None,
                        failure_kind=exc.kind.value,
                    )
                )
                state["cases"] = tuple(cases)
                raise ValueError("technical generation failure") from None
            name = f"case-{index:02d}-response.json"
            publish_immutable(run_dir / name, execution.evidence.body)
            summary = project_exact_llama_timing(
                execution.evidence.body, execution.timing
            )
            from .local_runtime_preflight_records import summarize_generation

            evidence = summarize_generation(
                execution.observation, execution.evidence.body
            )
            classification = classify_case(
                item,
                visible=execution.observation.raw_output,
                finish=execution.observation.termination_reason,
                reported_model_match=execution.observation.reported_model_matches_condition,
                reasoning_bytes=evidence.reasoning_bytes,
                completion_tokens=execution.observation.completion_tokens,
                limit=config.max_output_tokens,
                measurement_complete=(
                    summary.prompt_processing_seconds is not None
                    and summary.generation_seconds is not None
                    and execution.observation.prompt_tokens is not None
                    and execution.observation.completion_tokens is not None
                    and execution.observation.total_tokens is not None
                ),
            )
            cases.append(
                Q0CaseResult(
                    **common,
                    status=classification,
                    response_body_sha256=execution.evidence.body_sha256,
                    private_response_file=name,
                    finish_reason=execution.observation.finish_reason,
                    prompt_tokens=execution.observation.prompt_tokens,
                    completion_tokens=execution.observation.completion_tokens,
                    total_tokens=execution.observation.total_tokens,
                    request_wall_seconds=summary.total_request_wall_seconds,
                    prompt_processing_seconds=summary.prompt_processing_seconds,
                    generation_seconds=summary.generation_seconds,
                    visible_bytes=evidence.visible_output_bytes,
                    reasoning_bytes=evidence.reasoning_bytes,
                )
            )
            state["cases"] = tuple(cases)
            if classification not in ("PASS", "FAIL_ORACLE"):
                raise ValueError("technical output validity failure")
        state["failure_stage"] = None
        state["failure_code"] = None
    finally:
        monitor_done.set()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        stop_managed_server(handle)
        state["cleanup_verified"] = process.poll() is not None
        with acquire_resource_lease(lease_root, RESOURCE):
            pass
        state["cleanup_verified"] = True
        log_bytes = server_log.read_bytes()
        state["server_log_sha256"] = sha256_hex(log_bytes)
        state["control_token_128247_warning"] = b"128247" in log_bytes
        if state.get("actual_gpu_offload_layers") is None:
            state["actual_gpu_offload_layers"], state["actual_gpu_total_layers"] = (
                _offload_observation(log_bytes)
            )
    # A successful HTTP response does not establish unchanged source bytes.
    after_run = hash_model_artifact(artifact.path)
    after_runtime = inspect_llama_executable(runtime.path)
    if (after_run.sha256, after_run.size_bytes) != (
        artifact.sha256,
        artifact.size_bytes,
    ) or (
        after_runtime.sha256 != runtime.sha256
        or after_runtime.version != runtime.version
        or after_runtime.build != runtime.build
        or after_runtime.commit != runtime.commit
    ):
        raise ValueError("model or runtime bytes changed during Q0")


def run_q0(config: Q0Config) -> Q0Manifest:
    """One bounded, immutable development-run output; no retries or hidden fallback."""
    config = Q0Config.model_validate(config.model_dump())
    if config.public_manifest.exists() or config.public_manifest.is_symlink():
        raise FileExistsError("public manifest already exists")
    started = time.monotonic()
    commit, package_sha = software_identity()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + secrets.token_hex(4)
    )
    state: Q0Progress = {
        "status": "BLOCKED",
        "failure_stage": "identity",
        "failure_code": None,
        "cases": (),
    }
    private_dir: Path | None = None
    try:
        artifact, runtime, packet = validate_identity(config)
        state["failure_stage"] = "resource"
        config.private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (
            config.private_root.is_symlink()
            or config.private_root.stat().st_mode & 0o077
        ):
            raise PermissionError("private evidence root must be owner-only")
        private_dir = config.private_root / ("q0-" + run_id)
        private_dir.mkdir(mode=0o700, exist_ok=False)
        state["gpu_baseline_used_mib"] = _gpu_used_mib()
        state["host_baseline"] = _host_memory()
        asyncio.run(_collect(config, artifact, runtime, packet, private_dir, state))
        state["status"] = "COMPLETED"
    except (
        OSError,
        ValueError,
        RuntimeError,
        TimeoutError,
        subprocess.SubprocessError,
        KeyboardInterrupt,
    ) as exc:
        state["failure_code"] = type(exc).__name__
        if state.get("cases") or isinstance(exc, KeyboardInterrupt):
            state["status"] = "INCOMPLETE"
    state.setdefault("host_baseline", _host_memory())
    record = Q0Manifest.model_validate(
        dict(
            software_commit=commit,
            package_source_sha256=package_sha,
            run_id=run_id,
            repository=config.repository,
            revision=config.revision,
            artifact_filename=config.artifact.name,
            artifact_bytes=config.expected_size_bytes,
            artifact_sha256=config.expected_sha256,
            executable_sha256=config.expected_runtime.sha256,
            executable_version=config.expected_runtime.version,
            executable_build=config.expected_runtime.build,
            executable_commit=config.expected_runtime.commit,
            packet_sha256=Q0_PACKET_SHA256,
            alias=config.alias,
            wall_seconds=time.monotonic() - started,
            **state,
        )
    )
    serialized = record.model_dump_json(indent=2).encode() + b"\n"
    publish_immutable(config.public_manifest, serialized)
    if Q0Manifest.model_validate_json(config.public_manifest.read_bytes()) != record:
        raise RuntimeError("Q0 manifest round-trip verification failed")
    return record
