"""Measured local identities, bounded probes and narrowly scoped server facts."""

import re
import hashlib
import math
import time
import shlex
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from ..codec import parse_json_unique
from ..contracts import ClosedRecord
from ..timing import RequestTiming


class ExecutableIdentity(ClosedRecord):
    path: str
    sha256: str
    version: str
    build: int
    commit: str
    platform: str


class ArtifactIdentity(ClosedRecord):
    path: str
    sha256: str
    size_bytes: int


class ExternalConflictStatus(StrEnum):
    NO_KNOWN_CONFLICT = "no_known_conflict"
    CONFLICT_DETECTED = "conflict_detected"
    UNVERIFIED = "unverified"


class LlamaQualification(ClosedRecord):
    total_slots: int
    slot_count: int
    model_path: str
    alias: str
    chat_template_sha256: str
    chat_template_caps: tuple[tuple[str, bool], ...]
    build_info: str
    context_tokens: int


class JsonProbe(Protocol):
    async def request_json(self, method: str, path: str) -> tuple[Any, Any, float]: ...


def project_exact_llama_timing(evidence: bytes, timing: RequestTiming) -> RequestTiming:
    """Project only exact-build llama.cpp timing fields from bounded provider evidence."""
    payload = parse_json_unique(evidence)
    if not isinstance(payload, dict):
        raise ValueError("llama response object required")
    timings = payload.get("timings")
    if timings is None:
        return timing
    if not isinstance(timings, dict):
        raise ValueError("llama timings object required")

    def seconds(key: str) -> float | None:
        value = timings.get(key)
        if value is None:
            return None
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid direct llama timing")
        return value / 1000

    return timing.model_copy(update={
        "prompt_processing_seconds": seconds("prompt_ms"),
        "generation_seconds": seconds("predicted_ms"),
    })


def hash_model_artifact(
    path: Path | str, *, deadline: float | None = None
) -> ArtifactIdentity:
    import hashlib

    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("artifact hashing deadline exhausted")

    p = Path(path).resolve(strict=True)
    if not p.is_file():
        raise ValueError("regular model artifact required")
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("artifact hashing deadline exhausted")
            digest.update(chunk)
    return ArtifactIdentity(
        path=str(p), sha256=digest.hexdigest(), size_bytes=p.stat().st_size
    )


def inspect_llama_executable(
    path: Path | str, *, deadline: float | None = None
) -> ExecutableIdentity:
    p = Path(path).resolve(strict=True)
    left = 5.0 if deadline is None else min(5.0, deadline - time.monotonic())
    if left <= 0:
        raise TimeoutError("executable identity deadline exhausted")
    try:
        result = subprocess.run(
            [str(p), "--version"], capture_output=True, text=True, timeout=left
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError("executable identity deadline exhausted") from None
    text = (result.stdout + result.stderr)[:16384]
    match = re.search(
        r"version:\s*(\S+)\s*\(build\s+(\d+),\s*commit\s+([0-9a-f]+)\)", text
    )
    if result.returncode or match is None:
        raise ValueError("unrecognized executable identity")
    platform = text.splitlines()[-1]
    return ExecutableIdentity(
        path=str(p),
        sha256=hash_model_artifact(p, deadline=deadline).sha256,
        version=match[1],
        build=int(match[2]),
        commit=match[3],
        platform=platform,
    )


def classify_processes(text: str, exclude_pids: set[int]) -> ExternalConflictStatus:
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            return ExternalConflictStatus.UNVERIFIED
        pid = int(parts[0])
        command = parts[1].lower()
        if pid in exclude_pids:
            continue
        try:
            argv = shlex.split(command)
        except ValueError:
            return ExternalConflictStatus.UNVERIFIED
        if not argv:
            continue
        executable = Path(argv[0]).name
        is_llama_cli = executable in {"llama-cli", "llama-cli.exe"}
        llama_cli_model = is_llama_cli and any(
            (arg in {"-m", "--model"} and index + 1 < len(argv) and bool(argv[index + 1]))
            or (arg.startswith("--model=") and bool(arg.removeprefix("--model=")))
            for index, arg in enumerate(argv[1:], start=1)
        )
        if is_llama_cli and not llama_cli_model:
            return ExternalConflictStatus.UNVERIFIED
        if (
            executable in {"llama-server", "llama-server.exe"}
            or (executable == "llama" and len(argv) > 1 and argv[1] == "serve")
            or llama_cli_model
            or any(arg in {"mlx_lm.server", "mlx_vlm.server"} for arg in argv[1:])
        ):
            return ExternalConflictStatus.CONFLICT_DETECTED
    return ExternalConflictStatus.NO_KNOWN_CONFLICT


def _lm_studio_main_pids(text: str, exclude_pids: set[int]) -> tuple[int, ...] | None:
    pids: list[int] = []
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            return None
        pid = int(parts[0])
        if pid in exclude_pids:
            continue
        if re.match(
            r"^/.*/LM Studio\.app/Contents/MacOS/LM Studio(?:\s|$)", parts[1]
        ):
            pids.append(pid)
    return tuple(pids)


def _lm_studio_server_status() -> tuple[bool, int | None] | None:
    executable = shutil.which("lms")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "server", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode or len(result.stdout) > 8192:
        return None
    try:
        payload = parse_json_unique(result.stdout.encode("utf-8"))
    except ValueError:
        return None
    if not isinstance(payload, dict) or type(payload.get("running")) is not bool:
        return None
    if not payload["running"]:
        return False, None
    port = payload.get("port")
    if type(port) is not int or not 1 <= port <= 65535:
        return None
    return True, port


def probe_known_external_conflicts(
    exclude_pids: set[int] | None = None,
) -> ExternalConflictStatus:
    excluded = exclude_pids or set()
    try:
        p = subprocess.run(
            ["ps", "-axo", "pid=,args="], capture_output=True, text=True, timeout=3
        )
    except (OSError, subprocess.TimeoutExpired):
        return ExternalConflictStatus.UNVERIFIED
    if p.returncode or len(p.stdout) > 2 * 1024 * 1024:
        return ExternalConflictStatus.UNVERIFIED

    process_status = classify_processes(p.stdout, excluded)
    if process_status is not ExternalConflictStatus.NO_KNOWN_CONFLICT:
        return process_status

    lm_studio_pids = _lm_studio_main_pids(p.stdout, excluded)
    if lm_studio_pids is None:
        return ExternalConflictStatus.UNVERIFIED
    if not lm_studio_pids:
        return ExternalConflictStatus.NO_KNOWN_CONFLICT

    lm_status = _lm_studio_server_status()
    if lm_status is None:
        return ExternalConflictStatus.UNVERIFIED
    running, port = lm_status
    if not running:
        return ExternalConflictStatus.NO_KNOWN_CONFLICT
    assert port is not None

    for pid in lm_studio_pids:
        listener = observe_owned_listener(pid, port)
        if listener in {ListenerStatus.LOOPBACK_VERIFIED, ListenerStatus.WILDCARD_DETECTED}:
            return ExternalConflictStatus.CONFLICT_DETECTED
    return ExternalConflictStatus.UNVERIFIED


async def qualify_llama_server(
    probe: JsonProbe, *, artifact: Path | str, alias: str
) -> LlamaQualification:
    health, _, _ = await probe.request_json("GET", "/health")
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise ValueError("llama.cpp health response is invalid")
    props, _, _ = await probe.request_json("GET", "/props")
    slots, _, _ = await probe.request_json("GET", "/slots")
    models, _, _ = await probe.request_json("GET", "/v1/models")
    if (
        not isinstance(props, dict)
        or type(props.get("total_slots")) is not int
        or props["total_slots"] != 1
    ):
        raise ValueError("exactly one declared slot required")
    if not isinstance(slots, list) or len(slots) != 1 or not isinstance(slots[0], dict):
        raise ValueError("exactly one observed slot required")
    actual = props.get("model_path")
    if (
        not isinstance(actual, str)
        or Path(actual).resolve() != Path(artifact).resolve()
    ):
        raise ValueError("model path mismatch")
    if (
        not isinstance(models, dict)
        or not isinstance(models.get("data"), list)
        or sum(
            isinstance(m, dict) and m.get("id") == alias
            for m in models["data"]
        ) != 1
    ):
        raise ValueError("model alias missing or ambiguous")
    template = props.get("chat_template")
    if not isinstance(template, str) or not template or len(template.encode("utf-8")) > 65536:
        raise ValueError("bounded exact chat template required")
    caps = props.get("chat_template_caps")
    if (
        not isinstance(caps, dict) or len(caps) > 16
        or any(
            not isinstance(key, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key) is None
            or type(value) is not bool
            for key, value in caps.items()
        )
    ):
        raise ValueError("bounded chat template caps required")
    build = props.get("build_info")
    if not isinstance(build, str) or not build or len(build) > 512:
        raise ValueError("bounded build info required")
    defaults = props.get("default_generation_settings")
    if not isinstance(defaults, dict) or type(defaults.get("n_ctx")) is not int or defaults["n_ctx"] != 2048:
        raise ValueError("effective context mismatch")
    return LlamaQualification(
        total_slots=1, slot_count=1, model_path=str(Path(actual).resolve()), alias=alias,
        chat_template_sha256=hashlib.sha256(template.encode("utf-8")).hexdigest(),
        chat_template_caps=tuple(sorted(caps.items())), build_info=build, context_tokens=defaults["n_ctx"],
    )


class ListenerStatus(StrEnum):
    LOOPBACK_VERIFIED = "loopback_verified"
    WILDCARD_DETECTED = "wildcard_detected"
    UNVERIFIED = "unverified"


def classify_owned_listener(text: str, pid: int, port: int) -> ListenerStatus:
    """Classify only the PID and port supplied by this invocation."""
    seen_pid = False
    listen = False
    hosts: list[str] = []
    for line in text.splitlines():
        if line.startswith("p"):
            seen_pid = line[1:].isdigit() and int(line[1:]) == pid
        elif line == "TST=LISTEN" and seen_pid:
            listen = True
        elif line.startswith("n") and seen_pid:
            address = line[1:]
            if address.endswith(":" + str(port)):
                hosts.append(address.rsplit(":", 1)[0])
    if not seen_pid or not listen or not hosts:
        return ListenerStatus.UNVERIFIED
    if any(host in ("*", "0.0.0.0", "[::]", "::") for host in hosts):
        return ListenerStatus.WILDCARD_DETECTED
    if all(host in ("127.0.0.1", "[::1]", "::1") for host in hosts):
        return ListenerStatus.LOOPBACK_VERIFIED
    return ListenerStatus.UNVERIFIED


def observe_owned_listener(pid: int, port: int) -> ListenerStatus:
    if pid < 1 or not 1 <= port <= 65535:
        raise ValueError("valid owned PID and port required")
    try:
        observed = subprocess.run(
            ["lsof", "-nP", "-a", "-p", str(pid), "-iTCP:" + str(port),
             "-sTCP:LISTEN", "-FpnT"],
            capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ListenerStatus.UNVERIFIED
    if observed.returncode or len(observed.stdout) > 8192:
        return ListenerStatus.UNVERIFIED
    return classify_owned_listener(observed.stdout, pid, port)
