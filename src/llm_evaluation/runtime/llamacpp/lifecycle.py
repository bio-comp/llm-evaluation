"""One lifecycle owner; shared and formal servers contend on the same key."""

import asyncio
import os
import secrets
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Callable
from pydantic import Field
from ..contracts import ClosedRecord, OperationalEndpoint, HttpTimeouts
from ..resources import acquire_resource_lease, ResourceBusyError, canonical_resource
from ..transport.base import BearerAuthorization
from ..transport.openai_compatible import AsyncOpenAICompatibleTransport
from ..failures import RuntimeFailure


class ManagedServerMode(StrEnum):
    SHARED = "shared"
    FORMAL = "formal"


class ReasoningMode(StrEnum):
    AUTO = "auto"
    OFF = "off"


FORMAL_CACHE_PROMPT = False


class ManagedServerDeclaration(ClosedRecord):
    executable: str
    artifact: str
    alias: str
    mode: ManagedServerMode
    reasoning_mode: ReasoningMode | None = None
    resource: str = "llama-device:MTL0"
    lease_root: str = Field(
        default_factory=lambda: str(Path.home() / ".local/state/llm-runtime/leases")
    )


class FormalServerAuthorization(BearerAuthorization):
    pass


@dataclass
class ManagedServerHandle:
    process: subprocess.Popen[bytes] = field(repr=False)
    endpoint: OperationalEndpoint
    declaration: ManagedServerDeclaration

    def public_state(self) -> dict[str, object]:
        return {
            "pid": self.process.pid,
            "base_url": self.endpoint.base_url,
            "mode": self.declaration.mode.value,
            "resource": self.declaration.resource,
            "lease_scope": "cooperating_same_namespace",
        }


@dataclass(repr=False)
class ManagedServerLaunch:
    handle: ManagedServerHandle
    authorization: FormalServerAuthorization | None

    def __repr__(self) -> str:
        return "ManagedServerLaunch(" + repr(self.handle.public_state()) + ")"


def build_launch(
    declaration: ManagedServerDeclaration, port: int
) -> tuple[list[str], dict[str, str], FormalServerAuthorization | None]:
    if not 1 <= port <= 65535:
        raise ValueError("valid candidate port required")
    key = canonical_resource(declaration.resource)
    if key != "llama-device:MTL0":
        raise ValueError("initial managed profile supports only MTL0")
    if declaration.mode is ManagedServerMode.FORMAL and declaration.reasoning_mode is None:
        raise ValueError("formal reasoning profile required")
    executable = str(Path(declaration.executable).resolve(strict=True))
    artifact = str(Path(declaration.artifact).resolve(strict=True))
    if not declaration.alias.strip():
        raise ValueError("model alias required")
    args = [
        executable,
        "--device",
        "MTL0",
        "--parallel",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        artifact,
        "--alias",
        declaration.alias,
        "--ctx-size",
        "2048",
        "--predict",
        "128",
        "--reasoning",
        (declaration.reasoning_mode or ReasoningMode.AUTO).value,
        "--jinja",  # Pin the documented llama-server v0.7.0 enabled default.
        "--no-webui",
        "--no-agent",
        "--no-ui-mcp-proxy",
        "--slots",
        "--cache-ram",
        "0",
        "--no-cache-idle-slots",
        "--log-disable",
    ]
    if declaration.mode is ManagedServerMode.FORMAL and not FORMAL_CACHE_PROMPT:
        args.append("--no-cache-prompt")
    # No inherited LLAMA_*, HF_TOKEN, tool/MCP, router, save-path or proxy settings.
    env = {
        name: os.environ[name]
        for name in ("HOME", "PATH", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")
        if name in os.environ
    }
    auth = (
        FormalServerAuthorization(secrets.token_urlsafe(32))
        if declaration.mode is ManagedServerMode.FORMAL
        else None
    )
    if auth is not None:
        env["LLAMA_API_KEY"] = auth.reveal()
    return args, env, auth


def stop_managed_server(handle: ManagedServerHandle) -> None:
    """Only our Popen child is terminated; wait establishes process/lease release."""
    p = handle.process
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)
    else:
        p.wait()


async def start_managed_server(
    declaration: ManagedServerDeclaration,
    *,
    startup_seconds: float = 45.0,
    before_spawn: Callable[[], None] | None = None,
) -> ManagedServerLaunch:
    if startup_seconds <= 0 or startup_seconds > 45:
        raise ValueError("startup bound must be in (0,45]")
    # Early conflict reporting; child acquires the authoritative lock independently.
    with acquire_resource_lease(declaration.lease_root, declaration.resource):
        pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    args, env, auth = build_launch(declaration, port)
    command = [
        sys.executable,
        "-m",
        "llm_evaluation.runtime.llamacpp.lease_exec",
        declaration.lease_root,
        declaration.resource,
        *args,
    ]
    if before_spawn is not None:
        before_spawn()
    p = subprocess.Popen(
        command,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    handle = ManagedServerHandle(
        p, OperationalEndpoint(base_url=f"http://127.0.0.1:{port}"), declaration
    )
    probe = AsyncOpenAICompatibleTransport(
        handle.endpoint,
        authorization=auth,
        timeouts=HttpTimeouts(connect=0.3, read=0.5, write=0.5, pool=0.5, total=1.0),
    )
    try:
        async with asyncio.timeout(startup_seconds):
            while True:
                code = p.poll()
                if code is not None:
                    if code == 73:
                        raise ResourceBusyError("resource busy")
                    raise RuntimeError("managed server startup failed")
                try:
                    await probe.request_json("GET", "/health")
                    if p.poll() is None:
                        return ManagedServerLaunch(handle, auth)
                except RuntimeFailure:
                    pass
                await asyncio.sleep(0.05)
    except BaseException:
        stop_managed_server(handle)
        raise
