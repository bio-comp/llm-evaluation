"""Small auditable generation boundary for OpenAI-compatible HTTP endpoints.

This module deliberately implements only the portable single-turn text subset needed
for E-002 preflight.  A compatible transport is not evidence that two backends are
scientifically equivalent; effective requests and raw provider metadata are retained.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class CapabilityStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ACCEPTED_BUT_IGNORED = "accepted-but-ignored"
    SEMANTICS_DIFFER = "semantics-differ"
    UNKNOWN = "unknown"


class AttemptStatus(StrEnum):
    TRANSPORT_FAILURE = "transport_failure"
    PROVIDER_RUNTIME_FAILURE = "provider_runtime_failure"
    MODEL_GENERATION_COMPLETED = "model_generation_completed"
    ORACLE_FAILURE = "oracle_failure"


class CapabilityError(ValueError):
    """Requested scientific control is not known to have portable semantics."""


class CredentialError(ValueError):
    """Configured credential source is missing."""


class TransportFailure(RuntimeError):
    """The request did not reach a successful provider HTTP response."""


class ProviderRuntimeFailure(RuntimeError):
    """The provider returned an HTTP or response-shape failure."""


@dataclass(frozen=True)
class CapabilityProfile:
    statuses: Mapping[str, CapabilityStatus]

    def require(self, capability: str) -> None:
        status = self.statuses.get(capability, CapabilityStatus.UNKNOWN)
        if status is not CapabilityStatus.SUPPORTED:
            raise CapabilityError(f"{capability} is {status.value}")


@dataclass(frozen=True)
class BackendConfig:
    provider_id: str
    base_url: str
    model_id: str
    api_mode: str
    capability_profile: CapabilityProfile
    credential_env: str | None = None

    def __post_init__(self) -> None:
        if not self.provider_id or not self.model_id:
            raise ValueError("provider_id and model_id are required")
        if self.api_mode not in {"chat_completions", "responses"}:
            raise ValueError("api_mode must be chat_completions or responses")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be HTTP(S)")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))


@dataclass(frozen=True)
class BackendRegistry:
    """Tiny explicit lookup layer; provider behavior still lives in capability profiles."""

    backends: tuple[BackendConfig, ...]

    def __post_init__(self) -> None:
        ids = tuple(backend.provider_id for backend in self.backends)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate provider_id in backend registry")

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(backend.provider_id for backend in self.backends)

    def __getitem__(self, provider_id: str) -> BackendConfig:
        for backend in self.backends:
            if backend.provider_id == provider_id:
                return backend
        raise KeyError(provider_id)


@dataclass(frozen=True)
class ModelCondition:
    """Scientific condition identity kept separate from endpoint configuration."""

    condition_id: str
    weights_id: str
    artifact_sha256: str | None
    quantization: str | None
    provider_id: str
    backend_name: str
    backend_version: str
    api_mode: str
    endpoint_model_id: str
    prompt_template_id: str
    seed_behavior: str
    runtime_environment: Mapping[str, object]
    artifact_filename: str | None = None
    tokenizer_id: str | None = None
    artifact_metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not all((self.condition_id, self.weights_id, self.provider_id, self.backend_name,
                    self.backend_version, self.api_mode, self.endpoint_model_id,
                    self.prompt_template_id, self.seed_behavior)):
            raise ValueError("model condition identity fields must be nonempty")
        if self.artifact_sha256 is not None:
            digest = self.artifact_sha256.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("artifact_sha256 must be a 64-character hexadecimal digest")


@dataclass(frozen=True)
class GenerationRequest:
    condition_id: str
    messages: tuple[tuple[str, str], ...]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    stop: tuple[str, ...] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.condition_id:
            raise ValueError("condition_id is required")
        if not self.messages or any(role not in {"system", "user", "assistant"} or not isinstance(text, str)
                                    for role, text in self.messages):
            raise ValueError("messages must contain text system/user/assistant entries")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True)
class EffectiveRequest:
    endpoint: str
    api_mode: str
    serialized_body: str
    body_sha256: str
    prompt_packet_sha256: str


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    backend_model_id: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    provider_request_id: str | None
    raw_provider_metadata: dict[str, Any]
    elapsed_seconds: float
    effective_request: EffectiveRequest


@dataclass(frozen=True)
class GenerationAttempt:
    """One technical attempt; retry lineage is explicit and never implies a new sample."""

    attempt_index: int
    retry_of_attempt: int | None
    retry_count: int
    contributes_to_sample: bool
    status: AttemptStatus
    result: GenerationResult | None
    failure_detail: str | None

    def __post_init__(self) -> None:
        if self.attempt_index < 1 or self.retry_count < 0:
            raise ValueError("attempt indexes must be nonnegative/positive")
        if self.retry_of_attempt is not None and not 1 <= self.retry_of_attempt < self.attempt_index:
            raise ValueError("retry_of_attempt must reference an earlier attempt")
        if self.retry_of_attempt is None and self.retry_count != 0:
            raise ValueError("non-retry attempts require retry_count=0")
        if self.retry_of_attempt is not None and self.retry_count < 1:
            raise ValueError("retry attempts require positive retry_count")
        if self.status is AttemptStatus.MODEL_GENERATION_COMPLETED and self.result is None:
            raise ValueError("completed generation requires a result")
        if self.status is not AttemptStatus.MODEL_GENERATION_COMPLETED and self.result is not None:
            raise ValueError("failed attempt cannot carry a generation result")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _messages(request: GenerationRequest) -> list[dict[str, str]]:
    return [{"role": role, "content": text} for role, text in request.messages]


def _prompt_packet(request: GenerationRequest) -> dict[str, object]:
    return {
        "condition_id": request.condition_id,
        "messages": _messages(request),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "max_output_tokens": request.max_output_tokens,
        "stop": list(request.stop) if request.stop is not None else None,
        "seed": request.seed,
    }


class OpenAICompatibleTransport:
    """Exact-wire stdlib HTTP transport for a capability-gated compatible endpoint."""

    def __init__(self, config: BackendConfig, *, timeout_seconds: float = 120.0) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.credential_env is not None:
            secret = os.environ.get(self.config.credential_env)
            if not secret:
                raise CredentialError(f"missing credential environment variable {self.config.credential_env}")
            headers["Authorization"] = f"Bearer {secret}"
        return headers

    def _body(self, request: GenerationRequest) -> tuple[str, dict[str, object]]:
        profile = self.config.capability_profile
        profile.require(self.config.api_mode)
        body: dict[str, object] = {"model": self.config.model_id}
        if self.config.api_mode == "chat_completions":
            body["messages"] = _messages(request)
        else:
            body["input"] = _messages(request)
        controls = (
            ("temperature", request.temperature, "temperature"),
            ("top_p", request.top_p, "top_p"),
            ("top_k", request.top_k, "top_k"),
            ("seed", request.seed, "seed"),
            ("stop", list(request.stop) if request.stop is not None else None, "stop"),
        )
        for capability, value, wire_name in controls:
            if value is not None:
                profile.require(capability)
                body[wire_name] = value
        if request.max_output_tokens is not None:
            profile.require("max_output_tokens")
            wire_name = "max_tokens" if self.config.api_mode == "chat_completions" else "max_output_tokens"
            body[wire_name] = request.max_output_tokens
        endpoint = "/chat/completions" if self.config.api_mode == "chat_completions" else "/responses"
        return endpoint, body

    def _effective(self, request: GenerationRequest, endpoint: str, body: dict[str, object]) -> EffectiveRequest:
        serialized = canonical_json(body)
        prompt = canonical_json(_prompt_packet(request))
        return EffectiveRequest(
            endpoint=f"{self.config.base_url}{endpoint}",
            api_mode=self.config.api_mode,
            serialized_body=serialized,
            body_sha256=_sha256_text(serialized),
            prompt_packet_sha256=_sha256_text(prompt),
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        endpoint, body = self._body(request)
        effective = self._effective(request, endpoint, body)
        wire = effective.serialized_body.encode("utf-8")
        http_request = urllib.request.Request(
            effective.endpoint, data=wire, headers=self._headers(), method="POST"
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                header_request_id = response.headers.get("X-Request-ID")
        except urllib.error.HTTPError as exc:
            # Do not surface provider error bodies here: a hostile or buggy endpoint
            # can echo an Authorization value or other credential into its payload.
            raise ProviderRuntimeFailure(f"HTTP {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportFailure(str(exc)) from exc
        elapsed = time.perf_counter() - started
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderRuntimeFailure("provider returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderRuntimeFailure("provider response must be a JSON object")
        if self.config.api_mode == "chat_completions":
            parsed = self._parse_chat(payload)
        else:
            parsed = self._parse_responses(payload)
        request_id = payload.get("id") if isinstance(payload.get("id"), str) else header_request_id
        return GenerationResult(
            **parsed,
            provider_request_id=request_id,
            raw_provider_metadata=payload,
            elapsed_seconds=elapsed,
            effective_request=effective,
        )

    @staticmethod
    def _parse_chat(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderRuntimeFailure("invalid Chat Completions response shape") from exc
        if not isinstance(text, str):
            raise ProviderRuntimeFailure("Chat Completions text output is not a string")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return {
            "raw_output": text,
            "backend_model_id": payload.get("model") if isinstance(payload.get("model"), str) else None,
            "finish_reason": choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
            "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
            "completion_tokens": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
            "total_tokens": usage.get("total_tokens") if isinstance(usage.get("total_tokens"), int) else None,
        }

    @staticmethod
    def _parse_responses(payload: dict[str, Any]) -> dict[str, Any]:
        output = payload.get("output")
        if not isinstance(output, list):
            raise ProviderRuntimeFailure("Responses payload has no output list")
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                raise ProviderRuntimeFailure("Responses message content is not a list")
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return {
            "raw_output": "".join(texts),
            "backend_model_id": payload.get("model") if isinstance(payload.get("model"), str) else None,
            "finish_reason": payload.get("status") if isinstance(payload.get("status"), str) else None,
            "prompt_tokens": usage.get("input_tokens") if isinstance(usage.get("input_tokens"), int) else None,
            "completion_tokens": usage.get("output_tokens") if isinstance(usage.get("output_tokens"), int) else None,
            "total_tokens": usage.get("total_tokens") if isinstance(usage.get("total_tokens"), int) else None,
        }

    def list_models(self) -> tuple[str, ...]:
        self.config.capability_profile.require("models")
        request = urllib.request.Request(f"{self.config.base_url}/models", headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise TransportFailure(str(exc)) from exc
        try:
            values = tuple(item["id"] for item in payload["data"] if isinstance(item.get("id"), str))
        except (KeyError, TypeError) as exc:
            raise ProviderRuntimeFailure("invalid model-list response") from exc
        return values


def attempt_generation(
    transport: OpenAICompatibleTransport,
    request: GenerationRequest,
    *,
    attempt_index: int,
    retry_of_attempt: int | None = None,
    retry_count: int = 0,
    contributes_to_sample: bool = False,
) -> GenerationAttempt:
    """Execute one visible attempt without automatic retries.

    A caller may schedule a retry, but must link it to the earlier attempt and decide
    explicitly whether any completed attempt contributes to the statistical sample.
    """
    try:
        result = transport.generate(request)
    except TransportFailure as exc:
        return GenerationAttempt(
            attempt_index, retry_of_attempt, retry_count, contributes_to_sample,
            AttemptStatus.TRANSPORT_FAILURE, None, str(exc),
        )
    except ProviderRuntimeFailure as exc:
        return GenerationAttempt(
            attempt_index, retry_of_attempt, retry_count, contributes_to_sample,
            AttemptStatus.PROVIDER_RUNTIME_FAILURE, None, str(exc),
        )
    return GenerationAttempt(
        attempt_index, retry_of_attempt, retry_count, contributes_to_sample,
        AttemptStatus.MODEL_GENERATION_COMPLETED, result, None,
    )
