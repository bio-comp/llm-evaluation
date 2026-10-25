"""Small auditable generation boundary for OpenAI-compatible HTTP endpoints.

This module deliberately implements only the portable single-turn text subset needed
for E-002 preflight.  A compatible transport is not evidence that two backends are
scientifically equivalent; effective requests and raw provider metadata are retained.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class CapabilityStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ACCEPTED_BUT_IGNORED = "accepted-but-ignored"
    SEMANTICS_DIFFER = "semantics-differ"
    UNKNOWN = "unknown"


class GenerationStatus(StrEnum):
    TRANSPORT_FAILURE = "transport_failure"
    PROVIDER_RUNTIME_FAILURE = "provider_runtime_failure"
    COMPLETED = "model_generation_completed"
    MODEL_GENERATION_COMPLETED = "model_generation_completed"  # compatibility alias


AttemptStatus = GenerationStatus


class EvaluationStatus(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    PASSED = "passed"
    FAILED = "failed"
    ORACLE_ERROR = "oracle_error"


class ReasoningWireMode(StrEnum):
    CHAT_REASONING_EFFORT = "chat_reasoning_effort"
    RESPONSES_REASONING_EFFORT = "responses_reasoning_effort"


@dataclass(frozen=True)
class ReasoningControl:
    """Exact reasoning control semantics and wire shape for one model condition."""

    wire_mode: ReasoningWireMode
    effort: str

    def __post_init__(self) -> None:
        if not isinstance(self.wire_mode, ReasoningWireMode):
            raise ValueError("wire_mode must be ReasoningWireMode")
        if not isinstance(self.effort, str) or not self.effort.strip():
            raise ValueError("reasoning effort must be a nonempty string")

    @property
    def capability_name(self) -> str:
        return self.wire_mode.value


class CapabilityError(ValueError):
    """Requested scientific control is not known to have portable semantics."""


class CredentialError(ValueError):
    """Configured credential source is missing."""


class TransportFailure(RuntimeError):
    """The request did not reach a successful provider HTTP response."""


class ProviderRuntimeFailure(RuntimeError):
    """The provider returned an HTTP or response-shape failure."""


def _deep_freeze(value: object) -> object:
    """Copy canonical JSON-like provenance into recursively immutable containers."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("provenance mapping keys must be strings")
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("provenance numbers must be finite")
        return value
    raise ValueError("provenance values must be canonical JSON-like values")


_SENSITIVE_PROVIDER_METADATA_KEYS = {
    "authorization",
    "proxy_authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "client_secret",
}


def _reject_sensitive_provider_metadata(
    value: object, forbidden_strings: tuple[str, ...] = ()
) -> None:
    """Refuse response JSON that could persist an echoed credential."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SENSITIVE_PROVIDER_METADATA_KEYS:
                raise ProviderRuntimeFailure("provider response contains sensitive metadata key")
            _reject_sensitive_provider_metadata(item, forbidden_strings)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive_provider_metadata(item, forbidden_strings)
    elif isinstance(value, str) and any(secret in value for secret in forbidden_strings):
        raise ProviderRuntimeFailure("provider response contains configured credential")


@dataclass(frozen=True)
class CapabilityProfile:
    statuses: Mapping[str, CapabilityStatus]

    def __post_init__(self) -> None:
        statuses = dict(self.statuses)
        if any(not isinstance(status, CapabilityStatus) for status in statuses.values()):
            raise ValueError("capability statuses must be CapabilityStatus values")
        object.__setattr__(self, "statuses", MappingProxyType(statuses))

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
    config_id: str | None = None
    wire_parameter_names: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id.strip() or not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("provider_id and model_id are required")
        if self.config_id is not None and (not isinstance(self.config_id, str) or not self.config_id.strip()):
            raise ValueError("config_id must be nonempty when supplied")
        allowed_wire_overrides = {
            "temperature", "top_p", "top_k", "seed", "stop", "max_output_tokens"
        }
        if any(key not in allowed_wire_overrides for key in self.wire_parameter_names):
            raise ValueError("wire_parameter_names contains an unknown internal parameter")
        if self.api_mode not in {"chat_completions", "responses"}:
            raise ValueError("api_mode must be chat_completions or responses")
        supplied_wire_names = tuple(self.wire_parameter_names.values())
        if any(not isinstance(value, str) or not value for value in supplied_wire_names):
            raise ValueError("wire parameter names must be nonempty strings")
        if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None for value in supplied_wire_names):
            raise ValueError("wire parameter name must be a simple identifier")
        object.__setattr__(
            self, "wire_parameter_names", MappingProxyType(dict(self.wire_parameter_names))
        )
        default_wire_names = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "seed": "seed",
            "stop": "stop",
            "max_output_tokens": (
                "max_tokens" if self.api_mode == "chat_completions" else "max_output_tokens"
            ),
        }
        effective_wire_names = [
            self.wire_parameter_names.get(key, value)
            for key, value in default_wire_names.items()
        ]
        if len(effective_wire_names) != len(set(effective_wire_names)):
            raise ValueError("wire parameter names must be unique")
        reserved_protocol_fields = (
            {"model", "messages", "reasoning_effort"}
            if self.api_mode == "chat_completions"
            else {"model", "input", "reasoning"}
        )
        if any(name in reserved_protocol_fields for name in effective_wire_names):
            raise ValueError("wire parameter name collides with reserved protocol field")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not embed credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if (
            self.credential_env is not None
            and parsed.scheme != "https"
            and hostname not in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("credential requires HTTPS except for loopback endpoints")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def effective_config_id(self) -> str:
        """Registry identity; provider identity alone is not a model configuration."""
        return self.provider_id if self.config_id is None else self.config_id


@dataclass(frozen=True)
class BackendRegistry:
    """Tiny explicit lookup layer; provider behavior still lives in capability profiles."""

    backends: tuple[BackendConfig, ...]

    def __post_init__(self) -> None:
        ids = self.config_ids
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate config_id in backend registry")

    @property
    def config_ids(self) -> tuple[str, ...]:
        return tuple(backend.effective_config_id for backend in self.backends)

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(backend.provider_id for backend in self.backends))

    def __getitem__(self, config_id: str) -> BackendConfig:
        for backend in self.backends:
            if backend.effective_config_id == config_id:
                return backend
        raise KeyError(config_id)


@dataclass(frozen=True)
class ModelCondition:
    """Scientific condition identity kept separate from endpoint configuration."""

    condition_id: str
    weights_id: str
    artifact_sha256: str | None
    quantization: str | None
    provider_id: str
    backend_config_id: str
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
    reasoning_control: ReasoningControl | None = None

    def __post_init__(self) -> None:
        identity_fields = (
            self.condition_id, self.weights_id, self.provider_id, self.backend_config_id,
            self.backend_name, self.backend_version, self.api_mode, self.endpoint_model_id,
            self.prompt_template_id, self.seed_behavior,
        )
        if any(not isinstance(value, str) or not value.strip() for value in identity_fields):
            raise ValueError("model condition identity fields must be nonempty strings")
        if self.api_mode not in {"chat_completions", "responses"}:
            raise ValueError("model condition API mode must be chat_completions or responses")
        if not isinstance(self.runtime_environment, Mapping):
            raise ValueError("runtime_environment must be a mapping")
        if self.artifact_metadata is not None and not isinstance(self.artifact_metadata, Mapping):
            raise ValueError("artifact_metadata must be a mapping when supplied")
        if self.artifact_sha256 is not None:
            digest = self.artifact_sha256.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("artifact_sha256 must be a 64-character hexadecimal digest")
            object.__setattr__(self, "artifact_sha256", digest)
        object.__setattr__(self, "runtime_environment", _deep_freeze(self.runtime_environment))
        if self.artifact_metadata is not None:
            object.__setattr__(self, "artifact_metadata", _deep_freeze(self.artifact_metadata))
        if self.reasoning_control is not None:
            if not isinstance(self.reasoning_control, ReasoningControl):
                raise ValueError("reasoning_control must be ReasoningControl")
            expected_mode = (
                ReasoningWireMode.CHAT_REASONING_EFFORT
                if self.api_mode == "chat_completions"
                else ReasoningWireMode.RESPONSES_REASONING_EFFORT
            )
            if self.reasoning_control.wire_mode is not expected_mode:
                raise ValueError("reasoning wire mode does not match condition API mode")


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
        if not isinstance(self.condition_id, str) or not self.condition_id.strip():
            raise ValueError("condition_id is required")
        if not isinstance(self.messages, tuple) or not self.messages:
            raise ValueError("messages must be a nonempty tuple")
        for item in self.messages:
            if (
                not isinstance(item, tuple) or len(item) != 2
                or item[0] not in {"system", "user", "assistant"}
                or not isinstance(item[1], str)
            ):
                raise ValueError("messages must contain text system/user/assistant entries")
        if self.temperature is not None:
            if (
                isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(float(self.temperature))
                or self.temperature < 0
            ):
                raise ValueError("temperature must be a finite nonnegative number")
        if self.top_p is not None:
            if (
                isinstance(self.top_p, bool)
                or not isinstance(self.top_p, (int, float))
                or not math.isfinite(float(self.top_p))
                or not 0 < self.top_p <= 1
            ):
                raise ValueError("top_p must be finite and in (0, 1]")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1
        ):
            raise ValueError("top_k must be a positive integer")
        if self.max_output_tokens is not None and (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens < 1
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("seed must be an integer")
        if self.stop is not None and (
            not isinstance(self.stop, tuple)
            or not self.stop
            or any(not isinstance(value, str) or not value for value in self.stop)
        ):
            raise ValueError("stop must be a nonempty tuple of nonempty strings")


@dataclass(frozen=True)
class EffectiveRequest:
    endpoint: str
    api_mode: str
    serialized_body: str
    body_sha256: str
    prompt_packet_sha256: str
    http_method: str
    backend_config_id: str
    condition_id: str
    condition_sha256: str
    request_identity_sha256: str


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    backend_model_id: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    provider_request_id: str | None
    raw_provider_metadata: Mapping[str, object]
    reported_model_matches_condition: bool | None
    elapsed_seconds: float
    effective_request: EffectiveRequest
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        if self.termination_reason is not None and not isinstance(self.termination_reason, str):
            raise ValueError("termination_reason must be str or None")
        if self.reported_model_matches_condition is not None and not isinstance(
            self.reported_model_matches_condition, bool
        ):
            raise ValueError("reported_model_matches_condition must be bool or None")
        try:
            frozen = _deep_freeze(self.raw_provider_metadata)
        except ValueError as exc:
            raise ProviderRuntimeFailure("provider metadata is not canonical JSON-like data") from exc
        object.__setattr__(self, "raw_provider_metadata", frozen)


@dataclass(frozen=True)
class GenerationAttempt:
    """One technical generation attempt plus a distinct downstream evaluation state."""

    attempt_index: int
    retry_of_attempt: int | None
    retry_count: int
    generation_status: GenerationStatus
    result: GenerationResult | None
    failure_detail: str | None
    evaluation_status: EvaluationStatus = EvaluationStatus.NOT_EVALUATED
    contributes_to_sample: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.attempt_index, bool)
            or not isinstance(self.attempt_index, int)
            or self.attempt_index < 1
        ):
            raise ValueError("attempt_index must be a positive integer")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or self.retry_count < 0
        ):
            raise ValueError("retry_count must be a nonnegative integer")
        if self.retry_of_attempt is not None and (
            isinstance(self.retry_of_attempt, bool) or not isinstance(self.retry_of_attempt, int)
        ):
            raise ValueError("retry_of_attempt must be an integer when supplied")
        if not isinstance(self.generation_status, GenerationStatus):
            raise ValueError("generation_status must be GenerationStatus")
        if not isinstance(self.evaluation_status, EvaluationStatus):
            raise ValueError("evaluation_status must be EvaluationStatus")
        if not isinstance(self.contributes_to_sample, bool):
            raise ValueError("contributes_to_sample must be bool")
        if self.failure_detail is not None and not isinstance(self.failure_detail, str):
            raise ValueError("failure_detail must be a string when supplied")
        if self.result is not None and not isinstance(self.result, GenerationResult):
            raise ValueError("result must be GenerationResult when supplied")
        if self.retry_of_attempt is not None and not 1 <= self.retry_of_attempt < self.attempt_index:
            raise ValueError("retry_of_attempt must reference an earlier attempt")
        if self.retry_of_attempt is None and self.retry_count != 0:
            raise ValueError("non-retry attempts require retry_count=0")
        if self.retry_of_attempt is not None and self.retry_count < 1:
            raise ValueError("retry attempts require positive retry_count")
        completed = self.generation_status is GenerationStatus.COMPLETED
        if completed and self.result is None:
            raise ValueError("completed generation requires a result")
        if not completed and self.result is not None:
            raise ValueError("failed generation cannot carry a generation result")
        if not completed and self.evaluation_status is not EvaluationStatus.NOT_EVALUATED:
            raise ValueError("failed generation cannot have an oracle/evaluation outcome")
        if not completed and self.contributes_to_sample:
            raise ValueError("failed generation cannot contribute to the statistical sample")
        if self.evaluation_status is EvaluationStatus.ORACLE_ERROR and self.contributes_to_sample:
            raise ValueError("oracle error cannot contribute to the statistical sample")
        if self.evaluation_status is EvaluationStatus.NOT_EVALUATED and self.contributes_to_sample:
            raise ValueError("unevaluated generation cannot contribute to the statistical sample")

    @property
    def status(self) -> GenerationStatus:
        """Compatibility name for the technical generation status only."""
        return self.generation_status


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def _condition_packet(condition: ModelCondition) -> dict[str, object]:
    reasoning = condition.reasoning_control
    return {
        "condition_id": condition.condition_id,
        "weights_id": condition.weights_id,
        "artifact_sha256": condition.artifact_sha256,
        "quantization": condition.quantization,
        "provider_id": condition.provider_id,
        "backend_config_id": condition.backend_config_id,
        "backend_name": condition.backend_name,
        "backend_version": condition.backend_version,
        "api_mode": condition.api_mode,
        "endpoint_model_id": condition.endpoint_model_id,
        "prompt_template_id": condition.prompt_template_id,
        "seed_behavior": condition.seed_behavior,
        "runtime_environment": _json_ready(condition.runtime_environment),
        "artifact_filename": condition.artifact_filename,
        "tokenizer_id": condition.tokenizer_id,
        "artifact_metadata": _json_ready(condition.artifact_metadata),
        "reasoning_control": None if reasoning is None else {
            "wire_mode": reasoning.wire_mode.value,
            "effort": reasoning.effort,
        },
    }


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

    def __init__(
        self,
        config: BackendConfig,
        *,
        condition: ModelCondition | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        self.config = config
        self.condition = condition
        self.timeout_seconds = float(timeout_seconds)
        if condition is not None:
            if condition.provider_id != config.provider_id:
                raise ValueError("condition provider does not match backend provider")
            if condition.backend_config_id != config.effective_config_id:
                raise ValueError("condition backend config does not match selected backend config")
            if condition.endpoint_model_id != config.model_id:
                raise ValueError("condition model does not match backend model")
            if condition.api_mode != config.api_mode:
                raise ValueError("condition API mode does not match backend API mode")
            reasoning_capability = (
                ReasoningWireMode.CHAT_REASONING_EFFORT.value
                if config.api_mode == "chat_completions"
                else ReasoningWireMode.RESPONSES_REASONING_EFFORT.value
            )
            if (
                reasoning_capability in config.capability_profile.statuses
                and config.capability_profile.statuses[reasoning_capability]
                is not CapabilityStatus.UNSUPPORTED
                and condition.reasoning_control is None
            ):
                raise ValueError("reasoning-capable backend requires explicit reasoning condition control")

    def _require_bound_request(self, request: GenerationRequest) -> ModelCondition:
        if self.condition is None:
            raise ValueError("generation requires a bound ModelCondition")
        if request.condition_id != self.condition.condition_id:
            raise ValueError("request condition does not match bound model condition")
        return self.condition

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
        for capability, value, default_wire_name in controls:
            if value is not None:
                profile.require(capability)
                wire_name = self.config.wire_parameter_names.get(capability, default_wire_name)
                body[wire_name] = value
        if request.max_output_tokens is not None:
            profile.require("max_output_tokens")
            default_wire_name = (
                "max_tokens" if self.config.api_mode == "chat_completions" else "max_output_tokens"
            )
            wire_name = self.config.wire_parameter_names.get(
                "max_output_tokens", default_wire_name
            )
            body[wire_name] = request.max_output_tokens
        if self.condition is not None and self.condition.reasoning_control is not None:
            reasoning = self.condition.reasoning_control
            profile.require(reasoning.capability_name)
            if reasoning.wire_mode is ReasoningWireMode.CHAT_REASONING_EFFORT:
                body["reasoning_effort"] = reasoning.effort
            else:
                body["reasoning"] = {"effort": reasoning.effort}
        endpoint = "/chat/completions" if self.config.api_mode == "chat_completions" else "/responses"
        return endpoint, body

    def _effective(self, request: GenerationRequest, endpoint: str, body: dict[str, object]) -> EffectiveRequest:
        if self.condition is None:
            raise ValueError("effective request requires a bound ModelCondition")
        serialized = canonical_json(body)
        prompt = canonical_json(_prompt_packet(request))
        condition_json = canonical_json(_condition_packet(self.condition))
        endpoint_url = f"{self.config.base_url}{endpoint}"
        body_sha256 = _sha256_text(serialized)
        prompt_packet_sha256 = _sha256_text(prompt)
        condition_sha256 = _sha256_text(condition_json)
        identity = canonical_json({
            "http_method": "POST",
            "endpoint": endpoint_url,
            "api_mode": self.config.api_mode,
            "backend_config_id": self.config.effective_config_id,
            "condition_sha256": condition_sha256,
            "body_sha256": body_sha256,
            "prompt_packet_sha256": prompt_packet_sha256,
            "timeout_seconds": self.timeout_seconds,
        })
        return EffectiveRequest(
            endpoint=endpoint_url,
            api_mode=self.config.api_mode,
            serialized_body=serialized,
            body_sha256=body_sha256,
            prompt_packet_sha256=prompt_packet_sha256,
            http_method="POST",
            backend_config_id=self.config.effective_config_id,
            condition_id=self.condition.condition_id,
            condition_sha256=condition_sha256,
            request_identity_sha256=_sha256_text(identity),
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        condition = self._require_bound_request(request)
        endpoint, body = self._body(request)
        effective = self._effective(request, endpoint, body)
        wire = effective.serialized_body.encode("utf-8")
        headers = self._headers()
        authorization = headers.get("Authorization")
        forbidden_response_strings: tuple[str, ...] = ()
        if authorization is not None:
            forbidden_response_strings = (authorization, authorization.removeprefix("Bearer "))
        http_request = urllib.request.Request(
            effective.endpoint, data=wire, headers=headers, method="POST"
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                header_request_id = response.headers.get("X-Request-ID")
        except urllib.error.HTTPError as exc:
            # Do not surface provider-controlled error text or bodies here: a hostile
            # or buggy endpoint can echo an Authorization value or other credential.
            raise ProviderRuntimeFailure(f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportFailure(str(exc)) from exc
        elapsed = time.perf_counter() - started
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderRuntimeFailure("provider returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderRuntimeFailure("provider response must be a JSON object")
        _reject_sensitive_provider_metadata(header_request_id, forbidden_response_strings)
        _reject_sensitive_provider_metadata(payload, forbidden_response_strings)
        if self.config.api_mode == "chat_completions":
            parsed = self._parse_chat(payload)
        else:
            parsed = self._parse_responses(payload)
        body_request_id = payload.get("request_id")
        if not isinstance(body_request_id, str):
            body_request_id = None
        reported_model = parsed["backend_model_id"]
        model_matches = (
            None if reported_model is None else reported_model == condition.endpoint_model_id
        )
        return GenerationResult(
            **parsed,
            # OpenAI-style response-body `id` identifies the response object, not
            # the HTTP request. Preserve it in raw_provider_metadata. Normalize an
            # explicitly labeled body request_id first, then the request-ID header.
            provider_request_id=body_request_id or header_request_id,
            raw_provider_metadata=payload,
            reported_model_matches_condition=model_matches,
            elapsed_seconds=elapsed,
            effective_request=effective,
        )

    @staticmethod
    def _validated_usage(
        payload: dict[str, Any], input_key: str, output_key: str
    ) -> tuple[int | None, int | None, int | None]:
        raw_usage = payload.get("usage")
        if raw_usage is None:
            return None, None, None
        if not isinstance(raw_usage, dict):
            raise ProviderRuntimeFailure("provider usage must be a JSON object")

        def value(key: str) -> int | None:
            raw = raw_usage.get(key)
            if raw is None:
                return None
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ProviderRuntimeFailure("provider token usage must be nonnegative integers")
            return raw

        prompt_tokens = value(input_key)
        completion_tokens = value(output_key)
        total_tokens = value("total_tokens")
        components = [v for v in (prompt_tokens, completion_tokens) if v is not None]
        minimum_total = sum(components)
        if total_tokens is not None and components and total_tokens < minimum_total:
            raise ProviderRuntimeFailure("provider total token usage is inconsistent")
        return prompt_tokens, completion_tokens, total_tokens

    @staticmethod
    def _parse_chat(payload: dict[str, Any]) -> dict[str, Any]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ProviderRuntimeFailure("Chat Completions requires exactly one choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderRuntimeFailure("invalid Chat Completions response shape")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ProviderRuntimeFailure("Chat Completions finish_reason is not a string")
        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            raise ProviderRuntimeFailure("Chat Completions model id is not a string")
        prompt_tokens, completion_tokens, total_tokens = OpenAICompatibleTransport._validated_usage(
            payload, "prompt_tokens", "completion_tokens"
        )
        return {
            "raw_output": message["content"],
            "backend_model_id": model,
            "finish_reason": finish_reason,
            "termination_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _parse_responses(payload: dict[str, Any]) -> dict[str, Any]:
        output = payload.get("output")
        if not isinstance(output, list):
            raise ProviderRuntimeFailure("Responses payload has no output list")
        texts: list[str] = []
        message_count = 0
        for item in output:
            if not isinstance(item, dict):
                raise ProviderRuntimeFailure("Responses output items must be JSON objects")
            item_type = item.get("type")
            if item_type == "reasoning":
                continue
            if item_type != "message":
                raise ProviderRuntimeFailure("Responses output contains an unsupported item type")
            message_count += 1
            if message_count > 1:
                raise ProviderRuntimeFailure("Responses payload contains multiple message outputs")
            content = item.get("content")
            if not isinstance(content, list):
                raise ProviderRuntimeFailure("Responses message content is not a list")
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "output_text":
                    raise ProviderRuntimeFailure("Responses message contains unsupported content")
                if not isinstance(part.get("text"), str):
                    raise ProviderRuntimeFailure("Responses output_text is not a string")
                texts.append(part["text"])
        status = payload.get("status")
        if status is not None and not isinstance(status, str):
            raise ProviderRuntimeFailure("Responses status is not a string")
        incomplete_details = payload.get("incomplete_details")
        termination_reason: str | None = None
        if incomplete_details is not None:
            if not isinstance(incomplete_details, dict):
                raise ProviderRuntimeFailure("Responses incomplete_details is not a JSON object")
            reason = incomplete_details.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ProviderRuntimeFailure("Responses incomplete_details.reason is not a string")
            if status != "incomplete":
                raise ProviderRuntimeFailure(
                    "Responses incomplete_details is inconsistent with response status"
                )
            termination_reason = reason
        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            raise ProviderRuntimeFailure("Responses model id is not a string")
        prompt_tokens, completion_tokens, total_tokens = OpenAICompatibleTransport._validated_usage(
            payload, "input_tokens", "output_tokens"
        )
        return {
            "raw_output": "".join(texts),
            "backend_model_id": model,
            "finish_reason": status,
            "termination_reason": termination_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def list_models(self) -> tuple[str, ...]:
        self.config.capability_profile.require("models")
        request = urllib.request.Request(
            f"{self.config.base_url}/models", headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise ProviderRuntimeFailure(f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportFailure(str(exc)) from exc
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderRuntimeFailure("provider returned invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ProviderRuntimeFailure("invalid model-list response")
        values: list[str] = []
        for item in payload["data"]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ProviderRuntimeFailure("invalid model-list response")
            values.append(item["id"])
        return tuple(values)


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

    A caller may schedule a retry, but must link it to the earlier attempt. Statistical
    sample inclusion is attached only after downstream evaluation succeeds.
    """
    if transport.config.config_id is None:
        raise ValueError("scientific attempt requires explicit backend config_id")
    if contributes_to_sample:
        raise ValueError("statistical sample inclusion is decided only after evaluation")
    try:
        result = transport.generate(request)
    except TransportFailure as exc:
        return GenerationAttempt(
            attempt_index=attempt_index,
            retry_of_attempt=retry_of_attempt,
            retry_count=retry_count,
            generation_status=GenerationStatus.TRANSPORT_FAILURE,
            result=None,
            failure_detail=str(exc),
        )
    except ProviderRuntimeFailure as exc:
        return GenerationAttempt(
            attempt_index=attempt_index,
            retry_of_attempt=retry_of_attempt,
            retry_count=retry_count,
            generation_status=GenerationStatus.PROVIDER_RUNTIME_FAILURE,
            result=None,
            failure_detail=str(exc),
        )
    return GenerationAttempt(
        attempt_index=attempt_index,
        retry_of_attempt=retry_of_attempt,
        retry_count=retry_count,
        generation_status=GenerationStatus.COMPLETED,
        result=result,
        failure_detail=None,
    )


def record_evaluation(
    attempt: GenerationAttempt,
    evaluation_status: EvaluationStatus,
    *,
    contributes_to_sample: bool = False,
) -> GenerationAttempt:
    """Attach an oracle/evaluation outcome without discarding the generation result."""
    if attempt.generation_status is not GenerationStatus.COMPLETED:
        raise ValueError("only a completed generation can be evaluated")
    if attempt.evaluation_status is not EvaluationStatus.NOT_EVALUATED:
        raise ValueError("attempt is already evaluated")
    if evaluation_status is EvaluationStatus.NOT_EVALUATED:
        raise ValueError("record_evaluation requires a completed evaluation status")
    if contributes_to_sample and (
        attempt.result is None or attempt.result.reported_model_matches_condition is not True
    ):
        raise ValueError("statistical sample requires a matching provider-reported model id")
    return replace(
        attempt,
        evaluation_status=evaluation_status,
        contributes_to_sample=contributes_to_sample,
    )
