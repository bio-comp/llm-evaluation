"""Closed, strict, immutable records at the runtime trust boundary."""

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from .codec import freeze, json_ready

Nonblank = Annotated[str, Field(min_length=1, pattern=r"\S")]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
ApiMode = Literal["chat_completions", "responses"]


class ClosedRecord(BaseModel):
    model_config = ConfigDict(
        strict=True, frozen=True, extra="forbid", arbitrary_types_allowed=True
    )


class CapabilityStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ACCEPTED_BUT_IGNORED = "accepted-but-ignored"
    SEMANTICS_DIFFER = "semantics-differ"
    UNKNOWN = "unknown"


class ReasoningWireMode(StrEnum):
    CHAT_REASONING_EFFORT = "chat_reasoning_effort"
    RESPONSES_REASONING_EFFORT = "responses_reasoning_effort"


class ReasoningControl(ClosedRecord):
    wire_mode: ReasoningWireMode
    effort: Nonblank

    @property
    def capability_name(self) -> str:
        return self.wire_mode.value


class CapabilityProfile(ClosedRecord):
    statuses: Mapping[str, CapabilityStatus]

    @field_validator("statuses")
    @classmethod
    def immutable(
        cls, value: Mapping[str, CapabilityStatus]
    ) -> Mapping[str, CapabilityStatus]:
        from types import MappingProxyType

        return MappingProxyType(dict(value))

    @field_serializer("statuses")
    def serialize(self, value: Mapping[str, CapabilityStatus]) -> dict[str, str]:
        return {k: v.value for k, v in value.items()}

    def require(self, capability: str) -> None:
        if (
            self.statuses.get(capability, CapabilityStatus.UNKNOWN)
            is not CapabilityStatus.SUPPORTED
        ):
            raise ValueError("capability not supported: " + capability)


class ScientificEndpointIdentity(ClosedRecord):
    value: Nonblank


class OperationalEndpoint(ClosedRecord):
    base_url: Nonblank

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        p = urlsplit(value)
        if (
            p.scheme not in ("http", "https")
            or not p.hostname
            or p.username is not None
            or p.password is not None
            or p.query
            or p.fragment
        ):
            raise ValueError(
                "absolute HTTP(S) URL without credentials/query/fragment required"
            )
        _ = p.port
        return value.rstrip("/")


class HttpTimeouts(ClosedRecord):
    connect: Positive = 10.0
    read: Positive = 120.0
    write: Positive = 10.0
    pool: Positive = 10.0
    total: Positive = 180.0

    @field_validator("*", mode="before")
    @classmethod
    def reject_bool(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("boolean is not a timeout")
        return value


class BackendConfig(ClosedRecord):
    provider_id: Nonblank
    model_id: Nonblank
    api_mode: ApiMode
    capability_profile: CapabilityProfile
    config_id: Nonblank
    wire_parameter_names: Mapping[str, str] = Field(default_factory=dict)

    @field_validator("provider_id")
    @classmethod
    def reject_retired_provider(cls, value: str) -> str:
        if value.strip().lower() == "ollama":
            raise ValueError("Ollama is retired for new backend configurations")
        return value

    @model_validator(mode="after")
    def validate_wire(self) -> "BackendConfig":
        names = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "seed": "seed",
            "stop": "stop",
            "max_output_tokens": "max_tokens"
            if self.api_mode == "chat_completions"
            else "max_output_tokens",
        }
        reserved = {
            "model",
            "messages",
            "input",
            "reasoning",
            "reasoning_effort",
            "response_format",
            "stream",
        }
        if set(self.wire_parameter_names) - set(names):
            raise ValueError("unknown wire control")
        names.update(self.wire_parameter_names)
        if len(set(names.values())) != len(names) or any(
            n in reserved or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) is None
            for n in names.values()
        ):
            raise ValueError("wire control collision/invalid name")
        object.__setattr__(
            self, "wire_parameter_names", freeze(self.wire_parameter_names)
        )
        return self

    @field_serializer("wire_parameter_names")
    def serialize_wire(self, value: Mapping[str, str]) -> object:
        return dict(value)


class ModelCondition(ClosedRecord):
    condition_id: Nonblank
    weights_id: Nonblank
    artifact_sha256: Annotated[str, Field(pattern="^[0-9a-f]{64}$")] | None = None
    quantization: str | None = None
    provider_id: Nonblank
    backend_config_id: Nonblank
    backend_name: Nonblank
    backend_version: Nonblank
    api_mode: ApiMode
    endpoint_model_id: Nonblank
    prompt_template_id: Nonblank
    seed_behavior: Nonblank
    runtime_environment: Mapping[str, Any]
    artifact_filename: str | None = None
    tokenizer_id: str | None = None
    artifact_metadata: Mapping[str, Any] | None = None
    reasoning_control: ReasoningControl | None = None

    @field_validator("runtime_environment", "artifact_metadata")
    @classmethod
    def immutable(cls, value: Any) -> Any:
        return None if value is None else freeze(value)

    @field_serializer("runtime_environment", "artifact_metadata")
    def serialize(self, value: Any) -> Any:
        return json_ready(value)

    @model_validator(mode="after")
    def reasoning_matches(self) -> "ModelCondition":
        expected = (
            ReasoningWireMode.CHAT_REASONING_EFFORT
            if self.api_mode == "chat_completions"
            else ReasoningWireMode.RESPONSES_REASONING_EFFORT
        )
        if (
            self.reasoning_control is not None
            and self.reasoning_control.wire_mode is not expected
        ):
            raise ValueError("reasoning/API mismatch")
        return self


class GenerationRequest(ClosedRecord):
    condition_id: Nonblank
    messages: Annotated[
        tuple[tuple[Literal["system", "user", "assistant"], str], ...],
        Field(min_length=1),
    ]
    temperature: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    top_p: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] | None = None
    top_k: Annotated[int, Field(gt=0)] | None = None
    max_output_tokens: Annotated[int, Field(gt=0)] | None = None
    stop: (
        Annotated[tuple[Annotated[str, Field(min_length=1)], ...], Field(min_length=1)]
        | None
    ) = None
    seed: int | None = None

    @field_validator("temperature", "top_p", mode="before")
    @classmethod
    def reject_bool(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("boolean sampler")
        return value


class RuntimeBinding(ClosedRecord):
    backend: BackendConfig
    condition: ModelCondition
    scientific_endpoint: ScientificEndpointIdentity
    timeouts: HttpTimeouts = Field(default_factory=HttpTimeouts)

    @model_validator(mode="after")
    def bound(self) -> "RuntimeBinding":
        b, c = self.backend, self.condition
        if (b.provider_id, b.config_id, b.model_id, b.api_mode) != (
            c.provider_id,
            c.backend_config_id,
            c.endpoint_model_id,
            c.api_mode,
        ):
            raise ValueError("backend/condition mismatch")
        key = (
            "chat_reasoning_effort"
            if b.api_mode == "chat_completions"
            else "responses_reasoning_effort"
        )
        if (
            key in b.capability_profile.statuses
            and b.capability_profile.statuses[key] is not CapabilityStatus.UNSUPPORTED
            and c.reasoning_control is None
        ):
            raise ValueError("explicit reasoning control required")
        return self
