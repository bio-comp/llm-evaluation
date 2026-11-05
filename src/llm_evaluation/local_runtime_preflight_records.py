"""Infrastructure qualification record; deliberately outside the runtime."""

from typing import Literal
from pathlib import Path
from .runtime.evidence import GenerationObservation
from .runtime.timing import RequestTiming
from pydantic import Field, model_validator
from .runtime.contracts import ClosedRecord
from .runtime.llamacpp.qualification import (
    ExecutableIdentity,
    ArtifactIdentity,
    ExternalConflictStatus,
)


class PreflightBudget(ClosedRecord):
    context_tokens: Literal[2048] = 2048
    output_tokens: Literal[128] = 128
    startup_seconds: Literal[45] = 45
    request_seconds: Literal[30] = 30
    whole_seconds: Literal[240] = 240
    server_launches: int = Field(default=0, ge=0, le=3)
    generation_submissions: int = Field(default=0, ge=0, le=6)


class LegacyGateResults(ClosedRecord):
    shared_blocks_formal: bool | None = None
    formal_blocks_shared: bool | None = None
    no_key_rejected: bool | None = None
    wrong_key_rejected: bool | None = None
    qualified_model_slot: bool | None = None
    webui_disabled: bool | None = None
    tiny_generation: bool | None = None
    structured_generation: bool | None = None
    concurrent_requests: bool | None = None
    launcher_death_lease: bool | None = None
    cleanup: bool | None = None


class GateResults(LegacyGateResults):
    listener_loopback: bool | None = None
    agent_tools_disabled: bool | None = None


class LegacyPreflightRecord(ClosedRecord):
    record_type: Literal["local_llamacpp_runtime_preflight"] = (
        "local_llamacpp_runtime_preflight"
    )
    evidence_class: Literal["infrastructure_qualification_not_e002"]
    software_commit: str
    package_source_sha256: str
    executable: ExecutableIdentity | None
    artifact: ArtifactIdentity | None
    resource: Literal["llama-device:MTL0"] = "llama-device:MTL0"
    lease_scope: Literal["cooperating_same_namespace"] = "cooperating_same_namespace"
    external_conflict_status: ExternalConflictStatus
    status: Literal["passed", "failed", "incomplete"]
    failure_kind: Literal[
        "none", "conflict", "identity", "runtime", "deadline", "qualification"
    ]
    budget: PreflightBudget
    gates: LegacyGateResults
    slot_poll_interval_seconds: float = Field(default=0.02, gt=0)
    slot_observation_seconds: float = Field(default=0.0, ge=0)
    slot_poll_count: int = Field(default=0, ge=0)
    request_timings: tuple[RequestTiming, ...] = ()
    structured_domain_sha256: str | None = None
    structured_effective_sha256: str | None = None
    wall_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def complete_acceptance(self) -> "LegacyPreflightRecord":
        if self.status == "passed" and (
            self.executable is None
            or self.artifact is None
            or self.failure_kind != "none"
            or not all(v is True for v in self.gates.model_dump().values())
        ):
            raise ValueError(
                "passed preflight requires complete identities and positive gates"
            )
        return self


class PreflightRecord(LegacyPreflightRecord):
    """Version 2 retains stage and cleanup separately from gate progress."""

    gates: GateResults

    schema_version: Literal[2] = 2
    failure_stage: Literal[
        "identity", "external_conflict", "shared_exclusion", "formal_launch",
        "formal_exclusion", "authentication", "qualification",
        "tiny_generation", "structured_generation", "concurrency",
        "launcher_death", "cleanup", "deadline",
    ] | None = None
    failure_code: Literal[
        "http_status", "invalid_json", "invalid_response",
        "structured_parse", "structured_validation", "model_binding",
        "output_exhausted", "empty_visible_output", "deadline",
        "resource_busy", "startup_failure", "unknown_safe_failure",
    ] | None = None
    cleanup_status: Literal["verified", "failed", "unverified"] = "unverified"
    generation_summaries: tuple["GenerationProbeSummary", ...] = ()
    slot_samples: tuple["SlotSample", ...] = Field(default=(), max_length=4000)
    chat_template_sha256: str | None = None
    chat_template_caps: tuple[tuple[str, bool], ...] | None = None
    build_info_sha256: str | None = None
    qualified_condition_sha256: str | None = None
    reasoning_profile: Literal["auto", "off"] | None = None

    @model_validator(mode="after")
    def v2_consistency(self) -> "PreflightRecord":
        if self.status == "passed" and (
            self.cleanup_status != "verified"
            or self.failure_stage is not None
            or self.failure_code is not None
            or not self.slot_samples
        ):
            raise ValueError("passed v2 preflight requires verified cleanup")
        return self


def read_preflight_record(path: "Path") -> LegacyPreflightRecord | PreflightRecord:
    """Read historical unversioned evidence without modifying its bytes."""
    raw = Path(path).read_bytes()
    import json

    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("preflight object required")
    if "schema_version" not in obj:
        return LegacyPreflightRecord.model_validate_json(raw)
    return PreflightRecord.model_validate_json(raw)


class GenerationProbeSummary(ClosedRecord):
    """Closed, bounded counts from a credential-screened technical response."""

    reported_model_matches_condition: bool | None
    finish_reason: Literal["stop", "length", "other", "unknown"]
    termination_reason: Literal["stop", "length", "other", "unknown"]
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    visible_output_bytes: int = Field(ge=0, le=2 * 1024 * 1024)
    reasoning_present: bool
    reasoning_bytes: int = Field(ge=0, le=2 * 1024 * 1024)

    @model_validator(mode="after")
    def reasoning_presence_consistent(self) -> "GenerationProbeSummary":
        if self.reasoning_bytes > 0 and not self.reasoning_present:
            raise ValueError("positive reasoning bytes require reasoning presence")
        return self

    def within_reported_budget(self, *, context_tokens: int, output_tokens: int) -> bool:
        """Only reported token counts can establish a measured budget violation."""
        prompt, completion, total = self.prompt_tokens, self.completion_tokens, self.total_tokens
        return (
            (prompt is None or prompt <= context_tokens)
            and (completion is None or completion <= output_tokens)
            and (total is None or total <= context_tokens)
            and (prompt is None or completion is None or prompt + completion <= context_tokens)
        )


def summarize_generation(observation: "GenerationObservation", evidence: bytes) -> GenerationProbeSummary:
    from .runtime.codec import parse_json_unique
    payload = parse_json_unique(evidence)
    if not isinstance(payload, dict):
        raise ValueError("provider response object required")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("one chat choice required")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("chat message object required")
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("invalid reasoning diagnostic")
    def closed(value: str | None) -> Literal["stop", "length", "other", "unknown"]:
        if value is None:
            return "unknown"
        return value if value in ("stop", "length") else "other"  # type: ignore[return-value]

    return GenerationProbeSummary(
        reported_model_matches_condition=observation.reported_model_matches_condition,
        finish_reason=closed(observation.finish_reason),
        termination_reason=closed(observation.termination_reason),
        prompt_tokens=observation.prompt_tokens,
        completion_tokens=observation.completion_tokens,
        total_tokens=observation.total_tokens,
        visible_output_bytes=len(observation.raw_output.encode("utf-8")),
        reasoning_present=reasoning is not None,
        reasoning_bytes=len(reasoning.encode("utf-8")) if reasoning is not None else 0,
    )

class StructuredDiagnosticRecord(ClosedRecord):
    record_type: Literal["local_llamacpp_structured_diagnostic"] = "local_llamacpp_structured_diagnostic"
    schema_version: Literal[1] = 1
    evidence_class: Literal["infrastructure_qualification_not_e002"] = "infrastructure_qualification_not_e002"
    software_commit: str
    package_source_sha256: str
    executable: ExecutableIdentity | None
    artifact: ArtifactIdentity | None
    status: Literal["passed", "failed", "incomplete"]
    failure_stage: Literal["identity", "external_conflict", "launch", "qualification", "structured_generation", "cleanup", "deadline"] | None
    failure_code: Literal["http_status", "invalid_json", "invalid_response", "structured_parse", "structured_validation", "model_binding", "output_exhausted", "empty_visible_output", "deadline", "resource_busy", "startup_failure", "unknown_safe_failure"] | None
    cleanup_status: Literal["verified", "failed", "unverified"]
    server_launches: int = Field(ge=0, le=1)
    generation_submissions: int = Field(ge=0, le=1)
    reasoning_profile: Literal["auto", "off"]
    context_tokens: Literal[2048] = 2048
    output_tokens: Literal[128] = 128
    startup_seconds: Literal[45] = 45
    request_seconds: Literal[30] = 30
    whole_seconds: Literal[240] = 240
    chat_template_sha256: str | None = None
    qualified_condition_sha256: str | None = None
    structured_domain_sha256: str | None = None
    structured_effective_sha256: str | None = None
    structured_generation: bool | None = None
    summary: GenerationProbeSummary | None = None
    remote_completion_unknown: bool = False
    wall_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def passed_requires_evidence(self) -> "StructuredDiagnosticRecord":
        if self.status == "passed" and (
            self.executable is None
            or self.artifact is None
            or self.structured_domain_sha256 is None
            or self.structured_effective_sha256 is None
            or self.structured_generation is not True
            or self.summary is None
            or self.chat_template_sha256 is None
            or self.qualified_condition_sha256 is None
            or self.cleanup_status != "verified"
            or self.server_launches != 1
            or self.generation_submissions != 1
            or self.failure_stage is not None
            or self.failure_code is not None
            or self.remote_completion_unknown
            or self.summary.reported_model_matches_condition is not True
            or self.summary.finish_reason != "stop"
            or self.summary.termination_reason != "stop"
            or self.summary.visible_output_bytes == 0
            or not self.summary.within_reported_budget(
                context_tokens=self.context_tokens, output_tokens=self.output_tokens
            )
            or (self.reasoning_profile == "off" and self.summary.reasoning_bytes > 0)
        ):
            raise ValueError("passed structured diagnostic requires measured success and cleanup")
        return self


class SlotSample(ClosedRecord):
    monotonic_offset_seconds: float = Field(ge=0)
    slot_count: int = Field(ge=0, le=1)
    processing_count: int = Field(ge=0, le=1)

PreflightRecord.model_rebuild()
