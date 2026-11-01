"""Permitted raw bytes are separate from relied-upon provider observations."""

from pydantic import Field
from .contracts import ClosedRecord
from .identity import EffectiveRequest
from .timing import RequestTiming


class RawResponseEvidence(ClosedRecord):
    body: bytes
    body_sha256: str
    request_id: str | None = None


class GenerationObservation(ClosedRecord):
    raw_output: str
    backend_model_id: str | None
    finish_reason: str | None
    termination_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = None
    reported_model_matches_condition: bool | None = None


class GenerationExecution(ClosedRecord):
    observation: GenerationObservation
    evidence: RawResponseEvidence
    timing: RequestTiming
    effective_request: EffectiveRequest
