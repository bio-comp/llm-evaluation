"""Only directly observed timing is populated."""

from pydantic import Field
from .contracts import ClosedRecord


class RequestTiming(ClosedRecord):
    queue_admission_wait_seconds: float | None = Field(default=None, ge=0)
    server_processing_wait_seconds: float | None = Field(default=None, ge=0)
    prompt_processing_seconds: float | None = Field(default=None, ge=0)
    generation_seconds: float | None = Field(default=None, ge=0)
    total_request_wall_seconds: float = Field(ge=0)
