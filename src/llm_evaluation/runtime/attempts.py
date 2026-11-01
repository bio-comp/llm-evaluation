"""Technical attempt lineage only: no oracle or sample-inclusion decisions."""

from pydantic import Field, model_validator
from .contracts import ClosedRecord
from .evidence import GenerationExecution
from .failures import RuntimeFailure
from .identity import EffectiveRequest
from .transport.base import AsyncGenerationTransport


class AttemptLineage(ClosedRecord):
    attempt_index: int = Field(ge=1)
    retry_of_attempt: int | None = None
    retry_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid(self) -> "AttemptLineage":
        if self.retry_of_attempt is None:
            if self.retry_count != 0:
                raise ValueError("non-retry count must be zero")
        elif (
            not 1 <= self.retry_of_attempt < self.attempt_index or self.retry_count < 1
        ):
            raise ValueError("retry must link an earlier attempt")
        return self


class TechnicalAttempt(ClosedRecord):
    lineage: AttemptLineage
    execution: GenerationExecution | None = None
    failure_kind: str | None = None
    remote_completion_unknown: bool = False

    @model_validator(mode="after")
    def exclusive(self) -> "TechnicalAttempt":
        if (self.execution is None) == (self.failure_kind is None):
            raise ValueError("exactly one execution or failure required")
        return self


async def attempt_generation(
    transport: AsyncGenerationTransport,
    request: EffectiveRequest,
    *,
    lineage: AttemptLineage,
) -> TechnicalAttempt:
    try:
        result = await transport.generate(request)
    except RuntimeFailure as exc:
        return TechnicalAttempt(
            lineage=lineage,
            failure_kind=exc.kind.value,
            remote_completion_unknown=exc.remote_completion_unknown,
        )
    return TechnicalAttempt(lineage=lineage, execution=result)
