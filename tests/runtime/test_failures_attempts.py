import asyncio
import pytest
from llm_evaluation.runtime.attempts import AttemptLineage, attempt_generation
from llm_evaluation.runtime.failures import RuntimeFailure, FailureKind
from llm_evaluation.runtime.timing import RequestTiming


def test_retry_lineage_and_no_scientific_sample_state():
    with pytest.raises(ValueError):
        AttemptLineage(attempt_index=1, retry_of_attempt=1, retry_count=1)
    with pytest.raises(ValueError):
        AttemptLineage(attempt_index=1, retry_count=1)

    class Failing:
        calls = 0

        async def generate(self, request):
            self.calls += 1
            raise RuntimeFailure(
                FailureKind.READ_TIMEOUT, remote_completion_unknown=True
            )

    t = Failing()
    first = asyncio.run(
        attempt_generation(t, None, lineage=AttemptLineage(attempt_index=1))
    )
    second = asyncio.run(
        attempt_generation(
            t,
            None,
            lineage=AttemptLineage(attempt_index=2, retry_of_attempt=1, retry_count=1),
        )
    )
    assert t.calls == 2 and first.execution is None and second.lineage.retry_count == 1
    assert first.remote_completion_unknown
    assert "contributes_to_sample" not in first.model_dump()
    timing = RequestTiming(total_request_wall_seconds=1.0)
    assert timing.server_processing_wait_seconds is None
    assert timing.queue_admission_wait_seconds is None
