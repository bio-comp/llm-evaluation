import asyncio
import pytest
from test_model_runner_characterization import legacy_fixture
from llm_evaluation.model_runner import OpenAICompatibleTransport


def test_sync_facade_rejects_active_loop_before_network_or_coroutine_creation():
    b, c, r = legacy_fixture()

    async def run():
        with pytest.raises(RuntimeError, match="async"):
            OpenAICompatibleTransport(b, condition=c).generate(r)
        with pytest.raises(RuntimeError, match="async"):
            OpenAICompatibleTransport(b, condition=c).list_models()

    asyncio.run(run())


def test_public_attempt_preserves_remote_completion_unknown():
    from llm_evaluation.model_runner import attempt_generation, TransportFailure

    b, c, r = legacy_fixture()
    transport = OpenAICompatibleTransport(b, condition=c)

    def fail(request):
        error = TransportFailure("read_timeout")
        error.remote_completion_unknown = True
        raise error

    transport.generate = fail
    result = attempt_generation(transport, r, attempt_index=1)
    assert result.remote_completion_unknown is True
