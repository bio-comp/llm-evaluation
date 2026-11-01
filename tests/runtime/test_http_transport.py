import asyncio
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from llm_evaluation.runtime.transport.openai_compatible import (
    AsyncOpenAICompatibleTransport,
)
from llm_evaluation.runtime.transport.base import BearerAuthorization
from llm_evaluation.runtime.contracts import OperationalEndpoint, HttpTimeouts
from llm_evaluation.runtime.failures import RuntimeFailure, FailureKind


class Handler(BaseHTTPRequestHandler):
    calls: list[tuple[str, bytes, dict[str, str]]] = []
    payload = b"{}"
    status = 200
    drip = False

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.do_POST()

    def do_POST(self):
        type(self).calls.append(
            (
                self.path,
                self.rfile.read(int(self.headers.get("Content-Length", 0))),
                dict(self.headers),
            )
        )
        self.send_response(type(self).status)
        self.send_header("Location", "/redirect-target")
        self.send_header("X-Request-ID", "header-id")
        self.end_headers()
        try:
            if type(self).drip:
                for _ in range(50):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.02)
            else:
                self.wfile.write(type(self).payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def server():
    Handler.calls = []
    Handler.payload = b"{}"
    Handler.status = 200
    Handler.drip = False
    s = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:" + str(s.server_port)
    s.shutdown()
    s.server_close()
    t.join(2)


def get(url, **kwargs):
    return asyncio.run(
        AsyncOpenAICompatibleTransport(
            OperationalEndpoint(base_url=url), **kwargs
        ).request_json("POST", "/capture", b'{"x":1}')
    )


def test_exact_wire_proxy_ssl_isolation(server, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/ca.pem")
    get(server)
    assert Handler.calls[0][1] == b'{"x":1}'


@pytest.mark.parametrize("status", [307, 401, 503])
def test_no_redirect_or_retry(server, status):
    Handler.status = status
    with pytest.raises(RuntimeFailure) as e:
        get(server)
    assert e.value.kind is FailureKind.HTTP_STATUS
    assert len(Handler.calls) == 1


@pytest.mark.parametrize(
    "payload,kind",
    [
        (b'{"x":1,"x":2}', FailureKind.INVALID_JSON),
        (b"not JSON", FailureKind.INVALID_JSON),
        (b'{"secret":"x"}', FailureKind.CREDENTIAL_REFLECTION),
        (b'{"x":"test-key"}', FailureKind.CREDENTIAL_REFLECTION),
        (b" " * 1001, FailureKind.RESPONSE_TOO_LARGE),
    ],
)
def test_bounded_sanitized_evidence(server, payload, kind):
    Handler.payload = payload
    with pytest.raises(RuntimeFailure) as e:
        get(
            server,
            authorization=BearerAuthorization("test-key"),
            max_response_bytes=1000,
        )
    assert e.value.kind is kind
    assert "test-key" not in repr(e.value)
    assert e.value.__context__ is None


def test_total_deadline_catches_drip_response(server):
    Handler.drip = True
    with pytest.raises(RuntimeFailure) as e:
        get(server, timeouts=HttpTimeouts(read=0.2, total=0.12))
    assert e.value.kind is FailureKind.TOTAL_TIMEOUT
    assert e.value.remote_completion_unknown
    assert len(Handler.calls) == 1


def test_cancel_after_dispatch_propagates_with_uncertainty(server):
    Handler.drip = True

    async def run():
        t = asyncio.create_task(
            AsyncOpenAICompatibleTransport(
                OperationalEndpoint(base_url=server)
            ).request_json("POST", "/capture", b"{}")
        )
        while not Handler.calls:
            await asyncio.sleep(0.01)
        t.cancel()
        with pytest.raises(asyncio.CancelledError) as e:
            await t
        assert e.value.remote_completion_unknown

    asyncio.run(run())


def test_secret_repr_and_injection_boundary(server):
    assert "test-key" not in repr(BearerAuthorization("test-key"))
    with pytest.raises(TypeError):
        AsyncOpenAICompatibleTransport(
            OperationalEndpoint(base_url=server), client=object()
        )
    with pytest.raises(ValueError):
        AsyncOpenAICompatibleTransport(
            OperationalEndpoint(base_url="http://example.test"),
            authorization=BearerAuthorization("x"),
        )


def test_credential_in_unknown_mapping_key_is_not_returned_as_raw_evidence(server):
    Handler.payload = b'{"test-key":"value"}'
    with pytest.raises(RuntimeFailure) as failure:
        get(server, authorization=BearerAuthorization("test-key"))
    assert failure.value.kind is FailureKind.CREDENTIAL_REFLECTION


def test_outer_asyncio_deadline_is_not_bypassed_by_cancellation_wrapping(server):
    Handler.drip = True

    async def run():
        async with asyncio.timeout(0.1):
            await AsyncOpenAICompatibleTransport(
                OperationalEndpoint(base_url=server)
            ).request_json("POST", "/capture", b"{}")

    with pytest.raises(TimeoutError):
        asyncio.run(run())



def test_generic_transport_has_no_llamacpp_mode():
    import inspect

    assert "llamacpp_exact_build" not in inspect.signature(
        AsyncOpenAICompatibleTransport
    ).parameters


def test_generic_transport_leaves_provider_specific_timings_untrusted(server):
    import json
    from test_preparation import native_binding
    from llm_evaluation.runtime.preparation import prepare_generation

    binding, request = native_binding()
    Handler.payload = json.dumps({
        "model": binding.backend.model_id,
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "timings": {"prompt_ms": 12.5, "predicted_ms": 20},
    }).encode()
    execution = asyncio.run(AsyncOpenAICompatibleTransport(
        OperationalEndpoint(base_url=server), timeouts=binding.timeouts,
    ).generate(prepare_generation(binding, request)))
    assert execution.timing.prompt_processing_seconds is None
    assert execution.timing.generation_seconds is None
    assert execution.timing.queue_admission_wait_seconds is None
    assert execution.timing.server_processing_wait_seconds is None
