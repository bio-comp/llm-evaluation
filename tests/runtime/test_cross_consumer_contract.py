"""Offline Prompt Research comparator: mechanism tests, not scientific equivalence."""

import asyncio
import json

import pytest

from llm_evaluation.runtime.contracts import (
    BackendConfig,
    CapabilityProfile,
    CapabilityStatus,
    GenerationRequest,
    ModelCondition,
    RuntimeBinding,
    ScientificEndpointIdentity,
)
from llm_evaluation.runtime.preparation import prepare_generation
from llm_evaluation.runtime.transport.openai_compatible import (
    AsyncOpenAICompatibleTransport,
)
from llm_evaluation.runtime.contracts import OperationalEndpoint
from llm_evaluation.runtime.failures import FailureKind, RuntimeFailure
from llm_evaluation.runtime.attempts import AttemptLineage


def local_contract(reasoning_mode="off", capability=None):
    """Match prlab's documented bounded chat-control case, without importing prlab."""
    supported = {
        key: CapabilityStatus.SUPPORTED
        for key in (
            "chat_completions", "temperature", "top_p", "top_k",
            "seed", "max_output_tokens",
        )
    }
    if capability:
        supported.update(capability)
    backend = BackendConfig(
        provider_id="managed-llamacpp", model_id="approved-fixture",
        api_mode="chat_completions", config_id="local-fixture",
        capability_profile=CapabilityProfile(statuses=supported),
    )
    condition = ModelCondition(
        condition_id="prlab-comparison-fixture",
        weights_id="approved-fixture",
        artifact_sha256="a" * 64, quantization="Q4_K_M",
        provider_id=backend.provider_id, backend_config_id=backend.config_id,
        backend_name="llama.cpp", backend_version="fixture-build",
        api_mode=backend.api_mode, endpoint_model_id=backend.model_id,
        prompt_template_id="fixture-template",
        seed_behavior="seed-42-requested-not-verified",
        runtime_environment={
            "context_tokens": 2048, "max_output_tokens": 128,
            "jinja": False, "reasoning_mode": reasoning_mode,
            "tool_mode": "disabled", "parallel": 1,
        },
    )
    binding = RuntimeBinding(
        backend=backend, condition=condition,
        scientific_endpoint=ScientificEndpointIdentity(value="local-fixture"),
    )
    request = GenerationRequest(
        condition_id=condition.condition_id,
        messages=(("system", "System."), ("user", "Return OK.")),
        temperature=0.0, top_p=1.0, top_k=1, seed=42,
        max_output_tokens=128,
    )
    return binding, request


def test_prlab_second_consumer_chat_case_and_explicit_difference():
    """prlab's stream:false is extra; do not assert full wire parity."""
    binding, request = local_contract()
    effective = prepare_generation(binding, request)
    packet = json.loads(effective.wire_bytes)
    # Independently reproduced with prlab.llama_cpp_runtime.request_body
    # on its local checked-out comparator; frozen so CI needs no prlab import.
    prlab_expected = {
        "model": "approved-fixture",
        "messages": [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Return OK."},
        ],
        "temperature": 0.0, "top_p": 1.0, "top_k": 1,
        "seed": 42, "max_tokens": 128, "stream": False,
    }
    assert {k: v for k, v in prlab_expected.items() if k != "stream"} == packet
    assert prlab_expected != packet  # different effective request protocol
    changed, _ = local_contract("auto")
    assert prepare_generation(changed, request).condition_sha256 != effective.condition_sha256
    assert prepare_generation(changed, request).request_identity_sha256 != effective.request_identity_sha256


@pytest.mark.parametrize(
    "status",
    [CapabilityStatus.UNSUPPORTED, CapabilityStatus.ACCEPTED_BUT_IGNORED,
     CapabilityStatus.SEMANTICS_DIFFER, CapabilityStatus.UNKNOWN],
)
def test_prlab_control_is_not_silently_transmitted_when_unqualified(status):
    binding, request = local_contract(capability={"top_k": status})
    with pytest.raises(ValueError, match="capability not supported"):
        prepare_generation(binding, request)


def test_retry_lineage_does_not_relabel_uncertain_remote_completion():
    one = AttemptLineage(attempt_index=1)
    two = AttemptLineage(attempt_index=2, retry_of_attempt=1, retry_count=1)
    assert one.retry_of_attempt is None
    assert two.retry_count == 1
    with pytest.raises(ValueError, match="earlier"):
        AttemptLineage(attempt_index=2, retry_of_attempt=2, retry_count=1)


@pytest.mark.parametrize(
    "mutation",
    [
        {"role": "tool"},
        {"tool_calls": [{"id": "tool1", "type": "function"}]},
        {"function_call": {"name": "unrequested"}},
    ],
)
def test_unrequested_tool_or_nonassistant_response_fails_closed(tmp_path, mutation):
    """Use the public async transport with a deterministic local fake HTTP server."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = {
                "model": "approved-fixture",
                "choices": [{
                    "message": {"role": "assistant", "content": "OK", **mutation},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5
                },
            }
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        binding, request = local_contract()
        transport = AsyncOpenAICompatibleTransport(
            OperationalEndpoint(base_url=f"http://127.0.0.1:{server.server_port}"),
            timeouts=binding.timeouts,
        )
        with pytest.raises(RuntimeFailure) as exc:
            asyncio.run(transport.generate(prepare_generation(binding, request)))
        assert exc.value.kind is FailureKind.INVALID_RESPONSE
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)



@pytest.mark.parametrize("role", [None, "tool", "user"])
def test_chat_message_requires_explicit_assistant_role(role):
    """An otherwise valid text response must not acquire assistant identity by default."""
    from llm_evaluation.runtime.transport.response_models import ProviderParser

    payload = {
        "model": "approved-fixture",
        "choices": [{
            "message": {"content": "OK", **({} if role is None else {"role": role})},
            "finish_reason": "stop",
        }],
    }
    with pytest.raises(ValueError, match="role"):
        ProviderParser._parse_chat(payload)


def test_v1_structured_failure_codes_remain_old_reader_compatible():
    """A schema-v1 writer must not emit values rejected by an existing v1 reader."""
    from typing import get_args
    from llm_evaluation.local_runtime_preflight_records import StructuredDiagnosticRecord

    original_v1_codes = {
        "http_status", "invalid_json", "invalid_response", "structured_parse",
        "structured_validation", "model_binding", "output_exhausted",
        "empty_visible_output", "deadline", "resource_busy", "startup_failure",
        "unknown_safe_failure",
    }
    annotation = StructuredDiagnosticRecord.model_fields["failure_code"].annotation
    code_literal = next(arg for arg in get_args(annotation) if arg is not type(None))
    assert set(get_args(code_literal)) == original_v1_codes
