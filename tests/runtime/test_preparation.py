from dataclasses import fields
import pytest
from test_model_runner_characterization import legacy_fixture, effective
from llm_evaluation.runtime.contracts import (
    BackendConfig,
    CapabilityProfile,
    CapabilityStatus,
    ModelCondition,
    RuntimeBinding,
    ScientificEndpointIdentity,
    GenerationRequest,
)
from llm_evaluation.runtime.preparation import prepare_generation
from llm_evaluation.runtime.identity import request_identity_v1


def native_binding():
    b, c, r = legacy_fixture()
    config = BackendConfig(
        provider_id=b.provider_id,
        model_id=b.model_id,
        api_mode=b.api_mode,
        config_id=b.effective_config_id,
        capability_profile=CapabilityProfile(
            statuses={
                k: CapabilityStatus(v.value)
                for k, v in b.capability_profile.statuses.items()
            }
        ),
    )
    condition = ModelCondition(**{f.name: getattr(c, f.name) for f in fields(c)})
    request = GenerationRequest(**{f.name: getattr(r, f.name) for f in fields(r)})
    return RuntimeBinding(
        backend=config,
        condition=condition,
        scientific_endpoint=ScientificEndpointIdentity(value="legacy"),
    ), request


def test_native_body_prompt_condition_and_explicit_v1_match_legacy():
    binding, request = native_binding()
    new = prepare_generation(binding, request)
    old = effective(*legacy_fixture())
    assert new.wire_bytes == old.serialized_body.encode()
    assert (new.body_sha256, new.prompt_packet_sha256, new.condition_sha256) == (
        old.body_sha256,
        old.prompt_packet_sha256,
        old.condition_sha256,
    )
    assert (
        request_identity_v1(
            endpoint=old.endpoint,
            api_mode=old.api_mode,
            backend_config_id=old.backend_config_id,
            condition_sha256=new.condition_sha256,
            body_sha256=new.body_sha256,
            prompt_packet_sha256=new.prompt_packet_sha256,
            timeout_seconds=120.0,
        )
        == old.request_identity_sha256
    )
    assert new.request_identity_sha256 != old.request_identity_sha256


def test_invalid_bindings_and_unknown_capability_fail_before_network():
    binding, request = native_binding()
    with pytest.raises(ValueError):
        RuntimeBinding(
            backend=binding.backend,
            condition=binding.condition.model_copy(update={"provider_id": "other"}),
            scientific_endpoint=binding.scientific_endpoint,
        )
    with pytest.raises(ValueError):
        prepare_generation(
            binding, request.model_copy(update={"condition_id": "wrong"})
        )
    with pytest.raises(ValueError):
        prepare_generation(binding, request.model_copy(update={"top_k": 1}))


def test_unrequested_control_is_not_sent():
    binding, request = native_binding()
    prepared = prepare_generation(
        binding, request.model_copy(update={"temperature": None})
    )
    assert b"temperature" not in prepared.wire_bytes


def test_transport_rejects_timeout_policy_different_from_prepared_identity():
    import asyncio
    from llm_evaluation.runtime.contracts import OperationalEndpoint, HttpTimeouts
    from llm_evaluation.runtime.transport.openai_compatible import (
        AsyncOpenAICompatibleTransport,
    )

    binding, request = native_binding()
    prepared = prepare_generation(binding, request)
    client = AsyncOpenAICompatibleTransport(
        OperationalEndpoint(base_url="http://127.0.0.1:1"),
        timeouts=HttpTimeouts(total=1.0),
    )
    with pytest.raises(ValueError, match="timeout"):
        asyncio.run(client.generate(prepared))
