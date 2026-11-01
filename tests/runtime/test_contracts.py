import pytest
from pydantic import ValidationError
from hypothesis import given, strategies as st
from llm_evaluation.runtime.contracts import (
    BackendConfig,
    CapabilityProfile,
    CapabilityStatus,
    GenerationRequest,
    HttpTimeouts,
    ScientificEndpointIdentity,
    ModelCondition,
)


def test_closed_strict_frozen_contract():
    with pytest.raises(ValidationError):
        HttpTimeouts(extra=True)
    with pytest.raises(ValidationError):
        HttpTimeouts(read="1")
    with pytest.raises(ValidationError):
        HttpTimeouts(read=True)
    with pytest.raises(ValidationError):
        HttpTimeouts().read = 1.0
    with pytest.raises(ValidationError):
        ScientificEndpointIdentity(value=" ")


def test_nested_provenance_is_deeply_immutable():
    source = {"nested": [1, {"x": 2}]}
    c = ModelCondition(
        condition_id="c",
        weights_id="w",
        provider_id="p",
        backend_config_id="b",
        backend_name="n",
        backend_version="1",
        api_mode="chat_completions",
        endpoint_model_id="m",
        prompt_template_id="t",
        seed_behavior="unknown",
        runtime_environment=source,
    )
    source["nested"].append(3)
    assert len(c.runtime_environment["nested"]) == 2
    with pytest.raises(TypeError):
        c.runtime_environment["x"] = 1
    assert c.model_dump(mode="json")["runtime_environment"] == {"nested": [1, {"x": 2}]}


@given(
    st.sampled_from(
        [s for s in CapabilityStatus if s is not CapabilityStatus.SUPPORTED]
    )
)
def test_all_non_supported_capabilities_fail(status):
    p = CapabilityProfile(statuses={"seed": status})
    with pytest.raises(ValueError):
        p.require("seed")


def test_active_backend_config_rejects_retired_ollama():
    profile = CapabilityProfile(statuses={"models": CapabilityStatus.SUPPORTED})
    with pytest.raises(ValidationError, match="retired"):
        BackendConfig(
            provider_id="ollama",
            model_id="historical-model",
            api_mode="chat_completions",
            capability_profile=profile,
            config_id="new-ollama-run",
        )


@given(st.one_of(st.booleans(), st.integers(max_value=0)))
def test_invalid_integer_request_controls(value):
    with pytest.raises(ValidationError):
        GenerationRequest(condition_id="c", messages=(("user", "x"),), top_k=value)
