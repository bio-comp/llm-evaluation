import pytest
from llm_evaluation.runtime.codec import parse_json_unique, canonical_json_bytes
from llm_evaluation.runtime.contracts import ScientificEndpointIdentity, HttpTimeouts
from llm_evaluation.runtime.identity import request_identity_v2


def test_canonical_json_and_duplicate_keys():
    assert canonical_json_bytes({"z": 1, "a": "é"}) == b'{"a":"\xc3\xa9","z":1}'
    for b in (b'{"x":1,"x":2}', b'{"x":NaN}', b'"\\ud800"'):
        with pytest.raises(ValueError):
            parse_json_unique(b)


def test_v2_identity_separates_socket_from_scientific_identity():
    kwargs = dict(
        endpoint=ScientificEndpointIdentity(value="model-runtime"),
        api_mode="chat_completions",
        path="/chat/completions",
        backend_config_id="b",
        condition_sha256="a" * 64,
        body_sha256="b" * 64,
        prompt_packet_sha256="c" * 64,
        timeouts=HttpTimeouts(),
    )
    first = request_identity_v2(**kwargs)
    assert first == request_identity_v2(**kwargs)
    for key, value in [
        ("endpoint", ScientificEndpointIdentity(value="different")),
        ("body_sha256", "d" * 64),
        ("timeouts", HttpTimeouts(total=3.0)),
    ]:
        assert first != request_identity_v2(**(kwargs | {key: value}))
    assert "operational" not in kwargs
