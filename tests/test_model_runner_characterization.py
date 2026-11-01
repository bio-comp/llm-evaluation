"""Historical v1 identity: complete recovered inputs, no network."""

import os
import unittest
from dataclasses import replace
from unittest.mock import patch
from llm_evaluation.model_runner import (
    BackendConfig,
    CapabilityProfile,
    CapabilityStatus,
    GenerationRequest,
    ModelCondition,
    OpenAICompatibleTransport,
)


def legacy_fixture():
    profile = CapabilityProfile(
        {
            name: CapabilityStatus.SUPPORTED
            for name in (
                "chat_completions",
                "temperature",
                "top_p",
                "seed",
                "max_output_tokens",
                "stop",
            )
        }
    )
    config = BackendConfig(
        "legacy-fixture",
        "https://example.test/v1",
        "condition-model",
        "chat_completions",
        profile,
        config_id="legacy-config",
    )
    condition = ModelCondition(
        condition_id="legacy-condition",
        weights_id="weights-a",
        artifact_sha256="a" * 64,
        quantization="Q4_K_M",
        provider_id="legacy-fixture",
        backend_config_id="legacy-config",
        backend_name="legacy-runtime",
        backend_version="1.0",
        api_mode="chat_completions",
        endpoint_model_id="condition-model",
        prompt_template_id="template-a",
        seed_behavior="supported",
        runtime_environment={"concurrency": 1},
    )
    request = GenerationRequest(
        "legacy-condition",
        (("system", "be terse"), ("user", "ping")),
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=7,
        stop=("END",),
        seed=123,
    )
    return config, condition, request


def effective(config, condition, request):
    transport = OpenAICompatibleTransport(
        config, condition=condition, timeout_seconds=120.0
    )
    endpoint, body = transport._body(request)
    return transport._effective(request, endpoint, body)


class LegacyCharacterization(unittest.TestCase):
    def test_legacy_v1_identity_fixture_is_exact(self):
        result = effective(*legacy_fixture())
        expected = {
            "body_sha256": "7a80a2e4d4edf2fa2afff0de27063e2b4b7192315187f03a864580c5c7c5dfc3",
            "prompt_packet_sha256": "551460ef522bbee5a54e0106fe4093167db1141c22bcdc345568fc6af2f3010e",
            "condition_sha256": "917d2d742dddeceb3663ba73db9ca7725452af7a99122d40e37480d0ff3cbee6",
            "request_identity_sha256": "c52aa247d6922661e7281e8afc3a6ed1d6d27c1fbb40c14681c4da3bc2d8391e",
        }
        for name, digest in expected.items():
            self.assertEqual(getattr(result, name), digest)

    def test_legacy_v1_endpoint_change_changes_request_identity(self):
        b, c, r = legacy_fixture()
        self.assertNotEqual(
            effective(b, c, r).request_identity_sha256,
            effective(
                replace(b, base_url="https://other.test/v1"), c, r
            ).request_identity_sha256,
        )

    def test_legacy_v1_condition_change_changes_condition_and_request_identity(self):
        b, c, r = legacy_fixture()
        old, new = (
            effective(b, c, r),
            effective(b, replace(c, weights_id="different"), r),
        )
        self.assertNotEqual(old.condition_sha256, new.condition_sha256)
        self.assertNotEqual(old.request_identity_sha256, new.request_identity_sha256)
        self.assertEqual(old.body_sha256, new.body_sha256)

    def test_legacy_v1_body_order_is_canonical(self):
        b, c, r = legacy_fixture()
        t = OpenAICompatibleTransport(b, condition=c)
        ep, body = t._body(r)
        self.assertEqual(
            t._effective(r, ep, body),
            t._effective(r, ep, dict(reversed(list(body.items())))),
        )

    def test_legacy_v1_credential_does_not_enter_body_or_hash(self):
        b, c, r = legacy_fixture()
        with patch.dict(os.environ, {"FIXTURE_KEY": "fixture-secret"}):
            self.assertEqual(
                effective(b, c, r),
                effective(replace(b, credential_env="FIXTURE_KEY"), c, r),
            )
