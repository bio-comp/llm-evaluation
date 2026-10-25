"""Opt-in live OpenAI-compatible integration probe.

Public CI remains offline because this module skips unless LLM_EVAL_INTEGRATION=1.
The caller must explicitly name the endpoint, model and capabilities it wants to test.
"""
import os
import unittest

from llm_evaluation.model_runner import (
    BackendConfig,
    CapabilityProfile,
    CapabilityStatus,
    GenerationRequest,
    OpenAICompatibleTransport,
)


def _enabled() -> bool:
    return os.environ.get("LLM_EVAL_INTEGRATION") == "1"


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise unittest.SkipTest(f"{name} is required for live integration probing")
    return value


@unittest.skipUnless(_enabled(), "live model integration probe is opt-in")
class LiveOpenAICompatibleProbe(unittest.TestCase):
    def test_declared_local_capabilities_and_one_generation(self):
        base_url = _required("LLM_EVAL_INTEGRATION_BASE_URL")
        model = _required("LLM_EVAL_INTEGRATION_MODEL")
        provider_id = _required("LLM_EVAL_INTEGRATION_PROVIDER_ID")
        api_mode = os.environ.get("LLM_EVAL_INTEGRATION_API_MODE", "chat_completions")
        declared = {
            item.strip() for item in _required("LLM_EVAL_INTEGRATION_CAPABILITIES").split(",")
            if item.strip()
        }
        statuses = {name: CapabilityStatus.SUPPORTED for name in declared}
        profile = CapabilityProfile(statuses)
        config = BackendConfig(provider_id, base_url, model, api_mode, profile)
        transport = OpenAICompatibleTransport(config, timeout_seconds=60)

        if "models" in declared:
            self.assertIn(model, transport.list_models())

        kwargs = {}
        if "temperature" in declared:
            kwargs["temperature"] = 0.7
        if "top_p" in declared:
            kwargs["top_p"] = 0.9
        if "seed" in declared:
            kwargs["seed"] = 1701
        if "max_output_tokens" in declared:
            kwargs["max_output_tokens"] = 64
        if "stop" in declared:
            kwargs["stop"] = ("STOP_NEVER",)
        result = transport.generate(GenerationRequest(
            condition_id=f"integration-{provider_id}",
            messages=(("user", "Return exactly the token PROBE."),),
            **kwargs,
        ))
        self.assertIsInstance(result.raw_output, str)
        self.assertGreaterEqual(result.elapsed_seconds, 0)
        self.assertIsInstance(result.raw_provider_metadata, dict)
        self.assertEqual(result.effective_request.api_mode, api_mode)


if __name__ == "__main__":
    unittest.main()
