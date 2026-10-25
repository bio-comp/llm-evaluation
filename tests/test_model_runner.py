import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from llm_evaluation.model_runner import (
    BackendConfig,
    BackendRegistry,
    CapabilityError,
    CapabilityProfile,
    CapabilityStatus,
    AttemptStatus,
    GenerationRequest,
    ModelCondition,
    OpenAICompatibleTransport,
    attempt_generation,
    canonical_json,
)


class _FakeHandler(BaseHTTPRequestHandler):
    requests = []
    response_payload = {}
    response_status = 200

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.__class__.requests.append(("GET", self.path, dict(self.headers), None))
        body = json.dumps({"object": "list", "data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(size)
        self.__class__.requests.append(("POST", self.path, dict(self.headers), json.loads(raw)))
        body = json.dumps(self.__class__.response_payload).encode()
        self.send_response(self.__class__.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Request-ID", "header-request-id")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ModelRunnerTests(unittest.TestCase):
    def setUp(self):
        _FakeHandler.requests = []
        _FakeHandler.response_status = 200
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.profile = CapabilityProfile(
            {
                "models": CapabilityStatus.SUPPORTED,
                "chat_completions": CapabilityStatus.SUPPORTED,
                "responses": CapabilityStatus.SUPPORTED,
                "temperature": CapabilityStatus.SUPPORTED,
                "top_p": CapabilityStatus.SUPPORTED,
                "top_k": CapabilityStatus.UNSUPPORTED,
                "seed": CapabilityStatus.SUPPORTED,
                "max_output_tokens": CapabilityStatus.SUPPORTED,
                "stop": CapabilityStatus.SUPPORTED,
                "usage": CapabilityStatus.SUPPORTED,
                "finish_reason": CapabilityStatus.SUPPORTED,
            }
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    def test_chat_wire_schema_effective_identity_and_raw_metadata(self):
        _FakeHandler.response_payload = {
            "id": "body-request-id",
            "object": "chat.completion",
            "model": "backend-reported-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            "runtime": {"name": "fake-runtime", "version": "1.2.3"},
        }
        config = BackendConfig("local-fake", self.base_url, "condition-model", "chat_completions", self.profile)
        request = GenerationRequest(
            condition_id="condition-a",
            messages=(("system", "be terse"), ("user", "ping")),
            temperature=0.2, top_p=0.9, max_output_tokens=7, stop=("END",), seed=123,
        )
        result = OpenAICompatibleTransport(config).generate(request)
        method, path, headers, body = _FakeHandler.requests[-1]
        self.assertEqual((method, path), ("POST", "/v1/chat/completions"))
        self.assertEqual(body, {
            "model": "condition-model",
            "messages": [{"role": "system", "content": "be terse"}, {"role": "user", "content": "ping"}],
            "temperature": 0.2, "top_p": 0.9, "seed": 123,
            "max_tokens": 7, "stop": ["END"],
        })
        self.assertNotIn("Authorization", headers)
        self.assertEqual(result.raw_output, "answer")
        self.assertEqual(result.backend_model_id, "backend-reported-model")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual((result.prompt_tokens, result.completion_tokens, result.total_tokens), (5, 2, 7))
        self.assertEqual(result.provider_request_id, "body-request-id")
        self.assertEqual(result.raw_provider_metadata["runtime"]["version"], "1.2.3")
        self.assertEqual(result.effective_request.body_sha256, result.effective_request.body_sha256)
        self.assertEqual(result.effective_request.serialized_body, canonical_json(body))
        self.assertGreaterEqual(result.elapsed_seconds, 0)

    def test_credential_env_is_used_without_entering_config_or_effective_body(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        config = BackendConfig("hosted-fake", self.base_url, "m", "chat_completions", self.profile,
                               credential_env="FAKE_E002_KEY")
        with patch.dict(os.environ, {"FAKE_E002_KEY": "secret-value"}, clear=False):
            result = OpenAICompatibleTransport(config).generate(GenerationRequest(condition_id="test-condition", messages=(("user", "x"),)))
        _, _, headers, body = _FakeHandler.requests[-1]
        self.assertEqual(headers.get("Authorization"), "Bearer secret-value")
        self.assertNotIn("secret-value", repr(config))
        self.assertNotIn("secret-value", result.effective_request.serialized_body)
        self.assertNotIn("secret-value", canonical_json(body))

    def test_provider_error_cannot_echo_credential_into_failure_detail(self):
        _FakeHandler.response_status = 401
        _FakeHandler.response_payload = {"error": "credential=secret-value"}
        config = BackendConfig("hosted-fake", self.base_url, "m", "chat_completions", self.profile,
                               credential_env="FAKE_E002_KEY")
        with patch.dict(os.environ, {"FAKE_E002_KEY": "secret-value"}, clear=False):
            attempt = attempt_generation(
                OpenAICompatibleTransport(config),
                GenerationRequest(condition_id="test-condition", messages=(("user", "x"),)),
                attempt_index=1,
            )
        self.assertEqual(attempt.status, AttemptStatus.PROVIDER_RUNTIME_FAILURE)
        self.assertNotIn("secret-value", attempt.failure_detail or "")
        self.assertIn("HTTP 401", attempt.failure_detail or "")

    def test_unsupported_or_ignored_scientific_control_is_rejected_not_silently_sent(self):
        for status in (CapabilityStatus.UNSUPPORTED, CapabilityStatus.ACCEPTED_BUT_IGNORED,
                       CapabilityStatus.SEMANTICS_DIFFER, CapabilityStatus.UNKNOWN):
            profile = CapabilityProfile({**self.profile.statuses, "seed": status})
            config = BackendConfig("fake", self.base_url, "m", "chat_completions", profile)
            with self.subTest(status=status), self.assertRaises(CapabilityError):
                OpenAICompatibleTransport(config).generate(
                    GenerationRequest(condition_id="test-condition", messages=(("user", "x"),), seed=4)
                )
        self.assertEqual(_FakeHandler.requests, [])

    def test_none_and_unsupported_unrequested_parameters_are_omitted(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        OpenAICompatibleTransport(config).generate(GenerationRequest(condition_id="test-condition", messages=(("user", "x"),)))
        body = _FakeHandler.requests[-1][3]
        self.assertEqual(body, {"model": "m", "messages": [{"role": "user", "content": "x"}]})

    def test_responses_wire_schema_and_parser_are_explicitly_mode_gated(self):
        _FakeHandler.response_payload = {
            "id": "resp_1", "model": "m2", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]}],
            "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        }
        config = BackendConfig("fake", self.base_url, "m", "responses", self.profile)
        result = OpenAICompatibleTransport(config).generate(
            GenerationRequest(condition_id="test-condition", messages=(("user", "x"),), temperature=0.1, max_output_tokens=8)
        )
        body = _FakeHandler.requests[-1][3]
        self.assertEqual(_FakeHandler.requests[-1][1], "/v1/responses")
        self.assertEqual(body["input"], [{"role": "user", "content": "x"}])
        self.assertEqual(body["max_output_tokens"], 8)
        self.assertEqual(result.raw_output, "hello")
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual((result.prompt_tokens, result.completion_tokens, result.total_tokens), (4, 2, 6))

    def test_responses_reasoning_only_completion_is_empty_visible_output_not_runtime_failure(self):
        _FakeHandler.response_payload = {
            "id": "resp_reasoning", "model": "m2", "status": "completed",
            "output": [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thinking"}]}],
            "usage": {"input_tokens": 4, "output_tokens": 8, "total_tokens": 12},
        }
        config = BackendConfig("fake", self.base_url, "m", "responses", self.profile)
        result = OpenAICompatibleTransport(config).generate(
            GenerationRequest(condition_id="reasoning-only", messages=(("user", "x"),), max_output_tokens=8)
        )
        self.assertEqual(result.raw_output, "")
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual(result.completion_tokens, 8)

    def test_model_listing_is_capability_gated(self):
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        self.assertEqual(OpenAICompatibleTransport(config).list_models(), ("fake-model",))
        self.assertEqual(_FakeHandler.requests[-1][1], "/v1/models")

    def test_effective_request_hash_is_deterministic_across_dict_order(self):
        left = canonical_json({"b": 2, "a": {"y": 2, "x": 1}})
        right = canonical_json({"a": {"x": 1, "y": 2}, "b": 2})
        self.assertEqual(left, right)

    def test_condition_identity_changes_normalized_request_hash_not_wire_body(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        transport = OpenAICompatibleTransport(config)
        first = transport.generate(GenerationRequest(condition_id="condition-a", messages=(("user", "x"),)))
        first_body = _FakeHandler.requests[-1][3]
        second = transport.generate(GenerationRequest(condition_id="condition-b", messages=(("user", "x"),)))
        second_body = _FakeHandler.requests[-1][3]
        self.assertEqual(first_body, second_body)
        self.assertEqual(first.effective_request.body_sha256, second.effective_request.body_sha256)
        self.assertNotEqual(first.effective_request.prompt_packet_sha256,
                            second.effective_request.prompt_packet_sha256)

    def test_model_condition_keeps_weights_backend_provider_and_template_separate(self):
        condition = ModelCondition(
            condition_id="qwen-local-ollama",
            weights_id="qwen3.5-9b",
            artifact_sha256="a" * 64,
            quantization="Q4_K_M",
            provider_id="local-mac",
            backend_name="ollama",
            backend_version="0.40.8",
            api_mode="chat_completions",
            endpoint_model_id="qwen3.5:9b",
            prompt_template_id="ollama-qwen35-template",
            seed_behavior="supported-not-proven-identical-across-backends",
            runtime_environment={"hardware": "apple-silicon", "concurrency": 1},
            artifact_filename="sha256-" + "a" * 64,
            tokenizer_id="qwen3.5-tokenizer",
            artifact_metadata={"architecture": "qwen35", "context_length": 262144},
        )
        self.assertNotEqual(condition.weights_id, condition.endpoint_model_id)
        self.assertEqual(condition.provider_id, "local-mac")
        self.assertEqual(condition.backend_name, "ollama")
        self.assertEqual(condition.runtime_environment["concurrency"], 1)
        self.assertEqual(condition.artifact_metadata["architecture"], "qwen35")
        self.assertEqual(condition.tokenizer_id, "qwen3.5-tokenizer")

    def test_backend_registry_is_small_explicit_and_rejects_duplicate_ids(self):
        one = BackendConfig("one", self.base_url, "m1", "chat_completions", self.profile)
        two = BackendConfig("two", self.base_url, "m2", "responses", self.profile)
        registry = BackendRegistry((one, two))
        self.assertIs(registry["one"], one)
        self.assertEqual(registry.provider_ids, ("one", "two"))
        with self.assertRaisesRegex(ValueError, "duplicate provider_id"):
            BackendRegistry((one, one))
        with self.assertRaises(KeyError):
            _ = registry["missing"]

    def test_attempt_record_makes_transport_failure_and_retry_lineage_visible(self):
        config = BackendConfig(
            "offline", "http://127.0.0.1:1/v1", "m", "chat_completions", self.profile
        )
        request = GenerationRequest(condition_id="test-condition", messages=(("user", "x"),))
        first = attempt_generation(OpenAICompatibleTransport(config, timeout_seconds=0.05), request,
                                   attempt_index=1, contributes_to_sample=False)
        self.assertEqual(first.status, AttemptStatus.TRANSPORT_FAILURE)
        self.assertEqual(first.attempt_index, 1)
        self.assertIsNone(first.retry_of_attempt)
        self.assertFalse(first.contributes_to_sample)
        retry = attempt_generation(OpenAICompatibleTransport(config, timeout_seconds=0.05), request,
                                   attempt_index=2, retry_of_attempt=1, retry_count=1,
                                   contributes_to_sample=False)
        self.assertEqual(retry.retry_of_attempt, 1)
        self.assertEqual(retry.retry_count, 1)
        self.assertEqual(retry.status, AttemptStatus.TRANSPORT_FAILURE)
        later_retry = attempt_generation(OpenAICompatibleTransport(config, timeout_seconds=0.05), request,
                                         attempt_index=5, retry_of_attempt=2, retry_count=2,
                                         contributes_to_sample=False)
        self.assertEqual(later_retry.retry_count, 2)

    def test_successful_attempt_can_be_marked_as_statistical_sample(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        attempt = attempt_generation(OpenAICompatibleTransport(config),
                                     GenerationRequest(condition_id="test-condition", messages=(("user", "x"),)),
                                     attempt_index=1, contributes_to_sample=True)
        self.assertEqual(attempt.status, AttemptStatus.MODEL_GENERATION_COMPLETED)
        self.assertTrue(attempt.contributes_to_sample)
        self.assertIsNotNone(attempt.result)
        self.assertIsNone(attempt.failure_detail)


if __name__ == "__main__":
    unittest.main()
