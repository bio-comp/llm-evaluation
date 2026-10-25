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
    EvaluationStatus,
    GenerationAttempt,
    GenerationStatus,
    GenerationRequest,
    ModelCondition,
    OpenAICompatibleTransport,
    ProviderRuntimeFailure,
    ReasoningControl,
    ReasoningWireMode,
    attempt_generation,
    canonical_json,
    record_evaluation,
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
        self.send_response(self.__class__.response_status)
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

    def _condition(self, **overrides):
        values = {
            "condition_id": "condition-a",
            "weights_id": "weights-a",
            "artifact_sha256": None,
            "quantization": None,
            "provider_id": "fake",
            "backend_config_id": "fake",
            "backend_name": "fake-runtime",
            "backend_version": "1.0",
            "api_mode": "chat_completions",
            "endpoint_model_id": "m",
            "prompt_template_id": "template-a",
            "seed_behavior": "supported",
            "runtime_environment": {"concurrency": 1},
        }
        values.update(overrides)
        return ModelCondition(**values)

    def _transport(self, config, *, condition_id="condition-a", timeout_seconds=120.0):
        return OpenAICompatibleTransport(
            config,
            condition=self._condition(
                condition_id=condition_id,
                provider_id=config.provider_id,
                backend_config_id=config.effective_config_id,
                endpoint_model_id=config.model_id,
                api_mode=config.api_mode,
            ),
            timeout_seconds=timeout_seconds,
        )

    def test_execution_condition_must_match_request_and_backend_before_transmission(self):
        request = GenerationRequest(condition_id="condition-a", messages=(("user", "x"),))
        matching = self._condition()
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)

        for condition, expected in (
            (self._condition(condition_id="condition-b"), "request condition"),
            (self._condition(provider_id="other"), "provider"),
            (self._condition(backend_config_id="other-config"), "backend config"),
            (self._condition(endpoint_model_id="other-model"), "model"),
            (self._condition(api_mode="responses"), "API mode"),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                OpenAICompatibleTransport(config, condition=condition).generate(request)
            self.assertEqual(_FakeHandler.requests, [])

        _FakeHandler.response_payload = {
            "id": "id", "model": "m",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        result = OpenAICompatibleTransport(config, condition=matching).generate(request)
        self.assertEqual(result.raw_output, "ok")
        self.assertEqual(len(_FakeHandler.requests), 1)

    def test_reasoning_configuration_is_condition_bound_capability_gated_and_transmitted(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "m",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        chat_profile = CapabilityProfile({
            **self.profile.statuses,
            ReasoningWireMode.CHAT_REASONING_EFFORT.value: CapabilityStatus.SUPPORTED,
        })
        chat_config = BackendConfig("fake", self.base_url, "m", "chat_completions", chat_profile)
        chat_condition = self._condition(
            reasoning_control=ReasoningControl(ReasoningWireMode.CHAT_REASONING_EFFORT, "low")
        )
        result = OpenAICompatibleTransport(chat_config, condition=chat_condition).generate(
            GenerationRequest(condition_id="condition-a", messages=(("user", "x"),))
        )
        self.assertEqual(_FakeHandler.requests[-1][3]["reasoning_effort"], "low")
        self.assertIn('"reasoning_effort":"low"', result.effective_request.serialized_body)

        _FakeHandler.requests = []
        differing_profile = CapabilityProfile({
            **self.profile.statuses,
            ReasoningWireMode.CHAT_REASONING_EFFORT.value: CapabilityStatus.SEMANTICS_DIFFER,
        })
        differing = BackendConfig("fake", self.base_url, "m", "chat_completions", differing_profile)
        with self.assertRaises(CapabilityError):
            OpenAICompatibleTransport(differing, condition=chat_condition).generate(
                GenerationRequest(condition_id="condition-a", messages=(("user", "x"),))
            )
        self.assertEqual(_FakeHandler.requests, [])

    def test_reasoning_capable_backend_requires_explicit_condition_control(self):
        profile = CapabilityProfile({
            **self.profile.statuses,
            ReasoningWireMode.CHAT_REASONING_EFFORT.value: CapabilityStatus.SUPPORTED,
        })
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", profile)
        with self.assertRaisesRegex(ValueError, "explicit reasoning"):
            OpenAICompatibleTransport(config, condition=self._condition()).generate(
                GenerationRequest(condition_id="condition-a", messages=(("user", "x"),))
            )
        self.assertEqual(_FakeHandler.requests, [])

    def test_responses_reasoning_effort_uses_nested_wire_shape(self):
        _FakeHandler.response_payload = {
            "id": "resp", "model": "m", "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        }
        profile = CapabilityProfile({
            **self.profile.statuses,
            ReasoningWireMode.RESPONSES_REASONING_EFFORT.value: CapabilityStatus.SUPPORTED,
        })
        config = BackendConfig("fake", self.base_url, "m", "responses", profile)
        condition = self._condition(
            api_mode="responses",
            reasoning_control=ReasoningControl(ReasoningWireMode.RESPONSES_REASONING_EFFORT, "medium"),
        )
        OpenAICompatibleTransport(config, condition=condition).generate(
            GenerationRequest(condition_id="condition-a", messages=(("user", "x"),))
        )
        self.assertEqual(_FakeHandler.requests[-1][3]["reasoning"], {"effort": "medium"})

    def test_reported_backend_model_mismatch_is_preserved_but_cannot_enter_sample(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "other-model",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        config = BackendConfig(
            "fake", self.base_url, "m", "chat_completions", self.profile,
            config_id="fake-routing-test",
        )
        request = GenerationRequest(condition_id="condition-a", messages=(("user", "x"),))
        result = self._transport(config).generate(request)
        self.assertFalse(result.reported_model_matches_condition)
        self.assertEqual(result.backend_model_id, "other-model")
        attempt = attempt_generation(self._transport(config), request, attempt_index=1)
        self.assertIsNotNone(attempt.result)
        with self.assertRaisesRegex(ValueError, "reported model"):
            record_evaluation(attempt, EvaluationStatus.PASSED, contributes_to_sample=True)

        _FakeHandler.response_payload = {
            "id": "id",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        missing = attempt_generation(self._transport(config), request, attempt_index=2)
        self.assertIsNone(missing.result.reported_model_matches_condition if missing.result else None)
        with self.assertRaisesRegex(ValueError, "reported model"):
            record_evaluation(missing, EvaluationStatus.PASSED, contributes_to_sample=True)

    def test_chat_wire_schema_effective_identity_and_raw_metadata(self):
        _FakeHandler.response_payload = {
            "id": "body-request-id",
            "object": "chat.completion",
            "model": "condition-model",
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
        result = self._transport(config).generate(request)
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
        self.assertEqual(result.backend_model_id, "condition-model")
        self.assertTrue(result.reported_model_matches_condition)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual((result.prompt_tokens, result.completion_tokens, result.total_tokens), (5, 2, 7))
        self.assertEqual(result.provider_request_id, "header-request-id")
        self.assertEqual(result.raw_provider_metadata["runtime"]["version"], "1.2.3")
        self.assertEqual(result.effective_request.body_sha256, result.effective_request.body_sha256)
        self.assertEqual(result.effective_request.serialized_body, canonical_json(body))
        self.assertEqual(result.effective_request.http_method, "POST")
        self.assertEqual(result.effective_request.backend_config_id, "local-fake")
        self.assertEqual(result.effective_request.condition_id, "condition-a")
        self.assertEqual(len(result.effective_request.condition_sha256), 64)
        self.assertEqual(len(result.effective_request.request_identity_sha256), 64)
        self.assertNotEqual(
            result.effective_request.request_identity_sha256,
            result.effective_request.body_sha256,
        )
        self.assertGreaterEqual(result.elapsed_seconds, 0)

    def test_explicit_body_request_id_is_not_confused_with_response_object_id(self):
        _FakeHandler.response_payload = {
            "id": "chatcmpl-response-object",
            "request_id": "req-explicit-body",
            "model": "m",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        result = self._transport(config, condition_id="c").generate(
            GenerationRequest(condition_id="c", messages=(("user", "x"),))
        )
        self.assertEqual(result.provider_request_id, "req-explicit-body")
        self.assertEqual(result.raw_provider_metadata["id"], "chatcmpl-response-object")

    def test_base_url_cannot_embed_credentials_that_effective_request_would_expose(self):
        with self.assertRaisesRegex(ValueError, "credentials"):
            BackendConfig(
                "bad", "https://user:secret@example.test/v1", "m", "chat_completions", self.profile
            )

    def test_credential_env_is_used_without_entering_config_or_effective_body(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        config = BackendConfig("hosted-fake", self.base_url, "m", "chat_completions", self.profile,
                               credential_env="FAKE_E002_KEY")
        with patch.dict(os.environ, {"FAKE_E002_KEY": "secret-value"}, clear=False):
            result = self._transport(config, condition_id="test-condition").generate(GenerationRequest(condition_id="test-condition", messages=(("user", "x"),)))
        _, _, headers, body = _FakeHandler.requests[-1]
        self.assertEqual(headers.get("Authorization"), "Bearer secret-value")
        self.assertNotIn("secret-value", repr(config))
        self.assertNotIn("secret-value", result.effective_request.serialized_body)
        self.assertNotIn("secret-value", canonical_json(body))

    def test_provider_error_cannot_echo_credential_into_failure_detail(self):
        _FakeHandler.response_status = 401
        _FakeHandler.response_payload = {"error": "credential=secret-value"}
        config = BackendConfig(
            "hosted-fake", self.base_url, "m", "chat_completions", self.profile,
            credential_env="FAKE_E002_KEY", config_id="hosted-fake-test",
        )
        with patch.dict(os.environ, {"FAKE_E002_KEY": "secret-value"}, clear=False):
            attempt = attempt_generation(
                self._transport(config, condition_id="test-condition"),
                GenerationRequest(condition_id="test-condition", messages=(("user", "x"),)),
                attempt_index=1,
            )
        self.assertEqual(attempt.status, AttemptStatus.PROVIDER_RUNTIME_FAILURE)
        self.assertNotIn("secret-value", attempt.failure_detail or "")
        self.assertIn("HTTP 401", attempt.failure_detail or "")

    def test_sensitive_provider_metadata_is_rejected_before_result_persistence(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "m",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "debug": {"Authorization": "Bearer echoed-secret"},
        }
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        with self.assertRaisesRegex(ProviderRuntimeFailure, "sensitive metadata key") as caught:
            self._transport(config, condition_id="test-condition").generate(
                GenerationRequest(condition_id="test-condition", messages=(("user", "x"),))
            )
        self.assertNotIn("echoed-secret", str(caught.exception))

    def test_unsupported_or_ignored_scientific_control_is_rejected_not_silently_sent(self):
        for status in (CapabilityStatus.UNSUPPORTED, CapabilityStatus.ACCEPTED_BUT_IGNORED,
                       CapabilityStatus.SEMANTICS_DIFFER, CapabilityStatus.UNKNOWN):
            profile = CapabilityProfile({**self.profile.statuses, "seed": status})
            config = BackendConfig("fake", self.base_url, "m", "chat_completions", profile)
            with self.subTest(status=status), self.assertRaises(CapabilityError):
                self._transport(config, condition_id="test-condition").generate(
                    GenerationRequest(condition_id="test-condition", messages=(("user", "x"),), seed=4)
                )
        self.assertEqual(_FakeHandler.requests, [])

    def test_backend_rejects_wire_parameter_name_collisions(self):
        with self.assertRaisesRegex(ValueError, "wire parameter names must be unique"):
            BackendConfig(
                "bad", self.base_url, "m", "chat_completions", self.profile,
                wire_parameter_names={"max_output_tokens": "top_p"},
            )

    def test_backend_rejects_sampler_translation_that_steals_protocol_fields(self):
        for api_mode, reserved in (("chat_completions", "model"), ("chat_completions", "messages"),
                                   ("responses", "model"), ("responses", "input")):
            with self.subTest(api_mode=api_mode, reserved=reserved), self.assertRaisesRegex(
                ValueError, "reserved protocol field"
            ):
                BackendConfig(
                    "bad", self.base_url, "m", api_mode, self.profile,
                    wire_parameter_names={"temperature": reserved},
                )
        for invalid in (" model", "model-name", "messages.value"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "wire parameter name"):
                BackendConfig(
                    "bad", self.base_url, "m", "chat_completions", self.profile,
                    wire_parameter_names={"temperature": invalid},
                )

    def test_backend_can_translate_internal_max_output_tokens_to_effective_wire_name(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "m",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        config = BackendConfig(
            "hosted-fake", self.base_url, "m", "chat_completions", self.profile,
            wire_parameter_names={"max_output_tokens": "max_completion_tokens"},
        )
        self._transport(config, condition_id="c").generate(
            GenerationRequest(condition_id="c", messages=(("user", "x"),), max_output_tokens=17)
        )
        body = _FakeHandler.requests[-1][3]
        self.assertEqual(body["max_completion_tokens"], 17)
        self.assertNotIn("max_tokens", body)

    def test_normalized_request_and_transport_validation_rejects_invalid_values_before_network(self):
        base = {"condition_id": "c", "messages": (("user", "x"),)}
        invalid_requests = (
            {"condition_id": "   ", "messages": (("user", "x"),)},
            {**base, "temperature": True},
            {**base, "temperature": -0.1},
            {**base, "temperature": float("nan")},
            {**base, "top_p": True},
            {**base, "top_p": 0.0},
            {**base, "top_p": 1.1},
            {**base, "top_k": True},
            {**base, "top_k": 0},
            {**base, "seed": True},
            {**base, "max_output_tokens": True},
            {**base, "stop": ()},
            {**base, "stop": ("",)},
        )
        for kwargs in invalid_requests:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GenerationRequest(**kwargs)

        with self.assertRaises(ValueError):
            self._condition(condition_id="   ")
        with self.assertRaises(ValueError):
            BackendConfig("   ", self.base_url, "m", "chat_completions", self.profile)
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        for timeout in (True, 0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                OpenAICompatibleTransport(config, condition=self._condition(), timeout_seconds=timeout)
        self.assertEqual(_FakeHandler.requests, [])

    def test_none_and_unsupported_unrequested_parameters_are_omitted(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        self._transport(config, condition_id="test-condition").generate(GenerationRequest(condition_id="test-condition", messages=(("user", "x"),)))
        body = _FakeHandler.requests[-1][3]
        self.assertEqual(body, {"model": "m", "messages": [{"role": "user", "content": "x"}]})

    def test_responses_wire_schema_and_parser_are_explicitly_mode_gated(self):
        _FakeHandler.response_payload = {
            "id": "resp_1", "model": "m2", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]}],
            "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        }
        config = BackendConfig("fake", self.base_url, "m", "responses", self.profile)
        result = self._transport(config, condition_id="test-condition").generate(
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
        result = self._transport(config, condition_id="reasoning-only").generate(
            GenerationRequest(condition_id="reasoning-only", messages=(("user", "x"),), max_output_tokens=8)
        )
        self.assertEqual(result.raw_output, "")
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual(result.completion_tokens, 8)

    def test_model_listing_is_capability_gated(self):
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        self.assertEqual(OpenAICompatibleTransport(config).list_models(), ("fake-model",))
        self.assertEqual(_FakeHandler.requests[-1][1], "/v1/models")

    def test_equivalent_provider_http_errors_have_consistent_runtime_classification(self):
        _FakeHandler.response_status = 503
        _FakeHandler.response_payload = {"error": "unavailable"}
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        transport = self._transport(config, condition_id="condition-a")
        with self.assertRaises(ProviderRuntimeFailure):
            transport.generate(GenerationRequest(condition_id="condition-a", messages=(("user", "x"),)))
        with self.assertRaises(ProviderRuntimeFailure):
            transport.list_models()

    def test_response_parsers_reject_ambiguous_shapes_and_invalid_usage(self):
        chat_bad = (
            {"choices": []},
            {"choices": [{"message": {"content": "a"}}, {"message": {"content": "b"}}]},
            {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": -1}},
            {"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": True}},
            {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 6}},
        )
        for payload in chat_bad:
            with self.subTest(payload=payload), self.assertRaises(ProviderRuntimeFailure):
                OpenAICompatibleTransport._parse_chat(payload)

        responses_bad = (
            {"output": ["not-an-object"]},
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": 3}]}]},
            {"output": [
                {"type": "message", "content": [{"type": "output_text", "text": "a"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "b"}]},
            ]},
            {"output": [], "usage": {"input_tokens": -1}},
        )
        for payload in responses_bad:
            with self.subTest(payload=payload), self.assertRaises(ProviderRuntimeFailure):
                OpenAICompatibleTransport._parse_responses(payload)

        parsed = OpenAICompatibleTransport._parse_responses({
            "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "a"},
                {"type": "output_text", "text": "b"},
            ]}],
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        })
        self.assertEqual(parsed["raw_output"], "ab")

    def test_effective_request_hash_is_deterministic_across_dict_order(self):
        left = canonical_json({"b": 2, "a": {"y": 2, "x": 1}})
        right = canonical_json({"a": {"x": 1, "y": 2}, "b": 2})
        self.assertEqual(left, right)

    def test_condition_identity_changes_normalized_request_hash_not_wire_body(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        first = self._transport(config, condition_id="condition-a").generate(
            GenerationRequest(condition_id="condition-a", messages=(("user", "x"),))
        )
        first_body = _FakeHandler.requests[-1][3]
        second = self._transport(config, condition_id="condition-b").generate(
            GenerationRequest(condition_id="condition-b", messages=(("user", "x"),))
        )
        second_body = _FakeHandler.requests[-1][3]
        self.assertEqual(first_body, second_body)
        self.assertEqual(first.effective_request.body_sha256, second.effective_request.body_sha256)
        self.assertNotEqual(first.effective_request.prompt_packet_sha256,
                            second.effective_request.prompt_packet_sha256)

    def test_capability_and_condition_mappings_are_copied_and_read_only(self):
        statuses = {"chat_completions": CapabilityStatus.SUPPORTED}
        profile = CapabilityProfile(statuses)
        statuses["chat_completions"] = CapabilityStatus.UNSUPPORTED
        profile.require("chat_completions")
        with self.assertRaises(TypeError):
            profile.statuses["chat_completions"] = CapabilityStatus.UNSUPPORTED

        runtime = {"concurrency": 1}
        metadata = {"architecture": "qwen35"}
        condition = ModelCondition(
            condition_id="immutable", weights_id="weights", artifact_sha256=None,
            quantization=None, provider_id="local", backend_config_id="local-config", backend_name="ollama",
            backend_version="0.40.8", api_mode="chat_completions", endpoint_model_id="m",
            prompt_template_id="template", seed_behavior="unknown",
            runtime_environment=runtime, artifact_metadata=metadata,
        )
        runtime["concurrency"] = 4
        metadata["architecture"] = "changed"
        self.assertEqual(condition.runtime_environment["concurrency"], 1)
        self.assertEqual(condition.artifact_metadata["architecture"], "qwen35")
        with self.assertRaises(TypeError):
            condition.runtime_environment["concurrency"] = 2

    def test_provenance_structures_are_deeply_immutable_and_sha_is_normalized(self):
        runtime = {"hardware": {"gpu": "A"}, "devices": ["gpu0"]}
        metadata = {"layers": [{"name": "decoder"}]}
        condition = self._condition(
            runtime_environment=runtime,
            artifact_metadata=metadata,
            artifact_sha256="A" * 64,
        )
        runtime["hardware"]["gpu"] = "B"
        runtime["devices"].append("gpu1")
        metadata["layers"][0]["name"] = "changed"
        self.assertEqual(condition.runtime_environment["hardware"]["gpu"], "A")
        self.assertEqual(condition.runtime_environment["devices"], ("gpu0",))
        self.assertEqual(condition.artifact_metadata["layers"][0]["name"], "decoder")
        self.assertEqual(condition.artifact_sha256, "a" * 64)
        with self.assertRaises(TypeError):
            condition.runtime_environment["hardware"]["gpu"] = "C"
        for bad_runtime in ({"devices": {"gpu0"}}, {1: "non-string-key"}, {"x": float("nan")}):
            with self.subTest(bad_runtime=bad_runtime), self.assertRaises(ValueError):
                self._condition(runtime_environment=bad_runtime)

        _FakeHandler.response_payload = {
            "id": "id", "model": "m",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "runtime": {"hardware": {"gpu": "A"}, "devices": ["gpu0"]},
        }
        config = BackendConfig("fake", self.base_url, "m", "chat_completions", self.profile)
        result = self._transport(config, condition_id="condition-a").generate(
            GenerationRequest(condition_id="condition-a", messages=(("user", "x"),))
        )
        with self.assertRaises(TypeError):
            result.raw_provider_metadata["runtime"]["hardware"]["gpu"] = "B"
        self.assertEqual(result.raw_provider_metadata["runtime"]["devices"], ("gpu0",))

    def test_model_condition_keeps_weights_backend_provider_and_template_separate(self):
        condition = ModelCondition(
            condition_id="qwen-local-ollama",
            weights_id="qwen3.5-9b",
            artifact_sha256="a" * 64,
            quantization="Q4_K_M",
            provider_id="local-mac",
            backend_config_id="ollama-qwen35",
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

    def test_backend_registry_separates_provider_identity_from_config_identity(self):
        qwen = BackendConfig(
            "ollama", self.base_url, "qwen3.5:9b", "chat_completions", self.profile,
            config_id="ollama-qwen35",
        )
        gpt_oss = BackendConfig(
            "ollama", self.base_url, "gpt-oss:20b", "chat_completions", self.profile,
            config_id="ollama-gpt-oss20b",
        )
        registry = BackendRegistry((qwen, gpt_oss))
        self.assertIs(registry["ollama-qwen35"], qwen)
        self.assertIs(registry["ollama-gpt-oss20b"], gpt_oss)
        self.assertEqual(registry.config_ids, ("ollama-qwen35", "ollama-gpt-oss20b"))
        self.assertEqual(registry.provider_ids, ("ollama",))
        with self.assertRaisesRegex(ValueError, "duplicate config_id"):
            BackendRegistry((qwen, qwen))
        with self.assertRaises(KeyError):
            _ = registry["missing"]

    def test_attempt_state_rejects_raw_closed_vocabularies_and_boolish_indexes(self):
        base = {
            "attempt_index": 1,
            "retry_of_attempt": None,
            "retry_count": 0,
            "generation_status": GenerationStatus.TRANSPORT_FAILURE,
            "result": None,
            "failure_detail": "offline",
        }
        with self.assertRaisesRegex(ValueError, "generation_status"):
            GenerationAttempt(**{**base, "generation_status": "transport_failure"})
        with self.assertRaisesRegex(ValueError, "attempt_index"):
            GenerationAttempt(**{**base, "attempt_index": True})
        with self.assertRaisesRegex(ValueError, "contributes_to_sample"):
            GenerationAttempt(**{**base, "contributes_to_sample": 0})

    def test_attempt_record_makes_transport_failure_and_retry_lineage_visible(self):
        request = GenerationRequest(condition_id="test-condition", messages=(("user", "x"),))
        implicit = BackendConfig(
            "offline", "http://127.0.0.1:1/v1", "m", "chat_completions", self.profile
        )
        with self.assertRaisesRegex(ValueError, "explicit backend config_id"):
            attempt_generation(
                self._transport(implicit, condition_id="test-condition", timeout_seconds=0.05),
                request, attempt_index=1,
            )
        config = BackendConfig(
            "offline", "http://127.0.0.1:1/v1", "m", "chat_completions", self.profile,
            config_id="offline-test",
        )
        first = attempt_generation(self._transport(config, condition_id="test-condition", timeout_seconds=0.05), request,
                                   attempt_index=1, contributes_to_sample=False)
        self.assertEqual(first.status, AttemptStatus.TRANSPORT_FAILURE)
        self.assertEqual(first.attempt_index, 1)
        self.assertIsNone(first.retry_of_attempt)
        self.assertFalse(first.contributes_to_sample)
        with self.assertRaisesRegex(ValueError, "after evaluation"):
            attempt_generation(
                self._transport(config, condition_id="test-condition", timeout_seconds=0.05),
                request, attempt_index=3, contributes_to_sample=True,
            )
        retry = attempt_generation(self._transport(config, condition_id="test-condition", timeout_seconds=0.05), request,
                                   attempt_index=2, retry_of_attempt=1, retry_count=1,
                                   contributes_to_sample=False)
        self.assertEqual(retry.retry_of_attempt, 1)
        self.assertEqual(retry.retry_count, 1)
        self.assertEqual(retry.status, AttemptStatus.TRANSPORT_FAILURE)
        later_retry = attempt_generation(self._transport(config, condition_id="test-condition", timeout_seconds=0.05), request,
                                         attempt_index=5, retry_of_attempt=2, retry_count=2,
                                         contributes_to_sample=False)
        self.assertEqual(later_retry.retry_count, 2)

    def test_generation_and_oracle_state_are_separate_before_sample_inclusion(self):
        _FakeHandler.response_payload = {
            "id": "id", "model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        config = BackendConfig(
            "fake", self.base_url, "m", "chat_completions", self.profile,
            config_id="fake-evaluated",
        )
        attempt = attempt_generation(
            self._transport(config, condition_id="test-condition"),
            GenerationRequest(condition_id="test-condition", messages=(("user", "x"),)),
            attempt_index=1,
        )
        self.assertEqual(attempt.generation_status, GenerationStatus.COMPLETED)
        self.assertEqual(attempt.evaluation_status, EvaluationStatus.NOT_EVALUATED)
        self.assertFalse(attempt.contributes_to_sample)
        self.assertIsNotNone(attempt.result)

        passed = record_evaluation(attempt, EvaluationStatus.PASSED, contributes_to_sample=True)
        self.assertTrue(passed.contributes_to_sample)
        self.assertIs(passed.result, attempt.result)

        oracle_error = record_evaluation(attempt, EvaluationStatus.ORACLE_ERROR)
        self.assertEqual(oracle_error.generation_status, GenerationStatus.COMPLETED)
        self.assertEqual(oracle_error.evaluation_status, EvaluationStatus.ORACLE_ERROR)
        self.assertIs(oracle_error.result, attempt.result)
        self.assertFalse(oracle_error.contributes_to_sample)
        with self.assertRaisesRegex(ValueError, "oracle error"):
            record_evaluation(attempt, EvaluationStatus.ORACLE_ERROR, contributes_to_sample=True)
        with self.assertRaisesRegex(ValueError, "already evaluated"):
            record_evaluation(passed, EvaluationStatus.FAILED, contributes_to_sample=True)


if __name__ == "__main__":
    unittest.main()
