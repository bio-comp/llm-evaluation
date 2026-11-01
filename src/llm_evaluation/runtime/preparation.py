"""Pure request validation, capability translation, exact bytes and hashes."""

from .codec import canonical_json_bytes, sha256_hex
from .contracts import RuntimeBinding, GenerationRequest
from .identity import EffectiveRequest, request_identity_v2
from .structured import StructuredOutputContract


def prepare_generation(
    binding: RuntimeBinding,
    request: GenerationRequest,
    structured: StructuredOutputContract | None = None,
) -> EffectiveRequest:
    b, c = binding.backend, binding.condition
    if c.condition_id != request.condition_id:
        raise ValueError("request condition mismatch")
    b.capability_profile.require(b.api_mode)
    messages = [{"role": role, "content": text} for role, text in request.messages]
    body: dict[str, object] = {
        "model": b.model_id,
        "messages" if b.api_mode == "chat_completions" else "input": messages,
    }
    for key in ("temperature", "top_p", "top_k", "seed", "stop", "max_output_tokens"):
        value = getattr(request, key)
        if value is not None:
            b.capability_profile.require(key)
            default = (
                ("max_tokens" if b.api_mode == "chat_completions" else key)
                if key == "max_output_tokens"
                else key
            )
            body[b.wire_parameter_names.get(key, default)] = value
    if c.reasoning_control is not None:
        rc = c.reasoning_control
        b.capability_profile.require(rc.capability_name)
        body[
            "reasoning_effort" if b.api_mode == "chat_completions" else "reasoning"
        ] = rc.effort if b.api_mode == "chat_completions" else {"effort": rc.effort}
    if structured is not None:
        b.capability_profile.require("json_schema")
        if b.api_mode != "chat_completions":
            raise ValueError("structured Responses adapter not qualified")
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "strict": True,
                "schema": structured.effective_schema_json(),
            },
        }
    wire = canonical_json_bytes(body)
    prompt = request.model_dump()
    prompt["messages"] = messages
    body_sha = sha256_hex(wire)
    prompt_sha = sha256_hex(canonical_json_bytes(prompt))
    condition_sha = sha256_hex(canonical_json_bytes(c.model_dump()))
    path = "/chat/completions" if b.api_mode == "chat_completions" else "/responses"
    identity = request_identity_v2(
        endpoint=binding.scientific_endpoint,
        api_mode=b.api_mode,
        path=path,
        backend_config_id=b.config_id,
        condition_sha256=condition_sha,
        body_sha256=body_sha,
        prompt_packet_sha256=prompt_sha,
        timeouts=binding.timeouts,
    )
    return EffectiveRequest(
        path=path,
        api_mode=b.api_mode,
        wire_bytes=wire,
        body_sha256=body_sha,
        prompt_packet_sha256=prompt_sha,
        condition_sha256=condition_sha,
        request_identity_sha256=identity,
        backend_config_id=b.config_id,
        condition_id=c.condition_id,
        endpoint_model_id=c.endpoint_model_id,
        timeouts=binding.timeouts,
    )
