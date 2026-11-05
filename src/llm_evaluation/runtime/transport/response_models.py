"""Strict relied-upon projections; unknown fields stay untrusted raw evidence."""

from typing import Any
from pydantic import BaseModel, ConfigDict


class ProviderChatProjection(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")
    model: str | None = None
    choices: list[dict[str, Any]]
    usage: dict[str, Any] | None = None


class ProviderResponsesProjection(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")
    model: str | None = None
    output: list[dict[str, Any]]
    status: str | None = None


class ProviderParser:
    @staticmethod
    def _validated_usage(
        payload: dict[str, Any], input_key: str, output_key: str
    ) -> tuple[int | None, int | None, int | None]:
        raw_usage = payload.get("usage")
        if raw_usage is None:
            return None, None, None
        if not isinstance(raw_usage, dict):
            raise ValueError("provider usage must be a JSON object")

        def value(key: str) -> int | None:
            raw = raw_usage.get(key)
            if raw is None:
                return None
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ValueError("provider token usage must be nonnegative integers")
            return raw

        prompt_tokens = value(input_key)
        completion_tokens = value(output_key)
        total_tokens = value("total_tokens")
        components = [v for v in (prompt_tokens, completion_tokens) if v is not None]
        minimum_total = sum(components)
        if total_tokens is not None and components and total_tokens < minimum_total:
            raise ValueError("provider total token usage is inconsistent")
        return prompt_tokens, completion_tokens, total_tokens

    @staticmethod
    def _parse_chat(payload: dict[str, Any]) -> dict[str, Any]:
        choices = payload.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
        ):
            raise ValueError("Chat Completions requires exactly one choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("invalid Chat Completions response shape")
        # Tools are not part of this text-generation contract. Never silently
        # accept their calls as an ordinary visible completion.
        if message.get("role") not in (None, "assistant"):
            raise ValueError("unexpected Chat Completions message role")
        if message.get("tool_calls") not in (None, []) or message.get("function_call") is not None:
            raise ValueError("unexpected Chat Completions tool/function call")
        finish_reason = choice.get("finish_reason")
        if finish_reason in ("tool_calls", "function_call"):
            raise ValueError("unexpected tool/function termination")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ValueError("Chat Completions finish_reason is not a string")
        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError("Chat Completions model id is not a string")
        prompt_tokens, completion_tokens, total_tokens = (
            ProviderParser._validated_usage(
                payload, "prompt_tokens", "completion_tokens"
            )
        )
        return {
            "raw_output": message["content"],
            "backend_model_id": model,
            "finish_reason": finish_reason,
            "termination_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _parse_responses(payload: dict[str, Any]) -> dict[str, Any]:
        output = payload.get("output")
        if not isinstance(output, list):
            raise ValueError("Responses payload has no output list")
        texts: list[str] = []
        message_count = 0
        for item in output:
            if not isinstance(item, dict):
                raise ValueError("Responses output items must be JSON objects")
            item_type = item.get("type")
            if item_type == "reasoning":
                continue
            if item_type != "message":
                raise ValueError("Responses output contains an unsupported item type")
            message_count += 1
            if message_count > 1:
                raise ValueError("Responses payload contains multiple message outputs")
            content = item.get("content")
            if not isinstance(content, list):
                raise ValueError("Responses message content is not a list")
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "output_text":
                    raise ValueError("Responses message contains unsupported content")
                if not isinstance(part.get("text"), str):
                    raise ValueError("Responses output_text is not a string")
                texts.append(part["text"])
        status = payload.get("status")
        if status is not None and not isinstance(status, str):
            raise ValueError("Responses status is not a string")
        incomplete_details = payload.get("incomplete_details")
        termination_reason: str | None = None
        if incomplete_details is not None:
            if not isinstance(incomplete_details, dict):
                raise ValueError("Responses incomplete_details is not a JSON object")
            reason = incomplete_details.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError("Responses incomplete_details.reason is not a string")
            if status != "incomplete":
                raise ValueError(
                    "Responses incomplete_details is inconsistent with response status"
                )
            termination_reason = reason
        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError("Responses model id is not a string")
        prompt_tokens, completion_tokens, total_tokens = (
            ProviderParser._validated_usage(payload, "input_tokens", "output_tokens")
        )
        return {
            "raw_output": "".join(texts),
            "backend_model_id": model,
            "finish_reason": status,
            "termination_reason": termination_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
