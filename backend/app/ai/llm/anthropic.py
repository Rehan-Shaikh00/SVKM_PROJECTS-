"""Claude (Anthropic API) — the recommended reasoning engine.

Streaming is used so the first sentence reaches TTS (and therefore the caller's
ear) while the rest of the answer is still being generated. Tool calls
(`kb_search`, `escalate_to_human`, `offer_details`) arrive as `input_json_delta`
blocks and are assembled before being executed.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from ...config import settings
from .base import LLM, LLMResult, Message, StreamEvent, ToolCall, ToolSpec

logger = logging.getLogger("nims.llm.anthropic")


class AnthropicLLM(LLM):
    name = "anthropic"
    supports_tools = True

    def __init__(
        self, api_key: str | None = None, model: str | None = None, timeout: float = 20.0
    ) -> None:
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("anthropic package not installed") from exc
        self.api_key = api_key or settings.anthropic_api_key
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")
        self.model = model or settings.anthropic_model
        self.client = AsyncAnthropic(api_key=self.api_key, timeout=timeout, max_retries=1)

    # -- non-streaming ------------------------------------------------------ #
    async def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 400,
        temperature: float = 0.2,
        language: str = "en-IN",
    ) -> LLMResult:
        started = time.time()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": _convert_messages(messages),
        }
        if tools:
            kwargs["tools"] = [t.to_anthropic() for t in tools]
            kwargs["tool_choice"] = {"type": "auto"}
        response = await self.client.messages.create(**kwargs)
        latency = (time.time() - started) * 1000

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input or {}))
                )
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
        }
        return LLMResult(
            text="".join(text_parts).strip(),
            tool_calls=calls,
            stop_reason=str(response.stop_reason or "end_turn"),
            usage=usage,
            latency_ms=latency,
            provider=self.name,
            model=self.model,
        )

    # -- streaming ---------------------------------------------------------- #
    async def stream(  # pragma: no cover - network dependent
        self,
        system: str,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 400,
        temperature: float = 0.2,
        language: str = "en-IN",
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": _convert_messages(messages),
        }
        if tools:
            kwargs["tools"] = [t.to_anthropic() for t in tools]
            kwargs["tool_choice"] = {"type": "auto"}

        started = time.time()
        first_token_ms = 0.0
        tool_buffers: dict[int, dict[str, Any]] = {}
        stop_reason = "end_turn"
        usage: dict[str, Any] = {}

        try:
            async with self.client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    kind = getattr(event, "type", "")
                    if kind == "content_block_start":
                        block = event.content_block
                        if getattr(block, "type", "") == "tool_use":
                            tool_buffers[event.index] = {
                                "id": block.id, "name": block.name, "json": "",
                            }
                    elif kind == "content_block_delta":
                        delta = event.delta
                        delta_type = getattr(delta, "type", "")
                        if delta_type == "text_delta":
                            if not first_token_ms:
                                first_token_ms = (time.time() - started) * 1000
                            yield StreamEvent(type="text_delta", text=delta.text)
                        elif delta_type == "input_json_delta":
                            buffer = tool_buffers.get(event.index)
                            if buffer is not None:
                                buffer["json"] += delta.partial_json
                    elif kind == "content_block_stop":
                        buffer = tool_buffers.pop(event.index, None)
                        if buffer:
                            try:
                                payload = json.loads(buffer["json"] or "{}")
                            except json.JSONDecodeError:
                                payload = {}
                            yield StreamEvent(
                                type="tool_use",
                                tool_call=ToolCall(
                                    id=buffer["id"], name=buffer["name"], input=payload
                                ),
                            )
                    elif kind == "message_delta":
                        stop_reason = str(getattr(event.delta, "stop_reason", stop_reason))
                        raw_usage = getattr(event, "usage", None)
                        if raw_usage is not None:
                            usage["output_tokens"] = getattr(raw_usage, "output_tokens", 0)
                    elif kind == "message_start":
                        raw_usage = getattr(event.message, "usage", None)
                        if raw_usage is not None:
                            usage["input_tokens"] = getattr(raw_usage, "input_tokens", 0)
        except Exception as exc:
            logger.exception("claude streaming failed")
            yield StreamEvent(type="error", error=str(exc))
            return

        usage["first_token_ms"] = first_token_ms
        yield StreamEvent(
            type="usage", usage=usage, stop_reason=None
        )
        yield StreamEvent(type="stop", stop_reason=stop_reason, usage=usage)


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Map our neutral message list onto Anthropic's schema.

    Tool results are sent as user-role messages with `tool_result` blocks.
    """
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id or "unknown",
                            "content": message.content,
                        }
                    ],
                }
            )
        else:
            converted.append({"role": message.role, "content": message.content})
    return converted
