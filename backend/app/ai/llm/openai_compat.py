"""OpenAI-compatible chat LLM (OpenAI, Groq, Together, vLLM, Ollama, Azure).

Kept as a second provider so the deployment is not single-vendor: swap by
setting LLM_PROVIDER=openai and OPENAI_BASE_URL to any compatible endpoint.
Implemented over plain HTTP to avoid a hard SDK dependency.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...config import settings
from .base import LLM, LLMResult, Message, StreamEvent, ToolCall, ToolSpec

logger = logging.getLogger("nims.llm.openai")


class OpenAICompatibleLLM(LLM):
    name = "openai"
    supports_tools = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 25.0,
    ) -> None:
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")
        self.model = model or settings.openai_model
        self.base_url = (
            base_url or getattr(settings, "openai_base_url", "https://api.openai.com/v1")
        ).rstrip("/")
        self.timeout = timeout

    def _payload(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}]
            + [m.to_dict() if m.role != "tool" else {
                "role": "tool", "content": m.content, "tool_call_id": m.tool_call_id
            } for m in messages],
        }
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

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
        payload = self._payload(system, messages, tools, max_tokens, temperature, False)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        latency = (time.time() - started) * 1000
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = [
            ToolCall(
                id=call.get("id", ""),
                name=(call.get("function") or {}).get("name", ""),
                input=json.loads((call.get("function") or {}).get("arguments") or "{}"),
            )
            for call in (message.get("tool_calls") or [])
        ]
        return LLMResult(
            text=(message.get("content") or "").strip(),
            tool_calls=calls,
            stop_reason=str(choice.get("finish_reason") or "stop"),
            usage=data.get("usage") or {},
            latency_ms=latency,
            provider=self.name,
            model=self.model,
        )

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
        payload = self._payload(system, messages, tools, max_tokens, temperature, True)
        started = time.time()
        first_token_ms = 0.0
        tool_buffers: dict[int, dict[str, str]] = {}
        stop_reason = "stop"
        usage: dict[str, Any] = {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client, client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            if not first_token_ms:
                                first_token_ms = (time.time() - started) * 1000
                            yield StreamEvent(type="text_delta", text=delta["content"])
                        for call in delta.get("tool_calls") or []:
                            index = int(call.get("index") or 0)
                            buffer = tool_buffers.setdefault(
                                index, {"id": "", "name": "", "arguments": ""}
                            )
                            if call.get("id"):
                                buffer["id"] = call["id"]
                            function = call.get("function") or {}
                            if function.get("name"):
                                buffer["name"] = function["name"]
                            if function.get("arguments"):
                                buffer["arguments"] += function["arguments"]
                        if choice.get("finish_reason"):
                            stop_reason = choice["finish_reason"]
        except Exception as exc:
            logger.exception("openai-compatible streaming failed")
            yield StreamEvent(type="error", error=str(exc))
            return

        for index in sorted(tool_buffers):
            buffer = tool_buffers[index]
            if not buffer["name"]:
                continue
            try:
                payload_args = json.loads(buffer["arguments"] or "{}")
            except json.JSONDecodeError:
                payload_args = {}
            yield StreamEvent(
                type="tool_use",
                tool_call=ToolCall(id=buffer["id"], name=buffer["name"], input=payload_args),
            )
        usage["first_token_ms"] = first_token_ms
        yield StreamEvent(type="usage", usage=usage)
        yield StreamEvent(type="stop", stop_reason=stop_reason, usage=usage)
