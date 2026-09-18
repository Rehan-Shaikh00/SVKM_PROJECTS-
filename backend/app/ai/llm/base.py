"""LLM interface: chat completion, streaming and tool calling."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class StreamEvent:
    type: Literal["text_delta", "tool_use", "stop", "error", "usage"]
    text: str = ""
    tool_call: ToolCall | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class LLMResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    first_token_ms: float = 0.0
    provider: str = ""
    model: str = ""
    raw: Any = None

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class LLM(ABC):
    name = "base"
    model = ""
    supports_tools = False

    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 400,
        temperature: float = 0.2,
        language: str = "en-IN",
    ) -> LLMResult: ...

    async def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 400,
        temperature: float = 0.2,
        language: str = "en-IN",
    ) -> AsyncIterator[StreamEvent]:
        """Default: run `complete` and emit the text in one event."""
        started = time.time()
        result = await self.complete(
            system, messages, tools=tools, max_tokens=max_tokens,
            temperature=temperature, language=language,
        )
        for call in result.tool_calls:
            yield StreamEvent(type="tool_use", tool_call=call)
        if result.text:
            yield StreamEvent(type="text_delta", text=result.text)
        yield StreamEvent(
            type="stop", stop_reason=result.stop_reason or "end_turn", usage=result.usage
        )
        _ = started

    async def close(self) -> None:  # pragma: no cover
        return None
