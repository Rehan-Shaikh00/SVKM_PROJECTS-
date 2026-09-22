"""Claude LLM provider implementation."""

import asyncio
from typing import List, Optional, AsyncGenerator
from anthropic import AsyncAnthropic

from packages.providers.interfaces import LLMProvider, Message, LLMResponse
from packages.config.settings import get_settings


class ClaudeProvider(LLMProvider):
    """Claude API implementation."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncAnthropic(api_key=self.settings.anthropic.api_key)

    async def generate(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResponse:
        """Generate completion from messages."""
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = await self.client.messages.create(
            model=self.settings.anthropic.model,
            max_tokens=max_tokens or self.settings.anthropic.max_tokens,
            temperature=temperature or self.settings.anthropic.temperature,
            system=system_prompt or "",
            messages=formatted_messages,
            stop_sequences=stop_sequences or []
        )

        return LLMResponse(
            content=response.content[0].text,
            finish_reason=response.stop_reason or "end_turn",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            model=response.model
        )

    async def generate_stream(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion."""
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        async with self.client.messages.stream(
            model=self.settings.anthropic.model,
            max_tokens=max_tokens or self.settings.anthropic.max_tokens,
            temperature=temperature or self.settings.anthropic.temperature,
            system=system_prompt or "",
            messages=formatted_messages
        ) as stream:
            async for text in stream.text_stream:
                yield text


class FakeLLMProvider(LLMProvider):
    """Fake LLM provider for testing."""

    async def generate(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResponse:
        """Return canned response."""
        await asyncio.sleep(0.2)  # Simulate latency

        return LLMResponse(
            content="This is a fake grounded response based on retrieved evidence.",
            finish_reason="end_turn",
            usage={"input_tokens": 100, "output_tokens": 50},
            model="fake-model"
        )

    async def generate_stream(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Stream fake response word by word."""
        text = "This is a fake grounded response based on retrieved evidence."
        for word in text.split():
            await asyncio.sleep(0.05)
            yield word + " "
