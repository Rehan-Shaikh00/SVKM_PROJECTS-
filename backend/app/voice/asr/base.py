"""Streaming ASR interface shared by every provider.

The orchestrator only ever talks to this interface, which keeps provider swaps
(Deepgram → Google → Azure) to a config change.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptSegment:
    """One ASR result — interim (partial) or final."""

    text: str
    is_final: bool = True
    language: str | None = None
    confidence: float = 0.0
    start_ms: int = 0
    end_ms: int = 0
    words: list[dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    #: provider decided the utterance is complete (Deepgram `speech_final`)
    utterance_end: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class StreamingASR(ABC):
    """Feed audio in, get transcripts out.

    Implementations push results onto `self._queue`; `segments()` drains it.
    """

    name = "base"
    #: languages this engine can recognise (BCP-47). Empty = "unknown/any".
    supported_languages: tuple[str, ...] = ()
    #: accepts raw G.711 μ-law without server-side decoding
    accepts_mulaw: bool = False
    #: native sample rate the engine prefers
    preferred_rate: int = 8000
    streaming: bool = True

    def __init__(self) -> None:
        self._queue: asyncio.Queue[TranscriptSegment | None] = asyncio.Queue()
        self._closed = False
        self.language: str | None = None
        self.sample_rate = 8000

    # -- lifecycle --------------------------------------------------------- #
    async def start(self, language: str | None = None, sample_rate: int = 8000) -> None:
        self.language = language
        self.sample_rate = sample_rate

    async def stop(self) -> None:
        self._closed = True
        await self._queue.put(None)

    # -- data -------------------------------------------------------------- #
    @abstractmethod
    async def feed(self, audio: bytes, *, encoding: str = "mulaw") -> None:
        """Send one audio chunk (bytes in `encoding`)."""

    async def segments(self) -> AsyncIterator[TranscriptSegment]:
        """Yield transcript segments until the recognizer is stopped."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def next_segment(self, timeout: float | None = None) -> TranscriptSegment | None:
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            return None
        if item is None:
            return None
        return item

    # -- helpers ----------------------------------------------------------- #
    async def _emit(self, segment: TranscriptSegment) -> None:
        if not segment.provider:
            segment.provider = self.name
        await self._queue.put(segment)

    def supports(self, language: str | None) -> bool:
        if not language:
            return True
        if not self.supported_languages:
            return True
        base = language.split("-")[0].lower()
        return any(
            s.lower() == language.lower() or s.split("-")[0].lower() == base
            for s in self.supported_languages
        )


class NullASR(StreamingASR):
    """No-op recognizer: used when ASR_PROVIDER=none (text-only integrations)."""

    name = "null"
    streaming = False

    async def feed(self, audio: bytes, *, encoding: str = "mulaw") -> None:  # pragma: no cover
        return None


class ASRError(RuntimeError):
    """Raised when a provider fails; the orchestrator degrades to escalation."""
