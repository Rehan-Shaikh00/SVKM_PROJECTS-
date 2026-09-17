"""Media channel abstraction.

The orchestrator never knows whether it is talking to Twilio Media Streams, a
SIP/RTP bridge or the browser simulator. Everything it can *do* to a call is a
method here:

    send_audio      push μ-law frames to the caller
    clear_audio     barge-in: drop everything queued
    say             speak text (server TTS, or hand it to the client)
    transfer        hand the call to a human
    play_url        hold music / queue audio
    hangup          end the call
    emit            arbitrary state events (used by the simulator + dashboard)

and everything it can *receive* arrives through `inbound()`, an async iterator of
`InboundEvent`s (audio, transcripts, DTMF, hangup).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..voice.audio import FRAME_MS, iter_frames, pcm16_to_mulaw

logger = logging.getLogger("nims.channel")


@dataclass
class InboundEvent:
    type: str                     # audio | transcript | dtmf | hangup | mark | control
    audio: bytes = b""            # raw μ-law 8 kHz when type == "audio"
    text: str = ""                # transcript text
    is_final: bool = True
    language: str | None = None
    confidence: float = 0.0
    digit: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    received_at: float = field(default_factory=time.time)


class MediaChannel(ABC):
    """Bidirectional media path for one call."""

    name = "base"
    sample_rate = 8000

    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        self.closed = False
        self.audio_frames_sent = 0
        self.audio_bytes_sent = 0
        self.tts_frames_queued = 0
        self._playback_lock = asyncio.Lock()
        self._speaking = asyncio.Event()
        self._stop_speaking = asyncio.Event()
        self._outbox: asyncio.Queue[Any] = asyncio.Queue()

    # -- receive ----------------------------------------------------------- #
    @abstractmethod
    def inbound(self) -> AsyncIterator[InboundEvent]: ...

    # -- send -------------------------------------------------------------- #
    @abstractmethod
    async def send_audio(self, mulaw: bytes) -> None: ...

    @abstractmethod
    async def send_event(self, event_type: str, payload: dict[str, Any]) -> None: ...

    async def say(
        self,
        text: str,
        *,
        audio_chunks: list[bytes] | None = None,
        language: str = "en-IN",
        text_only: bool = False,
        mark: str | None = None,
    ) -> None:
        """Send synthesised audio (if any) plus the text for display/logging."""
        if text:
            await self.send_event(
                "assistant_text",
                {"text": text, "language": language, "mark": mark},
            )
        if text_only or not audio_chunks:
            await self.send_event(
                "assistant_speech",
                {"text": text, "language": language, "mode": "client_tts"},
            )
            return
        async with self._playback_lock:
            self._speaking.set()
            self._stop_speaking.clear()
            try:
                for chunk in audio_chunks:
                    if self._stop_speaking.is_set():
                        break
                    await self.send_audio(chunk)
            finally:
                self._speaking.clear()

    async def interrupt(self) -> bool:
        """Barge-in: stop whatever is playing. Returns True if something was cut."""
        was_speaking = self._speaking.is_set()
        self._stop_speaking.set()
        await self.clear_audio()
        if was_speaking:
            await self.send_event("playback_interrupted", {"call_id": self.call_id})
        return was_speaking

    @abstractmethod
    async def clear_audio(self) -> None: ...

    @abstractmethod
    async def transfer(self, target: str, *, target_type: str = "number",
                       whisper_text: str = "", context: dict[str, Any] | None = None) -> None: ...

    async def play_url(self, url: str, *, loop: bool = True) -> None:
        await self.send_event("play_url", {"url": url, "loop": loop})

    @abstractmethod
    async def hangup(self, reason: str = "completed") -> None: ...

    async def close(self) -> None:
        self.closed = True

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def begin_speaking(self) -> None:
        """Called by the session when playback starts (barge-in detection)."""
        self._speaking.set()
        self._stop_speaking.clear()

    def end_speaking(self) -> None:
        self._speaking.clear()


def pcm_to_channel_frames(pcm, sample_rate: int = 8000) -> list[bytes]:
    """Split PCM into the 20 ms μ-law frames the channel expects."""
    frames: list[bytes] = []
    for chunk in iter_frames(pcm, sample_rate, FRAME_MS):
        frames.append(pcm16_to_mulaw(chunk))
    return frames


class SimulatorChannel(MediaChannel):
    """WebSocket channel for the browser simulator / web widget.

    Audio (when produced by a server-side TTS) and text are both pushed to the
    client; the browser plays the audio if present, otherwise speaks the text
    with the Web Speech API. This is what lets the whole pipeline be demoed
    without a telephone number.
    """

    name = "simulator"

    def __init__(self, call_id: str, websocket: Any) -> None:
        super().__init__(call_id)
        self.websocket = websocket
        self._inbound_queue: asyncio.Queue[InboundEvent | None] = asyncio.Queue()
        self.state_history: list[dict[str, Any]] = []

    async def push_inbound(self, event: InboundEvent) -> None:
        await self._inbound_queue.put(event)

    async def inbound(self) -> AsyncIterator[InboundEvent]:
        while not self.closed:
            event = await self._inbound_queue.get()
            if event is None:
                return
            yield event

    async def _send(self, payload: dict[str, Any]) -> None:
        if self.closed or self.websocket is None:
            return
        try:
            await self.websocket.send_json(payload)
        except Exception as exc:  # pragma: no cover - client may vanish
            logger.info("simulator send failed: %s", exc)
            self.closed = True

    async def send_audio(self, mulaw: bytes) -> None:
        if not mulaw:
            return
        self.audio_frames_sent += 1
        self.audio_bytes_sent += len(mulaw)
        await self._send(
            {
                "type": "audio",
                "call_id": self.call_id,
                "encoding": "mulaw",
                "sample_rate": self.sample_rate,
                "frame_ms": FRAME_MS,
                "payload": base64.b64encode(mulaw).decode("ascii"),
            }
        )

    async def send_event(self, event_type: str, payload: dict[str, Any]) -> None:
        body = {"type": event_type, "call_id": self.call_id, "ts": time.time()}
        body.update(payload)
        self.state_history.append({"type": event_type, "ts": body["ts"]})
        await self._send(body)

    async def clear_audio(self) -> None:
        await self.send_event("clear_audio", {"barge_in": True})

    async def transfer(
        self, target: str, *, target_type: str = "number", whisper_text: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        await self.send_event(
            "transfer",
            {
                "target": target,
                "target_type": target_type,
                "whisper": whisper_text,
                "context": context or {},
            },
        )

    async def hangup(self, reason: str = "completed") -> None:
        await self.send_event("hangup", {"reason": reason})
        await self.close()

    async def close(self) -> None:
        if self.closed:
            return
        await super().close()
        await self._inbound_queue.put(None)


class NullChannel(MediaChannel):
    """Used by tests: collects everything, sends nothing."""

    name = "null"

    def __init__(self, call_id: str = "test-call") -> None:
        super().__init__(call_id)
        self.events: list[dict[str, Any]] = []
        self.audio: list[bytes] = []
        self.transfers: list[dict[str, Any]] = []
        self.hangups: list[str] = []
        self._inbound_queue: asyncio.Queue[InboundEvent | None] = asyncio.Queue()

    async def push(self, event: InboundEvent) -> None:
        await self._inbound_queue.put(event)

    async def inbound(self) -> AsyncIterator[InboundEvent]:
        while not self.closed:
            event = await self._inbound_queue.get()
            if event is None:
                return
            yield event

    async def send_audio(self, mulaw: bytes) -> None:
        self.audio.append(mulaw)
        self.audio_bytes_sent += len(mulaw)

    async def send_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, **payload})

    async def clear_audio(self) -> None:
        self.events.append({"type": "clear_audio"})

    async def transfer(self, target: str, *, target_type: str = "number",
                       whisper_text: str = "", context: dict[str, Any] | None = None) -> None:
        self.transfers.append(
            {"target": target, "target_type": target_type, "whisper": whisper_text,
             "context": context or {}}
        )

    async def hangup(self, reason: str = "completed") -> None:
        self.hangups.append(reason)
        await self.close()

    @property
    def spoken_texts(self) -> list[str]:
        return [e["text"] for e in self.events if e["type"] == "assistant_text"]


def hold_music_url() -> str:
    return settings.hold_music_url
