"""AssemblyAI Universal-Streaming adapter.

v3 WebSocket API: send a JSON config frame, then raw PCM binary frames; the
service returns `Turn` messages with partial and complete turns. It expects
16-bit PCM (we decode μ-law and keep 8 kHz, which Universal-Streaming accepts).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets

from ...config import settings
from ...voice.audio import mulaw_to_pcm16
from .base import ASRError, StreamingASR, TranscriptSegment

AA_WS = "wss://api.assemblyai.com/v3/ws"
SUPPORTED = ("en-IN", "en-US", "hi-IN", "es-ES", "fr-FR", "de-DE")


class AssemblyAIStreamingASR(StreamingASR):  # pragma: no cover - network dependent
    name = "assemblyai"
    accepts_mulaw = False
    preferred_rate = 8000
    supported_languages = SUPPORTED

    def __init__(self, api_key: str | None = None, word_boost: list[str] | None = None) -> None:
        super().__init__()
        self.api_key = api_key or settings.assemblyai_api_key
        #: domain vocabulary bias — big accuracy win for course/degree names
        self.word_boost = word_boost or [
            # Institution and place names a Dhule caller says constantly.
            "NMIMS", "SVKM", "Dhule", "Maharashtra", "Shirpur", "Nashik",
            # Degrees and programmes actually offered here.
            "B.Tech", "M.Tech", "BBA", "B.Com", "M.Com", "MBA", "BCA", "MCA",
            "B.Pharm", "D.Pharm", "M.Pharm", "LL.B", "Ph.D",
            # Entrance tests and domain vocabulary.
            "NPAT", "NMAT", "MHT-CET", "JEE Main", "CLAT", "LSAT", "GATE",
            "GPAT", "CAT", "hostel", "scholarship", "placement", "lateral entry",
        ]
        self._ws: Any = None
        self._reader: asyncio.Task[None] | None = None

    async def start(self, language: str | None = None, sample_rate: int = 8000) -> None:
        if not self.api_key:
            raise ASRError("ASSEMBLYAI_API_KEY is not configured")
        await super().start(language, sample_rate)
        try:
            self._ws = await websockets.connect(
                AA_WS, extra_headers={"Authorization": self.api_key}, max_size=2**22
            )
            await self._ws.send(
                json.dumps(
                    {
                        "format": "pcm",
                        "sample_rate": sample_rate,
                        "word_boost": self.word_boost,
                        "enable_extra_session_information": True,
                    }
                )
            )
        except Exception as exc:
            raise ASRError(f"assemblyai connect failed: {exc}") from exc
        self._reader = asyncio.create_task(self._read_loop(), name="aa-reader")

    async def feed(self, audio: bytes, *, encoding: str = "mulaw") -> None:
        if self._ws is None:
            raise ASRError("AssemblyAIStreamingASR not started")
        pcm = mulaw_to_pcm16(audio).tobytes() if encoding == "mulaw" else audio
        try:
            await self._ws.send(pcm)
        except Exception as exc:  # pragma: no cover
            raise ASRError(f"assemblyai send failed: {exc}") from exc

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                if message.get("message_type") == "Turn":
                    text = (message.get("text") or "").strip()
                    complete = bool(message.get("turn_is_formatted")) or bool(
                        message.get("end_of_turn_confidence", 0) > 0.5
                    )
                    if not text:
                        continue
                    await self._emit(
                        TranscriptSegment(
                            text=text,
                            is_final=complete,
                            language=message.get("language") or self.language,
                            confidence=float(message.get("end_of_turn_confidence") or 0.0),
                            start_ms=int(message.get("audio_start_ms") or 0),
                            end_ms=int(message.get("audio_end_ms") or 0),
                            utterance_end=complete,
                            provider=self.name,
                        )
                    )
                elif message.get("message_type") == "SessionInformation" or message.get("message_type") == "SessionBegins":
                    continue
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed:
            pass
        except Exception as exc:  # pragma: no cover
            raise ASRError(f"assemblyai read failed: {exc}") from exc
        finally:
            await super().stop()

    async def stop(self) -> None:
        if self._reader:
            self._reader.cancel()
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass
        await super().stop()
