"""Deepgram streaming ASR (recommended default for this deployment).

Why Deepgram here: native μ-law support (no transcoding of telephone audio),
low-latency interim results, `endpointing` for fast turn-taking, and
`language=multi` which returns a detected language per utterance — that gives us
spoken language identification and ASR from a single connection.

Protocol reference: POST/WS /v1/listen, JSON control + binary audio frames.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import websockets

from ...config import settings
from ...i18n.languages import asr_locale
from .base import ASRError, StreamingASR, TranscriptSegment

DEEPGRAM_WS = "wss://api.deepgram.com/v1/listen"
DEEPGRAM_MANAGE = "https://api.deepgram.com/v1/projects/usage/summary"

#: Nova-2/3 multilingual coverage for Indian languages (verify before rollout).
SUPPORTED = (
    "en-IN", "en-US", "en-GB", "hi-IN", "ta-IN", "bn-IN", "mr-IN", "gu-IN",
    "te-IN", "kn-IN", "ml-IN", "pa-IN", "ur-IN", "or-IN",
)


class DeepgramStreamingASR(StreamingASR):
    name = "deepgram"
    accepts_mulaw = True
    preferred_rate = 8000
    supported_languages = SUPPORTED

    def __init__(self, api_key: str | None = None, model: str = "nova-2-general") -> None:
        super().__init__()
        self.api_key = api_key or settings.deepgram_api_key
        self.model = model
        self._ws: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._multilingual = False
        self._bytes_sent = 0

    async def start(self, language: str | None = None, sample_rate: int = 8000) -> None:
        if not self.api_key:
            raise ASRError("DEEPGRAM_API_KEY is not configured")
        await super().start(language, sample_rate)

        params: dict[str, Any] = {
            "encoding": "mulaw",
            "sample_rate": str(sample_rate),
            "model": self.model,
            "smart_format": "true",
            "punctuate": "true",
            "interim_results": "true" if settings.asr_interim_results else "false",
            "endpointing": str(settings.asr_endpoint_silence_ms),
            "utterance_end_ms": "1200",
            "vad_events": "true",
        }
        if language is None or language == "multi":
            # language identification mode: let Deepgram pick per utterance
            params["language"] = "multi"
            params["detect_language"] = "true"
            self._multilingual = True
        else:
            params["language"] = asr_locale(language)
            self._multilingual = False

        try:
            self._ws = await websockets.connect(
                DEEPGRAM_WS,
                extra_headers={"Authorization": f"Token {self.api_key}"},
                subprotocols=["token"],
                max_size=2**22,
                ping_interval=20,
                ping_timeout=20,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise ASRError(f"deepgram connect failed: {exc}") from exc

        # extra_headers is the modern kwarg; older websockets used extra_headers=
        self._reader = asyncio.create_task(self._read_loop(), name="deepgram-reader")

    async def feed(self, audio: bytes, *, encoding: str = "mulaw") -> None:
        if self._ws is None:
            raise ASRError("DeepgramASR not started")
        if encoding != "mulaw":
            raise ASRError("DeepgramASR in this deployment expects mulaw frames")
        try:
            await self._ws.send(audio)
            self._bytes_sent += len(audio)
        except Exception as exc:  # pragma: no cover
            raise ASRError(f"deepgram send failed: {exc}") from exc

    async def stop(self) -> None:
        try:
            if self._ws is not None:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await asyncio.sleep(0.05)
        except Exception:
            pass
        if self._reader:
            self._reader.cancel()
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass
        await super().stop()

    # -- internals --------------------------------------------------------- #
    async def _read_loop(self) -> None:  # pragma: no cover - network dependent
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle(message)
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed:
            pass
        except Exception as exc:
            await self._emit(
                TranscriptSegment(text="", is_final=True, provider=self.name)
            )
            raise ASRError(f"deepgram read failed: {exc}") from exc
        finally:
            await super().stop()

    async def _handle(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind in ("Results",):
            channel = message.get("channel", {})
            alternatives = (channel.get("alternatives") or [{}])
            best = alternatives[0] if alternatives else {}
            text = (best.get("transcript") or "").strip()
            is_final = bool(message.get("is_final"))
            speech_final = bool(message.get("speech_final"))
            detected = message.get("channel", {}).get("detected_language") or (
                (message.get("metadata") or {}).get("detected_language")
            )
            confidence = float(best.get("confidence") or 0.0)
            if not text and not speech_final:
                return
            await self._emit(
                TranscriptSegment(
                    text=text,
                    is_final=is_final or speech_final,
                    language=detected or self.language,
                    confidence=confidence,
                    start_ms=int(message.get("start", 0) * 1000),
                    end_ms=int((message.get("start", 0) + message.get("duration", 0)) * 1000),
                    words=best.get("words") or [],
                    utterance_end=speech_final,
                    provider=self.name,
                )
            )
        elif kind == "UtteranceEnd":
            await self._emit(
                TranscriptSegment(text="", is_final=True, utterance_end=True, provider=self.name)
            )
        elif kind == "Metadata":
            return
        elif kind == "Error":
            raise ASRError(f"deepgram error: {message.get('description')}")

    @property
    def multilingual_mode(self) -> bool:
        return self._multilingual


async def check_key(api_key: str | None = None) -> dict[str, Any]:
    """Cheap health check used by the dashboard's provider status panel."""
    key = api_key or settings.deepgram_api_key
    if not key:
        return {"ok": False, "detail": "no api key"}
    try:  # pragma: no cover - network dependent
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.deepgram.com/v1/projects",
                headers={"Authorization": f"Token {key}"},
            )
        return {"ok": resp.status_code == 200, "detail": f"http {resp.status_code}"}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "detail": str(exc)}
