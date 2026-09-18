"""Client-provided ASR.

Two production-legitimate uses:

1. **Browser simulator / web widget** — the client captures the microphone and
   streams transcripts (Web Speech API or its own recognizer) so the same
   conversation core can be exercised without a telephone.
2. **Provider-side ASR** — some CPaaS platforms (Exotel AI, Ozonetel Kookoo,
   Amazon Connect) run their own speech recognition and post text to a webhook.
   In that mode this class receives text instead of audio, and the telephony
   adapter synthesises the reply back as provider TTS/audio.

Audio frames are counted but not decoded, and the endpointing logic still runs
on the energy VAD so barge-in behaves identically to the phone path.
"""

from __future__ import annotations

from typing import Any

from ...voice.audio import mulaw_to_pcm16
from .base import StreamingASR, TranscriptSegment


class ClientASR(StreamingASR):
    name = "client"
    streaming = True
    accepts_mulaw = True

    def __init__(self) -> None:
        super().__init__()
        self.audio_bytes_received = 0
        self.transcripts_received = 0

    async def feed(self, audio: bytes, *, encoding: str = "mulaw") -> None:
        self.audio_bytes_received += len(audio)

    async def push_transcript(
        self,
        text: str,
        *,
        is_final: bool = True,
        language: str | None = None,
        confidence: float = 0.0,
        utterance_end: bool | None = None,
        words: list[dict[str, Any]] | None = None,
    ) -> None:
        """Called by the websocket/webhook adapter when the client supplies text."""
        self.transcripts_received += 1
        await self._emit(
            TranscriptSegment(
                text=(text or "").strip(),
                is_final=is_final,
                language=language or self.language,
                confidence=confidence,
                utterance_end=bool(utterance_end if utterance_end is not None else is_final),
                words=words or [],
                provider=self.name,
            )
        )


class WhisperChunkASR(StreamingASR):
    """HTTP transcription of buffered utterances (OpenAI-compatible endpoint).

    Not streaming — used as a low-cost fallback or for languages the streaming
    engine misses. Audio is buffered until the VAD reports end-of-speech, then
    the utterance is transcribed in one request. Adds ~300–700 ms latency, so
    `ASR_ENDPOINT_SILENCE_MS` should stay low when this provider is active.
    """

    name = "whisper"
    accepts_mulaw = False
    preferred_rate = 16000
    streaming = False
    supported_languages = (
        "en", "hi", "ta", "bn", "mr", "gu", "te", "kn", "ml", "pa", "ur", "or",
    )

    def __init__(self, api_key: str = "", base_url: str = "", model: str = "") -> None:
        super().__init__()
        from ...config import settings

        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.whisper_base_url).rstrip("/")
        self.model = model or settings.whisper_model
        self._buffer: list[bytes] = []

    async def start(self, language: str | None = None, sample_rate: int = 8000) -> None:
        await super().start(language, sample_rate)

    async def feed(self, audio: bytes, *, encoding: str = "mulaw") -> None:
        if encoding == "mulaw":
            self._buffer.append(mulaw_to_pcm16(audio).tobytes())
        else:
            self._buffer.append(audio)

    async def transcribe_buffer(self) -> TranscriptSegment | None:
        """Flush the buffer and transcribe it. Returns None on empty audio."""
        import io
        import wave

        import httpx


        if not self._buffer:
            return None
        pcm = b"".join(self._buffer)
        self._buffer.clear()
        rate = self.sample_rate
        wav = io.BytesIO()
        with wave.open(wav, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(pcm)
        wav.seek(0)

        data = {"model": self.model, "response_format": "verbose_json"}
        if self.language:
            data["language"] = self.language.split("-")[0]
        try:  # pragma: no cover - network dependent
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": ("utterance.wav", wav.read(), "audio/wav")},
                    data=data,
                )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # pragma: no cover
            from .base import ASRError

            raise ASRError(f"whisper transcription failed: {exc}") from exc

        text = (payload.get("text") or "").strip()
        segment = TranscriptSegment(
            text=text,
            is_final=True,
            language=payload.get("language") or self.language,
            confidence=float(payload.get("probability") or 0.0),
            utterance_end=True,
            provider=self.name,
        )
        await self._emit(segment)
        return segment

    async def stop(self) -> None:  # pragma: no cover
        self._buffer.clear()
        await super().stop()
