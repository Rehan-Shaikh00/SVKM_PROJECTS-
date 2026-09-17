"""Google Cloud Speech-to-Text streaming adapter (gRPC).

Requires `google-cloud-speech` (see requirements.txt, optional block). Google is
a good pick when you already run on GCP, and it is the provider that supports
**spoken language identification from an explicit candidate list**
(`alternative_language_codes`), which is exactly the "no keypad" requirement.

If the SDK is missing, importing this module still works — instantiation raises
ASRError, and the registry falls back to another provider.
"""

from __future__ import annotations

import asyncio
import queue
from typing import Any

from ...config import settings
from .base import ASRError, StreamingASR, TranscriptSegment

SUPPORTED = (
    "en-IN", "hi-IN", "ta-IN", "bn-IN", "mr-IN", "gu-IN", "te-IN", "kn-IN",
    "ml-IN", "pa-IN", "ur-IN", "or-IN", "as-IN",
)


class GoogleStreamingASR(StreamingASR):  # pragma: no cover - requires GCP SDK + key
    name = "google"
    accepts_mulaw = True
    preferred_rate = 8000
    supported_languages = SUPPORTED

    def __init__(self, candidates: list[str] | None = None) -> None:
        super().__init__()
        self.candidates = candidates or list(SUPPORTED)
        self._client: Any = None
        self._requests: queue.Queue[Any] = queue.Queue()
        self._stream_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def start(self, language: str | None = None, sample_rate: int = 8000) -> None:
        try:
            from google.cloud import speech  # type: ignore
        except ImportError as exc:
            raise ASRError(
                "google-cloud-speech is not installed; add it to requirements.txt"
            ) from exc

        if settings.google_application_credentials:
            import os

            os.environ.setdefault(
                "GOOGLE_APPLICATION_CREDENTIALS", settings.google_application_credentials
            )
        try:
            self._client = speech.SpeechAsyncClient()
        except Exception as exc:
            raise ASRError(f"google speech client init failed: {exc}") from exc

        await super().start(language, sample_rate)
        self._loop = asyncio.get_running_loop()

        encoding = speech.RecognitionConfig.AudioEncoding.MULAW
        config_kwargs: dict[str, Any] = {
            "encoding": encoding,
            "sample_rate_hertz": sample_rate,
            "enable_automatic_punctuation": True,
            "model": "phone_call",
            "single_utterance": False,
            "enable_word_time_offsets": True,
        }
        if language:
            config_kwargs["language_code"] = language
            others = [c for c in self.candidates if c != language][:5]
            if others:
                config_kwargs["alternative_language_codes"] = others
        else:
            # Language identification: omit language_code, provide candidates.
            config_kwargs["alternative_language_codes"] = self.candidates[:8]

        config = speech.RecognitionConfig(**config_kwargs)
        self._config = config
        self._speech = speech
        self._stream_task = asyncio.create_task(self._stream_loop(), name="google-asr")

    async def _stream_loop(self) -> None:
        assert self._client is not None
        try:
            async for response in self._client.streaming_recognize(
                self._speech.StreamingRecognitionConfig(
                    config=self._config, interim_results=settings.asr_interim_results
                ),
                self._request_iterator(),
            ):
                for result in response.results:
                    if not result.alternatives:
                        continue
                    alt = result.alternatives[0]
                    await self._emit(
                        TranscriptSegment(
                            text=(alt.transcript or "").strip(),
                            is_final=result.is_final,
                            language=getattr(result, "language_code", None) or self.language,
                            confidence=float(alt.confidence or 0.0),
                            words=[
                                {
                                    "word": w.word,
                                    "start_ms": w.start_time.total_seconds() * 1000,
                                    "end_ms": w.end_time.total_seconds() * 1000,
                                }
                                for w in alt.words
                            ],
                            utterance_end=result.is_final,
                            provider=self.name,
                        )
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ASRError(f"google streaming failed: {exc}") from exc
        finally:
            await super().stop()

    def _request_iterator(self):
        speech = self._speech
        first = True
        while True:
            try:
                chunk = self._requests.get(timeout=30)
            except queue.Empty:
                continue
            if chunk is None:
                return
            if first:
                first = False
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    async def feed(self, audio: bytes, *, encoding: str = "mulaw") -> None:
        if encoding != "mulaw":
            raise ASRError("GoogleStreamingASR configured for mulaw telephone audio")
        self._requests.put_nowait(audio)

    async def stop(self) -> None:
        self._requests.put_nowait(None)
        if self._stream_task:
            self._stream_task.cancel()
        await super().stop()
