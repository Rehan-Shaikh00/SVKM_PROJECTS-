"""Azure Speech adapter (streaming ASR + automatic language identification).

Azure is the strongest option for the "no keypad" language-selection step:
`AutoDetectSourceLanguageConfig` runs real acoustic language ID over the
candidate locales and returns a confidence per result, and the same SDK gives
neural TTS in those languages.

Requires `azure-cognitiveservices-speech`. Audio is fed as 16-bit PCM at 8 kHz
(we decode μ-law before pushing).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from ...config import settings
from ...voice.audio import mulaw_to_pcm16
from .base import ASRError, StreamingASR, TranscriptSegment

SUPPORTED = (
    "en-IN", "hi-IN", "ta-IN", "bn-IN", "mr-IN", "gu-IN", "te-IN", "kn-IN",
    "ml-IN", "pa-IN", "ur-IN", "or-IN", "as-IN",
)


class AzureStreamingASR(StreamingASR):  # pragma: no cover - requires Azure SDK + key
    name = "azure"
    accepts_mulaw = False
    preferred_rate = 8000
    supported_languages = SUPPORTED

    def __init__(self, candidates: list[str] | None = None) -> None:
        super().__init__()
        self.candidates = candidates or list(SUPPORTED)
        self._sdk: Any = None
        self._recognizer: Any = None
        self._stream: Any = None
        self._lock = threading.Lock()

    async def start(self, language: str | None = None, sample_rate: int = 8000) -> None:
        try:
            import azure.cognitiveservices.speech as speechsdk  # type: ignore
        except ImportError as exc:
            raise ASRError(
                "azure-cognitiveservices-speech is not installed; add it to requirements.txt"
            ) from exc

        if not settings.azure_speech_key or not settings.azure_speech_region:
            raise ASRError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION not configured")

        self._sdk = speechsdk
        await super().start(language, sample_rate)

        audio_config = speechsdk.audio.AudioConfig(
            stream=speechsdk.audio.PushAudioInputStream(
                speechsdk.audio.AudioStreamFormat(
                    samples_per_second=sample_rate,
                    bits_per_sample=16,
                    channels=1,
                )
            )
        )
        self._stream = audio_config.stream

        if language:
            source_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=[language] + [c for c in self.candidates if c != language][:3]
            )
        else:
            source_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=self.candidates[:10]
            )

        speech_config = speechsdk.SpeechConfig(
            subscription=settings.azure_speech_key, region=settings.azure_speech_region
        )
        speech_config.speech_recognition_language = language or "en-IN"
        speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            str(settings.asr_endpoint_silence_ms),
        )

        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=source_config,
        )
        self._recognizer.recognizing.connect(self._on_recognizing)
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.session_stopped.connect(lambda *_: self._schedule_stop())
        self._recognizer.start_continuous_recognition()

    # -- callbacks (called on the SDK thread) ------------------------------- #
    def _on_recognizing(self, evt: Any) -> None:
        text = getattr(evt.result, "text", "") or ""
        if not text.strip():
            return
        self._emit_threadsafe(
            TranscriptSegment(
                text=text.strip(),
                is_final=False,
                language=self._detected_language(evt),
                provider=self.name,
            )
        )

    def _on_recognized(self, evt: Any) -> None:
        result = evt.result
        reason = self._sdk.SpeechRecognitionReason
        if result.reason == reason.NoMatch:
            self._emit_threadsafe(
                TranscriptSegment(text="", is_final=True, utterance_end=True, provider=self.name)
            )
            return
        text = (result.text or "").strip()
        self._emit_threadsafe(
            TranscriptSegment(
                text=text,
                is_final=True,
                language=self._detected_language(evt),
                confidence=float(
                    getattr(result, "properties", {})
                    .get(self._sdk.PropertyId.SpeechServiceResponse_Confidence, 0)
                    or 0
                ),
                utterance_end=True,
                provider=self.name,
            )
        )

    def _detected_language(self, evt: Any) -> str | None:
        try:
            auto = self._sdk.AutoDetectSourceLanguageResult.from_result(evt.result)
            return auto.language or self.language
        except Exception:
            return self.language

    def _emit_threadsafe(self, segment: TranscriptSegment) -> None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(self._emit(segment), loop)

    def _schedule_stop(self) -> None:
        self._emit_threadsafe(TranscriptSegment(text="", is_final=True, provider=self.name))

    async def feed(self, audio: bytes, *, encoding: str = "mulaw") -> None:
        if self._stream is None:
            raise ASRError("AzureStreamingASR not started")
        pcm = mulaw_to_pcm16(audio) if encoding == "mulaw" else audio
        with self._lock:
            self._stream.write(pcm.tobytes())

    async def stop(self) -> None:  # pragma: no cover
        try:
            if self._recognizer:
                self._recognizer.stop_continuous_recognition()
            if self._stream:
                self._stream.close()
        except Exception:
            pass
        await super().stop()
