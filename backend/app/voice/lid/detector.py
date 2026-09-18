"""Language identification orchestrator.

Combines, in priority order:

1. **Explicit name** — the caller said "Hindi" / "हिंदी" / "मारवाड़ी". Confidence ~0.97.
2. **Acoustic LID** — Deepgram `language=multi`, Azure AutoDetectSourceLanguage,
   or Google `alternative_language_codes`. Real signal from the audio itself,
   independent of what the ASR happened to transcribe.
3. **Lexical/script LID** on the transcript (app.voice.lid.local).

Scores are fused (weighted, capped) so a weak acoustic guess confirmed by the
transcript crosses the confidence threshold, while a single weak signal does not.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import numpy as np

from ...config import settings
from ...i18n.languages import get_language
from ...voice.audio import wav_header
from .local import LIDResult, detect_language_name, detect_language_text

logger = logging.getLogger("nims.lid")

DEEPGRAM_LISTEN = "https://api.deepgram.com/v1/listen"


class AcousticLID:
    """Interface for audio-only language identification."""

    name = "acoustic"

    async def detect(self, pcm16: np.ndarray, sample_rate: int) -> LIDResult | None:
        raise NotImplementedError


class DeepgramAcousticLID(AcousticLID):  # pragma: no cover - network dependent
    """Prerecorded Deepgram request with `language=multi&detect_language=true`."""

    name = "deepgram"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.deepgram_api_key

    async def detect(self, pcm16: np.ndarray, sample_rate: int) -> LIDResult | None:
        if not self.api_key or pcm16.size < sample_rate // 4:
            return None
        wav = wav_header(int(pcm16.size), sample_rate) + pcm16.tobytes()
        params = {
            "model": "nova-2-general",
            "language": "multi",
            "detect_language": "true",
            "smart_format": "false",
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    DEEPGRAM_LISTEN,
                    params=params,
                    content=wav,
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": "audio/wav",
                    },
                )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("deepgram LID failed: %s", exc)
            return None

        channels = (payload.get("results") or {}).get("channels") or []
        if not channels:
            return None
        alt = (channels[0].get("alternatives") or [{}])[0]
        code = alt.get("detected_language")
        confidence = float(alt.get("language_confidence") or 0.0)
        if not code:
            return None
        return LIDResult(
            language=code,
            confidence=min(0.95, confidence),
            method="acoustic",
            scores={code: confidence},
            detail=f"deepgram acoustic LID conf={confidence:.2f} "
            f"transcript='{(alt.get('transcript') or '')[:40]}'",
        )


class AzureAcousticLID(AcousticLID):  # pragma: no cover - requires Azure SDK
    name = "azure"

    def __init__(self, candidates: list[str] | None = None) -> None:
        self.candidates = candidates or settings.supported_language_list[:10]

    async def detect(self, pcm16: np.ndarray, sample_rate: int) -> LIDResult | None:
        if not settings.azure_speech_key:
            return None

        def _run() -> LIDResult | None:
            try:
                import azure.cognitiveservices.speech as speechsdk  # type: ignore
            except ImportError:
                logger.warning("azure speech SDK not installed; skipping acoustic LID")
                return None

            stream = speechsdk.audio.PushAudioInputStream(
                speechsdk.audio.AudioStreamFormat(
                    samples_per_second=sample_rate, bits_per_sample=16, channels=1
                )
            )
            stream.write(pcm16.tobytes())
            stream.close()
            audio_config = speechsdk.audio.AudioConfig(stream=stream)
            auto = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=self.candidates
            )
            config = speechsdk.SpeechConfig(
                subscription=settings.azure_speech_key,
                region=settings.azure_speech_region,
            )
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=config,
                audio_config=audio_config,
                auto_detect_source_language_config=auto,
            )
            result = recognizer.recognize_once()
            detected = speechsdk.AutoDetectSourceLanguageResult.from_result(result)
            if not detected.language:
                return None
            return LIDResult(
                language=detected.language,
                confidence=0.85 if result.text else 0.6,
                method="acoustic",
                scores={detected.language: 0.85},
                detail=f"azure acoustic LID text='{(result.text or '')[:40]}'",
            )

        return await asyncio.to_thread(_run)


class GoogleAcousticLID(AcousticLID):  # pragma: no cover - requires GCP SDK
    name = "google"

    def __init__(self, candidates: list[str] | None = None) -> None:
        self.candidates = candidates or settings.supported_language_list[:8]

    async def detect(self, pcm16: np.ndarray, sample_rate: int) -> LIDResult | None:
        def _run() -> LIDResult | None:
            try:
                from google.cloud import speech  # type: ignore
            except ImportError:
                logger.warning("google-cloud-speech not installed; skipping acoustic LID")
                return None
            client = speech.SpeechClient()
            audio = speech.RecognitionAudio(content=pcm16.tobytes())
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                alternative_language_codes=self.candidates,
                model="phone_call",
            )
            response = client.recognize(config=config, audio=audio)
            for result in response.results:
                code = getattr(result, "language_code", None)
                if code and result.alternatives:
                    return LIDResult(
                        language=code,
                        confidence=min(0.95, float(result.alternatives[0].confidence or 0.7)),
                        method="acoustic",
                        scores={code: float(result.alternatives[0].confidence or 0.7)},
                        detail="google acoustic LID",
                    )
            return None

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            logger.warning("google LID failed: %s", exc)
            return None


def build_acoustic_lid(provider: str | None = None) -> AcousticLID | None:
    provider = (provider or settings.lid_provider).lower()
    if provider == "deepgram" and settings.deepgram_api_key:
        return DeepgramAcousticLID()
    if provider == "azure" and settings.azure_speech_key:
        return AzureAcousticLID()
    if provider == "google":
        return GoogleAcousticLID()
    return None


class LanguageIdentifier:
    """Per-call identifier: accumulates audio, fuses signals, tracks attempts."""

    #: acoustic evidence is worth more than text, but neither alone is decisive
    ACOUSTIC_WEIGHT = 0.62
    LEXICAL_WEIGHT = 0.48
    EXPLICIT_WEIGHT = 1.0

    def __init__(
        self,
        candidates: list[str] | None = None,
        acoustic: AcousticLID | None = None,
        threshold: float | None = None,
    ) -> None:
        self.candidates = tuple(candidates or settings.supported_language_list)
        self.acoustic = acoustic if acoustic is not None else build_acoustic_lid()
        self.threshold = threshold or settings.lid_confidence_threshold
        self.audio_chunks: list[np.ndarray] = []
        self.sample_rate = 8000
        self.attempts = 0
        self.history: list[dict[str, Any]] = []

    # -- audio accumulation ------------------------------------------------ #
    def add_audio(self, pcm16: np.ndarray, sample_rate: int = 8000) -> None:
        self.sample_rate = sample_rate
        self.audio_chunks.append(np.asarray(pcm16, dtype=np.int16))

    def buffered_audio(self) -> np.ndarray:
        if not self.audio_chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self.audio_chunks)

    def reset_audio(self) -> None:
        self.audio_chunks.clear()

    # -- identification ---------------------------------------------------- #
    async def identify(self, transcript: str | None = None) -> LIDResult:
        """Fuse every available signal for the current utterance."""
        self.attempts += 1
        scores: dict[str, float] = {}
        methods: list[str] = []
        details: list[str] = []

        # 1. explicit name in the transcript
        explicit = detect_language_name(transcript or "", self.candidates)
        if explicit:
            scores[explicit.language] = scores.get(explicit.language, 0.0) + (
                explicit.confidence * self.EXPLICIT_WEIGHT
            )
            methods.append(explicit.method)
            details.append(explicit.detail)

        # 2. lexical / script analysis of the transcript
        lexical: LIDResult | None = None
        if transcript and transcript.strip():
            lexical = detect_language_text(transcript, self.candidates, prefer_name_match=False)
            for code, value in lexical.scores.items():
                scores[code] = scores.get(code, 0.0) + (value / 12.0) * self.LEXICAL_WEIGHT
            methods.append(lexical.method)
            details.append(lexical.detail)

        # 3. acoustic LID over the buffered audio
        acoustic: LIDResult | None = None
        audio = self.buffered_audio()
        if self.acoustic is not None and audio.size > int(self.sample_rate * 0.4):
            acoustic = await self.acoustic.detect(audio, self.sample_rate)
            if acoustic:
                scores[acoustic.language] = scores.get(acoustic.language, 0.0) + (
                    acoustic.confidence * self.ACOUSTIC_WEIGHT
                )
                methods.append("acoustic")
                details.append(acoustic.detail)

        if not scores:
            result = LIDResult(
                language="en-IN",
                confidence=0.0,
                method="none",
                candidates=self.candidates,
                detail="no signal",
            )
        else:
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            top_code, top_score = ranked[0]
            total = sum(max(0.0, v) for v in scores.values()) or 1.0
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            confidence = min(0.99, (top_score / total) * 0.75 + (top_score / 1.6) * 0.45)
            if top_score - runner_up < 0.12:
                confidence *= 0.85  # ambiguous — force a re-prompt

            # That share is not comparable across scripts. Hindi and Marathi both
            # collect the Devanagari script vote, so the most a Marathi sentence
            # can reach is roughly a two-thirds share however many Marathi words
            # it contains, which pinned confidence near 0.65 — under the 0.82 a
            # mid-call switch needs. English collects the whole Latin vote alone
            # and sailed to 0.99 on the same rule, so a caller could switch into
            # English mid-call but never into Marathi, on a line whose campus is
            # in Maharashtra. Where the lexical detector picked the winner, trust
            # the confidence it calibrated against the raw marker scores, and let
            # the other signals only adjust it.
            if lexical is not None and lexical.language == top_code:
                confidence = lexical.confidence
                if explicit is not None and explicit.language == top_code:
                    confidence = max(confidence, explicit.confidence)
                if acoustic is not None:
                    confidence = (
                        min(0.99, confidence + 0.08)
                        if acoustic.language == top_code
                        else confidence * 0.7
                    )
                confidence = min(0.99, confidence)
            method = "+".join(dict.fromkeys(methods)) or "lexicon"
            result = LIDResult(
                language=top_code,
                confidence=round(confidence, 3),
                method=method,
                scores=scores,
                candidates=self.candidates,
                detail="; ".join(details)[:400],
            )

        self.history.append(
            {
                "attempt": self.attempts,
                "transcript": (transcript or "")[:120],
                **result.to_dict(),
            }
        )
        logger.info(
            "LID attempt=%d result=%s conf=%.2f method=%s",
            self.attempts, result.language, result.confidence, result.method,
        )
        return result

    def confirm(self, transcript: str) -> LIDResult | None:
        """Match a confirmation utterance ("yes", "haan", "ठीक है") to accept/reject."""
        text = (transcript or "").strip().lower()
        if not text:
            return None
        affirmative = {
            "yes", "yeah", "yep", "correct", "right", "ok", "okay", "sure", "fine",
            "हां", "हाँ", "ठीक", "ठीक है", "सही", "जी", "जी हाँ", "हां जी", "theek",
            "theek hai", "haan", "ha", "sahi", "sahi hai", "ji haan", "ji",
            "हो", "हांजी", "ठीकआ", "थिक", "thik",
        }
        negative = {
            "no", "nope", "wrong", "not", "incorrect", "नहीं", "ना", "गलत", "कोनी",
            "नाजी", "नहीं जी", "nahi", "nahin", "galat", "koni", "illa", "వద్దు",
        }
        tokens = set(text.replace(",", " ").replace(".", " ").split())
        if tokens & negative:
            return LIDResult(language="", confidence=0.9, method="confirmation",
                             detail="rejected")
        if tokens & affirmative:
            return LIDResult(
                language=self.history[-1]["language"] if self.history else "en-IN",
                confidence=0.95,
                method="confirmation",
                detail="accepted",
            )
        # caller named another language instead of confirming → treat as new answer
        return detect_language_name(text, self.candidates)

    def language_name(self, code: str) -> str:
        return get_language(code).english_name
