"""Cloud TTS providers — Google, Azure, ElevenLabs.

All three are plain REST calls (no vendor SDK required), and all three can emit
telephone-ready audio:

* Google  → LINEAR16 @ 8 kHz with the `telephony-bandwidth` effect profile
* Azure   → `raw-8khz-8bit-1chan-mulaw` (native μ-law, zero transcoding)
* ElevenLabs → `pcm_16000`, resampled here to 8 kHz

Voices are chosen per language from app.i18n.languages.voice_for().
"""

from __future__ import annotations

import base64
import html
import time

import httpx

from ...config import settings
from ...i18n.languages import get_language, tts_locale, voice_for
from ..audio import pcm16_to_mulaw, resample
from .base import TTS, SynthesisResult, sanitize_for_speech

GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class GoogleTTS(TTS):  # pragma: no cover - network dependent
    name = "google"
    encodings = ("mulaw", "pcm16")
    supported_languages = tuple(
        get_language(c).code
        for c in settings.supported_language_list
    )

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self.api_key = api_key or settings.google_tts_api_key

    async def synthesize(
        self, text: str, language: str = "en-IN", *, sample_rate: int = 8000,
        encoding: str = "mulaw", speaking_rate: float | None = None,
    ) -> SynthesisResult:
        if not self.api_key:
            raise RuntimeError("GOOGLE_TTS_API_KEY not configured")
        clean = sanitize_for_speech(text)
        if not clean:
            return SynthesisResult(text="", provider=self.name, text_only=True, is_last=True)

        rate = sample_rate if encoding == "pcm16" else 8000
        body = {
            "input": {"text": clean},
            "voice": {
                "languageCode": tts_locale(language),
                "name": voice_for(language, "google"),
                "ssmlGender": "NEUTRAL",
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": rate,
                "speakingRate": speaking_rate or settings.tts_speaking_rate,
                "pitch": 0,
                "effectsProfileId": ["telephony-bandwidth"],
            },
        }
        started = time.time()
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                GOOGLE_TTS_URL, params={"key": self.api_key}, json=body
            )
            resp.raise_for_status()
            payload = resp.json()
        pcm = base64.b64decode(payload.get("audioContent", ""))
        latency = self._timed(started)
        return SynthesisResult(
            audio=pcm16_to_mulaw(__import__("numpy").frombuffer(pcm, dtype="<i2"))
            if encoding == "mulaw" else pcm,
            encoding="mulaw" if encoding == "mulaw" else "pcm16",
            sample_rate=rate,
            text=clean,
            voice=body["voice"]["name"],
            provider=self.name,
            language=language,
            latency_ms=latency,
            characters=len(clean),
        )


class AzureTTS(TTS):  # pragma: no cover - network dependent
    """Azure Neural TTS. Native μ-law output at 8 kHz = no transcoding at all."""

    name = "azure"
    encodings = ("mulaw", "pcm16")

    def __init__(self, key: str | None = None, region: str | None = None) -> None:
        super().__init__()
        self.key = key or settings.azure_speech_key
        self.region = region or settings.azure_speech_region

    @property
    def endpoint(self) -> str:
        return (
            f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        )

    async def synthesize(
        self, text: str, language: str = "en-IN", *, sample_rate: int = 8000,
        encoding: str = "mulaw", speaking_rate: float | None = None,
    ) -> SynthesisResult:
        if not self.key or not self.region:
            raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION not configured")
        clean = sanitize_for_speech(text)
        if not clean:
            return SynthesisResult(text="", provider=self.name, text_only=True, is_last=True)

        output_format = (
            "raw-8khz-8bit-1chan-mulaw" if encoding == "mulaw" else "raw-16khz-16bit-mono-pcm"
        )
        rate_pct = int((speaking_rate or settings.tts_speaking_rate) * 100) - 100
        ssml = (
            "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xml:lang='{tts_locale(language)}'>"
            f"<voice name='{voice_for(language, 'azure')}'>"
            f"<prosody rate='{rate_pct:+d}%'>{html.escape(clean)}</prosody>"
            "</voice></speak>"
        )
        started = time.time()
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                self.endpoint,
                content=ssml.encode("utf-8"),
                headers={
                    "Ocp-Apim-Subscription-Key": self.key,
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": output_format,
                    "User-Agent": "NIMSVoiceAssistant",
                },
            )
            resp.raise_for_status()
            audio = resp.content
        return SynthesisResult(
            audio=audio,
            encoding="mulaw" if encoding == "mulaw" else "pcm16",
            sample_rate=8000 if encoding == "mulaw" else 16000,
            text=clean,
            voice=voice_for(language, "azure"),
            provider=self.name,
            language=language,
            latency_ms=self._timed(started),
            characters=len(clean),
        )


class ElevenLabsTTS(TTS):  # pragma: no cover - network dependent
    """Highest naturalness; `eleven_multilingual_v2` covers all Indian languages
    we serve from a single voice, which keeps the assistant's persona stable
    when the caller switches language."""

    name = "elevenlabs"
    encodings = ("mulaw", "pcm16")

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        super().__init__()
        self.api_key = api_key or settings.elevenlabs_api_key
        self.model = model or settings.elevenlabs_model
        self.voice_id = settings.elevenlabs_voice_id

    async def synthesize(
        self, text: str, language: str = "en-IN", *, sample_rate: int = 8000,
        encoding: str = "mulaw", speaking_rate: float | None = None,
    ) -> SynthesisResult:
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not configured")
        clean = sanitize_for_speech(text)
        if not clean:
            return SynthesisResult(text="", provider=self.name, text_only=True, is_last=True)

        stability = 0.45
        similarity = 0.75
        speed = float(speaking_rate or settings.tts_speaking_rate)
        body = {
            "text": clean,
            "model_id": self.model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity,
                "style": 0.15,
                "use_speaker_boost": True,
                "speed": max(0.7, min(1.2, speed)),
            },
        }
        started = time.time()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                ELEVENLABS_URL.format(voice_id=self.voice_id),
                params={"output_format": "pcm_16000"},
                json=body,
                headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            pcm = resp.content
        latency = self._timed(started)

        import numpy as np

        audio16 = np.frombuffer(pcm, dtype="<i2")
        audio8 = resample(audio16, 16000, sample_rate).astype(np.int16)
        return SynthesisResult(
            audio=pcm16_to_mulaw(audio8) if encoding == "mulaw" else audio8.tobytes(),
            encoding="mulaw" if encoding == "mulaw" else "pcm16",
            sample_rate=sample_rate if encoding == "mulaw" else 16000,
            text=clean,
            voice=self.voice_id,
            provider=self.name,
            language=language,
            latency_ms=latency,
            characters=len(clean),
        )
