"""TTS registry with graceful degradation."""

from __future__ import annotations

import logging
from typing import Any

from ...config import settings
from .base import TTS
from .cloud import AzureTTS, ElevenLabsTTS, GoogleTTS
from .local import ClientTTS, LocalTTS

logger = logging.getLogger("nims.tts")

FALLBACK_ORDER = ("google", "azure", "elevenlabs", "local", "client")


def build_tts(provider: str | None = None) -> TTS:
    provider = (provider or settings.tts_provider or "client").lower()
    if provider == "google":
        if not settings.google_tts_api_key:
            raise RuntimeError("GOOGLE_TTS_API_KEY missing")
        return GoogleTTS()
    if provider == "azure":
        if not settings.azure_speech_key or not settings.azure_speech_region:
            raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION missing")
        return AzureTTS()
    if provider == "elevenlabs":
        if not settings.elevenlabs_api_key:
            raise RuntimeError("ELEVENLABS_API_KEY missing")
        return ElevenLabsTTS()
    if provider == "local":
        return LocalTTS()
    return ClientTTS()


def resolve_tts(requested: str | None = None) -> tuple[TTS, dict[str, Any]]:
    requested = (requested or settings.tts_provider or "client").lower()
    attempts: list[str] = []
    for candidate in [requested] + [p for p in FALLBACK_ORDER if p != requested]:
        try:
            engine = build_tts(candidate)
            info = {
                "requested": requested,
                "active": candidate,
                "degraded": candidate != requested,
                "text_only": engine.text_only,
                "attempts": attempts,
            }
            if info["degraded"]:
                logger.warning("TTS degraded: requested=%s active=%s", requested, candidate)
            return engine, info
        except Exception as exc:
            attempts.append(f"{candidate}:{exc}")
            continue
    return ClientTTS(), {"requested": requested, "active": "client", "degraded": True,
                         "text_only": True, "attempts": attempts}
