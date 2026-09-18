"""ASR registry — resolves ASR_PROVIDER into a concrete engine.

Providers are tried in the configured order with graceful degradation:
if the configured engine cannot start (missing key/SDK), the next one is used
and the degradation is recorded so the dashboard shows it.
"""

from __future__ import annotations

import logging
from typing import Any

from ...config import settings
from .base import ASRError, NullASR, StreamingASR
from .client import ClientASR, WhisperChunkASR

logger = logging.getLogger("nims.asr")

#: Fallback order used when the configured provider is unavailable.
FALLBACK_ORDER: tuple[str, ...] = (
    "deepgram",
    "assemblyai",
    "google",
    "azure",
    "whisper",
    "client",
    "none",
)


def build_asr(provider: str | None = None) -> StreamingASR:
    """Instantiate an ASR engine by name (does not connect yet)."""
    provider = (provider or settings.asr_provider or "client").lower()

    if provider == "deepgram":
        from .deepgram import DeepgramStreamingASR

        if not settings.deepgram_api_key:
            raise ASRError("DEEPGRAM_API_KEY missing")
        return DeepgramStreamingASR()

    if provider == "assemblyai":
        from .assemblyai import AssemblyAIStreamingASR

        if not settings.assemblyai_api_key:
            raise ASRError("ASSEMBLYAI_API_KEY missing")
        return AssemblyAIStreamingASR()

    if provider == "google":
        from .google import GoogleStreamingASR

        if not settings.google_application_credentials:
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set; relying on ADC")
        return GoogleStreamingASR(candidates=settings.supported_language_list)

    if provider == "azure":
        from .azure import AzureStreamingASR

        if not settings.azure_speech_key:
            raise ASRError("AZURE_SPEECH_KEY missing")
        return AzureStreamingASR(candidates=settings.supported_language_list)

    if provider == "whisper":
        if not settings.openai_api_key:
            raise ASRError("OPENAI_API_KEY missing")
        return WhisperChunkASR()

    if provider == "client":
        return ClientASR()

    return NullASR()


def resolve_asr(requested: str | None = None) -> tuple[StreamingASR, dict[str, Any]]:
    """Build the best available engine, degrading through FALLBACK_ORDER.

    Returns (engine, info) where info records what was requested vs used.
    """
    requested = (requested or settings.asr_provider or "client").lower()
    attempts: list[str] = []
    order = [requested] + [p for p in FALLBACK_ORDER if p != requested]

    for candidate in order:
        try:
            engine = build_asr(candidate)
            info = {
                "requested": requested,
                "active": candidate,
                "degraded": candidate != requested,
                "attempts": attempts,
                "streaming": engine.streaming,
                "accepts_mulaw": engine.accepts_mulaw,
            }
            if info["degraded"]:
                logger.warning("ASR degraded: requested=%s active=%s", requested, candidate)
            return engine, info
        except ASRError as exc:
            attempts.append(f"{candidate}:{exc}")
            logger.info("ASR provider %s unavailable: %s", candidate, exc)
            continue

    return ClientASR(), {
        "requested": requested,
        "active": "client",
        "degraded": True,
        "attempts": attempts,
        "streaming": True,
        "accepts_mulaw": True,
    }


def languages_for(provider_name: str) -> tuple[str, ...]:
    engine, _ = resolve_asr(provider_name)
    return engine.supported_languages
