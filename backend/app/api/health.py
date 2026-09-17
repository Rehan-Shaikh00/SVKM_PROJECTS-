"""Health, readiness and provider diagnostics."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from ..config import settings
from ..db import SessionLocal
from ..dependencies import get_retriever, get_tts, provider_summary
from ..models import KBChunk, KBRecord
from ..observability.logging import metrics
from ..telephony.session_runner import active_call_count
from .auth import require_admin

router = APIRouter(tags=["health"])

START_TIME = time.time()


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe — intentionally cheap (no DB access)."""
    return {
        "status": "ok",
        "environment": settings.environment,
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "active_calls": active_call_count(),
    }


@router.get("/health/ready")
async def ready() -> dict[str, Any]:
    """Readiness probe: DB reachable and the KB index has content."""
    problems: list[str] = []
    records = chunks = 0
    try:
        async with SessionLocal() as session:
            records = (await session.execute(select(func.count(KBRecord.id)))).scalar_one()
            chunks = (await session.execute(select(func.count(KBChunk.id)))).scalar_one()
    except Exception as exc:
        problems.append(f"database: {exc}")
    if records == 0:
        problems.append("knowledge base has no records")
    return {
        "status": "ready" if not problems else "degraded",
        "records": int(records),
        "chunks": int(chunks),
        "problems": problems,
    }


@router.get("/health/providers")
async def providers(_: str = Depends(require_admin)) -> dict[str, Any]:
    """What is configured, what is actually active, and what degraded."""
    engine = get_tts()
    retriever = await get_retriever()
    return {
        "summary": provider_summary(),
        "tts": {
            "provider": engine.name,
            "text_only": getattr(engine, "text_only", True),
            "synthesis_count": getattr(engine, "synthesis_count", 0),
            "last_latency_ms": round(getattr(engine, "last_latency_ms", 0.0), 1),
        },
        "retrieval": await retriever.stats(),
        "asr": {
            "configured": settings.asr_provider,
            "endpoint_silence_ms": settings.asr_endpoint_silence_ms,
            "interim_results": settings.asr_interim_results,
        },
        "lid": {
            "configured": settings.lid_provider,
            "threshold": settings.lid_confidence_threshold,
            "greeting_languages": settings.greeting_language_list,
            "supported_languages": settings.supported_language_list,
            "dtmf_fallback_allowed": settings.allow_dtmf_fallback,
        },
        "llm": {
            "configured": settings.llm_provider,
            "model": settings.anthropic_model if settings.llm_provider == "anthropic"
            else settings.openai_model,
            "streaming": settings.llm_streaming,
            "confidence_threshold": settings.answer_confidence_threshold,
        },
        "telephony": {
            "provider": settings.telephony_provider,
            "twilio_configured": settings.twilio_configured,
            "escalation_agents": len(settings.escalation_agent_list),
            "queue": settings.escalation_queue_name,
            "recording_consent": settings.recording_consent_announce,
            "call_recording": settings.call_recording_enabled,
        },
        "compliance": {
            "pii_redaction": settings.redact_pii,
            "recording_storage": settings.allow_call_recording_storage,
            "max_call_minutes": settings.max_call_minutes,
        },
        "metrics": metrics.snapshot(),
    }


@router.get("/metrics")
async def metrics_endpoint(_: str = Depends(require_admin)) -> dict[str, Any]:
    return metrics.snapshot()
