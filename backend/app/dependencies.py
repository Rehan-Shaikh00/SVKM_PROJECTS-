"""Process-wide singletons and FastAPI dependencies.

The retriever, answer engine and TTS engine are expensive to build (indexes,
websocket pools, model metadata), so they live for the lifetime of the process.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .ai.rag import AnswerEngine
from .config import settings
from .kb.retriever import HybridRetriever
from .orchestrator.call_logger import CallLogger, call_logger
from .voice.asr.registry import resolve_asr
from .voice.tts.registry import resolve_tts

logger = logging.getLogger("nims.deps")

_retriever: HybridRetriever | None = None
_retriever_lock = asyncio.Lock()
_answer_engine: AnswerEngine | None = None
_tts_engine = None
_tts_info: dict[str, Any] = {}
_asr_info: dict[str, Any] = {}
_kb_bootstrap: dict[str, Any] = {}


async def get_retriever() -> HybridRetriever:
    global _retriever
    async with _retriever_lock:
        if _retriever is None:
            _retriever = HybridRetriever()
            await _retriever.initialise()
        return _retriever


async def get_answer_engine() -> AnswerEngine:
    global _answer_engine
    if _answer_engine is None:
        retriever = await get_retriever()
        _answer_engine = AnswerEngine(retriever)
    return _answer_engine


def get_tts():
    global _tts_engine, _tts_info
    if _tts_engine is None:
        _tts_engine, _tts_info = resolve_tts()
    return _tts_engine


def tts_info() -> dict[str, Any]:
    get_tts()
    return _tts_info


def asr_info() -> dict[str, Any]:
    global _asr_info
    if not _asr_info:
        _engine, _asr_info = resolve_asr()
    return _asr_info


def get_call_logger() -> CallLogger:
    return call_logger


async def bootstrap_knowledge_base() -> dict[str, Any]:
    """Load/seed the KB once at startup."""
    global _kb_bootstrap
    if _kb_bootstrap:
        return _kb_bootstrap
    from .db import SessionLocal
    from .kb.ingest import bootstrap_knowledge_base as _bootstrap

    retriever = await get_retriever()
    async with SessionLocal() as session:
        _kb_bootstrap = await _bootstrap(session, retriever)
    return _kb_bootstrap


def kb_bootstrap_info() -> dict[str, Any]:
    return _kb_bootstrap


async def reset_singletons() -> None:
    """Used by tests."""
    global _retriever, _answer_engine, _tts_engine, _tts_info, _asr_info, _kb_bootstrap
    _retriever = None
    _answer_engine = None
    _tts_engine = None
    _tts_info = {}
    _asr_info = {}
    _kb_bootstrap = {}


def provider_summary() -> dict[str, Any]:
    """Everything the dashboard shows about how the system is currently wired."""
    engine = get_tts()
    return {
        "configured": settings.provider_status(),
        "active": {
            "asr": asr_info().get("active"),
            "tts": engine.name,
            "llm": (_answer_engine.llm_info.get("active") if _answer_engine else settings.llm_provider),
            "embeddings": (_retriever.embedder.name if _retriever and _retriever.embedder else settings.embedding_provider),
            "vector_store": (_retriever.store.name if _retriever and _retriever.store else settings.vector_store),
            "lid": settings.lid_provider,
            "telephony": settings.telephony_provider if settings.twilio_configured else "simulator",
        },
        "languages": settings.supported_language_list,
        "greeting_languages": settings.greeting_language_list,
        "academic_year": settings.kb_academic_year,
        "tts_text_only": getattr(engine, "text_only", True),
    }
