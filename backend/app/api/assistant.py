"""Stateless assistant endpoints.

`POST /assistant/query` runs one turn through the *same* retrieval + generation
pipeline a phone call uses, without any telephony. It is what the dashboard's
"test a question" panel and the QA scripts call, and it is the fastest way to
validate the knowledge base after an edit.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..ai.intents import detect_intent
from ..ai.rag import AnswerRequest
from ..db import get_session
from ..dependencies import get_answer_engine, get_retriever
from ..i18n.languages import LANGUAGES, get_language
from .auth import require_admin

logger = logging.getLogger("nims.assistant")

router = APIRouter(prefix="/assistant", tags=["assistant"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=600)
    language: str = Field("en-IN")
    call_id: str = Field("api-query")
    history: list[dict[str, str]] = Field(default_factory=list)
    top_k: int = Field(6, ge=1, le=12)
    explain: bool = Field(False, description="include retrieval + grounding debug info")


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=600)
    language: str = "en-IN"
    top_k: int = Field(8, ge=1, le=20)
    category: str | None = None
    verified_only: bool = False


@router.post("/query")
async def query(
    body: QueryRequest,
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    language = get_language(body.language).code
    engine = await get_answer_engine()
    started = time.perf_counter()

    from ..ai.llm.base import Message

    history = [
        Message(role=m.get("role", "user"), content=m.get("content", ""))  # type: ignore[arg-type]
        for m in body.history[-8:]
    ]
    answer = await engine.answer(
        AnswerRequest(
            call_id=body.call_id,
            question=body.question,
            language=language,
            history=history,
            stage="api_query",
        ),
        session,
    )
    payload: dict[str, Any] = {
        "question": body.question,
        "language": language,
        "answer": answer.text,
        "grounded": answer.grounded,
        "confidence": answer.confidence,
        "intent": answer.intent,
        "needs_escalation": answer.needs_escalation,
        "escalation_reason": answer.escalation_reason,
        "citations": answer.citations,
        "followup": answer.followup,
        "provider": answer.provider,
        "fallback_used": answer.fallback_used,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "retrieval_ms": round(answer.retrieval_ms, 1),
        "llm_ms": round(answer.llm_ms, 1),
        "warnings": answer.warnings,
    }
    if body.explain:
        payload["debug"] = {
            **answer.debug,
            "retrieval": {
                "query": answer.retrieval.query if answer.retrieval else body.question,
                "best_score": answer.retrieval.best_score if answer.retrieval else 0.0,
                "grounded": answer.retrieval.grounded if answer.retrieval else False,
                "items": [
                    {
                        "title": item.title,
                        "category": item.category,
                        "score": item.score,
                        "dense_score": item.dense_score,
                        "lexical_score": item.lexical_score,
                        "dense_rank": item.dense_rank,
                        "lexical_rank": item.lexical_rank,
                        "signals": item.signals,
                        "snippet": item.text[:300],
                    }
                    for item in (answer.retrieval.items if answer.retrieval else [])
                ],
            },
        }
    return payload


@router.post("/retrieve")
async def retrieve(body: RetrievalRequest, _: str = Depends(require_admin)) -> dict[str, Any]:
    """Debug endpoint: show exactly what the retriever returns and why."""
    retriever = await get_retriever()
    intent = detect_intent(body.query)
    result = await retriever.search(
        body.query,
        language=body.language,
        intent=intent,
        top_k=body.top_k,
        categories=(body.category,) if body.category else None,
        verified_only=body.verified_only,
    )
    return {
        "query": body.query,
        "intent": intent.to_dict(),
        "grounded": result.grounded,
        "best_score": result.best_score,
        "latency_ms": round(result.latency_ms, 2),
        "debug": result.debug,
        "items": [
            {
                "chunk_id": item.chunk_id,
                "record_id": item.record_id,
                "title": item.title,
                "category": item.category,
                "language": item.language,
                "verified": item.verified,
                "academic_year": item.academic_year,
                "score": item.score,
                "dense_score": item.dense_score,
                "lexical_score": item.lexical_score,
                "signals": item.signals,
                "citation": item.citation,
                "text": item.text,
            }
            for item in result.items
        ],
        "context_block": result.context_block(max_chars=3000),
    }


@router.get("/languages")
async def languages() -> dict[str, Any]:
    return {
        "languages": [
            {
                "code": lang.code,
                "english_name": lang.english_name,
                "native_name": lang.native_name,
                "asr_locale": lang.asr_locale or lang.code,
                "lid_supported": lang.lid_supported,
                "fallback": lang.fallback_code,
                "voices": lang.voices,
            }
            for lang in LANGUAGES.values()
        ]
    }
