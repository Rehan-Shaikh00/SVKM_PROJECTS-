"""Call log API — transcripts and quality review for the university team."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..ai.guardrails import hash_secret
from ..db import get_session
from ..models import (
    CallRecord,
    CallStatus,
    CallTurn,
    EscalationEvent,
    FollowUp,
    TurnRole,
)
from ..telephony.session_runner import ACTIVE_SESSIONS
from .auth import require_admin

logger = logging.getLogger("nims.api.calls")

router = APIRouter(prefix="/calls", tags=["calls"])


def _call_row(record: CallRecord) -> dict[str, Any]:
    started = record.started_at
    if started and started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return {
        "id": record.id,
        "provider": record.provider,
        "provider_call_sid": record.provider_call_sid,
        "status": record.status,
        "end_reason": record.end_reason,
        "from_number": record.from_number,
        "to_number": record.to_number,
        "started_at": started.isoformat() if started else None,
        "duration_seconds": round(record.duration_seconds or 0.0, 1),
        "detected_language": record.detected_language,
        "language_confidence": record.language_confidence,
        "language_method": record.language_method,
        "language_attempts": record.language_attempts,
        "resolved": record.resolved,
        "escalated": record.escalated,
        "escalation_reason": record.escalation_reason,
        "turn_count": record.turn_count,
        "unanswered_count": record.unanswered_count,
        "primary_intent": record.primary_intent,
        "barge_in_count": record.barge_in_count,
        "time_to_first_audio_ms": round(record.time_to_first_audio_ms or 0.0, 1),
        "avg_response_latency_ms": round(record.avg_response_latency_ms or 0.0, 1),
        "satisfaction": record.satisfaction,
        "summary": record.summary,
    }


@router.get("")
async def list_calls(
    status: str | None = Query(None),
    language: str | None = Query(None),
    escalated: bool | None = Query(None),
    resolved: bool | None = Query(None),
    search: str | None = Query(None),
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(days=days)
    query = select(CallRecord).where(CallRecord.started_at >= since)
    count_query = select(func.count(CallRecord.id)).where(CallRecord.started_at >= since)

    if status:
        query = query.where(CallRecord.status == status)
        count_query = count_query.where(CallRecord.status == status)
    if language:
        query = query.where(CallRecord.detected_language == language)
        count_query = count_query.where(CallRecord.detected_language == language)
    if escalated is not None:
        query = query.where(CallRecord.escalated.is_(escalated))
        count_query = count_query.where(CallRecord.escalated.is_(escalated))
    if resolved is not None:
        query = query.where(CallRecord.resolved.is_(resolved))
        count_query = count_query.where(CallRecord.resolved.is_(resolved))
    if search:
        like = f"%{search.strip()}%"
        condition = or_(
            CallRecord.summary.ilike(like),
            CallRecord.transcript_text.ilike(like),
            CallRecord.id.ilike(like),
            CallRecord.provider_call_sid.ilike(like),
        )
        query = query.where(condition)
        count_query = count_query.where(condition)

    total = (await session.execute(count_query)).scalar_one()
    rows = (
        await session.execute(
            query.order_by(CallRecord.started_at.desc()).limit(limit).offset(offset)
        )
    ).scalars().all()
    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "items": [_call_row(r) for r in rows],
        "statuses": [s.value for s in CallStatus],
    }


@router.get("/live")
async def live_calls(_: str = Depends(require_admin)) -> dict[str, Any]:
    calls = []
    for call_id, session in ACTIVE_SESSIONS.items():
        calls.append(
            {
                "call_id": call_id,
                "state": session.state.value,
                "language": session.language,
                "turns": len(session.transcript),
                "barge_ins": session.ctx.barge_in_count,
                "silence_count": session.ctx.silence_count,
                "provider": session.ctx.provider,
                "escalated": session.escalated,
            }
        )
    return {"active": len(calls), "calls": calls}


@router.get("/{call_id}")
async def get_call(
    call_id: str,
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    record = await session.get(CallRecord, call_id)
    if record is None:
        raise HTTPException(status_code=404, detail="call not found")

    turns = (
        await session.execute(
            select(CallTurn).where(CallTurn.call_id == call_id).order_by(CallTurn.seq, CallTurn.id)
        )
    ).scalars().all()
    escalations = (
        await session.execute(
            select(EscalationEvent)
            .where(EscalationEvent.call_id == call_id)
            .order_by(EscalationEvent.requested_at)
        )
    ).scalars().all()
    followups = (
        await session.execute(select(FollowUp).where(FollowUp.call_id == call_id))
    ).scalars().all()

    payload = _call_row(record)
    payload.update(
        {
            "recording_uri": record.recording_uri if record.recording_uri else None,
            "recording_consent": record.recording_consent,
            "intents": record.intents_json or [],
            "metadata": record.metadata_json or {},
            "transcript": [
                {
                    "seq": turn.seq,
                    "role": turn.role,
                    "text": turn.text,
                    "language": turn.language,
                    "interrupted": turn.interrupted,
                    "grounded": turn.grounded,
                    "confidence": turn.confidence,
                    "timings": {
                        "asr_ms": turn.asr_ms,
                        "retrieval_ms": turn.retrieval_ms,
                        "llm_ms": turn.llm_ms,
                        "tts_ms": turn.tts_ms,
                        "total_ms": turn.total_ms,
                    },
                    "citations": turn.citations_json or [],
                    "tool_calls": turn.tool_calls_json or [],
                    "metadata": turn.metadata_json or {},
                    "at": turn.started_at.isoformat() if turn.started_at else None,
                }
                for turn in turns
            ],
            "escalations": [
                {
                    "id": e.id,
                    "reason": e.reason,
                    "target": e.target,
                    "target_type": e.target_type,
                    "outcome": e.outcome,
                    "wait_seconds": e.wait_seconds,
                    "whisper_summary": e.whisper_summary,
                    "context": e.context_json or {},
                    "requested_at": e.requested_at.isoformat() if e.requested_at else None,
                }
                for e in escalations
            ],
            "followups": [
                {
                    "id": f.id,
                    "channel": f.channel,
                    "status": f.status,
                    "body": f.body,
                    "destination_ref": f.destination_ref,
                    "error": f.error,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in followups
            ],
        }
    )
    return payload


@router.get("/{call_id}/transcript.txt", response_class=PlainTextResponse)
async def transcript_text(
    call_id: str,
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> str:
    record = await session.get(CallRecord, call_id)
    if record is None:
        raise HTTPException(status_code=404, detail="call not found")
    if record.transcript_text:
        return record.transcript_text
    turns = (
        await session.execute(
            select(CallTurn).where(CallTurn.call_id == call_id).order_by(CallTurn.seq)
        )
    ).scalars().all()
    return "\n".join(f"{t.role}: {t.text}" for t in turns)


@router.post("/{call_id}/review")
async def review_call(
    call_id: str,
    satisfaction: str = Query("resolved_ok"),
    note: str = Query(""),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    """Manual QA annotation — feeds the knowledge-base backlog."""
    record = await session.get(CallRecord, call_id)
    if record is None:
        raise HTTPException(status_code=404, detail="call not found")
    record.satisfaction = satisfaction
    metadata = dict(record.metadata_json or {})
    reviews = metadata.setdefault("reviews", [])
    reviews.append(
        {"at": datetime.now(UTC).isoformat(), "rating": satisfaction, "note": note[:500]}
    )
    record.metadata_json = metadata
    await session.commit()
    return {"ok": True, "call_id": call_id, "satisfaction": satisfaction}


@router.get("/{call_id}/handoff-brief")
async def handoff_brief(
    call_id: str,
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    """The context package that would be handed to a human agent."""
    record = await session.get(CallRecord, call_id)
    if record is None:
        raise HTTPException(status_code=404, detail="call not found")
    escalation = (
        await session.execute(
            select(EscalationEvent)
            .where(EscalationEvent.call_id == call_id)
            .order_by(EscalationEvent.requested_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return {
        "call_id": call_id,
        "language": record.detected_language,
        "summary": record.summary,
        "whisper": escalation.whisper_summary if escalation else None,
        "escalation_reason": record.escalation_reason,
        "primary_intent": record.primary_intent,
        "caller_ref": record.from_number,
        "unresolved": (record.metadata_json or {}).get("unresolved", []),
        "transcript_tail": (record.transcript_text or "").splitlines()[-8:],
    }


def caller_reference(number: str) -> str:
    """Helper used by tests: the pseudonymised caller id stored on a call."""
    return f"hash:{hash_secret(number)}"


def assistant_turn_count(turns: list[CallTurn]) -> int:
    return sum(1 for t in turns if t.role == TurnRole.ASSISTANT.value)
