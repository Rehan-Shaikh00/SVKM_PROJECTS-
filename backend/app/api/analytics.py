"""Analytics for the university team.

Answers the questions the admissions office actually asks:
how many calls, in which languages, what did people ask, what could the AI not
answer, how fast was it, and how often did a human have to take over.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..ai.intents import detect_intent
from ..db import get_session
from ..kb.chunking import canonical_key
from ..models import (
    CallRecord,
    CallStatus,
    CallTurn,
    EscalationEvent,
    FollowUp,
    KBRecord,
    TurnRole,
    UnansweredQuestion,
)
from ..observability.logging import metrics
from .auth import require_admin

logger = logging.getLogger("nims.analytics")

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _since(days: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


# Turns captured while the call was still identifying the language are the
# caller's *answer* to "which language?" — usually a bare language name. They
# are not questions, and "hindi"/"english" legitimately look like the B.A.
# Hindi / B.A. English specialisations to the intent classifier, so counting
# them would invent a phantom top query.
_LANGUAGE_STAGES = frozenset(
    {"language_prompt", "language_reprompt", "language_confirm", "language_unsupported"}
)


def _is_language_turn(metadata: Any) -> bool:
    if not isinstance(metadata, dict):
        return False
    return str(metadata.get("stage") or "").lower() in _LANGUAGE_STAGES


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(pct * (len(ordered) - 1)))))
    return round(ordered[index], 1)


@router.get("/overview")
async def overview(
    days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    since = _since(days)
    rows = (
        await session.execute(select(CallRecord).where(CallRecord.started_at >= since))
    ).scalars().all()

    total = len(rows)
    escalated = sum(1 for r in rows if r.escalated)
    resolved = sum(1 for r in rows if r.resolved and not r.escalated)
    failed = sum(1 for r in rows if r.status == CallStatus.FAILED.value)
    abandoned = sum(1 for r in rows if r.status == CallStatus.ABANDONED.value)
    durations = [r.duration_seconds or 0.0 for r in rows if r.duration_seconds]
    turn_counts = [r.turn_count or 0 for r in rows]
    barge_ins = sum(r.barge_in_count or 0 for r in rows)
    latencies = [r.avg_response_latency_ms for r in rows if r.avg_response_latency_ms]
    ttfa = [r.time_to_first_audio_ms for r in rows if r.time_to_first_audio_ms]

    per_day: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "escalated": 0, "resolved": 0}
    )
    for record in rows:
        started = record.started_at
        if started is None:
            continue
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        key = started.astimezone(UTC).date().isoformat()
        per_day[key]["calls"] += 1
        per_day[key]["escalated"] += 1 if record.escalated else 0
        per_day[key]["resolved"] += 1 if record.resolved and not record.escalated else 0

    unanswered = (
        await session.execute(
            select(func.coalesce(func.sum(UnansweredQuestion.occurrences), 0)).where(
                UnansweredQuestion.last_seen_at >= since
            )
        )
    ).scalar_one()

    kb_total = (await session.execute(select(func.count(KBRecord.id)))).scalar_one()
    kb_verified = (
        await session.execute(select(func.count(KBRecord.id)).where(KBRecord.verified.is_(True)))
    ).scalar_one()

    return {
        "range_days": days,
        "since": since.isoformat(),
        "totals": {
            "calls": total,
            "resolved": resolved,
            "escalated": escalated,
            "failed": failed,
            "abandoned": abandoned,
            "resolution_rate": round(resolved / total, 3) if total else 0.0,
            "escalation_rate": round(escalated / total, 3) if total else 0.0,
            "avg_duration_seconds": round(sum(durations) / len(durations), 1) if durations else 0.0,
            "avg_turns": round(sum(turn_counts) / len(turn_counts), 2) if turn_counts else 0.0,
            "barge_ins": barge_ins,
            "unanswered_questions": int(unanswered),
        },
        "latency_ms": {
            "avg_response": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "p95_response": _percentile(latencies, 0.95),
            "avg_time_to_first_audio": round(sum(ttfa) / len(ttfa), 1) if ttfa else 0.0,
            "p95_time_to_first_audio": _percentile(ttfa, 0.95),
        },
        "per_day": [
            {"date": key, **value} for key, value in sorted(per_day.items())
        ],
        "knowledge_base": {
            "records": int(kb_total),
            "verified": int(kb_verified),
            "verification_rate": round(kb_verified / kb_total, 3) if kb_total else 0.0,
        },
        "live": metrics.snapshot()["calls"],
    }


@router.get("/languages")
async def languages(
    days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    since = _since(days)
    rows = (
        await session.execute(
            select(CallRecord).where(CallRecord.started_at >= since, CallRecord.ended_at.is_not(None))
        )
    ).scalars().all()

    per_language: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0, "escalated": 0, "resolved": 0, "durations": [],
            "attempts": [], "confidence": [], "methods": Counter(),
        }
    )
    for record in rows:
        code = record.detected_language or "unknown"
        bucket = per_language[code]
        bucket["calls"] += 1
        bucket["escalated"] += 1 if record.escalated else 0
        bucket["resolved"] += 1 if record.resolved and not record.escalated else 0
        if record.duration_seconds:
            bucket["durations"].append(record.duration_seconds)
        if record.language_attempts:
            bucket["attempts"].append(record.language_attempts)
        if record.language_confidence:
            bucket["confidence"].append(record.language_confidence)
        if record.language_method:
            bucket["methods"][record.language_method] += 1

    items = []
    total_calls = sum(bucket["calls"] for bucket in per_language.values()) or 1
    for code, bucket in sorted(per_language.items(), key=lambda kv: -kv[1]["calls"]):
        items.append(
            {
                "language": code,
                "calls": bucket["calls"],
                "share": round(bucket["calls"] / total_calls, 3),
                "escalated": bucket["escalated"],
                "resolved": bucket["resolved"],
                "resolution_rate": round(
                    bucket["resolved"] / bucket["calls"], 3
                ) if bucket["calls"] else 0.0,
                "avg_duration_seconds": round(
                    sum(bucket["durations"]) / len(bucket["durations"]), 1
                ) if bucket["durations"] else 0.0,
                "avg_lid_attempts": round(
                    sum(bucket["attempts"]) / len(bucket["attempts"]), 2
                ) if bucket["attempts"] else 0.0,
                "avg_lid_confidence": round(
                    sum(bucket["confidence"]) / len(bucket["confidence"]), 3
                ) if bucket["confidence"] else 0.0,
                "lid_methods": dict(bucket["methods"]),
            }
        )
    return {"items": items, "total_calls": total_calls}


@router.get("/intents")
async def intents(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    since = _since(days)
    rows = (
        await session.execute(
            select(CallTurn.text, CallTurn.metadata_json)
            .join(CallRecord, CallTurn.call_id == CallRecord.id)
            .where(CallTurn.role == TurnRole.CALLER.value, CallRecord.started_at >= since)
        )
    ).all()

    counter: Counter[str] = Counter()
    for text, meta in rows:
        if not text or _is_language_turn(meta):
            continue
        counter[detect_intent(text).intent] += 1

    # primary intent per call, from the call record (cheaper, more accurate)
    primary_rows = (
        await session.execute(
            select(CallRecord.primary_intent, func.count(CallRecord.id))
            .where(CallRecord.started_at >= since, CallRecord.primary_intent.is_not(None))
            .group_by(CallRecord.primary_intent)
        )
    ).all()

    return {
        "turn_intents": [{"intent": k, "count": v} for k, v in counter.most_common(limit)],
        "call_primary_intents": [
            {"intent": intent, "calls": int(count)} for intent, count in primary_rows
        ],
        "total_turns": sum(counter.values()),
    }


@router.get("/top-queries")
async def top_queries(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(25, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    """Most-asked questions, grouped by canonical form."""
    since = _since(days)
    rows = (
        await session.execute(
            select(CallTurn.text, CallTurn.language, CallTurn.metadata_json, CallRecord.started_at)
            .join(CallRecord, CallTurn.call_id == CallRecord.id)
            .where(CallTurn.role == TurnRole.CALLER.value, CallRecord.started_at >= since)
        )
    ).all()

    groups: dict[str, dict[str, Any]] = {}
    counted = 0
    for text, language, meta, _started in rows:
        if not text or len(text.strip()) < 4 or _is_language_turn(meta):
            continue
        counted += 1
        key = canonical_key(text)
        bucket = groups.setdefault(
            key, {"canonical": key, "count": 0, "examples": [], "languages": Counter()}
        )
        bucket["count"] += 1
        bucket["languages"][language or "unknown"] += 1
        if len(bucket["examples"]) < 3 and text not in bucket["examples"]:
            bucket["examples"].append(text[:160])

    items = sorted(groups.values(), key=lambda b: -b["count"])[:limit]
    return {
        "items": [
            {
                "canonical": item["canonical"],
                "count": item["count"],
                "examples": item["examples"],
                "languages": dict(item["languages"]),
                "intent": detect_intent(item["examples"][0]).intent if item["examples"] else "other",
            }
            for item in items
        ],
        "distinct": len(groups),
        "total": counted,
    }


@router.get("/unanswered")
async def unanswered(
    days: int = Query(60, ge=1, le=365),
    limit: int = Query(50, ge=1, le=500),
    resolution: str = Query("open"),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    """The knowledge-base backlog: what callers asked that we could not answer."""
    since = _since(days)
    query = select(UnansweredQuestion).where(UnansweredQuestion.last_seen_at >= since)
    if resolution and resolution != "all":
        query = query.where(UnansweredQuestion.resolution == resolution)
    rows = (
        await session.execute(query.order_by(UnansweredQuestion.occurrences.desc()).limit(limit))
    ).scalars().all()
    return {
        "items": [
            {
                "id": row.id,
                "question": row.question,
                "canonical_key": row.canonical_key,
                "occurrences": row.occurrences,
                "language": row.language,
                "intent": row.intent or detect_intent(row.question).intent,
                "best_score": row.best_score,
                "resolution": row.resolution,
                "resolution_note": row.resolution_note,
                "first_seen_at": row.first_seen_at.isoformat() if row.first_seen_at else None,
                "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
            }
            for row in rows
        ],
        "total": len(rows),
    }


@router.post("/unanswered/{question_id}/resolve")
async def resolve_unanswered(
    question_id: str,
    resolution: str = Query("kb_added"),
    note: str = Query(""),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    row = await session.get(UnansweredQuestion, question_id)
    if row is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="question not found")
    row.resolution = resolution
    row.resolution_note = note
    await session.commit()
    return {"ok": True, "id": question_id, "resolution": resolution}


@router.get("/latency")
async def latency(
    days: int = Query(7, ge=1, le=90),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    """Where the milliseconds go, per pipeline stage."""
    since = _since(days)
    rows = (
        await session.execute(
            select(
                CallTurn.asr_ms, CallTurn.retrieval_ms, CallTurn.llm_ms,
                CallTurn.tts_ms, CallTurn.total_ms, CallTurn.metadata_json,
            )
            .join(CallRecord, CallTurn.call_id == CallRecord.id)
            .where(CallTurn.role == TurnRole.ASSISTANT.value, CallRecord.started_at >= since)
        )
    ).all()

    stages: dict[str, list[float]] = {
        "asr": [], "retrieval": [], "llm": [], "tts": [], "total": [], "first_audio": []
    }
    for asr_ms, retrieval_ms, llm_ms, tts_ms, total_ms, metadata in rows:
        stages["asr"].append(asr_ms or 0.0)
        stages["retrieval"].append(retrieval_ms or 0.0)
        stages["llm"].append(llm_ms or 0.0)
        stages["tts"].append(tts_ms or 0.0)
        stages["total"].append(total_ms or 0.0)
        first_audio = (metadata or {}).get("first_audio_ms")
        if first_audio:
            stages["first_audio"].append(float(first_audio))

    providers: Counter[str] = Counter()
    for _a, _r, _l, _t, _total, metadata in rows:
        providers[str((metadata or {}).get("provider", "unknown"))] += 1

    return {
        "turns": len(rows),
        "stages": {
            name: {
                "avg": round(sum(values) / len(values), 1) if values else 0.0,
                "p50": _percentile(values, 0.5),
                "p95": _percentile(values, 0.95),
                "max": round(max(values), 1) if values else 0.0,
            }
            for name, values in stages.items()
        },
        "providers": dict(providers),
        "live": metrics.snapshot()["latency_ms"],
    }


@router.get("/escalations")
async def escalations(
    days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    since = _since(days)
    rows = (
        await session.execute(
            select(EscalationEvent).where(EscalationEvent.requested_at >= since)
        )
    ).scalars().all()
    by_reason: Counter[str] = Counter(r.reason for r in rows)
    by_outcome: Counter[str] = Counter(r.outcome for r in rows)
    by_target_type: Counter[str] = Counter(r.target_type for r in rows)
    waits = [r.wait_seconds or 0.0 for r in rows if r.wait_seconds]

    followups = (
        await session.execute(select(FollowUp).where(FollowUp.created_at >= since))
    ).scalars().all()
    followup_status: Counter[str] = Counter(f.status for f in followups)
    followup_channel: Counter[str] = Counter(f.channel for f in followups)

    return {
        "total": len(rows),
        "by_reason": [{"reason": k, "count": v} for k, v in by_reason.most_common()],
        "by_outcome": dict(by_outcome),
        "by_target_type": dict(by_target_type),
        "avg_wait_seconds": round(sum(waits) / len(waits), 1) if waits else 0.0,
        "p95_wait_seconds": _percentile(waits, 0.95),
        "recent": [
            {
                "id": r.id,
                "call_id": r.call_id,
                "reason": r.reason,
                "target": r.target,
                "target_type": r.target_type,
                "outcome": r.outcome,
                "wait_seconds": r.wait_seconds,
                "requested_at": r.requested_at.isoformat() if r.requested_at else None,
                "summary": (r.whisper_summary or "")[:300],
            }
            for r in sorted(rows, key=lambda x: x.requested_at or datetime.now(UTC),
                            reverse=True)[:20]
        ],
        "followups": {
            "total": len(followups),
            "by_status": dict(followup_status),
            "by_channel": dict(followup_channel),
        },
    }


@router.get("/quality")
async def quality(
    days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    """Grounding + guardrail health: how often did the model need a fallback,
    how often were answers ungrounded, how often did barge-in happen."""
    since = _since(days)
    rows = (
        await session.execute(
            select(CallTurn).join(CallRecord, CallTurn.call_id == CallRecord.id).where(
                CallTurn.role == TurnRole.ASSISTANT.value, CallRecord.started_at >= since
            )
        )
    ).scalars().all()

    grounded = sum(1 for t in rows if t.grounded)
    fallback = 0
    warnings: Counter[str] = Counter()
    confidences: list[float] = []
    for turn in rows:
        metadata = turn.metadata_json or {}
        if metadata.get("fallback_used"):
            fallback += 1
        for warning in metadata.get("warnings") or []:
            warnings[str(warning)[:80]] += 1
        if turn.confidence:
            confidences.append(turn.confidence)

    calls = (
        await session.execute(select(CallRecord).where(CallRecord.started_at >= since))
    ).scalars().all()
    barge_in_calls = sum(1 for c in calls if (c.barge_in_count or 0) > 0)

    return {
        "assistant_turns": len(rows),
        "grounded": grounded,
        "grounded_rate": round(grounded / len(rows), 3) if rows else 0.0,
        "template_fallback": fallback,
        "template_fallback_rate": round(fallback / len(rows), 3) if rows else 0.0,
        "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        "warnings": [{"warning": k, "count": v} for k, v in warnings.most_common(15)],
        "calls_with_barge_in": barge_in_calls,
        "barge_in_total": sum(c.barge_in_count or 0 for c in calls),
    }
