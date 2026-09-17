"""Asynchronous call/transcript persistence.

The media path must never block on the database, so writes go through a bounded
queue drained by a single worker. If the queue is full we drop the *oldest*
event and count the drop — losing one transcript line is better than adding
latency to a live call.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from ..config import settings
from ..db import SessionLocal
from ..kb.chunking import canonical_key
from ..models import (
    CallRecord,
    CallStatus,
    CallTurn,
    EscalationEvent,
    FollowUp,
    ProviderHealth,
    TurnRole,
    UnansweredQuestion,
    new_id,
)

logger = logging.getLogger("nims.calllog")

QUEUE_MAX = 500


class CallLogger:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue(
            maxsize=QUEUE_MAX
        )
        self._worker: asyncio.Task[None] | None = None
        self.dropped = 0
        self.written = 0
        self.errors = 0

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="call-logger")

    async def stop(self) -> None:
        if self._worker:
            await self._queue.put(None)
            try:
                await asyncio.wait_for(self._worker, timeout=5)
            except (TimeoutError, asyncio.CancelledError):  # pragma: no cover
                self._worker.cancel()

    def submit(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((kind, payload))
        except asyncio.QueueFull:
            self.dropped += 1
            logger.warning("call log queue full; dropped a %s event", kind)

    # -- worker ------------------------------------------------------------ #
    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            kind, payload = item
            handler = getattr(self, f"_do_{kind}", None)
            if handler is None:  # pragma: no cover
                logger.error("unknown log kind %s", kind)
                continue
            try:
                async with SessionLocal() as session:
                    await handler(session, payload)
                    await session.commit()
                self.written += 1
            except Exception as exc:
                self.errors += 1
                logger.exception("call log write failed (%s): %s", kind, exc)

    # -- handlers ---------------------------------------------------------- #
    def start_call(self, payload: dict[str, Any]) -> None:
        self.submit("start_call", payload)

    async def _do_start_call(self, session, payload: dict[str, Any]) -> None:
        record = CallRecord(
            id=payload["call_id"],
            provider=payload.get("provider", "twilio"),
            provider_call_sid=payload.get("provider_call_sid"),
            stream_sid=payload.get("stream_sid"),
            direction=payload.get("direction", "inbound"),
            from_number=payload.get("from_number"),
            to_number=payload.get("to_number"),
            status=CallStatus.IN_PROGRESS.value,
            started_at=datetime.now(UTC),
            answered_at=datetime.now(UTC),
            metadata_json=payload.get("metadata") or {},
        )
        session.add(record)

    def log_turn(self, payload: dict[str, Any]) -> None:
        self.submit("log_turn", payload)

    async def _do_log_turn(self, session, payload: dict[str, Any]) -> None:
        record = await session.get(CallRecord, payload["call_id"])
        if record is None:  # pragma: no cover - race with start_call
            logger.warning("log_turn for unknown call %s", payload["call_id"])
            return
        turn = CallTurn(
            id=new_id(),
            call_id=payload["call_id"],
            seq=payload.get("seq", 0),
            role=payload.get("role", TurnRole.CALLER.value),
            text=payload.get("text", "") or "",
            language=payload.get("language"),
            is_final=payload.get("is_final", True),
            interrupted=payload.get("interrupted", False),
            asr_ms=payload.get("asr_ms", 0.0),
            retrieval_ms=payload.get("retrieval_ms", 0.0),
            llm_ms=payload.get("llm_ms", 0.0),
            tts_ms=payload.get("tts_ms", 0.0),
            total_ms=payload.get("total_ms", 0.0),
            confidence=payload.get("confidence", 0.0),
            grounded=payload.get("grounded", False),
            citations_json=payload.get("citations") or [],
            tool_calls_json=payload.get("tool_calls") or [],
            metadata_json=payload.get("metadata") or {},
        )
        session.add(turn)
        record.turn_count = (record.turn_count or 0) + (
            1 if turn.role == TurnRole.CALLER.value else 0
        )
        if turn.role == TurnRole.ASSISTANT.value and not turn.grounded:
            record.unanswered_count = (record.unanswered_count or 0) + 1

    def update_call(self, payload: dict[str, Any]) -> None:
        self.submit("update_call", payload)

    async def _do_update_call(self, session, payload: dict[str, Any]) -> None:
        record = await session.get(CallRecord, payload["call_id"])
        if record is None:
            return
        for key, value in payload.items():
            if key == "call_id":
                continue
            if hasattr(record, key):
                setattr(record, key, value)

    def end_call(self, payload: dict[str, Any]) -> None:
        self.submit("end_call", payload)

    async def _do_end_call(self, session, payload: dict[str, Any]) -> None:
        record = await session.get(CallRecord, payload["call_id"])
        if record is None:
            return
        record.status = payload.get("status", CallStatus.COMPLETED.value)
        record.end_reason = payload.get("end_reason")
        record.ended_at = datetime.now(UTC)
        record.duration_seconds = float(payload.get("duration_seconds", 0.0))
        record.summary = payload.get("summary")
        record.transcript_text = payload.get("transcript")
        record.escalated = bool(payload.get("escalated", False))
        record.resolved = bool(payload.get("resolved", False))
        record.escalation_reason = payload.get("escalation_reason")
        record.detected_language = payload.get("detected_language") or record.detected_language
        record.language_confidence = float(
            payload.get("language_confidence") or record.language_confidence or 0.0
        )
        record.language_method = payload.get("language_method") or record.language_method
        record.language_attempts = int(payload.get("language_attempts") or record.language_attempts or 0)
        record.barge_in_count = int(payload.get("barge_in_count") or 0)
        record.time_to_first_audio_ms = float(payload.get("time_to_first_audio_ms") or 0.0)
        record.avg_response_latency_ms = float(payload.get("avg_response_latency_ms") or 0.0)
        record.primary_intent = payload.get("primary_intent") or record.primary_intent
        record.intents_json = payload.get("intents") or record.intents_json
        if payload.get("metadata"):
            merged = dict(record.metadata_json or {})
            merged.update(payload["metadata"])
            record.metadata_json = merged

    def log_escalation(self, payload: dict[str, Any]) -> None:
        self.submit("log_escalation", payload)

    async def _do_log_escalation(self, session, payload: dict[str, Any]) -> None:
        session.add(
            EscalationEvent(
                id=new_id(),
                call_id=payload["call_id"],
                reason=payload.get("reason", "unknown"),
                target=payload.get("target"),
                target_type=payload.get("target_type", "number"),
                outcome=payload.get("outcome", "pending"),
                whisper_summary=payload.get("whisper_summary"),
                context_json=payload.get("context") or {},
                connected_at=payload.get("connected_at"),
            )
        )

    def log_followup(self, payload: dict[str, Any]) -> None:
        self.submit("log_followup", payload)

    async def _do_log_followup(self, session, payload: dict[str, Any]) -> None:
        session.add(
            FollowUp(
                id=new_id(),
                call_id=payload["call_id"],
                channel=payload.get("channel", "sms"),
                destination_ref=payload.get("destination_ref", ""),
                body=payload.get("body", ""),
                attachments_json=payload.get("attachments") or [],
                status=payload.get("status", "queued"),
                provider_message_id=payload.get("provider_message_id"),
                error=payload.get("error"),
            )
        )

    def log_unanswered(self, payload: dict[str, Any]) -> None:
        self.submit("log_unanswered", payload)

    async def _do_log_unanswered(self, session, payload: dict[str, Any]) -> None:
        key = canonical_key(payload.get("question", ""))
        if not key:
            return
        existing = (
            await session.execute(
                select(UnansweredQuestion).where(UnansweredQuestion.canonical_key == key)
            )
        ).scalar_one_or_none()
        if existing:
            existing.occurrences += 1
            existing.last_seen_at = datetime.now(UTC)
            existing.best_score = max(existing.best_score or 0.0, float(payload.get("best_score") or 0.0))
            if payload.get("intent"):
                existing.intent = payload["intent"]
        else:
            session.add(
                UnansweredQuestion(
                    id=new_id(),
                    call_id=payload.get("call_id"),
                    question=payload.get("question", "")[:500],
                    canonical_key=key,
                    language=payload.get("language"),
                    intent=payload.get("intent"),
                    best_score=float(payload.get("best_score") or 0.0),
                )
            )

    def log_provider_health(self, payload: dict[str, Any]) -> None:
        self.submit("provider_health", payload)

    async def _do_provider_health(self, session, payload: dict[str, Any]) -> None:
        session.add(
            ProviderHealth(
                id=new_id(),
                provider=payload.get("provider", "unknown"),
                component=payload.get("component", "unknown"),
                ok=bool(payload.get("ok", True)),
                latency_ms=float(payload.get("latency_ms", 0.0)),
                detail=str(payload.get("detail", ""))[:500],
            )
        )

    def stats(self) -> dict[str, int]:
        return {"written": self.written, "dropped": self.dropped, "errors": self.errors,
                "queue": self._queue.qsize()}


call_logger = CallLogger()


def redacted_caller(number: str | None) -> str | None:
    from ..ai.guardrails import hash_secret

    if not number:
        return None
    if not settings.redact_pii:
        return number
    return f"hash:{hash_secret(number)}"
