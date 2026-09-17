"""Telephony webhooks: Twilio (primary), Exotel and Plivo adapters.

Twilio endpoints
    POST /telephony/twilio/voice         inbound call → open the media stream
    WS   /telephony/twilio/media-stream  bidirectional μ-law audio + events
    POST /telephony/twilio/status        call status callbacks (ringing/completed)
    POST /telephony/twilio/transfer      human handoff (Dial/Enqueue TwiML)
    POST /telephony/twilio/whisper       context brief spoken to the agent
    POST /telephony/twilio/queue-wait    hold music while queued
    POST /telephony/twilio/queue-result  outcome of the enqueue
    POST /telephony/twilio/dial-result   outcome of the dial
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import UTC
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from sqlalchemy import select, update

from ..config import settings
from ..db import SessionLocal
from ..models import CallRecord, CallStatus, EscalationEvent
from ..observability.logging import metrics
from ..orchestrator.escalation import peek_pending_transfer, pop_pending_transfer
from ..telephony import exotel, plivo, twiml
from ..telephony.media_stream import TwilioMediaChannel, validate_twilio_request
from ..telephony.session_runner import build_context, build_dependencies, run_session

logger = logging.getLogger("nims.telephony")

router = APIRouter(prefix="/telephony", tags=["telephony"])

XML = "application/xml"


def _xml(body: str) -> Response:
    return Response(content=body, media_type=XML)


async def _validate(request: Request) -> None:
    """Signature validation for provider callbacks (skipped without credentials)."""
    if not settings.twilio_configured:
        return
    body = await request.body()
    url = str(request.url)
    headers = dict(request.headers)
    if not validate_twilio_request(headers, url, body):
        logger.warning("rejected unverified webhook from %s", request.client.host if request.client else "?")
        raise HTTPException(status_code=403, detail="invalid provider signature")


# --------------------------------------------------------------------------- #
# Twilio
# --------------------------------------------------------------------------- #


@router.post("/twilio/voice")
async def twilio_voice(request: Request) -> Response:
    await _validate(request)
    form = await request.form()
    call_sid = str(form.get("CallSid") or "")
    from_number = str(form.get("From") or "")
    to_number = str(form.get("To") or "")
    call_id = f"tw-{call_sid or uuid.uuid4().hex[:16]}"

    metrics.calls_started += 1
    logger.info("inbound twilio call %s from %s", call_sid, from_number[-4:] if from_number else "?")

    # Persist immediately so the dashboard shows the call even if the websocket
    # never connects (e.g. a firewall blocking wss://).
    from ..orchestrator.call_logger import redacted_caller

    async with SessionLocal() as session:
        session.add(
            CallRecord(
                id=call_id,
                provider="twilio",
                provider_call_sid=call_sid,
                direction="inbound",
                from_number=redacted_caller(from_number),
                to_number=to_number,
                status=CallStatus.IN_PROGRESS.value,
                metadata_json={"webhook": "voice", "answered_by": "ai"},
            )
        )
        await session.commit()

    body = twiml.answer_with_stream(
        call_sid=call_sid,
        from_number=from_number,
        to_number=to_number,
        custom_parameters={"call_id": call_id},
    )
    return _xml(body)


@router.websocket("/twilio/media-stream")
async def twilio_media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    channel: TwilioMediaChannel | None = None
    reader_task: asyncio.Task[None] | None = None
    call_id = f"tw-stream-{uuid.uuid4().hex[:10]}"
    try:
        # The first message is always `connected`, then `start`.
        channel = TwilioMediaChannel(call_id, websocket)

        async def reader() -> None:
            assert channel is not None
            while not channel.closed:
                message = await websocket.receive_json()
                await channel.handle_message(message)

        reader_task = asyncio.create_task(reader(), name=f"{call_id}:twilio-reader")

        # Wait for the `start` event so we know the call SID and custom params.
        deadline = time.time() + 10
        while channel.call_sid is None and time.time() < deadline:
            await asyncio.sleep(0.02)
        if channel.call_sid:
            call_id = f"tw-{channel.call_sid}"
            channel.call_id = call_id

        context = build_context(
            call_id=call_id,
            provider="twilio",
            provider_call_sid=channel.call_sid,
            stream_sid=channel.stream_sid,
            from_number=channel.from_number,
            to_number=channel.to_number,
            metadata={"account_sid": channel.account_sid},
        )
        deps = await build_dependencies()
        session = await run_session(channel, context, deps=deps)
        logger.info("twilio media stream session started: %s", call_id)

        await session._done.wait()
    except WebSocketDisconnect:
        logger.info("twilio media stream disconnected: %s", call_id)
    except Exception as exc:
        logger.exception("twilio media stream error: %s", exc)
        try:
            await websocket.send_text(twiml.fallback_twiml())
        except Exception:  # pragma: no cover
            pass
    finally:
        if channel is not None:
            await channel.close()
        # Cancel by reference. Matching on the task *name* used to miss every
        # time, because `call_id` is reassigned to the Twilio CallSid after the
        # `start` event while the task keeps its original name — leaking one
        # reader task per call and swallowing its exceptions.
        if reader_task is not None:
            if not reader_task.done():
                reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("twilio media reader ended with: %s", exc)
        try:
            await websocket.close()
        except Exception:  # pragma: no cover
            pass


@router.post("/twilio/status")
async def twilio_status(request: Request) -> Response:
    """Call status callback — keeps the call record accurate for analytics."""
    form = await request.form()
    call_sid = str(form.get("CallSid") or "")
    status = str(form.get("CallStatus") or "")
    duration = form.get("CallDuration") or form.get("Duration")
    recording_url = str(form.get("RecordingUrl") or "") or None

    if not call_sid:
        return _xml("<Response/>")

    mapped = {
        "completed": CallStatus.COMPLETED.value,
        "busy": CallStatus.FAILED.value,
        "failed": CallStatus.FAILED.value,
        "no-answer": CallStatus.ABANDONED.value,
        "canceled": CallStatus.ABANDONED.value,
    }.get(status, CallStatus.IN_PROGRESS.value)

    if recording_url and not settings.allow_call_recording_storage:
        logger.info("dropping recording URL: call recording storage is disabled")
        recording_url = None

    async with SessionLocal() as session:
        await session.execute(
            update(CallRecord)
            .where(CallRecord.provider_call_sid == call_sid)
            .values(
                status=mapped,
                end_reason=f"provider:{status}",
                duration_seconds=float(duration) if duration else None,
                recording_uri=recording_url,
                ended_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            )
        )
        await session.commit()
    return _xml("<Response/>")


@router.post("/twilio/transfer")
async def twilio_transfer(
    call_id: str = Query(""),
    call_sid: str = Query(""),
    reason: str = Query(""),
) -> Response:
    """The redirect target after the AI decides to escalate."""
    plan = pop_pending_transfer(call_id)
    whisper_url = f"{settings.public_base_url}/telephony/twilio/whisper?call_id={call_id}"

    if plan is None:
        async with SessionLocal() as session:
            row = (
                await session.execute(
                    select(EscalationEvent)
                    .where(EscalationEvent.call_id == call_id)
                    .order_by(EscalationEvent.requested_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        if row is None:
            logger.warning("transfer requested for unknown call %s", call_id)
            return _xml(twiml.fallback_twiml())
        plan_target = row.target or ""
        plan_type = row.target_type
    else:
        plan_target = plan.target
        plan_type = plan.target_type

    metrics.calls_escalated += 1
    if plan_type == "queue" and settings.escalation_agent_list:
        body = twiml.transfer_twiml(
            target=plan_target,
            target_type="queue",
            whisper_url=whisper_url if settings.escalation_whisper_context else None,
        )
    else:
        body = twiml.transfer_twiml(
            target=plan_target,
            target_type="number",
            whisper_url=whisper_url if settings.escalation_whisper_context else None,
        )
    logger.info("transferring call %s (%s) to %s", call_id, reason, plan_target or "queue")
    return _xml(body)


@router.post("/twilio/whisper")
@router.get("/twilio/whisper")
async def twilio_whisper(call_id: str = Query("")) -> Response:
    """Context brief spoken to the agent before the caller is bridged."""
    plan = peek_pending_transfer(call_id)
    summary = plan.whisper if plan else ""
    if not summary:
        async with SessionLocal() as session:
            row = (
                await session.execute(
                    select(EscalationEvent.whisper_summary)
                    .where(EscalationEvent.call_id == call_id)
                    .order_by(EscalationEvent.requested_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        summary = row or ""
    if not summary:
        call = (
            await _fetch_call(call_id)
        )
        summary = (call.summary if call else "") or "The caller needs admissions help."
    return _xml(twiml.whisper_page(summary))


async def _fetch_call(call_id: str) -> CallRecord | None:
    async with SessionLocal() as session:
        return await session.get(CallRecord, call_id)


@router.post("/twilio/queue-wait")
async def twilio_queue_wait() -> Response:
    return _xml(twiml.queue_wait_twiml())


@router.post("/twilio/queue-result")
async def twilio_queue_result(request: Request) -> Response:
    form = await request.form()
    call_id = str(form.get("call_id") or "")
    outcome = str(form.get("QueueResult") or form.get("DequeueResult") or "unknown")
    await _record_escalation_outcome(call_id, outcome, target_type="queue")
    if outcome in ("hangup", "system-error", "queue-full"):
        return _xml(twiml.fallback_twiml(hangup=True))
    return _xml("<Response/>")


@router.post("/twilio/dial-result")
async def twilio_dial_result(request: Request) -> Response:
    form = await request.form()
    call_id = str(form.get("call_id") or "")
    dial_status = str(form.get("DialCallStatus") or form.get("DialStatus") or "unknown")
    await _record_escalation_outcome(call_id, dial_status, target_type="number")
    if dial_status in {"no-answer", "busy", "failed", "canceled"}:
        return _xml(
            twiml.fallback_twiml(
                "The advisor could not be reached. Please call the admissions office on "
                "1 800 120 1020.",
                hangup=True,
            )
        )
    return _xml("<Response/>")


async def _record_escalation_outcome(call_id: str, outcome: str, target_type: str) -> None:
    if not call_id:
        return
    from datetime import datetime

    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(EscalationEvent)
                .where(EscalationEvent.call_id == call_id)
                .order_by(EscalationEvent.requested_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if row is None:
            return
        now = datetime.now(UTC)
        row.outcome = outcome
        row.ended_at = now
        if row.requested_at:
            requested = row.requested_at
            if requested.tzinfo is None:
                requested = requested.replace(tzinfo=UTC)
            row.wait_seconds = max(0.0, (now - requested).total_seconds())
        if outcome in {"completed", "answered"}:
            row.connected_at = now
        await session.commit()
    logger.info("escalation outcome for %s (%s): %s", call_id, target_type, outcome)


# --------------------------------------------------------------------------- #
# Exotel
# --------------------------------------------------------------------------- #


@router.post("/exotel/voice")
async def exotel_voice(request: Request) -> Response:
    """Exotel inbound webhook (MODE B). See app/telephony/exotel.py for MODE A."""
    data = await _provider_payload(request)
    call_uuid = str(data.get("CallSid") or data.get("call_sid") or uuid.uuid4().hex[:16])
    metrics.calls_started += 1
    return _xml(exotel.build_greeting_exoml(f"ex-{call_uuid}", call_uuid))


@router.post("/exotel/language")
async def exotel_language(
    request: Request,
    call_id: str = Query(""),
    sid: str = Query(""),
) -> Response:
    data = await _provider_payload(request)
    transcript = str(data.get("Speech") or data.get("speech") or "")
    digits = str(data.get("Digits") or data.get("digits") or "")
    return _xml(exotel.build_language_result_exoml(call_id, sid, transcript=transcript, digits=digits))


@router.post("/exotel/conversation")
async def exotel_conversation(
    request: Request,
    call_id: str = Query(""),
    sid: str = Query(""),
    language: str = Query("en-IN"),
) -> Response:
    data = await _provider_payload(request)
    transcript = str(data.get("Speech") or data.get("speech") or data.get("text") or "")
    if not transcript.strip():
        return _xml(exotel.exoml_response(exotel.say("I did not catch that. Let me connect you to an advisor."),
                                          exotel._transfer_verbs()))
    body = await exotel.build_conversation_exoml_async(call_id, sid, language, transcript)
    return _xml(body)


@router.post("/exotel/events")
async def exotel_events(request: Request) -> dict[str, Any]:
    """Call-event callbacks (ringing / answered / completed / recording ready)."""
    data = await _provider_payload(request)
    normalised = exotel.call_event_payload(data)
    logger.info("exotel event: %s", normalised.get("event"))
    if normalised.get("recording_url") and not settings.allow_call_recording_storage:
        normalised["recording_url"] = None
    return {"ok": True, "received": normalised}


# --------------------------------------------------------------------------- #
# Plivo
# --------------------------------------------------------------------------- #


@router.post("/plivo/voice")
async def plivo_voice(request: Request) -> Response:
    data = await _provider_payload(request)
    call_uuid = str(data.get("CallUUID") or data.get("callUUID") or uuid.uuid4().hex[:16])
    metrics.calls_started += 1
    return _xml(plivo.build_inbound_xml(f"pl-{call_uuid}"))


@router.post("/plivo/language")
async def plivo_language(request: Request, uuid: str = Query("")) -> Response:
    data = await _provider_payload(request)
    speech = str(data.get("Speech") or "")
    digits = str(data.get("Digits") or "")
    return _xml(await plivo.build_language_xml(uuid, speech, digits))


@router.post("/plivo/conversation")
async def plivo_conversation(
    request: Request,
    uuid: str = Query(""),
    language: str = Query("en-IN"),
) -> Response:
    data = await _provider_payload(request)
    speech = str(data.get("Speech") or data.get("text") or "")
    if not speech.strip():
        return _xml(plivo.xml_response(plivo.speak("Let me connect you to an advisor."),
                                       plivo._transfer_verbs()))
    return _xml(await plivo.build_conversation_xml(uuid, language, speech))


# --------------------------------------------------------------------------- #
# SIP / generic media bridge (FreeSWITCH mod_audio_stream, Asterisk ARI, LiveKit)
# --------------------------------------------------------------------------- #


@router.websocket("/sip/media-stream")
async def sip_media_stream(websocket: WebSocket, provider: str = Query("sip")) -> None:
    """Generic websocket media bridge using the Twilio Media Streams framing.

    FreeSWITCH `mod_audio_stream`, Asterisk `res_audio_stream` and LiveKit SIP
    bridges can all be pointed here, which is how Exotel/Ozonetel SIP trunks get
    the same streaming experience as Twilio (8 kHz μ-law, JSON control frames).
    """
    await websocket.accept()
    call_id = f"{provider}-{uuid.uuid4().hex[:10]}"
    channel = TwilioMediaChannel(call_id, websocket)
    channel.name = provider
    try:
        await channel.handle_message({"event": "connected", "streamSid": call_id})
        await channel.handle_message({"event": "start", "start": {"callSid": call_id}})
        context = build_context(call_id=call_id, provider=provider,
                                provider_call_sid=call_id, stream_sid=call_id)
        deps = await build_dependencies()
        session = await run_session(channel, context, deps=deps)

        async def reader() -> None:
            while not channel.closed:
                message = await websocket.receive_json()
                await channel.handle_message(message)

        reader_task = asyncio.create_task(reader(), name=f"{call_id}:sip-reader")
        await session._done.wait()
        reader_task.cancel()
    except WebSocketDisconnect:
        logger.info("sip media stream disconnected: %s", call_id)
    except Exception as exc:  # pragma: no cover
        logger.exception("sip media stream error: %s", exc)
    finally:
        await channel.close()


# --------------------------------------------------------------------------- #


async def _provider_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "json" in content_type:
        try:
            return dict(await request.json())
        except Exception:
            return {}
    form = await request.form()
    return {key: value for key, value in form.items()}


@router.get("/twilio/config")
async def twilio_config() -> dict[str, Any]:
    """What to paste into the Twilio console — handy during setup."""
    base = settings.public_base_url
    return {
        "voice_webhook": f"{base}/telephony/twilio/voice",
        "voice_method": "POST",
        "status_callback": f"{base}/telephony/twilio/status",
        "media_stream_url": twiml.stream_websocket_url(),
        "sms_webhook": f"{base}/telephony/twilio/sms",
        "configured": settings.twilio_configured,
        "helpline": settings.twilio_helpline_number,
        "escalation_agents": settings.escalation_agent_list,
        "queue": settings.escalation_queue_name,
        "recording_enabled": settings.call_recording_enabled,
        "notes": [
            "The voice webhook must be reachable over HTTPS from Twilio.",
            "The media stream URL must be wss:// (TLS) in production.",
            "Enable 'Voice Geographic Permissions' for India on the number.",
        ],
    }


@router.post("/twilio/sms")
async def twilio_sms(request: Request) -> Response:
    """Inbound SMS: the same brain, text channel. Callers who prefer texting, or
    who were cut off mid-call, can continue the conversation here."""
    from ..ai.rag import AnswerRequest
    from ..dependencies import get_answer_engine

    form = await request.form()
    body = str(form.get("Body") or "")
    sender = str(form.get("From") or "")
    if not body.strip():
        return _xml(twiml.fallback_twiml("Please send your question.", hangup=False))
    engine = await get_answer_engine()
    async with SessionLocal() as session:
        result = await engine.answer(
            AnswerRequest(call_id=f"sms-{sender[-6:] or 'unknown'}", question=body, language="en-IN"),
            session,
        )
    response = twiml.VoiceResponse()
    response.message(result.text[:1400])
    return _xml(str(response))


@router.get("/health")
async def telephony_health() -> dict[str, Any]:
    return {
        "provider": settings.telephony_provider,
        "twilio_configured": settings.twilio_configured,
        "media_stream_url": twiml.stream_websocket_url(),
        "escalation_targets": len(settings.escalation_agent_list),
        "queue": settings.escalation_queue_name,
        "hold_music": settings.hold_music_url,
    }


def _unused_metrics_reference() -> None:  # pragma: no cover
    """Keep the import used for static analysis; metrics are updated in routes."""
    _ = metrics.snapshot
