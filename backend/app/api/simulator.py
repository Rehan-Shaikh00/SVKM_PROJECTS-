"""Browser call simulator.

This is the development and demo entry point: a web page that behaves like a
phone. It drives the *same* `CallSession` as a real Twilio call, so everything
you test here is what a caller experiences — greeting, spoken language
identification, RAG answers, barge-in, follow-up SMS offer, escalation.

Client → server messages
    {"type":"start","from":"+91…","to":"+91…","mode":"voice"|"text"}
    {"type":"audio","payload":"<base64>","encoding":"mulaw"|"pcm16","sample_rate":8000}
    {"type":"transcript","text":"…","is_final":true,"language":"hi-IN","confidence":0.9}
    {"type":"text","text":"…"}                     # typed instead of spoken
    {"type":"dtmf","digit":"1"}
    {"type":"control","name":"speech_done"}        # browser finished speaking
    {"type":"language_override","language":"hi-IN"}
    {"type":"hangup"}

Server → client: every `channel.send_event(...)` payload, plus
    {"type":"audio","payload":"<base64 mulaw>","sample_rate":8000}
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..observability.logging import metrics
from ..orchestrator.channel import InboundEvent, SimulatorChannel
from ..telephony.session_runner import build_context, build_dependencies, run_session
from ..voice.audio import resample

logger = logging.getLogger("nims.simulator")

router = APIRouter(prefix="/simulator", tags=["simulator"])


@router.get("/config")
async def simulator_config() -> dict[str, Any]:
    """What the browser needs before it places a call."""
    from ..dependencies import provider_summary

    summary = provider_summary()
    return {
        "assistant_name": "Saarthi",
        "university": "NMIMS Global University, Dhule",
        # Fallback is the verified campus number from svkmnmimsgu.ac.in/contact-us;
        # the university publishes no toll-free line of its own.
        "helpline": settings.twilio_helpline_number,
        "supported_languages": settings.supported_language_list,
        "greeting_languages": settings.greeting_language_list,
        "server_asr": summary["active"]["asr"],
        "server_tts": summary["active"]["tts"],
        "tts_text_only": summary["tts_text_only"],
        "llm": summary["active"]["llm"],
        "lid": settings.lid_provider,
        "lid_threshold": settings.lid_confidence_threshold,
        "dtmf_fallback": settings.allow_dtmf_fallback,
        "sample_rate": 8000,
        "frame_ms": 20,
        "academic_year": settings.kb_academic_year,
        "provider_summary": summary,
    }


@router.websocket("/ws")
async def simulator_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    call_id = f"sim-{uuid.uuid4().hex[:12]}"
    channel = SimulatorChannel(call_id, websocket)
    session = None
    reader: asyncio.Task[Any] | None = None

    try:
        first = await asyncio.wait_for(websocket.receive_json(), timeout=30)
        if first.get("type") != "start":
            await channel.send_event(
                "error", {"message": "first message must be {\"type\":\"start\"}"}
            )
            await websocket.close(code=1008)
            return
    except (TimeoutError, WebSocketDisconnect):
        await websocket.close(code=1008)
        return

    context = build_context(
        call_id=call_id,
        provider="simulator",
        from_number=first.get("from") or "+919000000000",
        to_number=first.get("to") or settings.twilio_helpline_number,
        metadata={"mode": first.get("mode", "voice"), "user_agent": first.get("user_agent", "")},
    )
    metrics.calls_started += 1
    deps = await build_dependencies()
    session = await run_session(channel, context, deps=deps)
    logger.info("simulator call %s started", call_id)

    async def pump() -> None:
        try:
            while True:
                message = await websocket.receive_json()
                await _dispatch(channel, message)
                if message.get("type") == "hangup":
                    break
        except WebSocketDisconnect:
            logger.info("simulator client disconnected: %s", call_id)
        except Exception as exc:  # pragma: no cover
            logger.warning("simulator pump error: %s", exc)
        finally:
            await channel.push_inbound(InboundEvent(type="hangup", payload={"reason": "ws_close"}))

    reader = asyncio.create_task(pump(), name=f"{call_id}:ws-pump")

    await session._done.wait()
    metrics.calls_completed += 1
    if reader and not reader.done():
        reader.cancel()
    try:
        await reader
    except (asyncio.CancelledError, Exception):  # pragma: no cover
        pass


async def _dispatch(channel: SimulatorChannel, message: dict[str, Any]) -> None:
    kind = message.get("type")
    if kind == "audio":
        payload = message.get("payload") or ""
        if not payload:
            return
        try:
            raw = base64.b64decode(payload)
        except Exception:
            return
        encoding = message.get("encoding", "mulaw")
        if encoding == "pcm16":
            rate = int(message.get("sample_rate") or 16000)
            pcm = np.frombuffer(raw, dtype="<i2")
            if rate != 8000:
                pcm = resample(pcm, rate, 8000).astype(np.int16)
            from ..voice.audio import pcm16_to_mulaw

            raw = pcm16_to_mulaw(pcm)
        await channel.push_inbound(InboundEvent(type="audio", audio=raw))
    elif kind == "transcript":
        await channel.push_inbound(
            InboundEvent(
                type="transcript",
                text=str(message.get("text") or ""),
                is_final=bool(message.get("is_final", True)),
                language=message.get("language"),
                confidence=float(message.get("confidence") or 0.0),
            )
        )
    elif kind == "text":
        await channel.push_inbound(
            InboundEvent(type="transcript", text=str(message.get("text") or ""), is_final=True)
        )
    elif kind == "dtmf":
        await channel.push_inbound(InboundEvent(type="dtmf", digit=str(message.get("digit", ""))))
    elif kind == "control":
        await channel.push_inbound(
            InboundEvent(type="control", payload={"name": message.get("name"), **message})
        )
    elif kind == "language_override":
        await channel.push_inbound(
            InboundEvent(
                type="control",
                payload={"name": "language_override", "language": message.get("language")},
            )
        )
    elif kind == "escalate":
        await channel.push_inbound(
            InboundEvent(type="control", payload={"name": "escalate",
                                                  "summary": message.get("summary", "")})
        )
    elif kind == "hangup":
        await channel.push_inbound(InboundEvent(type="hangup", payload={"reason": "client"}))
    else:
        await channel.send_event("error", {"message": f"unknown message type: {kind}"})


@router.post("/end/{call_id}")
async def end_simulator_call(call_id: str) -> dict[str, Any]:
    """Force-end a running simulated call (used by the dashboard)."""
    from ..telephony.session_runner import get_session

    session = get_session(call_id)
    if session is None:
        return {"ok": False, "reason": "no such active call"}
    await session.hangup("ended_from_dashboard")
    return {"ok": True, "call_id": call_id, "ended_at": time.time()}
