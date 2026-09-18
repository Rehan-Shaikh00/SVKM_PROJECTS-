"""Turns a MediaChannel into a running CallSession.

Shared by the Twilio websocket, the Exotel/Plivo adapters and the browser
simulator, so every entry point gets identical conversation behaviour.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..ai.llm.base import LLM
from ..dependencies import get_answer_engine, get_call_logger, get_retriever, get_tts
from ..orchestrator.channel import MediaChannel
from ..orchestrator.session import CallSession, SessionDependencies
from ..orchestrator.states import CallContext, CallState
from ..voice.asr.base import StreamingASR

logger = logging.getLogger("nims.runner")

#: live sessions by call id — used by the dashboard and the hangup endpoint
ACTIVE_SESSIONS: dict[str, CallSession] = {}


async def build_dependencies(summariser_llm: LLM | None = None) -> SessionDependencies:
    retriever = await get_retriever()
    engine = await get_answer_engine()
    return SessionDependencies(
        retriever=retriever,
        answer_engine=engine,
        tts=get_tts(),
        logger=get_call_logger(),
        summariser_llm=summariser_llm or (engine.llm if engine.llm else None),
    )


def build_context(
    *,
    call_id: str,
    provider: str,
    provider_call_sid: str | None = None,
    stream_sid: str | None = None,
    from_number: str | None = None,
    to_number: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CallContext:
    now = time.time()
    return CallContext(
        call_id=call_id,
        provider=provider,
        provider_call_sid=provider_call_sid,
        stream_sid=stream_sid,
        from_number=from_number,
        to_number=to_number,
        state=CallState.STARTED,
        started_at=now,
        answered_at=now,
        metadata=metadata or {},
    )


async def run_session(
    channel: MediaChannel,
    context: CallContext,
    *,
    asr: StreamingASR | None = None,
    deps: SessionDependencies | None = None,
) -> CallSession:
    """Start a session in the background and register it."""
    deps = deps or await build_dependencies()
    session = CallSession(channel, deps, context, asr=asr)
    ACTIVE_SESSIONS[context.call_id] = session
    task = asyncio.create_task(_supervise(session, context.call_id), name=f"session:{context.call_id}")
    session.task = task  # type: ignore[attr-defined]
    return session


async def _supervise(session: CallSession, call_id: str) -> None:
    try:
        await session.run()
    except asyncio.CancelledError:  # pragma: no cover
        raise
    except Exception as exc:
        logger.exception("session %s crashed: %s", call_id, exc)
    finally:
        ACTIVE_SESSIONS.pop(call_id, None)


def get_session(call_id: str) -> CallSession | None:
    return ACTIVE_SESSIONS.get(call_id)


def active_call_count() -> int:
    return len(ACTIVE_SESSIONS)


async def end_all_sessions(reason: str = "shutdown") -> None:  # pragma: no cover
    for session in list(ACTIVE_SESSIONS.values()):
        try:
            await session.hangup(reason)
        except Exception as exc:
            logger.warning("could not end session cleanly: %s", exc)
