"""Exotel adapter (Indian CPaaS — TRAI-compliant numbering, local presence).

Exotel does not expose a bidirectional raw-media websocket the way Twilio Media
Streams does, so there are two supported integration modes. Both are implemented
here; pick one per deployment.

MODE A — SIP trunk (recommended for this product)
    Exotel DID/toll-free → SIP trunk → your media server (FreeSWITCH / Asterisk /
    LiveKit SIP) → this service's websocket adapter.
    You get continuous 8 kHz μ-law in both directions, so streaming ASR,
    sub-second answers and barge-in all work exactly as on Twilio.
    FreeSWITCH config sketch: `mod_audio_stream` pointed at
    `wss://<host>/telephony/sip/media-stream?provider=exotel`.

MODE B — ExoML webhook (no media server)
    Exotel posts to /telephony/exotel/voice and we return ExoML verbs. Speech
    recognition is performed by Exotel's platform (their `Gather`/AI verbs), so
    turn latency is higher and barge-in is not available. Good enough for a
    pilot in a single language; use MODE A for production.

Verify the exact verb set and parameter names against your Exotel plan's
documentation before go-live — ExoML capabilities differ between accounts and
regions, and the speech-input verbs in particular are plan dependent.
"""

from __future__ import annotations

import logging
from typing import Any
from xml.sax.saxutils import escape

from ..config import settings

logger = logging.getLogger("nims.exotel")


def exoml_response(*verbs: str) -> str:
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response>" + "".join(verbs) + "</Response>"


def say(text: str, *, voice: str = "woman", language: str = "en") -> str:
    return f'<Say voice="{voice}" language="{language}">{escape(text)}</Say>'


def play(url: str, *, loop: int = 1) -> str:
    return f'<Play loop="{loop}">{escape(url)}</Play>'


def dial(number: str, *, caller_id: str = "", time_limit: int = 300) -> str:
    attributes = f' timeLimit="{time_limit}"'
    if caller_id:
        attributes += f' callerId="{escape(caller_id)}"'
    return f"<Dial{attributes}><Number>{escape(number)}</Number></Dial>"


def hangup() -> str:
    return "<Hangup/>"


def redirect(url: str) -> str:
    return f"<Redirect>{escape(url)}</Redirect>"


def gather_speech(
    action_url: str,
    *,
    language: str = "en-IN",
    timeout: int = 8,
    max_retries: int = 2,
) -> str:
    """MODE B speech input. Parameter names vary by Exotel plan — confirm with
    your account manager. Falls back to DTMF when speech is unavailable."""
    return (
        f'<Gather action="{escape(action_url)}" method="POST" timeout="{timeout}" '
        f'digits="1" retries="{max_retries}" speech="true" language="{language}"/>'
    )


def build_greeting_exoml(call_id: str, sid: str) -> str:
    """MODE B entry point: greet, then collect the language declaration."""
    action = f"{settings.public_base_url}/telephony/exotel/language?call_id={call_id}&sid={sid}"
    verbs = [
        say("Thank you for calling NMIMS Global University, Dhule. This call may be recorded."),
        say("Please tell me your preferred language. कृपया अपनी भाषा बताइए।"),
        gather_speech(action, language="en-IN", timeout=8),
        say("Sorry, we could not understand. Connecting you to an advisor."),
        _transfer_verbs(),
    ]
    return exoml_response(*verbs)


def _transfer_verbs() -> str:
    agents = settings.escalation_agent_list
    if not agents:
        return say("Our advisors are unavailable. Please call 1 800 120 1020.") + hangup()
    return dial(agents[0], caller_id=settings.twilio_helpline_number, time_limit=300)


def build_language_result_exoml(
    call_id: str,
    sid: str,
    *,
    transcript: str = "",
    digits: str = "",
) -> str:
    """Handle the caller's spoken language declaration (MODE B).

    Language identification runs on our side (app.voice.lid), then we redirect to
    the conversation collector with the language pinned.
    """
    from ..voice.lid.local import detect_language_name, detect_language_text

    text = transcript or digits
    named = detect_language_name(text)
    result = named or detect_language_text(text)
    language = result.language if result.confidence >= settings.lid_confidence_threshold else "en-IN"
    action = (
        f"{settings.public_base_url}/telephony/exotel/conversation"
        f"?call_id={call_id}&sid={sid}&language={language}"
    )
    return exoml_response(
        say(f"Great, I will continue in {language.split('-')[0]}."),
        gather_speech(action, language=language, timeout=10),
        say("I did not catch that. Let me connect you to an advisor."),
        _transfer_verbs(),
    )


def build_conversation_exoml(call_id: str, sid: str, language: str, transcript: str) -> str:
    """MODE B conversation loop: answer synchronously, then gather again."""
    import asyncio

    from ..ai.rag import AnswerRequest
    from ..dependencies import get_answer_engine

    async def _answer() -> str:
        engine = await get_answer_engine()
        result = await engine.answer(
            AnswerRequest(call_id=call_id, question=transcript, language=language)
        )
        return result.text

    try:
        answer = asyncio.run(_answer())
    except RuntimeError:
        # already inside an event loop (uvicorn) — the API layer uses the async
        # variant below instead
        logger.warning("exotel sync answer called inside a running loop")
        answer = ""
    next_action = (
        f"{settings.public_base_url}/telephony/exotel/conversation"
        f"?call_id={call_id}&sid={sid}&language={language}"
    )
    verbs = []
    if answer:
        verbs.append(say(answer))
    verbs.append(gather_speech(next_action, language=language, timeout=10))
    verbs.append(say("Let me connect you to an advisor."))
    verbs.append(_transfer_verbs())
    return exoml_response(*verbs)


async def build_conversation_exoml_async(
    call_id: str, sid: str, language: str, transcript: str
) -> str:
    """Async variant used by the FastAPI route (no nested event loop)."""
    from ..ai.rag import AnswerRequest
    from ..dependencies import get_answer_engine

    engine = await get_answer_engine()
    result = await engine.answer(
        AnswerRequest(call_id=call_id, question=transcript, language=language)
    )
    next_action = (
        f"{settings.public_base_url}/telephony/exotel/conversation"
        f"?call_id={call_id}&sid={sid}&language={language}"
    )
    verbs = [say(result.text or "Let me connect you to an advisor.")]
    if result.needs_escalation:
        verbs.append(say("Connecting you to the admissions team, please stay on the line."))
        verbs.append(_transfer_verbs())
    else:
        verbs.append(gather_speech(next_action, language=language, timeout=10))
    return exoml_response(*verbs)


def call_event_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Normalise an Exotel call-event callback into our internal shape."""
    return {
        "provider": "exotel",
        "provider_call_sid": data.get("CallSid") or data.get("call_sid") or data.get("Sid"),
        "event": data.get("event") or data.get("Event") or data.get("type"),
        "from_number": data.get("From") or data.get("caller_number"),
        "to_number": data.get("To") or data.get("virtual_number"),
        "status": data.get("Status") or data.get("call_status"),
        "duration": data.get("Duration") or data.get("call_duration"),
        "recording_url": data.get("RecordingUrl") or data.get("recording_url"),
        "raw": data,
    }
