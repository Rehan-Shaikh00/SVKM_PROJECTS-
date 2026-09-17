"""TwiML builders for the Twilio Voice integration.

Call flow:

    caller dials the helpline
      → Twilio POSTs /telephony/twilio/voice
      → we answer with <Connect><Stream> pointing at our websocket
      → the websocket carries bidirectional μ-law audio for the whole call
      → on escalation we send a `redirect` event to /telephony/twilio/transfer
      → that returns <Dial>/<Enqueue> TwiML and the AI call leg ends

Using Media Streams (instead of <Gather>/<Say> round trips) is what makes
barge-in and sub-second responses possible: audio flows continuously in both
directions and we decide turn boundaries ourselves.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from twilio.twiml.voice_response import (
    Connect,
    Dial,
    Enqueue,
    Hangup,
    Redirect,
    VoiceResponse,
)

from ..config import settings
from ..i18n.languages import get_language

#: Azure/Google neural voices are not usable in <Say>; Twilio's own voices are.
#: We only use <Say> for the whisper page and fallbacks, never for answers.
TWIML_VOICE = "Polly.Aditi"       # Hindi/English bilingual
TWIML_LANGUAGE = "en-IN"


def stream_websocket_url() -> str:
    base = settings.websocket_base_url
    return f"{base}/telephony/twilio/media-stream"


def transfer_url(call_id: str, call_sid: str, reason: str = "") -> str:
    query = urlencode({"call_id": call_id, "call_sid": call_sid, "reason": reason})
    return f"{settings.public_base_url}/telephony/twilio/transfer?{query}"


def answer_with_stream(
    *,
    call_sid: str = "",
    from_number: str = "",
    to_number: str = "",
    custom_parameters: dict[str, Any] | None = None,
    status_callback: bool = True,
) -> str:
    """The webhook response that opens the bidirectional media stream."""
    response = VoiceResponse()
    params = {
        "callSid": call_sid,
        "from": from_number,
        "to": to_number,
        **(custom_parameters or {}),
    }
    connect = Connect()
    stream = connect.stream(url=stream_websocket_url())
    for key, value in params.items():
        if value:
            stream.parameter(name=key, value=str(value))
    response.append(connect)
    # Safety net: if the websocket never produces audio the call should not hang
    # forever. Twilio keeps the call alive while <Connect> is active.
    return str(response)


def whisper_page(summary: str, *, language: str = "en-IN") -> str:
    """Played to the agent just before the caller is bridged.

    The bridging line is always English because the agent reads it, but the
    summary usually quotes the caller's own words, so it is synthesised in the
    call's language: Polly.Aditi is Hindi/English bilingual and only pronounces
    Devanagari correctly when the locale matches. (The previous
    ``if language == "en-IN" else`` had the same string on both arms.)
    """
    response = VoiceResponse()
    text = summary or "The AI assistant could not resolve this caller's question."
    response.say(text[:900], voice=TWIML_VOICE, language=language or TWIML_LANGUAGE)
    response.pause(length=1)
    response.say("Bridging the caller now.", voice=TWIML_VOICE, language=TWIML_LANGUAGE)
    return str(response)


def transfer_twiml(
    *,
    target: str,
    target_type: str = "number",
    whisper_url: str | None = None,
    hold_music: str | None = None,
    max_wait_seconds: int | None = None,
    caller_id: str | None = None,
) -> str:
    """<Dial> a number (with a whisper page) or <Enqueue> into a TaskRouter queue."""
    response = VoiceResponse()
    max_wait = max_wait_seconds or settings.escalation_max_wait_seconds

    if target_type == "queue" and target:
        response.say(
            "Connecting you to an admissions advisor, please stay on the line.",
            voice=TWIML_VOICE, language=TWIML_LANGUAGE,
        )
        enqueue = Enqueue(
            name=target,
            wait_url=f"{settings.public_base_url}/telephony/twilio/queue-wait",
            wait_url_method="POST",
            action=f"{settings.public_base_url}/telephony/twilio/queue-result",
        )
        response.append(enqueue)
        return str(response)

    if not target:
        response.say(
            "Our advisors are unavailable right now. Please call 1 800 120 1020.",
            voice=TWIML_VOICE, language=TWIML_LANGUAGE,
        )
        response.hangup()
        return str(response)

    dial = Dial(
        caller_id=caller_id or settings.twilio_helline_number or None,
        time_limit=max_wait,
        record="do-not-record" if not settings.call_recording_enabled else "record-from-ringing",
        action=f"{settings.public_base_url}/telephony/twilio/dial-result",
    )
    dial.number(target, url=whisper_url)
    response.append(dial)
    response.say(
        "The advisor is unavailable. Please call back on 1 800 120 1020.",
        voice=TWIML_VOICE, language=TWIML_LANGUAGE,
    )
    response.hangup()
    return str(response)


def queue_wait_twiml() -> str:
    """Music + periodic reassurance while the caller waits in the queue."""
    response = VoiceResponse()
    response.play(settings.hold_music_url, loop=3)
    response.say(
        "You are still in the queue. An admissions advisor will be with you shortly.",
        voice=TWIML_VOICE, language=TWIML_LANGUAGE,
    )
    response.pause(length=2)
    response.redirect(f"{settings.public_base_url}/telephony/twilio/queue-wait")
    return str(response)


def fallback_twiml(message: str | None = None, *, hangup: bool = True) -> str:
    response = VoiceResponse()
    response.say(
        message
        or "We are having a technical problem. Please call the admissions office on "
        "1 800 120 1020.",
        voice=TWIML_VOICE,
        language=TWIML_LANGUAGE,
    )
    if hangup:
        response.hangup()
    return str(response)


def recording_consent_twiml(language: str = "en-IN") -> str:
    """Used when the deployment wants Twilio (not the websocket) to open the call."""
    lang = get_language(language)
    response = VoiceResponse()
    response.say(
        "Thank you for calling NMIMS Global University, Dhule. This call is answered by an AI "
        "assistant and may be recorded for quality.",
        voice=TWIML_VOICE,
        language=TWIML_LANGUAGE,
    )
    response.pause(length=1)
    _ = lang
    return str(response)


def redirect_twiml(url: str) -> str:
    response = VoiceResponse()
    response.append(Redirect(url))
    return str(response)


def hangup_twiml() -> str:
    response = VoiceResponse()
    response.append(Hangup())
    return str(response)
