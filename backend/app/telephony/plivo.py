"""Plivo adapter.

Plivo supports speech input natively through the `<GetInput>` XML verb
(`inputType="dtmf speech"`), so a webhook-mode integration works without a media
server. Latency is higher than a continuous media stream (one HTTP round trip per
turn) and barge-in is not possible, so for production we recommend Plivo's
SIP trunking into a media server and the shared websocket adapter.

Response fields posted back to our `action` URL include `Speech`, `Digits`,
`CallUUID` and `From`/`To`.
"""

from __future__ import annotations

import logging
from typing import Any
from xml.sax.saxutils import escape

from ..config import settings

logger = logging.getLogger("nims.plivo")


def xml_response(*verbs: str) -> str:
    return "<?xml version=\"1.0\" encoding=\"utf-8\"?><Response>" + "".join(verbs) + "</Response>"


def speak(text: str, *, voice: str = "WOMAN", language: str = "en-IN") -> str:
    return f'<Speak voice="{voice}" language="{language}">{escape(text)}</Speak>'


def get_input(
    action: str,
    *,
    input_type: str = "speech",
    language: str = "en-IN",
    speech_timeout: int = 8,
    finish_on_key: str = "#",
    method: str = "POST",
    log: bool = False,
) -> str:
    return (
        f'<GetInput action="{escape(action)}" method="{method}" '
        f'inputType="{input_type}" language="{language}" '
        f'speechTimeout="{speech_timeout}" finishOnKey="{finish_on_key}" '
        f'log="{"true" if log else "false"}" />'
    )


def dial(number: str, *, caller_id: str = "", time_limit: int = 300) -> str:
    attributes = f' timeLimit="{time_limit}"'
    if caller_id:
        attributes += f' callerID="{escape(caller_id)}"'
    return f"<Dial{attributes}><Number>{escape(number)}</Number></Dial>"


def hangup(schedule: int = 0) -> str:
    return f'<Hangup schedule="{schedule}"/>'


def redirect(url: str) -> str:
    return f"<Redirect>{escape(url)}</Redirect>"


def play(url: str, *, loop: int = 1) -> str:
    return f'<Play loop="{loop}">{escape(url)}</Play>'


def _transfer_verbs() -> str:
    agents = settings.escalation_agent_list
    if not agents:
        return speak(
            "Our advisors are unavailable right now. Please call 1 800 120 1020.",
            language="en-IN",
        ) + hangup()
    return dial(agents[0], caller_id=settings.twilio_helline_number)


def build_inbound_xml(call_uuid: str) -> str:
    """First response to Plivo's inbound call webhook."""
    action = f"{settings.public_base_url}/telephony/plivo/language?uuid={call_uuid}"
    return xml_response(
        speak(
            "Thank you for calling NMIMS Global University, Dhule. This call is answered by an AI "
            "assistant and may be recorded for quality."
        ),
        speak("Please tell me your preferred language. कृपया अपनी भाषा बताइए।"),
        get_input(action, input_type="dtmf speech", language="en-IN", speech_timeout=8),
        speak("Sorry, we could not understand. Connecting you to an advisor."),
        _transfer_verbs(),
    )


async def build_language_xml(call_uuid: str, speech: str, digits: str) -> str:
    """Identify the spoken language, then open the conversation loop."""
    from ..voice.lid.local import detect_language_name, detect_language_text

    text = speech or digits
    named = detect_language_name(text)
    result = named or detect_language_text(text)
    language = (
        result.language if result.confidence >= settings.lid_confidence_threshold else "en-IN"
    )
    if digits and digits in {"1", "2", "3"} and not speech:
        language = {"1": "en-IN", "2": "hi-IN", "3": "raj-IN"}[digits]
    action = (
        f"{settings.public_base_url}/telephony/plivo/conversation"
        f"?uuid={call_uuid}&language={language}"
    )
    return xml_response(
        speak(f"Great, I will continue in {language.split('-')[0]}."),
        speak(
            "You can ask me about courses, admission, fees, eligibility, scholarships, "
            "hostels and placements. What would you like to know?"
        ),
        get_input(action, input_type="speech", language=language, speech_timeout=10),
        speak("Let me connect you to an advisor."),
        _transfer_verbs(),
    )


async def build_conversation_xml(
    call_uuid: str, language: str, speech: str, *, session: Any = None
) -> str:
    """Answer one turn with the RAG engine and gather the next utterance."""
    from ..ai.rag import AnswerRequest
    from ..dependencies import get_answer_engine

    engine = await get_answer_engine()
    result = await engine.answer(
        AnswerRequest(call_id=call_uuid, question=speech, language=language)
    )
    action = (
        f"{settings.public_base_url}/telephony/plivo/conversation"
        f"?uuid={call_uuid}&language={language}"
    )
    verbs = [speak(result.text or "Let me connect you to an advisor.", language=language)]
    if result.needs_escalation:
        verbs.append(speak("Connecting you to the admissions team, please stay on the line."))
        verbs.append(_transfer_verbs())
    elif result.followup and result.followup.get("items"):
        verbs.append(speak("I can send you the full details by SMS. Would you like that?"))
        verbs.append(get_input(action, language=language, speech_timeout=6))
    else:
        verbs.append(get_input(action, input_type="speech", language=language, speech_timeout=10))
    return xml_response(*verbs)
