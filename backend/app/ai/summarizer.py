"""Call summarisation for the human handoff.

When a call escalates, the agent must not have to ask the caller to repeat
themselves. The summary is generated in two ways:

* with an LLM configured → a short structured brief (who, what, what's missing)
* without → an extractive brief built from the transcript (always available)

The result is written to `escalations.whisper_summary` and, when
`ESCALATION_WHISPER_CONTEXT` is on, spoken to the agent on a whisper page before
the caller is bridged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .intents import detect_intent
from .llm.base import LLM, Message

logger = logging.getLogger("nims.summary")

WHISPER_MAX_CHARS = 420

SUMMARY_PROMPT = """You are writing a handover note for a university admissions
counsellor who is about to take over a phone call from an AI assistant.

Write 3-4 short sentences in plain English covering:
1. the caller's language
2. what the caller wants (programme, level, year if known)
3. what the AI already told them
4. exactly what the AI could NOT answer, and why it escalated

Do not invent details. Do not use bullet points or markdown. Maximum 60 words.

Call transcript:
{transcript}

Escalation reason: {reason}
"""


@dataclass
class CallSummary:
    text: str
    whisper: str
    language: str = "en-IN"
    caller_intent: str = "other"
    programmes: list[str] = field(default_factory=list)
    answered: list[str] = field(default_factory=list)
    unanswered: list[str] = field(default_factory=list)
    escalated_reason: str | None = None
    turn_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "whisper": self.whisper,
            "language": self.language,
            "caller_intent": self.caller_intent,
            "programmes": self.programmes,
            "answered": self.answered,
            "unanswered": self.unanswered,
            "escalated_reason": self.escalated_reason,
            "turn_count": self.turn_count,
        }


def _clip(text: str, limit: int) -> str:
    """Trim to `limit` characters on a word boundary.

    A hard ``text[:160]`` slice used to end agent hand-off briefs mid-word
    ("…marksheets and degree, whe"), which reads as corruption when it is
    spoken aloud to a human agent.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    if cut <= limit // 2:  # single long token — no usable break point
        cut = limit
    return f"{text[:cut].rstrip(' ,;:.')}…"


def extractive_summary(
    turns: list[dict[str, Any]],
    *,
    language: str,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CallSummary:
    """Build a brief from the transcript without an LLM."""
    caller_turns = [t for t in turns if t.get("role") == "caller" and t.get("text")]
    assistant_turns = [t for t in turns if t.get("role") == "assistant" and t.get("text")]

    programmes: list[str] = []
    intents: list[str] = []
    unanswered: list[str] = []
    for turn in caller_turns:
        intent = detect_intent(str(turn.get("text") or ""))
        intents.append(intent.intent)
        programmes.extend(intent.course_tokens)
        programmes.extend(s.title() for s in intent.specialisations)
        if turn.get("metadata", {}).get("needs_escalation") or turn.get("grounded") is False:
            unanswered.append(_clip(str(turn.get("text")), 120))

    answered: list[str] = []
    for turn in assistant_turns[-3:]:
        text = str(turn.get("text") or "").strip()
        if text:
            answered.append(_clip(text, 160))

    primary_intent = max(set(intents), key=intents.count) if intents else "other"
    unique_programmes = list(dict.fromkeys(p for p in programmes if p))[:4]

    lines = [f"Caller speaks {language}."]
    if unique_programmes:
        lines.append(f"Asked about {', '.join(unique_programmes)}.")
    else:
        lines.append(f"Primary topic: {primary_intent.replace('_', ' ')}.")
    if answered:
        lines.append(f"Assistant covered: {answered[-1]}")
    if unanswered:
        lines.append(f"Could not answer: {unanswered[-1]}")
    if reason:
        lines.append(f"Escalated because {reason.replace('_', ' ')}.")

    text = " ".join(lines)
    whisper = _clip(text, WHISPER_MAX_CHARS)
    return CallSummary(
        text=text,
        whisper=whisper,
        language=language,
        caller_intent=primary_intent,
        programmes=unique_programmes,
        answered=answered,
        unanswered=unanswered,
        escalated_reason=reason,
        turn_count=len(turns),
        metadata=metadata or {},
    )


async def summarise(
    turns: list[dict[str, Any]],
    *,
    language: str,
    reason: str | None = None,
    llm: LLM | None = None,
    metadata: dict[str, Any] | None = None,
) -> CallSummary:
    base = extractive_summary(turns, language=language, reason=reason, metadata=metadata)
    if llm is None or len(turns) < 2:
        return base

    transcript = "\n".join(
        f"{t.get('role', 'caller').upper()}: {t.get('text', '')}" for t in turns[-14:]
    )
    try:
        result = await llm.complete(
            system="You write concise call handover notes. Never invent facts.",
            messages=[
                Message(
                    role="user",
                    content=SUMMARY_PROMPT.format(transcript=transcript, reason=reason or "not stated"),
                )
            ],
            max_tokens=180,
            temperature=0.1,
            language="en-IN",
        )
        if result.text.strip():
            base.text = result.text.strip()
            base.whisper = _clip(result.text.strip(), WHISPER_MAX_CHARS)
            base.metadata["summariser"] = llm.name
        else:
            base.metadata["summariser"] = "extractive"
    except Exception as exc:
        logger.warning("LLM summarisation failed, keeping extractive brief: %s", exc)
        base.metadata["summariser"] = "extractive"
        base.metadata["summariser_error"] = str(exc)[:200]
    else:
        base.metadata.setdefault("summariser", "llm")
    return base


def whisper_twiml_text(summary: CallSummary) -> str:
    """Text spoken to the agent before bridging the caller."""
    prefix = "Whisper: an AI assistant handled this call first. "
    body = summary.whisper or summary.text
    if not body:
        return prefix + "The caller needs admissions help."
    return _clip(prefix + body, WHISPER_MAX_CHARS)
