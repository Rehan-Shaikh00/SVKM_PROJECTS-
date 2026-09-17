"""Call state machine definitions."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class CallState(str, enum.Enum):
    STARTED = "started"
    GREETING = "greeting"
    LANGUAGE_PROMPT = "language_prompt"       # waiting for the caller to speak a language
    LANGUAGE_REPROMPT = "language_reprompt"   # first attempt was not understood
    LANGUAGE_CONFIRM = "language_confirm"     # detected; asking the caller to confirm
    LANGUAGE_UNSUPPORTED = "language_unsupported"
    MENU = "menu"
    CONVERSATION = "conversation"
    FOLLOWUP_OFFER = "followup_offer"         # "should I send the details?"
    FOLLOWUP_DESTINATION = "followup_destination"  # asking for number/email
    ESCALATING = "escalating"
    TRANSFERRED = "transferred"
    CLOSING = "closing"
    ENDED = "ended"
    FAILED = "failed"


#: states in which a spoken utterance is interpreted as a language choice
LANGUAGE_STATES = {
    CallState.LANGUAGE_PROMPT,
    CallState.LANGUAGE_REPROMPT,
    CallState.LANGUAGE_CONFIRM,
    CallState.LANGUAGE_UNSUPPORTED,
}

#: states in which speech is a question for the knowledge base
CONVERSATION_STATES = {
    CallState.MENU,
    CallState.CONVERSATION,
    CallState.FOLLOWUP_OFFER,
    CallState.FOLLOWUP_DESTINATION,
}


class EscalationReason(str, enum.Enum):
    KB_NO_ANSWER = "kb_no_answer"
    LOW_CONFIDENCE = "low_confidence"
    CALLER_REQUESTED_HUMAN = "caller_requested_human"
    COMPLAINT = "complaint"
    PAYMENT_DISPUTE = "payment_dispute"
    SENSITIVE_OR_LEGAL = "sensitive_or_legal"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    TECHNICAL_FAILURE = "technical_failure"
    OUT_OF_SCOPE = "out_of_scope"
    SILENCE_TIMEOUT = "silence_timeout"
    MAX_DURATION = "max_duration"


@dataclass
class TurnTimings:
    asr_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0
    first_audio_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "asr_ms": round(self.asr_ms, 1),
            "retrieval_ms": round(self.retrieval_ms, 1),
            "llm_ms": round(self.llm_ms, 1),
            "tts_ms": round(self.tts_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "first_audio_ms": round(self.first_audio_ms, 1),
        }


@dataclass
class CallContext:
    """Everything the session knows about the live call."""

    call_id: str
    provider: str = "twilio"
    provider_call_sid: str | None = None
    stream_sid: str | None = None
    from_number: str | None = None
    to_number: str | None = None
    state: CallState = CallState.STARTED
    language: str | None = None
    language_confidence: float = 0.0
    language_method: str | None = None
    language_attempts: int = 0
    silence_count: int = 0
    turn_index: int = 0
    barge_in_count: int = 0
    started_at: float = 0.0
    answered_at: float = 0.0
    ended_at: float = 0.0
    last_assistant_text: str = ""
    last_answer_confidence: float = 0.0
    pending_followup: dict[str, Any] | None = None
    escalation_reason: str | None = None
    escalation_summary: str | None = None
    consecutive_low_confidence: int = 0
    unresolved_questions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def duration_seconds(self, now: float) -> float:
        return max(0.0, now - (self.answered_at or self.started_at))
