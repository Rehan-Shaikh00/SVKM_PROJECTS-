"""Escalation planning: who to transfer to, and with what context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import settings


@dataclass
class EscalationPlan:
    target: str
    target_type: str = "number"        # number | queue
    whisper: str = ""
    hold_music: str = ""
    max_wait_seconds: int = 300
    agents: list[str] = field(default_factory=list)
    reason: str = ""
    context: dict[str, Any] = field(default_factory=dict)


def plan_escalation(
    *,
    reason: str,
    summary: str = "",
    language: str = "en-IN",
    extra_context: dict[str, Any] | None = None,
) -> EscalationPlan:
    """Choose queue vs direct numbers.

    Production guidance: prefer a **queue** (Twilio TaskRouter / Exotel skill
    based routing) so calls are distributed and wait times are observable.
    Direct numbers are the fallback when no queue is configured.
    """
    agents = settings.escalation_agent_list
    queue = settings.escalation_queue_name if agents or settings.escalation_queue_name else ""
    use_queue = bool(queue and settings.telephony_provider == "twilio")

    context = {
        "reason": reason,
        "language": language,
        "summary": summary,
        "helpline": settings.twilio_helline_number,
    }
    if extra_context:
        context.update(extra_context)

    if use_queue:
        return EscalationPlan(
            target=queue,
            target_type="queue",
            whisper=summary,
            hold_music=settings.hold_music_url,
            max_wait_seconds=settings.escalation_max_wait_seconds,
            agents=agents,
            reason=reason,
            context=context,
        )
    if agents:
        return EscalationPlan(
            target=agents[0],
            target_type="number",
            whisper=summary,
            hold_music=settings.hold_music_url,
            max_wait_seconds=settings.escalation_max_wait_seconds,
            agents=agents,
            reason=reason,
            context=context,
        )
    # Nothing configured: tell the caller the office number instead of dead air.
    return EscalationPlan(
        target="",
        target_type="none",
        whisper=summary,
        reason=reason,
        context=context,
    )


NO_AGENT_LINE = {
    "en-IN": "Our advisors are on another call. Please call the admissions office on 1 800 120 1020, and they will help you.",
    "hi-IN": "हमारे सलाहकार दूसरी कॉल पर हैं। कृपया एडमिशन ऑफिस को 1 800 120 1020 पर कॉल करें, वे आपकी मदद करेंगे।",
    "raj-IN": "म्हारा सलाहकार दूसरी कॉल पर है। एडमिशन ऑफिस ने 1 800 120 1020 पर फोन करजो, ई मदद करसी।",
}


# --------------------------------------------------------------------------- #
# pending transfers
# --------------------------------------------------------------------------- #
# The session decides *who* to transfer to, then redirects the provider to a
# TwiML/webhook URL which must answer in a few milliseconds. Passing the plan
# through the database would be too slow (and racy), so it is parked here in
# memory keyed by call id and consumed by the transfer endpoint.

_PENDING: dict[str, tuple[float, EscalationPlan]] = {}
_PENDING_TTL_SECONDS = 600.0


def register_pending_transfer(call_id: str, plan: EscalationPlan) -> None:
    import time

    now = time.time()
    for key in [k for k, (stamp, _) in _PENDING.items() if now - stamp > _PENDING_TTL_SECONDS]:
        _PENDING.pop(key, None)
    _PENDING[call_id] = (now, plan)


def pop_pending_transfer(call_id: str) -> EscalationPlan | None:
    entry = _PENDING.pop(call_id, None)
    if entry is None:
        return None
    _stamp, plan = entry
    return plan


def peek_pending_transfer(call_id: str) -> EscalationPlan | None:
    entry = _PENDING.get(call_id)
    return entry[1] if entry else None
