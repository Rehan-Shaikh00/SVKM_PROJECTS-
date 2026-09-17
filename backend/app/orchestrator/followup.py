"""Post-call follow-up: SMS / WhatsApp / email.

Voice is a terrible medium for lists and fee tables. When the assistant offers
details and the caller accepts, we send them as a message instead of reading them
out. Channels:

* **SMS / WhatsApp** via the Twilio Messaging API (in India, WhatsApp Business
  API needs a pre-approved template for business-initiated messages; we send the
  session message inside the 24-hour customer-service window, which is allowed).
* **Email** via SendGrid.
* **Simulator** — the message is pushed to the browser so you can see exactly
  what the caller would receive.

Every send is logged in `followups` with a hashed destination (no raw phone
numbers or emails are stored — DPDP Act 2023 data-minimisation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..ai.guardrails import hash_secret
from ..config import settings
from .channel import MediaChannel

logger = logging.getLogger("nims.followup")

MAX_SMS_BODY = 1400
MAX_LINE_ITEMS = 12

#: Contact block printed at the foot of every follow-up message. These are the
#: numbers and the website published on svkmnmimsgu.ac.in/contact-us. The
#: university publishes **no email address and no toll-free line**, so neither is
#: printed here: an invented address bounces and an invented number sends an
#: applicant to the wrong office. Update this block only from the contact page.
CONTACT_BLOCK: tuple[str, ...] = (
    "School of Technology, Management & Engineering: 02562 350620",
    "School of Pharmacy & Technology Management: 02562 350640",
    "School of Commerce: 02562 350600",
    "www.svkmnmimsgu.ac.in",
)

INSTITUTION_LINE = "SVKM NMIMS Global University, Dhule"


@dataclass
class FollowUpMessage:
    channel: str                       # sms | whatsapp | email
    destination: str
    title: str = ""
    items: list[str] = field(default_factory=list)
    body: str = ""
    language: str = "en-IN"
    call_id: str = ""
    sender: str = ""

    def render_body(self) -> str:
        if self.body:
            return self.body
        lines: list[str] = []
        if self.title:
            lines.append(self.title)
        lines.append(INSTITUTION_LINE)
        for item in self.items[:MAX_LINE_ITEMS]:
            text = str(item).strip()
            if text:
                lines.append(f"- {text}" if self.channel == "email" else text)
        lines.append("")
        lines.extend(CONTACT_BLOCK)
        body = "\n".join(lines)
        if self.channel != "email" and len(body) > MAX_SMS_BODY:
            body = body[: MAX_SMS_BODY - 1].rsplit("\n", 1)[0] + "\n…"
        return body


async def send_sms(destination: str, body: str, sender: str | None = None) -> dict[str, Any]:
    if not settings.followup_sms_enabled or not settings.twilio_configured:
        return {"status": "skipped", "reason": "SMS disabled or Twilio not configured"}
    account_sid, token = settings.twilio_credentials
    sender = sender or settings.twilio_helpline_number
    if not sender:
        return {"status": "failed", "reason": "TWILIO_HELLINE_NUMBER not set"}
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    try:  # pragma: no cover - network dependent
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                url,
                auth=(account_sid, token),
                data={"To": destination, "From": sender, "Body": body},
            )
        payload = resp.json()
        if resp.status_code >= 400:
            return {"status": "failed", "reason": payload.get("message", str(resp.status_code)),
                    "code": payload.get("code")}
        return {"status": "sent", "provider_message_id": payload.get("sid")}
    except Exception as exc:  # pragma: no cover
        logger.warning("SMS send failed: %s", exc)
        return {"status": "failed", "reason": str(exc)}


async def send_whatsapp(destination: str, body: str, sender: str | None = None) -> dict[str, Any]:
    if not settings.followup_whatsapp_enabled or not settings.twilio_configured:
        return {"status": "skipped", "reason": "WhatsApp disabled or Twilio not configured"}
    account_sid, token = settings.twilio_credentials
    sender = sender or settings.twilio_helpline_number
    to = destination if destination.startswith("whatsapp:") else f"whatsapp:{destination}"
    frm = sender if sender.startswith("whatsapp:") else f"whatsapp:{sender}"
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    try:  # pragma: no cover - network dependent
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                url, auth=(account_sid, token), data={"To": to, "From": frm, "Body": body}
            )
        payload = resp.json()
        if resp.status_code >= 400:
            return {"status": "failed", "reason": payload.get("message", str(resp.status_code))}
        return {"status": "sent", "provider_message_id": payload.get("sid")}
    except Exception as exc:  # pragma: no cover
        logger.warning("WhatsApp send failed: %s", exc)
        return {"status": "failed", "reason": str(exc)}


async def send_email(destination: str, subject: str, body: str) -> dict[str, Any]:
    if not settings.followup_email_enabled or not settings.sendgrid_api_key:
        return {"status": "skipped", "reason": "Email disabled or SENDGRID_API_KEY missing"}
    try:  # pragma: no cover - network dependent
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={
                    "Authorization": f"Bearer {settings.sendgrid_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "personalizations": [{"to": [{"email": destination}]}],
                    "from": {"email": settings.followup_from_email},
                    "subject": subject[:200],
                    "content": [{"type": "text/plain", "value": body}],
                },
            )
        if resp.status_code >= 400:
            return {"status": "failed", "reason": resp.text[:300]}
        return {"status": "sent", "provider_message_id": resp.headers.get("X-Message-Id")}
    except Exception as exc:  # pragma: no cover
        logger.warning("Email send failed: %s", exc)
        return {"status": "failed", "reason": str(exc)}


async def deliver(
    channel: MediaChannel,
    message: FollowUpMessage,
) -> dict[str, Any]:
    """Send through the configured provider and always echo to the channel.

    Echoing matters for the simulator: you see the SMS the caller would get,
    which makes the offer-details flow testable end to end.
    """
    body = message.render_body()
    result: dict[str, Any]
    if message.channel == "sms":
        result = await send_sms(message.destination, body, message.sender or None)
    elif message.channel == "whatsapp":
        result = await send_whatsapp(message.destination, body, message.sender or None)
    elif message.channel == "email":
        result = await send_email(message.destination, message.title or "NMIMS Global University, Dhule details", body)
    else:
        result = {"status": "failed", "reason": f"unknown channel {message.channel}"}

    destination_ref = hash_secret(message.destination) if message.destination else ""
    await channel.send_event(
        "followup",
        {
            "channel": message.channel,
            "status": result.get("status"),
            "reason": result.get("reason"),
            "destination_ref": destination_ref,
            "preview": body,
            "title": message.title,
            "items": message.items[:MAX_LINE_ITEMS],
        },
    )
    return {
        **result,
        "channel": message.channel,
        "destination_ref": destination_ref,
        "body": body,
        "provider_message_id": result.get("provider_message_id"),
    }


def looks_like_phone(text: str) -> bool:
    import re

    digits = re.sub(r"\D", "", text or "")
    return 10 <= len(digits) <= 13


def looks_like_email(text: str) -> bool:
    import re

    return bool(re.match(r"^[\w.+-]+@[\w-]+\.[\w.-]{2,}$", (text or "").strip()))


def extract_phone(text: str) -> str:
    """Pull an Indian mobile number out of a spoken answer ('98 76 54 32 10')."""
    import re

    digits = re.sub(r"\D", "", text or "")
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == 10:
        return f"+91{digits}"
    match = re.search(r"\+?\d{10,13}", text or "")
    return match.group(0) if match else ""


def extract_email(text: str) -> str:
    import re

    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}", text or "")
    return match.group(0) if match else ""
