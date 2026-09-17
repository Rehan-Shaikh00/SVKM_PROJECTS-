"""Twilio Media Streams channel.

Implements the orchestrator's `MediaChannel` on top of a Twilio bidirectional
Media Streams websocket:

inbound events   connected / start / media / dtmf / mark / stop
outbound events  media / mark / clear / stop / redirect

Audio is 8 kHz G.711 μ-law, base64 encoded, 20 ms frames.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import WebSocket

from ..config import settings
from ..orchestrator.channel import InboundEvent, MediaChannel
from ..voice.audio import FRAME_MS

logger = logging.getLogger("nims.twilio.stream")


class TwilioMediaChannel(MediaChannel):
    name = "twilio"
    sample_rate = 8000

    def __init__(self, call_id: str, websocket: WebSocket) -> None:
        super().__init__(call_id)
        self.websocket = websocket
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self.account_sid: str | None = None
        self.custom_parameters: dict[str, Any] = {}
        self.from_number: str | None = None
        self.to_number: str | None = None
        self._inbound: asyncio.Queue[InboundEvent | None] = asyncio.Queue()
        self._send_lock = asyncio.Lock()
        self.marks_sent = 0
        self.redirected = False

    # -- receive ----------------------------------------------------------- #
    async def push_inbound(self, event: InboundEvent) -> None:
        await self._inbound.put(event)

    async def inbound(self) -> AsyncIterator[InboundEvent]:
        while not self.closed:
            event = await self._inbound.get()
            if event is None:
                return
            yield event

    async def handle_message(self, message: dict[str, Any]) -> None:
        """Translate one Twilio websocket message into an InboundEvent."""
        event = message.get("event")
        if event == "connected":
            self.stream_sid = message.get("streamSid") or self.stream_sid
            logger.info("twilio stream connected: %s", self.stream_sid)
            return
        if event == "start":
            start = message.get("start") or {}
            self.stream_sid = message.get("streamSid") or start.get("streamSid") or self.stream_sid
            self.call_sid = start.get("callSid") or self.call_sid
            self.account_sid = start.get("accountSid")
            self.custom_parameters = start.get("customParameters") or {}
            self.from_number = self.custom_parameters.get("from") or self.from_number
            self.to_number = self.custom_parameters.get("to") or self.to_number
            media_format = start.get("mediaFormat") or {}
            logger.info(
                "twilio stream start sid=%s call=%s format=%s",
                self.stream_sid, self.call_sid, media_format,
            )
            return
        if event == "media":
            media = message.get("media") or {}
            payload = media.get("payload")
            if not payload:
                return
            try:
                audio = base64.b64decode(payload)
            except Exception:  # pragma: no cover
                logger.warning("bad base64 media payload")
                return
            await self.push_inbound(InboundEvent(type="audio", audio=audio, payload=media))
            return
        if event == "dtmf":
            digit = (message.get("dtmf") or {}).get("digit", "")
            await self.push_inbound(InboundEvent(type="dtmf", digit=digit))
            return
        if event == "mark":
            name = (message.get("mark") or {}).get("name", "")
            await self.push_inbound(InboundEvent(type="mark", payload={"name": name}))
            return
        if event == "stop":
            logger.info("twilio stream stopped: %s", self.stream_sid)
            await self.push_inbound(InboundEvent(type="hangup", payload={"reason": "stream_stop"}))
            await self._inbound.put(None)
            self.closed = True
            return
        logger.debug("unhandled twilio event: %s", event)

    # -- send -------------------------------------------------------------- #
    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self.closed:
            return
        async with self._send_lock:
            try:
                await self.websocket.send_text(json.dumps(payload))
            except Exception as exc:  # pragma: no cover - client vanished
                logger.info("twilio websocket send failed: %s", exc)
                self.closed = True

    async def send_audio(self, mulaw: bytes) -> None:
        if not mulaw or not self.stream_sid:
            return
        self.audio_frames_sent += 1
        self.audio_bytes_sent += len(mulaw)
        await self._send_json(
            {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(mulaw).decode("ascii")},
            }
        )

    async def send_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Session state events are logged; only a few reach Twilio."""
        body = {"type": event_type, "call_id": self.call_id, "ts": time.time()}
        body.update(payload)
        logger.debug("twilio channel event %s", event_type)
        # the caller leg does not render JSON, so nothing is sent for most events

    async def clear_audio(self) -> None:
        """Barge-in: tell Twilio to drop everything it has buffered."""
        if not self.stream_sid:
            return
        await self._send_json({"event": "clear", "streamSid": self.stream_sid})

    async def mark(self, name: str) -> None:
        if not self.stream_sid:
            return
        self.marks_sent += 1
        await self._send_json(
            {"event": "mark", "streamSid": self.stream_sid, "mark": {"name": name}}
        )

    async def transfer(
        self, target: str, *, target_type: str = "number", whisper_text: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Hand the call over by redirecting Twilio to our transfer TwiML."""
        from . import twiml

        if not self.stream_sid:
            return
        url = twiml.transfer_url(
            self.call_id, self.call_sid or "", reason=str((context or {}).get("reason", ""))
        )
        self.redirected = True
        await self._send_json(
            {"event": "redirect", "streamSid": self.stream_sid, "redirect": {"url": url}}
        )
        _ = target, target_type, whisper_text

    async def play_url(self, url: str, *, loop: bool = True) -> None:
        # hold music is handled by the <Enqueue> waitUrl, not the stream
        logger.debug("hold music requested: %s (loop=%s)", url, loop)

    async def hangup(self, reason: str = "completed") -> None:
        if not self.stream_sid:
            return
        await self._send_json({"event": "stop", "streamSid": self.stream_sid})
        await self.close()

    async def close(self) -> None:
        if self.closed:
            return
        await super().close()
        await self._inbound.put(None)
        try:
            await self.websocket.close()
        except Exception:  # pragma: no cover
            pass


def frame_duration_ms(mulaw_bytes: int) -> float:
    """A Twilio media frame is 20 ms of 8 kHz μ-law = 160 bytes."""
    return mulaw_bytes / (8000 / 1000) if mulaw_bytes else 0.0


def validate_twilio_request(headers: dict[str, str], url: str, body: bytes | str) -> bool:
    """X-Twilio-Signature validation. Skipped only when credentials are absent
    (local development with the simulator)."""
    if not settings.twilio_configured:
        logger.debug("twilio credentials absent; skipping signature validation")
        return True
    signature = headers.get("x-twilio-signature") or headers.get("X-Twilio-Signature")
    if not signature:
        return False
    try:  # pragma: no cover - depends on credentials
        from twilio.request_validator import RequestValidator

        _, token = settings.twilio_credentials
        validator = RequestValidator(token)
        payload: Any = body
        if isinstance(body, bytes):
            payload = body.decode("utf-8")
        if payload:
            try:
                parsed = json.loads(payload)
                payload = {str(k): str(v) for k, v in parsed.items()}
            except json.JSONDecodeError:
                from urllib.parse import parse_qs

                payload = {k: v[0] for k, v in parse_qs(payload).items()}
        else:
            payload = {}
        return validator.validate(url, payload, signature)
    except Exception as exc:  # pragma: no cover
        logger.warning("twilio signature validation error: %s", exc)
        return False


def expected_frame_bytes() -> int:
    return int(8000 * FRAME_MS / 1000)
