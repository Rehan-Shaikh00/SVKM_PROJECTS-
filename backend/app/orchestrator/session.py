"""The call session — one instance per live call.

Responsibilities
----------------
* drive the state machine: greeting → spoken language selection → confirmation →
  conversation → follow-up/escalation → close
* keep latency low: audio frames are handed to the ASR immediately, LLM tokens
  are streamed into a speech queue so TTS starts on the first sentence
* barge-in: speech detected while the assistant is talking flushes the speech
  queue and tells the telephony provider to clear its audio buffer
* never leave the caller in silence: no-input reprompts, LLM failure → template
  fallback, unanswerable question → human transfer with a spoken brief
* log everything (transcript, timings, citations, escalations) asynchronously

The session is provider-agnostic: it only touches `MediaChannel`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..ai.intents import IntentResult, detect_intent, extract_entities, is_control_utterance
from ..ai.llm.base import LLM, Message
from ..ai.rag import AnswerEngine, AnswerRequest, AssistantAnswer
from ..ai.summarizer import summarise, whisper_twiml_text
from ..ai.templates import KNOWLEDGE_GAP_REASONS
from ..config import settings
from ..db import SessionLocal
from ..i18n.languages import (
    asr_locale,
    display_name,
    get_language,
    script_line,
)
from ..kb.retriever import HybridRetriever
from ..voice.asr.base import StreamingASR, TranscriptSegment
from ..voice.asr.client import ClientASR, WhisperChunkASR
from ..voice.audio import FRAME_MS, EnergyVAD, mulaw_to_pcm16
from ..voice.lid.detector import LanguageIdentifier
from ..voice.lid.local import named_language_only
from ..voice.tts.base import TTS
from .call_logger import CallLogger, redacted_caller
from .channel import MediaChannel
from .escalation import NO_AGENT_LINE, plan_escalation, register_pending_transfer
from .followup import (
    FollowUpMessage,
    deliver,
    extract_email,
    extract_phone,
    looks_like_email,
)
from .states import CONVERSATION_STATES, LANGUAGE_STATES, CallContext, CallState, TurnTimings

logger = logging.getLogger("nims.session")

#: seconds of silence before a reprompt
SILENCE_TIMEOUT_SECONDS = 7.0
#: confidence above which the detected language is adopted without asking again
HIGH_LID_CONFIDENCE = 0.82

#: Confidence a *mid-call* switch needs, on each of two consecutive turns.
#: Measured over 24 ordinary Hindi and Marathi helpline questions, the detector
#: picked the right language every time, but only a third of them reached the
#: 0.82 that immediate adoption asks for — so a caller who moved into Marathi
#: mid-call was almost never followed, and stayed answered in English. At 0.75
#: nearly all of them clear it while the genuinely ambiguous ones (0.56-0.61,
#: where Hindi and Marathi share every word in the sentence) still do not. The
#: two-turn requirement is what makes the lower bar safe here and not at the
#: language prompt, where one turn is all there is.
MID_CALL_SWITCH_CONFIDENCE = 0.75

#: How many turns a named programme stays the subject of the call. Long enough
#: for "and the fees?", "how many seats?", "what documents?" to follow one
#: another; short enough that a caller who has moved on is not silently
#: answered about a programme they asked about a minute ago.
PROGRAMME_CONTEXT_TURNS = 3
#: how long we wait for the client to finish speaking text (browser TTS)
WORDS_PER_SECOND = 2.6


@dataclass
class SessionDependencies:
    retriever: HybridRetriever
    answer_engine: AnswerEngine
    tts: TTS
    logger: CallLogger
    asr_factory: Any = None            # callable() -> StreamingASR
    lid_factory: Any = None            # callable(candidates) -> LanguageIdentifier
    summariser_llm: LLM | None = None


@dataclass
class _SpeechItem:
    text: str
    language: str
    done: asyncio.Event = field(default_factory=asyncio.Event)
    wait: bool = True
    first_audio_ms: float = 0.0


class CallSession:
    def __init__(
        self,
        channel: MediaChannel,
        deps: SessionDependencies,
        context: CallContext,
        *,
        asr: StreamingASR | None = None,
    ) -> None:
        self.channel = channel
        self.deps = deps
        self.ctx = context
        self.asr = asr
        #: The browser simulator can run in "text" mode, where nothing is played
        #: aloud and therefore no client will ever report `speech_done`. Waiting
        #: for one anyway stalled the call for the whole budget on every turn.
        self.text_mode = str(context.metadata.get("mode", "")).lower() == "text"
        self.lid: LanguageIdentifier | None = None
        self.vad = EnergyVAD(
            sample_rate=8000,
            endpoint_silence_ms=settings.asr_endpoint_silence_ms,
        )
        self.state = CallState.STARTED
        self.transcript: list[dict[str, Any]] = []
        self.history: list[Message] = []
        self.timings: list[TurnTimings] = []
        self.intents_seen: list[str] = []

        self._done = asyncio.Event()
        self._interrupt = asyncio.Event()
        self._speech_queue: asyncio.Queue[_SpeechItem | None] = asyncio.Queue()
        self._speech_idle = asyncio.Event()
        self._speech_idle.set()
        self._speaking = False
        self._turn_lock = asyncio.Lock()
        self._utterance_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._tasks: list[asyncio.Task[Any]] = []
        self._last_activity = time.time()
        self._last_turn_started = 0.0
        self._first_audio_at: dict[int, float] = {}
        self._pending_language: str | None = None
        self._pending_followup: dict[str, Any] | None = None
        self._awaiting_destination = False
        self._language_switch_streak = 0
        #: The programme record the caller was last talking about, and the turn
        #: it was established on. Follow-up questions inherit it.
        self._programme_context_title = ""
        self._programme_context_tokens: list[str] = []
        self._programme_context_specs: list[str] = []
        self._programme_context_turn = -99
        self._answer_turn_seq = 0
        self._asr_info: dict[str, Any] = {}
        self._client_speech_done = asyncio.Event()
        self._client_speech_done.set()
        self.session_language = "en-IN"
        self.escalated = False
        self.resolved = False

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    @property
    def language(self) -> str:
        return self.ctx.language or self.session_language

    async def run(self) -> CallContext:
        self.ctx.started_at = time.time()
        self.ctx.answered_at = self.ctx.started_at
        await self._open()
        self._tasks = [
            asyncio.create_task(self._media_loop(), name=f"{self.ctx.call_id}:media"),
            asyncio.create_task(self._asr_loop(), name=f"{self.ctx.call_id}:asr"),
            asyncio.create_task(self._utterance_loop(), name=f"{self.ctx.call_id}:turns"),
            asyncio.create_task(self._speech_worker(), name=f"{self.ctx.call_id}:speech"),
            asyncio.create_task(self._silence_watchdog(), name=f"{self.ctx.call_id}:silence"),
            asyncio.create_task(self._duration_watchdog(), name=f"{self.ctx.call_id}:duration"),
        ]
        try:
            await self._greeting_sequence()
            await self._done.wait()
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:
            logger.exception("call session crashed: %s", exc)
            await self._fail(str(exc))
        finally:
            await self._teardown()
        return self.ctx

    async def _open(self) -> None:
        self.deps.logger.start_call(
            {
                "call_id": self.ctx.call_id,
                "provider": self.ctx.provider,
                "provider_call_sid": self.ctx.provider_call_sid,
                "stream_sid": self.ctx.stream_sid,
                "direction": "inbound",
                "from_number": redacted_caller(self.ctx.from_number),
                "to_number": self.ctx.to_number,
                "metadata": self.ctx.metadata,
            }
        )
        self._build_asr(language=None)
        self._build_lid()
        await self.channel.send_event(
            "call_started",
            {
                "call_id": self.ctx.call_id,
                "state": self.state.value,
                "providers": {
                    "asr": self._asr_info.get("active"),
                    "tts": self.deps.tts.name,
                    "lid": settings.lid_provider,
                    "llm": self.deps.answer_engine.provider_name,
                },
                "supported_languages": settings.supported_language_list,
                "greeting_languages": settings.greeting_language_list,
            },
        )

    def _build_asr(self, language: str | None) -> None:
        if self.asr is not None and isinstance(self.asr, ClientASR):
            self._asr_info = {"active": "client", "requested": "client", "degraded": False}
            return
        factory = self.deps.asr_factory
        if factory is None:
            from ..voice.asr.registry import resolve_asr

            self.asr, self._asr_info = resolve_asr()
        else:
            self.asr = factory()
            self._asr_info = {"active": self.asr.name, "requested": self.asr.name,
                              "degraded": False}

    def _build_lid(self) -> None:
        factory = self.deps.lid_factory
        candidates = settings.supported_language_list
        if factory is not None:
            self.lid = factory(candidates)
        else:
            self.lid = LanguageIdentifier(candidates=candidates)

    async def _teardown(self) -> None:
        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        try:
            if self.asr is not None:
                await self.asr.stop()
        except Exception:  # pragma: no cover
            pass
        await self.channel.close()

    async def _fail(self, reason: str) -> None:
        self.state = CallState.FAILED
        self.ctx.escalation_reason = "technical_failure"
        await self._finish(status="failed", end_reason=f"error:{reason[:120]}")
        self._done.set()

    async def _finish(self, status: str, end_reason: str) -> None:
        if self.state == CallState.ENDED:
            return
        now = time.time()
        self.ctx.ended_at = now
        summary = await summarise(
            self.transcript,
            language=self.language,
            reason=self.ctx.escalation_reason,
            llm=self.deps.summariser_llm,
            metadata={"intents": self.intents_seen},
        )
        transcript_text = "\n".join(
            f"{t['role']}: {t['text']}" for t in self.transcript if t.get("text")
        )
        avg_latency = (
            sum(t.total_ms for t in self.timings) / len(self.timings) if self.timings else 0.0
        )
        ttfa = min((t.first_audio_ms for t in self.timings if t.first_audio_ms), default=0.0)
        primary_intent = summary.caller_intent
        self.deps.logger.end_call(
            {
                "call_id": self.ctx.call_id,
                "status": status,
                "end_reason": end_reason,
                "duration_seconds": self.ctx.duration_seconds(now),
                "summary": summary.text,
                "transcript": transcript_text,
                "escalated": self.escalated,
                "resolved": self.resolved and not self.escalated,
                "escalation_reason": self.ctx.escalation_reason,
                "detected_language": self.ctx.language,
                "language_confidence": self.ctx.language_confidence,
                "language_method": self.ctx.language_method,
                "language_attempts": self.ctx.language_attempts,
                "barge_in_count": self.ctx.barge_in_count,
                "time_to_first_audio_ms": ttfa,
                "avg_response_latency_ms": avg_latency,
                "primary_intent": primary_intent,
                "intents": self.intents_seen,
                "metadata": {
                    "asr": self._asr_info,
                    "tts": self.deps.tts.name,
                    "llm": self.deps.answer_engine.llm_info,
                    "lid_history": (self.lid.history if self.lid else []),
                    "unresolved": self.ctx.unresolved_questions,
                    "summary_detail": summary.to_dict(),
                },
            }
        )
        self.state = CallState.ENDED
        self.ctx.state = CallState.ENDED
        await self.channel.send_event(
            "call_ended",
            {
                "call_id": self.ctx.call_id,
                "status": status,
                "end_reason": end_reason,
                "duration_seconds": round(self.ctx.duration_seconds(now), 1),
                "escalated": self.escalated,
                "resolved": self.resolved,
                "language": self.ctx.language,
                "turns": len(self.transcript),
                "summary": summary.text,
            },
        )
        self._done.set()

    # ------------------------------------------------------------------ #
    # inbound media
    # ------------------------------------------------------------------ #
    async def _media_loop(self) -> None:
        try:
            async for event in self.channel.inbound():
                if self.state == CallState.ENDED:
                    break
                if event.type == "audio":
                    await self._on_audio(event.audio)
                elif event.type == "transcript":
                    self._last_activity = time.time()
                    await self._enqueue_utterance(
                        event.text,
                        {
                            "source": "client_asr",
                            "language": event.language,
                            "confidence": event.confidence,
                            "is_final": event.is_final,
                        },
                    )
                elif event.type == "dtmf":
                    await self._on_dtmf(event.digit)
                elif event.type == "control":
                    await self._on_control(event.payload)
                elif event.type == "hangup":
                    logger.info("caller hung up (%s)", self.ctx.call_id)
                    await self._finish(
                        status="completed" if self.resolved else "abandoned",
                        end_reason="caller_hangup",
                    )
                    break
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("media loop error: %s", exc)
            await self._fail(str(exc))

    async def _on_audio(self, data: bytes) -> None:
        if not data:
            return
        pcm = mulaw_to_pcm16(data)
        events = self.vad.process(pcm, now=time.time())
        if self.lid is not None:
            self.lid.add_audio(pcm, 8000)
        if self.asr is not None and not isinstance(self.asr, ClientASR):
            try:
                await self.asr.feed(data)
            except Exception as exc:
                logger.warning("asr feed failed: %s", exc)

        if events.get("speech_started"):
            self._last_activity = time.time()
            if self._speaking:
                await self._barge_in()
        if events.get("speech_ended"):
            self._last_activity = time.time()
            if isinstance(self.asr, WhisperChunkASR):
                try:
                    segment = await self.asr.transcribe_buffer()
                except Exception as exc:  # pragma: no cover
                    logger.warning("whisper transcription failed: %s", exc)
                    segment = None
                if segment and segment.text.strip():
                    await self._enqueue_utterance(
                        segment.text, {"source": "whisper", "language": segment.language,
                                       "confidence": segment.confidence}
                    )
            if self.lid is not None:
                self.lid.reset_audio()

    async def _on_control(self, payload: dict[str, Any]) -> None:
        name = payload.get("name") or payload.get("type")
        if name in ("speech_done", "playback_done"):
            self._client_speech_done.set()
        elif name == "hangup":
            await self._finish(status="completed", end_reason="client_hangup")
        elif name == "language_override":
            code = payload.get("language")
            if code in settings.supported_language_list:
                await self._adopt_language(code, method="explicit", confidence=1.0)
        elif name == "escalate":
            await self._escalate("caller_requested_human", payload.get("summary", ""))
        elif name in ("barge_in", "interrupt") and self._speaking:
            # Telephone callers barge in via VAD on the audio stream. A browser or
            # web-widget client does its own ASR and sends no audio, so it has to
            # say so explicitly -- otherwise it can never interrupt the assistant.
            await self._barge_in()

    async def _on_dtmf(self, digit: str) -> None:
        """Keypad is a last resort: only honoured in the language-selection flow
        when ALLOW_DTMF_FALLBACK is on and speech could not be understood."""
        self._last_activity = time.time()
        if not settings.allow_dtmf_fallback:
            await self.channel.send_event("dtmf_ignored", {"digit": digit})
            return
        if self.state not in LANGUAGE_STATES:
            return
        mapping = {"1": "en-IN", "2": "hi-IN", "3": "raj-IN", "4": "ta-IN", "5": "bn-IN"}
        code = mapping.get(digit)
        if code and code in settings.supported_language_list:
            await self._adopt_language(code, method="dtmf", confidence=1.0)
        else:
            await self.channel.send_event("dtmf_unmapped", {"digit": digit})

    # ------------------------------------------------------------------ #
    # ASR results
    # ------------------------------------------------------------------ #
    async def _asr_loop(self) -> None:
        if self.asr is None or isinstance(self.asr, ClientASR):
            return  # transcripts arrive as `transcript` events
        try:
            assert self.asr is not None
            async for segment in self.asr.segments():
                await self._on_asr_segment(segment)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("asr loop failed: %s", exc)

    async def _on_asr_segment(self, segment: TranscriptSegment) -> None:
        if segment.is_empty and not segment.utterance_end:
            return
        if not segment.is_final:
            if segment.text:
                self._last_activity = time.time()
                await self.channel.send_event(
                    "partial_transcript",
                    {"text": segment.text, "language": segment.language,
                     "provider": segment.provider},
                )
            return
        text = segment.text.strip()
        if not text:
            return
        self._last_activity = time.time()
        await self._enqueue_utterance(
            text,
            {
                "source": segment.provider,
                "language": segment.language,
                "confidence": segment.confidence,
                "asr_ms": max(0.0, (time.time() - segment.created_at) * 1000),
            },
        )

    async def _enqueue_utterance(self, text: str, meta: dict[str, Any]) -> None:
        text = (text or "").strip()
        if not text:
            return
        await self._utterance_queue.put((text, meta))

    async def _utterance_loop(self) -> None:
        while not self._done.is_set():
            text, meta = await self._utterance_queue.get()
            async with self._turn_lock:
                if self.state == CallState.ENDED:
                    continue
                try:
                    await self._handle_utterance(text, meta)
                except Exception as exc:
                    logger.exception("utterance handling failed: %s", exc)
                    await self._escalate("technical_failure", f"internal error: {exc}"[:200])

    # ------------------------------------------------------------------ #
    # speaking
    # ------------------------------------------------------------------ #
    async def _say(
        self,
        text: str,
        *,
        wait: bool = True,
        language: str | None = None,
        timeout: float = 25.0,
    ) -> None:
        """Queue text for speech. `wait=True` blocks until it has been played."""
        text = (text or "").strip()
        if not text:
            return
        item = _SpeechItem(text=text, language=language or self.language, wait=wait)
        self._speech_idle.clear()
        await self._speech_queue.put(item)
        if wait:
            try:
                await asyncio.wait_for(item.done.wait(), timeout)
            except TimeoutError:  # pragma: no cover
                logger.warning("speech timed out for: %s", text[:60])

    async def _speech_worker(self) -> None:
        tts = self.deps.tts
        while True:
            item = await self._speech_queue.get()
            if item is None:
                return
            started = time.perf_counter()
            self._interrupt.clear()
            self._speaking = True
            self.channel.begin_speaking()
            audio_frames = 0
            text_only = False
            try:
                if tts.text_only:
                    text_only = True
                    await self.channel.send_event(
                        "assistant_text",
                        {"text": item.text, "language": item.language, "seq": self.ctx.turn_index},
                    )
                    await self.channel.send_event(
                        "assistant_speech",
                        {"text": item.text, "language": item.language, "mode": "client_tts"},
                    )
                    await self._await_client_speech(item.text)
                else:
                    await self.channel.send_event(
                        "assistant_text",
                        {"text": item.text, "language": item.language, "seq": self.ctx.turn_index},
                    )
                    async for chunk in tts.synthesize_stream(
                        item.text,
                        item.language,
                        sample_rate=8000,
                        encoding="mulaw",
                        max_chunk_chars=settings.tts_max_chunk_chars,
                    ):
                        if self._interrupt.is_set():
                            break
                        if chunk.text_only or not chunk.audio:
                            text_only = True
                            await self.channel.send_event(
                                "assistant_speech",
                                {"text": chunk.text or item.text, "language": item.language,
                                 "mode": "client_tts"},
                            )
                            await self._await_client_speech(chunk.text or item.text)
                            continue
                        frames = _mulaw_frames(chunk.audio, chunk.sample_rate)
                        for frame in frames:
                            if self._interrupt.is_set():
                                break
                            await self.channel.send_audio(frame)
                            audio_frames += 1
                            if item.first_audio_ms == 0.0:
                                item.first_audio_ms = (time.perf_counter() - started) * 1000
            except asyncio.CancelledError:  # pragma: no cover
                raise
            except Exception as exc:
                logger.exception("synthesis failed: %s", exc)
                await self.channel.send_event("tts_error", {"error": str(exc)[:200]})
                text_only = True
                await self.channel.send_event(
                    "assistant_speech", {"text": item.text, "language": item.language,
                                         "mode": "client_tts"},
                )
            finally:
                self._speaking = False
                self.channel.end_speaking()
                item.done.set()
                if self._speech_queue.empty():
                    self._speech_idle.set()
                await self.channel.send_event(
                    "speech_finished",
                    {"text": item.text, "frames": audio_frames, "text_only": text_only,
                     "interrupted": self._interrupt.is_set(),
                     "duration_ms": round((time.perf_counter() - started) * 1000, 1)},
                )

    async def _await_client_speech(self, text: str) -> None:
        """Wait for the client to finish speaking (browser TTS path).

        Returns immediately in text mode: there is no audio, so nothing to wait
        for. Otherwise the budget tracks the spoken length but is capped, so a
        client that never sends the `speech_done` control cannot freeze the call.
        """
        if self.text_mode:
            return
        words = max(1, len((text or "").split()))
        budget = min(settings.client_speech_max_wait, words / WORDS_PER_SECOND + 0.9)
        self._client_speech_done.clear()
        try:
            await asyncio.wait_for(self._client_speech_done.wait(), budget)
        except TimeoutError:
            logger.info("client speech budget elapsed after %.1fs", budget)
        finally:
            self._client_speech_done.set()

    async def _await_speech_complete(self, timeout: float = 30.0) -> None:
        """Block until the speech queue has drained and nothing is playing."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self._speech_queue.empty() and not self._speaking:
                await asyncio.sleep(0.02)
                if self._speech_queue.empty() and not self._speaking:
                    return
            await asyncio.sleep(0.05)
        logger.warning("speech did not complete within %.0fs", timeout)

    async def _barge_in(self) -> None:
        """Caller started talking over the assistant — stop immediately."""
        if not self._speaking:
            return
        self.ctx.barge_in_count += 1
        self._interrupt.set()
        # drop everything still queued
        dropped = 0
        while not self._speech_queue.empty():
            try:
                item = self._speech_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                await self._speech_queue.put(None)
                break
            item.done.set()
            dropped += 1
        await self.channel.interrupt()
        if self.transcript and self.transcript[-1]["role"] == "assistant":
            self.transcript[-1]["interrupted"] = True
        logger.info("barge-in on call %s (dropped %d queued items)", self.ctx.call_id, dropped)
        await self.channel.send_event("barge_in", {"dropped": dropped})

    # ------------------------------------------------------------------ #
    # opening sequence
    # ------------------------------------------------------------------ #
    async def _greeting_sequence(self) -> None:
        self._set_state(CallState.GREETING)
        opening: list[str] = [script_line("en-IN", "greeting")]
        # TRAI / DPDP: identify ourselves first, then disclose the recording and
        # the opt-out before any caller speech is captured.
        if settings.recording_consent_announce:
            opening.append(script_line("en-IN", "recording_consent"))
        await self._say(" ".join(opening), language="en-IN")

        # The language prompt is spoken in every greeting language at once, so
        # the caller hears their own language and answers in it. No keypad.
        #
        # Say it ONCE. Each `language_prompt` script line is already trilingual
        # ("Please tell me your preferred language. कृपया अपनी भाषा बताइए। आप
        # अपणी भाषा बतावो।"), so looping over the greeting languages and joining
        # the lines read the whole trilingual prompt three times over -- nine
        # seconds of talking before the caller could answer a single question.
        self._set_state(CallState.LANGUAGE_PROMPT)
        prompt_language = (settings.greeting_language_list or ["en-IN"])[0]
        prompt_line = script_line(prompt_language, "language_prompt")
        if prompt_line:
            await self._say(prompt_line, language=prompt_language)
        await self._announce_asr_hints()
        self._last_activity = time.time()

    async def _announce_asr_hints(self) -> None:
        """Tell the client which locales to try while we are identifying language."""
        await self.channel.send_event(
            "asr_config",
            {
                "mode": "language_identification",
                # Some codes fall back to a shared ASR locale (raj-IN -> hi-IN),
                # so dedupe: the browser was being told ["en-IN","hi-IN","hi-IN"].
                "candidates": list(dict.fromkeys(
                    asr_locale(c) for c in settings.greeting_language_list
                )),
                "provider": self._asr_info.get("active"),
            },
        )

    # ------------------------------------------------------------------ #
    # routing
    # ------------------------------------------------------------------ #
    async def _handle_utterance(self, text: str, meta: dict[str, Any]) -> None:
        self._last_turn_started = time.perf_counter()
        self.ctx.turn_index += 1
        seq = self.ctx.turn_index
        self.ctx.silence_count = 0
        await self._log_caller_turn(text, seq, meta)

        if self.state in LANGUAGE_STATES:
            await self._handle_language_turn(text, meta)
            return
        if self.state in CONVERSATION_STATES:
            await self._handle_conversation_turn(text, seq)
            return
        logger.info("utterance ignored in state %s", self.state.value)

    async def _log_caller_turn(self, text: str, seq: int, meta: dict[str, Any]) -> None:
        clean_text, pii = _redact(text)
        self.transcript.append(
            {"role": "caller", "text": clean_text, "seq": seq, "language": self.language,
             "pii": pii, "meta": {k: v for k, v in meta.items() if k != "source"}}
        )
        self.deps.logger.log_turn(
            {
                "call_id": self.ctx.call_id,
                "seq": seq,
                "role": "caller",
                "text": clean_text,
                "language": meta.get("language") or self.language,
                "is_final": True,
                "confidence": float(meta.get("confidence") or 0.0),
                "asr_ms": float(meta.get("asr_ms") or 0.0),
                "metadata": {
                    "source": meta.get("source"),
                    "pii_redacted": list(pii.keys()),
                    # Lets analytics tell a real question from the caller's
                    # answer to "which language?" (that answer is often a bare
                    # language name, which the intent classifier legitimately
                    # reads as the B.A. Hindi / B.A. English specialisation).
                    "stage": self.state.value,
                },
            }
        )
        await self.channel.send_event(
            "caller_transcript", {"text": clean_text, "seq": seq, "language": self.language}
        )

    # ------------------------------------------------------------------ #
    # language selection
    # ------------------------------------------------------------------ #
    async def _handle_language_turn(self, text: str, meta: dict[str, Any]) -> None:
        assert self.lid is not None
        if self.state == CallState.LANGUAGE_CONFIRM and self._pending_language:
            confirmation = self.lid.confirm(text)
            if confirmation is not None and confirmation.method == "confirmation":
                if confirmation.detail == "accepted":
                    await self._adopt_language(
                        self._pending_language, method="confirmed",
                        confidence=max(confirmation.confidence, self.ctx.language_confidence),
                    )
                    return
                self._pending_language = None
                await self._reprompt_language(understood=False)
                return
            if confirmation is not None:
                # caller named a different language instead of confirming
                await self._evaluate_lid(confirmation, text)
                return
            await self._reprompt_language(understood=False)
            return

        result = await self.lid.identify(text)
        await self.channel.send_event("language_detected", result.to_dict() | {"text": text})
        await self._evaluate_lid(result, text)

    async def _evaluate_lid(self, result: Any, text: str) -> None:
        code = result.language
        supported = settings.supported_language_list
        if not code:
            await self._reprompt_language(understood=False)
            return
        if code not in supported:
            await self._unsupported_language(code)
            return
        if result.confidence >= HIGH_LID_CONFIDENCE or result.method in (
            "explicit_name", "confirmed", "dtmf"
        ):
            await self._adopt_language(code, method=result.method, confidence=result.confidence)
            return
        if result.confidence >= settings.lid_confidence_threshold:
            self._pending_language = code
            self._set_state(CallState.LANGUAGE_CONFIRM)
            question = _confirmation_question(code, self.language)
            await self._say(question, language=code)
            return
        await self._reprompt_language(understood=False, heard=text)

    async def _adopt_language(self, code: str, *, method: str, confidence: float) -> None:
        language = get_language(code)
        self.ctx.language = code
        self.ctx.language_confidence = float(confidence)
        self.ctx.language_method = method
        self.session_language = code
        self._pending_language = None
        self._set_state(CallState.MENU)
        logger.info(
            "call %s language=%s confidence=%.2f method=%s attempts=%d",
            self.ctx.call_id, code, confidence, method, self.ctx.language_attempts,
        )
        self.deps.logger.update_call(
            {
                "call_id": self.ctx.call_id,
                "detected_language": code,
                "language_confidence": float(confidence),
                "language_method": method,
                "language_attempts": self.ctx.language_attempts,
                "fallback_language_used": bool(language.fallback_code),
            }
        )
        await self._switch_asr_language(code)
        if self.lid is not None:
            self.lid.reset_audio()
        confirm = script_line(code, "language_confirm", language=language.native_name)
        menu = script_line(code, "menu")
        await self.channel.send_event(
            "language_switched",
            {"language": code, "name": language.english_name, "native": language.native_name,
             "confidence": confidence, "method": method},
        )
        await self._say(f"{confirm} {menu}", language=code)
        self._set_state(CallState.CONVERSATION)
        self._last_activity = time.time()

    async def _switch_asr_language(self, code: str) -> None:
        """Restart the recognizer pinned to the caller's language."""
        locale = asr_locale(code)
        await self.channel.send_event(
            "asr_config", {"mode": "conversation", "language": locale, "candidates": [locale]}
        )
        if self.asr is None or isinstance(self.asr, ClientASR):
            return
        if isinstance(self.asr, WhisperChunkASR):
            self.asr.language = locale
            return
        try:
            await self.asr.stop()
        except Exception:  # pragma: no cover
            pass
        old_task = next(
            (t for t in self._tasks if t.get_name().endswith(":asr") and not t.done()), None
        )
        if old_task:
            old_task.cancel()
        self._build_asr(locale)
        if self.asr is not None:
            try:
                await self.asr.start(language=locale, sample_rate=8000)
            except Exception as exc:  # pragma: no cover
                logger.warning("could not restart ASR for %s: %s", locale, exc)
                from ..voice.asr.client import ClientASR as _ClientASR

                self.asr = _ClientASR()
                self._asr_info = {"active": "client", "degraded": True,
                                  "reason": f"asr restart failed: {exc}"}
                return
            task = asyncio.create_task(self._asr_loop(), name=f"{self.ctx.call_id}:asr")
            self._tasks.append(task)

    async def _reprompt_language(self, *, understood: bool, heard: str = "") -> None:
        self.ctx.language_attempts += 1
        if self.ctx.language_attempts > 2:
            if settings.allow_dtmf_fallback:
                self._set_state(CallState.LANGUAGE_REPROMPT)
                await self._say(script_line("en-IN", "dtmf_fallback"), language="en-IN")
                await self._say(
                    " ".join(dict.fromkeys([
                        script_line("hi-IN", "language_unclear"),
                        script_line("mr-IN", "language_unclear"),
                    ])),
                    language="hi-IN",
                )
                return
            # Give up on detection, continue in English and offer a human.
            await self._adopt_language("en-IN", method="default", confidence=0.2)
            await self._say(script_line("en-IN", "escalate_offer"), language="en-IN")
            self._set_state(CallState.CONVERSATION)
            return
        self._set_state(CallState.LANGUAGE_REPROMPT)
        lines = [
            script_line("en-IN", "language_unclear"),
            script_line("hi-IN", "language_unclear"),
            script_line("mr-IN", "language_unclear"),
        ]
        if settings.allow_dtmf_fallback and self.ctx.language_attempts == 2:
            lines.append(script_line("en-IN", "dtmf_fallback"))
        await self._say(" ".join(dict.fromkeys(lines)), language="en-IN")
        await self._announce_asr_hints()
        self._last_activity = time.time()

    async def _unsupported_language(self, code: str) -> None:
        self._set_state(CallState.LANGUAGE_UNSUPPORTED)
        language = get_language(code)
        line = script_line("en-IN", "language_unsupported", language=language.english_name)
        await self.channel.send_event(
            "unsupported_language", {"language": code, "name": language.english_name}
        )
        await self._say(line, language="en-IN")
        await self._say(script_line("hi-IN", "language_unsupported",
                                    language=language.native_name), language="hi-IN")
        await self._say(script_line("mr-IN", "language_unsupported",
                                    language=language.native_name), language="mr-IN")
        # Fall back to Hindi: after Marathi and English it is the most widely
        # understood language on a Dhule helpline, and it has full ASR/TTS
        # support, unlike most of the codes that can reach this branch.
        await self._adopt_language("hi-IN", method="fallback", confidence=0.4)

    # ------------------------------------------------------------------ #
    # conversation
    # ------------------------------------------------------------------ #
    async def _handle_conversation_turn(self, text: str, seq: int) -> None:
        if self._awaiting_destination:
            await self._capture_destination(text)
            return

        _control, control_intent = is_control_utterance(text)
        intent = detect_intent(text)
        self.intents_seen.append(intent.intent)

        named_only = named_language_only(text, tuple(settings.supported_language_list))
        if named_only:
            # The caller named a language and said nothing else. That is a
            # choice, not a question: answering it as one gave "मराठी" the
            # academic calendar, because that record carries Marathi aliases.
            # Explicit beats inferred, so unlike a script-based detection this
            # does not wait for a second turn to agree.
            await self._adopt_language(
                named_only, method="explicit_name", confidence=0.97
            )
            return

        await self._maybe_switch_language(text)

        if control_intent == "human_request" or intent.intent == "human_request":
            await self._escalate(
                "caller_requested_human",
                f"Caller asked for a person. Language {self.language}. Last topic: "
                f"{intent.intent}.",
            )
            return

        if self.state == CallState.FOLLOWUP_OFFER:
            if control_intent in {"affirmation"} or intent.intent == "affirmation":
                await self._start_followup_delivery()
                return
            if control_intent in {"negation"} or intent.intent == "negation":
                self._pending_followup = None
                self._set_state(CallState.CONVERSATION)
                await self._say(script_line(self.language, "menu"), language=self.language)
                return

        if intent.intent == "repeat":
            if self.ctx.last_assistant_text:
                await self._say(self.ctx.last_assistant_text, language=self.language)
            else:
                await self._say(script_line(self.language, "menu"), language=self.language)
            return

        if intent.intent in {"greeting", "smalltalk"} and len(text.split()) <= 6:
            await self._say(script_line(self.language, "menu"), language=self.language)
            return

        if intent.intent == "negation" and len(text.split()) <= 3:
            await self._say(script_line(self.language, "menu"), language=self.language)
            return

        if intent.intent == "affirmation" and len(text.split()) <= 3:
            await self._say(script_line(self.language, "menu"), language=self.language)
            return

        await self._answer_question(text, intent, seq)

    async def _maybe_switch_language(self, text: str) -> None:
        """Offer to switch if the caller clearly moved to another language."""
        if self.lid is None or not text.strip():
            return
        result = await self.lid.identify(text)
        if (
            result.language
            and result.language != self.language
            and result.language in settings.supported_language_list
            and result.confidence >= MID_CALL_SWITCH_CONFIDENCE
        ):
            self._language_switch_streak += 1
        else:
            self._language_switch_streak = 0
        if self._language_switch_streak >= 2:
            self._language_switch_streak = 0
            await self._adopt_language(
                result.language, method="mid_call_switch", confidence=result.confidence
            )

    def _programme_context_for(self, seq: int) -> tuple[str, list[str], list[str]]:
        """The programme still under discussion, if any."""
        if not self._programme_context_title:
            return "", [], []
        if seq - self._programme_context_turn > PROGRAMME_CONTEXT_TURNS:
            self._forget_programme_context()
            return "", [], []
        return (
            self._programme_context_title,
            list(self._programme_context_tokens),
            list(self._programme_context_specs),
        )

    def _forget_programme_context(self) -> None:
        self._programme_context_title = ""
        self._programme_context_tokens = []
        self._programme_context_specs = []

    def _remember_programme(self, answer: AssistantAnswer, seq: int) -> None:
        """Note which programme this turn was about, so a follow-up can inherit it.

        Taken from the record that was actually answered from, not from the
        caller's words: "what is the eligibility for B.Tech Computer
        Engineering?" yields the token BTECH and nothing else, which cannot tell
        Computer Engineering from the five other B.Tech branches on this campus.
        The record title can.
        """
        retrieval = answer.retrieval
        if retrieval is None:
            return
        course = next(
            (item for item in retrieval.items if getattr(item, "category", "") == "course"),
            None,
        )
        if course is None:
            # Nothing programme-shaped was discussed — a question about the
            # calendar or the address must not wipe the programme under way.
            return
        title = (course.title or "").strip()
        if not title:
            return
        if "overview" in title.lower() or "at a glance" in title.lower():
            # "What about the pharmacy one?" is answered from a school overview,
            # which lists programmes rather than being one. Anchoring on it sent
            # the next "and its eligibility?" to whichever programme happened to
            # rank first inside that school. Drop the anchor instead, so the
            # follow-up asks which programme the caller means.
            self._forget_programme_context()
            return
        entities = extract_entities(course.title)
        self._programme_context_title = course.title
        self._programme_context_tokens = entities["course_tokens"]
        self._programme_context_specs = entities["specialisations"]
        self._programme_context_turn = seq

    async def _answer_question(self, text: str, intent: IntentResult, seq: int) -> None:
        engine = self.deps.answer_engine
        started = time.perf_counter()
        timings = TurnTimings()
        sentence_index = 0
        first_audio_recorded = False

        async def on_sentence(sentence: str) -> None:
            nonlocal sentence_index, first_audio_recorded
            sentence_index += 1
            await self._say(sentence, wait=False, language=self.language)
            if not first_audio_recorded:
                timings.first_audio_ms = (time.perf_counter() - started) * 1000
                first_audio_recorded = True

        context_title, context_tokens, context_specs = self._programme_context_for(seq)

        async with SessionLocal() as session:
            answer: AssistantAnswer = await engine.answer(
                AnswerRequest(
                    call_id=self.ctx.call_id,
                    question=text,
                    language=self.language,
                    history=list(self.history),
                    stage=self.state.value,
                    turn_index=seq,
                    caller_ref=redacted_caller(self.ctx.from_number),
                    context_record_title=context_title,
                    context_course_tokens=context_tokens,
                    context_specialisations=context_specs,
                ),
                session,
                on_sentence=on_sentence if settings.llm_streaming else None,
            )

        self._remember_programme(answer, seq)

        timings.retrieval_ms = answer.retrieval_ms
        timings.llm_ms = answer.llm_ms
        timings.total_ms = (time.perf_counter() - started) * 1000

        text_to_speak = answer.text or script_line(self.language, "not_found")
        if sentence_index == 0:
            await self._say(text_to_speak, wait=False, language=self.language)
            timings.first_audio_ms = (time.perf_counter() - started) * 1000
        await self._await_speech_complete()
        timings.total_ms = (time.perf_counter() - started) * 1000

        self.ctx.last_assistant_text = text_to_speak
        self.ctx.last_answer_confidence = answer.confidence
        self.timings.append(timings)
        self._answer_turn_seq += 1
        await self._log_assistant_turn(text_to_speak, answer, timings)

        self.history.append(Message(role="user", content=text))
        self.history.append(Message(role="assistant", content=text_to_speak))
        self.history = self.history[-10:]

        if answer.followup and (answer.followup.get("items") or answer.followup.get("title")):
            self._pending_followup = answer.followup
            await self._offer_followup(answer.followup)
        elif answer.needs_escalation:
            await self._escalate(
                answer.escalation_reason or "kb_no_answer",
                answer.escalation_summary or "",
                answer=answer,
            )
            return
        else:
            self.resolved = True
            self.ctx.consecutive_low_confidence = 0
            if answer.confidence < settings.answer_confidence_threshold:
                self.ctx.consecutive_low_confidence += 1
            else:
                self.ctx.consecutive_low_confidence = 0
            if self.ctx.consecutive_low_confidence >= 2:
                await self._escalate("low_confidence", "Two consecutive low-confidence answers.")
                return
            self._set_state(CallState.CONVERSATION)

        self._last_activity = time.time()

    async def _log_assistant_turn(
        self, text: str, answer: AssistantAnswer, timings: TurnTimings
    ) -> None:
        clean_text, _pii = _redact(text)
        self.transcript.append(
            {
                "role": "assistant",
                "text": clean_text,
                "language": self.language,
                "grounded": answer.grounded,
                "confidence": answer.confidence,
                "metadata": {"provider": answer.provider, "intent": answer.intent},
            }
        )
        self.deps.logger.log_turn(
            {
                "call_id": self.ctx.call_id,
                "seq": self.ctx.turn_index,
                "role": "assistant",
                "text": clean_text,
                "language": self.language,
                "retrieval_ms": timings.retrieval_ms,
                "llm_ms": timings.llm_ms,
                "tts_ms": timings.tts_ms,
                "total_ms": timings.total_ms,
                "confidence": answer.confidence,
                "grounded": answer.grounded,
                "citations": answer.citations,
                "tool_calls": answer.tool_calls,
                "metadata": {
                    "provider": answer.provider,
                    "fallback_used": answer.fallback_used,
                    "intent": answer.intent,
                    "warnings": answer.warnings,
                    "needs_escalation": answer.needs_escalation,
                    "escalation_reason": answer.escalation_reason,
                    "first_audio_ms": timings.first_audio_ms,
                },
            }
        )
        self.deps.logger.update_call(
            {
                "call_id": self.ctx.call_id,
                "primary_intent": answer.intent,
                "intents_json": self.intents_seen[-20:],
            }
        )
        if answer.needs_escalation and answer.escalation_reason in KNOWLEDGE_GAP_REASONS:
            question = self.transcript[-2]["text"] if len(self.transcript) >= 2 else text
            self.ctx.unresolved_questions.append(question)
            self.deps.logger.log_unanswered(
                {
                    "call_id": self.ctx.call_id,
                    "question": question,
                    "language": self.language,
                    "intent": answer.intent,
                    "best_score": (answer.retrieval.best_score if answer.retrieval else 0.0),
                }
            )

    # ------------------------------------------------------------------ #
    # follow-up details (SMS / WhatsApp / email)
    # ------------------------------------------------------------------ #
    async def _offer_followup(self, followup: dict[str, Any]) -> None:
        self._set_state(CallState.FOLLOWUP_OFFER)
        await self._say(script_line(self.language, "followup_offer"), language=self.language)
        await self.channel.send_event("followup_offer", followup)
        self._last_activity = time.time()

    async def _start_followup_delivery(self) -> None:
        followup = self._pending_followup or {}
        channel_name = str(followup.get("channel") or "sms").lower()
        destination = ""
        if (
            channel_name in {"sms", "whatsapp"}
            and self.ctx.from_number
            and (settings.followup_sms_enabled or settings.followup_whatsapp_enabled)
        ):
            destination = self.ctx.from_number
        if not destination:
            self._awaiting_destination = True
            self._set_state(CallState.FOLLOWUP_DESTINATION)
            ask = _ask_destination(self.language, channel_name)
            await self._say(ask, language=self.language)
            self._last_activity = time.time()
            return
        await self._deliver_followup(destination, channel_name)

    async def _capture_destination(self, text: str) -> None:
        followup = self._pending_followup or {}
        channel_name = str(followup.get("channel") or "sms").lower()
        if channel_name == "email" or looks_like_email(text):
            destination = extract_email(text)
            channel_name = "email" if destination else channel_name
        else:
            destination = extract_phone(text)
        if not destination:
            await self._say(_ask_destination(self.language, channel_name), language=self.language)
            self._last_activity = time.time()
            return
        self._awaiting_destination = False
        await self._deliver_followup(destination, channel_name)

    async def _deliver_followup(self, destination: str, channel_name: str) -> None:
        followup = self._pending_followup or {}
        message = FollowUpMessage(
            channel=channel_name,
            destination=destination,
            title=str(followup.get("title") or "NMIMS Global University, Dhule details"),
            items=[str(i) for i in (followup.get("items") or [])],
            language=self.language,
            call_id=self.ctx.call_id,
        )
        result = await deliver(self.channel, message)
        self.deps.logger.log_followup(
            {
                "call_id": self.ctx.call_id,
                "channel": channel_name,
                "destination_ref": result.get("destination_ref", ""),
                "body": result.get("body", ""),
                "status": result.get("status", "unknown"),
                "provider_message_id": result.get("provider_message_id"),
                "error": result.get("reason") if result.get("status") == "failed" else None,
            }
        )
        if result.get("status") in {"sent", "skipped"}:
            await self._say(_followup_sent(self.language, channel_name, result.get("status")),
                            language=self.language)
        else:
            await self._say(_followup_failed(self.language), language=self.language)
        self._pending_followup = None
        self._set_state(CallState.CONVERSATION)
        await self._say(script_line(self.language, "menu"), language=self.language)
        self._last_activity = time.time()

    # ------------------------------------------------------------------ #
    # escalation
    # ------------------------------------------------------------------ #
    async def _escalate(
        self, reason: str, summary: str = "", *, answer: AssistantAnswer | None = None
    ) -> None:
        if self.escalated or self.state in {CallState.TRANSFERRED, CallState.ENDED}:
            return
        self.escalated = True
        self.ctx.escalation_reason = reason
        self._set_state(CallState.ESCALATING)

        brief = await summarise(
            self.transcript,
            language=self.language,
            reason=reason,
            llm=self.deps.summariser_llm,
        )
        if summary:
            brief.text = f"{brief.text} Note from the assistant: {summary}"
            brief.whisper = brief.text[:420]
        self.ctx.escalation_summary = brief.text

        plan = plan_escalation(
            reason=reason,
            summary=brief.text,
            language=self.language,
            extra_context={
                "call_id": self.ctx.call_id,
                "unresolved": self.ctx.unresolved_questions[-3:],
                "intents": self.intents_seen[-8:],
                "confidence": (answer.confidence if answer else None),
                "citations": (answer.citations[:3] if answer else []),
            },
        )

        self.deps.logger.log_escalation(
            {
                "call_id": self.ctx.call_id,
                "reason": reason,
                "target": plan.target or None,
                "target_type": plan.target_type,
                "whisper_summary": whisper_twiml_text(brief),
                "context": plan.context,
            }
        )

        if plan.target_type == "none":
            line = NO_AGENT_LINE.get(self.language, NO_AGENT_LINE["en-IN"])
            await self._say(line, language=self.language)
            await self._finish(status="escalated", end_reason=f"no_agent:{reason}")
            await self.channel.hangup(reason="no_agent_available")
            return

        register_pending_transfer(self.ctx.call_id, plan)
        await self._say(script_line(self.language, "escalating"), language=self.language)
        if plan.target_type == "queue":
            await self.channel.play_url(plan.hold_music, loop=True)
        await self.channel.transfer(
            plan.target,
            target_type=plan.target_type,
            whisper_text=whisper_twiml_text(brief),
            context=plan.context,
        )
        self._set_state(CallState.TRANSFERRED)
        await self._finish(status="escalated", end_reason=f"transferred:{reason}")

    # ------------------------------------------------------------------ #
    # watchdogs
    # ------------------------------------------------------------------ #
    async def _silence_watchdog(self) -> None:
        while not self._done.is_set():
            await asyncio.sleep(2.0)
            if self._speaking or self.state in {CallState.ENDED, CallState.TRANSFERRED,
                                                CallState.ESCALATING}:
                self._last_activity = time.time()
                continue
            idle = time.time() - self._last_activity
            if idle < SILENCE_TIMEOUT_SECONDS:
                continue
            self._last_activity = time.time()
            self.ctx.silence_count += 1
            logger.info("silence #%d in state %s", self.ctx.silence_count, self.state.value)
            await self.channel.send_event("silence", {"count": self.ctx.silence_count})

            if self.state in LANGUAGE_STATES:
                if self.ctx.silence_count >= settings.silence_max_reprompts + 1:
                    await self._adopt_language("en-IN", method="default", confidence=0.15)
                    await self._say(script_line("en-IN", "menu"), language="en-IN")
                    self._set_state(CallState.CONVERSATION)
                else:
                    await self._reprompt_language(understood=False)
                continue

            if self.ctx.silence_count == 1:
                await self._say(script_line(self.language, "no_input_1"), language=self.language)
            elif self.ctx.silence_count == 2:
                await self._say(script_line(self.language, "no_input_2"), language=self.language)
            else:
                if self.ctx.turn_index == 0:
                    await self._escalate("silence_timeout", "Caller never spoke.")
                else:
                    await self._say(script_line(self.language, "farewell"), language=self.language)
                    await self._finish(status="completed", end_reason="silence_timeout")
                    await self.channel.hangup(reason="silence_timeout")
                break

    async def _duration_watchdog(self) -> None:
        limit = settings.max_call_minutes * 60
        while not self._done.is_set():
            await asyncio.sleep(10.0)
            if self.ctx.duration_seconds(time.time()) < limit:
                continue
            logger.info("call %s reached max duration", self.ctx.call_id)
            if self.state in CONVERSATION_STATES:
                await self._say(script_line(self.language, "farewell"), language=self.language)
                await self._finish(status="completed", end_reason="max_duration")
                await self.channel.hangup(reason="max_duration")
            else:
                await self._escalate("max_duration", "Call reached the maximum duration.")
            break

    # ------------------------------------------------------------------ #
    async def hangup(self, reason: str = "completed") -> None:
        if self.state in {CallState.CLOSING, CallState.ENDED}:
            return
        self._set_state(CallState.CLOSING)
        await self._say(script_line(self.language, "farewell"), language=self.language)
        await self._finish(status="completed", end_reason=reason)
        await self.channel.hangup(reason=reason)

    def _set_state(self, state: CallState) -> None:
        if state == self.state:
            return
        previous = self.state
        self.state = state
        self.ctx.state = state
        logger.info("call %s state %s -> %s", self.ctx.call_id, previous.value, state.value)
        asyncio.create_task(
            self.channel.send_event("state", {"from": previous.value, "to": state.value})
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _mulaw_frames(mulaw: bytes, sample_rate: int) -> list[bytes]:
    frame_bytes = int(sample_rate * FRAME_MS / 1000)
    if frame_bytes <= 0:
        return [mulaw]
    return [mulaw[i : i + frame_bytes] for i in range(0, len(mulaw), frame_bytes)] or [mulaw]


def _redact(text: str) -> tuple[str, dict[str, list[str]]]:
    from ..ai.guardrails import redact_pii

    return redact_pii(text)


def _confirmation_question(code: str, current_language: str) -> str:
    language = get_language(code)
    if code == "en-IN":
        return "Should I continue in English?"
    if code == "hi-IN":
        return "क्या मैं हिंदी में बात करूँ?"
    if code == "raj-IN":
        return "म्हूँ राजस्थानी में बात करूँ?"
    return f"Should I continue in {language.english_name}?"


def _ask_destination(language: str, channel_name: str) -> str:
    base = language.split("-")[0]
    if channel_name == "email":
        if base == "hi":
            return "कृपया अपना ईमेल पता बताइए।"
        if language == "raj-IN":
            return "आप रो ईमेल पतो बतावो जी।"
        return "Please tell me your email address."
    if base == "hi":
        return "कृपया अपना मोबाइल नंबर बताइए।"
    if language == "raj-IN":
        return "आप रो मोबाइल नंबर बतावो जी।"
    return "Please tell me the mobile number I should send it to."


def _followup_sent(language: str, channel_name: str, status: str) -> str:
    base = language.split("-")[0]
    if status == "skipped":
        english = "I have noted that down and our team will send it to you shortly."
        hindi = "मैंने नोट कर लिया है, हमारी टीम आपको जल्द ही भेज देगी।"
        raj = "म्हूँ नोट कर लीधो, म्हारी टीम थानै जल्द ही भेज देसी।"
    else:
        noun = "email" if channel_name == "email" else "message"
        english = f"Done, the {noun} is on its way."
        hindi = f"हो गया, {noun if channel_name == 'email' else 'संदेश'} भेज दिया गया है।"
        raj = f"हो ग्यो, {noun if channel_name == 'email' else 'संदेशो'} भेज दीधो।"
    if base == "hi":
        return hindi
    if language == "raj-IN":
        return raj
    return english


def _followup_failed(language: str) -> str:
    base = language.split("-")[0]
    if base == "hi":
        return "माफ़ कीजिए, संदेश नहीं भेजा जा सका। हमारी टीम आपको भेज देगी।"
    if language == "raj-IN":
        return "माफ करना, संदेश कोनी भेज सके। म्हारी टीम भेज देसी।"
    return "Sorry, I could not send that. Our team will send it to you instead."


def build_history_from_transcript(transcript: list[dict[str, Any]]) -> list[Message]:
    return [
        Message(role="user" if t["role"] == "caller" else "assistant", content=t["text"])
        for t in transcript[-8:]
        if t.get("text")
    ]


def language_display(code: str) -> str:
    return display_name(code)
