"""The answering engine: retrieval → grounding → generation → guardrails.

Flow for one caller turn:

1. intent + entity detection (app.ai.intents)
2. query augmentation (native-script course names → Latin codes, cycle filter)
3. hybrid retrieval (BM25 + dense, RRF fused, reranked)
4. topic guardrail: off-topic → short polite decline, sensitive → escalate
5. generation: Claude/OpenAI with tool calling, streamed sentence by sentence to
   TTS — or the template composer when no LLM key is configured
6. post-checks: voice scrub + numeric grounding check. An answer containing a
   number that exists nowhere in the retrieved context is dropped and the call
   escalates instead.

Any failure in step 5 falls back to step 5-alt (templates) so the caller never
hears silence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import ist_today, settings
from ..i18n.languages import get_language
from ..kb.retriever import HybridRetriever, RetrievalResult
from ..models import KBRecord
from . import guardrails, prompts
from .intents import IntentResult, detect_intent
from .llm.base import Message, ToolCall, ToolSpec
from .llm.registry import resolve_llm
from .templates import compose as template_compose

logger = logging.getLogger("nims.rag")

SentenceCallback = Callable[[str], Awaitable[None]]

#: native-script course names -> Latin tokens the index understands
CROSS_SCRIPT_COURSES: dict[str, str] = {
    "बीटेक": "B.Tech", "बी.टेक": "B.Tech", "बी टेक": "B.Tech", "बीटेक्": "B.Tech",
    "एमटेक": "M.Tech", "एमबीए": "MBA", "एमबीबीएस": "MBBS", "बीडीएस": "BDS",
    "बीफार्म": "B.Pharm", "फार्मसी": "Pharmacy", "नर्सिंग": "Nursing",
    "बीएससी": "B.Sc", "एमएससी": "M.Sc", "बीए": "B.A", "एमए": "M.A",
    "बीकॉम": "B.Com", "एमकॉम": "M.Com", "बीसीए": "BCA", "एमसीए": "MCA",
    "कंप्यूटर": "Computer Science CSE", "सिविल": "Civil", "मैकेनिकल": "Mechanical",
    "इलेक्ट्रिकल": "Electrical", "इलेक्ट्रॉनिक्स": "Electronics ECE",
    "कानून": "Law LLB", "होटल": "Hotel Management", "डिजाइन": "Design",
    "आर्किटेक्चर": "Architecture", "फिजियोथेरेपी": "Physiotherapy BPT",
    "कृषि": "Agriculture", "पीएचडी": "PhD", "पी एच डी": "PhD",
    "बीएड": "B.Ed", "बी.एड": "B.Ed", "पैरामेडिकल": "Paramedical",
    "रेडियोलॉजी": "Radiology", "लैब टेक्निशियन": "Medical Laboratory Technology",
    "पत्रकारिता": "Journalism Mass Communication", "अविएशन": "Aviation",
    "फैशन": "Fashion Design", "साइबर": "Cyber Security", "डेटा साइंस": "Data Science",
    "आर्टिफिशियल इंटेलिजेंस": "Artificial Intelligence AI",
    "एमबीबीएस डॉक्टर": "MBBS",
    # --- Marathi: the same Devanagari script, different words --------------- #
    # Dhule callers ask in Marathi, and these are the terms that appear in the
    # knowledge base under their English headings.
    "अभियांत्रिकी": "Engineering", "संगणक अभियांत्रिकी": "Computer Science CSE",
    "यांत्रिकी अभियांत्रिकी": "Mechanical", "स्थपती": "Architecture",
    "औषधनिर्माणशास्त्र": "Pharmacy", "औषधनिर्माण": "Pharmacy",
    "परिचारिका": "Nursing", "वाणिज्य": "Commerce B.Com", "व्यापार": "Commerce B.Com",
    "व्यवस्थापन": "Management BBA MBA", "प्रशासन": "Management",
    "विधी": "Law LLB", "कायदा": "Law LLB", "वकील": "Law LLB",
    "संगणक अनुप्रयोग": "Computer Applications BCA MCA",
    "डॉक्टरेट": "PhD", "संशोधन": "Research PhD",
    "शुल्क": "Fees", "फी": "Fees", "खर्च": "Fees",
    "प्रवेश": "Admission", "प्रवेश प्रक्रिया": "Admission Process",
    "अभ्यासक्रम": "Courses Programmes", "पदवी": "Degree", "पदव्युत्तर": "Postgraduate",
    "पात्रता": "Eligibility", "गुण": "Marks Percentage",
    "वसतिगृह": "Hostel", "राहण्याची सोय": "Hostel Accommodation",
    "शिष्यवृत्ती": "Scholarship", "कागदपत्रे": "Documents",
    "प्रमाणपत्र": "Certificate", "मुदत": "Deadline Dates",
    "शेवटची तारीख": "Last Date Deadline", "परीक्षा": "Exam",
    "नोकरी": "Placement Job", "प्लेसमेंट": "Placement", "पगार": "Salary Package",
    "सुविधा": "Facilities", "प्रयोगशाळा": "Laboratory", "ग्रंथालय": "Library",
    "वाहतूक": "Transport", "कसे पोहोचावे": "How to reach Transport",
    "कॅम्पस": "Campus", "संपर्क": "Contact", "पत्ता": "Address",
    "मान्यता": "Recognition Accreditation", "विद्यापीठ": "University",
    "ऋण": "Loan", "शैक्षणिक कर्ज": "Education Loan",
}


@dataclass
class AnswerRequest:
    call_id: str
    question: str
    language: str = "en-IN"
    history: list[Message] = field(default_factory=list)
    stage: str = "conversation"
    turn_index: int = 0
    caller_ref: str | None = None

    @property
    def language_name(self) -> str:
        return get_language(self.language).english_name


@dataclass
class AssistantAnswer:
    text: str
    grounded: bool = False
    confidence: float = 0.0
    needs_escalation: bool = False
    escalation_reason: str | None = None
    escalation_summary: str | None = None
    followup: dict[str, Any] | None = None
    followup_pending: bool = False
    intent: str = "other"
    citations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    provider: str = "template"
    fallback_used: bool = False
    latency_ms: float = 0.0
    first_token_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_turn_metadata(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "provider": self.provider,
            "fallback_used": self.fallback_used,
            "needs_escalation": self.needs_escalation,
            "escalation_reason": self.escalation_reason,
            "citations": self.citations[:6],
            "tool_calls": self.tool_calls[:6],
            "latency_ms": round(self.latency_ms, 1),
            "first_token_ms": round(self.first_token_ms, 1),
            "retrieval_ms": round(self.retrieval_ms, 1),
            "llm_ms": round(self.llm_ms, 1),
            "warnings": self.warnings,
            "followup": self.followup,
            "retrieval_debug": (self.retrieval.debug if self.retrieval else {}),
        }


def augment_query(question: str, intent: IntentResult) -> str:
    """Add Latin equivalents so a Hindi/Marwari query finds English records."""
    extra: list[str] = []
    for native, latin in CROSS_SCRIPT_COURSES.items():
        if native in question:
            extra.append(latin)
    extra.extend(intent.course_tokens)
    extra.extend(intent.specialisations)
    augmented = question.strip()
    if extra:
        augmented = f"{augmented} {' '.join(dict.fromkeys(extra))}"
    return augmented


class AnswerEngine:
    """Owns retrieval + generation for a call. One instance per process."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever
        self.llm, self.llm_info = resolve_llm()
        self.tools: list[ToolSpec] = []
        if self.llm is not None and self.llm.supports_tools:
            from .tools import ALL_TOOLS

            self.tools = list(ALL_TOOLS)
        logger.info(
            "answer engine ready: llm=%s tools=%d", self.llm_info.get("active"), len(self.tools)
        )

    @property
    def provider_name(self) -> str:
        return str(self.llm_info.get("active", "local"))

    async def _retrieve(
        self, question: str, intent: IntentResult, language: str, session: AsyncSession | None
    ) -> RetrievalResult:
        query = augment_query(question, intent)
        return await self.retriever.search(query, language=language, intent=intent)

    # ------------------------------------------------------------------ #
    async def answer(
        self,
        request: AnswerRequest,
        session: AsyncSession | None = None,
        *,
        on_sentence: SentenceCallback | None = None,
        deadline_seconds: float = 6.0,
    ) -> AssistantAnswer:
        started = time.perf_counter()
        question = (request.question or "").strip()
        intent = detect_intent(question)

        # --- topic guardrails ------------------------------------------- #
        topic, topic_confidence = guardrails.classify_topic(question)
        if topic == "sensitive":
            language_frames = prompts.SENSITIVE_ESCALATION
            text = language_frames.get(request.language, language_frames["en-IN"])
            return AssistantAnswer(
                text=text, grounded=False, confidence=0.9, needs_escalation=True,
                escalation_reason="sensitive_or_legal", intent=intent.intent,
                escalation_summary=(
                    f"Caller raised a sensitive/grievance topic: {question[:160]}"
                ),
                provider="guardrail",
                latency_ms=(time.perf_counter() - started) * 1000,
                warnings=["sensitive topic detected"],
            )
        if topic == "off_topic" and len(question.split()) > 3:
            text = prompts.OFF_TOPIC.get(request.language, prompts.OFF_TOPIC["en-IN"])
            return AssistantAnswer(
                text=text, grounded=False, confidence=0.6, intent=intent.intent,
                provider="guardrail",
                latency_ms=(time.perf_counter() - started) * 1000,
                warnings=["off-topic request declined"],
            )

        # --- explicit request for a person ------------------------------- #
        # Never retrieve for this. A caller who asks for a human must reach one,
        # whatever the knowledge base happens to contain. The session layer also
        # checks this, but the HTTP assistant API and the template fallback path
        # do not go through it, so the engine enforces it too.
        if intent.intent == "human_request" and intent.confidence >= 0.45:
            frames = prompts.HUMAN_HANDOFF
            return AssistantAnswer(
                text=frames.get(request.language, frames["en-IN"]),
                grounded=False, confidence=0.95, needs_escalation=True,
                escalation_reason="caller_requested_human", intent=intent.intent,
                provider="guardrail",
                escalation_summary=(
                    f"Caller asked for a human ({request.language}): {question[:160]}"
                ),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # --- retrieval --------------------------------------------------- #
        retrieval_started = time.perf_counter()
        retrieval = await self._retrieve(question, intent, request.language, session)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        if not retrieval.items:
            text = prompts.NOT_FOUND.get(request.language, prompts.NOT_FOUND["en-IN"])
            return AssistantAnswer(
                text=text, grounded=False, confidence=0.0, needs_escalation=True,
                escalation_reason="kb_no_answer", intent=intent.intent,
                retrieval=retrieval, provider="rag",
                latency_ms=(time.perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                warnings=["knowledge base is empty or nothing matched"],
            )

        # --- generation --------------------------------------------------- #
        answer: AssistantAnswer | None = None
        if self.llm is not None:
            try:
                answer = await asyncio.wait_for(
                    self._answer_with_llm(request, intent, retrieval, session, on_sentence),
                    timeout=deadline_seconds,
                )
            except TimeoutError:
                logger.warning("LLM deadline exceeded; falling back to templates")
                answer = None
            except Exception as exc:
                logger.exception("LLM generation failed: %s", exc)
                answer = None

        if answer is None:
            fallback = answer is None
            answer = self._answer_with_templates(request, intent, retrieval)
            answer.fallback_used = fallback
            if on_sentence and answer.text:
                for sentence in _sentence_chunks(answer.text):
                    await on_sentence(sentence)

        answer.retrieval = retrieval
        answer.retrieval_ms = retrieval_ms
        answer.intent = answer.intent or intent.intent
        answer.latency_ms = (time.perf_counter() - started) * 1000

        # --- voice scrub --------------------------------------------------- #
        scrubbed = guardrails.scrub_for_voice(answer.text, language=request.language)
        if scrubbed.text != answer.text:
            answer.text = scrubbed.text
        answer.warnings.extend(scrubbed.warnings)

        # --- numeric grounding check --------------------------------------- #
        if answer.grounded and answer.provider != "guardrail":
            context = retrieval.context_block(max_chars=6000)
            check = guardrails.numeric_grounding_check(answer.text, context)
            answer.debug["numeric_grounding"] = check
            if not check["ok"]:
                answer.warnings.append(
                    "ungrounded numbers: " + ", ".join(check["ungrounded_numbers"][:4])
                )
                answer.grounded = False
                answer.confidence = min(answer.confidence, 0.3)
                answer.needs_escalation = True
                answer.escalation_reason = answer.escalation_reason or "low_confidence"
                answer.text = prompts.NOT_FOUND.get(
                    request.language, prompts.NOT_FOUND["en-IN"]
                )
                answer.escalation_summary = (
                    "Model produced numbers that are not in the knowledge base; "
                    f"unmatched: {', '.join(check['ungrounded_numbers'][:5])}"
                )

        if answer.needs_escalation and not answer.escalation_summary:
            answer.escalation_summary = (
                f"Language {request.language}; intent {answer.intent}; "
                f"question: {question[:180]}"
            )
        answer.debug.update(
            {
                "topic": topic,
                "topic_confidence": topic_confidence,
                "intent": intent.to_dict(),
                "provider": answer.provider,
                "llm": self.llm_info,
            }
        )
        return answer

    # ------------------------------------------------------------------ #
    async def _answer_with_llm(
        self,
        request: AnswerRequest,
        intent: IntentResult,
        retrieval: RetrievalResult,
        session: AsyncSession | None,
        on_sentence: SentenceCallback | None,
    ) -> AssistantAnswer:
        assert self.llm is not None
        language = get_language(request.language)
        system = prompts.build_system_prompt(
            language_code=language.code,
            language_name=language.english_name,
            academic_year=settings.kb_academic_year,
        )
        context = retrieval.context_block(max_chars=4200)
        user_prompt = prompts.build_user_prompt(
            context=context,
            language_code=language.code,
            language_name=language.english_name,
            academic_year=settings.kb_academic_year,
            today=ist_today().isoformat(),
            stage=request.stage,
            question=request.question,
        )

        messages: list[Message] = list(request.history)[-8:]
        messages.append(Message(role="user", content=user_prompt))

        text_buffer: list[str] = []
        pending_sentence: list[str] = []
        tool_calls: list[ToolCall] = []
        executed: list[dict[str, Any]] = []
        followup: dict[str, Any] | None = None
        escalate: tuple[str, str] | None = None
        first_token_ms = 0.0
        llm_started = time.perf_counter()
        rounds = 0
        usage: dict[str, Any] = {}

        async def flush_sentence(force: bool = False) -> None:
            joined = "".join(pending_sentence)
            if not joined:
                return
            if on_sentence:
                await on_sentence(joined)
            pending_sentence.clear()
            _ = force

        while rounds < 2:
            rounds += 1
            stop_reason = "end_turn"
            async for event in self.llm.stream(
                system, messages, tools=self.tools or None,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                language=request.language,
            ):
                if event.type == "text_delta":
                    if not first_token_ms:
                        first_token_ms = (time.perf_counter() - llm_started) * 1000
                    text_buffer.append(event.text)
                    pending_sentence.append(event.text)
                    chunk = "".join(pending_sentence)
                    if _sentence_complete(chunk) and on_sentence:
                        await flush_sentence()
                elif event.type == "tool_use" and event.tool_call:
                    tool_calls.append(event.tool_call)
                elif event.type == "usage":
                    usage = event.usage or usage
                elif event.type == "stop":
                    stop_reason = event.stop_reason or stop_reason
                elif event.type == "error":
                    raise RuntimeError(event.error or "llm stream error")

            if not tool_calls:
                break

            # execute the tool calls, then continue the conversation
            messages.append(Message(role="assistant", content="".join(text_buffer)))
            for call in tool_calls:
                result = await self._run_tool(call, request, intent, retrieval, session)
                executed.append({"name": call.name, "input": call.input, "result": result})
                if call.name == "escalate_to_human":
                    escalate = (
                        str(call.input.get("reason") or "kb_no_answer"),
                        str(call.input.get("summary") or "")[:600],
                    )
                if call.name == "offer_details":
                    followup = {
                        "channel": call.input.get("channel", "sms"),
                        "title": call.input.get("title"),
                        "items": call.input.get("items") or [],
                        "ask_for_destination": bool(
                            call.input.get("ask_for_destination", True)
                        ),
                    }
                if call.name in {"search_knowledge_base", "lookup_course"}:
                    messages.append(
                        Message(
                            role="tool",
                            tool_call_id=call.id,
                            content=json.dumps(result, ensure_ascii=False, default=str)[:6000],
                        )
                    )
            if escalate:
                break
            tool_calls = []

        await flush_sentence(force=True)
        text = "".join(text_buffer).strip()
        llm_ms = (time.perf_counter() - llm_started) * 1000

        if escalate:
            reason, summary = escalate
            return AssistantAnswer(
                text=text or prompts.NOT_FOUND.get(request.language, prompts.NOT_FOUND["en-IN"]),
                grounded=bool(retrieval.grounded),
                confidence=retrieval.best_score,
                needs_escalation=True,
                escalation_reason=reason,
                escalation_summary=summary or f"question: {request.question[:180]}",
                followup=followup,
                intent=intent.intent,
                citations=retrieval.citations,
                tool_calls=[{"name": t["name"], "input": t["input"]} for t in executed],
                provider=self.provider_name,
                first_token_ms=first_token_ms,
                llm_ms=llm_ms,
                debug={"rounds": rounds, "usage": usage, "tools": executed},
            )

        if not text:
            raise RuntimeError("model returned no text")

        return AssistantAnswer(
            text=text,
            grounded=retrieval.grounded,
            confidence=min(0.97, 0.5 + retrieval.best_score * 0.5),
            needs_escalation=False,
            followup=followup,
            intent=intent.intent,
            citations=retrieval.citations,
            tool_calls=[{"name": t["name"], "input": t["input"]} for t in executed],
            provider=self.provider_name,
            first_token_ms=first_token_ms,
            llm_ms=llm_ms,
            debug={"rounds": rounds, "usage": usage, "tools": executed, "stop": stop_reason},
        )

    # ------------------------------------------------------------------ #
    def _answer_with_templates(
        self, request: AnswerRequest, intent: IntentResult, retrieval: RetrievalResult
    ) -> AssistantAnswer:
        composed = template_compose(
            retrieval, language=request.language, question=request.question, intent=intent,
            min_confidence=settings.answer_confidence_threshold,
        )
        return AssistantAnswer(
            text=composed.text,
            grounded=composed.grounded,
            confidence=composed.confidence,
            needs_escalation=composed.needs_escalation,
            escalation_reason=composed.escalation_reason,
            followup=composed.followup,
            intent=composed.intent,
            citations=composed.citations,
            provider="template",
            warnings=(["template engine (no LLM key configured)"] if not self.llm else []),
            debug={"template": composed.template, **composed.debug},
        )

    # ------------------------------------------------------------------ #
    async def _run_tool(
        self,
        call: ToolCall,
        request: AnswerRequest,
        intent: IntentResult,
        retrieval: RetrievalResult,
        session: AsyncSession | None,
    ) -> dict[str, Any]:
        name = call.name
        payload = call.input or {}
        if name == "search_knowledge_base":
            query = str(payload.get("query") or request.question)
            category = payload.get("category")
            sub_intent = detect_intent(query)
            result = await self.retriever.search(
                query,
                language=request.language,
                intent=sub_intent,
                top_k=int(payload.get("top_k") or 4),
                categories=(category,) if category else None,
            )
            return {
                "matches": [
                    {
                        "title": item.title,
                        "category": item.category,
                        "academic_year": item.academic_year,
                        "verified": item.verified,
                        "score": item.score,
                        "snippet": item.text[:420],
                    }
                    for item in result.items
                ],
                "grounded": result.grounded,
                "best_score": result.best_score,
            }

        if name == "lookup_course":
            course = str(payload.get("course") or "").strip()
            specialisation = str(payload.get("specialisation") or "").strip()
            query = f"{course} {specialisation}".strip()
            if not query:
                return {"error": "course name required"}
            found = await self._lookup_course_record(query, session)
            if not found:
                return {"error": "no record matched", "suggestion": "use search_knowledge_base"}
            return found

        if name == "escalate_to_human":
            return {"status": "escalation_queued", "reason": payload.get("reason")}

        if name == "offer_details":
            return {"status": "followup_scheduled", "channel": payload.get("channel")}

        return {"error": f"unknown tool {name}"}

    async def _lookup_course_record(
        self, query: str, session: AsyncSession | None
    ) -> dict[str, Any]:
        """Exact structured lookup so the model never has to read a number from prose."""
        from ..kb.chunking import CATEGORY_LABELS

        result = await self.retriever.search(
            query, language="en-IN", top_k=6, categories=("course", "specialisation", "fees",
                                                          "eligibility", "placements")
        )
        if not result.items:
            return {}
        record_ids = [item.record_id for item in result.items[:3]]
        facts: dict[str, Any] = {}
        if session is not None:
            rows = (
                await session.execute(select(KBRecord).where(KBRecord.id.in_(record_ids)))
            ).scalars().all()
            for row in rows:
                facts[row.title] = {
                    "category": CATEGORY_LABELS.get(row.category, row.category),
                    "academic_year": row.academic_year,
                    "verified": bool(row.verified),
                    "structured": row.structured or {},
                }
        else:
            for item in result.items[:3]:
                facts[item.title] = {
                    "category": item.category,
                    "academic_year": item.academic_year,
                    "verified": item.verified,
                    "structured": item.structured,
                }
        return {"course": query, "records": facts}


# --------------------------------------------------------------------------- #
# sentence streaming helpers
# --------------------------------------------------------------------------- #

_SENTENCE_END = re.compile(r"[.!?।؟]")


def _sentence_complete(buffer: str) -> bool:
    """True when the buffer ends with a terminator and is long enough to speak."""
    stripped = buffer.strip()
    if len(stripped) < 18:
        return False
    if not _SENTENCE_END.search(stripped[-1:]):
        return False
    # avoid splitting "B.Tech" / "₹1,50,000." style artefacts
    # "B.Tech" / "M.Sc." style abbreviations are not sentence ends.
    return not (stripped.endswith(".") and re.search(r"[A-Z]\.$", stripped.rstrip(".")))


def _sentence_chunks(text: str, max_chars: int = 180) -> list[str]:
    from ..voice.tts.base import split_for_speech

    return split_for_speech(text, max_chars=max_chars)
