"""Transport-neutral conversation state machine."""

import asyncio
from enum import Enum
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from dataclasses import dataclass, field

from packages.schemas.schemas import LanguageCode, DecisionType, Citation


class ConversationState(str, Enum):
    """Conversation state machine states."""
    INITIALIZING = "initializing"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    HANDOFF = "handoff"
    ENDED = "ended"


@dataclass
class ConversationContext:
    """Conversation context and memory."""
    session_id: UUID
    detected_language: Optional[LanguageCode] = None
    language_confidence: float = 0.0
    resolved_school: Optional[str] = None
    resolved_program: Optional[str] = None
    resolved_year: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_generation_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class Turn:
    """Single conversation turn."""
    id: UUID
    role: str
    content: str
    language: Optional[LanguageCode]
    decision_type: Optional[DecisionType]
    citations: List[Citation]
    latency_ms: float
    timestamp: datetime


class ConversationEngine:
    """Transport-neutral conversation state machine."""

    def __init__(
        self,
        session_id: UUID,
        speech_provider,
        llm_provider,
        retrieval_service,
        handoff_service
    ):
        self.session_id = session_id
        self.speech = speech_provider
        self.llm = llm_provider
        self.retrieval = retrieval_service
        self.handoff = handoff_service

        self.state = ConversationState.INITIALIZING
        self.context = ConversationContext(session_id=session_id)
        self._cancelled = False
        self._lock = asyncio.Lock()

    async def process_text_input(self, text: str, language: LanguageCode) -> Dict[str, Any]:
        """Process text input (for testing or text mode)."""
        async with self._lock:
            if self.state not in [ConversationState.LISTENING, ConversationState.INITIALIZING]:
                return {"error": "Invalid state for input"}

            self.state = ConversationState.THINKING
            self.context.detected_language = language

            # Retrieve evidence
            retrieval_results = await self.retrieval.search(
                query=text,
                language=language,
                context=self.context
            )

            # Generate grounded response
            response = await self._generate_response(text, retrieval_results, language)

            self.state = ConversationState.SPEAKING
            return response

    async def _generate_response(
        self,
        user_input: str,
        retrieval_results: List[Any],
        language: LanguageCode
    ) -> Dict[str, Any]:
        """Generate grounded response using LLM."""
        # Build system prompt
        system_prompt = self._build_system_prompt(language)

        # Build evidence context
        evidence_text = "\n\n".join([
            f"[Evidence {i+1}] {r.content}"
            for i, r in enumerate(retrieval_results[:5])
        ])

        # Build messages
        messages = []

        # Add conversation history
        for turn in self.context.conversation_history[-3:]:
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })

        # Add current query with evidence
        user_message = f"""Evidence from knowledge base:
{evidence_text}

User question: {user_input}

Provide a brief, spoken answer grounded in the evidence above."""

        from packages.providers.interfaces import Message
        messages.append(Message(role="user", content=user_message))

        # Generate response
        llm_response = await self.llm.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=512
        )

        # Validate response against evidence
        decision_type = self._validate_response(llm_response.content, retrieval_results)

        # Build citations
        citations = [
            Citation(
                source_id=str(r.chunk_id),
                chunk_id=str(r.chunk_id),
                text=r.content[:200],
                confidence=r.score
            )
            for r in retrieval_results[:3]
        ]

        # Update context
        self.context.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.context.conversation_history.append({
            "role": "assistant",
            "content": llm_response.content,
            "timestamp": datetime.utcnow().isoformat()
        })

        return {
            "text": llm_response.content,
            "language": language,
            "decision_type": decision_type,
            "citations": citations
        }

    def _build_system_prompt(self, language: LanguageCode) -> str:
        """Build system prompt for grounded responses."""
        return f"""You are a voice assistant for SVKM NMIMS Global University, Dhule.

CRITICAL RULES:
1. Answer ONLY from the provided evidence
2. Keep responses brief and conversational (2-3 sentences max)
3. For fees/dates/tables, offer to send details instead of reading them aloud
4. If evidence is missing or conflicting, say "I don't have that information" and offer to connect to admissions
5. Never invent fees, dates, eligibility, or scholarship information
6. Preserve exact values for phone numbers, dates, and amounts
7. Respond in {language.value} language

KNOWN GAPS (always escalate):
- Scholarships (not yet published)
- Engineering-specific fees (pending)
- Dhule hostel specifics (pending)

If uncertain, offer to connect the caller to the appropriate school contact."""

    def _validate_response(self, response_text: str, evidence: List[Any]) -> DecisionType:
        """Validate response is grounded in evidence."""
        # Simple heuristic: if response is very short or contains uncertainty phrases
        uncertainty_phrases = [
            "I don't have",
            "I'm not sure",
            "let me connect you",
            "transfer you",
            "speak with"
        ]

        if any(phrase in response_text.lower() for phrase in uncertainty_phrases):
            return DecisionType.ABSTAIN

        if len(evidence) < 2:
            return DecisionType.ABSTAIN

        if len(response_text) < 50:
            return DecisionType.CLARIFY

        return DecisionType.ANSWER

    async def cancel_generation(self):
        """Cancel current generation."""
        self._cancelled = True
        self.context.current_generation_id = str(uuid4())
        if self.state == ConversationState.SPEAKING:
            await self.speech.cancel_synthesis(self.context.current_generation_id)
        self.state = ConversationState.LISTENING

    async def end_conversation(self, reason: str = "user_ended"):
        """End conversation."""
        self.state = ConversationState.ENDED
        return {"reason": reason, "session_id": str(self.session_id)}
