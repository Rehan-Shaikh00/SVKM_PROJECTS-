"""ORM models — call logs, transcripts, knowledge base, escalations, follow-ups.

Everything the analytics dashboard needs is written here during the call, not
reconstructed afterwards.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class CallStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    FAILED = "failed"
    ABANDONED = "abandoned"


class TurnRole(str, enum.Enum):
    CALLER = "caller"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    AGENT = "agent"
    TOOL = "tool"


class KBRecordStatus(str, enum.Enum):
    PUBLISHED = "published"
    DRAFT = "draft"
    ARCHIVED = "archived"


class CallRecord(Base):
    """One inbound phone call."""

    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    provider: Mapped[str] = mapped_column(String(32), default="twilio", index=True)
    provider_call_sid: Mapped[str | None] = mapped_column(String(64), index=True)
    stream_sid: Mapped[str | None] = mapped_column(String(64), index=True)
    direction: Mapped[str] = mapped_column(String(16), default="inbound")
    #: Stored hashed when REDACT_PII is on; the raw value lives only in the
    #: telephony provider's own records.
    from_number: Mapped[str | None] = mapped_column(String(128), index=True)
    to_number: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(24), default=CallStatus.IN_PROGRESS.value, index=True)
    end_reason: Mapped[str | None] = mapped_column(String(64))

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    # --- language ---------------------------------------------------------
    detected_language: Mapped[str | None] = mapped_column(String(16), index=True)
    language_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    language_method: Mapped[str | None] = mapped_column(String(32))  # audio_lid | text_lid | explicit | dtmf | default
    language_attempts: Mapped[int] = mapped_column(Integer, default=0)
    fallback_language_used: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- outcome ----------------------------------------------------------
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    escalation_reason: Mapped[str | None] = mapped_column(String(64))
    caller_requested_human: Mapped[bool] = mapped_column(Boolean, default=False)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    unanswered_count: Mapped[int] = mapped_column(Integer, default=0)
    primary_intent: Mapped[str | None] = mapped_column(String(48), index=True)
    intents_json: Mapped[list | None] = mapped_column(JSON, default=list)

    # --- quality / latency -------------------------------------------------
    time_to_first_audio_ms: Mapped[float] = mapped_column(Float, default=0.0)
    avg_response_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    barge_in_count: Mapped[int] = mapped_column(Integer, default=0)
    satisfaction: Mapped[str | None] = mapped_column(String(16))

    # --- artefacts ---------------------------------------------------------
    summary: Mapped[str | None] = mapped_column(Text)
    transcript_text: Mapped[str | None] = mapped_column(Text)
    recording_uri: Mapped[str | None] = mapped_column(String(512))
    recording_consent: Mapped[str | None] = mapped_column(String(32))
    metadata_json: Mapped[dict | None] = mapped_column(JSON, default=dict)

    turns: Mapped[list[CallTurn]] = relationship(
        back_populates="call", cascade="all, delete-orphan", order_by="CallTurn.seq"
    )
    escalations: Mapped[list[EscalationEvent]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )
    followups: Mapped[list[FollowUp]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_calls_started_status", "started_at", "status"),)


class CallTurn(Base):
    """One utterance (caller) or response (assistant) inside a call."""

    __tablename__ = "call_turns"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    role: Mapped[str] = mapped_column(String(16), index=True)
    text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str | None] = mapped_column(String(16))
    is_final: Mapped[bool] = mapped_column(Boolean, default=True)
    interrupted: Mapped[bool] = mapped_column(Boolean, default=False)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # latency breakdown, milliseconds
    asr_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_ms: Mapped[float] = mapped_column(Float, default=0.0)
    llm_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tts_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_ms: Mapped[float] = mapped_column(Float, default=0.0)

    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    grounded: Mapped[bool] = mapped_column(Boolean, default=False)
    citations_json: Mapped[list | None] = mapped_column(JSON, default=list)
    tool_calls_json: Mapped[list | None] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, default=dict)

    call: Mapped[CallRecord] = relationship(back_populates="turns")


class EscalationEvent(Base):
    """Transfer of a call to a human agent, with the context handed over."""

    __tablename__ = "escalations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    reason: Mapped[str] = mapped_column(String(64), index=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    target: Mapped[str | None] = mapped_column(String(128))
    target_type: Mapped[str] = mapped_column(String(24), default="number")  # number | queue
    outcome: Mapped[str] = mapped_column(String(24), default="pending")      # pending | answered | no_answer | failed
    wait_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    whisper_summary: Mapped[str | None] = mapped_column(Text)
    context_json: Mapped[dict | None] = mapped_column(JSON, default=dict)

    call: Mapped[CallRecord] = relationship(back_populates="escalations")


class FollowUp(Base):
    """SMS / WhatsApp / email sent after a call so the caller gets full details."""

    __tablename__ = "followups"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    channel: Mapped[str] = mapped_column(String(16), index=True)  # sms | whatsapp | email
    destination_ref: Mapped[str] = mapped_column(String(128))     # hashed
    body: Mapped[str] = mapped_column(Text, default="")
    attachments_json: Mapped[list | None] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(24), default="queued")
    provider_message_id: Mapped[str | None] = mapped_column(String(128))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    call: Mapped[CallRecord] = relationship(back_populates="followups")


class KBRecord(Base):
    """A single source-of-truth knowledge item (course, fee row, policy, FAQ…)."""

    __tablename__ = "kb_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    slug: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    category: Mapped[str] = mapped_column(String(48), index=True)
    #: course | specialisation | eligibility | fees | admission_process | important_dates
    #: scholarship | hostel | placement | facility | contact | policy | faq | department
    subcategory: Mapped[str | None] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(512))
    title_localized: Mapped[dict | None] = mapped_column(JSON, default=dict)
    body: Mapped[str] = mapped_column(Text, default="")
    #: machine-readable facts, e.g. {"fees": {"year_1": 150000, "total": 600000},
    #: "duration_years": 4, "eligibility": "...", "seats": 120}
    structured: Mapped[dict | None] = mapped_column(JSON, default=dict)
    language: Mapped[str] = mapped_column(String(16), default="en-IN", index=True)
    tags: Mapped[list | None] = mapped_column(JSON, default=list)
    aliases: Mapped[list | None] = mapped_column(JSON, default=list)

    academic_year: Mapped[str | None] = mapped_column(String(16), index=True)
    source: Mapped[str | None] = mapped_column(String(256))
    source_uri: Mapped[str | None] = mapped_column(String(512))
    #: provenance — an answer may only be spoken from a verified record unless
    #: ALLOW_UNVERIFIED_ANSWERS is set (see guardrails).
    verified: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    verified_by: Mapped[str | None] = mapped_column(String(128))
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    status: Mapped[str] = mapped_column(
        String(16), default=KBRecordStatus.PUBLISHED.value, index=True
    )
    revision: Mapped[int] = mapped_column(Integer, default=1)
    created_by: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    chunks: Mapped[list[KBChunk]] = relationship(
        back_populates="record", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_kb_category_status", "category", "status"),)

    @property
    def is_stale(self) -> bool:
        from .config import settings

        if self.status != KBRecordStatus.PUBLISHED.value:
            return False
        reference = self.verified_at or self.updated_at or utcnow()
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)
        age_days = (utcnow() - reference).days
        return age_days > settings.kb_staleness_days


class KBChunk(Base):
    """Retrieval unit: a chunk of a record plus its embedding + provenance."""

    __tablename__ = "kb_chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    record_id: Mapped[str] = mapped_column(ForeignKey("kb_records.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    #: stable hash so re-ingesting identical content does not re-embed everything
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    language: Mapped[str] = mapped_column(String(16), default="en-IN")
    category: Mapped[str] = mapped_column(String(48), index=True)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str] = mapped_column(String(64), default="local")
    #: NULL for external vector stores (pgvector/qdrant hold the vectors)
    embedding_json: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    record: Mapped[KBRecord] = relationship(back_populates="chunks")

    __table_args__ = (Index("ix_kb_chunk_record_pos", "record_id", "position"),)


class KBRevision(Base):
    """Audit trail so staff edits can be reviewed and rolled back."""

    __tablename__ = "kb_revisions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    record_id: Mapped[str] = mapped_column(String(32), index=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    changed_by: Mapped[str | None] = mapped_column(String(128))
    change_note: Mapped[str | None] = mapped_column(Text)
    snapshot: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UnansweredQuestion(Base):
    """Questions the AI could not ground — the KB backlog for the next cycle."""

    __tablename__ = "unanswered_questions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    call_id: Mapped[str | None] = mapped_column(String(32), index=True)
    question: Mapped[str] = mapped_column(Text)
    #: canonicalised form used to group near-duplicates
    canonical_key: Mapped[str] = mapped_column(String(160), index=True)
    language: Mapped[str | None] = mapped_column(String(16))
    intent: Mapped[str | None] = mapped_column(String(48))
    best_score: Mapped[float] = mapped_column(Float, default=0.0)
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    resolution: Mapped[str] = mapped_column(String(32), default="open")  # open | kb_added | not_relevant
    resolution_note: Mapped[str | None] = mapped_column(Text)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProviderHealth(Base):
    """Rolling provider health, surfaced on the dashboard."""

    __tablename__ = "provider_health"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    component: Mapped[str] = mapped_column(String(32), index=True)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[str | None] = mapped_column(Text)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
