"""Database models and schema for SVKM Voice Assistant."""

from datetime import datetime
from typing import Optional
from uuid import uuid4
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey,
    Enum as SQLEnum, JSON, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from packages.schemas.schemas import (
    CallStatus, TurnRole, DecisionType, PublicationStatus,
    SourceType, School, IngestionStatus
)

Base = declarative_base()


# =============================================================================
# Knowledge Base Tables
# =============================================================================

class KBSource(Base):
    """Knowledge base source registry."""
    __tablename__ = "kb_sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_type = Column(SQLEnum(SourceType), nullable=False)
    url = Column(String(2048), nullable=True)
    title = Column(String(512), nullable=False)
    is_allowlisted = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    versions = relationship("KBSourceVersion", back_populates="source", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_kb_sources_active", "is_active"),
        Index("ix_kb_sources_type", "source_type"),
    )


class KBSourceVersion(Base):
    """Immutable source version snapshots."""
    __tablename__ = "kb_source_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("kb_sources.id"), nullable=False)
    content_hash = Column(String(64), nullable=False)
    raw_content = Column(Text, nullable=True)
    blob_path = Column(String(1024), nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=dict)

    source = relationship("KBSource", back_populates="versions")
    documents = relationship("KBDocument", back_populates="source_version")

    __table_args__ = (
        Index("ix_kb_source_versions_source", "source_id"),
        Index("ix_kb_source_versions_hash", "content_hash"),
    )


class KBDocument(Base):
    """Parsed documents from sources."""
    __tablename__ = "kb_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_version_id = Column(UUID(as_uuid=True), ForeignKey("kb_source_versions.id"), nullable=False)
    title = Column(String(512), nullable=False)
    content = Column(Text, nullable=False)
    status = Column(SQLEnum(PublicationStatus), default=PublicationStatus.DRAFT, nullable=False)
    school = Column(SQLEnum(School), nullable=True)
    academic_year = Column(String(16), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    published_at = Column(DateTime, nullable=True)

    source_version = relationship("KBSourceVersion", back_populates="documents")
    chunks = relationship("KBChunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_kb_documents_status", "status"),
        Index("ix_kb_documents_school", "school"),
        Index("ix_kb_documents_year", "academic_year"),
    )


class KBChunk(Base):
    """Text chunks with embeddings for retrieval."""
    __tablename__ = "kb_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("kb_documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    chunk_index = Column(Integer, nullable=False)
    status = Column(SQLEnum(PublicationStatus), default=PublicationStatus.DRAFT, nullable=False)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    document = relationship("KBDocument", back_populates="chunks")

    __table_args__ = (
        Index("ix_kb_chunks_status", "status"),
        Index("ix_kb_chunks_embedding", "embedding", postgresql_using="ivfflat"),
        Index("ix_kb_chunks_content_fts", "content", postgresql_using="gin", postgresql_ops={"content": "gin_trgm_ops"}),
    )


class KBFact(Base):
    """Structured facts extracted from content."""
    __tablename__ = "kb_facts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("kb_chunks.id"), nullable=False)
    fact_type = Column(String(64), nullable=False)
    entity = Column(String(256), nullable=False)
    value = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(SQLEnum(PublicationStatus), default=PublicationStatus.DRAFT, nullable=False)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_kb_facts_type", "fact_type"),
        Index("ix_kb_facts_entity", "entity"),
        Index("ix_kb_facts_status", "status"),
    )


class IngestionRun(Base):
    """Ingestion job tracking."""
    __tablename__ = "ingestion_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_type = Column(SQLEnum(SourceType), nullable=False)
    status = Column(SQLEnum(IngestionStatus), default=IngestionStatus.PENDING, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    documents_processed = Column(Integer, default=0, nullable=False)
    chunks_created = Column(Integer, default=0, nullable=False)
    errors = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)

    __table_args__ = (
        Index("ix_ingestion_runs_status", "status"),
        Index("ix_ingestion_runs_started", "started_at"),
    )


# =============================================================================
# Call & Session Tables
# =============================================================================

class VoiceSession(Base):
    """Voice call sessions."""
    __tablename__ = "voice_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    external_id = Column(String(256), nullable=True)
    status = Column(SQLEnum(CallStatus), default=CallStatus.ACTIVE, nullable=False)
    detected_language = Column(String(10), nullable=True)
    language_confidence = Column(Float, nullable=True)
    consent_given = Column(Boolean, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    metadata = Column(JSON, default=dict)

    calls = relationship("Call", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_voice_sessions_status", "status"),
        Index("ix_voice_sessions_started", "started_at"),
        Index("ix_voice_sessions_external", "external_id"),
    )


class Call(Base):
    """Individual calls within a session."""
    __tablename__ = "calls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("voice_sessions.id"), nullable=False)
    status = Column(SQLEnum(CallStatus), default=CallStatus.ACTIVE, nullable=False)
    school = Column(SQLEnum(School), nullable=True)
    program_name = Column(String(256), nullable=True)
    handoff_reason = Column(String(512), nullable=True)
    handoff_contact = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    session = relationship("VoiceSession", back_populates="calls")
    turns = relationship("Turn", back_populates="call", cascade="all, delete-orphan")
    events = relationship("CallEvent", back_populates="call", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_calls_session", "session_id"),
        Index("ix_calls_status", "status"),
        Index("ix_calls_school", "school"),
    )


class Turn(Base):
    """Conversation turns."""
    __tablename__ = "turns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id"), nullable=False)
    role = Column(SQLEnum(TurnRole), nullable=False)
    content = Column(Text, nullable=False)
    language = Column(String(10), nullable=True)
    decision_type = Column(SQLEnum(DecisionType), nullable=True)
    confidence = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=dict)

    call = relationship("Call", back_populates="turns")
    evidence = relationship("RetrievalEvidence", back_populates="turn", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_turns_call", "call_id"),
        Index("ix_turns_role", "role"),
        Index("ix_turns_created", "created_at"),
    )


class RetrievalEvidence(Base):
    """Retrieved chunks used as evidence."""
    __tablename__ = "retrieval_evidence"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    turn_id = Column(UUID(as_uuid=True), ForeignKey("turns.id"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("kb_chunks.id"), nullable=False)
    score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    used_in_response = Column(Boolean, default=False, nullable=False)

    turn = relationship("Turn", back_populates="evidence")

    __table_args__ = (
        Index("ix_retrieval_evidence_turn", "turn_id"),
        Index("ix_retrieval_evidence_chunk", "chunk_id"),
    )


class CallEvent(Base):
    """Append-only event log."""
    __tablename__ = "call_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id"), nullable=False)
    event_type = Column(String(64), nullable=False)
    event_data = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    call = relationship("Call", back_populates="events")

    __table_args__ = (
        Index("ix_call_events_call", "call_id"),
        Index("ix_call_events_timestamp", "timestamp"),
        Index("ix_call_events_type", "event_type"),
    )


# =============================================================================
# Contact & Routing Tables
# =============================================================================

class ContactRoute(Base):
    """School contact routing configuration."""
    __tablename__ = "contact_routes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    school = Column(SQLEnum(School), nullable=False, unique=True)
    primary_phone = Column(String(32), nullable=False)
    fallback_phones = Column(ARRAY(String), default=list, nullable=False)
    email = Column(String(256), nullable=True)
    business_hours = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_contact_routes_school", "school"),
        Index("ix_contact_routes_active", "is_active"),
    )


# =============================================================================
# Audit & Operations Tables
# =============================================================================

class AuditEvent(Base):
    """Append-only audit log."""
    __tablename__ = "audit_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(256), nullable=True)
    action = Column(String(128), nullable=False)
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(256), nullable=False)
    changes = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_audit_events_timestamp", "timestamp"),
        Index("ix_audit_events_user", "user_id"),
        Index("ix_audit_events_resource", "resource_type", "resource_id"),
    )
