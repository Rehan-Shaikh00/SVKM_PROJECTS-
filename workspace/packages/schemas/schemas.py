"""Shared schemas and types for SVKM Voice Assistant."""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, HttpUrl


# =============================================================================
# Enums
# =============================================================================

class LanguageCode(str, Enum):
    """Supported language codes."""
    EN_IN = "en-IN"
    HI_IN = "hi-IN"
    MR_IN = "mr-IN"


class CallStatus(str, Enum):
    """Call status states."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TRANSFERRED = "transferred"


class TurnRole(str, Enum):
    """Turn role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DecisionType(str, Enum):
    """Conversation decision types."""
    ANSWER = "answer"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"
    HANDOFF = "handoff"


class PublicationStatus(str, Enum):
    """Knowledge publication status."""
    DRAFT = "draft"
    STAGED = "staged"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class SourceType(str, Enum):
    """Knowledge source types."""
    WEBPAGE = "webpage"
    PDF = "pdf"
    MANUAL = "manual"
    JSON_IMPORT = "json_import"


class School(str, Enum):
    """University schools."""
    STME = "STME"  # School of Technology, Management & Engineering
    SPTM = "SPTM"  # School of Pharmacy & Technology Management
    SC = "SC"      # School of Commerce
    GENERAL = "GENERAL"


class IngestionStatus(str, Enum):
    """Ingestion run status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# WebSocket Event Models
# =============================================================================

class ClientConnectEvent(BaseModel):
    """Client connection event."""
    type: str = Field("connect", const=True)
    session_token: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ClientAudioEvent(BaseModel):
    """Client audio data event."""
    type: str = Field("audio", const=True)
    data: str  # Base64 PCM16 mono 16kHz
    is_final: Optional[bool] = None


class ClientTextEvent(BaseModel):
    """Client text input event (for testing)."""
    type: str = Field("text", const=True)
    text: str
    language: Optional[LanguageCode] = None


class ClientInterruptEvent(BaseModel):
    """Client interrupt/barge-in event."""
    type: str = Field("interrupt", const=True)


class ClientEndEvent(BaseModel):
    """Client end call event."""
    type: str = Field("end", const=True)


class ServerReadyEvent(BaseModel):
    """Server ready event."""
    type: str = Field("ready", const=True)
    session_id: UUID
    supported_languages: List[LanguageCode]


class ServerTranscriptEvent(BaseModel):
    """Server transcript event."""
    type: str = Field("transcript", const=True)
    text: str
    is_final: bool
    language: Optional[LanguageCode] = None
    confidence: Optional[float] = None


class ServerLanguageDetectedEvent(BaseModel):
    """Server language detection event."""
    type: str = Field("language_detected", const=True)
    language: LanguageCode
    confidence: float


class ServerThinkingEvent(BaseModel):
    """Server thinking/processing event."""
    type: str = Field("thinking", const=True)


class Citation(BaseModel):
    """Source citation."""
    source_id: str
    chunk_id: str
    text: str
    confidence: float


class ServerResponseEvent(BaseModel):
    """Server text response event."""
    type: str = Field("response", const=True)
    text: str
    language: LanguageCode
    decision_type: DecisionType
    citations: Optional[List[Citation]] = None


class ServerAudioEvent(BaseModel):
    """Server audio data event."""
    type: str = Field("audio", const=True)
    data: str  # Base64 PCM16 mono 16kHz
    generation_id: str


class ServerAudioMarkEvent(BaseModel):
    """Server audio playback marker event."""
    type: str = Field("audio_mark", const=True)
    mark_id: str
    generation_id: str


class ServerHandoffEvent(BaseModel):
    """Server handoff to human event."""
    type: str = Field("handoff", const=True)
    reason: str
    school: Optional[str] = None
    contact_number: Optional[str] = None
    summary: str


class ServerErrorEvent(BaseModel):
    """Server error event."""
    type: str = Field("error", const=True)
    code: str
    message: str
    recoverable: bool


class ServerEndEvent(BaseModel):
    """Server end call event."""
    type: str = Field("end", const=True)
    reason: str


class ServerLatencyEvent(BaseModel):
    """Server latency metrics event."""
    type: str = Field("latency", const=True)
    stage: str
    duration_ms: float
    timestamp: datetime


# =============================================================================
# Knowledge Base Models
# =============================================================================

class KnowledgeChunkMetadata(BaseModel):
    """Metadata for knowledge chunks."""
    school: Optional[School] = None
    program_name: Optional[str] = None
    academic_year: Optional[str] = None
    category: Optional[str] = None
    page_number: Optional[int] = None
    heading: Optional[str] = None
    source_url: Optional[HttpUrl] = None


class KnowledgeChunk(BaseModel):
    """Knowledge chunk model."""
    id: UUID
    document_id: UUID
    content: str
    embedding: Optional[List[float]] = None
    metadata: KnowledgeChunkMetadata
    status: PublicationStatus
    created_at: datetime
    updated_at: datetime


class RetrievalResult(BaseModel):
    """Search retrieval result."""
    chunk_id: UUID
    content: str
    score: float
    metadata: Dict[str, Any]


# =============================================================================
# Contact & Routing Models
# =============================================================================

class BusinessHours(BaseModel):
    """Business hours configuration."""
    start: str  # HH:MM format
    end: str    # HH:MM format
    days: List[int] = Field(..., description="0=Sunday through 6=Saturday")


class ContactRoute(BaseModel):
    """Contact routing configuration."""
    id: UUID
    school: School
    primary_phone: str
    fallback_phones: List[str]
    email: Optional[str] = None
    business_hours: Optional[BusinessHours] = None
    is_active: bool


# =============================================================================
# Ingestion Models
# =============================================================================

class IngestionError(BaseModel):
    """Ingestion error detail."""
    url: Optional[str] = None
    error: str
    timestamp: datetime


class IngestionRun(BaseModel):
    """Ingestion run status."""
    id: UUID
    status: IngestionStatus
    source_type: SourceType
    started_at: datetime
    completed_at: Optional[datetime] = None
    documents_processed: int
    chunks_created: int
    errors: List[IngestionError]


# =============================================================================
# Analytics Models
# =============================================================================

class CallAnalytics(BaseModel):
    """Call analytics aggregates."""
    total_calls: int
    completed_calls: int
    transferred_calls: int
    failed_calls: int
    average_duration: float  # seconds
    language_distribution: Dict[LanguageCode, int]
    school_distribution: Dict[School, int]
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


# =============================================================================
# Error Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: Optional[UUID] = None
