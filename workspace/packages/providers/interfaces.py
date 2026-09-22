"""Provider interfaces for speech, LLM, embeddings, storage, and cache."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from packages.schemas.schemas import LanguageCode


# =============================================================================
# Speech Provider Interface
# =============================================================================

class SpeechEvent(str, Enum):
    """Speech event types."""
    RECOGNIZING = "recognizing"
    RECOGNIZED = "recognized"
    LANGUAGE_DETECTED = "language_detected"
    SESSION_STARTED = "session_started"
    SESSION_STOPPED = "session_stopped"
    CANCELED = "canceled"


@dataclass
class SpeechRecognitionResult:
    """ASR result."""
    text: str
    is_final: bool
    confidence: Optional[float] = None
    language: Optional[LanguageCode] = None
    event_type: SpeechEvent = SpeechEvent.RECOGNIZED


@dataclass
class LanguageIdentificationResult:
    """Language identification result."""
    language: LanguageCode
    confidence: float


@dataclass
class TTSAudioChunk:
    """TTS audio chunk."""
    audio_data: bytes  # PCM16 mono 16kHz
    mark_id: Optional[str] = None
    is_final: bool = False


class SpeechProvider(ABC):
    """Abstract speech provider interface."""

    @abstractmethod
    async def recognize_continuous(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: Optional[LanguageCode] = None
    ) -> AsyncGenerator[SpeechRecognitionResult, None]:
        """Continuous speech recognition from audio stream."""
        pass

    @abstractmethod
    async def identify_language(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        candidate_languages: List[LanguageCode]
    ) -> LanguageIdentificationResult:
        """Identify spoken language from audio."""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        language: LanguageCode,
        voice_name: Optional[str] = None
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        """Text-to-speech synthesis."""
        pass

    @abstractmethod
    async def cancel_synthesis(self, generation_id: str) -> None:
        """Cancel ongoing synthesis."""
        pass


# =============================================================================
# LLM Provider Interface
# =============================================================================

@dataclass
class Message:
    """Chat message."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class LLMResponse:
    """LLM response."""
    content: str
    finish_reason: str
    usage: Dict[str, int]
    model: str


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResponse:
        """Generate completion from messages."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion."""
        pass


# =============================================================================
# Embedding Provider Interface
# =============================================================================

@dataclass
class EmbeddingResult:
    """Embedding result."""
    embedding: List[float]
    model: str
    usage: Dict[str, int]


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for batch of texts."""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        pass


# =============================================================================
# Blob Storage Provider Interface
# =============================================================================

@dataclass
class BlobMetadata:
    """Blob metadata."""
    path: str
    size: int
    content_type: str
    created_at: str
    etag: str
    metadata: Dict[str, str]


class BlobStorageProvider(ABC):
    """Abstract blob storage provider interface."""

    @abstractmethod
    async def upload(
        self,
        container: str,
        path: str,
        data: bytes,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> BlobMetadata:
        """Upload blob."""
        pass

    @abstractmethod
    async def download(self, container: str, path: str) -> bytes:
        """Download blob."""
        pass

    @abstractmethod
    async def delete(self, container: str, path: str) -> None:
        """Delete blob."""
        pass

    @abstractmethod
    async def exists(self, container: str, path: str) -> bool:
        """Check if blob exists."""
        pass

    @abstractmethod
    async def list_blobs(self, container: str, prefix: Optional[str] = None) -> List[BlobMetadata]:
        """List blobs in container."""
        pass


# =============================================================================
# Cache Provider Interface
# =============================================================================

class CacheProvider(ABC):
    """Abstract cache provider interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def expire(self, key: str, ttl: int) -> None:
        """Set expiration on key."""
        pass
