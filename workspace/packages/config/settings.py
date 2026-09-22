"""Configuration and settings management for SVKM Voice Assistant."""

from typing import Optional, Literal
from pydantic import Field, field_validator, PostgresDsn, RedisDsn, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    url: PostgresDsn = Field(
        default="postgresql://svkm_user:svkm_password@localhost:5432/svkm_voice_assistant",
        description="PostgreSQL connection URL"
    )
    pool_size: int = Field(default=20, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=100)
    echo: bool = Field(default=False, description="Log SQL statements")

    model_config = SettingsConfigDict(env_prefix="DATABASE_")


class RedisSettings(BaseSettings):
    """Redis cache configuration."""

    url: RedisDsn = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    max_connections: int = Field(default=50, ge=1, le=1000)
    socket_timeout: float = Field(default=5.0, gt=0)
    socket_connect_timeout: float = Field(default=5.0, gt=0)

    model_config = SettingsConfigDict(env_prefix="REDIS_")


class AzureSpeechSettings(BaseSettings):
    """Azure Speech Services configuration."""

    key: str = Field(default="", description="Azure Speech API key")
    region: str = Field(default="centralindia", description="Azure region")
    endpoint: Optional[HttpUrl] = Field(
        default="https://centralindia.api.cognitive.microsoft.com/",
        description="Custom endpoint"
    )

    model_config = SettingsConfigDict(env_prefix="AZURE_SPEECH_")


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI configuration for embeddings."""

    endpoint: HttpUrl = Field(
        default="https://your-resource.openai.azure.com",
        description="Azure OpenAI endpoint"
    )
    key: str = Field(default="", description="Azure OpenAI API key")
    embedding_deployment: str = Field(
        default="text-embedding-3-large",
        description="Embedding model deployment name"
    )
    embedding_dimensions: int = Field(default=1536, ge=256, le=3072)
    api_version: str = Field(default="2024-02-15-preview")

    model_config = SettingsConfigDict(env_prefix="AZURE_OPENAI_")


class AnthropicSettings(BaseSettings):
    """Anthropic Claude API configuration."""

    api_key: str = Field(default="", description="Anthropic API key")
    model: str = Field(default="claude-opus-4", description="Claude model to use")
    max_tokens: int = Field(default=4096, ge=256, le=8192)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_")


class AzureStorageSettings(BaseSettings):
    """Azure Blob Storage configuration."""

    connection_string: str = Field(default="", description="Azure Storage connection string")
    container_raw_sources: str = Field(default="raw-sources")
    container_recordings: str = Field(default="call-recordings")

    model_config = SettingsConfigDict(env_prefix="AZURE_STORAGE_")


class AzureKeyVaultSettings(BaseSettings):
    """Azure Key Vault configuration."""

    url: Optional[HttpUrl] = Field(
        default=None,
        description="Key Vault URL"
    )
    enabled: bool = Field(default=False, description="Enable Key Vault integration")

    model_config = SettingsConfigDict(env_prefix="AZURE_KEY_VAULT_")


class ExotelSettings(BaseSettings):
    """Exotel telephony configuration."""

    enabled: bool = Field(default=False, description="Enable Exotel adapter")
    account_sid: str = Field(default="", description="Exotel account SID")
    api_key: str = Field(default="", description="Exotel API key")
    api_token: str = Field(default="", description="Exotel API token")
    app_id: str = Field(default="", description="Exotel app ID")
    status_callback_url: Optional[HttpUrl] = None
    recording_callback_url: Optional[HttpUrl] = None

    model_config = SettingsConfigDict(env_prefix="EXOTEL_")


class PrivacySettings(BaseSettings):
    """Privacy and retention configuration."""

    recordings_enabled: bool = Field(
        default=False,
        description="Enable call recording storage"
    )
    transcript_retention_days: int = Field(
        default=90,
        ge=1,
        le=730,
        description="Days to retain transcripts"
    )
    analytics_retention_days: int = Field(
        default=365,
        ge=30,
        le=1825,
        description="Days to retain analytics"
    )
    require_consent: bool = Field(
        default=True,
        description="Require consent before recording"
    )

    model_config = SettingsConfigDict(env_prefix="")


class KnowledgeSettings(BaseSettings):
    """Knowledge base configuration."""

    embedding_batch_size: int = Field(default=100, ge=1, le=1000)
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=512)
    retrieval_top_k: int = Field(default=10, ge=1, le=50)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v

    model_config = SettingsConfigDict(env_prefix="")


class ConversationSettings(BaseSettings):
    """Conversation engine configuration."""

    language_detection_timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    asr_interim_results: bool = Field(default=True)
    tts_voice_en_in: str = Field(default="en-IN-NeerjaNeural")
    tts_voice_hi_in: str = Field(default="hi-IN-SwaraNeural")
    tts_voice_mr_in: str = Field(default="mr-IN-AarohiNeural")
    max_turn_duration_seconds: int = Field(default=300, ge=30, le=1800)

    model_config = SettingsConfigDict(env_prefix="")


class ObservabilitySettings(BaseSettings):
    """Observability and monitoring configuration."""

    otel_enabled: bool = Field(default=False, description="Enable OpenTelemetry")
    otel_exporter_otlp_endpoint: Optional[HttpUrl] = Field(
        default="http://localhost:4318",
        description="OTLP endpoint"
    )
    azure_monitor_connection_string: Optional[str] = Field(
        default=None,
        description="Azure Monitor connection string"
    )

    model_config = SettingsConfigDict(env_prefix="")


class APISettings(BaseSettings):
    """API service configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False, description="Enable auto-reload (dev only)")
    workers: int = Field(default=4, ge=1, le=32)

    model_config = SettingsConfigDict(env_prefix="API_")


class RealtimeSettings(BaseSettings):
    """Realtime gateway configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001, ge=1, le=65535)
    max_connections: int = Field(default=1000, ge=1, le=10000)
    session_timeout_seconds: int = Field(default=1800, ge=60, le=7200)

    model_config = SettingsConfigDict(env_prefix="REALTIME_")


class CORSSettings(BaseSettings):
    """CORS configuration."""

    origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    allow_credentials: bool = Field(default=True)

    model_config = SettingsConfigDict(env_prefix="CORS_")


class AuthSettings(BaseSettings):
    """Authentication configuration."""

    enabled: bool = Field(default=False, description="Enable authentication")
    azure_ad_tenant_id: str = Field(default="")
    azure_ad_client_id: str = Field(default="")
    azure_ad_client_secret: str = Field(default="")

    model_config = SettingsConfigDict(env_prefix="")


class Settings(BaseSettings):
    """Main application settings."""

    # Application
    app_name: str = Field(default="SVKM NMIMS Voice Assistant")
    app_version: str = Field(default="1.0.0")
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    debug: bool = Field(default=False)

    # Provider mode
    providers_mode: Literal["real", "fake"] = Field(
        default="real",
        description="Use real or fake providers"
    )

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    azure_speech: AzureSpeechSettings = Field(default_factory=AzureSpeechSettings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    azure_storage: AzureStorageSettings = Field(default_factory=AzureStorageSettings)
    azure_key_vault: AzureKeyVaultSettings = Field(default_factory=AzureKeyVaultSettings)
    exotel: ExotelSettings = Field(default_factory=ExotelSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    api: APISettings = Field(default_factory=APISettings)
    realtime: RealtimeSettings = Field(default_factory=RealtimeSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )

    def validate_required_credentials(self) -> list[str]:
        """
        Validate required credentials based on providers_mode.
        Returns list of missing credentials.
        """
        missing = []

        if self.providers_mode == "real":
            # Check Azure Speech
            if not self.azure_speech.key:
                missing.append("AZURE_SPEECH_KEY")

            # Check Azure OpenAI
            if not self.azure_openai.key:
                missing.append("AZURE_OPENAI_KEY")

            # Check Anthropic
            if not self.anthropic.api_key:
                missing.append("ANTHROPIC_API_KEY")

            # Check Azure Storage
            if not self.azure_storage.connection_string:
                missing.append("AZURE_STORAGE_CONNECTION_STRING")

            # Check Exotel if enabled
            if self.exotel.enabled:
                if not self.exotel.account_sid:
                    missing.append("EXOTEL_ACCOUNT_SID")
                if not self.exotel.api_key:
                    missing.append("EXOTEL_API_KEY")
                if not self.exotel.api_token:
                    missing.append("EXOTEL_API_TOKEN")

        return missing

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app_env == "production"

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.app_env == "development"


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings
