"""Central, environment-driven configuration.

Every provider selection is resolved here, together with an automatic
"degraded mode" decision: if a provider is configured but its credentials are
missing, we fall back to the local implementation instead of crashing. That is
what lets the whole system boot and run a complete call with zero API keys.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> repo root
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
REPO_ROOT = BACKEND_DIR.parent

# The university is in Dhule, Maharashtra. Every caller-facing notion of "today" —
# admission windows, application deadlines, result dates — is India Standard
# Time, not the server's local zone and not UTC (which is a day behind after
# 19:30 IST and would answer "is the deadline passed?" incorrectly).
IST = timezone(timedelta(hours=5, minutes=30))


def ist_today() -> date:
    """Today's date in India Standard Time."""
    return datetime.now(IST).date()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- core --------------------------------------------------------------
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    public_base_url: str = "http://localhost:8000"
    app_secret: str = "change-me-in-production"

    # ---- database ----------------------------------------------------------
    database_url: str = f"sqlite:///{REPO_ROOT}/data/runtime/nims_voice.db"

    # ---- telephony ---------------------------------------------------------
    telephony_provider: Literal["twilio", "exotel", "plivo", "simulator"] = "twilio"
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_api_key_sid: str = ""
    twilio_api_key_secret: str = ""
    # The outbound number and the escalation targets default to the phone numbers
    # the university publishes on its own contact page (svkmnmimsgu.ac.in
    # /contact-us): STME 02562 350620, School of Commerce 02562 350600 and SPTM
    # 02562 350640. Placeholder numbers here meant a real caller asking for a
    # human was transferred to a line that does not exist.
    twilio_helpline_number: str = Field(
        default="+912562350620",
        validation_alias=AliasChoices("twilio_helpline_number", "twilio_helline_number",
                                      "TWILIO_HELPLINE_NUMBER", "TWILIO_HELLINE_NUMBER"),
    )
    escalation_agents: str = "+912562350620,+912562350600,+912562350640"
    escalation_queue_name: str = "svkm-nmims-dhule-admissions-queue"
    escalation_max_wait_seconds: int = 300
    escalation_whisper_context: bool = True
    hold_music_url: str = "https://api.twilio.com/cowbell.mp3"

    # ---- language identification -------------------------------------------
    lid_provider: Literal["azure", "google", "deepgram", "local"] = "local"
    supported_languages: str = "en-IN,hi-IN,mr-IN,gu-IN,ta-IN,bn-IN,te-IN,kn-IN,ml-IN,pa-IN,ur-IN,raj-IN"
    # English, Hindi and Marathi are spoken in the opening prompt: the campus is
    # in Dhule, Maharashtra, where callers arrive in Marathi, Hindi, Ahirani or
    # Gujarati. The remaining codes stay *supported* — if a caller names one, we
    # switch to it — but they are not announced, and Rajasthani in particular is
    # no longer a greeting language for this campus.
    greeting_languages: str = "en-IN,hi-IN,mr-IN"
    lid_confidence_threshold: float = 0.62
    allow_dtmf_fallback: bool = True

    # ---- ASR ---------------------------------------------------------------
    asr_provider: Literal[
        "deepgram", "assemblyai", "google", "azure", "whisper", "client", "none"
    ] = "client"
    deepgram_api_key: str = ""
    assemblyai_api_key: str = ""
    google_application_credentials: str = ""
    azure_speech_key: str = ""
    azure_speech_region: str = ""
    openai_api_key: str = ""
    whisper_base_url: str = "https://api.openai.com/v1"
    whisper_model: str = "gpt-4o-mini-transcribe"
    asr_interim_results: bool = True
    asr_endpoint_silence_ms: int = 650

    # ---- TTS ---------------------------------------------------------------
    tts_provider: Literal[
        "google", "azure", "elevenlabs", "local", "client", "none"
    ] = "client"
    google_tts_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "eleven_multilingual_v2"
    #: premade voice used for every language (eleven_multilingual_v2 is language agnostic)
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    tts_speaking_rate: float = 1.0
    tts_sample_rate: int = 8000
    #: split long answers into sentences so the first audio leaves in <500 ms
    tts_sentence_streaming: bool = True
    #: Upper bound on how long the session waits for a browser/client TTS to
    #: report that it finished speaking. A stuck client must never stall a call.
    client_speech_max_wait: float = 8.0
    tts_max_chunk_chars: int = 180

    # ---- LLM ---------------------------------------------------------------
    llm_provider: Literal["anthropic", "openai", "local"] = "local"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    llm_max_tokens: int = 400
    llm_temperature: float = 0.2
    llm_streaming: bool = True
    answer_confidence_threshold: float = 0.55

    # ---- RAG ---------------------------------------------------------------
    embedding_provider: Literal["openai", "cohere", "local"] = "local"
    openai_embedding_model: str = "text-embedding-3-small"
    cohere_api_key: str = ""
    vector_store: Literal["local", "pgvector", "qdrant"] = "local"
    vector_dimensions: int = 768
    qdrant_url: str = "http://localhost:6333"
    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.22
    rag_citations_required: bool = True

    # ---- knowledge base ----------------------------------------------------
    kb_seed_dir: str = "data/kb"
    kb_auto_ingest: bool = True
    kb_google_sheet_csv_url: str = ""
    kb_sync_interval_minutes: int = 60
    kb_staleness_days: int = 180
    #: admission cycle the assistant should prefer when several years exist in the KB
    kb_academic_year: str = "2026-27"

    # ---- compliance / guardrails -------------------------------------------
    recording_consent_announce: bool = True
    call_recording_enabled: bool = False
    allow_call_recording_storage: bool = False
    redact_pii: bool = True
    max_call_minutes: int = 15
    silence_max_reprompts: int = 2

    # ---- follow-up channel -------------------------------------------------
    followup_sms_enabled: bool = False
    followup_whatsapp_enabled: bool = False
    followup_email_enabled: bool = False
    sendgrid_api_key: str = ""
    #: Sender identity for outbound follow-up email. The university website
    #: publishes no email address, so this must be an operator-owned, verified
    #: sender domain — it is never spoken to a caller or printed as a university
    #: contact. Point it at the admissions team's own mailbox in production.
    followup_from_email: str = "admissions@svkmnmimsgu.ac.in"

    # ---- admin -------------------------------------------------------------
    admin_username: str = "nims-admin"
    admin_password: str = "change-me"
    admin_auth_enabled: bool = True

    # ---- observability -----------------------------------------------------
    sentry_dsn: str = ""
    metrics_enabled: bool = True

    # ---------------------------------------------------------------- helpers
    @field_validator("public_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def kb_seed_path(self) -> Path:
        p = Path(self.kb_seed_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def runtime_dir(self) -> Path:
        p = REPO_ROOT / "data" / "runtime"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def index_dir(self) -> Path:
        p = REPO_ROOT / "data" / "index"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def websocket_base_url(self) -> str:
        return self.public_base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )

    @property
    def supported_language_list(self) -> list[str]:
        return [c.strip() for c in self.supported_languages.split(",") if c.strip()]

    @property
    def greeting_language_list(self) -> list[str]:
        codes = [c.strip() for c in self.greeting_languages.split(",") if c.strip()]
        return [c for c in codes if c in self.supported_language_list] or codes

    @property
    def escalation_agent_list(self) -> list[str]:
        return [a.strip() for a in self.escalation_agents.split(",") if a.strip()]

    @property
    def twilio_credentials(self) -> tuple[str, str]:
        """API key pair preferred over the master auth token."""
        if self.twilio_api_key_sid and self.twilio_api_key_secret:
            return self.twilio_api_key_sid, self.twilio_api_key_secret
        return self.twilio_account_sid, self.twilio_auth_token

    @property
    def twilio_configured(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def provider_status(self) -> dict[str, dict[str, object]]:
        """Diagnostics for the dashboard / `/health` endpoint."""

        def entry(configured: bool, provider: str, fallback: str) -> dict[str, object]:
            return {
                "provider": provider,
                "configured": configured,
                "active": provider if configured else fallback,
                "degraded": not configured and provider != fallback,
            }

        return {
            "asr": entry(
                self.asr_provider
                in ("deepgram", "assemblyai", "google", "azure", "whisper")
                and bool(
                    self.deepgram_api_key
                    or self.assemblyai_api_key
                    or self.google_application_credentials
                    or self.azure_speech_key
                    or self.openai_api_key
                ),
                self.asr_provider,
                "client",
            ),
            "lid": entry(
                self.lid_provider in ("azure", "google", "deepgram")
                and bool(
                    self.azure_speech_key
                    or self.google_application_credentials
                    or self.deepgram_api_key
                ),
                self.lid_provider,
                "local",
            ),
            "tts": entry(
                self.tts_provider in ("google", "azure", "elevenlabs")
                and bool(
                    self.google_tts_api_key
                    or self.azure_speech_key
                    or self.elevenlabs_api_key
                ),
                self.tts_provider,
                "client",
            ),
            "llm": entry(
                self.llm_provider in ("anthropic", "openai")
                and bool(self.anthropic_api_key or self.openai_api_key),
                self.llm_provider,
                "local",
            ),
            "embeddings": entry(
                self.embedding_provider in ("openai", "cohere")
                and bool(self.openai_api_key or self.cohere_api_key),
                self.embedding_provider,
                "local",
            ),
            "telephony": entry(
                self.twilio_configured, self.telephony_provider, "simulator"
            ),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # Make sure the sqlite directory exists before the engine is created.
    if settings.database_url.startswith("sqlite"):
        db_path = settings.database_url.split("///")[-1]
        if not Path(db_path).is_absolute():
            db_path = REPO_ROOT / db_path
        os.makedirs(Path(db_path).parent, exist_ok=True)
    return settings


settings = get_settings()
