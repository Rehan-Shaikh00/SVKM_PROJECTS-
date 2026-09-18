"""FastAPI application entry point.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Startup does four things: create tables, load or seed the knowledge base,
build the retrieval index, and start the async call logger + sheet sync.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import analytics, assistant, calls, health, kb_admin, simulator, telephony
from .config import INSECURE_APP_SECRETS, settings
from .db import dispose_db, init_db
from .dependencies import (
    bootstrap_knowledge_base,
    get_call_logger,
    kb_bootstrap_info,
    provider_summary,
)
from .kb.ingest import schedule_sheet_sync
from .observability.logging import configure_logging, metrics
from .telephony.session_runner import active_call_count, end_all_sessions

logger = logging.getLogger("nims.main")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
DASHBOARD_DIR = STATIC_DIR / "dashboard"

START_TIME = time.time()


@asynccontextmanager
def _enforce_credential_hygiene() -> None:
    """Refuse to serve callers with credentials that are printed in the repository.

    Every failure this catches is silent in use: the dashboard logs in, calls are
    answered, transcripts look right, and the caller pseudonyms in the database
    look hashed. Nothing would ever complain -- not until somebody recomputed a
    phone number from one of them. So it is caught at boot or never.
    """
    problems = settings.credential_problems()
    if not problems:
        return
    if settings.environment != "development":
        listed = "\n".join(f"  {i}. {problem}" for i, problem in enumerate(problems, start=1))
        raise RuntimeError(
            f"refusing to start in {settings.environment}: insecure credentials\n"
            f"{listed}\n"
            "Generate fresh values into .env on the deployment host (never into the\n"
            "repository, and never into a chat message):\n"
            '  python3 -c "import secrets; print(secrets.token_urlsafe(48))"  # APP_SECRET\n'
            '  python3 -c "import secrets; print(secrets.token_urlsafe(18))"  # ADMIN_PASSWORD'
        )
    logger.warning(
        "insecure credentials, tolerated only because ENVIRONMENT=development: %s",
        " | ".join(problems),
    )
    if settings.app_secret in INSECURE_APP_SECRETS:
        # Keying caller pseudonyms with a published constant is worse than not
        # hashing at all, because it looks protected. A per-process secret cannot
        # be recomputed from the repository; the cost is that follow-up SMS dedupe
        # starts from scratch after a restart, which development can live with.
        settings.app_secret = secrets.token_urlsafe(48)
        logger.warning(
            "generated an ephemeral APP_SECRET for this process; set APP_SECRET in "
            ".env to keep caller pseudonyms stable across restarts"
        )


async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    _enforce_credential_hygiene()
    logger.info(
        "starting NMIMS Dhule voice assistant (env=%s, telephony=%s, asr=%s, tts=%s, llm=%s)",
        settings.environment,
        settings.telephony_provider,
        settings.asr_provider,
        settings.tts_provider,
        settings.llm_provider,
    )
    await init_db()

    try:
        info = await bootstrap_knowledge_base()
        logger.info(
            "knowledge base ready (%s): %s records / %s chunks",
            info.get("mode"),
            info.get("stats", {}).get("records"),
            info.get("stats", {}).get("chunks"),
        )
    except Exception as exc:  # pragma: no cover - startup must not die on KB issues
        logger.exception("knowledge base bootstrap failed: %s", exc)

    call_logger = get_call_logger()
    await call_logger.start()
    sync_task = schedule_sheet_sync()
    app.state.sync_task = sync_task
    app.state.started_at = time.time()

    degraded = [
        name for name, entry in settings.provider_status().items() if entry["degraded"]
    ]
    if degraded:
        logger.warning(
            "running with local fallbacks for: %s (set the matching API keys in .env "
            "to use the cloud providers)", ", ".join(degraded)
        )

    try:
        yield
    finally:
        logger.info("shutting down")
        if sync_task:
            sync_task.cancel()
        await end_all_sessions("shutdown")
        await call_logger.stop()
        await dispose_db()


app = FastAPI(
    title="NMIMS Global University, Dhule — AI Voice Assistant",
    description=(
        "Multilingual conversational AI for inbound admissions calls: spoken "
        "language identification, streaming ASR, grounded RAG answers, neural TTS, "
        "barge-in and human handoff."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else [settings.public_base_url],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


@app.middleware("http")
async def request_timing(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    if elapsed > 3000 and request.url.path.startswith("/assistant"):
        logger.warning("slow request %s took %.0f ms", request.url.path, elapsed)
    return response


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "path": request.url.path},
    )


# --- routes --------------------------------------------------------------- #
app.include_router(health.router)
app.include_router(simulator.router, prefix="/api")
app.include_router(assistant.router, prefix="/api")
app.include_router(kb_admin.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(calls.router, prefix="/api")
app.include_router(telephony.router)


@app.get("/api/service", tags=["config"])
async def service_index() -> dict[str, Any]:
    """Machine-readable service index (the browser gets the dashboard at `/`)."""
    return {
        "service": "NMIMS Global University, Dhule AI Voice Assistant",
        "version": "1.0.0",
        "status": "ok",
        "dashboard": "/dashboard/" if DASHBOARD_DIR.exists() else "not built (cd frontend && npm run build)",
        "docs": "/api/docs",
        "health": "/health",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "active_calls": active_call_count(),
    }


@app.get("/", include_in_schema=False)
async def root() -> Any:
    """Land on the dashboard when it is built; fall back to the JSON index."""
    if DASHBOARD_DIR.exists():
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/dashboard/")
    return await service_index()


@app.get("/api/config", tags=["config"])
async def public_config() -> dict[str, Any]:
    """Non-sensitive runtime configuration for the dashboard."""
    summary = provider_summary()
    return {
        "assistant_name": "Saarthi",
        "university": "NMIMS Global University, Dhule",
        # Fallback is the verified campus number from svkmnmimsgu.ac.in/contact-us;
        # the university publishes no toll-free line of its own.
        "helpline": settings.twilio_helpline_number,
        "environment": settings.environment,
        "academic_year": settings.kb_academic_year,
        "languages": settings.supported_language_list,
        "greeting_languages": settings.greeting_language_list,
        "kb_bootstrap": {k: v for k, v in kb_bootstrap_info().items() if k != "ingest"},
        "providers": summary,
        "guardrails": {
            "pii_redaction": settings.redact_pii,
            "recording_consent": settings.recording_consent_announce,
            "call_recording": settings.call_recording_enabled,
            "dtmf_fallback": settings.allow_dtmf_fallback,
            "max_call_minutes": settings.max_call_minutes,
        },
    }


@app.get("/api/status", tags=["config"])
async def status() -> dict[str, Any]:
    return {
        "metrics": metrics.snapshot(),
        "active_calls": active_call_count(),
        "providers": provider_summary(),
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }


# --- dashboard ------------------------------------------------------------ #
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

    @app.get("/dashboard-redirect", include_in_schema=False)
    async def dashboard_redirect():  # pragma: no cover
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/dashboard/")
else:  # pragma: no cover - dev convenience
    logger.warning(
        "dashboard build not found at %s — run `cd frontend && npm install && npm run build`",
        DASHBOARD_DIR,
    )


async def _background_tasks() -> None:  # pragma: no cover - placeholder hook
    await asyncio.sleep(0)
