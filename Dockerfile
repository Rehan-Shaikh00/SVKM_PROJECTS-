# ---------------------------------------------------------------------------
# NMIMS Global University, Dhule — AI voice assistant
# Stage 1: build the React dashboard. Stage 2: run FastAPI + the built bundle.
#
# Note: this image was authored in an environment without a Docker daemon, so it
# has not been built here. `make docker-build` on a machine with Docker is the
# first validation step (see docs/DEPLOYMENT.md).
# ---------------------------------------------------------------------------

# ---- stage 1: dashboard --------------------------------------------------- #
FROM node:20-alpine AS dashboard

WORKDIR /build

# Install dependencies first so the layer is cached across source changes.
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci --no-audit --no-fund

COPY frontend/ ./frontend/
# vite.config.ts emits to ../backend/static/dashboard
RUN mkdir -p backend/static && cd frontend && npm run build


# ---- stage 2: runtime ----------------------------------------------------- #
FROM python:3.11-slim AS runtime

# espeak-ng gives the zero-key local TTS path a real voice; without it
# TTS_PROVIDER=local degrades to text-only. Harmless if you use a cloud TTS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/app ./backend/app
COPY backend/pytest.ini ./backend/pytest.ini
COPY backend/tests ./backend/tests
COPY data/kb ./data/kb
COPY scripts ./scripts
COPY .env.example ./

# Built dashboard from stage 1
COPY --from=dashboard /build/backend/static/dashboard ./backend/static/dashboard

# Runtime data (SQLite DB + vector index) lives on a volume so a redeploy does
# not throw away call history or the index.
RUN mkdir -p data/runtime data/index \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health/ready || exit 1

WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
