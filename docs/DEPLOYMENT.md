# Deployment

## What runs in production

One container (or one VM) serving:

- FastAPI API + WebSocket call sessions on **:8000**
- The built React dashboard at `/` → `/dashboard/`
- Telephony webhooks under `/telephony/...`
- OpenAPI docs at `/api/docs`

Plus a database. SQLite is fine for a single instance and low volume; use Postgres
for anything real.

```
                    PSTN
                     │
        Twilio / Exotel / Plivo
                     │  wss://voice.example.org/telephony/twilio/media-stream
                     ▼
        ┌────────────────────────┐        ┌──────────────────┐
        │  TLS proxy (nginx/ALB) │───────▶│  app (uvicorn)   │
        └────────────────────────┘        │  FastAPI + WS    │
                     ▲                    └───────┬──────────┘
        dashboard/API│(https)                     │
                     │                    ┌───────▼──────────┐
                   users                  │ Postgres+pgvector │
                                          └──────────────────┘
```

---

## 1. Docker

```bash
cp .env.example .env          # edit secrets + PUBLIC_BASE_URL
make docker-build             # or: docker build -t nims-voice-assistant .
make docker-up                # app + pgvector/pgvector:pg16
curl -fsS http://localhost:8000/health/ready
```

`docker-compose.yml` sets `DATABASE_URL`, `VECTOR_STORE=pgvector`,
`PGVECTOR_ENABLED=true`, `ENVIRONMENT=production` and `ADMIN_AUTH_ENABLED=true`
for the container network; everything else (provider keys, languages, guardrails)
comes from `.env` via `env_file`.

The image is multi-stage: Node 20 builds the dashboard, then `python:3.11-slim`
runs it with `espeak-ng` installed so the zero-key local TTS path has a real
voice. It runs as a non-root user, declares a volume at `/app/data` (SQLite +
vector index) and has a `HEALTHCHECK` against `/health/ready`.

> **Validation note.** The Dockerfile and compose file were authored in a sandbox
> with no Docker daemon, so they have not been built here. `make docker-build` is
> the first thing to run on a machine with Docker; the pinned base images and the
> dependency list are the only moving parts.

Single container without compose:

```bash
docker run -d --name nims-voice \
  -p 8000:8000 --env-file .env \
  -v nims-data:/app/data \
  nims-voice-assistant:latest
```

---

## 2. Bare metal / VM

```bash
sudo apt-get install -y python3.11 python3.11-venv nodejs npm espeak-ng
git clone <repo> && cd SVKM_PROJECTS-
make setup                     # venv + backend deps + dashboard build
cp .env.example .env           # then edit
make run                       # foreground; use systemd in production
```

systemd unit (`/etc/systemd/system/nims-voice.service`):

```ini
[Unit]
Description=NMIMS Global University, Dhule — AI Voice Assistant
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User=nims
WorkingDirectory=/opt/nims-voice/backend
EnvironmentFile=/opt/nims-voice/.env
ExecStart=/opt/nims-voice/.venv/bin/python -m uvicorn app.main:app \
          --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

**Run one worker per port, not `--workers N`.** A live call is a stateful
WebSocket session held in process memory; spreading workers behind one port means
a caller's audio can land on a different worker than their session. Scale by
running several instances behind the proxy with **sticky sessions by call**, each
sharing Postgres + pgvector (see §6).

---

## 3. TLS and public reachability

Twilio Media Streams requires a **publicly reachable `wss://` endpoint** with a
valid certificate. Self-signed will fail.

nginx:

```nginx
server {
    listen 443 ssl http2;
    server_name voice.example.org;

    ssl_certificate     /etc/letsencrypt/live/voice.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voice.example.org/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;      # WebSocket
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;                    # calls are long-lived
        proxy_send_timeout 3600s;
        proxy_buffering off;                         # latency matters
    }
}
```

Then set `PUBLIC_BASE_URL=https://voice.example.org` — it is used to build the
webhook/WebSocket URLs handed to the telephony provider.

For local development with a real provider, tunnel it:
`cloudflared tunnel --url http://localhost:8000` or
`ngrok http 8000`, then use the printed `https://` host as `PUBLIC_BASE_URL`.

---

## 4. Database

**SQLite (default).** Zero setup: `DATABASE_URL=sqlite:///data/runtime/nims_voice.db`.
Good for a pilot or a single-instance deployment. Keep `data/` on a real volume
and back it up.

**Postgres (recommended).**

```bash
docker run -d --name nims-db -e POSTGRES_USER=nims -e POSTGRES_PASSWORD=<pw> \
  -e POSTGRES_DB=nims_voice -p 5432:5432 -v pg-data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

```ini
DATABASE_URL=postgresql://nims:<pw>@db-host:5432/nims_voice
VECTOR_STORE=pgvector
PGVECTOR_ENABLED=true
```

Any `postgresql://` URL works — `app/db.py` normalises it to
`postgresql+asyncpg://`, so **asyncpg** is the driver that must be installed (it
is in `requirements.txt`; psycopg is not used). Tables are created on boot
(`init_db`), and `PgVectorStore` runs `CREATE EXTENSION IF NOT EXISTS vector`
itself — the DB role therefore needs `CREATE` on the database, or pre-create the
extension as a superuser:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Set `VECTOR_DIMENSIONS` to match your embedding model (1536 for OpenAI
`text-embedding-3-small`, 1024 for Cohere `embed-multilingual-v3.0`). Changing
dimensions later requires re-embedding: the local store detects the mismatch and
rebuilds; with pgvector, reindex via `POST /api/kb/reindex`.

---

## 5. Provider wiring

The system boots with **no keys at all** and logs every degradation. Add keys
incrementally — each one upgrades a single stage.

### Language ID
```ini
LID_PROVIDER=local            # lexical + script detection (default, no key)
# LID_PROVIDER=google         # GOOGLE_APPLICATION_CREDENTIALS
# LID_PROVIDER=azure          # AZURE_SPEECH_KEY + AZURE_SPEECH_REGION
SUPPORTED_LANGUAGES=en-IN,hi-IN,mr-IN,gu-IN,ta-IN,bn-IN,te-IN,kn-IN,ml-IN,pa-IN,ur-IN,raj-IN
GREETING_LANGUAGES=en-IN,hi-IN,mr-IN
LID_CONFIDENCE_THRESHOLD=0.45
ALLOW_DTMF_FALLBACK=false
```

### ASR (streaming)
```ini
ASR_PROVIDER=deepgram
DEEPGRAM_API_KEY=...
ASR_INTERIM_RESULTS=true
ASR_ENDPOINT_SILENCE_MS=450    # lower = faster turn-taking, higher = fewer cut-offs
```
Alternatives: `google`, `azure`, `assemblyai`, `openai` (Whisper chunks), or
`client` (browser does the ASR — what the simulator uses).

### TTS
```ini
TTS_PROVIDER=google           # or azure | elevenlabs | local | client
GOOGLE_TTS_API_KEY=...
ELEVENLABS_API_KEY=...        # most natural Indic voices
TTS_SPEAKING_RATE=1.0
TTS_SAMPLE_RATE=8000          # 8000 for telephony, 24000 for the browser
```
`local` needs `espeak-ng` or `piper` on the host (installed in the Docker image).

### LLM (reasoning)
```ini
LLM_PROVIDER=anthropic        # Claude — preferred
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-5
LLM_STREAMING=true            # sentence-level streaming into TTS
ANSWER_CONFIDENCE_THRESHOLD=0.5
```
Or `LLM_PROVIDER=openai` with `OPENAI_API_KEY`. Without a key the assistant uses
the deterministic template path — correct and grounded, but less natural in
non-English languages.

### Embeddings
```ini
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DIMENSIONS=1536
```
`local` (default) is a lexical hashing embedder: it works, boosted by synonym and
cross-lingual expansion plus BM25/RRF fusion, but semantic recall improves
markedly with a real embedding model.

### Telephony
```ini
TELEPHONY_PROVIDER=twilio
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_API_KEY_SID=...
TWILIO_API_KEY_SECRET=...
# Outbound caller ID and SMS sender. The university publishes three school
# offices on its contact page; no toll-free number is published, so do not
# invent one. STME 02562 350620 / Commerce 02562 350600 / SPTM 02562 350640.
TWILIO_HELPLINE_NUMBER=+912562350620
ESCALATION_AGENTS=+912562350620,+912562350600,+912562350640
ESCALATION_QUEUE_NAME=svkm-nmims-dhule-admissions
ESCALATION_MAX_WAIT_SECONDS=120
HOLD_MUSIC_URL=https://.../hold.mp3
ESCALATION_WHISPER_CONTEXT=true
```
In the Twilio console, point the number's **Voice** webhook at
`POST https://voice.example.org/telephony/twilio/voice` and the **Status**
callback at `/telephony/twilio/status`. Exotel and Plivo equivalents live under
`/telephony/exotel/*` and `/telephony/plivo/*`.

### Follow-up channel
```ini
FOLLOWUP_SMS_ENABLED=true
FOLLOWUP_WHATSAPP_ENABLED=false
FOLLOWUP_EMAIL_ENABLED=true
SENDGRID_API_KEY=...
# Sender identity for outbound follow-up email. The university website publishes
# NO email address, so this must be an operator-owned verified sender - it is
# never spoken to a caller nor printed as a university contact.
FOLLOWUP_FROM_EMAIL=admissions@svkmnmimsgu.ac.in
```

### Admin dashboard
```ini
ADMIN_AUTH_ENABLED=true       # MUST be true in production
ADMIN_USERNAME=<unique>
ADMIN_PASSWORD=<strong>
```
KB write endpoints, analytics and call QA sit behind `require_admin`. With auth
disabled they are open — acceptable only on a laptop.

---

## 6. Scaling and operations

- **Concurrency.** Each active call is one WebSocket plus a session object; the
  blocking work (embeddings, vector IO, TTS HTTP) is offloaded. A single instance
  handles a modest number of concurrent calls; scale horizontally with sticky
  routing per call and a shared Postgres/pgvector backend.
- **Call logging never blocks a call.** Events go onto a bounded in-process queue
  (500) drained by one background worker; if it fills, events are dropped and
  counted (`CallLogger.stats()` → `/metrics`) rather than applying back-pressure
  to the voice loop. Watch the drop count.
- **Metrics.** `GET /metrics` and `GET /api/status` expose counters and provider
  health; logs are structured JSON with correlation ids, ready for any collector.
  `SENTRY_DSN` wires error reporting.
- **Health.** `/health` (liveness), `/health/ready` (KB records/chunks/problems —
  use this for readiness gates), `/health/providers` (which providers degraded and
  why).
- **Backups.** Postgres dump + `data/kb` in git. `GET /api/kb/export.csv` gives a
  human-readable KB snapshot; `kb_revisions` records every content edit.
- **Zero-downtime content updates.** KB edits re-chunk and re-embed in place — no
  restart, no redeploy. Only code changes need a rollout.

---

## 7. Go-live checklist

- [ ] `PUBLIC_BASE_URL` is `https://` and the media-stream URL is `wss://` with a
      valid public certificate.
- [ ] `ENVIRONMENT=production`, `APP_SECRET` set to a fresh random value.
- [ ] `ADMIN_AUTH_ENABLED=true` with strong, unique credentials.
- [ ] Postgres reachable; `CREATE EXTENSION vector` present when using pgvector.
- [ ] `VECTOR_DIMENSIONS` matches the embedding model in use.
- [ ] LLM key set (`LLM_PROVIDER=anthropic`) — otherwise non-English answers fall
      back to templates with English fragments.
- [ ] Cloud ASR + TTS keys set and tested in the target languages.
- [ ] Provider webhook signatures verified; endpoints not publicly writable
      without them.
- [ ] All `verified: false` KB rows corrected, sourced and verified.
- [ ] `RECORDING_CONSENT_ANNOUNCE=true`; recording/storage flags match policy.
- [ ] `REDACT_PII=true`; confirmed with a test call reading out a number/email.
- [ ] Escalation agents reachable, queue configured, whisper context on.
- [ ] A real end-to-end call placed on the production number in each enabled
      language, transcript + analytics row present afterwards.
- [ ] Retention schedule and grievance process documented
      (see `docs/COMPLIANCE.md`).
- [ ] `make test` and `make lint` green; `/health/ready` reports no problems.
