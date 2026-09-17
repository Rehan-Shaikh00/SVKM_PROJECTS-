# SVKM's NMIMS Global University, Dhule — AI Voice Assistant for the Admissions Helpline

A conversational **AI voice assistant for inbound phone calls** to SVKM's NMIMS
Global University, Dhule (Maharashtra). Callers speak naturally — **no keypad
menus** — say their preferred language, and then ask about courses, admissions,
fees, eligibility, hostels, scholarships, placements and campus facilities.
Answers come from a **retrieval-augmented (RAG) knowledge base** that admissions
staff can edit without a redeploy. Anything the assistant cannot verify is handed
to a human with a spoken context brief.

Built for the Indian telephony reality: Marathi, Hindi and English on the opening
prompt with ten more codes recognised if a caller names one, code-mixed speech
("hostel mandatory hai kya", "शुल्क किती आहे"), Indian number formats
(lakh/crore, ₹1,40,000), sub-2-second perceived latency, and barge-in.

---

## Try it in 60 seconds (no API keys needed)

Every provider has a **zero-key local fallback**, so the whole system boots and
runs a complete call with no credentials at all.

```bash
# 1. backend
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt

# 2. dashboard (React/Vite -> backend/static/dashboard)
cd frontend && npm install && npm run build && cd ..

# 3. config + run
cp .env.example .env
cd backend && ../.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/** → redirects to the dashboard.

| What to open | Where |
|---|---|
| **Phone simulator** (make a real call in the browser) | `/dashboard/` → *Phone* tab |
| Analytics (call volume, languages, latency, escalations) | *Analytics* tab |
| Knowledge base editor (for non-technical staff) | *Knowledge base* tab |
| Call QA (full transcripts, hand-off briefs, review) | *Calls* tab |
| Interactive API docs | `/api/docs` |
| Live WebSocket call protocol | `/api/simulator/ws` |

The browser phone uses the **Web Speech API** for microphone input and speech
synthesis, and falls back to typing. It speaks to the same WebSocket session
engine that Twilio Media Streams uses, so what you rehearse in the browser is
what a real caller gets.

A text-only end-to-end run (no browser, no microphone):

```bash
.venv/bin/python scripts/smoke_ws.py
```

That script starts a call, answers the trilingual language prompt with "Hindi",
asks about fees/admission/hostel/documents, receives a follow-up offer, asks for
a human, and prints every server event through to `call_ended`.

---

## What the system does

### 1. Telephony integration
- **Twilio Media Streams** (bidirectional 8 kHz μ-law over WebSocket) — the
  reference implementation, with `<Connect>`/`<Stream>` TwiML generation.
- **Exotel** and **Plivo** webhook adapters (voice + conversation + events).
- **Simulator** provider for development and the browser phone.
- Toll-free/local number support, **queuing with hold music**, and
  **human-transfer fallback** via TwiML `<Dial>`/`<Queue>` with an agent
  *whisper* (a spoken summary played to the agent before bridging).
- Status callbacks keep call records accurate for analytics; recording URLs are
  dropped unless storage is explicitly enabled.

### 2. Spoken language identification (no DTMF)
The greeting asks once, trilingually:

> "Thank you for calling NMIMS Global University, Dhule. You are speaking with
> Saarthi, an AI assistant. For quality and training, this call may be recorded…"
> "Please tell me your preferred language. कृपया अपनी भाषा बताइए। कृपया आपली भाषा सांगा."

The caller answers **in their own language**; a layered detector (explicit
language names → script detection → lexicon → optional acoustic LID) picks it up
and switches **ASR + LLM + TTS locale together**. Confidence, method and attempt
count are logged per call. Marathi is detected both in Devanagari (`आहे`, `पाहिजे`,
`किती`) and **romanised** (`mala mahiti pahije`, `shulka kiti aahe`) — the second
case matters because a lot of Indian speech arrives as Latin-script transcript.
Devanagari is disambiguated three ways between Hindi, Marathi and Rajasthani.

- 14 languages in the registry; 12 enabled by default
  (`en, hi, mr, gu, ta, bn, te, kn, ml, pa, ur, raj`), with **`en, hi, mr`**
  announced in the greeting.
- Mid-call language switching is supported (two consecutive detections adopt the
  new language rather than flapping on one ambiguous utterance).
- Keypad is available only as a documented **last resort**
  (`ALLOW_DTMF_FALLBACK`, off by default) when speech cannot be understood.

### 3. Real-time streaming ASR
Adapters for **Deepgram, Google Cloud Speech, Azure Speech, AssemblyAI** and a
browser/client ASR path, all behind one `StreamingASR` interface with interim
results, endpointing silence tuning and per-turn latency capture.

### 4. LLM reasoning + RAG
- **Claude (Anthropic)** is the preferred reasoning layer; an OpenAI-compatible
  adapter and a deterministic local generator are also wired in.
- **Hybrid retrieval**: BM25 lexical + dense vectors fused with Reciprocal Rank
  Fusion (k=60), then reranked on category/language match, `verified` status,
  staleness and course-token overlap.
- **Vector stores**: local (numpy + JSONL, zero setup), **pgvector**, **Qdrant**.
- **Embeddings**: OpenAI/Cohere when keyed, otherwise a lexical hashing embedder
  with synonym + cross-lingual query expansion.
- Multi-turn context, tool loop, and a **grounding gate**: if the answer is not
  supported by retrieved chunks — or contains a number that does not appear in
  the context — the assistant refuses and escalates rather than inventing a fee.
- Answers are **short and voice-friendly** (≤ ~320 spoken characters); when the
  real answer is long, it offers **SMS / WhatsApp / email** follow-up instead of
  reading a list aloud.

### 5. Natural multilingual TTS
**Google, Azure, ElevenLabs** (REST) plus local (`espeak-ng`/`piper`) and browser
synthesis. Text is scrubbed for speech first: markdown/bullets/URLs removed,
`₹` → "rupees", numbers rendered the way Indians say them
("one lakh fifty thousand rupees per year"), and long sentences split at clause
boundaries **without deleting conjunctions**.

### 6. Knowledge base = single source of truth
`data/kb/*.yaml` (or CSV/JSON/Google Sheets) seeds the Dhule knowledge base —
nine topic files covering the university, engineering, pharmacy, commerce and
management, admissions, the academic calendar, fees and scholarships, campus
life, and placements/FAQs. 54 records, all sourced from the university's own
website, including the AY 2026-27 academic calendar transcribed from the PDF
signed by the Vice-Chancellor.
Staff update it through the dashboard: create/edit/delete records,
mark them **verified**, bulk-verify, import a CSV (with dry-run preview), sync a
Google Sheet, preview chunking, reindex, export. **No code change, no redeploy**
— content-hash deduplication means re-ingesting unchanged data is a no-op.

### 7. Escalation + full transcripts
Triggers: caller asks for a human, sensitive/legal/distress topics, ungrounded or
low-confidence answers, repeated silence, unsupported language, or a call-length
cap. Every call is logged with per-turn timings (ASR/retrieval/LLM/TTS),
citations, grounding status and confidence. The hand-off brief contains a
summary, a spoken whisper for the agent, unresolved points and the transcript
tail. QA staff can review and score calls from the dashboard.

### 8. Analytics dashboard
Call volume, resolution vs escalation, language distribution, top queries
(grouped by canonical form), unanswered-question backlog with inline resolve,
per-stage latency percentiles (p50/p95), intent mix, escalation reasons/outcomes
and answer-quality metrics (grounding rate, template-fallback rate, confidence).

### Compliance
Recording consent is announced before any caller speech is captured, PII is
redacted and hashed before it reaches logs, recording storage is off by default,
and retention/deletion controls exist. See **[docs/COMPLIANCE.md](docs/COMPLIANCE.md)**
for TRAI and DPDP Act 2023 mapping.

---

## Repository layout

```
backend/
  app/
    ai/            intents, prompts, LLM adapters, tools, guardrails, RAG engine,
                   spoken-number rendering, answer templates, summariser
    api/           health, assistant, simulator (WS), telephony webhooks,
                   kb_admin, analytics, calls
    i18n/          14-language registry, scripted prompts, ASR/TTS locale mapping
    kb/            chunking, embeddings, vector stores (local/pgvector/qdrant),
                   hybrid retriever, repository, ingest (CSV/YAML/JSON/Sheets)
    orchestrator/  CallSession state machine, channel abstraction, call logger,
                   escalation, follow-up delivery
    telephony/     TwiML builders, Twilio media stream, Exotel, Plivo
    voice/         audio (G.711 μ-law, resample, VAD, DTMF), ASR, LID, TTS
    config.py      environment-driven settings + degraded-mode resolution
    models.py      SQLAlchemy 2.0 models (calls, turns, escalations, KB, …)
  tests/           244 unit tests (no DB, no network, no keys required)
  static/dashboard built React app served at /dashboard
data/kb/           seed knowledge base (YAML)
frontend/          Vite + React + TypeScript dashboard
scripts/smoke_ws.py end-to-end WebSocket call driver
docs/              architecture, compliance, deployment, KB staff guide
```

---

## Configuration

Copy `.env.example` → `.env`. Every block degrades gracefully: if a provider is
named but its credentials are missing, the system logs the degradation and falls
back to a local implementation instead of crashing.

| Concern | Env var | Zero-key default | Production choice |
|---|---|---|---|
| Telephony | `TELEPHONY_PROVIDER` | `simulator` | `twilio` / `exotel` / `plivo` |
| Language ID | `LID_PROVIDER` | `local` (lexical+script) | `google` / `azure` / `deepgram` |
| ASR | `ASR_PROVIDER` | `client` (browser) | `deepgram` / `google` / `azure` / `assemblyai` |
| TTS | `TTS_PROVIDER` | `client` / `local` | `google` / `azure` / `elevenlabs` |
| LLM | `LLM_PROVIDER` | `local` (template path) | `anthropic` (Claude) / `openai` |
| Embeddings | `EMBEDDING_PROVIDER` | `local` (hashing) | `openai` / `cohere` |
| Vector store | `VECTOR_STORE` | `local` (numpy+JSONL) | `pgvector` / `qdrant` |
| Database | `DATABASE_URL` | SQLite file | Postgres |
| Admin auth | `ADMIN_AUTH_ENABLED` | `false` | `true` (+ `ADMIN_USERNAME/PASSWORD`) |

Other important settings: `PUBLIC_BASE_URL` (telephony websockets **must** be
`wss://` and publicly reachable), `SUPPORTED_LANGUAGES`, `GREETING_LANGUAGES`,
`RECORDING_CONSENT_ANNOUNCE`, `REDACT_PII`, `MAX_CALL_MINUTES`,
`ANSWER_CONFIDENCE_THRESHOLD`, `RETRIEVAL_TOP_K`, `KB_STALENESS_DAYS`,
`KB_GOOGLE_SHEET_CSV_URL`, escalation targets and hold music.

Escalation defaults to the three school offices the university publishes on its own
contact page — `TWILIO_HELPLINE_NUMBER=+912562350620` (STME, 02562 350620) and
`ESCALATION_AGENTS=+912562350620,+912562350600,+912562350640` (STME, School of
Commerce, SPTM). No toll-free number is published anywhere on the site, so none is
assumed: a caller who asks for a human is transferred to a line that exists. The
older misspelt `TWILIO_HELLINE_NUMBER` is still accepted as an alias.

---

## Development

```bash
# backend tests (298 tests, ~7s, no DB/network/keys)
cd backend && ../.venv/bin/pytest -q

# lint (ruff config in .ruff.toml; ignores are documented with reasons)
cd backend && ../.venv/bin/ruff check app/

# dashboard dev server with API proxy to :8000
cd frontend && npm run dev

# rebuild the dashboard into backend/static/dashboard
cd frontend && npm run build

# verify what a caller actually hears, end to end, against a running server
# (57 questions in English, Hindi and Marathi; exits non-zero on any failure)
.venv/bin/python scripts/verify_answers.py
```

The unit tests pin the composing rules against hand-built retrieval results.
`scripts/verify_answers.py` pins the whole path — retrieval, ranking, intent,
compose, guardrails — against a live server, so a knowledge-base edit or a
ranking tweak that changes what a caller hears is caught before it ships. Every
case in it was a real bad answer at some point, and each one carries a note
saying why it matters.

`make` targets wrap these — see the `Makefile`.

---

## Known limitations (read before production)

1. **Zero-key answer quality.** Without an LLM key the assistant answers through
   deterministic templates over retrieved KB rows. 51 sentence frames exist per
   language for English, Hindi, Marathi and Rajasthani, and money, duration and
   seat counts are rendered natively in all four, so the *shape* of an answer is
   in the caller's language. What is not translated is the KB payload itself: the
   seeded `structured` fields are English, so a Marathi caller asking about a
   specific eligibility clause or a round status hears a native frame around an
   English fact ("...saathi, schedules and merit lists published for Round I and
   Round II"). Anything the university publishes nothing about — fees, hostel,
   refunds, loans — has its own native frame in all four languages, so those never
   degrade. Set `LLM_PROVIDER=anthropic` (or `openai`) for production-quality
   multilingual phrasing of the payloads too.
2. **Local embeddings are lexical, not semantic.** Retrieval still works well
   because of synonym + cross-lingual expansion and BM25/RRF fusion, but
   `EMBEDDING_PROVIDER=openai` is a one-line upgrade to true semantic recall.
3. **Seeded fee and date rows are marked `verified: false`.** Public sources
   (aggregator sites) disagree heavily on fees for these programmes, so every
   fee/date record ships flagged *"SAMPLE — verify before production"* with its
   `source`. Where no figure could be sourced at all — engineering, pharmacy,
   BCA/MCA, MBA, M.Com and Ph.D. — the amount is **deliberately absent**, so the
   assistant escalates instead of inventing a number. Unverified records are
   ranked lower and labelled `[unverified]` in the prompt context. **Admissions
   staff must verify these in the dashboard before go-live.**
4. **Local server-side TTS needs `espeak-ng`/`piper` installed** (not present in
   every container); otherwise `TTS_PROVIDER=local` degrades to text-only. Use a
   cloud TTS provider in production.
5. **Telephony paths are untestable without credentials.** Twilio Media Streams,
   Exotel and Plivo adapters are implemented and unit-covered where possible, but
   a live number is required for end-to-end validation.
6. **Single-instance SQLite by default.** For production concurrency use Postgres
   (`DATABASE_URL`) and, for multi-replica deployments, `VECTOR_STORE=pgvector`
   or `qdrant` so the index is shared rather than per-process.

---

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — call flow, state machine,
  RAG pipeline, provider abstraction, latency budget, data model.
- **[docs/KNOWLEDGE_BASE.md](docs/KNOWLEDGE_BASE.md)** — guide for admissions
  staff: editing content, verification workflow, CSV/Sheets import, admission-cycle
  updates without a redeploy.
- **[docs/COMPLIANCE.md](docs/COMPLIANCE.md)** — TRAI recording-consent and
  DPDP Act 2023 mapping, PII handling, retention, caller rights.
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — Docker, Postgres/pgvector,
  provider wiring, public HTTPS/WSS, go-live checklist.
