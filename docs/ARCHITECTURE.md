# Architecture

## Design goals

| Goal | How it is met |
|---|---|
| Perceived latency < 1–2 s | Sentence-level streaming: retrieval and generation start while the caller is still finishing; the first sentence is synthesised and played before the rest is written |
| Never invent facts | Grounding gate on retrieved chunks + numeric grounding check against the context; failure escalates instead of answering |
| No keypad navigation | Trilingual spoken prompt + layered language identification |
| Staff-editable content | KB rows in the database, edited via API/dashboard, re-chunked and re-embedded on write — no redeploy |
| Runs with zero credentials | Every provider has a local fallback; degraded mode is logged, not fatal |
| Barge-in | Audio VAD on the media stream, plus an explicit control message for browser/text clients |

---

## Request path of a single call

```
Caller ──PSTN──> Twilio ──<Connect><Stream>──> WSS /telephony/twilio/media-stream
                                                     │
                                        TwilioMediaChannel (8 kHz μ-law, 20 ms frames)
                                                     │
                                              CallSession (state machine)
                                                     │
        ┌────────────────────────┬───────────────────┼────────────────────┬─────────────────┐
        │                        │                   │                    │                 │
   StreamingASR             LID detector        AnswerEngine (RAG)        TTS engine     CallLogger
   (Deepgram/Google/       (name → script →     ├─ intent classifier     (Google/Azure/  (queue → DB,
    Azure/AssemblyAI/       lexicon → acoustic)  ├─ topic guardrails       ElevenLabs/     never blocks
    browser)                                     ├─ HybridRetriever        local/browser)  the call)
                                                 │   BM25 + dense, RRF
                                                 ├─ LLM (Claude/OpenAI)
                                                 │   or template path
                                                 └─ numeric grounding gate
```

The channel abstraction is the only provider-specific piece: the browser
simulator, Twilio Media Streams, Exotel and Plivo all drive the *same*
`CallSession`, so behaviour is identical across transports.

---

## Call state machine

`app/orchestrator/states.py`

```
started → greeting → language_prompt ─┬→ menu → conversation ⇄ followup_offer
                     language_reprompt │                     ↓
                     language_confirm  │              followup_destination
                     language_unsupported                    │
                                      └──────────→ escalating → transferred → closing → ended
                                                                                   └→ failed
```

- `LANGUAGE_STATES` — an utterance is interpreted as a *language choice*.
- `CONVERSATION_STATES` — an utterance is interpreted as a *question*.
- Each caller turn is stamped with the state it arrived in (`metadata.stage`),
  which is what lets analytics exclude language answers from "top queries".

Transitions are logged with from/to state and emitted to the client as `state`
events, so the dashboard can render a live call.

---

## Language identification

`app/voice/lid/` — layered, cheapest first, fused with confidence:

1. **Explicit language names** across scripts ("Hindi", "हिन्दी", "অংগ্রেজি",
   "अंग्रेजी", "আপনি বাংলা", 72 name→code entries).
2. **Script detection** — Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil,
   Telugu, Kannada, Malayalam, Urdu/Arabic.
3. **Lexicon scoring** — function words and code-mixed Hinglish tokens.
4. **Acoustic LID adapters** (Google/Azure/Deepgram) when credentials exist.

Fusion requires `LID_CONFIDENCE_THRESHOLD`; below it the assistant reprompts
(`language_reprompt`) and, after repeated failure, offers the supported-language
list or escalates. Two consecutive mid-call detections of a different language
adopt it (`_language_switch_streak`), so a single ambiguous utterance cannot
flip a call.

For a Maharashtra campus the greeting languages are `en-IN`, `hi-IN` and `mr-IN`,
and **Marathi is a first-class conversation language**: its own prompt scripts,
its own answer-template frames, Marathi lakh/crore number words, Devanagari
markers (`आहे`, `पाहिजे`, `किती`) *and* romanised-Marathi markers (`mala`,
`aahe`, `pahije`) so a Latin-script transcript is still classified correctly.
Devanagari input is disambiguated three ways — Hindi, Marathi, Rajasthani — by
weighted function-word markers. `raj-IN` and `gu-IN` remain supported (a caller
who names one gets it) but are no longer announced; `raj-IN` is mapped to `hi-IN`
for ASR/TTS because no major vendor ships a Rajasthani acoustic model.

On detection, `asr_locale`, `tts_locale`, prompt scripts, template frames and the
LLM `LANGUAGE_RULE` all switch together.

---

## RAG pipeline

`app/ai/rag.py`, `app/kb/retriever.py`

```
question
  → intent classification (22 intents, EN + Indic patterns)
  → topic guardrails (sensitive → escalate; off-topic → decline)
  → query expansion (synonyms, transliteration, cross-lingual terms, course tokens)
  → hybrid retrieval
        BM25 over token-expanded chunks      ┐
        dense cosine over embeddings         ┴→ Reciprocal Rank Fusion (k=60)
  → rerank: category match for the intent, language match, verified > unverified,
            staleness penalty, course-token overlap, title > body weighting
  → context block assembly (budgeted characters, citations attached)
  → generation
        LLM path: streaming, sentence-completeness detection, per-sentence TTS
        template path (no key): composed from structured KB fields
  → guardrails on the output
        voice scrub (markdown/URLs/bullets, length caps)
        numeric grounding: every number in the answer must appear in the context
                           (understands Indian scale words and Devanagari digits)
        PII redaction + hashing
  → AssistantAnswer {text, grounded, confidence, citations, needs_escalation,
                     followup, timings, warnings}
```

**Grounding contract.** An answer is only returned as grounded when retrieval
produced supporting chunks above `RETRIEVAL_MIN_SCORE` *and* the numeric check
passed. Otherwise the assistant says it does not have that information and
escalates — the brief's hard requirement, and the reason fee rows in the seed KB
ship `verified: false`.

### Chunking and indexing
`app/kb/chunking.py` is structure-aware: records are split on headings, list
items and clause boundaries with a configurable overlap, preserving the record's
category, language, verification flag, academic year and source. Each chunk is
embedded and upserted; the retriever rebuilds its lexical index from published
chunks and **prunes vector-store orphans** (embeddings whose chunk was deleted or
unpublished) so a stale vector can never outrank a live one.

### Vector stores
`LocalVectorStore` (numpy matrix + JSONL metadata, `asyncio.to_thread` IO),
`PgVectorStore` (pgvector, `CREATE EXTENSION` on boot), `QdrantVectorStore`.
All implement `upsert / search / delete / delete_by_record / count / ids`.

---

## Voice path

`app/voice/audio.py` is pure numpy (no ffmpeg/sox/portaudio dependency):
G.711 μ-law encode/decode, linear resampling, 20 ms framing for Twilio, an
energy VAD with hangover, DTMF decode, and WAV header construction.

**Barge-in** works two ways:
- Telephone callers: VAD detects speech while the assistant is playing → the
  speech queue is dropped, `barge_in` is emitted, the interrupted turn is marked.
- Browser/text clients: they do their own ASR and send no audio, so they send an
  explicit `control: barge_in` / `interrupt` message.

**TTS chunking** (`split_for_speech`) splits on sentence boundaries first, then
clauses, splitting *before* conjunctions with a lookahead so "and"/"और" survive,
never splitting "per year", and hard-slicing only a single overflowing clause.
A previous implementation flushed its buffer and then sliced a candidate that
still contained it, so callers heard the same clause twice — now covered by
`tests/test_speech_splitting.py`.

**Spoken numbers** (`app/ai/spoken_numbers.py`) render amounts the way Indians
say them: `150000` → "one lakh fifty thousand rupees", `5600000` → "fifty-six
lakh rupees", with Devanagari output for Hindi.

---

## Escalation and hand-off

Triggers: `caller_requested_human`, `sensitive_or_legal`, `kb_no_answer`,
`low_confidence`, `unsupported_language`, `max_silence`, `max_call_minutes`,
`repeated_failure`.

`app/ai/summarizer.py` builds a `CallSummary` (extractive by default, LLM when
keyed): caller language, programmes discussed, what the assistant covered, what
it could not answer, and the escalation reason. `whisper_twiml_text` renders the
version spoken to the agent, clipped on word boundaries so nothing is cut
mid-word. Twilio transfer uses `<Dial>` with a `<Queue>` fallback, hold music and
a dial-result callback; the agent hears the whisper before bridging.

---

## Persistence

`app/models.py`, SQLAlchemy 2.0 async, SQLite by default / Postgres in production.

| Table | Purpose |
|---|---|
| `calls` | one row per call: provider, numbers (hashed), language + confidence + method, status, end reason, resolution, escalation, turn count, latency stats, summary, recording consent |
| `call_turns` | every turn: role, text, language, grounded, confidence, citations, tool calls, per-stage timings (asr/retrieval/llm/tts/total), stage |
| `escalation_events` | reason, target, wait, outcome, whisper summary, context |
| `follow_ups` | SMS/WhatsApp/email delivery of long details |
| `kb_records` / `kb_chunks` / `kb_revisions` | content, retrieval units, edit history |
| `unanswered_questions` | backlog for staff, canonicalised and resolvable |
| `provider_health` | degraded-mode observations per provider |

**Call logging is queue-based.** `CallLogger` wrappers enqueue onto a bounded
`asyncio.Queue` (500) and a single background worker writes and commits. The
wrappers are deliberately **plain functions, not coroutines**: the voice path must
never await a database write, and if they were `async def` an un-awaited call
would silently drop the event. When that happened, no call records, transcripts,
escalations or analytics rows were ever written. `tests/test_call_logger.py`
asserts the wrappers are not coroutine functions and that every enqueued kind has
a handler.

PII is redacted and hashed *before* it reaches the transcript, and caller numbers
are stored hashed unless explicitly configured otherwise.

---

## API surface (57 routes)

| Group | Prefix | Highlights |
|---|---|---|
| assistant | `/api/assistant` | `POST /query` (HTTP one-shot, used by the dashboard), `POST /retrieve` (debug retrieval), `GET /languages` |
| simulator | `/api/simulator` | `WS /ws` (full call protocol), `GET /config`, `POST /end/{call_id}` |
| knowledge-base | `/api/kb` | record CRUD, verify/bulk-verify, import (CSV/YAML/JSON, dry-run), `sync-sheet`, `preview-chunking`, `reindex`, `export.csv`, `template.csv`, `purge`, `stats`, `categories` |
| analytics | `/api/analytics` | `overview`, `languages`, `intents`, `top-queries`, `unanswered` (+ resolve), `latency`, `escalations`, `quality` |
| calls | `/api/calls` | list/filter, live calls, detail with transcript, `handoff-brief`, `transcript.txt`, QA `review` |
| telephony | `/telephony` | Twilio voice/status/transfer/queue/sms/whisper, Exotel + Plivo webhooks, health |
| config | `/api/config`, `/api/status`, `/api/service` | non-sensitive runtime config, metrics, service index |
| health | `/health`, `/health/ready`, `/health/providers`, `/metrics` | liveness, readiness (records/chunks/problems), per-provider degradation, metrics |

Interactive docs at `/api/docs`; OpenAPI schema at `/api/openapi.json`.
The dashboard is served from `/dashboard/` and `/` redirects there.

### WebSocket call protocol

Server → client: `call_started`, `asr_config`, `state`, `partial_transcript`,
`caller_transcript`, `language_detected`, `language_switched`,
`unsupported_language`, `assistant_text`, `assistant_speech`, `audio`,
`speech_finished`, `barge_in`, `silence`, `followup_offer`, `dtmf_ignored`,
`dtmf_unmapped`, `tts_error`, `error`, `call_ended`.

Client → server: `start`, `audio`, `transcript`, `text`, `dtmf`, `control`
(`speech_done`, `playback_done`, `barge_in`/`interrupt`, `hangup`,
`language_override`, `escalate`), `language_override`, `escalate`, `hangup`.

---

## Latency budget

Measured on the simulator path (text input, local providers), per turn:

| Stage | Typical |
|---|---|
| Retrieval (BM25 + dense + rerank over 155 chunks) | ~3 ms |
| Generation (template path) | ~1–3 ms |
| First audio/first sentence | < 30 ms |
| End-to-end turn | ~25 ms |

With cloud providers the budget is dominated by ASR endpointing and LLM first
token; streaming the first sentence to TTS before generation completes is what
keeps perceived latency inside the 1–2 s target. `/api/analytics/latency`
reports real p50/p95 per stage from logged calls.

---

## Frontend

Vite + React + TypeScript + Recharts, built into `backend/static/dashboard` and
served by FastAPI on the same origin (no CORS, no second port).

- **Phone** — browser phone: mic capture → Web Speech ASR → the same WS protocol,
  TTS via `speechSynthesis` or server μ-law audio, barge-in button, live state and
  transcript, follow-up handling, escalation.
- **Analytics** — KPI cards, language and outcome distributions, latency bars,
  intent mix, top queries, escalation breakdown, unanswered backlog with inline
  resolve.
- **Knowledge base** — staff editor: search/filter, CRUD, verify toggles, bulk
  verify, CSV import with dry-run diff, paste import, Google Sheet sync, reindex,
  export, chunk-preview drawer.
- **Calls** — call log with filters, live calls, detail drawer (turn-by-turn
  transcript with timings, citations, grounding, confidence), hand-off brief, QA
  review scoring.

All URLs are same-origin relative and the WebSocket uses `window.location.host`,
so the app works behind a path-prefixed proxy. Vite builds with `base: './'` for
the same reason.
