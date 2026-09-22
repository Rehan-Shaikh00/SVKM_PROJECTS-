# SVKM NMIMS Global University, Dhule — Voice Assistant

AI-powered multilingual voice assistant for handling inbound admissions calls with grounded, accurate responses from verified university data.

## Overview

This system answers calls in English, Hindi, and Marathi, using spoken language identification (no keypad navigation). It provides information about courses, admissions, fees, eligibility, hostel facilities, and contacts — all grounded in official university data with mandatory escalation when information is missing or uncertain.

## Architecture

### Core Components

- **apps/api** — FastAPI management API, simulator endpoints, admin/analytics backend
- **apps/realtime** — WebSocket gateway for browser/Exotel voice sessions
- **apps/worker** — Background jobs for ingestion, embedding, evaluation, retention
- **apps/web** — Next.js simulator UI and admin/analytics dashboard

### Shared Packages

- **packages/conversation** — Transport-neutral state machine and conversation policies
- **packages/knowledge** — Hybrid retrieval (pgvector + FTS), citations, confidence scoring
- **packages/providers** — Azure Speech, Claude API, Azure OpenAI embeddings, Blob Storage, Redis
- **packages/telephony** — Canonical events, browser adapter, Exotel adapter (ready for activation)
- **packages/schemas** — Shared TypeScript/Python API contracts

## Tech Stack

| Layer | Technology |
|---|---|
| Languages | Python 3.11+, TypeScript 5.x, Node 20+ |
| Backend | FastAPI, asyncio, Pydantic |
| Frontend | Next.js 14, React, TailwindCSS |
| Database | PostgreSQL 15+ with pgvector extension |
| Cache | Redis 7+ |
| Storage | Azure Blob Storage |
| Speech | Azure Speech Services (ASR, TTS, Language ID) |
| AI | Claude API (grounded generation), Azure OpenAI (embeddings) |
| Telephony | Browser WebSocket (active), Exotel AgentStream (ready) |
| Deployment | Azure Container Apps, managed PostgreSQL/Redis, Key Vault, Front Door |
| Observability | OpenTelemetry, Azure Monitor, Application Insights |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- pnpm 8+
- Docker & Docker Compose
- PostgreSQL 15+ (local or Docker)
- Redis 7+ (local or Docker)

### Local Development

```bash
# 1. Clone and enter workspace
cd workspace

# 2. Install Python dependencies
pip install -e ".[dev]"

# 3. Install Node dependencies
pnpm install

# 4. Copy environment template
cp .env.example .env.local

# 5. Start local stack (PostgreSQL, Redis, all services)
docker-compose up

# 6. Run migrations
alembic upgrade head

# 7. Seed initial data
python scripts/seed_contacts.py
python scripts/ingest_official_sources.py

# 8. Open simulator
open http://localhost:3000
```

### Environment Variables

Required for local development with real providers:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/svkm_voice_assistant

# Redis
REDIS_URL=redis://localhost:6379/0

# Azure Speech
AZURE_SPEECH_KEY=your_key_here
AZURE_SPEECH_REGION=centralindia

# Azure OpenAI (embeddings)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your_key_here
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Claude API
ANTHROPIC_API_KEY=your_key_here

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=your_connection_string

# Optional: Exotel (disabled by default)
EXOTEL_ACCOUNT_SID=your_sid
EXOTEL_API_KEY=your_key
EXOTEL_API_TOKEN=your_token
EXOTEL_ENABLED=false
```

### Fake Provider Mode

Run without Azure/Anthropic credentials using test doubles:

```bash
PROVIDERS_MODE=fake docker-compose up
```

This uses in-memory implementations suitable for development and CI.

## Development Workflow

### Running Services Individually

```bash
# API service
cd apps/api && uvicorn main:app --reload --port 8000

# Realtime gateway
cd apps/realtime && uvicorn main:app --reload --port 8001

# Worker
cd apps/worker && python -m worker.main

# Web UI
cd apps/web && pnpm dev
```

### Running Tests

```bash
# All tests
pnpm test

# Unit tests only
pytest tests/unit

# Integration tests (requires PostgreSQL/Redis)
pytest tests/integration

# Contract tests (provider adapters)
pytest tests/contract

# E2E tests (requires full stack)
pytest tests/e2e

# Golden evaluations
python scripts/run_evals.py
```

### Code Quality

```bash
# Python
ruff check .
ruff format .
mypy .

# TypeScript
pnpm lint
pnpm type-check

# Pre-commit (runs on git commit)
pre-commit install
pre-commit run --all-files
```

## Knowledge Base

### Official Source Ingestion

The system ingests from an allowlist of official university pages and PDFs:

- Institution pages (about, vision, mission)
- All program pages (courses, eligibility, duration)
- School-specific admission handouts
- Admissions process, FAQ, policy pages
- Hostel and contact information
- Official 2026-27 fee structure PDFs

```bash
# Trigger ingestion
python scripts/ingest_official_sources.py

# Check ingestion status
curl http://localhost:8000/admin/ingestion/status
```

### Manual Knowledge Upload

Upload the `svkm_nmims_dhule_knowledge_base.json` file:

1. Navigate to Admin Panel → Knowledge → Import
2. Upload JSON file
3. Review conflicts and staging preview
4. Approve for publication

The system detects conflicts between official sources, manual uploads, and existing data, requiring maker-checker approval before publish.

### Known Data Gaps

These topics **must escalate to human agents** (no data available):

- **Scholarships** — not published by university
- **Engineering-specific fees** — request detailed PDF from university
- **Dhule campus hostel specifics** — general NMIMS network info only

## Conversation Policies

### Grounded Answer Requirements

Every material claim must map to:
- A specific source chunk ID
- A published fact record
- Current academic year data
- Approved school/program

### Three-Way Decision Policy

1. **Answer** — Current authoritative evidence supports all claims
2. **Clarify** — School, program, year, or language is ambiguous
3. **Abstain/Handoff** — Evidence missing, stale, conflicting, or low confidence

### Strict Validation Rules

Extra scrutiny for:
- Fees and payment schedules
- Admission dates and deadlines
- Eligibility criteria and qualifying tests
- Scholarship availability
- Hostel capacity and availability
- Contact numbers and email addresses

### Voice Response Guidelines

- Keep responses brief and conversational
- Offer to send detailed tables/schedules via text/WhatsApp/email
- Never read long fee tables or admission schedules aloud
- Preserve exact values for dates, amounts, phone numbers
- Localize responses to caller's detected language

## Handoff & Escalation

### Contact Routes (Verified)

- **School of Technology, Management & Engineering**
  - Primary: 9425685966
  - Fallback: 9764405069

- **School of Pharmacy & Technology Management**
  - Primary: 9158947999
  - Fallback: 8668501496

- **School of Commerce**
  - Primary: 8788701642

### Escalation Triggers

- Caller explicitly asks for human agent
- Confidence below threshold
- Evidence conflicts or is stale
- Question about unpublished data (scholarships, etc.)
- Detected language not supported
- Technical failure or timeout

### Handoff Summary

Generated for receiving team:
- Detected language
- Caller's question (redacted for PII)
- Resolved school/program/year if any
- Escalation reason
- Cited evidence IDs

## Privacy & Compliance

### Default Privacy Posture

- **No raw audio recording retention by default**
- Localized consent prompt before any recording/storage
- Redacted transcripts retained for 90 days (configurable)
- Aggregate analytics for 365 days (configurable)
- PII redacted from routine logs
- Encryption at rest (Azure Blob, PostgreSQL TDE)
- Encryption in transit (TLS 1.3)

### RBAC Roles

- **editor** — Edit knowledge, sources, contacts
- **reviewer** — Approve staged publications
- **analyst** — View analytics, aggregated call data
- **operator** — View call status, trigger manual actions
- **auditor** — Access transcripts, audit logs, exports
- **administrator** — Full system access

### Audit Trail

All mutations logged with:
- User ID and role
- Timestamp (UTC)
- Action type
- Resource ID
- Before/after state (for knowledge changes)
- Client IP (redacted after 90 days)

### Data Retention & Deletion

- Operational events: 90 days (configurable)
- Transcripts: 90 days (configurable)
- Analytics aggregates: 365 days (configurable)
- Legal holds: bypass automatic deletion
- Export: JSON format with audit entry
- Deletion: soft delete with 30-day recovery window

## Deployment

### Azure Container Apps

The system deploys to Azure Container Apps with:

- **Web UI** (Next.js) — Front Door CDN, autoscale 1-10
- **API** (FastAPI) — Autoscale 2-20, health probes
- **Realtime Gateway** (FastAPI WebSocket) — Autoscale 2-20, sticky sessions, min warm replicas
- **Worker** (Python) — Job-based or always-on depending on workload

### Infrastructure as Code

```bash
# Deploy to dev environment
cd infra
az deployment group create \
  --resource-group svkm-voice-dev \
  --template-file main.bicep \
  --parameters @dev.parameters.json

# Deploy to production
az deployment group create \
  --resource-group svkm-voice-prod \
  --template-file main.bicep \
  --parameters @prod.parameters.json
```

### Required Azure Resources

- Container Apps environment
- PostgreSQL Flexible Server (with pgvector)
- Azure Cache for Redis
- Azure Blob Storage
- Azure Key Vault
- Azure Speech Services
- Azure OpenAI Service
- Front Door + WAF
- Log Analytics Workspace
- Application Insights

### Deployment Checklist

- [ ] Provision Azure resources via Bicep/Terraform
- [ ] Configure managed identities for Key Vault access
- [ ] Set environment variables in Container Apps
- [ ] Run database migrations
- [ ] Seed contact routes and initial knowledge
- [ ] Configure Front Door custom domain and TLS
- [ ] Set up Azure Monitor alerts
- [ ] Verify health checks pass
- [ ] Run smoke tests against staging
- [ ] Load test realtime gateway
- [ ] Verify backup/restore procedures
- [ ] Document runbooks for on-call

## Exotel Activation

The Exotel adapter is **implemented but feature-disabled** pending:

1. Account credentials (SID, API key, token)
2. Sandbox testing verification
3. AgentStream protocol certification
4. Call transfer API integration
5. Recording callback handling
6. Production number assignment

### Activation Steps

```bash
# 1. Set Exotel credentials in Key Vault

# 2. Enable Exotel adapter
export EXOTEL_ENABLED=true

# 3. Configure webhook URLs
export EXOTEL_STATUS_CALLBACK_URL=https://yourdomain.com/webhooks/exotel/status
export EXOTEL_RECORDING_CALLBACK_URL=https://yourdomain.com/webhooks/exotel/recording

# 4. Run contract tests against sandbox
pytest tests/contract/test_exotel_adapter.py --exotel-sandbox

# 5. Verify call flow in sandbox
python scripts/exotel_sandbox_test.py

# 6. Production cutover (requires university approval)
```

## Monitoring & Operations

### Key Metrics

- **Availability** — Service uptime, health check success rate
- **Latency** — End-of-speech to first audio (P50, P95, P99)
- **Quality** — Grounded vs. abstained rate, handoff rate, language detection accuracy
- **Volume** — Calls per hour, concurrent sessions, language distribution
- **Errors** — Provider failures, timeout rate, WebSocket disconnects

### Alerts

- Service health check failures
- P95 latency > 2.5 seconds
- Provider error rate > 5%
- WebSocket disconnect rate > 10%
- Database connection pool exhausted
- Knowledge ingestion failures

### Dashboards

- **Operations** — Service health, latency, error rates, provider status
- **Call Analytics** — Volume by time/language/school, top queries, handoff reasons
- **Knowledge Quality** — Coverage gaps, stale data, conflict trends
- **Privacy & Compliance** — Consent rate, retention policy adherence, audit activity

## Security

### Threat Model

- **Injection attacks** — SSRF in crawler, SQL injection, prompt injection
- **Data exfiltration** — Transcript access, PII leakage in logs
- **DoS/resource exhaustion** — WebSocket floods, large uploads, expensive queries
- **Privilege escalation** — RBAC bypass, token theft

### Mitigations

- Allowlist for knowledge sources (no arbitrary URLs)
- Parameterized SQL queries only
- Input validation and size limits on all endpoints
- Rate limiting on WebSocket connections and uploads
- WAF rules for common attack patterns
- Prompt injection detection and sanitization
- PII redaction in logs and error messages
- Secrets only from Key Vault (never in code/environment)
- Container vulnerability scanning in CI
- Regular dependency updates

## Limitations & Launch Gates

### Delivered System Capabilities

✅ Fully functional browser-based simulator
✅ Grounded RAG with hybrid retrieval
✅ Multilingual speech (English, Hindi, Marathi)
✅ Transport-neutral conversation engine
✅ Admin panel for knowledge management
✅ Privacy-by-default architecture
✅ Exotel adapter ready for activation

### Launch Blockers (External Dependencies)

❌ **Azure tenant & credentials** — Required for Speech, OpenAI, Blob Storage
❌ **Anthropic API key** — Required for Claude grounded generation
❌ **University content approval** — Legal must approve final knowledge base
❌ **TRAI compliance review** — Consent wording, recording retention, call ownership
❌ **Exotel account & sandbox** — Credentials, protocol certification, number assignment
❌ **Production contact verification** — School contacts must be verified as current

### Explicit Constraints

This system **cannot truthfully be declared production-ready** until:

1. University provides cloud credentials and approves deployment
2. Legal approves consent wording and retention policies
3. Admissions office verifies contact routes are current
4. Exotel sandbox certification passes
5. Production number is assigned and tested
6. University approves go-live for public calls

The delivered software is **deployment-ready**, but production **activation** requires completing these external gates.

## Troubleshooting

### Common Issues

**Simulator not connecting**
- Check realtime service is running on port 8001
- Verify WebSocket URL in web app config
- Check browser console for CORS errors

**Language detection not working**
- Verify Azure Speech credentials
- Check supported locales (en-IN, hi-IN, mr-IN)
- Ensure audio sample rate is 16kHz mono

**No search results**
- Run ingestion: `python scripts/ingest_official_sources.py`
- Check pgvector extension: `SELECT * FROM pg_extension WHERE extname = 'vector';`
- Verify embeddings were generated

**Provider timeouts**
- Check Azure/Anthropic service status
- Verify network connectivity from container
- Review timeout configuration in settings

**High latency**
- Check P95 latency metrics by stage
- Verify warm instances in Container Apps
- Review database query performance
- Check Redis cache hit rate

## Support & Contact

For technical issues or questions:

- Open an issue in the repository
- Contact the development team
- Review deployment runbooks in `/docs`

For university-specific configuration:

- Contact SVKM NMIMS Dhule IT/Admissions
- Review knowledge base admin panel
- Update contact routes in admin UI

---

**Version:** 1.0.0  
**Last Updated:** 2026-09-20  
**License:** Proprietary — SVKM NMIMS Global University, Dhule
