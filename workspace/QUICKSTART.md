# SVKM NMIMS Voice Assistant - Quick Start Guide

This guide will help you get the voice assistant system running locally.

## Prerequisites

- Python 3.11+
- Node.js 20+
- pnpm 8+
- Docker & Docker Compose
- PostgreSQL 15+ (via Docker or local)
- Redis 7+ (via Docker or local)

## Quick Start (Docker Compose)

The fastest way to get started:

```bash
cd workspace

# 1. Copy environment template
cp .env.example .env.local

# 2. Start all services with Docker Compose
docker-compose up

# 3. In another terminal, run migrations and seed data
docker-compose exec api alembic upgrade head
docker-compose exec api python scripts/seed_contacts.py
```

Services will be available at:
- **Web UI**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8001/ws/voice

## Manual Setup

If you prefer to run services individually:

```bash
cd workspace

# 1. Setup database and dependencies
./scripts/setup.sh

# 2. Start services in separate terminals

# Terminal 1: API Service
python -m apps.api.main

# Terminal 2: Realtime Gateway
python -m apps.realtime.main

# Terminal 3: Web UI
cd apps/web && pnpm dev

# Terminal 4: Worker (optional)
python -m apps.worker.main
```

## Configuration

### Using Fake Providers (No Credentials Required)

For development without Azure/Anthropic credentials:

```bash
# In .env.local
PROVIDERS_MODE=fake
```

This uses in-memory test doubles for all external services.

### Using Real Providers

Update `.env.local` with your credentials:

```bash
PROVIDERS_MODE=real

# Azure Speech
AZURE_SPEECH_KEY=your_key_here
AZURE_SPEECH_REGION=centralindia

# Azure OpenAI (embeddings)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=your_key_here

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

## Testing the System

### Browser Simulator

1. Open http://localhost:3000
2. Select language (English, Hindi, or Marathi)
3. Type a question like:
   - "What courses are available?"
   - "Tell me about MBA admission"
   - "What are the fees for B.Tech?"
4. View grounded responses with citations

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# List contact routes
curl http://localhost:8000/admin/contacts

# Analytics overview
curl http://localhost:8000/analytics/overview
```

### WebSocket Testing

Use a WebSocket client or the browser UI to test real-time communication.

## Development Workflow

### Running Tests

```bash
# All tests
pnpm test

# Python tests only
pytest

# TypeScript tests only
cd apps/web && pnpm test
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
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Architecture Overview

```
┌─────────────────┐
│   Next.js Web   │  Browser UI (simulator + admin)
│   localhost:3000│
└────────┬────────┘
         │
         ├─────────────────┬─────────────────┐
         │                 │                 │
┌────────▼────────┐ ┌─────▼──────┐ ┌───────▼───────┐
│  FastAPI API    │ │  Realtime  │ │   Worker      │
│  localhost:8000 │ │  WS:8001   │ │  (Background) │
└────────┬────────┘ └─────┬──────┘ └───────┬───────┘
         │                │                 │
         └────────────────┼─────────────────┘
                          │
              ┌───────────▼───────────┐
              │   PostgreSQL + Redis  │
              │   localhost:5432/6379 │
              └───────────────────────┘
                          │
              ┌───────────▼───────────┐
              │  Azure Services       │
              │  - Speech (ASR/TTS)   │
              │  - OpenAI (Embeddings)│
              │  - Blob Storage       │
              └───────────────────────┘
                          │
              ┌───────────▼───────────┐
              │  Claude API           │
              │  (Grounded Responses) │
              └───────────────────────┘
```

## Key Features Implemented

✅ **Foundation**
- Monorepo structure (Python + Node)
- PostgreSQL + pgvector schema
- Redis cache
- Configuration management
- Docker Compose stack

✅ **Provider Layer**
- Azure Speech (ASR/TTS/Language ID)
- Claude API (grounded generation)
- Azure OpenAI (embeddings)
- Fake providers for testing

✅ **Conversation Engine**
- Transport-neutral state machine
- Language detection
- Hybrid retrieval (vector + FTS)
- Grounded response generation
- Citation tracking
- Decision policy (answer/clarify/abstain)

✅ **Services**
- FastAPI management API
- WebSocket realtime gateway
- Background worker
- Contact routing

✅ **Web UI**
- Next.js simulator
- WebSocket integration
- Multilingual support
- Real-time messaging
- Citation display

✅ **DevOps**
- Docker Compose
- Alembic migrations
- Health checks
- Logging structure

## What's Next

### To Complete for Production

1. **Knowledge Ingestion** - Implement crawler and PDF parser
2. **Admin Panel** - Build full admin UI for knowledge management
3. **Exotel Integration** - Activate telephony adapter
4. **Azure Deployment** - Deploy to Container Apps
5. **Monitoring** - Set up Application Insights
6. **Testing** - Add comprehensive test suite
7. **Documentation** - Complete API docs and runbooks

### Development Priorities

1. Test with real Azure Speech credentials
2. Ingest official university content
3. Validate grounded responses
4. Test multilingual flows
5. Implement handoff logic
6. Add authentication
7. Deploy to staging

## Troubleshooting

### Services won't start

```bash
# Check if ports are already in use
lsof -i :3000  # Web
lsof -i :8000  # API
lsof -i :8001  # Realtime
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis

# Restart Docker services
docker-compose down
docker-compose up -d
```

### Database connection errors

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check connection
psql postgresql://svkm_user:svkm_password@localhost:5432/svkm_voice_assistant
```

### Missing dependencies

```bash
# Reinstall Python deps
pip install -e ".[dev]"

# Reinstall Node deps
pnpm install
```

## Support

For issues or questions:
- Check the main [README.md](README.md)
- Review logs: `docker-compose logs -f`
- Contact the development team

---

**Built for SVKM NMIMS Global University, Dhule**
