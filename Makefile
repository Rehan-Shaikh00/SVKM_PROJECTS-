# NMIMS Global University, Dhule — AI voice assistant developer tasks.
#
#   make install && make build && make run      # zero-key local run
#   make test && make lint                      # verification
#   make smoke                                  # end-to-end call over WebSocket
#
# Everything assumes the venv at .venv (repo root). Override with PY=... or
# PORT=... as needed.

# Anchor every path to this Makefile's directory: recipes `cd` into backend/ or
# frontend/, and a relative .venv path would then resolve to backend/.venv.
ROOT          := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PY            ?= $(ROOT)/.venv/bin/python
PIP           ?= $(ROOT)/.venv/bin/pip
RUFF          ?= $(ROOT)/.venv/bin/ruff
PYTEST        ?= $(ROOT)/.venv/bin/pytest
NPM           ?= npm
PORT          ?= 8000
HOST          ?= 0.0.0.0
BACKEND       := $(ROOT)/backend
FRONTEND      := $(ROOT)/frontend
UVICORN       := $(PY) -m uvicorn app.main:app

.DEFAULT_GOAL := help

## help: list available targets
.PHONY: help
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'

## venv: create the virtualenv (python3.11+)
.PHONY: venv
venv:
	cd $(ROOT) && (python3.11 -m venv .venv || python3 -m venv .venv)
	$(PIP) install --upgrade pip

## install: install backend dependencies
.PHONY: install
install:
	$(PIP) install -r $(BACKEND)/requirements.txt

## install-frontend: install dashboard dependencies
.PHONY: install-frontend
install-frontend:
	cd $(FRONTEND) && $(NPM) install

## build: build the React dashboard into backend/static/dashboard
.PHONY: build
build:
	cd $(FRONTEND) && $(NPM) run build

## run: serve the API + dashboard (PORT=8000 by default)
.PHONY: run
run:
	cd $(BACKEND) && $(UVICORN) --host $(HOST) --port $(PORT)

## dev: serve with auto-reload, backend only (PORT=8000 by default)
.PHONY: dev
dev:
	cd $(BACKEND) && $(UVICORN) --host $(HOST) --port $(PORT) --reload

## dev-frontend: Vite dev server with /api proxied to the backend
.PHONY: dev-frontend
dev-frontend:
	cd $(FRONTEND) && $(NPM) run dev

## test: run the backend test suite (no DB, network or API keys needed)
.PHONY: test
test:
	cd $(BACKEND) && $(PYTEST) -q

## lint: ruff check the backend
.PHONY: lint
lint:
	cd $(BACKEND) && $(RUFF) check app/ tests/

## format: ruff auto-fix + import sorting
.PHONY: format
format:
	cd $(BACKEND) && $(RUFF) check app/ tests/ --fix

## smoke: drive a full call over WebSocket (needs a running server)
.PHONY: smoke
smoke:
	cd $(ROOT) && $(PY) scripts/smoke_ws.py

## verify: assert what a caller hears for 57 questions (needs a running server)
.PHONY: verify
verify:
	cd $(ROOT) && $(PY) scripts/verify_answers.py

## reindex: rebuild chunks + embeddings from data/kb and the database
.PHONY: reindex
reindex:
	curl -fsS -X POST http://127.0.0.1:$(PORT)/api/kb/reindex | $(PY) -m json.tool

## kb-stats: knowledge base record/chunk/verification counts
.PHONY: kb-stats
kb-stats:
	curl -fsS http://127.0.0.1:$(PORT)/api/kb/stats | $(PY) -m json.tool

## health: readiness + per-provider degradation
.PHONY: health
health:
	@curl -fsS http://127.0.0.1:$(PORT)/health/ready; echo
	@curl -fsS http://127.0.0.1:$(PORT)/health/providers | $(PY) -m json.tool

## docker-build: build the runtime image (dashboard + FastAPI)
.PHONY: docker-build
docker-build:
	cd $(ROOT) && docker build -t nims-voice-assistant:latest .

## docker-up: run app + Postgres/pgvector via compose
.PHONY: docker-up
docker-up:
	cd $(ROOT) && docker compose up --build -d
	@echo "dashboard: http://localhost:$(PORT)/   docs: http://localhost:$(PORT)/api/docs"

## docker-down: stop the compose stack (volumes are kept)
.PHONY: docker-down
docker-down:
	cd $(ROOT) && docker compose down

## docker-logs: follow application logs
.PHONY: docker-logs
docker-logs:
	cd $(ROOT) && docker compose logs -f app

## clean: remove caches and build artefacts (keeps data/ and .env)
.PHONY: clean
clean:
	rm -rf $(BACKEND)/static/dashboard
	find $(ROOT) -path '*/node_modules' -prune -o -name '__pycache__' -print -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(ROOT)/.pytest_cache $(ROOT)/.ruff_cache $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache

## reset-data: delete the local DB and vector index (reseeds on next boot)
.PHONY: reset-data
reset-data:
	rm -rf $(ROOT)/data/runtime $(ROOT)/data/index
	@mkdir -p $(ROOT)/data/runtime $(ROOT)/data/index
	@echo "cleared; the next boot re-ingests data/kb and rebuilds the index"

## setup: venv + backend deps + dashboard deps + build (fresh clone)
.PHONY: setup
setup: venv install install-frontend build
	@echo "\nNext:  cp .env.example .env   then   make run"
