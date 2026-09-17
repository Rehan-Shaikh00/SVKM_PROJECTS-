"""Knowledge-base ingestion and indexing.

Sources, in order of how staff will actually use them:

1. **Google Sheet published as CSV** (`KB_GOOGLE_SHEET_CSV_URL`) — the
   recommended path for non-technical staff. One row per fact, re-synced hourly.
2. **CSV / YAML / JSON files in `data/kb/`** — version-controlled seeds.
3. **Admin dashboard** — direct record editing (app/api/kb_admin.py).

Ingestion is idempotent: content hashes decide whether a chunk has to be
re-embedded, so a nightly re-sync of 2 000 rows costs almost nothing.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models import KBChunk, KBRecord
from . import repository
from .chunking import chunk_record
from .retriever import HybridRetriever

logger = logging.getLogger("nims.ingest")


@dataclass
class IngestReport:
    source: str = "unknown"
    rows: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    embeddings: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    slugs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "rows": self.rows,
            "created": self.created,
            "updated": self.updated,
            "skipped": self.skipped,
            "failed": self.failed,
            "chunks_added": self.chunks_added,
            "chunks_removed": self.chunks_removed,
            "embeddings": self.embeddings,
            "duration_ms": round(self.duration_ms, 1),
            "errors": self.errors[:20],
            "slugs": self.slugs[:50],
        }


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #


def _read_csv(text: str) -> list[dict[str, Any]]:
    text = text.lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []
    for raw in reader:
        row = {(k or "").strip().lower(): (v.strip() if isinstance(v, str) else v)
               for k, v in raw.items() if k}
        if not row.get("title") and not row.get("slug"):
            continue
        rows.append(row)
    return rows


def _read_yaml(text: str) -> list[dict[str, Any]]:
    data = yaml.safe_load(text)
    if data is None:
        return []
    if isinstance(data, dict):
        for key in ("records", "items", "data", "courses", "faq"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    return [row for row in data if isinstance(row, dict)]


def _read_json(text: str) -> list[dict[str, Any]]:
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("records", "items", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    return [row for row in data if isinstance(row, dict)]


def parse_payload(text: str, filename: str = "payload") -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv" or text.lstrip("\ufeff").lower().startswith("slug,") or "," in text.split("\n")[0] and "\n" in text and "title" in text.split("\n")[0].lower():
        try:
            return _read_csv(text)
        except Exception:  # pragma: no cover
            pass
    if suffix in (".yaml", ".yml"):
        return _read_yaml(text)
    if suffix == ".json":
        return _read_json(text)
    # sniff
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        return _read_json(text)
    if stripped.startswith("- ") or ":" in stripped.split("\n")[0]:
        try:
            return _read_yaml(text)
        except yaml.YAMLError:
            pass
    return _read_csv(text)


def load_seed_files(directory: Path) -> list[tuple[str, dict[str, Any]]]:
    """Read every csv/yaml/json file in the seed directory."""
    out: list[tuple[str, dict[str, Any]]] = []
    if not directory.exists():
        logger.warning("KB seed directory %s does not exist", directory)
        return out
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".csv", ".yaml", ".yml", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            logger.error("could not read %s: %s", path, exc)
            continue
        try:
            rows = parse_payload(text, path.name)
        except Exception as exc:
            logger.error("could not parse %s: %s", path, exc)
            continue
        for row in rows:
            row.setdefault("source", f"seed:{path.name}")
            out.append((path.name, row))
    return out


async def fetch_google_sheet_csv(url: str) -> list[dict[str, Any]]:
    """Pull a Google Sheet published as CSV (`…/export?format=csv` or gviz)."""
    if not url:
        return []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    return _read_csv(resp.text)


# --------------------------------------------------------------------------- #
# indexing
# --------------------------------------------------------------------------- #


def _chunk_id(record_id: str, position: int, content_hash: str) -> str:
    digest = hashlib.sha1(f"{record_id}:{position}:{content_hash}".encode()).hexdigest()
    return digest[:28]


async def index_record(
    session: AsyncSession,
    retriever: HybridRetriever,
    record: KBRecord,
) -> tuple[int, int, int]:
    """Re-chunk one record. Returns (added, removed, embedded)."""
    chunks = chunk_record(
        {
            "title": record.title,
            "category": record.category,
            "subcategory": record.subcategory,
            "academic_year": record.academic_year,
            "structured": record.structured or {},
            "body": record.body or "",
            "language": record.language,
            "tags": record.tags or [],
            "aliases": record.aliases or [],
            "title_localized": record.title_localized or {},
        }
    )

    existing_rows = (
        (await session.execute(select(KBChunk).where(KBChunk.record_id == record.id)))
        .scalars()
        .all()
    )
    existing_by_hash = {row.content_hash: row for row in existing_rows}
    new_hashes = {chunk.content_hash for chunk in chunks}

    removed = 0
    for row in existing_rows:
        if row.content_hash not in new_hashes:
            await session.delete(row)
            removed += 1

    to_embed: list[dict[str, Any]] = []
    added = 0
    for chunk in chunks:
        if chunk.content_hash in existing_by_hash:
            continue  # unchanged content — keep the existing embedding
        row = KBChunk(
            id=_chunk_id(record.id, chunk.position, chunk.content_hash),
            record_id=record.id,
            position=chunk.position,
            text=chunk.text,
            content_hash=chunk.content_hash,
            language=chunk.metadata.get("language") or record.language,
            category=record.category,
            token_estimate=max(1, len(chunk.text) // 4),
        )
        session.add(row)
        added += 1
        to_embed.append(
            {
                "id": row.id,
                "record_id": record.id,
                "text": chunk.text,
                "title": record.title,
                "category": record.category,
                "subcategory": record.subcategory,
                "language": row.language,
                "verified": bool(record.verified),
                "status": record.status,
                "academic_year": record.academic_year,
                "source": record.source,
                "source_uri": record.source_uri,
                "structured": record.structured or {},
                "slug": record.slug,
                "updated_at": record.updated_at,
            }
        )

    await session.flush()
    if to_embed and record.status == "published":
        await retriever.upsert_chunks(session, to_embed)
    elif removed:
        await retriever.delete_record(record.id)
        # re-add remaining chunks so the lexical index stays in sync
        remaining = (
            (await session.execute(select(KBChunk).where(KBChunk.record_id == record.id)))
            .scalars()
            .all()
        )
        if remaining:
            await retriever.upsert_chunks(
                session,
                [
                    {
                        "id": row.id, "record_id": record.id, "text": row.text,
                        "title": record.title, "category": record.category,
                        "subcategory": record.subcategory, "language": row.language,
                        "verified": bool(record.verified), "status": record.status,
                        "academic_year": record.academic_year, "source": record.source,
                        "source_uri": record.source_uri, "structured": record.structured or {},
                        "slug": record.slug, "updated_at": record.updated_at,
                    }
                    for row in remaining
                ],
            )
    return added, removed, len(to_embed)


async def ingest_rows(
    session: AsyncSession,
    retriever: HybridRetriever,
    rows: list[dict[str, Any]],
    *,
    source: str = "upload",
    changed_by: str = "ingest",
    dry_run: bool = False,
) -> IngestReport:
    report = IngestReport(source=source, rows=len(rows))
    started = time.perf_counter()

    for row in rows:
        try:
            payload = repository.record_payload_from_row(row)
        except Exception as exc:
            report.failed += 1
            report.errors.append(f"parse error: {exc} :: {str(row)[:120]}")
            continue
        try:
            before = await repository.get_record_by_slug(session, payload["slug"])
            before_revision = before.revision if before else 0
            record, created = await repository.upsert_record(
                session, payload, changed_by=changed_by, change_note=f"sync:{source}"
            )
            if dry_run:
                await session.rollback()
                report.skipped += 1
                continue
            added, removed, embedded = await index_record(session, retriever, record)
            report.chunks_added += added
            report.chunks_removed += removed
            report.embeddings += embedded
            if created:
                report.created += 1
            elif record.revision != before_revision:
                report.updated += 1
            else:
                report.skipped += 1
            report.slugs.append(record.slug)
        except Exception as exc:
            report.failed += 1
            report.errors.append(f"{payload.get('slug', '?')}: {exc}")
            logger.exception("ingest failed for row %s", str(row)[:160])
            await session.rollback()

    if not dry_run:
        await session.commit()
    report.duration_ms = (time.perf_counter() - started) * 1000
    logger.info("ingest %s: %s", source, json.dumps({k: v for k, v in report.to_dict().items()
                                                     if k != "slugs"})[:600])
    return report


async def ingest_seed_directory(
    session: AsyncSession, retriever: HybridRetriever, directory: Path | None = None
) -> IngestReport:
    directory = directory or settings.kb_seed_path
    seeded = load_seed_files(directory)
    rows = [row for _name, row in seeded]
    report = await ingest_rows(session, retriever, rows, source=f"seed:{directory.name}")
    report.rows = len(rows)
    return report


async def sync_google_sheet(
    session: AsyncSession, retriever: HybridRetriever, url: str | None = None
) -> IngestReport:
    url = url or settings.kb_google_sheet_csv_url
    if not url:
        return IngestReport(source="google_sheet", errors=["KB_GOOGLE_SHEET_CSV_URL not set"])
    try:
        rows = await fetch_google_sheet_csv(url)
    except Exception as exc:
        logger.warning("google sheet sync failed: %s", exc)
        return IngestReport(source="google_sheet", errors=[str(exc)])
    return await ingest_rows(session, retriever, rows, source="google_sheet",
                             changed_by="sheet-sync")


async def purge(session: AsyncSession, retriever: HybridRetriever) -> dict[str, int]:
    """Drop every record + chunk + vector (used by tests and the admin reset)."""
    chunk_count = len((await session.execute(select(KBChunk.id))).scalars().all())
    record_count = len((await session.execute(select(KBRecord.id))).scalars().all())
    await session.execute(delete(KBChunk))
    await session.execute(delete(KBRecord))
    await session.commit()
    retriever.chunks.clear()
    retriever.bm25.build([])
    if retriever.store is not None:
        store_stats = await retriever.store.count()
        logger.info("purge: store still reports %s vectors", store_stats)
    return {"records": record_count, "chunks": chunk_count}


async def reindex_all(
    session: AsyncSession, retriever: HybridRetriever
) -> dict[str, Any]:
    """Rebuild the lexical + vector index from the records already in the DB."""
    started = time.perf_counter()
    records = (await session.execute(select(KBRecord))).scalars().all()
    added = removed = embedded = 0
    for record in records:
        a, r, e = await index_record(session, retriever, record)
        added += a
        removed += r
        embedded += e
    await session.commit()
    await retriever.load_from_db(session)
    return {
        "records": len(records),
        "chunks_added": added,
        "chunks_removed": removed,
        "embeddings": embedded,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "index_chunks": await retriever.count(),
    }


async def bootstrap_knowledge_base(
    session: AsyncSession, retriever: HybridRetriever
) -> dict[str, Any]:
    """Startup routine: seed (first run) or reload the index (subsequent runs)."""
    await retriever.initialise()
    existing = (await session.execute(select(KBRecord.id).limit(1))).scalar_one_or_none()

    # The seed directory is re-read on every boot, not just the first. Staff edit
    # these files (or the Google Sheet) between admission cycles and must not need
    # a redeploy or a manual purge for the change to take effect. Ingest is
    # content-hash deduped, so an unchanged file costs a hash comparison and
    # nothing else. Anything already in the database -- including records added
    # through the admin API -- is then loaded back so the in-memory index is
    # always a faithful copy of the store.
    result: dict[str, Any]
    if settings.kb_auto_ingest:
        report = await ingest_seed_directory(session, retriever)
        loaded = await retriever.load_from_db(session)
        result = {
            "mode": "seed" if existing is None else "refresh",
            "ingest": report.to_dict(),
            "chunks_loaded": loaded,
        }
    else:
        loaded = await retriever.load_from_db(session)
        result = {"mode": "reload", "chunks_loaded": loaded}

    if settings.kb_google_sheet_csv_url:
        sheet = await sync_google_sheet(session, retriever)
        result["google_sheet"] = sheet.to_dict()

    result["stats"] = await retriever.stats()
    logger.info(
        "knowledge base ready: %s",
        json.dumps(result.get("stats", {}), default=str)[:400],
    )
    return result


def schedule_sheet_sync() -> asyncio.Task[None] | None:
    """Background task that re-syncs the staff Google Sheet on an interval."""
    if not settings.kb_google_sheet_csv_url or settings.kb_sync_interval_minutes <= 0:
        return None

    async def _loop() -> None:  # pragma: no cover - long running
        from ..db import SessionLocal
        from ..dependencies import get_retriever

        while True:
            await asyncio.sleep(settings.kb_sync_interval_minutes * 60)
            try:
                async with SessionLocal() as session:
                    retriever = await get_retriever()
                    report = await sync_google_sheet(session, retriever)
                    await retriever.load_from_db(session)
                    logger.info("scheduled sheet sync: %s", report.to_dict()["updated"])
            except Exception as exc:
                logger.error("scheduled sheet sync failed: %s", exc)

    return asyncio.create_task(_loop(), name="kb-sheet-sync")
