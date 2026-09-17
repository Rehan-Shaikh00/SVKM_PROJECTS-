"""Knowledge-base admin API — what non-technical staff actually use.

Everything here re-indexes immediately, so an edit made in the dashboard or a
Google Sheet sync is live on the next call without a redeploy.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db import get_session
from ..dependencies import get_retriever
from ..kb import ingest, repository
from ..kb.chunking import chunk_record
from ..models import KBChunk, KBRecord, KBRevision
from .auth import require_admin

logger = logging.getLogger("nims.api.kb")

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

CSV_COLUMNS = [
    "slug", "category", "subcategory", "title", "academic_year", "language",
    "status", "verified", "verified_by", "source", "source_uri", "tags", "aliases",
    "degree", "school", "department", "level", "duration_years", "seats", "mode",
    "annual_fee", "total_fee", "first_year_fee", "hostel_fee", "application_fee",
    "caution_deposit", "fee_note", "eligibility", "minimum_marks", "age_limit",
    "entrance_exam", "selection_process", "curriculum", "career_options",
    "accreditation", "recognition", "highest_package", "average_package",
    "median_package", "recruiters", "placement_rate", "facilities", "room_types",
    "contact_phone", "contact_email", "address", "scholarship_types",
    "documents_required", "important_dates", "body",
]


class RecordCreate(BaseModel):
    slug: str | None = None
    title: str = Field(..., min_length=1, max_length=400)
    category: str = Field("faq")
    subcategory: str | None = None
    body: str = ""
    structured: dict[str, Any] = Field(default_factory=dict)
    language: str = "en-IN"
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    title_localized: dict[str, str] = Field(default_factory=dict)
    academic_year: str | None = None
    source: str | None = None
    source_uri: str | None = None
    verified: bool = False
    verified_by: str | None = None
    status: str = "published"
    change_note: str = ""


class RecordUpdate(RecordCreate):
    title: str = Field("", max_length=400)


class VerifyRequest(BaseModel):
    verified_by: str = ""
    change_note: str = ""


class BulkVerifyRequest(BaseModel):
    record_ids: list[str] = Field(default_factory=list)
    verified_by: str = ""


async def _chunk_counts(
    session: AsyncSession, record_ids: list[str]
) -> dict[str, int]:
    """Chunk counts in one grouped query.

    `KBRecord.chunks` is a lazy relationship; reading it from this sync
    serialiser made SQLAlchemy attempt blocking IO inside the async event loop
    and every list request failed with `MissingGreenlet`. Counting in SQL also
    avoids materialising every chunk row just to call `len()` on it.
    """
    if not record_ids:
        return {}
    rows = await session.execute(
        select(KBChunk.record_id, func.count(KBChunk.id))
        .where(KBChunk.record_id.in_(record_ids))
        .group_by(KBChunk.record_id)
    )
    return {record_id: int(count) for record_id, count in rows.all()}


def _serialise(record: KBRecord, chunk_count: int | None = None) -> dict[str, Any]:
    return {
        "id": record.id,
        "slug": record.slug,
        "category": record.category,
        "subcategory": record.subcategory,
        "title": record.title,
        "title_localized": record.title_localized or {},
        "body": record.body,
        "structured": record.structured or {},
        "language": record.language,
        "tags": record.tags or [],
        "aliases": record.aliases or [],
        "academic_year": record.academic_year,
        "source": record.source,
        "source_uri": record.source_uri,
        "verified": record.verified,
        "verified_by": record.verified_by,
        "verified_at": record.verified_at.isoformat() if record.verified_at else None,
        "status": record.status,
        "revision": record.revision,
        "stale": record.is_stale,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        "chunk_count": chunk_count,
    }


@router.get("/categories")
async def categories(_: str = Depends(require_admin)) -> dict[str, Any]:
    return {"categories": list(repository.CATEGORIES), "csv_columns": CSV_COLUMNS}


@router.get("/stats")
async def stats(
    session: AsyncSession = Depends(get_session), _: str = Depends(require_admin)
) -> dict[str, Any]:
    retriever = await get_retriever()
    return {
        "records": await repository.kb_stats(session),
        "index": await retriever.stats(),
        "seed_dir": str(settings.kb_seed_path),
        "google_sheet_configured": bool(settings.kb_google_sheet_csv_url),
        "sync_interval_minutes": settings.kb_sync_interval_minutes,
        "academic_year": settings.kb_academic_year,
        "staleness_days": settings.kb_staleness_days,
    }


@router.get("/records")
async def list_records(
    search: str | None = Query(None),
    category: str | None = Query(None),
    status: str | None = Query(None),
    language: str | None = Query(None),
    verified: bool | None = Query(None),
    academic_year: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    rows, total = await repository.list_records(
        session,
        search=search,
        category=category,
        status=status,
        language=language,
        verified=verified,
        academic_year=academic_year,
        limit=limit,
        offset=offset,
    )
    counts = await _chunk_counts(session, [r.id for r in rows])
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [
            _serialise(r, counts.get(r.id)) for r in rows
        ],
    }


@router.get("/records/{record_id}")
async def get_record(
    record_id: str,
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    record = await repository.get_record(session, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    chunks = (
        await session.execute(
            select(KBChunk)
            .where(KBChunk.record_id == record_id)
            .order_by(KBChunk.position)
        )
    ).scalars().all()
    payload = _serialise(record, len(chunks))
    payload["chunks"] = [
        {"id": chunk.id, "position": chunk.position, "text": chunk.text,
         "language": chunk.language, "tokens": chunk.token_estimate}
        for chunk in chunks
    ]
    revisions = (
        await session.execute(
            select(KBRevision)
            .where(KBRevision.record_id == record_id)
            .order_by(KBRevision.revision.desc())
            .limit(20)
        )
    ).scalars().all()
    payload["revisions"] = [
        {
            "revision": r.revision,
            "changed_by": r.changed_by,
            "change_note": r.change_note,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in revisions
    ]
    return payload


@router.post("/records", status_code=201)
async def create_record(
    body: RecordCreate,
    session: AsyncSession = Depends(get_session),
    admin: str = Depends(require_admin),
) -> dict[str, Any]:
    retriever = await get_retriever()
    payload = repository.record_payload_from_row(body.model_dump(exclude={"change_note"}))
    if body.verified:
        payload["verified_at"] = datetime.now(UTC)
    record, created = await repository.upsert_record(
        session, payload, changed_by=admin, change_note=body.change_note or "dashboard create"
    )
    if not created:
        raise HTTPException(status_code=409, detail=f"slug already exists: {record.slug}")
    added, removed, embedded = await ingest.index_record(session, retriever, record)
    await session.commit()
    await retriever.load_from_db(session)
    counts = await _chunk_counts(session, [record.id])
    return {"record": _serialise(record, counts.get(record.id)),
            "chunks_added": added, "chunks_removed": removed, "embeddings": embedded}


@router.put("/records/{record_id}")
async def update_record(
    record_id: str,
    body: RecordUpdate,
    session: AsyncSession = Depends(get_session),
    admin: str = Depends(require_admin),
) -> dict[str, Any]:
    retriever = await get_retriever()
    record = await repository.get_record(session, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    data = body.model_dump(exclude={"change_note"})
    data.pop("slug", None)  # slug is the stable identity
    if not data.get("title"):
        data["title"] = record.title
    payload = repository.record_payload_from_row(data)
    payload["slug"] = record.slug
    if payload["verified"] and not record.verified:
        payload["verified_at"] = datetime.now(UTC)
        payload["verified_by"] = payload.get("verified_by") or admin
    record, _created = await repository.upsert_record(
        session, payload, changed_by=admin, change_note=body.change_note or "dashboard update"
    )
    added, removed, embedded = await ingest.index_record(session, retriever, record)
    await session.commit()
    await retriever.load_from_db(session)
    counts = await _chunk_counts(session, [record.id])
    return {"record": _serialise(record, counts.get(record.id)),
            "chunks_added": added, "chunks_removed": removed, "embeddings": embedded}


@router.post("/records/{record_id}/verify")
async def verify_record(
    record_id: str,
    body: VerifyRequest | None = None,
    session: AsyncSession = Depends(get_session),
    admin: str = Depends(require_admin),
) -> dict[str, Any]:
    verified_by = (body.verified_by if body else "") or ""
    retriever = await get_retriever()
    record = await repository.get_record(session, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    record.verified = True
    record.verified_by = verified_by or admin
    record.verified_at = datetime.now(UTC)
    record.revision += 1
    await session.flush()
    await ingest.index_record(session, retriever, record)
    await session.commit()
    await retriever.load_from_db(session)
    counts = await _chunk_counts(session, [record.id])
    return {"record": _serialise(record, counts.get(record.id))}


@router.delete("/records/{record_id}")
async def delete_record(
    record_id: str,
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    retriever = await get_retriever()
    record = await repository.get_record(session, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    slug = record.slug
    await retriever.delete_record(record_id)
    await repository.delete_record(session, record_id)
    await session.commit()
    await retriever.load_from_db(session)
    return {"deleted": slug}


@router.post("/import")
async def import_records(
    file: UploadFile | None = File(None),
    text: str = Form(""),
    source: str = Form("dashboard-import"),
    dry_run: bool = Form(False),
    session: AsyncSession = Depends(get_session),
    admin: str = Depends(require_admin),
) -> dict[str, Any]:
    """Import CSV / YAML / JSON — the main path for bulk fee-table updates."""
    if file is not None and file.filename:
        raw = await file.read()
        content = raw.decode("utf-8-sig", errors="replace")
        filename = file.filename
    elif text.strip():
        content = text
        filename = "pasted.csv" if "," in content.split("\n")[0] else "pasted.yaml"
    else:
        raise HTTPException(status_code=400, detail="upload a file or paste content")

    try:
        rows = ingest.parse_payload(content, filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not parse: {exc}") from exc
    if not rows:
        raise HTTPException(status_code=400, detail="no records found in the payload")

    retriever = await get_retriever()
    report = await ingest.ingest_rows(
        session, retriever, rows, source=source, changed_by=admin, dry_run=dry_run
    )
    if not dry_run:
        await retriever.load_from_db(session)
    return report.to_dict()


@router.post("/sync-sheet")
async def sync_sheet(
    url: str = Form(""),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    retriever = await get_retriever()
    report = await ingest.sync_google_sheet(session, retriever, url or None)
    await retriever.load_from_db(session)
    return report.to_dict()


@router.post("/reindex")
async def reindex(
    session: AsyncSession = Depends(get_session), _: str = Depends(require_admin)
) -> dict[str, Any]:
    retriever = await get_retriever()
    return await ingest.reindex_all(session, retriever)


@router.post("/purge")
async def purge(
    confirm: str = Form(""),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict[str, Any]:
    if confirm.upper() != "PURGE":
        raise HTTPException(status_code=400, detail="send confirm=PURGE to delete every record")
    retriever = await get_retriever()
    return await ingest.purge(session, retriever)


@router.get("/export.csv")
async def export_csv(
    category: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> Any:
    """Download the KB as a CSV staff can open in Sheets, edit and re-import."""
    from fastapi.responses import StreamingResponse

    rows, _total = await repository.list_records(
        session, category=category, limit=5000
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for record in rows:
        structured = record.structured or {}
        # `fees` is usually a breakdown dict, but staff may write prose instead
        # ("NRI quota fees are denominated in US dollars…"). Export both shapes.
        raw_fees = structured.get("fees")
        if isinstance(raw_fees, dict):
            fees: dict[str, Any] = raw_fees
            fee_note = fees.get("note")
        elif raw_fees in (None, ""):
            fees = {}
            fee_note = None
        else:
            fees = {}
            fee_note = str(raw_fees)
        row: dict[str, Any] = {
            "slug": record.slug,
            "category": record.category,
            "subcategory": record.subcategory,
            "title": record.title,
            "academic_year": record.academic_year,
            "language": record.language,
            "status": record.status,
            "verified": "true" if record.verified else "false",
            "verified_by": record.verified_by,
            "source": record.source,
            "source_uri": record.source_uri,
            "tags": ";".join(record.tags or []),
            "aliases": ";".join(record.aliases or []),
            "body": record.body,
            "annual_fee": fees.get("annual"),
            "total_fee": fees.get("total"),
            "first_year_fee": fees.get("year_1"),
            "hostel_fee": fees.get("hostel"),
            "application_fee": fees.get("application"),
            "caution_deposit": fees.get("deposit"),
            "fee_note": fee_note,
        }
        for key, column in repository.STRUCTURED_COLUMNS.items():
            if "." in column:
                continue
            if key in structured:
                value = structured[key]
                row[key] = ";".join(str(v) for v in value) if isinstance(value, list) else value
        writer.writerow(row)
    buffer.seek(0)
    filename = f"nims-knowledge-base-{datetime.now(UTC):%Y%m%d}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/template.csv")
async def template_csv(_: str = Depends(require_admin)) -> Any:
    from fastapi.responses import StreamingResponse

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerow(
        {
            "slug": "course-btech-cse",
            "category": "course",
            "subcategory": "engineering",
            "title": "B.Tech Computer Science and Engineering",
            "academic_year": settings.kb_academic_year,
            "language": "en-IN",
            "status": "published",
            "verified": "true",
            "verified_by": "Admissions Office",
            "source": "Fee and Admission Handbook",
            "tags": "ug;engineering",
            "aliases": "B.Tech CSE;Computer Science Engineering;बीटेक सीएसई;संगणक अभियांत्रिकी",
            "degree": "B.Tech",
            "school": "School of Technology, Management & Engineering",
            "level": "UG",
            "duration_years": "4",
            "seats": "180",
            "mode": "Full time",
            "annual_fee": "150000",
            "total_fee": "600000",
            "hostel_fee": "85000",
            "eligibility": "10+2 with Physics, Chemistry and Mathematics, 50% aggregate",
            "entrance_exam": "NMIMS-NPAT or JEE Main",
            "body": "Overview of the programme, labs, faculty and career paths.",
        }
    )
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="nmims-dhule-kb-template.csv"'},
    )


@router.post("/preview-chunking")
async def preview_chunking(
    payload: RecordCreate, _: str = Depends(require_admin)
) -> dict[str, Any]:
    """Show staff exactly what the assistant will retrieve for this record."""
    record = payload.model_dump()
    record.setdefault("category", "faq")
    chunks = chunk_record(record)
    return {
        "chunk_count": len(chunks),
        "chunks": [
            {"position": c.position, "characters": len(c.text), "kind": c.metadata.get("chunk_kind"),
             "text": c.text}
            for c in chunks
        ],
    }


@router.post("/bulk-verify")
async def bulk_verify(
    body: BulkVerifyRequest,
    session: AsyncSession = Depends(get_session),
    admin: str = Depends(require_admin),
) -> dict[str, Any]:
    ids = body.record_ids
    verified_by = body.verified_by or ""
    retriever = await get_retriever()
    updated = 0
    now = datetime.now(UTC)
    for record_id in ids:
        record = await repository.get_record(session, record_id)
        if record is None:
            continue
        record.verified = True
        record.verified_by = verified_by or admin
        record.verified_at = now
        record.revision += 1
        await session.flush()
        await ingest.index_record(session, retriever, record)
        updated += 1
    await session.commit()
    await retriever.load_from_db(session)
    return {"updated": updated, "requested": len(ids)}
