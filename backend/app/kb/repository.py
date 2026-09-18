"""Knowledge-base repository: CRUD + querying used by the admin API and ingest.

Records are the unit staff edit; chunks are derived and never edited by hand.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, date, datetime, time
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import KBChunk, KBRecord, KBRecordStatus, KBRevision, new_id

logger = logging.getLogger("nims.kb")

CATEGORIES = (
    "university", "course", "specialisation", "eligibility", "fees",
    "admission_process", "important_dates", "entrance_exam", "scholarships",
    "hostel", "placements", "facilities", "documents", "contact", "transport",
    "loan_payment", "policy", "department", "faq",
)

#: CSV/Sheet column -> structured field mapping (keeps the sheet human friendly)
STRUCTURED_COLUMNS: dict[str, str] = {
    "degree": "degree", "programme": "programme", "school": "school",
    "department": "department", "duration_years": "duration_years",
    "duration": "duration", "seats": "seats", "intake": "seats", "mode": "mode",
    "medium": "medium", "level": "level",
    "annual_fee": "fees.annual", "first_year_fee": "fees.year_1",
    "total_fee": "fees.total", "hostel_fee": "fees.hostel",
    "caution_deposit": "fees.deposit", "application_fee": "fees.application",
    "exam_fee": "fees.exam", "fee_note": "fees.note",
    "eligibility": "eligibility", "minimum_marks": "minimum_marks",
    "age_limit": "age_limit", "entrance_exam": "entrance_exam",
    "selection_process": "selection_process", "curriculum": "curriculum",
    "career_options": "career_options", "accreditation": "accreditation",
    "recognition": "recognition", "highest_package": "highest_package",
    "average_package": "average_package", "median_package": "median_package",
    "recruiters": "recruiters", "placement_rate": "placement_rate",
    "facilities": "facilities", "room_types": "room_types",
    "contact_phone": "contact_phone", "contact_email": "contact_email",
    "address": "address", "scholarship_types": "scholarship_types",
    "documents_required": "documents_required", "important_dates": "important_dates",
}

INT_FIELDS = {"seats", "duration_years"}
FLOAT_FIELDS = {
    "fees.annual", "fees.year_1", "fees.total", "fees.hostel", "fees.deposit",
    "fees.application", "fees.exam", "application_fee",
}
LIST_FIELDS = {
    "curriculum", "career_options", "facilities", "room_types", "recruiters",
    "scholarship_types", "documents_required", "important_dates", "aliases", "tags",
}


def slugify(text: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", (text or "").strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:140] or hashlib.sha1((text or "record").encode()).hexdigest()[:12]


def _coerce(key: str, value: Any) -> Any:
    if value is None or value == "":
        return None
    if key in INT_FIELDS:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value
    if key in FLOAT_FIELDS:
        try:
            text = str(value).replace(",", "").replace("₹", "").replace("INR", "").strip()
            multiplier = 1.0
            lowered = text.lower()
            if "lakh" in lowered or "lac" in lowered:
                multiplier, text = 100_000.0, lowered.split("lakh")[0].split("lac")[0]
            elif "crore" in lowered or "cr" in lowered:
                multiplier, text = 10_000_000.0, lowered.split("crore")[0].split("cr")[0]
            return float(text.strip()) * multiplier
        except (TypeError, ValueError):
            return value
    if key in LIST_FIELDS:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [part.strip() for part in str(value).replace("|", ";").split(";") if part.strip()]
    if isinstance(value, str):
        return value.strip()
    return value


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def record_payload_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise a CSV/YAML/API row into a KBRecord payload."""
    structured: dict[str, Any] = {}
    if row.get("structured"):
        raw = row["structured"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        if isinstance(raw, dict):
            structured.update(raw)

    for column, path in STRUCTURED_COLUMNS.items():
        if column in row and row[column] not in (None, ""):
            value = _coerce(path, row[column])
            if value is not None:
                _set_path(structured, path, value)

    aliases = row.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.replace("|", ";").split(";") if a.strip()]
    tags = row.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.replace("|", ";").split(";") if t.strip()]

    title_localized = row.get("title_localized") or {}
    if isinstance(title_localized, str):
        try:
            title_localized = json.loads(title_localized)
        except json.JSONDecodeError:
            title_localized = {}

    title = str(row.get("title") or "").strip()
    slug = str(row.get("slug") or "").strip() or slugify(
        f"{title}-{row.get('academic_year') or ''}-{row.get('category') or 'record'}"
    )
    category = str(row.get("category") or "faq").strip().lower()
    if category not in CATEGORIES:
        logger.warning("unknown category '%s' for %s — keeping as-is", category, slug)

    verified = row.get("verified")
    if isinstance(verified, str):
        verified = verified.strip().lower() in {"true", "yes", "y", "1"}
    verified = bool(verified)

    payload: dict[str, Any] = {
        "slug": slug,
        "category": category,
        "subcategory": (row.get("subcategory") or None),
        "title": title or slug,
        "title_localized": title_localized or {},
        "body": str(row.get("body") or row.get("description") or "").strip(),
        "structured": structured,
        "language": str(row.get("language") or "en-IN"),
        "tags": tags,
        "aliases": aliases,
        "academic_year": (str(row.get("academic_year")).strip() if row.get("academic_year") else None),
        "source": (row.get("source") or None),
        "source_uri": (row.get("source_uri") or row.get("url") or None),
        "verified": verified,
        "verified_by": (row.get("verified_by") or None),
        "status": str(row.get("status") or KBRecordStatus.PUBLISHED.value).lower(),
    }
    verified_at = row.get("verified_at")
    if verified_at:
        payload["verified_at"] = _parse_dt(verified_at)
    if row.get("verification_note"):
        payload["verification_note"] = str(row["verification_note"]).strip()
    for key in ("effective_from", "effective_to"):
        if row.get(key):
            payload[key] = _parse_dt(row[key])
    return payload


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        # PyYAML resolves an unquoted `verified_at: 2026-09-17` to a date, not a
        # string. Falling through to None here dropped it silently, which is how
        # the whole seed KB ended up verified with no date attached -- and a
        # member of staff editing YAML by hand would write it unquoted.
        return datetime.combine(value, time.min, tzinfo=UTC)
    if isinstance(value, str):
        from dateutil import parser as date_parser

        try:
            parsed = date_parser.parse(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #


async def get_record(session: AsyncSession, record_id: str) -> KBRecord | None:
    return await session.get(KBRecord, record_id)


async def get_record_by_slug(session: AsyncSession, slug: str) -> KBRecord | None:
    result = await session.execute(select(KBRecord).where(KBRecord.slug == slug))
    return result.scalar_one_or_none()


async def list_records(
    session: AsyncSession,
    *,
    search: str | None = None,
    category: str | None = None,
    status: str | None = None,
    language: str | None = None,
    verified: bool | None = None,
    academic_year: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[KBRecord], int]:
    query = select(KBRecord)
    if category:
        query = query.where(KBRecord.category == category)
    if status:
        query = query.where(KBRecord.status == status)
    if language:
        query = query.where(KBRecord.language == language)
    if verified is not None:
        query = query.where(KBRecord.verified == verified)
    if academic_year:
        query = query.where(KBRecord.academic_year == academic_year)
    if search:
        like = f"%{search.strip()}%"
        query = query.where(
            or_(
                KBRecord.title.ilike(like),
                KBRecord.body.ilike(like),
                KBRecord.slug.ilike(like),
            )
        )
    total = (
        await session.execute(select(func.count()).select_from(query.subquery()))
    ).scalar_one()
    rows = (
        await session.execute(
            query.order_by(KBRecord.updated_at.desc()).limit(limit).offset(offset)
        )
    ).scalars().all()
    return list(rows), int(total)


async def upsert_record(
    session: AsyncSession,
    payload: dict[str, Any],
    *,
    changed_by: str = "ingest",
    change_note: str = "",
) -> tuple[KBRecord, bool]:
    """Insert or update by slug. Returns (record, created)."""
    payload = dict(payload)
    # A seed row may carry the evidence its verification rests on -- "the handouts
    # publish no fee column", "the site publishes no hostel information". That is
    # an audit note about how the record was checked, not a column on it.
    verification_note = str(payload.pop("verification_note", "") or "").strip()
    slug = payload["slug"]
    existing = await get_record_by_slug(session, slug)
    created = existing is None
    if created:
        record = KBRecord(id=new_id(), **payload, created_by=changed_by, revision=1)
        session.add(record)
        await session.flush()
        # Creation is audited as well. Without this row a seeded record showed
        # `revision: 1` against an empty history, so nobody could tell what it had
        # been compiled from, or when -- and the audit trail began at the first
        # edit, which is exactly backwards for a knowledge base whose whole
        # purpose is answering only from checked sources.
        session.add(
            KBRevision(
                id=new_id(),
                record_id=record.id,
                revision=record.revision,
                changed_by=changed_by,
                change_note=verification_note or change_note or "created",
                snapshot=_snapshot(record),
            )
        )
        await session.flush()
    elif content_fingerprint(existing) == content_fingerprint(payload):
        # Nothing actually changed. Leave revision/updated_at untouched so the
        # staleness clock keeps running from the real last edit.
        return existing, False
    else:
        assert existing is not None
        record = existing
        snapshot = _snapshot(record)
        for key, value in payload.items():
            if key in {"id", "created_at", "created_by"}:
                continue
            setattr(record, key, value)
        record.revision = (record.revision or 1) + 1
        record.updated_at = datetime.now(UTC)
        session.add(
            KBRevision(
                id=new_id(),
                record_id=record.id,
                revision=record.revision,
                changed_by=changed_by,
                change_note=change_note or ("created" if created else "updated"),
                snapshot=snapshot,
            )
        )
        await session.flush()
    return record, created


#: Fields that define the *content* of a record. Bookkeeping columns (revision,
#: timestamps, who changed it) are deliberately excluded.
_CONTENT_FIELDS = (
    "title", "category", "subcategory", "body", "structured", "academic_year",
    "verified", "verified_by", "status", "tags", "aliases", "language",
    "source", "source_uri", "effective_from", "effective_to",
)


def _normalise(value: Any) -> Any:
    """Canonicalise a value so the same content hashes identically.

    SQLite returns naive datetimes for columns written as timezone-aware ones, so
    "2025-04-01 00:00:00" from the database and "2025-04-01 00:00:00+00:00" from
    the seed YAML are the same instant spelled differently. Comparing them as
    strings made every record with an effective_from/effective_to look changed on
    each boot.
    """
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalise(v) for v in value]
    return value


def content_fingerprint(obj: KBRecord | dict[str, Any]) -> str:
    """Stable hash of a record's content, for change detection.

    Boot re-ingests the seed directory every time so staff edits land without a
    redeploy. Without this fingerprint an unchanged file still bumped `revision`,
    appended a `KBRevision` row and reset `updated_at` -- which would have made
    `updated_at` useless and silently defeated stale-content detection, the very
    guard that stops last year's fee figures being quoted as current.
    """
    def read(key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    payload = {key: _normalise(read(key)) for key in _CONTENT_FIELDS}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _snapshot(record: KBRecord) -> dict[str, Any]:
    return {
        "title": record.title,
        "category": record.category,
        "subcategory": record.subcategory,
        "body": record.body,
        "structured": record.structured,
        "academic_year": record.academic_year,
        "verified": record.verified,
        "verified_by": record.verified_by,
        "status": record.status,
        "tags": record.tags,
        "aliases": record.aliases,
        "revision": record.revision,
    }


def audit_snapshot(record: KBRecord) -> dict[str, Any]:
    """The record's values *before* a change, for the revision row about to be
    written. Callers outside :func:`upsert_record` -- verification, say -- take
    this first, then mutate, then hand it to :func:`note_revision`."""
    return _snapshot(record)


async def note_revision(
    session: AsyncSession,
    record: KBRecord,
    *,
    changed_by: str | None,
    change_note: str,
    snapshot: dict[str, Any] | None = None,
) -> None:
    """Append an audit row for a change this module did not make itself."""
    session.add(
        KBRevision(
            id=new_id(),
            record_id=record.id,
            revision=record.revision,
            changed_by=changed_by,
            change_note=change_note,
            snapshot=snapshot or _snapshot(record),
        )
    )
    await session.flush()


async def delete_record(session: AsyncSession, record_id: str) -> bool:
    record = await session.get(KBRecord, record_id)
    if not record:
        return False
    await session.delete(record)
    await session.flush()
    return True


async def kb_stats(session: AsyncSession) -> dict[str, Any]:
    total = (await session.execute(select(func.count(KBRecord.id)))).scalar_one()
    by_category_rows = (
        await session.execute(
            select(KBRecord.category, func.count(KBRecord.id)).group_by(KBRecord.category)
        )
    ).all()
    by_status_rows = (
        await session.execute(
            select(KBRecord.status, func.count(KBRecord.id)).group_by(KBRecord.status)
        )
    ).all()
    verified = (
        await session.execute(select(func.count(KBRecord.id)).where(KBRecord.verified.is_(True)))
    ).scalar_one()
    chunks = (await session.execute(select(func.count(KBChunk.id)))).scalar_one()
    stale_rows = (
        await session.execute(
            select(KBRecord).where(KBRecord.status == KBRecordStatus.PUBLISHED.value)
        )
    ).scalars().all()
    stale = sum(1 for row in stale_rows if row.is_stale)
    # Grounded at ingest from the university's own website, but never signed off
    # by a person. The assistant may speak from these -- that is what verified
    # means here -- yet a registrar should confirm them before an admission
    # cycle, and this count is what tells staff how much of the KB is in that
    # state. Anything still carrying a "seed:" actor has not been looked at.
    awaiting_signoff = (
        await session.execute(
            select(func.count(KBRecord.id)).where(
                KBRecord.verified.is_(True),
                KBRecord.verified_by.like("seed:%"),
            )
        )
    ).scalar_one()
    return {
        "records": int(total),
        "chunks": int(chunks),
        "verified": int(verified),
        "unverified": int(total) - int(verified),
        "awaiting_signoff": int(awaiting_signoff),
        "stale": int(stale),
        "by_category": {str(c): int(n) for c, n in by_category_rows},
        "by_status": {str(s): int(n) for s, n in by_status_rows},
    }
