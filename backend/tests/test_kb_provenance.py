"""Who checked this record, when, and against what.

The knowledge base is the single source of truth for what the assistant may say
aloud, so `verified: true` is a claim somebody has to be able to justify later —
to a registrar before an admission cycle, or to a caller who was told something
wrong. Three defects made that impossible:

* verifying a record bumped `revision` and wrote no revision row at all, and
  discarded the note the dashboard sent with it. The one act that decides what
  the AI is allowed to speak was the one that left no trace, so a record showed
  "revision 2" with an empty history behind it;
* creating a record wrote no revision row either, so a record's history began at
  its first edit and nobody could say what it had been compiled from;
* the seed KB marked 45 records verified with `verified_by` holding a URL — the
  evidence, not an actor — and no `verified_at`, so the dashboard's "verified"
  pill claimed a sign-off no person had given.

Unlike the rest of this suite these touch a database, because an audit trail is
only real once it has been written somewhere. It is an in-memory SQLite
database: no files, no network, no credentials, and nothing is written into the
developer's data/ directory.
"""

from __future__ import annotations

import inspect
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from app.api import kb_admin
from app.db import Base
from app.kb import repository
from app.models import KBRevision
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "kb"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session
    await engine.dispose()


def _payload(**overrides: Any) -> dict[str, Any]:
    row = {
        "slug": "hostel-wifi",
        "category": "faq",
        "title": "Campus Wi-Fi account activation",
        "body": "Every admitted student's account is activated within 24 hours.",
        "source": "https://www.svkmnmimsgu.ac.in/",
        "verified": True,
    }
    row.update(overrides)
    return repository.record_payload_from_row(row)


async def _revisions(session: AsyncSession, record_id: str) -> list[KBRevision]:
    rows = (
        await session.execute(
            select(KBRevision).where(KBRevision.record_id == record_id).order_by(KBRevision.revision)
        )
    ).scalars().all()
    return list(rows)


def _seed_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(KB_DIR.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        rows = document if isinstance(document, list) else (document or {}).get("records", [])
        records.extend(rows)
    return records


# --------------------------------------------------------------------------- #
# the audit trail
# --------------------------------------------------------------------------- #
async def test_creating_a_record_opens_its_audit_trail(session: AsyncSession) -> None:
    record, created = await repository.upsert_record(
        session, _payload(), changed_by="ingest", change_note="sync:seed:kb"
    )
    assert created is True
    history = await _revisions(session, record.id)
    assert len(history) == 1, "a new record's history used to begin at its first edit"
    assert history[0].revision == 1
    assert history[0].changed_by == "ingest"
    assert history[0].change_note == "sync:seed:kb"


async def test_the_evidence_a_record_was_checked_against_opens_its_trail(
    session: AsyncSession,
) -> None:
    """The seed's notes are worth keeping — several record what the university
    does *not* publish, which is why the assistant declines to answer it."""
    payload = _payload(
        verification_note="svkmnmimsgu.ac.in publishes no hostel information",
    )
    record, _ = await repository.upsert_record(
        session, payload, changed_by="ingest", change_note="sync:seed:kb"
    )
    history = await _revisions(session, record.id)
    assert history[0].change_note == "svkmnmimsgu.ac.in publishes no hostel information"
    # it is a note about the checking, not a column on the record
    assert not hasattr(record, "verification_note")


async def test_verifying_records_who_signed_off_and_against_what(
    session: AsyncSession,
) -> None:
    """The sequence the verify endpoint runs: snapshot, mutate, note."""
    record, _ = await repository.upsert_record(session, _payload(verified=False))
    before = repository.audit_snapshot(record)

    actor = "Registrar, SVKM NMIMS Global University"
    note = "checked against the AY 2026-27 technology admission handout"
    record.verified = True
    record.verified_by = actor
    record.verified_at = datetime(2026, 9, 18, 10, 30, tzinfo=UTC)
    record.revision += 1
    await repository.note_revision(
        session, record, changed_by=actor, change_note=note, snapshot=before
    )

    history = await _revisions(session, record.id)
    signoff = history[-1]
    assert signoff.revision == record.revision
    assert signoff.changed_by == actor
    assert signoff.change_note == note, "the note the dashboard sends must survive"
    # the snapshot is the state *before* the sign-off, so it can be rolled back to
    assert signoff.snapshot["verified"] is False
    assert record.verified is True


def test_both_verify_endpoints_write_the_trail() -> None:
    """Pinned by source, the way the call-logging tests pin their enqueue calls.

    These endpoints mutate the record directly rather than going through
    `upsert_record`, so nothing else would notice if the audit row went missing
    again — and a missing row is invisible in use, because verification still
    appears to work.
    """
    for endpoint in (kb_admin.verify_record, kb_admin.bulk_verify):
        source = inspect.getsource(endpoint)
        assert "audit_snapshot" in source, f"{endpoint.__name__} must snapshot first"
        assert "note_revision" in source, f"{endpoint.__name__} must write a revision"
        assert "change_note" in source, f"{endpoint.__name__} must keep the note"
        # the snapshot has to be taken before the record is mutated
        assert source.index("audit_snapshot") < source.index("record.verified = True")


# --------------------------------------------------------------------------- #
# dates
# --------------------------------------------------------------------------- #
def test_an_unquoted_yaml_date_is_not_silently_dropped() -> None:
    """PyYAML resolves `verified_at: 2026-09-18` to a date, not a string.

    A member of staff editing the seed by hand writes it unquoted, and the value
    used to vanish — which is how the whole KB ended up verified with no date.
    """
    parsed = repository._parse_dt(date(2026, 9, 18))
    assert isinstance(parsed, datetime)
    assert (parsed.year, parsed.month, parsed.day) == (2026, 9, 18)
    assert parsed.tzinfo is not None
    # a quoted ISO timestamp keeps its meaning too, and a string still works
    assert repository._parse_dt("2026-09-18T00:00:00+05:30") is not None
    row = repository.record_payload_from_row(
        {"slug": "x", "category": "faq", "title": "X", "verified": True,
         "verified_at": date(2026, 9, 18)}
    )
    assert row["verified_at"] is not None


# --------------------------------------------------------------------------- #
# the seed knowledge base
# --------------------------------------------------------------------------- #
def test_every_verified_seed_record_says_who_when_and_against_what() -> None:
    records = _seed_records()
    verified = [r for r in records if r.get("verified")]
    assert verified, "the seed KB is supposed to be grounded"
    for record in verified:
        slug = record["slug"]
        actor = str(record.get("verified_by") or "")
        # an actor, not the evidence: a URL answers "where from", which `source`
        # already answers, and cannot tell anyone who is accountable
        assert actor.startswith("seed:"), f"{slug}: verified_by must name an actor"
        assert record.get("verified_at"), f"{slug}: verified with no date"
        assert record.get("verification_note"), f"{slug}: the evidence was lost"
        source = str(record.get("source") or "")
        if actor == "seed:svkmnmimsgu.ac.in":
            # grounded in the university's own website, so it must point at it
            assert source.startswith("http"), f"{slug}: website-grounded with no URL"
        else:
            # the assistant's own boundaries come from nowhere on the website
            assert source.strip(), f"{slug}: no source at all"


def test_no_seed_record_claims_a_sign_off_nobody_gave() -> None:
    """`verified` on a seeded record means "compiled from the university's own
    website", not "a registrar read this". The dashboard counts anything still
    carrying a seed actor as awaiting sign-off, so the two must not be confused:
    a pill that says "verified" next to a person's name implies an approval that
    never happened."""
    people = {"admissions office", "registrar", "admin", "dev", ""}
    for record in _seed_records():
        if not record.get("verified"):
            continue
        actor = str(record.get("verified_by") or "").strip().lower()
        assert actor not in people, f"{record['slug']} claims a human sign-off"


async def test_stats_count_the_records_no_person_has_signed_off(
    session: AsyncSession,
) -> None:
    await repository.upsert_record(
        session, _payload(slug="seeded", verified_by="seed:svkmnmimsgu.ac.in")
    )
    await repository.upsert_record(
        session, _payload(slug="signed-off", title="Signed off", verified_by="Registrar")
    )
    await repository.upsert_record(session, _payload(slug="draft", verified=False))
    stats = await repository.kb_stats(session)
    assert stats["records"] == 3
    assert stats["verified"] == 2
    assert stats["unverified"] == 1
    assert stats["awaiting_signoff"] == 1, "grounded at ingest, but nobody has read it"


async def test_a_real_sign_off_takes_a_record_off_the_backlog(
    session: AsyncSession,
) -> None:
    record, _ = await repository.upsert_record(
        session, _payload(verified_by="seed:svkmnmimsgu.ac.in")
    )
    assert (await repository.kb_stats(session))["awaiting_signoff"] == 1
    before = repository.audit_snapshot(record)
    record.verified_by = "Admissions Office"
    record.revision += 1
    await repository.note_revision(
        session, record, changed_by="Admissions Office",
        change_note="confirmed for AY 2026-27", snapshot=before,
    )
    assert (await repository.kb_stats(session))["awaiting_signoff"] == 0
