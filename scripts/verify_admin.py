#!/usr/bin/env python3
"""Drive the knowledge-base admin workflow the way a member of staff would.

The brief asks for a knowledge base that non-technical staff can edit per
admission cycle without a redeploy. This walks that whole loop against a running
server and closes it: a record created through the API must be something a
caller hears on the very next question, an edit must change the answer, and a
deletion must take it away again.

Usage:
    cd backend && ../.venv/bin/python -m uvicorn app.main:app --port 8000
    .venv/bin/python scripts/verify_admin.py            # or --base-url ...

Everything it creates it deletes again, so it is safe against a real database.
Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000"

RESULTS: list[tuple[bool, str, list[str]]] = []


def record_count(stats: Any) -> int:
    """Read the KB record count out of whichever shape the endpoint returns."""
    if not isinstance(stats, dict):
        return 0
    inner = stats.get("records")
    if isinstance(inner, dict):
        return int(inner.get("records") or 0)
    return int(stats.get("records") or stats.get("total") or 0)


def check(name: str, failures: list[str]) -> None:
    RESULTS.append((not failures, name, failures))
    print(f"{'PASS' if not failures else 'FAIL'}  {name}")
    for reason in failures:
        print(f"        - {reason}")


class Client:
    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base = base_url.rstrip("/")
        self.token = token

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if content_type:
            headers["content-type"] = content_type
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        return headers

    def request(
        self, method: str, path: str, *, body: Any = None, form: dict[str, Any] | None = None,
        raw: bool = False,
    ) -> tuple[int, Any]:
        url = f"{self.base}{path}"
        data: bytes | None = None
        content_type: str | None = None
        if form is not None:
            data = urllib.parse.urlencode(form).encode()
            content_type = "application/x-www-form-urlencoded"
        elif body is not None:
            data = json.dumps(body).encode()
            content_type = "application/json"
        request = urllib.request.Request(
            url, data=data, headers=self._headers(content_type), method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
                if raw:
                    return response.status, payload.decode("utf-8", errors="replace")
                return response.status, json.loads(payload or b"null")
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            try:
                return exc.code, json.loads(payload)
            except json.JSONDecodeError:
                return exc.code, payload
        except (urllib.error.URLError, TimeoutError) as exc:
            return 0, f"cannot reach {url}: {exc}"

    def get(self, path: str, **kwargs: Any) -> tuple[int, Any]:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> tuple[int, Any]:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> tuple[int, Any]:
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> tuple[int, Any]:
        return self.request("DELETE", path, **kwargs)

    def ask(self, question: str, language: str = "en-IN") -> tuple[str, dict[str, Any]]:
        status, data = self.post(
            "/api/assistant/query",
            body={"question": question, "language": language, "explain": True},
        )
        if status != 200 or not isinstance(data, dict):
            return "", {"_status": status, "_body": data}
        return data.get("answer") or "", data.get("debug") or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token", default=None, help="admin bearer token, if auth is enabled")
    args = parser.parse_args()
    client = Client(args.base_url, args.token)

    # A distinctive fact, so there is no doubt whose record an answer came from.
    marker = uuid.uuid4().hex[:8]
    slug = f"tmp-wifi-{marker}"
    title = "Campus Wi-Fi account activation"
    first_fact = f"activated within 24 hours of registration, reference {marker}"
    second_fact = f"activated the same working day, reference {marker}"
    question = "How do I get my campus Wi-Fi account?"

    print(f"walking the staff workflow against {args.base_url}\n")

    # ---------------------------------------------------------------- #
    status, stats = client.get("/api/kb/stats")
    failures = []
    if status != 200:
        failures.append(f"GET /api/kb/stats -> {status} {str(stats)[:120]}")
    else:
        before = record_count(stats)
        if before <= 0:
            failures.append(f"knowledge base looks empty: {stats}")
    check("staff can see the state of the knowledge base", failures)

    # ---------------------------------------------------------------- #
    status, created = client.post("/api/kb/records", body={
        "slug": slug, "title": title, "category": "faq",
        "body": (f"NMIMS Global University, Dhule · FAQ\nEvery admitted student's campus "
                 f"Wi-Fi account is {first_fact}. Bring your admission letter to the IT desk "
                 f"in the library to collect the credentials."),
        "structured": {"published": True},
        "tags": ["wifi", "campus", "it-desk"],
        "aliases": ["internet", "network access"],
        "source": "verify_admin.py", "change_note": "created by the admin harness",
    })
    record_id = ""
    failures = []
    if status != 201:
        failures.append(f"POST /api/kb/records -> {status} {str(created)[:200]}")
    else:
        record = (created or {}).get("record") or {}
        record_id = record.get("id") or ""
        if not record_id:
            failures.append(f"no record id in response: {str(created)[:200]}")
        if (created or {}).get("chunks_added", 0) < 1:
            failures.append(f"record was stored but not chunked: {created}")
    check("staff can add a record from the dashboard", failures)

    # ---------------------------------------------------------------- #
    # The point of the whole requirement: no redeploy, no reindex by hand.
    answer, debug = client.ask(question)
    failures = []
    if marker not in answer:
        failures.append(f"the new fact is not in the answer: {answer[:160]!r}")
    if title.lower() not in str(debug.get("record_title") or "").lower():
        failures.append(f"answered from {debug.get('record_title')!r}, not the new record")
    check("a caller hears the new record on the very next question", failures)
    if answer:
        print(f"        heard: {answer[:150]}")

    # ---------------------------------------------------------------- #
    status, updated = client.put(f"/api/kb/records/{record_id}", body={
        "title": title, "category": "faq",
        "body": (f"NMIMS Global University, Dhule · FAQ\nEvery admitted student's campus "
                 f"Wi-Fi account is {second_fact}. Credentials are posted to the student "
                 f"portal; the IT desk no longer issues them in person."),
        "structured": {"published": True}, "source": "verify_admin.py",
        "change_note": "edited by the admin harness",
    })
    failures = []
    if status != 200:
        failures.append(f"PUT /api/kb/records/{record_id} -> {status} {str(updated)[:200]}")
    check("staff can edit that record", failures)

    answer, _debug = client.ask(question)
    failures = []
    if first_fact in answer:
        failures.append("the superseded wording is still being spoken")
    if second_fact not in answer:
        failures.append(f"the edit did not reach the caller: {answer[:160]!r}")
    check("an edit changes what the caller hears, with no redeploy", failures)

    # ---------------------------------------------------------------- #
    signoff_note = f"checked against the AY 2026-27 handout, reference {marker}"
    status, verified = client.post(
        f"/api/kb/records/{record_id}/verify",
        body={"verified_by": "registrar", "change_note": signoff_note},
    )
    failures = []
    if status != 200:
        failures.append(f"POST verify -> {status} {str(verified)[:200]}")
    else:
        record = (verified or {}).get("record") or {}
        if not record.get("verified"):
            failures.append(f"record is not marked verified: {record}")
        if record.get("verified_by") != "registrar":
            failures.append(f"verified_by is {record.get('verified_by')!r}, not the registrar")
        if not record.get("verified_at"):
            failures.append("verified with no verified_at — nobody can say when it was checked")
    check("a registrar can mark the record verified", failures)

    # ---------------------------------------------------------------- #
    # Verification is the act that decides what the assistant may say aloud, so
    # it has to leave the same trail an edit does: who, when, against what.
    status, audit = client.get(f"/api/kb/records/{record_id}")
    failures = []
    if status != 200:
        failures.append(f"GET record -> {status}")
    else:
        payload = audit if "revisions" in (audit or {}) else (audit or {}).get("record") or {}
        revisions = (audit or {}).get("revisions") or payload.get("revisions") or []
        signoffs = [r for r in revisions if (r or {}).get("changed_by") == "registrar"]
        if not signoffs:
            failures.append(
                f"the sign-off left no audit row (revisions={len(revisions)}) — "
                "the record says it is verified and nobody can say by whom"
            )
        elif signoff_note not in str(signoffs[-1].get("change_note") or ""):
            failures.append(
                f"the note the registrar wrote was dropped: {signoffs[-1].get('change_note')!r}"
            )
        if not any((r or {}).get("revision") == 1 for r in revisions):
            failures.append("creation itself is not in the trail, so it begins at the first edit")
    check("a sign-off records who, when and against what", failures)

    # ---------------------------------------------------------------- #
    # The seeded KB is grounded in the university's website, but no person has
    # read it. That has to be visible, or a "verified" pill implies an approval
    # nobody gave.
    status, listing = client.get("/api/kb/records?limit=500")
    failures = []
    seeded = []
    if status != 200:
        failures.append(f"GET /api/kb/records -> {status}")
    else:
        items = (listing or {}).get("items") or []
        seeded = [i for i in items if str((i or {}).get("verified_by") or "").startswith("seed:")]
        if not seeded:
            failures.append("no record names the actor that compiled it")
        for item in seeded[:5]:
            if not item.get("verified_at"):
                failures.append(f"{item.get('slug')}: verified with no date")
        odd = [i.get("slug") for i in items
               if i.get("verified")
               and not str(i.get("verified_by") or "").startswith(("seed:", "registrar"))]
        if odd:
            failures.append(f"verified_by holds something that is not an actor: {odd[:3]}")
    status, stats_now = client.get("/api/kb/stats")
    if status == 200:
        counts = (stats_now or {}).get("records") or {}
        backlog = counts.get("awaiting_signoff")
        if backlog is None:
            failures.append("stats does not report how many records await a person's sign-off")
        elif backlog != len(seeded):
            failures.append(f"awaiting_signoff={backlog} but {len(seeded)} records carry a seed actor")
    check("the dashboard can tell grounded-at-ingest from signed-off", failures)

    # ---------------------------------------------------------------- #
    status, listing = client.get(f"/api/kb/records?q={urllib.parse.quote(marker)}")
    failures = []
    if status != 200:
        failures.append(f"GET /api/kb/records -> {status} {str(listing)[:160]}")
    else:
        items = listing.get("items") if isinstance(listing, dict) else listing
        if not any((i or {}).get("id") == record_id for i in (items or [])):
            failures.append("the record does not come back in a search for its own marker")
    check("staff can find the record again by searching", failures)

    # ---------------------------------------------------------------- #
    status, audit = client.get(f"/api/kb/records/{record_id}")
    failures = []
    if status != 200:
        failures.append(f"GET record -> {status}")
    else:
        record = (audit or {}).get("record") or {}
        record = audit if "revisions" in audit else record
        if not record.get("revisions"):
            failures.append(
                "no change history on the record — staff cannot see who edited what"
            )
    check("every edit leaves an audit trail", failures)

    # ---------------------------------------------------------------- #
    import_buffer = io.StringIO()
    csv.writer(import_buffer).writerows([
        ["slug", "title", "category", "body", "verified"],
        ["tmp-import-" + marker, "Imported parking note", "faq",
         f"Visitors park at the north gate, reference {marker}", "false"],
    ])
    csv_text = import_buffer.getvalue()
    status, dry = client.post("/api/kb/import", form={
        "text": csv_text, "dry_run": "true", "source": "verify_admin.py",
    })
    failures = []
    if status != 200:
        failures.append(f"POST /api/kb/import (dry run) -> {status} {str(dry)[:200]}")
    else:
        rows = int((dry or {}).get("rows") or 0)
        if rows != 1:
            failures.append(f"dry run parsed {rows} rows, expected 1: {dry}")
        created_count = int((dry or {}).get("created") or 0)
        if created_count != 0:
            failures.append(f"a dry run wrote {created_count} records")
    check("a bulk CSV can be previewed without writing anything", failures)

    status, wet = client.post("/api/kb/import", form={
        "text": csv_text, "dry_run": "false", "source": "verify_admin.py",
    })
    imported_id = ""
    failures = []
    if status != 200:
        failures.append(f"POST /api/kb/import -> {status} {str(wet)[:200]}")
    elif int((wet or {}).get("created") or 0) != 1:
        failures.append(f"import did not create the row: {wet}")
    check("the same CSV imports for real", failures)

    if status == 200:
        _answer, debug = client.ask(f"Where do visitors park, reference {marker}?")
        _listing_status, listing = client.get(f"/api/kb/records?q={urllib.parse.quote(marker)}")
        items = (listing or {}).get("items") if isinstance(listing, dict) else listing
        for item in items or []:
            if (item or {}).get("slug") == f"tmp-import-{marker}":
                imported_id = item.get("id") or ""
        if not imported_id:
            check("the imported row is findable", ["imported record not found in the listing"])

    # ---------------------------------------------------------------- #
    status, exported = client.get("/api/kb/export.csv", raw=True)
    failures = []
    if status != 200:
        failures.append(f"GET /api/kb/export.csv -> {status}")
    else:
        rows = list(csv.DictReader(io.StringIO(exported)))
        if not rows:
            failures.append("export produced no rows")
        elif marker not in exported:
            failures.append("the record just added is missing from the export")
        else:
            headings = set(rows[0].keys())
            for needed in ("slug", "title", "category", "body"):
                if needed not in headings:
                    failures.append(f"export has no {needed!r} column: {sorted(headings)[:8]}")
    check("the whole KB exports as a CSV staff can open in Sheets", failures)

    status, template = client.get("/api/kb/template.csv", raw=True)
    failures = []
    if status != 200:
        failures.append(f"GET /api/kb/template.csv -> {status}")
    elif "slug" not in template.split("\n")[0]:
        failures.append(f"template has no header row: {template[:80]!r}")
    check("there is a blank template to start from", failures)

    # ---------------------------------------------------------------- #
    status, preview = client.post("/api/kb/preview-chunking", body={
        "title": title, "category": "faq", "body": "A short body. " * 40,
    })
    failures = []
    if status != 200:
        failures.append(f"POST /api/kb/preview-chunking -> {status} {str(preview)[:160]}")
    else:
        chunks = (preview or {}).get("chunks") or []
        if not chunks:
            failures.append(f"preview returned no chunks: {str(preview)[:160]}")
    check("staff can see how a record will be split before saving", failures)

    # ---------------------------------------------------------------- #
    status, reindexed = client.post("/api/kb/reindex")
    failures = []
    if status != 200:
        failures.append(f"POST /api/kb/reindex -> {status} {str(reindexed)[:160]}")
    else:
        if int((reindexed or {}).get("records") or (reindexed or {}).get("rows") or 0) <= 0:
            failures.append(f"reindex reported no records: {reindexed}")
    check("the index can be rebuilt after a bulk edit", failures)

    answer, _debug = client.ask(question)
    failures = []
    if marker not in answer:
        failures.append(f"the reindex lost the record: {answer[:160]!r}")
    check("the record still answers after a reindex", failures)

    # ---------------------------------------------------------------- #
    # Clean up everything this script created.
    for rid in filter(None, [record_id, imported_id]):
        status, body = client.delete(f"/api/kb/records/{rid}")
        if status not in (200, 204):
            check(f"cleanup of {rid}", [f"DELETE -> {status} {str(body)[:120]}"])

    answer, debug = client.ask(question)
    failures = []
    if marker in answer:
        failures.append(
            f"a deleted record is still being spoken: {answer[:140]!r} "
            f"(from {debug.get('record_title')!r})"
        )
    check("a deleted record stops being spoken", failures)

    status, after = client.get("/api/kb/stats")
    failures = []
    if status == 200:
        before = record_count(stats)
        now = record_count(after)
        if now != before:
            failures.append(f"record count moved {before} -> {now}; the harness leaked a record")
    check("the knowledge base is left exactly as it was found", failures)

    passed = sum(1 for ok, *_ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print(f"\n{passed} passed, {failed} failed, {len(RESULTS)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
