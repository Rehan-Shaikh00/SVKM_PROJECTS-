#!/usr/bin/env python3
"""Drive the live assistant and assert what a caller actually hears.

The unit tests in `backend/tests` pin the composing rules with hand-built
retrieval results. This script pins the *whole* path — retrieval, ranking,
intent, compose, guardrails — against a running server, in all three languages
the Dhule line serves. Every case here was a real bad answer at some point.

Usage:
    cd backend && ../.venv/bin/python -m uvicorn app.main:app --port 8000
    .venv/bin/python scripts/verify_answers.py            # or --base-url ...

With ADMIN_AUTH_ENABLED=true, pass --username/--password (or set the
ADMIN_USERNAME / ADMIN_PASSWORD env vars); with auth off nothing is needed.

Exits non-zero if any case fails, so it can gate a deploy.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000"


@dataclass
class Case:
    """One caller question and what the answer must and must not do."""

    question: str
    language: str = "en-IN"
    template: str | None = None
    record: str | None = None            # substring of the record title
    contains: tuple[str, ...] = ()
    avoids: tuple[str, ...] = ()
    escalate: bool | None = None
    intent: str | None = None
    max_chars: int = 340
    note: str = ""
    followup: bool | None = None
    _extra_checks: list[Any] = field(default_factory=list, repr=False)


# --------------------------------------------------------------------------- #
# The guarantees this assistant exists to keep
# --------------------------------------------------------------------------- #

CASES: list[Case] = [
    # --- never invent money, and never name a variant nobody asked for ------- #
    Case("What is the fee for B.Tech?", template="fees_not_recorded", escalate=True,
         contains=("B.Tech",), avoids=("Cosmetic", "rupees", "lakh"),
         note="a bare degree is not its dual-degree variant"),
    Case("What is the fee for B.Tech Computer Engineering?", template="fees_not_recorded",
         record="Computer Engineering", escalate=True, avoids=("Cosmetic",)),
    Case("B.Tech fee kitni hai?", "hi-IN", template="fees_not_recorded", escalate=True,
         contains=("B.Tech",), avoids=("Cosmetic",)),
    Case("What is the fee for B.Pharm?", contains=("B.Pharm",), avoids=("MBA",),
         escalate=True),
    Case("शुल्क किती आहे?", "mr-IN", escalate=True, avoids=("Cosmetic",)),
    Case("Can I get a refund if I withdraw?", template="refund_not_published",
         escalate=True, note="no refund policy is published"),
    Case("शिक्षण कर्ज मिळेल का?", "mr-IN", template="loan_not_published", escalate=True),

    # --- programmes this university does not run ---------------------------- #
    Case("Do you offer MBBS?", template="not_offered", escalate=False,
         contains=("does not run",), note="a denial speaks prose, not an intent frame"),
    Case("Is BDS available?", template="not_offered", contains=("does not run",)),
    Case("Do you offer B.Com?", template="not_offered", contains=("BBA", "BCA"),
         note="the denial offers what the school does run"),
    Case("एमबीए चा कोर्स आहे का?", "mr-IN", template="not_offered"),
    Case("Is BBA LL.B. available here?", template="not_offered",
         note="the interceptor beats the plain BBA record"),

    # --- what is actually on offer, and its verified numbers ---------------- #
    Case("How many seats in B.Pharm?", record="Bachelor of Pharmacy",
         contains=("sixty",), avoids=("forty",),
         note="the plain record, not the 40-seat dual degree"),
    Case("What is the intake for M.Pharm Pharmaceutics?", template="specialisation_intake",
         contains=("fifteen",), note="a specialisation has its own intake"),
    Case("What is the duration of MCA?", contains=("two years", "one hundred twenty")),
    Case("बी.टेक संगणक अभियांत्रिकी साठी किती जागा आहेत?", "mr-IN",
         contains=("180",), record="Computer Engineering"),
    Case("What courses do you offer?", template="catalog",
         avoids=("overview",), note="programme names, not school-overview titles"),
    Case("अभ्यासक्रमांची यादी सांगा", "mr-IN", template="catalog", avoids=("overview",)),
    Case("Which entrance exam for M.Pharm?", template="exam", contains=("GPAT",)),
    Case("प्रवेश के लिए कौन सी परीक्षा देनी होगी?", "hi-IN",
         template="entrance_tests_generic", contains=("MHT-CET",)),

    # --- eligibility and documents ------------------------------------------ #
    Case("बीबीए की पात्रता क्या है?", "hi-IN", template="eligibility_long",
         contains=("50%",), note="Maths is not compulsory for BBA"),
    Case("What is the eligibility?", template="ask_clarify", escalate=False,
         avoids=("Mechanical", "Computer", "Pharmacy", "B.Tech", "BBA"),
         note="no programme named, so it asks instead of quoting one it picked"),
    Case("पात्रता काय आहे?", "mr-IN", template="ask_clarify", escalate=False,
         contains=("अभ्यासक्रमाविषयी",),
         note="Marathi inflects the noun before the postposition"),
    Case("What documents do I need for admission?", template="documents",
         contains=("Class 10", "Class 12")),
    Case("डॉक्यूमेंट्स की लिस्ट एसएमएस से भेजो", "hi-IN", template="documents",
         record="Documents required",
         note="SMS is not M.A.: the word for SMS contains the word for M.A."),
    Case("कागदपत्रे कोणती लागतील?", "mr-IN", template="documents"),

    # --- admission cycle: rounds are published per programme, per PDF ------- #
    Case("Has the merit list come out for BBA?", template="round_status",
         contains=("BBA",), followup=True),
    Case("What is the last date to apply for B.Tech?", template="round_status",
         avoids=("31 December", "30 June"),
         note="no single deadline exists, so none may be invented"),
    Case("Is admission still open?", record="Admission dates"),

    # --- the signed academic calendar --------------------------------------- #
    Case("When do classes start?", template="important_dates",
         record="Academic calendar",
         contains=("13 July 2026",), avoids=(", Term end",),
         note="date sentences are spoken, not glued into a run-on"),
    Case("When are the term end exams?", record="Academic calendar",
         contains=("December 2026",), escalate=False),
    Case("परीक्षा कब होगी?", "hi-IN", record="Academic calendar", intent="important_dates",
         note="not the list of entrance tests"),
    Case("सेमेस्टर कधी सुरू होईल?", "mr-IN", record="Academic calendar",
         intent="important_dates", note="मेस inside सेमेस्टर is not a mess"),
    Case("हॉस्टेल में मेस की सुविधा है?", "hi-IN", intent="hostel", escalate=True,
         note="a real mess question still reaches the hostel branch"),

    # --- accommodation and facilities: nothing is published, so say so ------ #
    Case("Is hostel available?", template="hostel_unconfirmed", escalate=True,
         avoids=("mandatory", "per year")),
    Case("हॉस्टेल फी किती आहे?", "mr-IN", template="fees_not_published", escalate=True),
    Case("Do you have a hostel for girls?", template="hostel_unconfirmed", escalate=True),
    Case("क्या लाइब्रेरी है?", "hi-IN", record="Campus facilities"),

    # --- placements: an unpublished detail is not answered with a claim ----- #
    Case("What is the highest package?", template="placements_not_published",
         escalate=True, avoids=("100",),
         note="the homepage claim says nothing about a package"),
    Case("Which companies come for placement?", template="placements_not_published",
         escalate=True),
    Case("पैकेज कितना मिलता है?", "hi-IN", template="placements_not_published",
         intent="placements", escalate=True),
    Case("Do you have placement support?", template="placements", escalate=False,
         contains=("website",), note="the claim is attributed, not asserted"),

    # --- contact, address and travel ---------------------------------------- #
    Case("What is the contact number?", template="contact_phone",
         contains=("02562 350620",), note="a number that exists on the website"),
    Case("फोन नंबर क्या है?", "hi-IN", template="contact_phone", contains=("02562",)),
    Case("संपर्क क्रमांक काय आहे?", "mr-IN", template="contact_phone", contains=("02562",)),
    Case("What is the campus address?", template="campus_address",
         contains=("Survey No. 499", "424001"),
         note="the address, not the landmarks and a spelled-out pin code"),
    Case("धुले कैंपस का पता बताओ", "hi-IN", template="campus_address",
         contains=("499",)),
    Case("How do I reach the campus by train?", template="transport_rail",
         contains=("Bhusawal",), avoids=("airport",),
         note="answer the mode of travel that was asked about"),
    Case("Can I come by bus?", template="transport_road", contains=("330 km",),
         avoids=("airport",)),
    Case("How do I reach by flight?", template="transport_air", contains=("Aurangabad",),
         avoids=("Bhusawal",)),
    Case("गाडीने कसे यावे?", "mr-IN", template="transport_road", contains=("330 km",),
         note="Marathi inflects the mode and asks 'how to come'"),
    Case("रेल्वेने कसे यावे?", "mr-IN", template="transport_rail", contains=("Bhusawal",)),
    Case("How do I reach the campus?", template="transport_summary",
         contains=("Aurangabad", "330 km"), followup=True),

    # --- accreditation and identity ----------------------------------------- #
    Case("Which NBA accredited programmes do you have?", intent="university_info",
         record="Accreditation", note="not a catalogue question"),
    Case("Is the university UGC recognised?", record="Accreditation"),
    Case("Do you have NPTEL or SWAYAM?", contains=("NPTEL", "SWAYAM"),
         avoids=("credit", "certificat", "guarantee"),
         note="the page publishes logos, so no benefit may be claimed"),

    # --- escalation is the fallback, and it reaches a person ---------------- #
    Case("My daughter was harassed on campus", escalate=True,
         contains=("person",), note="sensitive topics go straight to a human"),
    Case("I want to talk to a person", escalate=True),
    Case("What is the weather in Dhule?", note="out of scope: no invented answer"),
]


def ask(
    base_url: str, case: Case, auth_header: str | None = None
) -> tuple[bool, list[str], dict[str, Any]]:
    payload = {"question": case.question, "language": case.language, "explain": True}
    headers = {"content-type": "application/json"}
    if auth_header:
        headers["authorization"] = auth_header
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/assistant/query",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:  # pragma: no cover - server problem
        if exc.code == 401 and not auth_header:
            return False, [
                f"HTTP {exc.code}: {exc.reason} (admin auth is enabled: pass "
                "--username/--password or set ADMIN_USERNAME/ADMIN_PASSWORD)"
            ], {}
        return False, [f"HTTP {exc.code}: {exc.reason}"], {}
    except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover
        return False, [f"cannot reach {base_url}: {exc}"], {}

    answer = data.get("answer") or ""
    debug = data.get("debug") or {}
    failures: list[str] = []

    if case.template and debug.get("template") != case.template:
        failures.append(f"template={debug.get('template')!r}, want {case.template!r}")
    if case.record and case.record.lower() not in str(debug.get("record_title") or "").lower():
        failures.append(f"record={debug.get('record_title')!r}, want ~{case.record!r}")
    if case.intent and (debug.get("intent") or {}).get("intent") != case.intent:
        failures.append(f"intent={(debug.get('intent') or {}).get('intent')!r}, want {case.intent!r}")
    if case.escalate is not None and bool(data.get("needs_escalation")) != case.escalate:
        failures.append(f"needs_escalation={data.get('needs_escalation')}, want {case.escalate}")
    for needle in case.contains:
        if needle.lower() not in answer.lower():
            failures.append(f"missing {needle!r}")
    for needle in case.avoids:
        if needle.lower() in answer.lower():
            failures.append(f"must not say {needle!r}")
    if len(answer) > case.max_chars:
        failures.append(f"{len(answer)} chars > {case.max_chars} (not voice-friendly)")
    if case.followup is not None and bool(data.get("followup")) != case.followup:
        failures.append(f"followup={bool(data.get('followup'))}, want {case.followup}")
    grounding = debug.get("numeric_grounding") or {}
    if grounding and grounding.get("ok") is False:
        failures.append(f"ungrounded numbers: {grounding.get('ungrounded_numbers', [])[:4]}")
    if not answer.strip():
        failures.append("empty answer")
    return not failures, failures, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--username", default=None,
        help="admin username when auth is enabled (default: $ADMIN_USERNAME)",
    )
    parser.add_argument(
        "--password", default=None,
        help="admin password when auth is enabled (default: $ADMIN_PASSWORD)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print every answer")
    parser.add_argument("-k", dest="needle", default=None, help="only cases whose question contains this")
    args = parser.parse_args()

    username = args.username or os.environ.get("ADMIN_USERNAME") or None
    password = args.password
    if password is None:
        password = os.environ.get("ADMIN_PASSWORD")
    auth_header: str | None = None
    if username:
        raw = f"{username}:{password or ''}".encode()
        auth_header = f"Basic {base64.b64encode(raw).decode('ascii')}"

    cases = [c for c in CASES if not args.needle or args.needle.lower() in c.question.lower()]
    passed = failed = 0
    print(f"verifying {len(cases)} caller questions against {args.base_url}\n")
    for case in cases:
        ok, failures, data = ask(args.base_url, case, auth_header)
        mark = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        label = f"[{case.language}] {case.question}"
        if not ok or args.verbose:
            print(f"{mark}  {label}")
            if ok and args.verbose:
                print(f"        {(data.get('answer') or '')[:160]}")
            for reason in failures:
                print(f"        - {reason}")
            print(f"        answer: {(data.get('answer') or '')[:160]}")
            debug = data.get("debug") or {}
            print(f"        tpl={debug.get('template')} rec={debug.get('record_title')}")
            if case.note:
                print(f"        why it matters: {case.note}")
        else:
            print(f"{mark}  {label}")

    print(f"\n{passed} passed, {failed} failed, {len(cases)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
