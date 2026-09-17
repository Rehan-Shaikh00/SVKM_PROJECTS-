"""Regression tests for what a Dhule caller actually hears.

These pin the defects found by driving the live assistant in English, Hindi and
Marathi against the seeded KB, rather than the plumbing around it. Each one was
a real bad answer:

* a facts chunk read aloud as a spreadsheet ("Degree: B.Tech / Seats: 60 / …");
* an English paragraph spliced onto the end of a Marathi sentence;
* a fee question answered with a number nobody could source;
* "how do I reach the campus" answered as "The campus has The campus is …";
* a contact question answered with the record's *title* as the department;
* staff-facing instructions ("the assistant must never …") spoken to a caller;
* an alias written for recall hijacking a neighbouring record's question.

Everything here is unit-level: no database, no network, no API keys.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from app.ai.intents import detect_intent
from app.ai.templates import (
    DEPARTMENT_FALLBACK,
    FRAMES,
    MAX_SPOKEN_CHARS,
    PROGRAMME_NEUTRAL,
    _script_compatible,
    _sentences,
    compose,
)
from app.kb.retriever import ChunkDoc, HybridRetriever, RetrievalResult, RetrievedChunk

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "kb"
DEVANAGARI = re.compile(r"[\u0900-\u097F]")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _chunk(
    *,
    text: str,
    title: str = "Hostel accommodation",
    category: str = "hostel",
    structured: dict[str, Any] | None = None,
    verified: bool = False,
    chunk_id: str = "c1",
    record_id: str = "r1",
    score: float = 0.85,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        record_id=record_id,
        text=text,
        title=title,
        category=category,
        language="en-IN",
        verified=verified,
        academic_year="2026-27",
        score=score,
        structured=structured or {},
    )


def _retrieval(question: str, chunk: RetrievedChunk) -> RetrievalResult:
    return RetrievalResult(
        query=question,
        items=[chunk],
        intent=detect_intent(question),
        best_score=chunk.score,
        grounded=True,
    )


def _kb_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(KB_DIR.glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        for record in loaded or []:
            record["_file"] = path.name
            records.append(record)
    return records


# --------------------------------------------------------------------------- #
# spoken text hygiene
# --------------------------------------------------------------------------- #
def test_label_and_field_dumps_are_never_spoken() -> None:
    facts = (
        "Also known as: hostel, वसतिगृह. Degree: B.Tech. Seats: 60. "
        "Fees — total fee INR 5.10 lakh. Eligibility: Class 12 with 50%."
    )
    assert _sentences(facts, limit=8) == []


def test_prose_survives_the_dump_filter() -> None:
    prose = "The university provides hostel accommodation with separate blocks. A mess is available."
    assert len(_sentences(prose, limit=8)) == 2


@pytest.mark.parametrize("language", ["mr-IN", "hi-IN", "raj-IN"])
def test_script_filter_rejects_latin_prose_for_devanagari_calls(language: str) -> None:
    assert not _script_compatible("Confirm the current room types with the hostel office.", language)
    assert _script_compatible("वसतिगृह उपलब्ध आहे.", language)
    # numbers and punctuation alone are script-neutral
    assert _script_compatible("1800 102 5138", language)


def test_script_filter_leaves_english_calls_alone() -> None:
    assert _script_compatible("Hostel accommodation is available.", "en-IN")
    assert _script_compatible("Hostel accommodation is available.", None)


def test_marathi_hostel_answer_is_pure_marathi() -> None:
    chunk = _chunk(
        text=(
            "NMIMS Global University, Dhule · Hostel\n"
            "The university provides hostel accommodation with separate blocks for male "
            "and female students. Confirm the current room types with the hostel office."
        ),
        structured={"hostel_available": True, "summary": "Separate blocks for boys and girls."},
    )
    answer = compose(_retrieval("हॉस्टेल ची सोय आहे का", chunk),
                     language="mr-IN", question="हॉस्टेल ची सोय आहे का")
    assert answer.text.startswith("होय, वसतिगृह उपलब्ध आहे")
    # no English prose tail, and no staff instruction lifted from the record
    assert "Confirm the current room types" not in answer.text
    assert not re.search(r"[A-Za-z]{4,}", answer.text.replace("SMS", ""))


def test_english_only_kb_degrades_to_flagged_english_not_an_escalation() -> None:
    chunk = _chunk(
        text=(
            "NMIMS Global University, Dhule · Transport\n"
            "The campus is directly on the Mumbai Agra Highway, National Highway 3, "
            "so it is straightforward to reach by road."
        ),
        title="Getting to campus",
        category="transport",
    )
    answer = compose(_retrieval("कॅम्पस ला कसे पोहोचावे", chunk),
                     language="mr-IN", question="कॅम्पस ला कसे पोहोचावे")
    assert not answer.needs_escalation
    assert answer.fallback_language is True
    assert "Mumbai Agra Highway" in answer.text
    # a bare record title is not an answer
    assert answer.text.strip() != "Getting to campus"


def test_transport_prose_is_not_wrapped_in_the_facilities_frame() -> None:
    chunk = _chunk(
        text=(
            "NMIMS Global University, Dhule · Facilities\n"
            "The campus is directly on the Mumbai Agra Highway, National Highway 3."
        ),
        title="Campus facilities",
        category="facilities",
        structured={},
    )
    question = "how do I reach the campus from Mumbai"
    answer = compose(_retrieval(question, chunk), language="en-IN", question=question)
    assert "The campus has The campus is" not in answer.text
    assert not answer.text.startswith("The campus has")
    assert "Mumbai Agra Highway" in answer.text


def test_facilities_list_is_spoken_from_the_structured_field() -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Facilities\nListed facilities include academic blocks.",
        title="Campus facilities",
        category="facilities",
        structured={"facilities": ["Academic blocks", "Laboratories", "Library", "Sports ground"]},
    )
    answer = compose(_retrieval("what facilities does the campus have", chunk),
                     language="en-IN", question="what facilities does the campus have")
    assert answer.text.startswith("The campus has Academic blocks, Laboratories and Library")
    assert answer.followup and answer.followup["items"] == [
        "Academic blocks", "Laboratories", "Library", "Sports ground",
    ]


# --------------------------------------------------------------------------- #
# fees: never invent a number
# --------------------------------------------------------------------------- #
def test_fee_question_with_a_recorded_total_is_spoken_in_words() -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Course\nBBA is a three year full time programme.",
        title="BBA (Bachelor of Business Administration)",
        category="course",
        structured={"fees": {"total": 450000}, "duration_years": 3, "eligibility": "Class 12"},
    )
    answer = compose(_retrieval("what is the BBA fee", chunk),
                     language="en-IN", question="what is the BBA fee")
    assert "four lakh fifty thousand" in answer.text
    assert answer.needs_escalation is False


@pytest.mark.parametrize(
    ("language", "question", "expected"),
    [
        ("mr-IN", "B.Tech CSE ची फी किती आहे", "नोंदलेली नाही"),
        ("hi-IN", "B.Tech CSE की फीस कितनी है", "नहीं है"),
        ("en-IN", "what is the B.Tech CSE fee", "do not have the confirmed fee"),
    ],
)
def test_fee_question_without_a_recorded_number_escalates_natively(
    language: str, question: str, expected: str
) -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Course\nB.Tech CSE is a four year programme.",
        title="B.Tech Computer Science & Engineering (CSE)",
        category="course",
        structured={"duration_years": 4, "eligibility": "Class 12 with PCM"},
    )
    answer = compose(_retrieval(question, chunk),
                     language=language, question=question)
    assert answer.needs_escalation is True
    assert answer.escalation_reason == "fee_not_in_kb"
    assert expected in answer.text
    assert "0" not in answer.text and "lakh" not in answer.text.lower()
    # no contradictory "last published figure" tail, and no eligibility trivia
    # bolted onto a hand-off
    assert "last published figure" not in answer.text
    assert "Class 12" not in answer.text


# --------------------------------------------------------------------------- #
# hostel, contact, scholarships frames
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("language", ["en-IN", "hi-IN", "mr-IN"])
@pytest.mark.parametrize("available", [True, False])
def test_hostel_availability_is_spoken_in_the_callers_language(
    language: str, available: bool
) -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Hostel\nThe university provides hostel accommodation.",
        structured={"hostel_available": available},
    )
    answer = compose(_retrieval("is hostel available", chunk),
                     language=language, question="is hostel available")
    assert answer.template in {"hostel_available", "hostel_none"}
    assert answer.text == FRAMES[language]["hostel_available" if available else "hostel_none"]
    if language != "en-IN":
        assert DEVANAGARI.search(answer.text)


def test_hostel_staff_instructions_are_not_spoken() -> None:
    chunk = _chunk(
        text=(
            "NMIMS Global University, Dhule · Hostel\n"
            "The university provides hostel accommodation with separate blocks."
        ),
        structured={
            "hostel_available": True,
            "assistant_instruction": "Do not quote a hostel charge; connect to the hostel office.",
        },
    )
    answer = compose(_retrieval("is hostel available", chunk),
                     language="en-IN", question="is hostel available")
    assert "Do not quote" not in answer.text
    assert "assistant" not in answer.text.lower()


@pytest.mark.parametrize(
    ("language", "expected_department"),
    [("en-IN", "admissions office"), ("mr-IN", "प्रवेश कार्यालय"), ("hi-IN", "प्रवेश कार्यालय")],
)
def test_contact_frame_uses_a_localised_department_not_the_record_title(
    language: str, expected_department: str
) -> None:
    """The website publishes school phone numbers and no email, so the frame names the
    department in the caller's language next to the number, and the whole card of
    school-wise numbers goes by SMS instead of being read aloud."""
    chunk = _chunk(
        text="SVKM NMIMS Global University, Dhule · Contact\nThe university publishes no email address.",
        title="Admissions contact details and helpline",
        category="contact",
        structured={
            "contact_phone": "02562 350620",
            "department_localized": {
                "en-IN": "admissions office",
                "hi-IN": "प्रवेश कार्यालय",
                "mr-IN": "प्रवेश कार्यालय",
            },
            "school_phones": [
                "School of Technology, Management & Engineering: 02562 350620",
                "School of Pharmacy & Technology Management: 02562 350640",
                "School of Commerce: 02562 350600",
            ],
            "admission_portal": "https://sdcappscs.svkm.ac.in:44300/irj/portal",
            "website": "https://www.svkmnmimsgu.ac.in",
        },
    )
    answer = compose(_retrieval("what is the contact number for admissions", chunk),
                     language=language, question="what is the contact number for admissions")
    assert expected_department in answer.text
    assert "Admissions contact details and helpline" not in answer.text
    assert "02562 350620" in answer.text
    # no email is published, so none may be invented
    assert "@" not in answer.text
    # the full card goes by SMS, not down the phone
    assert answer.followup and answer.followup["channel"] == "sms"
    assert any("02562 350640" in item for item in answer.followup["items"])
    assert any("sdcappscs.svkm.ac.in" in item for item in answer.followup["items"])


def test_department_fallback_covers_every_scripted_language() -> None:
    assert set(DEPARTMENT_FALLBACK) == {"en", "hi", "mr", "raj"}
    assert all(value.strip() for value in DEPARTMENT_FALLBACK.values())


def test_scholarships_frame_leaves_no_trailing_gap() -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Scholarships\nScholarships are announced each cycle.",
        title="Scholarships and concessions",
        category="scholarships",
        structured={"scholarship_types": ["Merit based concessions", "State post matric schemes"]},
    )
    answer = compose(_retrieval("शिष्यवृत्ती उपलब्ध आहे का", chunk),
                     language="mr-IN", question="शिष्यवृत्ती उपलब्ध आहे का")
    assert answer.template == "scholarships"
    assert "  " not in answer.text
    assert not answer.text.endswith(" .")
    assert not answer.text.endswith("। ")
    assert "Merit based concessions" in answer.text


def test_every_language_has_the_same_frames() -> None:
    keys = {lang: set(frames) for lang, frames in FRAMES.items()}
    assert len(set(map(frozenset, keys.values()))) == 1
    assert set(FRAMES) == {"en-IN", "hi-IN", "mr-IN", "raj-IN"}


# --------------------------------------------------------------------------- #
# intent routing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("what is the contact number for admissions", "contact"),
        ("प्रवेशासाठी संपर्क क्रमांक काय आहे", "contact"),
        ("एडमिशन का helpline number क्या है", "contact"),
        ("how do I reach the campus from Mumbai", "transport"),
        ("nearest railway station", "transport"),
        ("कॅम्पस ला कसे पोहोचावे", "transport"),
        ("कैंपस कैसे पहुंचें", "transport"),
        ("हॉस्टल की सुविधा है क्या", "hostel"),
        ("हॉस्टेल ची सोय आहे का", "hostel"),
        ("what facilities does the campus have", "facilities"),
        ("कॅम्पस मध्ये काय सुविधा आहेत", "facilities"),
        ("what is the admission process", "admission_process"),
        ("BBA की फीस कितनी है", "fees"),
    ],
)
def test_intent_routing(question: str, expected: str) -> None:
    assert detect_intent(question).intent == expected


# --------------------------------------------------------------------------- #
# retrieval ranking
# --------------------------------------------------------------------------- #
def _doc(chunk_id: str, title: str, category: str, body: str, structured: dict | None = None) -> ChunkDoc:
    return ChunkDoc(
        chunk_id=chunk_id,
        record_id=chunk_id,
        text=f"NMIMS Global University, Dhule · {title}\n{body}",
        title=title,
        category=category,
        subcategory=None,
        language="en-IN",
        verified=False,
        status="published",
        academic_year="2026-27",
        source=None,
        source_uri=None,
        structured=structured or {},
        updated_at=None,
        slug=chunk_id,
    )


def _rerank(
    docs: list[ChunkDoc],
    intent,
    categories: tuple[str, ...],
    *,
    equal_scores: bool = True,
    course_tokens: list[str] | None = None,
    query: str = "hostel",
    language: str = "en-IN",
):
    retriever = HybridRetriever()
    retriever.chunks = {d.chunk_id: d for d in docs}
    fused = {
        d.chunk_id: {"rrf_norm": 0.8, "dense": 0.8, "lexical": 4.0} for d in docs
    }
    return retriever._rerank(
        fused, intent=intent, language=language, categories=categories,
        course_tokens=course_tokens if course_tokens is not None else [], query=query,
    )


def test_alias_dump_chunks_rank_below_the_prose_they_advertise() -> None:
    intent = detect_intent("हॉस्टल की सुविधा है क्या")
    docs = [
        _doc("fac-alias", "Campus facilities", "facilities",
             "Also known as: facilities, library, सुविधा, प्रयोगशाळा."),
        _doc("hostel-prose", "Hostel accommodation", "hostel",
             "The university provides hostel accommodation with separate blocks."),
    ]
    ranked = _rerank(docs, intent, ("hostel", "facilities"))
    assert ranked[0].record_id == "hostel-prose"
    assert ranked[0].signals.get("primary_category_match") is True
    assert any(item.signals.get("alias_dump") for item in ranked)


def test_degree_prefix_exactness_beats_a_facts_chunk_mentioning_the_fee() -> None:
    intent = detect_intent("what is the fee for BBA")
    docs = [
        _doc("bba", "BBA (Bachelor of Business Administration)", "course",
             "BBA is a three year full time programme."),
        _doc("bballb", "BBA LL.B. (Honours) — integrated law", "course",
             "Fees — total fee INR 5.10 lakh."),
    ]
    retriever = HybridRetriever()
    retriever.chunks = {d.chunk_id: d for d in docs}
    ranked = retriever._rerank(
        {d.chunk_id: {"rrf_norm": 0.8, "dense": 0.8, "lexical": 4.0} for d in docs},
        intent=intent, language="en-IN", categories=("course",),
        course_tokens=intent.course_tokens, query="what is the fee for BBA",
    )
    assert ranked[0].record_id == "bba"


NOT_OFFERED_STRUCTURED = {
    "not_offered": [
        "B.Com (Hons) and its specialisations, including ACCA, CMA and International Accounting",
        "M.Com",
        "BBA LL.B. (Hons) or any law degree",
        "A standalone MBA",
    ],
    "offered_instead": ["BBA (3 years, intake 60)", "BCA (3 years, intake 60)"],
}


def _interceptor_docs() -> list[ChunkDoc]:
    return [
        _doc(
            "faq-not-offered",
            "Commerce, management and law programmes this university does not run",
            "faq",
            "Also known as: B.Com, M.Com, MBA, law degree.\n"
            "The university does not run these programmes at Dhule.",
            structured=NOT_OFFERED_STRUCTURED,
        ),
        _doc(
            "bba",
            "Bachelor of Business Administration (BBA)",
            "course",
            "BBA is a three year full time programme with an intake of sixty.",
            structured={"seats": 60, "duration_years": 3},
        ),
        _doc(
            "phd",
            "Ph.D. in Technology and Engineering",
            "course",
            "The doctoral programme runs for a minimum of three years.",
        ),
    ]


def test_a_not_offered_interceptor_outranks_the_degree_records_it_intercepts() -> None:
    """`not_offered` is a record's own declaration of the questions it intercepts.

    Without a decisive signal the intent layer routes "do you offer B.Com?" to
    `courses`, and the caller was told about the BBA record — the university's
    answer to that question is "we do not run it".
    """
    intent = detect_intent("Do you offer B.Com?")
    ranked = _rerank(
        _interceptor_docs(), intent, ("course", "faq"),
        course_tokens=intent.course_tokens, query="Do you offer B.Com?",
    )
    assert ranked[0].record_id == "faq-not-offered"
    assert ranked[0].signals.get("programme_not_offered")


def test_the_interceptor_matches_degree_names_not_its_own_prose() -> None:
    """Two regressions in one guard.

    A substring match over the whole `not_offered` prose let "BBA LL.B. (Hons) or
    any law degree" catch a plain BBA question, so a caller asking about BBA
    eligibility was told the university does not run BBA. And the loop that read
    the list shadowed `entry` from `fused.items()`, which raised AttributeError on
    every query that reached the signal at all.
    """
    for question, expected in (
        ("What is the eligibility for BBA?", "bba"),
        ("How many seats in BBA?", "bba"),
        ("Is BBA LL.B. available here?", "faq-not-offered"),
        ("Do you offer a standalone MBA?", "faq-not-offered"),
        ("What about M.Com?", "faq-not-offered"),
    ):
        intent = detect_intent(question)
        ranked = _rerank(
            _interceptor_docs(), intent, ("course", "faq"),
            course_tokens=intent.course_tokens, query=question,
        )
        assert ranked[0].record_id == expected, question


def test_seat_count_questions_are_not_read_as_fee_questions() -> None:
    """Marathi "किती" is only "how many/how much"; with "जागा" it counts seats."""
    for question in (
        "how many seats in B.Tech Computer Engineering?",
        "\u092c\u0940.\u091f\u0947\u0915 \u0938\u0902\u0917\u0923\u0915 \u0905\u092d\u093f\u092f\u093e\u0902\u0924\u094d\u0930\u093f\u0915\u0940 \u0938\u093e\u0920\u0940 \u0915\u093f\u0924\u0940 \u091c\u093e\u0917\u093e \u0906\u0939\u0947\u0924?",
    ):
        assert detect_intent(question).intent == "courses", question
    # and a real Marathi fee question still routes to fees
    assert detect_intent("\u092c\u0940\u092c\u0940\u090f \u091a\u0940 \u092b\u0940 \u0915\u093f\u0924\u0940 \u0906\u0939\u0947?").intent == "fees"


# --------------------------------------------------------------------------- #
# knowledge base hygiene
# --------------------------------------------------------------------------- #
ASSISTANT_DIRECTED = re.compile(r"\bthe assistant\b|\bmust never\b|\bcallers should be\b", re.IGNORECASE)
EXEMPT_SLUGS = {"policy-escalation-boundaries"}


def test_record_bodies_are_caller_facing() -> None:
    dirty = [
        f"{r['_file']}::{r['slug']}"
        for r in _kb_records()
        if r["slug"] not in EXEMPT_SLUGS and ASSISTANT_DIRECTED.search(r.get("body") or "")
    ]
    assert dirty == []


def test_rendered_structured_values_are_caller_facing() -> None:
    rendered_keys = {
        "facilities", "room_types", "contact_phone", "contact_email", "address",
        "important_dates", "scholarship_types", "documents_required", "eligibility",
        "selection_process", "curriculum", "career_options", "recognition",
        "accreditation", "programme", "school", "degree", "summary", "description",
    }
    dirty: list[str] = []
    for record in _kb_records():
        structured = record.get("structured") or {}
        fees = structured.get("fees") or {}
        values = {k: v for k, v in structured.items() if k in rendered_keys}
        if isinstance(fees, dict):
            values.update({f"fees.{k}": v for k, v in fees.items()})
        for key, value in values.items():
            if isinstance(value, str):
                flat = value
            elif isinstance(value, (list, tuple, dict)):
                flat = " ".join(map(str, value))
            else:
                continue  # ints/bools cannot carry staff instructions
            if ASSISTANT_DIRECTED.search(flat):
                dirty.append(f"{record['slug']}.{key}")
    assert dirty == []


def test_staff_instructions_live_in_keys_that_are_never_rendered() -> None:
    never_rendered = {"assistant_instruction", "assistant_must_not", "assistant_behaviour", "action"}
    for record in _kb_records():
        for key in never_rendered & set((record.get("structured") or {}).keys()):
            assert isinstance(record["structured"][key], (str, list, dict)), record["slug"]


def test_no_generic_alias_hijacks_a_neighbouring_record() -> None:
    """An alias exists for recall; a generic one steals another record's query."""
    records = {r["slug"]: r for r in _kb_records()}
    facilities = records["campus-facilities"]["aliases"]
    hostel = records["hostel-availability"]["aliases"]
    # hostel phrases stay with the hostel record in both scripts, and the two
    # records share no alias at all
    assert any("हॉस्ट" in a for a in hostel)
    assert "राहण्याची सोय" in hostel
    assert "कॅम्पस मधील सुविधा" in facilities
    assert not (set(facilities) & set(hostel))
    # no bare campus word may sit on the hostel record and pull facility queries in
    assert all(
        "हॉस्ट" in a or "होस्टल" in a or "वसतिगृह" in a or "hostel" in a.lower()
        or a in {"accommodation", "mess", "room", "खाना", "राहण्याची सोय"}
        for a in hostel
    )
    # every alias must be short enough to be a phrase, not a whole question
    assert all(len(a.split()) <= 6 for a in facilities + hostel)


def test_contact_record_exposes_the_keys_the_contact_frame_reads() -> None:
    """The official site publishes school-wise numbers, an enquiry form and the
    portal — and no email address and no toll-free line, so neither may appear."""
    record = next(r for r in _kb_records() if r["slug"] == "university-contact")
    structured = record["structured"]
    assert structured["contact_phone"].strip()
    # a spoken number is digits, not a sentence with English words in it
    assert not re.search(r"[A-Za-z]", structured["contact_phone"])
    assert re.search(r"\d", structured["contact_phone"])
    # the department is localised so a Marathi caller does not hear an English label
    assert structured["department_localized"]["mr-IN"].strip()
    # the SMS card carries every school number, the enquiry form and the portal
    assert len(structured["school_phones"]) == 3
    assert all(re.search(r"\d", line) for line in structured["school_phones"])
    assert structured["enquiry_form"].strip()
    assert structured["admission_portal"].startswith("https://")
    # nothing invented: no email, no toll-free number, no fabricated office hours
    assert "contact_email" not in structured
    assert "admissions_toll_free" not in structured
    assert "office_hours" not in structured
    assert not any(
        str(v).startswith("1800") for v in structured.values() if isinstance(v, str)
    )


def test_scholarship_record_exposes_speakable_types() -> None:
    record = next(r for r in _kb_records() if r["slug"] == "scholarships-available")
    types = record["structured"]["scholarship_types"]
    assert isinstance(types, list) and len(types) >= 3
    assert all(isinstance(t, str) and t.strip() for t in types)


def test_hostel_record_does_not_claim_accommodation_the_website_never_published() -> None:
    """Hostel availability used to come from aggregator listings for other institutes
    at Dhule. The university's own site publishes nothing about hostels, so the record
    says so explicitly and the answer escalates instead of promising a room."""
    record = next(r for r in _kb_records() if r["slug"] == "hostel-availability")
    structured = record["structured"]
    assert structured["hostel_status"] == "not_published"
    assert structured["published_on_website"] is False
    assert "hostel_available" not in structured
    assert "charges" not in structured
    assert "room_types" not in structured
    # the caller is handed the numbers that can actually confirm it
    assert len(structured["school_phones"]) == 3
    # and the record is honest about what it is: a verified absence, not a guess
    assert record["verified"] is True
    assert "does not publish" in record["body"]


def test_facilities_record_uses_the_rendered_key() -> None:
    record = next(r for r in _kb_records() if r["slug"] == "campus-facilities")
    assert isinstance(record["structured"].get("facilities"), list)
    assert "facilities_listed" not in record["structured"]


def test_unverified_records_carry_a_source_and_are_not_marked_verified() -> None:
    """The seeded KB is sample data: nothing unverified may look authoritative."""
    for record in _kb_records():
        if record.get("verified"):
            continue
        assert record.get("source"), f"{record['slug']} has no source note"


def test_no_published_record_invents_a_hostel_charge() -> None:
    """Hostel fees are deliberately absent — the assistant must offer, not quote."""
    for record in _kb_records():
        fees = (record.get("structured") or {}).get("fees") or {}
        if isinstance(fees, dict):
            hostel_fee = fees.get("hostel")
            assert hostel_fee in (None, ""), f"{record['slug']} quotes a hostel charge"


# --------------------------------------------------------------------------- #
# a policy record must never be spoken as a programme name
# --------------------------------------------------------------------------- #
def _retrieval_of(question: str, chunks: list[RetrievedChunk]) -> RetrievalResult:
    return RetrievalResult(
        query=question,
        items=chunks,
        intent=detect_intent(question),
        best_score=chunks[0].score,
        grounded=True,
    )


def test_fee_policy_title_is_not_spliced_into_the_fees_frame() -> None:
    """A fee question that lands on the fee-policy record used to say "Fees are not
    published on the university website ची अधिकृत फी माझ्याकडे नोंदलेली नाही." — the
    record's *title* was formatted into `{programme}` as though it were a course."""
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Fees\nNo fee table is published.",
        title="Fees are not published on the university website",
        category="fees",
        structured={"published_fee_table": False,
                    "where_the_fee_is_confirmed": "At registration on the SVKM admission portal"},
        verified=True,
    )
    question = "बीबीए ची फी किती आहे?"
    answer = compose(_retrieval(question, chunk), language="mr-IN", question=question)
    assert answer.text == FRAMES["mr-IN"]["fees_not_published"]
    assert "published on the university website" not in answer.text
    assert answer.needs_escalation and answer.escalation_reason == "fee_not_in_kb"
    assert answer.followup and "portal" in " ".join(answer.followup["items"])


def test_non_programme_records_get_a_neutral_subject_not_their_title() -> None:
    """Frames read "For {programme}, …". When a policy record answers, the subject
    has to be a neutral noun, never the record's own statement."""
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Policy\nTransfers are not published.",
        title="Programme transfer and deferral are not published",
        category="policy",
        structured={"published": False, "eligibility": "Class 10+2 from a recognised board"},
        verified=True,
    )
    question = "what is the eligibility for admission"
    answer = compose(_retrieval(question, chunk), language="en-IN", question=question)
    assert "Programme transfer and deferral" not in answer.text
    assert PROGRAMME_NEUTRAL["en"] in answer.text


def test_the_programme_catalogue_never_lists_a_policy_record() -> None:
    """A policy record ranking into a "what courses do you have" retrieval used to be
    read out as an offering: "We offer Programme transfer and deferral are not
    published, and many more programmes." """
    policy = _chunk(
        text="NMIMS Global University, Dhule · Policy\nTransfers are not published.",
        title="Programme transfer and deferral are not published",
        category="policy",
        structured={"published": False},
        verified=True, chunk_id="c1", record_id="r-policy", score=0.9,
    )
    course = _chunk(
        text="NMIMS Global University, Dhule · Course\nBBA runs for three years.",
        title="Bachelor of Business Administration (BBA)",
        category="course",
        structured={"programme": "Bachelor of Business Administration (BBA)"},
        verified=True, chunk_id="c2", record_id="r-bba", score=0.8,
    )
    question = "what courses do you offer"
    answer = compose(_retrieval_of(question, [policy, course]),
                     language="en-IN", question=question)
    assert answer.template == "catalog"
    assert "Bachelor of Business Administration (BBA)" in answer.text
    assert "Programme transfer and deferral" not in answer.text
    assert "Programme transfer and deferral" not in " ".join(answer.followup["items"])


def test_a_record_that_publishes_nothing_always_ends_with_a_human() -> None:
    """`published: false` means the fact does not exist. Even when the sentence is
    useful, a refund or a loan cannot be settled from an absence of data."""
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Loans\nNo loan scheme is published.",
        title="Scholarships and fee concessions",
        category="scholarships",
        structured={"published_by_university": False,
                    "scholarship_types": ["Government of Maharashtra post-matric schemes"]},
        verified=True,
    )
    answer = compose(_retrieval("are there scholarships", chunk),
                     language="en-IN", question="are there scholarships")
    assert answer.needs_escalation and answer.escalation_reason == "not_published"


@pytest.mark.parametrize("question,frame", [
    ("शिक्षण कर्ज मिळेल का?", "loan_not_published"),
    ("शिक्षण ऋण मिलेगा क्या?", "loan_not_published"),
])
def test_loan_questions_are_answered_in_the_callers_language(question: str, frame: str) -> None:
    """The loan record only carries English prose, so a Marathi or Hindi caller used
    to hear the bare "I don't have that information" line — or worse, a description
    of the engineering school picked up from an unrelated retrieved record."""
    language = "mr-IN" if "कर्ज" in question else "hi-IN"
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Loans\nAn education loan is the lender's decision.",
        title="Education loans",
        category="loan_payment",
        structured={"published_by_university": False,
                    "what_the_university_can_issue": "A bonafide certificate and the fee structure"},
        verified=True,
    )
    answer = compose(_retrieval(question, chunk), language=language, question=question)
    assert answer.text == FRAMES[language][frame]
    assert answer.needs_escalation and answer.escalation_reason == "not_published"
    assert not re.search(r"[A-Za-z]{4,}", answer.text.replace("SMS", ""))


def test_a_refund_question_is_not_answered_with_the_loan_sentence() -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Fees\nNo refund policy is published.",
        title="Refunds and withdrawal",
        category="loan_payment",
        structured={"published": False},
        verified=True,
    )
    answer = compose(_retrieval("Can I get a refund if I withdraw?", chunk),
                     language="en-IN", question="Can I get a refund if I withdraw?")
    assert answer.text == FRAMES["en-IN"]["refund_not_published"]
    assert "education loan" not in answer.text.lower()


# --------------------------------------------------------------------------- #
# the fact the caller asked for, from whichever retrieved record holds it
# --------------------------------------------------------------------------- #
def test_eligibility_is_taken_from_the_record_that_actually_carries_it() -> None:
    """A generic summary record outranked the MCA record on category match, and the
    summary has no `eligibility` field — so the call escalated with the fact sitting
    two ranks down in the same retrieval."""
    summary = _chunk(
        text="NMIMS Global University, Dhule · Eligibility\nPostgraduate: a bachelor's degree.",
        title="Eligibility summary by level",
        category="eligibility",
        structured={"postgraduate": "A bachelor's degree in a relevant field"},
        verified=True, chunk_id="c1", record_id="r-summary", score=0.92,
    )
    mca = _chunk(
        text="NMIMS Global University, Dhule · Course\nMCA eligibility.",
        title="Master of Computer Applications (MCA)",
        category="course",
        structured={"programme": "Master of Computer Applications (MCA)",
                    "eligibility": "A bachelor's degree in Computer Applications, or a B.Sc. with Mathematics"},
        verified=True, chunk_id="c2", record_id="r-mca", score=0.88,
    )
    question = "What is the eligibility for MCA?"
    answer = compose(_retrieval_of(question, [summary, mca]),
                     language="en-IN", question=question)
    assert "Computer Applications" in answer.text
    assert "Master of Computer Applications (MCA)" in answer.text
    assert answer.needs_escalation is False


def test_the_english_fallback_stays_inside_the_record_that_answered() -> None:
    """Reaching into whatever ranked next made a Marathi caller asking about loans
    hear a description of the engineering school."""
    facts = _chunk(
        text="Published by university: false\nWhat the university can issue: a bonafide certificate",
        title="Education loans",
        category="loan_payment",
        structured={"published_by_university": False},
        verified=True, chunk_id="c1", record_id="r-loan", score=0.9,
    )
    unrelated = _chunk(
        text="NMIMS Global University, Dhule · Course\nThe School of Technology started in 2025-26.",
        title="School of Technology, Management & Engineering",
        category="university",
        structured={},
        verified=True, chunk_id="c2", record_id="r-school", score=0.7,
    )
    question = "शिक्षण कर्ज मिळेल का?"
    answer = compose(_retrieval_of(question, [facts, unrelated]),
                     language="mr-IN", question=question)
    assert "School of Technology" not in answer.text
    assert "2025-26" not in answer.text


# --------------------------------------------------------------------------- #
# intake, exams and rounds: the number the caller asked for
# --------------------------------------------------------------------------- #
def test_a_named_specialisation_gets_its_own_intake() -> None:
    """"What is the intake for M.Pharm Pharmaceutics?" was answered with the 48 seats
    that cover all four specialisations, and the "Pharmaceutics, intake 15" label was
    then dropped by the sentence filter and shipped off in the SMS instead."""
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Course\nM.Pharm runs on the semester pattern.",
        title="Master of Pharmacy (M.Pharm)",
        category="course",
        structured={"programme": "Master of Pharmacy (M.Pharm)", "duration_years": 2, "seats": 48,
                    "school": "School of Pharmacy & Technology Management",
                    "specialisations": ["Pharmaceutics — intake 15",
                                        "Pharmaceutical Quality Assurance — intake 15",
                                        "Pharmacology — intake 9"]},
        verified=True,
    )
    question = "What is the intake for M.Pharm Pharmaceutics?"
    answer = compose(_retrieval(question, chunk), language="en-IN", question=question)
    assert answer.template == "specialisation_intake"
    assert "For Pharmaceutics, the intake is fifteen this cycle." in answer.text
    assert "forty eight" not in answer.text
    assert answer.followup and len(answer.followup["items"]) == 3


def test_a_generic_exam_question_offers_the_list_instead_of_one_programme() -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Entrance exam\nTests accepted by programme.",
        title="Entrance tests accepted, by programme",
        category="entrance_exam",
        structured={"tests": ["B.Tech and B.Tech + MBA: MHT-CET (PCM), JEE Main, SAT, PERA TEST",
                              "M.Pharm: GPAT or PERA TEST",
                              "MCA: MAH-MCA-CET or PERA TEST (MCA)"]},
        verified=True,
    )
    question = "Which entrance exam do I need to give?"
    answer = compose(_retrieval(question, chunk), language="en-IN", question=question)
    assert answer.text.startswith("The qualifying test depends on the programme.")
    assert answer.followup["channel"] == "sms"
    assert len(answer.followup["items"]) == 3


def test_an_exam_question_naming_a_programme_reads_that_programmes_entry() -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Entrance exam\nTests accepted by programme.",
        title="Entrance tests accepted, by programme",
        category="entrance_exam",
        structured={"tests": ["B.Tech and B.Tech + MBA: MHT-CET (PCM), JEE Main, SAT, PERA TEST",
                              "M.Pharm: GPAT or PERA TEST"]},
        verified=True,
    )
    question = "Which entrance exam for M.Pharm?"
    answer = compose(_retrieval(question, chunk), language="en-IN", question=question)
    assert answer.text == "M.Pharm: GPAT or PERA TEST"
    assert "MHT-CET" not in answer.text


def test_a_named_programme_gets_its_published_round_status() -> None:
    """Dates are published per programme and per round, so the honest answer to
    "which round is MCA admission in?" is the MCA entry, not the mechanism."""
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Dates\nRound-wise schedules are published.",
        title="Admission dates and how to get the current schedule",
        category="important_dates",
        structured={"how_it_works": "Every date is published as a per-programme PDF schedule.",
                    "rounds_running_for_ay_2026_27": [
                        "B.Tech and B.Tech + MBA: schedules and merit lists published for Round I, Round II and Round III",
                        "MCA: schedules and merit lists published for Round I and Round II",
                        "BBA and BCA: first, second and third round merit lists published",
                    ]},
        verified=True,
    )
    question = "Which round is MCA admission in?"
    answer = compose(_retrieval(question, chunk), language="en-IN", question=question)
    assert answer.template == "round_status"
    assert answer.text.startswith("For MCA, schedules and merit lists published for Round I and Round II.")
    assert "SMS" in answer.text


def test_an_unnamed_round_question_explains_the_mechanism_in_the_callers_language() -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Dates\nRound-wise schedules are published.",
        title="Admission dates and how to get the current schedule",
        category="important_dates",
        structured={"how_it_works": "Every date is published as a per-programme PDF schedule.",
                    "rounds_running_for_ay_2026_27": ["MCA: Round I and Round II"]},
        verified=True,
    )
    question = "मेरिट यादी आली आहे का?"
    answer = compose(_retrieval(question, chunk), language="mr-IN", question=question)
    assert answer.text == FRAMES["mr-IN"]["important_dates_mechanism"]
    assert not re.search(r"[A-Za-z]{4,}", answer.text.replace("SMS", ""))


# --------------------------------------------------------------------------- #
# sentence splitting and the spoken-length trim
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("BBA LL.B. honours is not offered here.", ["BBA LL.B. honours is not offered here."]),
    ("Survey No. 499 is the campus address. Call the office.",
     ["Survey No. 499 is the campus address.", "Call the office."]),
    ("A B.Sc. degree is accepted. Then an interview.",
     ["A B.Sc. degree is accepted.", "Then an interview."]),
])
def test_abbreviation_periods_are_not_sentence_boundaries(text: str, expected: list[str]) -> None:
    """`(?<=[.!?।])\\s+` cut "BBA LL.B. honours" and "Survey No. 499" in half, and the
    caller heard a fragment of a degree name."""
    from app.ai.templates import split_sentences

    assert split_sentences(text) == expected


def test_the_length_trim_never_ends_on_a_dangling_comma() -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule · Dates\n" + (
            "For this cycle the School of Technology has published schedules and merit lists "
            "up to round three for B.Tech and B.Tech plus MBA, up to round two for M.Tech, MCA "
            "and direct second year B.Tech, and a spot admission announcement for the school, "
            "while the School of Pharmacy has published final merit lists up to round three for "
            "D.Pharm, B.Pharm, B.Pharm plus MBA, B.Tech Cosmetic Technology and direct second "
            "year B.Pharm, and up to round two for M.Pharm in this admission cycle for Dhule."
        ),
        title="Admission dates and how to get the current schedule",
        category="important_dates",
        structured={"important_dates": ["Round-wise schedules are published per programme"]},
        verified=True,
    )
    answer = compose(_retrieval("when is the last date", chunk),
                     language="en-IN", question="when is the last date")
    assert not answer.text.endswith(",.")
    assert not answer.text.endswith(",")
    assert len(answer.text) <= MAX_SPOKEN_CHARS


# --------------------------------------------------------------------------- #
# intent routing for the questions callers actually ask
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,intent", [
    ("शिक्षण कर्ज मिळेल का?", "loan_payment"),
    ("पैसे परत मिळतील का?", "loan_payment"),
    ("Can I get a refund if I withdraw?", "loan_payment"),
    ("प्रवेश के लिए कौन सी परीक्षा देनी होगी?", "entrance_exam"),
    ("कोणती परीक्षा द्यावी लागेल?", "entrance_exam"),
    ("Has the merit list come out for BBA?", "important_dates"),
    ("मेरिट लिस्ट का आ गया है क्या?", "important_dates"),
    ("मेरिट यादी आली आहे का?", "important_dates"),
    ("Is admission still open for B.Tech?", "important_dates"),
    ("what is the admission process", "admission_process"),
])
def test_intent_routing_for_loan_exam_and_round_questions(question: str, intent: str) -> None:
    assert detect_intent(question).intent == intent


# --------------------------------------------------------------------------- #
# knowledge-base hygiene
# --------------------------------------------------------------------------- #
def test_facility_items_are_speakable_nouns_not_sentences() -> None:
    """The facilities frame joins the list into "The campus has X, Y and Z", so an
    item that is itself a sentence produced an unreadable answer."""
    record = next(r for r in _kb_records() if r["slug"] == "campus-facilities")
    for item in record["structured"]["facilities"]:
        assert len(item) < 48, item
        assert not item.endswith("."), item


def test_no_record_body_addresses_the_caller_in_the_third_person() -> None:
    """Bodies are spoken aloud. "A caller who wants to withdraw needs the accounts
    office" is staff-facing guidance, not something to say to a person on the phone."""
    for record in _kb_records():
        body = record.get("body") or ""
        assert "a caller" not in body.lower(), record["slug"]
        assert "the caller" not in body.lower(), record["slug"]


def test_the_dates_record_records_rounds_per_school() -> None:
    """Read off the homepage Admission Announcements tabs: each school is at a
    different round, so a blanket "rounds one to three for everything" is wrong."""
    record = next(r for r in _kb_records() if r["slug"] == "admission-important-dates")
    rounds = " ".join(record["structured"]["rounds_running_for_ay_2026_27"]).lower()
    for expected in ("b.tech", "mca", "pharmacy", "bba and bca", "ph.d"):
        assert expected in rounds
    assert record["verified"] is True
    assert record["source"] == "https://www.svkmnmimsgu.ac.in/"


def test_every_language_has_the_same_frame_keys() -> None:
    """A frame missing in one language is a KeyError on a live call."""
    keysets = {frozenset(frames) for frames in FRAMES.values()}
    assert len(keysets) == 1
    for key in ("fees_not_published", "loan_not_published", "refund_not_published",
                "specialisation_intake", "entrance_tests_generic", "round_status",
                "eligibility_long"):
        assert all(key in frames for frames in FRAMES.values()), key
