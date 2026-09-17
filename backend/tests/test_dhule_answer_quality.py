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
    PROGRAMME_NOUN,
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
    # Names a programme, so the answer composes rather than asking which one.
    question = "what is the eligibility for B.Tech"
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


# --------------------------------------------------------------------------- #
# a denial is an answer, not a payload for an intent frame
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "how do I take admission in MBBS",
    "what is the fee for MBBS",
    "what is the eligibility for BDS",
])
def test_a_not_offered_record_is_never_wrapped_in_an_intent_frame(question: str) -> None:
    """A Hindi caller asking how to get into MBBS used to hear "एडमिशन की प्रक्रिया
    है: SVKM NMIMS Global University does not run MBBS, BDS or any dental
    programme…" — the process frame wrapped a denial."""
    chunk = _chunk(
        text=(
            "NMIMS Global University, Dhule · FAQ\n"
            "SVKM NMIMS Global University does not run MBBS, BDS or any dental "
            "programme, nursing or AYUSH degrees. There is no entrance route, fee "
            "or seat count for a clinical medical programme here."
        ),
        title="Medical, dental, nursing and allied health programmes are not offered",
        category="faq",
        structured={"not_offered": ["MBBS", "BDS / dental", "B.Sc. Nursing"],
                    "offered_instead": ["Diploma in Pharmacy (D.Pharm)",
                                        "Bachelor of Pharmacy (B.Pharm)"]},
        verified=True,
    )
    answer = compose(_retrieval(question, chunk), language="en-IN", question=question)
    assert answer.template == "not_offered"
    assert not answer.text.startswith("The admission process is")
    assert not answer.text.startswith("For ")
    assert "does not run MBBS" in answer.text
    # the alternatives travel by message, not by voice
    assert answer.followup and "D.Pharm" in " ".join(answer.followup["items"])


# --------------------------------------------------------------------------- #
# the numbers a live call is actually transferred to
# --------------------------------------------------------------------------- #
def test_escalation_targets_are_the_published_school_offices() -> None:
    """The seeded env shipped placeholder agents (+911412345678) and a US helpline,
    so a caller who asked for a human was transferred to a line that does not
    exist. Only numbers published on svkmnmimsgu.ac.in/contact-us may be used."""
    from app.config import Settings

    defaults = Settings(_env_file=None)
    published = {"+912562350620", "+912562350600", "+912562350640"}
    assert set(defaults.escalation_agent_list) <= published
    assert defaults.twilio_helpline_number in published
    assert "nims-admissions-queue" != defaults.escalation_queue_name
    assert "dhule" in defaults.escalation_queue_name


def test_no_rejected_or_placeholder_number_survives_in_the_codebase() -> None:
    """Aggregator research produced a toll-free number (1800 102 5138 /
    +911800120000) that the university's own site does not publish. It must not
    come back as a fallback anywhere."""
    banned = ("+911800120000", "+911412345678", "+919876500000", "+15551234567",
              "1800 102 5138", "1800120102")
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.glob("app/**/*.py")):
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{path.name} still carries {needle}"


def test_the_typo_env_var_still_loads_for_existing_deployments() -> None:
    """`TWILIO_HELLINE_NUMBER` (missing its second "p") is what deployments already
    have written down, so the rename must not silently drop their number."""
    from app.config import Settings

    legacy = Settings(_env_file=None, TWILIO_HELLINE_NUMBER="+912562350640")
    assert legacy.twilio_helpline_number == "+912562350640"
    current = Settings(_env_file=None, TWILIO_HELPLINE_NUMBER="+912562350600")
    assert current.twilio_helpline_number == "+912562350600"


# --------------------------------------------------------------------------- #
# the follow-up offer must not describe a short answer as long
# --------------------------------------------------------------------------- #
def test_the_followup_offer_line_is_length_neutral_in_every_language() -> None:
    """It is spoken for every follow-up, including a one-line hostel answer, so
    "That is a long answer" was simply untrue on most calls."""
    from app.i18n.languages import LANGUAGES

    for code, lang in LANGUAGES.items():
        line = lang.scripts.get("followup_offer")
        if not line:
            continue
        lowered = line.lower()
        for claim in ("long answer", "लंबी", "लांब", "लम्बी"):
            assert claim not in line and claim not in lowered, f"{code}: {line}"
        assert "SMS" in line or "एसएमएस" in line, f"{code} offers no channel"


# --------------------------------------------------------------------------- #
# AY 2026-27 academic calendar (official signed PDF) and collaborations
# --------------------------------------------------------------------------- #

CALENDAR_TITLE = "Academic calendar for AY 2026-27"


def _record(slug: str) -> dict[str, Any]:
    found = next((r for r in _kb_records() if r.get("slug") == slug), None)
    assert found is not None, f"KB record {slug} is missing"
    return found


def _record_chunk(slug: str, *, kind: str = "facts") -> RetrievedChunk:
    """A chunk carrying the real record's prose and structured fields, as ingested."""
    record = _record(slug)
    body = " ".join(str(record.get("body") or "").split())
    return _chunk(
        text=f"NMIMS Global University, Dhule \u00b7 {record['title']}\n{body}",
        title=record["title"],
        category=record["category"],
        structured=record.get("structured") or {},
        verified=bool(record.get("verified")),
        chunk_id=f"{slug}:{kind}",
        record_id=slug,
    )


@pytest.mark.parametrize("question", [
    "When do classes start?",
    "When are the term end exams?",
    "what is the academic calendar",
])
def test_calendar_questions_speak_the_published_dates(question: str) -> None:
    chunk = _record_chunk("academic-calendar-ay-2026-27")
    answer = compose(_retrieval(question, chunk), language="en-IN", question=question)
    assert answer.template == "important_dates", answer.text
    assert answer.needs_escalation is False, "the calendar is published and verified"
    assert "13 July 2026" in answer.text and "24 December 2026" in answer.text, answer.text
    assert len(answer.text) <= MAX_SPOKEN_CHARS


def test_date_lines_are_spoken_as_sentences_not_a_run_on() -> None:
    """Three long date sentences joined with ", " and " and " made one run-on
    that the trim then cut in the middle of a date ("12 January to 28")."""
    chunk = _record_chunk("academic-calendar-ay-2026-27")
    answer = compose(_retrieval("When do classes start?", chunk),
                     language="en-IN", question="When do classes start?")
    text = answer.text.rstrip()
    assert text.endswith("."), text
    assert ", Term end" not in text and " and The semester" not in text, text
    assert " to 28\u201d" not in text and not text.endswith("to 28"), text
    # The rest of the calendar still reaches the caller by SMS.
    assert answer.followup and len(answer.followup["items"]) > 3, answer.followup


def test_calendar_dates_are_read_from_the_real_record() -> None:
    """Pin the transcription against the signed PDF the website publishes."""
    structured = _record("academic-calendar-ay-2026-27")["structured"]
    assert structured["term_end_exam"] == "01 December to 24 December 2026"
    assert structured["diwali_break"] == "07 November to 14 November 2026"
    assert structured["even_semester_term_end_exam"] == "15 May to 10 June 2027"
    assert structured["next_academic_year_commences"] == "12 July 2027"
    assert "Examination Department" in structured["calendar_caveat"]


def test_calendar_record_does_not_pass_teaching_dates_off_as_admission_deadlines() -> None:
    record = _record("academic-calendar-ay-2026-27")
    assert record["verified"] is True
    assert "svkmnmimsgu.ac.in" in (record.get("source") or "")
    assert "Admission deadlines are published separately" in record["body"]
    instruction = record["structured"]["assistant_instruction"]
    assert "never extrapolate" in instruction.lower()


# --- Indic routing: a calendar question must not be read as hostel or exams --- #

HI_EXAM_WHEN = "\u092A\u0930\u0940\u0915\u094D\u0937\u093E \u0915\u092C \u0939\u094B\u0917\u0940?"
HI_CLASS_START = "\u0915\u094D\u0932\u093E\u0938 \u0915\u092C \u0938\u0947 \u0936\u0941\u0930\u0942 \u0939\u094B\u0902\u0917\u0947?"
MR_SEMESTER = "\u0938\u0947\u092E\u0947\u0938\u094D\u091F\u0930 \u0915\u0927\u0940 \u0938\u0941\u0930\u0942 \u0939\u094B\u0908\u0932?"
MR_EXAM_DATES = "\u092A\u0930\u0940\u0915\u094D\u0937\u0947\u091A\u094D\u092F\u093E \u0924\u093E\u0930\u0916\u093E \u0915\u093E\u092F \u0906\u0939\u0947\u0924?"


@pytest.mark.parametrize("question", [HI_EXAM_WHEN, HI_CLASS_START, MR_SEMESTER, MR_EXAM_DATES])
def test_indic_calendar_questions_are_dates_questions(question: str) -> None:
    assert detect_intent(question).intent == "important_dates", detect_intent(question).scores


def test_marathi_mess_does_not_match_inside_semester() -> None:
    r"""The Marathi word for mess is a substring of the Marathi word for semester.

    Python's \b does not catch it: Devanagari matras are not \w, so a word
    boundary exists inside the word. A semester question was detected as a hostel
    question and escalated as unconfirmed accommodation.
    """
    assert detect_intent(MR_SEMESTER).intent != "hostel", detect_intent(MR_SEMESTER).scores
    # Real mess questions still reach the hostel branch.
    hi_mess = "\u0939\u0949\u0938\u094D\u091F\u0932 \u092E\u0947\u0902 \u092E\u0947\u0938 \u0915\u0940 \u0938\u0941\u0935\u093F\u0927\u093E \u0939\u0948?"
    mr_mess = "\u092E\u0947\u0938\u091A\u0940 \u0938\u094B\u092F \u0906\u0939\u0947 \u0915\u093E?"
    assert detect_intent(hi_mess).intent == "hostel", detect_intent(hi_mess).scores
    assert detect_intent(mr_mess).intent == "hostel", detect_intent(mr_mess).scores


def test_indic_exam_word_bridges_to_the_calendar_for_a_dates_question() -> None:
    """One Devanagari word, two meanings: which exams, or when the exams are."""
    from app.ai.rag import augment_query

    when = augment_query(HI_EXAM_WHEN, detect_intent(HI_EXAM_WHEN))
    assert "term end examination dates" in when, when
    which = augment_query(
        "\u092A\u0930\u0940\u0915\u094D\u0937\u093E \u0915\u094C\u0928 \u0938\u0940 \u0926\u0947\u0928\u0940 \u0939\u094B\u0917\u0940?",
        detect_intent("\u092A\u0930\u0940\u0915\u094D\u0937\u093E \u0915\u094C\u0928 \u0938\u0940 \u0926\u0947\u0928\u0940 \u0939\u094B\u0917\u0940?"),
    )
    assert "entrance test" in which and "term end" not in which, which


def test_the_intent_bridge_makes_the_calendar_chunk_outrank_admission_dates() -> None:
    """Scored with the real BM25 index over the augmented query.

    Before the intent-aware bridge, the Devanagari word for exam bridged to the
    bare English word "Exam", which lexically favoured the admission-schedules
    record, so "when is the exam?" was answered with admission round status.
    """
    from app.ai.rag import augment_query
    from app.kb.embeddings import expand
    from app.kb.retriever import BM25Index

    calendar_text = (
        "Academic calendar for AY 2026-27 Classes commence on 13 July 2026 for "
        "Pharmacy and term end examinations run from 01 December to 24 December 2026."
    )
    admission_text = (
        "Admission dates and how to get the current schedules Merit lists and "
        "admission schedules are published programme by programme and round by round."
    )
    index = BM25Index()
    index.build([("cal", expand(calendar_text)), ("adm", expand(admission_text))])

    intent = detect_intent(HI_EXAM_WHEN)
    bridged = index.score(expand(augment_query(HI_EXAM_WHEN, intent)), top_k=2)
    assert bridged and bridged[0][0] == "cal", bridged

    # The old generic bridge ranked the admission record first.
    legacy = index.score(expand(f"{HI_EXAM_WHEN} Exam"), top_k=2)
    assert legacy and legacy[0][0] == "adm", legacy


def test_entrance_exam_questions_are_not_stolen_by_the_calendar() -> None:
    from app.ai.rag import augment_query

    question = "\u092A\u0930\u0940\u0915\u094D\u0937\u093E \u0915\u094C\u0928 \u0938\u0940 \u0926\u0947\u0928\u0940 \u0939\u094B\u0917\u0940?"
    assert detect_intent(question).intent == "entrance_exam"
    assert "term end" not in augment_query(question, detect_intent(question))
    chunk = _record_chunk("admission-entrance-tests")
    answer = compose(_retrieval(question, chunk), language="en-IN", question=question)
    assert "MHT-CET" in answer.text, answer.text


# --- collaborations: logos only, so no invented benefit ---------------------- #

def test_collaborations_names_the_published_partnerships_without_inventing_benefit() -> None:
    chunk = _record_chunk("university-collaborations")
    answer = compose(_retrieval("Do you have NPTEL or SWAYAM?", chunk),
                     language="en-IN", question="Do you have NPTEL or SWAYAM?")
    for name in ("NPTEL", "SWAYAM", "e-Yantra", "Spoken Tutorial"):
        assert name in answer.text, answer.text
    # The page publishes logos with no description: no credit, certification,
    # free-access or placement benefit may be claimed from any of them.
    for claim in ("credit", "certificat", "free of cost", "guarantee", "placement"):
        assert claim not in answer.text.lower(), answer.text


def test_collaborations_record_does_not_claim_detail_the_page_never_published() -> None:
    record = _record("university-collaborations")
    assert record["verified"] is True
    assert record["structured"]["published_detail"] is False
    assert "without describing what each partnership means" in record["body"]
    assert len(record["structured"]["collaborations"]) == 7


# --- library page is a photo gallery, so no facility detail is invented ------- #

def test_facilities_record_says_the_library_page_is_a_photo_gallery() -> None:
    record = _record("campus-facilities")
    assert "gallery of library photographs" in record["body"]
    assert "no holdings, timings or capacity" in record["structured"]["library_page"]
    assert record["verified"] is True


# --------------------------------------------------------------------------- #
# transport: answer the mode the caller named
# --------------------------------------------------------------------------- #

TRANSPORT_STRUCTURED = {
    "nearest_airport": "Aurangabad (Chh. Sambhajinagar) Airport, about 157 km from Dhule",
    "by_air": "Aurangabad is the nearest airport at about 157 km by road.",
    "by_rail": (
        "Trains from New Delhi passing through Bhusawal: alight at Bhusawal "
        "Railway Station and continue by road via Jalgaon to Dhule, about 125 km."
    ),
    "road_distances": ["Mumbai \u2014 330 km", "Pune \u2014 355 km", "Indore \u2014 225 km"],
    "on_highway": "The campus is on the Mumbai Agra National Highway, behind the Gurudwara.",
}


def _transport_chunk() -> RetrievedChunk:
    return _chunk(
        text="NMIMS Global University, Dhule \u00b7 How to reach the Dhule campus",
        title="How to reach the Dhule campus",
        category="transport",
        structured=dict(TRANSPORT_STRUCTURED),
        verified=True,
        chunk_id="transport:facts",
        record_id="university-transport",
    )


def _ask_transport(question: str, language: str = "en-IN"):
    return compose(_retrieval(question, _transport_chunk()),
                   language=language, question=question)


@pytest.mark.parametrize("question", [
    "How do I reach the campus by train?",
    "which is the nearest railway station",
    "\u091F\u094D\u0930\u0947\u0928 \u0938\u0947 \u0915\u0948\u0938\u0947 \u092A\u0939\u0941\u0902\u091A\u0947\u0902?",
    "\u0930\u0947\u0932\u094D\u0935\u0947\u0928\u0947 \u0915\u0938\u0947 \u092A\u094B\u0939\u094B\u091A\u093E\u0935\u0947?",
])
def test_train_questions_are_answered_with_the_rail_route(question: str) -> None:
    """A rail question used to be answered with the airport and road distances."""
    answer = _ask_transport(question)
    assert answer.template == "transport_rail", answer.text
    assert "Bhusawal" in answer.text, answer.text
    assert "airport" not in answer.text.lower(), answer.text
    assert "330 km" not in answer.text, answer.text


@pytest.mark.parametrize("question", [
    "How do I reach by flight?",
    "which is the nearest airport",
    "\u0935\u093F\u092E\u093E\u0928 \u0938\u0947 \u0915\u0948\u0938\u0947 \u0906\u090A\u0902?",
])
def test_flight_questions_are_answered_with_the_air_route(question: str) -> None:
    answer = _ask_transport(question)
    assert answer.template == "transport_air", answer.text
    assert "Aurangabad" in answer.text, answer.text
    assert "Bhusawal" not in answer.text, answer.text


@pytest.mark.parametrize("question", [
    "Can I come by bus?",
    "how do I drive there",
    "\u092C\u0938 \u0938\u0947 \u0915\u0948\u0938\u0947 \u092A\u0939\u0941\u0902\u091A\u0947\u0902?",
])
def test_road_questions_give_distances_not_the_airport(question: str) -> None:
    answer = _ask_transport(question)
    assert answer.template == "transport_road", answer.text
    assert "330 km" in answer.text and "Mumbai Agra National Highway" in answer.text
    assert "airport" not in answer.text.lower(), answer.text


def test_a_general_how_to_reach_still_names_airport_and_distances() -> None:
    answer = _ask_transport("How do I reach the campus?")
    assert answer.template == "transport_summary", answer.text
    assert "Aurangabad" in answer.text and "330 km" in answer.text
    # The rail route still reaches the caller by SMS.
    assert answer.followup and any("Bhusawal" in i for i in answer.followup["items"])


def test_transport_frames_exist_in_every_language() -> None:
    for language in FRAMES:
        for key in ("transport_rail", "transport_air", "transport_road"):
            assert key in FRAMES[language], (language, key)
    # The rail/air frames must expose {detail}; road must expose {distances}.
    for language, frames in FRAMES.items():
        assert "{detail}" in frames["transport_rail"], language
        assert "{detail}" in frames["transport_air"], language
        assert "{distances}" in frames["transport_road"], language


@pytest.mark.parametrize("question", [
    "\u0917\u093E\u0921\u0940\u0928\u0947 \u0915\u0938\u0947 \u092F\u093E\u0935\u0947?",
    "\u0930\u0938\u094D\u0924\u094D\u092F\u093E\u0928\u0947 \u0915\u0938\u0947 \u092F\u093E\u0935\u0947?",
    "\u092C\u0938\u0928\u0947 \u0915\u0938\u0947 \u092F\u093E\u0935\u0947?",
])
def test_marathi_road_questions_use_the_road_route(question: str) -> None:
    """Marathi spells car/road with a short a, which the Hindi pattern missed."""
    answer = _ask_transport(question, "mr-IN")
    assert answer.template == "transport_road", answer.text
    assert "330 km" in answer.text, answer.text


# --------------------------------------------------------------------------- #
# a bare degree name must not be answered as one niche variant
# --------------------------------------------------------------------------- #

def test_bare_degree_fee_question_names_the_degree_the_caller_said() -> None:
    """'B.Tech' retrieved B.Tech (Cosmetic Technology), a pharmacy-school variant.

    Seven other B.Tech records exist and no fee is published for any of them, so
    naming the variant answered a question nobody asked.
    """
    from app.ai.templates import _caller_named_only_the_degree

    assert _caller_named_only_the_degree(
        "What is the fee for B.Tech?", "B.Tech (Cosmetic Technology)") == "B.Tech"
    assert _caller_named_only_the_degree(
        "B.Tech fee kitni hai?", "B.Tech (Cosmetic Technology)") == "B.Tech"
    assert _caller_named_only_the_degree(
        "What is the fee for B.Pharm?", "B.Pharm + MBA (Pharma Tech)") == "B.Pharm"


def test_a_specific_programme_question_keeps_the_record_title() -> None:
    from app.ai.templates import _caller_named_only_the_degree

    assert _caller_named_only_the_degree(
        "What is the fee for B.Tech Computer Engineering?",
        "B.Tech Computer Engineering") is None
    assert _caller_named_only_the_degree(
        "fee for B.Tech Cosmetic Technology",
        "B.Tech (Cosmetic Technology)") is None
    # No degree named at all: nothing to narrow.
    assert _caller_named_only_the_degree("what is the fee", "B.Tech (Cosmetic Technology)") is None


def test_fee_answer_speaks_the_bare_degree_not_the_variant() -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule \u00b7 B.Tech (Cosmetic Technology)",
        title="B.Tech (Cosmetic Technology)",
        category="course",
        structured={"programme": "B.Tech (Cosmetic Technology)", "duration_years": 4},
        verified=True,
        chunk_id="cosmetic:facts",
        record_id="course-btech-cosmetic-technology",
    )
    answer = compose(_retrieval("What is the fee for B.Tech?", chunk),
                     language="en-IN", question="What is the fee for B.Tech?")
    assert answer.template == "fees_not_recorded", answer.text
    assert "Cosmetic" not in answer.text, answer.text
    assert "B.Tech" in answer.text
    assert answer.needs_escalation is True and answer.escalation_reason == "fee_not_in_kb"
    # No number is ever invented for a fee that is not published.
    assert not re.search(r"\d{3,}", answer.text), answer.text


# --------------------------------------------------------------------------- #
# Devanagari word edges: SMS is not M.A.
# --------------------------------------------------------------------------- #

HI_DOCUMENTS_SMS = (
    "\u0921\u0949\u0915\u094D\u092F\u0942\u092E\u0947\u0902\u091F\u094D\u0938 \u0915\u0940 \u0932\u093F\u0938\u094D\u091F "
    "\u090F\u0938\u090F\u092E\u090F\u0938 \u0938\u0947 \u092D\u0947\u091C\u094B"
)


def test_the_word_for_sms_does_not_invent_an_ma_degree() -> None:
    """The Devanagari for SMS contains the Devanagari for M.A.

    A caller asking for the document list "by SMS" was given course tokens for
    M.A. and MS, the query was bridged to those, and retrieval answered with the
    travel record. Longest-match-first was not enough; matches now need a word
    edge. Only the leading edge, because Indian languages inflect the degree name
    itself.
    """
    intent = detect_intent(HI_DOCUMENTS_SMS)
    assert intent.intent == "documents", intent.scores
    assert intent.course_tokens == [], intent.course_tokens


def test_inflected_degree_names_still_match() -> None:
    """The word-edge rule must not drop genuine mentions."""
    assert detect_intent("\u092C\u0940\u091F\u0947\u0915\u0915\u0940 \u092B\u0940\u0938").course_tokens == ["BTECH"]
    # Marathi inflection: "बीटेकसाठी" (for B.Tech), "एमफार्मची" (M.Pharm's).
    assert "BTECH" in detect_intent("\u092C\u0940\u091F\u0947\u0915\u0938\u093E\u0920\u0940 \u092A\u093E\u091F\u094D\u0930\u0924\u093E").course_tokens
    assert detect_intent("\u090F\u092E\u092C\u0940\u092C\u0940\u090F\u0938 \u0939\u0948 \u0915\u094D\u092F\u093E?").course_tokens == ["MBBS"]


def test_the_cross_script_bridge_needs_a_word_edge_too() -> None:
    from app.ai.rag import _contains_native, augment_query

    intent = detect_intent(HI_DOCUMENTS_SMS)
    assert "M.A" not in augment_query(HI_DOCUMENTS_SMS, intent)
    assert "Documents" in augment_query(HI_DOCUMENTS_SMS, intent)
    assert _contains_native("\u0926\u0938\u094D\u0924\u093E\u0935\u0947\u091C \u091A\u093E\u0939\u093F\u090F", "\u0926\u0938\u094D\u0924\u093E\u0935\u0947\u091C")
    assert not _contains_native("\u090F\u0938\u090F\u092E\u090F\u0938", "\u090F\u092E\u090F")


# --------------------------------------------------------------------------- #
# catalogue: programme names, not school-overview titles
# --------------------------------------------------------------------------- #

def _overview_chunk() -> RetrievedChunk:
    return _chunk(
        text="NMIMS Global University, Dhule \u00b7 School of Commerce \u2014 programmes overview",
        title="School of Commerce \u2014 programmes overview",
        category="course",
        structured={
            "school": "School of Commerce",
            "levels": ["UG (BBA, BCA)"],
            "programmes": [
                "Bachelor of Business Administration (BBA) \u2014 3 years, semester, intake 60",
                "Bachelor in Computer Applications (BCA) \u2014 3 years, semester, intake 60",
            ],
        },
        verified=True,
        chunk_id="soc:facts",
        record_id="school-soc-overview",
    )


def test_catalogue_speaks_programme_names_not_overview_titles() -> None:
    """The catalogue used to offer "School of Commerce — programmes overview"
    as though it were a course a caller could enrol in."""
    question = "What courses do you offer?"
    answer = compose(_retrieval(question, _overview_chunk()),
                     language="en-IN", question=question)
    assert answer.template == "catalog", answer.text
    assert "overview" not in answer.text.lower(), answer.text
    assert "Bachelor of Business Administration (BBA)" in answer.text
    # The detail after the dash stays out of the spoken line but goes by message.
    assert "intake 60" not in answer.text
    assert answer.followup and any("BCA" in i for i in answer.followup["items"])


# --------------------------------------------------------------------------- #
# degree exactness: a bare degree is not its dual-degree variant
# --------------------------------------------------------------------------- #

def test_a_bare_degree_prefers_the_plain_record_over_a_dual_degree() -> None:
    """'How many seats in B.Pharm?' landed on B.Pharm + MBA and said forty seats.

    The plain record spells its degree out — "Bachelor of Pharmacy (B.Pharm)" — so
    the spoken prefix never matched the caller's token, while the dual degree's
    title merely started with it.
    """
    plain = _doc("bpharm:facts", "Bachelor of Pharmacy (B.Pharm)", "course",
                 "Four year semester programme.", {"seats": 60})
    dual = _doc("dual:facts", "B.Pharm + MBA (Pharma Tech) \u2014 five year dual degree",
                "course", "Five year dual degree.", {"seats": 40})
    intent = detect_intent("How many seats in B.Pharm?")
    ranked = _rerank([dual, plain], intent, ("course",),
                     course_tokens=intent.course_tokens,
                     query="How many seats in B.Pharm?")
    assert ranked[0].chunk_id == "bpharm:facts", [r.chunk_id for r in ranked]
    assert ranked[0].signals.get("degree_exact"), ranked[0].signals


def test_a_named_dual_degree_still_reaches_its_own_record() -> None:
    dual = _doc("dual:facts", "B.Pharm + MBA (Pharma Tech) \u2014 five year dual degree",
                "course", "Five year dual degree.", {"seats": 40})
    plain = _doc("bpharm:facts", "Bachelor of Pharmacy (B.Pharm)", "course",
                 "Four year semester programme.", {"seats": 60})
    question = "B.Pharm + MBA ki fees?"
    intent = detect_intent(question)
    ranked = _rerank([plain, dual], intent, ("course",),
                     course_tokens=intent.course_tokens, query=question)
    assert ranked[0].chunk_id == "dual:facts", [r.chunk_id for r in ranked]


# --------------------------------------------------------------------------- #
# placements: a detail nobody published is not answered with a claim
# --------------------------------------------------------------------------- #

PLACEMENTS_STRUCTURED = {
    "website_claim": "The university's homepage states '100% job placement.'",
    "published_detail": False,
    "not_published": [
        "A placement report, placement percentage or number of offers",
        "Highest, average or median package",
        "A recruiter list or the names of visiting companies",
    ],
}


def _placements_chunk() -> RetrievedChunk:
    return _chunk(
        text="NMIMS Global University, Dhule \u00b7 Placements",
        title="Placements",
        category="placements",
        structured=dict(PLACEMENTS_STRUCTURED),
        verified=True,
        chunk_id="placements:facts",
        record_id="placements-overview",
    )


@pytest.mark.parametrize("question", [
    "What is the highest package?",
    "what is the average salary",
    "Which companies come for placement?",
    "\u092A\u0948\u0915\u0947\u091C \u0915\u093F\u0924\u0928\u093E \u092E\u093F\u0932\u0924\u093E \u0939\u0948?",
])
def test_placement_details_nobody_published_are_not_answered_with_a_claim(question: str) -> None:
    answer = compose(_retrieval(question, _placements_chunk()),
                     language="en-IN", question=question)
    assert answer.template == "placements_not_published", answer.text
    assert answer.needs_escalation is True and answer.escalation_reason == "not_published"
    assert "100" not in answer.text and "placement report" in answer.text.lower()
    assert answer.followup and len(answer.followup["items"]) == 3


def test_a_general_placement_question_keeps_the_attributed_claim() -> None:
    question = "Do you have placement support?"
    answer = compose(_retrieval(question, _placements_chunk()),
                     language="en-IN", question=question)
    assert answer.template != "placements_not_published", answer.text
    assert answer.needs_escalation is False


# --------------------------------------------------------------------------- #
# address: an address question is answered with the address
# --------------------------------------------------------------------------- #

ADDRESS_STRUCTURED = {
    "address": "SVKM NMIMS Global University, Survey No. 499, Behind Gurudwara, Dhule 424001",
    "pin_code": "424001",
    "landmark": "Behind the Gurudwara, on the Mumbai Agra National Highway",
    "website": "https://www.svkmnmimsgu.ac.in",
}


def _address_chunk() -> RetrievedChunk:
    return _chunk(
        text="NMIMS Global University, Dhule \u00b7 Campus address",
        title="Campus address",
        category="contact",
        structured=dict(ADDRESS_STRUCTURED),
        verified=True,
        chunk_id="address:facts",
        record_id="university-address",
    )


@pytest.mark.parametrize("question,language", [
    ("What is the campus address?", "en-IN"),
    ("\u0927\u0941\u0932\u0947 \u0915\u0948\u0902\u092A\u0938 \u0915\u093E \u092A\u0924\u093E \u092C\u0924\u093E\u0913", "hi-IN"),
    ("\u0915\u0945\u092E\u094D\u092A\u0938\u091A\u093E \u092A\u0924\u094D\u0924\u093E \u0915\u093E\u092F \u0906\u0939\u0947?", "mr-IN"),
])
def test_an_address_question_speaks_the_address(question: str, language: str) -> None:
    """It used to read the landmarks and spell out the pin code in words."""
    answer = compose(_retrieval(question, _address_chunk()),
                     language=language, question=question)
    assert answer.template == "campus_address", answer.text
    assert "Survey No. 499" in answer.text and "424001" in answer.text
    # The frame is the whole answer: no extractive prose tail appended to it.
    assert answer.text.count("Gurudwara") == 1, answer.text
    assert answer.followup and any("424001" in i for i in answer.followup["items"])


def test_a_contact_number_question_is_not_answered_with_the_address() -> None:
    chunk = _address_chunk()
    chunk.structured["contact_phone"] = "02562 350620"
    answer = compose(_retrieval("What is the contact number?", chunk),
                     language="en-IN", question="What is the contact number?")
    assert answer.template == "contact_phone", answer.text
    assert "02562 350620" in answer.text


# --------------------------------------------------------------------------- #
# accreditation questions are not catalogue questions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question", [
    "Which NBA accredited programmes do you have?",
    "Is the university UGC recognised?",
    "Is it NAAC accredited?",
])
def test_accreditation_questions_are_university_questions(question: str) -> None:
    assert detect_intent(question).intent == "university_info", detect_intent(question).scores


def test_a_catalogue_question_is_still_a_catalogue_question() -> None:
    assert detect_intent("What courses do you offer?").intent == "courses"

# --------------------------------------------------------------------------- #
# Round 6 — a follow-up keeps its programme, and an ambiguous one asks
# --------------------------------------------------------------------------- #
def test_eligibility_without_a_programme_asks_instead_of_guessing() -> None:
    """Retrieval always ranks something first; that is not what they asked about.

    "What is the eligibility?" used to come back "For B.Tech Mechanical
    Engineering, the eligibility is…" to a caller who had never mentioned
    Mechanical. Eligibility is what someone decides whether to apply on.
    """
    chunk = _chunk(
        text="NMIMS Global University, Dhule · B.Tech Mechanical Engineering\n"
             "Eligibility: Class 10+2 with Physics and Mathematics, at least 45% in PCM.",
        title="B.Tech Mechanical Engineering",
        category="course",
        structured={"programme": "B.Tech Mechanical Engineering",
                    "eligibility": "Class 10+2 with Physics and Mathematics, at least 45% in PCM"},
        verified=True,
    )
    question = "what is the eligibility"
    answer = compose(_retrieval(question, chunk), language="en-IN", question=question)
    assert "Mechanical" not in answer.text, answer.text
    assert answer.text == FRAMES["en-IN"]["ask_clarify"].format(programme="programme")
    # Asking a question is not giving up: the caller can answer it.
    assert not answer.needs_escalation, answer.escalation_reason
    assert answer.template == "ask_clarify"


@pytest.mark.parametrize(
    ("question", "language", "noun"),
    [
        ("पात्रता क्या है?", "hi-IN", PROGRAMME_NOUN["hi"]),
        ("पात्रता काय आहे?", "mr-IN", PROGRAMME_NOUN["mr"]),
    ],
)
def test_the_clarifying_question_is_asked_in_the_callers_language(
    question: str, language: str, noun: str
) -> None:
    chunk = _chunk(
        text="NMIMS Global University, Dhule · B.Tech Mechanical Engineering\n"
             "Eligibility: Class 10+2 with Physics and Mathematics.",
        title="B.Tech Mechanical Engineering",
        category="course",
        structured={"eligibility": "Class 10+2 with Physics and Mathematics"},
        verified=True,
    )
    answer = compose(_retrieval(question, chunk), language=language, question=question)
    assert answer.text == FRAMES[language]["ask_clarify"].format(programme=noun)
    assert "Mechanical" not in answer.text
    assert not answer.needs_escalation


def test_eligibility_with_a_programme_named_still_answers() -> None:
    """Asking is for when we do not know; it must not become a habit."""
    chunk = _chunk(
        text="NMIMS Global University, Dhule · BBA\n"
             "Eligibility: Class 10+2 from a recognised board, at least 50% aggregate. "
             "Mathematics is not compulsory.",
        title="Bachelor of Business Administration (BBA)",
        category="course",
        structured={"programme": "BBA",
                    "eligibility": "Class 10+2 from a recognised board, at least 50% aggregate. "
                                   "Mathematics is not compulsory."},
        verified=True,
    )
    question = "what is the eligibility for BBA"
    answer = compose(_retrieval(question, chunk), language="en-IN", question=question)
    assert answer.template == "eligibility_long"
    assert "50%" in answer.text and "BBA" in answer.text


def test_a_follow_up_inherits_the_record_not_just_the_degree_token() -> None:
    """"How many seats are there?" must not be answered about another B.Tech.

    `extract_entities` reduces "B.Tech Computer Engineering" to the token BTECH,
    which cannot tell it from the five other B.Tech branches here. Measured
    against the real index, inheriting the token alone ranked the five-year
    B.Tech + MBA dual degree first — sixty seats quoted for a programme that has
    one hundred and eighty. Only the record title is precise enough to carry.
    """
    from app.ai.rag import AnswerRequest, _inherit_programme_context

    bare = detect_intent("How many seats are there?")
    request = AnswerRequest(
        call_id="c", question="How many seats are there?",
        context_record_title="B.Tech Computer Engineering",
        context_course_tokens=["BTECH"],
    )
    inherited = _inherit_programme_context(bare, request)
    assert inherited is not None
    assert inherited.title == "B.Tech Computer Engineering"
    assert inherited.tokens == ["BTECH"]

    # A bare degree with no record behind it is not enough to inherit: guessing
    # a branch is worse than asking.
    coarse = AnswerRequest(call_id="c", question="How many seats are there?",
                           context_course_tokens=["BTECH"])
    assert _inherit_programme_context(bare, coarse) is None

    # A turn that names its own programme is left alone.
    named = detect_intent("How many seats in B.Pharm?")
    assert _inherit_programme_context(named, request) is None

    # So is a question no programme would change the answer to.
    for question in ("How do I reach the campus?", "What is the contact number?",
                     "What is the highest package?"):
        assert _inherit_programme_context(detect_intent(question), request) is None, question
