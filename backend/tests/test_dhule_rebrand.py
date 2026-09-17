"""The Dhule rebrand, pinned by tests.

This suite exists because the product was originally built against a different
institution (NIMS University Jaipur, with Rajasthani as the regional language).
Re-pointing it at SVKM's NMIMS Global University, Dhule changes four things a
caller can actually hear:

1. **Who the assistant says it is** — greeting, farewell, prompt scripts and the
   retrieval context header.
2. **Which languages it speaks first** — English, Hindi and *Marathi*, with
   Marathi as a fully-scripted, fully-framed language rather than a fallback.
3. **What is in the knowledge base** — Dhule programmes, and crucially *no fee
   numbers that nobody could source*, so the assistant escalates instead of
   inventing one.
4. **What it must never say again** — the old institution name, city, helpline
   number or medical programmes that this campus does not run.

Every test here is unit-level: no database, no network, no API keys.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from app.ai.intents import detect_intent
from app.ai.spoken_numbers import duration_phrase, number_to_words_mr, spoken_count, spoken_money
from app.ai.templates import FRAMES, FULL_FRAMES, compose
from app.config import settings
from app.i18n.languages import (
    HINGLISH_MARATHI_MARKERS,
    get_language,
    script_line,
)
from app.kb.chunking import build_header
from app.kb.embeddings import expand
from app.kb.retriever import RetrievalResult, RetrievedChunk
from app.voice.lid.local import detect_language_text

BACKEND_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BACKEND_DIR.parent / "data" / "kb"

INSTITUTION = "NMIMS Global University, Dhule"
OLD_BRAND_PATTERNS = (
    "NIMS University",
    "Jaipur",
    "johra",
    "shobha",
    "Rajasthan",
    "NIMSEE",
    "1800 120 1020",
    "nimsuniversity",
)
HELPLINE = "1800 102 5138"


def _kb_records() -> list[tuple[str, dict]]:
    """Every seeded record, tagged with the file it came from."""
    rows: list[tuple[str, dict]] = []
    for path in sorted(KB_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, list), f"{path.name} must be a YAML list of records"
        rows.extend((path.name, record) for record in data)
    assert rows, "the seed KB is empty"
    return rows


# --------------------------------------------------------------------------- #
# 1. Identity: what the assistant calls itself
# --------------------------------------------------------------------------- #
def test_greeting_and_farewell_name_the_dhule_campus() -> None:
    for code in ("en-IN", "hi-IN", "mr-IN"):
        greeting = script_line(code, "greeting")
        farewell = script_line(code, "farewell")
        assert greeting and farewell, f"{code} is missing a greeting/farewell"
        assert re.search(r"Dhule|धुले|धुळे", greeting + farewell), (
            f"{code} greeting/farewell does not mention Dhule: {greeting!r}"
        )


def test_language_prompt_is_trilingual_english_hindi_marathi() -> None:
    """The opening prompt must ask in the three languages the campus hears."""
    prompt = script_line("en-IN", "language_prompt")
    assert "Please tell me your preferred language" in prompt
    assert "कृपया अपनी भाषा बताइए" in prompt
    # Marathi, not Rajasthani: "आपली" is Marathi, "अपणी" was the old Rajasthani line.
    assert "आपली भाषा" in prompt
    assert "अपणी" not in prompt


def test_unsupported_language_message_offers_marathi() -> None:
    for code in ("en-IN", "hi-IN", "mr-IN"):
        line = script_line(code, "language_unsupported", language="Tamil")
        assert re.search(r"Marathi|मराठी", line), f"{code} does not offer Marathi: {line!r}"


def test_greeting_languages_are_english_hindi_marathi() -> None:
    assert settings.greeting_language_list == ["en-IN", "hi-IN", "mr-IN"]
    # Every greeting language must actually have a script set, otherwise the
    # caller hears the generic English fallback in the middle of a Marathi call.
    for code in settings.greeting_language_list:
        assert get_language(code).scripts, f"{code} has no scripted prompts"


def test_marathi_scripts_cover_every_english_key() -> None:
    """A missing key silently falls back to English mid-call."""
    english = set(get_language("en-IN").scripts)
    marathi = set(get_language("mr-IN").scripts)
    assert english - marathi == set(), f"mr-IN is missing: {sorted(english - marathi)}"


def test_retrieval_context_header_is_branded_for_dhule() -> None:
    header = build_header(
        title="B.Tech Computer Science and Engineering",
        category="course",
        subcategory="engineering",
        academic_year="2026-27",
    )
    assert header.startswith(INSTITUTION), header


# --------------------------------------------------------------------------- #
# 2. Language identification — Marathi in Devanagari *and* in Latin script
# --------------------------------------------------------------------------- #
ALLOWED = tuple(settings.supported_language_list)


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        # Devanagari, disambiguated from Hindi by function words
        ("मला बीटेक ची माहिती पाहिजे, शुल्क किती आहे", "mr-IN"),
        ("प्रवेशासाठी किती गुण आवश्यक आहेत", "mr-IN"),
        ("मुझे बीटेक की फीस बताइए", "hi-IN"),
        # Romanised Marathi — the case that used to be scored as English
        ("mala marathi bolayche aahe", "mr-IN"),
        ("btech chi mahiti pahije shulka kiti aahe", "mr-IN"),
        ("mi marathi madhe bolu shakto", "mr-IN"),
        # Romanised Hindi and plain English must not drift into Marathi
        ("mujhe btech ki fees bataiye", "hi-IN"),
        ("what is the fee for computer science engineering", "en-IN"),
        # Naming a language always wins
        ("मराठी", "mr-IN"),
        ("ગુજરાતી", "gu-IN"),
    ],
)
def test_local_lid_routes_marathi_correctly(utterance: str, expected: str) -> None:
    result = detect_language_text(utterance, candidates=ALLOWED)
    assert result.language == expected, f"{utterance!r} -> {result.language} ({result.detail})"
    assert result.confidence >= settings.lid_confidence_threshold or result.method == "explicit_name"


# Hindi and Marathi share the Devanagari script, so the script vote is worth
# nothing on its own: both codes get it. These sentences are unambiguously
# Marathi, and every one of them used to tie Hindi at 6.0-6.0 and lose, because
# the winner was whichever code came first in the candidate list. A Marathi
# caller from Dhule was answered in Hindi.
MARATHI_ONLY_QUESTIONS = (
    "सेमेस्टर कधी सुरू होईल?",
    "कागदपत्रे कोणती लागतील?",
    "परीक्षा कधी होणार?",
    "प्रवेशासाठी कोणती परीक्षा द्यावी लागेल?",
    "त्याची पात्रता काय आहे?",
    "अभ्यासक्रमांची यादी सांगा",
)

HINDI_ONLY_QUESTIONS = (
    "परीक्षा कब होगी?",
    "कौन से डॉक्यूमेंट्स चाहिए?",
    "इसका शुल्क कितना लगेगा?",
    "एडमिशन कब तक खुला रहेगा?",
    "बीटेक की पात्रता क्या है?",
)


@pytest.mark.parametrize("utterance", MARATHI_ONLY_QUESTIONS)
def test_marathi_question_words_beat_the_shared_script(utterance: str) -> None:
    """A Marathi sentence must win on its own words, not on script order."""
    result = detect_language_text(utterance, candidates=ALLOWED)
    assert result.language == "mr-IN", f"{utterance!r} -> {result.language} ({result.detail})"
    # Above the bare script vote (6.0), i.e. it found Marathi evidence.
    assert result.scores.get("mr-IN", 0.0) > result.scores.get("hi-IN", 0.0), result.scores


@pytest.mark.parametrize("utterance", HINDI_ONLY_QUESTIONS)
def test_hindi_still_wins_its_own_sentences(utterance: str) -> None:
    """Expanding the Marathi list must not steal Hindi callers."""
    result = detect_language_text(utterance, candidates=ALLOWED)
    assert result.language == "hi-IN", f"{utterance!r} -> {result.language} ({result.detail})"


def test_a_devanagari_tie_breaks_toward_the_campus_language() -> None:
    """With no lexical evidence either way, order must not decide.

    "संगणक" is Devanagari and in neither marker list, so Hindi and Marathi score
    identically. The campus is in Maharashtra, so the tie goes to Marathi — and
    it must do that whichever way the candidate list happens to be ordered.
    """
    tied = "संगणक"
    for candidates in (("en-IN", "hi-IN", "mr-IN"), ("mr-IN", "hi-IN", "en-IN"),
                       ("hi-IN", "mr-IN")):
        result = detect_language_text(tied, candidates=candidates)
        assert result.language == settings.devanagari_preference, (
            f"{candidates} -> {result.language}; tie broken by candidate order, "
            "not by the configured regional language"
        )


def test_the_english_word_me_is_not_romanised_hindi() -> None:
    """"Tell me about placements." was detected as Hindi and answered in Devanagari.

    Bare "me" sat in the Hinglish marker list as the Hindi postposition, but it
    is also one of the commonest English words, and this sentence had no English
    marker in it either, so Hindi won with 0.98 confidence.
    """
    from app.i18n.languages import HINGLISH_MARKERS

    assert "me" not in HINGLISH_MARKERS
    # "mein", "mujhe" and "mera" carry the same signal without the collision.
    assert {"mein", "mujhe", "mera"} <= set(HINGLISH_MARKERS)
    for utterance in ("Tell me about placements.", "Can you send me the details?",
                      "Please text me the brochure."):
        result = detect_language_text(utterance, candidates=ALLOWED)
        assert result.language == "en-IN", f"{utterance!r} -> {result.language}"


@pytest.mark.parametrize(
    "utterance",
    [
        "When do classes start?",
        "Are scholarships available?",
        "Which documents should I bring?",
        "I want to talk to a person.",
        "Is parking available near campus?",
    ],
)
def test_an_english_question_never_comes_back_with_zero_confidence(utterance: str) -> None:
    """Zero confidence re-prompts a caller who already asked in English.

    English is the only Latin-script candidate on this line, so wholly Latin
    text with no romanised-Indic marker in it is evidence in itself.
    """
    result = detect_language_text(utterance, candidates=ALLOWED)
    assert result.language == "en-IN", f"{utterance!r} -> {result.language}"
    assert result.confidence >= settings.lid_confidence_threshold, (
        f"{utterance!r} scored {result.confidence} — the caller would be asked "
        "to choose a language again"
    )


def test_romanised_marathi_marker_list_is_marathi_specific() -> None:
    """Guards against adding a word that is really English or Hindi.

    If one of these appeared in romanised Hindi, every Hinglish caller on the
    Dhule line would be switched into Marathi.
    """
    from app.i18n.languages import HINGLISH_MARKERS, HINGLISH_RAJASTHANI_MARKERS

    marathi = set(HINGLISH_MARATHI_MARKERS)
    # Checked against the real lists, not a hand-picked subset: "nahi" was in
    # here once, and Hindi "नहीं" and Marathi "नाही" both romanise to it.
    assert not marathi & set(HINGLISH_MARKERS), sorted(marathi & set(HINGLISH_MARKERS))
    assert not marathi & set(HINGLISH_RAJASTHANI_MARKERS)
    english = {"the", "fee", "course", "and", "for", "what", "is", "year", "of", "to"}
    assert not marathi & english, sorted(marathi & english)


# --------------------------------------------------------------------------- #
# 3. Marathi answer composition (the zero-key template path)
# --------------------------------------------------------------------------- #
def test_marathi_frames_have_parity_with_english() -> None:
    assert "mr-IN" in FULL_FRAMES
    assert set(FRAMES["mr-IN"]) == set(FRAMES["en-IN"])


def _fee_retrieval(*, verified: bool, total: int) -> RetrievalResult:
    chunk = RetrievedChunk(
        chunk_id="c1",
        record_id="r1",
        text="BBA is a three year full time programme. The fee is published per cycle.",
        title="BBA (Bachelor of Business Administration)",
        category="course",
        language="en-IN",
        verified=verified,
        academic_year="2026-27",
        score=0.82,
        structured={"fees": {"total": total}, "duration_years": 3},
    )
    return RetrievalResult(
        query="BBA ki fees kitni hai",
        items=[chunk],
        intent=detect_intent("BBA ki fees kitni hai"),
        best_score=0.82,
        grounded=True,
    )


def test_marathi_fee_answer_is_spoken_in_marathi_numbers() -> None:
    answer = compose(
        _fee_retrieval(verified=True, total=444000),
        language="mr-IN",
        question="BBA chi fees kiti aahe",
    )
    assert "शुल्क" in answer.text
    # 4,44,000 -> "4 लाख 44 हजार रुपये", never "four lakh forty four thousand"
    assert "लाख" in answer.text and "रुपये" in answer.text
    assert answer.grounded is True


def test_unverified_fee_answer_carries_the_marathi_hedge() -> None:
    question = "BBA chi fees kiti aahe"
    answer = compose(
        _fee_retrieval(verified=False, total=444000), language="mr-IN", question=question
    )
    assert "प्रवेश कार्यालयाकडून" in answer.text, answer.text
    verified_answer = compose(
        _fee_retrieval(verified=True, total=444000), language="mr-IN", question=question
    )
    assert answer.confidence < verified_answer.confidence


@pytest.mark.parametrize(
    ("value", "language", "expected"),
    [
        (150000, "mr-IN", "1 लाख 50 हजार"),
        (4440000, "mr-IN", "44 लाख 40 हजार"),
        (0, "mr-IN", "शून्य"),
    ],
)
def test_marathi_number_words(value: int, language: str, expected: str) -> None:
    assert number_to_words_mr(value) == expected
    assert spoken_money(value, language).startswith(expected)


def test_marathi_duration_pluralises_the_noun() -> None:
    """Marathi says '1 वर्ष' but '4 वर्षे' — the noun changes, not the digit."""
    assert duration_phrase(1, "mr-IN") == "1 वर्ष"
    assert duration_phrase(4, "mr-IN") == "4 वर्षे"
    # and it must not leak the Hindi word
    assert duration_phrase(4, "hi-IN") == "4 साल"


def test_marathi_money_suffixes() -> None:
    assert spoken_money(150000, "mr-IN", per="year").endswith("प्रति वर्ष")
    assert "संपूर्ण अभ्यासक्रमासाठी" in spoken_money(444000, "mr-IN", per="total")


def test_marathi_counts_stay_as_digits() -> None:
    """A Marathi caller hears '120 जागा', never a spelled-out count."""
    assert spoken_count(120, "जागा", "mr-IN") == "120 जागा"


# --------------------------------------------------------------------------- #
# 4. Marathi intent routing (the intent picks the KB categories to retrieve)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("बीटेक चे शुल्क किती आहे", "fees"),
        ("हॉस्टेल फी किती", "fees"),
        ("प्रवेश प्रक्रिया काय आहे", "admission_process"),
        ("प्रवेशासाठी किती गुण आवश्यक आहेत", "eligibility"),
        ("वसतिगृह उपलब्ध आहे का", "hostel"),
        ("कोणते अभ्यासक्रम आहेत", "courses"),
        ("शिष्यवृत्ती उपलब्ध आहे का", "scholarships"),
        ("कोणती कागदपत्रे लागतील", "documents"),
        ("प्रवेशाची शेवटची तारीख काय आहे", "important_dates"),
        ("कॅम्पस ला कसे पोहोचावे", "transport"),
        ("प्लेसमेंट कशी आहे", "placements"),
        ("मला माणसाशी बोलायचे आहे", "human_request"),
    ],
)
def test_marathi_intents_route_correctly(question: str, expected: str) -> None:
    assert detect_intent(question).intent == expected


# --------------------------------------------------------------------------- #
# 5. The seeded knowledge base is Dhule's, and invents nothing
# --------------------------------------------------------------------------- #
def test_seed_kb_has_no_trace_of_the_old_institution() -> None:
    for path in sorted(KB_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for pattern in OLD_BRAND_PATTERNS:
            assert pattern not in text, f"{path.name} still mentions {pattern!r}"


def test_every_record_targets_the_current_cycle_and_cites_a_source() -> None:
    for filename, record in _kb_records():
        assert record.get("academic_year") == "2026-27", f"{filename}:{record.get('slug')}"
        assert record.get("source"), f"{filename}:{record.get('slug')} has no source"
        assert record.get("slug") and record.get("title") and record.get("body"), (
            f"{filename}:{record.get('slug')} is missing a core field"
        )


def test_record_slugs_are_globally_unique() -> None:
    slugs = [record["slug"] for _, record in _kb_records()]
    assert len(slugs) == len(set(slugs)), "duplicate slugs across KB files"


def test_unsourced_fees_are_absent_rather_than_guessed() -> None:
    """The single most dangerous failure mode on an admissions line.

    Engineering, pharmacy, computer applications, MBA, M.Com and Ph.D. fees could
    not be sourced, so those records must carry no number at all — the assistant
    then escalates. Any numeric fee that *is* present must be unverified and
    labelled indicative, so it can never be spoken as fact.
    """
    for filename, record in _kb_records():
        structured = record.get("structured") or {}
        numbers: list[float] = []
        for key, value in structured.items():
            if key in {"annual_fee", "total_fee", "hostel_fee", "application_fee"} and isinstance(
                value, (int, float)
            ):
                numbers.append(float(value))
            fees = value if key == "fees" and isinstance(value, dict) else {}
            numbers.extend(float(v) for v in fees.values() if isinstance(v, (int, float)))
        if not numbers:
            continue
        assert record.get("verified") is False, (
            f"{filename}:{record['slug']} states a fee as verified — a fee may only "
            "be verified by admissions staff against the official handbook"
        )
        blob = yaml.safe_dump(record, allow_unicode=True)
        assert "indicative" in blob.lower() or "SAMPLE" in blob, (
            f"{filename}:{record['slug']} quotes a fee without labelling it indicative"
        )


def test_engineering_and_pharmacy_records_quote_no_fee_number() -> None:
    for filename in ("10_courses_engineering.yaml", "20_courses_pharmacy.yaml"):
        path = KB_DIR / filename
        assert path.exists(), f"{filename} is missing from the seed KB"
        for record in yaml.safe_load(path.read_text(encoding="utf-8")):
            fees = (record.get("structured") or {}).get("fees") or {}
            numeric = [v for v in fees.values() if isinstance(v, (int, float))]
            assert not numeric, f"{filename}:{record['slug']} invented a fee: {numeric}"


def test_disambiguation_and_no_medical_faqs_are_seeded() -> None:
    """Three questions Dhule callers ask that an unbriefed assistant gets wrong."""
    slugs = {record["slug"] for _, record in _kb_records()}
    # Three SVKM NMIMS entities sit in Dhule district plus the Mumbai deemed
    # university; a caller who reaches this line may have applied to the wrong one.
    assert "faq-global-university-vs-other-svkm-nmims" in slugs
    # B.Com / M.Com / BBA-LL.B. / a standalone MBA belong to those other entities.
    assert "faq-programmes-not-offered-here" in slugs
    assert "faq-no-medical-programmes" in slugs

    record = next(
        r for _, r in _kb_records() if r["slug"] == "faq-no-medical-programmes"
    )
    blob = record.get("body", "") + str(record.get("structured", ""))
    for programme in ("MBBS", "BDS", "nursing"):
        assert programme.lower() in blob.lower(), f"{programme} is not named in the FAQ"

    # and the "not this university" FAQ names the entities it redirects to
    redirect = next(
        r for _, r in _kb_records()
        if r["slug"] == "faq-programmes-not-offered-here"
    ).get("body", "")
    assert "NMIMS Mumbai" in redirect or "Mumbai" in redirect


def test_escalation_boundaries_record_forbids_inventing_a_number() -> None:
    """The behaviour policy still forbids the two things a helpline must not guess."""
    records = {record["slug"]: record for _, record in _kb_records()}
    policy = records["policy-escalation-boundaries"]
    assert policy["structured"]["must_escalate_to_human"], "escalation list is empty"
    assert "fee" in policy["structured"]["never_state"].lower()
    assert "date" in policy["structured"]["never_state"].lower()


def test_every_verified_record_cites_the_university_or_is_internal_policy() -> None:
    """Verification now means 'read on svkmnmimsgu.ac.in', not 'plausible'.

    The seeded KB used to mark almost everything unverified because it came from
    aggregator listings; it has been rewritten from the university's own pages, so
    the bar moved: a record may only claim `verified: true` if it names the
    official site, or if it is the assistant's own behaviour policy, which makes no
    external claim at all.
    """
    internal = {"policy-escalation-boundaries"}
    verified = 0
    for _, record in _kb_records():
        if not record.get("verified"):
            continue
        verified += 1
        if record["slug"] in internal:
            continue
        citation = f"{record.get('source', '')} {record.get('verified_by', '')}"
        assert "svkmnmimsgu.ac.in" in citation, (
            f"{record['slug']} claims verification without citing the university site"
        )
    # the rewrite is meant to move the KB from guesses to sourced facts
    assert verified >= 40, f"only {verified} verified records — the rewrite regressed"


def test_contact_record_carries_only_the_numbers_the_website_publishes() -> None:
    """The contact page publishes school-wise numbers and an enquiry form — no
    toll-free line and no email address, so neither may appear in the KB."""
    contact = next(
        record for _, record in _kb_records() if record["slug"] == "university-contact"
    )
    structured = contact["structured"]
    assert "2562" in structured["contact_phone"]          # Dhule STD code
    assert len(structured["school_phones"]) == 3
    assert all("2562" in line or re.search(r"\d{5}", line) for line in structured["school_phones"])
    assert structured["website"] == "https://www.svkmnmimsgu.ac.in"
    assert structured["admission_portal"].startswith("https://sdcappscs.svkm.ac.in")
    assert "enquiry" in structured["enquiry_form"].lower()

    # the aggregator toll-free line and mailbox must not survive anywhere in the KB
    blob = "".join(
        record.get("body", "") + str(record.get("structured", "")) + str(record.get("aliases", ""))
        for _, record in _kb_records()
    )
    assert HELPLINE not in blob, "the unpublished toll-free number is still in the KB"
    assert "dhule@nmims.edu" not in blob, "the unpublished mailbox is still in the KB"


def test_address_is_the_dhule_campus_on_the_mumbai_agra_highway() -> None:
    address = next(
        record for _, record in _kb_records() if record["slug"] == "university-address"
    )
    structured = address["structured"]
    line = str(structured["address"])
    assert structured["pin_code"] == "424001"
    assert "Dhule" in line
    assert "Maharashtra" in line
    assert "Survey No. 499" in line or "Survey No 499" in line
    # the official wording, not the aggregator's "NH-3 campus" shorthand
    assert "Mumbai Agra National Highway" in line
    assert "Gurudwara" in str(structured["landmark"])
    assert structured["website"] == "https://www.svkmnmimsgu.ac.in"


# --------------------------------------------------------------------------- #
# 6. Retrieval vocabulary follows the campus, not the old city
# --------------------------------------------------------------------------- #
def _groups(text: str) -> set[str]:
    return {f for f in expand(text) if f.startswith("#g:")}


def test_place_and_institution_synonyms_are_dhule_ones() -> None:
    """Retrieval expands a Marathi place word and its English one to one feature.

    Without this, "धुळे कॅम्पस" and "the Dhule campus" retrieve different chunks.
    """
    assert "#g:dhule" in _groups("how do I reach the Dhule campus")
    assert "#g:dhule" in _groups("धुळे कॅम्पस ला कसे पोहोचावे")
    assert "#g:nmims" in _groups("NMIMS Global University admission")
    assert "#g:nmims" in _groups("SVKM एनएमआयएमएस प्रवेश")
    # the old city is no longer a synonym of anything
    assert "#g:dhule" not in _groups("campus in Jaipur")


def test_common_marathi_words_are_not_synonym_group_members() -> None:
    """Regression guard on the synonym table.

    Members are split on spaces and every piece becomes a token -> group mapping,
    so adding "माहिती तंत्रज्ञान" (information technology) silently tagged every
    question containing "माहिती" — Marathi for plain "information" — as computer
    science, and "प्रवेश प्रक्रिया काय" did the same with "काय" ("what").
    """
    for word in ("माहिती", "काय", "तंत्रज्ञान", "आहे", "किती", "मला", "पाहिजे", "द्या"):
        assert not _groups(word), f"{word!r} should not expand to any group"


def test_marathi_cross_script_vocabulary_reaches_english_records() -> None:
    """Marathi aliases in the KB and Marathi synonyms in retrieval must agree."""
    from app.ai.rag import CROSS_SCRIPT_COURSES

    for marathi, latin in (
        ("अभियांत्रिकी", "Engineering"),
        ("औषधनिर्माणशास्त्र", "Pharmacy"),
        ("वसतिगृह", "Hostel"),
        ("शिष्यवृत्ती", "Scholarship"),
        ("शुल्क", "Fees"),
    ):
        assert CROSS_SCRIPT_COURSES.get(marathi) == latin


def test_kb_records_carry_marathi_aliases_where_callers_will_use_them() -> None:
    """A Marathi caller says 'शुल्क' while the record is titled 'Fees'."""
    devanagari = re.compile(r"[\u0900-\u097F]")
    hits = 0
    for _, record in _kb_records():
        aliases = record.get("aliases") or []
        if any(devanagari.search(str(a)) for a in aliases):
            hits += 1
    total = len(_kb_records())
    # Nearly every record: a Marathi caller says "शुल्क" while the record is
    # titled "Fees", and Indic aliases are what bridge the two.
    assert hits >= total * 0.9, f"only {hits}/{total} records have Indic aliases"
