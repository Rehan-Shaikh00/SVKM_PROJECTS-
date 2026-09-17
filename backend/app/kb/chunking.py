"""Structure-aware chunking of knowledge-base records.

Chunking strategy matters more than the embedding model for this use case: a fee
question must retrieve a chunk that contains *both* the course name and the fee.
So every chunk carries a **context header** ("NMIMS Global University, Dhule · Fees · B.Tech
Computer Science and Engineering · 2025-26") and structured fields are rendered
into natural sentences before chunking.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

TARGET_CHARS = 620
MAX_CHARS = 900
MIN_CHARS = 120
OVERLAP_CHARS = 90

CATEGORY_LABELS = {
    "course": "Course",
    "specialisation": "Specialisation",
    "eligibility": "Eligibility",
    "fees": "Fees",
    "admission_process": "Admission process",
    "important_dates": "Important dates",
    "entrance_exam": "Entrance exam",
    "scholarships": "Scholarship",
    "hostel": "Hostel",
    "placements": "Placements",
    "facilities": "Campus facilities",
    "documents": "Documents",
    "contact": "Contact",
    "transport": "Transport",
    "loan_payment": "Fees and payment",
    "policy": "Policy",
    "department": "Department",
    "faq": "FAQ",
    "university": "About the university",
}


@dataclass
class Chunk:
    text: str
    position: int
    content_hash: str
    header: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _money(value: Any) -> str:
    """Render a number the way it appears in Indian fee tables."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number >= 10_000_000:
        return f"INR {number / 10_000_000:.2f} crore"
    if number >= 100_000:
        return f"INR {number / 100_000:.2f} lakh"
    return f"INR {number:,.0f}"


#: Keys that hold instructions *to the assistant*. They must never be rendered
#: into chunk text: that text is what the composer reads aloud to a caller.
NEVER_RENDER_KEYS = frozenset({
    "assistant_instruction", "assistant_must_not", "assistant_behaviour", "action",
})


def render_structured(structured: dict[str, Any] | None, category: str) -> str:
    """Turn machine-readable fields into sentences the retriever can match."""
    if not structured:
        return ""
    lines: list[str] = []
    handled: set[str] = set()

    def add(label: str, value: Any) -> None:
        if value in (None, "", [], {}):
            return
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        lines.append(f"{label}: {value}")

    for key in (
        "degree", "programme", "program", "school", "department", "duration_years",
        "duration", "seats", "intake", "mode", "medium", "level",
    ):
        if key in structured:
            label = key.replace("_", " ").title()
            value = structured[key]
            if key == "duration_years":
                value = f"{value} years"
            add(label, value)
            handled.add(key)

    fees = structured.get("fees") or structured.get("fee") or {}
    handled |= {"fees", "fee"}
    if isinstance(fees, dict):
        parts = []
        for label, key in (
            ("annual fee", "annual"), ("first year fee", "year_1"),
            ("total fee", "total"), ("hostel fee", "hostel"),
            ("caution deposit", "deposit"), ("application fee", "application"),
            ("exam fee", "exam"),
        ):
            if key in fees and fees[key] not in (None, ""):
                parts.append(f"{label} {_money(fees[key])}")
        extra = {k: v for k, v in fees.items() if k not in {
            "annual", "year_1", "total", "hostel", "deposit", "application", "exam", "note"}}
        for key, value in extra.items():
            parts.append(f"{key.replace('_', ' ')} {_money(value) if isinstance(value, (int, float)) else value}")
        if parts:
            lines.append("Fees — " + "; ".join(parts) + ".")
        if fees.get("note"):
            lines.append(f"Fee note: {fees['note']}")
    elif isinstance(fees, (int, float, str)):
        lines.append(f"Fees: {_money(fees)}")

    for key, label in (
        ("eligibility", "Eligibility"),
        ("minimum_marks", "Minimum marks"),
        ("entrance_exam", "Entrance exam"),
        ("age_limit", "Age limit"),
        ("selection_process", "Selection process"),
        ("curriculum", "Curriculum highlights"),
        ("career_options", "Career options"),
        ("accreditation", "Accreditation"),
        ("recognition", "Recognition"),
        ("highest_package", "Highest package"),
        ("average_package", "Average package"),
        ("median_package", "Median package"),
        ("recruiters", "Top recruiters"),
        ("placement_rate", "Placement rate"),
        ("facilities", "Facilities"),
        ("room_types", "Room types"),
        ("contact_phone", "Contact phone"),
        ("contact_email", "Contact email"),
        ("address", "Address"),
        ("application_fee", "Application fee"),
        ("important_dates", "Important dates"),
        ("scholarship_types", "Scholarship types"),
        ("documents_required", "Documents required"),
    ):
        if key in structured:
            value = structured[key]
            if isinstance(value, dict):
                value = "; ".join(
                    f"{str(k).replace('_', ' ')} {v}" for k, v in value.items()
                )
            add(label, value)
            handled.add(key)

    # Generic pass over anything the explicit lists above do not know about.
    #
    # A whitelist silently drops new fields, and the drop is invisible until a
    # caller asks: the rewritten KB stores verified travel data under
    # `road_distances` and school numbers under `school_phones`, neither of which
    # was listed, so the facts chunk never contained "330 km" and the numeric
    # grounding guard then refused to let the composer say it out loud — a correct
    # answer came back as an escalation. Staff edit these YAML files (or the admin
    # panel) between admission cycles without a redeploy, so any new key has to be
    # retrievable and has to ground the numbers it carries.
    #
    # The assistant-directed keys stay out: they are instructions to the model, and
    # `render_structured` output can reach the caller.
    for key, value in structured.items():
        if key in handled or key in NEVER_RENDER_KEYS:
            continue
        label = key.replace("_", " ").strip()
        label = label[:1].upper() + label[1:] if label else key
        add(label, value)

    return "\n".join(lines)


def build_header(
    *,
    title: str,
    category: str,
    subcategory: str | None = None,
    academic_year: str | None = None,
    school: str | None = None,
) -> str:
    parts = ["NMIMS Global University, Dhule"]
    label = CATEGORY_LABELS.get(category, category.replace("_", " ").title())
    parts.append(label)
    if school:
        parts.append(school)
    if title and title.lower() not in {label.lower()}:
        parts.append(title)
    if subcategory:
        parts.append(subcategory.replace("_", " "))
    if academic_year:
        parts.append(str(academic_year))
    return " · ".join(dict.fromkeys(parts))


#: See `app.ai.templates.split_sentences` for why abbreviations need protecting:
#: "BBA LL.B. honours" and "Survey No. 499" are not sentence ends, and cutting on
#: them split a single thought across two chunks.
_ABBREV_PERIOD_RE = re.compile(r"(?<![A-Za-z])([A-Z][A-Za-z]{0,2})\.(?=\s+[a-z0-9])")
_ABBREV_SENTINEL = "\x00"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u0964])\s+")


def split_sentences(text: str) -> list[str]:
    protected = _ABBREV_PERIOD_RE.sub(lambda m: m.group(1) + _ABBREV_SENTINEL, text)
    return [
        part.replace(_ABBREV_SENTINEL, ".").strip()
        for part in _SENTENCE_SPLIT_RE.split(protected)
    ]


def _split_text(text: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= MAX_CHARS:
        return [text] if text else []

    # split on paragraph then sentence boundaries
    blocks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= target:
            blocks.append(paragraph)
            continue
        sentences = split_sentences(paragraph)
        buffer = ""
        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip()
            if len(candidate) <= target:
                buffer = candidate
            else:
                if buffer:
                    blocks.append(buffer)
                buffer = sentence
        if buffer:
            blocks.append(buffer)

    # pack blocks into chunks, carrying a small overlap for context continuity
    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n{block}".strip() if current else block
        if len(candidate) <= MAX_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = block
    if current:
        chunks.append(current)

    if overlap and len(chunks) > 1:
        overlapped = [chunks[0]]
        for previous, chunk in itertools.pairwise(chunks):
            tail = previous[-overlap:].strip()
            if tail:
                # avoid cutting mid-word
                cut = tail.find(" ")
                tail = tail[cut + 1 :] if cut > 0 else tail
            overlapped.append(f"…{tail} {chunk}".strip() if tail else chunk)
        chunks = overlapped
    return [c for c in chunks if len(c) >= MIN_CHARS] or chunks


def chunk_text(text: str) -> list[str]:
    """Public helper used by tests and the admin "preview chunking" endpoint."""
    return _split_text(text)


def chunk_record(record: dict[str, Any]) -> list[Chunk]:
    """Chunk one KB record (dict or ORM-like) into retrieval units.

    Emits one *facts chunk* (header + structured rendering) first, because that
    is the highest-signal text for fee/eligibility/date questions, then chunks
    the free-text body.
    """
    title = str(record.get("title") or "").strip()
    category = str(record.get("category") or "faq")
    subcategory = record.get("subcategory")
    academic_year = record.get("academic_year")
    structured = record.get("structured") or {}
    body = str(record.get("body") or "").strip()
    language = record.get("language") or "en-IN"
    tags = record.get("tags") or []
    aliases = record.get("aliases") or []
    localized = record.get("title_localized") or {}

    header = build_header(
        title=title,
        category=category,
        subcategory=subcategory,
        academic_year=academic_year,
        school=(structured.get("school") if isinstance(structured, dict) else None),
    )

    chunks: list[Chunk] = []
    facts = render_structured(structured if isinstance(structured, dict) else {}, category)
    alias_line = ""
    if aliases:
        alias_line = f"\nAlso known as: {', '.join(str(a) for a in aliases)}."
    # Localised titles ride along inside the facts chunk instead of getting a chunk
    # of their own. Separate tiny chunks outscored the chunk that actually holds the
    # data: for "what is the contact number", "संपर्क तपशील" ranked 1.41 while the
    # facts chunk carrying 02562 350620 ranked 1.12 and fell outside the grounding
    # context — so a correct answer was thrown away as an ungrounded number.
    # Merged, the Devanagari title is still indexed for recall on a Marathi or Hindi
    # query, and it sits in the same chunk as the facts that answer it.
    local_line = ""
    local_titles = [
        str(t).strip()
        for t in (localized or {}).values()
        if str(t).strip() and str(t).strip() != title
    ]
    if local_titles:
        # The label is chosen to match the composer's field-dump filter
        # ("Label: value", <=4 words, no apostrophes): these titles exist for
        # retrieval, and a title read aloud is not an answer.
        local_line = "\nIn other languages: " + ", ".join(dict.fromkeys(local_titles)) + "."
    fact_text = f"{header}{alias_line}{local_line}\n{facts}".strip()
    # Aliases are the only Marathi/Hindi surface form many records have, so the
    # facts chunk must exist even when `render_structured` produced nothing:
    # without it 26 of 48 records had their aliases indexed nowhere at all, and
    # a query in the caller's own language could not reach them.
    if facts or alias_line or local_line:
        chunks.append(
            Chunk(
                text=fact_text,
                position=0,
                content_hash=_hash(fact_text),
                header=header,
                metadata={
                    "title": title, "category": category, "chunk_kind": "facts",
                    "language": language, "academic_year": academic_year, "tags": tags,
                },
            )
        )

    if body:
        for index, piece in enumerate(
            _split_text(body), start=1 if (facts or alias_line or local_line) else 0
        ):
            text = f"{header}\n{piece}"
            chunks.append(
                Chunk(
                    text=text,
                    position=index,
                    content_hash=_hash(text),
                    header=header,
                    metadata={
                        "title": title, "category": category, "chunk_kind": "body",
                        "language": language, "academic_year": academic_year, "tags": tags,
                    },
                )
            )

    return chunks


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:32]


def canonical_key(question: str) -> str:
    """Group near-duplicate unanswered questions for the analytics backlog.

    Combining marks (Unicode category ``M*``) are preserved: Devanagari matras
    are not matched by ``\\w``, so a punctuation-only strip used to turn
    "बीटेक की फीस कितनी है" into the unreadable "ब ट क क फ स क तन" on the
    dashboard.
    """
    text = (question or "").lower()
    text = "".join(
        ch
        if (ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M"))
        else " "
        for ch in text
    )
    text = re.sub(r"\b(a|an|the|is|are|do|does|of|for|to|in|at|please|tell|me|i|want|"
                  r"kya|hai|ki|ka|ke|mujhe|bataye|batao|about)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()[:8]
    return " ".join(tokens)[:160] or (question or "")[:60].lower()
