"""Voice-friendly sentence splitting.

Regression cover for a bug that made the assistant *repeat itself on every
long answer*: `_split_long` flushed its buffer and then sliced a candidate that
still contained that buffer, so callers heard the same clause twice.
"""

from __future__ import annotations

import pytest
from app.voice.tts.base import sanitize_for_speech, split_for_speech

HI_DOCUMENTS = (
    "आपको Class 10 marksheet and certificate (proof of date of birth), "
    "Class 12 marksheet and certificate और Graduation and postgraduation "
    "marksheets and degree, where applicable चाहिए होंगे।"
)
EN_DOCUMENTS = (
    "You will need Class 10 marksheet and certificate, Class 12 marksheet and "
    "certificate, graduation marksheets and degree, caste certificate and income "
    "proof, plus the entrance exam scorecard where applicable."
)


def _no_duplicates(chunks: list[str]) -> bool:
    """True when no chunk contains another chunk's full text."""
    for i, chunk in enumerate(chunks):
        for other in chunks[i + 1 :]:
            if chunk and chunk in other:
                return False
    return True


def test_short_text_is_not_split() -> None:
    text = "Fees are 1.4 lakh rupees per year."
    assert split_for_speech(text, max_chars=180) == [text]


def test_empty_text_returns_no_chunks() -> None:
    assert split_for_speech("") == []
    assert split_for_speech("   ") == []


@pytest.mark.parametrize("text", [HI_DOCUMENTS, EN_DOCUMENTS])
def test_long_answer_is_never_repeated(text: str) -> None:
    chunks = split_for_speech(text, max_chars=180)
    assert len(chunks) > 1, "expected the long answer to be split for streaming"
    assert _no_duplicates(chunks), f"assistant would repeat itself: {chunks}"


@pytest.mark.parametrize("text", [HI_DOCUMENTS, EN_DOCUMENTS])
def test_splitting_preserves_every_word(text: str) -> None:
    """Concatenating the chunks must reproduce the source (modulo spacing)."""
    chunks = split_for_speech(text, max_chars=180)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_conjunctions_survive_the_split() -> None:
    """A consuming split silently deleted every "and"/"और" from long answers."""
    chunks = split_for_speech(EN_DOCUMENTS, max_chars=180)
    assert " ".join(chunks).count(" and ") == EN_DOCUMENTS.count(" and ")

    hi_chunks = split_for_speech(HI_DOCUMENTS, max_chars=180)
    assert " ".join(hi_chunks).count("और") == HI_DOCUMENTS.count("और")


def test_unit_words_are_not_split_points() -> None:
    """"per" is not a conjunction: "per year" must keep its unit."""
    text = (
        "B.Tech fees are 1.4 lakh rupees per year and the total programme cost is "
        "about 5.6 lakh rupees, excluding hostel charges which are billed separately."
    )
    joined = " ".join(split_for_speech(text, max_chars=120))
    assert "per year" in joined


def test_very_long_single_clause_is_hard_sliced_within_budget() -> None:
    text = "word " * 200
    chunks = split_for_speech(text.strip(), max_chars=180)
    assert len(chunks) > 1
    assert all(len(c) <= 180 for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.strip().replace(" ", "")


def test_chunks_respect_the_budget_for_real_answers() -> None:
    for text in (HI_DOCUMENTS, EN_DOCUMENTS):
        for chunk in split_for_speech(text, max_chars=180):
            # split_for_speech merges tiny trailing fragments, so allow the
            # documented slack rather than demanding a hard 180.
            assert len(chunk) <= 180 + 40


def test_sanitize_strips_markdown_and_symbols() -> None:
    dirty = "**B.Tech** fees: ₹1,40,000/yr\n- hostel included\nhttps://dhule.nmims.edu/fees"
    clean = sanitize_for_speech(dirty)
    assert "**" not in clean
    assert "\n" not in clean
    assert "https://" not in clean
    # The degree must survive in a *speakable* form: TTS reads "B.Tech" as
    # "B dot Tech", so dropping the period is deliberate.
    assert "B Tech" in clean or "B.Tech" in clean
    # Currency is expanded the way Indians say it, digits untouched.
    assert "rupees" in clean
    assert "1,40,000" in clean
    # Bullets are stripped but their content is kept.
    assert "hostel included" in clean
