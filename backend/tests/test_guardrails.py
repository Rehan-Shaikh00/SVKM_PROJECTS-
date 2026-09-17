"""Guardrails: numeric grounding, PII redaction, topic safety, voice scrubbing.

These are the guarantees the brief makes explicit — never invent a fee, never
log personal data in the clear, and hand a distressed caller to a human.
"""

from __future__ import annotations

import pytest
from app.ai import guardrails

CONTEXT = "B.Tech Information Technology fee is 1,40,000 per year. Total programme cost 5,60,000."


# --------------------------------------------------------------------------- #
# numeric grounding — the anti-hallucination gate
# --------------------------------------------------------------------------- #
def test_grounded_fee_passes() -> None:
    check = guardrails.numeric_grounding_check(
        "The fee is 1.4 lakh rupees per year.", CONTEXT
    )
    assert check["ok"] is True
    assert check["ungrounded_numbers"] == []


def test_invented_fee_is_rejected() -> None:
    """The single most important guarantee: no fee may be stated that is not in
    the retrieved knowledge base."""
    check = guardrails.numeric_grounding_check(
        "The fee is 9 lakh rupees per year.", CONTEXT
    )
    assert check["ok"] is False
    assert "900000" in check["ungrounded_numbers"]


def test_hindi_scale_words_are_understood() -> None:
    """लाख/करोड़/हज़ार compose a number; a naive digit match would miss it."""
    check = guardrails.numeric_grounding_check(
        "फीस 1 लाख 40 हज़ार रुपये प्रति वर्ष है।", CONTEXT
    )
    assert check["ok"] is True
    assert "140000" in check["answer_numbers"]


def test_hindi_invented_amount_is_rejected() -> None:
    check = guardrails.numeric_grounding_check("फीस 9 लाख रुपये प्रति वर्ष है।", CONTEXT)
    assert check["ok"] is False


# --------------------------------------------------------------------------- #
# PII redaction
# --------------------------------------------------------------------------- #
def test_each_pii_type_is_labelled_correctly() -> None:
    """Regression: the redaction lambda closed over the loop variable, so every
    match was tagged with the *last* label ([pan:...] for emails and URLs)."""
    text = (
        "email ram@nmims.edu, see https://dhule.nmims.edu/admissions, "
        "PAN ABCDE1234F"
    )
    redacted, found = guardrails.redact_pii(text)
    assert "[email:" in redacted
    assert "[url:" in redacted
    assert "[pan:" in redacted
    assert set(found) == {"email", "url", "pan"}
    assert "ram@nmims.edu" not in redacted
    assert "ABCDE1234F" not in redacted


def test_phone_is_masked_but_tail_kept_for_the_agent() -> None:
    redacted, found = guardrails.redact_pii("call me on 9876543210 tomorrow")
    assert "[phone:" in redacted
    assert "9876543210" not in redacted
    assert found["phone"], "the hashed value must be kept for correlation"


def test_redaction_is_deterministic() -> None:
    first, _ = guardrails.redact_pii("my pan is ABCDE1234F")
    second, _ = guardrails.redact_pii("my pan is ABCDE1234F")
    assert first == second


# --------------------------------------------------------------------------- #
# topic safety
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "my daughter was harassed on campus",
        "a student was molested in the hostel",
        "he was assaulted by seniors",
        "there is caste discrimination in the hostel",
        "I was threatened by an agent",
        "I want to file a complaint against the admissions office",
        "I will go to consumer court",
        "I am feeling hopeless and depressed",
        "मेरी बेटी के साथ छेड़छाड़ हुई",
        "मुझे शिकायत दर्ज करनी है",
        "रैगिंग हुई है",
    ],
)
def test_harm_and_grievance_reach_a_human(text: str) -> None:
    """Verb forms matter: callers say "was harassed", not "harassment"."""
    assert guardrails.is_sensitive(text) is True
    assert guardrails.classify_topic(text)[0] == "sensitive"


@pytest.mark.parametrize(
    "text",
    [
        "what is the fee for B.Tech CSE",
        "am I eligible for B.Pharm",
        "hostel mandatory hai kya",
        "do you have a scholarship for OBC students",
        "how to reach the campus from Dhule station",
        "बीटेक की फीस कितनी है",
    ],
)
def test_ordinary_admission_queries_are_not_flagged(text: str) -> None:
    """Over-escalating would defeat the purpose of the assistant."""
    assert guardrails.is_sensitive(text) is False
    assert guardrails.classify_topic(text)[0] != "sensitive"


def test_off_topic_is_detected() -> None:
    assert guardrails.is_off_topic("what is the weather today") is True
    assert guardrails.is_off_topic("what is the hostel fee") is False


# --------------------------------------------------------------------------- #
# voice scrubbing
# --------------------------------------------------------------------------- #
def test_long_answers_are_trimmed_for_speech() -> None:
    result = guardrails.scrub_for_voice("Fees are 1.4 lakh. " + "Details follow. " * 80)
    assert len(result.text) <= guardrails.MAX_SPOKEN_CHARS + 40
    assert any("truncated" in w for w in result.warnings)


def test_markdown_is_stripped_before_speech() -> None:
    result = guardrails.scrub_for_voice("**B.Tech** fee is `1,40,000` per year")
    assert "**" not in result.text
    assert "`" not in result.text
