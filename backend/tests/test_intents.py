"""Intent classification and control-utterance detection.

The intent decides which knowledge-base categories are retrieved, so a
misclassification silently produces a plausible-but-wrong answer.
"""

from __future__ import annotations

import pytest
from app.ai.intents import INTENTS, detect_intent, is_control_utterance


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("what is the fee for B.Tech CSE", "fees"),
        ("बीटेक की फीस कितनी है", "fees"),
        ("am I eligible for B.Pharm", "eligibility"),
        ("what documents do I need", "documents"),
        ("how do I apply", "admission_process"),
        ("hostel mandatory hai kya", "hostel"),
        ("what are the placement records", "placements"),
        ("MHT-CET cutoff kya hai", "entrance_exam"),
        ("last date of admission", "important_dates"),
        ("scholarship mil sakti hai", "scholarships"),
        ("hello", "greeting"),
        ("yes", "affirmation"),
    ],
)
def test_intent_routing(question: str, expected: str) -> None:
    assert detect_intent(question).intent == expected


def test_ask_for_a_human_is_detected_in_hindi() -> None:
    """Escalation must trigger without the caller knowing a magic phrase."""
    result = detect_intent("मुझे किसी व्यक्ति से बात करनी है")
    assert result.intent == "human_request"
    assert result.confidence >= 0.45, "guardrail threshold would not fire"


def test_degree_token_does_not_hijack_a_fee_question() -> None:
    """"B.Tech" pushes `courses`; the caller is really asking about fees."""
    result = detect_intent("what is the fee for B.Tech CSE")
    assert result.intent == "fees"
    assert "BTECH" in result.course_tokens


def test_two_degree_tokens_imply_a_comparison() -> None:
    result = detect_intent("which is better BTech CSE or IT")
    assert result.intent == "comparison"


def test_bare_language_name_reads_as_a_specialisation() -> None:
    """B.A. Hindi / B.A. English are real courses, so this is correct — which is
    why analytics must exclude the caller's *answer* to the language prompt
    rather than "fixing" the classifier."""
    assert detect_intent("Hindi").intent == "courses"
    assert "hindi" in detect_intent("Hindi").specialisations


def test_course_catalogue_question() -> None:
    assert detect_intent("do you offer BBA").intent == "courses"
    assert "BBA" in detect_intent("do you offer BBA").course_tokens


def test_unrelated_text_falls_back_to_other() -> None:
    assert detect_intent("zzz qqq xyzzy").intent == "other"
    assert detect_intent("").intent == "other"


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("please repeat", "repeat"),
        ("एक बार फिर", "repeat"),
        ("talk to a human", "human_request"),
        ("yes", "affirmation"),
        ("no thanks", "negation"),
    ],
)
def test_control_utterances(utterance: str, expected: str) -> None:
    is_control, kind = is_control_utterance(utterance)
    assert is_control is True
    assert kind == expected


def test_control_detection_does_not_swallow_questions() -> None:
    is_control, _ = is_control_utterance("what is the fee for B.Tech CSE")
    assert is_control is False


def test_every_intent_is_declared() -> None:
    """A pattern table entry for an unknown intent would be dead code."""
    from app.ai.intents import _PATTERNS, INTENT_TO_CATEGORIES

    assert set(_PATTERNS) <= set(INTENTS)
    assert set(INTENT_TO_CATEGORIES) <= set(INTENTS)
