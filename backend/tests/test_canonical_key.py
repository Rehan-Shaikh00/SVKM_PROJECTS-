"""Question canonicalisation for the analytics backlog and top-queries panel."""

from __future__ import annotations

from app.kb.chunking import canonical_key


def test_devanagari_vowel_signs_survive() -> None:
    """Regression: matras are Unicode category Mn, which ``\\w`` does not match,
    so "बीटेक की फीस कितनी है" was rendered unreadable as "ब ट क क फ स क तन"."""
    key = canonical_key("बीटेक की फीस कितनी है")
    assert key == "बीटेक की फीस कितनी है"
    assert "बीटेक" in key


def test_english_stopwords_are_dropped() -> None:
    assert canonical_key("What is the fee for B.Tech CSE?") == "what fee b tech cse"


def test_near_duplicates_group_together() -> None:
    """Same question, different casing/punctuation -> same bucket."""
    a = canonical_key("What is the fee for B.Tech CSE?")
    b = canonical_key("what is the fee for b.tech cse")
    assert a == b


def test_transliterated_hindi_keeps_content_words() -> None:
    assert canonical_key("hostel mandatory hai kya") == "hostel mandatory"


def test_punctuation_is_removed_but_words_are_kept() -> None:
    """canonical_key is a grouping key, not a content filter: punctuation and
    stopwords go, the caller's own words stay (a URL becomes plain tokens)."""
    key = canonical_key("Fee for B.Tech CSE??")
    assert key == "fee b tech cse"

    url_key = canonical_key("see: https://dhule.nmims.edu/fees !!")
    assert ":" not in url_key
    assert "/" not in url_key
    assert "!" not in url_key
    assert "nmims" in url_key


def test_long_questions_are_capped() -> None:
    key = canonical_key(" ".join(f"word{i}" for i in range(60)))
    assert len(key) <= 160
    assert len(key.split()) <= 8


def test_empty_input_is_safe() -> None:
    assert canonical_key("") == ""
    assert canonical_key("   ") == "   " or canonical_key("   ") == ""
