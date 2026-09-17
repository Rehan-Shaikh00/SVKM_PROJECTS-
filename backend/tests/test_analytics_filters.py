"""Analytics turn filtering.

The caller's *answer* to "which language do you speak?" is not a question. It is
usually a bare language name, and "hindi"/"english" legitimately look like the
B.A. Hindi / B.A. English specialisations to the intent classifier — so counting
those turns invents a phantom top query and a phantom `courses` intent.
"""

from __future__ import annotations

import pytest
from app.api.analytics import _is_language_turn


@pytest.mark.parametrize(
    "stage",
    ["language_prompt", "language_reprompt", "language_confirm", "language_unsupported"],
)
def test_language_selection_turns_are_excluded(stage: str) -> None:
    assert _is_language_turn({"stage": stage}) is True


@pytest.mark.parametrize(
    "stage", ["menu", "conversation", "followup_offer", "followup_destination", "escalating"]
)
def test_conversation_turns_are_kept(stage: str) -> None:
    assert _is_language_turn({"stage": stage}) is False


def test_stage_match_is_case_insensitive() -> None:
    assert _is_language_turn({"stage": "LANGUAGE_PROMPT"}) is True


def test_legacy_rows_without_a_stage_are_kept() -> None:
    """Turns logged before the stage marker existed must not vanish from the
    dashboard — the filter may only exclude what it can prove is a language
    turn."""
    assert _is_language_turn({}) is False
    assert _is_language_turn(None) is False
    assert _is_language_turn("language_prompt") is False
    assert _is_language_turn({"stage": None}) is False
