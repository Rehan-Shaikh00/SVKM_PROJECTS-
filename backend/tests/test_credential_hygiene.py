"""Credentials that are printed in the repository must not reach a caller.

`APP_SECRET` is the key that caller phone numbers — and any Aadhaar or card
number dictated mid-call — are pseudonymised with, and `ADMIN_PASSWORD` guards
the panel that rewrites what the assistant says aloud. Both shipped with
placeholder defaults, in a public repository, and nothing objected: the
dashboard logged in, calls were answered, transcripts looked right, and the
pseudonyms in the database looked hashed. Every failure here is silent in use,
so it is caught at boot or never.

These build `Settings` explicitly rather than reading `.env`, so they say the
same thing on a laptop, in CI and on a deployment host.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from app.config import (
    INSECURE_ADMIN_PASSWORDS,
    INSECURE_APP_SECRETS,
    MIN_ADMIN_PASSWORD_LENGTH,
    MIN_APP_SECRET_LENGTH,
    Settings,
)
from app.main import _enforce_credential_hygiene

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPO_ROOT / ".env.example"

STRONG_SECRET = "k3Vn9sQ1xZ7bR4mT8wY2cL6pD0fH5jA9nE3uG7iO1s"
STRONG_PASSWORD = "correct-horse-battery"


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "production",
        "app_secret": STRONG_SECRET,
        "admin_password": STRONG_PASSWORD,
        "admin_auth_enabled": True,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _example_value(key: str) -> str:
    match = re.search(rf"^{key}=(.*)$", ENV_EXAMPLE.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, f"{key} is not documented in .env.example"
    return match.group(1).strip()


# --------------------------------------------------------------------------- #
# what counts as unsafe
# --------------------------------------------------------------------------- #
def test_the_published_placeholders_are_rejected() -> None:
    problems = _settings(
        app_secret="change-me-in-production", admin_password="change-me"
    ).credential_problems()
    assert any("APP_SECRET" in p for p in problems)
    assert any("ADMIN_PASSWORD" in p for p in problems)


def test_an_open_admin_api_is_rejected() -> None:
    """The admin API rewrites the knowledge base, so 'no auth' is not a
    configuration anybody should reach production with -- whatever the password
    happens to be."""
    problems = _settings(admin_auth_enabled=False).credential_problems()
    assert any("ADMIN_AUTH_ENABLED" in p for p in problems)


def test_short_credentials_are_rejected_without_being_the_placeholder() -> None:
    """Rotating to something merely different is not rotating to something
    strong; a short key is enumerable whatever it says."""
    problems = _settings(app_secret="abc123", admin_password="hunter2").credential_problems()
    assert any(f"at least {MIN_APP_SECRET_LENGTH}" in p for p in problems)
    assert any(f"at least {MIN_ADMIN_PASSWORD_LENGTH}" in p for p in problems)


def test_credentials_that_were_actually_generated_pass() -> None:
    assert _settings().credential_problems() == []


def test_staging_is_held_to_the_same_standard_as_production() -> None:
    """Staging usually points at a real telephone number and stores real caller
    transcripts, so it is not a place to run a public password either."""
    problems = _settings(environment="staging", admin_password="change-me").credential_problems()
    assert any("ADMIN_PASSWORD" in p for p in problems)


def test_the_rejection_says_what_the_secret_protects() -> None:
    """The reason matters: 'insecure secret' gets ignored, 'every caller phone
    number in the database can be recomputed from a public constant' does not."""
    problems = _settings(app_secret="change-me-in-production").credential_problems()
    joined = " ".join(problems)
    assert "phone number" in joined
    assert "repository" in joined


# --------------------------------------------------------------------------- #
# what happens at boot
# --------------------------------------------------------------------------- #
def test_production_refuses_to_start_and_says_how_to_fix_it(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(
        main.settings, "environment", "production", raising=False
    )
    monkeypatch.setattr(main.settings, "app_secret", "change-me-in-production", raising=False)
    monkeypatch.setattr(main.settings, "admin_password", "change-me", raising=False)
    monkeypatch.setattr(main.settings, "admin_auth_enabled", True, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        _enforce_credential_hygiene()
    message = str(excinfo.value)
    assert "refusing to start in production" in message
    assert "APP_SECRET" in message and "ADMIN_PASSWORD" in message
    # the fix has to be in the message, not just the complaint
    assert "secrets.token_urlsafe" in message
    assert ".env" in message


def test_an_open_admin_api_stops_a_production_boot(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(main.settings, "environment", "production", raising=False)
    monkeypatch.setattr(main.settings, "app_secret", STRONG_SECRET, raising=False)
    monkeypatch.setattr(main.settings, "admin_password", STRONG_PASSWORD, raising=False)
    monkeypatch.setattr(main.settings, "admin_auth_enabled", False, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        _enforce_credential_hygiene()
    assert "ADMIN_AUTH_ENABLED" in str(excinfo.value)


def test_development_boots_but_does_not_keep_the_public_key(monkeypatch) -> None:
    """Copying `.env.example` and running the app has to keep working -- that is
    the documented 60-second start -- but the caller pseudonyms must not stay
    keyed by a constant the whole world can read."""
    from app import main

    monkeypatch.setattr(main.settings, "environment", "development", raising=False)
    monkeypatch.setattr(main.settings, "app_secret", "change-me-in-production", raising=False)
    monkeypatch.setattr(main.settings, "admin_password", "change-me", raising=False)
    monkeypatch.setattr(main.settings, "admin_auth_enabled", False, raising=False)

    _enforce_credential_hygiene()  # must not raise

    replacement = main.settings.app_secret
    assert replacement not in INSECURE_APP_SECRETS
    assert len(replacement) >= MIN_APP_SECRET_LENGTH


def test_development_leaves_a_real_secret_alone(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(main.settings, "environment", "development", raising=False)
    monkeypatch.setattr(main.settings, "app_secret", STRONG_SECRET, raising=False)
    monkeypatch.setattr(main.settings, "admin_password", STRONG_PASSWORD, raising=False)
    monkeypatch.setattr(main.settings, "admin_auth_enabled", True, raising=False)

    _enforce_credential_hygiene()
    assert main.settings.app_secret == STRONG_SECRET, "a generated secret must survive"


# --------------------------------------------------------------------------- #
# the example file itself
# --------------------------------------------------------------------------- #
def test_every_credential_the_example_ships_is_on_the_reject_list() -> None:
    """Keeps `.env.example` and the guard from drifting apart.

    If the placeholder in the example is ever changed, the guard has to learn
    the new one -- otherwise a fresh deployment copies a value that passes.
    """
    assert _example_value("APP_SECRET") in INSECURE_APP_SECRETS
    assert _example_value("ADMIN_PASSWORD") in INSECURE_ADMIN_PASSWORDS


def test_the_example_says_what_app_secret_is_for() -> None:
    """It was documented as signing webhook callbacks, which it never did --
    Twilio signature validation uses TWILIO_AUTH_TOKEN. A secret described as
    something it is not does not get rotated."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    app_secret_block = text.split("APP_SECRET=")[0].rsplit("# ----", 1)[-1]
    assert "pseudonymis" in app_secret_block, "must say what the key protects"
    assert "TWILIO_AUTH_TOKEN" in app_secret_block, "must say what it is not for"
    assert "refuses to start" in app_secret_block, "must warn that booting fails"
