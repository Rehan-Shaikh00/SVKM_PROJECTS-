"""Shared fixtures.

These tests are deliberately unit-level: no database, no network, no API keys.
They pin the behaviours that a live call depends on, especially the ones that
have broken before (see the comment on each regression test).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Keep tests deterministic and side-effect free: no provider credentials, no
# writes into the developer's real data/ directory. ENVIRONMENT must be one of
# the Literal values Settings accepts.
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("KB_AUTO_INGEST", "false")
os.environ.setdefault("CALL_RECORDING_ENABLED", "false")


@pytest.fixture
def tmp_index(tmp_path: pytest.TempPathFactory) -> Path:
    """A scratch directory for vector-store files."""
    return Path(str(tmp_path)) / "index"
