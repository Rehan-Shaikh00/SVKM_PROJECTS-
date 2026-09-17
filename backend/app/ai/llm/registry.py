"""LLM registry with graceful degradation to the local grounded engine."""

from __future__ import annotations

import logging
from typing import Any

from ...config import settings
from .base import LLM

logger = logging.getLogger("nims.llm")


def build_llm(provider: str | None = None) -> LLM | None:
    provider = (provider or settings.llm_provider or "local").lower()
    if provider == "anthropic":
        from .anthropic import AnthropicLLM

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing")
        return AnthropicLLM()
    if provider == "openai":
        from .openai_compat import OpenAICompatibleLLM

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        return OpenAICompatibleLLM(base_url=settings.openai_base_url)
    return None  # "local" — handled by TemplateAnswerEngine, no LLM client


def resolve_llm(requested: str | None = None) -> tuple[LLM | None, dict[str, Any]]:
    requested = (requested or settings.llm_provider or "local").lower()
    attempts: list[str] = []
    order = [requested] + [p for p in ("anthropic", "openai") if p != requested]
    if requested == "local":
        return None, {
            "requested": "local", "active": "local", "degraded": False,
            "attempts": attempts, "tools": False,
        }
    for candidate in order:
        if candidate == "local":
            continue
        try:
            llm = build_llm(candidate)
            if llm is None:
                continue
            return llm, {
                "requested": requested,
                "active": candidate,
                "degraded": candidate != requested,
                "attempts": attempts,
                "tools": llm.supports_tools,
                "model": llm.model,
            }
        except Exception as exc:
            attempts.append(f"{candidate}:{exc}")
            logger.warning("LLM provider %s unavailable: %s", candidate, exc)
    logger.warning("no cloud LLM available — using the local grounded template engine")
    return None, {
        "requested": requested, "active": "local", "degraded": True,
        "attempts": attempts, "tools": False,
    }
