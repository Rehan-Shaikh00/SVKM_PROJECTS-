"""Structured logging + lightweight in-process metrics.

Voice quality problems are latency problems, so every turn emits a timing record
(`nims.turn`) with the ASR / retrieval / LLM / TTS breakdown. Those land in the
`call_turns` table and feed the dashboard's latency panel.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"


class JsonFormatter(logging.Formatter):
    """JSON lines in production (log aggregators), human readable in dev."""

    def __init__(self, as_json: bool) -> None:
        super().__init__(LOG_FORMAT)
        self.as_json = as_json

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - IO
        if not self.as_json:
            return super().format(record)
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("call_id", "provider", "state", "language", "latency_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", as_json: bool | None = None) -> None:
    from ..config import settings

    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    use_json = as_json if as_json is not None else settings.is_production
    handler.setFormatter(JsonFormatter(use_json))
    root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "websockets.client", "anthropic._base_client",
                  "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@dataclass
class Metrics:
    """Rolling in-process counters. Cheap, no external dependency.

    For a multi-instance deployment export these to Prometheus/OTel; the shape
    here maps 1:1 onto counters and histograms.
    """

    calls_started: int = 0
    calls_completed: int = 0
    calls_escalated: int = 0
    calls_failed: int = 0
    turns: int = 0
    grounded_turns: int = 0
    barge_ins: int = 0
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    ttfa: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    asr_latency: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    llm_latency: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    tts_latency: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    retrieval_latency: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    languages: Counter[str] = field(default_factory=Counter)
    intents: Counter[str] = field(default_factory=Counter)
    escalation_reasons: Counter[str] = field(default_factory=Counter)
    provider_errors: Counter[str] = field(default_factory=Counter)
    started_at: float = field(default_factory=time.time)

    def record_turn(
        self,
        *,
        grounded: bool,
        latency_ms: float,
        first_audio_ms: float = 0.0,
        asr_ms: float = 0.0,
        llm_ms: float = 0.0,
        tts_ms: float = 0.0,
        retrieval_ms: float = 0.0,
        intent: str = "other",
    ) -> None:
        self.turns += 1
        if grounded:
            self.grounded_turns += 1
        self.latencies.append(latency_ms)
        if first_audio_ms:
            self.ttfa.append(first_audio_ms)
        if asr_ms:
            self.asr_latency.append(asr_ms)
        if llm_ms:
            self.llm_latency.append(llm_ms)
        if tts_ms:
            self.tts_latency.append(tts_ms)
        if retrieval_ms:
            self.retrieval_latency.append(retrieval_ms)
        self.intents[intent] += 1

    def snapshot(self) -> dict[str, Any]:
        def stats(values: deque[float]) -> dict[str, float]:
            if not values:
                return {"count": 0, "p50": 0.0, "p95": 0.0, "avg": 0.0}
            ordered = sorted(values)
            return {
                "count": len(ordered),
                "p50": round(ordered[len(ordered) // 2], 1),
                "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
                "avg": round(sum(ordered) / len(ordered), 1),
            }

        uptime = time.time() - self.started_at
        return {
            "uptime_seconds": round(uptime, 1),
            "calls": {
                "started": self.calls_started,
                "completed": self.calls_completed,
                "escalated": self.calls_escalated,
                "failed": self.calls_failed,
                "escalation_rate": round(
                    self.calls_escalated / max(1, self.calls_started), 3
                ),
            },
            "turns": {
                "total": self.turns,
                "grounded": self.grounded_turns,
                "grounded_rate": round(self.grounded_turns / max(1, self.turns), 3),
                "barge_ins": self.barge_ins,
            },
            "latency_ms": {
                "end_to_end": stats(self.latencies),
                "time_to_first_audio": stats(self.ttfa),
                "asr": stats(self.asr_latency),
                "retrieval": stats(self.retrieval_latency),
                "llm": stats(self.llm_latency),
                "tts": stats(self.tts_latency),
            },
            "languages": dict(self.languages),
            "intents": dict(self.intents.most_common(20)),
            "escalation_reasons": dict(self.escalation_reasons),
            "provider_errors": dict(self.provider_errors),
        }


metrics = Metrics()
