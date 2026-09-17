"""Call logging pipeline.

Regression cover for the worst bug found in this codebase: the enqueue wrappers
were declared ``async def`` but only put an item on a queue. Call sites did not
await them (by design — logging must never block a live call), so *every* event
was dropped: no call records, no transcripts, no escalations, no analytics.
"""

from __future__ import annotations

import inspect

from app.orchestrator.call_logger import QUEUE_MAX, CallLogger

WRAPPERS = (
    "start_call",
    "log_turn",
    "update_call",
    "end_call",
    "log_escalation",
    "log_followup",
    "log_unanswered",
    "log_provider_health",
)


def test_enqueue_wrappers_are_plain_functions() -> None:
    """If any of these is a coroutine again, un-awaited calls silently drop."""
    for name in WRAPPERS:
        method = getattr(CallLogger, name)
        assert not inspect.iscoroutinefunction(method), (
            f"{name} must not be async: call sites deliberately do not await it"
        )


def test_db_handlers_stay_async() -> None:
    for name in WRAPPERS:
        logger_ = CallLogger()
        kinds: list[str] = []
        logger_.submit = lambda kind, payload, _k=kinds: _k.append(kind)  # type: ignore[method-assign]
        getattr(logger_, name)({"call_id": "call-1"})
        assert len(kinds) == 1, f"{name} enqueued {kinds}"
        handler = getattr(logger_, f"_do_{kinds[0]}", None)
        assert handler is not None, f"no _do_{kinds[0]} handler for {name}"
        assert inspect.iscoroutinefunction(handler)


def test_submit_enqueues_without_blocking() -> None:
    logger_ = CallLogger()
    logger_.submit("start_call", {"call_id": "call-1"})
    logger_.submit("log_turn", {"call_id": "call-1", "seq": 1})
    assert logger_._queue.qsize() == 2
    assert logger_.dropped == 0


def test_full_queue_drops_events_instead_of_stalling_the_call() -> None:
    """A stalled logging queue must never apply back-pressure to the voice loop."""
    logger_ = CallLogger()
    for i in range(QUEUE_MAX + 5):
        logger_.submit("log_turn", {"call_id": "call-1", "seq": i})
    assert logger_._queue.qsize() == QUEUE_MAX
    assert logger_.dropped == 5


def test_stats_reports_the_drop_count() -> None:
    logger_ = CallLogger()
    logger_.dropped = 3
    logger_.written = 7
    stats = logger_.stats()
    assert stats["dropped"] == 3
    assert stats["written"] == 7
