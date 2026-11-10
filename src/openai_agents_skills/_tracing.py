"""Tracing spans recorded for skill routing and triggered skills.

Users query these span names and data keys in their trace exporters, so the
README lists them. Update it when either changes.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

_ROUTING_SPAN_NAME = "skill_routing"
_TRIGGERED_SPAN_NAME = "skill_triggered"


def _custom_span(name: str) -> AbstractContextManager[Any]:
    """Return an SDK ``custom_span`` called *name* in the host application's trace.

    When no trace is active — as in unit tests, or when the host has not
    configured SDK tracing — a null context is returned instead. This avoids the
    SDK's "No active trace" error log and adds zero overhead, so spans are only
    emitted when the caller has already opted into tracing.
    """
    from agents.tracing import custom_span, get_current_trace

    if get_current_trace() is None:
        return nullcontext(None)
    return custom_span(name)


def _record_routing(
    span: Any | None,
    *,
    candidates: list[str],
    selected: list[str],
    cache_hit: bool,
    always_on: list[str] | None = None,
    error: Exception | None = None,
) -> None:
    """Attach routing-decision attributes to *span*.

    No-op when *span* is ``None`` (no active trace). Records the candidate skills
    considered, those selected and rejected, the always-on skills injected
    unconditionally (so the trace reflects every skill active this turn, not just
    the routed ones), whether the result came from the LRU cache, and — on
    failure — the error so silent fallbacks remain queryable.

    Routing is otherwise invisible: the internal model call in
    :class:`~openai_agents_skills.router.LLMSkillRouter` runs with tracing
    disabled, and cache hits make no model call at all, so without this span a
    silent router failure is indistinguishable from "correctly selected nothing".
    """
    if span is None:
        return
    data = span.span_data.data
    data["candidates"] = candidates
    data["candidate_count"] = len(candidates)
    data["selected"] = selected
    data["selected_count"] = len(selected)
    data["rejected"] = [name for name in candidates if name not in selected]
    data["cache_hit"] = cache_hit
    if always_on is not None:
        data["always_on"] = always_on
        data["always_on_count"] = len(always_on)
    if error is not None:
        data["error"] = str(error)
        data["error_type"] = type(error).__name__


def _record_triggered(
    span: Any | None,
    *,
    triggered: list[str],
    collected: list[str],
    already_claimed: list[str],
    disabled: list[str],
    failed: list[str],
    error: Exception | None = None,
) -> None:
    """Attach the outcome of a drain to *span*.  No-op when *span* is ``None``."""
    if span is None:
        return
    data = span.span_data.data
    data["triggered"] = triggered
    data["triggered_count"] = len(triggered)
    # Loaded, not proof of injection: blocks may be empty or the LLM call may abort.
    data["collected"] = collected
    data["collected_count"] = len(collected)
    # Claimed by another path this call. Its load may still be running, or fail.
    data["already_claimed"] = already_claimed
    data["disabled"] = disabled
    data["failed"] = failed
    if error is not None:
        data["error"] = str(error)
        data["error_type"] = type(error).__name__
