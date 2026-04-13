"""Per-run state for skill hooks, scoped to the current asyncio task via ContextVar.

RunState is a mutable dataclass held in a ContextVar. Each Runner.run() call runs
in its own asyncio task; when asyncio.gather creates child tasks for on_llm_start
calls, those tasks inherit a copy of the parent context that points to the SAME
RunState object (passed by reference). This allows the double-injection guard
(injected_this_call) to work correctly across both RunSkillHooks and SkillHooks
firing on the same LLM call, provided that _get_run_state() is called once in the
parent task before the gather (e.g. from on_start / on_agent_start).
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

from .skills import Skill


@dataclass
class RunState:
    """Mutable per-run state for skill hooks.

    All per-run state is consolidated here so there is one place to read and test.
    The instance is stored in a ContextVar so it is automatically scoped to the
    current asyncio task and concurrent runs cannot interfere with each other.

    Attributes:
        manifest_injected: True once the available-skills manifest has been
            prepended for this run. Prevents re-injection on subsequent turns.
        injected_this_call: Names of skills already injected for the current
            LLM call. Shared across RunSkillHooks and SkillHooks via the mutable
            object reference so duplicate injection is prevented even when both
            hooks fire concurrently via asyncio.gather.
        pending_skills: Skills queued for injection on the next LLM call (reserved
            for Phase 4 tool-result triggers).
        invoke_skill_calls: Running count of invoke_skill tool invocations this run.
            Used to enforce the max_calls_per_run guard.
    """

    manifest_injected: bool = False
    injected_this_call: set[str] = field(default_factory=set)
    pending_skills: list[Skill] = field(default_factory=list)
    invoke_skill_calls: int = 0


_run_state: ContextVar[RunState | None] = ContextVar("_run_state", default=None)


def _get_run_state() -> RunState:
    """Return the current run's RunState, creating one lazily if needed.

    Call this once from the parent asyncio task (e.g. in on_start / on_agent_start)
    before any asyncio.gather that fires on_llm_start. Child tasks created by the
    gather inherit a context copy that references the same RunState object, so
    mutations (e.g. adding to injected_this_call) are visible across both hook
    instances for the same LLM call.

    Returns:
        The RunState for the current async context.
    """
    state = _run_state.get()
    if state is None:
        state = RunState()
        _run_state.set(state)
    return state
