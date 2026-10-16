"""Per-run, per-agent state for skill hooks.

The SDK runs each hook in its own asyncio task, so a ContextVar set inside one
hook is invisible to the next. RunState is therefore stored on the run's
``Usage`` object, the one object every hook context of a run shares, with one
RunState per agent: a sub-agent used as a tool shares its parent's ``Usage``,
and a handoff continues the same run. Callers without a real run context
(direct hook calls, tests) fall back to a ContextVar.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agents.usage import Usage

if TYPE_CHECKING:
    from agents import Agent, RunContextWrapper

    from .skills import Skill


@dataclass
class RunState:
    """Mutable state for one agent within one run.

    All per-run state is consolidated here so there is one place to read and test.
    See the module docstring for how it is scoped.

    Attributes:
        manifest_injected: True once the available-skills manifest has been
            prepended for this agent in this run.  Prevents re-injection on
            subsequent turns.
        injected_this_call: Names of skills already injected during the current
            LLM call.  Shared across RunSkillHooks and SkillHooks via the mutable
            object reference so duplicate injection is prevented even when both
            hooks fire concurrently via asyncio.gather.  Cleared after each LLM
            call by on_llm_end so that skills re-inject on every subsequent turn.
        invoke_skill_calls: Running count of invoke_skill tool invocations by this
            agent in this run.  Used to enforce the max_calls_per_run guard.
        pending_skills: Skills queued by tool-result triggers (on_tool_end) or
            post-turn triggers (on_llm_end).  Drained and cleared at this
            agent's next on_llm_start; RunSkillHooks moves its own to the
            next agent on handoff.
        last_context: The RunContextWrapper from the most recent on_llm_start call.
            Stored so that deferred callbacks (_drain_pending) have access to
            runtime context without requiring it to be threaded through every
            intermediate function.
        last_agent: The Agent from the most recent on_llm_start call.
    """

    manifest_injected: bool = False
    injected_this_call: set[str] = field(default_factory=set)
    invoke_skill_calls: int = 0
    pending_skills: list[Skill] = field(default_factory=list)
    last_context: RunContextWrapper[Any] | None = None
    last_agent: Agent[Any] | None = None


_run_state: ContextVar[RunState | None] = ContextVar("_run_state", default=None)

_SLOT_ATTRIBUTE = "_openai_agents_skills_run_state"


class _RunStateSlot:
    """Holds a run's per-agent RunStates on its Usage.

    Copies start empty: the SDK deep-copies Usage for resumable checkpoints, and
    RunState references the agent and context, which must not be deep-copied.
    """

    __slots__ = ("states",)

    def __init__(self) -> None:
        # Keyed by id(agent); the agent is held so its id cannot be reused mid-run.
        self.states: dict[int, tuple[Any, RunState]] = {}

    def for_agent(self, agent: Any) -> RunState:
        entry = self.states.get(id(agent))
        if entry is None:
            entry = (agent, RunState())
            self.states[id(agent)] = entry
        return entry[1]

    def __deepcopy__(self, memo: dict[int, Any]) -> _RunStateSlot:
        return _RunStateSlot()


def _get_run_state(context: Any = None, agent: Any = None) -> RunState:
    """Return the RunState for *agent* in the current run, creating it if needed.

    Args:
        context: The hook's run context.  When it carries the run's ``Usage``,
            state is shared by every hook of that run regardless of which
            asyncio task the SDK runs the hook in.
        agent: The agent the hook fired for.  Each agent in a run gets its own
            state.

    Returns:
        The RunState for *agent* in the current run.
    """
    usage = getattr(context, "usage", None)
    if isinstance(usage, Usage):
        slot = getattr(usage, _SLOT_ATTRIBUTE, None)
        if not isinstance(slot, _RunStateSlot):
            slot = _RunStateSlot()
            object.__setattr__(usage, _SLOT_ATTRIBUTE, slot)
        return slot.for_agent(agent)

    state = _run_state.get()
    if state is None:
        state = RunState()
        _run_state.set(state)
    return state
