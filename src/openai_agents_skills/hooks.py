"""SkillHooks and RunSkillHooks — inject skills into the agent loop.

The injection mechanism:
- The SDK passes a mutable ``input_items`` list to ``AgentHooks.on_llm_start`` before
  every model invocation, then passes the same list to ``model.get_response()``.
- Blocks prepended inside the hook are seen by the LLM on that call.
- The SDK fires both ``run_hooks.on_llm_start`` and ``agent.hooks.on_llm_start`` via
  ``asyncio.gather``.  The per-call double-injection guard in ``RunState.injected_this_call``
  prevents duplicate injection when both hook types are active simultaneously.

Usage::

    from agents import Agent, Runner
    from openai_agents_skills import Skill, SkillHooks, RunSkillHooks, SkillRegistry

    # Simple list-based usage (Phase 1 compatible)
    agent = Agent(
        name="Assistant",
        instructions="You are helpful.",
        hooks=SkillHooks([BulletSkill()]),
    )

    # Registry-based usage with routing (Phase 2)
    registry = SkillRegistry()
    registry.register(BulletSkill())
    registry.register(CitationSkill())

    result = await Runner.run(
        agent,
        input="...",
        hooks=RunSkillHooks(registry=registry),
    )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from agents import (
    Agent,
    AgentHookContext,
    AgentHooks,
    FunctionTool,
    RunContextWrapper,
    RunHooks,
    function_tool,
)

from ._state import RunState, _get_run_state
from .registry import SkillRegistry
from .skills import Skill

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_routing_context(input_items: list[Any], turns: int | None = 1) -> str:
    """Return the last *turns* user messages joined by ``' | '``, newest last.

    Args:
        input_items: The LLM input items list from ``on_llm_start``.
        turns: Number of recent user messages to include.  Pass ``None`` to
            include all user messages in the conversation (appropriate only for
            short sessions — the context string grows unboundedly and LRU caching
            will be less effective).

    Returns:
        A single routing context string, or ``""`` if no user messages are found.
    """
    user_texts = [
        item["content"]
        for item in input_items
        if isinstance(item, dict)
        and item.get("role") == "user"
        and isinstance(item.get("content"), str)
    ]
    recent = user_texts[-turns:] if turns is not None else user_texts
    return " | ".join(recent)


def _deduplicate(skills: list[Skill]) -> list[Skill]:
    """Deduplicate skills by name, keeping the **last** occurrence of each name.

    When called as ``_deduplicate([*direct_skills, *registry_skills])``, registry
    skills win on name conflicts because they appear last.

    Args:
        skills: Skills to deduplicate.

    Returns:
        Deduplicated list preserving original relative order of surviving items.
    """
    seen: set[str] = set()
    result: list[Skill] = []
    for skill in reversed(skills):
        if skill.name not in seen:
            seen.add(skill.name)
            result.append(skill)
    return list(reversed(result))


def _build_manifest(skills: list[Skill], max_skills: int | None = None) -> str:
    """Build a text manifest of available skills for the first-call system prompt.

    Args:
        skills: All registered skills to describe.
        max_skills: Optional cap.  When set, only the first *max_skills* entries
            are included.

    Returns:
        A markdown-formatted manifest block, or ``""`` if there are no skills.
    """
    entries = skills[:max_skills] if max_skills is not None else list(skills)
    if not entries:
        return ""
    lines = ["## Available Skills"]
    for skill in entries:
        line = f"- {skill.name}: {skill.description}"
        if skill.when_to_use:
            line += f"\n  When to use: {skill.when_to_use}"
        lines.append(line)
    return "\n".join(lines)


def _default_on_skill_error(skill: Skill, exc: Exception) -> None:
    """Default error handler — log a WARNING and continue."""
    _log.warning(
        "Skill %r raised during get_prompt_blocks(); skipping. Error: %s",
        skill.name,
        exc,
    )


async def _collect_blocks(
    skills: list[Skill],
    seen: set[str],
    on_error: Callable[[Skill, Exception], None],
) -> list[Any]:
    """Iterate *skills*, skip duplicates and disabled ones, collect prompt blocks.

    Adds each injected skill's name to *seen* before awaiting ``get_prompt_blocks``
    so that the double-injection guard is effective even when coroutines interleave
    at the await point.

    Args:
        skills: Deduplicated candidate skills.
        seen: The ``RunState.injected_this_call`` set shared across hook instances.
        on_error: Error handler called when ``get_prompt_blocks`` raises.

    Returns:
        Flat list of prompt blocks to prepend to ``input_items``.
    """
    all_blocks: list[Any] = []
    for skill in skills:
        if skill.name in seen:
            continue
        if not skill.is_enabled():
            continue
        seen.add(skill.name)
        try:
            blocks = await skill.get_prompt_blocks()
            all_blocks.extend(blocks)
        except Exception as exc:
            on_error(skill, exc)
    return all_blocks


def _resolve_candidates(
    direct_skills: list[Skill],
    registry: SkillRegistry | None,
    routed: list[Skill],
) -> list[Skill]:
    """Merge direct skills, always-on registry skills, and routed skills.

    Args:
        direct_skills: Skills passed directly to the hook constructor.
        registry: Optional registry to query for always-on skills.
        routed: Skills the router selected for the current message.

    Returns:
        Deduplicated candidate list.  Registry skills win over direct skills on
        name conflicts; routed skills win over everything on conflicts.
    """
    base: list[Skill]
    if registry is not None:
        always_on = registry.get_always_on()
        base = _deduplicate([*direct_skills, *always_on])
    else:
        base = list(direct_skills)
    return _deduplicate([*base, *routed])


# ---------------------------------------------------------------------------
# SkillHooks — per-agent hook
# ---------------------------------------------------------------------------


class SkillHooks(AgentHooks):
    """``AgentHooks`` implementation that injects skills before each LLM call.

    Supports both the simple Phase 1 interface (``skills`` list) and Phase 2
    registry-based routing.

    Args:
        skills: Skill instances to inject unconditionally.  When *registry* is
            also provided, these are merged with ``registry.get_always_on()``
            and registry skills win on name conflicts.
        registry: Optional :class:`~openai_agents_skills.registry.SkillRegistry`.
            Enables manifest injection and dynamic per-message routing.
        routing_context_turns: Number of recent user messages to concatenate into
            the routing context string.  ``1`` (default) routes on the latest
            message only.  Pass ``None`` to include all user messages.
        on_skill_error: Callback invoked when ``get_prompt_blocks()`` raises.
            Defaults to logging at ``WARNING`` and continuing.  Pass
            ``lambda s, e: None`` to silence all errors, or a re-raise function
            for tests.
        max_manifest_skills: Optional cap on skills listed in the first-call
            manifest.  Default: unlimited.

    Example::

        # Phase 1 style — direct skill list
        agent = Agent(hooks=SkillHooks([CitationSkill(), BulletSkill()]))

        # Phase 2 style — registry with routing
        registry = SkillRegistry(router=LLMSkillRouter(client=AsyncOpenAI()))
        registry.register(CitationSkill())
        agent = Agent(hooks=SkillHooks(registry=registry))
    """

    def __init__(
        self,
        skills: list[Skill] | None = None,
        registry: SkillRegistry | None = None,
        routing_context_turns: int | None = 1,
        on_skill_error: Callable[[Skill, Exception], None] | None = None,
        max_manifest_skills: int | None = None,
    ) -> None:
        self._skills: list[Skill] = list(skills or [])
        self._registry = registry
        self._routing_context_turns = routing_context_turns
        self._on_error: Callable[[Skill, Exception], None] = (
            on_skill_error if on_skill_error is not None else _default_on_skill_error
        )
        self._max_manifest_skills = max_manifest_skills

    async def on_start(
        self,
        context: AgentHookContext[Any],
        agent: Agent[Any],
    ) -> None:
        """Initialise RunState in the parent task before any on_llm_start gather.

        Called by the SDK in the agent's main task before the first LLM call.
        Eagerly calling ``_get_run_state()`` here ensures the RunState is stored
        in the parent context so that child tasks created by ``asyncio.gather``
        inherit a reference to the same object — making the double-injection guard
        effective.
        """
        _get_run_state()

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        """Prepend skill prompt blocks to ``input_items`` before the LLM is called.

        On the first call of the run, also prepends a manifest block listing all
        registered skills if a registry is configured.

        Args:
            context: The run context wrapper provided by the SDK.
            agent: The agent currently executing.
            system_prompt: The agent's system prompt (read-only).
            input_items: Mutable list of input items.  Blocks prepended here are
                seen by the model.
        """
        state = _get_run_state()

        manifest_blocks: list[Any] = _maybe_build_manifest_blocks(
            state, self._registry, self._max_manifest_skills
        )

        routed: list[Skill] = []
        if self._registry is not None:
            routing_ctx = _extract_routing_context(input_items, turns=self._routing_context_turns)
            if routing_ctx:
                routed = await self._registry.select_for_message(routing_ctx)

        candidates = _resolve_candidates(self._skills, self._registry, routed)
        skill_blocks = await _collect_blocks(candidates, state.injected_this_call, self._on_error)

        input_items[0:0] = manifest_blocks + skill_blocks


# ---------------------------------------------------------------------------
# RunSkillHooks — run-level hook spanning all agents
# ---------------------------------------------------------------------------


class RunSkillHooks(RunHooks):
    """``RunHooks`` variant that applies skills across every agent in a run.

    Attach to ``Runner.run(..., hooks=RunSkillHooks(...))`` to enable skills without
    configuring each ``Agent`` individually.  Uses the same per-run double-injection
    guard as :class:`SkillHooks` — a skill registered in both ``RunSkillHooks`` and
    a per-agent ``SkillHooks`` injects only once per LLM call.

    Args:
        skills: Skill instances to inject unconditionally.
        registry: Optional :class:`~openai_agents_skills.registry.SkillRegistry`.
        routing_context_turns: Number of recent user messages for routing context.
        on_skill_error: Callback invoked when ``get_prompt_blocks()`` raises.
        max_manifest_skills: Optional cap on the first-call manifest size.

    Example::

        result = await Runner.run(
            triage_agent,
            input="...",
            hooks=RunSkillHooks(registry=registry),
        )
    """

    def __init__(
        self,
        skills: list[Skill] | None = None,
        registry: SkillRegistry | None = None,
        routing_context_turns: int | None = 1,
        on_skill_error: Callable[[Skill, Exception], None] | None = None,
        max_manifest_skills: int | None = None,
    ) -> None:
        self._skills: list[Skill] = list(skills or [])
        self._registry = registry
        self._routing_context_turns = routing_context_turns
        self._on_error: Callable[[Skill, Exception], None] = (
            on_skill_error if on_skill_error is not None else _default_on_skill_error
        )
        self._max_manifest_skills = max_manifest_skills

    async def on_agent_start(
        self,
        context: AgentHookContext[Any],
        agent: Agent[Any],
    ) -> None:
        """Initialise RunState in the parent task before any on_llm_start gather.

        Called by the SDK in the run's main task before the first LLM call.
        Eagerly calling ``_get_run_state()`` here ensures the double-injection
        guard works correctly when both RunSkillHooks and a per-agent SkillHooks
        are active simultaneously.
        """
        _get_run_state()

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        """Prepend skill prompt blocks for all agents in the run."""
        state = _get_run_state()

        manifest_blocks: list[Any] = _maybe_build_manifest_blocks(
            state, self._registry, self._max_manifest_skills
        )

        routed: list[Skill] = []
        if self._registry is not None:
            routing_ctx = _extract_routing_context(input_items, turns=self._routing_context_turns)
            if routing_ctx:
                routed = await self._registry.select_for_message(routing_ctx)

        candidates = _resolve_candidates(self._skills, self._registry, routed)
        skill_blocks = await _collect_blocks(candidates, state.injected_this_call, self._on_error)

        input_items[0:0] = manifest_blocks + skill_blocks


# ---------------------------------------------------------------------------
# make_invoke_skill_tool
# ---------------------------------------------------------------------------


def make_invoke_skill_tool(
    registry: SkillRegistry,
    max_calls_per_run: int = 10,
) -> FunctionTool:
    """Create a ``FunctionTool`` that lets the model invoke skills by name.

    The model can call this tool to retrieve the prompt content of any registered
    skill.  Add it to ``agent.tools`` alongside a :class:`SkillHooks` so the model
    can explicitly request skill content beyond what automatic routing provides.

    A runaway-invocation guard prevents the model from calling the tool more than
    *max_calls_per_run* times in a single run.  Set ``max_calls_per_run=0`` to
    disable the guard.

    Args:
        registry: The :class:`~openai_agents_skills.registry.SkillRegistry` to look
            up skills in.
        max_calls_per_run: Maximum invocations per run.  Default: 10.

    Returns:
        A :class:`~agents.FunctionTool` ready to add to an agent.

    Example::

        from openai_agents_skills import make_invoke_skill_tool

        agent = Agent(
            name="Assistant",
            tools=[make_invoke_skill_tool(registry)],
            hooks=SkillHooks(registry=registry),
        )
    """

    @function_tool(strict_mode=False)
    async def invoke_skill(skill_name: str, args: str = "") -> str:
        """Invoke a skill by name and return its prompt content as text.

        Args:
            skill_name: Name of the skill to invoke.
            args: Optional arguments passed through to the skill.
        """
        state = _get_run_state()
        if max_calls_per_run > 0 and state.invoke_skill_calls >= max_calls_per_run:
            return "invoke_skill limit reached for this run"
        state.invoke_skill_calls += 1
        try:
            skill = registry.get(skill_name)
        except KeyError:
            available = ", ".join(registry.skill_names) or "(none)"
            return f"Unknown skill: {skill_name!r}. Available: {available}"
        blocks = await skill.get_prompt_blocks(args)
        return "\n".join(
            block.get("content", "") if isinstance(block, dict) else str(block) for block in blocks
        )

    return invoke_skill


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _maybe_build_manifest_blocks(
    state: RunState,
    registry: SkillRegistry | None,
    max_manifest_skills: int | None,
) -> list[Any]:
    """Build and return the manifest block list if this is the first LLM call.

    Mutates ``state.manifest_injected`` to prevent re-injection on subsequent
    turns.

    Args:
        state: The current run's ``RunState``.
        registry: The registry to build the manifest from.  Returns ``[]`` if
            ``None``.
        max_manifest_skills: Optional cap on the number of skills listed.

    Returns:
        A one-element list containing the manifest user message dict, or ``[]``.
    """
    if state.manifest_injected or registry is None:
        return []
    state.manifest_injected = True
    all_skills = list(registry._skills.values())
    manifest_text = _build_manifest(all_skills, max_manifest_skills)
    if not manifest_text:
        return []
    return [{"role": "user", "content": manifest_text}]
