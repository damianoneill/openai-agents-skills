"""SkillRegistry — central store and optional routing layer for skills."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .skills import Skill

if TYPE_CHECKING:
    from agents import Agent, RunContextWrapper

    from .router import SkillRouter


class SkillRegistry:
    """Central store for skills with optional LLM-based per-message routing.

    Register skills once; SkillHooks or RunSkillHooks query the registry on every
    LLM call to decide which skills to inject.

    When *router* is ``None`` (the default), :meth:`select_for_message` always
    returns ``[]`` and every active skill comes from :meth:`get_always_on`.  Pass
    a :class:`~openai_agents_skills.router.SkillRouter` implementation to enable
    dynamic per-message selection.

    Args:
        router: Optional routing strategy.  Pass ``None`` (default) to disable
            routing — all skills are treated as unconditional (always-on).

    Example::

        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        from openai import AsyncOpenAI
        from openai_agents_skills import LLMSkillRouter, SkillRegistry, SkillHooks

        model = OpenAIChatCompletionsModel("gpt-4o-mini", AsyncOpenAI())
        router = LLMSkillRouter(model=model)
        registry = SkillRegistry(router=router)
        registry.register(MySkill())

        hooks = SkillHooks(registry=registry)
    """

    def __init__(self, router: SkillRouter | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        self._router: SkillRouter | None = router

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, skill: Skill) -> None:
        """Register a skill.  Overwrites any existing skill with the same name.

        Args:
            skill: The skill instance to register.

        Raises:
            ValueError: If the skill's ``name`` attribute is empty or whitespace-only.
                Every registered skill must have a non-empty name because the registry
                uses ``name`` as its primary key.
        """
        if not skill.name or not skill.name.strip():
            raise ValueError(
                f"{type(skill).__name__!r} has an empty or whitespace-only name. "
                "Set the 'name' class attribute to a non-empty string before registering."
            )
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        """Remove a skill by name.  Silently ignored if the name is not registered.

        Args:
            name: The skill name to remove.
        """
        self._skills.pop(name, None)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str) -> Skill:
        """Return the skill with the given name.

        Args:
            name: The skill name to look up.

        Returns:
            The registered :class:`~openai_agents_skills.skills.Skill` instance.

        Raises:
            KeyError: If no skill with that name is registered.
        """
        return self._skills[name]

    @property
    def skill_names(self) -> list[str]:
        """Sorted list of all registered skill names."""
        return sorted(self._skills.keys())

    @property
    def all_skills(self) -> list[Skill]:
        """All registered skills in registration order."""
        return list(self._skills.values())

    def get_always_on(
        self,
        context: RunContextWrapper[Any] | None = None,
        agent: Agent[Any] | None = None,
    ) -> list[Skill]:
        """Return skills that should always be injected regardless of the message.

        A skill is *always-on* when its ``when_to_use`` attribute is empty **and**
        ``is_enabled()`` returns ``True``.  Skills with a non-empty ``when_to_use``
        are excluded — they are only injected when the router selects them.

        Args:
            context: The current run context, or ``None`` if not available.
            agent: The current agent, or ``None`` if not available.

        Returns:
            List of enabled, unconditional skills in registration order.
        """
        return [
            s for s in self._skills.values() if not s.when_to_use and s.is_enabled(context, agent)
        ]

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def select_for_message(
        self,
        message: str,
        context: RunContextWrapper[Any] | None = None,
        agent: Agent[Any] | None = None,
    ) -> list[Skill]:
        """Return skills selected by the router for this message.

        Only skills with a non-empty ``when_to_use`` are passed to the router.
        Returns ``[]`` when no router is configured (all active skills then come
        from :meth:`get_always_on`).

        Args:
            message: The routing context string — typically the last user message
                or a multi-turn summary produced by
                :func:`~openai_agents_skills.hooks._extract_routing_context`.
            context: The current run context, or ``None`` if not available.
            agent: The current agent, or ``None`` if not available.

        Returns:
            List of skills the router selected, in the order the router returned
            their names.  Names not found in the registry are silently skipped.
        """
        if self._router is None:
            return []
        routable = [
            s for s in self._skills.values() if s.when_to_use and s.is_enabled(context, agent)
        ]
        if not routable:
            return []
        names = await self._router.select(message, routable)
        return [self._skills[n] for n in names if n in self._skills]

    def get_triggered_by_tool(
        self,
        tool_name: str,
        context: RunContextWrapper[Any] | None = None,
        agent: Agent[Any] | None = None,
    ) -> list[Skill]:
        """Return enabled skills whose ``triggers_after_tools`` includes *tool_name*.

        Called by ``on_tool_end`` to determine which skills to queue in
        ``RunState.pending_skills`` after the named tool completes.

        Args:
            tool_name: The name of the tool that just finished executing.
            context: The current run context, or ``None`` if not available.
            agent: The current agent, or ``None`` if not available.

        Returns:
            List of enabled skills triggered by *tool_name*, in registration order.
            Returns an empty list when no skills declare this tool as a trigger.
        """
        return [
            s
            for s in self._skills.values()
            if tool_name in s.triggers_after_tools and s.is_enabled(context, agent)
        ]

    def get_post_turn(
        self,
        context: RunContextWrapper[Any] | None = None,
        agent: Agent[Any] | None = None,
    ) -> list[Skill]:
        """Return enabled skills that fire after every model response.

        Called by ``on_llm_end`` to queue skills with ``triggers_after_turn=True``
        in ``RunState.pending_skills`` so they inject at the start of the next turn.

        Args:
            context: The current run context, or ``None`` if not available.
            agent: The current agent, or ``None`` if not available.

        Returns:
            List of enabled skills with ``triggers_after_turn=True``, in
            registration order.
        """
        return [
            s
            for s in self._skills.values()
            if s.triggers_after_turn and s.is_enabled(context, agent)
        ]
