"""Skills for the OpenAI Agents SDK.

A Skill is a named, reusable prompt fragment injected into the LLM's context
at the right moment in the agent loop via AgentHooks.on_llm_start.

The SDK passes a mutable input_items list to AgentHooks.on_llm_start before
every model invocation. SkillHooks prepends each skill's prompt blocks to that
list so the model receives the skill's instructions on every call.

Example::

    from openai_agents_skills import Skill, SkillHooks
    from agents import Agent, Runner

    class CitationSkill(Skill):
        name = "citation"
        description = "Always cite sources when making factual claims."

        async def get_prompt_blocks(self, context, agent, args=""):
            return [{"role": "user", "content": "Always cite your sources."}]

    agent = Agent(
        name="Assistant",
        instructions="You are helpful.",
        hooks=SkillHooks([CitationSkill()]),
    )

    result = await Runner.run(agent, "What is the speed of light?")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents import Agent, RunContextWrapper


class Skill(ABC):
    """Abstract base class for skills.

    Subclass and override :meth:`get_prompt_blocks` to provide skill content.
    The returned blocks are prepended to the LLM's ``input_items`` list before
    each model invocation by :class:`~openai_agents_skills.hooks.SkillHooks`.

    Both :meth:`get_prompt_blocks` and :meth:`is_enabled` receive the current
    ``RunContextWrapper`` and ``Agent`` so that skills can inject dynamic content
    (organisation IDs, user preferences, feature flags) and gate themselves on
    runtime state without requiring separate configuration.

    Class attributes:

    Attributes:
        name: Unique identifier for the skill.
        description: Short human-readable summary. Shown in the manifest and
            used by routing logic.
        when_to_use: Prose description of when the skill should be triggered,
            including example phrases. Used by routing.
        allowed_tools: Tools this skill may invoke. Defaults to an empty list
            (no restriction). Surfaced in the skill manifest in Phase 3.
            Instances that need their own list must set ``self.allowed_tools``
            in ``__init__`` to avoid sharing the class-level default.
        user_invocable: Whether this skill appears in the manifest shown to
            the model. Set to ``False`` to hide internal or helper skills
            from the manifest while still allowing them to be injected.
            Defaults to ``True``.
        triggers_after_tools: Tool names that should queue this skill for injection
            on the next LLM call when the tool completes.  Empty list (default)
            means no tool-result triggers.  Registered via
            :class:`~openai_agents_skills.registry.SkillRegistry`; queued by
            ``on_tool_end`` in :class:`~openai_agents_skills.hooks.SkillHooks` /
            :class:`~openai_agents_skills.hooks.RunSkillHooks`.
        triggers_after_turn: When ``True``, this skill is queued in
            ``pending_skills`` after every model response (``on_llm_end``) and
            injected at the start of the next turn.  Useful for quality checks,
            memory consolidation, or review workflows.  Defaults to ``False``.

    Example::

        class ReplyInBulletsSkill(Skill):
            name = "reply_in_bullets"
            description = "Instructs the agent to respond using bullet points."

            async def get_prompt_blocks(self, context, agent, args=""):
                return [{"role": "user", "content": "Always respond using bullet points."}]
    """

    name: str = ""
    description: str = ""
    when_to_use: str = ""
    allowed_tools: list[str] = []
    user_invocable: bool = True
    triggers_after_tools: list[str] = []
    triggers_after_turn: bool = False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

    def is_enabled(
        self,
        context: RunContextWrapper[Any] | None = None,
        agent: Agent[Any] | None = None,
    ) -> bool:
        """Return True if this skill should be injected on the current call.

        Override to gate a skill dynamically on feature flags, runtime context,
        or any other condition. Disabled skills are silently skipped by
        :class:`~openai_agents_skills.hooks.SkillHooks`.

        Args:
            context: The current run context, or ``None`` if not available.
            agent: The current agent, or ``None`` if not available.

        Returns:
            True if the skill is active; False to suppress injection.
        """
        return True

    @abstractmethod
    async def get_prompt_blocks(
        self,
        context: RunContextWrapper[Any] | None,
        agent: Agent[Any] | None,
        args: str = "",
    ) -> list[Any]:
        """Return prompt blocks to prepend to the LLM's input_items.

        Each block should be a response-input-item dict, for example::

            {"role": "user", "content": "Always be concise."}

        Both ``context`` and ``agent`` are ``None`` when called outside a live
        agent run (e.g. in tests or via the ``invoke_skill`` tool).
        Implementations that depend on them must handle the ``None`` case.

        Args:
            context: The current ``RunContextWrapper``, or ``None``.
            agent: The current ``Agent``, or ``None``.
            args: Optional space-separated arguments passed to the skill.
                  Used by file-based skills for ``$arg_name`` substitution.
                  Bundled skills may ignore this parameter.

        Returns:
            A list of input-item dicts to prepend to the LLM input.
        """
        ...
