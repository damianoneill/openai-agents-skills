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
        description = "Always cite sources."

        async def get_prompt_blocks(self, args: str = "") -> list:
            return [{"role": "user", "content": "Always cite your sources."}]

    agent = Agent(
        name="Assistant",
        instructions="You are helpful.",
        hooks=SkillHooks([CitationSkill()]),
    )

    result = await Runner.run(agent, "What is the speed of light?")
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SkillProtocol(Protocol):
    """Protocol that any skill implementation must satisfy.

    Any object with ``name``, ``description``, and ``get_prompt_blocks`` can be
    used with :class:`~openai_agents_skills.hooks.SkillHooks` — subclassing
    :class:`Skill` is not required.
    """

    name: str
    description: str

    async def get_prompt_blocks(self, args: str = "") -> list[Any]:
        """Return prompt blocks to inject into the LLM's input_items list."""
        ...


class Skill:
    """Base class for bundled (code-defined) skills.

    Subclass and override :meth:`get_prompt_blocks` to provide skill content.
    The returned blocks are prepended to the LLM's ``input_items`` list before
    each model invocation by :class:`~openai_agents_skills.hooks.SkillHooks`.

    Class attributes:

    Attributes:
        name: Unique identifier for the skill.
        description: Short human-readable summary. Shown in the manifest and
            used by future routing logic.
        when_to_use: Prose description of when the skill should be triggered,
            including example phrases. Used by Phase 2 routing.

    Example::

        class ReplyInBulletsSkill(Skill):
            name = "reply_in_bullets"
            description = "Instructs the agent to respond using bullet points."

            async def get_prompt_blocks(self, args: str = "") -> list:
                return [{"role": "user", "content": "Always respond using bullet points."}]
    """

    name: str = ""
    description: str = ""
    when_to_use: str = ""

    def is_enabled(self) -> bool:
        """Return True if this skill should be injected on the current call.

        Override to gate a skill dynamically on feature flags, runtime context,
        or any other condition. Disabled skills are silently skipped by
        :class:`~openai_agents_skills.hooks.SkillHooks`.

        Returns:
            True if the skill is active; False to suppress injection.
        """
        return True

    async def get_prompt_blocks(self, args: str = "") -> list[Any]:
        """Return prompt blocks to prepend to the LLM's input_items.

        Each block should be a response-input-item dict, for example::

            {"role": "user", "content": "Always be concise."}

        Args:
            args: Optional space-separated arguments passed to the skill.
                  Used by file-based skills for ``$arg_name`` substitution
                  (Phase 3). Bundled skills may ignore this parameter.

        Returns:
            A list of input-item dicts to prepend to the LLM input.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement get_prompt_blocks()")


def skill_factory(
    name: str,
    description: str = "",
) -> Any:
    """Decorator that tags a factory function as a skill factory.

    The decorated function must return a :class:`Skill` instance. This
    decorator is a lightweight annotation only — it does not call the function
    or register the skill automatically. Its primary use is skill discovery
    tooling (Phase 3+).

    Has no runtime effect in the current version. To register a skill, call
    the factory and pass the result to ``SkillRegistry.register``.

    Args:
        name: The skill name. Should match the returned :attr:`Skill.name`.
        description: Optional description surfaced in tooling and docs.

    Returns:
        A decorator that attaches ``__skill_name__`` and
        ``__skill_description__`` to the factory function.

    Example::

        @skill_factory(name="summariser", description="Summarise long documents.")
        def make_summariser() -> Skill:
            return MySummariserSkill()

        # Metadata available without calling the factory:
        print(make_summariser.__skill_name__)         # "summariser"
        print(make_summariser.__skill_description__)  # "Summarise long documents."
    """

    def decorator(fn: Any) -> Any:
        fn.__skill_name__ = name
        fn.__skill_description__ = description
        return fn

    return decorator
