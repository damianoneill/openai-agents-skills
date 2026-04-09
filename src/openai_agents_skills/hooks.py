"""SkillHooks — AgentHooks implementation that injects skills into the agent loop.

The injection mechanism relies on the SDK passing a mutable ``input_items`` list
to ``AgentHooks.on_llm_start`` before every model invocation, and then passing
the same list object directly to ``model.get_response()``.  Any items prepended
inside the hook are therefore visible to the LLM on that call.

Usage::

    from agents import Agent, Runner
    from openai_agents_skills import Skill, SkillHooks

    class BulletSkill(Skill):
        name = "bullet"
        description = "Always reply in bullet points."

        async def get_prompt_blocks(self, args: str = "") -> list:
            return [{"role": "user", "content": "Always respond using bullet points."}]

    agent = Agent(
        name="Assistant",
        instructions="You are helpful.",
        hooks=SkillHooks([BulletSkill()]),
    )

    result = await Runner.run(agent, "Tell me about Python.")
"""

from __future__ import annotations

from typing import Any, Optional

from agents import Agent, AgentHooks, RunContextWrapper

from .skills import Skill, SkillProtocol


class SkillHooks(AgentHooks):
    """AgentHooks implementation that prepends skill prompts before each LLM call.

    Attach an instance to an ``Agent`` via ``agent.hooks`` to enable skills for
    that specific agent.  All enabled skills are evaluated on every
    ``on_llm_start`` call; their prompt blocks are prepended to ``input_items``
    in registration order so the first skill in the list ends up closest to the
    top of the conversation.

    Skills that are ``Skill`` subclass instances and whose ``is_enabled()``
    returns ``False`` are silently skipped.  Any object that satisfies
    :class:`~openai_agents_skills.SkillProtocol` is accepted, including
    duck-typed classes that do not inherit from :class:`~openai_agents_skills.Skill`.

    Example::

        hooks = SkillHooks([CitationSkill(), BulletSkill()])

        agent = Agent(
            name="Assistant",
            instructions="You are helpful.",
            hooks=hooks,
        )
    """

    def __init__(self, skills: list[SkillProtocol]) -> None:
        """Initialise with a list of skills to inject before each LLM call.

        Args:
            skills: Skills to inject.  Must satisfy :class:`SkillProtocol`.
                    :class:`Skill` subclasses whose ``is_enabled()`` returns
                    ``False`` at call time are skipped automatically.
        """
        self._skills: list[SkillProtocol] = list(skills)

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: Optional[str],
        input_items: list[Any],
    ) -> None:
        """Prepend skill prompt blocks to ``input_items`` before the LLM is called.

        Iterates over all registered skills in order.  For each enabled skill,
        calls ``get_prompt_blocks()`` and prepends the returned blocks to the
        front of ``input_items``.  The list is mutated in place; the same object
        is then used by the SDK for the model call, so the injected content is
        visible to the LLM without any further intervention.

        Skills are injected in registration order: the *first* skill in the list
        ends up at position 0 after all injections complete.

        Args:
            context: The run context wrapper provided by the SDK.
            agent: The agent currently executing.
            system_prompt: The agent's system prompt (informational; read-only).
            input_items: Mutable list of input items for the upcoming LLM call.
                         Blocks prepended here are seen by the model.
        """
        all_blocks: list[Any] = []
        for skill in self._skills:
            if isinstance(skill, Skill) and not skill.is_enabled():
                continue
            blocks = await skill.get_prompt_blocks()
            all_blocks.extend(blocks)
        input_items[0:0] = all_blocks
