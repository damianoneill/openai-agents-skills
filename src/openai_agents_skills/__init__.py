"""OpenAI Agents Skills.

Skills extension for the OpenAI Agents SDK. A Skill is a named, reusable
prompt fragment injected into the LLM's context at the right moment in the
agent loop via AgentHooks.

Usage::

    from agents import Agent, Runner
    from openai_agents_skills import Skill, SkillHooks

    class MyConciseSkill(Skill):
        name = "concise"
        description = "Instructs the agent to be concise."

        async def get_prompt_blocks(self, args: str = "") -> list:
            return [{"role": "user", "content": "Be concise in all responses."}]

    agent = Agent(
        name="Assistant",
        instructions="You are helpful.",
        hooks=SkillHooks([MyConciseSkill()]),
    )

    result = await Runner.run(agent, "Tell me about Python.")

Phase 2 — registry and routing::

    from openai import AsyncOpenAI
    from openai_agents_skills import (
        LLMSkillRouter,
        SkillRegistry,
        RunSkillHooks,
        make_invoke_skill_tool,
    )

    router = LLMSkillRouter(client=AsyncOpenAI(), model="gpt-4o-mini")
    registry = SkillRegistry(router=router)
    registry.register(MyConciseSkill())

    result = await Runner.run(
        agent,
        input="...",
        hooks=RunSkillHooks(registry=registry),
    )

Requirements::

    pip install openai-agents-skills
"""

from openai_agents_skills._version import __version__
from openai_agents_skills.hooks import RunSkillHooks, SkillHooks, make_invoke_skill_tool
from openai_agents_skills.registry import SkillRegistry
from openai_agents_skills.router import LLMSkillRouter, SkillRouter
from openai_agents_skills.skills import Skill

__all__ = [
    "__version__",
    "LLMSkillRouter",
    "RunSkillHooks",
    "Skill",
    "SkillHooks",
    "SkillRegistry",
    "SkillRouter",
    "make_invoke_skill_tool",
]
