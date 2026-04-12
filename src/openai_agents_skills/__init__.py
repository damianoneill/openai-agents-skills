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

Requirements::

    pip install openai-agents-skills
"""

from openai_agents_skills._version import __version__
from openai_agents_skills.hooks import SkillHooks
from openai_agents_skills.skills import Skill, SkillProtocol, skill_factory

__all__ = [
    "__version__",
    "Skill",
    "SkillHooks",
    "SkillProtocol",
    "skill_factory",
]
