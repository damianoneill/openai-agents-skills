"""OpenAI Agents Skills.

Skills extension for the OpenAI Agents SDK, providing reusable, composable
capabilities that integrate with the agent loop — inspired by Claude Skills.

A Skill bundles tools, instructions, and behaviours into a named, shareable
unit that can be attached to any Agent without duplicating configuration.

Usage:
    from agents import Agent
    from openai_agents_skills import Skill, skill

    # Define a skill using the decorator
    @skill(name="web_search", description="Search the web for information")
    class WebSearchSkill(Skill):
        ...

    # Attach the skill to an agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        tools=[*WebSearchSkill.tools],
    )

Requirements:
    pip install openai-agents-skills
"""

from openai_agents_skills._version import __version__
from openai_agents_skills.skills import Skill, SkillRegistry, skill

__all__ = [
    "__version__",
    "Skill",
    "SkillRegistry",
    "skill",
]
