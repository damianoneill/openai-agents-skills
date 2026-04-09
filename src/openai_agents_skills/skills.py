"""Skills for the OpenAI Agents SDK.

A Skill is a reusable, composable capability that can be applied to an Agent,
bundling together tools, instructions, and context into a named unit — similar
to Claude Skills but designed to work within the OpenAI Agents SDK agent loop.

Usage::

    from agents import Agent
    from openai_agents_skills import Skill, SkillRegistry, skill

    @skill(name="web_search", description="Search the web for information")
    def web_search_skill() -> Skill:
        from agents import function_tool

        @function_tool
        def search(query: str) -> str:
            # TODO: implement
            return f"Results for: {query}"

        return Skill(
            name="web_search",
            description="Search the web for information",
            tools=[search],
            instructions="You can search the web to find up-to-date information.",
        )

    # Apply a skill to an agent
    registry = SkillRegistry()
    registry.register(web_search_skill())

    agent = Agent(name="Assistant", instructions="You are helpful.")
    agent_with_skills = registry.apply(agent, skill_names=["web_search"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A reusable, composable capability for an OpenAI Agents SDK agent.

    A Skill bundles tools, instructions, and metadata into a single unit that
    can be registered in a :class:`SkillRegistry` and applied to any
    :class:`agents.Agent` at construction time or before a run.

    Attributes:
        name: Unique identifier for the skill.
        description: Human-readable description of what the skill provides.
        tools: List of function tools exposed by this skill.
        instructions: Additional system-prompt fragment injected when the skill
            is applied to an agent.
        metadata: Arbitrary key/value pairs for extension points.
    """

    name: str
    description: str
    tools: list[Any] = field(default_factory=list)
    instructions: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Skill name must not be empty.")


class SkillRegistry:
    """Registry for managing and applying :class:`Skill` instances.

    Skills are keyed by their :attr:`Skill.name`.  Applying a registry to an
    agent merges the selected skills' tools and instructions into a new
    :class:`agents.Agent` instance, leaving the original unchanged.

    Example::

        registry = SkillRegistry()
        registry.register(my_skill)
        enriched_agent = registry.apply(agent, skill_names=["my_skill"])
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill: Skill) -> None:
        """Register a :class:`Skill`, replacing any existing entry with the same name.

        Args:
            skill: The skill instance to register.
        """
        if skill.name in self._skills:
            logger.debug("Overwriting existing skill %r in registry.", skill.name)
        self._skills[skill.name] = skill
        logger.debug("Registered skill %r.", skill.name)

    def unregister(self, name: str) -> None:
        """Remove a skill from the registry by name.

        Args:
            name: The :attr:`Skill.name` of the skill to remove.

        Raises:
            KeyError: If no skill with *name* is registered.
        """
        if name not in self._skills:
            raise KeyError(f"No skill named {name!r} is registered.")
        del self._skills[name]
        logger.debug("Unregistered skill %r.", name)

    def get(self, name: str) -> Skill:
        """Retrieve a registered skill by name.

        Args:
            name: The :attr:`Skill.name` to look up.

        Returns:
            The matching :class:`Skill`.

        Raises:
            KeyError: If no skill with *name* is registered.
        """
        if name not in self._skills:
            raise KeyError(f"No skill named {name!r} is registered.")
        return self._skills[name]

    @property
    def skill_names(self) -> list[str]:
        """Return a sorted list of all registered skill names."""
        return sorted(self._skills.keys())

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(self, agent: Any, skill_names: list[str] | None = None) -> Any:
        """Apply skills to an agent, returning a new agent with merged capabilities.

        The original *agent* is not mutated.  Tools from each requested skill
        are appended to the agent's existing tool list, and each skill's
        :attr:`Skill.instructions` fragment is appended to the agent's system
        instructions separated by a newline.

        Args:
            agent: An :class:`agents.Agent` instance to enrich.
            skill_names: Names of skills to apply.  If *None*, all registered
                skills are applied.

        Returns:
            A new :class:`agents.Agent` with the merged tools and instructions.

        Raises:
            KeyError: If a requested skill name is not registered.
        """
        names = skill_names if skill_names is not None else self.skill_names
        skills = [self.get(n) for n in names]

        merged_tools = list(agent.tools or [])
        merged_instructions = agent.instructions or ""

        for sk in skills:
            merged_tools.extend(sk.tools)
            if sk.instructions:
                separator = "\n" if merged_instructions else ""
                merged_instructions = merged_instructions + separator + sk.instructions
            logger.debug("Applied skill %r to agent %r.", sk.name, agent.name)

        # Import here to avoid a hard dependency at module load time —
        # allows the module to be imported even in environments where
        # openai-agents is not yet installed (e.g. during packaging).
        from agents import Agent

        return (
            agent.clone(tools=merged_tools, instructions=merged_instructions)
            if hasattr(agent, "clone")
            else Agent(
                name=agent.name,
                instructions=merged_instructions,
                tools=merged_tools,
                model=getattr(agent, "model", None),
                output_type=getattr(agent, "output_type", None),
                handoffs=getattr(agent, "handoffs", []),
            )
        )


# ---------------------------------------------------------------------------
# Decorator helper
# ---------------------------------------------------------------------------


def skill(
    name: str,
    description: str = "",
) -> Any:
    """Decorator that tags a factory function as a skill factory.

    The decorated function must return a :class:`Skill` instance.  The
    decorator is a lightweight annotation only — it does **not** call the
    function or register the skill automatically.

    Args:
        name: The skill name (must match the returned :attr:`Skill.name`).
        description: Optional description surfaced in tooling / docs.

    Returns:
        A decorator that annotates the factory function with skill metadata.

    Example::

        @skill(name="summariser", description="Summarise long documents")
        def summariser_skill() -> Skill:
            ...
            return Skill(name="summariser", description="...", tools=[...])
    """

    def decorator(fn: Any) -> Any:
        fn.__skill_name__ = name
        fn.__skill_description__ = description
        return fn

    return decorator
