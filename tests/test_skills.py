"""Tests for openai-agents-skills."""

import pytest

from openai_agents_skills import Skill, skill
from openai_agents_skills.skills import SkillRegistry

# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------


class TestSkill:
    """Tests for the Skill dataclass."""

    def test_skill_minimal(self) -> None:
        """A Skill can be created with only name and description."""
        sk = Skill(name="my_skill", description="Does something useful.")
        assert sk.name == "my_skill"
        assert sk.description == "Does something useful."
        assert sk.tools == []
        assert sk.instructions == ""
        assert sk.metadata == {}

    def test_skill_full(self) -> None:
        """A Skill accepts tools, instructions, and metadata."""
        dummy_tool = object()
        sk = Skill(
            name="full_skill",
            description="Full skill.",
            tools=[dummy_tool],
            instructions="Extra instructions.",
            metadata={"version": "1"},
        )
        assert sk.tools == [dummy_tool]
        assert sk.instructions == "Extra instructions."
        assert sk.metadata == {"version": "1"}

    def test_skill_empty_name_raises(self) -> None:
        """An empty skill name should raise ValueError."""
        with pytest.raises(ValueError, match="name"):
            Skill(name="", description="No name skill.")


# ---------------------------------------------------------------------------
# SkillRegistry — registration
# ---------------------------------------------------------------------------


class TestSkillRegistryRegistration:
    """Tests for SkillRegistry register / unregister / get."""

    def test_register_and_get(self) -> None:
        registry = SkillRegistry()
        sk = Skill(name="alpha", description="Alpha skill.")
        registry.register(sk)
        assert registry.get("alpha") is sk

    def test_register_overwrites(self) -> None:
        registry = SkillRegistry()
        sk1 = Skill(name="alpha", description="First.")
        sk2 = Skill(name="alpha", description="Second.")
        registry.register(sk1)
        registry.register(sk2)
        assert registry.get("alpha") is sk2

    def test_get_unknown_raises(self) -> None:
        registry = SkillRegistry()
        with pytest.raises(KeyError, match="unknown"):
            registry.get("unknown")

    def test_unregister(self) -> None:
        registry = SkillRegistry()
        sk = Skill(name="beta", description="Beta skill.")
        registry.register(sk)
        registry.unregister("beta")
        with pytest.raises(KeyError):
            registry.get("beta")

    def test_unregister_unknown_raises(self) -> None:
        registry = SkillRegistry()
        with pytest.raises(KeyError, match="ghost"):
            registry.unregister("ghost")

    def test_skill_names_sorted(self) -> None:
        registry = SkillRegistry()
        for name in ("charlie", "alpha", "bravo"):
            registry.register(Skill(name=name, description=f"{name} skill."))
        assert registry.skill_names == ["alpha", "bravo", "charlie"]

    def test_skill_names_empty(self) -> None:
        registry = SkillRegistry()
        assert registry.skill_names == []


# ---------------------------------------------------------------------------
# SkillRegistry — apply
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Minimal stand-in for agents.Agent used in apply() tests."""

    def __init__(
        self,
        name: str = "TestAgent",
        instructions: str = "",
        tools: list | None = None,
        model: str | None = None,
        output_type: type | None = None,
        handoffs: list | None = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.model = model
        self.output_type = output_type
        self.handoffs = handoffs or []


class TestSkillRegistryApply:
    """Tests for SkillRegistry.apply()."""

    def test_apply_adds_tools(self) -> None:
        registry = SkillRegistry()
        tool_a = object()
        sk = Skill(name="tool_skill", description="Adds a tool.", tools=[tool_a])
        registry.register(sk)

        agent = _FakeAgent(name="Agent")
        result = registry.apply(agent, skill_names=["tool_skill"])

        assert tool_a in result.tools

    def test_apply_appends_instructions(self) -> None:
        registry = SkillRegistry()
        sk = Skill(
            name="inst_skill",
            description="Adds instructions.",
            instructions="Be extra helpful.",
        )
        registry.register(sk)

        agent = _FakeAgent(name="Agent", instructions="Base instructions.")
        result = registry.apply(agent, skill_names=["inst_skill"])

        assert "Base instructions." in result.instructions
        assert "Be extra helpful." in result.instructions

    def test_apply_empty_instructions_skips_separator(self) -> None:
        """When agent has no instructions, no leading newline is prepended."""
        registry = SkillRegistry()
        sk = Skill(name="sk", description=".", instructions="Only instructions.")
        registry.register(sk)

        agent = _FakeAgent(name="Agent", instructions="")
        result = registry.apply(agent, skill_names=["sk"])

        assert result.instructions == "Only instructions."
        assert not result.instructions.startswith("\n")

    def test_apply_all_skills_when_names_is_none(self) -> None:
        registry = SkillRegistry()
        tool_x = object()
        tool_y = object()
        registry.register(Skill(name="x", description="X.", tools=[tool_x]))
        registry.register(Skill(name="y", description="Y.", tools=[tool_y]))

        agent = _FakeAgent(name="Agent")
        result = registry.apply(agent, skill_names=None)

        assert tool_x in result.tools
        assert tool_y in result.tools

    def test_apply_unknown_skill_raises(self) -> None:
        registry = SkillRegistry()
        agent = _FakeAgent(name="Agent")
        with pytest.raises(KeyError, match="missing"):
            registry.apply(agent, skill_names=["missing"])

    def test_apply_preserves_existing_tools(self) -> None:
        registry = SkillRegistry()
        existing_tool = object()
        new_tool = object()
        sk = Skill(name="new", description="New.", tools=[new_tool])
        registry.register(sk)

        agent = _FakeAgent(name="Agent", tools=[existing_tool])
        result = registry.apply(agent, skill_names=["new"])

        assert existing_tool in result.tools
        assert new_tool in result.tools

    def test_apply_does_not_mutate_original_agent(self) -> None:
        registry = SkillRegistry()
        tool = object()
        sk = Skill(name="sk", description=".", tools=[tool])
        registry.register(sk)

        agent = _FakeAgent(name="Agent", tools=[])
        registry.apply(agent, skill_names=["sk"])

        # Original agent's tools must be unchanged
        assert agent.tools == []


# ---------------------------------------------------------------------------
# @skill decorator
# ---------------------------------------------------------------------------


class TestSkillDecorator:
    """Tests for the @skill factory decorator."""

    def test_decorator_attaches_metadata(self) -> None:
        @skill(name="decorated", description="A decorated factory.")
        def make_skill() -> Skill:
            return Skill(name="decorated", description="A decorated factory.")

        assert make_skill.__skill_name__ == "decorated"
        assert make_skill.__skill_description__ == "A decorated factory."

    def test_decorator_factory_still_callable(self) -> None:
        @skill(name="callable_skill", description="Still callable.")
        def make_skill() -> Skill:
            return Skill(name="callable_skill", description="Still callable.")

        result = make_skill()
        assert isinstance(result, Skill)
        assert result.name == "callable_skill"

    def test_decorator_default_description(self) -> None:
        @skill(name="no_desc")
        def make_skill() -> Skill:
            return Skill(name="no_desc", description="")

        assert make_skill.__skill_description__ == ""
