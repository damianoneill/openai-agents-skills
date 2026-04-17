"""Tests for SkillRegistry — CRUD, skill_names, get_always_on, and select_for_message."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import MockRouter

from openai_agents_skills import Skill, SkillRegistry

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _AlwaysOnSkill(Skill):
    """Enabled, unconditional skill (when_to_use is empty)."""

    name = "always_on"
    description = "Always injects."
    when_to_use = ""

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": "always on content"}]


class _RoutableSkill(Skill):
    """Skill with a non-empty when_to_use — eligible for routing only."""

    name = "routable"
    description = "Routable skill."
    when_to_use = "Use when the user asks about routing topics."

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": "routed content"}]


class _DisabledSkill(Skill):
    """Unconditional skill whose is_enabled() always returns False."""

    name = "disabled"
    description = "Never active."
    when_to_use = ""

    def is_enabled(self, context=None, agent=None) -> bool:
        return False

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": "should not appear"}]


class _SecondAlwaysOnSkill(Skill):
    """A second always-on skill with a different name for multi-skill tests."""

    name = "second"
    description = "Second always-on skill."
    when_to_use = ""

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": "second content"}]


class _RoutableB(Skill):
    name = "routable_b"
    description = "B"
    when_to_use = "Use for B."

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return []


# ---------------------------------------------------------------------------
# register / get / unregister
# ---------------------------------------------------------------------------


class TestRegisterGetUnregister:
    def test_register_and_get_returns_same_instance(self) -> None:
        registry = SkillRegistry()
        skill = _AlwaysOnSkill()

        registry.register(skill)

        assert registry.get("always_on") is skill

    def test_get_unknown_name_raises_keyerror(self) -> None:
        registry = SkillRegistry()

        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_duplicate_register_overwrites_previous_skill(self) -> None:
        registry = SkillRegistry()
        first = _AlwaysOnSkill()
        second = _AlwaysOnSkill()

        registry.register(first)
        registry.register(second)

        # The second registration replaces the first.
        assert registry.get("always_on") is second
        assert registry.get("always_on") is not first

    def test_unregister_removes_skill(self) -> None:
        registry = SkillRegistry()
        registry.register(_AlwaysOnSkill())

        registry.unregister("always_on")

        with pytest.raises(KeyError):
            registry.get("always_on")

    def test_unregister_unknown_name_is_silent(self) -> None:
        registry = SkillRegistry()
        # Should not raise even though the name was never registered.
        registry.unregister("does_not_exist")

    def test_register_multiple_different_skills(self) -> None:
        registry = SkillRegistry()
        always = _AlwaysOnSkill()
        routable = _RoutableSkill()

        registry.register(always)
        registry.register(routable)

        assert registry.get("always_on") is always
        assert registry.get("routable") is routable

    def test_unregister_leaves_other_skills_intact(self) -> None:
        registry = SkillRegistry()
        registry.register(_AlwaysOnSkill())
        registry.register(_RoutableSkill())

        registry.unregister("always_on")

        with pytest.raises(KeyError):
            registry.get("always_on")
        # Other skill is still accessible.
        assert registry.get("routable") is not None

    def test_register_empty_name_raises_value_error(self) -> None:
        """Registering a skill with an empty name must raise ValueError."""

        class _UnnamedSkill(Skill):
            name = ""
            description = "No name."

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                return []

        registry = SkillRegistry()
        with pytest.raises(ValueError, match="empty"):
            registry.register(_UnnamedSkill())

    def test_register_whitespace_only_name_raises_value_error(self) -> None:
        """Registering a skill with a whitespace-only name must raise ValueError."""

        class _WhitespaceSkill(Skill):
            name = "   "
            description = "Whitespace name."

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                return []

        registry = SkillRegistry()
        with pytest.raises(ValueError, match="empty"):
            registry.register(_WhitespaceSkill())


# ---------------------------------------------------------------------------
# skill_names
# ---------------------------------------------------------------------------


class TestSkillNames:
    def test_skill_names_empty_registry_returns_empty_list(self) -> None:
        registry = SkillRegistry()

        assert registry.skill_names == []

    def test_skill_names_returns_sorted_list(self) -> None:
        registry = SkillRegistry()
        # Register in non-alphabetical order.
        registry.register(_RoutableSkill())  # "routable"
        registry.register(_AlwaysOnSkill())  # "always_on"
        registry.register(_SecondAlwaysOnSkill())  # "second"

        names = registry.skill_names

        assert names == ["always_on", "routable", "second"]

    def test_skill_names_single_skill(self) -> None:
        registry = SkillRegistry()
        registry.register(_AlwaysOnSkill())

        assert registry.skill_names == ["always_on"]

    def test_skill_names_after_unregister_excludes_removed_name(self) -> None:
        registry = SkillRegistry()
        registry.register(_AlwaysOnSkill())
        registry.register(_RoutableSkill())

        registry.unregister("always_on")

        assert registry.skill_names == ["routable"]

    def test_skill_names_after_overwrite_still_unique(self) -> None:
        registry = SkillRegistry()
        registry.register(_AlwaysOnSkill())
        # Register again under the same name — no duplicates in the list.
        registry.register(_AlwaysOnSkill())

        assert registry.skill_names == ["always_on"]


# ---------------------------------------------------------------------------
# get_always_on
# ---------------------------------------------------------------------------


class TestGetAlwaysOn:
    def test_empty_registry_returns_empty_list(self) -> None:
        registry = SkillRegistry()

        assert registry.get_always_on() == []

    def test_returns_enabled_unconditional_skill(self) -> None:
        registry = SkillRegistry()
        skill = _AlwaysOnSkill()
        registry.register(skill)

        result = registry.get_always_on()

        assert skill in result

    def test_excludes_skill_with_non_empty_when_to_use(self) -> None:
        registry = SkillRegistry()
        registry.register(_RoutableSkill())  # when_to_use is non-empty

        result = registry.get_always_on()

        assert result == []

    def test_excludes_disabled_skill(self) -> None:
        registry = SkillRegistry()
        registry.register(_DisabledSkill())  # is_enabled() returns False

        result = registry.get_always_on()

        assert result == []

    def test_mix_returns_only_eligible_skills(self) -> None:
        registry = SkillRegistry()
        always = _AlwaysOnSkill()
        routable = _RoutableSkill()
        disabled = _DisabledSkill()

        registry.register(always)
        registry.register(routable)
        registry.register(disabled)

        result = registry.get_always_on()

        assert always in result
        assert routable not in result
        assert disabled not in result

    def test_multiple_always_on_skills_all_returned(self) -> None:
        registry = SkillRegistry()
        first = _AlwaysOnSkill()
        second = _SecondAlwaysOnSkill()
        registry.register(first)
        registry.register(second)

        result = registry.get_always_on()

        assert first in result
        assert second in result
        assert len(result) == 2

    def test_dynamically_disabled_skill_excluded(self) -> None:
        """A skill that returns False from is_enabled() is excluded even if registered."""

        class _ToggleSkill(Skill):
            name = "toggle"
            description = "Toggleable."
            when_to_use = ""

            def __init__(self, active: bool) -> None:
                self._active = active

            def is_enabled(self, context=None, agent=None) -> bool:
                return self._active

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                return []

        registry = SkillRegistry()
        registry.register(_ToggleSkill(active=False))

        assert registry.get_always_on() == []

        # Re-register the enabled variant; now it should appear.
        registry.register(_ToggleSkill(active=True))
        result = registry.get_always_on()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# select_for_message — no router
# ---------------------------------------------------------------------------


class TestSelectForMessageNoRouter:
    async def test_no_router_returns_empty_list(self) -> None:
        registry = SkillRegistry()  # router=None by default
        registry.register(_AlwaysOnSkill())

        result = await registry.select_for_message("any message")

        assert result == []

    async def test_no_router_ignores_message_content(self) -> None:
        registry = SkillRegistry(router=None)
        registry.register(_RoutableSkill())

        result = await registry.select_for_message("use when routing")

        assert result == []

    async def test_no_router_empty_registry_returns_empty_list(self) -> None:
        registry = SkillRegistry(router=None)

        result = await registry.select_for_message("test")

        assert result == []

    async def test_no_router_with_multiple_skills_still_returns_empty(self) -> None:
        registry = SkillRegistry(router=None)
        registry.register(_AlwaysOnSkill())
        registry.register(_RoutableSkill())
        registry.register(_SecondAlwaysOnSkill())

        result = await registry.select_for_message("test message")

        assert result == []


# ---------------------------------------------------------------------------
# select_for_message — with router
# ---------------------------------------------------------------------------


class TestSelectForMessageWithRouter:
    async def test_router_called_with_message_and_skills_returned(self) -> None:
        router = MockRouter(names=["routable"])
        registry = SkillRegistry(router=router)
        skill = _RoutableSkill()
        registry.register(skill)

        result = await registry.select_for_message("route this")

        assert skill in result
        assert len(router.calls) == 1
        assert router.calls[0][0] == "route this"

    async def test_unknown_names_in_router_response_silently_skipped(self) -> None:
        router = MockRouter(names=["nonexistent_skill"])
        registry = SkillRegistry(router=router)
        registry.register(_RoutableSkill())

        result = await registry.select_for_message("test")

        # The router returned an unknown name; it must be silently ignored.
        assert result == []

    async def test_only_routable_skills_passed_to_router(self) -> None:
        """Skills with empty when_to_use must NOT be forwarded to the router."""
        router = MockRouter(names=[])
        registry = SkillRegistry(router=router)
        registry.register(_AlwaysOnSkill())  # when_to_use="" — not routable
        registry.register(_RoutableSkill())  # when_to_use!="" — routable

        await registry.select_for_message("test message")

        assert len(router.calls) == 1
        passed_names = router.calls[0][1]
        assert "always_on" not in passed_names
        assert "routable" in passed_names

    async def test_mix_of_known_and_unknown_router_names(self) -> None:
        """Router may return a mix; only names found in the registry are included."""
        router = MockRouter(names=["routable", "ghost"])
        registry = SkillRegistry(router=router)
        skill = _RoutableSkill()
        registry.register(skill)

        result = await registry.select_for_message("test")

        assert skill in result
        # "ghost" is not registered; only the known skill appears.
        assert len(result) == 1

    async def test_router_not_called_when_no_routable_skills_registered(self) -> None:
        """When all registered skills have empty when_to_use the router is bypassed."""
        router = MockRouter(names=["always_on"])
        registry = SkillRegistry(router=router)
        registry.register(_AlwaysOnSkill())  # not routable

        result = await registry.select_for_message("test")

        assert result == []
        # Router should not have been called.
        assert len(router.calls) == 0

    async def test_router_returns_multiple_skills_in_order(self) -> None:
        """The order of names returned by the router is preserved."""

        router = MockRouter(names=["routable", "routable_b"])
        registry = SkillRegistry(router=router)
        skill_a = _RoutableSkill()
        skill_b = _RoutableB()
        registry.register(skill_a)
        registry.register(skill_b)

        result = await registry.select_for_message("test")

        assert result[0] is skill_a
        assert result[1] is skill_b

    async def test_disabled_always_on_not_in_always_on_but_routable_still_routed(
        self,
    ) -> None:
        """A disabled skill with when_to_use="" is excluded from always-on and never routed."""
        router = MockRouter(names=[])
        registry = SkillRegistry(router=router)
        registry.register(_DisabledSkill())

        result = await registry.select_for_message("test")

        assert result == []
        # Disabled skill has empty when_to_use, so router is not called at all.
        assert len(router.calls) == 0

    async def test_disabled_routable_skill_not_forwarded_to_router(self) -> None:
        """A routable skill that returns False from is_enabled() must not be
        forwarded to the router — disabled skills waste routing tokens."""

        class _DisabledRoutable(Skill):
            name = "disabled_routable"
            description = "Disabled but routable."
            when_to_use = "Use when routing."

            def is_enabled(self, context=None, agent=None) -> bool:
                return False

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                return []

        router = MockRouter(names=["disabled_routable"])
        registry = SkillRegistry(router=router)
        registry.register(_DisabledRoutable())

        result = await registry.select_for_message("test")

        # The router should not have been called because there are no enabled
        # routable skills to forward.
        assert len(router.calls) == 0
        assert result == []
