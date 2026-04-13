"""Tests for SkillHooks injection behaviour."""

from __future__ import annotations

from typing import Any

from openai_agents_skills import Skill, SkillHooks

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _AlwaysOnSkill(Skill):
    """Skill that is always enabled and returns a single configurable block."""

    name = "always_on"
    description = "Always injects."

    def __init__(self, content: str = "injected") -> None:
        self._content = content

    async def get_prompt_blocks(self, args: str = "") -> list[Any]:
        return [{"role": "user", "content": self._content}]


class _DisabledSkill(Skill):
    """Skill whose is_enabled() always returns False."""

    name = "disabled"
    description = "Never injects."

    def is_enabled(self) -> bool:
        return False

    async def get_prompt_blocks(self, args: str = "") -> list[Any]:
        return [{"role": "user", "content": "should not appear"}]


class _MultiBlockSkill(Skill):
    """Skill that returns two blocks per call."""

    name = "multi_block"
    description = "Returns two blocks."

    async def get_prompt_blocks(self, args: str = "") -> list[Any]:
        return [
            {"role": "user", "content": "block one"},
            {"role": "user", "content": "block two"},
        ]


async def _fire(hooks: SkillHooks, input_items: list[Any]) -> None:
    """Invoke on_llm_start with the minimal required arguments."""
    await hooks.on_llm_start(
        context=None,  # type: ignore[arg-type]
        agent=None,  # type: ignore[arg-type]
        system_prompt=None,
        input_items=input_items,
    )


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


class TestSkillHooksInjection:
    async def test_single_skill_prepends_block(self) -> None:
        hooks = SkillHooks([_AlwaysOnSkill("skill content")])
        items: list[Any] = [{"role": "user", "content": "user message"}]

        await _fire(hooks, items)

        assert items[0] == {"role": "user", "content": "skill content"}
        assert items[1] == {"role": "user", "content": "user message"}

    async def test_skill_block_appears_before_existing_items(self) -> None:
        hooks = SkillHooks([_AlwaysOnSkill("prepended")])
        original = {"role": "user", "content": "original"}
        items: list[Any] = [original]

        await _fire(hooks, items)

        assert items[-1] is original

    async def test_empty_skills_list_is_noop(self) -> None:
        hooks = SkillHooks([])
        items: list[Any] = [{"role": "user", "content": "unchanged"}]

        await _fire(hooks, items)

        assert items == [{"role": "user", "content": "unchanged"}]

    async def test_empty_input_items_receives_blocks(self) -> None:
        hooks = SkillHooks([_AlwaysOnSkill("block")])
        items: list[Any] = []

        await _fire(hooks, items)

        assert items == [{"role": "user", "content": "block"}]

    async def test_original_list_object_is_mutated_not_replaced(self) -> None:
        hooks = SkillHooks([_AlwaysOnSkill("injected")])
        items: list[Any] = []
        original_id = id(items)

        await _fire(hooks, items)

        assert id(items) == original_id
        assert len(items) == 1


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestSkillHooksOrdering:
    async def test_multiple_skills_inject_in_registration_order(self) -> None:
        hooks = SkillHooks(
            [
                _AlwaysOnSkill("first"),
                _AlwaysOnSkill("second"),
                _AlwaysOnSkill("third"),
            ]
        )
        items: list[Any] = []

        await _fire(hooks, items)

        contents = [item["content"] for item in items]
        assert contents == ["first", "second", "third"]

    async def test_skill_blocks_precede_original_items_after_multi_skill_injection(
        self,
    ) -> None:
        hooks = SkillHooks([_AlwaysOnSkill("a"), _AlwaysOnSkill("b")])
        items: list[Any] = [{"role": "user", "content": "original"}]

        await _fire(hooks, items)

        assert items[0]["content"] == "a"
        assert items[1]["content"] == "b"
        assert items[2]["content"] == "original"


# ---------------------------------------------------------------------------
# Multi-block skills
# ---------------------------------------------------------------------------


class TestMultiBlockSkills:
    async def test_multiple_blocks_from_one_skill_are_all_prepended(self) -> None:
        hooks = SkillHooks([_MultiBlockSkill()])
        items: list[Any] = [{"role": "user", "content": "original"}]

        await _fire(hooks, items)

        assert items[0]["content"] == "block one"
        assert items[1]["content"] == "block two"
        assert items[2]["content"] == "original"

    async def test_multi_block_skill_followed_by_single_block_skill(self) -> None:
        hooks = SkillHooks([_MultiBlockSkill(), _AlwaysOnSkill("single")])
        items: list[Any] = []

        await _fire(hooks, items)

        assert items[0]["content"] == "block one"
        assert items[1]["content"] == "block two"
        assert items[2]["content"] == "single"


# ---------------------------------------------------------------------------
# Disabled skills
# ---------------------------------------------------------------------------


class TestDisabledSkills:
    async def test_disabled_skill_is_not_injected(self) -> None:
        hooks = SkillHooks([_DisabledSkill()])
        items: list[Any] = [{"role": "user", "content": "user message"}]

        await _fire(hooks, items)

        assert len(items) == 1
        assert items[0]["content"] == "user message"

    async def test_enabled_and_disabled_skills_mixed(self) -> None:
        hooks = SkillHooks([_AlwaysOnSkill("enabled"), _DisabledSkill()])
        items: list[Any] = []

        await _fire(hooks, items)

        assert len(items) == 1
        assert items[0]["content"] == "enabled"

    async def test_all_disabled_leaves_items_unchanged(self) -> None:
        hooks = SkillHooks([_DisabledSkill(), _DisabledSkill()])
        items: list[Any] = [{"role": "user", "content": "original"}]

        await _fire(hooks, items)

        assert items == [{"role": "user", "content": "original"}]
