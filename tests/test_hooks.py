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

    def __init__(self, content: str = "injected", name_override: str | None = None) -> None:
        self._content = content
        if name_override is not None:
            self.name = name_override

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


async def _fire_end(hooks: SkillHooks, input_items: list[Any]) -> None:
    """Invoke on_llm_end with the minimal required arguments."""
    await hooks.on_llm_end(
        context=None,  # type: ignore[arg-type]
        agent=None,  # type: ignore[arg-type]
        response=None,  # type: ignore[arg-type]
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
                _AlwaysOnSkill("first", name_override="skill_first"),
                _AlwaysOnSkill("second", name_override="skill_second"),
                _AlwaysOnSkill("third", name_override="skill_third"),
            ]
        )
        items: list[Any] = []

        await _fire(hooks, items)

        contents = [item["content"] for item in items]
        assert contents == ["first", "second", "third"]

    async def test_skill_blocks_precede_original_items_after_multi_skill_injection(
        self,
    ) -> None:
        hooks = SkillHooks(
            [
                _AlwaysOnSkill("a", name_override="skill_a"),
                _AlwaysOnSkill("b", name_override="skill_b"),
            ]
        )
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


# ---------------------------------------------------------------------------
# Multi-turn injection
# ---------------------------------------------------------------------------


class TestMultiTurnInjection:
    async def test_skill_reinjects_after_on_llm_end_clears_guard(self) -> None:
        """Skills re-inject on every turn; on_llm_end resets the per-call guard."""
        hooks = SkillHooks([_AlwaysOnSkill("content")])

        # Turn 1.
        items_turn1: list[Any] = []
        await _fire(hooks, items_turn1)
        assert items_turn1 == [{"role": "user", "content": "content"}]

        # on_llm_end resets the injection guard.
        await _fire_end(hooks, items_turn1)

        # Turn 2: skill must inject again on the new call.
        items_turn2: list[Any] = []
        await _fire(hooks, items_turn2)
        assert items_turn2 == [{"role": "user", "content": "content"}]

    async def test_without_on_llm_end_skill_not_reinjected_in_same_run(self) -> None:
        """Without on_llm_end the per-call guard is not cleared between turns.

        This documents the behaviour when the guard accumulates across calls in the
        same run (e.g. if the SDK does not fire on_llm_end in an error path).
        """
        hooks = SkillHooks([_AlwaysOnSkill("content")])

        items_turn1: list[Any] = []
        await _fire(hooks, items_turn1)

        # No on_llm_end call — guard is still populated.
        items_turn2: list[Any] = []
        await _fire(hooks, items_turn2)

        # The guard was not cleared; injection is skipped on the second call.
        assert items_turn2 == []

    async def test_multiple_turns_each_injects_after_clear(self) -> None:
        """Each turn (separated by on_llm_end) injects independently."""
        hooks = SkillHooks([_AlwaysOnSkill("block")])

        for _ in range(3):
            items: list[Any] = []
            await _fire(hooks, items)
            assert items == [{"role": "user", "content": "block"}]
            await _fire_end(hooks, items)
