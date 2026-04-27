"""Tests for SkillHooks and RunSkillHooks injection behaviour.

Covers: basic injection, ordering, multi-block skills, disabled skills,
multi-turn injection, _extract_routing_context, _deduplicate,
SkillHooks with registry (no router), SkillHooks with router,
deduplication, error resilience, manifest injection, RunSkillHooks,
double-injection guard, make_invoke_skill_tool, invoke_skill content
delivery, and router + invoke_skill coexistence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
from agents.tool_context import ToolContext
from conftest import (
    MockRouter,
    SimpleSkill,
    extract_contents,
    fire_llm_end,
    fire_llm_start,
)

from openai_agents_skills import (
    RunSkillHooks,
    Skill,
    SkillHooks,
    SkillRegistry,
    make_invoke_skill_tool,
)
from openai_agents_skills._state import RunState, _get_run_state, _run_state
from openai_agents_skills.hooks import _deduplicate, _extract_routing_context

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _MultiBlockSkill(Skill):
    """Skill that returns two blocks per call."""

    def __init__(
        self, name: str = "multi", block_a: str = "block_a", block_b: str = "block_b"
    ) -> None:
        self.name = name
        self.description = f"Multi-block skill {name}"
        self._block_a = block_a
        self._block_b = block_b

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [
            {"role": "user", "content": self._block_a},
            {"role": "user", "content": self._block_b},
        ]


class _ErrorSkill(Skill):
    """Skill that raises ValueError from get_prompt_blocks."""

    def __init__(self, name: str = "error_skill") -> None:
        self.name = name
        self.description = "Always raises."

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        raise ValueError(f"Skill {self.name!r} intentionally failed")


async def _invoke(tool: Any, skill_name: str, args: str = "") -> str:
    """Helper to invoke a FunctionTool returned by make_invoke_skill_tool."""
    args_json = json.dumps({"skill_name": skill_name, "args": args})
    ctx = ToolContext(
        context=None,
        tool_name="invoke_skill",
        tool_call_id="test_call_id",
        tool_arguments=args_json,
    )
    return await tool.on_invoke_tool(ctx, args_json)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Basic injection
# ---------------------------------------------------------------------------


class TestSkillHooksInjection:
    async def test_single_skill_prepends_block(self) -> None:
        hooks = SkillHooks([SimpleSkill("s", content="skill content", always_on=True)])
        items: list[Any] = [{"role": "user", "content": "user message"}]

        await fire_llm_start(hooks, items)

        assert items[0] == {"role": "user", "content": "skill content"}
        assert items[1] == {"role": "user", "content": "user message"}

    async def test_skill_block_appears_before_existing_items(self) -> None:
        hooks = SkillHooks([SimpleSkill("s", content="prepended", always_on=True)])
        original = {"role": "user", "content": "original"}
        items: list[Any] = [original]

        await fire_llm_start(hooks, items)

        assert items[-1] is original

    async def test_empty_skills_list_is_noop(self) -> None:
        hooks = SkillHooks([])
        items: list[Any] = [{"role": "user", "content": "unchanged"}]

        await fire_llm_start(hooks, items)

        assert items == [{"role": "user", "content": "unchanged"}]

    async def test_empty_input_items_receives_blocks(self) -> None:
        hooks = SkillHooks([SimpleSkill("s", content="block", always_on=True)])
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        assert items == [{"role": "user", "content": "block"}]

    async def test_original_list_object_is_mutated_not_replaced(self) -> None:
        hooks = SkillHooks([SimpleSkill("s", content="injected", always_on=True)])
        items: list[Any] = []
        original_id = id(items)

        await fire_llm_start(hooks, items)

        assert id(items) == original_id
        assert len(items) == 1


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestSkillHooksOrdering:
    async def test_multiple_skills_inject_in_registration_order(self) -> None:
        hooks = SkillHooks(
            [
                SimpleSkill("skill_first", content="first", always_on=True),
                SimpleSkill("skill_second", content="second", always_on=True),
                SimpleSkill("skill_third", content="third", always_on=True),
            ]
        )
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        contents = [item["content"] for item in items]
        assert contents == ["first", "second", "third"]

    async def test_skill_blocks_precede_original_items_after_multi_skill_injection(
        self,
    ) -> None:
        hooks = SkillHooks(
            [
                SimpleSkill("skill_a", content="a", always_on=True),
                SimpleSkill("skill_b", content="b", always_on=True),
            ]
        )
        items: list[Any] = [{"role": "user", "content": "original"}]

        await fire_llm_start(hooks, items)

        assert items[0]["content"] == "a"
        assert items[1]["content"] == "b"
        assert items[2]["content"] == "original"


# ---------------------------------------------------------------------------
# Multi-block skills
# ---------------------------------------------------------------------------


class TestMultiBlockSkills:
    async def test_multiple_blocks_from_one_skill_are_all_prepended(self) -> None:
        hooks = SkillHooks([_MultiBlockSkill("m", block_a="block one", block_b="block two")])
        items: list[Any] = [{"role": "user", "content": "original"}]

        await fire_llm_start(hooks, items)

        assert items[0]["content"] == "block one"
        assert items[1]["content"] == "block two"
        assert items[2]["content"] == "original"

    async def test_multi_block_skill_followed_by_single_block_skill(self) -> None:
        hooks = SkillHooks(
            [
                _MultiBlockSkill("m", block_a="block one", block_b="block two"),
                SimpleSkill("s", content="single", always_on=True),
            ]
        )
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        assert items[0]["content"] == "block one"
        assert items[1]["content"] == "block two"
        assert items[2]["content"] == "single"


# ---------------------------------------------------------------------------
# Disabled skills
# ---------------------------------------------------------------------------


class TestDisabledSkills:
    async def test_disabled_skill_is_not_injected(self) -> None:
        hooks = SkillHooks([SimpleSkill("s", content="bad", enabled=False, always_on=True)])
        items: list[Any] = [{"role": "user", "content": "user message"}]

        await fire_llm_start(hooks, items)

        assert len(items) == 1
        assert items[0]["content"] == "user message"

    async def test_enabled_and_disabled_skills_mixed(self) -> None:
        hooks = SkillHooks(
            [
                SimpleSkill("on", content="enabled", always_on=True),
                SimpleSkill("off", content="bad", enabled=False, always_on=True),
            ]
        )
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        assert len(items) == 1
        assert items[0]["content"] == "enabled"

    async def test_all_disabled_leaves_items_unchanged(self) -> None:
        hooks = SkillHooks(
            [
                SimpleSkill("off1", enabled=False, always_on=True),
                SimpleSkill("off2", enabled=False, always_on=True),
            ]
        )
        items: list[Any] = [{"role": "user", "content": "original"}]

        await fire_llm_start(hooks, items)

        assert items == [{"role": "user", "content": "original"}]


# ---------------------------------------------------------------------------
# Multi-turn injection
# ---------------------------------------------------------------------------


class TestMultiTurnInjection:
    async def test_skill_reinjects_after_on_llm_end_clears_guard(self) -> None:
        """Skills re-inject on every turn; on_llm_end resets the per-call guard."""
        hooks = SkillHooks([SimpleSkill("s", content="content", always_on=True)])

        items_turn1: list[Any] = []
        await fire_llm_start(hooks, items_turn1)
        assert items_turn1 == [{"role": "user", "content": "content"}]

        await fire_llm_end(hooks)

        items_turn2: list[Any] = []
        await fire_llm_start(hooks, items_turn2)
        assert items_turn2 == [{"role": "user", "content": "content"}]

    async def test_without_on_llm_end_skill_not_reinjected_in_same_run(self) -> None:
        """Without on_llm_end the per-call guard is not cleared between turns."""
        hooks = SkillHooks([SimpleSkill("s", content="content", always_on=True)])

        items_turn1: list[Any] = []
        await fire_llm_start(hooks, items_turn1)

        # No on_llm_end call — guard is still populated.
        items_turn2: list[Any] = []
        await fire_llm_start(hooks, items_turn2)

        assert items_turn2 == []

    async def test_multiple_turns_each_injects_after_clear(self) -> None:
        """Each turn separated by on_llm_end injects independently."""
        hooks = SkillHooks([SimpleSkill("s", content="block", always_on=True)])

        for _ in range(3):
            items: list[Any] = []
            await fire_llm_start(hooks, items)
            assert items == [{"role": "user", "content": "block"}]
            await fire_llm_end(hooks)


# ---------------------------------------------------------------------------
# _extract_routing_context
# ---------------------------------------------------------------------------


class TestExtractRoutingContext:
    def test_single_user_message_returns_that_message(self) -> None:
        items = [{"role": "user", "content": "hello world"}]

        result = _extract_routing_context(items, turns=1)

        assert result == "hello world"

    def test_multiple_messages_turns_2_returns_last_two(self) -> None:
        items = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "user", "content": "third"},
        ]

        result = _extract_routing_context(items, turns=2)

        assert result == "second | third"

    def test_non_user_items_are_ignored(self) -> None:
        items = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user message"},
            {"role": "assistant", "content": "assistant reply"},
        ]

        result = _extract_routing_context(items, turns=1)

        assert result == "user message"

    def test_turns_none_returns_all_user_messages(self) -> None:
        items = [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
            {"role": "user", "content": "three"},
        ]

        result = _extract_routing_context(items, turns=None)

        assert result == "one | two | three"

    def test_empty_list_returns_empty_string(self) -> None:
        result = _extract_routing_context([])

        assert result == ""

    def test_no_user_messages_returns_empty_string(self) -> None:
        items = [
            {"role": "system", "content": "system only"},
            {"role": "assistant", "content": "assistant only"},
        ]

        result = _extract_routing_context(items)

        assert result == ""

    def test_default_turns_is_one(self) -> None:
        items = [
            {"role": "user", "content": "older"},
            {"role": "user", "content": "newest"},
        ]

        result = _extract_routing_context(items)

        assert result == "newest"

    def test_non_string_content_items_are_skipped(self) -> None:
        items = [
            {"role": "user", "content": ["complex", "content"]},
            {"role": "user", "content": "plain string"},
        ]

        result = _extract_routing_context(items, turns=None)

        assert result == "plain string"

    def test_non_dict_items_are_skipped(self) -> None:
        items: list[Any] = [
            "raw string item",
            {"role": "user", "content": "valid"},
        ]

        result = _extract_routing_context(items)

        assert result == "valid"

    def test_turns_larger_than_history_returns_all(self) -> None:
        items = [
            {"role": "user", "content": "alpha"},
            {"role": "user", "content": "beta"},
        ]

        result = _extract_routing_context(items, turns=10)

        assert result == "alpha | beta"

    def test_messages_joined_with_pipe_separator(self) -> None:
        items = [
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "msg2"},
        ]

        result = _extract_routing_context(items, turns=None)

        assert " | " in result
        assert result == "msg1 | msg2"


# ---------------------------------------------------------------------------
# _deduplicate
# ---------------------------------------------------------------------------


class TestDeduplicate:
    def test_empty_list_returns_empty_list(self) -> None:
        result = _deduplicate([])

        assert result == []

    def test_no_duplicates_preserves_original_order(self) -> None:
        a = SimpleSkill("alpha")
        b = SimpleSkill("beta")
        c = SimpleSkill("gamma")

        result = _deduplicate([a, b, c])

        assert result == [a, b, c]

    def test_duplicate_keeps_last_occurrence(self) -> None:
        a1 = SimpleSkill("alpha", content="a1_content")
        a2 = SimpleSkill("alpha", content="a2_content")
        b = SimpleSkill("beta")

        result = _deduplicate([a1, b, a2])

        assert len(result) == 2
        names = [s.name for s in result]
        assert names == ["beta", "alpha"]
        assert result[1] is a2

    def test_all_duplicates_keeps_last_only(self) -> None:
        a1 = SimpleSkill("alpha", content="first")
        a2 = SimpleSkill("alpha", content="second")
        a3 = SimpleSkill("alpha", content="third")

        result = _deduplicate([a1, a2, a3])

        assert len(result) == 1
        assert result[0] is a3

    def test_duplicates_at_start_keeps_last(self) -> None:
        a1 = SimpleSkill("alpha")
        a2 = SimpleSkill("alpha")
        b = SimpleSkill("beta")

        result = _deduplicate([a1, a2, b])

        assert len(result) == 2
        assert result[0] is a2
        assert result[1] is b

    def test_single_element_list_returned_unchanged(self) -> None:
        skill = SimpleSkill("solo")

        result = _deduplicate([skill])

        assert result == [skill]

    def test_relative_order_of_surviving_items_preserved(self) -> None:
        """Skills that are not deduped away must keep their relative order."""
        skills = [
            SimpleSkill("alpha"),
            SimpleSkill("beta"),
            SimpleSkill("gamma"),
            SimpleSkill("alpha"),  # duplicate; first "alpha" removed
        ]

        result = _deduplicate(skills)

        assert [s.name for s in result] == ["beta", "gamma", "alpha"]


# ---------------------------------------------------------------------------
# SkillHooks with registry (no router)
# ---------------------------------------------------------------------------


class TestSkillHooksWithRegistryNoRouter:
    async def test_always_on_skill_is_injected(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("always", content="always content", always_on=True))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        assert any(item.get("content") == "always content" for item in items)

    async def test_routable_skill_not_injected_without_router(self) -> None:
        """A skill with always_on=False is excluded when no router is configured."""
        registry = SkillRegistry()
        registry.register(SimpleSkill("routed", content="routed content"))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "route this"}]

        await fire_llm_start(hooks, items)

        assert not any(item.get("content") == "routed content" for item in items)

    async def test_routable_skill_not_in_always_on(self) -> None:
        """Only the always-on skill is injected; the routable one is skipped without router."""
        registry = SkillRegistry()
        registry.register(SimpleSkill("always", content="always content", always_on=True))
        registry.register(SimpleSkill("routed", content="routed content"))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "any message"}]

        await fire_llm_start(hooks, items)

        contents = extract_contents(items)
        assert "always content" in contents
        assert "routed content" not in contents

    async def test_disabled_skill_not_injected(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("off", content="bad", enabled=False, always_on=True))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        assert not any(item.get("content") == "bad" for item in items)

    async def test_direct_skills_and_registry_always_on_both_injected(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("reg_skill", content="reg content", always_on=True))

        hooks = SkillHooks(
            skills=[SimpleSkill("direct", content="direct content", always_on=True)],
            registry=registry,
        )
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        contents = extract_contents(items)
        assert "direct content" in contents
        assert "reg content" in contents


# ---------------------------------------------------------------------------
# SkillHooks with registry and mock router
# ---------------------------------------------------------------------------


class TestSkillHooksWithMockRouter:
    async def test_routed_skill_is_injected_when_router_selects_it(self) -> None:
        router = MockRouter(names=["routed"])
        registry = SkillRegistry(router=router)
        registry.register(SimpleSkill("routed", content="routed content"))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "route this"}]

        await fire_llm_start(hooks, items)

        assert any(item.get("content") == "routed content" for item in items)

    async def test_both_always_on_and_routed_skills_injected(self) -> None:
        router = MockRouter(names=["routed"])
        registry = SkillRegistry(router=router)
        registry.register(SimpleSkill("always", content="always content", always_on=True))
        registry.register(SimpleSkill("routed", content="routed content"))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "route this"}]

        await fire_llm_start(hooks, items)

        contents = extract_contents(items)
        assert "always content" in contents
        assert "routed content" in contents

    async def test_router_not_called_when_no_user_message_in_items(self) -> None:
        """Routing context is empty when there are no user messages; router stays idle."""
        router = MockRouter(names=["routed"])
        registry = SkillRegistry(router=router)
        registry.register(SimpleSkill("routed", content="rc"))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        assert len(router.calls) == 0

    async def test_unselected_routable_skill_not_injected(self) -> None:
        """A routable skill the router did not select must be skipped."""
        router = MockRouter(names=[])
        registry = SkillRegistry(router=router)
        registry.register(SimpleSkill("routed", content="rc"))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "test"}]

        await fire_llm_start(hooks, items)

        assert not any(item.get("content") == "rc" for item in items)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestSkillHooksDeduplication:
    async def test_skill_in_direct_and_routed_injected_only_once(self) -> None:
        """A skill passed as both a direct skill and selected by the router injects once."""
        router = MockRouter(names=["shared"])
        registry = SkillRegistry(router=router)
        shared = SimpleSkill("shared", content="shared content")
        registry.register(shared)

        hooks = SkillHooks(skills=[shared], registry=registry)
        items: list[Any] = [{"role": "user", "content": "share this"}]

        await fire_llm_start(hooks, items)

        shared_blocks = [item for item in items if item.get("content") == "shared content"]
        assert len(shared_blocks) == 1

    async def test_registry_skill_wins_over_direct_skill_on_name_conflict(self) -> None:
        """When direct and registry skills share a name, the registry instance wins."""
        direct_instance = SimpleSkill("alpha", content="direct content", always_on=True)
        registry_instance = SimpleSkill("alpha", content="registry content", always_on=True)

        registry = SkillRegistry()
        registry.register(registry_instance)

        hooks = SkillHooks(skills=[direct_instance], registry=registry)
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        contents = extract_contents(items)
        assert "registry content" in contents
        assert "direct content" not in contents

    async def test_duplicate_direct_skills_injected_only_once(self) -> None:
        skill = SimpleSkill("dup", content="dup content", always_on=True)
        hooks = SkillHooks(skills=[skill, skill])
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        dup_blocks = [item for item in items if item.get("content") == "dup content"]
        assert len(dup_blocks) == 1


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:
    async def test_error_in_one_skill_does_not_prevent_others(self) -> None:
        """A ValueError from get_prompt_blocks must not abort injection of other skills."""
        errors: list[tuple[Skill, Exception]] = []
        error_skill = _ErrorSkill("broken")
        good_skill = SimpleSkill("good", content="good content", always_on=True)

        hooks = SkillHooks(
            skills=[error_skill, good_skill],
            on_skill_error=lambda s, e: errors.append((s, e)),
        )
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        assert any(item.get("content") == "good content" for item in items)
        assert len(errors) == 1
        assert errors[0][0] is error_skill
        assert isinstance(errors[0][1], ValueError)

    async def test_all_error_skills_leaves_items_unchanged_except_user_msg(self) -> None:
        hooks = SkillHooks(skills=[_ErrorSkill("e1"), _ErrorSkill("e2")])
        original = {"role": "user", "content": "original"}
        items: list[Any] = [original]

        await fire_llm_start(hooks, items)

        assert items == [original]

    async def test_on_skill_error_callback_receives_skill_and_exception(self) -> None:
        received: list[tuple[Skill, Exception]] = []
        error_skill = _ErrorSkill("my_broken_skill")

        hooks = SkillHooks(
            skills=[error_skill],
            on_skill_error=lambda s, e: received.append((s, e)),
        )

        await fire_llm_start(hooks, [])

        assert len(received) == 1
        skill_arg, exc_arg = received[0]
        assert skill_arg is error_skill
        assert isinstance(exc_arg, ValueError)
        assert "my_broken_skill" in str(exc_arg)

    async def test_default_on_skill_error_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Default handler must emit a WARNING containing the skill name."""
        hooks = SkillHooks(skills=[_ErrorSkill("warn_skill")])

        with caplog.at_level(logging.WARNING, logger="openai_agents_skills.hooks"):
            await fire_llm_start(hooks, [])

        assert any("warn_skill" in r.getMessage() for r in caplog.records)
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    async def test_silent_error_handler_suppresses_all_logging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        hooks = SkillHooks(
            skills=[_ErrorSkill("quiet")],
            on_skill_error=lambda s, e: None,
        )

        with caplog.at_level(logging.WARNING, logger="openai_agents_skills.hooks"):
            await fire_llm_start(hooks, [])

        hook_warnings = [
            r
            for r in caplog.records
            if r.name == "openai_agents_skills.hooks" and r.levelno >= logging.WARNING
        ]
        assert hook_warnings == []


# ---------------------------------------------------------------------------
# Manifest injection
# ---------------------------------------------------------------------------


class TestManifestInjection:
    async def test_manifest_injected_on_first_llm_call(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("myskill", always_on=True))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        manifest_items = [
            item
            for item in items
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        ]
        assert len(manifest_items) == 1

    async def test_manifest_not_reinjected_on_second_call(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("myskill", content="skill content", always_on=True))

        hooks = SkillHooks(registry=registry)
        items_first: list[Any] = []
        items_second: list[Any] = []

        await fire_llm_start(hooks, items_first)
        await fire_llm_end(hooks)
        await fire_llm_start(hooks, items_second)

        manifest_count_second = sum(
            1
            for item in items_second
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        )
        assert manifest_count_second == 0

        skill_count_second = sum(
            1 for item in items_second if item.get("content") == "skill content"
        )
        assert skill_count_second == 1

    async def test_manifest_contains_registered_skill_names(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("skill_alpha", always_on=True))
        registry.register(SimpleSkill("skill_beta", always_on=True))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        manifest_text = next(
            item["content"]
            for item in items
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        )
        assert "skill_alpha" in manifest_text
        assert "skill_beta" in manifest_text

    async def test_manifest_not_injected_without_registry(self) -> None:
        hooks = SkillHooks(skills=[SimpleSkill("standalone", always_on=True)])
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        manifest_items = [
            item
            for item in items
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        ]
        assert manifest_items == []

    async def test_max_manifest_skills_caps_entries(self) -> None:
        registry = SkillRegistry()
        for i in range(5):
            registry.register(SimpleSkill(f"skill_{i}", always_on=True))

        hooks = SkillHooks(registry=registry, max_manifest_skills=2)
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        manifest_text = next(
            item["content"]
            for item in items
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        )
        skill_lines = [line for line in manifest_text.split("\n") if line.startswith("- ")]
        assert len(skill_lines) == 2

    async def test_manifest_injected_flag_persists_across_calls(self) -> None:
        """RunState.manifest_injected is True after the first call and never reset."""
        registry = SkillRegistry()
        registry.register(SimpleSkill("skill", always_on=True))

        hooks = SkillHooks(registry=registry)
        await fire_llm_start(hooks, [])

        state = _get_run_state()
        assert state.manifest_injected is True

    async def test_manifest_prepended_before_skill_blocks(self) -> None:
        """The manifest block must appear before any skill prompt blocks."""
        registry = SkillRegistry()
        registry.register(SimpleSkill("myskill", content="skill block", always_on=True))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        manifest_idx = next(
            i
            for i, item in enumerate(items)
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        )
        skill_idx = next(i for i, item in enumerate(items) if item.get("content") == "skill block")
        assert manifest_idx < skill_idx

    async def test_empty_registry_produces_no_manifest_block(self) -> None:
        """If the registry has no skills, no manifest block is prepended."""
        registry = SkillRegistry()
        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        assert items == []


# ---------------------------------------------------------------------------
# RunSkillHooks
# ---------------------------------------------------------------------------


class TestRunSkillHooks:
    async def test_injects_direct_skill(self) -> None:
        hooks = RunSkillHooks(skills=[SimpleSkill("s", content="run content", always_on=True)])
        items: list[Any] = [{"role": "user", "content": "user message"}]

        await fire_llm_start(hooks, items)

        assert items[0]["content"] == "run content"
        assert items[1]["content"] == "user message"

    async def test_injects_always_on_registry_skill(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("reg_skill", content="reg content", always_on=True))

        hooks = RunSkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "test"}]

        await fire_llm_start(hooks, items)

        assert any(item.get("content") == "reg content" for item in items)

    async def test_routable_skill_injected_with_router(self) -> None:
        router = MockRouter(names=["routed"])
        registry = SkillRegistry(router=router)
        registry.register(SimpleSkill("routed", content="routed content"))

        hooks = RunSkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "route this"}]

        await fire_llm_start(hooks, items)

        assert any(item.get("content") == "routed content" for item in items)

    async def test_injects_manifest_on_first_call(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("mskill", always_on=True))

        hooks = RunSkillHooks(registry=registry)
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        manifest_items = [
            item
            for item in items
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        ]
        assert len(manifest_items) == 1

    async def test_disabled_skill_not_injected(self) -> None:
        hooks = RunSkillHooks(
            skills=[SimpleSkill("off", content="bad", enabled=False, always_on=True)]
        )
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        assert items == []

    async def test_error_skill_does_not_abort_other_skills(self) -> None:
        good = SimpleSkill("good", content="good content", always_on=True)
        hooks = RunSkillHooks(
            skills=[_ErrorSkill("bad"), good],
            on_skill_error=lambda s, e: None,
        )
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        assert any(item.get("content") == "good content" for item in items)

    async def test_on_agent_start_initialises_run_state(self) -> None:
        """on_agent_start must prime the RunState so the guard works correctly."""
        hooks = RunSkillHooks(skills=[SimpleSkill("s", always_on=True)])

        await hooks.on_agent_start(
            context=None,  # type: ignore[arg-type]
            agent=None,  # type: ignore[arg-type]
        )

        state = _run_state.get()
        assert isinstance(state, RunState)


# ---------------------------------------------------------------------------
# Double-injection guard
# ---------------------------------------------------------------------------


class TestDoubleInjectionGuard:
    async def test_same_skill_in_run_and_agent_hooks_injected_only_once(self) -> None:
        """Simulates RunSkillHooks and SkillHooks both firing for the same LLM call."""
        skill = SimpleSkill("shared", content="shared content", always_on=True)

        run_hooks = RunSkillHooks(skills=[skill])
        agent_hooks = SkillHooks(skills=[skill])

        _get_run_state()

        items: list[Any] = [{"role": "user", "content": "original"}]

        await fire_llm_start(run_hooks, items)
        await fire_llm_start(agent_hooks, items)

        shared_blocks = [item for item in items if item.get("content") == "shared content"]
        assert len(shared_blocks) == 1

    async def test_injected_this_call_set_populated_during_injection(self) -> None:
        skill = SimpleSkill("tracked", content="tracked", always_on=True)

        hooks = SkillHooks(skills=[skill])
        await fire_llm_start(hooks, [])

        state = _get_run_state()
        assert "tracked" in state.injected_this_call

    async def test_skill_skipped_if_already_in_injected_this_call_guard(self) -> None:
        """Manually pre-populating injected_this_call prevents re-injection."""
        skill = SimpleSkill("pre_seen", content="pre_seen content", always_on=True)

        state = _get_run_state()
        state.injected_this_call.add("pre_seen")

        hooks = SkillHooks(skills=[skill])
        items: list[Any] = []

        await fire_llm_start(hooks, items)

        assert items == []

    async def test_different_skills_in_run_and_agent_hooks_both_injected(self) -> None:
        """Distinct skills in each hook must both be injected (no false-positive guard)."""
        run_skill = SimpleSkill("run_only", content="run content", always_on=True)
        agent_skill = SimpleSkill("agent_only", content="agent content", always_on=True)

        run_hooks = RunSkillHooks(skills=[run_skill])
        agent_hooks = SkillHooks(skills=[agent_skill])

        _get_run_state()

        items: list[Any] = []

        await fire_llm_start(run_hooks, items)
        await fire_llm_start(agent_hooks, items)

        contents = extract_contents(items)
        assert "run content" in contents
        assert "agent content" in contents

    async def test_concurrent_gather_injects_shared_skill_only_once(self) -> None:
        """When RunSkillHooks and SkillHooks fire concurrently via asyncio.gather,
        a skill registered in both injects exactly once."""
        skill = SimpleSkill("shared", content="shared content", always_on=True)
        run_hooks = RunSkillHooks(skills=[skill])
        agent_hooks = SkillHooks(skills=[skill])

        _get_run_state()

        items: list[Any] = [{"role": "user", "content": "question"}]

        async def fire_run() -> None:
            await fire_llm_start(run_hooks, items)

        async def fire_agent() -> None:
            await fire_llm_start(agent_hooks, items)

        await asyncio.gather(fire_run(), fire_agent())

        shared_blocks = [item for item in items if item.get("content") == "shared content"]
        assert len(shared_blocks) == 1


# ---------------------------------------------------------------------------
# make_invoke_skill_tool
# ---------------------------------------------------------------------------


class TestInvokeSkillTool:
    async def test_returns_content_for_known_skill(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("myskill", content="skill content"))
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "myskill")

        assert result == "skill content"

    async def test_returns_error_string_for_unknown_skill(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("known"))
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "unknown_skill")

        assert "Unknown skill" in result
        assert "unknown_skill" in result

    async def test_error_message_includes_available_skills(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("alpha"))
        registry.register(SimpleSkill("beta"))
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "ghost")

        assert "alpha" in result
        assert "beta" in result

    async def test_content_concatenated_from_multiple_blocks(self) -> None:
        registry = SkillRegistry()
        registry.register(_MultiBlockSkill("multi", block_a="first block", block_b="second block"))
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "multi")

        assert "first block" in result
        assert "second block" in result
        assert result.index("first block") < result.index("second block")

    async def test_args_passed_through_to_skill(self) -> None:
        class _EchoSkill(Skill):
            name = "echo"
            description = "Echoes args."

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                return [{"role": "user", "content": f"args={args}"}]

        registry = SkillRegistry()
        registry.register(_EchoSkill())
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "echo", args="hello world")

        assert "args=hello world" in result

    async def test_max_calls_per_run_guard_enforced(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("s", content="ok"))
        tool = make_invoke_skill_tool(registry, max_calls_per_run=2)

        r1 = await _invoke(tool, "s")
        r2 = await _invoke(tool, "s")
        r3 = await _invoke(tool, "s")

        assert r1 == "ok"
        assert r2 == "ok"
        assert "limit reached" in r3

    async def test_max_calls_zero_disables_guard(self) -> None:
        """max_calls_per_run=0 means the guard is disabled entirely."""
        registry = SkillRegistry()
        registry.register(SimpleSkill("s", content="ok"))
        tool = make_invoke_skill_tool(registry, max_calls_per_run=0)

        for _ in range(20):
            result = await _invoke(tool, "s")
            assert result == "ok"

    async def test_invoke_skill_counter_increments_per_call(self) -> None:
        registry = SkillRegistry()
        registry.register(SimpleSkill("s", content="ok"))
        tool = make_invoke_skill_tool(registry, max_calls_per_run=10)

        await _invoke(tool, "s")
        await _invoke(tool, "s")
        await _invoke(tool, "s")

        state = _get_run_state()
        assert state.invoke_skill_calls == 3

    async def test_invoke_skill_empty_registry_returns_error(self) -> None:
        registry = SkillRegistry()
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "anything")

        assert "Unknown skill" in result

    async def test_invoke_skill_empty_registry_shows_none_available(self) -> None:
        registry = SkillRegistry()
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "missing")

        assert "(none)" in result


# ---------------------------------------------------------------------------
# invoke_skill content delivery
#
# Documented claim: "invoke_skill returns the skill's prompt content as a tool
# result string. The model receives the guidance and the diagnostic evidence
# together in the same LLM call and reasons about them simultaneously — no
# extra round-trip is needed."
# ---------------------------------------------------------------------------


class TestInvokeSkillContentDelivery:
    """invoke_skill delivers skill content as a plain string in the same turn."""

    async def test_invoke_skill_returns_string_not_prompt_blocks(self) -> None:
        """The return value is a str, not a list of blocks — it arrives as a tool
        result in the current turn, not as a prepended instruction block."""
        registry = SkillRegistry()
        registry.register(SimpleSkill("guidance", content="remediation steps"))
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "guidance")

        assert isinstance(result, str)
        assert "remediation steps" in result

    async def test_invoke_skill_result_and_separate_tool_result_both_present_before_llm_call(
        self,
    ) -> None:
        """After invoke_skill is called and a diagnostic tool has also run, both
        results are present in input_items before on_llm_start fires — confirming
        no extra round-trip is needed for the model to see both together."""
        registry = SkillRegistry()
        registry.register(SimpleSkill("nexthop-unresolvable", content="NEXTHOP: check route table"))
        hooks = SkillHooks(registry=registry)
        tool = make_invoke_skill_tool(registry)

        items: list[Any] = [{"role": "user", "content": "Diagnose the failure"}]
        await hooks.on_llm_start(None, None, None, items)  # type: ignore[arg-type]

        skill_content = await _invoke(tool, "nexthop-unresolvable")
        assert "NEXTHOP" in skill_content

        diagnostic_result = '{"root_cause": "NEXTHOP_UNRESOLVABLE", "summary": "route missing"}'
        items.append({"role": "tool", "content": skill_content})
        items.append({"role": "tool", "content": diagnostic_result})

        await hooks.on_llm_end(None, None, None)  # type: ignore[arg-type]
        await hooks.on_llm_start(None, None, None, items)  # type: ignore[arg-type]

        all_content = " ".join(
            i["content"] for i in items if isinstance(i, dict) and "content" in i
        )
        assert "NEXTHOP" in all_content, "Skill guidance must be in items before the LLM call"
        assert "NEXTHOP_UNRESOLVABLE" in all_content, (
            "Diagnostic evidence must be in items before the LLM call"
        )

    async def test_unknown_skill_name_returns_descriptive_error_not_exception(self) -> None:
        """A bad skill name returns a descriptive error string so the model can
        self-correct; the agent loop must not crash."""
        registry = SkillRegistry()
        registry.register(SimpleSkill("real-skill"))
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "nonexistent")

        assert "Unknown skill" in result
        assert "nonexistent" in result
        assert "real-skill" in result


# ---------------------------------------------------------------------------
# Router and invoke_skill coexistence
#
# Documented claim: "Both mechanisms can coexist in the same agent. A common
# pattern is to use the router for broad topic skills and invoke_skill for
# narrow use-case skills whose activation depends on what a tool returned."
# ---------------------------------------------------------------------------


class TestRouterAndInvokeSkillCoexist:
    """Router-injected skills and invoke_skill-fetched skills do not interfere."""

    async def test_router_injects_broad_skill_while_invoke_skill_fetches_narrow_skill(
        self,
    ) -> None:
        """The router selects and injects a broad topic skill via on_llm_start.
        invoke_skill independently fetches a narrow use-case skill by name.
        Both sets of content are present without conflict."""
        router = MockRouter(names=["bgp-troubleshooting"])
        registry = SkillRegistry(router=router)
        registry.register(SimpleSkill("bgp-troubleshooting", content="BGP CHECKLIST"))
        registry.register(SimpleSkill("nexthop-unresolvable", content="NEXTHOP REMEDIATION"))

        hooks = SkillHooks(registry=registry)
        tool = make_invoke_skill_tool(registry)

        items: list[Any] = [{"role": "user", "content": "BGP session is flapping"}]
        await fire_llm_start(hooks, items)

        injected = [i["content"] for i in items if isinstance(i, dict)]
        assert "BGP CHECKLIST" in injected

        narrow_result = await _invoke(tool, "nexthop-unresolvable")
        assert "NEXTHOP REMEDIATION" in narrow_result

        items.append({"role": "tool", "content": narrow_result})

        all_content = " ".join(i["content"] for i in items if isinstance(i, dict))
        assert "BGP CHECKLIST" in all_content
        assert "NEXTHOP REMEDIATION" in all_content

    async def test_invoke_skill_does_not_affect_router_selection_on_next_turn(self) -> None:
        """Calling invoke_skill does not alter which skills the router is offered
        on the next on_llm_start — the two mechanisms are fully independent."""
        router = MockRouter(names=["broad-skill"])
        registry = SkillRegistry(router=router)
        registry.register(SimpleSkill("broad-skill"))
        registry.register(SimpleSkill("narrow-skill"))

        hooks = SkillHooks(registry=registry)
        tool = make_invoke_skill_tool(registry)

        items: list[Any] = [{"role": "user", "content": "first question"}]
        await fire_llm_start(hooks, items)

        await _invoke(tool, "narrow-skill")

        await hooks.on_llm_end(None, None, None)  # type: ignore[arg-type]

        items2: list[Any] = [{"role": "user", "content": "follow-up question"}]
        await fire_llm_start(hooks, items2)

        assert len(router.calls) == 2
        second_turn_skill_names = router.calls[1][1]
        assert "broad-skill" in second_turn_skill_names
        assert "narrow-skill" in second_turn_skill_names

    async def test_router_selected_and_invoked_skill_both_appear_in_items(self) -> None:
        """Neither router-injected content nor invoke_skill content crowds out the
        other in input_items — both reach the model in the same call."""
        router = MockRouter(names=["topic-skill"])
        registry = SkillRegistry(router=router)
        registry.register(SimpleSkill("topic-skill", content="TOPIC GUIDANCE"))
        registry.register(SimpleSkill("usecase-skill", content="USECASE GUIDANCE"))

        hooks = SkillHooks(registry=registry)
        tool = make_invoke_skill_tool(registry)

        items: list[Any] = [{"role": "user", "content": "user query"}]
        await fire_llm_start(hooks, items)

        invoked = await _invoke(tool, "usecase-skill")
        items.append({"role": "tool", "content": invoked})

        all_content = " ".join(i["content"] for i in items if isinstance(i, dict))
        assert "TOPIC GUIDANCE" in all_content
        assert "USECASE GUIDANCE" in all_content
