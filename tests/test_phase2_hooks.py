"""Tests for Phase 2 hook functions.

Covers: _extract_routing_context, _deduplicate, SkillHooks (registry-backed),
RunSkillHooks, double-injection guard, and make_invoke_skill_tool.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from agents.tool_context import ToolContext
from conftest import MockRouter

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


class _SimpleSkill(Skill):
    """Configurable skill used throughout Phase 2 tests."""

    def __init__(
        self,
        name: str,
        content: str = "injected",
        when_to_use: str = "",
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.description = f"Skill {name}"
        self.when_to_use = when_to_use
        self._content = content
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    async def get_prompt_blocks(self, args: str = "") -> list[Any]:
        return [{"role": "user", "content": self._content}]


class _MultiBlockSkill(Skill):
    """Skill that returns two blocks per call."""

    def __init__(
        self, name: str = "multi", block_a: str = "block_a", block_b: str = "block_b"
    ) -> None:
        self.name = name
        self.description = f"Multi-block skill {name}"
        self.when_to_use = ""
        self._block_a = block_a
        self._block_b = block_b

    async def get_prompt_blocks(self, args: str = "") -> list[Any]:
        return [
            {"role": "user", "content": self._block_a},
            {"role": "user", "content": self._block_b},
        ]


class _ErrorSkill(Skill):
    """Skill that raises ValueError from get_prompt_blocks."""

    def __init__(self, name: str = "error_skill") -> None:
        self.name = name
        self.description = "Always raises."
        self.when_to_use = ""

    async def get_prompt_blocks(self, args: str = "") -> list[Any]:
        raise ValueError(f"Skill {self.name!r} intentionally failed")


async def _fire(
    hooks: SkillHooks | RunSkillHooks,
    input_items: list[Any],
) -> None:
    """Call on_llm_start with minimal SDK arguments."""
    await hooks.on_llm_start(
        context=None,  # type: ignore[arg-type]
        agent=None,  # type: ignore[arg-type]
        system_prompt=None,
        input_items=input_items,
    )


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


def _injected_contents(items: list[Any]) -> list[str]:
    """Extract the content strings from all user-role items in the list."""
    return [item["content"] for item in items if isinstance(item, dict) and "content" in item]


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
        a = _SimpleSkill("alpha")
        b = _SimpleSkill("beta")
        c = _SimpleSkill("gamma")

        result = _deduplicate([a, b, c])

        assert result == [a, b, c]

    def test_duplicate_keeps_last_occurrence(self) -> None:
        a1 = _SimpleSkill("alpha", content="a1_content")
        a2 = _SimpleSkill("alpha", content="a2_content")
        b = _SimpleSkill("beta")

        # a1 and a2 have the same name; a2 (last) should survive.
        result = _deduplicate([a1, b, a2])

        assert len(result) == 2
        names = [s.name for s in result]
        assert names == ["beta", "alpha"]
        assert result[1] is a2

    def test_all_duplicates_keeps_last_only(self) -> None:
        a1 = _SimpleSkill("alpha", content="first")
        a2 = _SimpleSkill("alpha", content="second")
        a3 = _SimpleSkill("alpha", content="third")

        result = _deduplicate([a1, a2, a3])

        assert len(result) == 1
        assert result[0] is a3

    def test_duplicates_at_start_keeps_last(self) -> None:
        a1 = _SimpleSkill("alpha")
        a2 = _SimpleSkill("alpha")
        b = _SimpleSkill("beta")

        result = _deduplicate([a1, a2, b])

        # a2 (last "alpha") survives; relative order: a2 before b.
        assert len(result) == 2
        assert result[0] is a2
        assert result[1] is b

    def test_single_element_list_returned_unchanged(self) -> None:
        skill = _SimpleSkill("solo")

        result = _deduplicate([skill])

        assert result == [skill]

    def test_relative_order_of_surviving_items_preserved(self) -> None:
        """Skills that are not deduped away must keep their relative order."""
        skills = [
            _SimpleSkill("alpha"),
            _SimpleSkill("beta"),
            _SimpleSkill("gamma"),
            _SimpleSkill("alpha"),  # duplicate; first "alpha" removed
        ]

        result = _deduplicate(skills)

        # Survivors are the last "alpha" (index 3) and beta, gamma (indices 1, 2).
        # Relative order after dedup: [beta, gamma, alpha_last].
        assert [s.name for s in result] == ["beta", "gamma", "alpha"]


# ---------------------------------------------------------------------------
# SkillHooks with registry (no router)
# ---------------------------------------------------------------------------


class TestSkillHooksWithRegistryNoRouter:
    async def test_always_on_skill_is_injected(self) -> None:
        registry = SkillRegistry()
        skill = _SimpleSkill("always", content="always content")
        registry.register(skill)

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await _fire(hooks, items)

        assert any(item.get("content") == "always content" for item in items)

    async def test_routable_skill_not_injected_without_router(self) -> None:
        """A skill with non-empty when_to_use is excluded when no router is configured."""
        registry = SkillRegistry()
        skill = _SimpleSkill("routed", content="routed content", when_to_use="Use for routing.")
        registry.register(skill)

        hooks = SkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "route this"}]

        await _fire(hooks, items)

        assert not any(item.get("content") == "routed content" for item in items)

    async def test_skill_with_non_empty_when_to_use_not_in_always_on(self) -> None:
        """Only the always-on skill is injected; the routable one is skipped without router."""
        registry = SkillRegistry()
        always = _SimpleSkill("always", content="always content")
        routed = _SimpleSkill("routed", content="routed content", when_to_use="Use when X.")
        registry.register(always)
        registry.register(routed)

        hooks = SkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "any message"}]

        await _fire(hooks, items)

        contents = _injected_contents(items)
        assert "always content" in contents
        assert "routed content" not in contents

    async def test_disabled_skill_not_injected(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("disabled", content="bad", enabled=False))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await _fire(hooks, items)

        # The disabled skill's own content block must NOT be injected.
        # The manifest block may still appear (it lists all registered skills).
        assert not any(item.get("content") == "bad" for item in items)

    async def test_direct_skills_and_registry_always_on_both_injected(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("reg_skill", content="reg content"))

        hooks = SkillHooks(
            skills=[_SimpleSkill("direct", content="direct content")],
            registry=registry,
        )
        items: list[Any] = []

        await _fire(hooks, items)

        contents = _injected_contents(items)
        assert "direct content" in contents
        assert "reg content" in contents


# ---------------------------------------------------------------------------
# SkillHooks with registry and mock router
# ---------------------------------------------------------------------------


class TestSkillHooksWithMockRouter:
    async def test_routed_skill_is_injected_when_router_selects_it(self) -> None:
        router = MockRouter(names=["routed"])
        registry = SkillRegistry(router=router)
        registry.register(
            _SimpleSkill("routed", content="routed content", when_to_use="Use for routing.")
        )

        hooks = SkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "route this"}]

        await _fire(hooks, items)

        assert any(item.get("content") == "routed content" for item in items)

    async def test_both_always_on_and_routed_skills_injected(self) -> None:
        router = MockRouter(names=["routed"])
        registry = SkillRegistry(router=router)
        registry.register(_SimpleSkill("always", content="always content"))
        registry.register(
            _SimpleSkill("routed", content="routed content", when_to_use="Use for routing.")
        )

        hooks = SkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "route this"}]

        await _fire(hooks, items)

        contents = _injected_contents(items)
        assert "always content" in contents
        assert "routed content" in contents

    async def test_router_not_called_when_no_user_message_in_items(self) -> None:
        """Routing context is empty when there are no user messages; router stays idle."""
        router = MockRouter(names=["routed"])
        registry = SkillRegistry(router=router)
        registry.register(_SimpleSkill("routed", content="rc", when_to_use="When X."))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []  # no user message → empty routing context

        await _fire(hooks, items)

        assert len(router.calls) == 0

    async def test_unselected_routable_skill_not_injected(self) -> None:
        """A routable skill the router did not select must be skipped."""
        router = MockRouter(names=[])  # selects nothing
        registry = SkillRegistry(router=router)
        registry.register(_SimpleSkill("routed", content="rc", when_to_use="When X."))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "test"}]

        await _fire(hooks, items)

        # Only the user message itself should remain; no injected blocks.
        assert not any(item.get("content") == "rc" for item in items)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestSkillHooksDeduplication:
    async def test_skill_in_direct_and_routed_injected_only_once(self) -> None:
        """A skill passed as both a direct skill and selected by the router injects once."""
        router = MockRouter(names=["shared"])
        registry = SkillRegistry(router=router)
        shared = _SimpleSkill("shared", content="shared content", when_to_use="Use when sharing.")
        registry.register(shared)

        hooks = SkillHooks(skills=[shared], registry=registry)
        items: list[Any] = [{"role": "user", "content": "share this"}]

        await _fire(hooks, items)

        shared_blocks = [item for item in items if item.get("content") == "shared content"]
        assert len(shared_blocks) == 1

    async def test_registry_skill_wins_over_direct_skill_on_name_conflict(self) -> None:
        """When direct and registry skills share a name, the registry instance wins."""
        direct_instance = _SimpleSkill("alpha", content="direct content")
        registry_instance = _SimpleSkill("alpha", content="registry content")

        registry = SkillRegistry()
        registry.register(registry_instance)

        hooks = SkillHooks(skills=[direct_instance], registry=registry)
        items: list[Any] = []

        await _fire(hooks, items)

        contents = _injected_contents(items)
        # Registry skill (last in dedup input) wins.
        assert "registry content" in contents
        assert "direct content" not in contents

    async def test_duplicate_direct_skills_injected_only_once(self) -> None:
        skill = _SimpleSkill("dup", content="dup content")
        # Same instance listed twice.
        hooks = SkillHooks(skills=[skill, skill])
        items: list[Any] = []

        await _fire(hooks, items)

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
        good_skill = _SimpleSkill("good", content="good content")

        hooks = SkillHooks(
            skills=[error_skill, good_skill],
            on_skill_error=lambda s, e: errors.append((s, e)),
        )
        items: list[Any] = []

        await _fire(hooks, items)

        # Good skill's block must still be present.
        assert any(item.get("content") == "good content" for item in items)
        # Error skill was reported.
        assert len(errors) == 1
        assert errors[0][0] is error_skill
        assert isinstance(errors[0][1], ValueError)

    async def test_all_error_skills_leaves_items_unchanged_except_user_msg(self) -> None:
        hooks = SkillHooks(skills=[_ErrorSkill("e1"), _ErrorSkill("e2")])
        original = {"role": "user", "content": "original"}
        items: list[Any] = [original]

        await _fire(hooks, items)

        # Original item untouched; no skill blocks injected.
        assert items == [original]

    async def test_on_skill_error_callback_receives_skill_and_exception(self) -> None:
        received: list[tuple[Skill, Exception]] = []
        error_skill = _ErrorSkill("my_broken_skill")

        hooks = SkillHooks(
            skills=[error_skill],
            on_skill_error=lambda s, e: received.append((s, e)),
        )

        await _fire(hooks, [])

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
            await _fire(hooks, [])

        assert any("warn_skill" in r.getMessage() for r in caplog.records)
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    async def test_silent_error_handler_suppresses_all_logging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        hooks = SkillHooks(
            skills=[_ErrorSkill("quiet")],
            on_skill_error=lambda s, e: None,  # silent
        )

        with caplog.at_level(logging.WARNING, logger="openai_agents_skills.hooks"):
            await _fire(hooks, [])

        # No WARNING records from the hooks module.
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
        registry.register(_SimpleSkill("myskill"))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await _fire(hooks, items)

        manifest_items = [
            item
            for item in items
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        ]
        assert len(manifest_items) == 1

    async def test_manifest_not_reinjected_on_second_call(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("myskill", content="skill content"))

        hooks = SkillHooks(registry=registry)
        items_first: list[Any] = []
        items_second: list[Any] = []

        await _fire(hooks, items_first)

        # on_llm_end clears the per-call guard so skills can re-inject.
        await hooks.on_llm_end(
            context=None,  # type: ignore[arg-type]
            agent=None,  # type: ignore[arg-type]
            response=None,  # type: ignore[arg-type]
        )

        await _fire(hooks, items_second)

        # Manifest must NOT appear on the second call (manifest_injected persists).
        manifest_count_second = sum(
            1
            for item in items_second
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        )
        assert manifest_count_second == 0

        # Skill content MUST appear on the second call (injection guard was reset).
        skill_count_second = sum(
            1 for item in items_second if item.get("content") == "skill content"
        )
        assert skill_count_second == 1

    async def test_manifest_contains_registered_skill_names(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("skill_alpha"))
        registry.register(_SimpleSkill("skill_beta"))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await _fire(hooks, items)

        manifest_text = next(
            item["content"]
            for item in items
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        )
        assert "skill_alpha" in manifest_text
        assert "skill_beta" in manifest_text

    async def test_manifest_not_injected_without_registry(self) -> None:
        skill = _SimpleSkill("standalone")
        hooks = SkillHooks(skills=[skill])
        items: list[Any] = []

        await _fire(hooks, items)

        manifest_items = [
            item
            for item in items
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        ]
        assert manifest_items == []

    async def test_max_manifest_skills_caps_entries(self) -> None:
        registry = SkillRegistry()
        for i in range(5):
            registry.register(_SimpleSkill(f"skill_{i}"))

        hooks = SkillHooks(registry=registry, max_manifest_skills=2)
        items: list[Any] = []

        await _fire(hooks, items)

        manifest_text = next(
            item["content"]
            for item in items
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        )
        # Bullet lines start with "- ".
        skill_lines = [line for line in manifest_text.split("\n") if line.startswith("- ")]
        assert len(skill_lines) == 2

    async def test_manifest_injected_flag_persists_across_calls(self) -> None:
        """RunState.manifest_injected is True after the first call."""
        registry = SkillRegistry()
        registry.register(_SimpleSkill("skill"))

        hooks = SkillHooks(registry=registry)
        await _fire(hooks, [])

        state = _get_run_state()
        assert state.manifest_injected is True

    async def test_manifest_prepended_before_skill_blocks(self) -> None:
        """The manifest user message must appear before any skill prompt blocks."""
        registry = SkillRegistry()
        registry.register(_SimpleSkill("myskill", content="skill block"))

        hooks = SkillHooks(registry=registry)
        items: list[Any] = []

        await _fire(hooks, items)

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

        await _fire(hooks, items)

        assert items == []


# ---------------------------------------------------------------------------
# RunSkillHooks
# ---------------------------------------------------------------------------


class TestRunSkillHooks:
    async def test_injects_direct_skill(self) -> None:
        skill = _SimpleSkill("run_skill", content="run content")
        hooks = RunSkillHooks(skills=[skill])
        items: list[Any] = [{"role": "user", "content": "user message"}]

        await _fire(hooks, items)

        assert items[0]["content"] == "run content"
        assert items[1]["content"] == "user message"

    async def test_injects_always_on_registry_skill(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("reg_skill", content="reg content"))

        hooks = RunSkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "test"}]

        await _fire(hooks, items)

        assert any(item.get("content") == "reg content" for item in items)

    async def test_routable_skill_injected_with_router(self) -> None:
        router = MockRouter(names=["routed"])
        registry = SkillRegistry(router=router)
        registry.register(
            _SimpleSkill("routed", content="routed content", when_to_use="Use for routing.")
        )

        hooks = RunSkillHooks(registry=registry)
        items: list[Any] = [{"role": "user", "content": "route this"}]

        await _fire(hooks, items)

        assert any(item.get("content") == "routed content" for item in items)

    async def test_injects_manifest_on_first_call(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("mskill"))

        hooks = RunSkillHooks(registry=registry)
        items: list[Any] = []

        await _fire(hooks, items)

        manifest_items = [
            item
            for item in items
            if isinstance(item.get("content"), str) and "Available Skills" in item["content"]
        ]
        assert len(manifest_items) == 1

    async def test_disabled_skill_not_injected(self) -> None:
        hooks = RunSkillHooks(skills=[_SimpleSkill("off", content="bad", enabled=False)])
        items: list[Any] = []

        await _fire(hooks, items)

        assert items == []

    async def test_error_skill_does_not_abort_other_skills(self) -> None:
        good = _SimpleSkill("good", content="good content")
        hooks = RunSkillHooks(
            skills=[_ErrorSkill("bad"), good],
            on_skill_error=lambda s, e: None,
        )
        items: list[Any] = []

        await _fire(hooks, items)

        assert any(item.get("content") == "good content" for item in items)

    async def test_on_agent_start_initialises_run_state(self) -> None:
        """on_agent_start must prime the RunState so the guard works correctly."""
        hooks = RunSkillHooks(skills=[_SimpleSkill("s")])

        await hooks.on_agent_start(
            context=None,  # type: ignore[arg-type]
            agent=None,  # type: ignore[arg-type]
        )

        # After on_agent_start the RunState must already exist (not None).
        state = _run_state.get()
        assert isinstance(state, RunState)


# ---------------------------------------------------------------------------
# Double-injection guard
# ---------------------------------------------------------------------------


class TestDoubleInjectionGuard:
    async def test_same_skill_in_run_and_agent_hooks_injected_only_once(self) -> None:
        """Simulates RunSkillHooks and SkillHooks both firing for the same LLM call."""
        skill = _SimpleSkill("shared", content="shared content")

        run_hooks = RunSkillHooks(skills=[skill])
        agent_hooks = SkillHooks(skills=[skill])

        # Initialise RunState in the current task so both hooks share it.
        _get_run_state()

        items: list[Any] = [{"role": "user", "content": "original"}]

        # Sequential calls share the same RunState (same ContextVar value).
        await _fire(run_hooks, items)
        await _fire(agent_hooks, items)

        shared_blocks = [item for item in items if item.get("content") == "shared content"]
        assert len(shared_blocks) == 1

    async def test_injected_this_call_set_populated_during_injection(self) -> None:
        skill = _SimpleSkill("tracked", content="tracked")

        hooks = SkillHooks(skills=[skill])
        await _fire(hooks, [])

        state = _get_run_state()
        assert "tracked" in state.injected_this_call

    async def test_skill_skipped_if_already_in_injected_this_call_guard(self) -> None:
        """Manually pre-populating injected_this_call prevents re-injection."""
        skill = _SimpleSkill("pre_seen", content="pre_seen content")

        state = _get_run_state()
        state.injected_this_call.add("pre_seen")

        hooks = SkillHooks(skills=[skill])
        items: list[Any] = []

        await _fire(hooks, items)

        # The skill was already marked as injected; it must be skipped.
        assert items == []

    async def test_different_skills_in_run_and_agent_hooks_both_injected(self) -> None:
        """Distinct skills in each hook must both be injected (no false-positive guard)."""
        run_skill = _SimpleSkill("run_only", content="run content")
        agent_skill = _SimpleSkill("agent_only", content="agent content")

        run_hooks = RunSkillHooks(skills=[run_skill])
        agent_hooks = SkillHooks(skills=[agent_skill])

        _get_run_state()

        items: list[Any] = []

        await _fire(run_hooks, items)
        await _fire(agent_hooks, items)

        contents = _injected_contents(items)
        assert "run content" in contents
        assert "agent content" in contents

    async def test_concurrent_gather_injects_shared_skill_only_once(self) -> None:
        """When RunSkillHooks and SkillHooks fire concurrently via asyncio.gather,
        a skill registered in both injects exactly once."""
        import asyncio

        skill = _SimpleSkill("shared", content="shared content")
        run_hooks = RunSkillHooks(skills=[skill])
        agent_hooks = SkillHooks(skills=[skill])

        # Prime RunState in the parent task so both hooks share the same object.
        _get_run_state()

        items: list[Any] = [{"role": "user", "content": "question"}]

        async def fire_run() -> None:
            await _fire(run_hooks, items)

        async def fire_agent() -> None:
            await _fire(agent_hooks, items)

        await asyncio.gather(fire_run(), fire_agent())

        shared_blocks = [item for item in items if item.get("content") == "shared content"]
        assert len(shared_blocks) == 1


# ---------------------------------------------------------------------------
# make_invoke_skill_tool
# ---------------------------------------------------------------------------


class TestInvokeSkillTool:
    async def test_returns_content_for_known_skill(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("myskill", content="skill content"))
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "myskill")

        assert result == "skill content"

    async def test_returns_error_string_for_unknown_skill(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("known"))
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "unknown_skill")

        assert "Unknown skill" in result
        assert "unknown_skill" in result

    async def test_error_message_includes_available_skills(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("alpha"))
        registry.register(_SimpleSkill("beta"))
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
            when_to_use = ""

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return [{"role": "user", "content": f"args={args}"}]

        registry = SkillRegistry()
        registry.register(_EchoSkill())
        tool = make_invoke_skill_tool(registry)

        result = await _invoke(tool, "echo", args="hello world")

        assert "args=hello world" in result

    async def test_max_calls_per_run_guard_enforced(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("s", content="ok"))
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
        registry.register(_SimpleSkill("s", content="ok"))
        tool = make_invoke_skill_tool(registry, max_calls_per_run=0)

        for _ in range(20):
            result = await _invoke(tool, "s")
            assert result == "ok"

    async def test_invoke_skill_counter_increments_per_call(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("s", content="ok"))
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
