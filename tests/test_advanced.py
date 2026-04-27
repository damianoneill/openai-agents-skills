"""Tests for Phase 4 — Advanced Triggering.

Covers: triggers_after_tools, triggers_after_turn, SkillRegistry.get_triggered_by_tool,
SkillRegistry.get_post_turn, on_tool_end queuing, on_llm_end post-turn queuing,
_drain_pending concurrency safety, cross-agent drain, and deduplication.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from openai_agents_skills import Skill, SkillHooks, SkillRegistry
from openai_agents_skills._state import RunState, _get_run_state
from openai_agents_skills.hooks import RunSkillHooks, _drain_pending

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _SimpleSkill(Skill):
    """Configurable skill used throughout Phase 4 tests."""

    def __init__(
        self,
        name: str,
        content: str = "injected",
        always_on: bool = False,
        enabled: bool = True,
        triggers_after_tools: list[str] | None = None,
        triggers_after_turn: bool = False,
    ) -> None:
        self.name = name
        self.description = f"Skill {name}"
        self.always_on = always_on
        self._content = content
        self._enabled = enabled
        self.triggers_after_tools = list(triggers_after_tools or [])
        self.triggers_after_turn = triggers_after_turn

    def is_enabled(self, context=None, agent=None) -> bool:
        return self._enabled

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": self._content}]


def _mock_tool(name: str) -> Any:
    """Return a minimal mock with a .name attribute, matching the SDK Tool interface."""
    tool = MagicMock()
    tool.name = name
    return tool


def _mock_context() -> Any:
    return MagicMock()


def _mock_agent() -> Any:
    return MagicMock()


def _mock_response() -> Any:
    return MagicMock()


def _hooks_with_registry(registry: SkillRegistry) -> SkillHooks:
    """Return a SkillHooks bound to *registry* with a re-raise error handler."""
    return SkillHooks(
        registry=registry,
        on_skill_error=lambda s, e: (_ for _ in ()).throw(e),
    )


def _run_hooks_with_registry(registry: SkillRegistry) -> RunSkillHooks:
    return RunSkillHooks(
        registry=registry,
        on_skill_error=lambda s, e: (_ for _ in ()).throw(e),
    )


# ---------------------------------------------------------------------------
# Skill class attributes — Phase 4 defaults
# ---------------------------------------------------------------------------


class TestSkillPhase4Attributes:
    def test_triggers_after_tools_default_is_empty_list(self) -> None:
        sk = _SimpleSkill("x")
        assert sk.triggers_after_tools == []

    def test_triggers_after_turn_default_is_false(self) -> None:
        sk = _SimpleSkill("x")
        assert sk.triggers_after_turn is False

    def test_triggers_after_tools_can_be_set(self) -> None:
        sk = _SimpleSkill("x", triggers_after_tools=["write_file", "edit_file"])
        assert sk.triggers_after_tools == ["write_file", "edit_file"]

    def test_triggers_after_turn_can_be_set(self) -> None:
        sk = _SimpleSkill("x", triggers_after_turn=True)
        assert sk.triggers_after_turn is True

    def test_triggers_after_tools_instance_list_does_not_share_class_default(self) -> None:
        """Two instances must not share the same class-level list."""
        sk_a = _SimpleSkill("a", triggers_after_tools=["t1"])
        sk_b = _SimpleSkill("b")
        sk_b.triggers_after_tools.append("t2")
        assert sk_a.triggers_after_tools == ["t1"]


# ---------------------------------------------------------------------------
# SkillRegistry.get_triggered_by_tool
# ---------------------------------------------------------------------------


class TestRegistryGetTriggeredByTool:
    def test_returns_skill_whose_triggers_after_tools_matches(self) -> None:
        registry = SkillRegistry()
        skill = _SimpleSkill("post-write", triggers_after_tools=["write_file"])
        registry.register(skill)

        result = registry.get_triggered_by_tool("write_file")
        assert result == [skill]

    def test_returns_empty_list_when_no_skills_match(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("other"))
        assert registry.get_triggered_by_tool("write_file") == []

    def test_returns_empty_list_on_empty_registry(self) -> None:
        assert SkillRegistry().get_triggered_by_tool("any_tool") == []

    def test_only_returns_skills_matching_the_tool_name(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("a", triggers_after_tools=["write_file"]))
        registry.register(_SimpleSkill("b", triggers_after_tools=["read_file"]))

        result = registry.get_triggered_by_tool("write_file")
        assert len(result) == 1
        assert result[0].name == "a"

    def test_skill_triggerable_by_multiple_tools(self) -> None:
        registry = SkillRegistry()
        skill = _SimpleSkill("multi", triggers_after_tools=["tool_a", "tool_b"])
        registry.register(skill)

        assert skill in registry.get_triggered_by_tool("tool_a")
        assert skill in registry.get_triggered_by_tool("tool_b")

    def test_disabled_skill_not_returned(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("off", triggers_after_tools=["t"], enabled=False))
        assert registry.get_triggered_by_tool("t") == []

    def test_multiple_skills_triggered_by_same_tool(self) -> None:
        registry = SkillRegistry()
        skill_a = _SimpleSkill("a", triggers_after_tools=["deploy"])
        skill_b = _SimpleSkill("b", triggers_after_tools=["deploy"])
        registry.register(skill_a)
        registry.register(skill_b)

        result = registry.get_triggered_by_tool("deploy")
        assert len(result) == 2
        assert {s.name for s in result} == {"a", "b"}


# ---------------------------------------------------------------------------
# SkillRegistry.get_post_turn
# ---------------------------------------------------------------------------


class TestRegistryGetPostTurn:
    def test_returns_skills_with_triggers_after_turn_true(self) -> None:
        registry = SkillRegistry()
        skill = _SimpleSkill("review", triggers_after_turn=True)
        registry.register(skill)
        assert registry.get_post_turn() == [skill]

    def test_does_not_return_skills_without_flag(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("normal"))
        assert registry.get_post_turn() == []

    def test_empty_registry_returns_empty(self) -> None:
        assert SkillRegistry().get_post_turn() == []

    def test_disabled_post_turn_skill_excluded(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("off", triggers_after_turn=True, enabled=False))
        assert registry.get_post_turn() == []

    def test_mixed_registry_only_returns_post_turn_flagged(self) -> None:
        registry = SkillRegistry()
        always = _SimpleSkill("always")
        post = _SimpleSkill("post", triggers_after_turn=True)
        registry.register(always)
        registry.register(post)

        result = registry.get_post_turn()
        assert result == [post]


# ---------------------------------------------------------------------------
# on_tool_end — queuing via SkillHooks
# ---------------------------------------------------------------------------


class TestToolResultTriggersSkillHooks:
    async def test_matching_tool_queues_skill_in_pending(self) -> None:
        registry = SkillRegistry()
        skill = _SimpleSkill("post-write", triggers_after_tools=["write_file"])
        registry.register(skill)
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("write_file"), "ok")

        state = _get_run_state()
        assert skill in state.pending_skills

    async def test_non_matching_tool_does_not_queue_skill(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("post-write", triggers_after_tools=["write_file"]))
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("read_file"), "ok")

        assert _get_run_state().pending_skills == []

    async def test_no_registry_does_nothing(self) -> None:
        hooks = SkillHooks([_SimpleSkill("x")])
        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("any"), "ok")
        assert _get_run_state().pending_skills == []

    async def test_same_skill_not_queued_twice_for_same_tool(self) -> None:
        """Calling on_tool_end twice for the same tool must not double-enqueue."""
        registry = SkillRegistry()
        skill = _SimpleSkill("s", triggers_after_tools=["t"])
        registry.register(skill)
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")

        state = _get_run_state()
        assert state.pending_skills.count(skill) == 1

    async def test_disabled_skill_not_queued(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("off", triggers_after_tools=["t"], enabled=False))
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")

        assert _get_run_state().pending_skills == []


# ---------------------------------------------------------------------------
# on_llm_end — post-turn queuing via SkillHooks
# ---------------------------------------------------------------------------


class TestPostTurnTriggersSkillHooks:
    async def test_post_turn_skill_queued_after_llm_end(self) -> None:
        registry = SkillRegistry()
        skill = _SimpleSkill("review", triggers_after_turn=True)
        registry.register(skill)
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        state = _get_run_state()
        assert skill in state.pending_skills

    async def test_non_post_turn_skill_not_queued(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("normal"))
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        assert _get_run_state().pending_skills == []

    async def test_injected_this_call_cleared_by_llm_end(self) -> None:
        registry = SkillRegistry()
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        state = _get_run_state()
        state.injected_this_call.add("some-skill")

        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        assert state.injected_this_call == set()

    async def test_no_registry_still_clears_injected_this_call(self) -> None:
        hooks = SkillHooks([_SimpleSkill("x")])
        await hooks.on_start(_mock_context(), _mock_agent())
        state = _get_run_state()
        state.injected_this_call.add("x")

        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        assert state.injected_this_call == set()

    async def test_post_turn_skill_not_queued_twice_when_called_twice(self) -> None:
        """Calling on_llm_end twice (e.g. both SkillHooks and RunSkillHooks) must not
        double-enqueue the same post-turn skill."""
        registry = SkillRegistry()
        skill = _SimpleSkill("review", triggers_after_turn=True)
        registry.register(skill)
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        state = _get_run_state()
        assert state.pending_skills.count(skill) == 1

    async def test_disabled_post_turn_skill_not_queued(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("off", triggers_after_turn=True, enabled=False))
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        assert _get_run_state().pending_skills == []


# ---------------------------------------------------------------------------
# Pending drain — injected at next on_llm_start
# ---------------------------------------------------------------------------


class TestPendingDrain:
    async def test_pending_skill_injected_at_next_llm_start(self) -> None:
        registry = SkillRegistry()
        skill = _SimpleSkill("triggered", triggers_after_tools=["t"], content="triggered!")
        registry.register(skill)
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())

        # Simulate tool fires → skill queued
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")
        assert len(_get_run_state().pending_skills) == 1

        # Simulate LLM end (clears injected_this_call)
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        # Simulate next LLM start — pending should be drained
        input_items: list[Any] = [{"role": "user", "content": "hello"}]
        await hooks.on_llm_start(_mock_context(), _mock_agent(), None, input_items)

        contents = [b["content"] for b in input_items if isinstance(b, dict)]
        assert "triggered!" in contents

    async def test_pending_queue_cleared_after_drain(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("s", triggers_after_tools=["t"]))
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        input_items: list[Any] = [{"role": "user", "content": "x"}]
        await hooks.on_llm_start(_mock_context(), _mock_agent(), None, input_items)

        # After drain, pending must be empty
        assert _get_run_state().pending_skills == []

    async def test_pending_not_reinjected_on_subsequent_llm_start(self) -> None:
        """Draining is a one-shot operation; pending must not cause a second injection.

        The skill has ``always_on=False`` so it is routable rather than
        always-on.  With no router configured it is never selected by routing
        either, meaning the *only* path into input_items is the pending drain.
        After that drain the skill must not re-appear.
        """
        registry = SkillRegistry()
        # always_on=False → excluded from get_always_on(); no router → never routed.
        # The only injection path is the pending drain triggered by the tool.
        registry.register(_SimpleSkill("s", triggers_after_tools=["t"], content="ONCE"))
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        # First call drains pending
        items1: list[Any] = [{"role": "user", "content": "x"}]
        await hooks.on_llm_start(_mock_context(), _mock_agent(), None, items1)

        # Second call — pending is empty; skill must NOT re-appear
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())
        items2: list[Any] = [{"role": "user", "content": "y"}]
        await hooks.on_llm_start(_mock_context(), _mock_agent(), None, items2)
        contents2 = [b["content"] for b in items2 if isinstance(b, dict)]
        assert "ONCE" not in contents2

    async def test_multiple_pending_skills_all_injected(self) -> None:
        registry = SkillRegistry()
        skill_a = _SimpleSkill("a", triggers_after_tools=["t"], content="block_a")
        skill_b = _SimpleSkill("b", triggers_after_turn=True, content="block_b")
        registry.register(skill_a)
        registry.register(skill_b)
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        input_items: list[Any] = [{"role": "user", "content": "hello"}]
        await hooks.on_llm_start(_mock_context(), _mock_agent(), None, input_items)

        contents = [b["content"] for b in input_items if isinstance(b, dict)]
        assert "block_a" in contents
        assert "block_b" in contents

    async def test_disabled_pending_skill_not_injected(self) -> None:
        """A skill that becomes disabled between queueing and draining is skipped."""
        registry = SkillRegistry()
        skill = _SimpleSkill("s", triggers_after_tools=["t"], content="should-not-appear")
        registry.register(skill)
        hooks = _hooks_with_registry(registry)

        await hooks.on_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        # Disable the skill between queueing and draining
        skill._enabled = False

        input_items: list[Any] = [{"role": "user", "content": "x"}]
        await hooks.on_llm_start(_mock_context(), _mock_agent(), None, input_items)

        contents = [b["content"] for b in input_items if isinstance(b, dict)]
        assert "should-not-appear" not in contents


# ---------------------------------------------------------------------------
# _drain_pending — unit tests for the module-level helper
# ---------------------------------------------------------------------------


class TestDrainPending:
    async def test_returns_empty_list_when_no_pending(self) -> None:
        state = RunState()
        result = await _drain_pending(state, lambda s, e: None)
        assert result == []

    async def test_drains_and_returns_blocks(self) -> None:
        skill = _SimpleSkill("s", content="drained!")
        state = RunState()
        state.pending_skills.append(skill)

        blocks = await _drain_pending(state, lambda s, e: None)
        assert blocks == [{"role": "user", "content": "drained!"}]

    async def test_pending_cleared_after_drain(self) -> None:
        state = RunState()
        state.pending_skills.append(_SimpleSkill("s"))
        await _drain_pending(state, lambda s, e: None)
        assert state.pending_skills == []

    async def test_already_injected_skill_skipped(self) -> None:
        """Skills already in injected_this_call must not be drained again."""
        skill = _SimpleSkill("s", content="should-not-appear")
        state = RunState()
        state.pending_skills.append(skill)
        state.injected_this_call.add("s")  # already claimed

        blocks = await _drain_pending(state, lambda s, e: None)
        assert blocks == []

    async def test_error_in_get_prompt_blocks_calls_on_error(self) -> None:
        class BrokenSkill(Skill):
            name = "broken"
            description = "always raises"

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                raise RuntimeError("boom")

        errors: list[tuple[Skill, Exception]] = []
        state = RunState()
        state.pending_skills.append(BrokenSkill())

        result = await _drain_pending(state, lambda s, e: errors.append((s, e)))
        assert result == []
        assert len(errors) == 1
        assert isinstance(errors[0][1], RuntimeError)

    async def test_error_removes_skill_from_injected_this_call(self) -> None:
        """On error the skill name is removed from injected_this_call so it retries."""

        class BrokenSkill(Skill):
            name = "broken"
            description = "raises"

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                raise RuntimeError("boom")

        state = RunState()
        state.pending_skills.append(BrokenSkill())

        await _drain_pending(state, lambda s, e: None)
        assert "broken" not in state.injected_this_call

    async def test_concurrent_drain_only_processes_each_skill_once(self) -> None:
        """Simulates two concurrent _drain_pending calls sharing the same RunState."""
        import asyncio

        call_count = 0

        class CountingSkill(Skill):
            name = "counted"
            description = "counts calls"

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(0)  # yield to let the other coroutine run
                return [{"role": "user", "content": "counted"}]

        state = RunState()
        state.pending_skills.append(CountingSkill())

        # Run two concurrent drains sharing the same RunState
        results = await asyncio.gather(
            _drain_pending(state, lambda s, e: None),
            _drain_pending(state, lambda s, e: None),
        )

        # Only one drain should have processed the skill
        all_blocks = results[0] + results[1]
        assert call_count == 1
        assert len(all_blocks) == 1


# ---------------------------------------------------------------------------
# Cross-agent drain: tool on Agent A → inject on Agent B's on_llm_start
# ---------------------------------------------------------------------------


class TestCrossAgentDrain:
    async def test_tool_trigger_on_agent_a_drains_into_agent_b_llm_start(self) -> None:
        """Pending skills are run-scoped; a tool firing on one agent injects into
        the next LLM call regardless of which agent handles it."""
        registry = SkillRegistry()
        skill = _SimpleSkill(
            "cross-agent", triggers_after_tools=["deploy"], content="cross-agent-block"
        )
        registry.register(skill)
        hooks = _run_hooks_with_registry(registry)

        agent_a = _mock_agent()
        agent_a.configure_mock(name="AgentA")
        agent_b = _mock_agent()
        agent_b.configure_mock(name="AgentB")

        # Agent A fires tool → skill queued
        await hooks.on_agent_start(_mock_context(), agent_a)
        await hooks.on_tool_end(_mock_context(), agent_a, _mock_tool("deploy"), "ok")
        await hooks.on_llm_end(_mock_context(), agent_a, _mock_response())

        # Agent B's on_llm_start should drain the pending skill
        input_items: list[Any] = [{"role": "user", "content": "continue"}]
        await hooks.on_llm_start(_mock_context(), agent_b, None, input_items)

        contents = [b["content"] for b in input_items if isinstance(b, dict)]
        assert "cross-agent-block" in contents

    async def test_pending_cleared_after_cross_agent_drain(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("s", triggers_after_tools=["t"]))
        hooks = _run_hooks_with_registry(registry)

        await hooks.on_agent_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        input_items: list[Any] = [{"role": "user", "content": "x"}]
        await hooks.on_llm_start(_mock_context(), _mock_agent(), None, input_items)

        assert _get_run_state().pending_skills == []


# ---------------------------------------------------------------------------
# RunSkillHooks — same trigger behaviour as SkillHooks
# ---------------------------------------------------------------------------


class TestRunSkillHooksTriggers:
    async def test_tool_trigger_queues_skill(self) -> None:
        registry = SkillRegistry()
        skill = _SimpleSkill("s", triggers_after_tools=["t"])
        registry.register(skill)
        hooks = _run_hooks_with_registry(registry)

        await hooks.on_agent_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")

        assert skill in _get_run_state().pending_skills

    async def test_post_turn_queued_after_llm_end(self) -> None:
        registry = SkillRegistry()
        skill = _SimpleSkill("review", triggers_after_turn=True)
        registry.register(skill)
        hooks = _run_hooks_with_registry(registry)

        await hooks.on_agent_start(_mock_context(), _mock_agent())
        await hooks.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        assert skill in _get_run_state().pending_skills

    async def test_non_matching_tool_does_not_queue(self) -> None:
        registry = SkillRegistry()
        registry.register(_SimpleSkill("s", triggers_after_tools=["write_file"]))
        hooks = _run_hooks_with_registry(registry)

        await hooks.on_agent_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("read_file"), "ok")

        assert _get_run_state().pending_skills == []

    async def test_no_registry_on_tool_end_does_nothing(self) -> None:
        hooks = RunSkillHooks([_SimpleSkill("x")])
        await hooks.on_agent_start(_mock_context(), _mock_agent())
        await hooks.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("any"), "ok")
        assert _get_run_state().pending_skills == []


# ---------------------------------------------------------------------------
# Deduplication between SkillHooks and RunSkillHooks active simultaneously
# ---------------------------------------------------------------------------


class TestHookDeduplication:
    async def test_tool_trigger_not_double_queued_when_both_hooks_active(self) -> None:
        """If both SkillHooks and RunSkillHooks share the same registry and call
        on_tool_end for the same tool, the skill must appear in pending only once."""
        registry = SkillRegistry()
        skill = _SimpleSkill("s", triggers_after_tools=["t"])
        registry.register(skill)

        sh = _hooks_with_registry(registry)
        rh = _run_hooks_with_registry(registry)

        # Initialise RunState via both hooks
        await sh.on_start(_mock_context(), _mock_agent())
        await rh.on_agent_start(_mock_context(), _mock_agent())

        # Both hooks fire on_tool_end for the same tool
        await sh.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")
        await rh.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")

        state = _get_run_state()
        assert state.pending_skills.count(skill) == 1

    async def test_post_turn_skill_not_double_queued_when_both_hooks_active(self) -> None:
        """Same deduplication guarantee for on_llm_end."""
        registry = SkillRegistry()
        skill = _SimpleSkill("review", triggers_after_turn=True)
        registry.register(skill)

        sh = _hooks_with_registry(registry)
        rh = _run_hooks_with_registry(registry)

        await sh.on_start(_mock_context(), _mock_agent())
        await rh.on_agent_start(_mock_context(), _mock_agent())

        await sh.on_llm_end(_mock_context(), _mock_agent(), _mock_response())
        await rh.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        state = _get_run_state()
        assert state.pending_skills.count(skill) == 1

    async def test_pending_blocks_not_double_injected_when_both_hooks_drain(self) -> None:
        """When both hooks call on_llm_start concurrently, each pending skill's
        blocks appear in input_items exactly once."""
        import asyncio

        registry = SkillRegistry()
        skill = _SimpleSkill("s", triggers_after_tools=["t"], content="unique-block")
        registry.register(skill)

        sh = _hooks_with_registry(registry)
        rh = _run_hooks_with_registry(registry)

        await sh.on_start(_mock_context(), _mock_agent())
        await rh.on_agent_start(_mock_context(), _mock_agent())

        await sh.on_tool_end(_mock_context(), _mock_agent(), _mock_tool("t"), "ok")
        await sh.on_llm_end(_mock_context(), _mock_agent(), _mock_response())

        input_items: list[Any] = [{"role": "user", "content": "x"}]

        # Simulate concurrent on_llm_start (as asyncio.gather does in the SDK)
        await asyncio.gather(
            sh.on_llm_start(_mock_context(), _mock_agent(), None, input_items),
            rh.on_llm_start(_mock_context(), _mock_agent(), None, input_items),
        )

        unique_block_count = sum(
            1 for b in input_items if isinstance(b, dict) and b.get("content") == "unique-block"
        )
        assert unique_block_count == 1


# ---------------------------------------------------------------------------
# triggers_after_tools fires on tool name, not tool output content
#
# Documented claim: "triggers_after_tools is the right mechanism when a tool
# always warrants the same skill regardless of what it returned. When a tool
# can return different classifications and each classification warrants a
# different skill, see the next section."
# ---------------------------------------------------------------------------


class TestTriggersAfterToolsIgnoresContent:
    """triggers_after_tools dispatch is based solely on tool name, never on
    the content of the tool's return value."""

    async def test_same_skill_queued_for_different_result_strings_from_same_tool(
        self,
    ) -> None:
        """Two on_tool_end calls with the same tool name but completely different
        result strings both queue the same skill — the result is never inspected."""
        registry = SkillRegistry()
        skill = _SimpleSkill("classifier", triggers_after_tools=["run_pipeline"])
        registry.register(skill)

        hooks = _hooks_with_registry(registry)
        ctx = _mock_context()
        agent = _mock_agent()

        await hooks.on_start(ctx, agent)
        state = _get_run_state()

        # First call — result indicates one classification
        await hooks.on_tool_end(
            ctx, agent, _mock_tool("run_pipeline"), '{"root_cause": "NEXTHOP_UNRESOLVABLE"}'
        )
        assert any(s.name == "classifier" for s in state.pending_skills)

        # Reset pending and call again with a completely different result
        state.pending_skills.clear()
        await hooks.on_tool_end(
            ctx, agent, _mock_tool("run_pipeline"), '{"root_cause": "QUEUE_CONGESTION"}'
        )
        # Same skill still queued — content made no difference
        assert any(s.name == "classifier" for s in state.pending_skills)

    async def test_different_tool_name_does_not_queue_skill_regardless_of_result_content(
        self,
    ) -> None:
        """A tool with the wrong name never queues the skill, even when its result
        content looks identical to results produced by the registered tool."""
        registry = SkillRegistry()
        registry.register(_SimpleSkill("my-skill", triggers_after_tools=["tool-a"]))

        hooks = _hooks_with_registry(registry)
        ctx = _mock_context()
        agent = _mock_agent()

        await hooks.on_start(ctx, agent)
        state = _get_run_state()

        # Run tool-b — content is identical to what tool-a would return
        await hooks.on_tool_end(
            ctx, agent, _mock_tool("tool-b"), '{"root_cause": "NEXTHOP_UNRESOLVABLE"}'
        )

        assert not any(s.name == "my-skill" for s in state.pending_skills)

    async def test_all_skills_for_a_tool_name_are_queued_regardless_of_result(
        self,
    ) -> None:
        """Every skill declaring a given tool name is queued unconditionally —
        the result content cannot suppress any of them."""
        registry = SkillRegistry()
        registry.register(_SimpleSkill("skill-one", triggers_after_tools=["pipeline"]))
        registry.register(_SimpleSkill("skill-two", triggers_after_tools=["pipeline"]))
        registry.register(_SimpleSkill("unrelated", triggers_after_tools=["other-tool"]))

        hooks = _hooks_with_registry(registry)
        ctx = _mock_context()
        agent = _mock_agent()

        await hooks.on_start(ctx, agent)
        state = _get_run_state()

        await hooks.on_tool_end(ctx, agent, _mock_tool("pipeline"), "any result whatsoever")

        queued = {s.name for s in state.pending_skills}
        assert "skill-one" in queued
        assert "skill-two" in queued
        assert "unrelated" not in queued  # registered for a different tool name
