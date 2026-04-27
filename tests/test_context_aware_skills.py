"""Tests for context/agent forwarding in Skill.get_prompt_blocks and Skill.is_enabled."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from agents.tool_context import ToolContext
from conftest import (
    fire_llm_end,
    fire_llm_start,
    fire_tool_end,
    make_hooks,
    make_mock_agent,
    make_mock_context,
    make_run_hooks,
)

from openai_agents_skills import Skill, SkillHooks, SkillRegistry, make_invoke_skill_tool
from openai_agents_skills._state import _get_run_state  # noqa: F401
from openai_agents_skills.hooks import RunSkillHooks

# ---------------------------------------------------------------------------
# 1 & 3.  get_prompt_blocks forwarding through SkillHooks (identity checks)
# ---------------------------------------------------------------------------


class TestContextForwardingSkillHooks:
    async def test_get_prompt_blocks_receives_exact_context_object(self) -> None:
        # on_llm_start context is forwarded by identity to get_prompt_blocks
        received: list[Any] = []

        class _CapturingSkill(Skill):
            name = "capture"
            description = "Captures context"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received.append(context)
                return []

        hooks = SkillHooks(skills=[_CapturingSkill()])
        ctx = make_mock_context()
        await hooks.on_start(make_mock_context(), make_mock_agent())
        await fire_llm_start(hooks, [], context=ctx)

        assert len(received) == 1
        assert received[0] is ctx

    async def test_get_prompt_blocks_receives_exact_agent_object(self) -> None:
        # on_llm_start agent is forwarded by identity to get_prompt_blocks
        received: list[Any] = []

        class _CapturingSkill(Skill):
            name = "capture"
            description = "Captures agent"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received.append(agent)
                return []

        hooks = SkillHooks(skills=[_CapturingSkill()])
        agent = make_mock_agent()
        await hooks.on_start(make_mock_context(), make_mock_agent())
        await fire_llm_start(hooks, [], agent=agent)

        assert len(received) == 1
        assert received[0] is agent

    async def test_skill_can_read_context_attributes_and_include_in_blocks(self) -> None:
        # A skill reads context.context.org_id and returns it in blocks

        class _OrgSkill(Skill):
            name = "org"
            description = "Injects org ID"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                if context is not None:
                    org_id = context.context.org_id
                    return [{"role": "user", "content": f"org={org_id}"}]
                return [{"role": "user", "content": "no-context"}]

        ctx = MagicMock()
        ctx.context.org_id = "acme-123"
        hooks = SkillHooks(skills=[_OrgSkill()])
        await hooks.on_start(make_mock_context(), make_mock_agent())
        input_items: list[Any] = []
        await fire_llm_start(hooks, input_items, context=ctx)

        contents = [b["content"] for b in input_items if isinstance(b, dict)]
        assert any("acme-123" in c for c in contents)

    async def test_skill_can_read_agent_attributes_and_include_in_blocks(self) -> None:
        # A skill reads agent.name and returns it in blocks

        class _AgentNameSkill(Skill):
            name = "agent_name"
            description = "Injects agent name"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                if agent is not None:
                    return [{"role": "user", "content": f"agent={agent.name}"}]
                return [{"role": "user", "content": "no-agent"}]

        agent = MagicMock()
        agent.name = "ResearchBot"
        hooks = SkillHooks(skills=[_AgentNameSkill()])
        await hooks.on_start(make_mock_context(), make_mock_agent())
        input_items: list[Any] = []
        await fire_llm_start(hooks, input_items, agent=agent)

        contents = [b["content"] for b in input_items if isinstance(b, dict)]
        assert any("ResearchBot" in c for c in contents)

    async def test_skill_returns_fallback_when_context_and_agent_are_none(self) -> None:
        # A well-written skill returns sensible fallback content when both are None

        class _FallbackSkill(Skill):
            name = "fallback"
            description = "Returns fallback when no context"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                if context is None:
                    return [{"role": "user", "content": "fallback-content"}]
                return [{"role": "user", "content": "live-content"}]

        hooks = SkillHooks(skills=[_FallbackSkill()])
        input_items: list[Any] = []
        await fire_llm_start(hooks, input_items, context=None, agent=None)

        contents = [b["content"] for b in input_items if isinstance(b, dict)]
        assert "fallback-content" in contents
        assert "live-content" not in contents


# ---------------------------------------------------------------------------
# 4.  get_prompt_blocks forwarding through RunSkillHooks (identity checks)
# ---------------------------------------------------------------------------


class TestContextForwardingRunSkillHooks:
    async def test_get_prompt_blocks_receives_exact_context_object(self) -> None:
        # RunSkillHooks forwards on_llm_start context by identity
        received: list[Any] = []

        class _CapturingSkill(Skill):
            name = "capture"
            description = "Captures context"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received.append(context)
                return []

        hooks = RunSkillHooks(skills=[_CapturingSkill()])
        ctx = make_mock_context()
        await hooks.on_agent_start(make_mock_context(), make_mock_agent())
        await fire_llm_start(hooks, [], context=ctx)

        assert len(received) == 1
        assert received[0] is ctx

    async def test_get_prompt_blocks_receives_exact_agent_object(self) -> None:
        # RunSkillHooks forwards on_llm_start agent by identity
        received: list[Any] = []

        class _CapturingSkill(Skill):
            name = "capture"
            description = "Captures agent"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received.append(agent)
                return []

        hooks = RunSkillHooks(skills=[_CapturingSkill()])
        agent = make_mock_agent()
        await hooks.on_agent_start(make_mock_context(), make_mock_agent())
        await fire_llm_start(hooks, [], agent=agent)

        assert len(received) == 1
        assert received[0] is agent

    async def test_context_and_agent_arrive_together_from_same_on_llm_start_call(self) -> None:
        # Both context AND agent received by the skill are from the same call
        received_pairs: list[tuple[Any, Any]] = []

        class _PairCapturingSkill(Skill):
            name = "pair_capture"
            description = "Captures both context and agent"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received_pairs.append((context, agent))
                return []

        hooks = RunSkillHooks(skills=[_PairCapturingSkill()])
        ctx = make_mock_context()
        agent = make_mock_agent()
        await hooks.on_agent_start(make_mock_context(), make_mock_agent())
        await fire_llm_start(hooks, [], context=ctx, agent=agent)

        assert len(received_pairs) == 1
        got_ctx, got_agent = received_pairs[0]
        assert got_ctx is ctx
        assert got_agent is agent


# ---------------------------------------------------------------------------
# 2.  is_enabled receives context and agent
# ---------------------------------------------------------------------------


class TestContextAwareIsEnabled:
    async def test_skill_enabled_when_context_carries_feature_flag(self) -> None:
        # A skill gated on a context attribute injects when the flag is True

        class _FlagGatedSkill(Skill):
            name = "flag_gated"
            description = "Only active when feature_on is True"

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                return context is not None and bool(context.context.feature_on)

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return [{"role": "user", "content": "feature-content"}]

        ctx = MagicMock()
        ctx.context.feature_on = True
        hooks = SkillHooks(skills=[_FlagGatedSkill()])
        input_items: list[Any] = []
        await fire_llm_start(hooks, input_items, context=ctx)

        contents = [b["content"] for b in input_items if isinstance(b, dict)]
        assert "feature-content" in contents

    async def test_skill_disabled_when_context_is_none(self) -> None:
        # A skill that requires context is skipped when context=None

        class _RequiresContextSkill(Skill):
            name = "requires_ctx"
            description = "Requires context"

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                return context is not None

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return [{"role": "user", "content": "context-content"}]

        hooks = SkillHooks(skills=[_RequiresContextSkill()])
        input_items: list[Any] = []
        await fire_llm_start(hooks, input_items, context=None)

        assert input_items == []

    async def test_skill_disabled_when_context_flag_is_false(self) -> None:
        # A skill gated on a context attribute is skipped when the flag is False

        class _FlagGatedSkill(Skill):
            name = "flag_gated"
            description = "Context-gated"

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                return context is not None and bool(context.context.feature_on)

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return [{"role": "user", "content": "should-not-appear"}]

        ctx = MagicMock()
        ctx.context.feature_on = False
        hooks = SkillHooks(skills=[_FlagGatedSkill()])
        input_items: list[Any] = []
        await fire_llm_start(hooks, input_items, context=ctx)

        assert input_items == []

    async def test_skill_enabled_when_agent_name_matches(self) -> None:
        # A skill that reads agent.name in is_enabled injects for matching agent

        class _AgentGatedSkill(Skill):
            name = "agent_gated"
            description = "Only for specific agents"

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                return agent is not None and agent.name == "AllowedAgent"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return [{"role": "user", "content": "agent-specific-content"}]

        agent = MagicMock()
        agent.name = "AllowedAgent"
        hooks = SkillHooks(skills=[_AgentGatedSkill()])
        input_items: list[Any] = []
        await fire_llm_start(hooks, input_items, agent=agent)

        contents = [b["content"] for b in input_items if isinstance(b, dict)]
        assert "agent-specific-content" in contents

    async def test_skill_disabled_when_agent_name_does_not_match(self) -> None:
        # is_enabled with wrong agent name → skill skipped

        class _AgentGatedSkill(Skill):
            name = "agent_gated"
            description = "Only for specific agents"

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                return agent is not None and agent.name == "AllowedAgent"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return [{"role": "user", "content": "should-not-appear"}]

        agent = MagicMock()
        agent.name = "SomeOtherAgent"
        hooks = SkillHooks(skills=[_AgentGatedSkill()])
        input_items: list[Any] = []
        await fire_llm_start(hooks, input_items, agent=agent)

        assert input_items == []

    async def test_is_enabled_receives_context_by_identity(self) -> None:
        # The context object in is_enabled is identical to the on_llm_start context
        received_in_is_enabled: list[Any] = []

        class _InspectingSkill(Skill):
            name = "inspector"
            description = "Records context in is_enabled"

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                received_in_is_enabled.append(context)
                return True

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return []

        ctx = make_mock_context()
        hooks = SkillHooks(skills=[_InspectingSkill()])
        await fire_llm_start(hooks, [], context=ctx)

        assert len(received_in_is_enabled) == 1
        assert received_in_is_enabled[0] is ctx

    async def test_is_enabled_receives_agent_by_identity(self) -> None:
        # The agent object in is_enabled is identical to the on_llm_start agent
        received_in_is_enabled: list[Any] = []

        class _InspectingSkill(Skill):
            name = "inspector"
            description = "Records agent in is_enabled"

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                received_in_is_enabled.append(agent)
                return True

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return []

        agent = make_mock_agent()
        hooks = SkillHooks(skills=[_InspectingSkill()])
        await fire_llm_start(hooks, [], agent=agent)

        assert len(received_in_is_enabled) == 1
        assert received_in_is_enabled[0] is agent


# ---------------------------------------------------------------------------
# 5.  Deferred (pending) paths - context forwarded via state.last_context
# ---------------------------------------------------------------------------


class TestContextAwareDeferredPaths:
    async def test_tool_triggered_pending_skill_receives_context_from_next_llm_start(self) -> None:
        # Skill queued by tool trigger receives context of the next on_llm_start
        received_context: list[Any] = []

        class _PendingCaptureSkill(Skill):
            name = "pending_capture"
            description = "Captures context when drained"
            triggers_after_tools = ["trigger_tool"]

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received_context.append(context)
                return [{"role": "user", "content": "pending-block"}]

        registry = SkillRegistry()
        registry.register(_PendingCaptureSkill())
        hooks = make_hooks(registry)
        await hooks.on_start(make_mock_context(), make_mock_agent())

        tool_ctx = make_mock_context()
        await fire_tool_end(hooks, "trigger_tool", context=tool_ctx)
        await fire_llm_end(hooks, context=tool_ctx)

        next_ctx = make_mock_context()
        assert next_ctx is not tool_ctx
        input_items: list[Any] = [{"role": "user", "content": "next-turn"}]
        await fire_llm_start(hooks, input_items, context=next_ctx)

        assert len(received_context) == 1
        assert received_context[0] is next_ctx

    async def test_tool_triggered_pending_skill_receives_agent_from_next_llm_start(self) -> None:
        # Skill queued by tool trigger receives agent of the next on_llm_start
        received_agent: list[Any] = []

        class _PendingCaptureSkill(Skill):
            name = "pending_agent_capture"
            description = "Captures agent when drained"
            triggers_after_tools = ["my_tool"]

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received_agent.append(agent)
                return []

        registry = SkillRegistry()
        registry.register(_PendingCaptureSkill())
        hooks = make_hooks(registry)
        await hooks.on_start(make_mock_context(), make_mock_agent())

        tool_agent = make_mock_agent()
        await fire_tool_end(hooks, "my_tool", agent=tool_agent)
        await fire_llm_end(hooks, agent=tool_agent)

        drain_agent = make_mock_agent()
        assert drain_agent is not tool_agent
        await fire_llm_start(hooks, [], agent=drain_agent)

        assert len(received_agent) == 1
        assert received_agent[0] is drain_agent

    async def test_post_turn_skill_receives_context_from_next_llm_start(self) -> None:
        # Post-turn skill queued in on_llm_end receives context from the next on_llm_start
        received_context: list[Any] = []

        class _PostTurnCaptureSkill(Skill):
            name = "post_turn_capture"
            description = "Captures context in post-turn drain"
            triggers_after_turn = True

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received_context.append(context)
                return [{"role": "user", "content": "post-turn-block"}]

        registry = SkillRegistry()
        registry.register(_PostTurnCaptureSkill())
        hooks = make_hooks(registry)
        await hooks.on_start(make_mock_context(), make_mock_agent())

        turn1_ctx = make_mock_context()
        items1: list[Any] = [{"role": "user", "content": "first"}]
        await fire_llm_start(hooks, items1, context=turn1_ctx)
        received_context.clear()  # discard direct injection from turn 1

        await fire_llm_end(hooks, context=turn1_ctx)

        turn2_ctx = make_mock_context()
        assert turn2_ctx is not turn1_ctx
        items2: list[Any] = [{"role": "user", "content": "second"}]
        await fire_llm_start(hooks, items2, context=turn2_ctx)

        assert len(received_context) >= 1
        assert received_context[-1] is turn2_ctx

    async def test_drain_pending_uses_state_last_context_not_tool_end_context(self) -> None:
        # Pending drain uses state.last_context, not context when tool fired
        received_context: list[Any] = []

        class _ToolTriggeredSkill(Skill):
            name = "tool_triggered"
            description = "Triggered by tool"
            triggers_after_tools = ["some_tool"]

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received_context.append(context)
                return []

        registry = SkillRegistry()
        registry.register(_ToolTriggeredSkill())
        hooks = make_hooks(registry)
        await hooks.on_start(make_mock_context(), make_mock_agent())

        old_ctx = make_mock_context()
        await fire_tool_end(hooks, "some_tool", context=old_ctx)
        await fire_llm_end(hooks, context=old_ctx)

        new_ctx = make_mock_context()
        assert new_ctx is not old_ctx
        await fire_llm_start(hooks, [], context=new_ctx)

        assert len(received_context) == 1
        assert received_context[0] is new_ctx
        assert received_context[0] is not old_ctx

    async def test_context_gated_pending_skill_respects_drain_time_context_in_is_enabled(
        self,
    ) -> None:
        # Pending skill is_enabled uses state.last_context at drain time
        # A blocking context prevents injection even though the skill was queued
        injected: list[str] = []

        class _GatedPendingSkill(Skill):
            name = "gated_pending"
            description = "Context-gated pending skill"
            triggers_after_tools = ["gate_tool"]

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                return context is not None and bool(context.context.allow_skill)

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                injected.append("gated-content")
                return [{"role": "user", "content": "gated-content"}]

        registry = SkillRegistry()
        registry.register(_GatedPendingSkill())
        hooks = make_hooks(registry)
        await hooks.on_start(make_mock_context(), make_mock_agent())
        await fire_tool_end(hooks, "gate_tool")
        await fire_llm_end(hooks)

        blocking_ctx = MagicMock()
        blocking_ctx.context.allow_skill = False
        items: list[Any] = []
        await fire_llm_start(hooks, items, context=blocking_ctx)

        assert injected == []
        assert not any(b.get("content") == "gated-content" for b in items if isinstance(b, dict))

    async def test_run_skill_hooks_pending_skill_receives_context_from_next_llm_start(self) -> None:
        # Same deferred-context guarantee holds for RunSkillHooks
        received_context: list[Any] = []

        class _PendingCaptureSkill(Skill):
            name = "run_pending_capture"
            description = "Captures context in RunSkillHooks drain"
            triggers_after_tools = ["run_tool"]

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received_context.append(context)
                return []

        registry = SkillRegistry()
        registry.register(_PendingCaptureSkill())
        hooks = make_run_hooks(registry)
        await hooks.on_agent_start(make_mock_context(), make_mock_agent())

        tool_ctx = make_mock_context()
        await fire_tool_end(hooks, "run_tool", context=tool_ctx)
        await fire_llm_end(hooks, context=tool_ctx)

        drain_ctx = make_mock_context()
        assert drain_ctx is not tool_ctx
        await fire_llm_start(hooks, [], context=drain_ctx)

        assert len(received_context) == 1
        assert received_context[0] is drain_ctx


# ---------------------------------------------------------------------------
# 6.  SkillRegistry.get_always_on passes context/agent to is_enabled
# ---------------------------------------------------------------------------


class TestContextForwardingViaRegistryGetAlwaysOn:
    def test_get_always_on_passes_context_to_is_enabled(self) -> None:
        # get_always_on forwards the context argument to each skill is_enabled
        received_contexts: list[Any] = []

        class _RecordingSkill(Skill):
            name = "recorder"
            description = "Records context received in is_enabled"
            always_on = True

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                received_contexts.append(context)
                return True

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return []

        registry = SkillRegistry()
        registry.register(_RecordingSkill())
        ctx = make_mock_context()
        registry.get_always_on(ctx)

        assert len(received_contexts) == 1
        assert received_contexts[0] is ctx

    def test_get_always_on_passes_agent_to_is_enabled(self) -> None:
        # get_always_on forwards the agent argument to each skill is_enabled
        received_agents: list[Any] = []

        class _RecordingSkill(Skill):
            name = "recorder"
            description = "Records agent received in is_enabled"
            always_on = True

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                received_agents.append(agent)
                return True

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return []

        registry = SkillRegistry()
        registry.register(_RecordingSkill())
        agent = make_mock_agent()
        registry.get_always_on(agent=agent)

        assert len(received_agents) == 1
        assert received_agents[0] is agent

    def test_context_gated_always_on_skill_excluded_when_context_denies(self) -> None:
        # An always-on skill with context-based is_enabled is excluded when context says no

        class _ContextGatedSkill(Skill):
            name = "ctx_gated"
            description = "Only when allowed"
            always_on = True

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                return context is not None and bool(context.context.allowed)

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return []

        registry = SkillRegistry()
        registry.register(_ContextGatedSkill())

        disallowing_ctx = MagicMock()
        disallowing_ctx.context.allowed = False
        assert registry.get_always_on(disallowing_ctx) == []

        allowing_ctx = MagicMock()
        allowing_ctx.context.allowed = True
        assert len(registry.get_always_on(allowing_ctx)) == 1

    def test_context_gated_always_on_skill_excluded_when_context_is_none(self) -> None:
        # An always-on skill that requires context is excluded when context=None

        class _RequiresContextSkill(Skill):
            name = "needs_ctx"
            description = "Requires non-None context"
            always_on = True

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                return context is not None

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return []

        registry = SkillRegistry()
        registry.register(_RequiresContextSkill())

        assert registry.get_always_on(None) == []

    async def test_get_always_on_context_passed_through_full_hooks_pipeline(self) -> None:
        # End-to-end: context-gated always-on skill excluded when context disables it

        class _ContextGatedSkill(Skill):
            name = "pipeline_gated"
            description = "Context-gated always-on"
            always_on = True

            def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
                return context is not None and bool(context.context.active)

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                return [{"role": "user", "content": "active-content"}]

        registry = SkillRegistry()
        registry.register(_ContextGatedSkill())
        hooks = SkillHooks(registry=registry)

        inactive_ctx = MagicMock()
        inactive_ctx.context.active = False
        inactive_items: list[Any] = []
        await fire_llm_start(hooks, inactive_items, context=inactive_ctx)
        await fire_llm_end(hooks, context=inactive_ctx)

        active_ctx = MagicMock()
        active_ctx.context.active = True
        active_items: list[Any] = []
        await fire_llm_start(hooks, active_items, context=active_ctx)

        # manifest may be present; verify the skill content is absent for the inactive context
        assert not any(
            b.get("content") == "active-content" for b in inactive_items if isinstance(b, dict)
        )
        contents = [b["content"] for b in active_items if isinstance(b, dict)]
        assert "active-content" in contents


# ---------------------------------------------------------------------------
# 7.  invoke_skill tool passes its RunContextWrapper ctx to get_prompt_blocks
# ---------------------------------------------------------------------------


class TestInvokeSkillContext:
    async def test_invoke_skill_tool_passes_tool_context_to_get_prompt_blocks(self) -> None:
        # ToolContext passed to on_invoke_tool is forwarded by identity to get_prompt_blocks
        received_contexts: list[Any] = []

        class _CapturingSkill(Skill):
            name = "ctx_capture"
            description = "Captures context from invoke_skill"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received_contexts.append(context)
                return [{"role": "user", "content": "captured"}]

        registry = SkillRegistry()
        registry.register(_CapturingSkill())
        tool = make_invoke_skill_tool(registry)

        args_json = json.dumps({"skill_name": "ctx_capture"})
        tool_ctx = ToolContext(
            context=None,
            tool_name="invoke_skill",
            tool_call_id="test-call-id",
            tool_arguments=args_json,
        )

        result = await tool.on_invoke_tool(tool_ctx, args_json)

        assert "captured" in result
        assert len(received_contexts) == 1
        assert received_contexts[0] is tool_ctx

    async def test_invoke_skill_tool_inner_context_attribute_accessible_in_get_prompt_blocks(
        self,
    ) -> None:
        # ToolContext wrapping a non-None inner context makes .context accessible
        received_contexts: list[Any] = []

        class _InnerContextSkill(Skill):
            name = "inner_ctx"
            description = "Reads inner context"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received_contexts.append(context)
                if context is not None:
                    val = context.context.session_id
                    return [{"role": "user", "content": f"session={val}"}]
                return []

        registry = SkillRegistry()
        registry.register(_InnerContextSkill())
        tool = make_invoke_skill_tool(registry)

        inner_ctx = MagicMock()
        inner_ctx.session_id = "sess-42"
        args_json = json.dumps({"skill_name": "inner_ctx"})
        tool_ctx = ToolContext(
            context=inner_ctx,
            tool_name="invoke_skill",
            tool_call_id="test-call-id-2",
            tool_arguments=args_json,
        )

        result = await tool.on_invoke_tool(tool_ctx, args_json)

        assert len(received_contexts) == 1
        assert received_contexts[0] is tool_ctx
        assert "sess-42" in result

    async def test_invoke_skill_tool_agent_parameter_is_none(self) -> None:
        # The agent parameter in get_prompt_blocks is always None for invoke_skill
        received_agents: list[Any] = []

        class _AgentCapturingSkill(Skill):
            name = "agent_capture"
            description = "Captures agent param"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received_agents.append(agent)
                return [{"role": "user", "content": "ok"}]

        registry = SkillRegistry()
        registry.register(_AgentCapturingSkill())
        tool = make_invoke_skill_tool(registry)

        args_json = json.dumps({"skill_name": "agent_capture"})
        tool_ctx = ToolContext(
            context=None,
            tool_name="invoke_skill",
            tool_call_id="test-call-id-3",
            tool_arguments=args_json,
        )
        await tool.on_invoke_tool(tool_ctx, args_json)

        assert len(received_agents) == 1
        assert received_agents[0] is None

    async def test_invoke_skill_tool_args_forwarded_alongside_context(self) -> None:
        # Context and args are both forwarded correctly to get_prompt_blocks
        received: list[tuple[Any, str]] = []

        class _ContextAndArgsSkill(Skill):
            name = "ctx_and_args"
            description = "Captures both context and args"

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                received.append((context, args))
                return [{"role": "user", "content": f"args={args}"}]

        registry = SkillRegistry()
        registry.register(_ContextAndArgsSkill())
        tool = make_invoke_skill_tool(registry)

        args_json = json.dumps({"skill_name": "ctx_and_args", "args": "foo bar"})
        tool_ctx = ToolContext(
            context=None,
            tool_name="invoke_skill",
            tool_call_id="test-call-id-4",
            tool_arguments=args_json,
        )
        result = await tool.on_invoke_tool(tool_ctx, args_json)

        assert len(received) == 1
        ctx_received, args_received = received[0]
        assert ctx_received is tool_ctx
        assert args_received == "foo bar"
        assert "foo bar" in result
