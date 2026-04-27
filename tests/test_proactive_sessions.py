"""Tests for proactive (agent-initiated) session entry.

Documented claims:
- A synthetic first message constructed by calling code is treated identically
  to a human-authored user message by the injection lifecycle.
- on_llm_start fires, always-on skills inject unconditionally, and the router
  receives the synthetic prompt as its routing context string.
- Skills re-inject after tool calls in a proactive session just as they do in
  an interactive session.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from openai_agents_skills import (
    Skill,
    SkillHooks,
    SkillRegistry,
)
from openai_agents_skills._state import _get_run_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleSkill(Skill):
    def __init__(
        self,
        name: str,
        content: str = "injected",
        always_on: bool = False,
        triggers_after_tools: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = f"Description for {name}"
        self.always_on = always_on
        self._content = content
        self.triggers_after_tools = list(triggers_after_tools or [])

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": self._content}]


class _RecordingRouter:
    """Router that records every call and returns a pre-configured list of names."""

    def __init__(self, names: list[str]) -> None:
        self._names = names
        self.calls: list[tuple[str, list[str]]] = []

    async def select(self, message: str, skills: list[Skill]) -> list[str]:
        self.calls.append((message, [s.name for s in skills]))
        return list(self._names)


def _hooks(registry: SkillRegistry) -> SkillHooks:
    return SkillHooks(
        registry=registry,
        on_skill_error=lambda s, e: (_ for _ in ()).throw(e),
    )


def _mock_tool(name: str) -> Any:
    tool = MagicMock()
    tool.name = name
    return tool


def _mock_response() -> Any:
    return MagicMock()


def _synthetic_prompt(
    device: str = "qfx-spine-01",
    event: str = "sustained packet drop on interface et-0/0/0",
    root_cause: str = "NEXTHOP_UNRESOLVABLE",
) -> str:
    return (
        f"Automated alert from monitoring system.\n"
        f"Device: {device}\n"
        f"Event: {event}\n"
        f"Root cause: {root_cause}\n"
        f"Analyse the evidence and produce a remediation recommendation."
    )


def _contents(items: list[Any]) -> list[str]:
    return [i["content"] for i in items if isinstance(i, dict) and "content" in i]


# ---------------------------------------------------------------------------
# Always-on injection — synthetic vs human entry path
# ---------------------------------------------------------------------------


class TestAlwaysOnSkillInjectsForSyntheticPrompt:
    """Always-on skills fire unconditionally regardless of how the session started."""

    @pytest.mark.asyncio
    async def test_always_on_skill_injects_for_synthetic_prompt(self) -> None:
        """An always-on skill injects when the first message is a synthetic alert,
        not a human query."""
        registry = SkillRegistry()
        registry.register(
            _SimpleSkill("escalation-policy", content="ESCALATION POLICY", always_on=True)
        )

        hooks = _hooks(registry)
        ctx = None  # type: ignore[arg-type]
        agent = None  # type: ignore[arg-type]

        items: list[Any] = [{"role": "user", "content": _synthetic_prompt()}]
        await hooks.on_llm_start(ctx, agent, None, items)

        assert "ESCALATION POLICY" in _contents(items)

    @pytest.mark.asyncio
    async def test_synthetic_and_human_prompts_produce_identical_always_on_injection(
        self,
    ) -> None:
        """The injection result for a synthetic prompt is identical to the result
        for a human message — the entry path is irrelevant to the hooks lifecycle."""

        def _make_registry() -> SkillRegistry:
            r = SkillRegistry()
            r.register(_SimpleSkill("policy", content="POLICY CONTENT", always_on=True))
            return r

        human_items: list[Any] = [{"role": "user", "content": "My BGP session is flapping."}]
        synthetic_items: list[Any] = [{"role": "user", "content": _synthetic_prompt()}]

        hooks_h = _hooks(_make_registry())
        hooks_s = _hooks(_make_registry())

        ctx = None  # type: ignore[arg-type]
        agent = None  # type: ignore[arg-type]

        await hooks_h.on_start(ctx, agent)
        await hooks_h.on_llm_start(ctx, agent, None, human_items)

        # Reset RunState so hooks_s gets a clean slate independent of hooks_h
        from openai_agents_skills._state import _run_state

        _run_state.set(None)

        await hooks_s.on_start(ctx, agent)
        await hooks_s.on_llm_start(ctx, agent, None, synthetic_items)

        assert "POLICY CONTENT" in _contents(human_items)
        assert "POLICY CONTENT" in _contents(synthetic_items)

    @pytest.mark.asyncio
    async def test_multiple_always_on_skills_all_inject_for_synthetic_prompt(self) -> None:
        """All always-on skills inject, not just the first one registered."""
        registry = SkillRegistry()
        registry.register(_SimpleSkill("policy-a", content="POLICY A", always_on=True))
        registry.register(_SimpleSkill("policy-b", content="POLICY B", always_on=True))

        hooks = _hooks(registry)
        ctx = None  # type: ignore[arg-type]
        agent = None  # type: ignore[arg-type]

        items: list[Any] = [{"role": "user", "content": _synthetic_prompt()}]
        await hooks.on_llm_start(ctx, agent, None, items)

        contents = _contents(items)
        assert "POLICY A" in contents
        assert "POLICY B" in contents


# ---------------------------------------------------------------------------
# Router receives the full synthetic prompt as routing context
# ---------------------------------------------------------------------------


class TestRouterReceivesSyntheticPrompt:
    """The router is handed the synthetic message text verbatim."""

    @pytest.mark.asyncio
    async def test_router_receives_full_synthetic_text_as_routing_context(self) -> None:
        """The router is called with the complete synthetic prompt string, including
        all structured fields — it is not summarised or transformed."""
        router = _RecordingRouter(names=[])
        registry = SkillRegistry(router=router)
        registry.register(_SimpleSkill("bgp-troubleshooting"))

        hooks = _hooks(registry)
        ctx = None  # type: ignore[arg-type]
        agent = None  # type: ignore[arg-type]

        prompt = _synthetic_prompt(root_cause="NEXTHOP_UNRESOLVABLE", device="qfx-spine-01")
        items: list[Any] = [{"role": "user", "content": prompt}]
        await hooks.on_llm_start(ctx, agent, None, items)

        assert len(router.calls) == 1
        routing_context = router.calls[0][0]
        assert "NEXTHOP_UNRESOLVABLE" in routing_context
        assert "qfx-spine-01" in routing_context

    @pytest.mark.asyncio
    async def test_router_selected_skill_injects_for_synthetic_prompt(self) -> None:
        """A skill selected by the router for a synthetic prompt is injected into
        input_items just as it would be for a conversational user message."""
        router = _RecordingRouter(names=["bgp-troubleshooting"])
        registry = SkillRegistry(router=router)
        registry.register(_SimpleSkill("bgp-troubleshooting", content="BGP CHECKLIST"))

        hooks = _hooks(registry)
        ctx = None  # type: ignore[arg-type]
        agent = None  # type: ignore[arg-type]

        prompt = _synthetic_prompt(event="BGP peer flap", root_cause="BGP_PEER_UNREACHABLE")
        items: list[Any] = [{"role": "user", "content": prompt}]
        await hooks.on_llm_start(ctx, agent, None, items)

        assert "BGP CHECKLIST" in _contents(items)

    @pytest.mark.asyncio
    async def test_router_not_called_when_synthetic_prompt_has_no_routable_skills(
        self,
    ) -> None:
        """When the registry has no routable skills, the router is never called —
        this holds for synthetic prompts just as for human messages."""
        router = _RecordingRouter(names=[])
        registry = SkillRegistry(router=router)
        # Register only an always-on skill — nothing routable
        registry.register(_SimpleSkill("policy", content="POLICY", always_on=True))

        hooks = _hooks(registry)
        ctx = None  # type: ignore[arg-type]
        agent = None  # type: ignore[arg-type]

        items: list[Any] = [{"role": "user", "content": _synthetic_prompt()}]
        await hooks.on_llm_start(ctx, agent, None, items)

        assert len(router.calls) == 0

    @pytest.mark.asyncio
    async def test_two_different_synthetic_prompts_each_trigger_router_call(self) -> None:
        """Different synthetic prompts (e.g. two separate proactive events) each
        produce a distinct router call — the LRU cache does not conflate them."""
        router = _RecordingRouter(names=[])
        registry = SkillRegistry(router=router)
        registry.register(_SimpleSkill("some-skill"))

        hooks = _hooks(registry)
        ctx = None  # type: ignore[arg-type]
        agent = None  # type: ignore[arg-type]

        prompt_a = _synthetic_prompt(device="device-a", root_cause="NEXTHOP_UNRESOLVABLE")
        prompt_b = _synthetic_prompt(device="device-b", root_cause="QUEUE_CONGESTION")

        items_a: list[Any] = [{"role": "user", "content": prompt_a}]
        await hooks.on_llm_start(ctx, agent, None, items_a)

        await hooks.on_llm_end(ctx, agent, _mock_response())  # type: ignore[arg-type]

        items_b: list[Any] = [{"role": "user", "content": prompt_b}]
        await hooks.on_llm_start(ctx, agent, None, items_b)

        assert len(router.calls) == 2
        assert router.calls[0][0] != router.calls[1][0]


# ---------------------------------------------------------------------------
# Skills re-inject after tool calls in a proactive session
# ---------------------------------------------------------------------------


class TestSkillReinjectionAfterToolCallInProactiveSession:
    """The re-injection lifecycle after tool calls is identical regardless of
    how the session was started."""

    @pytest.mark.asyncio
    async def test_always_on_skill_reinjects_after_tool_call_in_proactive_session(
        self,
    ) -> None:
        """In a proactive session, always-on skills re-inject after every tool call
        just as they do in an interactive session — on_llm_end clears the guard."""
        registry = SkillRegistry()
        registry.register(
            _SimpleSkill("escalation-policy", content="ESCALATION POLICY", always_on=True)
        )

        hooks = _hooks(registry)
        ctx = None  # type: ignore[arg-type]
        agent = None  # type: ignore[arg-type]

        items: list[Any] = [{"role": "user", "content": _synthetic_prompt()}]

        # First LLM call
        await hooks.on_llm_start(ctx, agent, None, items)
        first_call_count = sum(
            1 for i in items if isinstance(i, dict) and i.get("content") == "ESCALATION POLICY"
        )
        assert first_call_count == 1

        # Tool executes; on_llm_end clears the guard
        await hooks.on_llm_end(ctx, agent, _mock_response())  # type: ignore[arg-type]
        items.append({"role": "tool", "content": "tool output"})

        # Post-tool LLM call — skill must re-inject
        await hooks.on_llm_start(ctx, agent, None, items)
        second_call_count = sum(
            1 for i in items if isinstance(i, dict) and i.get("content") == "ESCALATION POLICY"
        )
        assert second_call_count == 2

    @pytest.mark.asyncio
    async def test_tool_triggered_skill_drains_into_next_llm_start_in_proactive_session(
        self,
    ) -> None:
        """A skill declared with triggers_after_tools is queued by on_tool_end and
        drained into the next on_llm_start — this works identically in proactive
        sessions where no human sent the initial message."""
        registry = SkillRegistry()
        registry.register(
            _SimpleSkill(
                "log-parser",
                content="LOG PARSER GUIDANCE",
                triggers_after_tools=["run_show_command"],
            )
        )

        hooks = _hooks(registry)
        ctx = None  # type: ignore[arg-type]
        agent = None  # type: ignore[arg-type]

        items: list[Any] = [{"role": "user", "content": _synthetic_prompt()}]

        await hooks.on_llm_start(ctx, agent, None, items)
        await hooks.on_llm_end(ctx, agent, _mock_response())  # type: ignore[arg-type]

        # on_tool_end queues the triggered skill
        await hooks.on_tool_end(ctx, agent, _mock_tool("run_show_command"), "show output")  # type: ignore[arg-type]

        state = _get_run_state()
        assert any(s.name == "log-parser" for s in state.pending_skills)

        # Post-tool on_llm_start drains the pending queue
        items.append({"role": "tool", "content": "show output"})
        await hooks.on_llm_start(ctx, agent, None, items)

        assert "LOG PARSER GUIDANCE" in _contents(items)

    @pytest.mark.asyncio
    async def test_without_on_llm_end_skill_does_not_reinject_in_proactive_session(
        self,
    ) -> None:
        """If on_llm_end is not called between LLM invocations, the per-call guard
        is not cleared and the skill does not re-inject — same as interactive sessions."""
        registry = SkillRegistry()
        registry.register(_SimpleSkill("policy", content="POLICY", always_on=True))

        hooks = _hooks(registry)
        ctx = None  # type: ignore[arg-type]
        agent = None  # type: ignore[arg-type]

        items: list[Any] = [{"role": "user", "content": _synthetic_prompt()}]

        await hooks.on_llm_start(ctx, agent, None, items)
        first_count = sum(1 for i in items if isinstance(i, dict) and i.get("content") == "POLICY")

        # Second on_llm_start WITHOUT calling on_llm_end first
        await hooks.on_llm_start(ctx, agent, None, items)
        second_count = sum(1 for i in items if isinstance(i, dict) and i.get("content") == "POLICY")

        # Guard was not cleared — count must not have grown
        assert second_count == first_count
