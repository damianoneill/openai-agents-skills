"""Skill run state driven through a real ``Runner``.

The SDK runs every hook in its own asyncio task, so run state must be shared
across tasks for a skill queued in ``on_tool_end`` to reach the next model call.
It must also stay separate per agent: a sub-agent used as a tool shares the
parent run's ``Usage``, and a handoff continues the same run.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest
from agents import Agent, RunContextWrapper, Runner, function_tool
from conftest import FakeModel, SimpleSkill, make_handoff_call, make_message, make_tool_call

from openai_agents_skills import RunSkillHooks, SkillHooks, SkillRegistry
from openai_agents_skills._state import _get_run_state, _run_state

GUIDANCE = "TRIGGERED-GUIDANCE"

AFTER_TOOL = SimpleSkill("guide", content=GUIDANCE, triggers_after_tools=["health_check"])
AFTER_TURN = SimpleSkill("guide", content=GUIDANCE, triggers_after_turn=True)


@function_tool
def health_check() -> str:
    return "{}"


def _agent(model: FakeModel, skill: SimpleSkill = AFTER_TOOL, name: str = "a") -> Agent:
    registry = SkillRegistry()
    registry.register(skill)
    return Agent(name=name, model=model, tools=[health_check], hooks=SkillHooks(registry=registry))


def _manifest_agent(name: str, model: FakeModel, skill_name: str, **kwargs: Any) -> Agent:
    registry = SkillRegistry()
    registry.register(SimpleSkill(skill_name))
    return Agent(name=name, model=model, hooks=SkillHooks(registry=registry), **kwargs)


def _guidance_per_call(model: FakeModel) -> list[bool]:
    return [GUIDANCE in str(call_input) for call_input in model.inputs]


async def test_triggered_skill_reaches_the_call_after_the_tool() -> None:
    model = FakeModel([[make_tool_call("health_check", "c1")], [make_message("done")]])
    await Runner.run(_agent(model), "check it")
    assert _guidance_per_call(model) == [False, True]


@pytest.mark.parametrize("skill", [AFTER_TOOL, AFTER_TURN], ids=["after_tool", "after_turn"])
async def test_triggered_skill_reaches_the_call_after_the_tool_streamed(skill: SimpleSkill) -> None:
    model = FakeModel([[make_tool_call("health_check", "c1")], [make_message("done")]])
    result = Runner.run_streamed(_agent(model, skill), "check it")
    async for _ in result.stream_events():
        pass
    assert _guidance_per_call(model) == [False, True]


async def test_nested_run_does_not_share_pending_skills() -> None:
    """A tool call inside the child must not queue the parent's skill, nor vice versa."""
    child_model = FakeModel([[make_tool_call("health_check", "c1")], [make_message("child")]])
    child = _agent(child_model, AFTER_TURN, name="child")
    parent_model = FakeModel(
        [
            [make_tool_call("ask_child", "p1", '{"input": "hi"}')],
            [make_tool_call("health_check", "p2")],
            [make_message("done")],
        ]
    )
    parent = _agent(parent_model, AFTER_TOOL)
    parent.tools.append(child.as_tool(tool_name="ask_child", tool_description="Ask the child."))

    await Runner.run(parent, "go")

    assert _guidance_per_call(child_model) == [False, True]
    assert _guidance_per_call(parent_model) == [False, False, True]


async def test_concurrent_runs_do_not_share_pending_skills() -> None:
    triggered = FakeModel([[make_tool_call("health_check", "c1")], [make_message("done")]])
    plain = FakeModel([[make_message("a")], [make_message("b")]])
    plain_agent = _agent(plain)

    async def plain_twice() -> None:
        await Runner.run(plain_agent, "one")
        await Runner.run(plain_agent, "two")

    await asyncio.gather(Runner.run(_agent(triggered), "check it"), plain_twice())
    assert _guidance_per_call(triggered) == [False, True]
    assert _guidance_per_call(plain) == [False, False]


async def test_agent_used_as_tool_gets_its_own_manifest() -> None:
    child_model = FakeModel([[make_message("child done")]])
    child = _manifest_agent("child", child_model, "child-skill")
    parent_model = FakeModel(
        [[make_tool_call("ask_child", "c1", '{"input": "hi"}')], [make_message("done")]]
    )
    parent = _manifest_agent(
        "parent",
        parent_model,
        "parent-skill",
        tools=[child.as_tool(tool_name="ask_child", tool_description="Ask the child.")],
    )

    await Runner.run(parent, "go")

    assert "parent-skill" in str(parent_model.inputs[0])
    assert "child-skill" in str(child_model.inputs[0])
    assert "parent-skill" not in str(child_model.inputs[0])


async def test_handoff_target_gets_its_own_manifest() -> None:
    model = FakeModel([[make_handoff_call("b")], [make_message("done")]])
    receiver = _manifest_agent("b", model, "b-skill")
    starter = _manifest_agent("a", model, "a-skill", handoffs=[receiver])

    await Runner.run(starter, "go")

    assert "a-skill" in str(model.inputs[0])
    assert "b-skill" in str(model.inputs[1])


@pytest.mark.parametrize(
    ("skill", "first_turn"),
    [
        (
            SimpleSkill("guide", content=GUIDANCE, triggers_after_turn=True),
            [make_handoff_call("b")],
        ),
        (
            SimpleSkill("guide", content=GUIDANCE, triggers_after_tools=["health_check"]),
            [make_tool_call("health_check", "c1"), make_handoff_call("b")],
        ),
    ],
    ids=["post_turn", "after_tool"],
)
async def test_run_hooks_pending_skill_follows_handoff(
    skill: SimpleSkill, first_turn: list[Any]
) -> None:
    registry = SkillRegistry()
    registry.register(skill)
    model = FakeModel([first_turn, [make_message("done")]])
    receiver = Agent(name="b", model=model)
    starter = Agent(name="a", model=model, tools=[health_check], handoffs=[receiver])

    await Runner.run(starter, "go", hooks=RunSkillHooks(registry=registry))

    assert _guidance_per_call(model) == [False, True]


async def test_agent_hooks_pending_skill_stays_with_its_agent_on_handoff() -> None:
    agent_registry = SkillRegistry()
    agent_registry.register(
        SimpleSkill("agent-only", content="AGENT-ONLY", triggers_after_turn=True)
    )
    run_registry = SkillRegistry()
    run_registry.register(SimpleSkill("run-wide", content="RUN-WIDE", triggers_after_turn=True))
    model = FakeModel([[make_handoff_call("b")], [make_message("done")]])
    receiver = Agent(name="b", model=model)
    starter = Agent(
        name="a", model=model, handoffs=[receiver], hooks=SkillHooks(registry=agent_registry)
    )

    await Runner.run(starter, "go", hooks=RunSkillHooks(registry=run_registry))

    assert "RUN-WIDE" in str(model.inputs[1])
    assert "AGENT-ONLY" not in str(model.inputs[1])


async def test_run_hooks_handoff_to_same_agent_keeps_pending_skills() -> None:
    registry = SkillRegistry()
    registry.register(SimpleSkill("guide"))
    context = RunContextWrapper(context=None)
    agent = Agent(name="a")
    _get_run_state(context, agent).pending_skills.append(registry.get("guide"))

    await RunSkillHooks(registry=registry).on_handoff(context, agent, agent)

    assert [s.name for s in _get_run_state(context, agent).pending_skills] == ["guide"]


async def test_run_does_not_leave_state_in_the_caller_context() -> None:
    model = FakeModel([[make_tool_call("health_check", "c1")], [make_message("done")]])
    await Runner.run(_agent(model), "check it")
    assert _run_state.get() is None


def test_copied_usage_starts_with_fresh_state() -> None:
    """The SDK deep-copies Usage for resumable checkpoints; run state must not follow."""
    context = RunContextWrapper(context=None)
    _get_run_state(context).pending_skills.append(SimpleSkill("queued"))

    copied = RunContextWrapper(context=None, usage=copy.deepcopy(context.usage))

    assert _get_run_state(copied).pending_skills == []
    assert [s.name for s in _get_run_state(context).pending_skills] == ["queued"]
