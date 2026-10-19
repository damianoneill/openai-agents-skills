"""Skill run state driven through a real ``Runner``.

The SDK runs every hook in its own asyncio task, so run state must be shared
across tasks for a skill queued in ``on_tool_end`` to reach the next model call.
It must also stay separate per run and per agent: each agent-as-tool call is its
own run that shares the parent's ``Usage``, and a handoff continues the same run.
"""

from __future__ import annotations

import asyncio
import copy
import gc
import weakref
from typing import Any

import pytest
from agents import Agent, RunContextWrapper, Runner, function_tool
from agents.items import ModelResponse
from conftest import FakeModel, SimpleSkill, make_handoff_call, make_message, make_tool_call

from openai_agents_skills import RunSkillHooks, SkillHooks, SkillRegistry, make_invoke_skill_tool
from openai_agents_skills._state import _SLOT_ATTRIBUTE, _get_run_state, _run_state

GUIDANCE = "TRIGGERED-GUIDANCE"

AFTER_TOOL = SimpleSkill("guide", content=GUIDANCE, triggers_after_tools=["health_check"])
AFTER_TURN = SimpleSkill("guide", content=GUIDANCE, triggers_after_turn=True)
ALWAYS = SimpleSkill("always", content=GUIDANCE, always_on=True)


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


def _listed_per_call(model: FakeModel, skill_name: str) -> list[int]:
    return [str(call_input).count(f"- {skill_name}:") for call_input in model.inputs]


async def _run(agent: Agent, streamed: bool, **kwargs: Any) -> None:
    if streamed:
        result = Runner.run_streamed(agent, "go", **kwargs)
        async for _ in result.stream_events():
            pass
    else:
        await Runner.run(agent, "go", **kwargs)


class _FailFirstModel(FakeModel):
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        response = await super().get_response(*args, **kwargs)
        if len(self.inputs) == 1:
            raise RuntimeError("model unavailable")
        return response


def _ask_child(call_id: str) -> Any:
    return make_tool_call("ask_child", call_id, '{"input": "hi"}')


def _parent_of(child_model: FakeModel, child_skill: SimpleSkill, turns: list[list[Any]]) -> Agent:
    child = _agent(child_model, child_skill, name="child")
    return Agent(
        name="parent",
        model=FakeModel(turns),
        tools=[child.as_tool(tool_name="ask_child", tool_description="Ask the child.")],
    )


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


async def test_same_child_called_twice_starts_fresh_each_time() -> None:
    child_model = FakeModel([[make_message("one")], [make_message("two")]])
    parent = _parent_of(
        child_model, AFTER_TURN, [[_ask_child("p1")], [_ask_child("p2")], [make_message("done")]]
    )

    await Runner.run(parent, "go")

    assert _guidance_per_call(child_model) == [False, False]


async def test_same_child_called_in_parallel_gets_its_skills_in_both_calls() -> None:
    child_model = FakeModel([[make_message("one")], [make_message("two")]])
    parent = _parent_of(
        child_model, ALWAYS, [[_ask_child("p1"), _ask_child("p2")], [make_message("done")]]
    )

    await Runner.run(parent, "go")

    assert _guidance_per_call(child_model) == [True, True]


async def test_same_child_retried_after_a_failed_call_gets_its_skills() -> None:
    child_model = _FailFirstModel([[make_message("lost")], [make_message("ok")]])
    parent = _parent_of(
        child_model, ALWAYS, [[_ask_child("p1")], [_ask_child("p2")], [make_message("done")]]
    )

    await Runner.run(parent, "go")

    assert _guidance_per_call(child_model) == [True, True]


class _RunRecordingHooks(SkillHooks):
    """Keeps the run's Usage alive and weak references to its context and run state."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.refs: list[weakref.ref[Any]] = []
        self.usages: list[Any] = []

    async def on_llm_start(
        self, context: Any, agent: Any, system_prompt: Any, input_items: list[Any]
    ) -> None:
        await super().on_llm_start(context, agent, system_prompt, input_items)
        self.refs += [weakref.ref(context), weakref.ref(_get_run_state(context, agent))]
        self.usages.append(context.usage)


class _HangingModel(FakeModel):
    def __init__(self, started: asyncio.Event) -> None:
        super().__init__()
        self._started = started

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        self._started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _recorded_child_and_parent(child_model: FakeModel) -> tuple[_RunRecordingHooks, Agent]:
    registry = SkillRegistry()
    registry.register(ALWAYS)
    hooks = _RunRecordingHooks(registry=registry)
    child = Agent(name="child", model=child_model, hooks=hooks)
    parent = Agent(
        name="parent",
        model=FakeModel([[_ask_child("p1")], [make_message("done")]]),
        tools=[child.as_tool(tool_name="ask_child", tool_description="Ask the child.")],
    )
    return hooks, parent


@pytest.mark.parametrize("model_cls", [FakeModel, _FailFirstModel], ids=["completed", "failed"])
async def test_finished_child_call_releases_its_context_and_state(
    model_cls: type[FakeModel],
) -> None:
    hooks, parent = _recorded_child_and_parent(model_cls([[make_message("one")]]))

    result = await Runner.run(parent, "go")
    gc.collect()

    assert result.final_output == "done"
    assert len(hooks.refs) == 2
    assert [ref() for ref in hooks.refs] == [None, None]


async def test_cancelled_child_call_releases_its_context_and_state() -> None:
    started = asyncio.Event()
    hooks, parent = _recorded_child_and_parent(_HangingModel(started))

    task = asyncio.create_task(Runner.run(parent, "go"))
    await started.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    del task
    for _ in range(3):
        await asyncio.sleep(0)  # let the cancelled tasks' done callbacks drop their references
    gc.collect()

    assert len(hooks.refs) == 2
    assert [ref() for ref in hooks.refs] == [None, None]


@function_tool(needs_approval=True)
def guarded() -> str:
    return "{}"


async def test_run_paused_for_approval_keeps_its_state() -> None:
    registry = SkillRegistry()
    registry.register(ALWAYS)
    hooks = _RunRecordingHooks(registry=registry)
    model = FakeModel([[make_tool_call("guarded", "c1")]])
    agent = Agent(name="a", model=model, tools=[guarded], hooks=hooks)

    result = await Runner.run(agent, "go")
    gc.collect()

    assert result.interruptions
    assert hooks.refs[1]() is not None


def _failing_instructions(context: Any, agent: Any) -> str:
    raise RuntimeError("instructions unavailable")


async def test_child_call_failing_before_its_model_call_leaves_no_state() -> None:
    registry = SkillRegistry()
    registry.register(ALWAYS)
    child = Agent(
        name="child",
        model=FakeModel(),
        instructions=_failing_instructions,
        hooks=SkillHooks(registry=registry),
    )
    parent = Agent(
        name="parent",
        model=FakeModel([[_ask_child("p1")], [make_message("done")]]),
        tools=[child.as_tool(tool_name="ask_child", tool_description="Ask the child.")],
    )

    result = await Runner.run(parent, "go")
    gc.collect()

    assert result.final_output == "done"
    slot = getattr(result.context_wrapper.usage, _SLOT_ATTRIBUTE, None)
    assert slot is None or slot.states == {}


async def test_invoke_skill_limit_counts_every_call_in_one_run() -> None:
    registry = SkillRegistry()
    registry.register(SimpleSkill("s", content="ok"))
    args = '{"skill_name": "s"}'
    model = FakeModel(
        [
            [make_tool_call("invoke_skill", "c1", args)],
            [make_tool_call("invoke_skill", "c2", args)],
            [make_message("done")],
        ]
    )
    agent = Agent(
        name="a", model=model, tools=[make_invoke_skill_tool(registry, max_calls_per_run=1)]
    )

    await Runner.run(agent, "go")

    assert "limit reached" not in str(model.inputs[1])
    assert "limit reached" in str(model.inputs[2])


@pytest.mark.parametrize("streamed", [False, True], ids=["run", "run_streamed"])
async def test_invoke_skill_limit_is_per_agent_across_a_handoff(streamed: bool) -> None:
    registry = SkillRegistry()
    registry.register(SimpleSkill("s", content="SKILL-BODY"))
    tool = make_invoke_skill_tool(registry, max_calls_per_run=1)
    args = '{"skill_name": "s"}'
    model = FakeModel(
        [
            [make_tool_call("invoke_skill", "c1", args)],
            [make_handoff_call("b")],
            [make_tool_call("invoke_skill", "c2", args)],
            [make_message("done")],
        ]
    )
    receiver = Agent(name="b", model=model, tools=[tool])
    starter = Agent(name="a", model=model, tools=[tool], handoffs=[receiver])

    await _run(starter, streamed)

    assert str(model.inputs[3]).count("SKILL-BODY") == 2


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


@pytest.mark.parametrize("streamed", [False, True], ids=["run", "run_streamed"])
async def test_manifest_is_in_every_request(streamed: bool) -> None:
    model = FakeModel([[make_tool_call("health_check", "c1")], [make_message("done")]])
    agent = _manifest_agent("a", model, "listed", tools=[health_check])

    await _run(agent, streamed)

    assert _listed_per_call(model, "listed") == [1, 1]


@pytest.mark.parametrize(
    "run_skills", [[], ["run-skill"]], ids=["empty_run_registry", "run_registry"]
)
async def test_manifest_lists_agent_and_run_registries(run_skills: list[str]) -> None:
    model = FakeModel([[make_tool_call("health_check", "c1")], [make_message("done")]])
    agent = _manifest_agent("a", model, "agent-skill", tools=[health_check])
    run_registry = SkillRegistry()
    for name in run_skills:
        run_registry.register(SimpleSkill(name))

    await Runner.run(agent, "go", hooks=RunSkillHooks(registry=run_registry))

    for name in ["agent-skill", *run_skills]:
        assert _listed_per_call(model, name) == [1, 1]


async def test_manifest_lists_a_shared_skill_once_per_request() -> None:
    registry = SkillRegistry()
    registry.register(SimpleSkill("shared"))
    model = FakeModel([[make_tool_call("health_check", "c1")], [make_message("done")]])
    agent = Agent(name="a", model=model, tools=[health_check], hooks=SkillHooks(registry=registry))

    await Runner.run(agent, "go", hooks=RunSkillHooks(registry=registry))

    assert _listed_per_call(model, "shared") == [1, 1]


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


async def test_agent_hooks_pending_skill_stays_when_run_registry_has_same_name() -> None:
    agent_registry = SkillRegistry()
    agent_registry.register(SimpleSkill("guide", content="AGENT-ONLY", triggers_after_turn=True))
    run_registry = SkillRegistry()
    run_registry.register(SimpleSkill("guide", content="RUN-GUIDE"))
    model = FakeModel([[make_handoff_call("b")], [make_message("done")]])
    receiver = Agent(name="b", model=model)
    starter = Agent(
        name="a", model=model, handoffs=[receiver], hooks=SkillHooks(registry=agent_registry)
    )

    await Runner.run(starter, "go", hooks=RunSkillHooks(registry=run_registry))

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
