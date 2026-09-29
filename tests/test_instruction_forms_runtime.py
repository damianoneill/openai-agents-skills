"""Lifecycle tests for instruction-form scope propagation through the SDK.

These drive real ``Runner.run`` / ``Runner.run_streamed`` executions with a
recording fake model, plus the explicit invocation tool and the deferred-drain
hook path, to prove that ``instruction_form_scope`` is observed everywhere a
skill renders — not just when ``get_prompt_blocks`` is called directly.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from agents import Agent, Runner
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.tool_context import ToolContext
from agents.usage import Usage
from conftest import (
    make_hooks,
    make_mock_agent,
    make_mock_context,
    make_mock_response,
    make_run_hooks,
)
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from openai_agents_skills import (
    InstructionForm,
    Skill,
    SkillRegistry,
    get_instruction_form,
    instruction_form_scope,
    make_invoke_skill_tool,
)

# ---------------------------------------------------------------------------
# Recording fake model
# ---------------------------------------------------------------------------


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="m",
        role="assistant",
        status="completed",
        type="message",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
    )


def _handoff_call(agent_name: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id="h",
        call_id="c",
        name=f"transfer_to_{agent_name}",
        arguments="{}",
        type="function_call",
    )


def _tool_call(name: str, call_id: str, arguments: str = "{}") -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=call_id, call_id=call_id, name=name, arguments=arguments, type="function_call"
    )


def _response(output: list[Any]) -> Response:
    return Response(
        id="r",
        created_at=0.0,
        model="fake",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


class _FakeModel(Model):
    """Records the ``input`` seen on every model call and replays scripted turns."""

    def __init__(self, turns: list[list[Any]] | None = None) -> None:
        self._turns = turns
        self._i = 0
        self.inputs: list[Any] = []

    def _next(self) -> list[Any]:
        if self._turns is None:
            return [_message("done")]
        out = self._turns[self._i]
        self._i += 1
        return out

    async def get_response(
        self,
        system_instructions: Any,
        input: Any,
        model_settings: Any,
        tools: Any,
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        *,
        previous_response_id: Any = None,
        conversation_id: Any = None,
        prompt: Any = None,
    ) -> ModelResponse:
        self.inputs.append(input)
        return ModelResponse(output=self._next(), usage=Usage(), response_id=None)

    async def stream_response(
        self,
        system_instructions: Any,
        input: Any,
        model_settings: Any,
        tools: Any,
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        *,
        previous_response_id: Any = None,
        conversation_id: Any = None,
        prompt: Any = None,
    ) -> Any:
        self.inputs.append(input)
        yield ResponseCompletedEvent(
            type="response.completed", response=_response(self._next()), sequence_number=0
        )


# ---------------------------------------------------------------------------
# Skills that record the active instruction form
# ---------------------------------------------------------------------------


class _FormMarkerSkill(Skill):
    """Always-on skill whose content encodes the form active when it renders."""

    name = "form-marker"
    description = "Marks the active instruction form."
    always_on = True

    async def get_prompt_blocks(self, context: Any, agent: Any, args: str = "") -> list[Any]:
        return [{"role": "user", "content": f"FORM={get_instruction_form().value}"}]


class _DeferredMarkerSkill(Skill):
    """Post-turn skill whose content encodes the form active when it renders."""

    name = "deferred-marker"
    description = "Marks the active form on the next turn."
    triggers_after_turn = True

    async def get_prompt_blocks(self, context: Any, agent: Any, args: str = "") -> list[Any]:
        return [{"role": "user", "content": f"DEFERRED={get_instruction_form().value}"}]


class _InvokableMarkerSkill(Skill):
    """Non-always-on skill rendered only when invoked, encoding the active form."""

    name = "invokable"
    description = "Marks the active form when explicitly invoked."

    async def get_prompt_blocks(self, context: Any, agent: Any, args: str = "") -> list[Any]:
        return [{"role": "user", "content": f"INVOKED={get_instruction_form().value}"}]


def _flatten(inp: Any) -> str:
    return json.dumps(inp, default=str)


def _contents(items: list[Any]) -> list[str]:
    return [
        i["content"] for i in items if isinstance(i, dict) and isinstance(i.get("content"), str)
    ]


def _marker_values(items: list[Any], prefix: str) -> list[str]:
    return [c.split("=", 1)[1] for c in _contents(items) if c.startswith(prefix)]


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(_FormMarkerSkill())
    return registry


# ---------------------------------------------------------------------------
# Runner.run — scope propagation through SDK-created tasks
# ---------------------------------------------------------------------------


class TestRunnerRunPropagation:
    async def test_default_form_without_scope(self) -> None:
        model = _FakeModel()
        agent = Agent(name="a", instructions="x", model=model, hooks=make_hooks(_registry()))
        await Runner.run(agent, "hello")
        assert _marker_values(model.inputs[0], "FORM=") == ["default"]

    async def test_agent_hooks_observe_concise_scope(self) -> None:
        model = _FakeModel()
        agent = Agent(name="a", instructions="x", model=model, hooks=make_hooks(_registry()))
        with instruction_form_scope(InstructionForm.CONCISE):
            await Runner.run(agent, "hello")
        assert _marker_values(model.inputs[0], "FORM=") == ["concise"]

    async def test_run_hooks_observe_concise_scope(self) -> None:
        model = _FakeModel()
        agent = Agent(name="a", instructions="x", model=model)
        with instruction_form_scope(InstructionForm.CONCISE):
            await Runner.run(agent, "hello", hooks=make_run_hooks(_registry()))
        assert _marker_values(model.inputs[0], "FORM=") == ["concise"]

    async def test_both_hook_types_observe_concise_scope(self) -> None:
        registry = _registry()
        model = _FakeModel()
        agent = Agent(name="a", instructions="x", model=model, hooks=make_hooks(registry))
        with instruction_form_scope(InstructionForm.CONCISE):
            await Runner.run(agent, "hello", hooks=make_run_hooks(registry))
        # Both hook instances render under the scope: every injection is concise.
        values = _marker_values(model.inputs[0], "FORM=")
        assert values
        assert set(values) == {"concise"}


# ---------------------------------------------------------------------------
# Runner.run_streamed — scope propagation and restoration
# ---------------------------------------------------------------------------


class TestRunnerStreamedPropagation:
    async def test_streamed_observes_concise_scope(self) -> None:
        model = _FakeModel()
        agent = Agent(name="a", instructions="x", model=model, hooks=make_hooks(_registry()))
        with instruction_form_scope(InstructionForm.CONCISE):
            result = Runner.run_streamed(agent, "hello")
            async for _event in result.stream_events():
                pass
        assert _marker_values(model.inputs[0], "FORM=") == ["concise"]

    async def test_streamed_both_hook_types_observe_scope(self) -> None:
        registry = _registry()
        model = _FakeModel()
        agent = Agent(name="a", instructions="x", model=model, hooks=make_hooks(registry))
        with instruction_form_scope(InstructionForm.CONCISE):
            result = Runner.run_streamed(agent, "hello", hooks=make_run_hooks(registry))
            async for _event in result.stream_events():
                pass
        values = _marker_values(model.inputs[0], "FORM=")
        assert values
        assert set(values) == {"concise"}

    async def test_streamed_cancellation_drains_and_restores(self) -> None:
        model = _FakeModel()
        agent = Agent(name="a", instructions="x", model=model, hooks=make_hooks(_registry()))
        with instruction_form_scope(InstructionForm.CONCISE):
            result = Runner.run_streamed(agent, "hello")
            result.cancel()
            # Drain any remaining events after cancellation, per the SDK's guidance.
            try:
                async for _event in result.stream_events():
                    pass
            except asyncio.CancelledError:
                pass
        assert get_instruction_form() is InstructionForm.DEFAULT


# ---------------------------------------------------------------------------
# Concurrency and handoff
# ---------------------------------------------------------------------------


class TestConcurrencyAndHandoff:
    async def test_concurrent_runs_do_not_leak_form(self) -> None:
        async def run_with(form: InstructionForm) -> list[str]:
            model = _FakeModel()
            agent = Agent(name="a", instructions="x", model=model, hooks=make_hooks(_registry()))
            with instruction_form_scope(form):
                await Runner.run(agent, "hello")
            return _marker_values(model.inputs[0], "FORM=")

        concise, default = await asyncio.gather(
            run_with(InstructionForm.CONCISE), run_with(InstructionForm.DEFAULT)
        )
        assert concise == ["concise"]
        assert default == ["default"]

    async def test_concurrent_runs_share_instances_isolated(self) -> None:
        # One registry, skill instance and hooks shared across both concurrent runs.
        shared_registry = _registry()
        shared_hooks = make_run_hooks(shared_registry)

        async def run_with(form: InstructionForm) -> list[str]:
            model = _FakeModel()
            agent = Agent(name="a", instructions="x", model=model)
            with instruction_form_scope(form):
                await Runner.run(agent, "hello", hooks=shared_hooks)
            return _marker_values(model.inputs[0], "FORM=")

        concise, default = await asyncio.gather(
            run_with(InstructionForm.CONCISE), run_with(InstructionForm.DEFAULT)
        )
        assert concise == ["concise"]
        assert default == ["default"]

    async def test_handoff_receiving_agent_observes_form(self) -> None:
        model = _FakeModel(turns=[[_handoff_call("b")], [_message("final")]])
        receiver = Agent(name="b", instructions="x", model=model)
        starter = Agent(name="a", instructions="x", model=model, handoffs=[receiver])
        with instruction_form_scope(InstructionForm.CONCISE):
            await Runner.run(starter, "hello", hooks=make_run_hooks(_registry()))
        # inputs[1] is the receiving agent's turn after the handoff.
        assert _marker_values(model.inputs[1], "FORM=") == ["concise"]


# ---------------------------------------------------------------------------
# Explicit invocation and deferred drain
# ---------------------------------------------------------------------------


class TestExplicitAndDeferred:
    async def _invoke(self, tool: Any, skill_name: str) -> str:
        args_json = json.dumps({"skill_name": skill_name, "args": ""})
        ctx = ToolContext(
            context=None,
            tool_name="invoke_skill",
            tool_call_id="call",
            tool_arguments=args_json,
        )
        return await tool.on_invoke_tool(ctx, args_json)  # type: ignore[arg-type]

    async def test_invoke_skill_tool_observes_form(self) -> None:
        tool = make_invoke_skill_tool(_registry())
        assert await self._invoke(tool, "form-marker") == "FORM=default"
        with instruction_form_scope(InstructionForm.CONCISE):
            assert await self._invoke(tool, "form-marker") == "FORM=concise"

    async def test_invoke_skill_dispatched_by_model_observes_form(self) -> None:
        registry = SkillRegistry()
        registry.register(_InvokableMarkerSkill())
        tool = make_invoke_skill_tool(registry)
        model = _FakeModel(
            turns=[
                [_tool_call("invoke_skill", "c1", json.dumps({"skill_name": "invokable"}))],
                [_message("final")],
            ]
        )
        agent = Agent(
            name="a",
            instructions="x",
            model=model,
            tools=[tool],
        )
        with instruction_form_scope(InstructionForm.CONCISE):
            await Runner.run(agent, "hello", hooks=make_run_hooks(registry))
        # The invokable skill is not always-on, so its marker only reaches the
        # model when the tool the model dispatched rendered it under the scope.
        assert "INVOKED=concise" in _flatten(model.inputs[1])

    async def test_deferred_post_turn_skill_observes_form(self) -> None:
        # Exercises the real _drain_pending rendering path directly. End-to-end
        # deferred injection across a real Runner turn/handoff does not propagate
        # in this SDK (a pre-existing RunState-lifecycle limitation, tracked
        # separately), so this is driven at the hook level rather than via Runner.
        registry = SkillRegistry()
        registry.register(_DeferredMarkerSkill())
        hooks = make_hooks(registry)

        await hooks.on_start(make_mock_context(), make_mock_agent())
        await hooks.on_llm_end(make_mock_context(), make_mock_agent(), make_mock_response())

        input_items: list[Any] = [{"role": "user", "content": "hello"}]
        with instruction_form_scope(InstructionForm.CONCISE):
            await hooks.on_llm_start(make_mock_context(), make_mock_agent(), None, input_items)

        assert _marker_values(input_items, "DEFERRED=") == ["concise"]


# ---------------------------------------------------------------------------
# ContextVar inheritance — scope exit does not clear an inherited child
# ---------------------------------------------------------------------------


class TestChildTaskInheritance:
    async def test_child_retains_inherited_form_after_parent_scope_exit(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        observed: list[InstructionForm] = []

        async def child() -> None:
            started.set()
            await release.wait()
            observed.append(get_instruction_form())

        with instruction_form_scope(InstructionForm.CONCISE):
            task = asyncio.create_task(child())
            await started.wait()
        # Parent scope has exited; the child was created inside it.
        assert get_instruction_form() is InstructionForm.DEFAULT
        release.set()
        await task
        assert observed == [InstructionForm.CONCISE]

    async def test_cancelling_owning_task_restores_form(self) -> None:
        entered = asyncio.Event()
        observed: list[InstructionForm] = []

        async def owner() -> None:
            try:
                with instruction_form_scope(InstructionForm.CONCISE):
                    entered.set()
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                # The scope's finally has already restored the previous binding.
                observed.append(get_instruction_form())
                raise

        task = asyncio.create_task(owner())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert observed == [InstructionForm.DEFAULT]
