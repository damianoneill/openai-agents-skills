"""Shared pytest configuration and fixtures for the openai-agents-skills test suite."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from openai_agents_skills import Skill, SkillHooks, SkillRegistry
from openai_agents_skills._state import _run_state
from openai_agents_skills.hooks import RunSkillHooks

# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------


def _reraise(skill: Any, exc: Exception) -> None:  # noqa: ARG001
    """Re-raise *exc* — use as ``on_skill_error`` to surface errors in tests."""
    raise exc


# ---------------------------------------------------------------------------
# MockRouter
# ---------------------------------------------------------------------------


class MockRouter:
    """Simple mock router that records calls and returns pre-configured names.

    Shared across test files to eliminate duplication.
    """

    def __init__(self, names: list[str]) -> None:
        self._names = names
        # Each entry is (message, [skill_name, ...]) so callers can inspect what was passed.
        self.calls: list[tuple[str, list[str]]] = []

    async def select(self, message: str, skills: list[Skill]) -> list[str]:
        self.calls.append((message, [s.name for s in skills]))
        return list(self._names)


# ---------------------------------------------------------------------------
# SimpleSkill — maximally-parameterised concrete Skill for use in tests
# ---------------------------------------------------------------------------


class SimpleSkill(Skill):
    """Configurable concrete Skill subclass usable as a fixture or imported directly.

    Covers all Phase 1–4 knobs so individual test files do not need their own copies.
    """

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

    def is_enabled(self, context: Any = None, agent: Any = None) -> bool:
        return self._enabled

    async def get_prompt_blocks(self, context: Any, agent: Any, args: str = "") -> list[Any]:
        return [{"role": "user", "content": self._content}]


# ---------------------------------------------------------------------------
# Mock object factories
# ---------------------------------------------------------------------------


def make_mock_tool(name: str) -> Any:
    """Return a minimal MagicMock with a .name attribute, matching the SDK Tool interface."""
    tool = MagicMock()
    tool.name = name
    return tool


def make_mock_context() -> Any:
    """Return a MagicMock suitable for use as an agent context."""
    return MagicMock()


def make_mock_agent() -> Any:
    """Return a MagicMock suitable for use as an agent."""
    return MagicMock()


def make_mock_response() -> Any:
    """Return a MagicMock suitable for use as an LLM response."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Hooks factories
# ---------------------------------------------------------------------------


def make_hooks(registry: SkillRegistry) -> SkillHooks:
    """Return a SkillHooks bound to *registry* with a re-raising error handler."""
    return SkillHooks(registry=registry, on_skill_error=_reraise)


def make_run_hooks(registry: SkillRegistry) -> RunSkillHooks:
    """Return a RunSkillHooks bound to *registry* with a re-raising error handler."""
    return RunSkillHooks(registry=registry, on_skill_error=_reraise)


# ---------------------------------------------------------------------------
# Async fire helpers — thin wrappers around the hook lifecycle calls
# ---------------------------------------------------------------------------


async def fire_llm_start(
    hooks: Any,
    input_items: list[Any],
    context: Any = None,
    agent: Any = None,
) -> None:
    """Invoke ``hooks.on_llm_start`` with minimal / optional SDK arguments."""
    await hooks.on_llm_start(
        context=context,
        agent=agent,
        system_prompt=None,
        input_items=input_items,
    )


async def fire_llm_end(
    hooks: Any,
    context: Any = None,
    agent: Any = None,
    response: Any = None,
) -> None:
    """Invoke ``hooks.on_llm_end`` with minimal / optional SDK arguments."""
    await hooks.on_llm_end(
        context=context,
        agent=agent,
        response=response,
    )


async def fire_tool_end(
    hooks: Any,
    tool_name: str,
    context: Any = None,
    agent: Any = None,
    result: str = "",
) -> None:
    """Invoke ``hooks.on_tool_end`` with a minimal mock tool."""
    tool = make_mock_tool(tool_name)
    await hooks.on_tool_end(context=context, agent=agent, tool=tool, result=result)


# ---------------------------------------------------------------------------
# Content extraction helper
# ---------------------------------------------------------------------------


def extract_contents(items: list[Any]) -> list[str]:
    """Return the ``content`` value from every dict item in *items* that has one."""
    return [item["content"] for item in items if isinstance(item, dict) and "content" in item]


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_run_state() -> pytest.Generator[None, None, None]:
    """Reset the RunState ContextVar to None before and after every test.

    This ensures each test starts with a clean RunState so that per-run flags
    (manifest_injected, injected_this_call, invoke_skill_calls) cannot leak
    between tests that share the same asyncio event loop.
    """
    token = _run_state.set(None)
    yield
    _run_state.reset(token)


@pytest.fixture
def mock_context() -> Any:
    """Pytest fixture that returns a fresh MagicMock context object."""
    return make_mock_context()


@pytest.fixture
def mock_agent() -> Any:
    """Pytest fixture that returns a fresh MagicMock agent object."""
    return make_mock_agent()


@pytest.fixture
def mock_response() -> Any:
    """Pytest fixture that returns a fresh MagicMock response object."""
    return make_mock_response()
