"""Shared pytest configuration and fixtures for the openai-agents-skills test suite."""

from __future__ import annotations

import pytest

from openai_agents_skills._state import _run_state
from openai_agents_skills.skills import Skill


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
