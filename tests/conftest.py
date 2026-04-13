"""Shared pytest configuration and fixtures for the openai-agents-skills test suite."""

from __future__ import annotations

import pytest

from openai_agents_skills._state import _run_state


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
