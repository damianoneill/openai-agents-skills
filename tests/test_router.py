"""Tests for LLMSkillRouter — protocol conformance, select parsing, caching, and error handling."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from agents.model_settings import ModelSettings
from agents.tracing import Span, Trace, TracingProcessor, set_trace_processors, trace

from openai_agents_skills import BaseSkillRouter, LLMSkillRouter, Skill, SkillRouter
from openai_agents_skills.router import _extract_json

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _RoutableSkill(Skill):
    """Skill with always_on=False (default) — eligible for LLM routing."""

    name = "skill_a"
    description = "Does topic A things."

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": "skill_a content"}]


class _AnotherRoutableSkill(Skill):
    """A second routable skill for multi-skill tests."""

    name = "skill_b"
    description = "Does topic B things."

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": "skill_b content"}]


class _NonRoutableSkill(Skill):
    """Skill with always_on=True — always-on, never routed by LLM."""

    name = "non_routable"
    description = "Always on."
    always_on = True

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": "non_routable content"}]


def _make_mock_model(response_json: str) -> tuple[MagicMock, AsyncMock]:
    """Return *(mock_model, mock_get_response)* where *mock_get_response* records every call.

    The mock satisfies ``model.get_response(...)`` and returns a response object whose
    ``output[0].content[0].text`` equals *response_json*.
    """
    mock_block = MagicMock()
    mock_block.text = response_json

    mock_item = MagicMock()
    mock_item.content = [mock_block]

    mock_response = MagicMock()
    mock_response.output = [mock_item]

    mock_get_response = AsyncMock(return_value=mock_response)

    mock_model = MagicMock()
    mock_model.get_response = mock_get_response

    return mock_model, mock_get_response


def _make_error_model(error: Exception) -> tuple[MagicMock, AsyncMock]:
    """Return *(mock_model, mock_get_response)* where *mock_get_response* raises *error*."""
    mock_get_response = AsyncMock(side_effect=error)
    mock_model = MagicMock()
    mock_model.get_response = mock_get_response
    return mock_model, mock_get_response


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestSkillRouterProtocol:
    def test_llm_skill_router_satisfies_skill_router_protocol(self) -> None:
        mock_model, _ = _make_mock_model("{}")
        router = LLMSkillRouter(model=mock_model)

        assert isinstance(router, SkillRouter)

    def test_llm_skill_router_has_select_method(self) -> None:
        mock_model, _ = _make_mock_model("{}")
        router = LLMSkillRouter(model=mock_model)

        assert callable(getattr(router, "select", None))


# ---------------------------------------------------------------------------
# select — happy path
# ---------------------------------------------------------------------------


class TestLLMSkillRouterSelect:
    async def test_select_parses_json_response_and_returns_names(self) -> None:
        mock_model, _ = _make_mock_model('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(model=mock_model)

        result = await router.select("test message", [_RoutableSkill()])

        assert result == ["skill_a"]

    async def test_select_returns_multiple_names_in_order(self) -> None:
        mock_model, _ = _make_mock_model('{"selected": ["skill_a", "skill_b"]}')
        router = LLMSkillRouter(model=mock_model)

        result = await router.select("test message", [_RoutableSkill(), _AnotherRoutableSkill()])

        assert result == ["skill_a", "skill_b"]

    async def test_select_returns_empty_list_when_none_selected(self) -> None:
        mock_model, _ = _make_mock_model('{"selected": []}')
        router = LLMSkillRouter(model=mock_model)

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []

    async def test_select_filters_non_string_entries_in_selected(self) -> None:
        """Non-string values in the ``selected`` list must be silently dropped."""
        mock_model, _ = _make_mock_model('{"selected": ["ok", 123, null, true]}')
        router = LLMSkillRouter(model=mock_model)

        class _OkSkill(Skill):
            name = "ok"
            description = "OK"

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                return []

        result = await router.select("test", [_OkSkill()])

        assert result == ["ok"]

    async def test_select_empty_skills_list_skips_llm_call(self) -> None:
        mock_model, mock_get_response = _make_mock_model('{"selected": []}')
        router = LLMSkillRouter(model=mock_model)

        result = await router.select("test", [])

        assert result == []
        mock_get_response.assert_not_called()

    async def test_select_all_non_routable_skills_skips_llm_call(self) -> None:
        """When every skill has always_on=True the LLM is never invoked."""
        mock_model, mock_get_response = _make_mock_model('{"selected": []}')
        router = LLMSkillRouter(model=mock_model)

        result = await router.select("test", [_NonRoutableSkill()])

        assert result == []
        mock_get_response.assert_not_called()


# ---------------------------------------------------------------------------
# select — always-on skill exclusion
# ---------------------------------------------------------------------------


class TestLLMSkillRouterNonRoutableExclusion:
    async def test_always_on_skill_not_in_manifest_sent_to_llm(self) -> None:
        """Skills with always_on=True must not appear in the prompt."""
        mock_model, mock_get_response = _make_mock_model('{"selected": []}')
        router = LLMSkillRouter(model=mock_model)

        await router.select("test", [_RoutableSkill(), _NonRoutableSkill()])

        # A routable skill exists so the LLM was called.
        assert mock_get_response.called
        call_kwargs = mock_get_response.call_args
        prompt_content: str = call_kwargs.kwargs["input"]
        assert "non_routable" not in prompt_content

    async def test_routable_skill_is_included_in_manifest(self) -> None:
        mock_model, mock_get_response = _make_mock_model('{"selected": []}')
        router = LLMSkillRouter(model=mock_model)

        await router.select("test", [_RoutableSkill()])

        call_kwargs = mock_get_response.call_args
        prompt_content: str = call_kwargs.kwargs["input"]
        assert "skill_a" in prompt_content

    async def test_description_text_included_in_manifest(self) -> None:
        mock_model, mock_get_response = _make_mock_model('{"selected": []}')
        router = LLMSkillRouter(model=mock_model)

        await router.select("test", [_RoutableSkill()])

        call_kwargs = mock_get_response.call_args
        prompt_content: str = call_kwargs.kwargs["input"]
        assert "Does topic A things" in prompt_content


# ---------------------------------------------------------------------------
# select — error handling
# ---------------------------------------------------------------------------


class TestLLMSkillRouterErrorHandling:
    async def test_client_error_returns_empty_list(self) -> None:
        mock_model, _ = _make_error_model(RuntimeError("network error"))
        router = LLMSkillRouter(model=mock_model)

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []

    async def test_client_error_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_model, _ = _make_error_model(RuntimeError("network error"))
        router = LLMSkillRouter(model=mock_model)

        with caplog.at_level(logging.WARNING, logger="openai_agents_skills.router"):
            await router.select("test message", [_RoutableSkill()])

        assert any("SkillRouter.select failed" in r.getMessage() for r in caplog.records)

    async def test_malformed_json_returns_empty_list(self) -> None:
        mock_model, _ = _make_mock_model("not-json{{{{")
        router = LLMSkillRouter(model=mock_model)

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []

    async def test_malformed_json_no_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """_extract_json converts unrecognised input to '{}' silently — no warning expected."""
        mock_model, _ = _make_mock_model("not-json{{{{")
        router = LLMSkillRouter(model=mock_model)

        with caplog.at_level(logging.WARNING, logger="openai_agents_skills.router"):
            await router.select("test message", [_RoutableSkill()])

        assert not any("SkillRouter.select failed" in r.getMessage() for r in caplog.records)

    async def test_non_dict_json_returns_empty_list(self) -> None:
        """A JSON array at the top level is invalid — must return []."""
        mock_model, _ = _make_mock_model('["skill_a"]')
        router = LLMSkillRouter(model=mock_model)

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []

    async def test_non_dict_json_no_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """_extract_json finds no JSON object in a bare array — falls back to '{}' silently."""
        mock_model, _ = _make_mock_model('["skill_a"]')
        router = LLMSkillRouter(model=mock_model)

        with caplog.at_level(logging.WARNING, logger="openai_agents_skills.router"):
            await router.select("test message", [_RoutableSkill()])

        assert not any("SkillRouter.select failed" in r.getMessage() for r in caplog.records)

    async def test_missing_selected_key_returns_empty_list(self) -> None:
        """JSON object without the ``selected`` key yields an empty result."""
        mock_model, _ = _make_mock_model('{"skills": ["skill_a"]}')
        router = LLMSkillRouter(model=mock_model)

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []

    async def test_selected_value_is_non_list_returns_empty(self) -> None:
        """If ``selected`` is not a list, return empty without crashing."""
        mock_model, _ = _make_mock_model('{"selected": "skill_a"}')
        router = LLMSkillRouter(model=mock_model)

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []


# ---------------------------------------------------------------------------
# select — caching
# ---------------------------------------------------------------------------


class TestLLMSkillRouterCaching:
    async def test_same_message_calls_llm_only_once(self) -> None:
        mock_model, mock_get_response = _make_mock_model('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(model=mock_model)
        skills = [_RoutableSkill()]

        result1 = await router.select("identical message", skills)
        result2 = await router.select("identical message", skills)

        assert result1 == result2 == ["skill_a"]
        assert mock_get_response.call_count == 1

    async def test_cached_result_is_independent_copy(self) -> None:
        """Mutating the returned list must not corrupt the cache."""
        mock_model, mock_get_response = _make_mock_model('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(model=mock_model)
        skills = [_RoutableSkill()]

        first = await router.select("msg", skills)
        first.append("injected")

        second = await router.select("msg", skills)

        assert "injected" not in second
        assert mock_get_response.call_count == 1

    async def test_different_messages_each_call_llm(self) -> None:
        mock_model, mock_get_response = _make_mock_model('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(model=mock_model)
        skills = [_RoutableSkill()]

        await router.select("message one", skills)
        await router.select("message two", skills)

        assert mock_get_response.call_count == 2

    async def test_cache_eviction_removes_oldest_entry(self) -> None:
        """With cache_size=2, the third distinct message evicts the first."""
        mock_model, mock_get_response = _make_mock_model('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(model=mock_model, cache_size=2)
        skills = [_RoutableSkill()]

        await router.select("msg1", skills)  # cache: {msg1}          — call count: 1
        await router.select("msg2", skills)  # cache: {msg1, msg2}    — call count: 2
        await router.select("msg3", skills)  # evict msg1; cache: {msg2, msg3} — count: 3

        call_count_after_three = mock_get_response.call_count
        assert call_count_after_three == 3

        # msg1 was evicted — must trigger a fresh LLM call.
        await router.select("msg1", skills)
        assert mock_get_response.call_count == call_count_after_three + 1

    async def test_cache_hit_promotes_entry_to_survive_next_eviction(self) -> None:
        """A cache hit moves the entry to MRU so it outlives older (unaccessed) entries.

        Trace (cache_size=2):
          1. miss msg1  → cache: [msg1]                        client calls: 1
          2. miss msg2  → cache: [msg1(oldest), msg2]          client calls: 2
          3. hit  msg1  → move_to_end; cache: [msg2(oldest), msg1]  client calls: 2
          4. miss msg3  → evict msg2 (oldest); cache: [msg1, msg3]  client calls: 3

        After step 4:
          - msg1 is still cached (promoted past msg2 in step 3).
          - msg2 was evicted (it became oldest after the promotion).
          - msg3 is freshly cached.
        """
        mock_model, mock_get_response = _make_mock_model('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(model=mock_model, cache_size=2)
        skills = [_RoutableSkill()]

        await router.select("msg1", skills)  # step 1
        await router.select("msg2", skills)  # step 2
        await router.select("msg1", skills)  # step 3: hit — promotes msg1
        await router.select("msg3", skills)  # step 4: evicts msg2

        assert mock_get_response.call_count == 3

        count_before = mock_get_response.call_count  # = 3

        # msg1 was promoted — it must still be in cache; no new client call.
        await router.select("msg1", skills)
        assert mock_get_response.call_count == count_before

        # msg2 was evicted in step 4 — must trigger a fresh client call.
        await router.select("msg2", skills)
        assert mock_get_response.call_count == count_before + 1

    async def test_cache_size_one_keeps_only_latest_message(self) -> None:
        mock_model, mock_get_response = _make_mock_model('{"selected": []}')
        router = LLMSkillRouter(model=mock_model, cache_size=1)
        skills = [_RoutableSkill()]

        await router.select("alpha", skills)  # cache: {alpha}
        await router.select("beta", skills)  # evict alpha; cache: {beta}

        count_before = mock_get_response.call_count

        # alpha evicted → new LLM call.
        await router.select("alpha", skills)
        assert mock_get_response.call_count == count_before + 1

        count_before2 = mock_get_response.call_count

        # After re-adding alpha, cache = {alpha}.  Beta was evicted.
        await router.select("beta", skills)
        assert mock_get_response.call_count == count_before2 + 1


class TestLLMSkillRouterCacheSizeZero:
    async def test_cache_size_zero_never_caches_results(self) -> None:
        """cache_size=0 disables the cache entirely; every call hits the LLM."""
        mock_model, mock_get_response = _make_mock_model('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(model=mock_model, cache_size=0)
        skills = [_RoutableSkill()]

        await router.select("same message", skills)
        await router.select("same message", skills)
        await router.select("same message", skills)

        # Every call is a cache miss because caching is disabled.
        assert mock_get_response.call_count == 3

    async def test_cache_size_zero_does_not_raise(self) -> None:
        """cache_size=0 must not raise KeyError or any other exception."""
        mock_model, _ = _make_mock_model('{"selected": []}')
        router = LLMSkillRouter(model=mock_model, cache_size=0)
        skills = [_RoutableSkill()]

        # Should complete without any exception.
        result = await router.select("any message", skills)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _extract_json — unit tests for the JSON extraction helper
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_none_returns_empty_object(self) -> None:
        assert _extract_json(None) == "{}"

    def test_empty_string_returns_empty_object(self) -> None:
        assert _extract_json("") == "{}"

    def test_plain_json_returned_as_is(self) -> None:
        raw = '{"selected": ["skill_a"]}'
        assert _extract_json(raw) == raw

    def test_plain_json_with_empty_selected(self) -> None:
        raw = '{"selected": []}'
        assert _extract_json(raw) == raw

    def test_prose_wrapped_json_extracts_object(self) -> None:
        raw = 'Based on the message, here are the skills: {"selected": ["skill_a"]}'
        result = _extract_json(raw)
        assert result == '{"selected": ["skill_a"]}'

    def test_prose_with_newline_before_json(self) -> None:
        raw = 'I will select the relevant skills.\n\n{"selected": ["skill_b"]}'
        result = _extract_json(raw)
        assert result == '{"selected": ["skill_b"]}'

    def test_no_json_in_prose_returns_empty_object(self) -> None:
        assert _extract_json("There are no matching skills.") == "{}"

    def test_thinking_block_list_extracts_text_json(self) -> None:
        """Extended-thinking models return a list of typed content blocks."""
        content = [
            {"type": "thinking", "thinking": "Let me analyse the user message..."},
            {"type": "text", "text": '{"selected": ["router-diagnostics"]}'},
        ]
        result = _extract_json(content)
        assert result == '{"selected": ["router-diagnostics"]}'

    def test_thinking_block_list_with_empty_selection(self) -> None:
        content = [
            {"type": "thinking", "thinking": "Nothing relevant."},
            {"type": "text", "text": '{"selected": []}'},
        ]
        result = _extract_json(content)
        assert result == '{"selected": []}'

    def test_list_with_non_dict_entries_handled_gracefully(self) -> None:
        """Non-dict entries in the content list are converted via str()."""
        content = ["some string", '{"selected": ["x"]}']
        result = _extract_json(content)
        # The joined string contains the JSON — it should be extractable
        assert '"selected"' in result

    def test_whitespace_only_string_returns_empty_object(self) -> None:
        assert _extract_json("   \n\t  ") == "{}"

    def test_integer_content_returns_empty_object(self) -> None:
        assert _extract_json(42) == "{}"


# ---------------------------------------------------------------------------
# LLMSkillRouter — prose-wrapped and thinking-block responses (Issue 2 regression)
# ---------------------------------------------------------------------------


class TestLLMSkillRouterProseAndThinkingResponses:
    async def test_prose_wrapped_json_is_parsed_correctly(self) -> None:
        """Model returns prose before the JSON object — must still parse."""
        prose_response = (
            "Based on the user's message about BGP issues, I'll select:\n"
            '{"selected": ["bgp-troubleshooting"]}'
        )
        mock_model, _ = _make_mock_model(prose_response)

        class _BgpSkill(Skill):
            name = "bgp-troubleshooting"
            description = "BGP diagnostics."

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                return []

        router = LLMSkillRouter(model=mock_model)
        result = await router.select("BGP session flapping", [_BgpSkill()])

        assert result == ["bgp-troubleshooting"]

    async def test_model_with_reasoning_item_returns_text_block(self) -> None:
        """Model returning a leading reasoning item — text block must still be found."""
        mock_reasoning = MagicMock()
        mock_reasoning.content = None  # reasoning items have no text content list

        mock_block = MagicMock()
        mock_block.text = '{"selected": ["skill_a"]}'
        mock_message = MagicMock()
        mock_message.content = [mock_block]

        mock_response = MagicMock()
        mock_response.output = [mock_reasoning, mock_message]

        mock_get_response = AsyncMock(return_value=mock_response)
        mock_model = MagicMock()
        mock_model.get_response = mock_get_response

        router = LLMSkillRouter(model=mock_model)
        result = await router.select("test message", [_RoutableSkill()])

        assert result == ["skill_a"]


# ---------------------------------------------------------------------------
# BaseSkillRouter — custom subclass conformance
# ---------------------------------------------------------------------------


class TestBaseSkillRouter:
    def test_base_skill_router_satisfies_skill_router_protocol(self) -> None:
        """BaseSkillRouter.select satisfies the SkillRouter protocol."""

        class _ConcreteRouter(BaseSkillRouter):
            async def _call_model(self, prompt: str) -> str:
                return '{"selected": []}'

        router = _ConcreteRouter()
        assert isinstance(router, SkillRouter)

    async def test_custom_subclass_routes_correctly(self) -> None:
        """A minimal _call_model implementation routes skills end-to-end."""

        class _EchoRouter(BaseSkillRouter):
            """Always selects skill_a."""

            async def _call_model(self, prompt: str) -> str:
                return '{"selected": ["skill_a"]}'

        router = _EchoRouter()
        result = await router.select("any message", [_RoutableSkill(), _AnotherRoutableSkill()])

        assert result == ["skill_a"]

    async def test_custom_subclass_prose_response_extracted(self) -> None:
        """_extract_json is applied to _call_model output for custom subclasses too."""

        class _ProseRouter(BaseSkillRouter):
            async def _call_model(self, prompt: str) -> str:
                return 'I selected the following: {"selected": ["skill_b"]}'

        router = _ProseRouter()
        result = await router.select("some query", [_RoutableSkill(), _AnotherRoutableSkill()])

        assert result == ["skill_b"]

    async def test_call_model_not_implemented_returns_empty_list(self) -> None:
        """Calling BaseSkillRouter directly (without override) returns [] and logs."""
        router = BaseSkillRouter()

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []

    async def test_custom_subclass_inherits_lru_cache(self) -> None:
        """The inherited LRU cache prevents duplicate _call_model calls."""
        call_count = 0

        class _CountingRouter(BaseSkillRouter):
            async def _call_model(self, prompt: str) -> str:
                nonlocal call_count
                call_count += 1
                return '{"selected": ["skill_a"]}'

        router = _CountingRouter()
        skills = [_RoutableSkill()]

        await router.select("same message", skills)
        await router.select("same message", skills)

        assert call_count == 1

    async def test_custom_subclass_cache_size_zero_disables_cache(self) -> None:
        """cache_size=0 on a custom subclass bypasses the cache."""
        call_count = 0

        class _CountingRouter(BaseSkillRouter):
            async def _call_model(self, prompt: str) -> str:
                nonlocal call_count
                call_count += 1
                return '{"selected": []}'

        router = _CountingRouter(cache_size=0)
        skills = [_RoutableSkill()]

        await router.select("same message", skills)
        await router.select("same message", skills)

        assert call_count == 2


# ---------------------------------------------------------------------------
# LLMSkillRouter — model_settings parameter
# ---------------------------------------------------------------------------


class TestLLMSkillRouterModelSettings:
    async def test_custom_model_settings_passed_to_get_response(self) -> None:
        """When model_settings is provided, it must be forwarded to model.get_response()."""
        mock_model, mock_get_response = _make_mock_model('{"selected": ["skill_a"]}')
        sentinel_settings = MagicMock()

        router = LLMSkillRouter(model=mock_model, model_settings=sentinel_settings)
        await router.select("test message", [_RoutableSkill()])

        call_kwargs = mock_get_response.call_args.kwargs
        assert call_kwargs["model_settings"] is sentinel_settings

    async def test_none_model_settings_uses_default_model_settings(self) -> None:
        """When model_settings is None (default), an empty ModelSettings() is created."""
        mock_model, mock_get_response = _make_mock_model('{"selected": ["skill_a"]}')

        router = LLMSkillRouter(model=mock_model)  # model_settings defaults to None
        await router.select("test message", [_RoutableSkill()])

        call_kwargs = mock_get_response.call_args.kwargs
        assert isinstance(call_kwargs["model_settings"], ModelSettings)

    async def test_default_model_settings_use_temperature_zero(self) -> None:
        """The default routing settings pin temperature to 0 for deterministic selection."""
        mock_model, mock_get_response = _make_mock_model('{"selected": ["skill_a"]}')

        router = LLMSkillRouter(model=mock_model)  # model_settings defaults to None
        await router.select("test message", [_RoutableSkill()])

        call_kwargs = mock_get_response.call_args.kwargs
        assert call_kwargs["model_settings"].temperature == 0.0

    async def test_custom_model_settings_temperature_not_overridden(self) -> None:
        """Caller-supplied settings are passed verbatim; their temperature is preserved."""
        mock_model, mock_get_response = _make_mock_model('{"selected": ["skill_a"]}')
        custom_settings = ModelSettings(temperature=0.7)

        router = LLMSkillRouter(model=mock_model, model_settings=custom_settings)
        await router.select("test message", [_RoutableSkill()])

        call_kwargs = mock_get_response.call_args.kwargs
        assert call_kwargs["model_settings"] is custom_settings
        assert call_kwargs["model_settings"].temperature == 0.7

    async def test_explicit_none_model_settings_produces_fresh_model_settings(self) -> None:
        """Passing model_settings=None explicitly behaves the same as omitting it."""
        mock_model, mock_get_response = _make_mock_model('{"selected": ["skill_a"]}')

        router = LLMSkillRouter(model=mock_model, model_settings=None)
        await router.select("test message", [_RoutableSkill()])

        call_kwargs = mock_get_response.call_args.kwargs
        assert isinstance(call_kwargs["model_settings"], ModelSettings)

    async def test_same_settings_object_reused_across_calls(self) -> None:
        """The provided settings object is reused on every call — no accidental copies."""
        mock_model, mock_get_response = _make_mock_model('{"selected": []}')
        sentinel_settings = MagicMock()

        router = LLMSkillRouter(model=mock_model, model_settings=sentinel_settings)
        # Two distinct messages to avoid cache hit.
        await router.select("msg1", [_RoutableSkill()])
        await router.select("msg2", [_RoutableSkill()])

        for call in mock_get_response.call_args_list:
            assert call.kwargs["model_settings"] is sentinel_settings


# ---------------------------------------------------------------------------
# select — routing observability (tracing span)
# ---------------------------------------------------------------------------


class _SpanCollector(TracingProcessor):
    """Trace processor that records every span ended during a test."""

    def __init__(self) -> None:
        self.spans: list[Span[Any]] = []

    def on_trace_start(self, trace: Trace) -> None:  # noqa: D102
        pass

    def on_trace_end(self, trace: Trace) -> None:  # noqa: D102
        pass

    def on_span_start(self, span: Span[Any]) -> None:  # noqa: D102
        pass

    def on_span_end(self, span: Span[Any]) -> None:  # noqa: D102
        self.spans.append(span)

    def force_flush(self) -> None:  # noqa: D102
        pass

    def shutdown(self) -> None:  # noqa: D102
        pass

    def routing_spans(self) -> list[Span[Any]]:
        """Return only the ``skill_routing`` custom spans collected."""
        return [s for s in self.spans if getattr(s.span_data, "name", None) == "skill_routing"]


@pytest.fixture
def span_collector() -> Any:
    """Install a span-collecting trace processor for the duration of a test."""
    collector = _SpanCollector()
    set_trace_processors([collector])
    try:
        yield collector
    finally:
        set_trace_processors([])


class TestLLMSkillRouterTracing:
    async def test_select_emits_skill_routing_span_with_selected_and_rejected(
        self, span_collector: _SpanCollector
    ) -> None:
        mock_model, _ = _make_mock_model('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(model=mock_model)

        with trace("test-workflow"):
            result = await router.select(
                "test message", [_RoutableSkill(), _AnotherRoutableSkill()]
            )

        assert result == ["skill_a"]
        routing = span_collector.routing_spans()
        assert len(routing) == 1
        data = routing[0].span_data.data
        assert data["selected"] == ["skill_a"]
        assert data["rejected"] == ["skill_b"]
        assert data["candidates"] == ["skill_a", "skill_b"]
        assert data["candidate_count"] == 2
        assert data["selected_count"] == 1
        assert data["cache_hit"] is False
        assert "error" not in data

    async def test_cache_hit_records_cache_hit_true(self, span_collector: _SpanCollector) -> None:
        mock_model, _ = _make_mock_model('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(model=mock_model)
        skills = [_RoutableSkill()]

        with trace("test-workflow"):
            await router.select("same message", skills)  # miss → populates cache
            await router.select("same message", skills)  # hit

        routing = span_collector.routing_spans()
        assert len(routing) == 2
        assert routing[0].span_data.data["cache_hit"] is False
        assert routing[1].span_data.data["cache_hit"] is True
        assert routing[1].span_data.data["selected"] == ["skill_a"]

    async def test_error_path_records_error_on_span(self, span_collector: _SpanCollector) -> None:
        mock_model, _ = _make_error_model(RuntimeError("network error"))
        router = LLMSkillRouter(model=mock_model)

        with trace("test-workflow"):
            result = await router.select("test message", [_RoutableSkill()])

        assert result == []
        routing = span_collector.routing_spans()
        assert len(routing) == 1
        data = routing[0].span_data.data
        assert data["selected"] == []
        assert data["cache_hit"] is False
        assert data["error"] == "network error"
        assert data["error_type"] == "RuntimeError"

    async def test_no_active_trace_emits_no_span_and_no_error_log(
        self, span_collector: _SpanCollector, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without a trace, routing must not create a span nor log a 'No active trace' error."""
        mock_model, _ = _make_mock_model('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(model=mock_model)

        with caplog.at_level(logging.ERROR):
            result = await router.select("test message", [_RoutableSkill()])

        assert result == ["skill_a"]
        assert span_collector.routing_spans() == []
        assert not any("No active trace" in r.getMessage() for r in caplog.records)
