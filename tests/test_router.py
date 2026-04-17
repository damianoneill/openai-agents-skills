"""Tests for LLMSkillRouter — protocol conformance, select parsing, caching, and error handling."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from openai_agents_skills import BaseSkillRouter, LLMSkillRouter, Skill, SkillRouter
from openai_agents_skills.router import _extract_json

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _RoutableSkill(Skill):
    """Skill with a non-empty when_to_use — eligible for LLM routing."""

    name = "skill_a"
    description = "Does topic A things."
    when_to_use = "Use when the user asks about topic A."

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": "skill_a content"}]


class _AnotherRoutableSkill(Skill):
    """A second routable skill for multi-skill tests."""

    name = "skill_b"
    description = "Does topic B things."
    when_to_use = "Use when the user asks about topic B."

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": "skill_b content"}]


class _NonRoutableSkill(Skill):
    """Skill with empty when_to_use — always-on, never routed by LLM."""

    name = "non_routable"
    description = "Always on."
    when_to_use = ""

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
        return [{"role": "user", "content": "non_routable content"}]


def _make_mock_client(response_json: str) -> tuple[MagicMock, AsyncMock]:
    """Return *(mock_client, mock_create)* where *mock_create* records every call.

    The mock chain satisfies ``client.chat.completions.create(...)`` and returns
    a response object whose ``choices[0].message.content`` equals *response_json*.
    """
    mock_message = MagicMock()
    mock_message.content = response_json

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_create = AsyncMock(return_value=mock_response)

    mock_completions = MagicMock()
    mock_completions.create = mock_create

    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    mock_client = MagicMock()
    mock_client.chat = mock_chat

    return mock_client, mock_create


def _make_error_client(error: Exception) -> tuple[MagicMock, AsyncMock]:
    """Return *(mock_client, mock_create)* where *mock_create* raises *error*."""
    mock_create = AsyncMock(side_effect=error)

    mock_completions = MagicMock()
    mock_completions.create = mock_create

    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    mock_client = MagicMock()
    mock_client.chat = mock_chat

    return mock_client, mock_create


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestSkillRouterProtocol:
    def test_llm_skill_router_satisfies_skill_router_protocol(self) -> None:
        mock_client, _ = _make_mock_client("{}")
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        assert isinstance(router, SkillRouter)

    def test_llm_skill_router_has_select_method(self) -> None:
        mock_client, _ = _make_mock_client("{}")
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        assert callable(getattr(router, "select", None))


# ---------------------------------------------------------------------------
# select — happy path
# ---------------------------------------------------------------------------


class TestLLMSkillRouterSelect:
    async def test_select_parses_json_response_and_returns_names(self) -> None:
        mock_client, _ = _make_mock_client('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        result = await router.select("test message", [_RoutableSkill()])

        assert result == ["skill_a"]

    async def test_select_returns_multiple_names_in_order(self) -> None:
        mock_client, _ = _make_mock_client('{"selected": ["skill_a", "skill_b"]}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        result = await router.select("test message", [_RoutableSkill(), _AnotherRoutableSkill()])

        assert result == ["skill_a", "skill_b"]

    async def test_select_returns_empty_list_when_none_selected(self) -> None:
        mock_client, _ = _make_mock_client('{"selected": []}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []

    async def test_select_filters_non_string_entries_in_selected(self) -> None:
        """Non-string values in the ``selected`` list must be silently dropped."""
        mock_client, _ = _make_mock_client('{"selected": ["ok", 123, null, true]}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        class _OkSkill(Skill):
            name = "ok"
            description = "OK"
            when_to_use = "Use when OK."

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                return []

        result = await router.select("test", [_OkSkill()])

        assert result == ["ok"]

    async def test_select_empty_skills_list_skips_llm_call(self) -> None:
        mock_client, mock_create = _make_mock_client('{"selected": []}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        result = await router.select("test", [])

        assert result == []
        mock_create.assert_not_called()

    async def test_select_all_non_routable_skills_skips_llm_call(self) -> None:
        """When every skill has empty when_to_use the LLM is never invoked."""
        mock_client, mock_create = _make_mock_client('{"selected": []}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        result = await router.select("test", [_NonRoutableSkill()])

        assert result == []
        mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# select — non-routable skill exclusion
# ---------------------------------------------------------------------------


class TestLLMSkillRouterNonRoutableExclusion:
    async def test_non_routable_skill_not_in_manifest_sent_to_llm(self) -> None:
        """Skills with empty when_to_use must not appear in the prompt."""
        mock_client, mock_create = _make_mock_client('{"selected": []}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        await router.select("test", [_RoutableSkill(), _NonRoutableSkill()])

        # A routable skill exists so the LLM was called.
        assert mock_create.called
        call_kwargs = mock_create.call_args[1]
        prompt_content: str = call_kwargs["messages"][0]["content"]
        assert "non_routable" not in prompt_content

    async def test_routable_skill_is_included_in_manifest(self) -> None:
        mock_client, mock_create = _make_mock_client('{"selected": []}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        await router.select("test", [_RoutableSkill()])

        call_kwargs = mock_create.call_args[1]
        prompt_content: str = call_kwargs["messages"][0]["content"]
        assert "skill_a" in prompt_content

    async def test_when_to_use_text_included_in_manifest(self) -> None:
        mock_client, mock_create = _make_mock_client('{"selected": []}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        await router.select("test", [_RoutableSkill()])

        call_kwargs = mock_create.call_args[1]
        prompt_content: str = call_kwargs["messages"][0]["content"]
        assert "topic A" in prompt_content


# ---------------------------------------------------------------------------
# select — error handling
# ---------------------------------------------------------------------------


class TestLLMSkillRouterErrorHandling:
    async def test_client_error_returns_empty_list(self) -> None:
        mock_client, _ = _make_error_client(RuntimeError("network error"))
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []

    async def test_client_error_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_client, _ = _make_error_client(RuntimeError("network error"))
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        with caplog.at_level(logging.WARNING, logger="openai_agents_skills.router"):
            await router.select("test message", [_RoutableSkill()])

        assert any("SkillRouter.select failed" in r.getMessage() for r in caplog.records)

    async def test_malformed_json_returns_empty_list(self) -> None:
        mock_client, _ = _make_mock_client("not-json{{{{")
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []

    async def test_malformed_json_no_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """_extract_json converts unrecognised input to '{}' silently — no warning expected."""
        mock_client, _ = _make_mock_client("not-json{{{{")
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        with caplog.at_level(logging.WARNING, logger="openai_agents_skills.router"):
            await router.select("test message", [_RoutableSkill()])

        assert not any("SkillRouter.select failed" in r.getMessage() for r in caplog.records)

    async def test_non_dict_json_returns_empty_list(self) -> None:
        """A JSON array at the top level is invalid — must return []."""
        mock_client, _ = _make_mock_client('["skill_a"]')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []

    async def test_non_dict_json_no_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """_extract_json finds no JSON object in a bare array — falls back to '{}' silently."""
        mock_client, _ = _make_mock_client('["skill_a"]')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        with caplog.at_level(logging.WARNING, logger="openai_agents_skills.router"):
            await router.select("test message", [_RoutableSkill()])

        assert not any("SkillRouter.select failed" in r.getMessage() for r in caplog.records)

    async def test_missing_selected_key_returns_empty_list(self) -> None:
        """JSON object without the ``selected`` key yields an empty result."""
        mock_client, _ = _make_mock_client('{"skills": ["skill_a"]}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []

    async def test_selected_value_is_non_list_returns_empty(self) -> None:
        """If ``selected`` is not a list, return empty without crashing."""
        mock_client, _ = _make_mock_client('{"selected": "skill_a"}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []


# ---------------------------------------------------------------------------
# select — caching
# ---------------------------------------------------------------------------


class TestLLMSkillRouterCaching:
    async def test_same_message_calls_llm_only_once(self) -> None:
        mock_client, mock_create = _make_mock_client('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")
        skills = [_RoutableSkill()]

        result1 = await router.select("identical message", skills)
        result2 = await router.select("identical message", skills)

        assert result1 == result2 == ["skill_a"]
        assert mock_create.call_count == 1

    async def test_cached_result_is_independent_copy(self) -> None:
        """Mutating the returned list must not corrupt the cache."""
        mock_client, mock_create = _make_mock_client('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")
        skills = [_RoutableSkill()]

        first = await router.select("msg", skills)
        first.append("injected")

        second = await router.select("msg", skills)

        assert "injected" not in second
        assert mock_create.call_count == 1

    async def test_different_messages_each_call_llm(self) -> None:
        mock_client, mock_create = _make_mock_client('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")
        skills = [_RoutableSkill()]

        await router.select("message one", skills)
        await router.select("message two", skills)

        assert mock_create.call_count == 2

    async def test_cache_eviction_removes_oldest_entry(self) -> None:
        """With cache_size=2, the third distinct message evicts the first."""
        mock_client, mock_create = _make_mock_client('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini", cache_size=2)
        skills = [_RoutableSkill()]

        await router.select("msg1", skills)  # cache: {msg1}          — call count: 1
        await router.select("msg2", skills)  # cache: {msg1, msg2}    — call count: 2
        await router.select("msg3", skills)  # evict msg1; cache: {msg2, msg3} — count: 3

        call_count_after_three = mock_create.call_count
        assert call_count_after_three == 3

        # msg1 was evicted — must trigger a fresh LLM call.
        await router.select("msg1", skills)
        assert mock_create.call_count == call_count_after_three + 1

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
        mock_client, mock_create = _make_mock_client('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini", cache_size=2)
        skills = [_RoutableSkill()]

        await router.select("msg1", skills)  # step 1
        await router.select("msg2", skills)  # step 2
        await router.select("msg1", skills)  # step 3: hit — promotes msg1
        await router.select("msg3", skills)  # step 4: evicts msg2

        assert mock_create.call_count == 3

        count_before = mock_create.call_count  # = 3

        # msg1 was promoted — it must still be in cache; no new client call.
        await router.select("msg1", skills)
        assert mock_create.call_count == count_before

        # msg2 was evicted in step 4 — must trigger a fresh client call.
        await router.select("msg2", skills)
        assert mock_create.call_count == count_before + 1

    async def test_cache_size_one_keeps_only_latest_message(self) -> None:
        mock_client, mock_create = _make_mock_client('{"selected": []}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini", cache_size=1)
        skills = [_RoutableSkill()]

        await router.select("alpha", skills)  # cache: {alpha}
        await router.select("beta", skills)  # evict alpha; cache: {beta}

        count_before = mock_create.call_count

        # alpha evicted → new LLM call.
        await router.select("alpha", skills)
        assert mock_create.call_count == count_before + 1

        count_before2 = mock_create.call_count

        # After re-adding alpha, cache = {alpha}.  Beta was evicted.
        await router.select("beta", skills)
        assert mock_create.call_count == count_before2 + 1


class TestLLMSkillRouterCacheSizeZero:
    async def test_cache_size_zero_never_caches_results(self) -> None:
        """cache_size=0 disables the cache entirely; every call hits the LLM."""
        mock_client, mock_create = _make_mock_client('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini", cache_size=0)
        skills = [_RoutableSkill()]

        await router.select("same message", skills)
        await router.select("same message", skills)
        await router.select("same message", skills)

        # Every call is a cache miss because caching is disabled.
        assert mock_create.call_count == 3

    async def test_cache_size_zero_does_not_raise(self) -> None:
        """cache_size=0 must not raise KeyError or any other exception."""
        mock_client, _ = _make_mock_client('{"selected": []}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini", cache_size=0)
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
# LLMSkillRouter — response_format flag
# ---------------------------------------------------------------------------


class TestLLMSkillRouterResponseFormat:
    async def test_default_does_not_pass_response_format(self) -> None:
        """By default, response_format must NOT be sent to the client."""
        mock_client, mock_create = _make_mock_client('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")

        await router.select("test message", [_RoutableSkill()])

        call_kwargs = mock_create.call_args[1]
        assert "response_format" not in call_kwargs

    async def test_use_response_format_true_passes_json_object(self) -> None:
        """When use_response_format=True, response_format is forwarded to the client."""
        mock_client, mock_create = _make_mock_client('{"selected": ["skill_a"]}')
        router = LLMSkillRouter(
            client=mock_client, model="gpt-4o-mini", use_response_format=True
        )

        await router.select("test message", [_RoutableSkill()])

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    async def test_provider_error_on_response_format_returns_empty_list(self) -> None:
        """Simulates a provider raising UnsupportedParamsError for response_format."""
        mock_client, _ = _make_error_client(
            Exception("bedrock does not support parameters: ['response_format']")
        )
        router = LLMSkillRouter(
            client=mock_client, model="some-bedrock-model", use_response_format=True
        )

        result = await router.select("test message", [_RoutableSkill()])

        assert result == []


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
        mock_client, _ = _make_mock_client(prose_response)

        class _BgpSkill(Skill):
            name = "bgp-troubleshooting"
            description = "BGP diagnostics."
            when_to_use = "Use for BGP issues."

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                return []

        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")
        result = await router.select("BGP session flapping", [_BgpSkill()])

        assert result == ["bgp-troubleshooting"]

    async def test_thinking_block_list_response_is_parsed_correctly(self) -> None:
        """Extended-thinking models return a list of content blocks — must parse."""
        thinking_content: list[Any] = [
            {"type": "thinking", "thinking": "Analysing the message..."},
            {"type": "text", "text": '{"selected": ["skill_a"]}'},
        ]

        mock_message = MagicMock()
        mock_message.content = thinking_content
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_create = AsyncMock(return_value=mock_response)
        mock_completions = MagicMock()
        mock_completions.create = mock_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        router = LLMSkillRouter(client=mock_client, model="gpt-4o-mini")
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
