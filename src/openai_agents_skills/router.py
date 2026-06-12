"""Skill routing — SkillRouter protocol, BaseSkillRouter, and LLMSkillRouter."""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .skills import Skill

if TYPE_CHECKING:
    from agents.model_settings import ModelSettings

_log = logging.getLogger(__name__)

# Name of the tracing span emitted for each routing decision.
_ROUTING_SPAN_NAME = "skill_routing"


def _routing_span() -> AbstractContextManager[Any]:
    """Return a tracing-span context manager for a single routing decision.

    Uses the openai-agents SDK ``custom_span`` so routing decisions appear in the
    host application's trace, alongside the agent/chat spans of the run that
    triggered routing.  Routing is otherwise invisible: the internal model call
    in :class:`LLMSkillRouter` runs with tracing disabled, and cache hits make no
    model call at all, so without this span a silent router failure is
    indistinguishable from "correctly selected nothing".

    When no trace is active — as in unit tests, or when the host has not
    configured SDK tracing — a null context is returned instead. This avoids the
    SDK's "No active trace" error log and adds zero overhead, so routing only
    emits spans when the caller has already opted into tracing.
    """
    from agents.tracing import custom_span, get_current_trace

    if get_current_trace() is None:
        return nullcontext(None)
    return custom_span(_ROUTING_SPAN_NAME)


def _record_routing(
    span: Any | None,
    *,
    candidates: list[str],
    selected: list[str],
    cache_hit: bool,
    error: Exception | None = None,
) -> None:
    """Attach routing-decision attributes to *span*.

    No-op when *span* is ``None`` (no active trace). Records the candidate skills
    considered, those selected and rejected, whether the result came from the LRU
    cache, and — on failure — the error so silent fallbacks remain queryable.
    """
    if span is None:
        return
    data = span.span_data.data
    data["candidates"] = candidates
    data["candidate_count"] = len(candidates)
    data["selected"] = selected
    data["selected_count"] = len(selected)
    data["rejected"] = [name for name in candidates if name not in selected]
    data["cache_hit"] = cache_hit
    if error is not None:
        data["error"] = str(error)
        data["error_type"] = type(error).__name__


_ROUTER_PROMPT_TEMPLATE = (
    "You are a skill router. Given a user message, select which skills (if any) apply.\n"
    'Return JSON: {{"selected": ["skill_name", ...]}}.\n'
    "Only select skills clearly relevant to the message. "
    'Return {{"selected": []}} if none apply.\n\n'
    "User message: {message}\n\n"
    "Available skills:\n{manifest}"
)


def _extract_json(content: Any) -> str:
    """Extract a JSON object string from a model response.

    Handles all observed response shapes:

    - ``None`` or empty string returns ``"{}"``
    - Plain JSON string starting with ``{`` is returned as-is
    - Prose-wrapped JSON (e.g. ``"Here are the results: {...}"``) has the first
      ``{...}`` block extracted
    - List of content blocks (extended-thinking models such as Claude) has the
      ``"text"`` fields joined first, then the JSON object extracted

    Note: the extraction uses a non-recursive regex and will not match nested
    JSON objects.  This is intentional — the expected routing response shape
    ``{"selected": ["name1", "name2"]}`` has no nested objects.

    Args:
        content: Raw ``message.content`` from a chat completions response.

    Returns:
        A string suitable for ``json.loads``.  Always returns at least ``"{}"``.
    """
    if not content:
        return "{}"

    # Extended-thinking models (e.g. Claude Haiku/Sonnet) may return a list of
    # typed content blocks: [{"type": "thinking", ...}, {"type": "text", "text": "..."}]
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )

    if not isinstance(content, str) or not content.strip():
        return "{}"

    stripped = content.strip()

    # Fast path: response is already raw JSON
    if stripped.startswith("{"):
        return stripped

    # Slow path: JSON object embedded in surrounding prose
    match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
    return match.group() if match else "{}"


@runtime_checkable
class SkillRouter(Protocol):
    """Protocol for skill routing strategies.

    Any object that implements :meth:`select` satisfies this protocol and can be
    used as a ``router`` argument to
    :class:`~openai_agents_skills.registry.SkillRegistry`.
    """

    async def select(
        self,
        message: str,
        skills: list[Skill],
    ) -> list[str]:
        """Return names of skills to activate for this message.

        Args:
            message: The routing context string (user message or multi-turn summary).
            skills: Candidate skills — typically only those with
                ``always_on=False`` are passed.

        Returns:
            A list of skill names from *skills* that should be activated.
        """
        ...


class BaseSkillRouter:
    """Base class for custom skill routing implementations.

    Provides the shared routing logic — prompt building from the skill manifest,
    robust JSON extraction, and an LRU cache — so that subclasses only need to
    implement :meth:`_call_model`.

    This is the recommended starting point when integrating with a non-OpenAI
    provider (AWS Bedrock, Azure OpenAI, Vertex AI, Ollama, etc.).  Subclass
    :class:`BaseSkillRouter`, implement :meth:`_call_model`, and pass an
    instance to :class:`~openai_agents_skills.registry.SkillRegistry`.

    Args:
        cache_size: Maximum number of ``(message to names)`` entries kept in the
            in-process LRU cache.  Pass ``0`` to disable caching entirely.
            Default: 256.

    Example::

        from openai_agents_skills import BaseSkillRouter

        class MyBedrockRouter(BaseSkillRouter):
            def __init__(self, bedrock_client, model_id: str) -> None:
                super().__init__()
                self._client = bedrock_client
                self._model_id = model_id

            async def _call_model(self, prompt: str) -> str:
                response = await self._client.invoke_model(
                    modelId=self._model_id, body={"prompt": prompt}
                )
                return response["body"]
    """

    def __init__(self, cache_size: int = 256) -> None:
        self._cache_size = cache_size
        self._cache: OrderedDict[tuple[str, frozenset[str]], list[str]] = OrderedDict()

    async def select(self, message: str, skills: list[Skill]) -> list[str]:
        """Select skills relevant to *message*.

        Skills with ``always_on=True`` are excluded from the manifest —
        they inject unconditionally and need no routing signal.

        On any exception (network error, JSON parse failure, unexpected response
        shape) logs a WARNING and returns ``[]`` so the run continues with
        unconditional skills only.

        When an SDK trace is active, the decision (candidates, selected,
        rejected, cache-hit, and any error) is recorded on a ``skill_routing``
        custom span so routing is observable in the host application's trace.

        Args:
            message: The routing context string.
            skills: Candidate skill instances.

        Returns:
            List of skill names the model selected, or ``[]`` on any error.
        """
        routable = [s for s in skills if not s.always_on]
        # Built unconditionally (not lazily behind a span check): the names are
        # load-bearing for the cache key below, not just telemetry.
        candidates = [s.name for s in routable]
        # Key the cache on the candidate set as well as the message, not the
        # message alone. The same routing context can be routed for agents whose
        # routable skills differ (``Skill.is_enabled`` gates per agent/context
        # upstream of the router). Keying on message alone would return one
        # agent's selection for another's distinct candidate set, silently
        # skipping the router for the second and never selecting its skills.
        cache_key = (message, frozenset(candidates))

        with _routing_span() as span:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                cached = list(self._cache[cache_key])
                _record_routing(span, candidates=candidates, selected=cached, cache_hit=True)
                return cached

            if not routable:
                _record_routing(span, candidates=candidates, selected=[], cache_hit=False)
                return []

            manifest = "\n".join(f"- {s.name}: {s.description}" for s in routable)
            prompt = _ROUTER_PROMPT_TEMPLATE.format(message=message, manifest=manifest)

            try:
                raw = await self._call_model(prompt)
                data: Any = json.loads(_extract_json(raw))
                if not isinstance(data, dict):
                    raise ValueError(f"Expected JSON object, got {type(data).__name__}")
                names = data.get("selected", [])
                if not isinstance(names, list):
                    names = []
                result: list[str] = [n for n in names if isinstance(n, str)]
            except Exception as exc:
                _log.warning(
                    "SkillRouter.select failed; returning []. Error: %s",
                    exc,
                )
                _record_routing(
                    span, candidates=candidates, selected=[], cache_hit=False, error=exc
                )
                return []

            if self._cache_size > 0:
                if len(self._cache) >= self._cache_size:
                    self._cache.popitem(last=False)
                self._cache[cache_key] = result
            _record_routing(span, candidates=candidates, selected=list(result), cache_hit=False)
            return list(result)

    async def _call_model(self, prompt: str) -> str:
        """Call the backing model with *prompt* and return the raw text response.

        Override this method in subclasses to integrate with any model provider.
        The return value is processed by :func:`_extract_json` before JSON
        parsing, so prose-wrapped or thinking-block responses are handled
        automatically.

        Args:
            prompt: The complete routing prompt, including the skill manifest.

        Returns:
            The raw model response as a plain string.  May contain prose before
            or after the JSON object — :func:`_extract_json` will handle it.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _call_model(prompt: str) -> str"
        )


class LLMSkillRouter(BaseSkillRouter):
    """Skill router backed by any openai-agents SDK ``Model`` instance.

    The default routing implementation.  Sends the user message and a skill
    manifest to the configured model via the SDK's ``Model.get_response()``
    interface and parses the ``{"selected": [...]}`` JSON response.  Results
    are cached using an in-process LRU cache keyed on the message and its
    candidate skill set, so repeated routing of an identical message against the
    same candidates avoids redundant LLM calls while a differing candidate set
    (e.g. another agent's routable skills) still routes afresh.

    Since ``openai-agents`` is a required dependency, pass the same ``Model``
    instance you use for your ``Agent`` -- no separate client configuration is
    needed.  Any SDK model works: ``OpenAIChatCompletionsModel``,
    ``LitellmModel``, ``AnyLLMModel``, and so on.

    For integrations not covered by any SDK model implementation, see
    :class:`BaseSkillRouter`, which lets you implement just the model call
    while reusing the prompt building, JSON extraction, and caching logic.

    Args:
        model: Any openai-agents SDK ``Model`` instance.  Pass the same model
            you give your ``Agent`` for zero extra configuration.
        model_settings: Optional ``ModelSettings`` to use for routing calls.
            When provided, these settings (including ``extra_args`` for AWS
            credentials, ``extra_body`` for inference profiles, etc.) are passed
            verbatim to every ``model.get_response()`` invocation — including
            whatever ``temperature`` they specify.  When ``None`` (the default),
            ``ModelSettings(temperature=0.0)`` is used so that routing is
            deterministic: skill selection is a classification task, and a
            temperature of ``0`` keeps results stable and consistent with the
            per-message LRU cache.
        cache_size: Maximum number of ``(message -> names)`` entries in the LRU
            cache.  Default: 256.

    Example::

        from agents import Agent
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        from openai import AsyncOpenAI
        from openai_agents_skills import LLMSkillRouter, SkillRegistry

        model = OpenAIChatCompletionsModel("gpt-4o-mini", AsyncOpenAI())
        agent = Agent(model=model, ...)
        router = LLMSkillRouter(model=model)
        registry = SkillRegistry(router=router)
    """

    def __init__(
        self,
        model: Any,
        model_settings: ModelSettings | None = None,
        cache_size: int = 256,
    ) -> None:
        super().__init__(cache_size=cache_size)
        self._model = model
        self._model_settings = model_settings

    async def _call_model(self, prompt: str) -> str:
        """Call the SDK Model and return the raw text response.

        Invokes ``model.get_response()`` with the routing prompt and no tools,
        handoffs, or output schema.  Tracing is disabled to avoid polluting the
        agent's trace with internal routing calls.  When no ``model_settings``
        was supplied to the router, a deterministic ``temperature=0.0`` setting
        is used.

        Args:
            prompt: The complete routing prompt.

        Returns:
            The first text block found in the model response, or ``""`` if the
            response contains no text content.
        """
        from agents.model_settings import ModelSettings
        from agents.models.interface import ModelTracing

        settings = (
            self._model_settings
            if self._model_settings is not None
            else ModelSettings(temperature=0.0)
        )
        response = await self._model.get_response(
            system_instructions=None,
            input=prompt,
            model_settings=settings,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
        for item in response.output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    return text
        return ""
