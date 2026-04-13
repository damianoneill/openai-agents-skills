"""Skill routing — SkillRouter protocol and LLMSkillRouter default implementation."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI

from .skills import Skill

_log = logging.getLogger(__name__)

_ROUTER_PROMPT_TEMPLATE = (
    "You are a skill router. Given a user message, select which skills (if any) apply.\n"
    'Return JSON: {{"selected": ["skill_name", ...]}}.\n'
    "Only select skills clearly relevant to the message. "
    'Return {{"selected": []}} if none apply.\n\n'
    "User message: {message}\n\n"
    "Available skills:\n{manifest}"
)


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
            skills: Candidate skills — typically only those with a non-empty
                ``when_to_use`` are passed.

        Returns:
            A list of skill names from *skills* that should be activated.
        """
        ...


class LLMSkillRouter:
    """Default routing implementation that uses an LLM to select relevant skills.

    Sends the user message and a skill manifest (name + description + ``when_to_use``
    for each routable skill) to an LLM via ``chat.completions.create``.  Parses the
    ``{"selected": [...]}`` JSON response and returns skill names.

    Results are cached per message string using an in-process LRU cache, so repeated
    routing of identical messages within a session avoids redundant LLM calls.

    Uses ``response_format={"type": "json_object"}`` for maximum provider
    compatibility — works with OpenAI, LiteLLM, Bedrock, Vertex, Ollama, and any
    other OpenAI-compatible endpoint.

    Args:
        client: An ``AsyncOpenAI`` or AsyncOpenAI-compatible async client.
        model: Model identifier the client accepts.  Default: ``"gpt-4o-mini"``.
        cache_size: Maximum number of (message → names) entries in the LRU cache.
            Default: 256.

    Example::

        from openai import AsyncOpenAI
        from openai_agents_skills import LLMSkillRouter, SkillRegistry

        router = LLMSkillRouter(client=AsyncOpenAI(), model="gpt-4o-mini")
        registry = SkillRegistry(router=router)
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        cache_size: int = 256,
    ) -> None:
        self._client = client
        self._model = model
        self._cache_size = cache_size
        # OrderedDict used as an LRU cache: oldest entry at the front.
        self._cache: OrderedDict[str, list[str]] = OrderedDict()

    async def select(self, message: str, skills: list[Skill]) -> list[str]:
        """Select skills relevant to *message*.

        Skills with an empty ``when_to_use`` are excluded from the manifest sent
        to the LLM — they are always-on and need no routing signal.

        On any exception (network error, JSON parse failure, unexpected response
        shape) logs a WARNING and returns ``[]`` so the run continues with
        unconditional skills only.

        Args:
            message: The routing context string.
            skills: Candidate skill instances.

        Returns:
            List of skill names the LLM selected, or ``[]`` on any error.
        """
        if message in self._cache:
            self._cache.move_to_end(message)
            return list(self._cache[message])

        routable = [s for s in skills if s.when_to_use]
        if not routable:
            return []

        manifest = "\n".join(
            f"- {s.name}: {s.description}\n  When to use: {s.when_to_use}" for s in routable
        )
        prompt = _ROUTER_PROMPT_TEMPLATE.format(message=message, manifest=manifest)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            data: object = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got {type(data).__name__}")
            names = data.get("selected", [])
            if not isinstance(names, list):
                names = []
            result: list[str] = [n for n in names if isinstance(n, str)]
        except Exception as exc:
            _log.warning(
                "LLMSkillRouter.select failed; returning []. Error: %s",
                exc,
            )
            return []

        # Store result; evict the oldest entry when the cache is full.
        # Guard against cache_size=0 (disabled caching) to avoid popping an empty dict.
        if self._cache_size > 0:
            if len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)
            self._cache[message] = result
        return list(result)
