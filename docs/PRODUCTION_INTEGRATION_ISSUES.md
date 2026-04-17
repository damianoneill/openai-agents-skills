# Production Integration Issues

Findings from integrating `openai-agents-skills` into a real production agent system
(JINO — a Juniper Networks network management AI using AWS Bedrock / Claude).
Each issue was encountered in live usage, not in theory.  Evidence, root cause,
and a concrete proposed fix are provided for each.

---

## Table of Contents

1. [Issue 1 — `LLMSkillRouter`: `response_format` breaks non-OpenAI providers](#issue-1--llmskillrouter-response_format-breaks-non-openai-providers)
2. [Issue 2 — `LLMSkillRouter`: fragile JSON parsing](#issue-2--llmskillrouter-fragile-json-parsing)
3. [Issue 3 — No built-in observability](#issue-3--no-built-in-observability)
4. [Issue 4 — Custom `SkillRouter` is undocumented as an integration path](#issue-4--custom-skillrouter-is-undocumented-as-an-integration-path)
5. [Issue 5 — Manifest re-injected on every call (suspected bug)](#issue-5--manifest-re-injected-on-every-call-suspected-bug)

---

## Issue 1 — `LLMSkillRouter`: `response_format` breaks non-OpenAI providers

### Severity
**Blocking** — the router silently fails and no device skills inject on every turn.

### Evidence
```
WARNING JinoSkillRouter.select failed; returning [].
Error: litellm.UnsupportedParamsError: bedrock does not support parameters:
['response_format'], for model=arn:aws:bedrock:us-west-2:...:application-inference-profile/...
```

### Root cause
`LLMSkillRouter.select` passes `response_format={"type": "json_object"}` to
`client.chat.completions.create`.  This is an OpenAI Chat Completions parameter.
Bedrock (via LiteLLM), Azure in some configurations, and Ollama do not support it.

LiteLLM raises `UnsupportedParamsError` rather than silently dropping the parameter.
`LLMSkillRouter.select` catches all exceptions and returns `[]`, so every routing call
fails and all routable skills are silently skipped for the entire session.  The user
sees no error and no device skills inject — indistinguishable from the router returning
an empty selection.

### Affected code
`src/openai_agents_skills/router.py` — `LLMSkillRouter.select`:
```python
response = await self._client.chat.completions.create(
    model=self._model,
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},   # <-- breaks Bedrock, some Azure configs
)
```

### Proposed fix
Remove `response_format` entirely.  The routing prompt already instructs the model to
return `{"selected": [...]}` JSON — capable models (GPT-4o, Claude 3+, Gemini 1.5+)
follow this reliably without a format hint.  `response_format` is only needed for weak
models that ignore prompt instructions.

```python
response = await self._client.chat.completions.create(
    model=self._model,
    messages=[{"role": "user", "content": prompt}],
    # No response_format — the prompt instruction is sufficient for capable models
    # and response_format is not supported by all providers (e.g. Bedrock).
)
```

**Alternative** (if `response_format` is desired for OpenAI): add a constructor flag
`use_response_format: bool = False` and only pass it when explicitly enabled.  Default
should be `False` for maximum provider compatibility.

### Testing
Add a test that constructs an `LLMSkillRouter` with a mock client that raises
`Exception("response_format not supported")` and verifies routing falls back to `[]`
— but more importantly, add a test that verifies the call is made *without*
`response_format` in the default case.

---

## Issue 2 — `LLMSkillRouter`: fragile JSON parsing

### Severity
**Blocking** — identical silent failure mode as Issue 1; all routable skills skipped.

### Evidence
```
WARNING JinoSkillRouter.select failed; returning [].
Error: Expecting value: line 1 column 1 (char 0)
```

### Root cause
After removing `response_format` (Issue 1), the model is free to return JSON wrapped
in prose, or in thinking blocks (Claude extended-thinking models), or preceded by a
sentence.  Examples of real Claude responses to the routing prompt without
`response_format`:

```
Based on the user's message about switch port issues, I'll select the relevant skills:

{"selected": ["switch-diagnostics"]}
```

```
{"selected": []}
```

The current parsing code is:
```python
content = response.choices[0].message.content or "{}"
data: Any = json.loads(content)
```

`json.loads` on `"Based on the user's message..."` raises `JSONDecodeError`.  The
`or "{}"` guard only handles `None` and empty string — it does not help when the model
returns non-empty non-JSON text.

Additionally, Claude extended-thinking models (e.g. claude-haiku-4-5, claude-sonnet-4-5)
may return `message.content` as a **list of content blocks** rather than a plain string:
```python
[
    {"type": "thinking", "thinking": "Let me analyse the message..."},
    {"type": "text", "text": '{"selected": ["router-diagnostics"]}'},
]
```
`json.loads` on a list raises `TypeError`.

### Affected code
`src/openai_agents_skills/router.py` — `LLMSkillRouter.select`, the content parsing
section.

### Proposed fix
Add a `_extract_json` helper that handles all observed response shapes and call it
instead of `json.loads(content or "{}")`:

```python
import re
from typing import Any


def _extract_json(content: Any) -> str:
    """Extract a JSON object from a model response.

    Handles:
    - None / empty string  → returns "{}"
    - Plain JSON string    → returned as-is
    - Prose + JSON         → extracts the first {...} block
    - List of content blocks (extended-thinking models) → joins text fields first
    """
    if not content:
        return "{}"

    # Thinking-block models return a list of {"type": "text"|"thinking", ...} dicts
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    if not isinstance(content, str) or not content.strip():
        return "{}"

    stripped = content.strip()

    # Fast path: response starts with { — treat as raw JSON
    if stripped.startswith("{"):
        return stripped

    # Slow path: find the first {...} object embedded in prose
    match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
    return match.group() if match else "{}"
```

Replace the parsing block in `select`:
```python
raw = response.choices[0].message.content
data: Any = json.loads(_extract_json(raw))
result: list[str] = [n for n in data.get("selected", []) if isinstance(n, str)]
```

### Testing
Unit-test `_extract_json` directly against all observed shapes:
- `None` → `"{}"`
- `""` → `"{}"`
- `'{"selected": ["ap"]}'` → unchanged
- `'Here are the skills: {"selected": ["ap"]}'` → `'{"selected": ["ap"]}'`
- `[{"type": "thinking", "thinking": "..."}, {"type": "text", "text": '{"selected": []}'}]`
  → `'{"selected": []}'`
- Non-JSON prose with no `{` → `"{}"`

---

## Issue 3 — No built-in observability

### Severity
**Moderate** — not a correctness issue but a significant barrier to production adoption.
Without observability, operators cannot verify routing is working or diagnose why the
wrong content is being injected.

### Evidence
To answer "which skills are being injected on this call?" in JINO, we had to:

1. Import a private API: `from openai_agents_skills._state import _get_run_state`
2. Subclass `SkillHooks` and override `on_llm_start`
3. Snapshot `state.injected_this_call` before and after `_do_llm_start`
4. Compute the diff and log it

This is ~20 lines of boilerplate reaching into library internals.  Any production
deployment will need this and will write roughly the same code.

### Root cause
`RunState.injected_this_call` is the ground truth for which skills injected on a
given call, but `RunState` and `_get_run_state` are private (`_` prefix, not exported).
There is no callback or hook for skill injection events.

### Proposed fix

**Option A (preferred): add `on_skill_injected` callback to `SkillHooks`**

Mirror the existing `on_skill_error` callback pattern:

```python
class SkillHooks(AgentHooks, _SkillInjectionMixin):
    def __init__(
        self,
        skills: list[Skill] | None = None,
        registry: SkillRegistry | None = None,
        routing_context_turns: int | None = 1,
        on_skill_error: Callable[[Skill, Exception], None] | None = None,
        on_skill_injected: Callable[[list[Skill], int], None] | None = None,
        max_manifest_skills: int | None = None,
    ) -> None: ...
```

`on_skill_injected(skills, blocks_added)` is called after each `on_llm_start` with
the list of `Skill` instances that were actually injected (returned non-empty blocks)
and the number of blocks prepended.  This covers the key use case:

```python
import logging
logger = logging.getLogger(__name__)

hooks = SkillHooks(
    skills=[...],
    registry=registry,
    on_skill_injected=lambda skills, n: logger.info(
        "Skills injected: %s (%d blocks)", [s.name for s in skills], n
    ),
)
```

**Option B: make `get_run_state` public**

Export `get_run_state()` (removing the `_` prefix) so that subclasses can access
`RunState` without reaching into private APIs.  More flexible but exposes more surface
area and still requires boilerplate subclassing.

Option A is the better default.  Option B can be offered as an escape hatch for
advanced use cases.

### Implementation notes

The callback receives `list[Skill]` (instances), not just names, so callers can inspect
`skill.name`, `skill.description`, `skill.when_to_use`, etc.  The `blocks_added` count
allows callers to distinguish between "skill ran but returned `[]`" (blocks_added=0)
and "skill injected content" (blocks_added>0).

To determine `blocks_added` accurately, `_do_llm_start` should snapshot
`len(input_items)` before injection and compute the delta after.  This is already done
in the JINO workaround and is straightforward to add.

The callback should also apply to `RunSkillHooks` (same `_SkillInjectionMixin` base,
same change).

---

## Issue 4 — Custom `SkillRouter` is undocumented as an integration path

### Severity
**Low** (documentation gap) — but causes significant wasted effort for any non-OpenAI
deployment.  Without documentation, engineers assume `LLMSkillRouter` is the only
option and spend time trying to make it work with their provider rather than
implementing the protocol directly.

### Evidence
JINO required a custom `JinoSkillRouter` (~100 lines) to route via AWS Bedrock.  The
`SkillRouter` Protocol exists and is the right abstraction, but:

- The README only mentions `LLMSkillRouter` by name
- No example shows how to implement `SkillRouter` directly
- The boilerplate (prompt template, JSON extraction, LRU cache) is not shared — every
  implementor re-writes it

### What a custom router involves
A working custom `SkillRouter` implementation needs:
1. The routing prompt template (same as `LLMSkillRouter`'s `_ROUTER_PROMPT_TEMPLATE`)
2. A robust JSON extractor (see Issue 2)
3. An LRU cache (same pattern as `LLMSkillRouter._cache`)
4. Provider-specific completion call (the only genuinely custom part)

Items 1, 2, and 3 are identical across all implementations.

### Proposed fix

**Option A (documentation only):** Add a "Custom SkillRouter" section to the README
with a minimal working example:

```python
import json
from collections import OrderedDict
from openai_agents_skills import Skill

class MyCustomRouter:
    """Minimal custom SkillRouter using any async completion API."""

    _PROMPT = (
        "You are a skill router. Given a user message, select which skills apply.\n"
        'Return JSON: {{"selected": ["skill_name", ...]}}.\n'
        "Only select skills clearly relevant to the message.\n\n"
        "User message: {message}\n\nAvailable skills:\n{manifest}"
    )

    def __init__(self, cache_size: int = 256) -> None:
        self._cache: OrderedDict[str, list[str]] = OrderedDict()
        self._cache_size = cache_size

    async def select(self, message: str, skills: list[Skill]) -> list[str]:
        if message in self._cache:
            self._cache.move_to_end(message)
            return list(self._cache[message])

        manifest = "\n".join(
            f"- {s.name}: {s.description}\n  When to use: {s.when_to_use}"
            for s in skills if s.when_to_use
        )
        prompt = self._PROMPT.format(message=message, manifest=manifest)

        try:
            raw = await self._call_model(prompt)   # implement this
            data = json.loads(_extract_json(raw))  # _extract_json from Issue 2
            result = [n for n in data.get("selected", []) if isinstance(n, str)]
        except Exception:
            return []

        if self._cache_size > 0:
            if len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)
            self._cache[message] = result
        return list(result)

    async def _call_model(self, prompt: str) -> str:
        raise NotImplementedError
```

**Option B (library support):** Extract the shared boilerplate into a
`BaseSkillRouter` class that handles the prompt, JSON extraction, and LRU cache.
Subclasses implement only `_call_model(prompt: str) -> str`.

`LLMSkillRouter` becomes a thin subclass:
```python
class LLMSkillRouter(BaseSkillRouter):
    def __init__(self, client: Any, model: str = "gpt-4o-mini", ...) -> None: ...

    async def _call_model(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
```

Option B is more engineering work but eliminates the repeated boilerplate for every
custom implementation and makes `LLMSkillRouter` itself simpler.  Worth doing alongside
Issue 2 since `_extract_json` would live in `BaseSkillRouter`.

---

## Issue 5 — Manifest re-injected on every call (suspected bug)

### Severity
**Low** — minor token waste, does not break functionality.  Needs investigation to
confirm whether it is a bug or a measurement artefact.

### Evidence
The JINO logging hook (`_LoggingSkillHooks`) measures `blocks_added` as
`len(input_items) - before_len` after `_do_llm_start`.  For a conversation with 4
active skills (each returning 1 block), 5 blocks were observed on **every** LLM call
throughout the session — not just the first call.

Expected behaviour: `_maybe_build_manifest_blocks` sets `state.manifest_injected = True`
after the first call and returns `[]` on all subsequent calls, giving 4 blocks from
call 2 onwards.

### Potential causes

**Cause A — `RunState` reset between calls**: If `RunState` is being re-created (e.g.
`_run_state` ContextVar is reset) between tool-call LLM invocations within the same
`Runner.run()`, `manifest_injected` would be `False` at the start of each call.

**Cause B — Measurement artefact**: The logging hook snapshots `before_len =
len(input_items)` before calling `_do_llm_start`.  If the SDK passes a fresh
`input_items` list (not the accumulated conversation) on each invocation, and the
manifest block from the previous call is not in scope, `before_len` would be 0 each
time and the 5-block count would reflect all injected blocks including the manifest on
every call — even if `manifest_injected` is working correctly.

**Cause C — Double `on_llm_start` firing**: Both `agent.hooks.on_llm_start` and
`run_hooks.on_llm_start` fire on each LLM call via `asyncio.gather`.  If `SkillHooks`
is set as the agent hook AND a separate `RunHooks` is also active (e.g. JINO's audit
log hooks), the `RunState` initialisation in `on_start` / `on_agent_start` may not
fire for the second hook instance, causing `_get_run_state()` to lazily create a
new `RunState` mid-run.

### Investigation steps
1. Add a debug log inside `_maybe_build_manifest_blocks` that prints
   `state.manifest_injected` on every call to confirm whether it's being reset.
2. Check whether the `id()` of the `RunState` object is the same across all
   `on_llm_start` calls within one `Runner.run()` — if it differs, the ContextVar
   is being reset.
3. Verify that `on_start` (for `SkillHooks`) fires before the first `on_llm_start`
   in the session under JINO's hook composition (`CompositeRunHooks`).

### Proposed fix
Depends on root cause.  If Cause A or C: ensure `_get_run_state()` is called in the
parent asyncio task before any `asyncio.gather` that fires `on_llm_start`.  The
existing `on_start` / `on_agent_start` hooks do this, but they may not fire when JINO
wraps `SkillHooks` in a subclass or composes hooks externally.  Making `on_start`
defensive (call `_get_run_state()` at the start of `on_llm_start` as well as in
`on_start`) would eliminate the race condition.

---

## Summary table

| # | Issue | Severity | Fix complexity |
|---|-------|----------|----------------|
| 1 | `response_format` breaks Bedrock / non-OpenAI providers | Blocking | Low — remove one parameter |
| 2 | Fragile JSON parsing (prose, thinking blocks) | Blocking | Low — add `_extract_json` helper |
| 3 | No built-in observability (`on_skill_injected` callback) | Moderate | Medium — new callback + threading through `_do_llm_start` |
| 4 | Custom `SkillRouter` undocumented | Low | Low (docs) / Medium (BaseSkillRouter) |
| 5 | Manifest may re-inject on every call | Low | Unknown — needs investigation first |

Issues 1 and 2 should be fixed together since they both affect `LLMSkillRouter` and
the fix for Issue 2 (`_extract_json`) is also the right foundation for the
`BaseSkillRouter` proposed in Issue 4.
