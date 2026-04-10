# openai-agents-skills: Implementation Plan

> **Status:** Phase 1 complete — Phase 2 next

---

## Table of Contents

1. [Design Overview](#design-overview)
2. [SDK Hook Anatomy](#sdk-hook-anatomy)
3. [The Injection Mechanism](#the-injection-mechanism)
4. [Phase 1 — Proto ✅](#phase-1--proto-)
5. [Phase 2 — Registry & Routing](#phase-2--registry--routing)
6. [Phase 3 — File-Based Skills](#phase-3--file-based-skills)
7. [Phase 4 — Advanced Triggering](#phase-4--advanced-triggering)
8. [Phase 5 — Security Hardening](#phase-5--security-hardening)
9. [Module Layout](#module-layout)
10. [Delivery Summary](#delivery-summary)

---

## Design Overview

A **Skill** is a named, reusable prompt fragment injected into the LLM's context at
the right moment in the agent loop. Skills are _instructions for the model_ — they
describe workflows, checklists, and procedures the model should follow, delivered as
additional content blocks prepended to the LLM's input before each call.

Skills are _not_ wrappers around the agent, tool bundles, or static configuration
applied at construction time. They are dynamic — evaluated and injected on every LLM
call via `AgentHooks.on_llm_start`.

| Concept           | Value                                                           |
| ----------------- | --------------------------------------------------------------- |
| What a Skill is   | A prompt fragment provider: `async get_prompt_blocks() -> list` |
| When it applies   | At each LLM call, decided dynamically                           |
| How it integrates | `agent = Agent(..., hooks=SkillHooks([skill1, skill2]))`        |
| Key SDK surface   | `AgentHooks.on_llm_start` → mutate `input_items`                |

---

## SDK Hook Anatomy

The OpenAI Agents SDK exposes two hook classes.

### `AgentHooks` — per-agent hooks

Set on `agent.hooks`. Fires only for that specific agent instance.

```python
class AgentHooks:
    async def on_start(context: AgentHookContext, agent: Agent) -> None: ...
    async def on_end(context: AgentHookContext, agent: Agent, output: Any) -> None: ...
    async def on_handoff(context: RunContextWrapper, agent: Agent, source: Agent) -> None: ...
    async def on_tool_start(context: RunContextWrapper, agent: Agent, tool: Tool) -> None: ...
    async def on_tool_end(context: RunContextWrapper, agent: Agent, tool: Tool, result: str) -> None: ...
    async def on_llm_start(context: RunContextWrapper, agent: Agent,
                           system_prompt: str | None,
                           input_items: list[TResponseInputItem]) -> None: ...
    async def on_llm_end(context: RunContextWrapper, agent: Agent,
                         response: ModelResponse) -> None: ...
```

### `RunHooks` — run-level hooks

Passed to `Runner.run(..., hooks=run_hooks)`. Fires for every agent involved in the
run, making it the right integration point for skills that should span a multi-agent
workflow without configuring each agent individually.

### Hook-to-Skill-Feature mapping

| SDK Hook                  | Skill feature                                                              |
| ------------------------- | -------------------------------------------------------------------------- |
| `AgentHooks.on_llm_start` | **Primary injection point** — prepend skill prompt blocks to `input_items` |
| `AgentHooks.on_start`     | Session-level setup; inject system prompt manifest on first turn           |
| `AgentHooks.on_tool_end`  | Tool-result triggers — activate skills after specific tools complete       |
| `AgentHooks.on_llm_end`   | Post-turn skills — quality checks, memory review                           |
| `AgentHooks.on_end`       | Session cleanup                                                            |
| `RunHooks`                | Run-level equivalent, spanning all agents in a run                         |

---

## The Injection Mechanism

Reading the SDK source (`run_loop.py`), the call sequence before each LLM invocation
is:

```
1. filtered = await maybe_filter_model_input(agent, input_items, system_prompt)
2. filtered.input = deduplicate_input_items_preferring_latest(filtered.input)
3. await asyncio.gather(
       run_hooks.on_llm_start(ctx, agent, filtered.instructions, filtered.input),
       agent.hooks.on_llm_start(ctx, agent, filtered.instructions, filtered.input),
   )
4. response = await model.get_response(input=filtered.input, ...)  # ← SAME list object
```

`filtered.input` is a **mutable Python list** passed by reference into `on_llm_start`
and then passed directly to `model.get_response()`. Items prepended inside the hook
are seen by the LLM on that call.

The injection pattern:

```python
async def on_llm_start(self, context, agent, system_prompt, input_items):
    all_blocks: list = []
    for skill in self._active_skills:
        blocks = await skill.get_prompt_blocks()
        all_blocks.extend(blocks)
    input_items[0:0] = all_blocks  # single prepend — preserves registration order
```

No monkey-patching. No subclassing of internal SDK types. No modifications to the
`Agent` object. A standard `AgentHooks` subclass writing to a mutable list.

---

## Phase 1 — Proto ✅

**Delivered:** Core injection mechanism. A skill is a prompt fragment provider;
`SkillHooks` injects enabled skills before every LLM call.

### Public API

```python
from openai_agents_skills import Skill, SkillProtocol, SkillHooks, skill
```

### `Skill` — base class

```python
class Skill:
    name: str = ""
    description: str = ""
    when_to_use: str = ""           # used by Phase 2 routing

    def is_enabled(self) -> bool:   # override to gate dynamically
        return True

    async def get_prompt_blocks(self, args: str = "") -> list:
        raise NotImplementedError
```

### `SkillProtocol` — structural typing

Any object with `name`, `description`, and `get_prompt_blocks` satisfies the protocol.
Subclassing `Skill` is not required.

### `SkillHooks` — the injection engine

```python
agent = Agent(
    name="Assistant",
    instructions="You are helpful.",
    hooks=SkillHooks([CitationSkill(), BulletSkill()]),
)
```

- Subclasses `AgentHooks`
- Prepends all enabled skills' blocks at `on_llm_start`
- `Skill` subclasses with `is_enabled() == False` are silently skipped
- Duck-typed objects satisfying `SkillProtocol` are always injected

### `@skill` decorator

Lightweight factory annotation for tooling and discovery (Phase 2+). Does not call
or register the factory.

```python
@skill(name="summariser", description="Summarise long documents.")
def make_summariser() -> Skill:
    return MySummariserSkill()
```

---

## Phase 2 — Registry & Routing

**Goal:** Skills are selected dynamically per-turn based on trigger conditions.
Introduce `SkillRegistry` — a routing layer that decides which skills fire and when.

### `SkillRegistry`

```python
# src/openai_agents_skills/registry.py

class SkillRegistry:
    def register(self, skill: Skill) -> None: ...
    def unregister(self, name: str) -> None: ...
    def get(self, name: str) -> Skill: ...

    @property
    def skill_names(self) -> list[str]: ...

    def get_unconditional(self) -> list[Skill]:
        """Skills whose is_enabled() returns True."""

    async def select_for_message(self, message: str) -> list[Skill]:
        """Return skills selected by the router for this message."""
```

### Skill routing via `SkillRouter`

`select_for_message` delegates to a pluggable `SkillRouter` — an async component that
decides which skills are relevant to a given user message using semantic understanding
rather than lexical matching.

#### `SkillRouter` protocol

```python
# router.py
class SkillRouter(Protocol):
    async def select(
        self,
        message: str,
        skills: list[SkillProtocol],
    ) -> list[str]:
        """Return names of skills to activate for this message."""
        ...
```

Any object satisfying this protocol can be injected. Two implementations ship by default.

#### `LLMSkillRouter` — default implementation

Sends the user message and a skill manifest (name + description + `when_to_use` for
each registered skill) to an LLM. Returns a JSON list of skill names to activate.

```python
class LLMSkillRouter:
    def __init__(
        self,
        client: AsyncOpenAI,        # AsyncOpenAI or any AsyncOpenAI-compatible client
        model: str = "gpt-4o-mini", # any model string the client accepts
    ) -> None: ...

    async def select(self, message: str, skills: list[SkillProtocol]) -> list[str]:
        # Build manifest from skills
        # Call client.chat.completions.create with JSON response_format
        # Parse {"selected": [...]} and return name list
        # On any error: log WARNING, return []
        ...
```

Uses `chat.completions.create` with `response_format={"type": "json_object"}` for
maximum provider compatibility — works with OpenAI, LiteLLM, Bedrock, Vertex, Ollama,
and any other OpenAI-compatible endpoint. Does not use the Responses API or
`beta.chat.completions.parse`.

**Prompt skeleton:**

```
You are a skill router. Given a user message, select which skills (if any) apply.
Return JSON: {"selected": ["skill_name", ...]}.
Only select skills clearly relevant to the message. Return {"selected": []} if none apply.

User message: {message}

Available skills:
{skill_manifest}
```

**Result caching:** responses cached by `message` string. Same message across turns
returns cached selection without an additional LLM call.

**Fallback on error:** any exception is caught, logged at WARNING, and returns `[]` —
the run continues with unconditional skills only.

**LiteLLM / Bedrock usage:**

```python
from openai import AsyncOpenAI
from openai_agents_skills import LLMSkillRouter, SkillRegistry

# OpenAI
router = LLMSkillRouter(client=AsyncOpenAI(), model="gpt-4o-mini")

# Bedrock/Claude via LiteLLM (pass litellm AsyncOpenAI-compatible client)
litellm_client = AsyncOpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="...",
)
router = LLMSkillRouter(
    client=litellm_client,
    model="anthropic/claude-3-5-haiku-20241022",
)

registry = SkillRegistry(router=router)
```

The `openai-agents` SDK's own LiteLLM integration (`set_litellm_model`) applies only
to the agent's primary model. The skill router uses its own injected client, so the
two are independent and can use different providers.

#### `NullSkillRouter` — no routing

Returns `[]` for every message. All skills are treated as unconditional. Useful in
tests and when routing is not desired.

```python
class NullSkillRouter:
    async def select(self, message: str, skills: list[SkillProtocol]) -> list[str]:
        return []
```

#### `SkillRegistry` with router

Router is optional. When omitted, `select_for_message` always returns `[]` and all
active skills come from `get_unconditional()`.

```python
class SkillRegistry:
    def __init__(self, router: SkillRouter | None = None) -> None: ...

    async def select_for_message(self, message: str) -> list[SkillProtocol]:
        if self._router is None:
            return []
        names = await self._router.select(message, list(self._skills.values()))
        return [self._skills[n] for n in names if n in self._skills]
```

Skills with an empty `when_to_use` string are never passed to the router — the router
manifest only includes skills that have declared routing intent.

### Routing-aware `SkillHooks`

`SkillHooks` is extended to accept a `SkillRegistry`. At `on_llm_start` it extracts
the last user message from `input_items` and uses `select_for_message` to identify
which skills to fire alongside the always-on unconditional set.

```python
class SkillHooks(AgentHooks):
    def __init__(
        self,
        skills: list[SkillProtocol] | None = None,
        registry: SkillRegistry | None = None,
    ) -> None: ...

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        base = self._registry.get_unconditional() if self._registry else self._skills
        last_user_msg = _extract_last_user_message(input_items)
        routed = await self._registry.select_for_message(last_user_msg) if self._registry else []
        all_blocks: list = []
        for skill in _deduplicate([*base, *routed]):
            blocks = await skill.get_prompt_blocks()
            all_blocks.extend(blocks)
        input_items[0:0] = all_blocks  # single prepend — preserves registration order
```

### System prompt manifest

On the first LLM call of each run, inject a concise manifest listing all registered
skills and their `when_to_use` descriptions into `input_items`, so the model is aware
of available capabilities even when automatic routing does not select them.

Guarded per `Runner.run()` invocation using `id(context)` as a proxy for run identity —
each `Runner.run()` call creates a new `RunContextWrapper` instance with a unique object
id. `SkillHooks` stores `_manifest_injected: set[int]` and injects only when
`id(context)` is not already in the set, then adds it. The set is never cleared —
entries accumulate across the lifetime of the `SkillHooks` instance, which is
intentional: if the same hooks object is reused across multiple `Runner.run()` calls,
each call gets exactly one manifest injection.

Note: `RunContextWrapper` does not expose a `run_id` field in the current SDK. `id(context)`
is safe here because the context object stays alive for the full duration of the run.

```
## Available Skills
- verify: Use when checking that changes work end-to-end.
  Example: "does this work", "check my changes", "verify"
- simplify: Use after implementation to review and clean up code.
  Example: "clean this up", "review the code"
```

### `invoke_skill` function tool

An optional `function_tool` the model can call explicitly for model-driven skill
selection. Users opt in by adding it to `agent.tools`:

```python
from openai_agents_skills import make_invoke_skill_tool

agent = Agent(
    name="Assistant",
    tools=[make_invoke_skill_tool(registry)],
    hooks=SkillHooks(registry=registry),
)
```

`make_invoke_skill_tool(registry)` returns a `@function_tool` that looks up
`skill_name` in the registry, calls `skill.get_prompt_blocks(args)`, and returns
the concatenated text as a tool result the model can act on.

### `is_enabled` gate

`Skill.is_enabled()` (already present in Phase 1) is used by `get_unconditional()`
to exclude skills that are disabled at call time.

### Error resilience in `get_prompt_blocks`

If `skill.get_prompt_blocks()` raises, the default behaviour is to let the exception
propagate, which would crash the entire agent run. Instead, `SkillHooks` wraps each
call in a try/except and logs a warning, allowing the run to continue with the
remaining skills.

```python
import logging
_log = logging.getLogger(__name__)

async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
    all_blocks: list = []
    for skill in _active_skills:
        try:
            blocks = await skill.get_prompt_blocks()
            all_blocks.extend(blocks)
        except Exception as exc:
            _log.warning(
                "Skill %r raised during get_prompt_blocks(); skipping. Error: %s",
                skill.name, exc,
            )
    input_items[0:0] = all_blocks
```

`SkillHooks` accepts an optional `on_skill_error` callback for callers that need
custom handling (e.g., re-raise in test environments):

```python
SkillHooks(
    skills=[...],
    on_skill_error=lambda skill, exc: None,   # swallow silently
)
```

The default callback logs at WARNING and continues. Passing
`on_skill_error=lambda skill, exc: (_ for _ in ()).throw(exc)` restores the
raise-on-error behaviour for tests.

Tests must cover: skill raising `ValueError` does not prevent other skills from
injecting; the `on_skill_error` callback is called with the skill and exception;
a raising skill's blocks are absent from `input_items`.

### `RunSkillHooks`

A `RunHooks` variant that applies skills across every agent in a multi-agent run
without requiring each `Agent` to be individually configured:

```python
from openai_agents_skills import RunSkillHooks

result = await Runner.run(
    triage_agent,
    input="...",
    hooks=RunSkillHooks(registry=registry),
)
```

**Double-injection guard.** The SDK fires both `run_hooks.on_llm_start` and
`agent.hooks.on_llm_start` via `asyncio.gather` for every LLM call. If a skill is
registered in both a `RunSkillHooks` and a per-agent `SkillHooks`, it will inject twice.
To prevent this, both hook classes use a module-level `dict[int, set[str]]` named
`_INJECTED_THIS_CALL` keyed by `id(input_items)`. Before injecting a skill, its `name`
is checked against the set for the current call; if present it is skipped; otherwise the
name is added and injection proceeds. The entry is removed in a `finally` block after all
skills have been processed, so stale IDs never accumulate. Because Python's async
event loop is single-threaded, no additional locking is needed for this dictionary.

Tests must cover: same skill registered in both `RunSkillHooks` and `SkillHooks` injects
only once; two _different_ skills registered separately each inject once.

### New modules

`src/openai_agents_skills/registry.py` — `SkillRegistry`.
`src/openai_agents_skills/router.py` — `SkillRouter` protocol, `LLMSkillRouter`, `NullSkillRouter`.

### Tests for Phase 2

- Registry: register / get / unregister, duplicate name overwrite, `skill_names` sorted
- `get_unconditional` excludes skills where `is_enabled()` returns `False`
- `NullSkillRouter.select` always returns `[]`
- `LLMSkillRouter.select` calls client with correct prompt and parses `{"selected": [...]}` response
- `LLMSkillRouter.select` returns `[]` and logs warning on client error (no crash)
- `LLMSkillRouter` caches: second call with same message skips client call
- `LLMSkillRouter` cache: different messages produce independent entries
- `SkillRegistry.select_for_message` returns `[]` when no router configured
- `SkillRegistry.select_for_message` delegates to router; unknown names in response silently skipped
- Skills with empty `when_to_use` excluded from router manifest
- `SkillHooks` with registry injects unconditional + routed skills
- `SkillHooks` deduplicates when skill appears in both unconditional and routed sets
- `invoke_skill` tool returns expected content; unknown skill name raises a clear error
- Manifest injected exactly once across multiple turns (guard on `id(context)`)
- `RunSkillHooks` injects for all agents in a two-agent handoff scenario

---

## Phase 3 — File-Based Skills

**Goal:** Load skills from `.agent/skills/<name>/SKILL.md` files on disk. Teams can
distribute skills as files alongside their projects without writing Python.

### Directory conventions

Two layers, in priority order (user wins on name conflict):

```
User (personal, cross-repo):
  ~/.agent/skills/<name>/SKILL.md

Project (repo-specific, checked in):
  <cwd>/.agent/skills/<name>/SKILL.md
```

Additional directories can be supplied via `SkillConfig.extra_dirs`.

### Frontmatter fields

| Field            | Type               | Required | Notes                                           |
| ---------------- | ------------------ | -------- | ----------------------------------------------- |
| `name`           | `str`              | No       | Overrides the directory name as the skill label |
| `description`    | `str`              | Yes      | Short summary; used in manifest and routing     |
| `when_to_use`    | `str`              | No       | Prose trigger description with example phrases  |
| `allowed-tools`  | `list[str]`        | No       | Informational in Phase 3; enforced in Phase 5   |
| `argument-hint`  | `str`              | No       | Human-readable hint, e.g. `"[branch] [env]"`    |
| `arguments`      | `list[str]`        | No       | Named args for `$arg_name` substitution in body |
| `context`        | `inline` \| `fork` | No       | `fork` implemented in Phase 4; default `inline` |
| `user-invocable` | `bool`             | No       | Default `true`; `false` hides from manifest     |

### `FileSkill`

A concrete `Skill` subclass built from a parsed SKILL.md. `get_prompt_blocks` returns
the Markdown body (after the frontmatter block) as a user-role message, with
`$arg_name` positional substitutions and caller-supplied `${VAR}` variable substitutions
applied (see `### Argument substitution`).

```python
class FileSkill(Skill):
    def __init__(self, fields: SkillFields, body: str, file_path: Path) -> None: ...

    async def get_prompt_blocks(self, args: str = "") -> list:
        body = substitute_args(self._body, args, self._arg_names, self._variables)
        return [{"role": "user", "content": body}]
```

### FileSkill prompt-block caching

`FileSkill.get_prompt_blocks` is called on every LLM invocation. The raw body is
already parsed at construction time and stored as `self._body`. Argument substitution
is cheap for small files, but caching avoids redundant string work on hot loops and
ensures identical `args` inputs always return the same list object reference for easy
equality testing.

Implementation:

```python
class FileSkill(Skill):
    def __init__(self, fields: SkillFields, body: str, file_path: Path) -> None:
        ...
        self._cache: dict[str, list[Any]] = {}

    async def get_prompt_blocks(self, args: str = "") -> list[Any]:
        if args not in self._cache:
            body = substitute_args(self._body, args, self._arg_names, self._variables)
            self._cache[args] = [{"role": "user", "content": body}]
        return self._cache[args]
```

The cache is keyed by the `args` string. It is instance-local and never invalidated
(file content is immutable once loaded — reload the registry to pick up changes).
Bundled `Skill` subclasses do not cache by default; override `get_prompt_blocks` with
the same pattern if a subclass has expensive construction.

Tests must cover: second call with identical `args` returns the cached list object
(`is` check); different `args` values produce independent cache entries; empty `args`
is cached separately from a non-empty value.

### Loaders

```python
async def load_skills_from_dir(
    dir_path: Path,
    source: SkillSource,
) -> list[tuple[Skill, Path]]:
    """Load all SKILL.md files from immediate subdirectories of dir_path."""

async def load_all_skills(
    cwd: Path,
    config: SkillConfig | None = None,
) -> SkillRegistry:
    """Load user and project skills, deduplicate, return populated registry."""
```

`load_all_skills`:

1. Resolves user dir (`~/.agent/skills/`) and project dir (`<cwd>/.agent/skills/`)
   plus any `config.extra_dirs`
2. Loads all dirs in parallel via `asyncio.gather`
3. Deduplicates by `realpath(file_path)` — canonical path, never inodes
4. User-layer skills win over project-layer on name conflict
5. Returns a `SkillRegistry` ready to pass to `SkillHooks`

### Argument substitution

```python
def substitute_args(
    template: str,
    raw_args: str,
    arg_names: list[str],
    variables: dict[str, str] | None = None,
) -> str:
    """Replace $arg_name, $1/$2/... positional, and ${VAR} caller-supplied variables."""
```

Substitution order:

1. **Named args:** split `raw_args` by whitespace, map positionally to `arg_names`,
   replace `$arg_name` throughout the body.
2. **Positional fallback:** replace `$1`, `$2`, … with the split parts.
3. **Caller-supplied variables:** replace `${KEY}` and `$KEY` patterns for each key in
   the `variables` dict (if provided). Common uses: `${RUN_ID}`, `${USER}`,
   `${PROJECT}`. The dict is passed down from `SkillConfig.variables`.

`variables` replaces the previously proposed `${SESSION_ID}` built-in. There is no
implicitly-resolved variable; callers supply whatever context they need:

```python
config = SkillConfig(variables={"RUN_ID": str(uuid.uuid4()), "USER": "alice"})
registry = await load_all_skills(cwd=Path.cwd(), config=config)
```

`SkillConfig` gains the field:

```python
@dataclass
class SkillConfig:
    extra_dirs: list[Path] = field(default_factory=list)
    user_dir: Path | None = None
    project_dir: Path | None = None
    variables: dict[str, str] = field(default_factory=dict)  # NEW
```

### `SkillConfig`

```python
@dataclass
class SkillConfig:
    extra_dirs: list[Path] = field(default_factory=list)
    user_dir: Path | None = None    # default: Path.home() / ".agent" / "skills"
    project_dir: Path | None = None # default: cwd / ".agent" / "skills"
```

### Typical usage

```python
from openai_agents_skills import load_all_skills, SkillHooks
from agents import Agent, Runner

registry = await load_all_skills(cwd=Path.cwd())

agent = Agent(
    name="Assistant",
    instructions="You are helpful.",
    hooks=SkillHooks(registry=registry),
)

result = await Runner.run(agent, "Run the payments workflow.")
```

### New modules

`src/openai_agents_skills/loader.py` — `load_skills_from_dir`, `load_all_skills`,
deduplication logic. `src/openai_agents_skills/substitution.py` — `substitute_args`.

### Tests for Phase 3

- `load_skills_from_dir` returns a `FileSkill` per subdirectory containing `SKILL.md`
- Subdirectories without `SKILL.md` are silently skipped
- Missing top-level directory is silently skipped (no error on fresh project)
- Frontmatter fields parsed correctly; missing optional fields use defaults
- `realpath` deduplication prevents loading the same file twice via symlinks
- User-layer skill wins over project-layer skill with the same name
- Argument substitution for named args, positional fallback, and caller-supplied `${VAR}` variables
- `substitute_args` with `variables={}` leaves `${UNKNOWN}` patterns unchanged
- `SkillConfig.variables` dict is threaded through to `substitute_args`
- `load_all_skills` returns a populated registry wired to `SkillHooks`

---

## Phase 4 — Advanced Triggering

**Goal:** Richer triggering patterns beyond always-on and message-routed injection.
Skills fire in response to tool results, after model turns, and in isolated sub-agents.

### `context: fork` — isolated sub-agent

Skills with `context: fork` run in a fully isolated `Runner.run()` call rather than
injecting inline into the current conversation. Useful for self-contained tasks that
should not be contaminated by the parent conversation history.

Implementation:

- When `SkillHooks` detects a triggered skill with `context == "fork"`, instead of
  prepending blocks to `input_items`, it schedules a nested `Runner.run()` call using
  a temporary `Agent` whose `instructions` contain only the skill's prompt body.
- The forked run receives the current user message as its input.
- The result is injected back into the parent conversation as a synthetic
  `function_call_output` item so the parent agent can act on it.

```python
fork_agent = Agent(
    name=f"skill:{skill.name}",
    instructions=skill_prompt_body,
)
fork_result = await Runner.run(fork_agent, input=last_user_message)
# inject fork_result.final_output back into parent input_items
```

### Tool-result triggers

Skills declare which tools should trigger them after completion. After a matching tool
finishes, the skill's prompt is queued for injection on the _next_ LLM call.

```python
class Skill:
    triggers_after_tools: list[str] = []
    # e.g. triggers_after_tools = ["write_file", "edit_file"]
```

Implementation in `SkillHooks`:

```python
async def on_tool_end(self, context, agent, tool, result) -> None:
    for skill in self._registry.get_triggered_by_tool(tool.name):
        self._pending.append(skill)

async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
    # ... existing unconditional + routed injection ...
    for skill in self._pending:
        blocks = await skill.get_prompt_blocks()
        input_items[0:0] = blocks
    self._pending.clear()
```

Injecting at the _next_ `on_llm_start` rather than the same turn avoids confusing
the model with mid-turn context changes.

**Concurrency safety.** `_pending` is mutated from both `on_tool_end` and
`on_llm_start`. In Python's asyncio single-threaded event loop these never run
truly in parallel, but if a `SkillHooks` instance is shared across a
`RunSkillHooks`-driven multi-agent run (multiple agents firing `on_tool_end`
concurrently via `asyncio.gather`), appends can interleave unpredictably. Guard
`_pending` with `asyncio.Lock`:

```python
def __init__(self, ...) -> None:
    ...
    self._pending: list[Skill] = []
    self._pending_lock = asyncio.Lock()

async def on_tool_end(self, context, agent, tool, result) -> None:
    async with self._pending_lock:
        for skill in self._registry.get_triggered_by_tool(tool.name):
            self._pending.append(skill)

async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
    async with self._pending_lock:
        pending_snapshot = list(self._pending)
        self._pending.clear()
    # inject snapshot outside the lock so get_prompt_blocks can await freely
    all_blocks: list = []
    for skill in pending_snapshot:
        blocks = await skill.get_prompt_blocks()
        all_blocks.extend(blocks)
    input_items[0:0] = all_blocks
```

The snapshot-outside-lock pattern is critical: holding the lock while calling
`await skill.get_prompt_blocks()` would deadlock if that skill itself triggers
another `on_tool_end`.

### Post-turn skills via `on_llm_end`

Skills can opt in to firing _after_ each model response, useful for quality checks,
memory consolidation, or review workflows.

```python
class Skill:
    triggers_after_turn: bool = False
```

In `on_llm_end`, skills with `triggers_after_turn = True` are queued in `_pending`
(same drain mechanism), causing them to inject at the start of the next turn.

### `SkillRegistry` additions for Phase 4

```python
class SkillRegistry:
    def get_triggered_by_tool(self, tool_name: str) -> list[Skill]:
        """Return skills whose triggers_after_tools includes tool_name."""

    def get_post_turn(self) -> list[Skill]:
        """Return skills with triggers_after_turn == True."""
```

### Tests for Phase 4

- `context: fork` spawns an isolated `Runner.run()` (verified via mock runner)
- Fork result injected into parent `input_items` as expected content
- `triggers_after_tools`: skill queued in `_pending` after matching tool fires
- `triggers_after_tools`: skill not queued when tool name does not match
- `_pending` queue is drained and cleared at next `on_llm_start`
- `triggers_after_turn`: skill queued in `_pending` after `on_llm_end`
- Multiple pending skills from different trigger sources all inject correctly
- `get_triggered_by_tool` returns empty list when no skills match
- `get_post_turn` returns only skills with the flag set
- `_pending_lock` prevents lost appends when two `on_tool_end` callbacks fire concurrently (use `asyncio.gather` over two mock tool completions in the test)

---

## Phase 5 — Security Hardening

**Goal:** The extension is safe, well-documented, and production-quality.

### Source trust levels

```python
class SkillSource(str, Enum):
    BUNDLED = "bundled"   # compiled into the package — fully trusted
    USER = "user"         # ~/.agent/skills — user-trusted
    PROJECT = "project"   # <cwd>/.agent/skills — project-trusted
    EXTRA = "extra"       # caller-supplied extra_dirs — caller-trusted
```

Trust level is stored on `FileSkill` and surfaced in log output. Reserved for future
enforcement (e.g. restricting certain features to higher-trust sources).

### Path traversal validation

When resolving SKILL.md file paths during loading, normalise and assert the resolved
path is still a descendant of the expected base directory. Any `name` or `arguments`
field value containing path separators or `..` is rejected at parse time.

```python
def assert_within_base(path: Path, base: Path) -> None:
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Path {path} escapes base directory {base}")
```

### Deduplication correctness

`realpath()` for canonical path resolution. Never use inode numbers — unreliable on
NFS, ExFAT, overlayfs, and container filesystems. First occurrence wins (user >
project > extra).

### Argument injection safety

`substitute_args` must not allow a skill body to inject content that looks like a
system message boundary. Substituted values are validated as plain text before
insertion into the prompt body.

### `allowed-tools` enforcement

The `allowed-tools` frontmatter field (informational in Phase 3) is enforced here.
When a `FileSkill` with `allowed-tools` is applied via `context: fork`, the forked
agent is constructed with only those tools available, regardless of what the parent
agent has.

### Coverage & documentation targets

- 90% test coverage enforced (already in CI)
- All public API symbols have Google-style docstrings
- README reflects the full Phase 5 API
- `CHANGELOG.md` updated per release

---

## Module Layout

```
src/openai_agents_skills/
    __init__.py          # Public API — grows with each phase
    _version.py
    py.typed
    skills.py            # Skill, SkillProtocol, @skill decorator           ✅ Phase 1
    hooks.py             # SkillHooks, RunSkillHooks — injection engine      ✅ Phase 1 / Phase 2
    registry.py          # SkillRegistry — routing, tool triggers            Phase 2 / Phase 4
    router.py            # SkillRouter protocol, LLMSkillRouter, NullSkillRouter  Phase 2
    loader.py            # load_skills_from_dir, load_all_skills, dedup      Phase 3
    substitution.py      # substitute_args, substitute_vars                  Phase 3
    bundled/
        __init__.py      # register_bundled_skills()
        # concrete Skill subclasses added as needed

tests/
    __init__.py
    test_version.py
    test_skills.py       # Skill, SkillProtocol, @skill                      ✅ Phase 1
    test_hooks.py        # SkillHooks injection                               ✅ Phase 1
    test_registry.py     # SkillRegistry routing, tool triggers               Phase 2 / Phase 4
    test_router.py       # LLMSkillRouter, NullSkillRouter, SkillRouter protocol  Phase 2
    test_loader.py       # SKILL.md loading, dedup, priority                  Phase 3
    test_substitution.py # argument and variable substitution                 Phase 3
    test_advanced.py     # fork, tool-result triggers, post-turn skills       Phase 4
    fixtures/
        .agent/
            skills/
                test-skill/
                    SKILL.md    # fixture for file-based loading tests
```

---

## Delivery Summary

| Phase                       | Status      | What ships                                                                         | Key SDK surface                        |
| --------------------------- | ----------- | ---------------------------------------------------------------------------------- | -------------------------------------- |
| **1 — Proto**               | ✅ Complete | `Skill`, `SkillProtocol`, `SkillHooks`, `@skill`, always-on injection              | `AgentHooks.on_llm_start`              |
| **2 — Routing**             | 🟡 Planned  | `SkillRegistry`, `when_to_use` matching, manifest, `invoke_skill`, `RunSkillHooks` | `AgentHooks.on_start`, `RunHooks`      |
| **3 — File skills**         | 🟡 Planned  | SKILL.md loading, frontmatter parsing, arg substitution, user + project dirs       | —                                      |
| **4 — Advanced triggering** | 🟡 Planned  | `context: fork`, tool-result triggers, post-turn skills                            | `AgentHooks.on_tool_end`, `on_llm_end` |
| **5 — Hardening**           | 🟡 Planned  | Source trust levels, path validation, `allowed-tools` enforcement, full docs       | —                                      |
