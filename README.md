# openai-agents-skills

[![PyPI version](https://img.shields.io/pypi/v/openai-agents-skills.svg)](https://pypi.org/project/openai-agents-skills/)
[![CI](https://github.com/damianoneill/openai-agents-skills/actions/workflows/ci.yml/badge.svg)](https://github.com/damianoneill/openai-agents-skills/actions/workflows/ci.yml)
[![Compatibility](https://github.com/damianoneill/openai-agents-skills/actions/workflows/compatibility.yml/badge.svg)](https://github.com/damianoneill/openai-agents-skills/actions/workflows/compatibility.yml)
[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)](https://github.com/damianoneill/openai-agents-skills)

Skills extension for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

A **Skill** is a named, reusable prompt fragment injected into the LLM's context at the right moment
in the agent loop via `AgentHooks`. This lets you package workflow instructions, checklists, or
procedures as composable named units that can be shared across agents without duplicating configuration.

> **All planned phases complete.** See [docs/PLAN.md](docs/PLAN.md) for the full implementation history.

---

## Installation

```bash
pip install openai-agents-skills
```

---

## Quick Start

Define a skill by subclassing `Skill` and overriding `get_prompt_blocks`:

```python
from openai_agents_skills import Skill

class CitationSkill(Skill):
    name = "citation"
    description = "Always cite sources when making factual claims."

    async def get_prompt_blocks(self, args: str = "") -> list:
        return [{"role": "user", "content": "Always cite your sources when making factual claims."}]
```

Attach it to an agent via `SkillHooks`:

```python
from agents import Agent, Runner
from openai_agents_skills import SkillHooks

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    hooks=SkillHooks([CitationSkill()]),
)

result = await Runner.run(agent, "What is the speed of light?")
print(result.final_output)
```

Before each LLM call, `SkillHooks` prepends the skill's prompt blocks to the model's input list.
The model receives and acts on those instructions transparently — no changes to the agent's
`instructions`, `tools`, or any other configuration.

---

## How It Works

The SDK passes a mutable `input_items` list to `AgentHooks.on_llm_start` before every model
invocation, then passes the same list object directly to `model.get_response()`. `SkillHooks`
prepends skill prompt blocks to that list, so the model sees the skill's instructions on every call.

```
Runner.run()
  └─ on_llm_start(context, agent, system_prompt, input_items)
       └─ SkillHooks prepends skill blocks to input_items here
  └─ model.get_response(input=input_items)   ← model sees skill content
```

This is a standard `AgentHooks` subclass — no monkey-patching, no internal SDK changes.

---

## Defining Skills

Subclass `Skill` and override `get_prompt_blocks`. The method receives an optional `args` string
and returns a list of input-item dicts:

```python
from openai_agents_skills import Skill

class ReplyInBulletsSkill(Skill):
    name = "reply_in_bullets"
    description = "Instructs the agent to respond using bullet points."
    when_to_use = "Use when structured, scannable output is preferred."

    async def get_prompt_blocks(self, context, agent, args=""):
        return [{"role": "user", "content": "Always respond using bullet points."}]
```

### Class attributes

| Attribute     | Type  | Purpose                                             |
| ------------- | ----- | --------------------------------------------------- |
| `name`        | `str` | Unique identifier                                   |
| `description` | `str` | Human-readable summary                              |
| `when_to_use` | `str` | Prose trigger description (used by Phase 2 routing) |

### Context-aware skills

Both `get_prompt_blocks` and `is_enabled` receive the SDK's `RunContextWrapper` and
`Agent` from the hook, so skills can inject dynamic content or gate themselves on
runtime state:

```python
class OrgContextSkill(Skill):
    name = "org-context"
    description = "Injects the current organisation ID."

    async def get_prompt_blocks(self, context, agent, args=""):
        org_id = context.context.org_id if context else None
        if not org_id:
            return []
        return [{"role": "user", "content": f"Your org_id is `{org_id}`."}]
```

Skills must handle `context=None` and `agent=None` — both are `None` when the skill
is called outside a live agent run (e.g. via `make_invoke_skill_tool` or in tests).

### Gating a skill with `is_enabled`

Override `is_enabled()` to activate or suppress a skill at runtime:

```python
import os
from openai_agents_skills import Skill

class FeatureFlagSkill(Skill):
    name = "feature_flag_skill"
    description = "Only active when ENABLE_SKILL=1."

    def is_enabled(self, context=None, agent=None) -> bool:
        return os.getenv("ENABLE_SKILL") == "1"

    async def get_prompt_blocks(self, args: str = "") -> list:
        return [{"role": "user", "content": "The feature flag skill is active."}]
```

Disabled skills are silently skipped by `SkillHooks` on every call.

---

## Attaching Multiple Skills

Pass a list of skills to `SkillHooks`. All enabled skills inject in registration order:

```python
from openai_agents_skills import SkillHooks

hooks = SkillHooks([CitationSkill(), ReplyInBulletsSkill(), FeatureFlagSkill()])

agent = Agent(
    name="Assistant",
    instructions="You are helpful.",
    hooks=hooks,
)
```

---

## Registry & Routing

For dynamic per-turn skill selection, use `SkillRegistry` with `LLMSkillRouter`.
Skills with a non-empty `when_to_use` are forwarded to the router, which uses a
lightweight LLM call to decide which skills are relevant for the current message.
Skills with an empty `when_to_use` are always-on and inject unconditionally.

```python
from openai import AsyncOpenAI
from agents import Agent, Runner
from openai_agents_skills import LLMSkillRouter, SkillHooks, SkillRegistry

router = LLMSkillRouter(client=AsyncOpenAI(), model="gpt-4o-mini")
# See "Custom SkillRouter" below for non-OpenAI providers (Bedrock, Azure, Ollama).
registry = SkillRegistry(router=router)
registry.register(CitationSkill())        # non-empty when_to_use — routed selectively
registry.register(SafetyReminderSkill())  # empty when_to_use   — always injected

agent = Agent(
    name="Assistant",
    instructions="You are helpful.",
    hooks=SkillHooks(registry=registry),
)

result = await Runner.run(agent, "What is the speed of light?")
```

For multi-agent runs (handoffs), pass `RunSkillHooks` to `Runner.run` instead —
it fires for every agent in the handoff chain:

```python
from openai_agents_skills import RunSkillHooks

result = await Runner.run(
    agent,
    "What is the speed of light?",
    hooks=RunSkillHooks(registry=registry),
)
```

---

## Custom SkillRouter

`LLMSkillRouter` works with any `AsyncOpenAI`-compatible client but requires
`chat.completions.create` — which some providers (AWS Bedrock, Azure in certain
configurations, Ollama) don't fully support. For those cases, subclass
`BaseSkillRouter` and implement only `_call_model`. All routing logic — prompt
building, JSON extraction, and LRU caching — is inherited automatically.

```python
from openai_agents_skills import BaseSkillRouter


class MyBedrockRouter(BaseSkillRouter):
    """Route skills via AWS Bedrock Converse API."""

    def __init__(self, bedrock_client, model_id: str) -> None:
        super().__init__()          # inherits cache_size=256
        self._client = bedrock_client
        self._model_id = model_id

    async def _call_model(self, prompt: str) -> str:
        # Only _call_model needs to be implemented.
        # The response may contain prose before/after the JSON — BaseSkillRouter
        # handles extraction automatically.
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.converse(
                modelId=self._model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
            ),
        )
        return response["output"]["message"]["content"][0]["text"]


router = MyBedrockRouter(bedrock_client=boto3.client("bedrock-runtime"), model_id="...")
registry = SkillRegistry(router=router)
```

Use this with `SkillRegistry` exactly as you would `LLMSkillRouter`.

If you are on OpenAI and want stricter JSON-mode enforcement for weaker models,
pass `use_response_format=True` to `LLMSkillRouter` — it is opt-in and disabled
by default for maximum provider compatibility:

```python
router = LLMSkillRouter(client=AsyncOpenAI(), model="gpt-4o-mini", use_response_format=True)
```

---

## Roadmap

| Phase               | Status      | What ships                                                               |
| ------------------- | ----------- | ------------------------------------------------------------------------ |
| **1 — Proto**       | ✅ Complete | `Skill`, `SkillHooks`, always-on injection                               |
| **2 — Routing**     | ✅ Complete | `SkillRegistry`, LLM-based routing, `invoke_skill` tool, `RunSkillHooks` |
| **3 — File skills** | ✅ Complete | `FileSkill`, SKILL.md loading, YAML frontmatter, argument substitution   |
| **4 — Advanced**    | ✅ Complete | Tool-result triggers, post-turn skills                                   |
| **5 — Hardening**   | ✅ Complete | Source trust levels, path validation, coverage, docs                     |

See [docs/PLAN.md](docs/PLAN.md) for the full phased plan.

---

## Compatibility

Tested weekly against the latest OpenAI Agents SDK to ensure compatibility.

---

## License

MIT
