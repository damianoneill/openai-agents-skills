# openai-agents-skills

[![PyPI version](https://img.shields.io/pypi/v/openai-agents-skills.svg)](https://pypi.org/project/openai-agents-skills/)
[![CI](https://github.com/damianoneill/openai-agents-skills/actions/workflows/ci.yml/badge.svg)](https://github.com/damianoneill/openai-agents-skills/actions/workflows/ci.yml)
[![Compatibility](https://github.com/damianoneill/openai-agents-skills/actions/workflows/compatibility.yml/badge.svg)](https://github.com/damianoneill/openai-agents-skills/actions/workflows/compatibility.yml)
[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)](https://github.com/damianoneill/openai-agents-skills)

Skills extension for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

A **Skill** is a named, reusable prompt fragment injected into the LLM's context at the right moment
in the agent loop via `AgentHooks`. This lets you package workflow instructions, checklists, or
procedures as composable named units that can be shared across agents without duplicating configuration.

File-based skills follow the [agentskills.io](https://agentskills.io/) open standard, making
`SKILL.md` files portable across any compatible agent tool.

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

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list:
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

    async def get_prompt_blocks(self, context, agent, args=""):
        return [{"role": "user", "content": "Always respond using bullet points."}]
```

### Class attributes

| Attribute              | Type        | Purpose                                                                                   |
| ---------------------- | ----------- | ----------------------------------------------------------------------------------------- |
| `name`                 | `str`       | Unique identifier                                                                         |
| `description`          | `str`       | Human-readable summary; used by routing logic to decide relevance                         |
| `always_on`            | `bool`      | `True` → injects on every call; `False` (default) → routed by description                 |
| `allowed_tools`        | `list[str]` | Tools this skill may invoke; surfaced in the manifest. Default: `[]`                      |
| `user_invocable`       | `bool`      | `False` hides the skill from the manifest while still allowing injection. Default: `True` |
| `triggers_after_tools` | `list[str]` | Tool names that queue this skill for injection after the tool completes. Default: `[]`    |
| `triggers_after_turn`  | `bool`      | `True` → queued after every model response for quality checks or review. Default: `False` |

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

    async def get_prompt_blocks(self, context, agent, args: str = "") -> list:
        return [{"role": "user", "content": "The feature flag skill is active."}]
```

Disabled skills are silently skipped by `SkillHooks` on every call.

This pattern requires defining the skill as a Python subclass so you can override the
method. Skills loaded from a `SKILL.md` file have no such override point; gate them by
registry membership, or assign a predicate to their `enabled_when` attribute after
loading — see the scoping note under [Registry & Routing](#registry--routing).

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

## File-based Skills

Skills can be defined as `SKILL.md` files on disk — no Python required. This follows the
[agentskills.io](https://agentskills.io/) open standard, so the same skill folders work across
any compatible agent tool (Cursor, VS Code, GitHub Copilot, Claude Code, and others).

### Directory layout

```
~/.agent/skills/          # user layer  (personal, cross-repo)
  my-skill/
    SKILL.md

<project>/.agent/skills/  # project layer (checked in)
  payments-workflow/
    SKILL.md
```

### SKILL.md format

```yaml
---
name: payments-workflow # required: lowercase letters, digits, hyphens; max 64 chars
# description (required): what the skill does and when to use it; max 1024 chars
description: >
  Guides the agent through the
  end-to-end payments workflow.
license: MIT # optional
compatibility: Requires Python 3.11+ # optional; max 500 chars
metadata: # optional: arbitrary key-value pairs
  author: my-team
  version: "1.0"
allowed-tools: Bash(git:*) Read # optional: space-separated list of pre-approved tools
always-on: false # extension: inject unconditionally (default false)
user-invocable: true # extension: show in manifest (default true)
---
Step-by-step instructions the agent follows when handling a payment request...
```

Standard fields (`name`, `description`, `license`, `compatibility`, `metadata`, `allowed-tools`)
are defined by the [agentskills.io specification](https://agentskills.io/specification).
Fields prefixed "extension" are specific to this library.

### Loading file-based skills

```python
from pathlib import Path
from openai_agents_skills import load_all_skills, SkillHooks
from agents import Agent

registry = await load_all_skills(cwd=Path.cwd())

agent = Agent(
    name="Assistant",
    instructions="You are helpful.",
    hooks=SkillHooks(registry=registry),
)
```

`load_all_skills` searches the user layer (`~/.agent/skills/`) then the project layer
(`.agent/skills/` relative to `cwd`). User-layer skills win on name conflicts.

---

## Registry & Routing

For dynamic per-turn skill selection, use `SkillRegistry` with `LLMSkillRouter`.
Skills without `always_on=True` are forwarded to the router, which uses `description`
to decide relevance. Skills with `always_on=True` inject unconditionally.

```python
from agents import Agent, Runner
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from openai_agents_skills import LLMSkillRouter, SkillHooks, SkillRegistry

model = OpenAIChatCompletionsModel("gpt-4o-mini", AsyncOpenAI())
router = LLMSkillRouter(model=model)
# Pass any SDK Model instance — LitellmModel, AnyLLMModel, or OpenAIChatCompletionsModel all work.
registry = SkillRegistry(router=router)
registry.register(CitationSkill())        # always_on=False (default) — routed by description
registry.register(SafetyReminderSkill())  # always_on=True — always injected

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

> **Scoping across agents:** `RunSkillHooks` shares one registry across every agent in
> the chain, so an `always_on` skill injects for *all* of them — registry membership is
> only a per-agent lever when each agent has its own registry via per-agent `SkillHooks`.
> To scope a skill to a subset of agents on the shared-registry path, gate it on the
> active agent. A Python `Skill` overrides `is_enabled(context, agent)`; a skill loaded
> from a `SKILL.md` file has no subclass to override, so assign a predicate to its
> `enabled_when` attribute after loading:
>
> ```python
> for skill, _path in await load_skills_from_dir(skills_dir, SkillSource.PROJECT):
>     if skill.name == "markdown-guidelines":
>         skill.enabled_when = lambda context, agent: getattr(agent, "name", None) == "triage"
>     registry.register(skill)
> ```

---

## Writing Effective Descriptions

For routed skills (`always_on=False`), the `description` is the **only** text the
router sees when deciding whether a skill is relevant — it doubles as human
documentation and as the routing signal. A vague description is the most common
cause of a skill that never gets selected.

Write the description for a human deciding when to reach for the skill; the router
benefits as a side effect. A few habits cover almost every case:

- **Lead with the symptom in the user's words.** Users describe problems
  ("clients aren't getting addresses"), not internals ("DHCPv4 binding subsystem").
  The router matches against the user's message, so mirror their vocabulary.
- **Add an explicit "use when…" clause** that enumerates the triggering
  conditions. This is the single highest-leverage habit.
- **Disambiguate from sibling skills with `Not for` lines.** Most routing
  mistakes are confusion between near-neighbours, not a failure to find anything.
  When two skills overlap, name what each is _not_ for and point to the right
  one — e.g. `Not for AP-side loops — use ap-loop-troubleshooting instead.`
  A single exclusion line prevents the most common class of mis-routing.
- **Keep the core short; add exclusions only when needed.**
  Start with the symptom and the "use when" clause; only add `Not for` lines and
  scope notes when a skill has close neighbours. If the core description grows
  past a short paragraph, that is usually a signal to split the skill. Remember
  the hard budget below — the whole field must fit in 1024 characters.

Avoid keyword-stuffing the description with bare terms (`dhcp lease pool relay
option82 renew nak ...`). It reads worse for humans, drifts from the
[agentskills.io](https://agentskills.io/) standard that other tools rely on, and
actually routes _worse_ — a model reasons better over a clean "use when" sentence
than over a bag of words.

```yaml
# Weak — only matches the literal word "DHCP"
description: DHCP troubleshooting for Junos.

# Strong — names the symptoms a user would actually type, and disambiguates
description: >
  Diagnose Junos DHCP server and relay problems. Use when a user reports leases
  not renewing, clients not receiving addresses, address pool exhaustion, relay
  forwarding misconfiguration, or Option 82 mismatches.
  Not for DHCPv6 prefix delegation — use dhcpv6-troubleshooting instead.
```

This guidance applies equally to Python `Skill` subclasses and file-based
`SKILL.md` skills.

> **Length budget.** File-based skills have a hard limit of **1024 characters**
> for `description`, set by the [agentskills.io](https://agentskills.io/)
> specification. The loader rejects a longer description and skips the skill
> (logged at debug level), so an over-long description silently disappears rather
> than failing loudly. If you are porting skills from a tool that allows longer
> descriptions, trim to 1024 characters and move detail into the body.
>
> **Field names.** This library reads `allowed-tools` (not `tools`) for the list
> of tools a skill may invoke, and ignores non-standard fields such as `title`,
> `headers`, `tags`, and `platforms` — they are parsed but have no effect on
> loading or routing.

---

## Custom SkillRouter

`LLMSkillRouter` accepts any `agents.models.interface.Model` instance from the
`openai-agents` SDK, so Bedrock, Azure, Ollama, and any other supported provider
work out of the box. Pass the same model object you already have configured for
your agent — no extra client setup required:

```python
from agents.extensions.models.litellm_model import LitellmModel
from openai_agents_skills import LLMSkillRouter, SkillRegistry

router = LLMSkillRouter(model=LitellmModel(model="bedrock/anthropic.claude-3-haiku-..."))
registry = SkillRegistry(router=router)
```

For providers that require extra configuration on the model call — such as AWS
Bedrock credentials via `extra_args` or inference profiles via `extra_body` —
pass a `ModelSettings` instance to `model_settings`:

```python
from agents.model_settings import ModelSettings
from openai_agents_skills import LLMSkillRouter, SkillRegistry

router = LLMSkillRouter(
    model=model,
    model_settings=ModelSettings(
        extra_args={"aws_access_key_id": "...", "aws_secret_access_key": "..."}
    ),
)
registry = SkillRegistry(router=router)
```

For truly custom integrations not covered by the SDK (e.g. a proprietary API with
a non-standard interface), subclass `BaseSkillRouter` and implement only
`_call_model`. All routing logic — prompt building, JSON extraction, and LRU
caching — is inherited automatically:

```python
from openai_agents_skills import BaseSkillRouter


class MyCustomRouter(BaseSkillRouter):
    async def _call_model(self, prompt: str) -> str:
        # Call your custom model API here.
        # The response may contain prose before/after the JSON —
        # BaseSkillRouter handles extraction automatically.
        ...


registry = SkillRegistry(router=MyCustomRouter())
```

---

## Compatibility

Tested weekly against the latest OpenAI Agents SDK to ensure compatibility.

---

## License

MIT
