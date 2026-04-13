# openai-agents-skills

[![PyPI version](https://img.shields.io/pypi/v/openai-agents-skills.svg)](https://pypi.org/project/openai-agents-skills/)
[![CI](https://github.com/damianoneill/openai-agents-skills/actions/workflows/ci.yml/badge.svg)](https://github.com/damianoneill/openai-agents-skills/actions/workflows/ci.yml)
[![Compatibility](https://github.com/damianoneill/openai-agents-skills/actions/workflows/compatibility.yml/badge.svg)](https://github.com/damianoneill/openai-agents-skills/actions/workflows/compatibility.yml)
[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)](https://github.com/damianoneill/openai-agents-skills)

Skills extension for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

A **Skill** is a named, reusable prompt fragment injected into the LLM's context at the right moment
in the agent loop via `AgentHooks`. This lets you package workflow instructions, checklists, or
procedures as composable named units that can be shared across agents without duplicating configuration.

> **Alpha — Phase 3:** Core injection, registry & routing, and file-based skills are complete.
> Advanced triggering and hardening are coming in later phases — see [docs/PLAN.md](docs/PLAN.md).

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

    async def get_prompt_blocks(self, args: str = "") -> list:
        return [{"role": "user", "content": "Always respond using bullet points."}]
```

### Class attributes

| Attribute     | Type  | Purpose                                             |
| ------------- | ----- | --------------------------------------------------- |
| `name`        | `str` | Unique identifier                                   |
| `description` | `str` | Human-readable summary                              |
| `when_to_use` | `str` | Prose trigger description (used by Phase 2 routing) |

### Gating a skill with `is_enabled`

Override `is_enabled()` to activate or suppress a skill at runtime:

```python
import os
from openai_agents_skills import Skill

class FeatureFlagSkill(Skill):
    name = "feature_flag_skill"
    description = "Only active when ENABLE_SKILL=1."

    def is_enabled(self) -> bool:
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

## `SkillProtocol`

`SkillHooks` accepts any object that satisfies `SkillProtocol` — you do not have to subclass
`Skill`. Any class with `name: str`, `description: str`, and `async def get_prompt_blocks()`
qualifies:

```python
from openai_agents_skills import SkillProtocol, SkillHooks

class MyDuckSkill:
    name = "duck"
    description = "Duck-typed skill."

    async def get_prompt_blocks(self, args: str = "") -> list:
        return [{"role": "user", "content": "Quack."}]

assert isinstance(MyDuckSkill(), SkillProtocol)  # True

hooks = SkillHooks([MyDuckSkill()])
```

Duck-typed skills have no `is_enabled()` method and are always injected.

---

## `@skill_factory` Decorator

Tag factory functions with skill metadata for tooling and discovery (used by Phase 3+).
This is a forward-looking marker — it has no runtime effect in the current version.

```python
from openai_agents_skills import skill_factory, Skill

@skill_factory(name="summariser", description="Summarise long documents into bullet points.")
def make_summariser() -> Skill:
    return MySummariserSkill()

# Metadata is available without calling the factory:
print(make_summariser.__skill_name__)         # "summariser"
print(make_summariser.__skill_description__)  # "Summarise long documents into bullet points."

# Call when you need the Skill instance:
registry.register(make_summariser())
```

---

## Roadmap

| Phase               | Status      | What ships                                                               |
| ------------------- | ----------- | ------------------------------------------------------------------------ |
| **1 — Proto**       | ✅ Complete | `Skill`, `SkillProtocol`, `SkillHooks`, always-on injection              |
| **2 — Routing**     | ✅ Complete | `SkillRegistry`, LLM-based routing, `invoke_skill` tool, `RunSkillHooks` |
| **3 — File skills** | ✅ Complete | `FileSkill`, SKILL.md loading, YAML frontmatter, argument substitution   |
| **4 — Advanced**    | 🟡 Planned  | Tool-result triggers, post-turn skills                                   |
| **5 — Hardening**   | 🟡 Planned  | Source trust levels, path validation, `allowed-tools` enforcement        |

See [docs/PLAN.md](docs/PLAN.md) for the full phased plan.

---

## Compatibility

Tested weekly against the latest OpenAI Agents SDK to ensure compatibility.

---

## License

MIT
