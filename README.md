# openai-agents-skills

[![PyPI version](https://img.shields.io/pypi/v/openai-agents-skills.svg)](https://pypi.org/project/openai-agents-skills/)
[![CI](https://github.com/damianoneill/openai-agents-skills/actions/workflows/ci.yml/badge.svg)](https://github.com/damianoneill/openai-agents-skills/actions/workflows/ci.yml)
[![Compatibility](https://github.com/damianoneill/openai-agents-skills/actions/workflows/compatibility.yml/badge.svg)](https://github.com/damianoneill/openai-agents-skills/actions/workflows/compatibility.yml)
[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)](https://github.com/damianoneill/openai-agents-skills)

Skills extension for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), bringing reusable, composable capabilities to the agent loop — inspired by Claude Skills.

---

## Problem

When building microservices with the OpenAI Agents SDK, the same tools, instructions, and behaviours are often duplicated across many agents:

- **Repetition** – copy-pasting tool lists and instruction fragments across agent definitions
- **Drift** – updates to a capability must be applied in every agent that uses it
- **No composition model** – the SDK provides no first-class concept for packaging a capability as a reusable unit
- **Hard to share** – there is no standard way to distribute and consume pre-built agent capabilities

---

## Solution

This package introduces a **Skill** — a named, self-contained unit that bundles tools, instructions, and metadata — and a **SkillRegistry** that applies selected skills to any agent at construction time.

Skills are designed to integrate cleanly with the SDK's existing agent loop:  no monkey-patching, no subclassing of internal types.

> ⚠️ **Early-stage alpha:** The current release provides the core `Skill` / `SkillRegistry` primitives and the `@skill` factory decorator.  Higher-level features such as skill discovery, lazy loading, and cross-agent skill sharing are planned (see [Roadmap](#roadmap)).

---

## Installation

```bash
pip install openai-agents-skills
```

---

## Quick Start

```python
from agents import Agent, Runner, function_tool
from openai_agents_skills import Skill, SkillRegistry, skill

# 1. Define a tool
@function_tool
def search_web(query: str) -> str:
    """Search the web and return a summary."""
    # TODO: implement with a real search API
    return f"Results for: {query}"

# 2. Wrap it in a Skill
web_search = Skill(
    name="web_search",
    description="Search the web for up-to-date information.",
    tools=[search_web],
    instructions="You can search the web to find current information. Always cite your sources.",
)

# 3. Register and apply
registry = SkillRegistry()
registry.register(web_search)

base_agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
agent = registry.apply(base_agent, skill_names=["web_search"])

# 4. Run as normal — skills are fully transparent to the agent loop
result = await Runner.run(agent, "What is the latest Python release?")
print(result.final_output)
```

---

## Core Concepts

### `Skill`

A `Skill` is a plain dataclass that carries:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique identifier |
| `description` | `str` | Human-readable summary |
| `tools` | `list` | Function tools contributed by this skill |
| `instructions` | `str` | System-prompt fragment appended when the skill is applied |
| `metadata` | `dict` | Extension point for tooling and discovery |

```python
from openai_agents_skills import Skill

calculator = Skill(
    name="calculator",
    description="Perform arithmetic calculations.",
    tools=[add, subtract, multiply, divide],
    instructions="You can perform arithmetic. Always show your working.",
)
```

### `SkillRegistry`

A `SkillRegistry` holds named skills and applies them to agents.

```python
from openai_agents_skills import SkillRegistry

registry = SkillRegistry()

# Register skills
registry.register(calculator)
registry.register(web_search)

# List registered skills
print(registry.skill_names)  # ['calculator', 'web_search']

# Apply a subset of skills to an agent
agent = registry.apply(base_agent, skill_names=["calculator"])

# Apply all registered skills
agent = registry.apply(base_agent)

# Remove a skill
registry.unregister("calculator")
```

`apply()` returns a **new** agent — the original is never mutated.  Tools are appended to the agent's existing tool list, and instruction fragments are concatenated with a newline separator.

### `@skill` Decorator

The `@skill` decorator tags a factory function with skill metadata without calling it.  This is useful for lazy initialisation and skill discovery tooling.

```python
from openai_agents_skills import skill, Skill

@skill(name="summariser", description="Summarise long documents into bullet points.")
def make_summariser() -> Skill:
    from agents import function_tool

    @function_tool
    def summarise(text: str) -> str:
        """Return a bullet-point summary of the provided text."""
        ...

    return Skill(
        name="summariser",
        description="Summarise long documents into bullet points.",
        tools=[summarise],
        instructions="When asked to summarise, produce concise bullet points.",
    )

# Metadata is available without calling the factory
print(make_summariser.__skill_name__)         # "summariser"
print(make_summariser.__skill_description__)  # "Summarise long documents..."

# Call when you need the actual Skill instance
registry.register(make_summariser())
```

---

## Applying Multiple Skills

```python
from agents import Agent, Runner
from openai_agents_skills import Skill, SkillRegistry

registry = SkillRegistry()
registry.register(web_search)
registry.register(calculator)
registry.register(code_interpreter)

# Compose an agent with exactly the skills it needs
research_agent = registry.apply(
    Agent(name="Researcher", instructions="You research topics thoroughly."),
    skill_names=["web_search", "calculator"],
)

# A different agent with a different skill set
dev_agent = registry.apply(
    Agent(name="Developer", instructions="You write and explain code."),
    skill_names=["code_interpreter", "web_search"],
)
```

---

## Sharing Skills Across a Microservice

```python
# skills.py — define once, import everywhere
from openai_agents_skills import Skill, SkillRegistry

REGISTRY = SkillRegistry()
REGISTRY.register(web_search)
REGISTRY.register(calculator)

# agent_a.py
from skills import REGISTRY
from agents import Agent

agent_a = REGISTRY.apply(Agent(name="A", instructions="..."), skill_names=["web_search"])

# agent_b.py
from skills import REGISTRY
from agents import Agent

agent_b = REGISTRY.apply(Agent(name="B", instructions="..."), skill_names=["calculator"])
```

---

## Roadmap

| Feature | Status |
|---------|--------|
| `Skill` dataclass | ✅ Implemented |
| `SkillRegistry` (register / apply / unregister) | ✅ Implemented |
| `@skill` factory decorator | ✅ Implemented |
| Skill discovery via entry points | 🟡 Planned |
| Async skill initialisation | 🟡 Planned |
| Skill versioning and dependency resolution | 🟡 Planned |
| Pre-built skill library | 🟡 Planned |
| Skill lifecycle hooks (on_attach / on_detach) | 🟡 Planned |

---

## Compatibility

Tested weekly against the latest OpenAI Agents SDK to ensure compatibility.

---

## License

MIT
