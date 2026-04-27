# Example: Juniper Networks Troubleshooting Agent

This document walks through a realistic end-to-end scenario for a Juniper Networks
troubleshooting agent built with `openai-agents-skills`. It is written for both
engineers who will implement it and managers who want to understand what the library
does and why.

---

## What Problem Does This Solve?

As AI agents grow in capability, their system prompts tend to grow with them — absorbing
every checklist, procedure, and contextual rule the agent might ever need. This creates
three problems:

- **Token cost**: Every LLM call sends the full prompt, including guidance that is
  irrelevant to the current conversation.
- **Maintenance**: Updating a procedure means editing a monolithic prompt string shared
  across every interaction.
- **Quality**: Models perform worse when the context is cluttered with content that does
  not apply to the current task.

`openai-agents-skills` solves this by making instructions **composable and selective**.
A _Skill_ is a named block of instructions that knows when it is relevant. The library
injects only the skills that matter for each user message, leaving the rest out of the
model's context entirely.

**In production this means:** a BGP checklist is only sent when the user is asking
about BGP. A log-parsing reference only appears when log output is present. A
documentation-handoff instruction is always there. The main agent's system prompt
stays lean and focused.

---

## Key Concepts

| Concept                    | What it does                                                                                                                                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`Skill`**                | A named unit of instructions. Implements `get_prompt_blocks()` which returns the content to prepend to the model's input.                                                                                                                   |
| **Always-on skill**        | A skill with `always_on=True`. Injects on every LLM call unconditionally. Used for standing policies, org context, handoff rules.                                                                                                           |
| **Routed skill**           | A skill with `always_on=False` (default). Only injects when a router selects it as relevant to the current message.                                                                                                                         |
| **`SkillRegistry`**        | Holds all registered skills and knows which are always-on vs routable. One registry instance per agent — registries are not shared across agents.                                                                                           |
| **`LLMSkillRouter`**       | Sends the user's message and a skill manifest to a lightweight model. Returns which routed skills to activate. Results are LRU-cached so repeated messages within a session pay the routing cost only once.                                 |
| **`SkillHooks`**           | An `AgentHooks` subclass that wires the registry into the OpenAI Agents SDK loop. No monkey-patching — it uses the SDK's standard extension point.                                                                                          |
| **`FileSkill` / SKILL.md** | A skill loaded from a Markdown file with YAML frontmatter. The body may contain `$ARGUMENTS` as a substitution placeholder (matching the [agentskills.io](https://agentskills.io) specification), plus `${KEY}` / `$KEY` variable patterns. |

---

## Why It Works This Way

Skills inject via `AgentHooks.on_llm_start`, a standard SDK hook that fires before
every `model.get_response()` call. The SDK passes a mutable list of input items;
`SkillHooks` prepends the selected skill blocks to that list before the model sees it.
This means:

- **No changes to the agent's `instructions`, `tools`, or `model` configuration** —
  skills are additive and transparent.
- **Skills are re-evaluated on every LLM call**, not once per user turn. If the agent
  calls a tool and re-invokes the LLM to reason about the result, the same skills
  inject again. This keeps the model grounded throughout a multi-step investigation.
- **Routing uses an LRU cache**: if the same message context triggers a second LLM call
  (e.g. after a tool result), the router is not called again — the cached selection is
  reused at zero cost.

---

## When Does `on_llm_start` Fire?

`on_llm_start` fires **before every single `model.get_response()` call** in the agent
loop — not just once per user turn. The SDK's `run_loop.py` sequence is:

```python
# Simplified from run_loop.py
filtered = await maybe_filter_model_input(agent, input_items, system_prompt)
filtered.input = deduplicate_input_items_preferring_latest(filtered.input)
await asyncio.gather(
    run_hooks.on_llm_start(ctx, agent, filtered.instructions, filtered.input),
    agent.hooks.on_llm_start(ctx, agent, filtered.instructions, filtered.input),
)
response = await model.get_response(input=filtered.input, ...)  # same list object
await asyncio.gather(
    run_hooks.on_llm_end(ctx, agent, response),
    agent.hooks.on_llm_end(ctx, agent, response),
)
```

`filtered.input` is a **mutable Python list passed by reference**. Items prepended
inside `on_llm_start` are visible to the model on that call. This means `on_llm_start`
fires:

| Situation                                                                     | Fires?            |
| ----------------------------------------------------------------------------- | ----------------- |
| User sends a message (Turn 1, 2, 3 …)                                         | ✅ Every time     |
| Agent calls a tool and the loop re-invokes the LLM to reason about the result | ✅ Every time     |
| A handoff occurs and the receiving agent's LLM is called                      | ✅ For that agent |

Skill injection is therefore **re-evaluated on every LLM invocation** — `on_llm_end`
resets the injection guard after each model call so skills inject fresh on the next
turn. This is why routing and the `is_enabled()` gate are load-bearing, not optional —
you do not want to blindly dump every skill into every call.

---

## Scenario Setup

### Agent

```python
from agents import Agent, Runner
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from openai_agents_skills import SkillHooks, SkillRegistry, LLMSkillRouter

model = OpenAIChatCompletionsModel("gpt-4o-mini", AsyncOpenAI())
registry = SkillRegistry(
    router=LLMSkillRouter(model=model),
)
registry.register(BGPTroubleshootingSkill())
registry.register(LogParserSkill())
registry.register(JunosCliReferenceSkill())
registry.register(EscalationPolicySkill())   # always-on

agent = Agent(
    name="Juniper Troubleshooter",
    model="gpt-4o",
    instructions="You are an expert Juniper Networks support engineer.",
    tools=[run_show_command],                 # NETCONF tool for live device queries
    hooks=SkillHooks(registry=registry, routing_context_turns=3),
)
```

Each agent has its own `SkillRegistry` instance. Skills registered with one agent are
invisible to any other agent's router — if you run multiple specialised agents, construct
a separate registry for each and attach it via its own `SkillHooks`.

### Skills Registered

| Skill name            | `description` (routing signal)                                                                                                                           | Routing                   |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `bgp-troubleshooting` | "BGP session diagnostics and peer flap analysis. Use when user reports BGP session issues, peer flapping, route withdrawal, or AS-path problems."        | Routed (selective)        |
| `log-parser`          | "Parse and interpret Junos syslog messages and error codes. Use when user provides log output, syslog messages, or Juniper error codes."                 | Routed (selective)        |
| `junos-cli-reference` | "Junos CLI commands, commit syntax, and operational-mode reference. Use when user needs show commands, commit syntax, or operational-mode CLI guidance." | Routed (selective)        |
| `escalation-policy`   | "When and how to escalate to Juniper TAC." (`always_on=True`)                                                                                            | Always-on (unconditional) |

Because `escalation-policy` has **`always_on=True`**, it is never passed to the
router — it injects on every LLM call as long as `is_enabled()` returns `True`.

The other three skills use `always_on=False` (default), so they only inject when the
`LLMSkillRouter` selects them for the current message.

> **SKILL.md note:** Skill names must follow the [agentskills.io](https://agentskills.io)
> specification — lowercase letters, digits, and hyphens only; 1–64 characters; no
> leading/trailing hyphen; no consecutive hyphens (`--`). All four names above are valid.
> Optional frontmatter fields `license`, `compatibility` (≤ 500 chars), and `metadata`
> (key/value map) are supported for documentation and tooling purposes but do not affect
> routing or injection behaviour.

---

## Turn 1 — User's Initial Question

> **User:** "My BGP session to 10.1.1.1 keeps flapping. The peer is an MX480. Here's
> the log:
> `rpd[1234]: bgp_listen_accept: Connection attempt from unconfigured neighbor: 10.1.1.1`"

### `on_llm_start` fires (first time)

**Step 1 — Always-on pass**

`registry.get_always_on()` returns `[escalation-policy]`. Its `is_enabled()` returns
`True`, so it is included unconditionally.

**Step 2 — Routing pass**

`registry.select_for_message(message)` calls `LLMSkillRouter.select()`. A routing
request goes to `gpt-4o-mini` with the user message and this skill manifest:

```
You are a skill router. Given a user message, select which skills (if any) apply.
Return JSON: {"selected": ["skill_name", ...]}.
Only select skills clearly relevant to the message. Return {"selected": []} if none apply.

User message: "My BGP session to 10.1.1.1 keeps flapping..."

Available skills:
- bgp-troubleshooting: BGP session diagnostics and peer flap analysis.
  When to use: Use when user reports BGP session issues, peer flapping, route withdrawal, or AS-path problems.
- log-parser: Parse and interpret Junos syslog messages and error codes.
  When to use: Use when user provides log output, syslog messages, or Juniper error codes.
- junos-cli-reference: Junos CLI commands, commit syntax, and operational-mode reference.
  When to use: Use when user needs show commands, commit syntax, or operational-mode CLI guidance.
```

Router responds: `{"selected": ["bgp-troubleshooting", "log-parser"]}`

`junos-cli-reference` is not selected — the user has not asked for CLI syntax yet.

**Step 3 — Merge, deduplicate, gate**

Final set after deduplication and `is_enabled()` check:
`[escalation-policy, bgp-troubleshooting, log-parser]`

**Step 4 — `get_prompt_blocks()` called for each skill**

`escalation-policy` returns:

```
[ESCALATION POLICY]
If you cannot resolve the issue within 3 diagnostic steps, advise the user to open
a Juniper TAC case. Recommended priority: S2 for production-impacting BGP issues.
TAC portal: https://supportportal.juniper.net
```

`bgp-troubleshooting` returns:

```
[BGP TROUBLESHOOTING CHECKLIST]
1. Confirm the neighbor IP in the log matches the configured peer address:
   show bgp neighbor <peer-ip>
2. Check for hold-time or keepalive mismatch between peers.
3. Verify the configured AS number matches the remote peer's AS:
   show bgp neighbor <peer-ip> | match "AS number"
4. Look for authentication (MD5) mismatches in the neighbor configuration.
5. Check for route-policy or import/export policy rejecting all prefixes.
6. Review interface flapping on the peering link: show interfaces <iface> extensive
```

`log-parser` returns:

```
[JUNOS LOG REFERENCE]
Message: bgp_listen_accept: Connection attempt from unconfigured neighbor: <ip>
Meaning: The device received a TCP SYN on port 179 from an IP address that does not
         appear in the BGP neighbor table.
Likely causes:
  - The neighbor IP in the configuration does not match the source IP of the peer.
  - The peer is using a loopback or update-source that differs from the expected IP.
  - The session was deleted or never configured on this device.
Recommended next step: run `show bgp neighbor` and compare configured peer IPs.
```

**Step 5 — Prepend to `input_items`**

```python
input_items[0:0] = all_blocks   # single prepend, preserves registration order
```

**Step 6 — `model.get_response()` called**

`gpt-4o` now sees:

```
[escalation-policy block]
[bgp-troubleshooting block]
[log-parser block]
[user message: "My BGP session to 10.1.1.1 keeps flapping..."]
```

The model produces a structured response: it interprets the log message (using the
`log-parser` block), walks through steps 1–3 of the BGP checklist, and asks the user
to confirm the configured peer IP.

---

## Tool Call — Agent Queries the Device

The model decides to run a live check and calls the `run_show_command` tool:

```
Tool call: run_show_command("show bgp neighbor 10.1.1.1")
```

The SDK fires `on_tool_start`, executes the tool over NETCONF, then fires `on_tool_end`
with the device output. The tool result is appended to `input_items`. The agent loop
now calls the LLM **again** to reason about the result.

### `on_llm_start` fires (second time)

- **Routing context string** extracted from `input_items` is still built from the same
  user messages as before (tool results are not user messages, so the window of recent
  user turns has not changed).
- **Router cache hit** — the same routing context string was already routed this turn.
  The LRU cache returns `["bgp-troubleshooting", "log-parser"]` without a second
  `gpt-4o-mini` call.
- The same three skill blocks are prepended again.
- `model.get_response()` now sees:

```
[escalation-policy block]
[bgp-troubleshooting block]
[log-parser block]
[user message: "My BGP session to 10.1.1.1 keeps flapping..."]
[tool result: show bgp neighbor 10.1.1.1 output]
```

The model reasons about the device output in the context of the BGP checklist and
produces the next diagnostic step.

---

## Turn 2 — User Follow-up

> **User:** "The peer IP looks right. Could it be an AS number mismatch?"

### `on_llm_start` fires (third time)

**Step 1 — Always-on pass**

`escalation-policy` injects as before.

**Step 2 — Routing pass**

This is a **different routing context string** — the window now includes both the
original BGP flapping message and this new follow-up, so the LRU cache misses and a
new routing call goes to `gpt-4o-mini`.

Router responds: `{"selected": ["bgp-troubleshooting", "junos-cli-reference"]}`

- `bgp-troubleshooting` selected — still an active BGP diagnosis session.
- `junos-cli-reference` now selected — the user is asking about AS numbers, and the
  checklist will recommend `show bgp neighbor detail` commands; the CLI reference skill
  provides correct syntax.
- `log-parser` not selected — the user is no longer providing new log output.

**Step 3 — Final injected context**

```
[escalation-policy block]
[bgp-troubleshooting block]
[junos-cli-reference block]
[turn 1 user message]
[tool result]
[turn 1 assistant response]
[turn 2 user message: "The peer IP looks right. Could it be an AS number mismatch?"]
```

The model now answers with the exact `show bgp neighbor` command to check AS numbers,
referencing the correct JunOS syntax from the `junos-cli-reference` block.

---

## Full Call Flow Summary

```
User: "My BGP session to 10.1.1.1 keeps flapping... [log output]"
│
├─ on_llm_start (Turn 1)
│   ├─ Always-on:  [escalation-policy]
│   ├─ Router →   gpt-4o-mini → ["bgp-troubleshooting", "log-parser"]   (cache MISS)
│   ├─ Inject:    [escalation-policy] + [bgp-troubleshooting] + [log-parser]
│   └─ model.get_response() → "Confirm peer IP, try: show bgp neighbor 10.1.1.1"
│
├─ on_llm_end   → injected_this_call cleared ✦ skills will re-inject on next call
│
├─ on_tool_start  → run_show_command("show bgp neighbor 10.1.1.1")
├─ on_tool_end    → result appended to input_items (also queues any skills whose triggers_after_tools matches)
│
├─ on_llm_start (post-tool)
│   ├─ Always-on:  [escalation-policy]
│   ├─ Router →   cache HIT (same message) → ["bgp-troubleshooting", "log-parser"]
│   ├─ Inject:    [escalation-policy] + [bgp-troubleshooting] + [log-parser]
│   └─ model.get_response() → "Output shows AS 65001; expected 65002. Likely mismatch."
│
├─ on_llm_end   → injected_this_call cleared ✦ skills will re-inject on next call
│
User: "The peer IP looks right. Could it be an AS number mismatch?"
│
└─ on_llm_start (Turn 2)
    ├─ Always-on:  [escalation-policy]
    ├─ Router →   gpt-4o-mini → ["bgp-troubleshooting", "junos-cli-reference"]  (cache MISS)
    ├─ Inject:    [escalation-policy] + [bgp-troubleshooting] + [junos-cli-reference]
    └─ model.get_response() → "Run: show bgp neighbor 10.1.1.1 | match 'AS number'"
```

---

## Token Efficiency

| Turn      | Skills injected                                             | Skills skipped      | Why                     |
| --------- | ----------------------------------------------------------- | ------------------- | ----------------------- |
| Turn 1    | escalation-policy, bgp-troubleshooting, log-parser          | junos-cli-reference | No CLI syntax question  |
| Post-tool | escalation-policy, bgp-troubleshooting, log-parser          | junos-cli-reference | Same message, cache hit |
| Turn 2    | escalation-policy, bgp-troubleshooting, junos-cli-reference | log-parser          | No new log output       |

`junos-cli-reference` may be hundreds of tokens of JunOS command syntax. Skipping it
on Turn 1 and the tool-call re-invocation saves those tokens on two of three LLM calls
while still delivering the guidance exactly when it is needed.

---

## Tool-Result Triggers

`SkillHooks.on_tool_end` queues skills based on _which tool just ran_, giving even
finer-grained control than message routing alone. Skills declare the tools that should
trigger them via a class attribute; `SkillHooks` handles all queuing and drains the
pending list at the next `on_llm_start`:

```python
class LogParserSkill(Skill):
    name = "log-parser"
    description = "Parses Junos log output and error codes."
    triggers_after_tools = ["run_show_command"]  # queue after this tool completes

    async def get_prompt_blocks(self, args: str = "") -> list:
        ...  # return log-parsing guidance blocks as normal
```

With this, `log-parser` is automatically queued after every `run_show_command`
invocation — even if the user message that triggered the tool call did not mention
logs — and its prompt blocks are prepended at the start of the next LLM call.

`triggers_after_tools` is the right mechanism when a tool always warrants the same skill
regardless of what it returned. When a tool can return different classifications and each
classification warrants a different skill, see the next section.

---

## Content-Driven Skill Dispatch with `make_invoke_skill_tool`

The router selects skills based on the user's message. Tool-result triggers select
skills based on which tool ran. Neither mechanism can select a skill based on a value
_inside_ a tool's output — for example, a `root_cause` field returned by a diagnostic
pipeline that classifies the failure differently on each invocation.

`make_invoke_skill_tool` solves this. It creates a standard `FunctionTool` that the
model can call to retrieve any registered skill by name. The dispatch logic lives in
the agent's `instructions`, making it explicit and deterministic rather than
probabilistic:

```python
from openai_agents_skills import make_invoke_skill_tool

# Register one skill per failure class
registry.register(NexthopUnresolvableSkill())   # name = "nexthop-unresolvable"
registry.register(QueueCongestionSkill())        # name = "queue-congestion"
registry.register(EcmpBlackholeSkill())          # name = "ecmp-blackhole"
registry.register(UnclassifiedSkill())           # name = "unclassified-investigation"

agent = Agent(
    name="Diagnostic Agent",
    model="gpt-4o",
    instructions="""
        You are an expert network diagnostic agent.

        After any tool returns a result containing a root_cause field, immediately
        call invoke_skill with the matching skill name:
          NEXTHOP_UNRESOLVABLE  → invoke_skill("nexthop-unresolvable")
          QUEUE_CONGESTION      → invoke_skill("queue-congestion")
          ECMP_BLACKHOLE        → invoke_skill("ecmp-blackhole")
          UNKNOWN               → invoke_skill("unclassified-investigation")

        The skill will return interpretation and remediation guidance. Apply it to
        the diagnostic evidence before responding to the user.
    """,
    tools=[
        run_diagnostic_pipeline,
        make_invoke_skill_tool(registry),
    ],
    hooks=SkillHooks(registry=registry),
)
```

`invoke_skill` returns the skill's prompt content as a tool result string. The model
receives the guidance and the diagnostic evidence together in the same LLM call and
reasons about them simultaneously — no extra round-trip is needed.

The mapping table in `instructions` is the only thing that changes when a new failure
class is added: register the new skill, add one line to the table. No library or
application code changes are required.

### Router vs. `invoke_skill` — when to use each

| Mechanism                | Best for                                                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `LLMSkillRouter`         | Broad topic gating from conversational user messages — "use BGP checklist when the user mentions BGP". Tolerates ambiguity.  |
| `triggers_after_tools`   | Unconditional injection after a specific tool runs, regardless of message content.                                           |
| `make_invoke_skill_tool` | Precise, deterministic dispatch driven by structured values in tool output — classification codes, enum fields, error types. |

Both mechanisms can coexist in the same agent. A common pattern is to use the router
for broad topic skills (`bgp-troubleshooting`, `junos-cli-reference`) and
`invoke_skill` for narrow use-case skills whose activation depends on what a tool
returned.

---

## Proactive Sessions — Agent-Initiated Entry

The examples above assume a human sends the first message. Some deployments need the
agent to open a session autonomously — for example, when an on-device monitoring system
detects an anomaly and notifies the agent before any user interaction has occurred.

In this pattern the calling code constructs a synthetic first message and passes it
directly to `Runner.run()`:

```python
synthetic_prompt = (
    "Automated alert from monitoring system.\n"
    "Device: qfx-spine-01\n"
    "Event: sustained packet drop on interface et-0/0/0\n"
    "Root cause: NEXTHOP_UNRESOLVABLE\n"
    "Evidence: <pre-collected diagnostic summary>\n"
    "Analyse the evidence and produce a remediation recommendation."
)

result = await Runner.run(agent, input=synthetic_prompt)
```

The agent loop treats this exactly like a user message. `on_llm_start` fires, always-on
skills inject unconditionally, and the router receives the synthetic prompt as its
routing context string.

### Router behaviour with structured synthetic prompts

`LLMSkillRouter` routes on the text of the message passed to it. A well-formed
synthetic prompt that names a failure class clearly (as above) is sufficient for the
router to select broadly scoped topic skills. However, the router is **not reliable**
for selecting tightly scoped use-case skills (e.g. `nexthop-unresolvable` vs.
`ecmp-blackhole`) from a structured field value — descriptions of closely related
failure-class skills are too similar for the probabilistic router to distinguish
reliably.

The recommended approach for structured synthetic prompts:

- Use the router for **broad topic skills** whose descriptions match naturally against
  free text (e.g. `bgp-troubleshooting` when the prompt mentions BGP).
- Use `invoke_skill` in the agent's `instructions` for **precise failure-class
  dispatch** driven by a `root_cause` or equivalent classification field.

This hybrid works for both entry paths — a human asking "prefix X is dropping traffic"
and a monitoring system delivering a pre-classified `DiagnosticBundle` — because
`invoke_skill` dispatch is driven by structured field values that are present in both
cases.
