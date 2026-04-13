# Example: Juniper Networks Troubleshooting Agent

This document walks through a realistic end-to-end scenario for a Juniper Networks
troubleshooting agent built with `openai-agents-skills`. It shows what skills are
registered, when `on_llm_start` fires, how routing selects the right skills per turn,
and what the injected context looks like at each step.

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
from openai import AsyncOpenAI
from agents import Agent, Runner
from openai_agents_skills import SkillHooks, SkillRegistry, LLMSkillRouter

registry = SkillRegistry(
    router=LLMSkillRouter(client=AsyncOpenAI(), model="gpt-4o-mini"),
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

### Skills Registered

| Skill name            | `when_to_use`                                                                                     | Routing                   |
| --------------------- | ------------------------------------------------------------------------------------------------- | ------------------------- |
| `bgp-troubleshooting` | "Use when user reports BGP session issues, peer flapping, route withdrawal, or AS-path problems." | Routed (selective)        |
| `log-parser`          | "Use when user provides log output, syslog messages, or Juniper error codes."                     | Routed (selective)        |
| `junos-cli-reference` | "Use when user needs show commands, commit syntax, or operational-mode CLI guidance."             | Routed (selective)        |
| `escalation-policy`   | _(empty)_                                                                                         | Always-on (unconditional) |

Because `escalation-policy` has an **empty `when_to_use`**, it is never passed to the
router — it injects on every LLM call as long as `is_enabled()` returns `True`.

The other three skills have `when_to_use` filled in, so they only inject when the
`LLMSkillRouter` selects them for the current message.

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
- bgp-troubleshooting: Use when user reports BGP session issues, peer flapping,
  route withdrawal, or AS-path problems.
- log-parser: Use when user provides log output, syslog messages, or Juniper error codes.
- junos-cli-reference: Use when user needs show commands, commit syntax, or
  operational-mode CLI guidance.
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
