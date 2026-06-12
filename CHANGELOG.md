# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.2.0 (2026-06-12)

### Feat

- add model_settings parameter to LLMSkillRouter

### Refactor

- use TYPE_CHECKING guard for ModelSettings annotation

## v0.1.0 (2026-06-12)

### Feat

- **agentskills.io standard alignment** — `SKILL.md` files align with the
  [agentskills.io](https://agentskills.io/) open specification:
  - `allowed-tools` accepts a space-separated string (`"Bash(git:*) Read"`) per spec, as well as a
    YAML list for backward compatibility.
  - Standard frontmatter fields parsed and stored on `FileSkill`: `license`, `compatibility`
    (max 500 chars), and `metadata` (arbitrary key-value map).
  - `name` validated per spec: lowercase letters, digits, and hyphens only; no leading/trailing
    hyphen; no consecutive hyphens (`--`); max 64 characters.
  - `description` max 1024 characters enforced.
  - **Argument substitution aligned with Claude Code standard** — `substitute_args` uses
    `$ARGUMENTS` (full argument string) instead of named/positional `$arg_name` / `$N` patterns.
    If `$ARGUMENTS` is absent from the body and arguments are supplied, they are appended as
    `ARGUMENTS: <value>`. `argument-hint` is retained as a display-only extension field.

- **`BaseSkillRouter`** — public base class that exposes the shared routing
  pipeline (prompt building, `_extract_json`, LRU cache).  Subclass it and implement
  only `_call_model(prompt: str) -> str` to integrate any model provider without
  re-implementing the boilerplate.  `LLMSkillRouter` is a thin subclass of
  `BaseSkillRouter`.  `BaseSkillRouter` is exported from the package top-level.

- **Phase 1 — Core injection:** `Skill` abstract base class, `SkillHooks`
  (`AgentHooks` subclass) that prepends skill prompt blocks to `input_items` before
  every LLM call, `is_enabled()` gate.

- **Phase 2 — Registry & routing:** `SkillRegistry` as the central skill store;
  `SkillRouter` protocol and `LLMSkillRouter` (LRU-cached LLM-based routing via
  `when_to_use`); always-on vs routed skill distinction; first-call manifest injection;
  `RunSkillHooks` (`RunHooks` subclass) spanning all agents in a run; double-injection
  guard via `RunState.injected_this_call`; `make_invoke_skill_tool` for model-initiated
  skill invocation.

- **Phase 3 — File-based skills:** `FileSkill` concrete `Skill` subclass loaded from
  `SKILL.md` files; YAML frontmatter parsing (`description`, `allowed-tools`,
  `argument-hint`, `user-invocable`); `SkillSource` enum (`BUNDLED`, `USER`, `PROJECT`,
  `EXTRA`); `SkillConfig` dataclass for loader configuration; `load_skills_from_dir` and
  `load_all_skills` with concurrent async I/O, `realpath()` deduplication, and user >
  project > extra priority ordering; `substitute_args` with `$ARGUMENTS` and `${VAR}`
  variable substitution; pre-substitution safety validation rejecting null bytes, YAML
  boundaries, bidirectional override characters, and role-header sequences; path traversal
  guard (`assert_within_base`); `get_prompt_blocks` result cache keyed by `args` string;
  `allowed-tools` and `user-invocable` surfaced in manifest.

- **Phase 4 — Advanced triggering:** `triggers_after_tools` and `triggers_after_turn`
  attributes on `Skill`; `RunState.pending_skills` list (ContextVar-scoped);
  `SkillRegistry.get_triggered_by_tool` and `get_post_turn`; `on_tool_end` on both
  `SkillHooks` and `RunSkillHooks` queues triggered skills into `pending_skills`;
  `on_llm_end` queues post-turn skills; `_drain_pending` drains and injects pending
  blocks at the next `on_llm_start` with concurrency-safe claim-before-await
  semantics; name-based deduplication prevents double-queuing when both hook types
  are active simultaneously.

- **Phase 5 — Hardening:** `SkillSource` trust level stored on every `FileSkill`
  and surfaced in debug log output; path traversal protection on all file loads;
  canonical `realpath()` deduplication across layers; argument injection safety
  validation; `allowed-tools` surfaced in manifest as informational guidance for
  the model; 96% test coverage across 324 tests; Google-style docstrings on all
  public API symbols.

### Fix

- **`LLMSkillRouter`: no `response_format` default** — `response_format={"type":
  "json_object"}` is not sent by default because it is unsupported by several
  providers (AWS Bedrock via LiteLLM, some Azure configurations, Ollama).  Use the
  `use_response_format=True` constructor flag to opt in explicitly if you need it.

- **`LLMSkillRouter`: robust JSON extraction** — the router does not raise on
  prose-wrapped responses or extended-thinking content-block lists (e.g. Claude
  `claude-haiku-4-5` / `claude-sonnet-4-5`).  A `_extract_json` helper handles
  all observed response shapes: plain JSON, JSON embedded in prose, `None` / empty,
  and `list[dict]` content blocks.
