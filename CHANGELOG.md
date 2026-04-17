# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.1.0 (2025-01-01)

### Feat

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
  `SKILL.md` files; YAML frontmatter parsing (`description`, `when_to_use`,
  `allowed-tools`, `arguments`, `argument-hint`, `user-invocable`); `SkillSource`
  enum (`BUNDLED`, `USER`, `PROJECT`, `EXTRA`); `SkillConfig` dataclass for loader
  configuration; `load_skills_from_dir` and `load_all_skills` with concurrent async
  I/O, `realpath()` deduplication, and user > project > extra priority ordering;
  `substitute_args` with named, positional, and `${VAR}` variable substitution;
  pre-substitution safety validation rejecting null bytes, YAML boundaries,
  bidirectional override characters, and role-header sequences; path traversal guard
  (`assert_within_base`); `get_prompt_blocks` result cache keyed by `args` string;
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
