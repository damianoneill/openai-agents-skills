---
paths:
  - "**/*_test*"
---

# Testing Rules

- Use table-driven / parameterised tests for any function with more than one meaningful case
- Test files live alongside the code they test
- Integration tests that hit external services are gated with a build tag or environment flag — they must not run as part of the default unit suite
- Use interfaces and constructor injection so tests can swap real implementations for fakes — never patch globals
- Test the `usecase` layer against fake implementations of domain ports, not real infrastructure
- Test names follow a consistent `<Function>_<Scenario>` convention (e.g. `test_send_message_returns_error_on_timeout`)
- Aim for a fast unit suite — most tests should complete in milliseconds
- Write tests to verify behaviour, not to hit a coverage number — do not add tests solely to increase coverage
