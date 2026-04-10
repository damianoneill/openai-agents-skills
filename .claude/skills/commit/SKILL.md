---
name: commit
description: Stage all changes and create a conventional commit. Use when the user asks to commit work.
argument-hint: "[optional commit message override]"
allowed-tools: Bash(git *)
---

# Create a Conventional Commit

Review the current changes and create a well-formed commit.

## Current status
!`git status`

## Staged and unstaged diff
!`git diff HEAD`

## Recent commits (for style reference)
!`git log --oneline -5`

---

Follow these steps:

1. Review the diff above to understand what changed.

2. **Design review** — before staging anything, assess the approach against the requirement:
   - Is the design proportionate to the problem? Would a simpler design have delivered the same behaviour?
   - Does the implementation solve only what was asked, or did it solve a broader, hypothetical problem?
   - Were any design patterns, layers, or indirections introduced that are not justified by a current requirement?
   - Does the scope of the change match the scope of the feature, or has it grown beyond it?
   - Is the implementation placed correctly — could the logic live in a better location?
   - Are there edge cases or error conditions that are unhandled or incorrectly handled?
   - Do the tests assert meaningful behaviour, or do they only verify that a fake was called?
   - If any of these reveal a problem, raise it and propose a fix before committing.

3. **README check** — review `README.md` against the diff:
   - Does the diff add, remove, or rename a public interface, type, or port?
   - Does it add a new package or example?
   - Does it change observable behaviour?
   - If yes to any of the above, update `README.md` before staging. If no update is needed, state why briefly.

4. Stage all relevant changes: `git add -A` (exclude secrets and generated files).

5. Draft a commit message following conventional commits:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation only
   - `chore:` tooling, dependencies, config
   - `refactor:` code change that neither fixes a bug nor adds a feature
   - `test:` adding or updating tests

6. If `$ARGUMENTS` was provided, use that as the commit message body.

7. Create the commit.

Keep the subject line under 72 characters. Add a body only if the "why" needs explanation.

**Never** add a `Co-Authored-By` trailer or any attribution to Claude or Anthropic.
