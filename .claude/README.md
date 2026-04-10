# Claude Code Configuration

## Settings

- `settings.json` — shared project permissions (committed)
- `settings.local.json` — your local overrides (gitignored, not committed)

## Rules

Path-scoped instructions that load automatically based on the file being worked on:

| File               | Applies to   |
| ------------------ | ------------ |
| `rules/testing.md` | `**/*_test*` |

## Skills (Custom Slash Commands)

Invoke these with `/skill-name` in Claude Code:

| Command   | Description                                    |
| --------- | ---------------------------------------------- |
| `/commit` | Stage changes and create a conventional commit |

## Environment Variables

Copy `.env.example` to `.env` and fill in your values. Never commit `.env`.
