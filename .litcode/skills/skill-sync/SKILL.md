---
name: skill-sync
description: Share installed Agent Skills between LitCode and supported local coding agents. Use when the user asks to sync, expose, or reuse Skills across Codex, Claude Code, OpenCode, Cursor, Gemini CLI, or GitHub Copilot.
---

# Skill Sync

Use LitCode's Skill manager rather than copying Skill contents by hand.

- List the canonical Skills first with `litcode skill list` or `/skill list`.
- For project Skills, run `litcode skill sync [names...] --agent <agent>` or `/skill sync [names...] --agent <agent>`.
- Add `--scope user` to share user-level Skills from `~/.agents/skills/`.
- Supported agent names are `codex`, `claude-code`, `opencode`, `cursor`, `gemini-cli`, and `github-copilot`.
- If no `--agent` is provided, sync only to detected local Agent directories.

The canonical Skill remains the single source of truth. Sync creates directory symlinks and never overwrites an existing real directory or a link to another source. If a target conflicts, report the exact path and let the user decide how to resolve it; do not delete or replace it automatically.
