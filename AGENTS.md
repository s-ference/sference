# Coding agents (Cursor, Claude Code, …)

Use sference from **your** app repo—not by editing this OSS repo. Two layers:

| Layer | File | Use when |
|-------|------|----------|
| **Prompt** (start here) | [PROMPT.txt](PROMPT.txt) | Vibe coding: paste into project rules, `CLAUDE.md`, or the first agent message |
| **Full skill** | [SKILL.md](SKILL.md) | Persistent Cursor skill, or when the agent needs examples and a method map |

Raw URLs on GitHub `main` (no clone required):

- Prompt: `https://raw.githubusercontent.com/s-ference/sference/main/PROMPT.txt`
- Skill: `https://raw.githubusercontent.com/s-ference/sference/main/SKILL.md`

## 1. Short prompt (recommended)

Open [PROMPT.txt](PROMPT.txt), copy all lines, paste into:

- **Cursor** — Project Rules, or User Rules
- **Claude Code** — `CLAUDE.md` or custom instructions
- **Other tools** — `AGENTS.md` in your repo, or the tool’s system / custom instructions field

Optional one-liner to the agent:

```text
Follow the sference SDK rules in PROMPT.txt (sference-sdk, background responses, stream presets).
```

## 2. Full skill (Cursor install)

For automatic discovery in **Cursor**, copy the entire [SKILL.md](SKILL.md) (keep the YAML frontmatter) into your project:

```text
.cursor/skills/sference-sdk/SKILL.md
```

**Claude Code** — append [SKILL.md](SKILL.md) to `CLAUDE.md`, or use your project’s skill mechanism.

Do not duplicate rules in multiple places in the same project—pick **prompt only**, or **prompt + skill file**, not conflicting copies.

## Human docs

- [README.md](README.md) — install CLI / SDK
- [sdk-python/README.md](sdk-python/README.md) — library reference
- [https://sference.com/docs/sdk](https://sference.com/docs/sdk) — website quickstart for agents
