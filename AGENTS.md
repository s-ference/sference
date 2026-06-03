# Agent instructions (sference SDK)

**Copy [SKILL.md](SKILL.md)** into your coding agent so it knows how to write `sference-sdk` Python (Responses API with `background=True`, streams as presets, batches when needed).

You do not need to clone this repository to use the skill.

## Get the file

| Source | Link |
|--------|------|
| In this repo | [SKILL.md](SKILL.md) — open and copy all contents |
| GitHub (raw) | https://raw.githubusercontent.com/s-ference/sference/main/SKILL.md |

## Where to paste

**Cursor** — in your app project (not this repo):

```text
.cursor/skills/sference-sdk/SKILL.md
```

Create the `sference-sdk` folder, paste the full file (keep the YAML frontmatter at the top).

**Claude Code** — paste into your project’s **`CLAUDE.md`**, or add as a project skill / custom instructions per your Claude Code setup.

**Other agents** — paste into project rules, `AGENTS.md`, or the tool’s “custom instructions” field.

Tell the agent: *“Follow the sference SDK skill”* or @-mention the file if your editor supports it.

## Human docs

- Install and quick start: [README.md](README.md)
- SDK details: [sdk-python/README.md](sdk-python/README.md)
