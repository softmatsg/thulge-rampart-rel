# Example seed files

This directory ships example instruction-block seed files for the
quick-start in the top-level [README.md](../README.md).

Each file uses the YAML-frontmatter format the RAMPART parser
accepts: blocks are delimited by `---` lines, every block carries a
`name`, an optional `priority` (0.0–1.0), and optional `tags`. A file
with no frontmatter at all loads as a single anonymous block named
after the filename stem — that is the `SKILL.md` / `CLAUDE.md`
compatibility path.

| File | Description |
|---|---|
| `assistant_style.md` | Six style and output-format blocks (language, length, tone, code-block format, hedging, meta-commentary), in YAML-frontmatter format. Load with `BlockRegistry.from_files`. |
| `SKILL.md` | An example skill file in `SKILL.md` / `CLAUDE.md` shape — five `##`-delimited sections with one inline `[priority=…, tags=…]` hint. Demonstrates the Path B (header-split) import path. Load with `SeedRegistry.from_skill_file`. |

Drop your own seed files alongside these. `BlockRegistry.from_files`
accepts a list of paths and parses them all into one registry,
preserving the in-file block order.
