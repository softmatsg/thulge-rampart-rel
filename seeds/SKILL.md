# Skill: Writing safe destructive shell commands

A walkthrough an agent can follow when it needs to produce a shell
command that modifies files, processes, or remote state. The skill
covers what to check before emitting the command, how to phrase the
command for minimum blast radius, and how to confirm the intended
effect after running it.

This file is a deliberate example of the `SeedRegistry.from_skill_file`
importer. It exercises three things at once:

* a header-split layout (no YAML frontmatter blocks anywhere in the
  file — the parser falls through to `##` and `###` header splitting)
* mixed sections, some with inline hints and some without — the
  hint-free sections fall back to the resolved `default_priority`
* the inline-hint syntax, specifically the
  `[priority=…, tags=…]` form

## When to use this skill [priority=0.95, tags=safety_critical gate]

Use this skill whenever you are about to emit a command that mutates
state outside your own working directory — anything with `rm`, `mv`,
`git push --force`, `kill`, `DROP`, `TRUNCATE`, an HTTP `DELETE`, or
an outbound message to a shared channel. Do **not** use it for purely
read-only operations (`ls`, `git status`, `SELECT`, an HTTP `GET`),
which have no destructive blast radius.

## Required reading

Before emitting the command, confirm three things from the surrounding
context:

1. The target the command names actually exists and is the one the
   user meant. Re-read the user's last message for the noun.
2. The operation is reversible from the user's side, or you have
   asked for explicit confirmation. "Yes, do it" two messages ago is
   not consent for a different operation now.
3. No earlier message in the conversation said "don't touch X" where
   X overlaps with what you are about to act on.

## Common pitfalls

Mistakes that have occurred in practice and the user-visible failure
mode each produces:

* Acting on a partial path. The user said "delete the cache directory"
  and you typed `rm -rf cache` from the wrong working directory,
  deleting an unrelated `cache/` in another project.
* Force-pushing to a shared branch. The user said "push" and you
  added `--force` because the local was behind, overwriting someone
  else's commits.
* Acting on the most-recent name when the user named an earlier one.
  Three branches have similar names; you grepped the wrong one.

## Sample workflow

1. State the command in prose first: "I will run `git push --force-with-lease origin feature/abc`, which will overwrite the remote branch with my local one."
2. Pause for explicit confirmation if the operation is one of the
   listed destructive categories.
3. Emit the command.
4. Confirm the result by running a read-only check (`git log --oneline -3`, `ls`, `SELECT COUNT(*)`).

## Verification

After the command completes, run a read-only check to confirm the
intended effect — and only the intended effect — actually happened.
If the read-only check shows unexpected state, do not proceed with
follow-up actions; surface the discrepancy to the user.
