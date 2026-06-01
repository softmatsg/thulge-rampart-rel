"""Standalone helpers for inspecting user-provided skill-file inputs.

Currently houses :func:`validate_splitter`, a one-shot probe for the
``llm_splitter`` callable accepted by
:meth:`rampart.seed_registry.SeedRegistry.from_skill_file`. The probe
runs the user's callable on a sample text and reports whether the
return shape is what the import path expects, so authors can verify
their implementation in isolation before letting it loose on a full
skill library.

The probe never imports an LLM client of its own; the caller is
fully responsible for what the callable does internally.
"""

from __future__ import annotations

from typing import Any

from rampart.skill import SkillSplitter


def validate_splitter(
    callable_: SkillSplitter,
    sample_text: str,
) -> dict[str, Any]:
    """Run ``callable_`` against ``sample_text`` and report on the return value.

    The library treats two return shapes as legitimate:

    * a string — content only, all other fields default
    * a dict — must contain ``content``; may additionally contain
      ``name`` (str), ``priority`` (float), ``tags`` (list[str]),
      ``evictable`` (bool)

    Anything else is reported as a warning. The probe collects every
    issue it finds (rather than short-circuiting on the first) so a
    user iterating on their splitter sees the full set of fixes
    needed in one round.

    Args:
        callable_: The user's ``llm_splitter`` callable. Will be
            invoked exactly once with ``sample_text`` as its only
            positional argument.
        sample_text: The string to feed to ``callable_``. A few
            sentences is plenty; the goal is to exercise the
            callable's return shape, not its semantic correctness.

    Returns:
        A dict with the following keys:

        * ``valid`` (``bool``) — ``True`` if every entry in the
          returned list is either a string or a dict with at least
          a ``content`` field of ``str``-compatible type. ``False``
          otherwise (including when the callable raises, returns
          ``None``, or returns something other than a list).
        * ``n_blocks`` (``int``) — number of usable entries the
          callable returned. Excludes malformed entries.
        * ``fields_present`` (``list[str]``) — sorted list of every
          dict key seen across all returned entries. Useful for
          verifying that a custom ``priority``/``tags`` implementation
          is actually emitting those fields.
        * ``warnings`` (``list[str]``) — human-readable diagnostic
          strings describing each issue found.
    """
    warnings: list[str] = []
    fields_present: set[str] = set()
    n_blocks = 0
    valid = True

    try:
        result = callable_(sample_text)
    except Exception as exc:  # noqa: BLE001 — diagnostic helper
        return {
            "valid": False,
            "n_blocks": 0,
            "fields_present": [],
            "warnings": [
                f"callable raised {type(exc).__name__}: {exc}"
            ],
        }

    if result is None:
        return {
            "valid": False,
            "n_blocks": 0,
            "fields_present": [],
            "warnings": [
                "callable returned None; expected a list of strings "
                "or dicts. If you have not implemented llm_splitter, "
                "pass llm_splitter=None to from_skill_file()."
            ],
        }

    if not isinstance(result, list):
        return {
            "valid": False,
            "n_blocks": 0,
            "fields_present": [],
            "warnings": [
                f"callable returned {type(result).__name__}; "
                "expected list[dict | str]."
            ],
        }

    if not result:
        warnings.append(
            "callable returned an empty list; from_skill_file() "
            "would log a warning and fall back to a single block."
        )
        valid = False

    for idx, entry in enumerate(result):
        if isinstance(entry, str):
            n_blocks += 1
            continue
        if isinstance(entry, dict):
            fields_present.update(entry.keys())
            if "content" not in entry:
                warnings.append(
                    f"entry {idx} is a dict without 'content'; "
                    "from_skill_file() would skip it."
                )
                valid = False
                continue
            content = entry["content"]
            if not isinstance(content, str):
                warnings.append(
                    f"entry {idx}'s 'content' is "
                    f"{type(content).__name__}; expected str."
                )
                valid = False
                continue
            tags = entry.get("tags")
            if tags is not None and not (
                isinstance(tags, list)
                and all(isinstance(t, str) for t in tags)
            ):
                warnings.append(
                    f"entry {idx}'s 'tags' is "
                    f"{type(tags).__name__}; expected list[str]. "
                    "from_skill_file() would log a warning and "
                    "ignore tags for this entry."
                )
            priority = entry.get("priority")
            if priority is not None:
                try:
                    float(priority)
                except (TypeError, ValueError):
                    warnings.append(
                        f"entry {idx}'s 'priority' "
                        f"({priority!r}) is not numeric; "
                        "from_skill_file() would fall back to "
                        "default_priority for this entry."
                    )
            n_blocks += 1
            continue
        warnings.append(
            f"entry {idx} is {type(entry).__name__}; expected "
            "str or dict. from_skill_file() would skip it."
        )
        valid = False

    return {
        "valid": valid,
        "n_blocks": n_blocks,
        "fields_present": sorted(fields_present),
        "warnings": warnings,
    }


__all__ = ["validate_splitter"]
