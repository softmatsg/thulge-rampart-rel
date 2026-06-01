"""Format-extensible seed file parser. Session-agnostic by construction.

The parser is a pure file reader. It knows nothing about runtime identifiers,
embeddings, or the live registry. Every block it returns has
``runtime_id=""`` as a sentinel — ``BlockRegistry.from_files`` overwrites it
with a fresh UUID4 during insertion. Two consequences follow:

* Address randomisation stays the registry's responsibility, where it
  belongs (a new UUID per session, not per file). The parser does not import
  ``rampart.security``.
* The parser is trivially testable from any file path without spinning up
  a registry or generating UUIDs. ``parse_file(path)`` is a pure function.

The grammar supports multiple frontmatter sections per file. Each
section is delimited by lines containing only ``---``; content runs
until the next delimiter or EOF. A file with no frontmatter at all
loads as a single anonymous block named after the filename stem —
this is the SKILL.md compatibility path that lets existing
Claude Code / Gemini CLI skill files import without modification.

Format dispatch is via the module-level ``PARSERS`` dict. Users can register
additional file extensions before cold start::

    from rampart.parser import PARSERS
    PARSERS[".myext"] = my_custom_parser
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rampart.block import InstructionBlock


class ParseError(Exception):
    """Raised when a seed file cannot be parsed into valid InstructionBlocks.

    Covers malformed YAML, frontmatter that is not a YAML mapping, missing
    required keys, and ill-typed values for known optional keys.
    """


class UnsupportedFormatError(Exception):
    """Raised when ``parse_file`` is given a path whose extension has no
    registered parser in ``PARSERS``.
    """


@dataclass
class BlockSource:
    """Descriptor for a seed file plus an optional name filter.

    Used by ``BlockRegistry.from_files`` and ``SeedRegistry.from_files`` so
    callers can express "load only these blocks from this file" without
    splitting the file or post-filtering the registry. Plain ``str`` and
    ``Path`` arguments to those entry points are coerced to a default
    ``BlockSource(path=p, names=None, must_match=True)``, so existing
    call sites that pass bare paths keep working unchanged.

    Not to be confused with ``rampart.block.BlockSourceKind`` — that is
    the literal type tag (``"seed"`` / ``"agent"`` / ``"orchestrator"``)
    on each ``InstructionBlock``. ``BlockSource`` here is a *file-level*
    descriptor; ``BlockSourceKind`` is a *block-level* author class.

    Attributes:
        path: Filesystem path to the seed file. ``str`` and ``Path`` both
            accepted; coerced to ``Path`` lazily by callers.
        names: Optional whitelist of ``semantic_name`` values to keep
            from the parsed file. ``None`` (default) means "load every
            block from the file".
        must_match: When ``True`` (default), if any name in ``names`` is
            not found in the file, the loader raises
            ``BlockNotFoundError`` listing the missing names. When
            ``False``, missing names are silently skipped — useful when
            the seed library is shared across deployments and a given
            consumer only wants the subset that happens to be present.
    """

    path: str | Path
    names: list[str] | None = None
    must_match: bool = True

    def filter_blocks(
        self, blocks: list[InstructionBlock]
    ) -> tuple[list[InstructionBlock], list[str]]:
        """Apply this source's name filter to a parsed block list.

        Each entry in ``self.names`` may be either a **bare** semantic
        name (matches the block's ``semantic_name`` field directly) or
        a **namespaced** ``stem.semantic_name`` form (matches when the
        file's path stem and the block's semantic name together form
        that string). The two forms are equivalent at the file level
        because the within-file index is keyed by ``semantic_name``,
        which is unique per file; the namespaced form exists so the
        same name list can be reused unchanged across the
        ``SeedRegistry`` API where cross-file ambiguity matters.

        A block is included in the result if **either** form matches.
        A block is included at most once even if both forms in
        ``self.names`` reference it.

        Args:
            blocks: Blocks as returned by ``parse_file(self.path)``.

        Returns:
            ``(kept_blocks, missing_names)`` — ``kept_blocks`` preserves
            the input order, ``missing_names`` is the requested-but-
            not-found subset (empty if ``self.names is None``). The
            caller decides whether ``missing_names`` is fatal based on
            ``self.must_match``.
        """
        if self.names is None:
            return list(blocks), []
        stem = Path(self.path).stem
        wanted = set(self.names)
        kept: list[InstructionBlock] = []
        matched: set[str] = set()
        for b in blocks:
            bare = b.semantic_name
            namespaced = f"{stem}.{bare}"
            block_matched = False
            if bare in wanted:
                matched.add(bare)
                block_matched = True
            if namespaced in wanted:
                matched.add(namespaced)
                block_matched = True
            if block_matched:
                kept.append(b)
        missing = [n for n in self.names if n not in matched]
        return kept, missing


def _coerce_block_source(item: BlockSource | str | Path) -> BlockSource:
    """Promote a plain path to a default ``BlockSource``.

    Keeps the loader entry points polymorphic over
    ``BlockSource | str | Path`` without forcing every existing caller
    to wrap their paths.
    """
    if isinstance(item, BlockSource):
        return item
    return BlockSource(path=item)


_BLOCK_SPLIT = re.compile(r"(?m)^---\s*$")
"""Split file text on lines containing only ``---`` (and optional whitespace)."""

_NAME_SAFE = re.compile(r"[\s\-]+")
"""Match runs of whitespace or hyphens for the anonymous-block name fallback."""


def parse_markdown_frontmatter(path: Path) -> list[InstructionBlock]:
    """Parse a YAML-frontmatter markdown file into one or more seed blocks.

    A file with no leading ``---`` is treated as a single anonymous block:
    its content is the entire file text and its semantic name is the
    filename stem with whitespace and hyphens replaced by underscores.

    A file that does start with ``---`` is split into ``(frontmatter, content)``
    pairs at every line that contains only ``---``. Each pair must include
    a YAML mapping with at least a ``name`` key. Optional keys are
    ``priority`` (float in ``[0.0, 1.0]``, default 0.5) and ``tags``
    (list of strings, default empty). Other keys (``description``, ``version``)
    are accepted and silently ignored — they are documented in the grammar
    for human authors but do not map onto any field of ``InstructionBlock``.

    Args:
        path: Path to the seed file. Read as UTF-8 with BOM-stripping.

    Returns:
        Ordered list of seed blocks. Each has ``source='seed'``,
        ``runtime_id=''`` (sentinel), and ``created_at=0``. The caller —
        ``BlockRegistry.from_files`` — fills in ``runtime_id`` and
        ``created_at`` before insertion.

    Raises:
        ParseError: If the file starts with ``---`` but has no closing
            delimiter, if a frontmatter section is malformed YAML or not a
            mapping, if the required ``name`` key is missing, or if a
            documented optional key has the wrong type or out-of-range value.
    """
    text = path.read_text(encoding="utf-8-sig")
    if not text.lstrip().startswith("---"):
        return [_anonymous_block(path, text)]

    parts = _BLOCK_SPLIT.split(text)
    # parts[0] is the pre-amble before the first delimiter (whitespace only).
    # Subsequent pairs at indices (1, 2), (3, 4), ... are (frontmatter, content).
    if len(parts) < 3:
        raise ParseError(
            f"File {path} starts with '---' but has no closing delimiter."
        )

    blocks: list[InstructionBlock] = []
    i = 1
    while i < len(parts):
        if i + 1 >= len(parts):
            raise ParseError(
                f"Unterminated frontmatter section near the end of {path}."
            )
        meta_text = parts[i]
        content_text = parts[i + 1].strip()
        meta = _parse_meta(meta_text, path)
        blocks.append(_block_from_meta(meta, content_text, path))
        i += 2

    return blocks


PARSERS: dict[str, Callable[[Path], list[InstructionBlock]]] = {
    ".md": parse_markdown_frontmatter,
}
"""Module-level format registry. Maps file extension (with leading dot,
lowercase) to a callable that returns the file's seed blocks. Mutate before
cold start to register custom file types.
"""


def parse_file(path: Path) -> list[InstructionBlock]:
    """Dispatch a path to the parser registered for its file extension.

    Args:
        path: Seed file path. Extension is read from ``path.suffix`` and
            lowercased before lookup, so ``.MD`` and ``.md`` resolve to the
            same parser.

    Returns:
        Whatever the matched parser returns — a list of seed
        ``InstructionBlock`` instances with ``runtime_id=""``.

    Raises:
        UnsupportedFormatError: If no parser is registered for ``path.suffix``.
    """
    suffix = path.suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        raise UnsupportedFormatError(
            f"No parser registered for extension {suffix!r}; "
            f"register one in rampart.parser.PARSERS before cold start."
        )
    return parser(path)


def _parse_meta(meta_text: str, path: Path) -> dict[str, Any]:
    """Parse a single frontmatter YAML section into a Python dict."""
    try:
        meta = yaml.safe_load(meta_text)
    except yaml.YAMLError as exc:
        raise ParseError(f"Malformed YAML frontmatter in {path}: {exc}") from exc
    if meta is None:
        raise ParseError(f"Empty frontmatter section in {path}.")
    if not isinstance(meta, dict):
        raise ParseError(
            f"Frontmatter must be a YAML mapping in {path}; "
            f"got {type(meta).__name__}."
        )
    return meta


def _block_from_meta(
    meta: dict[str, Any],
    content: str,
    path: Path,
) -> InstructionBlock:
    """Build a seed ``InstructionBlock`` from a parsed frontmatter dict.

    Validates the documented keys (``name``, ``priority``, ``tags``) and
    raises ``ParseError`` on type or range errors. Other keys are tolerated
    silently so authors can include ``description`` or ``version`` for
    human readers without breaking the parser.
    """
    if "name" not in meta:
        raise ParseError(f"Missing required 'name' key in frontmatter of {path}.")
    name = meta["name"]
    if not isinstance(name, str):
        raise ParseError(
            f"'name' must be a string in {path}; got {type(name).__name__}."
        )

    priority_raw = meta.get("priority", 0.5)
    if isinstance(priority_raw, bool) or not isinstance(priority_raw, int | float):
        raise ParseError(
            f"'priority' must be a number in {path}; "
            f"got {type(priority_raw).__name__}."
        )
    priority = float(priority_raw)
    if not 0.0 <= priority <= 1.0:
        raise ParseError(
            f"'priority' must be in [0.0, 1.0] in {path}; got {priority}."
        )

    tags_raw = meta.get("tags", [])
    if not isinstance(tags_raw, list):
        raise ParseError(
            f"'tags' must be a list in {path}; got {type(tags_raw).__name__}."
        )
    if not all(isinstance(t, str) for t in tags_raw):
        raise ParseError(f"'tags' must contain only strings in {path}.")

    return InstructionBlock(
        semantic_name=name,
        runtime_id="",
        content=content,
        source="seed",
        priority=priority,
        tags=list(tags_raw),
    )


def _anonymous_block(path: Path, text: str) -> InstructionBlock:
    """Build a single seed block from a file with no frontmatter.

    The semantic name is the filename stem with runs of whitespace and
    hyphens collapsed to single underscores, then trimmed. This makes a
    Claude Code SKILL.md (typically headerless and using dashes in its
    filename, e.g. ``database-migration-helper.md``) load with a clean
    Python-friendly identifier (``database_migration_helper``).
    """
    stem = path.stem
    name = _NAME_SAFE.sub("_", stem).strip("_") or "anonymous"
    return InstructionBlock(
        semantic_name=name,
        runtime_id="",
        content=text.strip(),
        source="seed",
        priority=0.5,
        tags=[],
    )


__all__ = [
    "PARSERS",
    "ParseError",
    "UnsupportedFormatError",
    "parse_file",
    "parse_markdown_frontmatter",
]
