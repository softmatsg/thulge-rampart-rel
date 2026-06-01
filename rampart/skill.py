"""Skill-file import path. Backs ``SeedRegistry.from_skill_file()``.

The library entry point handles three structural file shapes:

* **Path A — frontmatter blocks.** A file containing one or more YAML
  frontmatter sections (``---``-delimited) is parsed via
  ``parse_markdown_frontmatter``. Each block's name is prefixed with
  the resolved namespace; frontmatter priority and tags take
  precedence over every other source.
* **Path B — header split.** A file with no frontmatter but
  containing ``##`` or ``###`` headers is split on those headers.
  The block name is the header text, lowercased, with spaces
  replaced by underscores and non-alphanumeric characters stripped,
  then prefixed with the namespace. Inline hints of the form
  ``[priority=0.9]`` or ``[priority=0.9, tags=hardware timing]``
  in the header text override the resolved ``default_priority`` for
  that section but are themselves overridden by frontmatter (which
  cannot coexist with header-split mode in a single file).
* **Path C — single block.** A file with neither frontmatter nor
  headers loads as one block. The block name is
  ``namespace.filename_stem`` or just ``namespace`` when the two
  match. Priority defaults to the resolved ``default_priority``;
  tags default to the empty list. If a caller supplies an
  ``llm_splitter`` callable, it is invoked here in place of the
  single-block fallback so the user's own LLM can split the file
  into multiple blocks.

Namespace resolution chain (highest precedence first):

1. Explicit ``namespace`` parameter passed to ``from_skill_file``.
2. ``root``-relative path. Given ``root="/project"`` and
   ``path="/project/api/skills/SKILL.md"``, the namespace becomes
   ``api.skills``. The filename itself does not contribute when
   root is supplied.
3. Immediate parent directory name (``path.parent.name``).
4. Filename stem (``path.stem``) as last resort. The stem is also
   used as the *content namespace* for Path C when none of the
   higher-precedence sources apply.

Priority resolution chain (highest precedence first):

1. Block-level frontmatter ``priority`` field (Path A only).
2. Inline header hint ``[priority=...]`` (Path B only). These two
   are mutually exclusive in a single file by construction —
   frontmatter delimits Path A, headers delimit Path B, and a file
   cannot be both at once.
3. ``default_priority`` parameter passed to ``from_skill_file``.
4. **Library default**: ``0.9`` if the filename stem (case-
   insensitive) is one of ``{"skill", "claude"}`` — matching the
   convention for Anthropic SKILL.md and CLAUDE.md files where the
   author has marked the file as the canonical "how to behave"
   document and the priority should be high. Otherwise ``0.5``.

Collisions are handled defensively: if a resolved block name is
already present in the target ``SeedRegistry``, a ``WARNING`` is
logged naming both the existing source file and the incoming file,
and the incoming block is skipped. Existing blocks are never
overwritten silently.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rampart.block import InstructionBlock
from rampart.parser import (
    ParseError,
    _BLOCK_SPLIT,
    _block_from_meta,
    _parse_meta,
)


_LOG = logging.getLogger("rampart.skill")


# Splitter callable signature. Each element of the returned list is
# either a string (content-only — all other fields fall back to
# defaults) or a dict carrying any subset of {content, name, priority,
# tags, evictable}; only ``content`` is required when the dict form is
# used.
SkillSplitter = Callable[[str], list[dict[str, Any] | str] | None]


# ----- namespace resolution --------------------------------------------------


def resolve_namespace(
    path: Path,
    root: Path | str | None,
    namespace: str | None,
) -> str:
    """Resolve the dot-notation namespace for a skill file.

    Precedence chain (highest first):

    1. Explicit ``namespace`` parameter, returned verbatim.
    2. Root-relative path translation. ``root="/project"`` and
       ``path="/project/api/skills/SKILL.md"`` resolve to
       ``"api.skills"``. The filename does not contribute under
       this branch.
    3. Immediate parent directory name (``path.parent.name``).
    4. Filename stem (``path.stem``) as last resort.

    Args:
        path: The skill file's path.
        root: Optional anchor for branch 2. When set, the path must
            be relative to ``root`` after both are made absolute. If
            not, the loader falls through to branches 3-4 rather
            than raising.
        namespace: Optional explicit namespace. Trumps every other
            source.

    Returns:
        The resolved namespace as a dot-separated string. Logged at
        DEBUG level so users can verify which branch fired.
    """
    if namespace is not None:
        resolved = namespace
        _LOG.debug(
            "namespace resolved via explicit parameter: %r", resolved,
        )
        return resolved
    if root is not None:
        try:
            relative = Path(path).resolve().relative_to(
                Path(root).resolve(),
            )
            parts = list(relative.parts[:-1])  # drop filename
            if parts:
                resolved = ".".join(parts)
                _LOG.debug(
                    "namespace resolved via root-relative path: %r",
                    resolved,
                )
                return resolved
        except ValueError:
            # path is not under root — fall through.
            pass
    parent = path.parent.name
    if parent:
        _LOG.debug(
            "namespace resolved via parent directory name: %r", parent,
        )
        return parent
    resolved = path.stem
    _LOG.debug(
        "namespace resolved via filename stem (last resort): %r",
        resolved,
    )
    return resolved


# ----- priority resolution ---------------------------------------------------


def library_default_priority(path: Path) -> float:
    """Library-default priority for a skill file.

    Returns ``0.9`` if the filename stem is ``"skill"`` or ``"claude"``
    (case-insensitive), ``0.5`` otherwise. The high default for
    SKILL.md / CLAUDE.md mirrors the Anthropic convention where the
    file represents the canonical "how to behave" instruction set
    and should sit near the top of the registry.
    """
    return 0.9 if path.stem.lower() in {"skill", "claude"} else 0.5


def resolve_default_priority(
    path: Path, default_priority: float | None,
) -> float:
    """Pick the effective ``default_priority`` for a skill file.

    Returns the ``default_priority`` parameter when set, otherwise
    falls back to :func:`library_default_priority`.
    """
    if default_priority is not None:
        return default_priority
    return library_default_priority(path)


# ----- inline hint parser ----------------------------------------------------


_INLINE_HINT = re.compile(r"\[([^\[\]]+)\]")
"""Match the bracketed hint suffix in a header line."""


def parse_inline_hint(
    header_text: str,
) -> tuple[str, dict[str, Any]]:
    """Strip the bracketed hint from a header and parse its key=value pairs.

    Recognised keys:

    * ``priority`` — float in ``[0.0, 1.0]``. Out-of-range or
      non-numeric values are ignored with a logged WARNING.
    * ``tags`` — space-separated list of strings. Empty list means
      "no tags".

    Unknown keys are tolerated but ignored — they are logged at
    DEBUG level so authors who experiment with new hint shapes do
    not get spammed but can still verify the parser saw their input.

    Args:
        header_text: The full header text, e.g.
            ``"Section name [priority=0.9, tags=hardware timing]"``.

    Returns:
        ``(stripped_text, hint_dict)``. ``stripped_text`` has the
        ``[...]`` removed and trailing whitespace trimmed. Empty
        ``hint_dict`` if no hint was present or the hint was
        unparseable.
    """
    match = _INLINE_HINT.search(header_text)
    if match is None:
        return header_text.strip(), {}
    stripped = (
        header_text[: match.start()] + header_text[match.end():]
    ).strip()
    hint_body = match.group(1)
    out: dict[str, Any] = {}
    for raw_pair in hint_body.split(","):
        pair = raw_pair.strip()
        if not pair or "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "priority":
            try:
                pf = float(value)
            except ValueError:
                _LOG.warning(
                    "ignoring inline hint with malformed priority "
                    "value %r in header %r",
                    value, header_text,
                )
                continue
            if not 0.0 <= pf <= 1.0:
                _LOG.warning(
                    "ignoring inline hint with priority %r out of "
                    "[0.0, 1.0] in header %r",
                    pf, header_text,
                )
                continue
            out["priority"] = pf
        elif key == "tags":
            out["tags"] = [t for t in value.split() if t]
        else:
            _LOG.debug(
                "unrecognised inline hint key %r in header %r — ignoring",
                key, header_text,
            )
    return stripped, out


# ----- name normalisation ----------------------------------------------------


_NON_ALNUM = re.compile(r"[^a-z0-9_]")


def normalise_header_name(text: str) -> str:
    """Header text → block name.

    Lowercase, spaces to underscores, then drop every non-alnum (and
    non-underscore) character. Empty input returns ``"section"`` so
    a stray bracketed-only header still produces a valid identifier.
    """
    lowered = text.strip().lower().replace(" ", "_")
    cleaned = _NON_ALNUM.sub("", lowered)
    return cleaned or "section"


# ----- file content type detection ------------------------------------------


_HEADER_LINE = re.compile(r"(?m)^(##+)\s+(.+)$")
"""Match an H2/H3+ header anchored at line start."""


def detect_path(text: str) -> str:
    """Pick which import path applies to a file's text content.

    Returns one of ``"frontmatter"`` (Path A — leading ``---``),
    ``"headers"`` (Path B — at least one ``##`` or ``###`` header
    after the leading whitespace), or ``"single"`` (Path C — neither).
    """
    if text.lstrip().startswith("---"):
        return "frontmatter"
    if _HEADER_LINE.search(text):
        return "headers"
    return "single"


# ----- Path A: frontmatter ---------------------------------------------------


def parse_frontmatter_blocks(
    text: str, path: Path, namespace: str,
) -> list[InstructionBlock]:
    """Parse frontmatter sections via the parser helpers and prefix names.

    Reuses ``rampart.parser._parse_meta`` and ``_block_from_meta`` so
    the parsing/validation rules stay in one place. The namespace is
    prepended to each block's ``semantic_name`` (``"<ns>.<name>"``).
    """
    parts = _BLOCK_SPLIT.split(text)
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
        block = _block_from_meta(meta, content_text, path)
        block.semantic_name = f"{namespace}.{block.semantic_name}"
        blocks.append(block)
        i += 2
    return blocks


# ----- Path B: header split --------------------------------------------------


def parse_header_blocks(
    text: str,
    path: Path,
    namespace: str,
    default_priority: float,
) -> list[InstructionBlock]:
    """Split on ``##``/``###`` headers and turn each section into a block.

    Inline hints in the header override ``default_priority`` and add
    tags. The block name is the namespaced normalised header
    text. If two headers in the same file produce the same normalised
    name, a numeric suffix is appended (``_2``, ``_3``, ...) so the
    in-file index stays unambiguous; collision against the *target
    SeedRegistry* is handled at insert time, not here.
    """
    matches = list(_HEADER_LINE.finditer(text))
    blocks: list[InstructionBlock] = []
    seen_names: dict[str, int] = {}
    for idx, match in enumerate(matches):
        header_text = match.group(2)
        section_start = match.end()
        section_end = (
            matches[idx + 1].start()
            if idx + 1 < len(matches)
            else len(text)
        )
        body = text[section_start:section_end].strip()
        stripped_header, hint = parse_inline_hint(header_text)
        base_name = normalise_header_name(stripped_header)
        count = seen_names.get(base_name, 0) + 1
        seen_names[base_name] = count
        suffix = "" if count == 1 else f"_{count}"
        unique_local = f"{base_name}{suffix}"
        priority = float(hint.get("priority", default_priority))
        tags = list(hint.get("tags", []))
        blocks.append(
            InstructionBlock(
                semantic_name=f"{namespace}.{unique_local}",
                runtime_id="",
                content=body,
                source="seed",
                priority=priority,
                tags=tags,
            )
        )
    return blocks


# ----- Path C: single block (with optional llm_splitter) --------------------


def parse_single_block(
    text: str,
    path: Path,
    namespace: str,
    default_priority: float,
    llm_splitter: SkillSplitter | None,
) -> list[InstructionBlock]:
    """Treat the file as one block, optionally split via ``llm_splitter``.

    When ``llm_splitter`` is provided this function calls it on the
    file text and converts the returned chunks into blocks. The
    callable owns any LLM call; the library never makes one itself.
    Unparseable returns trigger a warning and fall back to the
    single-block path.
    """
    if llm_splitter is None:
        return [_default_single_block(
            text, path, namespace, default_priority,
        )]

    try:
        chunks = llm_splitter(text)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning(
            "llm_splitter raised %s for %s; falling back to single "
            "block. Error: %s",
            type(exc).__name__, path, exc,
        )
        return [_default_single_block(
            text, path, namespace, default_priority,
        )]

    if not chunks:
        _LOG.warning(
            "llm_splitter returned no chunks for %s; importing as "
            "single block. If you have not implemented llm_splitter, "
            "pass llm_splitter=None.",
            path,
        )
        return [_default_single_block(
            text, path, namespace, default_priority,
        )]

    blocks: list[InstructionBlock] = []
    seen_names: dict[str, int] = {}
    for idx, raw in enumerate(chunks):
        block = _splitter_chunk_to_block(
            raw=raw,
            idx=idx,
            path=path,
            namespace=namespace,
            default_priority=default_priority,
            seen_names=seen_names,
        )
        if block is not None:
            blocks.append(block)
    if not blocks:
        _LOG.warning(
            "llm_splitter produced no usable chunks for %s after "
            "validation; importing as single block.",
            path,
        )
        return [_default_single_block(
            text, path, namespace, default_priority,
        )]
    return blocks


def _default_single_block(
    text: str,
    path: Path,
    namespace: str,
    default_priority: float,
) -> InstructionBlock:
    """Construct the single-block fallback for Path C."""
    if namespace == path.stem:
        name = namespace
    else:
        name = f"{namespace}.{path.stem}"
    return InstructionBlock(
        semantic_name=name,
        runtime_id="",
        content=text.strip(),
        source="seed",
        priority=default_priority,
        tags=[],
    )


def _splitter_chunk_to_block(
    *,
    raw: dict[str, Any] | str,
    idx: int,
    path: Path,
    namespace: str,
    default_priority: float,
    seen_names: dict[str, int],
) -> InstructionBlock | None:
    """Validate one ``llm_splitter`` chunk and turn it into a block.

    Returns ``None`` when the chunk is malformed (logged as WARNING).
    """
    if isinstance(raw, str):
        content = raw
        name_hint = None
        priority = default_priority
        tags: list[str] = []
        evictable = True
    elif isinstance(raw, dict):
        if "content" not in raw:
            _LOG.warning(
                "llm_splitter chunk %d for %s missing 'content' "
                "field; skipping. Chunk: %r",
                idx, path, raw,
            )
            return None
        content = str(raw["content"])
        name_hint = raw.get("name")
        priority_raw = raw.get("priority", default_priority)
        try:
            priority = float(priority_raw)
        except (TypeError, ValueError):
            _LOG.warning(
                "llm_splitter chunk %d for %s has non-numeric "
                "priority %r; falling back to default.",
                idx, path, priority_raw,
            )
            priority = default_priority
        tags_raw = raw.get("tags", [])
        if isinstance(tags_raw, list) and all(
            isinstance(t, str) for t in tags_raw
        ):
            tags = list(tags_raw)
        else:
            _LOG.warning(
                "llm_splitter chunk %d for %s has non-string-list "
                "tags %r; ignoring.",
                idx, path, tags_raw,
            )
            tags = []
        evictable = bool(raw.get("evictable", True))
    else:
        _LOG.warning(
            "llm_splitter chunk %d for %s is neither a string nor "
            "a dict (got %s); skipping.",
            idx, path, type(raw).__name__,
        )
        return None

    base_local = (
        normalise_header_name(name_hint)
        if isinstance(name_hint, str) and name_hint
        else f"{path.stem}_chunk_{idx}"
    )
    count = seen_names.get(base_local, 0) + 1
    seen_names[base_local] = count
    suffix = "" if count == 1 else f"_{count}"
    semantic_name = f"{namespace}.{base_local}{suffix}"
    return InstructionBlock(
        semantic_name=semantic_name,
        runtime_id="",
        content=content.strip(),
        source="seed",
        priority=priority,
        tags=tags,
        evictable=evictable,
    )


# ----- top-level entry point -------------------------------------------------


def import_skill_file(
    path: Path,
    *,
    root: Path | str | None = None,
    namespace: str | None = None,
    default_priority: float | None = None,
    llm_splitter: SkillSplitter | None = None,
) -> list[InstructionBlock]:
    """Read a skill file and return one or more parsed blocks.

    See module docstring for the namespace, priority, and path-A/B/C
    rules. The result is a list of blocks with ``runtime_id=""``;
    consumers (typically ``SeedRegistry.from_skill_file``) handle
    insertion + collision detection.
    """
    text = path.read_text(encoding="utf-8-sig")
    ns = resolve_namespace(path, root, namespace)
    dp = resolve_default_priority(path, default_priority)
    kind = detect_path(text)
    if kind == "frontmatter":
        return parse_frontmatter_blocks(text, path, ns)
    if kind == "headers":
        return parse_header_blocks(text, path, ns, dp)
    return parse_single_block(text, path, ns, dp, llm_splitter)


__all__ = [
    "SkillSplitter",
    "detect_path",
    "import_skill_file",
    "library_default_priority",
    "normalise_header_name",
    "parse_frontmatter_blocks",
    "parse_header_blocks",
    "parse_inline_hint",
    "parse_single_block",
    "resolve_default_priority",
    "resolve_namespace",
]
