"""SeedRegistry — read-only library of parsed seed blocks.

A ``SeedRegistry`` is the **library** end of the seed pipeline: parsed
once from disk at process start, then kept in RAM as a name-keyed map
of immutable ``InstructionBlock`` templates. It does not assign runtime
IDs, does not maintain ordering, does not protect against eviction,
and does not embed.

A ``BlockRegistry`` is the **working set** end: an ordered, mutable,
session-local collection of blocks with UUID4 runtime IDs and the full
priority / eviction / rollback machinery. Blocks flow from a
``SeedRegistry`` into a ``BlockRegistry`` via
``BlockRegistry.from_seed`` (cold start) or
``BlockRegistry.add_from_seed`` (hot path).

This split lets a long-running process pay the disk-and-parse cost
exactly once, then build many short-lived working registries for
different tasks or callers without re-reading files. The
``SeedRegistry`` is the right place for any cross-session caching of
parser output; the ``BlockRegistry`` stays focused on the
session-local concerns of ordering, eviction, and trajectory tracking.

**Key format.** Every block is keyed by ``f"{path.stem}.{semantic_name}"``,
mirroring the Python import mental model: ``import skills.coder.python``
is unambiguous regardless of what other files contain. Two files with
different stems (``skills.md`` and ``tools.md``) carrying blocks named
``coder.python`` produce distinct keys ``skills.coder.python`` and
``tools.coder.python`` — both retained, no collision.

Two **different** files that happen to share the same stem (typically
``~/prompts/skills.md`` plus ``~/work/skills.md``) would generate
silently-overlapping namespaced keys. ``from_files`` rejects this at
load time with a ``ValueError`` listing both paths so the operator
can rename one file rather than have blocks evaporate into a
last-wins overwrite.

**Lookup.** ``get`` / ``add_from_seed`` / ``from_seed`` accept either
the full namespaced key (``skills.coder.python``) or a bare name
(``coder.python``). A bare name resolves when it matches exactly one
key across the entire library. If it matches more than one, the lookup
raises ``ValueError`` and tells the caller to use the namespaced form.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from rampart.parser import BlockSource, _coerce_block_source, parse_file
from rampart.registry import BlockNotFoundError
from rampart.skill import SkillSplitter, import_skill_file

if TYPE_CHECKING:
    from rampart.block import InstructionBlock


_LOG = logging.getLogger("rampart.seed_registry")


class SeedRegistry:
    """Read-only in-RAM library of parsed seed blocks.

    Build via :meth:`from_files`. Query via :meth:`get` (single block by
    name) or :meth:`list_names` (full inventory, optionally tag-filtered).
    The blocks returned by both query methods carry the parser sentinel
    ``runtime_id=""``; assigning a real session UUID is the responsibility
    of the consumer (typically ``BlockRegistry.from_seed`` or
    ``BlockRegistry.add_from_seed``).
    """

    def __init__(self) -> None:
        """Construct an empty library. Use ``SeedRegistry.from_files``
        in normal code; the bare constructor is exposed for tests that
        want to inject ``InstructionBlock`` instances directly via
        ``library._blocks[key] = ...``.
        """
        self._blocks: dict[str, InstructionBlock] = {}
        # Per-key source-file map. Populated by ``add_skill_file`` so
        # collision warnings can name both files involved. Direct
        # assignments to ``_blocks`` (test-only) bypass this map; the
        # collision warning falls back to "<unknown source>" in that
        # case rather than refusing to operate.
        self._block_sources: dict[str, str] = {}

    def __len__(self) -> int:
        """Number of unique namespaced keys currently in the library."""
        return len(self._blocks)

    def __contains__(self, name: object) -> bool:
        """Membership check accepting both bare and namespaced forms.

        A bare name returns ``True`` only if it resolves
        unambiguously. Ambiguous bare names (matching more than one
        block) return ``False`` here so ``name in library`` does not
        raise; use :meth:`get` to surface the explicit ``ValueError``.
        """
        if not isinstance(name, str):
            return False
        try:
            self._resolve(name)
        except (BlockNotFoundError, ValueError):
            return False
        return True

    @classmethod
    def from_files(
        cls,
        sources: Iterable[BlockSource | str | Path],
    ) -> SeedRegistry:
        """Parse every source into a name-keyed library.

        Bare ``str`` and ``Path`` arguments are coerced to
        ``BlockSource(path=p, names=None, must_match=True)``. The
        ``names`` and ``must_match`` fields of an explicit
        ``BlockSource`` are honoured exactly as in
        ``BlockRegistry.from_files``: missing names raise
        ``BlockNotFoundError`` when ``must_match=True``.

        Each block is stored under the key
        ``f"{path.stem}.{semantic_name}"``. Different files with
        different stems produce distinct keys even when they share a
        ``semantic_name``.

        Two **different** source paths with the same stem are rejected
        up front: any blocks they contributed would generate
        silently-overlapping namespaced keys. The same path supplied
        twice is allowed (idempotent reload).

        Args:
            sources: Iterable of seed file paths or
                ``BlockSource`` descriptors.

        Returns:
            A populated :class:`SeedRegistry`. No file handles remain
            open after this call returns.

        Raises:
            BlockNotFoundError: If a ``BlockSource`` with
                ``must_match=True`` requests a name not present in
                the corresponding file.
            ValueError: If two distinct source paths share the same
                ``Path.stem``. The error message lists both paths and
                the colliding stem so the operator can rename one
                file rather than debug missing blocks at runtime.
        """
        library = cls()
        seen_stems: dict[str, Path] = {}
        for raw in sources:
            source = _coerce_block_source(raw)
            path = Path(source.path)
            stem = path.stem
            prior = seen_stems.get(stem)
            if prior is not None and prior != path:
                raise ValueError(
                    f"Two source files share the stem {stem!r}: "
                    f"{prior!s} and {path!s}. Their namespaced keys "
                    f"would collide silently. Rename one of the files "
                    f"so each source has a distinct stem."
                )
            seen_stems[stem] = path
            blocks = parse_file(path)
            blocks, missing = source.filter_blocks(blocks)
            if missing and source.must_match:
                raise BlockNotFoundError(
                    f"BlockSource(path={source.path!r}) requested "
                    f"semantic_names {missing!r} but the file does not "
                    f"contain them."
                )
            for block in blocks:
                key = f"{stem}.{block.semantic_name}"
                library._blocks[key] = block
        return library

    @classmethod
    def from_skill_file(
        cls,
        path: str | Path,
        *,
        root: str | Path | None = None,
        namespace: str | None = None,
        default_priority: float | None = None,
        llm_splitter: SkillSplitter | None = None,
    ) -> SeedRegistry:
        """Cold-start a library from a single skill file.

        See :mod:`rampart.skill` for the full namespace / priority /
        path-A/B/C resolution rules. Each block is keyed under its
        resolved ``semantic_name`` directly (skipping the
        ``stem.semantic_name`` wrapper used by :meth:`from_files`),
        because the importer already prepends the namespace prefix —
        wrapping again would produce double-prefixed keys like
        ``api.skills.api.skills.section``.

        Convenience entry point for the common case of a single
        skill file. To merge multiple skill files into one library,
        call this classmethod once and then :meth:`add_skill_file`
        for each subsequent file.

        Args:
            path: Path to the skill file. ``.md`` is the only
                extension currently supported by the importer; the
                file's content shape (frontmatter / headers / single)
                is auto-detected.
            root: Optional anchor for namespace resolution. When set,
                the file's path-relative-to-root is converted to
                dot-notation and used as the namespace.
            namespace: Optional explicit namespace override.
            default_priority: Optional override for the priority
                assigned to bare sections. ``None`` (default) lets
                the library pick: ``0.9`` for SKILL.md / CLAUDE.md
                stems, ``0.5`` otherwise.
            llm_splitter: Optional callable that splits a single-block
                file into multiple chunks. The library never invokes
                an LLM itself — see :mod:`rampart.skill` for the
                callable signature and validation rules.

        Returns:
            A populated :class:`SeedRegistry`.
        """
        library = cls()
        library.add_skill_file(
            path,
            root=root,
            namespace=namespace,
            default_priority=default_priority,
            llm_splitter=llm_splitter,
        )
        return library

    def add_skill_file(
        self,
        path: str | Path,
        *,
        root: str | Path | None = None,
        namespace: str | None = None,
        default_priority: float | None = None,
        llm_splitter: SkillSplitter | None = None,
    ) -> list[str]:
        """Merge another skill file into this library.

        Resolves blocks via :mod:`rampart.skill` and inserts each
        under its ``semantic_name`` key. **Collisions are skipped
        with a logged WARNING** — if the resolved key already
        exists, the new block is dropped and the existing one is
        preserved. The warning names both the existing block's
        source path (cached at insert time) and the incoming file
        path so an operator can rename or namespace one of them.

        Args:
            path: Path to the skill file.
            root: See :meth:`from_skill_file`.
            namespace: See :meth:`from_skill_file`.
            default_priority: See :meth:`from_skill_file`.
            llm_splitter: See :meth:`from_skill_file`.

        Returns:
            Ordered list of keys that were *newly* inserted by this
            call. Skipped collisions are not included.
        """
        path_obj = Path(path)
        blocks = import_skill_file(
            path_obj,
            root=root,
            namespace=namespace,
            default_priority=default_priority,
            llm_splitter=llm_splitter,
        )
        added: list[str] = []
        for block in blocks:
            key = block.semantic_name
            if key in self._blocks:
                prior = self._block_sources.get(key, "<unknown source>")
                _LOG.warning(
                    "skill-file collision: block %r already exists in "
                    "the library (from %s); skipping incoming block "
                    "from %s",
                    key, prior, path_obj,
                )
                continue
            self._blocks[key] = block
            self._block_sources[key] = str(path_obj)
            added.append(key)
        return added

    def get(self, name: str) -> InstructionBlock:
        """Retrieve a seed block by namespaced or bare name.

        Args:
            name: Either the full ``stem.semantic_name`` key or the
                bare ``semantic_name``. The bare form succeeds only
                when exactly one key in the library has that name as
                its trailing component.

        Returns:
            The parsed ``InstructionBlock`` template. The returned
            object is the library's internal reference — callers that
            mutate it (set ``runtime_id``, ``created_at``, etc.)
            should make a copy first or use the
            ``BlockRegistry.from_seed`` / ``add_from_seed`` paths
            which copy on insert.

        Raises:
            BlockNotFoundError: If ``name`` is not present in the
                library under any form.
            ValueError: If ``name`` is a bare name that matches more
                than one key. The error message lists the candidates
                so the caller can pick the right namespaced form.
        """
        return self._blocks[self._resolve(name)]

    def list_names(
        self,
        tags: list[str] | None = None,
    ) -> list[str]:
        """List every namespaced key in the library.

        Args:
            tags: Optional filter. ``None`` (default) returns every
                key. A non-empty list returns only the keys of blocks
                whose ``tags`` contain **at least one** of the
                requested tags (set-intersection semantics — the
                more permissive of the two reasonable choices,
                because libraries tend to be inspected with
                disjunctive queries: "show me anything tagged
                hardware OR critical"). An empty list (``[]``) is
                treated as "match nothing", not as "no filter".

        Returns:
            Namespaced keys in insertion order (later overwrites of an
            existing key preserve the original insertion slot).
        """
        if tags is None:
            return list(self._blocks)
        wanted = set(tags)
        return [
            key
            for key, block in self._blocks.items()
            if wanted.intersection(block.tags)
        ]

    def _resolve(self, name: str) -> str:
        """Resolve a bare or namespaced name to its full key.

        Lookup order:

        1. Direct hit on the namespaced key (``stem.semantic_name``)
           returns immediately.
        2. Otherwise, scan every key for ones ending in ``.{name}``:

           * Exactly one match → return that key.
           * Two or more matches → raise ``ValueError`` listing the
             candidates and recommending the namespaced form.
           * Zero matches → raise ``BlockNotFoundError``.

        Args:
            name: Either a full key or a bare semantic name.

        Returns:
            The full namespaced key that uniquely identifies the
            requested block.

        Raises:
            BlockNotFoundError: If no key matches.
            ValueError: If a bare name matches more than one key.
        """
        if name in self._blocks:
            return name
        suffix = f".{name}"
        candidates = [k for k in self._blocks if k.endswith(suffix)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"Bare name {name!r} matches multiple blocks: "
                f"{sorted(candidates)!r}. Use the namespaced form "
                f"(e.g. {sorted(candidates)[0]!r}) to disambiguate."
            )
        raise BlockNotFoundError(
            f"SeedRegistry has no block named {name!r}."
        )


__all__ = [
    "SeedRegistry",
]
