"""Ordered in-RAM registry of InstructionBlocks with the full mutation API.

The ordering is explicit, mutable, and the sole determinant of compile-time
position in the assembled context string. ``compile()`` never modifies
ordering; only the public mutation operations in this module do.

Three invariants are enforced here rather than by convention:

* Blocks with ``evictable=False`` cannot be removed by any caller via the
  default ``evict()`` path. Only ``evict(force=True)`` from the block's
  authoring registry can remove them; any other registry calling
  ``evict(force=True)`` against a non-evictable block authored elsewhere
  raises ``PermissionError``. This is the structural defence against
  context collapse: human-authored seed knowledge and any other
  caller-pinned coordination content stays present for the lifetime
  of the session unless the author explicitly chooses to drop it.
* Every block carries an ``author_id`` recording the registry instance
  that wrote it. Seed blocks loaded via ``from_files`` / ``from_seed``
  get the loading registry's ``registry_id``; blocks written through
  ``write_block`` / ``write_agent_block`` get the writing registry's.
  The author check above is what makes the ``force=True`` channel safe
  across multiple registries sharing a process.
* ``promote()`` requires exactly one of its directive keyword arguments
  (``steps``, ``to_position``, ``before``, ``after``, ``to_front``,
  ``to_back``). Zero or more than one raises ``ValueError`` so an ambiguous
  call ("did you mean steps or to_position?") fails immediately rather than
  silently picking a default.

The registry has **no token budget**. The only token ceiling in the
system is ``compile(max_tokens)``, which defaults to
``RAMPARTConfig.default_max_tokens`` if not passed. The registry's
``memory_limit_mb`` is a soft warning threshold for runaway growth;
exceeding it logs a ``RuntimeWarning`` but does not raise or evict.

A process-wide label registry tracks unique human-readable names per
``BlockRegistry`` instance. Pass ``label="my-registry"`` to the constructor
to claim that name; omit it to receive an auto-generated
``"registry-N"`` label. ``rampart.registry.get_by_label(label)``
recovers the instance from anywhere in the process. Call
``registry.release()`` to free the label for reuse (typically in test
teardown or session shutdown).
"""

from __future__ import annotations

import sys
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from rampart.block import BlockSourceKind, InstructionBlock
from rampart.config import RAMPARTConfig, default_tokeniser
from rampart.eviction import DefaultEvictionPolicy, EvictionPolicy
from rampart.parser import BlockSource, _coerce_block_source, parse_file
from rampart.security import generate_runtime_id

if TYPE_CHECKING:
    from rampart.compiler import CompileResult, DryRunResult
    from rampart.seed_registry import SeedRegistry


# Process-wide registry of label -> BlockRegistry instance. Enforces label
# uniqueness across the process and supports get_by_label() lookup. Not
# thread-safe; consistent with the rest of the library's single-threaded
# contract.
_GLOBAL_REGISTRY_LABELS: dict[str, BlockRegistry] = {}
_LABEL_COUNTER: int = 0


def get_by_label(label: str) -> BlockRegistry:
    """Return the ``BlockRegistry`` claiming ``label``.

    Args:
        label: The human-readable label assigned at construction time
            (or auto-generated as ``"registry-N"``).

    Returns:
        The ``BlockRegistry`` instance currently registered under
        ``label``.

    Raises:
        KeyError: If no registry holds this label. Either the label
            was never claimed or the registry has been ``release()``d.
    """
    if label not in _GLOBAL_REGISTRY_LABELS:
        raise KeyError(
            f"No BlockRegistry registered under label {label!r}. "
            f"Either the label was never claimed or the owning "
            f"registry has been released."
        )
    return _GLOBAL_REGISTRY_LABELS[label]


@runtime_checkable
class EmbeddingModel(Protocol):
    """Duck-type for an embedding model usable by ``BlockRegistry``.

    Anything exposing a single-text ``encode`` that returns a numpy 1-D vector
    satisfies this. ``sentence_transformers.SentenceTransformer`` matches
    without modification. Defined here as a Protocol so the registry does not
    import sentence-transformers itself — the heavy embedding-model wiring
    lives in ``rampart.scorer`` and only that module needs the
    sentence-transformers dependency at import time.
    """

    def encode(self, text: str) -> NDArray[Any]: ...


class BlockNotFoundError(KeyError):
    """Raised when a runtime_id is not present in the registry."""


class RegistryFullError(Exception):
    """Raised when an operation would push the registry past its token budget.

    Reserved for the eviction-policy path. Defined here so the public
    exception surface is stable from the start.
    """


class EvictionError(Exception):
    """Raised when ``evict()`` targets a non-evictable block without force.

    The block carries ``evictable=False`` (typically a seed block or
    a caller-pinned coordination block) and the caller did not pass
    ``force=True``. The error message names the block and its
    authoring registry so the operator can choose between releasing
    the protection or contacting the author. Permission is data-driven
    via the ``evictable`` flag, not tied to a specific source enum
    value.
    """


class BlockRegistry:
    """Ordered in-RAM mapping from runtime UUID to InstructionBlock.

    Single-threaded by default. A future thread-safe subclass adds an
    RLock for cross-thread coordination; that work lives outside the
    library API surface. Every method here is O(1) or O(N) in the
    number of blocks.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        tokeniser: Callable[[str], int] | None = None,
        eviction_policy: EvictionPolicy | None = None,
        *,
        label: str | None = None,
        default_max_tokens: int = 16384,
        memory_limit_mb: float = 100.0,
    ) -> None:
        """Construct an empty registry.

        Args:
            embedding_model: Object exposing ``encode(text) -> ndarray``.
                Required to call ``find_relevant``; ``score_block`` works
                without one if the caller provides a precomputed task vector.
            tokeniser: ``Callable[[str], int]`` that returns the token count
                of a string. ``None`` falls back to
                ``rampart.config.default_tokeniser`` (lazy tiktoken
                cl100k_base). Token counts are cached per block in
                ``block.token_count`` after the first computation.
            eviction_policy: Policy object whose ``select(registry)``
                returns IDs to evict, highest-score first. Default is
                ``DefaultEvictionPolicy()``.
            label: Human-readable identifier, unique across the
                process. ``None`` (default) auto-generates
                ``"registry-N"`` from a module-level counter. A
                user-supplied label that collides with a still-claimed
                label raises ``ValueError`` with the existing
                registry's UUID in the message; call ``release()`` on
                the prior owner first or pick a different name.
            default_max_tokens: Token ceiling used by ``compile()``
                when the caller does not pass ``max_tokens``
                explicitly. Set to your model's context window minus
                the expected task and response allowance.
            memory_limit_mb: Soft warning threshold for total registry
                size in megabytes. When a write pushes the registry
                across this threshold a ``RuntimeWarning`` is emitted;
                the registry never auto-evicts and never raises on
                growth. Set to ``float("inf")`` to disable the warning.

        Raises:
            ValueError: If ``label`` is already registered to another
                instance.
        """
        global _LABEL_COUNTER
        if label is None:
            _LABEL_COUNTER += 1
            candidate = f"registry-{_LABEL_COUNTER}"
            while candidate in _GLOBAL_REGISTRY_LABELS:
                _LABEL_COUNTER += 1
                candidate = f"registry-{_LABEL_COUNTER}"
            label = candidate
        else:
            if label in _GLOBAL_REGISTRY_LABELS:
                existing = _GLOBAL_REGISTRY_LABELS[label]
                raise ValueError(
                    f"Label {label!r} is already claimed by registry "
                    f"with registry_id {existing.registry_id!r}. Call "
                    f"release() on the existing instance or pick a "
                    f"different label."
                )

        self._blocks: OrderedDict[str, InstructionBlock] = OrderedDict()
        self._name_index: dict[str, list[str]] = {}
        self._embedding_model: EmbeddingModel | None = embedding_model
        self._tokeniser: Callable[[str], int] = tokeniser or default_tokeniser
        self._eviction_policy: EvictionPolicy = (
            eviction_policy if eviction_policy is not None
            else DefaultEvictionPolicy()
        )
        self.label: str = label
        self.registry_id: str = generate_runtime_id()
        self.default_max_tokens: int = default_max_tokens
        self.memory_limit_mb: float = memory_limit_mb
        self._memory_bytes: int = 0
        _GLOBAL_REGISTRY_LABELS[label] = self

    def release(self) -> None:
        """Remove this registry's label from the global label registry.

        After release, the label can be reclaimed by another
        ``BlockRegistry`` constructor call. Idempotent — calling
        ``release()`` twice is a no-op. The block content remains in
        memory for as long as the instance is alive; ``release()``
        only affects the cross-process label-uniqueness contract, not
        the registry's own state.
        """
        _GLOBAL_REGISTRY_LABELS.pop(self.label, None)

    @classmethod
    def from_files(
        cls,
        paths: Iterable[BlockSource | str | Path],
        embedding_model: EmbeddingModel | None = None,
        tokeniser: Callable[[str], int] | None = None,
        eviction_policy: EvictionPolicy | None = None,
        *,
        label: str | None = None,
        config: RAMPARTConfig | None = None,
    ) -> BlockRegistry:
        """Cold-start a registry from one or more seed files.

        Per-file flow: dispatch the path through ``rampart.parser.parse_file``,
        receive seed blocks with ``runtime_id=""`` (the parser sentinel), then
        for each block assign a fresh UUID4 and the current Unix timestamp
        before insertion. Address randomisation lives here, not in the parser:
        the parser produces session-agnostic data and the registry imposes
        the per-session identity. Two cold starts of the same file therefore
        yield disjoint runtime-id sets, satisfying invariant 4.

        File order is preserved in the resulting ordering. Within a single
        file, blocks appear in the order their frontmatter sections appear.

        Args:
            paths: Iterable of either bare ``Path`` / ``str`` or
                :class:`rampart.parser.BlockSource` descriptors. Bare
                paths load every block in the file; ``BlockSource``
                descriptors carry an optional ``names`` whitelist and
                ``must_match`` flag for selective loading. Each path is
                dispatched through ``rampart.parser.PARSERS`` by
                extension; unknown extensions raise
                ``UnsupportedFormatError``.
            embedding_model: See :meth:`__init__`.
            tokeniser: Optional override for the registry's tokeniser.
                ``None`` means "use ``config.tokeniser``" (which itself
                defaults to a lazy tiktoken cl100k_base counter).
            eviction_policy: See :meth:`__init__`.
            label: See :meth:`__init__`. Optional human-readable name
                claimed by the new registry.
            config: Source of defaults for ``tokeniser``,
                ``default_max_tokens``, and ``memory_limit_mb``. ``None``
                instantiates a fresh ``RAMPARTConfig()`` with library
                defaults. Per-call overrides above always win over the
                config's value.

        Returns:
            A populated :class:`BlockRegistry`. No file handle is held open
            after this call returns.

        Raises:
            BlockNotFoundError: If any ``BlockSource`` with
                ``must_match=True`` requests a name that is not present
                in the corresponding file. The error message lists the
                file path and the missing names so the operator can
                fix the call site or the seed file without further
                debugging.
        """
        cfg = config if config is not None else RAMPARTConfig()
        registry = cls(
            embedding_model=embedding_model,
            tokeniser=tokeniser if tokeniser is not None else cfg.tokeniser,
            eviction_policy=(
                eviction_policy
                if eviction_policy is not None
                else cfg.eviction_policy
            ),
            label=label,
            default_max_tokens=cfg.default_max_tokens,
            memory_limit_mb=cfg.memory_limit_mb,
        )
        now = int(time.time())
        for raw in paths:
            source = _coerce_block_source(raw)
            blocks = parse_file(Path(source.path))
            blocks, missing = source.filter_blocks(blocks)
            if missing and source.must_match:
                raise BlockNotFoundError(
                    f"BlockSource(path={source.path!r}) requested "
                    f"semantic_names {missing!r} but the file does not "
                    f"contain them."
                )
            for block in blocks:
                block.runtime_id = generate_runtime_id()
                block.created_at = now
                # Seed blocks loaded from disk are non-evictable by
                # default; the loading registry is the author of record.
                block.author_id = registry.registry_id
                block.evictable = False
                registry._blocks[block.runtime_id] = block
                registry._index_add(block.semantic_name, block.runtime_id)
                registry._memory_bytes += _estimate_block_bytes(block)
        registry._maybe_warn_memory()
        # Cold-start batch embedding: only after all blocks are inserted
        # and runtime_ids are assigned. Lazy import — rampart.scorer imports
        # from this module at type-check time only.
        if registry._embedding_model is not None:
            from rampart.scorer import embed_all

            embed_all(registry)
            # Re-account for embedding bytes now that they exist.
            registry._memory_bytes = sum(
                _estimate_block_bytes(b) for b in registry._blocks.values()
            )
            registry._maybe_warn_memory()
        return registry

    @classmethod
    def from_seed(
        cls,
        seed_library: SeedRegistry,
        names: Iterable[str],
        *,
        must_match: bool = True,
        embedding_model: EmbeddingModel | None = None,
        tokeniser: Callable[[str], int] | None = None,
        eviction_policy: EvictionPolicy | None = None,
        label: str | None = None,
        config: RAMPARTConfig | None = None,
    ) -> BlockRegistry:
        """Build a working registry from a named subset of a seed library.

        The seed library has already paid the disk-and-parse cost; this
        method just copies the requested blocks into a fresh registry,
        assigns session-local UUID4 runtime IDs, and stamps
        ``created_at`` on every block. Two calls with the same ``names``
        list against the same library yield disjoint runtime-id sets
        (invariant 4) so distinct working registries stay isolated.

        ``names`` order is preserved in the returned registry, which
        means the caller controls the initial ordering by listing
        blocks in the order they should appear at compile time. Mirrors
        the ``BlockRegistry.from_files`` per-file ordering contract.

        Args:
            seed_library: The :class:`SeedRegistry` to draw from.
            names: Ordered iterable of ``semantic_name`` values to
                pull from the library. Order becomes registry order.
            must_match: When ``True`` (default), missing names raise
                ``BlockNotFoundError`` with the full missing list.
                When ``False``, missing names are silently skipped —
                the returned registry contains only the blocks that
                were actually present.
            embedding_model: See :meth:`__init__`. If provided, every
                block in the resulting registry is batch-embedded
                after insertion (same code path as ``from_files``).
            tokeniser: See :meth:`__init__` / :meth:`from_files`.
            eviction_policy: See :meth:`__init__`.
            label: See :meth:`__init__`.
            config: Source of defaults for ``tokeniser``,
                ``default_max_tokens``, and ``memory_limit_mb``.
                ``None`` instantiates a fresh ``RAMPARTConfig()``.

        Returns:
            A populated :class:`BlockRegistry`.

        Raises:
            BlockNotFoundError: If ``must_match`` is ``True`` and any
                requested name is not in ``seed_library``. Error message
                lists the full missing subset so the operator can fix
                the call site or the library in one pass.
            ValueError: If any requested name is a bare ``semantic_name``
                that matches more than one block in the library.
                Propagates from ``seed_library.get`` and is raised
                regardless of ``must_match`` — ambiguity is a
                programmer error, not a missing-data condition. The
                error message lists the candidate namespaced keys.
        """
        cfg = config if config is not None else RAMPARTConfig()
        registry = cls(
            embedding_model=embedding_model,
            tokeniser=tokeniser if tokeniser is not None else cfg.tokeniser,
            eviction_policy=(
                eviction_policy
                if eviction_policy is not None
                else cfg.eviction_policy
            ),
            label=label,
            default_max_tokens=cfg.default_max_tokens,
            memory_limit_mb=cfg.memory_limit_mb,
        )
        # Resolve every requested name up front so the registry is not
        # half-built if a later name turns out to be ambiguous or
        # missing. ValueError on ambiguity propagates immediately;
        # BlockNotFoundError on missing names is collected and
        # surfaced after the whole list is checked.
        wanted = list(names)
        resolved: list[InstructionBlock | None] = []
        missing: list[str] = []
        for name in wanted:
            try:
                resolved.append(seed_library.get(name))
            except BlockNotFoundError:
                missing.append(name)
                resolved.append(None)
        if missing and must_match:
            raise BlockNotFoundError(
                f"SeedRegistry is missing requested blocks: {missing!r}."
            )
        now = int(time.time())
        for template in resolved:
            if template is None:
                continue  # must_match=False: silently skip
            registry._insert_seed_copy(template, now=now)
        if registry._embedding_model is not None:
            from rampart.scorer import embed_all

            embed_all(registry)
            registry._memory_bytes = sum(
                _estimate_block_bytes(b) for b in registry._blocks.values()
            )
        registry._maybe_warn_memory()
        return registry

    def add_from_seed(
        self,
        seed_library: SeedRegistry,
        name: str,
    ) -> str:
        """Add one block from a seed library into this live registry.

        The block keeps its ``source="seed"`` provenance — it came from
        a seed file even if it is being inserted mid-session — and is
        stamped ``evictable=False`` with ``author_id`` set to this
        registry's ``registry_id``. ``evict()`` therefore raises
        ``EvictionError`` against it unless the same registry calls
        ``evict(force=True)``.

        If the registry has an ``embedding_model`` configured, the
        new block is embedded immediately on insertion. Without an
        embedding model the block joins the registry without an
        embedding (matching the cold-start behaviour for
        ``from_files`` without an embedder).

        Args:
            seed_library: The library to draw the block from.
            name: Either the full ``stem.semantic_name`` key or the
                bare ``semantic_name`` of the block to add. The bare
                form succeeds only when it resolves unambiguously
                across the library; see :meth:`SeedRegistry.get`.

        Returns:
            The newly assigned ``runtime_id`` of the inserted block.

        Raises:
            BlockNotFoundError: If ``name`` is not in ``seed_library``.
            ValueError: If ``name`` is a bare name that matches more
                than one block. Propagates from ``seed_library.get``
                with the candidate namespaced keys in the message.
        """
        template = seed_library.get(name)
        runtime_id = self._insert_seed_copy(
            template, now=int(time.time())
        )
        if self._embedding_model is not None:
            from rampart.scorer import embed_text

            block = self._blocks[runtime_id]
            block.embedding = embed_text(block.content, self._embedding_model)
            # Re-account for the embedding bytes after assignment.
            self._memory_bytes += (
                int(block.embedding.nbytes)
                if block.embedding is not None
                else 0
            )
        self._maybe_warn_memory()
        return runtime_id

    def _insert_seed_copy(
        self,
        template: InstructionBlock,
        *,
        now: int,
    ) -> str:
        """Copy a parsed seed template into ``self`` with a fresh UUID.

        Internal helper shared by ``from_seed`` and ``add_from_seed``.
        Copies every primitive field; ``embedding``, ``token_count``,
        and ``kv_cache`` are reset to their post-parse defaults so the
        new block re-derives them under this registry's tokeniser /
        embedder. ``author_id`` becomes this registry's id and
        ``evictable`` is forced to ``False`` so seed-from-library
        copies share the cold-start contract: only the loading
        registry can ``evict(force=True)`` them.
        """
        runtime_id = generate_runtime_id()
        copy = InstructionBlock(
            semantic_name=template.semantic_name,
            runtime_id=runtime_id,
            content=template.content,
            source=template.source,
            author_id=self.registry_id,
            evictable=False,
            priority=template.priority,
            trajectory_id=template.trajectory_id,
            created_at=now,
            access_count=0,
            embedding=None,
            kv_cache=None,
            tags=list(template.tags),
            token_count=None,
            wrap_prefix=template.wrap_prefix,
            wrap_suffix=template.wrap_suffix,
        )
        self._blocks[runtime_id] = copy
        self._index_add(copy.semantic_name, runtime_id)
        self._memory_bytes += _estimate_block_bytes(copy)
        return runtime_id

    # ----- minimal query primitives required to test mutations -----

    def __len__(self) -> int:
        """Number of blocks currently in the registry."""
        return len(self._blocks)

    def __contains__(self, runtime_id: object) -> bool:
        """Membership check by runtime_id."""
        return runtime_id in self._blocks

    def get_by_id(self, runtime_id: str) -> InstructionBlock:
        """Return the block at ``runtime_id`` or raise.

        Args:
            runtime_id: The block's UUID4 storage key.

        Returns:
            The InstructionBlock at this id.

        Raises:
            BlockNotFoundError: If no block with this id exists.
        """
        block = self._blocks.get(runtime_id)
        if block is None:
            raise BlockNotFoundError(f"No block with runtime_id {runtime_id!r}")
        return block

    def runtime_ids(self) -> list[str]:
        """Return the ordered list of runtime ids.

        Used by tests and by callers that need to inspect ordering without
        holding a reference to the underlying OrderedDict.
        """
        return list(self._blocks.keys())

    def token_count_of(self, block: InstructionBlock) -> int:
        """Return the cached token count of a block, computing it if absent.

        First call on a block runs the registry's tokeniser over the block
        content and stores the result in ``block.token_count``. Subsequent
        calls return the cached value. ``update_block_content`` already
        invalidates the cache by setting ``token_count`` back to ``None``,
        so the cache cannot serve stale counts.

        Args:
            block: A block managed by this registry. The method does not
                verify membership; callers that need that check should call
                ``get_by_id`` first.

        Returns:
            The block's token count.
        """
        if block.token_count is None:
            block.token_count = self._tokeniser(block.content)
        return block.token_count

    def count_tokens(self, text: str) -> int:
        """Token count of ``text`` under the registry's configured tokeniser.

        Used by the compiler to size separator and wrap-decoration strings
        against the same tokenisation choice as the blocks themselves. For
        block content prefer :meth:`token_count_of`, which caches the
        result on the block.
        """
        return self._tokeniser(text)

    def compile(
        self,
        max_tokens: int | None = None,
        task_text: str | None = None,
        relevance_threshold: float | None = None,
        top_k: int | None = None,
        separator: str | None = None,
    ) -> CompileResult:
        """Compile this registry into a :class:`CompileResult`.

        Thin wrapper over ``rampart.compiler.compile``. ``max_tokens``
        defaults to ``self.default_max_tokens`` when not supplied.
        See ``rampart.compiler.compile`` for full argument semantics
        and the returned dataclass shape.
        """
        # Lazy import: rampart.compiler imports from rampart.registry only
        # under TYPE_CHECKING; importing at module level here would still
        # trigger compiler module evaluation on every registry import.
        from rampart.compiler import compile as _compile

        return _compile(
            self,
            max_tokens,
            task_text=task_text,
            relevance_threshold=relevance_threshold,
            top_k=top_k,
            separator=separator,
        )

    def compile_dry_run(
        self,
        max_tokens: int | None = None,
        task_text: str | None = None,
        relevance_threshold: float | None = None,
        top_k: int | None = None,
        separator: str | None = None,
    ) -> DryRunResult:
        """Same selection logic as :meth:`compile` but without side effects.

        Returns a :class:`DryRunResult` carrying the would-be inclusions
        plus the excluded names. Does not increment ``access_count`` and
        does not build the output string.
        """
        from rampart.compiler import compile_dry_run as _dry

        return _dry(
            self,
            max_tokens,
            task_text=task_text,
            relevance_threshold=relevance_threshold,
            top_k=top_k,
            separator=separator,
        )

    # ----- query: name lookup, listing, relevance scoring -----

    def get_by_name(self, semantic_name: str) -> list[InstructionBlock]:
        """Return all blocks with the given semantic name in registry order.

        Names can collide — writes that pick the same ``semantic_name``
        produce multiple registry entries. The returned list is ordered by
        registry position (front-to-back), not by write order, so callers
        that care about "the most prominent block with this name" can take
        the first element regardless of when each was written.

        Args:
            semantic_name: Human-readable identifier to look up.

        Returns:
            Ordered list of matching blocks. Empty if no block has this name.
        """
        ids = self._name_index.get(semantic_name)
        if not ids:
            return []
        id_set = set(ids)
        return [self._blocks[rid] for rid in self._blocks if rid in id_set]

    def list_blocks(
        self,
        source_filter: BlockSourceKind | None = None,
    ) -> list[InstructionBlock]:
        """Return blocks in registry order, optionally filtered by source.

        Args:
            source_filter: Restrict to one of ``"seed"``, ``"agent"``, or
                ``"orchestrator"``. ``None`` returns every block.

        Returns:
            Ordered list of matching blocks.
        """
        if source_filter is None:
            return list(self._blocks.values())
        return [b for b in self._blocks.values() if b.source == source_filter]

    def find_by_tag(self, tag: str) -> list[InstructionBlock]:
        """Return every block whose ``tags`` list contains ``tag``.

        Linear scan over the registry. Compile order is preserved so
        downstream code that cares about ordering (e.g. position-aware
        prompt assembly) does not need to re-sort the result.

        Args:
            tag: Single tag to match exactly. Empty string matches no
                blocks because tags-as-empty-strings are not produced
                by the parser or by ``write_block``.

        Returns:
            Ordered list of blocks tagged ``tag``. Empty list if none.
        """
        return [b for b in self._blocks.values() if tag in b.tags]

    def find_by_tags(
        self,
        tags: list[str],
        match_all: bool = False,
    ) -> list[InstructionBlock]:
        """Return blocks whose ``tags`` list intersects ``tags``.

        Args:
            tags: Tags to match against. An empty input list returns
                an empty result list under both modes — a query with
                no positive criterion produces no hits.
            match_all: When ``True``, a block must contain every tag
                in ``tags`` (set-superset). When ``False`` (default),
                a block matches if at least one tag overlaps
                (set-intersection non-empty).

        Returns:
            Ordered list of matching blocks. Compile order is
            preserved.
        """
        if not tags:
            return []
        wanted = set(tags)
        if match_all:
            return [
                b for b in self._blocks.values()
                if wanted.issubset(b.tags)
            ]
        return [
            b for b in self._blocks.values()
            if not wanted.isdisjoint(b.tags)
        ]

    def score_block(
        self,
        runtime_id: str,
        task_embedding: NDArray[Any],
    ) -> float | None:
        """Cosine similarity between a block's embedding and a task vector.

        Returns ``None`` rather than raising when the block has no stored
        embedding. This keeps callers that score many blocks in a loop free
        of try/except scaffolding when some blocks are mid-lifecycle (e.g.
        a freshly written block awaiting the next batch encode).

        Args:
            runtime_id: The block to score.
            task_embedding: A 1-D numpy vector of compatible dimensionality
                with ``block.embedding``.

        Returns:
            Cosine similarity in ``[-1.0, 1.0]``, or ``None`` if the block
            has no embedding.

        Raises:
            BlockNotFoundError: If ``runtime_id`` does not exist.
        """
        block = self.get_by_id(runtime_id)
        if block.embedding is None:
            return None
        return _cosine(block.embedding, task_embedding)

    def find_relevant(
        self,
        task_text: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[tuple[InstructionBlock, float]]:
        """Embed ``task_text``, score every embedded block, return ranked matches.

        Blocks with no stored embedding are silently skipped — that state is
        valid (a block written via ``write_block`` / ``write_agent_block``
        after cold start has no embedding until the next batch encode, and
        a registry loaded without an embedding model has no embeddings
        at all). The skip behaviour means a registry where every block
        lacks an embedding returns an empty list rather than raising
        mid-loop.

        Args:
            task_text: Free-form text to embed and compare against.
            top_k: If set, cap the result to the top-K best matches by score.
            threshold: If set, drop matches with cosine similarity below this.

        Returns:
            List of ``(block, score)`` pairs in descending score order.
            Always contains only blocks with a stored embedding.

        Raises:
            RuntimeError: If no embedding model is configured on this
                registry. ``task_text`` cannot be embedded without one;
                use ``score_block`` instead when only a precomputed task
                vector is available.
        """
        if self._embedding_model is None:
            raise RuntimeError(
                "find_relevant requires an embedding model; "
                "construct BlockRegistry(embedding_model=...) first."
            )
        task_embedding = np.asarray(self._embedding_model.encode(task_text))
        scored: list[tuple[InstructionBlock, float]] = []
        for block in self._blocks.values():
            if block.embedding is None:
                continue
            scored.append((block, _cosine(block.embedding, task_embedding)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        if threshold is not None:
            scored = [p for p in scored if p[1] >= threshold]
        if top_k is not None:
            scored = scored[:top_k]
        return scored

    # ----- mutation: ordering -----

    def promote(
        self,
        runtime_id: str,
        *,
        steps: int | None = None,
        to_position: int | None = None,
        before: str | None = None,
        after: str | None = None,
        to_front: bool = False,
        to_back: bool = False,
    ) -> int:
        """Move a block within the registry. Exactly one directive must be set.

        Directives (mutually exclusive — supplying zero or more than one raises
        ``ValueError``):

        * ``steps``: relative move toward position 0 by the given count
          (clamped at the front).
        * ``to_position``: absolute target index in the final ordering
          (clamped to ``[0, len(self) - 1]``).
        * ``before``: place adjacent to and before another runtime_id.
        * ``after``: place adjacent to and after another runtime_id.
        * ``to_front``: shorthand for ``to_position=0``.
        * ``to_back``: shorthand for ``to_position=len(self) - 1``. Logically
          a demotion; supported here so the flexible signature covers both
          directions. ``demote_to_back`` is a clearer call site.

        Args:
            runtime_id: The block to move.
            steps: See above.
            to_position: See above.
            before: See above. Must not equal ``runtime_id``.
            after: See above. Must not equal ``runtime_id``.
            to_front: See above.
            to_back: See above.

        Returns:
            The new index of the block in the ordering.

        Raises:
            ValueError: If the number of directives is not exactly 1, or if
                ``before``/``after`` equals ``runtime_id``.
            BlockNotFoundError: If ``runtime_id``, ``before``, or ``after``
                does not exist in the registry.
        """
        self._validate_promote_directive_count(
            steps, to_position, before, after, to_front, to_back
        )
        self.get_by_id(runtime_id)

        keys = list(self._blocks.keys())
        cur = keys.index(runtime_id)
        keys.pop(cur)

        if to_front:
            new_index = 0
        elif to_back:
            new_index = len(keys)
        elif to_position is not None:
            new_index = max(0, min(to_position, len(keys)))
        elif steps is not None:
            new_index = max(0, cur - steps)
        elif before is not None:
            if before == runtime_id:
                raise ValueError("'before' cannot equal the moving runtime_id")
            self.get_by_id(before)
            new_index = keys.index(before)
        else:
            assert after is not None  # validation guarantees coverage
            if after == runtime_id:
                raise ValueError("'after' cannot equal the moving runtime_id")
            self.get_by_id(after)
            new_index = keys.index(after) + 1

        keys.insert(new_index, runtime_id)
        self._rebuild_order(keys)
        return new_index

    def demote(self, runtime_id: str, steps: int = 1) -> int:
        """Move a block toward the back by ``steps`` positions.

        Args:
            runtime_id: The block to move.
            steps: Number of positions to move toward the end. Clamped so
                the block does not pass the last index.

        Returns:
            The new index of the block in the ordering.

        Raises:
            BlockNotFoundError: If ``runtime_id`` does not exist.
            ValueError: If ``steps`` is negative.
        """
        if steps < 0:
            raise ValueError(
                "demote() steps must be non-negative; use promote() to move forward"
            )
        self.get_by_id(runtime_id)
        keys = list(self._blocks.keys())
        cur = keys.index(runtime_id)
        keys.pop(cur)
        new_index = min(len(keys), cur + steps)
        keys.insert(new_index, runtime_id)
        self._rebuild_order(keys)
        return new_index

    def promote_to_front(self, runtime_id: str) -> int:
        """Move a block to position 0. Convenience wrapper over ``promote``."""
        return self.promote(runtime_id, to_front=True)

    def demote_to_back(self, runtime_id: str) -> int:
        """Move a block to the last index. Convenience wrapper over ``promote``."""
        return self.promote(runtime_id, to_back=True)

    def reorder(self, id_list: list[str]) -> None:
        """Place the listed blocks at the front in the given order.

        Blocks not in ``id_list`` retain their relative order after the
        listed blocks. O(k + N).

        Args:
            id_list: Ordered list of runtime ids to move to the front.

        Raises:
            BlockNotFoundError: If any id in ``id_list`` is unknown.
            ValueError: If ``id_list`` contains duplicate ids.
        """
        if len(set(id_list)) != len(id_list):
            raise ValueError("id_list contains duplicate runtime ids")
        for rid in id_list:
            self.get_by_id(rid)
        listed = set(id_list)
        tail = [k for k in self._blocks if k not in listed]
        self._rebuild_order(list(id_list) + tail)

    # ----- mutation: writes -----

    def write_agent_block(
        self,
        content: str,
        trajectory_id: str,
        semantic_name: str | None = None,
        priority: float = 0.5,
        tags: list[str] | None = None,
        *,
        position: int | None = None,
        evictable: bool = True,
    ) -> str:
        """Insert a new block tagged ``source="agent"``.

        Convenience helper over :meth:`write_block` that pins
        ``source="agent"`` and requires a ``trajectory_id`` so the
        block can later be retracted via ``rollback_trajectory``.

        Args:
            content: The natural-language text of the new block.
            trajectory_id: The run UUID associated with this write.
            semantic_name: Human-readable identifier. If omitted,
                derived from ``trajectory_id`` so writes from different
                runs do not accidentally collide in the name index.
            priority: Eviction-scoring weight in [0.0, 1.0].
            tags: String tags. A fresh list is allocated each call;
                not shared between blocks.
            position: Target index in the final ordering. ``None``
                (default) appends at the end. An integer is clamped to
                ``[0, len(self)]``.
            evictable: When ``True`` (default), the block is removable
                by any caller via the eviction policy or
                ``evict(rid)``. When ``False``, only ``evict(rid,
                force=True)`` from this same registry can remove it.

        Returns:
            The newly-assigned runtime_id.

        Raises:
            ValueError: If ``priority`` is outside [0.0, 1.0].
        """
        if semantic_name is None:
            semantic_name = f"agent_{trajectory_id}_{time.time_ns()}"
        return self.write_block(
            content=content,
            semantic_name=semantic_name,
            source="agent",
            priority=priority,
            trajectory_id=trajectory_id,
            tags=tags,
            position=position,
            evictable=evictable,
        )

    def write_block(
        self,
        content: str,
        semantic_name: str,
        source: BlockSourceKind = "agent",
        priority: float = 0.5,
        trajectory_id: str | None = None,
        tags: list[str] | None = None,
        *,
        position: int | None = None,
        evictable: bool = True,
    ) -> str:
        """Insert a new block with full control over provenance and ordering.

        The unified write path. ``write_agent_block`` is a thin wrapper
        that fixes ``source="agent"`` and demands a ``trajectory_id``.
        Coordination-block usage is just ``write_block(content,
        semantic_name=..., source="orchestrator", evictable=False,
        position=0)``.

        ``author_id`` is stamped automatically from ``self.registry_id``
        so the ``evict(force=True)`` path can authenticate the caller.

        Args:
            content: The natural-language text of the new block.
            semantic_name: Human-readable identifier.
            source: Provenance tag. Defaults to ``"agent"``.
            priority: Eviction-scoring weight in [0.0, 1.0].
            trajectory_id: Optional run UUID for trajectory-rollback
                bookkeeping. ``None`` is fine for any write that does
                not participate in ``rollback_trajectory``.
            tags: String tags. A fresh list is allocated each call.
            position: Target index. ``None`` appends; an integer
                clamps to ``[0, len(self)]``.
            evictable: When ``False``, the block becomes non-evictable
                except via ``evict(force=True)`` from this same
                registry.

        Returns:
            The newly-assigned runtime_id.

        Raises:
            ValueError: If ``priority`` is outside [0.0, 1.0].
        """
        self._validate_priority(priority)
        runtime_id = generate_runtime_id()
        block = InstructionBlock(
            semantic_name=semantic_name,
            runtime_id=runtime_id,
            content=content,
            source=source,
            author_id=self.registry_id,
            evictable=evictable,
            priority=priority,
            trajectory_id=trajectory_id,
            created_at=int(time.time()),
            tags=list(tags) if tags else [],
        )
        self._blocks[runtime_id] = block
        self._index_add(semantic_name, runtime_id)
        if position is not None:
            keys = list(self._blocks.keys())
            keys.remove(runtime_id)
            clamped = max(0, min(position, len(keys)))
            keys.insert(clamped, runtime_id)
            self._rebuild_order(keys)
        self._memory_bytes += _estimate_block_bytes(block)
        self._maybe_warn_memory()
        return runtime_id

    def update_block_content(self, runtime_id: str, new_content: str) -> None:
        """Replace the content of a block tagged ``source="agent"``.

        Delegates to ``InstructionBlock.update_content`` so the
        immutability rules for ``"seed"`` and ``"orchestrator"``
        sources are enforced in a single place.

        Args:
            runtime_id: The block to modify.
            new_content: The new natural-language text.

        Raises:
            BlockNotFoundError: If ``runtime_id`` does not exist.
            SeedMutationError: If the block has ``source="seed"``.
            OrchestratorMutationError: If the block has
                ``source="orchestrator"``.
        """
        block = self.get_by_id(runtime_id)
        block.update_content(new_content)

    def update_priority(self, runtime_id: str, new_priority: float) -> None:
        """Set the priority of any block.

        Does not auto-reorder. Ordering is always explicit; the new priority
        is consulted by the eviction policy on its next run.

        Args:
            runtime_id: The block to modify.
            new_priority: New priority value in [0.0, 1.0].

        Raises:
            BlockNotFoundError: If ``runtime_id`` does not exist.
            ValueError: If ``new_priority`` is outside [0.0, 1.0].
        """
        self._validate_priority(new_priority)
        block = self.get_by_id(runtime_id)
        block.priority = new_priority

    def set_wrap(
        self,
        runtime_id: str,
        prefix: str | None = None,
        suffix: str | None = None,
    ) -> None:
        """Set or clear compile-time decoration for a block.

        The wrap fields are applied only during ``compile()`` and are
        not stored in the block's content. This is what lets a caller decorate
        a seed block (e.g. wrap it in ``<critical>...</critical>`` for a
        single task) without violating seed immutability.

        Passing ``None`` for either argument clears that field.

        Args:
            runtime_id: The block to decorate.
            prefix: Text to emit before the block's content. ``None`` clears.
            suffix: Text to emit after the block's content. ``None`` clears.

        Raises:
            BlockNotFoundError: If ``runtime_id`` does not exist.
        """
        block = self.get_by_id(runtime_id)
        block.wrap_prefix = prefix
        block.wrap_suffix = suffix

    # ----- mutation: removal -----

    def evict(self, runtime_id: str, *, force: bool = False) -> None:
        """Remove a block from the registry.

        Permission model:

        * ``block.evictable is True`` → any caller can remove the block
          regardless of ``force``. The default eviction policy already
          filters non-evictable blocks out of its candidate set, so
          this branch covers the typical caller-written-block case.
        * ``block.evictable is False`` and ``force=False`` → raises
          :class:`EvictionError`. The error message names the block
          and the registry that authored it so the operator has a
          clear path to releasing the protection or contacting the
          author.
        * ``block.evictable is False`` and ``force=True`` →
          authenticated channel. Succeeds only when
          ``self.registry_id == block.author_id``. Any other registry
          calling ``force=True`` against a foreign-authored block
          raises :class:`PermissionError`.

        Args:
            runtime_id: The block to remove.
            force: Permit removal of non-evictable blocks authored by
                this registry. No effect on evictable blocks.

        Raises:
            BlockNotFoundError: If ``runtime_id`` does not exist.
            EvictionError: If the block is non-evictable and
                ``force=False``.
            PermissionError: If the block is non-evictable and
                ``force=True`` but the calling registry is not the
                block's author.
        """
        block = self.get_by_id(runtime_id)
        if not block.evictable:
            if not force:
                raise EvictionError(
                    f"Block {block.semantic_name!r} is non-evictable "
                    f"(authored by registry {block.author_id!r}). "
                    f"Use evict(..., force=True) from the authoring "
                    f"registry to remove it."
                )
            if block.author_id != self.registry_id:
                raise PermissionError(
                    f"Only the authoring registry "
                    f"{block.author_id!r} may force-evict the "
                    f"non-evictable block {block.semantic_name!r}; "
                    f"this registry is {self.registry_id!r}."
                )
        self._memory_bytes -= _estimate_block_bytes(block)
        if self._memory_bytes < 0:
            self._memory_bytes = 0
        self._remove(runtime_id, block)

    def evict_by_policy(self) -> list[str]:
        """Score every evictable block and remove them in score order.

        The configured policy is consulted only for ``score(block)``;
        the filter to evictable blocks happens here, before the
        policy is asked to score anything, so the policy never sees a
        non-evictable block. After scoring, the registry sorts in
        descending score order and removes each block via
        :meth:`evict` — keeping the same evictable + author_id guards
        on the actual mutation step.

        Returns:
            Ordered list of runtime_ids that were actually removed.
            Stops at the first failure; the partial-removal state is
            committed.

        Raises:
            EvictionError: If a block became non-evictable between
                the filter step and the call to :meth:`evict` — not
                possible under the current single-threaded
                implementation but kept on the contract so the
                guarantee survives a future RLock-backed subclass.
            PermissionError: As above for cross-author force evicts;
                also currently unreachable from the default flow.
            BlockNotFoundError: As above for races between the score
                step and the evict step.
        """
        candidates: list[tuple[InstructionBlock, float]] = [
            (block, self._eviction_policy.score(block))
            for block in self._blocks.values()
            if block.evictable
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        evicted: list[str] = []
        for block, _score in candidates:
            self.evict(block.runtime_id)
            evicted.append(block.runtime_id)
        return evicted

    def memory_bytes(self) -> int:
        """Estimate of in-RAM bytes consumed by this registry's blocks.

        Walks every block and sums an approximation of each field's
        size (string lengths plus embedding ``nbytes`` plus a small
        constant overhead per block). Used by :meth:`__init__` to
        seed the running counter, by tests, and by the soft
        ``memory_limit_mb`` warning.

        Returns:
            Bytes. The estimate is intentionally cheap rather than
            exact; it should be within ~30% of the true Python object
            graph size for typical block content.
        """
        return sum(
            _estimate_block_bytes(b) for b in self._blocks.values()
        )

    def _maybe_warn_memory(self) -> None:
        """Emit a ``RuntimeWarning`` if memory crossed the soft limit.

        Compares the running ``self._memory_bytes`` against
        ``self.memory_limit_mb``. The warning fires every time the
        threshold is exceeded — callers may dedupe via
        ``warnings.simplefilter`` if they want one-shot behaviour.
        """
        if self.memory_limit_mb == float("inf"):
            return
        threshold = self.memory_limit_mb * 1024 * 1024
        if self._memory_bytes > threshold:
            warnings.warn(
                (
                    f"BlockRegistry {self.label!r} memory estimate "
                    f"({self._memory_bytes / (1024 * 1024):.1f} MB) "
                    f"exceeds memory_limit_mb "
                    f"({self.memory_limit_mb:.1f} MB). No automatic "
                    f"eviction will occur; this is a soft warning only."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

    def rollback_trajectory(self, trajectory_id: str) -> int:
        """Author retraction of every ``source="agent"`` block in this trajectory.

        **This is author retraction, not policy eviction.** The two
        paths are deliberately separate:

        * ``evict()`` and ``evict_by_policy()`` represent third-party
          removal under the registry's protection rules. They honour
          the ``evictable`` flag and the ``author_id`` check.
        * ``rollback_trajectory()`` represents the *author* deciding
          to undo their own writes. It bypasses the ``evictable``
          flag because the author already had permission to write the
          block; gating their own retraction behind the same flag
          would leave them no way to drop a diagnosis they later
          decided was wrong. Concretely: callers that write
          ``evictable=False`` blocks to protect them from the
          eviction policy still have ``rollback_trajectory`` as the
          explicit retraction channel.

        Only blocks with ``source="agent"`` are considered. Blocks
        tagged ``"seed"`` or ``"orchestrator"`` are never touched by
        this operation regardless of trajectory_id, since their
        source means they were not written by this trajectory.

        Args:
            trajectory_id: The trajectory whose blocks should be
                retracted.

        Returns:
            The number of blocks removed (zero if the trajectory has
            no matching ``source="agent"`` blocks in this registry).
        """
        to_remove = [
            rid
            for rid, b in self._blocks.items()
            if b.source == "agent"
            and b.trajectory_id == trajectory_id
        ]
        for rid in to_remove:
            block = self._blocks[rid]
            self._memory_bytes -= _estimate_block_bytes(block)
            if self._memory_bytes < 0:  # pragma: no cover - clamp guard
                self._memory_bytes = 0
            self._remove(rid, block)
        return len(to_remove)

    # ----- observability -----

    def registry_summary(self) -> dict[str, Any]:
        """Aggregate snapshot of the registry's current state.

        Cheap, side-effect-free pass over every block. Intended for
        operator dashboards, debugging, and notebook walkthroughs that
        need a single call returning everything at-a-glance.

        ``avg_priority`` is computed as the arithmetic mean of every
        block's ``priority`` field; for an empty registry it is ``0.0``
        rather than ``NaN`` so the return value remains
        JSON-serialisable without special-casing.

        ``first_block_runtime_id`` is the runtime_id at position 0 in
        compile order, or ``None`` if the registry has no blocks. The
        front position is the most attention-favoured slot per Liu et
        al. 2024, so this field is the natural anchor for "what does
        the model see first?" inspection.

        Returns:
            Dict with the following keys, all JSON-serialisable:

            * ``block_count`` (``int``): Total number of blocks.
            * ``by_source`` (``dict[str, int]``): Count per ``source``
              value (``"seed"``, ``"agent"``, ``"orchestrator"``).
              Sources with zero blocks are omitted.
            * ``total_tokens`` (``int``): Sum of ``token_count_of(b)``
              across every block. Uses the registry's configured
              tokeniser; results are cached per-block.
            * ``avg_priority`` (``float``): Arithmetic mean of every
              block's priority, rounded to three decimals. ``0.0`` for
              an empty registry.
            * ``total_access_count`` (``int``): Sum of
              ``access_count`` across every block.
            * ``first_block_runtime_id`` (``str | None``): Runtime ID
              of the block at position 0, or ``None`` if empty.
            * ``by_tag`` (``dict[str, int]``): Count of blocks per
              distinct tag. A block contributes to every tag in its
              ``tags`` list. Tags with zero blocks are omitted; the
              dict is empty when no block carries any tag.
        """
        blocks = list(self._blocks.values())
        by_source: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        total_tokens = 0
        total_access = 0
        priority_sum = 0.0
        for b in blocks:
            by_source[b.source] = by_source.get(b.source, 0) + 1
            for tag in b.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1
            total_tokens += self.token_count_of(b)
            total_access += b.access_count
            priority_sum += b.priority
        avg_priority = priority_sum / len(blocks) if blocks else 0.0
        first_id = blocks[0].runtime_id if blocks else None
        return {
            "block_count": len(blocks),
            "by_source": by_source,
            "total_tokens": total_tokens,
            "avg_priority": round(avg_priority, 3),
            "total_access_count": total_access,
            "first_block_runtime_id": first_id,
            "by_tag": by_tag,
        }

    def provenance_report(
        self,
        trajectory_id: str | None = None,
    ) -> dict[str, Any]:
        """Provenance breakdown by source and by trajectory.

        Walks every block once and aggregates the source counts plus
        a ``trajectory_id -> [semantic_name, ...]`` map. Blocks
        without a trajectory_id (typically blocks tagged ``"seed"``
        or ``"orchestrator"``) are counted in ``by_source`` but
        omitted from ``trajectories``.

        When ``trajectory_id`` is supplied, the report is restricted
        to blocks whose ``trajectory_id`` exactly equals that value:
        ``by_source`` and ``total_blocks`` reflect only the matching
        subset, and ``trajectories`` contains either one entry (the
        requested trajectory) or zero (no matching blocks).

        Args:
            trajectory_id: Optional filter. ``None`` (default) reports
                on every block in the registry. A string value
                restricts the report to blocks whose ``trajectory_id``
                equals that string.

        Returns:
            Dict with the following keys:

            * ``total_blocks`` (``int``): Number of blocks included
              after the optional trajectory filter.
            * ``by_source`` (``dict[str, int]``): Count per ``source``
              value among the included blocks. Sources with zero
              included blocks are omitted.
            * ``trajectories`` (``dict[str, list[str]]``): Mapping
              from trajectory_id to the semantic_names of the blocks
              that carry that trajectory_id, in registry order.
              Blocks with ``trajectory_id is None`` do not appear.
        """
        included = [
            b
            for b in self._blocks.values()
            if trajectory_id is None or b.trajectory_id == trajectory_id
        ]
        by_source: dict[str, int] = {}
        trajectories: dict[str, list[str]] = {}
        for b in included:
            by_source[b.source] = by_source.get(b.source, 0) + 1
            if b.trajectory_id is not None:
                trajectories.setdefault(b.trajectory_id, []).append(
                    b.semantic_name
                )
        return {
            "total_blocks": len(included),
            "by_source": by_source,
            "trajectories": trajectories,
        }

    # ----- internals -----

    def _validate_promote_directive_count(
        self,
        steps: int | None,
        to_position: int | None,
        before: str | None,
        after: str | None,
        to_front: bool,
        to_back: bool,
    ) -> None:
        """Reject ``promote()`` calls that do not supply exactly one directive."""
        provided = [
            steps is not None,
            to_position is not None,
            before is not None,
            after is not None,
            to_front,
            to_back,
        ]
        n = sum(provided)
        if n != 1:
            raise ValueError(
                f"promote() requires exactly one of "
                f"{{steps, to_position, before, after, to_front, to_back}}; "
                f"got {n}."
            )

    def _validate_priority(self, priority: float) -> None:
        """Reject priorities outside the [0.0, 1.0] interval."""
        if not 0.0 <= priority <= 1.0:
            raise ValueError(f"priority must be in [0.0, 1.0]; got {priority}")

    def _rebuild_order(self, ordered_keys: list[str]) -> None:
        """Reorder the underlying OrderedDict to match ``ordered_keys``.

        Walks the desired order and calls ``move_to_end`` on each key in turn.
        The contract: ``ordered_keys`` must be a permutation of the current
        keys. Callers in this module always satisfy that.
        """
        for k in ordered_keys:
            self._blocks.move_to_end(k)

    def _index_add(self, semantic_name: str, runtime_id: str) -> None:
        """Append ``runtime_id`` to the name-index slot for ``semantic_name``."""
        self._name_index.setdefault(semantic_name, []).append(runtime_id)

    def _index_remove(self, semantic_name: str, runtime_id: str) -> None:
        """Remove ``runtime_id`` from the name-index slot. Drop empty slots.

        Internal contract: every block in ``_blocks`` has a corresponding
        ``runtime_id`` entry under its ``semantic_name`` in ``_name_index``,
        and this method is only ever called from ``_remove`` for a block that
        was just looked up via ``_blocks``. The two structures are kept in
        sync by every public mutation, so no defensive branches are needed.
        """
        ids = self._name_index[semantic_name]
        ids.remove(runtime_id)
        if not ids:
            del self._name_index[semantic_name]

    def _remove(self, runtime_id: str, block: InstructionBlock) -> None:
        """Drop a block from both the ordering and the name index."""
        del self._blocks[runtime_id]
        self._index_remove(block.semantic_name, runtime_id)


def _cosine(a: NDArray[Any], b: NDArray[Any]) -> float:
    """Cosine similarity between two 1-D numpy vectors.

    Returns ``0.0`` if either vector is the zero vector. Choosing 0.0 over
    a divide-by-zero exception keeps relevance loops simple: the natural
    interpretation of "no signal" is "no similarity", and a zero score is
    correctly filtered out by any positive threshold the caller supplies.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _estimate_block_bytes(block: InstructionBlock) -> int:
    """Approximate in-RAM byte cost of an ``InstructionBlock``.

    Sums ``sys.getsizeof`` over the heaviest fields plus a constant
    overhead for the dataclass instance and the shorter strings. The
    estimate is intentionally cheap (no recursive object-graph walk)
    and within ~30% of the true Python footprint for typical block
    content. Used by the running ``_memory_bytes`` counter that drives
    the soft ``memory_limit_mb`` warning.
    """
    total = sys.getsizeof(block.content)
    total += sys.getsizeof(block.semantic_name)
    total += sys.getsizeof(block.runtime_id)
    total += sys.getsizeof(block.author_id)
    total += sys.getsizeof(block.tags)
    if block.embedding is not None:
        total += int(block.embedding.nbytes)
    # Constant overhead: dataclass __dict__, source enum string,
    # priority/created_at/access_count primitives, etc.
    return total + 256


__all__ = [
    "BlockNotFoundError",
    "BlockRegistry",
    "EmbeddingModel",
    "EvictionError",
    "RegistryFullError",
    "get_by_label",
]
