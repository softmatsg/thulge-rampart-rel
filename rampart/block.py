"""InstructionBlock dataclass and the provenance-mutation exceptions it enforces.

The dataclass is the atomic, addressable unit of the registry. Two structural
defences live here rather than at the registry layer:

* ``SeedMutationError`` — seed text is the canonical record of human-authored
  domain knowledge. Allowing it to be rewritten in place would reproduce the
  context-collapse failure mode named in ACE (Zhang et al., 2025), where
  iterative summarisation erodes the original instructions over time.
* ``OrchestratorMutationError`` — blocks tagged ``source="orchestrator"`` are
  coordination signals authored by some other registry. From the receiving
  registry's perspective they are immutable in content; the authoring
  registry retracts them through its own retraction channels rather than
  through this dataclass.

The ``runtime_id`` field is intentionally a fresh UUID4 (never the
human-readable ``semantic_name``) so the storage key is unpredictable to an
attacker who knows only the seed file content. See ``security.py`` for the
generator and the ASLR-style rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from numpy.typing import NDArray

BlockSourceKind = Literal["seed", "agent", "orchestrator"]
"""Provenance tag identifying the author class of an InstructionBlock.

The name ``BlockSource`` is reserved for the file-source descriptor
in ``rampart.parser`` (a dataclass that carries a path plus optional
name filter). The literal here identifies the *kind* of source —
author class — which the file-source descriptor does not.
"""


class SeedMutationError(Exception):
    """Raised when an attempt is made to modify the content of a seed block.

    Seed blocks are loaded from source files at cold start and are immutable
    in content for the lifetime of the session. This protects against context
    collapse, where iterative rewriting erodes original human-authored
    domain knowledge.
    """


class OrchestratorMutationError(Exception):
    """Raised when a registry tries to modify a block authored elsewhere.

    Blocks tagged ``source="orchestrator"`` are coordination signals
    authored by some other registry. Their content is immutable from
    the receiving registry's perspective; the authoring registry
    retracts them through its own retraction channels rather than
    via ``update_content`` on the receiving side.
    """


@dataclass
class InstructionBlock:
    """Atomic addressable unit of the RAMPART block registry.

    A single behavioural directive, domain rule, skill procedure, or learned
    heuristic expressed in natural language, with accompanying metadata.

    Attributes:
        semantic_name: Human-readable identifier from the source file or
            from the caller at write time. Stable across sessions.
        runtime_id: UUID4 storage key. Regenerated on every cold start and
            snapshot restore. Never appears in any seed file.
        content: The natural-language text of the instruction. Immutable
            for blocks with ``source == "seed"`` or ``source == "orchestrator"``
            (from the receiving registry's perspective).
        source: Provenance tag. ``"seed"`` for human-authored boot blocks,
            ``"agent"`` for blocks written through ``write_agent_block`` (the
            convenience method that requires a ``trajectory_id``),
            ``"orchestrator"`` for coordination blocks tagged for retraction
            only by the authoring registry.
        author_id: UUID of the registry instance that wrote this block.
            For seed blocks loaded from files, ``author_id`` equals the
            loading registry's own ID. The empty string ``""`` is the
            parser sentinel — overwritten at insert time.
        evictable: Whether this block can be removed by any caller. When
            ``False``, only ``evict(force=True)`` from the authoring
            registry can remove it. Default ``True`` — any non-seed
            block is fair game for the eviction policy; seed blocks
            load with ``evictable=False``.
        priority: Eviction-scoring weight in the closed interval [0.0, 1.0].
            Updated by the caller. Drives the default eviction policy.
            Does not auto-reorder; ordering is always explicit.
        trajectory_id: Run UUID associated with a write that participates in
            ``rollback_trajectory`` retraction. ``None`` for seed blocks
            and any block written without a trajectory tag.
        created_at: Unix timestamp at creation. Recorded for diagnostics
            and trajectory inspection; the default eviction policy
            does not use it for age-based gating.
        access_count: Number of times this block has been emitted by
            ``BlockRegistry.compile()``.
        embedding: Optional float vector. Computed at boot time using the
            configured embedding model, ``None`` otherwise.
        kv_cache: Reserved for a future KV-tensor extension. Always
            ``None`` in the current release.
        tags: String tags from YAML frontmatter or caller metadata at
            write time. Used for filtering and organisation.
        token_count: Cached token count of ``content``. Computed lazily and
            invalidated on content mutation.
        wrap_prefix: Compile-time decoration applied before ``content`` in
            the assembled prompt. Not stored in ``content`` itself.
        wrap_suffix: Compile-time decoration applied after ``content`` in
            the assembled prompt. Not stored in ``content`` itself.
    """

    semantic_name: str
    runtime_id: str
    content: str
    source: BlockSourceKind
    author_id: str = ""
    evictable: bool = True
    priority: float = 0.5
    trajectory_id: str | None = None
    created_at: int = 0
    access_count: int = 0
    embedding: NDArray[Any] | None = None
    kv_cache: Any | None = None
    tags: list[str] = field(default_factory=list)
    token_count: int | None = None
    wrap_prefix: str | None = None
    wrap_suffix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise this block to a JSON-compatible dict.

        ``embedding`` is omitted because numpy arrays are not directly
        JSON-serialisable; embeddings are recomputed on snapshot restore.
        ``kv_cache`` is omitted because it is reserved for a future
        extension and is always ``None`` in the current release.

        Returns:
            Dict containing every primitive field of this block. Suitable
            for direct JSON encoding.
        """
        return {
            "semantic_name": self.semantic_name,
            "runtime_id": self.runtime_id,
            "content": self.content,
            "source": self.source,
            "author_id": self.author_id,
            "evictable": self.evictable,
            "priority": self.priority,
            "trajectory_id": self.trajectory_id,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "tags": list(self.tags),
            "token_count": self.token_count,
            "wrap_prefix": self.wrap_prefix,
            "wrap_suffix": self.wrap_suffix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstructionBlock:
        """Reconstruct an InstructionBlock from its dict representation.

        Inverse of :meth:`to_dict`. The reconstructed block always has
        ``embedding=None`` and ``kv_cache=None``; the registry recomputes
        embeddings as needed on restore.

        Args:
            data: Dict produced by a prior call to :meth:`to_dict`. Must
                contain at least ``semantic_name``, ``runtime_id``,
                ``content``, and ``source``. All other fields fall back to
                their dataclass defaults if absent.

        Returns:
            A new InstructionBlock instance.

        Raises:
            KeyError: If a required field is missing from ``data``.
        """
        source: BlockSourceKind = data["source"]
        return cls(
            semantic_name=data["semantic_name"],
            runtime_id=data["runtime_id"],
            content=data["content"],
            source=source,
            author_id=data.get("author_id", ""),
            evictable=data.get("evictable", True),
            priority=data.get("priority", 0.5),
            trajectory_id=data.get("trajectory_id"),
            created_at=data.get("created_at", 0),
            access_count=data.get("access_count", 0),
            embedding=None,
            kv_cache=None,
            tags=list(data.get("tags", [])),
            token_count=data.get("token_count"),
            wrap_prefix=data.get("wrap_prefix"),
            wrap_suffix=data.get("wrap_suffix"),
        )

    def update_content(self, new_content: str) -> None:
        """Replace the content of this block in place.

        Permitted only for blocks with ``source == "agent"``. Blocks
        tagged ``"seed"`` or ``"orchestrator"`` raise on mutation;
        this is the structural defence against context collapse
        described in ACE (Zhang et al., 2025) — the canonical
        instructions stay byte-stable for the lifetime of the
        session.

        Args:
            new_content: The new natural-language text for this block.

        Raises:
            SeedMutationError: If this block has ``source == "seed"``.
            OrchestratorMutationError: If this block has
                ``source == "orchestrator"``.
        """
        if self.source == "seed":
            raise SeedMutationError(
                f"Cannot modify content of seed block '{self.semantic_name}'."
            )
        if self.source == "orchestrator":
            raise OrchestratorMutationError(
                f"Cannot modify content of orchestrator-source block "
                f"'{self.semantic_name}' from a receiving registry."
            )
        self.content = new_content
        self.token_count = None


__all__ = [
    "BlockSourceKind",
    "InstructionBlock",
    "OrchestratorMutationError",
    "SeedMutationError",
]
