"""Eviction policy interface and the default scoring policy.

The default policy implements a two-component scoring formula::

    score = (1 - priority)         * priority_weight   (default 0.5)
          + 1 / (access_count + 1) * recency_weight    (default 0.5)

Higher score = higher eviction priority. Two structural decisions:

* **Non-evictable blocks are never returned by the default policy.**
  Blocks with ``evictable=False`` are filtered out before scoring. The
  registry's ``evict()`` separately enforces evictable + author_id
  gates regardless of policy, so this filter is the first line of
  defence rather than the only one.
* **No age gating, no token budget.** Age and the registry's total
  token cost do not participate in the policy. The registry is
  unbounded; the only token ceiling in the system is ``compile()``'s
  ``max_tokens`` argument. Eviction is immediate when called and never
  triggered automatically.

The :class:`EvictionPolicy` protocol's only required method is
:meth:`score`. The registry walks every evictable block, queries
``policy.score(block)``, and sorts in descending score order to
produce the eviction list. :class:`DefaultEvictionPolicy` also
exposes a :meth:`select` convenience method that returns the
ordered list directly; the two surfaces produce identical orderings.

Worked example — a tag-based protective policy. Any block tagged
``"safety_critical"`` returns ``float('-inf')`` regardless of
priority, so it sits at the bottom of every eviction ranking and is
never selected. The example shows how to compose the protection
on top of the default formula::

    from rampart.eviction import DefaultEvictionPolicy

    class TagProtectivePolicy(DefaultEvictionPolicy):
        \"\"\"Default scoring + a hard veto for safety_critical blocks.\"\"\"

        def score(self, block):
            if "safety_critical" in block.tags:
                return float("-inf")
            return super().score(block)

    registry = BlockRegistry(eviction_policy=TagProtectivePolicy())

The override returns ``-inf``, not just a small number, so the
protected blocks remain at the bottom of the ranking even after
arbitrary perturbations to the score formula in subclasses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rampart.block import InstructionBlock
    from rampart.registry import BlockRegistry


@runtime_checkable
class EvictionPolicy(Protocol):
    """Strategy interface for ranking blocks by eviction priority.

    The only required method is :meth:`score`. The registry walks
    every evictable block, queries ``policy.score(block)``, and
    sorts in descending score order so the highest scores are
    evicted first. Non-evictable blocks (``block.evictable is
    False``) are filtered out before the policy is consulted — the
    policy never sees them, so an implementation does not need a
    defensive guard.

    Implementations are pure: ``score`` must not mutate the block
    or the registry. The registry calls ``score`` once per
    evictable block per ``evict_by_policy()`` call; caching state
    on the policy across calls is permitted but not required.
    """

    def score(self, block: InstructionBlock) -> float:
        """Score a single evictable block. Higher means evict first.

        Args:
            block: The block to rank. The registry guarantees
                ``block.evictable is True``.

        Returns:
            Eviction priority. ``float('-inf')`` is the canonical
            "never evict this" sentinel; the default formula
            returns values in ``[0.0, 1.0]``. Negative finite
            values are valid but unusual.
        """
        ...


class DefaultEvictionPolicy:
    """Score-based default eviction policy. Skips non-evictable blocks.

    See module docstring for the scoring formula. The policy returns
    every evictable block sorted by score, highest first; the registry
    decides how many to actually remove (typically all of them via
    ``evict_by_policy()``).
    """

    def __init__(
        self,
        priority_weight: float = 0.5,
        recency_weight: float = 0.5,
    ) -> None:
        """Construct a default eviction policy.

        Args:
            priority_weight: Coefficient for ``(1 - block.priority)``.
                Higher value penalises low-priority blocks more strongly.
            recency_weight: Coefficient for ``1 / (access_count + 1)``.
                Higher value penalises rarely-touched blocks more
                strongly.
        """
        self.priority_weight = priority_weight
        self.recency_weight = recency_weight

    def score(self, block: InstructionBlock) -> float:
        """Compute the eviction score for a single (eligible) block.

        Each component is in ``[0, 1]`` so the weighted sum is too,
        which keeps cross-block comparison and threshold tuning
        intuitive.

        Args:
            block: The block to score. The caller is expected to have
                already filtered out non-evictable blocks; this method
                does not check ``block.evictable``.

        Returns:
            Eviction priority in ``[0.0, 1.0]`` under default weights.
        """
        priority_term = (1.0 - block.priority) * self.priority_weight
        recency_term = (
            (1.0 / (block.access_count + 1)) * self.recency_weight
        )
        return priority_term + recency_term

    def select(self, registry: BlockRegistry) -> list[str]:
        """Return runtime_ids of evictable blocks ordered by score.

        Convenience entry point that returns the full ordered list
        directly. Equivalent to consulting :meth:`score` per block via
        the registry's ``evict_by_policy()``; ``select`` is
        implemented as a thin wrapper over ``score`` so any subclass
        that overrides
        ``score`` automatically reorders ``select`` correctly.
        """
        candidates: list[tuple[InstructionBlock, float]] = [
            (block, self.score(block))
            for block in registry.list_blocks()
            if block.evictable
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return [block.runtime_id for block, _ in candidates]


__all__ = [
    "DefaultEvictionPolicy",
    "EvictionPolicy",
]
