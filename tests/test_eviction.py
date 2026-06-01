"""Tests for the eviction policy interface and DefaultEvictionPolicy.

The simplified policy scores blocks by ``(1 - priority)`` and
``1 / (access_count + 1)`` only. Non-evictable blocks are filtered
out before scoring. There is no age gate, no clock, and no token
budget — the policy is purely a ranking function over the eligible
subset, and the registry decides how many of the returned ids to
actually remove.
"""

from __future__ import annotations

from rampart.block import InstructionBlock
from rampart.eviction import DefaultEvictionPolicy, EvictionPolicy
from rampart.registry import BlockRegistry
from rampart.security import generate_runtime_id


def _block(
    *,
    name: str,
    priority: float = 0.5,
    access_count: int = 0,
    evictable: bool = True,
    source: str = "agent",
    content: str = "x" * 10,
) -> InstructionBlock:
    """Build a populated block. ``source`` accepts the literal strings
    ``"seed"`` / ``"agent"`` / ``"orchestrator"`` — typed as plain str
    here so tests can mix freely without casts.
    """
    return InstructionBlock(  # type: ignore[arg-type]
        semantic_name=name,
        runtime_id=generate_runtime_id(),
        content=content,
        source=source,
        evictable=evictable,
        priority=priority,
        trajectory_id="t" if source == "agent" else None,
        access_count=access_count,
    )


def _populate(registry: BlockRegistry, blocks: list[InstructionBlock]) -> None:
    """Insert pre-built blocks bypassing the public write methods so
    tests can pin priority and access_count to exact values.
    """
    for block in blocks:
        registry._blocks[block.runtime_id] = block
        registry._index_add(block.semantic_name, block.runtime_id)


def _registry() -> BlockRegistry:
    return BlockRegistry(tokeniser=len)


# --- protocol satisfaction ---------------------------------------------------


class TestProtocol:
    def test_default_policy_satisfies_eviction_protocol(self) -> None:
        # The Protocol is runtime-checkable; pin the structural
        # subtype check so a future refactor that drops a method
        # surfaces here rather than in a downstream test.
        policy: EvictionPolicy = DefaultEvictionPolicy()
        assert isinstance(policy, EvictionPolicy)


# --- candidate filtering -----------------------------------------------------


class TestCandidateFiltering:
    def test_non_evictable_blocks_are_excluded(self) -> None:
        # Seeds (and any other evictable=False block) never appear in
        # the policy's output. The registry's evict() guard would
        # refuse them anyway, but filtering early keeps the policy
        # clean of "nothing to recommend" branches.
        r = _registry()
        keep = _block(name="keep", evictable=False, source="seed")
        drop = _block(name="drop", evictable=True)
        _populate(r, [keep, drop])
        ids = DefaultEvictionPolicy().select(r)
        assert ids == [drop.runtime_id]

    def test_orchestrator_block_excluded_when_non_evictable(self) -> None:
        # Orchestrator-source is no longer special-cased; the
        # exclusion is purely the evictable flag. A user who marks
        # an orchestrator block evictable=True would see it returned
        # by the policy (legal under the new contract).
        r = _registry()
        orch = _block(
            name="orch", source="orchestrator", evictable=False, priority=0.8
        )
        agent = _block(name="agent", evictable=True, priority=0.2)
        _populate(r, [orch, agent])
        ids = DefaultEvictionPolicy().select(r)
        assert ids == [agent.runtime_id]

    def test_evictable_orchestrator_block_included(self) -> None:
        r = _registry()
        orch = _block(
            name="orch", source="orchestrator", evictable=True, priority=0.5
        )
        _populate(r, [orch])
        ids = DefaultEvictionPolicy().select(r)
        assert ids == [orch.runtime_id]

    def test_empty_registry_returns_empty_list(self) -> None:
        assert DefaultEvictionPolicy().select(_registry()) == []

    def test_registry_with_only_non_evictable_returns_empty(self) -> None:
        r = _registry()
        _populate(r, [_block(name="s", evictable=False, source="seed")])
        assert DefaultEvictionPolicy().select(r) == []


# --- scoring -----------------------------------------------------------------


class TestScoring:
    def test_lower_priority_scores_higher(self) -> None:
        # Higher score = higher eviction priority. The lower-priority
        # block must come first in the returned order.
        r = _registry()
        low = _block(name="low", priority=0.1)
        high = _block(name="high", priority=0.9)
        _populate(r, [high, low])  # registry order intentionally reversed
        ids = DefaultEvictionPolicy().select(r)
        assert ids == [low.runtime_id, high.runtime_id]

    def test_lower_access_count_scores_higher(self) -> None:
        # When priorities tie, the rarely-touched block scores higher.
        r = _registry()
        rare = _block(name="rare", priority=0.5, access_count=0)
        warm = _block(name="warm", priority=0.5, access_count=10)
        _populate(r, [warm, rare])
        ids = DefaultEvictionPolicy().select(r)
        assert ids == [rare.runtime_id, warm.runtime_id]

    def test_priority_and_access_combine(self) -> None:
        # Default weights are (0.5, 0.5). A high-priority block with
        # zero access (recency_term = 0.5) scores 0.05 + 0.5 = 0.55,
        # versus a low-priority block with 10 accesses
        # ((1 - 0.1) * 0.5 + (1/11) * 0.5 ≈ 0.495). The high-priority
        # block ranks first.
        r = _registry()
        a = _block(name="a", priority=0.9, access_count=0)
        b = _block(name="b", priority=0.1, access_count=10)
        _populate(r, [a, b])
        ids = DefaultEvictionPolicy().select(r)
        # a's score (~0.55) > b's score (~0.495)
        assert ids[0] == a.runtime_id

    def test_custom_weights_override_defaults(self) -> None:
        r = _registry()
        a = _block(name="a", priority=0.9, access_count=0)
        b = _block(name="b", priority=0.1, access_count=10)
        _populate(r, [a, b])
        # All weight on priority: low-priority block wins.
        ids = DefaultEvictionPolicy(
            priority_weight=1.0, recency_weight=0.0
        ).select(r)
        assert ids[0] == b.runtime_id

    def test_score_is_deterministic(self) -> None:
        r = _registry()
        for i in range(5):
            _populate(
                r, [_block(name=f"n{i}", priority=0.5 + i * 0.05)]
            )
        first = DefaultEvictionPolicy().select(r)
        second = DefaultEvictionPolicy().select(r)
        assert first == second


# --- V2 item 2: score()-based protocol --------------------------------------


class TestScoreContract:
    """The V2 protocol's only required method is ``score(block) -> float``.
    Custom policies that implement only ``score`` plug into the registry
    via ``evict_by_policy`` without needing ``select``.
    """

    def test_default_policy_score_returns_formula_value(self) -> None:
        # priority=0.4, access=0, default weights 0.5/0.5
        # priority_term = (1 - 0.4) * 0.5 = 0.30
        # recency_term  = (1 / 1)   * 0.5 = 0.50
        # total = 0.80
        block = _block(name="x", priority=0.4, access_count=0)
        s = DefaultEvictionPolicy().score(block)
        assert s == 0.8

    def test_custom_score_only_policy_drives_eviction_order(
        self,
    ) -> None:
        # A policy that scores by tag membership alone — no select()
        # method anywhere on the class — must order evictions
        # correctly when used via evict_by_policy().
        from rampart.block import InstructionBlock

        class TagScorePolicy:
            def score(self, block: InstructionBlock) -> float:
                return 1.0 if "evict_first" in block.tags else 0.0

        r = BlockRegistry(eviction_policy=TagScorePolicy())
        last = r.write_agent_block("a", trajectory_id="t")
        first = r.write_agent_block(
            "b", trajectory_id="t", tags=["evict_first"],
        )
        evicted = r.evict_by_policy()
        assert evicted == [first, last]

    def test_runtime_checkable_protocol_accepts_score_only_class(
        self,
    ) -> None:
        # The Protocol's runtime_checkable behaviour: a class with
        # only score() must satisfy isinstance(EvictionPolicy).
        from rampart.block import InstructionBlock

        class ScoreOnly:
            def score(self, block: InstructionBlock) -> float:
                return 0.0

        assert isinstance(ScoreOnly(), EvictionPolicy)


class TestTagProtectivePolicyExample:
    """Documented worked example: any block tagged ``safety_critical``
    returns ``float('-inf')`` so it never reaches the front of the
    eviction ranking, regardless of how aggressively the rest of the
    formula scores it.
    """

    def test_safety_critical_block_sinks_to_bottom_of_ranking(
        self,
    ) -> None:
        from rampart.block import InstructionBlock

        class TagProtectivePolicy(DefaultEvictionPolicy):
            def score(self, block: InstructionBlock) -> float:
                if "safety_critical" in block.tags:
                    return float("-inf")
                return super().score(block)

        r = BlockRegistry(eviction_policy=TagProtectivePolicy())
        # Even a 0.0-priority block (which would normally score
        # highest under the default formula) is protected when
        # tagged safety_critical.
        protected = r.write_agent_block(
            "p", trajectory_id="t", priority=0.0,
            tags=["safety_critical"],
        )
        normal_low = r.write_agent_block(
            "low", trajectory_id="t", priority=0.2,
        )
        normal_high = r.write_agent_block(
            "high", trajectory_id="t", priority=0.8,
        )
        evicted = r.evict_by_policy()
        # Protected block is removed last (lowest score); the two
        # normal blocks come first, lower-priority of the pair on
        # top.
        assert evicted == [normal_low, normal_high, protected]

    def test_safety_critical_blocks_are_still_evicted_eventually(
        self,
    ) -> None:
        # The protective sentinel returns -inf, not "skip" — the
        # policy still ranks the block, just at the very bottom.
        # A caller that drains evict_by_policy() to completion still
        # removes it. To make protection structural, mark the block
        # evictable=False instead of relying on score sentinels.
        from rampart.block import InstructionBlock

        class TagProtectivePolicy(DefaultEvictionPolicy):
            def score(self, block: InstructionBlock) -> float:
                if "safety_critical" in block.tags:
                    return float("-inf")
                return super().score(block)

        r = BlockRegistry(eviction_policy=TagProtectivePolicy())
        protected = r.write_agent_block(
            "p", trajectory_id="t",
            tags=["safety_critical"],
        )
        evicted = r.evict_by_policy()
        assert evicted == [protected]
        assert protected not in r


# --- V2 item 2: RAMPARTConfig wiring ----------------------------------------


class TestConfigEvictionPolicy:
    def test_config_default_is_a_default_eviction_policy(self) -> None:
        from rampart.config import RAMPARTConfig

        cfg = RAMPARTConfig()
        assert isinstance(cfg.eviction_policy, DefaultEvictionPolicy)

    def test_each_config_carries_independent_policy_instance(
        self,
    ) -> None:
        from rampart.config import RAMPARTConfig

        a = RAMPARTConfig()
        b = RAMPARTConfig()
        assert a.eviction_policy is not b.eviction_policy

    def test_from_files_uses_config_eviction_policy(
        self, tmp_path,
    ) -> None:
        from rampart.block import InstructionBlock
        from rampart.config import RAMPARTConfig

        # A custom policy threaded through config.eviction_policy
        # must reach the registry built by from_files() when the
        # caller does not pass an explicit eviction_policy.
        class MarkerPolicy:
            def score(
                self, block: InstructionBlock,  # noqa: ARG002
            ) -> float:
                return 0.0

        marker = MarkerPolicy()
        cfg = RAMPARTConfig()
        cfg.eviction_policy = marker

        seed = tmp_path / "seed.md"
        seed.write_text(
            "---\nname: only\n---\nbody\n", encoding="utf-8",
        )
        registry = BlockRegistry.from_files(
            [str(seed)], config=cfg,
        )
        assert registry._eviction_policy is marker

    def test_explicit_eviction_policy_overrides_config(
        self, tmp_path,
    ) -> None:
        from rampart.block import InstructionBlock
        from rampart.config import RAMPARTConfig

        class A:
            def score(
                self, block: InstructionBlock,  # noqa: ARG002
            ) -> float:
                return 0.0

        class B:
            def score(
                self, block: InstructionBlock,  # noqa: ARG002
            ) -> float:
                return 1.0

        cfg = RAMPARTConfig()
        cfg.eviction_policy = A()
        explicit = B()
        seed = tmp_path / "seed.md"
        seed.write_text(
            "---\nname: only\n---\nbody\n", encoding="utf-8",
        )
        registry = BlockRegistry.from_files(
            [str(seed)], config=cfg, eviction_policy=explicit,
        )
        assert registry._eviction_policy is explicit
