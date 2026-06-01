"""Tests for BlockRegistry mutations, validations, and edge cases.

Four named edge cases each have a dedicated test:

* ``test_promote_already_at_front_is_noop`` — promote a block already at front.
* ``test_evict_unknown_id_raises`` — evict a non-existent runtime_id.
* ``test_rollback_empty_trajectory_returns_zero`` — rollback an empty trajectory.
* ``test_reorder_partial_list_keeps_others_in_relative_order`` — partial reorder.

Seed blocks are injected via a small helper that bypasses the parser (the
parser arrives later). The helper writes through ``_index_add`` so the
name-index invariant is preserved even for direct injections.
"""

from __future__ import annotations

import uuid as _uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from rampart.block import (
    InstructionBlock,
    OrchestratorMutationError,
    SeedMutationError,
)
from rampart.parser import UnsupportedFormatError
from rampart.registry import (
    BlockNotFoundError,
    BlockRegistry,
    EvictionError,
)
from rampart.security import generate_runtime_id


def _inject_seed(reg: BlockRegistry, name: str, content: str = "x") -> str:
    """Write a seed block directly into ``reg``, bypassing the parser.

    Mirrors cold-start semantics: ``evictable=False`` and ``author_id``
    set to the receiving registry's ``registry_id``. Tests that need
    a different setting flip the attribute on the returned block via
    ``reg._blocks[rid]``.
    """
    rid = generate_runtime_id()
    block = InstructionBlock(
        semantic_name=name,
        runtime_id=rid,
        content=content,
        source="seed",
        author_id=reg.registry_id,
        evictable=False,
    )
    reg._blocks[rid] = block
    reg._index_add(name, rid)
    return rid


def _vec(*values: float) -> NDArray[Any]:
    """Build a small float32 vector for embedding-related tests."""
    return np.array(values, dtype=np.float32)


class _FakeEmbedder:
    """Minimal deterministic embedder for these tests.

    Avoids loading sentence-transformers in unit tests. Real embedding model
    wiring lives in ``rampart.scorer``. The mapping lets each test
    assert exact cosine scores against known vectors.
    """

    def __init__(self, mapping: dict[str, NDArray[Any]] | None = None) -> None:
        self.mapping = mapping or {}

    def encode(self, text: str) -> NDArray[Any]:
        return self.mapping.get(text, np.zeros(3, dtype=np.float32))


# --- minimal query primitives -------------------------------------------------


class TestQueryPrimitives:
    def test_empty_registry_is_zero_length(self) -> None:
        r = BlockRegistry()
        assert len(r) == 0
        assert r.runtime_ids() == []

    def test_contains_after_write(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("hello", trajectory_id="t1")
        assert rid in r
        assert "missing" not in r

    def test_get_by_id_returns_block(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("hello", trajectory_id="t1")
        block = r.get_by_id(rid)
        assert block.runtime_id == rid
        assert block.source == "agent"
        assert block.content == "hello"

    def test_get_by_id_missing_raises(self) -> None:
        r = BlockRegistry()
        with pytest.raises(BlockNotFoundError):
            r.get_by_id("nonexistent")


# --- write_agent_block --------------------------------------------------------


class TestWriteAgentBlock:
    def test_write_assigns_uuid4(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("foo", trajectory_id="t1")
        assert _uuid.UUID(rid).version == 4

    def test_write_appends_to_end(self) -> None:
        r = BlockRegistry()
        first = r.write_agent_block("a", trajectory_id="t1")
        second = r.write_agent_block("b", trajectory_id="t1")
        assert r.runtime_ids() == [first, second]

    def test_write_records_provenance(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("foo", trajectory_id="run-7")
        block = r.get_by_id(rid)
        assert block.source == "agent"
        assert block.trajectory_id == "run-7"
        assert block.created_at > 0

    def test_write_default_priority_is_half(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("foo", trajectory_id="t1")
        assert r.get_by_id(rid).priority == 0.5

    def test_write_invalid_priority_raises(self) -> None:
        r = BlockRegistry()
        with pytest.raises(ValueError):
            r.write_agent_block("foo", trajectory_id="t1", priority=1.5)
        with pytest.raises(ValueError):
            r.write_agent_block("foo", trajectory_id="t1", priority=-0.1)

    def test_default_tags_are_not_shared_between_blocks(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block("a", trajectory_id="t1")
        b = r.write_agent_block("b", trajectory_id="t1")
        r.get_by_id(a).tags.append("x")
        assert r.get_by_id(b).tags == []

    def test_explicit_semantic_name_is_used(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("c", trajectory_id="t1", semantic_name="my_heuristic")
        assert r.get_by_id(rid).semantic_name == "my_heuristic"

    def test_explicit_tags_are_copied_not_aliased(self) -> None:
        r = BlockRegistry()
        tags = ["hardware"]
        rid = r.write_agent_block("c", trajectory_id="t1", tags=tags)
        tags.append("mutated_after_write")
        assert r.get_by_id(rid).tags == ["hardware"]


# --- write_block (unified write surface) -------------------------------------


class TestWriteBlock:
    """The unified write_block() replaces write_orchestrator_block().
    Orchestrator-style usage = ``write_block(content, semantic_name=...,
    source="orchestrator", evictable=False, position=0)``.
    """

    def test_default_appends_at_end_with_agent_source(self) -> None:
        r = BlockRegistry()
        first = r.write_agent_block("a", trajectory_id="t1")
        rid = r.write_block("b", semantic_name="b")
        assert r.runtime_ids() == [first, rid]
        assert r.get_by_id(rid).source == "agent"

    def test_orchestrator_style_position_zero_lands_at_front(self) -> None:
        r = BlockRegistry()
        first = r.write_agent_block("a", trajectory_id="t1")
        orch = r.write_block(
            "urgent",
            semantic_name="alert",
            source="orchestrator",
            evictable=False,
            position=0,
        )
        assert r.runtime_ids() == [orch, first]

    def test_explicit_position_lands_in_middle(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block("a", trajectory_id="t1")
        b = r.write_agent_block("b", trajectory_id="t1")
        c = r.write_agent_block("c", trajectory_id="t1")
        rid = r.write_block(
            "brief", semantic_name="brief", position=2
        )
        assert r.runtime_ids() == [a, b, rid, c]

    def test_position_clamped_at_back(self) -> None:
        r = BlockRegistry()
        r.write_agent_block("a", trajectory_id="t1")
        rid = r.write_block("x", semantic_name="x", position=999)
        assert r.runtime_ids()[-1] == rid

    def test_explicit_source_orchestrator(self) -> None:
        r = BlockRegistry()
        rid = r.write_block(
            "x", semantic_name="x", source="orchestrator"
        )
        assert r.get_by_id(rid).source == "orchestrator"

    def test_default_evictable_is_true(self) -> None:
        r = BlockRegistry()
        rid = r.write_block("x", semantic_name="x")
        assert r.get_by_id(rid).evictable is True

    def test_evictable_false_propagates_to_block(self) -> None:
        r = BlockRegistry()
        rid = r.write_block(
            "x", semantic_name="x", evictable=False
        )
        assert r.get_by_id(rid).evictable is False

    def test_author_id_matches_writing_registry(self) -> None:
        r = BlockRegistry()
        rid = r.write_block("x", semantic_name="x")
        assert r.get_by_id(rid).author_id == r.registry_id

    def test_invalid_priority_raises(self) -> None:
        r = BlockRegistry()
        with pytest.raises(ValueError):
            r.write_block("x", semantic_name="x", priority=2.0)


# --- write_agent_block: position + evictable additions -----------------------


class TestWriteAgentBlockPositionAndEvictable:
    def test_position_none_appends(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block("a", trajectory_id="t1")
        b = r.write_agent_block("b", trajectory_id="t1")
        assert r.runtime_ids() == [a, b]

    def test_position_zero_lands_at_front(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block("a", trajectory_id="t1")
        b = r.write_agent_block("b", trajectory_id="t1", position=0)
        assert r.runtime_ids() == [b, a]

    def test_evictable_default_is_true(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        assert r.get_by_id(rid).evictable is True

    def test_evictable_false_carries_through(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block(
            "a", trajectory_id="t1", evictable=False
        )
        assert r.get_by_id(rid).evictable is False

    def test_author_id_set_from_registry(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        assert r.get_by_id(rid).author_id == r.registry_id


# --- promote() directive validation ------------------------------------------


class TestPromoteDirectiveValidation:
    def test_zero_directives_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        with pytest.raises(ValueError):
            r.promote(rid)

    def test_two_directives_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        with pytest.raises(ValueError):
            r.promote(rid, steps=2, to_front=True)

    def test_to_front_and_to_back_together_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        with pytest.raises(ValueError):
            r.promote(rid, to_front=True, to_back=True)

    def test_three_directives_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        with pytest.raises(ValueError):
            r.promote(rid, steps=1, to_position=0, to_front=True)

    def test_each_single_directive_succeeds(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block("a", trajectory_id="t1")
        b = r.write_agent_block("b", trajectory_id="t1")
        # Each call is one valid directive in isolation.
        assert r.promote(b, to_front=True) == 0
        assert r.promote(a, to_back=True) == 1
        assert r.promote(a, steps=1) == 0
        assert r.promote(b, to_position=1) == 1


# --- promote() semantics ------------------------------------------------------


class TestPromoteSemantics:
    def test_promote_steps_moves_toward_front(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abcde"]
        new = r.promote(ids[3], steps=2)
        assert new == 1
        assert r.runtime_ids() == [ids[0], ids[3], ids[1], ids[2], ids[4]]

    def test_promote_steps_clamps_at_front(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abc"]
        new = r.promote(ids[1], steps=99)
        assert new == 0
        assert r.runtime_ids()[0] == ids[1]

    def test_promote_already_at_front_is_noop(self) -> None:
        # named edge case.
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abc"]
        original = r.runtime_ids()
        new = r.promote(ids[0], to_front=True)
        assert new == 0
        assert r.runtime_ids() == original

    def test_promote_to_position_zero(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abcd"]
        new = r.promote(ids[2], to_position=0)
        assert new == 0
        assert r.runtime_ids() == [ids[2], ids[0], ids[1], ids[3]]

    def test_promote_to_position_clamps_high(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abc"]
        new = r.promote(ids[0], to_position=999)
        assert new == 2  # len(keys after pop) == 2, insertion at 2 → final index 2
        assert r.runtime_ids()[-1] == ids[0]

    def test_promote_before(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abcd"]
        new = r.promote(ids[3], before=ids[1])
        assert new == 1
        assert r.runtime_ids() == [ids[0], ids[3], ids[1], ids[2]]

    def test_promote_after(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abcd"]
        new = r.promote(ids[3], after=ids[0])
        assert new == 1
        assert r.runtime_ids() == [ids[0], ids[3], ids[1], ids[2]]

    def test_promote_before_self_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        with pytest.raises(ValueError):
            r.promote(rid, before=rid)

    def test_promote_after_self_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        with pytest.raises(ValueError):
            r.promote(rid, after=rid)

    def test_promote_unknown_id_raises(self) -> None:
        r = BlockRegistry()
        with pytest.raises(BlockNotFoundError):
            r.promote("nonexistent", to_front=True)

    def test_promote_unknown_before_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        with pytest.raises(BlockNotFoundError):
            r.promote(rid, before="nonexistent")

    def test_promote_unknown_after_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        with pytest.raises(BlockNotFoundError):
            r.promote(rid, after="nonexistent")

    def test_promote_to_back_via_promote(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abc"]
        new = r.promote(ids[0], to_back=True)
        assert new == 2
        assert r.runtime_ids() == [ids[1], ids[2], ids[0]]


# --- convenience wrappers ----------------------------------------------------


class TestPromoteToFrontConvenience:
    def test_promote_to_front_delegates(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block("a", trajectory_id="t1")
        b = r.write_agent_block("b", trajectory_id="t1")
        new = r.promote_to_front(b)
        assert new == 0
        assert r.runtime_ids() == [b, a]


class TestDemoteToBackConvenience:
    def test_demote_to_back_delegates(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block("a", trajectory_id="t1")
        b = r.write_agent_block("b", trajectory_id="t1")
        new = r.demote_to_back(a)
        assert new == 1
        assert r.runtime_ids() == [b, a]


# --- demote() -----------------------------------------------------------------


class TestDemote:
    def test_demote_steps_moves_toward_back(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abcd"]
        new = r.demote(ids[0], steps=2)
        assert new == 2
        assert r.runtime_ids() == [ids[1], ids[2], ids[0], ids[3]]

    def test_demote_default_steps_one(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abc"]
        new = r.demote(ids[0])
        assert new == 1
        assert r.runtime_ids() == [ids[1], ids[0], ids[2]]

    def test_demote_clamps_at_back(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abc"]
        new = r.demote(ids[0], steps=99)
        assert new == 2
        assert r.runtime_ids()[-1] == ids[0]

    def test_demote_negative_steps_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        with pytest.raises(ValueError):
            r.demote(rid, steps=-1)

    def test_demote_unknown_id_raises(self) -> None:
        r = BlockRegistry()
        with pytest.raises(BlockNotFoundError):
            r.demote("nonexistent")


# --- reorder() ----------------------------------------------------------------


class TestReorder:
    def test_reorder_full_list(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abcd"]
        r.reorder([ids[2], ids[0], ids[3], ids[1]])
        assert r.runtime_ids() == [ids[2], ids[0], ids[3], ids[1]]

    def test_reorder_partial_list_keeps_others_in_relative_order(self) -> None:
        # named edge case.
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abcde"]
        r.reorder([ids[3], ids[1]])
        assert r.runtime_ids() == [ids[3], ids[1], ids[0], ids[2], ids[4]]

    def test_reorder_empty_list_noop(self) -> None:
        r = BlockRegistry()
        for c in "abc":
            r.write_agent_block(c, trajectory_id="t")
        original = r.runtime_ids()
        r.reorder([])
        assert r.runtime_ids() == original

    def test_reorder_unknown_id_raises(self) -> None:
        r = BlockRegistry()
        r.write_agent_block("a", trajectory_id="t1")
        with pytest.raises(BlockNotFoundError):
            r.reorder(["nonexistent"])

    def test_reorder_duplicate_id_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("a", trajectory_id="t1")
        with pytest.raises(ValueError):
            r.reorder([rid, rid])


# --- update_block_content -----------------------------------------------------


class TestUpdateBlockContent:
    def test_update_agent_content(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("old", trajectory_id="t1")
        r.update_block_content(rid, "new")
        assert r.get_by_id(rid).content == "new"

    def test_update_seed_content_raises(self) -> None:
        r = BlockRegistry()
        rid = _inject_seed(r, "constraint")
        with pytest.raises(SeedMutationError):
            r.update_block_content(rid, "new")

    def test_update_orchestrator_content_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_block(
            "brief",
            semantic_name="brief",
            source="orchestrator",
            evictable=False,
            position=0,
        )
        with pytest.raises(OrchestratorMutationError):
            r.update_block_content(rid, "new")

    def test_update_unknown_id_raises(self) -> None:
        r = BlockRegistry()
        with pytest.raises(BlockNotFoundError):
            r.update_block_content("nonexistent", "x")


# --- update_priority ----------------------------------------------------------


class TestUpdatePriority:
    def test_update_priority_changes_field(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        r.update_priority(rid, 0.95)
        assert r.get_by_id(rid).priority == 0.95

    def test_update_priority_does_not_reorder(self) -> None:
        r = BlockRegistry()
        ids = [r.write_agent_block(c, trajectory_id="t") for c in "abc"]
        original = r.runtime_ids()
        r.update_priority(ids[2], 0.99)
        assert r.runtime_ids() == original

    def test_update_priority_out_of_range_raises(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        with pytest.raises(ValueError):
            r.update_priority(rid, 1.5)
        with pytest.raises(ValueError):
            r.update_priority(rid, -0.1)

    def test_update_priority_unknown_id_raises(self) -> None:
        r = BlockRegistry()
        with pytest.raises(BlockNotFoundError):
            r.update_priority("nonexistent", 0.5)

    def test_update_priority_works_for_seed_and_orchestrator(self) -> None:
        r = BlockRegistry()
        seed = _inject_seed(r, "s")
        orch = r.write_block(
            "o",
            semantic_name="o",
            source="orchestrator",
            evictable=False,
            position=0,
        )
        r.update_priority(seed, 0.1)
        r.update_priority(orch, 0.9)
        assert r.get_by_id(seed).priority == 0.1
        assert r.get_by_id(orch).priority == 0.9


# --- set_wrap -----------------------------------------------------------------


class TestSetWrap:
    def test_set_prefix_and_suffix(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        r.set_wrap(rid, prefix="<<", suffix=">>")
        block = r.get_by_id(rid)
        assert block.wrap_prefix == "<<"
        assert block.wrap_suffix == ">>"

    def test_clear_wrap(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        r.set_wrap(rid, prefix="<<", suffix=">>")
        r.set_wrap(rid)
        block = r.get_by_id(rid)
        assert block.wrap_prefix is None
        assert block.wrap_suffix is None

    def test_set_wrap_on_seed_block_does_not_violate_immutability(self) -> None:
        # Wrap is a compile-time decoration; setting it on a seed must not
        # require touching the seed's content.
        r = BlockRegistry()
        rid = _inject_seed(r, "seed_with_wrap", content="critical-rule")
        r.set_wrap(rid, prefix="<critical>", suffix="</critical>")
        block = r.get_by_id(rid)
        assert block.content == "critical-rule"
        assert block.wrap_prefix == "<critical>"

    def test_set_wrap_unknown_id_raises(self) -> None:
        r = BlockRegistry()
        with pytest.raises(BlockNotFoundError):
            r.set_wrap("nonexistent", prefix="x")


# --- evict --------------------------------------------------------------------


class TestEvict:
    def test_evict_agent_block(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        r.evict(rid)
        assert rid not in r
        assert len(r) == 0

    def test_evict_unknown_id_raises(self) -> None:
        # named edge case.
        r = BlockRegistry()
        with pytest.raises(BlockNotFoundError):
            r.evict("nonexistent")

    def test_evict_non_evictable_block_raises_eviction_error(self) -> None:
        r = BlockRegistry()
        rid = _inject_seed(r, "x")
        # Inject helper sets evictable=False below; pin the assertion
        # via the public attribute to keep the test robust against
        # changes to _inject_seed.
        r._blocks[rid].evictable = False
        with pytest.raises(EvictionError, match="non-evictable"):
            r.evict(rid)
        assert rid in r

    def test_evict_force_true_succeeds_for_authoring_registry(self) -> None:
        r = BlockRegistry()
        rid = _inject_seed(r, "x")
        r._blocks[rid].evictable = False
        r._blocks[rid].author_id = r.registry_id
        r.evict(rid, force=True)
        assert rid not in r

    def test_force_true_from_other_registry_raises_permission_error(
        self,
    ) -> None:
        a = BlockRegistry()
        b = BlockRegistry()
        rid = _inject_seed(a, "x")
        a._blocks[rid].evictable = False
        a._blocks[rid].author_id = a.registry_id
        # Smuggle the same block into b's index for this test.
        block = a._blocks[rid]
        b._blocks[rid] = block
        b._index_add(block.semantic_name, rid)
        with pytest.raises(PermissionError, match="authoring registry"):
            b.evict(rid, force=True)
        assert rid in b
        a.release()
        b.release()

    def test_evict_evictable_block_ignores_force_flag(self) -> None:
        # force=True is harmless for ordinary evictable blocks.
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        r.evict(rid, force=True)
        assert rid not in r

    def test_evict_clears_name_index(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1", semantic_name="my_block")
        assert "my_block" in r._name_index
        r.evict(rid)
        assert "my_block" not in r._name_index


# --- rollback_trajectory ------------------------------------------------------


class TestRollbackTrajectory:
    def test_rollback_removes_only_matching_agent_blocks(self) -> None:
        r = BlockRegistry()
        seed = _inject_seed(r, "fixed")
        a = r.write_agent_block("x", trajectory_id="run-A")
        b = r.write_agent_block("y", trajectory_id="run-A")
        c = r.write_agent_block("z", trajectory_id="run-B")
        n = r.rollback_trajectory("run-A")
        assert n == 2
        assert seed in r
        assert c in r
        assert a not in r
        assert b not in r

    def test_rollback_empty_trajectory_returns_zero(self) -> None:
        # named edge case.
        r = BlockRegistry()
        r.write_agent_block("x", trajectory_id="run-A")
        n = r.rollback_trajectory("run-Z")
        assert n == 0
        assert len(r) == 1

    def test_rollback_skips_seed_and_orchestrator(self) -> None:
        r = BlockRegistry()
        seed_id = _inject_seed(r, "seed_name")
        orch = r.write_block(
            "brief",
            semantic_name="brief",
            source="orchestrator",
            evictable=False,
            position=0,
        )
        agent = r.write_agent_block("a", trajectory_id="t")
        n = r.rollback_trajectory("t")
        assert n == 1
        assert seed_id in r
        assert orch in r
        assert agent not in r

    def test_rollback_clears_name_index_for_removed_blocks(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block(
            "x", trajectory_id="t1", semantic_name="my_block"
        )
        r.rollback_trajectory("t1")
        assert "my_block" not in r._name_index
        assert rid not in r

    def test_evictable_false_survives_evict_by_policy_but_rollback_retracts(
        self,
    ) -> None:
        """Pin the policy-vs-author-retraction split.

        ``evictable=False`` agent blocks are the canonical case for
        agent-loop diagnoses: the policy must not reap them
        (third-party eviction) but the trajectory's author
        (``rollback_trajectory``) must still be able to retract
        them. The two paths are deliberately separate.
        """
        r = BlockRegistry()
        rid = r.write_agent_block(
            "diagnosis: always rinse before scrub",
            trajectory_id="run-A",
            semantic_name="rinse_rule",
            evictable=False,
        )

        # Policy run: the block is non-evictable, so the policy
        # filters it out before scoring. evict_by_policy() removes
        # nothing.
        evicted = r.evict_by_policy()
        assert evicted == []
        assert rid in r
        assert r.get_by_id(rid).evictable is False

        # Author retraction: rollback_trajectory bypasses the
        # evictable flag because the agent that wrote the block has
        # always had permission to retract its own writes.
        n = r.rollback_trajectory("run-A")
        assert n == 1
        assert rid not in r
        assert "rinse_rule" not in r._name_index


# --- name index maintenance --------------------------------------------------


class TestNameIndexMaintenance:
    def test_write_populates_index(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1", semantic_name="rule")
        assert r._name_index["rule"] == [rid]

    def test_multiple_writes_with_same_name_accumulate(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block("x", trajectory_id="t1", semantic_name="rule")
        b = r.write_agent_block("y", trajectory_id="t2", semantic_name="rule")
        assert r._name_index["rule"] == [a, b]

    def test_evict_one_of_many_keeps_others_in_index(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block("x", trajectory_id="t1", semantic_name="rule")
        b = r.write_agent_block("y", trajectory_id="t2", semantic_name="rule")
        r.evict(a)
        assert r._name_index["rule"] == [b]


# --- get_by_name ------------------------------------------------------


class TestGetByName:
    def test_unknown_name_returns_empty_list(self) -> None:
        r = BlockRegistry()
        r.write_agent_block("x", trajectory_id="t1", semantic_name="known")
        assert r.get_by_name("missing") == []

    def test_returns_single_match(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1", semantic_name="rule")
        result = r.get_by_name("rule")
        assert len(result) == 1
        assert result[0].runtime_id == rid

    def test_multiple_matches_returned_in_registry_order(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block("x", trajectory_id="t1", semantic_name="rule")
        unrelated = r.write_agent_block(
            "y", trajectory_id="t2", semantic_name="other"
        )
        b = r.write_agent_block("z", trajectory_id="t3", semantic_name="rule")
        # Promote the second-written 'rule' block ahead of the first.
        r.promote(b, to_front=True)
        result = r.get_by_name("rule")
        assert [block.runtime_id for block in result] == [b, a]
        # The unrelated block must not appear.
        assert all(block.runtime_id != unrelated for block in result)

    def test_get_by_name_after_evict_drops_entry(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1", semantic_name="rule")
        r.evict(rid)
        assert r.get_by_name("rule") == []


# --- list_blocks ------------------------------------------------------


class TestListBlocks:
    def test_no_filter_returns_every_block_in_order(self) -> None:
        r = BlockRegistry()
        seed = _inject_seed(r, "s")
        agent = r.write_agent_block("a", trajectory_id="t1")
        orch = r.write_block(
            "o",
            semantic_name="o",
            source="orchestrator",
            evictable=False,
            position=0,
        )
        # Default position=0 puts the orchestrator block at the front.
        result = r.list_blocks()
        assert [b.runtime_id for b in result] == [orch, seed, agent]

    def test_filter_seed_returns_only_seed_blocks(self) -> None:
        r = BlockRegistry()
        seed = _inject_seed(r, "s")
        r.write_agent_block("a", trajectory_id="t1")
        r.write_block(
            "o",
            semantic_name="o",
            source="orchestrator",
            evictable=False,
            position=0,
        )
        result = r.list_blocks(source_filter="seed")
        assert [b.runtime_id for b in result] == [seed]

    def test_filter_agent_returns_only_agent_blocks(self) -> None:
        r = BlockRegistry()
        _inject_seed(r, "s")
        a = r.write_agent_block("a", trajectory_id="t1")
        b = r.write_agent_block("b", trajectory_id="t2")
        r.write_block(
            "o",
            semantic_name="o",
            source="orchestrator",
            evictable=False,
            position=0,
        )
        result = r.list_blocks(source_filter="agent")
        # Registry order is [orch, seed, a, b]; the agent filter keeps a, b.
        assert [bl.runtime_id for bl in result] == [a, b]

    def test_filter_orchestrator_returns_only_orchestrator_blocks(self) -> None:
        r = BlockRegistry()
        _inject_seed(r, "s")
        r.write_agent_block("a", trajectory_id="t1")
        orch = r.write_block(
            "o",
            semantic_name="o",
            source="orchestrator",
            evictable=False,
            position=0,
        )
        result = r.list_blocks(source_filter="orchestrator")
        assert [b.runtime_id for b in result] == [orch]

    def test_filter_no_match_returns_empty_list(self) -> None:
        r = BlockRegistry()
        r.write_agent_block("a", trajectory_id="t1")
        assert r.list_blocks(source_filter="orchestrator") == []

    def test_empty_registry_returns_empty_list(self) -> None:
        r = BlockRegistry()
        assert r.list_blocks() == []
        assert r.list_blocks(source_filter="agent") == []


# --- V2 item 1: tag queries --------------------------------------------------


class TestFindByTag:
    def test_returns_blocks_carrying_the_tag_in_compile_order(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block(
            "a", trajectory_id="t", semantic_name="alpha",
            tags=["isr", "timing"],
        )
        b = r.write_agent_block(
            "b", trajectory_id="t", semantic_name="beta",
            tags=["timing"],
        )
        r.write_agent_block(
            "c", trajectory_id="t", semantic_name="gamma",
            tags=["unrelated"],
        )
        timing = [blk.runtime_id for blk in r.find_by_tag("timing")]
        assert timing == [a, b]

    def test_unknown_tag_returns_empty_list(self) -> None:
        r = BlockRegistry()
        r.write_agent_block(
            "a", trajectory_id="t", tags=["isr"],
        )
        assert r.find_by_tag("does_not_exist") == []

    def test_block_without_tags_is_skipped(self) -> None:
        r = BlockRegistry()
        r.write_agent_block("a", trajectory_id="t")  # no tags
        assert r.find_by_tag("anything") == []

    def test_empty_string_matches_no_blocks(self) -> None:
        # The parser and writers never produce empty-string tags, so
        # find_by_tag("") is equivalent to a query for an absent tag.
        r = BlockRegistry()
        r.write_agent_block("a", trajectory_id="t", tags=["x"])
        assert r.find_by_tag("") == []

    def test_empty_registry_returns_empty_list(self) -> None:
        assert BlockRegistry().find_by_tag("isr") == []


class TestFindByTags:
    def test_match_any_returns_union_in_compile_order(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block(
            "a", trajectory_id="t", semantic_name="alpha",
            tags=["isr"],
        )
        b = r.write_agent_block(
            "b", trajectory_id="t", semantic_name="beta",
            tags=["timing"],
        )
        r.write_agent_block(
            "c", trajectory_id="t", semantic_name="gamma",
            tags=["unrelated"],
        )
        result = [
            blk.runtime_id
            for blk in r.find_by_tags(["isr", "timing"])
        ]
        assert result == [a, b]

    def test_match_all_requires_every_tag(self) -> None:
        r = BlockRegistry()
        a = r.write_agent_block(
            "a", trajectory_id="t", semantic_name="alpha",
            tags=["isr", "timing", "critical"],
        )
        r.write_agent_block(
            "b", trajectory_id="t", semantic_name="beta",
            tags=["isr"],
        )
        result = [
            blk.runtime_id
            for blk in r.find_by_tags(
                ["isr", "timing"], match_all=True,
            )
        ]
        assert result == [a]

    def test_empty_input_list_returns_empty_under_either_mode(
        self,
    ) -> None:
        r = BlockRegistry()
        r.write_agent_block(
            "a", trajectory_id="t", tags=["isr"],
        )
        assert r.find_by_tags([]) == []
        assert r.find_by_tags([], match_all=True) == []

    def test_match_all_with_single_tag_is_equivalent_to_find_by_tag(
        self,
    ) -> None:
        r = BlockRegistry()
        r.write_agent_block(
            "a", trajectory_id="t", semantic_name="alpha",
            tags=["isr"],
        )
        r.write_agent_block(
            "b", trajectory_id="t", semantic_name="beta",
            tags=["timing"],
        )
        single = [blk.runtime_id for blk in r.find_by_tag("isr")]
        multi = [
            blk.runtime_id
            for blk in r.find_by_tags(["isr"], match_all=True)
        ]
        assert single == multi

    def test_no_matches_returns_empty_under_either_mode(self) -> None:
        r = BlockRegistry()
        r.write_agent_block(
            "a", trajectory_id="t", tags=["isr"],
        )
        assert r.find_by_tags(["nope1", "nope2"]) == []
        assert r.find_by_tags(
            ["nope1", "nope2"], match_all=True,
        ) == []


# --- score_block ------------------------------------------------------


class TestScoreBlock:
    def test_identical_unit_vectors_score_one(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        r.get_by_id(rid).embedding = _vec(1.0, 0.0, 0.0)
        score = r.score_block(rid, _vec(1.0, 0.0, 0.0))
        assert score is not None
        assert score == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        r.get_by_id(rid).embedding = _vec(1.0, 0.0, 0.0)
        score = r.score_block(rid, _vec(0.0, 1.0, 0.0))
        assert score is not None
        assert score == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        r.get_by_id(rid).embedding = _vec(1.0, 0.0, 0.0)
        score = r.score_block(rid, _vec(-1.0, 0.0, 0.0))
        assert score is not None
        assert score == pytest.approx(-1.0)

    def test_scale_invariant(self) -> None:
        # Cosine ignores magnitude, only direction matters.
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        r.get_by_id(rid).embedding = _vec(2.0, 0.0, 0.0)
        score = r.score_block(rid, _vec(7.0, 0.0, 0.0))
        assert score == pytest.approx(1.0)

    def test_returns_none_when_block_has_no_embedding(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        # No embedding assigned; default is None.
        assert r.score_block(rid, _vec(1.0, 0.0, 0.0)) is None

    def test_unknown_id_raises(self) -> None:
        r = BlockRegistry()
        with pytest.raises(BlockNotFoundError):
            r.score_block("nonexistent", _vec(1.0, 0.0, 0.0))

    def test_zero_block_embedding_returns_zero_not_nan(self) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("x", trajectory_id="t1")
        r.get_by_id(rid).embedding = _vec(0.0, 0.0, 0.0)
        score = r.score_block(rid, _vec(1.0, 1.0, 1.0))
        assert score == 0.0


# --- find_relevant ----------------------------------------------------


class TestFindRelevant:
    def test_no_embedding_model_raises(self) -> None:
        r = BlockRegistry()  # default embedding_model=None
        rid = r.write_agent_block("x", trajectory_id="t1")
        r.get_by_id(rid).embedding = _vec(1.0, 0.0, 0.0)
        with pytest.raises(RuntimeError):
            r.find_relevant("any task")

    def test_returns_empty_when_no_blocks_have_embedding(self) -> None:
        # Embedding model present, but blocks have no stored embeddings —
        # the Phase 1 default state. Must not raise.
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = BlockRegistry(embedding_model=embedder)
        r.write_agent_block("x", trajectory_id="t1")
        r.write_agent_block("y", trajectory_id="t2")
        assert r.find_relevant("task") == []

    def test_skips_blocks_without_embedding(self) -> None:
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = BlockRegistry(embedding_model=embedder)
        with_embed = r.write_agent_block("a", trajectory_id="t1")
        without_embed = r.write_agent_block("b", trajectory_id="t2")
        r.get_by_id(with_embed).embedding = _vec(1.0, 0.0, 0.0)
        # without_embed.embedding stays None.
        result = r.find_relevant("task")
        assert [block.runtime_id for block, _ in result] == [with_embed]
        assert without_embed not in {block.runtime_id for block, _ in result}

    def test_orders_results_by_score_descending(self) -> None:
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = BlockRegistry(embedding_model=embedder)
        far = r.write_agent_block("a", trajectory_id="t1")
        near = r.write_agent_block("b", trajectory_id="t2")
        mid = r.write_agent_block("c", trajectory_id="t3")
        r.get_by_id(far).embedding = _vec(0.0, 1.0, 0.0)  # cosine 0
        r.get_by_id(near).embedding = _vec(1.0, 0.0, 0.0)  # cosine 1
        r.get_by_id(mid).embedding = _vec(1.0, 1.0, 0.0)  # cosine ~0.707
        result = r.find_relevant("task")
        ordered_ids = [block.runtime_id for block, _ in result]
        assert ordered_ids == [near, mid, far]
        # Scores must be monotonically non-increasing.
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_caps_results(self) -> None:
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = BlockRegistry(embedding_model=embedder)
        for i in range(5):
            rid = r.write_agent_block(f"b{i}", trajectory_id=f"t{i}")
            # Distinct embeddings so all five are scored.
            r.get_by_id(rid).embedding = _vec(float(i + 1), 0.0, 0.0)
        result = r.find_relevant("task", top_k=2)
        assert len(result) == 2

    def test_threshold_filters_low_scores(self) -> None:
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = BlockRegistry(embedding_model=embedder)
        a = r.write_agent_block("a", trajectory_id="t1")
        b = r.write_agent_block("b", trajectory_id="t2")
        r.get_by_id(a).embedding = _vec(1.0, 0.0, 0.0)  # cosine 1
        r.get_by_id(b).embedding = _vec(0.0, 1.0, 0.0)  # cosine 0
        result = r.find_relevant("task", threshold=0.5)
        assert [block.runtime_id for block, _ in result] == [a]

    def test_top_k_and_threshold_combine(self) -> None:
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = BlockRegistry(embedding_model=embedder)
        ids = [r.write_agent_block(f"b{i}", trajectory_id=f"t{i}") for i in range(4)]
        # Decreasing alignment with the task vector.
        r.get_by_id(ids[0]).embedding = _vec(1.0, 0.0, 0.0)  # 1.0
        r.get_by_id(ids[1]).embedding = _vec(2.0, 1.0, 0.0)  # ~0.894
        r.get_by_id(ids[2]).embedding = _vec(1.0, 1.0, 0.0)  # ~0.707
        r.get_by_id(ids[3]).embedding = _vec(1.0, 5.0, 0.0)  # ~0.196
        # Threshold 0.5 keeps three, top_k=2 trims to two.
        result = r.find_relevant("task", top_k=2, threshold=0.5)
        assert [block.runtime_id for block, _ in result] == [ids[0], ids[1]]

    def test_returns_pair_with_block_object(self) -> None:
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = BlockRegistry(embedding_model=embedder)
        rid = r.write_agent_block("a", trajectory_id="t1")
        r.get_by_id(rid).embedding = _vec(1.0, 0.0, 0.0)
        result = r.find_relevant("task")
        assert len(result) == 1
        block, score = result[0]
        assert isinstance(block, InstructionBlock)
        assert block.runtime_id == rid
        assert isinstance(score, float)


# --- BlockRegistry.from_files -----------------------------------------


def _seed_file(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestFromFilesSingleFile:
    def test_single_block_file_loads_one_block(self, tmp_path: Path) -> None:
        path = _seed_file(
            tmp_path,
            "rule.md",
            "---\nname: alpha\n---\nbody\n",
        )
        r = BlockRegistry.from_files([path])
        assert len(r) == 1
        block = next(iter(r.list_blocks()))
        assert block.semantic_name == "alpha"
        assert block.source == "seed"

    def test_assigns_uuid4_runtime_id(self, tmp_path: Path) -> None:
        path = _seed_file(tmp_path, "rule.md", "---\nname: a\n---\nbody\n")
        r = BlockRegistry.from_files([path])
        rid = r.list_blocks()[0].runtime_id
        assert rid != ""
        assert _uuid.UUID(rid).version == 4

    def test_stamps_created_at(self, tmp_path: Path) -> None:
        path = _seed_file(tmp_path, "rule.md", "---\nname: a\n---\nbody\n")
        r = BlockRegistry.from_files([path])
        block = r.list_blocks()[0]
        assert block.created_at > 0

    def test_populates_name_index(self, tmp_path: Path) -> None:
        path = _seed_file(tmp_path, "rule.md", "---\nname: alpha\n---\nbody\n")
        r = BlockRegistry.from_files([path])
        result = r.get_by_name("alpha")
        assert len(result) == 1
        assert result[0].semantic_name == "alpha"

    def test_skill_md_no_frontmatter_loads_as_anonymous_seed_block(
        self,
    ) -> None:
        # Phase 1 go criterion, end-to-end via from_files() rather than the
        # parser alone.
        fixture = (
            Path(__file__).parent / "fixtures" / "skill-no-frontmatter.md"
        )
        r = BlockRegistry.from_files([fixture])
        assert len(r) == 1
        block = r.list_blocks()[0]
        assert block.source == "seed"
        assert block.semantic_name == "skill_no_frontmatter"
        assert _uuid.UUID(block.runtime_id).version == 4
        assert "Database migration helper" in block.content


class TestFromFilesMultipleFiles:
    def test_preserves_file_order_and_within_file_order(
        self, tmp_path: Path
    ) -> None:
        a = _seed_file(
            tmp_path,
            "first.md",
            "---\nname: a1\n---\n1a\n---\nname: a2\n---\n2a\n",
        )
        b = _seed_file(
            tmp_path,
            "second.md",
            "---\nname: b1\n---\n1b\n",
        )
        r = BlockRegistry.from_files([a, b])
        names = [block.semantic_name for block in r.list_blocks()]
        assert names == ["a1", "a2", "b1"]

    def test_assigns_distinct_uuids_for_every_block(self, tmp_path: Path) -> None:
        path = _seed_file(
            tmp_path,
            "many.md",
            "---\nname: a\n---\nx\n---\nname: b\n---\ny\n---\nname: c\n---\nz\n",
        )
        r = BlockRegistry.from_files([path])
        ids = {block.runtime_id for block in r.list_blocks()}
        assert len(ids) == 3


class TestFromFilesAddressRandomisationInvariant:
    """Hard invariant 4: runtime IDs are UUIDs regenerated each session.

    Two cold starts of the same seed file must yield disjoint runtime-id
    sets.
    """

    def test_two_cold_starts_share_no_runtime_ids(self, tmp_path: Path) -> None:
        path = _seed_file(
            tmp_path,
            "rules.md",
            "---\nname: a\n---\nx\n---\nname: b\n---\ny\n---\nname: c\n---\nz\n",
        )
        first = BlockRegistry.from_files([path])
        second = BlockRegistry.from_files([path])
        first_ids = {block.runtime_id for block in first.list_blocks()}
        second_ids = {block.runtime_id for block in second.list_blocks()}
        assert first_ids.isdisjoint(second_ids)
        # Sanity: semantic names persist across cold starts.
        first_names = [block.semantic_name for block in first.list_blocks()]
        second_names = [block.semantic_name for block in second.list_blocks()]
        assert first_names == second_names == ["a", "b", "c"]


class TestFromFilesConfigPlumbing:
    def test_default_max_tokens_propagates_from_config(
        self, tmp_path: Path
    ) -> None:
        from rampart.config import RAMPARTConfig

        path = _seed_file(tmp_path, "x.md", "---\nname: a\n---\nb\n")
        cfg = RAMPARTConfig(default_max_tokens=4096)
        r = BlockRegistry.from_files([path], config=cfg)
        assert r.default_max_tokens == 4096

    def test_memory_limit_mb_propagates_from_config(
        self, tmp_path: Path
    ) -> None:
        from rampart.config import RAMPARTConfig

        path = _seed_file(tmp_path, "x.md", "---\nname: a\n---\nb\n")
        cfg = RAMPARTConfig(memory_limit_mb=42.5)
        r = BlockRegistry.from_files([path], config=cfg)
        assert r.memory_limit_mb == 42.5

    def test_loaded_seeds_carry_evictable_false_and_author_id(
        self, tmp_path: Path
    ) -> None:
        # Seeds become non-evictable on load and the loading registry
        # is recorded as the author of record. Pin both since both
        # gate the evict() permission model downstream.
        path = _seed_file(tmp_path, "x.md", "---\nname: a\n---\nb\n")
        r = BlockRegistry.from_files([path])
        block = r.list_blocks()[0]
        assert block.evictable is False
        assert block.author_id == r.registry_id

    def test_embedding_model_passed_through(self, tmp_path: Path) -> None:
        path = _seed_file(tmp_path, "x.md", "---\nname: a\n---\nb\n")
        embedder = _FakeEmbedder({"q": _vec(1.0, 0.0, 0.0)})
        r = BlockRegistry.from_files([path], embedding_model=embedder)
        # A later refactor wired embed_all into from_files: every block now has an
        # embedding after cold start (zeros for content the fake embedder
        # has no mapping for). find_relevant should return one (block,
        # score) pair, not raise.
        result = r.find_relevant("q")
        assert len(result) == 1
        assert result[0][0].semantic_name == "a"


class TestFromFilesErrorPropagation:
    def test_unsupported_extension_propagates(self, tmp_path: Path) -> None:
        path = _seed_file(tmp_path, "weird.xyz", "anything")
        with pytest.raises(UnsupportedFormatError):
            BlockRegistry.from_files([path])

    def test_empty_paths_yields_empty_registry(self) -> None:
        r = BlockRegistry.from_files([])
        assert len(r) == 0


class TestFromFilesBlockSourceFiltering:
    """``from_files`` accepts ``BlockSource`` descriptors with optional
    name filters. Plain ``Path`` and ``str`` paths still work and are
    coerced to ``BlockSource(path=p, names=None)`` with default
    ``must_match=True``.
    """

    def test_block_source_loads_named_subset(self, tmp_path: Path) -> None:
        from rampart.parser import BlockSource

        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\n---\n1\n"
            "---\nname: b\n---\n2\n"
            "---\nname: c\n---\n3\n",
        )
        r = BlockRegistry.from_files(
            [BlockSource(path=path, names=["a", "c"])]
        )
        assert sorted(b.semantic_name for b in r.list_blocks()) == ["a", "c"]

    def test_block_source_must_match_true_raises_on_missing_name(
        self, tmp_path: Path
    ) -> None:
        from rampart.parser import BlockSource

        path = _seed_file(
            tmp_path, "lib.md", "---\nname: a\n---\n1\n"
        )
        with pytest.raises(BlockNotFoundError, match="missing"):
            BlockRegistry.from_files(
                [BlockSource(path=path, names=["a", "missing"])]
            )

    def test_block_source_must_match_false_silently_skips_missing(
        self, tmp_path: Path
    ) -> None:
        from rampart.parser import BlockSource

        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\n---\n1\n"
            "---\nname: b\n---\n2\n",
        )
        r = BlockRegistry.from_files(
            [BlockSource(path=path, names=["a", "ghost"], must_match=False)]
        )
        names = sorted(b.semantic_name for b in r.list_blocks())
        assert names == ["a"]  # ghost silently dropped

    def test_plain_string_path_still_works(self, tmp_path: Path) -> None:
        # Backwards compatibility: every existing call site that passes
        # a bare Path or str must continue to load every block from
        # the file with default must_match semantics.
        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nbody\n")
        r = BlockRegistry.from_files([str(path)])
        assert len(r) == 1
        assert r.list_blocks()[0].semantic_name == "a"

    def test_mixed_block_source_and_plain_path(self, tmp_path: Path) -> None:
        from rampart.parser import BlockSource

        a = _seed_file(
            tmp_path,
            "a.md",
            "---\nname: a1\n---\n1\n---\nname: a2\n---\n2\n",
        )
        b = _seed_file(
            tmp_path, "b.md", "---\nname: b1\n---\n1\n"
        )
        r = BlockRegistry.from_files([
            BlockSource(path=a, names=["a2"]),
            b,  # plain Path: load everything
        ])
        names = [block.semantic_name for block in r.list_blocks()]
        assert names == ["a2", "b1"]


# --- BlockRegistry.from_seed and add_from_seed -----------------------------


class TestFromSeed:
    """``BlockRegistry.from_seed`` builds a working registry from a
    pre-parsed ``SeedRegistry`` library. Same address-randomisation
    contract as ``from_files``: every block gets a fresh UUID4.
    """

    def test_from_seed_loads_named_subset_in_requested_order(
        self, tmp_path: Path
    ) -> None:
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\n---\nA\n"
            "---\nname: b\n---\nB\n"
            "---\nname: c\n---\nC\n",
        )
        library = SeedRegistry.from_files([path])
        r = BlockRegistry.from_seed(library, names=["c", "a"])
        names = [block.semantic_name for block in r.list_blocks()]
        # Order in the registry follows the requested ``names`` order,
        # not the library insertion order.
        assert names == ["c", "a"]

    def test_from_seed_assigns_fresh_uuids(self, tmp_path: Path) -> None:
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files([path])
        r1 = BlockRegistry.from_seed(library, names=["a"])
        r2 = BlockRegistry.from_seed(library, names=["a"])
        rid1 = r1.list_blocks()[0].runtime_id
        rid2 = r2.list_blocks()[0].runtime_id
        assert rid1 != rid2
        assert _uuid.UUID(rid1).version == 4
        assert _uuid.UUID(rid2).version == 4

    def test_from_seed_preserves_seed_provenance(self, tmp_path: Path) -> None:
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files([path])
        r = BlockRegistry.from_seed(library, names=["a"])
        assert r.list_blocks()[0].source == "seed"

    def test_from_seed_must_match_true_raises_on_missing(
        self, tmp_path: Path
    ) -> None:
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files([path])
        with pytest.raises(BlockNotFoundError, match="ghost"):
            BlockRegistry.from_seed(library, names=["a", "ghost"])

    def test_from_seed_must_match_false_silently_skips(
        self, tmp_path: Path
    ) -> None:
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files([path])
        r = BlockRegistry.from_seed(
            library, names=["a", "ghost"], must_match=False
        )
        names = [block.semantic_name for block in r.list_blocks()]
        assert names == ["a"]

    def test_from_seed_empty_library_with_must_match_false_yields_empty(
        self,
    ) -> None:
        from rampart.seed_registry import SeedRegistry

        library = SeedRegistry()
        r = BlockRegistry.from_seed(
            library, names=["nope"], must_match=False
        )
        assert len(r) == 0

    def test_from_seed_with_embedding_model_embeds_every_block(
        self, tmp_path: Path
    ) -> None:
        # Cold-start embedding behaviour mirrors from_files: pass an
        # embedder, every block gets a vector before the call returns.
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\n---\nA\n---\nname: b\n---\nB\n",
        )
        library = SeedRegistry.from_files([path])

        embedder = _FakeEmbedder({"A": _vec(1.0, 0.0), "B": _vec(0.0, 1.0)})
        r = BlockRegistry.from_seed(
            library,
            names=["a", "b"],
            embedding_model=embedder,
            tokeniser=len,
        )
        for block in r.list_blocks():
            assert block.embedding is not None

    def test_from_seed_accepts_namespaced_keys(
        self, tmp_path: Path
    ) -> None:
        # The namespaced form is the unambiguous one and matches the
        # Python-imports mental model. Same library, same blocks,
        # explicit stem prefix — equivalent to passing bare names
        # when nothing in the library shadows them.
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\n---\nA\n---\nname: b\n---\nB\n",
        )
        library = SeedRegistry.from_files([path])
        r = BlockRegistry.from_seed(library, names=["lib.b", "lib.a"])
        names = [block.semantic_name for block in r.list_blocks()]
        # Order in the registry follows the requested order, with the
        # namespace prefix stripped (the semantic_name on the block
        # itself never includes the stem).
        assert names == ["b", "a"]

    def test_from_seed_ambiguous_bare_name_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        # Two files contribute a block named `coder`. A bare-name
        # request must surface as ValueError, not pick one silently.
        from rampart.seed_registry import SeedRegistry

        a = _seed_file(tmp_path, "skills.md", "---\nname: coder\n---\nA\n")
        b = _seed_file(tmp_path, "tools.md", "---\nname: coder\n---\nB\n")
        library = SeedRegistry.from_files([a, b])
        with pytest.raises(ValueError, match="namespaced"):
            BlockRegistry.from_seed(library, names=["coder"])

    def test_from_seed_ambiguity_raises_even_with_must_match_false(
        self, tmp_path: Path
    ) -> None:
        # Ambiguity is a programmer error, not a missing-data
        # condition. must_match=False suppresses BlockNotFoundError
        # for absent names but does not suppress ValueError for
        # ambiguous names.
        from rampart.seed_registry import SeedRegistry

        a = _seed_file(tmp_path, "skills.md", "---\nname: coder\n---\nA\n")
        b = _seed_file(tmp_path, "tools.md", "---\nname: coder\n---\nB\n")
        library = SeedRegistry.from_files([a, b])
        with pytest.raises(ValueError, match="namespaced"):
            BlockRegistry.from_seed(
                library, names=["coder"], must_match=False
            )

    def test_from_seed_namespaced_form_resolves_when_bare_is_ambiguous(
        self, tmp_path: Path
    ) -> None:
        # The escape hatch the ValueError points at: the namespaced
        # form is always unambiguous because the stem is included.
        from rampart.seed_registry import SeedRegistry

        a = _seed_file(tmp_path, "skills.md", "---\nname: coder\n---\nA\n")
        b = _seed_file(tmp_path, "tools.md", "---\nname: coder\n---\nB\n")
        library = SeedRegistry.from_files([a, b])
        r = BlockRegistry.from_seed(
            library, names=["skills.coder"]
        )
        assert len(r) == 1
        assert r.list_blocks()[0].content == "A"


class TestAddFromSeed:
    """``BlockRegistry.add_from_seed`` adds a single block mid-session.

    Source stays ``"seed"`` so the registry's seed-protection guard
    treats the new block exactly like a cold-start seed: the default
    eviction policy ignores it and ``evict()`` raises against it.
    """

    def test_add_from_seed_appends_block_with_fresh_uuid(
        self, tmp_path: Path
    ) -> None:
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\n---\nA\n---\nname: b\n---\nB\n",
        )
        library = SeedRegistry.from_files([path])
        r = BlockRegistry.from_seed(library, names=["a"])
        rid_added = r.add_from_seed(library, "b")
        names = [block.semantic_name for block in r.list_blocks()]
        assert names == ["a", "b"]
        assert _uuid.UUID(rid_added).version == 4
        assert r.get_by_id(rid_added).source == "seed"

    def test_add_from_seed_unknown_name_raises(
        self, tmp_path: Path
    ) -> None:
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files([path])
        r = BlockRegistry.from_seed(library, names=["a"])
        with pytest.raises(BlockNotFoundError, match="ghost"):
            r.add_from_seed(library, "ghost")

    def test_add_from_seed_block_is_seed_protected(
        self, tmp_path: Path
    ) -> None:
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files([path])
        r = BlockRegistry()
        rid = r.add_from_seed(library, "a")
        with pytest.raises(EvictionError):
            r.evict(rid)

    def test_add_from_seed_with_embedder_embeds_the_new_block(
        self, tmp_path: Path
    ) -> None:
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files([path])
        embedder = _FakeEmbedder({"A": _vec(1.0, 0.0)})
        r = BlockRegistry(embedding_model=embedder, tokeniser=len)
        rid = r.add_from_seed(library, "a")
        assert r.get_by_id(rid).embedding is not None

    def test_add_from_seed_accepts_namespaced_form(
        self, tmp_path: Path
    ) -> None:
        from rampart.seed_registry import SeedRegistry

        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files([path])
        r = BlockRegistry()
        rid = r.add_from_seed(library, "lib.a")
        assert r.get_by_id(rid).semantic_name == "a"

    def test_add_from_seed_ambiguous_bare_name_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        from rampart.seed_registry import SeedRegistry

        a = _seed_file(tmp_path, "skills.md", "---\nname: coder\n---\nA\n")
        b = _seed_file(tmp_path, "tools.md", "---\nname: coder\n---\nB\n")
        library = SeedRegistry.from_files([a, b])
        r = BlockRegistry()
        with pytest.raises(ValueError, match="namespaced"):
            r.add_from_seed(library, "coder")
        # Registry is untouched after the failed lookup.
        assert len(r) == 0


# --- token_count_of and evict_by_policy ------------------------------


class TestTokenCountOf:
    def test_computes_via_tokeniser_on_first_call(self) -> None:
        r = BlockRegistry(tokeniser=lambda s: len(s))
        rid = r.write_agent_block("hello world", trajectory_id="t1")
        block = r.get_by_id(rid)
        assert block.token_count is None
        assert r.token_count_of(block) == 11
        assert block.token_count == 11

    def test_returns_cached_value_on_second_call(self) -> None:
        calls = [0]

        def counter(text: str) -> int:
            calls[0] += 1
            return len(text)

        r = BlockRegistry(tokeniser=counter)
        rid = r.write_agent_block("hello", trajectory_id="t1")
        block = r.get_by_id(rid)
        r.token_count_of(block)
        r.token_count_of(block)
        assert calls[0] == 1

    def test_default_tokeniser_is_config_default(self) -> None:
        # The registry's default tokeniser is the lazy tiktoken cl100k_base
        # callable from rampart.config. Asserting identity (not behaviour)
        # keeps this test independent of cl100k_base's specific counts.
        from rampart.config import default_tokeniser

        r = BlockRegistry()
        assert r._tokeniser is default_tokeniser

    def test_default_tokeniser_returns_positive_count_for_nonempty_text(
        self,
    ) -> None:
        r = BlockRegistry()
        rid = r.write_agent_block("hello world", trajectory_id="t1")
        block = r.get_by_id(rid)
        assert r.token_count_of(block) > 0

    def test_update_block_content_invalidates_cache(self) -> None:
        r = BlockRegistry(tokeniser=lambda s: len(s))
        rid = r.write_agent_block("short", trajectory_id="t1")
        block = r.get_by_id(rid)
        r.token_count_of(block)
        assert block.token_count == 5
        r.update_block_content(rid, "much longer content")
        assert block.token_count is None
        assert r.token_count_of(block) == 19


class TestEvictByPolicyHappyPath:
    def test_empty_registry_yields_no_evictions(self) -> None:
        r = BlockRegistry()
        assert r.evict_by_policy() == []

    def test_evicts_every_evictable_block(self) -> None:
        # Without a token budget, the default policy returns every
        # evictable block sorted by score; evict_by_policy then
        # removes all of them. Non-evictable blocks survive.
        r = BlockRegistry(tokeniser=lambda s: len(s))
        seed = _inject_seed(r, "seed_keep")
        agents = [
            r.write_agent_block(
                "x" * 10, trajectory_id=f"t{i}", priority=i * 0.1
            )
            for i in range(5)
        ]
        evicted = r.evict_by_policy()
        assert set(evicted) == set(agents)
        assert seed in r
        assert all(rid not in r for rid in agents)

    def test_eviction_order_follows_policy_score(self) -> None:
        # Lower-priority block must come first in the evicted list;
        # the policy emits highest-score-first.
        r = BlockRegistry()
        high = r.write_agent_block("h", trajectory_id="t", priority=0.9)
        low = r.write_agent_block("l", trajectory_id="t", priority=0.1)
        evicted = r.evict_by_policy()
        assert evicted[0] == low
        assert evicted[1] == high


class TestEvictByPolicySafetyNet:
    """Under the V2 score-based contract the policy never receives
    non-evictable blocks — the registry filters before scoring. These
    tests pin that contract: a custom policy that *would* protect (or
    fail to protect) a seed block never gets a chance because the
    block is filtered out before ``score()`` is called.
    """

    def test_non_evictable_blocks_are_filtered_before_scoring(
        self,
    ) -> None:
        # A policy that would return float('inf') (always-evict) for
        # every block it sees must still leave non-evictable blocks
        # untouched — the registry never lets the policy see them.
        seed_block = InstructionBlock(
            semantic_name="protected",
            runtime_id=generate_runtime_id(),
            content="critical",
            source="seed",
            evictable=False,
        )
        seen: list[str] = []

        class GreedyPolicy:
            def score(
                self, block: InstructionBlock,
            ) -> float:
                seen.append(block.semantic_name)
                return float("inf")

        r = BlockRegistry(eviction_policy=GreedyPolicy())
        seed_block.author_id = r.registry_id
        r._blocks[seed_block.runtime_id] = seed_block
        r._index_add(seed_block.semantic_name, seed_block.runtime_id)
        # An evictable block to confirm the policy IS being called for
        # legitimate candidates — the test is meaningful only if score
        # ran at least once.
        evictable_rid = r.write_agent_block(
            "a", trajectory_id="t", semantic_name="evictable_one",
        )

        evicted = r.evict_by_policy()
        assert evicted == [evictable_rid]
        assert seen == ["evictable_one"]
        assert "protected" not in seen
        assert seed_block.runtime_id in r

    def test_score_signature_matches_protocol(self) -> None:
        # A policy that only implements score() (no select()) plugs
        # in cleanly via evict_by_policy. This is the canonical
        # "minimal custom policy" path documented on the Protocol.
        class ScoreOnlyPolicy:
            def score(
                self, block: InstructionBlock,
            ) -> float:
                return 1.0 if "evict_me" in block.tags else 0.0

        r = BlockRegistry(eviction_policy=ScoreOnlyPolicy())
        a = r.write_agent_block(
            "a", trajectory_id="t", tags=["evict_me"],
        )
        b = r.write_agent_block("b", trajectory_id="t")
        evicted = r.evict_by_policy()
        # Both are evictable so both get removed; only the order
        # depends on score(). The "evict_me"-tagged block scores
        # higher and so is removed first.
        assert evicted == [a, b]


class TestEvictByPolicyConfig:
    def test_default_eviction_policy_used_when_none_passed(self) -> None:
        from rampart.eviction import DefaultEvictionPolicy

        r = BlockRegistry()
        assert isinstance(r._eviction_policy, DefaultEvictionPolicy)

    def test_custom_policy_used_when_passed(self) -> None:
        class NoopPolicy:
            def score(
                self, block: InstructionBlock,  # noqa: ARG002
            ) -> float:
                return 0.0

        policy = NoopPolicy()
        r = BlockRegistry(eviction_policy=policy)
        assert r._eviction_policy is policy
        # Empty registry: nothing to score, nothing to evict.
        assert r.evict_by_policy() == []


class TestRegistrySummary:
    def test_empty_registry_returns_zeroed_summary(self) -> None:
        r = BlockRegistry()
        summary = r.registry_summary()
        assert summary == {
            "block_count": 0,
            "by_source": {},
            "total_tokens": 0,
            "avg_priority": 0.0,
            "total_access_count": 0,
            "first_block_runtime_id": None,
            "by_tag": {},
        }

    def test_single_seed_block_summary(self) -> None:
        r = BlockRegistry(tokeniser=len)
        rid = _inject_seed(r, "only", content="abcdef")
        r._blocks[rid].priority = 0.7
        r._blocks[rid].access_count = 4
        summary = r.registry_summary()
        assert summary["block_count"] == 1
        assert summary["by_source"] == {"seed": 1}
        assert summary["total_tokens"] == 6
        assert summary["avg_priority"] == 0.7
        assert summary["total_access_count"] == 4
        assert summary["first_block_runtime_id"] == rid

    def test_mixed_sources_aggregated_correctly(self) -> None:
        r = BlockRegistry(tokeniser=len)
        seed_rid = _inject_seed(r, "seed_one", content="seedseed")  # 8 tokens
        r._blocks[seed_rid].priority = 0.9
        r._blocks[seed_rid].access_count = 2
        agent_rid = r.write_agent_block(
            content="ag",  # 2 tokens
            trajectory_id="t1",
            priority=0.4,
        )
        r._blocks[agent_rid].access_count = 5
        orch_rid = r.write_block(
            content="orch",  # 4 tokens
            semantic_name="orch",
            source="orchestrator",
            evictable=False,
            priority=0.5,
            position=2,
        )
        r._blocks[orch_rid].access_count = 1
        summary = r.registry_summary()
        assert summary["block_count"] == 3
        assert summary["by_source"] == {"seed": 1, "agent": 1, "orchestrator": 1}
        assert summary["total_tokens"] == 14
        assert summary["avg_priority"] == 0.6  # (0.9 + 0.4 + 0.5) / 3
        assert summary["total_access_count"] == 8
        assert summary["first_block_runtime_id"] == seed_rid

    def test_avg_priority_is_rounded_to_three_decimals(self) -> None:
        r = BlockRegistry(tokeniser=len)
        for i in range(3):
            rid = _inject_seed(r, f"b{i}", content="x")
            r._blocks[rid].priority = 0.123456789 if i == 0 else 0.5
        summary = r.registry_summary()
        # Mean of (0.123456789, 0.5, 0.5) = 0.374485596...
        # Rounded to 3 decimals is 0.374.
        assert summary["avg_priority"] == 0.374

    def test_first_block_runtime_id_tracks_position_zero_after_promote(
        self,
    ) -> None:
        r = BlockRegistry(tokeniser=len)
        rid_a = _inject_seed(r, "first")
        rid_b = _inject_seed(r, "second")
        assert r.registry_summary()["first_block_runtime_id"] == rid_a
        # Use the public reorder API rather than reaching into
        # internals; promote() refuses to operate on seed blocks
        # via the public surface, so reorder is the right tool here.
        r.reorder([rid_b, rid_a])
        assert r.registry_summary()["first_block_runtime_id"] == rid_b

    def test_by_tag_counts_each_tag_per_block(self) -> None:
        r = BlockRegistry(tokeniser=len)
        r.write_agent_block(
            "a", trajectory_id="t", tags=["isr", "timing"],
        )
        r.write_agent_block(
            "b", trajectory_id="t", tags=["isr"],
        )
        r.write_agent_block("c", trajectory_id="t")  # no tags
        summary = r.registry_summary()
        assert summary["by_tag"] == {"isr": 2, "timing": 1}

    def test_by_tag_is_empty_when_no_block_has_tags(self) -> None:
        r = BlockRegistry(tokeniser=len)
        r.write_agent_block("a", trajectory_id="t")
        r.write_agent_block("b", trajectory_id="t")
        assert r.registry_summary()["by_tag"] == {}


class TestProvenanceReport:
    def test_empty_registry_returns_empty_report(self) -> None:
        r = BlockRegistry()
        report = r.provenance_report()
        assert report == {
            "total_blocks": 0,
            "by_source": {},
            "trajectories": {},
        }

    def test_seeds_only_have_no_trajectories(self) -> None:
        r = BlockRegistry()
        _inject_seed(r, "a")
        _inject_seed(r, "b")
        report = r.provenance_report()
        assert report["total_blocks"] == 2
        assert report["by_source"] == {"seed": 2}
        assert report["trajectories"] == {}

    def test_mixed_sources_grouped_by_trajectory(self) -> None:
        r = BlockRegistry()
        _inject_seed(r, "seed_a")
        r.write_agent_block(content="x", semantic_name="a1", trajectory_id="t1")
        r.write_agent_block(content="y", semantic_name="a2", trajectory_id="t1")
        r.write_agent_block(content="z", semantic_name="a3", trajectory_id="t2")
        report = r.provenance_report()
        assert report["total_blocks"] == 4
        assert report["by_source"] == {"seed": 1, "agent": 3}
        assert report["trajectories"] == {
            "t1": ["a1", "a2"],
            "t2": ["a3"],
        }

    def test_trajectory_filter_matching_returns_only_subset(self) -> None:
        r = BlockRegistry()
        _inject_seed(r, "seed_a")
        r.write_agent_block(content="x", semantic_name="a1", trajectory_id="t1")
        r.write_agent_block(content="y", semantic_name="a2", trajectory_id="t1")
        r.write_agent_block(content="z", semantic_name="a3", trajectory_id="t2")
        report = r.provenance_report(trajectory_id="t1")
        assert report["total_blocks"] == 2
        assert report["by_source"] == {"agent": 2}
        assert report["trajectories"] == {"t1": ["a1", "a2"]}

    def test_trajectory_filter_no_match_returns_empty_report(self) -> None:
        r = BlockRegistry()
        _inject_seed(r, "seed_a")
        r.write_agent_block(content="x", semantic_name="a1", trajectory_id="t1")
        report = r.provenance_report(trajectory_id="never_existed")
        assert report == {
            "total_blocks": 0,
            "by_source": {},
            "trajectories": {},
        }

    def test_orchestrator_block_appears_in_trajectories_if_tagged(self) -> None:
        r = BlockRegistry()
        _inject_seed(r, "seed_a")
        # write_block accepts trajectory_id but the typical orchestrator
        # call site omits it; tag it manually via the dataclass attribute
        # to verify provenance_report treats orchestrator blocks the same
        # as agent blocks once tagged.
        rid = r.write_block(
            content="orch_text",
            semantic_name="orch1",
            source="orchestrator",
            evictable=False,
            position=1,
        )
        r._blocks[rid].trajectory_id = "orchestrator_session_42"
        report = r.provenance_report(trajectory_id="orchestrator_session_42")
        assert report["total_blocks"] == 1
        assert report["by_source"] == {"orchestrator": 1}
        assert report["trajectories"] == {
            "orchestrator_session_42": ["orch1"],
        }

    def test_agent_block_with_no_trajectory_is_counted_but_not_grouped(
        self,
    ) -> None:
        # write_agent_block requires trajectory_id, so we synthesise a
        # block directly to exercise the "trajectory_id is None" branch.
        r = BlockRegistry()
        rid = generate_runtime_id()
        r._blocks[rid] = InstructionBlock(
            semantic_name="orphan",
            runtime_id=rid,
            content="x",
            source="agent",
            trajectory_id=None,
        )
        r._index_add("orphan", rid)
        report = r.provenance_report()
        assert report["total_blocks"] == 1
        assert report["by_source"] == {"agent": 1}
        assert report["trajectories"] == {}


# --- global label registry ---------------------------------------------------


class TestLabelRegistry:
    """``BlockRegistry`` claims a unique human-readable label at
    construction time. ``get_by_label()`` is the cross-process
    lookup; ``release()`` frees a label for reuse.
    """

    def test_auto_label_increments(self) -> None:
        from rampart.registry import (
            _GLOBAL_REGISTRY_LABELS,
            _LABEL_COUNTER,
        )

        # Snapshot the current state so the assertions are
        # independent of how many other tests have run before this
        # one (the counter is module-scoped).
        baseline = _LABEL_COUNTER  # noqa: F841 — only the relative gap matters
        a = BlockRegistry()
        b = BlockRegistry()
        try:
            assert a.label != b.label
            assert a.label.startswith("registry-")
            assert b.label.startswith("registry-")
            # Both labels are claimed in the global dict.
            assert _GLOBAL_REGISTRY_LABELS[a.label] is a
            assert _GLOBAL_REGISTRY_LABELS[b.label] is b
        finally:
            a.release()
            b.release()

    def test_explicit_label_claimed(self) -> None:
        from rampart.registry import _GLOBAL_REGISTRY_LABELS, get_by_label

        r = BlockRegistry(label="explicit-name")
        try:
            assert r.label == "explicit-name"
            assert _GLOBAL_REGISTRY_LABELS["explicit-name"] is r
            assert get_by_label("explicit-name") is r
        finally:
            r.release()

    def test_duplicate_label_raises_value_error_with_existing_uuid(
        self,
    ) -> None:
        first = BlockRegistry(label="dup")
        try:
            with pytest.raises(ValueError) as info:
                BlockRegistry(label="dup")
            assert first.registry_id in str(info.value)
        finally:
            first.release()

    def test_release_then_reuse(self) -> None:
        first = BlockRegistry(label="reusable")
        first.release()
        second = BlockRegistry(label="reusable")
        try:
            assert second.label == "reusable"
            assert second is not first
        finally:
            second.release()

    def test_release_is_idempotent(self) -> None:
        r = BlockRegistry(label="idempotent")
        r.release()
        r.release()  # second call is a no-op
        # Subsequent reuse is fine.
        BlockRegistry(label="idempotent").release()

    def test_get_by_label_miss_raises_key_error(self) -> None:
        from rampart.registry import get_by_label

        with pytest.raises(KeyError, match="never claimed"):
            get_by_label("nobody-here")

    def test_auto_label_skips_already_claimed_counter_value(self) -> None:
        # Pre-claim "registry-99999" so the auto-label loop has to
        # advance past it. Pins the while-loop branch in __init__
        # that exists specifically to handle this collision.
        from rampart.registry import _LABEL_COUNTER

        squatter = BlockRegistry(label=f"registry-{_LABEL_COUNTER + 1}")
        try:
            other = BlockRegistry()  # auto-label
            try:
                assert other.label != squatter.label
                assert other.label.startswith("registry-")
            finally:
                other.release()
        finally:
            squatter.release()


# --- evict(force=True) -------------------------------------------------------


class TestEvictForce:
    def test_force_evicts_non_evictable_block_authored_by_self(
        self,
    ) -> None:
        r = BlockRegistry()
        rid = r.write_block(
            "x", semantic_name="x", evictable=False
        )
        r.evict(rid, force=True)
        assert rid not in r

    def test_force_against_foreign_authored_block_raises(self) -> None:
        a = BlockRegistry()
        b = BlockRegistry()
        try:
            rid = a.write_block(
                "x", semantic_name="x", evictable=False
            )
            block = a._blocks[rid]
            # Smuggle the same block into b's index so b can attempt
            # the force-evict against a foreign-authored block.
            b._blocks[rid] = block
            b._index_add(block.semantic_name, rid)
            with pytest.raises(PermissionError):
                b.evict(rid, force=True)
            # The block must remain present in b after the failed
            # force-evict.
            assert rid in b
        finally:
            a.release()
            b.release()

    def test_force_false_against_non_evictable_raises_eviction_error(
        self,
    ) -> None:
        r = BlockRegistry()
        rid = r.write_block(
            "x", semantic_name="x", evictable=False
        )
        with pytest.raises(EvictionError, match="non-evictable"):
            r.evict(rid)


# --- memory_limit_mb soft warning -------------------------------------------


class TestMemoryLimitWarning:
    def test_below_limit_does_not_warn(self) -> None:
        import warnings

        r = BlockRegistry(memory_limit_mb=100.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r.write_agent_block("small", trajectory_id="t")
        assert not any(
            issubclass(w.category, RuntimeWarning) for w in caught
        )

    def test_crossing_limit_emits_runtime_warning(self) -> None:
        import warnings

        # Tiny limit (1 KB) trips on the first non-trivial write
        # because a block + 256 byte overhead exceeds 1024 bytes.
        r = BlockRegistry(memory_limit_mb=0.001)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r.write_agent_block(
                "x" * 2000, trajectory_id="t"
            )
        msgs = [
            str(w.message)
            for w in caught
            if issubclass(w.category, RuntimeWarning)
        ]
        assert any("memory_limit_mb" in m for m in msgs)

    def test_infinite_limit_disables_warning(self) -> None:
        import warnings

        r = BlockRegistry(memory_limit_mb=float("inf"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r.write_agent_block(
                "x" * 100_000, trajectory_id="t"
            )
        assert not any(
            issubclass(w.category, RuntimeWarning) for w in caught
        )

    def test_memory_bytes_estimate_grows_with_writes(self) -> None:
        r = BlockRegistry(memory_limit_mb=float("inf"))
        before = r.memory_bytes()
        for i in range(5):
            r.write_agent_block("x" * 200, trajectory_id=f"t{i}")
        after = r.memory_bytes()
        assert after > before


# --- compile() default_max_tokens fallback ----------------------------------


class TestCompileDefaultMaxTokens:
    def test_compile_without_max_tokens_uses_registry_default(self) -> None:
        r = BlockRegistry(tokeniser=len, default_max_tokens=4)
        r.write_agent_block("AAAA", trajectory_id="t1")
        r.write_agent_block("BBBB", trajectory_id="t2")
        # default_max_tokens=4 fits one block but not two.
        result = r.compile()
        assert result.included == ["AAAA"][:1] or len(result.included) == 1
        # Pin via prompt content too; semantic name was auto-generated.
        assert result.prompt == "AAAA"

    def test_compile_dry_run_without_max_tokens_uses_registry_default(
        self,
    ) -> None:
        r = BlockRegistry(tokeniser=len, default_max_tokens=4)
        r.write_agent_block("AAAA", trajectory_id="t1")
        r.write_agent_block("BBBB", trajectory_id="t2")
        result = r.compile_dry_run()
        assert len(result.included) == 1
