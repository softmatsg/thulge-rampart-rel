"""Tests for InstructionBlock contract and security.generate_runtime_id.

Covers the four block-level invariants directly: round-trip serialisation that
omits embedding/kv_cache, seed and orchestrator content immutability, the
token-count cache invalidation on agent edit, and UUID4 uniqueness across a
large batch of generations.
"""

from __future__ import annotations

import uuid
from typing import cast

import numpy as np
import pytest

from rampart.block import (
    BlockSourceKind,
    InstructionBlock,
    OrchestratorMutationError,
    SeedMutationError,
)
from rampart.security import generate_runtime_id


def _make_block(source: BlockSourceKind = "agent") -> InstructionBlock:
    """Build a populated InstructionBlock for use in tests."""
    return InstructionBlock(
        semantic_name="timing_constraints",
        runtime_id=generate_runtime_id(),
        content="Always check timer overflow before GPIO write.",
        source=source,
        author_id="author-uuid",
        evictable=source == "agent",
        priority=0.7,
        trajectory_id="traj-1" if source == "agent" else None,
        created_at=1_700_000_000,
        access_count=3,
        tags=["hardware", "timing"],
        token_count=12,
        wrap_prefix="<<<",
        wrap_suffix=">>>",
    )


class TestSerialisationRoundTrip:
    """to_dict / from_dict must preserve every primitive field."""

    def test_round_trip_preserves_primitive_fields(self) -> None:
        block = _make_block(source="agent")
        restored = InstructionBlock.from_dict(block.to_dict())

        assert restored.semantic_name == block.semantic_name
        assert restored.runtime_id == block.runtime_id
        assert restored.content == block.content
        assert restored.source == block.source
        assert restored.author_id == block.author_id
        assert restored.evictable == block.evictable
        assert restored.priority == block.priority
        assert restored.trajectory_id == block.trajectory_id
        assert restored.created_at == block.created_at
        assert restored.access_count == block.access_count
        assert restored.tags == block.tags
        assert restored.token_count == block.token_count
        assert restored.wrap_prefix == block.wrap_prefix
        assert restored.wrap_suffix == block.wrap_suffix

    def test_to_dict_omits_embedding_and_kv_cache(self) -> None:
        block = _make_block()
        block.embedding = np.zeros(384, dtype=np.float32)
        d = block.to_dict()

        assert "embedding" not in d
        assert "kv_cache" not in d

    def test_from_dict_resets_embedding_and_kv_cache_to_none(self) -> None:
        block = _make_block()
        block.embedding = np.zeros(384, dtype=np.float32)
        block.kv_cache = object()

        restored = InstructionBlock.from_dict(block.to_dict())

        assert restored.embedding is None
        assert restored.kv_cache is None

    def test_round_trip_with_seed_source(self) -> None:
        block = _make_block(source="seed")
        restored = InstructionBlock.from_dict(block.to_dict())
        assert restored.source == "seed"

    def test_round_trip_with_orchestrator_source(self) -> None:
        block = _make_block(source="orchestrator")
        restored = InstructionBlock.from_dict(block.to_dict())
        assert restored.source == "orchestrator"

    def test_from_dict_falls_back_to_defaults_for_optional_fields(self) -> None:
        minimal = {
            "semantic_name": "x",
            "runtime_id": generate_runtime_id(),
            "content": "y",
            "source": cast(BlockSourceKind, "agent"),
        }
        restored = InstructionBlock.from_dict(minimal)

        assert restored.author_id == ""
        assert restored.evictable is True
        assert restored.priority == 0.5
        assert restored.trajectory_id is None
        assert restored.created_at == 0
        assert restored.access_count == 0
        assert restored.tags == []
        assert restored.token_count is None
        assert restored.wrap_prefix is None
        assert restored.wrap_suffix is None

    def test_from_dict_missing_required_field_raises(self) -> None:
        partial = {
            "semantic_name": "x",
            "content": "y",
            "source": cast(BlockSourceKind, "agent"),
        }
        with pytest.raises(KeyError):
            InstructionBlock.from_dict(partial)

    def test_tags_round_trip_through_dict_and_are_not_aliased(self) -> None:
        original = InstructionBlock(
            semantic_name="x",
            runtime_id="rid",
            content="c",
            source=cast(BlockSourceKind, "agent"),
            tags=["safety_critical", "isr"],
        )
        d = original.to_dict()
        # The dict representation must hold an independent list, so
        # mutating the source block after serialisation cannot leak
        # into a previously-snapshotted dict.
        original.tags.append("mutated_after_snapshot")
        assert d["tags"] == ["safety_critical", "isr"]
        restored = InstructionBlock.from_dict(d)
        assert restored.tags == ["safety_critical", "isr"]
        # And from_dict must defensively copy too — mutating the dict
        # after restore must not perturb the restored block.
        d["tags"].append("mutated_after_restore")
        assert restored.tags == ["safety_critical", "isr"]


class TestSeedImmutability:
    """Seed and orchestrator blocks must reject content mutation."""

    def test_update_content_on_seed_raises(self) -> None:
        block = _make_block(source="seed")
        with pytest.raises(SeedMutationError):
            block.update_content("new text")

    def test_seed_content_unchanged_after_failed_update(self) -> None:
        block = _make_block(source="seed")
        original = block.content
        with pytest.raises(SeedMutationError):
            block.update_content("new text")
        assert block.content == original

    def test_update_content_on_orchestrator_raises(self) -> None:
        block = _make_block(source="orchestrator")
        with pytest.raises(OrchestratorMutationError):
            block.update_content("new text")

    def test_orchestrator_content_unchanged_after_failed_update(self) -> None:
        block = _make_block(source="orchestrator")
        original = block.content
        with pytest.raises(OrchestratorMutationError):
            block.update_content("new text")
        assert block.content == original

    def test_update_content_on_agent_succeeds(self) -> None:
        block = _make_block(source="agent")
        block.update_content("revised heuristic")
        assert block.content == "revised heuristic"

    def test_update_content_on_agent_invalidates_token_count_cache(self) -> None:
        block = _make_block(source="agent")
        assert block.token_count == 12
        block.update_content("revised heuristic")
        assert block.token_count is None


class TestRuntimeIdGeneration:
    """Address randomisation: every runtime_id is a fresh UUID4."""

    def test_runtime_id_is_string(self) -> None:
        rid = generate_runtime_id()
        assert isinstance(rid, str)

    def test_runtime_id_is_uuid4(self) -> None:
        rid = generate_runtime_id()
        parsed = uuid.UUID(rid)
        assert parsed.version == 4

    def test_runtime_ids_unique_across_10000_calls(self) -> None:
        ids = {generate_runtime_id() for _ in range(10_000)}
        assert len(ids) == 10_000


class TestDataclassDefaults:
    """A minimally-constructed block must have spec-correct defaults."""

    def test_minimal_block_has_default_field_values(self) -> None:
        block = InstructionBlock(
            semantic_name="x",
            runtime_id=generate_runtime_id(),
            content="y",
            source="agent",
        )
        assert block.priority == 0.5
        assert block.trajectory_id is None
        assert block.created_at == 0
        assert block.access_count == 0
        assert block.embedding is None
        assert block.kv_cache is None
        assert block.tags == []
        assert block.token_count is None
        assert block.wrap_prefix is None
        assert block.wrap_suffix is None

    def test_default_tags_list_is_not_shared_between_instances(self) -> None:
        a = InstructionBlock(
            semantic_name="a",
            runtime_id=generate_runtime_id(),
            content="x",
            source="agent",
        )
        b = InstructionBlock(
            semantic_name="b",
            runtime_id=generate_runtime_id(),
            content="y",
            source="agent",
        )
        a.tags.append("hardware")
        assert b.tags == []
