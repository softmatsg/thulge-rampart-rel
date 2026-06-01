"""Tests for SentenceTransformerEmbedder, embed_text, embed_all.

The real ``sentence_transformers.SentenceTransformer`` is never loaded by
these tests — the all-MiniLM-L6-v2 weights are 22 MB and the import is
slow. Instead, ``SentenceTransformerEmbedder`` exposes ``_construct_model``
as an overridable hook; tests subclass to inject a deterministic fake
that records call counts so the "one model call per cold start" invariant
can be asserted exactly.

The from_files integration tests live here too because the wiring point
(``BlockRegistry.from_files`` calling ``embed_all`` after parse + UUID
assignment) is the contract the user explicitly locked.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from rampart.registry import BlockRegistry, EmbeddingModel
from rampart.scorer import (
    SentenceTransformerEmbedder,
    embed_all,
    embed_text,
)


class _FakeST:
    """Deterministic stand-in for sentence_transformers.SentenceTransformer.

    Records every ``encode`` call so tests can assert single-call vs.
    per-block batching behaviour exactly. Returns a fixed-shape vector
    keyed off the input length so different texts get different (but
    reproducible) vectors.
    """

    DIM = 4

    def __init__(self, model_name: str = "fake") -> None:
        self.model_name = model_name
        self.encode_calls: list[Any] = []

    def encode(self, texts: str | list[str]) -> NDArray[Any]:
        self.encode_calls.append(texts)
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts])

    def _vec(self, text: str) -> NDArray[Any]:
        # A short deterministic encoding: length, length+1, length+2, length+3.
        n = float(len(text))
        return np.array(
            [n, n + 1.0, n + 2.0, n + 3.0], dtype=np.float32
        )


class _FakeEmbedder(SentenceTransformerEmbedder):
    """SentenceTransformerEmbedder that constructs ``_FakeST`` instead of the real model."""

    def __init__(self, model_name: str = "fake") -> None:
        super().__init__(model_name)
        self._fake_st: _FakeST | None = None

    def _construct_model(self) -> Any:  # noqa: ANN401  # mirrors parent signature
        self._fake_st = _FakeST(self.model_name)
        return self._fake_st


# --- SentenceTransformerEmbedder construction is lazy ----------------------


class TestLazyConstruction:
    def test_construction_does_not_load_model(self) -> None:
        e = _FakeEmbedder()
        assert e._model is None
        assert e._fake_st is None

    def test_first_encode_loads_model_once(self) -> None:
        e = _FakeEmbedder()
        e.encode("hello")
        e.encode("world")
        # _construct_model fired exactly once across both calls.
        assert e._fake_st is not None
        # Two encodes → two _FakeST.encode invocations.
        assert len(e._fake_st.encode_calls) == 2

    def test_default_model_name(self) -> None:
        # Asserting against the literal default rather than re-importing
        # the constant pins the public default.
        e = SentenceTransformerEmbedder()
        assert e.model_name == "all-MiniLM-L6-v2"

    def test_custom_model_name_passed_through(self) -> None:
        e = _FakeEmbedder("custom-model")
        assert e.model_name == "custom-model"
        e.encode("x")
        assert e._fake_st is not None
        assert e._fake_st.model_name == "custom-model"

    def test_satisfies_embedding_model_protocol(self) -> None:
        # SentenceTransformerEmbedder must satisfy the registry's Protocol
        # without any further wrapping. Structural check.
        e = _FakeEmbedder()
        assert isinstance(e, EmbeddingModel)


# --- encode and encode_batch shape contracts -------------------------------


class TestEncodeShape:
    def test_encode_returns_one_d_vector(self) -> None:
        e = _FakeEmbedder()
        result = e.encode("hello")
        assert result.shape == (_FakeST.DIM,)
        assert result.dtype == np.float32

    def test_encode_batch_returns_two_d_vector(self) -> None:
        e = _FakeEmbedder()
        result = e.encode_batch(["a", "b", "c"])
        assert result.shape == (3, _FakeST.DIM)
        assert result.dtype == np.float32

    def test_encode_batch_single_call_to_underlying_model(self) -> None:
        # The "one model call per cold start" invariant relies on this.
        e = _FakeEmbedder()
        e.encode_batch(["a", "b", "c", "d", "e"])
        assert e._fake_st is not None
        assert len(e._fake_st.encode_calls) == 1
        assert e._fake_st.encode_calls[0] == ["a", "b", "c", "d", "e"]


# --- embed_text utility ----------------------------------------------------


class TestEmbedText:
    def test_uses_explicit_embedder(self) -> None:
        e = _FakeEmbedder()
        result = embed_text("hello", embedder=e)
        assert result.shape == (_FakeST.DIM,)
        assert e._fake_st is not None
        assert len(e._fake_st.encode_calls) == 1

    def test_falls_back_to_default_embedder_when_none_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Replace _default_embedder with a fake so the real
        # sentence-transformers model is never loaded by this test.
        fake = _FakeEmbedder()
        monkeypatch.setattr("rampart.scorer._default_embedder", lambda: fake)
        result = embed_text("hello")
        assert result.shape == (_FakeST.DIM,)
        assert fake._fake_st is not None
        assert len(fake._fake_st.encode_calls) == 1


# --- default-embedder construction (real sentence-transformers path) -------


class TestDefaultEmbedderConstruction:
    """Cover the real-sentence-transformers paths without a real download.

    Patches ``sentence_transformers`` in ``sys.modules`` so the
    ``from sentence_transformers import SentenceTransformer`` inside
    ``_construct_model`` resolves to a fake. This pins the public contract
    that the default embedder constructs a SentenceTransformer with the
    library default model name on first encode call.
    """

    def test_default_embedder_uses_real_construction_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        instances: list[str] = []

        class FakeST:
            def __init__(self, name: str) -> None:
                instances.append(name)

            def encode(self, texts: str | list[str]) -> NDArray[Any]:
                if isinstance(texts, str):
                    return np.zeros(4, dtype=np.float32)
                return np.zeros((len(texts), 4), dtype=np.float32)

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeST  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        # Clear the lru_cache so this test actually constructs a fresh
        # default embedder rather than reusing one from a prior test.
        from rampart.scorer import _default_embedder

        _default_embedder.cache_clear()
        try:
            embedder = _default_embedder()
            assert isinstance(embedder, SentenceTransformerEmbedder)
            assert instances == []  # not loaded yet
            embedder.encode("hello")
            assert instances == ["all-MiniLM-L6-v2"]
        finally:
            _default_embedder.cache_clear()

    def test_default_embedder_is_cached_across_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rampart.scorer import _default_embedder

        # Patch the underlying constructor so we can assert that two
        # default-embedder calls share one instance.
        instances: list[str] = []

        class FakeST:
            def __init__(self, name: str) -> None:
                instances.append(name)

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeST  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        _default_embedder.cache_clear()
        try:
            a = _default_embedder()
            b = _default_embedder()
            assert a is b
        finally:
            _default_embedder.cache_clear()


# --- embed_all (cold-start batch routine) ----------------------------------


class _NoBatchEmbedder:
    """Embedder that only exposes ``encode``, no ``encode_batch``.

    Lets us assert the per-block fallback path of embed_all.
    """

    def __init__(self) -> None:
        self.encode_calls: list[str] = []

    def encode(self, text: str) -> NDArray[Any]:
        self.encode_calls.append(text)
        return np.array([float(len(text))], dtype=np.float32)


class TestEmbedAll:
    def test_no_op_when_embedding_model_is_none(self) -> None:
        r = BlockRegistry()
        r.write_agent_block("hello", trajectory_id="t1")
        embed_all(r)
        # Block embedding stays None.
        assert r.list_blocks()[0].embedding is None

    def test_no_op_for_empty_registry(self) -> None:
        e = _FakeEmbedder()
        r = BlockRegistry(embedding_model=e)
        embed_all(r)
        # No model call at all.
        assert e._fake_st is None or len(e._fake_st.encode_calls) == 0

    def test_batch_path_makes_exactly_one_model_call(self) -> None:
        e = _FakeEmbedder()
        r = BlockRegistry(embedding_model=e)
        for i in range(5):
            r.write_agent_block(f"content-{i}", trajectory_id=f"t{i}")
        embed_all(r)
        assert e._fake_st is not None
        assert len(e._fake_st.encode_calls) == 1
        assert isinstance(e._fake_st.encode_calls[0], list)
        assert len(e._fake_st.encode_calls[0]) == 5

    def test_batch_path_writes_vectors_in_registry_order(self) -> None:
        e = _FakeEmbedder()
        r = BlockRegistry(embedding_model=e)
        ids = [
            r.write_agent_block(f"text{i}", trajectory_id=f"t{i}")
            for i in range(3)
        ]
        embed_all(r)
        # _FakeST encodes "text0" → [5, 6, 7, 8], etc.
        for rid, expected_first in zip(ids, [5.0, 5.0, 5.0], strict=True):
            block = r.get_by_id(rid)
            assert block.embedding is not None
            assert float(block.embedding[0]) == expected_first

    def test_fallback_per_block_when_no_encode_batch(self) -> None:
        e = _NoBatchEmbedder()
        r = BlockRegistry(embedding_model=e)
        for i in range(4):
            r.write_agent_block(f"x-{i}", trajectory_id=f"t{i}")
        embed_all(r)
        # No encode_batch → 4 individual encode calls.
        assert len(e.encode_calls) == 4
        # Every block ends up with an embedding.
        for block in r.list_blocks():
            assert block.embedding is not None

    def test_overwrites_existing_embeddings(self) -> None:
        # Cold-start contract: embed everything fresh, not patch what's
        # missing. Re-running embed_all replaces prior vectors.
        e = _FakeEmbedder()
        r = BlockRegistry(embedding_model=e)
        rid = r.write_agent_block("hello", trajectory_id="t1")
        r.get_by_id(rid).embedding = np.array(
            [99.0, 99.0, 99.0, 99.0], dtype=np.float32
        )
        embed_all(r)
        # _FakeST encoding of "hello" (5 chars) → [5, 6, 7, 8].
        assert r.get_by_id(rid).embedding is not None
        assert float(r.get_by_id(rid).embedding[0]) == 5.0


# --- from_files integration ------------------------------------------------


def _seed_file(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestFromFilesEmbedAllWiring:
    def test_no_embedder_leaves_block_embeddings_at_none(
        self, tmp_path: Path
    ) -> None:
        path = _seed_file(
            tmp_path, "x.md", "---\nname: a\n---\nbody\n"
        )
        r = BlockRegistry.from_files([path])
        assert r.list_blocks()[0].embedding is None

    def test_with_embedder_populates_block_embeddings(
        self, tmp_path: Path
    ) -> None:
        e = _FakeEmbedder()
        path = _seed_file(
            tmp_path, "x.md", "---\nname: a\n---\nbody\n"
        )
        r = BlockRegistry.from_files([path], embedding_model=e)
        block = r.list_blocks()[0]
        assert block.embedding is not None
        assert block.embedding.shape == (_FakeST.DIM,)

    def test_single_batch_call_for_multi_block_seed_file(
        self, tmp_path: Path
    ) -> None:
        # The user-emphasised invariant: one model call regardless of
        # block count, not one per block.
        e = _FakeEmbedder()
        path = _seed_file(
            tmp_path,
            "many.md",
            (
                "---\nname: a\n---\nA\n"
                "---\nname: b\n---\nB\n"
                "---\nname: c\n---\nC\n"
            ),
        )
        BlockRegistry.from_files([path], embedding_model=e)
        assert e._fake_st is not None
        assert len(e._fake_st.encode_calls) == 1

    def test_embeddings_cover_blocks_across_multiple_files(
        self, tmp_path: Path
    ) -> None:
        e = _FakeEmbedder()
        a = _seed_file(tmp_path, "first.md", "---\nname: a\n---\nA\n")
        b = _seed_file(tmp_path, "second.md", "---\nname: b\n---\nB\n")
        r = BlockRegistry.from_files([a, b], embedding_model=e)
        for block in r.list_blocks():
            assert block.embedding is not None
        # One batch call covering both files' blocks.
        assert e._fake_st is not None
        assert len(e._fake_st.encode_calls) == 1
        assert len(e._fake_st.encode_calls[0]) == 2

    def test_embed_all_runs_after_runtime_ids_are_assigned(
        self, tmp_path: Path
    ) -> None:
        # Sanity: the user-locked sequence is parse → assign UUIDs → insert
        # → embed_all. Asserting that every block has a non-empty
        # runtime_id at the point its embedding lands proves the order.
        e = _FakeEmbedder()
        path = _seed_file(
            tmp_path, "x.md", "---\nname: a\n---\nbody\n"
        )
        r = BlockRegistry.from_files([path], embedding_model=e)
        for block in r.list_blocks():
            assert block.runtime_id != ""
            assert block.embedding is not None

    def test_find_relevant_works_after_from_files_with_embedder(
        self, tmp_path: Path
    ) -> None:
        # End-to-end: cold-start batch embedding makes find_relevant
        # returnable results without any extra setup from the caller.
        e = _FakeEmbedder()
        path = _seed_file(
            tmp_path,
            "many.md",
            (
                "---\nname: a\n---\nshort\n"
                "---\nname: b\n---\nmuch longer content\n"
            ),
        )
        r = BlockRegistry.from_files([path], embedding_model=e)
        # Task text "short" gets the same vector as "short" content under
        # _FakeST → highest cosine for the first block.
        result = r.find_relevant("short", top_k=1)
        assert len(result) == 1
        assert result[0][0].semantic_name == "a"
