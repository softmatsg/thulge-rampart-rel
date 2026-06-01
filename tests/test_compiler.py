"""Tests for the compile pipeline, including hard invariant 5.

Three behaviours are exercised by dedicated test classes:

* ``TestInvariantFiveOrderingPreserved`` — records the ordered key list
  before each ``compile()`` call and re-records after. Asserted for both
  the full-compile path and the relevance-gated path; those take
  different code routes through ``_select_blocks``.
* ``TestAccessCountIsConfirmedInclusionOnly`` — sets up a registry where
  several blocks score above the relevance threshold but only one fits
  the token budget. Verifies that the cut blocks' ``access_count`` stays
  at zero while the included block's increments to one.
* ``TestCompileDryRunSkipsStageFour`` — verifies dry-run never touches
  ``access_count`` and never builds an output string.

Tests use a deterministic token counter (``len``) so budget arithmetic is
exact and assertions are not at the mercy of cl100k_base behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from rampart.compiler import compile as compile_fn
from rampart.compiler import compile_dry_run, token_count
from rampart.config import RAMPARTConfig, default_tokeniser
from rampart.registry import BlockRegistry


def _vec(*values: float) -> NDArray[Any]:
    return np.array(values, dtype=np.float32)


class _FakeEmbedder:
    """Deterministic mapping of strings to vectors for relevance tests."""

    def __init__(self, mapping: dict[str, NDArray[Any]]) -> None:
        self.mapping = mapping

    def encode(self, text: str) -> NDArray[Any]:
        return self.mapping.get(text, np.zeros(3, dtype=np.float32))


def _registry(
    *,
    embedder: _FakeEmbedder | None = None,
) -> BlockRegistry:
    """Build a registry with a deterministic ``len`` tokeniser.

    Using ``len`` makes per-block costs equal to the character length of
    each block's content — easy to size against a budget exactly without
    depending on tiktoken's behaviour or the default 4-char heuristic.
    """
    return BlockRegistry(tokeniser=len, embedding_model=embedder)


def _agent_block(
    reg: BlockRegistry,
    content: str,
    name: str,
    tags: list[str] | None = None,
) -> str:
    return reg.write_agent_block(
        content, trajectory_id="t", semantic_name=name, tags=tags,
    )


# --- compile() basic behaviour ----------------------------------------------


class TestCompileBasic:
    def test_empty_registry_returns_empty_string(self) -> None:
        r = _registry()
        assert r.compile(max_tokens=100).prompt == ""

    def test_single_block_emitted_within_budget(self) -> None:
        r = _registry()
        _agent_block(r, "hello", name="a")
        result = r.compile(max_tokens=100)
        assert result.prompt == "hello"
        assert result.included == ["a"]
        assert result.excluded == []

    def test_two_blocks_joined_by_default_separator(self) -> None:
        r = _registry()
        _agent_block(r, "AAA", name="a")
        _agent_block(r, "BBB", name="b")
        result = r.compile(max_tokens=100)
        assert result.prompt == "AAA\n\n---\n\nBBB"
        assert result.included == ["a", "b"]

    def test_custom_separator_is_emitted(self) -> None:
        r = _registry()
        _agent_block(r, "AAA", name="a")
        _agent_block(r, "BBB", name="b")
        assert (
            r.compile(max_tokens=100, separator=" | ").prompt == "AAA | BBB"
        )

    def test_wrap_prefix_and_suffix_emitted(self) -> None:
        r = _registry()
        rid = _agent_block(r, "core", name="a")
        r.set_wrap(rid, prefix="<<", suffix=">>")
        assert r.compile(max_tokens=100).prompt == "<<core>>"


# --- token-budget cutoff (stage 3) ------------------------------------------


class TestTokenBudgetCutoff:
    def test_fits_when_budget_exact(self) -> None:
        r = _registry()
        _agent_block(r, "AAAA", name="a")  # 4 tokens
        _agent_block(r, "BBBB", name="b")  # 4 tokens
        # Default separator "\n\n---\n\n" is 7 chars → 7 tokens.
        # Total: 4 + 7 + 4 = 15.
        result = r.compile(max_tokens=15)
        assert result.prompt == "AAAA\n\n---\n\nBBBB"
        assert result.total_tokens == 15

    def test_drops_block_that_pushes_over_budget(self) -> None:
        r = _registry()
        _agent_block(r, "AAAA", name="a")  # 4
        _agent_block(r, "BBBB", name="b")  # 4
        # Budget 14 < 15; the second block does not fit.
        result = r.compile(max_tokens=14)
        assert result.prompt == "AAAA"
        assert result.included == ["a"]
        assert result.excluded == ["b"]

    def test_walk_does_not_skip_too_big_block_to_find_smaller_block(self) -> None:
        # Strict ordered walk: the first block that doesn't fit halts
        # the walk. The smaller behind block lands in `excluded`.
        r = _registry()
        _agent_block(r, "x" * 10, name="big")
        _agent_block(r, "y", name="small")
        result = r.compile(max_tokens=5)
        assert result.prompt == ""
        assert result.included == []
        assert result.excluded == ["big", "small"]

    def test_zero_budget_emits_nothing(self) -> None:
        r = _registry()
        _agent_block(r, "AAAA", name="a")
        assert r.compile(max_tokens=0).prompt == ""

    def test_wrap_decoration_counted_against_budget(self) -> None:
        r = _registry()
        rid = _agent_block(r, "core", name="a")  # 4 tokens
        r.set_wrap(rid, prefix="<<", suffix=">>")  # 2 + 2 = 4 tokens
        # Total cost 8. Budget 7 should drop the block.
        assert r.compile(max_tokens=7).prompt == ""
        # Budget 8 should fit it exactly.
        assert r.compile(max_tokens=8).prompt == "<<core>>"


# --- relevance gate (stage 1) -----------------------------------------------


class TestRelevanceGate:
    def test_no_gate_when_neither_threshold_nor_top_k_passed(self) -> None:
        # Even with task_text, no gate parameter means every block is a
        # candidate. No embedding lookup happens at all.
        r = _registry()  # no embedding model
        _agent_block(r, "AAA", name="a")
        # Should not raise (would raise if find_relevant were called).
        assert (
            r.compile(max_tokens=100, task_text="anything").prompt == "AAA"
        )

    def test_threshold_without_task_text_raises(self) -> None:
        r = _registry()
        with pytest.raises(ValueError, match="task_text"):
            r.compile(max_tokens=100, relevance_threshold=0.5)

    def test_top_k_without_task_text_raises(self) -> None:
        r = _registry()
        with pytest.raises(ValueError, match="task_text"):
            r.compile(max_tokens=100, top_k=3)

    def test_threshold_filters_out_low_scoring_blocks(self) -> None:
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = _registry(embedder=embedder)
        a = _agent_block(r, "AAA", name="a")
        b = _agent_block(r, "BBB", name="b")
        r.get_by_id(a).embedding = _vec(1.0, 0.0, 0.0)  # cosine 1
        r.get_by_id(b).embedding = _vec(0.0, 1.0, 0.0)  # cosine 0
        result = r.compile(
            max_tokens=100, task_text="task", relevance_threshold=0.5
        )
        assert result.prompt == "AAA"
        assert "b" in result.excluded

    def test_top_k_caps_relevance_candidates(self) -> None:
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = _registry(embedder=embedder)
        a = _agent_block(r, "AAA", name="a")
        b = _agent_block(r, "BBB", name="b")
        c = _agent_block(r, "CCC", name="c")
        r.get_by_id(a).embedding = _vec(1.0, 0.0, 0.0)
        r.get_by_id(b).embedding = _vec(2.0, 1.0, 0.0)
        r.get_by_id(c).embedding = _vec(0.0, 1.0, 0.0)  # least relevant
        result = r.compile(max_tokens=100, task_text="task", top_k=2)
        # c is excluded by top_k. Output includes a and b in registry order.
        assert "AAA" in result.prompt
        assert "BBB" in result.prompt
        assert "CCC" not in result.prompt
        assert "c" in result.excluded


# --- access_count semantics (stage 4) ---------------------------------------


class TestAccessCountIsConfirmedInclusionOnly:
    def test_only_included_blocks_increment(self) -> None:
        r = _registry()
        a = _agent_block(r, "AAAA", name="a")  # 4
        b = _agent_block(r, "BBBB", name="b")  # 4
        c = _agent_block(r, "CCCC", name="c")  # 4
        # Budget 4 — only block a fits.
        r.compile(max_tokens=4)
        assert r.get_by_id(a).access_count == 1
        assert r.get_by_id(b).access_count == 0
        assert r.get_by_id(c).access_count == 0

    def test_blocks_passing_relevance_but_cut_by_budget_do_not_increment(
        self,
    ) -> None:
        # The user-specified guarantee: a block scored as relevant but cut
        # by the token budget MUST NOT have its access_count bumped, since
        # the eviction policy uses access_count to judge "useful" blocks
        # and one that never reached the model is not evidence either way.
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = _registry(embedder=embedder)
        a = _agent_block(r, "AAAA", name="a")  # 4 tokens
        b = _agent_block(r, "BBBB", name="b")  # 4 tokens
        c = _agent_block(r, "CCCC", name="c")  # 4 tokens
        # All three score 1.0 (identical vectors) so all pass the gate.
        r.get_by_id(a).embedding = _vec(1.0, 0.0, 0.0)
        r.get_by_id(b).embedding = _vec(1.0, 0.0, 0.0)
        r.get_by_id(c).embedding = _vec(1.0, 0.0, 0.0)
        # Budget 4 fits only one block. The other two passed the gate but
        # never reached the model.
        r.compile(
            max_tokens=4, task_text="task", relevance_threshold=0.5
        )
        assert r.get_by_id(a).access_count == 1
        assert r.get_by_id(b).access_count == 0
        assert r.get_by_id(c).access_count == 0

    def test_repeated_compile_accumulates_counts_per_call(self) -> None:
        r = _registry()
        rid = _agent_block(r, "x", name="a")
        r.compile(max_tokens=100)
        r.compile(max_tokens=100)
        r.compile(max_tokens=100)
        assert r.get_by_id(rid).access_count == 3


# --- compile_dry_run skips stage 4 ------------------------------------------


class TestCompileDryRunSkipsStageFour:
    def test_dry_run_does_not_increment_any_access_count(self) -> None:
        r = _registry()
        rid = _agent_block(r, "x", name="a")
        r.compile_dry_run(max_tokens=100)
        r.compile_dry_run(max_tokens=100)
        assert r.get_by_id(rid).access_count == 0

    def test_dry_run_returns_dry_run_result_with_semantic_names(self) -> None:
        r = _registry()
        _agent_block(r, "AAAA", name="a")  # 4
        _agent_block(r, "BBBB", name="b")  # 4
        result = r.compile_dry_run(max_tokens=100)
        assert result.included == [("a", 4), ("b", 4)]
        assert result.excluded == []
        # Two 4-token blocks plus one 7-token separator = 15.
        assert result.total_tokens == 15

    def test_dry_run_selection_matches_compile_selection(self) -> None:
        r = _registry()
        _agent_block(r, "AAAA", name="a")  # 4
        _agent_block(r, "BBBB", name="b")  # 4 — would push over budget 14
        # Independent dry_run BEFORE compile so access_count starts at 0.
        dry = r.compile_dry_run(max_tokens=14)
        compiled = r.compile(max_tokens=14)
        assert dry.included == [("a", 4)]
        assert dry.excluded == ["b"]
        assert compiled.prompt == "AAAA"
        assert compiled.included == ["a"]

    def test_dry_run_respects_relevance_gate(self) -> None:
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = _registry(embedder=embedder)
        a = _agent_block(r, "AAAA", name="a")
        b = _agent_block(r, "BBBB", name="b")
        r.get_by_id(a).embedding = _vec(1.0, 0.0, 0.0)
        r.get_by_id(b).embedding = _vec(0.0, 1.0, 0.0)
        result = r.compile_dry_run(
            max_tokens=100, task_text="task", relevance_threshold=0.5
        )
        assert result.included == [("a", 4)]
        assert result.excluded == ["b"]

    def test_dry_run_aggregates_tag_counts_for_included_blocks(
        self,
    ) -> None:
        r = _registry()
        _agent_block(r, "AAAA", name="a", tags=["isr", "timing"])
        _agent_block(r, "BBBB", name="b", tags=["isr"])
        result = r.compile_dry_run(max_tokens=100)
        assert result.included_tag_counts == {"isr": 2, "timing": 1}

    def test_dry_run_tag_counts_omit_excluded_blocks(self) -> None:
        # Block "a" fits the budget but "b" does not — its tags must
        # NOT appear in included_tag_counts.
        r = _registry()
        _agent_block(r, "AAAA", name="a", tags=["isr"])
        _agent_block(r, "BBBB", name="b", tags=["timing"])
        result = r.compile_dry_run(max_tokens=4)
        assert result.included == [("a", 4)]
        assert result.excluded == ["b"]
        assert result.included_tag_counts == {"isr": 1}

    def test_dry_run_tag_counts_empty_when_no_tags(self) -> None:
        r = _registry()
        _agent_block(r, "AAAA", name="a")
        _agent_block(r, "BBBB", name="b")
        result = r.compile_dry_run(max_tokens=100)
        assert result.included_tag_counts == {}


# --- hard invariant 5: compile() never reorders ------------------------------


class TestInvariantFiveOrderingPreserved:
    """Hard invariant 5: ``compile()`` never modifies registry ordering.

    Tested for both code routes through the walk:
    * full path (no relevance gate)
    * relevance-gated path (find_relevant called for the candidate set)
    """

    def test_full_compile_preserves_key_order(self) -> None:
        r = _registry()
        for i in range(8):
            _agent_block(r, f"block-{i}-content", name=f"b{i}")
        before = list(r._blocks.keys())
        r.compile(max_tokens=100)
        after = list(r._blocks.keys())
        assert before == after

    def test_full_compile_with_budget_cutoff_preserves_key_order(self) -> None:
        r = _registry()
        for i in range(8):
            _agent_block(r, f"block-{i}-content", name=f"b{i}")
        before = list(r._blocks.keys())
        # Budget that forces an early cutoff still must not reorder.
        r.compile(max_tokens=20)
        after = list(r._blocks.keys())
        assert before == after

    def test_relevance_gated_compile_preserves_key_order(self) -> None:
        embedder = _FakeEmbedder({"task": _vec(1.0, 0.0, 0.0)})
        r = _registry(embedder=embedder)
        ids = []
        for i in range(8):
            rid = _agent_block(r, f"block-{i}-content", name=f"b{i}")
            ids.append(rid)
            # Random-ish but deterministic embeddings so find_relevant
            # has work to do.
            r.get_by_id(rid).embedding = _vec(
                float((i * 7) % 5), float((i * 3) % 5), 0.0
            )
        before = list(r._blocks.keys())
        r.compile(
            max_tokens=100, task_text="task", relevance_threshold=0.0
        )
        after = list(r._blocks.keys())
        assert before == after

    def test_dry_run_preserves_key_order(self) -> None:
        r = _registry()
        for i in range(5):
            _agent_block(r, f"x{i}", name=f"b{i}")
        before = list(r._blocks.keys())
        r.compile_dry_run(max_tokens=50)
        after = list(r._blocks.keys())
        assert before == after


# --- standalone token_count helper ------------------------------------------


class TestStandaloneTokenCount:
    def test_default_uses_tiktoken_cl100k(self) -> None:
        # Sanity: a short English string under cl100k_base should produce
        # a count close to a few tokens, not zero, not hundreds.
        n = token_count("hello world")
        assert 1 <= n <= 5

    def test_custom_tokeniser_overrides_default(self) -> None:
        assert token_count("hello world", tokeniser=len) == 11

    def test_empty_string(self) -> None:
        assert token_count("") == 0


# --- RAMPARTConfig + default_tokeniser --------------------------------------


class TestRAMPARTConfigDefaults:
    def test_default_tokeniser_is_lazy_tiktoken(self) -> None:
        cfg = RAMPARTConfig()
        assert cfg.tokeniser is default_tokeniser
        assert default_tokeniser("hello") > 0

    def test_default_max_tokens_is_16k(self) -> None:
        assert RAMPARTConfig().default_max_tokens == 16384

    def test_default_memory_limit_mb_is_100(self) -> None:
        assert RAMPARTConfig().memory_limit_mb == 100.0

    def test_default_block_separator_matches_compiler_default(self) -> None:
        # Drift between RAMPARTConfig.block_separator and the compiler's
        # default would be a silent semantic change — test pins them.
        from rampart.compiler import _DEFAULT_SEPARATOR

        assert RAMPARTConfig().block_separator == _DEFAULT_SEPARATOR

    def test_lazy_encoder_is_cached(self) -> None:
        from rampart.config import _cl100k_encoder

        first = _cl100k_encoder()
        second = _cl100k_encoder()
        assert first is second


# --- from_files config plumbing ---------------------------------------------


class TestFromFilesConfigPlumbing:
    def test_config_tokeniser_used_when_kwarg_absent(self, tmp_path: Path) -> None:
        cfg = RAMPARTConfig(tokeniser=len)
        path = tmp_path / "x.md"
        path.write_text("---\nname: a\n---\nbody\n", encoding="utf-8")
        r = BlockRegistry.from_files([path], config=cfg)
        assert r._tokeniser is len

    def test_explicit_tokeniser_overrides_config(self, tmp_path: Path) -> None:
        cfg = RAMPARTConfig(tokeniser=len)

        def custom(text: str) -> int:
            return 42

        path = tmp_path / "x.md"
        path.write_text("---\nname: a\n---\nbody\n", encoding="utf-8")
        r = BlockRegistry.from_files([path], tokeniser=custom, config=cfg)
        assert r._tokeniser is custom

    def test_config_default_max_tokens_propagates(
        self, tmp_path: Path
    ) -> None:
        cfg = RAMPARTConfig(default_max_tokens=4096)
        path = tmp_path / "x.md"
        path.write_text("---\nname: a\n---\nbody\n", encoding="utf-8")
        r = BlockRegistry.from_files([path], config=cfg)
        assert r.default_max_tokens == 4096

    def test_default_config_tokeniser_is_tiktoken(self, tmp_path: Path) -> None:
        path = tmp_path / "x.md"
        path.write_text("---\nname: a\n---\nbody\n", encoding="utf-8")
        r = BlockRegistry.from_files([path])
        # No explicit config or tokeniser — should be the lazy tiktoken one.
        assert r._tokeniser is default_tokeniser


# --- guard against compile_fn import alias collision ------------------------


class TestModuleLevelCompileFunction:
    def test_compile_callable_via_module_path(self) -> None:
        r = _registry()
        _agent_block(r, "hello", name="a")
        assert compile_fn(r, max_tokens=100).prompt == "hello"

    def test_compile_dry_run_callable_via_module_path(self) -> None:
        r = _registry()
        _agent_block(r, "hello", name="a")
        result = compile_dry_run(r, max_tokens=100)
        assert result.included == [("a", 5)]
