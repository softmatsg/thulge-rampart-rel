"""Compile-time pipeline: relevance gate → ordered walk → budget cutoff → emit.

The compiler is the central output function of the library. It walks the
registry's explicit ordering and produces a single concatenated prompt
string that fits within a hard token budget. Five non-obvious decisions
live here rather than at the caller:

* **compile() never modifies registry ordering.** Hard invariant 5. Tests
  record ``list(registry._blocks.keys())`` before and after a ``compile()``
  call (for both the full and the relevance-gated code routes) and assert
  exact equality. Only mutation methods reorder the registry.

* **access_count is bumped only for confirmed-included blocks.** A block
  that scored above the relevance threshold but was then cut by the token
  budget never reached the model. Treating it as "used" would mislead the
  eviction policy into protecting evidence that was never actually
  consulted. Stage 4 increments access_count strictly for the blocks that
  made it into the final string.

* **compile_dry_run() runs stages 1–3 and stops.** It returns the
  ``(runtime_id, token_count)`` tuples that would have been included.
  Critically: it does not increment any access_count and does not build
  an output string. That makes it safe to call from instrumentation, UI
  previews, and test harnesses without committing to the side effect.

* **Token costs include wrap_prefix/wrap_suffix and inter-block
  separators.** The budget is what the assembled string actually costs,
  not what the bare block content would. Separator cost is added once
  per block-after-the-first; the first block has no leading separator.

* **Tokenisation flows through the registry, never through tiktoken
  directly.** Compile calls ``registry.token_count_of`` for blocks (cached
  on the block) and ``registry.count_tokens`` for arbitrary strings
  (separator, wrap decoration). The registry was constructed with a
  tokeniser callable — by default, lazy tiktoken cl100k_base via
  ``RAMPARTConfig`` — and the compiler stays ignorant of which one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rampart.block import InstructionBlock
    from rampart.registry import BlockRegistry


_DEFAULT_SEPARATOR = "\n\n---\n\n"


@dataclass(frozen=True)
class CompileResult:
    """Outcome of a successful :func:`compile` call.

    Attributes:
        prompt: The assembled prompt string. Never exceeds the requested
            ``max_tokens`` under the registry's configured tokeniser.
            Empty string when nothing fit.
        included: Semantic names of the blocks that were emitted, in
            compile order. Useful for logging and audit trails without
            re-parsing the prompt.
        excluded: Semantic names of blocks that were considered but
            could not be included — either filtered out by the relevance
            gate or cut by the token-budget walk. Order matches the
            registry order so a reviewer can scan top-to-bottom.
        total_tokens: Total token count of the assembled output,
            including inter-block separators and wrap decoration.
    """

    prompt: str
    included: list[str]
    excluded: list[str]
    total_tokens: int


@dataclass(frozen=True)
class DryRunResult:
    """Outcome of a :func:`compile_dry_run` call.

    Attributes:
        included: ``(semantic_name, token_count)`` tuples in compile
            order. ``token_count`` is the block's effective cost
            (content + wrap decoration) excluding the inter-block
            separator, which is global rather than per-block.
        excluded: Semantic names of blocks that were considered but
            could not be included.
        total_tokens: What the assembled output would have cost,
            including separators. Compute-only — no string is built.
        included_tag_counts: Distribution of tags across the included
            blocks. Each tag in any included block's ``tags`` list
            contributes one to the count under that tag. Tags with
            zero included blocks are absent. Empty dict when no
            included block carries any tag.
    """

    included: list[tuple[str, int]]
    excluded: list[str]
    total_tokens: int
    included_tag_counts: dict[str, int] = field(default_factory=dict)


def compile(
    registry: BlockRegistry,
    max_tokens: int | None = None,
    task_text: str | None = None,
    relevance_threshold: float | None = None,
    top_k: int | None = None,
    separator: str | None = None,
) -> CompileResult:
    """Compile a registry into a :class:`CompileResult` within ``max_tokens``.

    Stages:

    1. **Relevance gate** (optional). When ``task_text`` is provided
       together with ``relevance_threshold`` and/or ``top_k``, the
       registry's ``find_relevant`` is called and only the returned blocks
       remain candidates. When neither gate parameter is set, every block
       is a candidate.
    2. **Ordered walk**. Walk the registry front-to-back in current
       ordering and apply the relevance gate.
    3. **Token budget cutoff**. Stop the walk when adding the next block
       (plus its wrap decoration plus a separator) would exceed
       ``max_tokens``. The walk does not skip a too-big block to look for
       smaller blocks behind it; ordering is the determining axis.
    4. **Emit and account**. Concatenate the surviving blocks with
       ``separator``, applying ``wrap_prefix`` and ``wrap_suffix`` to each.
       Increment ``access_count`` on each emitted block.

    Args:
        registry: The block registry to compile.
        max_tokens: Hard token budget for the assembled output. ``None``
            falls back to the registry's configured
            ``default_max_tokens`` (see ``RAMPARTConfig``).
        task_text: Optional. Free-form task description used to gate the
            walk by relevance. Required if ``relevance_threshold`` or
            ``top_k`` is set.
        relevance_threshold: Optional. Minimum cosine similarity for a
            block to remain a candidate. Requires ``task_text``.
        top_k: Optional. Cap on the number of relevance-gated candidates.
            Requires ``task_text``.
        separator: Inter-block separator. ``None`` uses the default
            ``"\\n\\n---\\n\\n"``.

    Returns:
        :class:`CompileResult` carrying the assembled string plus the
        included / excluded semantic names and the realised token total.

    Raises:
        ValueError: If ``relevance_threshold`` or ``top_k`` is set without
            ``task_text``.
        RuntimeError: If the relevance gate is requested but no embedding
            model is configured on the registry (raised by
            ``find_relevant``).
    """
    if max_tokens is None:
        max_tokens = registry.default_max_tokens
    sep = _DEFAULT_SEPARATOR if separator is None else separator
    selected, excluded_names = _select_blocks(
        registry, max_tokens, task_text, relevance_threshold, top_k, sep
    )
    if not selected:
        return CompileResult(
            prompt="",
            included=[],
            excluded=excluded_names,
            total_tokens=0,
        )
    parts: list[str] = []
    included_names: list[str] = []
    for block, _cost in selected:
        block.access_count += 1
        parts.append(_render_block(block))
        included_names.append(block.semantic_name)
    prompt = sep.join(parts)
    total_tokens = registry.count_tokens(prompt)
    return CompileResult(
        prompt=prompt,
        included=included_names,
        excluded=excluded_names,
        total_tokens=total_tokens,
    )


def compile_dry_run(
    registry: BlockRegistry,
    max_tokens: int | None = None,
    task_text: str | None = None,
    relevance_threshold: float | None = None,
    top_k: int | None = None,
    separator: str | None = None,
) -> DryRunResult:
    """Run stages 1–3 of :func:`compile` and return what would be included.

    Identical selection logic to :func:`compile`, including the strict
    ordered-walk-with-budget-cutoff. Stops before stage 4: no access_count
    increments, no output string.

    Args:
        registry: The block registry to inspect.
        max_tokens: Hard token budget that would be applied. ``None``
            falls back to the registry's configured
            ``default_max_tokens``.
        task_text: See :func:`compile`.
        relevance_threshold: See :func:`compile`.
        top_k: See :func:`compile`.
        separator: See :func:`compile`. Used only to size separator costs;
            no string is emitted.

    Returns:
        :class:`DryRunResult` carrying the would-be inclusions, the
        excluded semantic names, and the total token estimate.

    Raises:
        ValueError: As in :func:`compile`.
        RuntimeError: As in :func:`compile`.
    """
    if max_tokens is None:
        max_tokens = registry.default_max_tokens
    sep = _DEFAULT_SEPARATOR if separator is None else separator
    selected, excluded_names = _select_blocks(
        registry, max_tokens, task_text, relevance_threshold, top_k, sep
    )
    sep_cost = registry.count_tokens(sep) if selected else 0
    block_total = sum(cost for _, cost in selected)
    separator_total = sep_cost * max(0, len(selected) - 1)
    tag_counts: dict[str, int] = {}
    for block, _cost in selected:
        for tag in block.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return DryRunResult(
        included=[
            (block.semantic_name, cost) for block, cost in selected
        ],
        excluded=excluded_names,
        total_tokens=block_total + separator_total,
        included_tag_counts=tag_counts,
    )


def token_count(
    text: str,
    tokeniser: Callable[[str], int] | None = None,
) -> int:
    """Count tokens in ``text`` under ``tokeniser`` (default cl100k_base).

    Standalone helper for code that has a string but no registry handy.
    The default callable is the same lazy-loaded tiktoken cl100k_base
    encoder that ``RAMPARTConfig`` uses, so counts are consistent with
    the rest of the library.

    Args:
        text: String to count.
        tokeniser: Optional callable. ``None`` falls back to
            ``rampart.config.default_tokeniser``.

    Returns:
        Token count.
    """
    if tokeniser is None:
        from rampart.config import default_tokeniser

        tokeniser = default_tokeniser
    return tokeniser(text)


def _select_blocks(
    registry: BlockRegistry,
    max_tokens: int,
    task_text: str | None,
    relevance_threshold: float | None,
    top_k: int | None,
    separator: str,
) -> tuple[list[tuple[InstructionBlock, int]], list[str]]:
    """Stages 1–3: relevance gate, ordered walk, token-budget cutoff.

    Returns ``(selected, excluded_names)``. ``selected`` carries the
    blocks that would be emitted with each block's effective cost
    (content + wrap_prefix + wrap_suffix). ``excluded_names`` is the
    semantic names of the blocks that were considered but did not make
    the cut, in registry order. Used by both :func:`compile` (which
    then runs stage 4) and :func:`compile_dry_run` (which does not).
    """
    if (relevance_threshold is not None or top_k is not None) and task_text is None:
        raise ValueError(
            "relevance_threshold or top_k requires task_text to be set"
        )

    # Stage 1: relevance gate (optional).
    allowed_ids: set[str] | None
    if task_text is not None and (
        relevance_threshold is not None or top_k is not None
    ):
        scored = registry.find_relevant(
            task_text, top_k=top_k, threshold=relevance_threshold
        )
        allowed_ids = {block.runtime_id for block, _ in scored}
    else:
        allowed_ids = None

    sep_cost = registry.count_tokens(separator)

    # Stages 2 + 3: walk in registry order, apply gate, stop at budget.
    selected: list[tuple[InstructionBlock, int]] = []
    excluded: list[str] = []
    used_tokens = 0
    budget_exhausted = False
    for block in registry.list_blocks():
        if allowed_ids is not None and block.runtime_id not in allowed_ids:
            excluded.append(block.semantic_name)
            continue
        if budget_exhausted:
            excluded.append(block.semantic_name)
            continue
        block_cost = registry.token_count_of(block)
        if block.wrap_prefix is not None:
            block_cost += registry.count_tokens(block.wrap_prefix)
        if block.wrap_suffix is not None:
            block_cost += registry.count_tokens(block.wrap_suffix)
        added_cost = block_cost + (sep_cost if selected else 0)
        if used_tokens + added_cost > max_tokens:
            excluded.append(block.semantic_name)
            budget_exhausted = True
            continue
        selected.append((block, block_cost))
        used_tokens += added_cost
    return selected, excluded


def _render_block(block: InstructionBlock) -> str:
    """Apply wrap_prefix and wrap_suffix to a block's content."""
    return (
        (block.wrap_prefix or "") + block.content + (block.wrap_suffix or "")
    )


__all__ = [
    "CompileResult",
    "DryRunResult",
    "compile",
    "compile_dry_run",
    "token_count",
]
