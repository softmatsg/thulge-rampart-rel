"""RAMPART session-level configuration object and the lazy default tokeniser.

``RAMPARTConfig`` is the single source of truth for session-level tunables —
token budgets, embedding model name, eviction-policy thresholds, the
tokeniser callable. Cold-start entry points (``BlockRegistry.from_files``)
read defaults from a ``RAMPARTConfig()`` instance so the per-call
overrides at every other layer remain pure overrides rather than a second
default-pinning surface.

The ``tokeniser`` field defaults to ``default_tokeniser``, which counts
tokens via tiktoken's ``cl100k_base`` encoder. Two non-obvious choices
live in this module:

* The ``cl100k_base`` encoder is fetched through a
  ``functools.lru_cache(maxsize=1)``-wrapped helper. Importing
  ``tiktoken`` and loading the encoder costs ~100ms of startup latency.
  Wrapping the import + load in a cached call means the cost is paid
  once, on the first ``default_tokeniser`` invocation, and only when
  the user actually needs the default. Callers that ship their own
  tokeniser (Qwen3 native, llama-cpp tokenizer, mock) never trigger
  the import at all.
* The compiler does not import ``tiktoken`` directly. It calls into
  the registry's ``count_tokens`` / ``token_count_of``, both of which
  delegate to whatever callable the registry was constructed with. The
  tokenisation choice is therefore a configuration concern, not a
  compiler concern, and swapping tokenisers requires no compiler
  changes.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rampart.eviction import DefaultEvictionPolicy, EvictionPolicy


@functools.lru_cache(maxsize=1)
def _cl100k_encoder() -> Any:  # noqa: ANN401  # tiktoken does not ship stubs
    """Return the tiktoken cl100k_base encoder, loaded once and cached.

    Wrapped in ``lru_cache`` so the encoder is constructed lazily on first
    call and reused for the lifetime of the process. The ``import tiktoken``
    statement lives inside the function so simply importing this module
    does not pull tiktoken into memory.

    Return type is ``Any`` because tiktoken does not ship type stubs;
    pinning a more specific type would require either vendoring stubs or
    importing tiktoken at module-evaluation time, both of which defeat
    the lazy-loading purpose of this helper.
    """
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def default_tokeniser(text: str) -> int:
    """Count tokens in ``text`` via tiktoken's cl100k_base encoder.

    The encoder is loaded on first call (and cached for the rest of the
    process lifetime) by ``_cl100k_encoder``. Subsequent calls are pure
    encode-and-count. cl100k_base is the GPT-4 / GPT-3.5 tokenisation
    family; it is within ~10% of the Qwen3 and Llama 3 tokenisers for
    typical English content, which is close enough for token-budget
    management without pulling in the heavier transformers dependency.

    Args:
        text: Arbitrary string to count.

    Returns:
        Token count under cl100k_base.
    """
    return len(_cl100k_encoder().encode(text))


@dataclass
class RAMPARTConfig:
    """Session-level tunables for a RAMPART instance.

    Defaults are sized for a typical local-inference target (a single
    8–14B instruction-tuned model on a 16 GB GPU); override the
    relevant fields per call for larger contexts, cloud backends, or
    ablation runs.
    """

    default_max_tokens: int = 16384
    """Default token ceiling for ``compile()``.

    The only token limit in the system. ``compile(max_tokens=N)`` overrides
    per call; ``compile()`` with no argument falls back to this. Set this
    to the model's context window minus the expected task and response
    token allowance — e.g. an 8K-context model with a 1K task and 1K
    response budget should run with ``default_max_tokens=6144``.
    """

    memory_limit_mb: float = 100.0
    """Soft warning threshold for total registry size, in megabytes.

    When the registry exceeds this size during a write operation a
    warning is emitted via the standard library ``warnings`` module.
    Nothing is auto-evicted and no exception is raised — the threshold
    exists so an operator running with the default policy gets a
    visible heads-up before the registry grows large enough to dominate
    process RSS. Set to ``float("inf")`` to disable the warning.
    """

    embedding_model_name: str = "all-MiniLM-L6-v2"
    """HuggingFace identifier loaded by ``rampart.scorer``."""

    relevance_threshold: float | None = None
    """Cosine threshold for the lazy-render gate in ``compile()``."""

    top_k_promote: int = 3
    """How many blocks BlockAgent's score-and-promote step lifts per task."""

    promote_steps: int = 5
    """Default ``steps`` argument for BlockAgent promotion calls."""

    block_separator: str = "\n\n---\n\n"
    """String emitted between blocks in compiled output."""

    thinking_budget_tokens: int | None = None
    """Cap on Qwen3 ``<think>`` block size when thinking mode is enabled."""

    tokeniser: Callable[[str], int] = default_tokeniser
    """Token-count callable. Defaults to lazy tiktoken cl100k_base."""

    eviction_policy: EvictionPolicy = field(
        default_factory=DefaultEvictionPolicy,
    )
    """Per-session eviction policy. Defaults to a fresh
    :class:`DefaultEvictionPolicy` instance per ``RAMPARTConfig``.

    Constructed via ``default_factory`` so each config carries an
    independent policy object — necessary because subclasses may
    cache state across calls (e.g. an adaptive policy that tracks
    recent eviction history). Override at construction time to plug
    in custom logic; see :class:`rampart.eviction.EvictionPolicy`
    for the protocol contract and a worked tag-based protective
    example.
    """


__all__ = [
    "RAMPARTConfig",
    "default_tokeniser",
]
