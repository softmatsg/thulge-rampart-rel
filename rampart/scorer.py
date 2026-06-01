"""Embedding-model wiring and the cold-start batch-embedding routine.

The Protocol that defines what counts as an embedding model lives in
``rampart.registry`` (``EmbeddingModel``). This module imports it; it does
not redefine it. Keeping the Protocol in a single place is the difference
between two definitions drifting silently and one definition staying
authoritative.

Three pieces of functionality live here:

* ``SentenceTransformerEmbedder`` — concrete ``EmbeddingModel`` backed by
  the ``sentence-transformers`` library with the all-MiniLM-L6-v2 default
  (22 MB, Apache 2.0, 384-dim output). The
  underlying model loads lazily on first ``encode`` call so simply
  constructing an embedder costs no I/O or memory. The lazy hook is
  isolated in ``_construct_model`` so tests can override without touching
  ``sys.modules``.

* ``embed_all(registry)`` — the cold-start batch-embedding routine called
  by ``BlockRegistry.from_files`` after parsing and runtime-id assignment.
  Iterates ``registry.list_blocks()``, calls the embedder with the entire
  list of contents in a single ``encode_batch`` invocation, and writes the
  resulting vectors back into ``block.embedding`` in place. One model call
  for the whole seed set regardless of block count — that is the
  economical-at-boot invariant.

  Embedders that don't expose ``encode_batch`` fall through to per-block
  ``encode``. The fallback preserves correctness (every block still gets
  embedded) but pays N model calls. ``SentenceTransformerEmbedder``
  provides ``encode_batch`` and is the recommended embedder when batch
  efficiency matters.

* ``embed_text(text, embedder=None)`` — utility for callers (an
  application-layer loop, tests, the UI) that have a single string
  and want a vector without going through the registry. The default
  embedder is created once and cached so repeated ``embed_text`` calls
  do not reload the underlying model.

Note on what does NOT live here: ``score_block`` and ``find_relevant``
are already methods on ``BlockRegistry``. They use the registry's
configured embedder for their own embedding calls. This module exposes
the wiring those methods compose against.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from rampart.registry import EmbeddingModel

if TYPE_CHECKING:
    from rampart.registry import BlockRegistry


_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """``EmbeddingModel`` backed by sentence-transformers, with lazy loading.

    Constructs no underlying model until ``encode`` or ``encode_batch`` is
    first called. This keeps the import-time cost of ``rampart.scorer``
    near zero and lets tests construct embedders without paying download
    latency. Override ``_construct_model`` in a subclass to inject a fake
    for unit tests.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        """Construct the embedder. Does not load the model.

        Args:
            model_name: HuggingFace identifier of the sentence-transformers
                model. Default ``"all-MiniLM-L6-v2"`` matches the
                ``RAMPARTConfig.embedding_model_name`` default.
        """
        self.model_name = model_name
        self._model: Any = None

    def encode(self, text: str) -> NDArray[Any]:
        """Encode a single string into a 1-D float32 vector.

        Loads the underlying model on first call, caches it for subsequent
        calls. Returns the embedding as a numpy array regardless of the
        underlying model's native return type.
        """
        result = self._load().encode([text])
        return np.asarray(result[0], dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> NDArray[Any]:
        """Encode many strings in a single model call. Returns a 2-D array.

        This is the method ``embed_all`` invokes to satisfy the "one model
        call per cold start" invariant. Returns shape ``(len(texts), dim)``.
        """
        result = self._load().encode(texts)
        return np.asarray(result, dtype=np.float32)

    def _load(self) -> Any:  # noqa: ANN401  # sentence-transformers untyped
        """Return the underlying model, constructing it on first call."""
        if self._model is None:
            self._model = self._construct_model()
        return self._model

    def _construct_model(self) -> Any:  # noqa: ANN401  # sentence-transformers untyped
        """Build the underlying sentence-transformers model.

        Isolated as a separate method so tests can override the loader
        without monkeypatching ``sys.modules``. Subclass and override to
        return a fake; production code uses the real loader below.
        """
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)


@functools.lru_cache(maxsize=1)
def _default_embedder() -> SentenceTransformerEmbedder:
    """Return a process-wide cached default embedder.

    Wrapping construction in ``lru_cache(maxsize=1)`` ensures that
    ``embed_text`` calls without an explicit embedder share one
    underlying model rather than reloading on every call.
    """
    return SentenceTransformerEmbedder()


def embed_text(
    text: str,
    embedder: EmbeddingModel | None = None,
) -> NDArray[Any]:
    """Embed a single string. Convenience wrapper over ``EmbeddingModel.encode``.

    Args:
        text: String to embed.
        embedder: Embedder to use. ``None`` falls back to the process-wide
            cached default ``SentenceTransformerEmbedder()``.

    Returns:
        1-D embedding vector.
    """
    if embedder is None:
        embedder = _default_embedder()
    return embedder.encode(text)


def embed_all(registry: BlockRegistry) -> None:
    """Batch-embed every block in ``registry`` via its configured embedder.

    No-op if the registry has no embedding model configured, or has no
    blocks. Otherwise:

    * If the embedder exposes ``encode_batch``, calls it once with the full
      list of block contents. One model call regardless of block count.
    * Otherwise falls back to per-block ``encode``. Correct but pays N
      model calls; recommended only for embedders that lack a batch path.

    The resulting vectors are written into ``block.embedding`` in place,
    in registry order. Existing embeddings are overwritten — the cold-start
    contract is "embed everything fresh", not "patch what's missing".

    Args:
        registry: The registry whose blocks should be embedded.
    """
    embedder = registry._embedding_model
    if embedder is None:
        return
    blocks = registry.list_blocks()
    if not blocks:
        return
    contents = [block.content for block in blocks]
    if hasattr(embedder, "encode_batch"):
        vectors = embedder.encode_batch(contents)
        for block, vec in zip(blocks, vectors, strict=True):
            block.embedding = np.asarray(vec)
    else:
        for block, content in zip(blocks, contents, strict=True):
            block.embedding = np.asarray(embedder.encode(content))


__all__ = [
    "SentenceTransformerEmbedder",
    "embed_all",
    "embed_text",
]
