"""LlamaCppBackend — llama-cpp-python wrapper for local quantised inference.

Three boundaries this module enforces:

* **No module-level llama-cpp-python import.** The ``from llama_cpp import
  Llama`` lives inside ``__init__``. Importing this module is safe in any
  environment; constructing ``LlamaCppBackend`` is what fails (with a
  typed ImportError pointing at ``pip install rampart[local]``) when the
  optional dependency is missing. This is what lets the test suite import
  ``blockagent.backends.LlamaCppBackend`` and assert the helpful failure
  message without llama-cpp-python being on the dev box.

* **Qwen3 thinking toggle is documented as best-effort.** llama-cpp-python
  0.2.x does not expose a clean ``enable_thinking`` parameter. The
  user-facing toggle for Qwen3 is the ``/think`` and ``/no_think``
  prompt commands, which Qwen3 interprets via its chat template at
  tokenisation time. We prepend the appropriate command to the prompt
  as the best-effort fallback. For non-Qwen3 models the command is
  just literal text the model ignores — correct degradation.

* **Model load happens at construction time.** Loading Qwen3-4B Q4_K_M
  takes a few seconds; doing it lazily on first ``generate`` would shift
  that latency to the first task, surprising callers. Eager construction
  matches the user expectation that "the backend is ready when the
  constructor returns".
"""

from __future__ import annotations

from typing import Any, Literal, overload

from blockagent.backends.base import Backend, GenerateResult, apply_thinking_toggle


class LlamaCppBackend:
    """Local-inference backend wrapping ``llama_cpp.Llama``.

    Implements the ``Backend`` protocol. The wrapped Llama instance is
    constructed once at ``__init__`` and reused for every ``generate``
    call. Construction does the lazy import of llama-cpp-python; if the
    package is missing the constructor raises a typed ImportError telling
    the user how to install the optional extra.
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        seed: int | None = None,
    ) -> None:
        """Construct a LlamaCppBackend backed by a GGUF model file.

        Args:
            model_path: Path to a GGUF-format model file (e.g. a
                Qwen3-4B Q4_K_M weights file).
            n_gpu_layers: Number of transformer layers to offload to GPU.
                Default ``0`` runs CPU-only. Tune to fit your VRAM
                budget — a full offload (``-1``) at Q4_K_M on a typical
                7–8B model can exceed 16 GB once the KV cache is included.
            n_ctx: Context window in tokens. Default ``4096`` is a
                practical setting for typical instruction-tuned models;
                raise if you compile prompts that exceed it.
            seed: Reproducibility seed for sampling. ``None`` (default)
                lets llama-cpp-python pick a fresh seed per session;
                pass an integer to make completion deterministic.

        Raises:
            ImportError: If llama-cpp-python is not installed. The error
                message tells the caller exactly which extra to install.
        """
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "LlamaCppBackend requires llama-cpp-python. "
                "Install with: pip install rampart[local]"
            ) from exc

        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.seed = seed
        self._model: Any = self._construct_model(Llama)

    def _construct_model(self, llama_cls: Any) -> Any:  # noqa: ANN401  # llama-cpp-python untyped
        """Instantiate the wrapped Llama. Subclassable for tests."""
        return llama_cls(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            seed=self.seed if self.seed is not None else -1,
            verbose=False,
        )

    @overload
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        stop_sequences: list[str] | None = None,
        thinking: bool = False,
        system_prompt: str | None = None,
        return_usage: Literal[False] = False,
    ) -> str: ...

    @overload
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        stop_sequences: list[str] | None = None,
        thinking: bool = False,
        system_prompt: str | None = None,
        *,
        return_usage: Literal[True],
    ) -> GenerateResult: ...

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        stop_sequences: list[str] | None = None,
        thinking: bool = False,
        system_prompt: str | None = None,
        return_usage: bool = False,
    ) -> str | GenerateResult:
        """Generate a completion. See ``Backend.generate`` for the contract."""
        # llama-cpp-python 0.2.x does not expose a clean enable_thinking
        # kwarg or chat-template hook. The best-effort fallback (shared
        # across all backends) is to prepend Qwen3's documented /think
        # or /no_think prompt command; on Qwen3 the chat template
        # interprets it at tokenisation time, on other models it
        # degrades to literal text the model ignores.
        effective_prompt = apply_thinking_toggle(prompt, thinking)
        if system_prompt:
            # No native system-role slot on the raw text-completion API —
            # surface the instruction as a SYSTEM: marker the model can
            # treat as an instruction the way most instruction-tuned
            # checkpoints recognise leading meta-prompts.
            effective_prompt = f"SYSTEM: {system_prompt}\n\n{effective_prompt}"

        output = self._model(
            effective_prompt,
            max_tokens=max_new_tokens,
            stop=stop_sequences if stop_sequences is not None else [],
        )

        text = self._extract_text(output)
        if not return_usage:
            return text

        prompt_tokens, completion_tokens = self._extract_usage(output)
        return GenerateResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @staticmethod
    def _extract_text(output: Any) -> str:  # noqa: ANN401  # llama-cpp-python untyped
        """Pull the completion text out of llama-cpp-python's response dict."""
        text = output["choices"][0]["text"]
        return str(text)

    @staticmethod
    def _extract_usage(output: Any) -> tuple[int, int]:  # noqa: ANN401  # llama-cpp-python untyped
        """Pull (prompt_tokens, completion_tokens) out of the response dict.

        Some llama-cpp-python builds omit the usage block in certain modes;
        in that case both counts surface as zero rather than crashing the
        whole generate call. The session-report consumer treats zero as
        "unknown" and aggregates accordingly.
        """
        usage = output.get("usage") or {}
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


# Structural type-check that LlamaCppBackend satisfies the Backend protocol.
# Triggers a mypy error if the overload signatures drift from base.py.
_: type[Backend] = LlamaCppBackend


__all__ = [
    "LlamaCppBackend",
]
