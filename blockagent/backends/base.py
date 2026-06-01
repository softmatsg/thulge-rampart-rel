"""Backend Protocol and the GenerateResult shape returned with token usage.

Three load-bearing decisions live in this module:

* **Zero optional dependencies at import time.** This file imports nothing
  beyond the standard library so that ``from blockagent.backends.base
  import Backend`` is safe in environments with no llama-cpp-python, no
  httpx, no anthropic SDK. The concrete backends each handle their own
  heavy dependency in their own module's class constructors. The base
  module is the contract; the concrete modules are the integrations.

* **`generate` overloads on `return_usage`.** A single signature returning
  ``str`` would force an awkward later migration when Benchmark 5 needs
  per-coordination token counts. A single signature returning a union
  forces every str-only caller into ``isinstance`` ceremony. The
  ``Literal[False]`` / ``Literal[True]`` overload pair keeps both call
  sites clean: ``backend.generate(p)`` returns ``str``,
  ``backend.generate(p, return_usage=True)`` returns ``GenerateResult``,
  and mypy resolves the precise return type at each call site.

* **Frozen dataclass over NamedTuple for GenerateResult.** Both work; the
  frozen dataclass with ``slots=True`` gives clearer attribute access
  errors, allows future-additive fields with defaults, and is consistent
  with the other small value objects in the project. ``frozen=True``
  prevents callers from mutating returned usage counts in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, overload, runtime_checkable


@dataclass(frozen=True, slots=True)
class GenerateResult:
    """Generation output bundled with prompt and completion token counts.

    Returned by ``Backend.generate`` when ``return_usage=True``. Counts are
    measured by the backend itself (not re-tokenised by the caller) so they
    reflect exactly what the underlying model paid in input/output tokens.

    Attributes:
        text: The generated text — same string ``generate`` would have
            returned with ``return_usage=False``.
        prompt_tokens: Tokens the model consumed in the prompt.
        completion_tokens: Tokens the model produced as output.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int


@runtime_checkable
class Backend(Protocol):
    """Strategy interface for an LLM completion backend.

    Implementations wrap a specific runtime (llama-cpp-python, Ollama HTTP,
    OpenAI-compatible HTTP, ...) and expose a single ``generate`` entry
    point. The overloaded return type is the core of the contract: callers
    that want plain text get ``str``, callers that need accounting pass
    ``return_usage=True`` and get a ``GenerateResult``.
    """

    @overload
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = ...,
        stop_sequences: list[str] | None = ...,
        thinking: bool = ...,
        system_prompt: str | None = ...,
        return_usage: Literal[False] = ...,
    ) -> str: ...

    @overload
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = ...,
        stop_sequences: list[str] | None = ...,
        thinking: bool = ...,
        system_prompt: str | None = ...,
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
        """Run the model over ``prompt`` and return the completion.

        Args:
            prompt: Input text to complete.
            max_new_tokens: Hard cap on completion length in tokens.
            stop_sequences: Strings that, when generated, halt completion.
                ``None`` means no extra stops beyond the model's defaults.
            thinking: For models with a thinking-mode toggle (Qwen3 family),
                ``True`` requests the explicit reasoning trace and
                ``False`` requests a direct answer. Backends with no such
                toggle ignore this argument.
            system_prompt: Optional system-message instruction prepended
                ahead of the user prompt. Implementation varies by
                backend: chat-completions backends emit a ``system``
                role message; raw-text backends prepend a ``SYSTEM:``
                marker. Use case is suppressing residual reasoning
                traces on small reasoning-capable models when
                ``thinking=False`` — a brief instruction like
                "Answer directly without reasoning traces." stops the
                model from spending tokens on unsolicited ``<think>``
                blocks. Default ``None`` means no system message,
                preserving prior behaviour.
            return_usage: When ``False`` (default), return the completion
                as a plain ``str``. When ``True``, return a
                ``GenerateResult`` carrying both the text and the
                backend-reported prompt and completion token counts.

        Returns:
            ``str`` when ``return_usage=False``, ``GenerateResult`` when
            ``return_usage=True``. Overloads above pin the type for mypy
            so each call site sees the precise return type rather than
            the union.
        """
        ...


def apply_thinking_toggle(prompt: str, thinking: bool) -> str:
    """Prepend Qwen3's ``/think`` or ``/no_think`` prompt command.

    Shared across every backend (LlamaCpp, Ollama, OpenAICompat) because the
    semantics are identical regardless of how the model is reached. On Qwen3
    the chat template interprets the command at tokenisation time. On other
    model families the command degrades to literal text the model ignores —
    that is the deliberate fallback behaviour, not a bug.

    Args:
        prompt: User prompt to prepend the command to.
        thinking: True for ``/think`` (explicit reasoning trace), False for
            ``/no_think`` (direct answer).

    Returns:
        Prompt string with the command prepended on its own line.
    """
    command = "/think" if thinking else "/no_think"
    return f"{command}\n{prompt}"


__all__ = [
    "Backend",
    "GenerateResult",
    "apply_thinking_toggle",
]
