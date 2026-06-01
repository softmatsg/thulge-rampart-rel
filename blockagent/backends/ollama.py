"""OllamaBackend — synchronous httpx wrapper for the Ollama HTTP API.

Three boundaries this module enforces:

* **No module-level httpx import.** ``from httpx import ...`` lives inside
  ``__init__`` and re-raises a typed ImportError pointing at
  ``pip install rampart[http]`` if httpx is not installed. The base
  ``Backend`` Protocol stays importable in environments with zero optional
  dependencies; only constructing this concrete backend pays the cost.

* **Synchronous httpx.Client, never AsyncClient.** The agent loop is
  fundamentally synchronous on a single GPU: model inference is blocking,
  tool execution is blocking, and there is no parallelism to gain from
  async on hardware that runs one inference at a time. Adding async for
  the HTTP backend would force await ceremony through the entire call
  stack with no benefit. If async support is needed later it is a clean
  addition; baking it in now creates obligations not yet justified.

* **Transport injection for testing.** The ``transport`` constructor
  parameter accepts an ``httpx.BaseTransport`` (typically
  ``httpx.MockTransport``) so tests can intercept requests at the
  Client.send level. Production callers leave it as ``None`` and httpx
  uses the real network transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from blockagent.backends.base import Backend, GenerateResult

if TYPE_CHECKING:
    import httpx


class OllamaBackend:
    """Backend talking to a local or remote Ollama server over HTTP.

    Posts to the Ollama ``/api/generate`` endpoint with ``stream=False``
    so a single request returns the full completion plus token-count
    metadata. Token counts come from Ollama's ``prompt_eval_count`` and
    ``eval_count`` fields.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        num_ctx: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Construct an OllamaBackend.

        Args:
            model: Ollama model identifier (e.g. ``"qwen3:4b"``).
            base_url: Ollama server URL. Default is the local install
                default (``http://localhost:11434``).
            timeout: Per-request timeout in seconds. Defaults to 120 to
                accommodate first-token latency on cold-loaded models.
            num_ctx: Ollama context window in tokens. Default ``None``
                lets Ollama use the Modelfile's value (4096 for stock
                Qwen3 / Llama Modelfiles). Set explicitly when the
                prompt may exceed the Modelfile default — Ollama
                silently truncates to the last ``num_ctx`` tokens of
                the prompt rather than erroring, so under-sized
                contexts produce confidently-wrong answers with no
                surface signal. For the position-sensitivity KV sweep
                a 36941-token haystack needs ``num_ctx=40960`` to
                fit qwen3:8b's full window.
            transport: Optional ``httpx.BaseTransport`` for tests.
                Production callers leave as ``None``; tests pass an
                ``httpx.MockTransport`` to intercept requests.

        Raises:
            ImportError: If httpx is not installed. The error message
                tells the caller exactly which extra to install.
        """
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "OllamaBackend requires httpx. "
                "Install with: pip install rampart[http]"
            ) from exc

        self.model = model
        self.base_url = base_url
        self.num_ctx = num_ctx
        self._client = _httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
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
        """Generate a completion via Ollama. See ``Backend.generate``."""
        options: dict[str, Any] = {"num_predict": max_new_tokens}
        if stop_sequences:
            options["stop"] = list(stop_sequences)
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx

        # Ollama 0.21+ exposes a top-level `think` boolean that toggles
        # the reasoning channel for thinking-capable models (Qwen3,
        # DeepSeek, etc.). The legacy /no_think and /think prompt
        # prefixes used by apply_thinking_toggle are silently ignored
        # by recent Qwen3 builds in Ollama, so we send the explicit
        # field instead.
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": thinking,
            "options": options,
        }
        if system_prompt:
            # Ollama's /api/generate accepts a top-level `system` field
            # that the server applies via the model's chat template.
            body["system"] = system_prompt

        response = self._client.post("/api/generate", json=body)
        response.raise_for_status()
        data = response.json()

        text = self._extract_text(data, thinking=thinking)
        if not return_usage:
            return text

        return GenerateResult(
            text=text,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
        )

    @staticmethod
    def _extract_text(data: dict[str, Any], *, thinking: bool) -> str:
        """Pull the response text out of Ollama's /api/generate JSON.

        With ``think: true`` Ollama splits the assistant output into two
        top-level fields: ``response`` for the user-facing answer and
        ``thinking`` for the reasoning trace. With ``think: false`` the
        ``thinking`` field is absent or empty and ``response`` carries
        the direct answer.

        Behaviour:

        * ``thinking=False``: return ``response`` only. Ollama has
          already suppressed the reasoning trace at the model level, so
          ``response`` is the direct answer with no ``<think>``
          contamination to strip.
        * ``thinking=True``: if ``thinking`` is present and non-empty,
          wrap it as ``<think>...</think>`` and prepend to ``response``.
          This restores the conventional Qwen3-style single-stream
          output the agent loop's diagnosis path expects.

        If both fields are missing or empty, returns the empty string.

        Args:
            data: Parsed JSON body of the /api/generate response.
            thinking: Whether the call was made with ``thinking=True``.

        Returns:
            Merged text string, possibly empty.
        """
        response_text = data.get("response") or ""
        if not thinking:
            return str(response_text)
        reasoning = data.get("thinking") or ""
        if reasoning:
            return f"<think>{reasoning}</think>{response_text}"
        return str(response_text)

    def close(self) -> None:
        """Close the underlying httpx client. Optional but tidy."""
        self._client.close()


# Structural type-check that OllamaBackend satisfies the Backend Protocol.
# Triggers a mypy error if the overload signatures drift from base.py.
_: type[Backend] = OllamaBackend


__all__ = [
    "OllamaBackend",
]
