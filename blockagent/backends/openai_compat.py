"""OpenAICompatBackend — sync httpx wrapper for any OpenAI-compatible endpoint.

Tested call sites: Gemini via Google's OpenAI compatibility layer,
OpenRouter, vLLM-served local models, llama.cpp's ``llama-server``,
and the OpenAI API itself. The backend is genuinely generic — anything
that speaks the standard ``POST /chat/completions`` contract works.

Two non-obvious decisions:

* **``extra_headers`` is a first-class constructor parameter.** Some
  providers (most notably Gemini) require provider-specific authentication
  headers in addition to (or instead of) the standard
  ``Authorization: Bearer`` header. Hard-coding either assumption would
  ship a backend that "works for OpenAI but quietly 401s on Gemini". The
  ``extra_headers`` dict is merged into the httpx Client's default
  headers at construction time and applied to every request.
* **Synchronous httpx.Client, never AsyncClient.** Same reasoning as
  ``OllamaBackend``: the agent loop is sync, single-GPU inference is
  one-at-a-time, async would force ceremony with no parallelism payoff.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from blockagent.backends.base import Backend, GenerateResult, apply_thinking_toggle

if TYPE_CHECKING:
    import httpx


class OpenAICompatBackend:
    """Backend for any OpenAI-compatible chat-completions endpoint.

    Posts to ``{base_url}/chat/completions`` with the standard OpenAI
    request shape: ``{ model, messages, max_tokens, stop }``. Parses
    the standard response shape: ``choices[0].message.content`` for
    text and ``usage.prompt_tokens`` / ``usage.completion_tokens`` for
    accounting.

    Authentication defaults to ``Authorization: Bearer {api_key}``,
    which is what the OpenAI API itself, OpenRouter, and most
    OpenAI-compat layers expect. Providers with non-standard auth pass
    their requirements via ``extra_headers``.

    Example for Gemini 2.5 Pro via Google's OpenAI compat endpoint —
    if ``Authorization: Bearer`` alone is not accepted, Google also
    accepts an ``x-goog-api-key`` header::

        OpenAICompatBackend(
            model="gemini-2.5-pro",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key=key,
            extra_headers={"x-goog-api-key": key},
        )

    The two headers do not conflict; sending both is the safest default
    when uncertain which the endpoint requires.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Construct an OpenAICompatBackend.

        Args:
            model: Model identifier as the endpoint expects it (e.g.
                ``"gpt-4o"``, ``"gemini-2.5-pro"``).
            base_url: Endpoint root WITHOUT the ``/chat/completions``
                suffix (e.g. ``"https://api.openai.com/v1"``).
            api_key: Bearer token used in the default ``Authorization``
                header. Pass an empty string if your endpoint takes auth
                exclusively via ``extra_headers``.
            timeout: Per-request timeout in seconds. Defaults to 120 to
                accommodate frontier-model first-token latency.
            extra_headers: Provider-specific headers merged into every
                request. Use this for ``x-goog-api-key`` (Gemini),
                custom organisation headers, etc. Headers here override
                the default ``Authorization`` header if they share the
                same key.
            transport: Optional ``httpx.BaseTransport`` for tests.
                Production callers leave as ``None``.

        Raises:
            ImportError: If httpx is not installed. The error message
                tells the caller exactly which extra to install.
        """
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatBackend requires httpx. "
                "Install with: pip install rampart[http]"
            ) from exc

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.extra_headers = dict(extra_headers) if extra_headers else {}

        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Extra headers win on key collision so callers can override the
        # default Authorization header for endpoints that reject it.
        headers.update(self.extra_headers)

        self._client = _httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
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
        """Generate a completion via the chat-completions endpoint."""
        effective_prompt = apply_thinking_toggle(prompt, thinking)

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": effective_prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
        }
        if stop_sequences:
            body["stop"] = list(stop_sequences)

        response = self._client.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()

        text = self._extract_text(data, thinking=thinking)
        if not return_usage:
            return text

        usage = data.get("usage") or {}
        return GenerateResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    @staticmethod
    def _extract_text(data: dict[str, Any], *, thinking: bool) -> str:
        """Pull the response text out of the chat-completions JSON.

        Reasoning-capable servers (llama.cpp's llama-server, DeepSeek,
        OpenAI's o-series models, etc.) split the assistant message into
        two fields: ``content`` for the user-facing answer and
        ``reasoning_content`` for the thinking trace. Plain OpenAI-style
        servers return only ``content``.

        Behaviour:

        * ``thinking=False``: return ``content`` only. Reasoning servers
          have already routed the trace into ``reasoning_content``, so
          ``content`` is clean — no ``<think>`` contamination to strip.
        * ``thinking=True``: if ``reasoning_content`` is present and
          non-empty, wrap it as ``<think>...</think>`` and prepend to
          ``content``. This restores the conventional Qwen3-style
          single-stream output the agent loop's diagnosis path expects.
          The leading newline that llama-server emits at the start of
          its reasoning is preserved inside the wrapper.

        If both fields are missing or empty, returns the empty string.
        """
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        if not thinking:
            return str(content)
        reasoning = message.get("reasoning_content") or ""
        if reasoning:
            return f"<think>{reasoning}</think>{content}"
        return str(content)

    def close(self) -> None:
        """Close the underlying httpx client. Optional but tidy."""
        self._client.close()


# Structural type-check that OpenAICompatBackend satisfies Backend.
# Triggers a mypy error if the overload signatures drift from base.py.
_: type[Backend] = OpenAICompatBackend


__all__ = [
    "OpenAICompatBackend",
]
