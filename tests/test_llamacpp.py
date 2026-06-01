"""Tests for LlamaCppBackend.

Three test surfaces:

* The ImportError path. llama-cpp-python is not installed in the dev
  environment, so a bare ``LlamaCppBackend(model_path=...)`` would raise
  the typed ImportError naturally. To keep the test stable across
  environments where the user might install the optional extra, the
  test patches ``sys.modules['llama_cpp'] = None`` to force the import
  to fail regardless of installed state.
* The behavioural path. Tests use ``monkeypatch.setitem(sys.modules,
  'llama_cpp', fake_module)`` to inject a deterministic ``_FakeLlama``
  that records every call. This pins the wire-level contract
  (``stop`` mapped from ``stop_sequences``, ``max_tokens`` from
  ``max_new_tokens``, ``seed`` flowed through, thinking-toggle prompt
  prefix applied) without ever loading a real GGUF model.
* The Backend Protocol satisfaction is verified by a class-level type
  alias inside ``llamacpp.py`` itself; that line would fail mypy if the
  overload signatures drifted from ``base.py``.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from blockagent.backends.base import Backend, GenerateResult
from blockagent.backends.llamacpp import LlamaCppBackend


class _FakeLlama:
    """Deterministic stand-in for ``llama_cpp.Llama``.

    Records every constructor and call, so tests can assert exactly which
    parameters were forwarded. Returns the OpenAI-compatible completion
    dict shape that real llama-cpp-python emits.
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401  # mocks llama_cpp.Llama untyped kwargs
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401  # mocks llama_cpp.Llama untyped kwargs
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stop": stop,
                **kwargs,
            }
        )
        return {
            "choices": [
                {
                    "text": f"echo: {prompt}",
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 7,
                "total_tokens": len(prompt.split()) + 7,
            },
        }


def _install_fake_llama_cpp(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLlama]:
    """Inject a fake ``llama_cpp`` module into ``sys.modules``."""
    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    return _FakeLlama


def _backend(
    monkeypatch: pytest.MonkeyPatch,
    **overrides: Any,  # noqa: ANN401  # forwarded to LlamaCppBackend's heterogeneous kwargs
) -> LlamaCppBackend:
    """Build a LlamaCppBackend with the fake llama-cpp-python active."""
    _install_fake_llama_cpp(monkeypatch)
    kwargs: dict[str, Any] = {"model_path": "fake.gguf"}
    kwargs.update(overrides)
    return LlamaCppBackend(**kwargs)


# --- ImportError path ------------------------------------------------------


class TestImportError:
    def test_constructor_raises_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Setting sys.modules['llama_cpp'] = None is the documented way to
        # force `from llama_cpp import ...` to raise ModuleNotFoundError
        # regardless of whether the package is actually installed.
        monkeypatch.setitem(sys.modules, "llama_cpp", None)
        with pytest.raises(ImportError, match=r"rampart\[local\]"):
            LlamaCppBackend(model_path="anywhere.gguf")

    def test_error_message_names_the_dependency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "llama_cpp", None)
        with pytest.raises(ImportError, match="llama-cpp-python"):
            LlamaCppBackend(model_path="anywhere.gguf")


# --- constructor wires Llama() correctly -----------------------------------


class TestConstructor:
    def test_default_kwargs_passed_to_llama(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(monkeypatch)
        assert isinstance(backend._model, _FakeLlama)
        kw = backend._model.init_kwargs
        assert kw["model_path"] == "fake.gguf"
        assert kw["n_gpu_layers"] == 0
        assert kw["n_ctx"] == 4096
        assert kw["seed"] == -1  # None → -1 sentinel for llama-cpp-python
        assert kw["verbose"] is False

    def test_custom_kwargs_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(
            monkeypatch,
            model_path="qwen3.gguf",
            n_gpu_layers=20,
            n_ctx=8192,
            seed=42,
        )
        kw = backend._model.init_kwargs
        assert kw["model_path"] == "qwen3.gguf"
        assert kw["n_gpu_layers"] == 20
        assert kw["n_ctx"] == 8192
        assert kw["seed"] == 42


# --- generate(): default str return ----------------------------------------


class TestGenerateStr:
    def test_returns_str_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(monkeypatch)
        result = backend.generate("hello")
        assert isinstance(result, str)
        assert "hello" in result

    def test_max_new_tokens_maps_to_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(monkeypatch)
        backend.generate("p", max_new_tokens=128)
        call = backend._model.calls[0]
        assert call["max_tokens"] == 128

    def test_stop_sequences_passed_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(monkeypatch)
        backend.generate("p", stop_sequences=["</end>", "###"])
        call = backend._model.calls[0]
        assert call["stop"] == ["</end>", "###"]

    def test_no_stop_sequences_passes_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # llama-cpp-python accepts [] as "no extra stops"; passing None
        # could be interpreted differently. Pin the empty-list contract.
        backend = _backend(monkeypatch)
        backend.generate("p")
        call = backend._model.calls[0]
        assert call["stop"] == []


# --- generate(): return_usage=True path ------------------------------------


class TestGenerateUsage:
    def test_returns_generate_result_when_usage_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(monkeypatch)
        result = backend.generate("hello world", return_usage=True)
        assert isinstance(result, GenerateResult)
        assert "hello world" in result.text
        # _FakeLlama emits prompt_tokens = number of whitespace-split words.
        # "/no_think\nhello world" → 3 words.
        assert result.prompt_tokens == 3
        assert result.completion_tokens == 7

    def test_missing_usage_block_yields_zero_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Some llama-cpp-python builds omit the usage dict in particular
        # generation modes; the backend treats missing counts as zero
        # rather than crashing the whole call.
        class _FakeLlamaNoUsage:
            def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
                self.kwargs = kwargs

            def __call__(
                self,
                prompt: str,
                max_tokens: int = 512,
                stop: list[str] | None = None,
            ) -> dict[str, Any]:
                return {
                    "choices": [
                        {"text": "no usage here", "finish_reason": "stop"}
                    ],
                }

        fake_module = types.ModuleType("llama_cpp")
        fake_module.Llama = _FakeLlamaNoUsage  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

        backend = LlamaCppBackend(model_path="x")
        result = backend.generate("p", return_usage=True)
        assert result.text == "no usage here"
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


# --- thinking toggle -------------------------------------------------------


class TestThinkingToggle:
    def test_thinking_false_prepends_no_think_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(monkeypatch)
        backend.generate("hello", thinking=False)
        call = backend._model.calls[0]
        assert call["prompt"].startswith("/no_think\n")
        assert call["prompt"].endswith("hello")

    def test_thinking_true_prepends_think_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(monkeypatch)
        backend.generate("hello", thinking=True)
        call = backend._model.calls[0]
        assert call["prompt"].startswith("/think\n")
        assert call["prompt"].endswith("hello")

    def test_default_thinking_is_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(monkeypatch)
        backend.generate("hello")
        call = backend._model.calls[0]
        assert call["prompt"].startswith("/no_think\n")


class TestSystemPrompt:
    def test_system_prompt_prepended_with_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Raw text-completion API has no system-role slot. The backend
        # surfaces the instruction as a "SYSTEM: ..." marker prepended
        # ahead of the thinking-toggle command and the user prompt.
        backend = _backend(monkeypatch)
        backend.generate("hello", system_prompt="Be concise.")
        prompt = backend._model.calls[0]["prompt"]
        assert prompt.startswith("SYSTEM: Be concise.\n\n")
        # The thinking-toggle command still appears after the marker.
        assert "/no_think\n" in prompt
        assert prompt.endswith("hello")

    def test_no_marker_when_system_prompt_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(monkeypatch)
        backend.generate("hello")
        prompt = backend._model.calls[0]["prompt"]
        assert not prompt.startswith("SYSTEM:")

    def test_empty_system_prompt_treated_as_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(monkeypatch)
        backend.generate("hello", system_prompt="")
        prompt = backend._model.calls[0]["prompt"]
        assert not prompt.startswith("SYSTEM:")


# --- Protocol satisfaction at runtime --------------------------------------


class TestProtocolSatisfaction:
    def test_llamacpp_backend_is_a_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend(monkeypatch)
        assert isinstance(backend, Backend)
