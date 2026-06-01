"""Tests for the Backend Protocol and the GenerateResult value type.

This module imports ``blockagent.backends.base`` directly to verify the
contract is reachable with zero optional dependencies installed —
llama-cpp-python and httpx are both absent in the dev environment, so
any accidental module-level import in the chain would surface as an
ImportError here.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import pytest

from blockagent.backends.base import Backend, GenerateResult

# --- GenerateResult dataclass ----------------------------------------------


class TestGenerateResult:
    def test_holds_text_and_token_counts(self) -> None:
        r = GenerateResult(
            text="hello", prompt_tokens=12, completion_tokens=3
        )
        assert r.text == "hello"
        assert r.prompt_tokens == 12
        assert r.completion_tokens == 3

    def test_is_frozen(self) -> None:
        # Frozen dataclasses raise FrozenInstanceError on attribute
        # assignment. The point is to keep returned usage counts
        # immutable so consumers can pass the result through the session
        # report without defensive copy.
        r = GenerateResult(text="x", prompt_tokens=1, completion_tokens=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.text = "mutated"  # type: ignore[misc]

    def test_equality_is_value_based(self) -> None:
        a = GenerateResult(text="x", prompt_tokens=1, completion_tokens=2)
        b = GenerateResult(text="x", prompt_tokens=1, completion_tokens=2)
        c = GenerateResult(text="y", prompt_tokens=1, completion_tokens=2)
        assert a == b
        assert a != c


# --- Backend Protocol structural satisfaction -------------------------------


class _FakeBackend:
    """Minimal Backend implementation used to verify the Protocol shape.

    Defines the same overload signatures as the Protocol so a mypy-strict
    structural check accepts it. The runtime body returns deterministic
    output keyed off ``return_usage``.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        stop_sequences: list[str] | None = None,
        thinking: bool = False,
        system_prompt: str | None = None,
        return_usage: bool = False,
    ) -> str | GenerateResult:
        self.calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "stop_sequences": stop_sequences,
                "thinking": thinking,
                "system_prompt": system_prompt,
                "return_usage": return_usage,
            }
        )
        text = f"reply to: {prompt[:20]}"
        if return_usage:
            return GenerateResult(
                text=text,
                prompt_tokens=len(prompt.split()),
                completion_tokens=4,
            )
        return text


class TestBackendProtocol:
    def test_fake_backend_satisfies_protocol_at_runtime(self) -> None:
        backend: Backend = _FakeBackend()
        assert isinstance(backend, Backend)

    def test_default_call_returns_str(self) -> None:
        backend: Backend = _FakeBackend()
        result = backend.generate("hello")
        assert isinstance(result, str)

    def test_return_usage_true_returns_generate_result(self) -> None:
        backend: Backend = _FakeBackend()
        result = backend.generate("hello", return_usage=True)
        assert isinstance(result, GenerateResult)
        assert result.text.startswith("reply to:")
        assert result.prompt_tokens >= 1
        assert result.completion_tokens == 4

    def test_kwargs_propagate_to_implementation(self) -> None:
        backend = _FakeBackend()
        backend.generate(
            "p",
            max_new_tokens=128,
            stop_sequences=["</end>"],
            thinking=True,
        )
        call = backend.calls[0]
        assert call["max_new_tokens"] == 128
        assert call["stop_sequences"] == ["</end>"]
        assert call["thinking"] is True
        assert call["return_usage"] is False


# --- import isolation ------------------------------------------------------


class TestImportIsolation:
    def test_base_module_imports_with_no_optional_deps(self) -> None:
        # If base.py grew an accidental top-level import of an optional
        # dependency, this test would fail at module import time with
        # an ImportError (since llama-cpp-python is not installed in
        # the dev/test environment). The fact that the import succeeded
        # at the top of the file is the test; the assertion is a sanity
        # check that the symbols we need are present.
        from blockagent.backends import base

        assert hasattr(base, "Backend")
        assert hasattr(base, "GenerateResult")

    def test_overload_discrimination_via_literal(self) -> None:
        # Sanity-check that the Literal[True/False] overloads are imported
        # cleanly. The overloads exist for mypy; at runtime, both call
        # paths execute the same body. This test pins that the symbols
        # the Protocol relies on are still importable.
        backend: Backend = _FakeBackend()
        false_path: Literal[False] = False
        true_path: Literal[True] = True
        assert isinstance(backend.generate("p", return_usage=false_path), str)
        assert isinstance(
            backend.generate("p", return_usage=true_path), GenerateResult
        )
