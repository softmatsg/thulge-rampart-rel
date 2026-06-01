"""Tests for OllamaBackend.

Mock at the ``httpx.Client.send`` level via ``httpx.MockTransport`` so
the test catches body-shape and header bugs without coupling to the
exact URL path the backend constructs. A typo in the URL path that
hits the mock is still a real failure mode at runtime against a real
Ollama server, but pinning the URL in tests creates brittleness when
endpoints version or move.

The ImportError path uses ``sys.modules['httpx'] = None`` to force the
import inside ``OllamaBackend.__init__`` to fail regardless of whether
httpx is actually installed in the test environment.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx
import pytest

from blockagent.backends.base import Backend, GenerateResult
from blockagent.backends.ollama import OllamaBackend


def _capture_handler(
    response_body: dict[str, Any],
) -> tuple[list[httpx.Request], httpx.MockTransport]:
    """Build a MockTransport that records every request it receives."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=response_body)

    return captured, httpx.MockTransport(handler)


def _ok_response() -> dict[str, Any]:
    return {
        "model": "qwen3:4b",
        "response": "hello world",
        "done": True,
        "prompt_eval_count": 5,
        "eval_count": 3,
    }


# --- ImportError path ------------------------------------------------------


class TestImportError:
    def test_constructor_raises_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force `import httpx` inside __init__ to fail regardless of
        # whether httpx is actually installed in the dev environment.
        monkeypatch.setitem(sys.modules, "httpx", None)
        with pytest.raises(ImportError, match=r"rampart\[http\]"):
            OllamaBackend(model="qwen3:4b")

    def test_error_message_names_the_dependency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "httpx", None)
        with pytest.raises(ImportError, match="httpx"):
            OllamaBackend(model="qwen3:4b")


# --- request body shape ----------------------------------------------------


class TestRequestBody:
    def test_post_carries_model_and_prompt(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        backend.generate("hi")

        assert len(captured) == 1
        request = captured[0]
        assert request.method == "POST"
        body = json.loads(request.content)
        assert body["model"] == "qwen3:4b"
        # Prompt is sent verbatim. Thinking is now controlled by the
        # top-level `think` boolean (see TestThinkingFieldToggle), not
        # by a /no_think or /think prompt prefix.
        assert body["prompt"] == "hi"

    def test_stream_false_in_body(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        backend.generate("hi")
        body = json.loads(captured[0].content)
        # Without stream=False Ollama would return a chunked JSONL stream
        # which the backend cannot parse with response.json().
        assert body["stream"] is False

    def test_max_new_tokens_maps_to_options_num_predict(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        backend.generate("hi", max_new_tokens=128)
        body = json.loads(captured[0].content)
        assert body["options"]["num_predict"] == 128

    def test_stop_sequences_map_to_options_stop(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        backend.generate("hi", stop_sequences=["</end>", "###"])
        body = json.loads(captured[0].content)
        assert body["options"]["stop"] == ["</end>", "###"]

    def test_no_stop_key_when_stop_sequences_empty(self) -> None:
        # Pinning that the backend does not send `stop: []`, since some
        # servers treat an empty list differently from "no stop sequences".
        captured, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        backend.generate("hi")
        body = json.loads(captured[0].content)
        assert "stop" not in body["options"]

    def test_thinking_false_sends_top_level_think_field_false(self) -> None:
        # Pinned wire format: with recent Qwen3 builds in Ollama the
        # legacy /no_think prompt prefix is silently ignored; only the
        # top-level `think` boolean actually disables the reasoning
        # channel. Locking this in so a future revert to
        # apply_thinking_toggle does not regress.
        captured, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        backend.generate("hi", thinking=False)
        body = json.loads(captured[0].content)
        assert body["think"] is False
        assert body["prompt"] == "hi"  # no /no_think prefix
        assert "/no_think" not in body["prompt"]

    def test_thinking_true_sends_top_level_think_field_true(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        backend.generate("hi", thinking=True)
        body = json.loads(captured[0].content)
        assert body["think"] is True
        assert body["prompt"] == "hi"  # no /think prefix
        assert "/think" not in body["prompt"]

    def test_system_prompt_emitted_as_top_level_system_field(self) -> None:
        # Ollama's /api/generate accepts a top-level `system` field that
        # the server applies via the model's chat template. Pin that
        # mapping rather than embedding the system text inside the
        # prompt body.
        captured, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        backend.generate("hi", system_prompt="Be concise.")
        body = json.loads(captured[0].content)
        assert body["system"] == "Be concise."

    def test_no_system_field_when_system_prompt_absent(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        backend.generate("hi")
        body = json.loads(captured[0].content)
        assert "system" not in body

    def test_empty_system_prompt_treated_as_absent(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        backend.generate("hi", system_prompt="")
        body = json.loads(captured[0].content)
        assert "system" not in body


# --- response parsing ------------------------------------------------------


class TestResponseParsing:
    def test_returns_string_by_default(self) -> None:
        _, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        assert backend.generate("hi") == "hello world"

    def test_returns_generate_result_with_token_counts(self) -> None:
        _, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        result = backend.generate("hi", return_usage=True)
        assert isinstance(result, GenerateResult)
        assert result.text == "hello world"
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 3

    def test_missing_token_counts_yield_zero(self) -> None:
        # Some Ollama versions / endpoints omit the eval_count fields in
        # certain modes. Backend treats absence as zero rather than crash.
        _, transport = _capture_handler({"response": "x", "done": True})
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        result = backend.generate("hi", return_usage=True)
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


# --- reasoning-channel merge (Ollama splits response/thinking) -------------


class TestReasoningChannel:
    """Ollama's /api/generate splits assistant output into two top-level
    fields when ``think: true``: ``response`` (user-facing answer) and
    ``thinking`` (reasoning trace). With ``think: false`` the
    ``thinking`` field is absent or empty and ``response`` carries the
    direct answer.
    """

    def test_thinking_false_returns_response_only(self) -> None:
        body = {
            "response": "Paris.",
            "thinking": "",  # think:false suppresses; field may be absent or empty
            "done": True,
        }
        _, transport = _capture_handler(body)
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        text = backend.generate("capital?", thinking=False)
        assert text == "Paris."

    def test_thinking_false_ignores_thinking_field_if_present(self) -> None:
        # If a server quirk leaves `thinking` populated despite
        # think:false, the non-thinking branch must not surface it.
        body = {
            "response": "Paris.",
            "thinking": "leaked reasoning that should not appear",
            "done": True,
        }
        _, transport = _capture_handler(body)
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        text = backend.generate("capital?", thinking=False)
        assert text == "Paris."

    def test_thinking_true_merges_thinking_into_think_wrapper(self) -> None:
        body = {
            "response": "Paris.",
            "thinking": "France is in Europe; its capital is Paris.",
            "done": True,
        }
        _, transport = _capture_handler(body)
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        text = backend.generate("capital?", thinking=True)
        assert text == (
            "<think>France is in Europe; its capital is Paris.</think>Paris."
        )

    def test_thinking_true_falls_back_to_response_when_thinking_empty(
        self,
    ) -> None:
        # Some calls never enter the reasoning channel (trivial prompts);
        # the wrapper must not appear with empty content.
        body = {
            "response": "Paris.",
            "thinking": "",
            "done": True,
        }
        _, transport = _capture_handler(body)
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        text = backend.generate("capital?", thinking=True)
        assert text == "Paris."

    def test_thinking_true_with_missing_thinking_key_returns_response(
        self,
    ) -> None:
        # Older Ollama versions / non-thinking models omit the field
        # entirely rather than emitting an empty string.
        body = {"response": "Paris.", "done": True}
        _, transport = _capture_handler(body)
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        text = backend.generate("capital?", thinking=True)
        assert text == "Paris."

    def test_both_fields_empty_returns_empty_string(self) -> None:
        # Pinned because in production a reasoning model can exhaust
        # its budget inside the thinking block, leaving response=='' and
        # thinking populated. Without the fallback this branch returned
        # '' silently and broke downstream code-extraction; with
        # thinking=False the response should arrive populated, but the
        # empty-fallback branch still needs explicit coverage.
        body = {"response": "", "thinking": "", "done": True}
        _, transport = _capture_handler(body)
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        text = backend.generate("anything", thinking=False)
        assert text == ""


# --- HTTP error propagation ------------------------------------------------


class TestHTTPErrors:
    def test_non_2xx_raises_httpx_status_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "model not loaded"})

        backend = OllamaBackend(
            model="qwen3:4b", transport=httpx.MockTransport(handler)
        )
        with pytest.raises(httpx.HTTPStatusError):
            backend.generate("hi")


# --- Protocol satisfaction at runtime --------------------------------------


class TestProtocolSatisfaction:
    def test_ollama_backend_is_a_backend(self) -> None:
        _, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        assert isinstance(backend, Backend)


# --- close() tidies the client --------------------------------------------


class TestClose:
    def test_close_does_not_raise(self) -> None:
        _, transport = _capture_handler(_ok_response())
        backend = OllamaBackend(model="qwen3:4b", transport=transport)
        backend.close()
        # Idempotent — second close is fine too.
        backend.close()
