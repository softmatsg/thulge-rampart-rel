"""Tests for OpenAICompatBackend.

Heavier emphasis on header behaviour than ``test_ollama.py`` because the
``extra_headers`` merge is the load-bearing piece for endpoints that need
non-standard auth (notably Gemini's ``x-goog-api-key``).

Same MockTransport pattern as the Ollama tests: assertions check request
body shape and headers, not URL paths.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx
import pytest

from blockagent.backends.base import Backend, GenerateResult
from blockagent.backends.openai_compat import OpenAICompatBackend


def _capture_handler(
    response_body: dict[str, Any],
) -> tuple[list[httpx.Request], httpx.MockTransport]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=response_body)

    return captured, httpx.MockTransport(handler)


def _ok_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello world"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 4,
            "total_tokens": 11,
        },
    }


# --- ImportError path ------------------------------------------------------


class TestImportError:
    def test_constructor_raises_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "httpx", None)
        with pytest.raises(ImportError, match=r"rampart\[http\]"):
            OpenAICompatBackend(
                model="x", base_url="https://api.example.com/v1", api_key="k"
            )

    def test_error_message_names_the_dependency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "httpx", None)
        with pytest.raises(ImportError, match="httpx"):
            OpenAICompatBackend(
                model="x", base_url="https://api.example.com/v1", api_key="k"
            )


# --- request body shape (OpenAI chat-completions schema) -------------------


class TestRequestBody:
    def test_messages_have_user_role_and_prompt_content(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="gpt-4o",
            base_url="https://api.example.com/v1",
            api_key="key",
            transport=transport,
        )
        backend.generate("hi")

        body = json.loads(captured[0].content)
        assert body["model"] == "gpt-4o"
        assert isinstance(body["messages"], list)
        assert body["messages"][0]["role"] == "user"
        # Thinking-toggle prefix lives inside the user message content.
        assert body["messages"][0]["content"].endswith("hi")

    def test_max_tokens_field(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        backend.generate("hi", max_new_tokens=256)
        body = json.loads(captured[0].content)
        assert body["max_tokens"] == 256

    def test_stop_sequences_when_present(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        backend.generate("hi", stop_sequences=["</end>"])
        body = json.loads(captured[0].content)
        assert body["stop"] == ["</end>"]

    def test_no_stop_key_when_no_stop_sequences(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        backend.generate("hi")
        body = json.loads(captured[0].content)
        assert "stop" not in body

    def test_thinking_toggle_prepends_command(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        backend.generate("hi", thinking=True)
        body = json.loads(captured[0].content)
        assert body["messages"][0]["content"].startswith("/think\n")


# --- request headers and the extra_headers merge ---------------------------


class TestRequestHeaders:
    def test_default_authorization_header_present(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="my-key",
            transport=transport,
        )
        backend.generate("hi")
        headers = captured[0].headers
        assert headers["Authorization"] == "Bearer my-key"

    def test_content_type_is_application_json(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        backend.generate("hi")
        assert captured[0].headers["Content-Type"] == "application/json"

    def test_extra_headers_merged_into_request(self) -> None:
        # The Gemini case: server expects an x-goog-api-key header in
        # addition to (or instead of) the standard Authorization header.
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="gemini-2.5-pro",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key="google-key",
            extra_headers={"x-goog-api-key": "google-key"},
            transport=transport,
        )
        backend.generate("hi")
        headers = captured[0].headers
        assert headers["Authorization"] == "Bearer google-key"
        assert headers["x-goog-api-key"] == "google-key"

    def test_extra_headers_can_override_authorization(self) -> None:
        # Some endpoints don't accept Bearer; extra_headers wins on the
        # collision so callers can replace the default Authorization.
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="ignored",
            extra_headers={"Authorization": "Custom override-token"},
            transport=transport,
        )
        backend.generate("hi")
        assert (
            captured[0].headers["Authorization"]
            == "Custom override-token"
        )

    def test_no_extra_headers_when_none_passed(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        backend.generate("hi")
        # Sanity: no Google-specific header leaked from a prior test.
        assert "x-goog-api-key" not in captured[0].headers

    def test_extra_headers_dict_is_copied_not_aliased(self) -> None:
        # Mutating the caller's dict after construction must not change
        # what the backend sends. A shallow copy at __init__ time is the
        # contract.
        my_headers = {"x-custom": "v1"}
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            extra_headers=my_headers,
            transport=transport,
        )
        my_headers["x-custom"] = "MUTATED"
        backend.generate("hi")
        assert captured[0].headers["x-custom"] == "v1"


# --- response parsing ------------------------------------------------------


class TestResponseParsing:
    def test_returns_first_choice_message_content(self) -> None:
        _, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        assert backend.generate("hi") == "hello world"

    def test_returns_generate_result_with_usage(self) -> None:
        _, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        result = backend.generate("hi", return_usage=True)
        assert isinstance(result, GenerateResult)
        assert result.text == "hello world"
        assert result.prompt_tokens == 7
        assert result.completion_tokens == 4

    def test_missing_usage_block_yields_zero_counts(self) -> None:
        body = _ok_response()
        del body["usage"]
        _, transport = _capture_handler(body)
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        result = backend.generate("hi", return_usage=True)
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


# --- reasoning_content channel handling (llama-server, DeepSeek, o-series) -


def _reasoning_response(
    *, content: str = "", reasoning: str = "the trace"
) -> dict[str, Any]:
    """Response shape from a reasoning-capable OpenAI-compat server.

    llama.cpp's llama-server splits the assistant message into two
    fields when serving Qwen3 with the thinking template active:
    ``content`` for the user-facing answer and ``reasoning_content``
    for the trace. Same shape used by DeepSeek and OpenAI's o-series.
    """
    body = _ok_response()
    body["choices"][0]["message"]["content"] = content
    body["choices"][0]["message"]["reasoning_content"] = reasoning
    return body


class TestReasoningContentChannel:
    def test_thinking_false_returns_content_only_ignoring_reasoning(
        self,
    ) -> None:
        # Even if the server emitted reasoning_content (some servers do
        # this regardless of the prompt-level toggle), thinking=False
        # surfaces only the clean content channel. Symmetric with how
        # llama-server already routes Qwen3's <think> trace away from
        # `content` when /no_think is in the prompt.
        body = _reasoning_response(
            content="Paris", reasoning="some thinking"
        )
        _, transport = _capture_handler(body)
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        result = backend.generate("hi", thinking=False)
        assert result == "Paris"

    def test_thinking_true_wraps_reasoning_in_think_tags(self) -> None:
        # When the caller asked for the trace, surface it as the
        # conventional <think>...</think> wrapper concatenated with
        # the answer. Restores the single-stream Qwen3 format the
        # agent loop's diagnosis path expects.
        body = _reasoning_response(
            content="The answer is Paris.",
            reasoning="The user asked about France.",
        )
        _, transport = _capture_handler(body)
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        result = backend.generate("hi", thinking=True)
        assert result == (
            "<think>The user asked about France.</think>"
            "The answer is Paris."
        )

    def test_thinking_true_returns_just_reasoning_when_content_empty(
        self,
    ) -> None:
        # The case the live llama-server smoke test surfaced: the model
        # spent its whole token budget on the trace and never reached the
        # answer channel. We still want the trace returned so the agent
        # loop's diagnosis path can salvage something usable.
        body = _reasoning_response(
            content="", reasoning="The user asked about France."
        )
        _, transport = _capture_handler(body)
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        result = backend.generate("hi", thinking=True)
        assert result == "<think>The user asked about France.</think>"

    def test_thinking_true_no_reasoning_field_returns_content(self) -> None:
        # Plain OpenAI servers (no reasoning_content field at all)
        # still work — fall back to content under thinking=True with no
        # wrapper added.
        _, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        result = backend.generate("hi", thinking=True)
        assert result == "hello world"

    def test_thinking_true_empty_reasoning_does_not_emit_empty_tag(
        self,
    ) -> None:
        # Reasoning field present but blank: do not emit "<think></think>"
        # which would just be noise the downstream parser has to ignore.
        body = _reasoning_response(content="Paris", reasoning="")
        _, transport = _capture_handler(body)
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        result = backend.generate("hi", thinking=True)
        assert result == "Paris"
        assert "<think>" not in result

    def test_reasoning_propagates_via_return_usage_path(self) -> None:
        # The reasoning merge happens in _extract_text; verify it also
        # applies on the return_usage=True path so the GenerateResult.text
        # carries the trace.
        body = _reasoning_response(
            content="answer", reasoning="trace"
        )
        _, transport = _capture_handler(body)
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        result = backend.generate("hi", thinking=True, return_usage=True)
        assert isinstance(result, GenerateResult)
        assert result.text == "<think>trace</think>answer"


# --- system_prompt parameter ----------------------------------------------


class TestSystemPrompt:
    def test_no_system_prompt_means_no_system_message_in_body(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        backend.generate("hi")
        body = json.loads(captured[0].content)
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["user"]

    def test_system_prompt_emits_system_message_before_user(self) -> None:
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        backend.generate(
            "hi", system_prompt="Answer directly without reasoning traces."
        )
        body = json.loads(captured[0].content)
        assert body["messages"][0] == {
            "role": "system",
            "content": "Answer directly without reasoning traces.",
        }
        assert body["messages"][1]["role"] == "user"

    def test_empty_system_prompt_treated_as_absent(self) -> None:
        # `system_prompt=""` is falsy and should not emit an empty
        # system message — that would be wasted tokens with no signal.
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        backend.generate("hi", system_prompt="")
        body = json.loads(captured[0].content)
        assert [m["role"] for m in body["messages"]] == ["user"]

    def test_system_prompt_works_with_thinking_true(self) -> None:
        # The conditional "only suppress thinking when thinking=False"
        # lives in BlockAgent; the backend itself always honours the
        # provided system_prompt regardless of thinking. That keeps the
        # backend's responsibility narrow.
        captured, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        backend.generate("hi", system_prompt="Be concise.", thinking=True)
        body = json.loads(captured[0].content)
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "Be concise."


# --- HTTP error propagation ------------------------------------------------


class TestHTTPErrors:
    def test_non_2xx_raises_httpx_status_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "invalid api key"})

        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="bad",
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(httpx.HTTPStatusError):
            backend.generate("hi")


# --- Protocol satisfaction at runtime --------------------------------------


class TestProtocolSatisfaction:
    def test_openai_compat_backend_is_a_backend(self) -> None:
        _, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        assert isinstance(backend, Backend)


# --- close() tidies the client --------------------------------------------


class TestClose:
    def test_close_does_not_raise(self) -> None:
        _, transport = _capture_handler(_ok_response())
        backend = OpenAICompatBackend(
            model="m",
            base_url="https://api.example.com/v1",
            api_key="k",
            transport=transport,
        )
        backend.close()
