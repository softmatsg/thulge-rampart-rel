"""Tests for LLMEvaluator.

Verifies the YES/NO parsing, the fail-closed default for malformed
responses, and that the evaluator passes ``thinking=False`` explicitly
on its inference call (per the user-locked discipline that mechanical
steps must not consume thinking tokens).
"""

from __future__ import annotations

from typing import Any

from blockagent.backends.base import GenerateResult
from blockagent.eval.success import LLMEvaluator, SuccessEvaluator


class _ScriptedBackend:
    """Backend stub returning a fixed string and recording call kwargs."""

    def __init__(self, response: str = "YES") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

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
        if return_usage:
            return GenerateResult(
                text=self.response, prompt_tokens=10, completion_tokens=2
            )
        return self.response


class TestLLMEvaluatorJudgement:
    def test_yes_returns_true(self) -> None:
        ev = LLMEvaluator(_ScriptedBackend("YES"))
        assert ev.evaluate("task", "output", []) is True

    def test_yes_with_trailing_text_still_true(self) -> None:
        ev = LLMEvaluator(_ScriptedBackend("YES, the agent finished."))
        assert ev.evaluate("task", "output", []) is True

    def test_lowercase_yes_returns_true(self) -> None:
        ev = LLMEvaluator(_ScriptedBackend("yes"))
        assert ev.evaluate("task", "output", []) is True

    def test_yes_with_leading_whitespace_returns_true(self) -> None:
        ev = LLMEvaluator(_ScriptedBackend("\n  YES "))
        assert ev.evaluate("task", "output", []) is True

    def test_no_returns_false(self) -> None:
        ev = LLMEvaluator(_ScriptedBackend("NO"))
        assert ev.evaluate("task", "output", []) is False

    def test_malformed_response_returns_false(self) -> None:
        # Anything not starting with YES is treated as failure. Failing
        # closed is the right default for the agent's learning signal:
        # missing a real success wastes one diagnosis call, but a wrong
        # success suppresses an agent block that would have prevented
        # the next failure.
        ev = LLMEvaluator(_ScriptedBackend("Maybe? It depends."))
        assert ev.evaluate("task", "output", []) is False

    def test_empty_response_returns_false(self) -> None:
        ev = LLMEvaluator(_ScriptedBackend(""))
        assert ev.evaluate("task", "output", []) is False


class TestLLMEvaluatorThinkingDiscipline:
    def test_evaluator_call_uses_thinking_false_explicitly(self) -> None:
        backend = _ScriptedBackend("YES")
        ev = LLMEvaluator(backend)
        ev.evaluate("task", "output", [])
        assert backend.calls[0]["thinking"] is False


class TestLLMEvaluatorPromptShape:
    def test_prompt_includes_task_output_and_observations(self) -> None:
        backend = _ScriptedBackend("YES")
        ev = LLMEvaluator(backend)
        ev.evaluate(
            "do the thing",
            "I did the thing",
            ["STDOUT: thing-done"],
        )
        prompt = backend.calls[0]["prompt"]
        assert "do the thing" in prompt
        assert "I did the thing" in prompt
        assert "STDOUT: thing-done" in prompt

    def test_no_observations_uses_explicit_marker(self) -> None:
        # The marker matters because a blank observation block looks
        # like a tool call with no output, which is a different signal
        # from "no tool was called".
        backend = _ScriptedBackend("YES")
        ev = LLMEvaluator(backend)
        ev.evaluate("task", "output", [])
        prompt = backend.calls[0]["prompt"]
        assert "no tool calls were made" in prompt


class TestSuccessEvaluatorProtocol:
    def test_llm_evaluator_satisfies_protocol(self) -> None:
        ev: SuccessEvaluator = LLMEvaluator(_ScriptedBackend("YES"))
        assert isinstance(ev, SuccessEvaluator)
