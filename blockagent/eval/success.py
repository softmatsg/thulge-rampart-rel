"""Success-evaluator interface and the LLM-backed default.

A success evaluator turns the (task, model output, observations) triple
into a binary judgement that drives whether the agent loop writes a
diagnostic agent block. The interface is a Protocol so applications with
deterministic success criteria (compile checks, exact-match assertions,
domain-specific verifiers) can plug in without subclassing.

The LLM evaluator is non-thinking by design — the success check is
mechanical, not meta-cognitive, and consuming thinking tokens for a
binary YES/NO would inflate cost on every task.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from blockagent.backends.base import Backend


@runtime_checkable
class SuccessEvaluator(Protocol):
    """Strategy interface for judging task success."""

    def evaluate(
        self,
        task_text: str,
        model_output: str,
        observations: list[str],
    ) -> bool:
        """Return True iff the task was completed successfully."""
        ...


class LLMEvaluator:
    """Asks the backend (in non-thinking mode) whether a task succeeded.

    Returns True iff the backend's response, after stripping and upper-
    casing, starts with ``"YES"``. Anything else — ``"NO"``, malformed
    answers, blank output, hallucinated explanations — is treated as
    failure. Failing closed is the right default for the agent's
    learning signal: a wrong "success" suppresses the diagnostic agent
    block that would have prevented the next failure, which is the
    expensive direction. A wrong "failure" merely wastes one diagnostic
    inference call.
    """

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def evaluate(
        self,
        task_text: str,
        model_output: str,
        observations: list[str],
    ) -> bool:
        """Generate a YES/NO judgement and return ``True`` for YES."""
        prompt = self._build_prompt(task_text, model_output, observations)
        # thinking=False explicit per the user-locked discipline:
        # success evaluation is mechanical and should never consume
        # thinking tokens.
        response = self.backend.generate(
            prompt,
            max_new_tokens=16,
            thinking=False,
        )
        return response.strip().upper().startswith("YES")

    @staticmethod
    def _build_prompt(
        task_text: str,
        model_output: str,
        observations: list[str],
    ) -> str:
        """Assemble the YES/NO judgement prompt."""
        obs_text = (
            "\n---\n".join(observations)
            if observations
            else "(no tool calls were made)"
        )
        return (
            "Decide whether the agent below completed the task. Answer "
            "with a single word: YES or NO. Do not explain.\n\n"
            f"TASK:\n{task_text}\n\n"
            f"AGENT OUTPUT:\n{model_output}\n\n"
            f"TOOL OBSERVATIONS:\n{obs_text}\n\n"
            "ANSWER (YES or NO):"
        )


__all__ = ["LLMEvaluator", "SuccessEvaluator"]
