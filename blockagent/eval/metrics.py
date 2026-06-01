"""TaskResult — the shape returned by ``BlockAgent.run_task``.

Defined before the agent loop is implemented so the loop has a concrete
output contract to build toward. Frozen with ``slots=True`` so consumers
can pass results through ``SessionReport`` aggregation without defensive
copy and so accidental mutation of accumulated metrics fails loudly
rather than silently corrupting the learning curve.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Result of one ``BlockAgent.run_task`` invocation.

    Attributes:
        task_text: The original task string the agent was given.
        success: Whether the success evaluator judged the task complete.
        compiled_context_tokens: Token count of the assembled context that
            was actually sent to the model (after the relevance gate, the
            ordered walk, and the budget cutoff). Useful for the per-task
            efficiency metric in benchmarks.
        tool_calls_made: Number of tool calls extracted from the model's
            output and dispatched. Includes calls that returned errors.
        agent_block_written: True iff the loop reached the
            ``write_block_on_failure`` step and successfully appended a
            heuristic block to the registry. Always False when ``success``
            is True.
        trajectory_id: UUID assigned at the start of this run. Tags any
            agent block written during the run so ``rollback_trajectory``
            can later remove them as a unit.
        prompt_tokens: Backend-reported prompt tokens for the task call
            (not the success-evaluator or diagnosis calls). Counted by the
            backend itself, not re-tokenised, so it reflects exactly what
            the model paid.
        completion_tokens: Backend-reported completion tokens for the task
            call. Same accounting note as ``prompt_tokens``.
    """

    task_text: str
    success: bool
    compiled_context_tokens: int
    tool_calls_made: int
    agent_block_written: bool
    trajectory_id: str
    prompt_tokens: int
    completion_tokens: int


__all__ = ["TaskResult"]
