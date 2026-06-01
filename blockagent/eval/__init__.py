"""Agent-facing result type and the evaluator interface.

``TaskResult`` is the shape ``BlockAgent.run_task`` returns;
``SuccessEvaluator`` is the protocol a caller implements to turn a
(task, model output, observations) triple into a binary success flag
that drives the agent's write-back-on-failure loop. ``LLMEvaluator``
is the LLM-backed default for users without a domain-specific checker.

Domain-specific evaluators (compile checks, environment wrappers,
exact-match assertions) are caller-implementations of the protocol
and are not shipped here; this module exports only the interfaces.
"""

from blockagent.eval.metrics import TaskResult
from blockagent.eval.success import LLMEvaluator, SuccessEvaluator

__all__ = [
    "LLMEvaluator",
    "SuccessEvaluator",
    "TaskResult",
]
