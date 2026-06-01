r"""BlockAgent — the agent loop that ties RAMPART to a backend and tools.

Three load-bearing decisions live in this module:

* **``parse_tool_calls`` is a module-level function, not a method.** The
  parsing of structured tool calls out of model output is a pure
  text-to-data transformation; making it a method would couple it to a
  ``BlockAgent`` instance for no benefit and force test code to spin up
  an agent just to test the parser. The function only handles JSON
  fenced in ``\`\`\`json`` or ``\`\`\`tool_call`` code blocks; other
  formats can be parsed by the caller's own function and passed in via
  composition.

* **Thinking-mode discipline is explicit at every call site.** The
  task call passes ``thinking=False`` and the diagnosis call passes
  ``thinking=True``. Neither relies on the backend's default — the rule
  (non-thinking for mechanical steps, thinking only when meta-cognition
  is needed) is surfaced where a code reviewer can see it.

* **Diagnosis prompt includes all four pieces of context.** The original
  task, the compiled context the model saw, the model output that
  failed, and the tool observations. The prompt asks for a single
  actionable rule (not a description of what went wrong). A diagnosis
  written without the four pieces produces a generic "be more careful"
  block that adds noise to the registry without preventing recurrence.

The agent itself is a thin orchestrator: it composes the registry,
backend, tool dispatcher, and success evaluator and walks them through
the eight-stage loop documented in ``run_task``. None of the stages
live here in detail — they all delegate to the right module.
"""

from __future__ import annotations

import json
import re
import uuid

from blockagent.backends.base import Backend
from blockagent.eval.metrics import TaskResult
from blockagent.eval.success import SuccessEvaluator
from blockagent.tools.base import ToolCall, ToolDispatcher
from rampart.registry import BlockRegistry

_TOOL_CALL_PATTERN = re.compile(
    r"```(?:tool_call|json)\s*\n(.*?)\n```",
    re.DOTALL,
)
# Match a complete <think>...</think> block (including the tags). Used to
# strip reasoning traces out of diagnosis output before it lands in the
# registry — see _write_block_on_failure. Non-greedy so multiple blocks in
# one response are stripped individually rather than collapsed.
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_tool_calls(model_output: str) -> list[ToolCall]:
    r"""Extract tool calls from a model's text output.

    Looks for ``\`\`\`json`` or ``\`\`\`tool_call`` fenced blocks
    containing JSON objects with a string ``"name"`` field and an
    optional ``"arguments"`` object. Multiple blocks in one output are
    all returned in document order.

    Tolerates parse failures: a block whose content is not valid JSON,
    or a JSON object missing ``"name"``, or one whose ``"arguments"`` is
    not an object, is silently skipped. The agent loop sees the
    surviving subset and treats the remainder of the output as text.

    Args:
        model_output: Raw text from the backend's ``generate`` call.

    Returns:
        Ordered list of parsed ``ToolCall`` instances. Empty if no
        well-formed call is present.
    """
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_PATTERN.finditer(model_output):
        text = match.group(1).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("name")
        if not isinstance(name, str):
            continue
        raw_args = data.get("arguments", {})
        arguments = raw_args if isinstance(raw_args, dict) else {}
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls


class BlockAgent:
    """Top-level agent loop. Composes registry + backend + tools + evaluator."""

    def __init__(
        self,
        registry: BlockRegistry,
        backend: Backend,
        tools: ToolDispatcher,
        evaluator: SuccessEvaluator,
        max_compile_tokens: int = 4096,
        max_new_tokens: int = 512,
        top_k_promote: int = 3,
        system_prompt: str | None = None,
    ) -> None:
        """Construct an agent.

        Args:
            registry: The RAMPART block registry the agent reads from
                and writes to.
            backend: The LLM backend used for both the task call and the
                failure-diagnosis call.
            tools: Tool dispatcher pre-loaded with whatever tools the
                agent should be able to invoke.
            evaluator: Success evaluator. ``LLMEvaluator`` is the
                default; callers plug in custom evaluators implementing
                the ``SuccessEvaluator`` protocol.
            max_compile_tokens: Hard cap on the compiled context size
                handed to the backend per turn.
            max_new_tokens: Cap on completion length passed to the
                backend on the task call. Default 512 preserves prior
                behaviour. Raise to 1024 (or higher) for benchmarks
                where the model emits long structured output — e.g.
                C functions with ISR handlers regularly exceed 512
                tokens, and a 512-cap clips the response mid-function
                so even logically-correct code fails the structural
                checker.
            top_k_promote: How many task-relevant blocks to promote to
                the front before compile. Mirrors
                ``RAMPARTConfig.top_k_promote``.
            system_prompt: Optional system-message instruction passed to
                the backend on the task call only. The diagnosis call
                deliberately omits this so the failure-analysis prompt
                is not constrained by suppression instructions. Typical
                use: pass "Answer directly without reasoning traces." to
                stop reasoning-capable small models from spending tokens
                on unsolicited ``<think>`` blocks during benchmarks.
                Default ``None`` preserves prior single-shot behaviour.
        """
        self.registry = registry
        self.backend = backend
        self.tools = tools
        self.evaluator = evaluator
        self.max_compile_tokens = max_compile_tokens
        self.max_new_tokens = max_new_tokens
        self.top_k_promote = top_k_promote
        self.system_prompt = system_prompt

    def run_task(self, task_text: str) -> TaskResult:
        """Run one task end-to-end and return a ``TaskResult``.

        Stages:

        1. Score blocks by relevance to the task and promote the
           top-K to the front of the registry. Skipped without raising
           when no embedding model is configured.
        2. Compile the registry into a prompt within the token budget.
        3. Call the backend in non-thinking mode for the task itself.
        4. Parse tool calls from the model output.
        5. Execute each tool call in order, collecting observations.
        6. Ask the success evaluator whether the task succeeded.
        7. On failure, generate a single actionable rule via a
           thinking-mode diagnosis call and write it to the registry as
           an agent block tagged with this run's trajectory id.
        8. Run the configured eviction policy.

        Returns:
            A ``TaskResult`` populated with the metrics needed by the
            session report.
        """
        trajectory_id = str(uuid.uuid4())

        self._score_and_promote(task_text)

        compile_result = self.registry.compile(
            self.max_compile_tokens, task_text=task_text
        )
        compiled = compile_result.prompt
        compiled_tokens = compile_result.total_tokens

        # Stage 3 — task call. thinking=False is explicit so the
        # mechanical "produce output from context" step does not spend
        # tokens on a <think> trace. system_prompt is passed (when
        # configured) so reasoning-capable small models honour the
        # non-thinking request — see BlockAgent.__init__ docstring.
        full_prompt = self._assemble_prompt(compiled, task_text)
        result = self.backend.generate(
            full_prompt,
            max_new_tokens=self.max_new_tokens,
            thinking=False,
            system_prompt=self.system_prompt,
            return_usage=True,
        )

        tool_calls = parse_tool_calls(result.text)
        observations = self._execute_tool_calls(tool_calls)

        success = self.evaluator.evaluate(
            task_text, result.text, observations
        )

        agent_block_written = False
        if not success:
            # Optional duck-typed hook: evaluators that retain per-check
            # state after evaluate() (e.g. a multi-criterion checker
            # storing last-result state) can expose a formatted
            # bullet-list of which specific checks failed via
            # failure_diagnostics_text(). The
            # diagnosis prompt threads this in so the model writes a rule
            # targeting the actual failure rather than guessing
            # generically. Returning empty string (or absence of the hook)
            # falls back to prior behaviour.
            checker_diagnostics = ""
            hook = getattr(self.evaluator, "failure_diagnostics_text", None)
            if callable(hook):
                checker_diagnostics = hook() or ""
            agent_block_written = self._write_block_on_failure(
                task_text=task_text,
                compiled_context=compiled,
                model_output=result.text,
                observations=observations,
                trajectory_id=trajectory_id,
                checker_diagnostics=checker_diagnostics,
            )

        # Eviction is explicit, not automatic. The simplified default
        # policy returns every evictable block at every call; running
        # it at the end of every task would delete user-written agent
        # blocks the moment they were created. Operators that want
        # bounded-size registries call ``registry.evict_by_policy()``
        # themselves on a schedule that fits their use case.

        return TaskResult(
            task_text=task_text,
            success=success,
            compiled_context_tokens=compiled_tokens,
            tool_calls_made=len(tool_calls),
            agent_block_written=agent_block_written,
            trajectory_id=trajectory_id,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    # ----- internals -----

    def _score_and_promote(self, task_text: str) -> None:
        """Promote the top-K relevance-ranked blocks to the front.

        Skips silently when no embedding model is configured — a fresh
        registry without an embedder should still be usable end-to-end.
        ``find_relevant`` raises ``RuntimeError`` in that case; the
        catch is the loop's contract with the registry.
        """
        try:
            relevant = self.registry.find_relevant(
                task_text, top_k=self.top_k_promote
            )
        except RuntimeError:
            return
        if not relevant:
            return
        # reorder() places the listed ids at the front in the given
        # order, so the top-1 most-relevant block ends up at position 0
        # (the U-shaped attention favoured slot per Liu et al. 2024).
        self.registry.reorder([block.runtime_id for block, _ in relevant])

    def _execute_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[str]:
        """Dispatch each tool call and collect observation text."""
        observations: list[str] = []
        for call in tool_calls:
            try:
                observation = self.tools.execute(call)
            except ValueError as exc:
                observation = f"ERROR: {exc}"
            observations.append(observation)
        return observations

    def _write_block_on_failure(
        self,
        task_text: str,
        compiled_context: str,
        model_output: str,
        observations: list[str],
        trajectory_id: str,
        checker_diagnostics: str = "",
    ) -> bool:
        """Generate an actionable rule from the failure and write it.

        Builds the diagnosis prompt with all four pieces of context and
        calls the backend in thinking mode (the one place
        meta-cognition is genuinely needed). Returns ``True`` iff a
        non-empty rule was actually appended to the registry; an empty
        diagnosis is dropped silently rather than polluting the registry
        with empty agent blocks.

        The diagnosis output is stripped of any ``<think>...</think>``
        block before storage. Reasoning-capable models in thinking-mode
        (qwen3, deepseek, ...) emit a verbose trace alongside the final
        rule; storing the trace inflates the agent block by 10-20x with
        text that is only useful at write time, not at re-compile time.
        Strict pattern (only complete pairs); a malformed half-open
        ``<think>`` is left intact so the operator can see the issue.
        """
        prompt = self._build_diagnosis_prompt(
            task_text=task_text,
            compiled_context=compiled_context,
            model_output=model_output,
            observations=observations,
            checker_diagnostics=checker_diagnostics,
        )
        # thinking=True is explicit: meta-cognitive diagnosis is the
        # one step in the loop where the cost of a thinking trace is
        # justified by the quality of the resulting rule.
        diagnosis = self.backend.generate(prompt, thinking=True)
        rule = _THINK_BLOCK_PATTERN.sub("", diagnosis).strip()
        if not rule:
            return False
        # evictable=False so the trajectory-learned heuristic survives
        # the end-of-loop evict_by_policy() pass. The whole point of
        # writing the block is to retain it across runs; making it
        # immediately evictable would erase the learning.
        # rollback_trajectory remains the explicit retraction path.
        self.registry.write_agent_block(
            content=rule,
            trajectory_id=trajectory_id,
            evictable=False,
        )
        return True

    @staticmethod
    def _assemble_prompt(compiled_context: str, task_text: str) -> str:
        """Combine the compiled context with the task into one prompt."""
        if not compiled_context:
            return f"TASK:\n{task_text}"
        return f"{compiled_context}\n\nTASK:\n{task_text}"

    @staticmethod
    def _build_diagnosis_prompt(
        task_text: str,
        compiled_context: str,
        model_output: str,
        observations: list[str],
        checker_diagnostics: str = "",
    ) -> str:
        """Assemble the failure-diagnosis prompt.

        Includes all four pieces of context per the user-locked spec:
        the task, the instructions the agent saw, what the agent
        actually tried, and the tool observations. Asks for a single
        actionable rule (not a description of what went wrong); a
        description would produce a useless agent block on the next
        compile.

        ``checker_diagnostics`` is an optional fifth section: when the
        evaluator exposes per-check booleans (via
        ``failure_diagnostics_text()``), the formatted bullet list is
        injected so the model sees *which specific check failed* rather
        than guessing generically. Empty string omits the section
        entirely, preserving the prior four-piece layout.
        """
        obs_text = (
            "\n---\n".join(observations)
            if observations
            else "(no tool calls were made)"
        )
        checker_section = (
            f"CHECKER VERDICTS:\n{checker_diagnostics}\n\n"
            if checker_diagnostics
            else ""
        )
        return (
            "A previous task attempt failed. Your job is to write a "
            "single actionable rule that, if added to future agent "
            "instructions, would prevent this failure mode. Output "
            "ONLY the rule, as a single sentence stating what to do. "
            "Do not describe what went wrong. Do not apologise. Do not "
            "list multiple steps.\n\n"
            f"TASK:\n{task_text}\n\n"
            f"INSTRUCTIONS THE AGENT WAS GIVEN:\n{compiled_context}\n\n"
            f"WHAT THE AGENT TRIED:\n{model_output}\n\n"
            f"TOOL OBSERVATIONS:\n{obs_text}\n\n"
            f"{checker_section}"
            "RULE (single actionable sentence):"
        )


__all__ = ["BlockAgent", "parse_tool_calls"]
