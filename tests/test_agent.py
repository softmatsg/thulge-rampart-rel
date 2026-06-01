"""Tests for parse_tool_calls and BlockAgent.run_task.

Three behaviours pinned by dedicated test classes:

* ``TestThinkingModeDiscipline`` — the user-locked rule that the task
  call must pass ``thinking=False`` and the diagnosis call must pass
  ``thinking=True``, both explicitly. Asserts the kwarg captured at
  each backend call.
* ``TestDiagnosisPromptContent`` — verifies the diagnosis prompt
  contains all four pieces of context (task, compiled context, model
  output, tool observations) and instructs the model to produce a
  single actionable rule rather than describe what went wrong.
* ``TestRunTaskRegistryIntegration`` — end-to-end via real RAMPART
  registry plus mock backend, verifying access_count increments on
  successful compile and that the agent block written on failure is
  retrievable by trajectory id.
"""

from __future__ import annotations

import uuid as _uuid
from typing import Any

from numpy.typing import NDArray

from blockagent.agent import BlockAgent, parse_tool_calls
from blockagent.backends.base import GenerateResult
from blockagent.eval.metrics import TaskResult
from blockagent.eval.success import SuccessEvaluator
from blockagent.tools.base import Tool, ToolCall, ToolDispatcher
from rampart.registry import BlockRegistry

# --- parse_tool_calls -------------------------------------------------------


class TestParseToolCalls:
    def test_empty_output_returns_empty_list(self) -> None:
        assert parse_tool_calls("") == []

    def test_no_fenced_block_returns_empty_list(self) -> None:
        assert parse_tool_calls("just text, no tool calls") == []

    def test_extracts_single_json_fenced_call(self) -> None:
        text = (
            "I'll check the file:\n"
            "```json\n"
            '{"name": "file_read", "arguments": {"path": "/etc/hosts"}}\n'
            "```\n"
            "Then process the result."
        )
        calls = parse_tool_calls(text)
        assert calls == [
            ToolCall(name="file_read", arguments={"path": "/etc/hosts"})
        ]

    def test_tool_call_fence_label_also_recognised(self) -> None:
        text = (
            "```tool_call\n"
            '{"name": "code_exec", "arguments": {"command": "ls"}}\n'
            "```"
        )
        calls = parse_tool_calls(text)
        assert calls == [
            ToolCall(name="code_exec", arguments={"command": "ls"})
        ]

    def test_multiple_calls_returned_in_document_order(self) -> None:
        text = (
            "```json\n"
            '{"name": "file_read", "arguments": {"path": "a.txt"}}\n'
            "```\n"
            "and then\n"
            "```json\n"
            '{"name": "code_exec", "arguments": {"command": "echo ok"}}\n'
            "```"
        )
        calls = parse_tool_calls(text)
        assert [c.name for c in calls] == ["file_read", "code_exec"]

    def test_missing_arguments_defaults_to_empty_dict(self) -> None:
        text = '```json\n{"name": "ping"}\n```'
        calls = parse_tool_calls(text)
        assert calls == [ToolCall(name="ping", arguments={})]

    def test_malformed_json_block_silently_skipped(self) -> None:
        text = (
            "```json\nthis is not json at all\n```\n"
            "```json\n"
            '{"name": "ok", "arguments": {}}\n'
            "```"
        )
        calls = parse_tool_calls(text)
        assert calls == [ToolCall(name="ok", arguments={})]

    def test_non_object_top_level_skipped(self) -> None:
        text = '```json\n["just", "an", "array"]\n```'
        assert parse_tool_calls(text) == []

    def test_missing_name_skipped(self) -> None:
        text = '```json\n{"arguments": {"x": 1}}\n```'
        assert parse_tool_calls(text) == []

    def test_non_string_name_skipped(self) -> None:
        text = '```json\n{"name": 42}\n```'
        assert parse_tool_calls(text) == []

    def test_non_dict_arguments_replaced_by_empty_dict(self) -> None:
        text = '```json\n{"name": "x", "arguments": "not a dict"}\n```'
        assert parse_tool_calls(text) == [ToolCall(name="x", arguments={})]

    def test_python_fence_is_not_parsed_as_tool_call(self) -> None:
        # Code blocks of language `python` are code samples, not tool
        # calls — only ``json`` and ``tool_call`` fences are recognised.
        text = '```python\nprint("hi")\n```'
        assert parse_tool_calls(text) == []


# --- helpers shared by the run_task tests ----------------------------------


class _ScriptedBackend:
    """Backend that returns scripted responses in order, recording calls.

    Each entry in ``responses`` may be a ``str`` (returned as text or
    wrapped in a default GenerateResult depending on the call's
    ``return_usage``) or a ``GenerateResult`` (returned verbatim when
    ``return_usage=True``, or its ``.text`` when False).
    """

    def __init__(self, responses: list[str | GenerateResult]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.index = 0

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
        response = self.responses[self.index]
        self.index += 1
        if isinstance(response, GenerateResult):
            return response if return_usage else response.text
        if return_usage:
            return GenerateResult(
                text=response, prompt_tokens=42, completion_tokens=13
            )
        return response


class _StaticEvaluator:
    """SuccessEvaluator that returns a fixed verdict and records inputs."""

    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.calls: list[dict[str, Any]] = []

    def evaluate(
        self,
        task_text: str,
        model_output: str,
        observations: list[str],
    ) -> bool:
        self.calls.append(
            {
                "task_text": task_text,
                "model_output": model_output,
                "observations": list(observations),
            }
        )
        return self.verdict


def _agent(
    *,
    backend: _ScriptedBackend,
    evaluator: _StaticEvaluator,
    tools: ToolDispatcher | None = None,
    registry: BlockRegistry | None = None,
    system_prompt: str | None = None,
) -> BlockAgent:
    # `registry or BlockRegistry(...)` would silently create a fresh
    # registry when an empty one is passed, because BlockRegistry
    # defines __len__ and an empty registry is falsy. Explicit None
    # check avoids that footgun.
    return BlockAgent(
        registry=registry if registry is not None else BlockRegistry(tokeniser=len),
        backend=backend,
        tools=tools if tools is not None else ToolDispatcher(),
        evaluator=evaluator,
        system_prompt=system_prompt,
    )


# --- TaskResult shape on the success path ---------------------------------


class TestRunTaskSuccessPath:
    def test_returns_task_result(self) -> None:
        backend = _ScriptedBackend(["task done"])
        ev = _StaticEvaluator(verdict=True)
        result = _agent(backend=backend, evaluator=ev).run_task("do it")
        assert isinstance(result, TaskResult)

    def test_success_field_reflects_evaluator(self) -> None:
        backend = _ScriptedBackend(["task done"])
        ev = _StaticEvaluator(verdict=True)
        result = _agent(backend=backend, evaluator=ev).run_task("do it")
        assert result.success is True
        assert result.agent_block_written is False

    def test_no_diagnosis_call_on_success(self) -> None:
        backend = _ScriptedBackend(["task done"])
        ev = _StaticEvaluator(verdict=True)
        _agent(backend=backend, evaluator=ev).run_task("do it")
        # Only the task call (the evaluator is a static stub, not the
        # backend). No diagnosis call on the success path.
        assert len(backend.calls) == 1

    def test_token_counts_propagated_from_backend(self) -> None:
        backend = _ScriptedBackend(
            [GenerateResult(text="ok", prompt_tokens=99, completion_tokens=7)]
        )
        ev = _StaticEvaluator(verdict=True)
        result = _agent(backend=backend, evaluator=ev).run_task("do it")
        assert result.prompt_tokens == 99
        assert result.completion_tokens == 7

    def test_trajectory_id_is_uuid4(self) -> None:
        backend = _ScriptedBackend(["task done"])
        ev = _StaticEvaluator(verdict=True)
        result = _agent(backend=backend, evaluator=ev).run_task("do it")
        assert _uuid.UUID(result.trajectory_id).version == 4


# --- TaskResult on the failure path ---------------------------------------


class TestRunTaskFailurePath:
    def test_failure_writes_agent_block_and_records_flag(self) -> None:
        backend = _ScriptedBackend(["bad attempt", "always rinse before scrub"])
        ev = _StaticEvaluator(verdict=False)
        registry = BlockRegistry(tokeniser=len)
        result = _agent(
            backend=backend, evaluator=ev, registry=registry
        ).run_task("clean the pan")
        assert result.success is False
        assert result.agent_block_written is True
        # The newly written block lives in the registry tagged with the
        # run's trajectory id.
        agent_blocks = [
            b for b in registry.list_blocks() if b.source == "agent"
        ]
        assert len(agent_blocks) == 1
        assert agent_blocks[0].trajectory_id == result.trajectory_id

    def test_empty_diagnosis_does_not_pollute_registry(self) -> None:
        # If the diagnosis call returns an all-whitespace rule, the
        # block is dropped and the flag is False.
        backend = _ScriptedBackend(["bad attempt", "   \n   "])
        ev = _StaticEvaluator(verdict=False)
        registry = BlockRegistry(tokeniser=len)
        result = _agent(
            backend=backend, evaluator=ev, registry=registry
        ).run_task("clean the pan")
        assert result.success is False
        assert result.agent_block_written is False
        assert all(b.source != "agent" for b in registry.list_blocks())


# --- thinking-mode discipline (user-locked) -------------------------------


class TestThinkingModeDiscipline:
    def test_task_call_uses_thinking_false_explicitly(self) -> None:
        backend = _ScriptedBackend(["task done"])
        ev = _StaticEvaluator(verdict=True)
        _agent(backend=backend, evaluator=ev).run_task("do it")
        assert backend.calls[0]["thinking"] is False

    def test_diagnosis_call_uses_thinking_true_explicitly(self) -> None:
        backend = _ScriptedBackend(["bad attempt", "rule: do X"])
        ev = _StaticEvaluator(verdict=False)
        _agent(backend=backend, evaluator=ev).run_task("do it")
        # Two calls: index 0 is the task (thinking=False), index 1 is
        # the diagnosis (thinking=True). The user-locked rule.
        assert backend.calls[0]["thinking"] is False
        assert backend.calls[1]["thinking"] is True

    def test_task_call_requests_token_usage(self) -> None:
        backend = _ScriptedBackend(["task done"])
        ev = _StaticEvaluator(verdict=True)
        _agent(backend=backend, evaluator=ev).run_task("do it")
        # Token counts are needed for the TaskResult; the task call
        # asks for them. The diagnosis call does not.
        assert backend.calls[0]["return_usage"] is True


# --- system_prompt parameter (suppress thinking on task call only) --------


class TestSystemPrompt:
    def test_default_system_prompt_is_none(self) -> None:
        # Existing tests rely on this default. Pin it.
        backend = _ScriptedBackend(["task done"])
        ev = _StaticEvaluator(verdict=True)
        _agent(backend=backend, evaluator=ev).run_task("do it")
        assert backend.calls[0]["system_prompt"] is None

    def test_configured_system_prompt_passed_on_task_call(self) -> None:
        backend = _ScriptedBackend(["task done"])
        ev = _StaticEvaluator(verdict=True)
        _agent(
            backend=backend,
            evaluator=ev,
            system_prompt="Answer directly without reasoning traces.",
        ).run_task("do it")
        assert (
            backend.calls[0]["system_prompt"]
            == "Answer directly without reasoning traces."
        )

    def test_system_prompt_omitted_on_diagnosis_call(self) -> None:
        # The user-locked rule: thinking=True diagnosis call must NOT
        # carry the suppression instruction, otherwise the failure-
        # analysis path is constrained by a directive aimed at the
        # task path.
        backend = _ScriptedBackend(["bad attempt", "rule: do X"])
        ev = _StaticEvaluator(verdict=False)
        _agent(
            backend=backend,
            evaluator=ev,
            system_prompt="Answer directly without reasoning traces.",
        ).run_task("clean the pan")
        assert (
            backend.calls[0]["system_prompt"]
            == "Answer directly without reasoning traces."
        )
        # Diagnosis call (index 1) goes via _write_block_on_failure
        # which calls backend.generate(prompt, thinking=True) without
        # passing system_prompt — defaults to None.
        assert backend.calls[1]["system_prompt"] is None


# --- diagnosis prompt content (user-locked) -------------------------------


class TestDiagnosisPromptContent:
    def test_diagnosis_prompt_contains_all_four_context_pieces(self) -> None:
        backend = _ScriptedBackend(
            ["I will run ls", "rule: check file existence first"]
        )
        ev = _StaticEvaluator(verdict=False)
        registry = BlockRegistry(tokeniser=len)
        # Pre-load the registry with a seed-ish block so the compiled
        # context is non-empty.
        rid = registry.write_agent_block(
            "always check timer overflow", trajectory_id="seed"
        )
        # Make this block deterministic in the prompt by promoting to front.
        registry.promote_to_front(rid)

        tools = ToolDispatcher()
        agent = _agent(
            backend=backend, evaluator=ev, tools=tools, registry=registry
        )
        agent.run_task("clean the pan")

        diagnosis_prompt = backend.calls[1]["prompt"]
        # All four pieces must be present.
        assert "clean the pan" in diagnosis_prompt  # task text
        assert "always check timer overflow" in diagnosis_prompt  # compiled context
        assert "I will run ls" in diagnosis_prompt  # model output
        # No tool calls happened so the marker stands in for observations.
        assert "no tool calls were made" in diagnosis_prompt

    def test_diagnosis_prompt_includes_observations_when_tools_ran(
        self,
    ) -> None:
        backend = _ScriptedBackend(
            [
                # A tool-call output so observations are non-empty.
                '```json\n{"name": "rec", "arguments": {}}\n```',
                "rule: avoid X",
            ]
        )
        ev = _StaticEvaluator(verdict=False)
        tools = ToolDispatcher()

        class _Echo:
            name = "rec"

            def execute(self, arguments: dict[str, Any]) -> str:
                return "STDOUT: tool-said-something"

        tools.register(_Echo())
        agent = _agent(backend=backend, evaluator=ev, tools=tools)
        agent.run_task("a task")

        diagnosis_prompt = backend.calls[1]["prompt"]
        assert "STDOUT: tool-said-something" in diagnosis_prompt

    def test_diagnosis_prompt_asks_for_actionable_rule_not_description(
        self,
    ) -> None:
        # The rule-vs-description framing matters: a description writes
        # an unhelpful "we failed because X" agent block. Pin the
        # instruction wording so future edits don't silently regress.
        backend = _ScriptedBackend(["bad attempt", "rule"])
        ev = _StaticEvaluator(verdict=False)
        _agent(backend=backend, evaluator=ev).run_task("a task")
        diagnosis_prompt = backend.calls[1]["prompt"]
        assert "single actionable rule" in diagnosis_prompt.lower()
        assert "do not describe" in diagnosis_prompt.lower()


# --- diagnosis output sanitisation (thinking-trace stripping) -------------


class TestDiagnosisOutputSanitisation:
    """The diagnosis call uses ``thinking=True``; reasoning-capable models
    emit a ``<think>...</think>`` trace alongside the final rule. The
    block written to the registry must contain only the post-think text
    so the rule isn't drowned by hundreds of tokens of reasoning every
    future compile.
    """

    def test_thinking_trace_stripped_from_agent_block(self) -> None:
        diagnosis = (
            "<think>Okay, let's see. The agent tried X but Y was wrong "
            "because of Z, so the rule should be...</think>"
            "Always check the timer overflow flag before clearing."
        )
        backend = _ScriptedBackend(["bad attempt", diagnosis])
        ev = _StaticEvaluator(verdict=False)
        registry = BlockRegistry(tokeniser=len)
        _agent(
            backend=backend, evaluator=ev, registry=registry,
        ).run_task("do it")
        agent_blocks = [
            b for b in registry.list_blocks() if b.source == "agent"
        ]
        assert len(agent_blocks) == 1
        content = agent_blocks[0].content
        assert "<think>" not in content
        assert "</think>" not in content
        assert "Okay, let's see" not in content
        assert content == "Always check the timer overflow flag before clearing."

    def test_multiple_think_blocks_all_stripped(self) -> None:
        # Some models emit multiple <think>...</think> spans. Each must go.
        diagnosis = (
            "<think>first reasoning</think>part one. "
            "<think>second reasoning</think>part two."
        )
        backend = _ScriptedBackend(["bad attempt", diagnosis])
        ev = _StaticEvaluator(verdict=False)
        registry = BlockRegistry(tokeniser=len)
        _agent(
            backend=backend, evaluator=ev, registry=registry,
        ).run_task("do it")
        [block] = [b for b in registry.list_blocks() if b.source == "agent"]
        assert "<think>" not in block.content
        assert "first reasoning" not in block.content
        assert "second reasoning" not in block.content
        assert "part one." in block.content
        assert "part two." in block.content

    def test_diagnosis_with_only_thinking_trace_drops_block(self) -> None:
        # If the entire diagnosis is a <think> block with no post-think
        # text, the resulting rule is empty and no block should be
        # written — same contract as test_empty_diagnosis_does_not_pollute.
        diagnosis = "<think>I have nothing useful to say.</think>"
        backend = _ScriptedBackend(["bad attempt", diagnosis])
        ev = _StaticEvaluator(verdict=False)
        registry = BlockRegistry(tokeniser=len)
        result = _agent(
            backend=backend, evaluator=ev, registry=registry,
        ).run_task("do it")
        assert result.agent_block_written is False
        assert all(b.source != "agent" for b in registry.list_blocks())

    def test_diagnosis_without_think_tags_passes_through(self) -> None:
        # Backends that don't emit thinking traces (or models that
        # decline to use them) must not be silently mangled.
        diagnosis = "Always increment the counter atomically."
        backend = _ScriptedBackend(["bad attempt", diagnosis])
        ev = _StaticEvaluator(verdict=False)
        registry = BlockRegistry(tokeniser=len)
        _agent(
            backend=backend, evaluator=ev, registry=registry,
        ).run_task("do it")
        [block] = [b for b in registry.list_blocks() if b.source == "agent"]
        assert block.content == diagnosis


# --- diagnosis prompt threads checker verdicts ----------------------------


class _DiagnosingEvaluator:
    """Static evaluator that also exposes a ``failure_diagnostics_text``
    duck-typed hook, the contract a multi-criterion checker uses to
    surface which specific checks failed.
    """

    def __init__(self, verdict: bool, diagnostics: str) -> None:
        self.verdict = verdict
        self.diagnostics = diagnostics

    def evaluate(
        self,
        task_text: str,
        model_output: str,
        observations: list[str],
    ) -> bool:
        return self.verdict

    def failure_diagnostics_text(self) -> str:
        return self.diagnostics


class TestCheckerDiagnosticsThreading:
    def test_diagnostics_appear_in_diagnosis_prompt(self) -> None:
        diagnostics = (
            "- no_isr_malloc: True\n"
            "- correct_prescaler: False  (FAILED)\n"
            "- timer_overflow_guard_present: True"
        )
        backend = _ScriptedBackend(["bad attempt", "rule: do X"])
        ev = _DiagnosingEvaluator(verdict=False, diagnostics=diagnostics)
        agent = BlockAgent(
            registry=BlockRegistry(tokeniser=len),
            backend=backend,
            tools=ToolDispatcher(),
            evaluator=ev,
        )
        agent.run_task("do it")
        diagnosis_prompt = backend.calls[1]["prompt"]
        assert "CHECKER VERDICTS:" in diagnosis_prompt
        assert "correct_prescaler: False" in diagnosis_prompt
        assert "(FAILED)" in diagnosis_prompt

    def test_no_diagnostics_section_when_evaluator_lacks_hook(self) -> None:
        # _StaticEvaluator does NOT define failure_diagnostics_text;
        # the agent must omit the section cleanly rather than crashing.
        backend = _ScriptedBackend(["bad attempt", "rule: do X"])
        ev = _StaticEvaluator(verdict=False)
        _agent(backend=backend, evaluator=ev).run_task("do it")
        diagnosis_prompt = backend.calls[1]["prompt"]
        assert "CHECKER VERDICTS:" not in diagnosis_prompt

    def test_empty_diagnostics_string_omits_section(self) -> None:
        # Hook present but returns empty string (e.g. a multi-criterion
        # checker before its first evaluate() call, or after an
        # evaluation that produced no extractable check verdicts).
        backend = _ScriptedBackend(["bad attempt", "rule: do X"])
        ev = _DiagnosingEvaluator(verdict=False, diagnostics="")
        agent = BlockAgent(
            registry=BlockRegistry(tokeniser=len),
            backend=backend,
            tools=ToolDispatcher(),
            evaluator=ev,
        )
        agent.run_task("do it")
        diagnosis_prompt = backend.calls[1]["prompt"]
        assert "CHECKER VERDICTS:" not in diagnosis_prompt


# --- max_new_tokens parameter on the task call ----------------------------


class TestTaskCallMaxNewTokens:
    """The task call must pass ``max_new_tokens`` explicitly so the
    backend doesn't silently truncate long outputs at its default.
    The 512-token Ollama default clips C-function emissions mid-body.
    """

    def test_default_max_new_tokens_is_512(self) -> None:
        backend = _ScriptedBackend(["task done"])
        ev = _StaticEvaluator(verdict=True)
        _agent(backend=backend, evaluator=ev).run_task("do it")
        assert backend.calls[0]["max_new_tokens"] == 512

    def test_constructor_override_propagates_to_task_call(self) -> None:
        backend = _ScriptedBackend(["task done"])
        ev = _StaticEvaluator(verdict=True)
        agent = BlockAgent(
            registry=BlockRegistry(tokeniser=len),
            backend=backend,
            tools=ToolDispatcher(),
            evaluator=ev,
            max_new_tokens=1024,
        )
        agent.run_task("do it")
        assert backend.calls[0]["max_new_tokens"] == 1024


# --- tool execution semantics ---------------------------------------------


class _CountingTool:
    name = "ct"

    def __init__(self) -> None:
        self.invocations: list[dict[str, Any]] = []

    def execute(self, arguments: dict[str, Any]) -> str:
        self.invocations.append(arguments)
        return f"observed: {arguments}"


class TestRunTaskToolExecution:
    def test_tool_calls_dispatched_and_counted(self) -> None:
        tool = _CountingTool()
        tools = ToolDispatcher()
        tools.register(tool)
        backend = _ScriptedBackend(
            [
                (
                    '```json\n{"name": "ct", "arguments": {"k": 1}}\n```\n'
                    '```json\n{"name": "ct", "arguments": {"k": 2}}\n```'
                ),
            ]
        )
        ev = _StaticEvaluator(verdict=True)
        result = _agent(backend=backend, evaluator=ev, tools=tools).run_task(
            "x"
        )
        assert result.tool_calls_made == 2
        assert tool.invocations == [{"k": 1}, {"k": 2}]

    def test_unknown_tool_observation_recorded_not_raised(self) -> None:
        backend = _ScriptedBackend(
            ['```json\n{"name": "missing", "arguments": {}}\n```']
        )
        ev = _StaticEvaluator(verdict=True)
        result = _agent(
            backend=backend, evaluator=ev, tools=ToolDispatcher()
        ).run_task("x")
        # The tool dispatcher raised ValueError; the loop converted it
        # to observation text and continued.
        assert result.tool_calls_made == 1
        assert ev.calls[0]["observations"][0].startswith("ERROR:")


# --- registry integration --------------------------------------------------


class TestRunTaskRegistryIntegration:
    def test_score_and_promote_skipped_without_embedder(self) -> None:
        # Default registry has no embedding model; find_relevant raises
        # RuntimeError, which the loop swallows. The task should still
        # run end-to-end.
        backend = _ScriptedBackend(["done"])
        ev = _StaticEvaluator(verdict=True)
        registry = BlockRegistry(tokeniser=len)
        registry.write_agent_block("seed", trajectory_id="t")
        result = _agent(
            backend=backend, evaluator=ev, registry=registry
        ).run_task("anything")
        assert result.success is True

    def test_score_and_promote_reorders_when_embedder_present(self) -> None:
        # With an embedder configured, the loop calls find_relevant and
        # reorders the top-K relevant blocks to the front via
        # registry.reorder(). Verify the most-relevant ends up at
        # position 0.
        import numpy as np

        class _FixedEmbedder:
            def encode(self, text: str) -> NDArray[Any]:
                # "task" embeds to a vector aligned with block "match";
                # "other" embeds to an unrelated vector.
                if text == "the task":
                    return np.array([1.0, 0.0, 0.0], dtype=np.float32)
                if text == "match":
                    return np.array([1.0, 0.0, 0.0], dtype=np.float32)
                return np.array([0.0, 1.0, 0.0], dtype=np.float32)

            def encode_batch(self, texts: list[str]) -> NDArray[Any]:
                return np.stack([self.encode(t) for t in texts])

        registry = BlockRegistry(
            tokeniser=len, embedding_model=_FixedEmbedder()
        )
        far = registry.write_agent_block("first written", trajectory_id="t")
        near = registry.write_agent_block("match", trajectory_id="t")
        # Pre-embed each block to known vectors so find_relevant has
        # something to score against.
        registry.get_by_id(far).embedding = np.array(
            [0.0, 1.0, 0.0], dtype=np.float32
        )
        registry.get_by_id(near).embedding = np.array(
            [1.0, 0.0, 0.0], dtype=np.float32
        )
        before_order = registry.runtime_ids()
        assert before_order == [far, near]  # write order

        backend = _ScriptedBackend(["done"])
        ev = _StaticEvaluator(verdict=True)
        _agent(
            backend=backend, evaluator=ev, registry=registry
        ).run_task("the task")
        # After score_and_promote, "near" (most relevant) is at the front.
        assert registry.runtime_ids()[0] == near

    def test_score_and_promote_noop_when_no_relevant_blocks(self) -> None:
        # Embedder configured but registry has no embedded blocks →
        # find_relevant returns [] → early return, no reorder call.
        # Sanity: the loop completes without raising.
        import numpy as np

        class _NoMatchEmbedder:
            def encode(self, text: str) -> NDArray[Any]:
                return np.zeros(3, dtype=np.float32)

            def encode_batch(self, texts: list[str]) -> NDArray[Any]:
                return np.stack([self.encode(t) for t in texts])

        registry = BlockRegistry(
            tokeniser=len, embedding_model=_NoMatchEmbedder()
        )
        # Block has no embedding (None); find_relevant skips it,
        # returning an empty list.
        registry.write_agent_block("x", trajectory_id="t")
        backend = _ScriptedBackend(["done"])
        ev = _StaticEvaluator(verdict=True)
        result = _agent(
            backend=backend, evaluator=ev, registry=registry
        ).run_task("anything")
        assert result.success is True

    def test_compile_increments_access_count_for_included_blocks(self) -> None:
        backend = _ScriptedBackend(["done"])
        ev = _StaticEvaluator(verdict=True)
        registry = BlockRegistry(tokeniser=len)
        rid = registry.write_agent_block("a block", trajectory_id="t")
        before = registry.get_by_id(rid).access_count
        _agent(backend=backend, evaluator=ev, registry=registry).run_task("x")
        # Access_count incremented because the block fit in the
        # compiled context.
        assert registry.get_by_id(rid).access_count == before + 1

    def test_compiled_context_token_count_recorded(self) -> None:
        backend = _ScriptedBackend(["done"])
        ev = _StaticEvaluator(verdict=True)
        registry = BlockRegistry(tokeniser=len)
        registry.write_agent_block("hello world", trajectory_id="t")
        result = _agent(
            backend=backend, evaluator=ev, registry=registry
        ).run_task("x")
        # tokeniser=len, "hello world" is 11 chars → 11 tokens.
        assert result.compiled_context_tokens == 11

    def test_run_task_does_not_auto_evict_existing_blocks(self) -> None:
        # The simplified default policy returns every evictable block
        # at every call, so calling evict_by_policy() at the end of
        # run_task would wipe user-written agent blocks the moment
        # they were created. Pin that ``run_task`` does NOT auto-evict.
        # Operators that want pruning call evict_by_policy explicitly.
        backend = _ScriptedBackend(["done"])
        ev = _StaticEvaluator(verdict=True)
        registry = BlockRegistry(tokeniser=len)
        rid = registry.write_agent_block("x", trajectory_id="t")
        result = _agent(
            backend=backend, evaluator=ev, registry=registry
        ).run_task("x")
        assert result.success is True
        # Block is still there.
        assert rid in registry


# --- type-level Protocol satisfaction --------------------------------------


class TestProtocolSatisfaction:
    def test_static_evaluator_satisfies_success_evaluator_protocol(self) -> None:
        ev: SuccessEvaluator = _StaticEvaluator(verdict=True)
        assert isinstance(ev, SuccessEvaluator)

    def test_counting_tool_satisfies_tool_protocol(self) -> None:
        tool: Tool = _CountingTool()
        assert isinstance(tool, Tool)
