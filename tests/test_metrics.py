"""Tests for the TaskResult dataclass.

Pin the public field set, immutability, and value-equality so consumers
that aggregate TaskResults into SessionReports cannot accidentally
mutate counts or rely on object identity.
"""

from __future__ import annotations

import dataclasses

import pytest

from blockagent.eval.metrics import TaskResult


def _result(**overrides: object) -> TaskResult:
    fields: dict[str, object] = {
        "task_text": "do the thing",
        "success": True,
        "compiled_context_tokens": 123,
        "tool_calls_made": 2,
        "agent_block_written": False,
        "trajectory_id": "abc-123",
        "prompt_tokens": 80,
        "completion_tokens": 40,
    }
    fields.update(overrides)
    return TaskResult(**fields)  # type: ignore[arg-type]


class TestTaskResultFields:
    def test_all_eight_fields_are_set(self) -> None:
        r = _result()
        assert r.task_text == "do the thing"
        assert r.success is True
        assert r.compiled_context_tokens == 123
        assert r.tool_calls_made == 2
        assert r.agent_block_written is False
        assert r.trajectory_id == "abc-123"
        assert r.prompt_tokens == 80
        assert r.completion_tokens == 40

    def test_field_set_matches_spec(self) -> None:
        # Pin the field set so a future addition or rename surfaces in
        # this test rather than as silent breakage downstream.
        names = {f.name for f in dataclasses.fields(TaskResult)}
        assert names == {
            "task_text",
            "success",
            "compiled_context_tokens",
            "tool_calls_made",
            "agent_block_written",
            "trajectory_id",
            "prompt_tokens",
            "completion_tokens",
        }


class TestTaskResultImmutability:
    def test_is_frozen(self) -> None:
        r = _result()
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.success = False  # type: ignore[misc]

    def test_value_equality(self) -> None:
        a = _result()
        b = _result()
        c = _result(success=False)
        assert a == b
        assert a != c
