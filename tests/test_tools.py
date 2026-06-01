"""Tests for the tool layer: ToolCall, ToolDispatcher, CodeExecTool, FileReadTool.

CodeExecTool tests use ``echo`` and ``false`` (or ``cmd``-equivalents
on Windows) which are present on every supported dev platform. File-
related failure-path tests use ``tmp_path`` so they never depend on
real files outside the test sandbox.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import pytest

from blockagent.tools.base import Tool, ToolCall, ToolDispatcher
from blockagent.tools.code_exec import CodeExecTool
from blockagent.tools.file_read import FileReadTool

# --- ToolCall dataclass -----------------------------------------------------


class TestToolCall:
    def test_default_arguments_is_empty_dict(self) -> None:
        call = ToolCall(name="x")
        assert call.arguments == {}

    def test_default_arguments_dicts_are_not_shared(self) -> None:
        a = ToolCall(name="x")
        b = ToolCall(name="y")
        # Frozen dataclass blocks reassignment but the dict itself is
        # mutable. Mutating one must not bleed into the other.
        a.arguments["k"] = 1
        assert b.arguments == {}

    def test_is_frozen(self) -> None:
        call = ToolCall(name="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            call.name = "y"  # type: ignore[misc]


# --- ToolDispatcher ---------------------------------------------------------


class _RecordingTool:
    def __init__(self, name: str = "rec") -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def execute(self, arguments: dict[str, Any]) -> str:
        self.calls.append(arguments)
        return f"recorded {self.name}"


class TestToolDispatcher:
    def test_register_then_execute(self) -> None:
        d = ToolDispatcher()
        tool = _RecordingTool("foo")
        d.register(tool)
        result = d.execute(ToolCall(name="foo", arguments={"a": 1}))
        assert result == "recorded foo"
        assert tool.calls == [{"a": 1}]

    def test_unknown_name_raises_value_error(self) -> None:
        d = ToolDispatcher()
        with pytest.raises(ValueError, match="missing"):
            d.execute(ToolCall(name="missing"))

    def test_re_register_overwrites(self) -> None:
        d = ToolDispatcher()
        first = _RecordingTool("foo")
        second = _RecordingTool("foo")
        d.register(first)
        d.register(second)
        d.execute(ToolCall(name="foo"))
        assert first.calls == []
        assert second.calls == [{}]

    def test_has_returns_true_only_for_registered(self) -> None:
        d = ToolDispatcher()
        d.register(_RecordingTool("foo"))
        assert d.has("foo") is True
        assert d.has("bar") is False

    def test_names_returns_registration_order(self) -> None:
        d = ToolDispatcher()
        d.register(_RecordingTool("a"))
        d.register(_RecordingTool("b"))
        d.register(_RecordingTool("c"))
        assert d.names() == ["a", "b", "c"]

    def test_recording_tool_satisfies_protocol(self) -> None:
        # Structural Protocol check on the test stand-in keeps the
        # Protocol shape from drifting silently.
        tool: Tool = _RecordingTool()
        assert isinstance(tool, Tool)


# --- CodeExecTool -----------------------------------------------------------


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _echo_command(text: str) -> str:
    """Return a shell command that echoes ``text`` cross-platform."""
    if _is_windows():
        return f"cmd /c echo {text}"
    return f"echo '{text}'"


def _false_command() -> str:
    """Return a shell command that exits non-zero cross-platform."""
    if _is_windows():
        return "cmd /c exit 7"
    return "false"


class TestCodeExecTool:
    def test_name_attribute(self) -> None:
        assert CodeExecTool().name == "code_exec"

    def test_executes_command_and_captures_stdout(self) -> None:
        tool = CodeExecTool()
        result = tool.execute({"command": _echo_command("hello")})
        assert "hello" in result
        assert "STDOUT:" in result

    def test_missing_command_returns_error(self) -> None:
        tool = CodeExecTool()
        result = tool.execute({})
        assert result.startswith("ERROR:")

    def test_nonzero_exit_recorded_in_observation(self) -> None:
        tool = CodeExecTool()
        result = tool.execute({"command": _false_command()})
        assert "EXIT:" in result

    def test_stderr_recorded_in_observation(self) -> None:
        # python -c is the cross-platform way to make a process write to
        # stderr without depending on platform-specific shell builtins.
        tool = CodeExecTool()
        result = tool.execute(
            {
                "command": (
                    'python -c "import sys; sys.stderr.write(\'oh no\')"'
                )
            }
        )
        assert "STDERR:" in result
        assert "oh no" in result

    def test_timeout_returns_error(self) -> None:
        tool = CodeExecTool(default_timeout=1)
        if _is_windows():
            cmd = "cmd /c ping 127.0.0.1 -n 5 > NUL"
        else:
            cmd = "sleep 5"
        result = tool.execute({"command": cmd})
        assert result.startswith("ERROR:") and "timed out" in result

    def test_custom_timeout_argument(self) -> None:
        # Just verify the int conversion path works; behavioural timeout
        # already covered above.
        tool = CodeExecTool()
        result = tool.execute(
            {"command": _echo_command("ok"), "timeout": 5}
        )
        assert "ok" in result

    def test_satisfies_tool_protocol(self) -> None:
        tool: Tool = CodeExecTool()
        assert isinstance(tool, Tool)


# --- FileReadTool -----------------------------------------------------------


class TestFileReadTool:
    def test_name_attribute(self) -> None:
        assert FileReadTool().name == "file_read"

    def test_reads_file(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        path.write_text("hello world", encoding="utf-8")
        result = FileReadTool().execute({"path": str(path)})
        assert result == "hello world"

    def test_missing_path_argument_returns_error(self) -> None:
        result = FileReadTool().execute({})
        assert result.startswith("ERROR:")

    def test_missing_file_returns_error(self, tmp_path: Path) -> None:
        ghost = tmp_path / "does-not-exist.txt"
        result = FileReadTool().execute({"path": str(ghost)})
        assert result.startswith("ERROR:") and "not found" in result

    def test_directory_returns_error(self, tmp_path: Path) -> None:
        # POSIX raises IsADirectoryError; Windows raises PermissionError.
        # Both are caught and surfaced as ERROR observations; the test
        # accepts either marker so it stays honest cross-platform.
        result = FileReadTool().execute({"path": str(tmp_path)})
        assert result.startswith("ERROR:")
        assert "is a directory" in result or "permission denied" in result

    def test_truncates_at_max_chars(self, tmp_path: Path) -> None:
        path = tmp_path / "big.txt"
        path.write_text("x" * 1000, encoding="utf-8")
        result = FileReadTool().execute(
            {"path": str(path), "max_chars": 100}
        )
        assert result.startswith("x" * 100)
        assert "truncated" in result

    def test_no_truncation_when_under_max_chars(self, tmp_path: Path) -> None:
        path = tmp_path / "small.txt"
        path.write_text("hi", encoding="utf-8")
        result = FileReadTool().execute(
            {"path": str(path), "max_chars": 1000}
        )
        assert result == "hi"

    def test_strips_utf8_bom(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.txt"
        path.write_bytes(b"\xef\xbb\xbfhello")
        result = FileReadTool().execute({"path": str(path)})
        assert result == "hello"

    def test_non_utf8_returns_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bin.dat"
        path.write_bytes(b"\xff\xfe\x00invalid")
        result = FileReadTool().execute({"path": str(path)})
        assert result.startswith("ERROR:") and "UTF-8" in result

    def test_satisfies_tool_protocol(self) -> None:
        tool: Tool = FileReadTool()
        assert isinstance(tool, Tool)
