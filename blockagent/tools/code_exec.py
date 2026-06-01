"""CodeExecTool — subprocess-based code/command execution.

Sharp edge: this tool runs whatever command the model emits via
``shell=True``. That is the entire point of a code-execution tool for an
agentic coding benchmark — the agent's job is to produce and run code.
The risk is the agent's own capability, not external prompt injection;
operators sandbox the runtime (Docker container, ephemeral VM, etc.)
rather than trying to filter shell strings inside the tool itself.

Recoverable failures (timeouts, non-zero exits, missing executables)
are converted into observation text so the agent loop continues. The
tool only raises if a programming-level invariant is violated, which
the dispatcher does not currently produce.
"""

from __future__ import annotations

import subprocess
from typing import Any


class CodeExecTool:
    """Execute a shell command and return combined stdout/stderr/exit info."""

    name: str = "code_exec"

    def __init__(self, default_timeout: int = 30) -> None:
        """Construct the tool.

        Args:
            default_timeout: Seconds. Used when a tool call does not
                supply its own ``timeout`` argument.
        """
        self.default_timeout = default_timeout

    def execute(self, arguments: dict[str, Any]) -> str:
        """Run the command described by ``arguments``.

        Recognised arguments:
            command: str — shell command to run. Required.
            timeout: int — seconds before the command is killed. Default
                falls back to ``self.default_timeout``.

        Returns:
            Multi-line observation containing whichever of STDOUT,
            STDERR, and EXIT are present. Returns ``"ERROR: ..."`` text
            for missing command, timeout, or other recoverable failure.
        """
        command = str(arguments.get("command", ""))
        if not command:
            return "ERROR: no 'command' argument provided"

        timeout = int(arguments.get("timeout", self.default_timeout))

        try:
            result = subprocess.run(
                command,
                shell=True,  # noqa: S602  # agentic code execution by design
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {timeout}s"
        except OSError as exc:  # pragma: no cover - OS-conditional
            return f"ERROR: failed to spawn command: {exc}"

        parts: list[str] = []
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout.rstrip()}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr.rstrip()}")
        if result.returncode != 0:
            parts.append(f"EXIT: {result.returncode}")
        return "\n".join(parts) if parts else "(no output)"


__all__ = ["CodeExecTool"]
