"""Tool Protocol, ToolCall value type, and ToolDispatcher.

A ``ToolCall`` is the agent's parsed request to invoke a named tool with
arguments. The dispatcher maps the name to a registered ``Tool`` and
hands off execution. Unknown tool names raise ``ValueError`` immediately
so a misnamed call surfaces in the observation rather than silently
becoming a no-op — silent skips would let the agent loop think it
"executed" something it didn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A parsed tool invocation extracted from model output.

    Attributes:
        name: Tool identifier, matched against ``Tool.name`` at dispatch.
        arguments: Keyword arguments to forward to ``Tool.execute``.
            Always a dict (possibly empty); the parser substitutes ``{}``
            for malformed or missing argument blocks.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    """Strategy interface for an executable tool."""

    @property
    def name(self) -> str:
        """Stable identifier used by the dispatcher and by model output."""
        ...

    def execute(self, arguments: dict[str, Any]) -> str:
        """Run the tool with the given arguments and return an observation.

        Tools should convert recoverable failures (missing files,
        non-zero exit codes, malformed arguments) into observation text
        rather than raising. Raising is reserved for programming errors
        the dispatcher cannot meaningfully recover from.
        """
        ...


class ToolDispatcher:
    """Routes ``ToolCall`` instances to registered ``Tool`` implementations.

    Tools are registered by their ``.name`` property. Re-registering an
    existing name overwrites the previous tool — useful for tests that
    want to swap a real CodeExecTool for a mock without rebuilding the
    dispatcher.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Subsequent registrations of the same name win."""
        self._tools[tool.name] = tool

    def execute(self, call: ToolCall) -> str:
        """Dispatch ``call`` to the tool registered under ``call.name``.

        Args:
            call: The parsed tool invocation.

        Returns:
            Observation text returned by the tool.

        Raises:
            ValueError: If no tool is registered under ``call.name``.
                Raised loudly so a typo or hallucinated tool name shows
                up in the trace rather than silently becoming a no-op.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            raise ValueError(f"No tool registered for {call.name!r}")
        return tool.execute(call.arguments)

    def has(self, name: str) -> bool:
        """Return True if a tool is registered under ``name``."""
        return name in self._tools

    def names(self) -> list[str]:
        """Return the registered tool names in registration order."""
        return list(self._tools.keys())


__all__ = ["Tool", "ToolCall", "ToolDispatcher"]
