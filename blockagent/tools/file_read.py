"""FileReadTool — read a file's contents and return as observation text.

Recoverable I/O failures (missing file, permission error, undecodable
bytes) are converted into observation text so the agent loop continues
to the next step. The tool reads with ``utf-8-sig`` to silently strip
a BOM, mirroring the parser's seed-file handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class FileReadTool:
    """Read a UTF-8 text file and return its contents (truncated if large)."""

    name: str = "file_read"

    def __init__(self, default_max_chars: int = 8000) -> None:
        """Construct the tool.

        Args:
            default_max_chars: Hard cap on returned content length.
                Used when a tool call does not supply its own ``max_chars``.
                The default is sized for a few thousand tokens of context
                without overwhelming a small-model prompt.
        """
        self.default_max_chars = default_max_chars

    def execute(self, arguments: dict[str, Any]) -> str:
        """Read the file at ``arguments['path']``.

        Recognised arguments:
            path: str — file path. Required.
            max_chars: int — cap on returned content length. Default
                falls back to ``self.default_max_chars``.

        Returns:
            File contents up to ``max_chars`` (with a truncation marker
            appended if cut). Returns ``"ERROR: ..."`` text for missing
            file, permission denied, or non-UTF-8 content.
        """
        raw_path = str(arguments.get("path", ""))
        if not raw_path:
            return "ERROR: no 'path' argument provided"

        max_chars = int(arguments.get("max_chars", self.default_max_chars))
        path = Path(raw_path)

        try:
            content = path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return f"ERROR: file not found: {raw_path}"
        except PermissionError:
            return f"ERROR: permission denied: {raw_path}"
        except UnicodeDecodeError:
            return f"ERROR: file is not valid UTF-8: {raw_path}"
        except IsADirectoryError:  # pragma: no cover - OS-conditional
            return f"ERROR: path is a directory: {raw_path}"

        if len(content) > max_chars:
            content = (
                content[:max_chars]
                + f"\n... (truncated at {max_chars} chars)"
            )
        return content


__all__ = ["FileReadTool"]
