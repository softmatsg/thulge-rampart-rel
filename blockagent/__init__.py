"""Reference agent built on top of RAMPART.

Re-exports the public surface of the backend layer for convenience.
Importing this package pulls in nothing beyond the standard library and
the rampart core; concrete backends with heavy optional dependencies
(llama-cpp-python, httpx, ...) defer those imports to their constructors.
"""

from blockagent.agent import BlockAgent, parse_tool_calls
from blockagent.backends import (
    Backend,
    GenerateResult,
    LlamaCppBackend,
    OllamaBackend,
    OpenAICompatBackend,
)
from blockagent.eval import (
    LLMEvaluator,
    SuccessEvaluator,
    TaskResult,
)
from blockagent.tools import (
    CodeExecTool,
    FileReadTool,
    Tool,
    ToolCall,
    ToolDispatcher,
)

__all__ = [
    "Backend",
    "BlockAgent",
    "CodeExecTool",
    "FileReadTool",
    "GenerateResult",
    "LLMEvaluator",
    "LlamaCppBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "SuccessEvaluator",
    "TaskResult",
    "Tool",
    "ToolCall",
    "ToolDispatcher",
    "parse_tool_calls",
]
