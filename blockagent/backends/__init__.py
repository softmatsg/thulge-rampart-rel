"""Public surface of the backend layer.

Re-exports the Backend Protocol, the GenerateResult shape, and the
concrete backend classes. Importing this module does NOT pull in any
heavy optional dependency — each concrete backend defers its own import
to its constructor and raises a typed ImportError if the optional extra
is missing.
"""

from blockagent.backends.base import Backend, GenerateResult
from blockagent.backends.llamacpp import LlamaCppBackend
from blockagent.backends.ollama import OllamaBackend
from blockagent.backends.openai_compat import OpenAICompatBackend

__all__ = [
    "Backend",
    "GenerateResult",
    "LlamaCppBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
]
