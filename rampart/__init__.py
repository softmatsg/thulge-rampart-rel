"""Public API of the RAMPART instruction-block registry library.

Re-exports the small surface that downstream callers bind against.
Internal helpers stay in their submodules so that ``import rampart``
does not advertise more than the contract.
"""

from rampart.block import (
    BlockSourceKind,
    InstructionBlock,
    SeedMutationError,
)
from rampart.compiler import (
    CompileResult,
    DryRunResult,
    compile,
    compile_dry_run,
    token_count,
)
from rampart.config import RAMPARTConfig, default_tokeniser
from rampart.eviction import DefaultEvictionPolicy, EvictionPolicy
from rampart.parser import (
    PARSERS,
    BlockSource,
    ParseError,
    UnsupportedFormatError,
    parse_file,
)
from rampart.registry import (
    BlockNotFoundError,
    BlockRegistry,
    EmbeddingModel,
    EvictionError,
    get_by_label,
)
from rampart.scorer import (
    SentenceTransformerEmbedder,
    embed_all,
    embed_text,
)
from rampart.security import generate_runtime_id
from rampart.seed_registry import SeedRegistry

__all__ = [
    "PARSERS",
    "BlockNotFoundError",
    "BlockRegistry",
    "BlockSource",
    "BlockSourceKind",
    "CompileResult",
    "DefaultEvictionPolicy",
    "DryRunResult",
    "EmbeddingModel",
    "EvictionError",
    "EvictionPolicy",
    "InstructionBlock",
    "ParseError",
    "RAMPARTConfig",
    "SeedMutationError",
    "SeedRegistry",
    "SentenceTransformerEmbedder",
    "UnsupportedFormatError",
    "compile",
    "compile_dry_run",
    "default_tokeniser",
    "embed_all",
    "embed_text",
    "generate_runtime_id",
    "get_by_label",
    "parse_file",
    "token_count",
]

__version__ = "0.1.0"
