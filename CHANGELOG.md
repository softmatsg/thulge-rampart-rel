# Changelog

All notable changes to this project are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/) and the project
uses semantic versioning.

## [0.1.0] — initial public release

First public release of the RAMPART library.

### Added

* `rampart` core package:
  * `BlockRegistry` — ordered, mutable, in-RAM store of instruction
    blocks with promote / demote / evict / reorder / tag-query
    primitives.
  * `compile(max_tokens)` — order-walking prompt assembly under a
    token budget.
  * `RAMPARTConfig` — session-level tunables (token budget, embedding
    model, eviction policy, tokeniser).
  * YAML-frontmatter parser, SKILL.md / CLAUDE.md compatibility path,
    and a pluggable LLM-splitter hook for free-form files.
  * `SeedRegistry` — namespaced library facade for loading and
    composing seed files.
  * `DefaultEvictionPolicy` and an `EvictionPolicy` protocol for
    custom ranking.
  * `SentenceTransformerEmbedder` and `embed_text` / `embed_all`
    helpers (optional dependency `[embed]`).

* `blockagent` reference agent package:
  * `BlockAgent.run_task` — eight-stage agent loop with optional
    write-back of diagnostic blocks on failure.
  * Backend adapters: `OllamaBackend`, `OpenAICompatBackend`,
    `LlamaCppBackend`. All implement the `Backend` protocol in
    `blockagent.backends.base`.
  * `Tool` / `ToolCall` / `ToolDispatcher` with reference
    implementations for code execution and file reading.
  * `TaskResult` result type and the `SuccessEvaluator` protocol,
    plus an `LLMEvaluator` default.

* `seeds/assistant_style.md` — example seed file showing the
  YAML-frontmatter format with six commonly-useful style and
  output-format blocks.

* Test suite under `tests/` covering registry mutations, the
  compiler, the parser, the scorer, the seed-registry facade, the
  skill-file path, the agent loop, and each backend with
  `httpx.MockTransport` (no live LLM required).
