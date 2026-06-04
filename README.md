# RAMPART

**Registry-based Agentic Memory with Priority-Aware Runtime Transformation**

Nikodem Tomczak, Thulge Labs, Singapore

[![arXiv](https://img.shields.io/badge/arXiv-2606.04628-b31b1b.svg)](https://arxiv.org/abs/2606.04628)

Copyright (c) 2026 Nikodem Tomczak, Thulge Labs. All rights reserved.

---

A small Python library for managing an ordered, mutable set of
natural-language instruction blocks that compile into a single prompt
string under a token budget. The order is the lever: blocks at the
front of the registry are reliably attended to; blocks buried mid-context
are forgotten ("Lost in the Middle"). RAMPART makes the order
programmatically manipulable so an application can promote, demote, or
cluster blocks in response to the task at hand.

The library does no I/O of its own beyond reading seed files at cold
start. Embeddings, LLM calls, and tool execution are caller-controlled.
A reference agent loop (`BlockAgent`) and a small set of backend
adapters (Ollama, any OpenAI-compatible HTTP endpoint, llama-cpp-python)
ship alongside the core registry but are entirely optional.

> **Status**: alpha (0.1.0). The public API is stable but evolving;
> breaking changes will be flagged in `CHANGELOG.md`.

## Installation

```bash
pip install rampart                  # core: registry, parser, compiler, scorer
pip install "rampart[embed]"         # add sentence-transformers for relevance scoring
pip install "rampart[agent]"         # add httpx + the BlockAgent loop
pip install "rampart[dev]"           # tests + linting tooling
```

Requires Python 3.11+.

## Quick start

```python
from rampart import BlockRegistry

# Load a seed file (YAML-frontmatter format — see seeds/ for examples).
registry = BlockRegistry.from_files(["seeds/assistant_style.md"])

# Compile into a single prompt string within a token budget.
result = registry.compile(max_tokens=4000)
print(result.prompt)

# Append your task and send to your LLM of choice.
prompt = f"{result.prompt}\n\nTASK:\n{user_task}"
```

## The block file format

A seed file contains one or more instruction blocks delimited by
`---` lines, each with YAML frontmatter:

```markdown
---
name: language
priority: 0.9
tags: [style, output_format]
---
All replies must be in English regardless of the input language.

---
name: length
priority: 0.7
tags: [style]
---
Keep replies focused and brief. Default to under 200 words.
```

* `name` — unique within the file. Used for promote/evict operations.
* `priority` (0.0–1.0, default 0.5) — input to the eviction-policy ranking.
* `tags` (optional list) — used by `find_by_tags` and tag-aware eviction policies.

Files with no frontmatter at all load as a single anonymous block named
after the filename stem. That is the `SKILL.md` / `CLAUDE.md`
compatibility path; for those files RAMPART splits on `##` and `###`
headers automatically via `SeedRegistry.from_skill_file`.

## Three usage patterns

### 1. Compile a fixed registry

```python
from rampart import BlockRegistry

registry = BlockRegistry.from_files(["seeds/assistant_style.md"])
prompt = registry.compile(max_tokens=4000).prompt
```

### 2. Score and promote the most relevant blocks for a task

```python
from rampart import BlockRegistry
from rampart.scorer import SentenceTransformerEmbedder, embed_text

embedder = SentenceTransformerEmbedder()
registry = BlockRegistry.from_files(
    ["seeds/assistant_style.md"],
    embedding_model=embedder,
)

# Pick the top-3 most-relevant blocks for the task and promote them
# to the front of the registry so they survive the token budget.
task = "Write a short blog post about distributed tracing."
task_vector = embed_text(task, embedder)
top = registry.find_relevant(task_vector, k=3)
for block in top:
    registry.promote(block.runtime_id, to_front=True)

prompt = registry.compile(max_tokens=4000).prompt
```

### 3. Run a task end-to-end with the reference agent loop

```python
from blockagent import BlockAgent, OllamaBackend, LLMEvaluator
from rampart import BlockRegistry

registry = BlockRegistry.from_files(["seeds/assistant_style.md"])
backend = OllamaBackend(model="llama3.1:8b-instruct-q4_K_M")
evaluator = LLMEvaluator(backend=backend)

agent = BlockAgent(
    registry=registry,
    backend=backend,
    success_evaluator=evaluator,
)

result = agent.run_task("Summarise the attached document in three bullets.")
print(result.output)
print(f"success={result.success}, blocks_used={len(result.compiled_blocks)}")
```

The agent walks an eight-stage loop documented in
`BlockAgent.run_task`: score-and-promote, compile, task call, success
check, (on failure) diagnosis call, write-back of a diagnostic block
into the registry, and metric capture.

## Backends

| Backend | Module | Optional dependency |
|---|---|---|
| Ollama (local HTTP) | `blockagent.backends.OllamaBackend` | `pip install "rampart[http]"` |
| OpenAI-compatible HTTP | `blockagent.backends.OpenAICompatBackend` | `pip install "rampart[http]"` |
| llama-cpp-python (in-process) | `blockagent.backends.LlamaCppBackend` | install `llama-cpp-python` manually with the right CUDA/Metal flavour |

All three implement the `Backend` protocol in `blockagent.backends.base`,
so swapping models is a one-line change at the call site.

## Project layout

```
rampart/                   # the core registry + compiler + parser
    block.py               # InstructionBlock dataclass
    compiler.py            # compile() — order-walking prompt assembly
    config.py              # RAMPARTConfig, default tokeniser
    eviction.py            # DefaultEvictionPolicy + protocol
    parser.py              # YAML-frontmatter + skill-file parsers
    registry.py            # BlockRegistry — the ordered store with mutations
    scorer.py              # embedding wiring (sentence-transformers)
    security.py            # runtime-id generation (UUID4)
    seed_registry.py       # SeedRegistry — namespaced library facade
    skill.py               # SKILL.md / CLAUDE.md import path
    utils.py               # validate_splitter and other helpers

blockagent/                # the reference agent loop and backend adapters
    agent.py               # BlockAgent.run_task
    backends/              # Ollama, OpenAI-compat, llama-cpp-python
    tools/                 # Tool, ToolCall, ToolDispatcher + reference tools
    eval/                  # TaskResult + SuccessEvaluator protocol
                           # (LLMEvaluator is the default)

tests/                     # pytest suite covering both packages
seeds/                     # example seed files for the quick start
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

The suite uses `httpx.MockTransport` for HTTP-backend tests, so it
runs offline without a live Ollama or llama-server process.

## License

Source-available, non-commercial. Free to use, copy, and modify for
academic research, personal projects, and evaluation. Commercial use
in production systems requires a separate commercial licence from
Thulge Labs. See [LICENSE](LICENSE) for the full text.

## Citation

```bibtex
@article{tomczak2026rampart,
  title={RAMPART: Registry-based Agentic Memory with Priority-Aware
         Runtime Transformation},
  author={Tomczak, Nikodem},
  journal={arXiv preprint arXiv:2606.04628},
  year={2026},
  url={https://arxiv.org/abs/2606.04628}
}
```
