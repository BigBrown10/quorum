# Quorum

Quorum is a consensus engine for distributed agent systems with a quantum-ready optimization path.

This repository currently contains:

- Core data models for agent outputs and consensus results
- Baseline consensus modes for simple and confidence-weighted majority voting
- Graph/QUBO consensus with exact classical solving for small problems
- Semantic text embeddings with TF-IDF default similarity and an optional local SentenceTransformer backend
- QuorumX package scaffold with a mock debate loop, config models, and a decorator API
- Optional quantum backends for Qiskit and D-Wave via environment selection
- A thin HTTP JSON API wrapper over the core
- A minimal MCP server that exposes `quorum_consensus`
- A TypeScript client for the HTTP API
- A result shape that can mark unstable outcomes explicitly

## Status

The Python core, HTTP API, TypeScript client, and MCP server are in place. The consensus engine supports classical and quantum-ready optimization paths, with TF-IDF text embeddings by default, an optional local SentenceTransformer backend, and optional Qiskit and D-Wave backends selected at runtime. QuorumX now has a real-mode-first SDK, a mock backend for CI and offline use, and HTTP and MCP entry points for agent integrations.

## V2 Direction

Quorum Core is the low-level consensus backend. QuorumX now sits on top as the reasoning trust layer with stance-based multi-agent debate, sparse disagreement summaries, a Python SDK, adapter helpers, an HTTP gateway, and an MCP server. Framework-specific adapters and usage examples are the next layer of work.

See [docs/v2-vision.md](docs/v2-vision.md) for the research basis, mini-PRD, and integration notes, and [docs/quorumx-implementation-plan.md](docs/quorumx-implementation-plan.md) for the build checklist.

## QuorumX

QuorumX exposes two primary entry points:

- `POST /v1/quorumx` for the native QuorumX request format
- `POST /v1/chat/completions` for OpenAI-compatible chat completion calls that route through QuorumX

The default `QuorumXConfig` path is real-backend oriented. Use `backend="mock"` only for tests, demos, or offline runs.

```python
import os

from quorumx import QuorumX, QuorumXConfig, run_langchain_consensus

config = QuorumXConfig(api_key=os.getenv("OPENAI_API_KEY"))
result = QuorumX(config).run("Review this patch for correctness and regressions.")

consensus = run_langchain_consensus([
    {"content": "Ship the patch"},
    {"content": "Ship the patch"},
    {"content": "Add tests first"},
])
```

## Local API

Run the HTTP API with:

```bash
python -m quorum_core.api
```

The API exposes:

- `GET /health`
- `POST /resolve`

The request body for `POST /resolve` is a JSON object with `candidates` and optional `mode`.

## MCP Server

Run the MCP server with:

```bash
python -m quorum_mcp.server
```

It exposes a single tool named `quorum_consensus` and routes requests through the same core consensus logic as the HTTP API.

See [docs/mcp-config.md](docs/mcp-config.md) for sample Claude Desktop and Cursor configuration snippets.

## Orchestration Guides

Integration guidance lives in [docs/agent-orchestrations.md](docs/agent-orchestrations.md), QuorumX usage examples live in [docs/usage.md](docs/usage.md), and runnable notebooks are in [notebooks](notebooks).

## Embeddings

Quorum uses semantic text similarity to build disagreement graphs.

By default, the Python core uses TF-IDF embeddings for a dependency-light local path.

For higher-quality semantic matching, install the optional embeddings extra and use the local SentenceTransformer backend. The model downloads on first use:

```bash
pip install ".[embeddings]"
```

```python
from quorum_core.embeddings import SentenceTransformerBackend, embed_texts

backend = SentenceTransformerBackend()
vectors = embed_texts(
    ["Paris is the capital of France", "Paris is the capital city of France"],
    backend=backend,
)
```

The hash-based embedding backend has been removed.

## Quantum Backends

Optional backend selection is controlled through environment variables:

- `QUORUM_OPTIMIZER=qiskit` or `QUORUM_OPTIMIZER=dwave`
- `QUORUM_ALLOW_CLASSICAL_FALLBACK=true` to fall back to the classical optimizer when a requested quantum backend is unavailable

The quantum path maps consensus to a QUBO objective and can use Qiskit or D-Wave when those packages and credentials are available.

## TypeScript Client

The TypeScript client lives in `clients/ts` and calls the same HTTP API contract exposed by the Python core.

## Release Checklist

See [docs/release-checklist.md](docs/release-checklist.md) before a private push or branch cut.
