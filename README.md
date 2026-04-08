# Quorum

Quorum is a consensus engine for distributed agent systems with a quantum-ready optimization path.

## Who Should Use What

Use QuorumX if you want a reasoning trust layer in front of an LLM or multi-agent system. It is the developer-facing V2 surface: Python SDK, HTTP gateway, MCP tool, and adapter helpers.

Use Quorum Core directly if you are building custom consensus algorithms, research tooling, or a lower-level consensus primitive.

## Entry Points by Layer

| Layer | Python | HTTP | MCP | TypeScript |
| --- | --- | --- | --- | --- |
| Core | `from quorum_core import resolve_consensus, AgentOutput, ConsensusResult` | `python -m quorum_core.api` -> `/health`, `/resolve` | `python -m quorum_mcp.server` -> `quorum_consensus` | `clients/ts` core client for `/resolve` |
| QuorumX | `from quorumx import QuorumX, quorum_x, QuorumXConfig` | `python -m quorumx.http` or `create_server()` -> `/v1/quorumx`, `/v1/chat/completions` | `python -m quorumx.mcp` or `QuorumXMCPServer.run()` -> `quorumx.run` | `clients/ts/src/quorumx.ts` helper for QuorumX endpoints |

## QuorumX

QuorumX runs stance-based multi-agent debate on top of Quorum Core. It preserves the full message history, accepts `system_instructions`, and returns a scored answer with explicit instability handling.

```python
import os

from quorumx import QuorumX, QuorumXConfig

config = QuorumXConfig(
    n_agents=3,
    max_rounds=2,
    consensus_mode="quantum_ready",
    api_key=os.getenv("OPENAI_API_KEY"),
)

result = QuorumX(config).run(
    "Review this patch for correctness and regressions.",
    messages=[
        {"role": "system", "content": "You are a careful reviewer."},
        {"role": "user", "content": "Focus on correctness, edge cases, and tests."},
    ],
    system_instructions="Be concise and concrete.",
)

print(result.answer)
print(result.unstable)
print(result.agreement_score)
```

Use `backend="mock"` only for tests, demos, or offline runs. See [docs/quorumx-config.md](docs/quorumx-config.md) for config, persona, and telemetry details.

See [docs/v2-vision.md](docs/v2-vision.md) for the research basis and [docs/quorumx-implementation-plan.md](docs/quorumx-implementation-plan.md) for the historical V2 checklist.

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

The TypeScript client in `clients/ts` still targets the core `/resolve` API.

For QuorumX, use `createQuorumXClient` from [clients/ts/src/quorumx.ts](clients/ts/src/quorumx.ts) against `/v1/quorumx` or `/v1/chat/completions`. The helper is JSON-first; if you need `stream=true`, call `fetch` directly so you can consume the SSE response.

## Release Checklist

See [docs/release-checklist.md](docs/release-checklist.md) before a private push or branch cut.
