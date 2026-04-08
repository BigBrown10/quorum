# Quorum Usage Guide

## Python Core

```python
from quorum_core import AgentOutput, resolve_consensus

candidates = [
    AgentOutput(id="a1", content="42", confidence=0.7),
    AgentOutput(id="a2", content="42", confidence=0.9),
    AgentOutput(id="a3", content="7", confidence=0.2),
]

result = resolve_consensus(candidates, mode="quantum_ready")
print(result.consensus_answer)
print(result.unstable)
```

## HTTP API

POST a JSON body like this to `/resolve`:

```json
{
  "candidates": [
    {"id": "a1", "content": "42", "confidence": 0.7},
    {"id": "a2", "content": "42", "confidence": 0.9}
  ],
  "mode": "weighted_majority"
}
```

## TypeScript Client

```ts
import { createQuorumClient } from "./src/index.js";

const client = createQuorumClient("http://127.0.0.1:8000");
const result = await client.resolveConsensus({
  candidates: [
    { id: "a1", content: "42", confidence: 0.7 },
    { id: "a2", content: "42", confidence: 0.9 },
  ],
  mode: "quantum_ready",
});
```

## MCP Server

Start the server with:

```bash
python -m quorum_mcp.server
```

It exposes `quorum_consensus` and returns the same JSON result shape as the HTTP API.

## QuorumX

QuorumX is the V2 reasoning trust layer. It runs a small debate across stance-based agents and returns a scored answer with explicit instability handling.

### Python SDK

Use the real backend by default and pass your API key through the config when you want to call a live model:

```python
import os

from quorumx import QuorumX, QuorumXConfig

config = QuorumXConfig(
    n_agents=3,
    max_rounds=2,
    consensus_mode="quantum_ready",
    model="gpt-4o-mini",
    quorum_model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

result = QuorumX(config).run(
    "Review this Python patch for correctness, edge cases, and regressions."
)

print(result.answer)
print(result.unstable)
```

For offline tests or CI, switch the backend explicitly:

```python
from quorumx import QuorumX, QuorumXConfig

config = QuorumXConfig(mock_mode=True)
result = QuorumX(config).run(
    "Draft a concise SDR follow-up email for a prospect who asked about pricing and security."
)
```

### HTTP Gateway

POST a task to `/v1/quorumx` when you want the native QuorumX request shape:

```json
{
  "task": "Review this Python patch for correctness, edge cases, and regressions.",
  "config": {
    "n_agents": 3,
    "max_rounds": 2,
    "model": "gpt-4o-mini",
    "quorum_model": "gpt-4o-mini"
  }
}
```

POST chat messages to `/v1/chat/completions` when you want an OpenAI-compatible facade:

```json
{
  "messages": [
    {"role": "system", "content": "You are a careful assistant."},
    {"role": "user", "content": "Draft a concise SDR follow-up email for a prospect who asked about pricing and security."}
  ],
  "config": {
    "n_agents": 3,
    "max_rounds": 2,
    "model": "gpt-4o-mini"
  }
}
```

### MCP Server

Start the QuorumX MCP server with:

```bash
python -m quorumx.mcp
```

It exposes the `quorumx.run` tool and returns the structured QuorumX result payload.

### Model Selection

- `model` is the primary debate model.
- `quorum_model` is the synthesis model used by the gateway path when you want a cheaper or smaller final-step model.
- If `quorum_model` is omitted, QuorumX falls back to `model`.
- Use `backend="mock"` only for tests, demos, or explicit offline runs.
- `request_timeout_seconds` controls how long the real backend waits before failing fast.

### Telemetry Hooks

The HTTP gateway and MCP server accept an optional telemetry callable for logging or metrics.

Pass a function shaped like `def hook(event: str, payload: dict[str, Any]) -> None` to:

- `resolve_quorumx_payload(..., telemetry=hook)`
- `chat_completions_payload(..., telemetry=hook)`
- `create_server(..., telemetry=hook)`
- `QuorumXMCPServer(telemetry=hook)`
