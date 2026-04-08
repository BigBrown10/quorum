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
