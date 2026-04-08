# Quorum TypeScript Client

This package is a thin TypeScript client for the Quorum HTTP API.

## Usage

```ts
import { createQuorumClient } from "./src/index.js";

const client = createQuorumClient("http://127.0.0.1:8000");
const result = await client.resolveConsensus({
  candidates: [
    { id: "a1", content: "42", confidence: 0.7 },
    { id: "a2", content: "42", confidence: 0.9 },
    { id: "a3", content: "7", confidence: 0.2 },
  ],
  mode: "weighted_majority",
});

console.log(result.consensus_answer);
```
