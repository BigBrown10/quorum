# Quorum TypeScript Client

This package exposes two separate client surfaces:

- The core client for the Quorum `/resolve` HTTP API.
- The QuorumX helper for `/v1/quorumx` and `/v1/chat/completions`.

## Core Usage

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

## QuorumX Usage

```ts
import { createQuorumXClient } from "./src/quorumx.js";

const client = createQuorumXClient("http://127.0.0.1:8010");

const result = await client.run({
  task: "Review this patch for correctness and regressions.",
  config: {
    n_agents: 3,
    max_rounds: 2,
    model: "gpt-4o-mini",
    quorum_model: "gpt-4o-mini",
  },
});

console.log(result.answer);
console.log(result.unstable);
console.log(result.prompt_tokens);
console.log(result.completion_tokens);
```

Use `client.chatCompletions(...)` when you want the OpenAI-compatible chat envelope. The helper is JSON-first; if you need `stream=true`, call `fetch` directly so you can consume the SSE response from `/v1/chat/completions`.
