# Quorum Usage Guide

## QuorumX Golden Path

QuorumX is the framework-agnostic reasoning trust layer. Start here if you want a stable, scored answer in front of an LLM or multi-agent workflow.

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
    "Review this Python patch for correctness, edge cases, and regressions.",
    messages=[
        {"role": "system", "content": "You are a careful reviewer."},
        {"role": "user", "content": "Focus on correctness, tests, and failure modes."},
    ],
    system_instructions="Be concise and concrete.",
)

print({
    "answer": result.answer,
    "unstable": result.unstable,
    "agreement_score": result.agreement_score,
    "rounds_used": result.rounds_used,
    "prompt_tokens": result.prompt_tokens,
    "completion_tokens": result.completion_tokens,
    "total_tokens": result.total_tokens,
})
```

A typical `QuorumXResult` includes the final answer, instability flag, agreement score, token accounting, and the per-agent benchmark snapshot. For configuration details, personas, and telemetry hooks, see [docs/quorumx-config.md](docs/quorumx-config.md).

## HTTP Gateway

Run the QuorumX HTTP gateway with:

```bash
python -m quorumx.http
```

The gateway exposes:

- `POST /v1/quorumx`
- `POST /v1/chat/completions`

A native QuorumX request looks like this:

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

An OpenAI-compatible chat request looks like this:

```json
{
  "messages": [
    {"role": "system", "content": "You are a careful assistant."},
    {"role": "user", "content": "Draft a concise SDR follow-up email for a prospect who asked about pricing and security."}
  ],
  "system_instructions": "Be direct and avoid fluff.",
  "config": {
    "n_agents": 3,
    "max_rounds": 2,
    "model": "gpt-4o-mini"
  }
}
```

If you send `"stream": true` to `/v1/chat/completions`, the gateway returns SSE with a full-answer chunk and a final `[DONE]` event so OpenAI-style UIs do not hang.

Example response fields:

```json
{
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 124,
    "completion_tokens": 58,
    "total_tokens": 182
  },
  "quorumx": {
    "unstable": false,
    "agreement_score": 0.84,
    "rounds_used": 2
  }
}
```

## MCP Server

Run the QuorumX MCP server with:

```bash
python -m quorumx.mcp
```

It exposes the `quorumx.run` tool. If you need to host it inside another process, use `QuorumXMCPServer.run()` or `create_server()` from `quorumx.mcp`.

## Quorum Core

Use Quorum Core directly when you want the lower-level consensus engine without the debate layer.

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

The core HTTP API still lives at `python -m quorum_core.api` with `GET /health` and `POST /resolve`.

## TypeScript

The existing TypeScript client in `clients/ts` targets the core `/resolve` API.

For QuorumX, use `createQuorumXClient` from [clients/ts/src/quorumx.ts](clients/ts/src/quorumx.ts) against `/v1/quorumx` or `/v1/chat/completions`.

If you need raw streaming support, call `fetch` directly so you can read the SSE response from `/v1/chat/completions` when `"stream": true` is set.

## Adapters

After you have the QuorumX golden path working, add orchestration-specific wrappers with `quorumx.adapters`.

Use the adapter helpers for LangChain, LangGraph, CrewAI, AutoGen, and OpenClaw-style runtimes when you want to keep the framework logic outside QuorumX and only normalize candidate outputs at the edge.
