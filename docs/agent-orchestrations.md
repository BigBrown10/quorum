# Quorum Across Agent Orchestrations

Quorum Core is the consensus backend. QuorumX-style orchestration layers can wrap it, but the integration surface stays the same:

1. Gather candidate agent outputs.
2. Normalize them into `AgentOutput` records.
3. Call `resolve_consensus(...)` or the HTTP `/resolve` endpoint.
4. Use the returned `unstable` flag to decide whether to proceed, retry, or escalate.

The same consensus engine can be used with CrewAI, LangGraph, AutoGen, OpenClaw-style setups, and other orchestration frameworks without changing the core API.

## Integration Pattern

The recommended integration pattern is a thin adapter:

- The orchestration framework generates responses.
- The adapter maps each response into `AgentOutput`.
- Quorum returns the consensus result and diagnostics.
- The framework continues with a stable answer or handles `NO CONSENSUS` explicitly.

There are two common ways to wire that adapter:

- Function-level wrapper: wrap the final `task -> answer` function and feed its outputs to Quorum.
- Gateway or proxy: expose Quorum as a service and point the agent runtime at the HTTP API or an OpenAI-compatible facade.

## CrewAI

Use Quorum after the agent crew produces candidate responses. Keep the orchestration logic in CrewAI and let Quorum decide whether the crew reached a stable answer.

## LangGraph

Use Quorum as a post-node or reducer step. LangGraph nodes can emit outputs into a shared list, then the reducer can call Quorum once the graph has enough candidates.

## AutoGen

Use Quorum after a round of multi-agent discussion or before final user-facing output. AutoGen can keep the conversation logic, while Quorum handles convergence and instability detection.

## OpenClaw and Similar Systems

OpenClaw can integrate in either of the same two patterns:

- As an HTTP tool or skill that calls Quorum with the current task or candidate answer.
- As an upstream proxy so one agent's model calls route through Quorum first.

Any orchestration that can emit a list of candidate answers can use Quorum. The required shape is candidate content, optional confidence, and optional metadata.

## Practical Advice

- Prefer semantic embeddings when you already have them. The core now defaults to TF-IDF embeddings and also supports a local SentenceTransformer backend.
- Use the `unstable` flag to avoid pretending there is consensus when there is not.
- Keep the adapter thin so the same integration works across frameworks.
- If you are building your own orchestration layer, call the HTTP API so Python and TypeScript clients share the same wire format.
