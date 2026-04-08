# QuorumX V2 Vision

Quorum Core is the consensus backend in this repository. QuorumX is the higher-level reasoning trust layer that can sit on top of it for stance-based multi-agent debate, sparse disagreement summaries, and framework adapters.

## Research Basis

- Mixture-of-Agents is real and useful: ensemble-style systems with a judge model have shown strong benchmark performance over single-model baselines in some regimes. [arxiv](https://arxiv.org/html/2406.04692v1)
- Judge LLMs are fragile: they add token cost, can favor long or polished outputs, and are vulnerable to sycophancy and prompt injection. [aclanthology](https://aclanthology.org/2025.naacl-long.475.pdf)
- Sparse multi-agent debate can reduce token waste: passing only disagreement edges instead of full transcripts keeps consensus quality while cutting communication overhead. [arxiv](https://arxiv.org/html/2502.04790v2)
- Enterprises are already buying AI trust layers for masking, policy enforcement, toxicity detection, and audit logging, but not reasoning consensus. [linkedin](https://www.linkedin.com/pulse/salesforce-einstein-trust-layer-how-works-why-matters-uz9uc)
- Quantum advantage is expected first in optimisation workloads, so consensus formulated as QUBO keeps the system ready for future offload while classical backends remain the default. [comparethecloud](https://www.comparethecloud.net/opinions/ibm-2026-quantum-computing-beats-classical)

## Mini-PRD

### Problem

- Single-agent outputs can be wrong without obvious warning.
- Multi-agent systems are powerful but expensive and hard to coordinate.
- Security trust layers exist, but there is no standard reasoning trust layer.

### Vision

QuorumX is the reasoning trust layer for AI agents. It wraps existing agent runtimes, spins up a small swarm, runs efficient debate and consensus, and returns a scored answer that is either stable or flagged for human review.

### Target Users

- AI engineers building LangChain, LangGraph, CrewAI, AutoGen, Swarm, or custom coding agents.
- Product teams shipping coding copilots, SDR agents, and internal copilots.
- Platform teams building AI gateways or trust layers that want a reasoning module alongside their security module.

### Core Use Cases

- Coding copilots can verify generated code or diffs before surfacing them.
- SDR agents can critique outbound email before sending.
- Internal copilots can mark contested answers as unstable so they do not auto-execute tools or transactions.

### Functional Requirements

- QuorumX.run(task: str) -> QuorumXResult.
- Configurable agent count and debate rounds.
- Sparse debate where only disagreement summaries flow between rounds.
- Pluggable consensus backend, with classical weighted majority and quantum-ready graph/QUBO modes.
- Decorator or adapter APIs for existing agent functions.

### Non-Functional Requirements

- Works offline in a deterministic mock mode.
- Clear typed Python API with no framework lock-in.
- Reasonable default token usage.
- Easy to self-host.

### Out of Scope

- Full enterprise security trust layer features such as PII classification, legal-grade auditing, or identity systems.
- World-scale swarm simulation.

### Success Metrics

- Integrations that use QuorumX as middleware or a decorator.
- Reduction in silent wrong answers versus a single-agent baseline.
- Token cost reduction versus naive debate.
- Later: enterprise proof-of-concepts that plug QuorumX into an existing AI gateway.

## Repo Structure

- `quorum_core/` remains the low-level consensus backend.
- `quorumx/` now provides the initial QuorumX scaffold and wraps `quorum_core` for higher-level debate workflows.
- Thin adapters should keep framework integrations isolated from core consensus logic.

## Integration Notes

- Function-level wrappers are the simplest integration point, and the first QuorumX decorator implementation already follows this pattern.
- Gateway or proxy integrations work well when the agent runtime already routes model calls through a central service.
- Framework adapters for LangChain, LangGraph, CrewAI, AutoGen, and OpenClaw should stay thin and call the same Quorum Core contract.