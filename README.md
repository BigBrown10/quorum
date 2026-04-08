# Quorum

Quorum is a consensus engine for distributed agent systems.

This repository currently contains:

- Core data models for agent outputs and consensus results
- Baseline consensus modes for simple and confidence-weighted majority voting
- A thin HTTP JSON API wrapper over the core
- A minimal MCP server that exposes `quorum_consensus`
- A TypeScript client for the HTTP API
- A result shape that can mark unstable outcomes explicitly

## Status

The Python core is being built first. HTTP, TypeScript, and MCP layers are being added on top of this core.

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

Integration guidance lives in [docs/agent-orchestrations.md](docs/agent-orchestrations.md), and runnable notebooks are in [notebooks](notebooks).

## Quantum Backends

Optional backend selection is controlled through environment variables:

- `QUORUM_OPTIMIZER=qiskit` or `QUORUM_OPTIMIZER=dwave`
- `QUORUM_ALLOW_CLASSICAL_FALLBACK=true` to fall back to the classical optimizer when a requested quantum backend is unavailable

## TypeScript Client

The TypeScript client lives in `clients/ts` and calls the same HTTP API contract exposed by the Python core.

## Release Checklist

See [docs/release-checklist.md](docs/release-checklist.md) before a private push or branch cut.
