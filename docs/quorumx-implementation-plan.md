# QuorumX V2 Implementation Plan

This tracks the QuorumX V2 buildout, the reasoning trust layer that sits on top of Quorum Core.

QuorumX now ships with a Python SDK, a real-mode-first debate engine, adapter helpers, an HTTP gateway, and an MCP server. The tracked V2 checklist is complete, and any further work is now optional hardening or future integrations.

## Phase 1: Core Package Scaffold

- [x] Create the `quorumx` package and public entry points
- [x] Define `QuorumXConfig`, `AgentBenchmark`, and `QuorumXResult`
- [x] Add a backend protocol for LLM providers
- [x] Keep `quorum_core/` as the low-level consensus backend and `quorumx/` as the V2 trust layer

## Phase 2: Debate Engine

- [x] Implement stance-based agents such as asserter, skeptic, synthesiser, contrarian, and verifier
- [x] Implement multi-round debate
- [x] Pass sparse disagreement summaries between rounds
- [x] Bridge the run loop to Quorum Core consensus modes where appropriate
- [x] Add the `@quorum_x(...)` decorator API

## Phase 3: Real LLM Backend

- [x] Add a `ChatOpenAI` backend using `langchain-openai`
- [x] Support a generic `call_llm(prompt, config) -> str` hook for custom providers
- [x] Keep the real backend optional and lazy-loaded
- [x] Provide a mock backend for tests and CI
- [x] Return `unstable=True` on backend failure instead of crashing
- [x] Add request timeouts to the LLM client path
- [x] Add API-key usage examples to the docs
- [x] Add guidance for selecting the primary QuorumX model and the cheaper quorum model

## Phase 4: Framework Adapters

- [x] Add reusable adapter normalization helpers in `quorumx.adapters`
- [x] Add thin adapters for LangChain
- [x] Add thin adapters for LangGraph
- [x] Add thin adapters for CrewAI
- [x] Add thin adapters for AutoGen
- [x] Add an OpenClaw integration example
- [x] Keep adapters focused on input/output translation only

## Phase 5: Validation and Documentation

- [x] Add unit tests for stable and unstable QuorumX outcomes
- [x] Add decorator tests and backend-selection tests
- [x] Add gateway tests for the HTTP and MCP surfaces
- [x] Add a usage note for a hello-world coding agent example
- [x] Add a usage note for an SDR email example
- [x] Keep README examples aligned with the public QuorumX API

## Phase 6: MCP Server and HTTP Gateway

- [x] Add `POST /v1/quorumx`
- [x] Add `POST /v1/chat/completions`
- [x] Add the `quorumx.run` MCP tool
- [x] Add basic structured logging for each call
- [x] Ship example configs for Claude Desktop, Cursor, and similar clients
- [x] Add hooks for app logging and metrics systems

## Status

QuorumX is now usable as a Python SDK and as a gateway-backed tool surface. The real backend is the default path, mock mode is reserved for tests and explicit offline use, and the tracked V2 implementation work is complete.