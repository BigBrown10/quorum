# QuorumX Implementation Plan

This document tracks the V2 buildout for QuorumX, the reasoning trust layer that sits on top of Quorum Core.

## Phase 1: Core Package Scaffold

- [x] Create the `quorumx` package and public entry points
- [x] Define `QuorumXConfig`, `AgentBenchmark`, and `QuorumXResult`
- [x] Add a lightweight adapter protocol for backends

## Phase 2: Mock Debate Engine

- [x] Implement a deterministic offline mock backend
- [x] Add the QuorumX run loop with sparse disagreement summaries
- [x] Bridge the run loop to Quorum Core consensus modes
- [x] Add the `@quorum_x(...)` decorator API

## Phase 3: Real LLM Backend

- [ ] Add a `ChatOpenAI` backend using `langchain-openai`
- [x] Keep the real backend optional and lazy-loaded
- [ ] Add configuration examples for API-key-based usage

## Phase 4: Framework Adapters

- [ ] Add thin adapters for LangChain, LangGraph, CrewAI, AutoGen, and OpenClaw-style runtimes
- [ ] Keep the adapters focused on input/output translation only

## Phase 5: Validation

- [x] Add unit tests for stable and unstable QuorumX outcomes
- [x] Add decorator tests and backend-selection tests
- [ ] Add usage notes and an example notebook for QuorumX

## Status

The V2 implementation is now starting with the package scaffold, a deterministic mock mode, and the decorator entry point. The real LLM backend and framework adapters remain to be added.