# Quorum Implementation Plan

This file is the working tracker for the repository. Mark items complete as the implementation lands.

## Done

- [x] Initialize the repo skeleton and Python packaging
- [x] Define the core data models for agent outputs and consensus results
- [x] Implement baseline consensus modes: `simple_majority` and `weighted_majority`
- [x] Add explicit `unstable` handling that returns `NO CONSENSUS`
- [x] Add the HTTP JSON API wrapper for `/health` and `/resolve`
- [x] Add the MCP server scaffold with the `quorum_consensus` tool
- [x] Add a TypeScript client for the HTTP API
- [x] Add initial unit tests for core, API, and MCP behavior
- [x] Add the graph/QUBO consensus primitives and classical optimizer path
- [x] Add a real optimizer implementation for the graph/QUBO path beyond the current classical greedy/local-search baseline
- [x] Add optional quantum backends such as Qiskit and D-Wave behind extras or environment flags
- [x] Add richer clustering and embedding support for text-based candidate similarity
- [x] Add benchmark tests comparing quorum against majority-vote baselines on synthetic tasks
- [x] Add example usage guides for CrewAI, LangGraph, and AutoGen
- [x] Add example notebooks for CrewAI, LangGraph, and AutoGen
- [x] Add packaging and release wiring for the TypeScript client if it becomes a published package
- [x] Add deployment notes and sample configs for MCP clients like Claude Desktop and Cursor

## Remaining
None.

## Current API Surface

- Python core: `src/quorum_core`
- HTTP API: `src/quorum_core/api.py`
- MCP server: `src/quorum_mcp/server.py`
- TypeScript client: `clients/ts`

## Update Rule

When a task is finished, change its checkbox to `[x]` and keep this document as the source of truth for project progress.
