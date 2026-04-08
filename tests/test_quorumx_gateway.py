from __future__ import annotations

import asyncio
from typing import Any

from quorumx import QuorumXResult
from quorumx.http import chat_completions_payload, resolve_quorumx_payload
from quorumx.mcp import MCP_TOOL_NAME, QuorumXMCPServer


def test_resolve_quorumx_payload_emits_telemetry() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    response = resolve_quorumx_payload(
        {
            "task": "Draft a launch email for a product update.",
            "config": {"mock_mode": True, "consensus_mode": "quantum_ready"},
        },
        telemetry=lambda event, payload: events.append((event, payload)),
    )

    assert response["answer"]
    assert events[0][0] == "quorumx.resolve"
    assert events[0][1]["unstable"] in {True, False}


def test_chat_completions_payload_emits_telemetry() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    response = chat_completions_payload(
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Summarize this RFC in one sentence."},
            ],
            "config": {"mock_mode": True, "consensus_mode": "quantum_ready"},
        },
        telemetry=lambda event, payload: events.append((event, payload)),
    )

    assert response["object"] == "chat.completion"
    assert events[0][0] == "quorumx.chat_completions"
    total_tokens = events[0][1]["total_tokens"]
    assert isinstance(total_tokens, int)
    assert total_tokens > 0


def test_chat_completions_payload_preserves_message_history(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeEngine:
        def __init__(self, config: Any, backend: Any = None, personas: Any = None) -> None:
            captured["config"] = config

        def run(
            self,
            task: str,
            *,
            messages: list[dict[str, Any]] | None = None,
            system_instructions: str | None = None,
        ) -> QuorumXResult:
            captured["task"] = task
            captured["messages"] = messages
            captured["system_instructions"] = system_instructions
            return QuorumXResult(
                answer="Preserved",
                agreement_score=0.91,
                unstable=False,
                rounds_used=1,
                total_tokens=12,
                tokens_per_round=[12],
                benchmark=[],
                selected_agent_ids=["asserter_1"],
                consensus_mode="quantum_ready",
                rationale="ok",
            )

    monkeypatch.setattr("quorumx.http.QuorumX", FakeEngine)

    original_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Summarize this RFC in one sentence."},
        {"role": "assistant", "content": "Here is a draft."},
        {"role": "user", "content": "Revise it for concision."},
    ]

    response = chat_completions_payload(
        {
            "messages": original_messages,
            "system_instructions": "Follow the company style guide.",
            "config": {"mock_mode": True, "consensus_mode": "quantum_ready"},
        }
    )

    assert response["choices"][0]["message"]["content"] == "Preserved"
    assert captured["messages"] == original_messages
    assert captured["system_instructions"] == "Follow the company style guide."
    assert captured["task"] == "Revise it for concision."


def test_mcp_server_exposes_quorumx_run_tool() -> None:
    async def scenario() -> tuple[list[Any], Any]:
        server = QuorumXMCPServer()
        tools = await server.list_tools()
        result = await server.call_tool(
            MCP_TOOL_NAME,
            {
                "task": "Draft a launch email for a product update.",
                "config": {
                    "n_agents": 2,
                    "max_rounds": 1,
                    "mock_mode": True,
                },
            },
        )
        return tools, result

    tools, result = asyncio.run(scenario())

    assert len(tools) == 1
    assert tools[0].name == MCP_TOOL_NAME
    assert "task" in tools[0].inputSchema["required"]
    assert result.structured_content is not None
    assert result.structured_content["answer"]
    assert result.structured_content["consensus_mode"] == "quantum_ready"


def test_mcp_server_emits_telemetry() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    async def scenario() -> Any:
        server = QuorumXMCPServer(telemetry=lambda event, payload: events.append((event, payload)))
        return await server.call_tool(
            MCP_TOOL_NAME,
            {
                "task": "Draft a launch email for a product update.",
                "config": {"mock_mode": True, "max_rounds": 1},
            },
        )

    result = asyncio.run(scenario())

    assert result.structured_content is not None
    assert events[0][0] == "quorumx.resolve"
    assert events[1][0] == "quorumx.mcp.tool_call"
