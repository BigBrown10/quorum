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


def test_resolve_quorumx_payload_returns_structured_result() -> None:
    response = resolve_quorumx_payload(
        {
            "task": "Draft a launch email for a product update.",
            "config": {
                "n_agents": 3,
                "max_rounds": 2,
                "consensus_mode": "quantum_ready",
                "mock_mode": True,
            },
        }
    )

    assert response["answer"]
    assert response["consensus_mode"] == "quantum_ready"
    assert response["rounds_used"] >= 1
    assert response["total_tokens"] > 0
    assert response["benchmark"]


def test_chat_completions_payload_wraps_quorumx_result() -> None:
    response = chat_completions_payload(
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Summarize this RFC in one sentence."},
            ],
            "config": {"mock_mode": True, "consensus_mode": "quantum_ready"},
        }
    )

    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"]
    assert response["quorumx"]["consensus_mode"] == "quantum_ready"


def test_mcp_server_exposes_quorumx_run_tool() -> None:
    server = QuorumXMCPServer()

    tools = server.list_tools()["tools"]

    assert len(tools) == 1
    assert tools[0]["name"] == MCP_TOOL_NAME
    assert "task" in tools[0]["inputSchema"]["required"]

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": MCP_TOOL_NAME,
                "arguments": {
                    "task": "Draft a launch email for a product update.",
                    "config": {
                        "n_agents": 2,
                        "max_rounds": 1,
                        "mock_mode": True,
                    },
                },
            },
        }
    )

    assert response is not None
    structured = response["result"]["structuredContent"]
    assert structured["answer"]
    assert structured["consensus_mode"] == "quantum_ready"


def test_mcp_server_emits_telemetry() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    server = QuorumXMCPServer(telemetry=lambda event, payload: events.append((event, payload)))

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": MCP_TOOL_NAME,
                "arguments": {
                    "task": "Draft a launch email for a product update.",
                    "config": {"mock_mode": True, "max_rounds": 1},
                },
            },
        }
    )

    assert response is not None
    assert events[0][0] == "quorumx.resolve"
    assert events[1][0] == "quorumx.mcp.tool_call"