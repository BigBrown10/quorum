from quorum_mcp.server import TOOL_NAME, QuorumMCPServer


def test_list_tools_exposes_quorum_consensus() -> None:
    server = QuorumMCPServer()

    tools = server.list_tools()["tools"]

    assert len(tools) == 1
    assert tools[0]["name"] == TOOL_NAME
    assert "candidates" in tools[0]["inputSchema"]["required"]


def test_tools_call_returns_structured_consensus_payload() -> None:
    server = QuorumMCPServer()

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": TOOL_NAME,
                "arguments": {
                    "candidates": [
                        {"id": "a1", "content": "win", "confidence": 0.2},
                        {"id": "a2", "content": "win", "confidence": 0.8},
                        {"id": "a3", "content": "lose", "confidence": 0.9},
                    ],
                    "mode": "weighted_majority",
                },
            },
        }
    )

    assert response is not None
    result = response["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["consensus_answer"] == "win"
    assert result["structuredContent"]["unstable"] is False
