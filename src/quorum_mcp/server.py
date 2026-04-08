from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from typing import Any

from quorum_core.api import resolve_consensus_payload

LOGGER = logging.getLogger(__name__)

TOOL_NAME = "quorum_consensus"
TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "content": {"type": "string"},
                    "confidence": {"type": "number"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "embedding": {"type": "array", "items": {"type": "number"}},
                    "stats": {"type": "object"},
                },
                "required": ["id", "content"],
            },
        },
        "mode": {
            "type": "string",
            "enum": ["simple_majority", "weighted_majority", "quantum_ready"],
            "default": "quantum_ready",
        },
    },
    "required": ["candidates"],
}


@dataclass(slots=True)
class QuorumMCPServer:
    """Minimal MCP-style request handler for quorum-consensus tooling."""

    def list_tools(self) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": TOOL_NAME,
                    "description": "Resolve consensus across multiple AI agent outputs.",
                    "inputSchema": TOOL_SCHEMA,
                }
            ]
        }

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name != TOOL_NAME:
            raise ValueError(f"Unknown tool: {name}")

        payload = resolve_consensus_payload(arguments)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False),
                }
            ],
            "isError": False,
            "structuredContent": payload,
        }

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "quorum-mcp", "version": "0.1.0"},
                    "capabilities": {"tools": {"listChanged": False}},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": self.list_tools(),
            }

        if method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = self.call_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }

        if request_id is None:
            return None

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }


def main() -> None:
    server = QuorumMCPServer()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        headers: dict[str, str] = {}
        while True:
            line = stdin.readline()
            if not line:
                return
            decoded = line.decode("utf-8").strip()
            if not decoded:
                break
            if ":" in decoded:
                key, value = decoded.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", "0"))
        if content_length <= 0:
            continue

        body = stdin.read(content_length)
        if not body:
            return

        try:
            request = json.loads(body.decode("utf-8"))
            response = server.handle_request(request)
        except Exception as exc:  # pragma: no cover - defensive server boundary
            LOGGER.exception("Unexpected MCP server error")
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(exc)},
            }
        if response is None:
            continue

        response_body = json.dumps(response, ensure_ascii=False).encode("utf-8")
        stdout.write(f"Content-Length: {len(response_body)}\r\n\r\n".encode("utf-8"))
        stdout.write(response_body)
        stdout.flush()


if __name__ == "__main__":
    main()
