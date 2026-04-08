from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from .http import resolve_quorumx_payload
from .telemetry import TelemetryHook, emit_telemetry

TOOL_NAME = "quorumx.run"
MCP_TOOL_NAME = TOOL_NAME


def create_server(*, telemetry: TelemetryHook | None = None) -> FastMCP:
    server = FastMCP("QuorumX", json_response=True)

    @server.tool(name=TOOL_NAME)
    def quorumx_run(
        task: str,
        system_instructions: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        roles: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> CallToolResult:
        payload = {
            "task": task,
            "system_instructions": system_instructions,
            "messages": messages,
            "roles": roles,
            "config": config or {},
        }
        result = resolve_quorumx_payload(payload, telemetry=telemetry)
        emit_telemetry(
            telemetry,
            "quorumx.mcp.tool_call",
            {
                "tool": TOOL_NAME,
                "agreement_score": result["agreement_score"],
                "unstable": result["unstable"],
                "rounds_used": result["rounds_used"],
                "total_tokens": result["total_tokens"],
            },
        )
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
            structured_content=result,
        )

    return server


mcp = create_server()


@dataclass(slots=True)
class QuorumXMCPServer:
    telemetry: TelemetryHook | None = None
    _server: FastMCP = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._server = create_server(telemetry=self.telemetry)

    async def list_tools(self) -> list[Any]:
        return await self._server.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        return await self._server.call_tool(name, arguments)

    def run(self, transport: str = "stdio") -> None:
        self._server.run(transport=transport)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
