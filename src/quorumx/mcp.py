from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

try:  # pragma: no cover - exercised in CI when mcp is unavailable
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult, TextContent
    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for lightweight environments
    MCP_AVAILABLE = False

    @dataclass(slots=True)
    class TextContent:
        type: str
        text: str


    @dataclass(slots=True)
    class CallToolResult:
        content: list[Any]
        structured_content: dict[str, Any] | None = None
        isError: bool = False


    @dataclass(slots=True)
    class _ToolSpec:
        name: str
        description: str
        inputSchema: dict[str, Any]


    class FastMCP:  # type: ignore[no-redef]
        def __init__(self, name: str, json_response: bool = True) -> None:
            self.name = name
            self.json_response = json_response
            self._tool_handlers: dict[str, Callable[..., Any]] = {}
            self._tool_specs: dict[str, _ToolSpec] = {}

        def tool(self, name: str | None = None):
            def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
                tool_name = name or function.__name__
                self._tool_handlers[tool_name] = function
                self._tool_specs[tool_name] = _ToolSpec(
                    name=tool_name,
                    description=(function.__doc__ or "").strip(),
                    inputSchema=_tool_input_schema(),
                )
                return function

            return decorator

        async def list_tools(self) -> list[Any]:
            return list(self._tool_specs.values())

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
            if name not in self._tool_handlers:
                raise ValueError(f"Unknown tool: {name}")
            return self._tool_handlers[name](**arguments)

        def run(self, transport: str = "stdio") -> None:
            raise ImportError(
                "The optional 'mcp' dependency is required to run the QuorumX MCP server"
            )

from .http import resolve_quorumx_payload
from .telemetry import TelemetryHook, emit_telemetry

TOOL_NAME = "quorumx.run"
MCP_TOOL_NAME = TOOL_NAME


def _tool_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "system_instructions": {"type": ["string", "null"]},
            "messages": {"type": ["array", "null"]},
            "roles": {"type": ["array", "null"]},
            "config": {"type": ["object", "null"]},
        },
        "required": ["task"],
    }


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
