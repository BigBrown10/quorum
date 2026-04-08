from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from .http import resolve_quorumx_payload
from .telemetry import TelemetryHook, emit_telemetry

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP as FastMCPType
    from mcp.types import CallToolResult as CallToolResultType
    from mcp.types import TextContent as TextContentType
else:
    FastMCPType = Any
    CallToolResultType = Any
    TextContentType = Any

try:  # pragma: no cover - exercised in CI when mcp is unavailable
    from mcp.server.fastmcp import FastMCP as _ImportedFastMCP
    from mcp.types import (
        CallToolResult as _ImportedCallToolResult,
    )
    from mcp.types import (
        TextContent as _ImportedTextContent,
    )
    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for lightweight environments
    MCP_AVAILABLE = False

    @dataclass(slots=True)
    class _ToolSpec:
        name: str
        description: str
        inputSchema: dict[str, Any]

    @dataclass(slots=True)
    class _FallbackTextContent:
        type: str
        text: str
        annotations: Any | None = None
        meta: dict[str, Any] | None = None

    @dataclass(slots=True)
    class _FallbackCallToolResult:
        _meta: dict[str, Any] | None = None
        content: list[Any] = field(default_factory=list)
        structuredContent: dict[str, Any] | None = None
        isError: bool = False

        @property
        def structured_content(self) -> dict[str, Any] | None:
            return self.structuredContent

    class _FallbackFastMCP:
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

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FallbackCallToolResult:
            if name not in self._tool_handlers:
                raise ValueError(f"Unknown tool: {name}")
            result = self._tool_handlers[name](**arguments)
            return cast(_FallbackCallToolResult, _coerce_call_tool_result(result))

        def run(self, transport: str = "stdio") -> None:
            raise ImportError(
                "The optional 'mcp' dependency is required to run the QuorumX MCP server"
            )


TOOL_NAME = "quorumx.run"
MCP_TOOL_NAME = TOOL_NAME


if MCP_AVAILABLE:
    _MCP_FASTMCP: Any = _ImportedFastMCP
    _MCP_CALL_TOOL_RESULT: Any = _ImportedCallToolResult
    _MCP_TEXT_CONTENT: Any = _ImportedTextContent
else:
    _MCP_FASTMCP = _FallbackFastMCP
    _MCP_CALL_TOOL_RESULT = _FallbackCallToolResult
    _MCP_TEXT_CONTENT = _FallbackTextContent


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


def _make_text_content(text: str) -> TextContentType:
    if MCP_AVAILABLE:
        return cast(Any, _ImportedTextContent(type="text", text=text))

    return cast(Any, _FallbackTextContent(type="text", text=text))


def _make_call_tool_result(
    *,
    content: list[Any],
    structured_content: dict[str, Any],
    is_error: bool = False,
) -> CallToolResultType:
    if MCP_AVAILABLE:
        return cast(
            Any,
            _ImportedCallToolResult(
                content=content,
                structuredContent=structured_content,
                isError=is_error,
            ),
        )

    return cast(
        Any,
        _FallbackCallToolResult(
            content=content,
            structuredContent=structured_content,
            isError=is_error,
        ),
    )


def create_server(*, telemetry: TelemetryHook | None = None) -> FastMCPType:
    server = _MCP_FASTMCP("QuorumX", json_response=True)

    @server.tool(name=TOOL_NAME)
    def quorumx_run(
        task: str,
        system_instructions: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        roles: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> CallToolResultType:
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
        return _make_call_tool_result(
            content=[_make_text_content(json.dumps(result, ensure_ascii=False))],
            structured_content=result,
        )

    return cast(Any, server)


mcp = create_server()


@dataclass(slots=True)
class QuorumXMCPServer:
    telemetry: TelemetryHook | None = None
    _server: FastMCPType = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._server = cast(Any, create_server(telemetry=self.telemetry))

    async def list_tools(self) -> list[Any]:
        return await self._server.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResultType:
        result = await self._server.call_tool(name, arguments)
        return _coerce_call_tool_result(result)

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
    ) -> None:
        self._server.run(transport=transport)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()


def _coerce_call_tool_result(value: Any) -> CallToolResultType:
    if MCP_AVAILABLE and isinstance(value, _ImportedCallToolResult):
        return cast(CallToolResultType, value)

    if not MCP_AVAILABLE and isinstance(value, _FallbackCallToolResult):
        return cast(CallToolResultType, value)

    if isinstance(value, dict):
        return _make_call_tool_result(
            content=_coerce_content_items(value.get("content")),
            structured_content=_structured_content_from_value(
                value.get("structuredContent", value.get("structured_content"))
            ) or {},
            is_error=bool(value.get("isError", value.get("is_error", False))),
        )

    content = getattr(value, "content", None)
    structured_content = getattr(value, "structuredContent", None)
    if structured_content is None:
        structured_content = getattr(value, "structured_content", None)
    is_error = bool(getattr(value, "isError", getattr(value, "is_error", False)))

    return _make_call_tool_result(
        content=_coerce_content_items(content),
        structured_content=_structured_content_from_value(structured_content) or {},
        is_error=is_error,
    )


def _coerce_content_items(content: Any) -> list[Any]:
    if content is None:
        return []
    if isinstance(content, list):
        return content
    return [content]


def _structured_content_from_value(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except TypeError:
            pass
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
            if isinstance(dumped, dict):
                return dumped
        except TypeError:
            pass
    return None
