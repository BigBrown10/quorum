# MCP Deployment Notes

## Claude Desktop

Add Quorum as a local MCP server entry in the Claude Desktop config:

```json
{
  "mcpServers": {
    "quorum": {
      "command": "c:/Users/edogu/OneDrive/Documents/dequorum/.venv/Scripts/python.exe",
      "args": ["-m", "quorum_mcp.server"]
    }
  }
}
```

## Cursor

Use the same command and args in Cursor's MCP server configuration:

```json
{
  "name": "quorum",
  "command": "c:/Users/edogu/OneDrive/Documents/dequorum/.venv/Scripts/python.exe",
  "args": ["-m", "quorum_mcp.server"]
}
```

## Notes

- The MCP server is stateless.
- The tool name is `quorum_consensus`.
- The server returns the same JSON consensus shape as the HTTP API.
- The Python virtual environment must have the project dependencies installed.
