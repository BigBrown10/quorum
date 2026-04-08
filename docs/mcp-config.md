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

## QuorumX

Use the QuorumX MCP server when you want the reasoning trust layer instead of the lower-level consensus tool:

```json
{
  "mcpServers": {
    "quorumx": {
      "command": "c:/Users/edogu/OneDrive/Documents/dequorum/.venv/Scripts/python.exe",
      "args": ["-m", "quorumx.mcp"]
    }
  }
}
```

For Cursor, use the same command and args with a QuorumX server name:

```json
{
  "name": "quorumx",
  "command": "c:/Users/edogu/OneDrive/Documents/dequorum/.venv/Scripts/python.exe",
  "args": ["-m", "quorumx.mcp"]
}
```

## Notes

- The MCP server is stateless.
- The tool name is `quorum_consensus`.
- The server returns the same JSON consensus shape as the HTTP API.
- QuorumX exposes the `quorumx.run` tool and returns the QuorumX result payload.
- The Python virtual environment must have the project dependencies installed.
