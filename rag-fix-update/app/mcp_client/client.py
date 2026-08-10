"""Async MCP client. Connects to the MCP server defined in
mcp_server/server.py, discovers its tools, and calls them. This is the ONLY
code path the agent uses to reach the HR tools -- the orchestrator never
imports mcp_server.tools directly.

Two transports are supported, selected via MCP_TRANSPORT:

- "memory" (default): the MCP server runs IN-PROCESS via the MCP SDK's
  in-memory transport (real JSON-RPC ClientSession <-> Server messages over
  anyio memory streams, not a hard-coded function call). This avoids
  spawning a second OS process that would re-import the whole
  chromadb/onnxruntime stack a second time -- on Render's free 512MB
  instance, running two full Python processes each holding a copy of that
  stack is what pushes memory usage over the limit. This is still a
  legitimate MCP transport per the project's "or another MCP-compatible
  approach" allowance.
- "stdio": the original approach -- spawns `python mcp_server/server.py` as
  a genuine separate OS subprocess and talks to it over stdio. Useful for
  local debugging or if you want to demonstrate MCP running as a literal
  separate process; set MCP_TRANSPORT=stdio to use it.
"""
import json
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from app import config


class MCPToolError(Exception):
    """Raised when an MCP tool call fails or the server is unreachable."""


class MCPClient:
    def __init__(self):
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._tools_cache = None

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        self._stack = AsyncExitStack()
        try:
            if config.MCP_TRANSPORT == "stdio":
                server_params = StdioServerParameters(
                    command=config.MCP_SERVER_CMD,
                    args=[config.MCP_SERVER_SCRIPT],
                )
                read, write = await self._stack.enter_async_context(
                    stdio_client(server_params)
                )
                self._session = await self._stack.enter_async_context(
                    ClientSession(read, write)
                )
                await self._session.initialize()
            else:
                # Local import: only pull in mcp_server.server (and its
                # heavier deps like the RAG stack) once we actually need to
                # connect, not at module import time.
                from mcp_server.server import mcp as fastmcp_instance

                # create_connected_server_and_client_session already calls
                # ClientSession.initialize() internally.
                self._session = await self._stack.enter_async_context(
                    create_connected_server_and_client_session(
                        fastmcp_instance._mcp_server
                    )
                )
            await self.list_tools(refresh=True)
        except Exception as exc:  # noqa: BLE001
            await self.close()
            raise MCPToolError(f"Failed to start/connect MCP server: {exc}") from exc

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
        self._tools_cache = None

    async def list_tools(self, refresh: bool = False):
        if self._tools_cache is not None and not refresh:
            return self._tools_cache
        if self._session is None:
            raise MCPToolError("MCP client is not connected.")
        result = await self._session.list_tools()
        self._tools_cache = result.tools
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict) -> dict:
        if self._session is None:
            raise MCPToolError("MCP client is not connected.")
        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001
            raise MCPToolError(f"Tool call '{name}' failed: {exc}") from exc

        for block in getattr(result, "content", []):
            if getattr(block, "type", None) == "text":
                try:
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    return {"raw_text": block.text}
        return {"error": "Tool returned no content."}

    def openai_tool_schemas(self) -> list[dict]:
        """Convert cached MCP tool definitions to OpenAI/Groq function-calling
        schema so the LLM can be given the exact tools discovered from MCP."""
        if self._tools_cache is None:
            raise MCPToolError("Call list_tools() before openai_tool_schemas().")
        schemas = []
        for tool in self._tools_cache:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema
                        or {"type": "object", "properties": {}},
                    },
                }
            )
        return schemas
