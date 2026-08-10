"""Async MCP client: spawns mcp_server/server.py over stdio, discovers its
tools, and calls them. This is the ONLY code path the agent uses to reach the
HR tools -- the orchestrator never imports mcp_server.tools directly.
"""
import json
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
        server_params = StdioServerParameters(
            command=config.MCP_SERVER_CMD,
            args=[config.MCP_SERVER_SCRIPT],
        )
        self._stack = AsyncExitStack()
        try:
            read, write = await self._stack.enter_async_context(
                stdio_client(server_params)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
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
