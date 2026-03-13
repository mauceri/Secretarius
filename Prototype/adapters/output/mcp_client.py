import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.ports import ToolClientInterface

logger = logging.getLogger(__name__)

class StdioMCPClient(ToolClientInterface):
    def __init__(
        self,
        command: str,
        args: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self.command = command
        self.args = args
        self.cwd = cwd
        self.env = env
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()

    async def connect(self):
        server_parameters = StdioServerParameters(
            command=self.command,
            args=self.args,
            env={**os.environ, **self.env} if self.env else None,
            cwd=Path(self.cwd) if self.cwd else None
        )
        try:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_parameters)
            )
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            logger.info(f"Connected to MCP Server with {self.command} {self.args}")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.command}: {e}")
            raise

    async def list_tools(self) -> List[Dict[str, Any]]:
        if not self._session:
            return []
            
        try:
            response = await self._session.list_tools()
            tools_schema = []
            for tool in response.tools:
                tools_schema.append({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                })
            return tools_schema
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if not self._session:
            raise RuntimeError("Not connected to any tool server.")
        
        try:
            result = await self._session.call_tool(name, arguments)
            # A typical CallToolResult contains content objects.
            content = []
            if result.content:
                for item in result.content:
                    # Item could be text, image, etc. Assuming text.
                    if item.type == "text":
                        content.append(item.text)
            return "\n".join(content) if content else "Tool executed smoothly with no text output."
        except Exception as e:
            logger.error(f"Failed to call tool {name}: {e}")
            return f"Error executing tool {name}: {e}"

    async def disconnect(self):
        await self._exit_stack.aclose()
        self._session = None
