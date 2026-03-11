import sys
import unittest
from pathlib import Path

from adapters.output.mcp_client import StdioMCPClient


class TestStdioMCPClient(unittest.IsolatedAsyncioTestCase):
    async def test_connect_list_and_call_tool(self):
        repo_root = Path(__file__).resolve().parents[1]
        client = StdioMCPClient(
            command=sys.executable,
            args=["tools/secretarius_server.py"],
            cwd=str(repo_root),
        )

        try:
            await client.connect()
            tools = await client.list_tools()
            self.assertTrue(any(t.get("name") == "ask_oracle" for t in tools))

            output = await client.call_tool("ask_oracle", {"question": "Le test passe-t-il ?"})
            self.assertIn("answered:", output)
        finally:
            await client.disconnect()
