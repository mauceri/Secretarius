import unittest

from secretarius_local.mcp_server import handle_mcp_message


class TestMCPToolsCatalog(unittest.TestCase):
    def test_public_tools_include_index_and_search_and_hide_embeddings(self):
        response = handle_mcp_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            }
        )
        self.assertIsInstance(response, dict)
        result = response.get("result", {})
        tools = result.get("tools", [])
        tool_names = {t.get("name") for t in tools if isinstance(t, dict)}

        self.assertIn("extract_expressions", tool_names)
        self.assertIn("semantic_graph_search", tool_names)
        self.assertIn("index_text", tool_names)
        self.assertIn("search_text", tool_names)
        self.assertNotIn("expressions_to_embeddings", tool_names)


if __name__ == "__main__":
    unittest.main()
