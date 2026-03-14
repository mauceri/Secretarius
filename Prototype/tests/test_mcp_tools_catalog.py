import unittest

from secretarius_local.mcp_server import handle_mcp_message


class TestMCPToolsCatalog(unittest.TestCase):
    def test_public_tools_hide_low_level_semantic_tool(self):
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
        self.assertIn("index_text", tool_names)
        self.assertIn("search_text", tool_names)
        self.assertIn("update_text", tool_names)
        self.assertNotIn("semantic_graph_search", tool_names)
        self.assertNotIn("expressions_to_embeddings", tool_names)

    def test_public_tool_schemas_stay_minimal(self):
        response = handle_mcp_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            }
        )
        tools = response.get("result", {}).get("tools", [])
        by_name = {tool.get("name"): tool for tool in tools if isinstance(tool, dict)}

        extract_props = by_name["extract_expressions"]["inputSchema"]["properties"]
        index_props = by_name["index_text"]["inputSchema"]["properties"]
        search_props = by_name["search_text"]["inputSchema"]["properties"]
        update_props = by_name["update_text"]["inputSchema"]["properties"]

        self.assertEqual(set(extract_props.keys()), {"text"})
        self.assertEqual(set(index_props.keys()), {"text"})
        self.assertEqual(set(search_props.keys()), {"query"})
        self.assertEqual(set(update_props.keys()), {"text"})


if __name__ == "__main__":
    unittest.main()
