import json
import unittest
from unittest.mock import patch

from secretarius_local import mcp_server


def _extract_payload(tool_result: dict) -> dict:
    return tool_result.get("structuredContent", {})


class TestMCPServerCompactResponses(unittest.TestCase):
    @patch("secretarius_local.mcp_server.extract_expressions")
    def test_extract_expressions_is_minimal_by_default(self, mock_extract):
        mock_extract.return_value = {
            "chunks": ["bonjour"],
            "by_chunk": [{"id": 0, "expressions": ["a", "b"]}],
            "expressions": ["a", "b"],
            "request_fingerprint": "fp",
            "inference_params": {"seed": 42},
            "warning": None,
        }

        result = mcp_server._handle_extract_expressions({"text": "bonjour"})
        payload = _extract_payload(result)

        self.assertEqual(payload, {"expressions": ["a", "b"]})

    @patch("secretarius_local.mcp_server.extract_expressions")
    def test_extract_expressions_moves_diagnostics_to_debug(self, mock_extract):
        mock_extract.return_value = {
            "chunks": ["bonjour"],
            "by_chunk": [{"id": 0, "expressions": ["a"]}],
            "expressions": ["a"],
            "request_fingerprint": "fp",
            "inference_params": {"seed": 42},
            "raw_llm_outputs": [{"content": "raw"}],
            "warning": "filtered",
        }

        result = mcp_server._handle_extract_expressions({"text": "bonjour", "debug_return_raw": True})
        payload = _extract_payload(result)

        self.assertEqual(payload.get("expressions"), ["a"])
        self.assertEqual(payload.get("warning"), "filtered")
        self.assertEqual(payload.get("tool"), "extract_expressions")
        self.assertEqual(payload.get("backend"), "llama_cpp_direct")
        self.assertEqual(payload.get("chunk_count"), 1)
        self.assertIn("by_chunk", payload)
        self.assertIn("request_fingerprint", payload)
        self.assertIn("inference_params", payload)
        self.assertIn("raw_llm_outputs", payload)

    @patch("secretarius_local.mcp_server._handle_semantic_graph_search")
    @patch("secretarius_local.mcp_server._handle_extract_expressions")
    def test_index_text_is_compact_by_default(self, mock_extract, mock_search):
        mock_extract.return_value = {
            "structuredContent": {
                "expressions": ["a", "b"],
                "warning": None,
            }
        }
        mock_search.return_value = {
            "structuredContent": {
                "collection_name": "secretarius_semantic_graph",
                "inserted_count": 2,
                "query_count": 2,
                "hits": [[], []],
                "warning": None,
            }
        }

        result = mcp_server._handle_index_text({"text": "bonjour"})
        payload = _extract_payload(result)
        self.assertEqual(payload.get("tool"), "index_text")
        self.assertIn("summary", payload)
        self.assertNotIn("extract", payload)
        self.assertNotIn("index", payload)

    @patch("secretarius_local.mcp_server._handle_semantic_graph_search")
    @patch("secretarius_local.mcp_server._handle_extract_expressions")
    def test_index_text_debug_full_includes_details(self, mock_extract, mock_search):
        mock_extract.return_value = {"structuredContent": {"expressions": ["a"], "warning": None}}
        mock_search.return_value = {
            "structuredContent": {
                "collection_name": "secretarius_semantic_graph",
                "inserted_count": 1,
                "query_count": 1,
                "hits": [[]],
                "warning": None,
            }
        }

        result = mcp_server._handle_index_text({"text": "bonjour", "debug_full": True})
        payload = _extract_payload(result)
        self.assertIn("extract", payload)
        self.assertIn("index", payload)

    @patch("secretarius_local.mcp_server._handle_semantic_graph_search")
    @patch("secretarius_local.mcp_server._handle_extract_expressions")
    def test_index_text_enriches_document_before_indexing(self, mock_extract, mock_search):
        def fake_extract(arguments):
            return {
                "structuredContent": {
                    "expressions": ["corps documentaire"],
                    "warning": None,
                    "document": arguments.get("document"),
                }
            }

        mock_extract.side_effect = fake_extract
        mock_search.return_value = {
            "structuredContent": {
                "collection_name": "secretarius_semantic_graph",
                "inserted_count": 1,
                "query_count": 1,
                "hits": [[]],
                "warning": None,
            }
        }

        result = mcp_server._handle_index_text(
            {"text": "Titre\n#archive 11-03-2026 https://example.com\nCorps documentaire"}
        )
        payload = _extract_payload(result)

        extract_args = mock_extract.call_args.args[0]
        document = extract_args.get("document")
        self.assertEqual(extract_args.get("text"), "Corps documentaire")
        self.assertEqual(document.get("content", {}).get("text"), "Corps documentaire")
        self.assertEqual(document.get("source", {}).get("url"), "https://example.com")
        self.assertEqual(document.get("user_fields", {}).get("title"), "Titre")
        self.assertEqual(document.get("user_fields", {}).get("document_date"), "11-03-2026")
        self.assertEqual(document.get("user_fields", {}).get("keywords"), ["#archive"])

        graph_args = mock_search.call_args.args[0]
        self.assertEqual(graph_args.get("document"), document)
        self.assertEqual(payload.get("summary", {}).get("inserted_count"), 1)

    @patch("secretarius_local.mcp_server._handle_semantic_graph_search")
    def test_search_text_is_compact_by_default(self, mock_search):
        mock_search.return_value = {
            "structuredContent": {
                "collection_name": "secretarius_semantic_graph",
                "query_count": 1,
                "hits": [[{"id": "x", "score": 0.91, "entity": {"payload_json": "{\"doc_id\":\"d1\",\"content\":\"bonjour\"}"}}]],
                "warning": None,
            }
        }

        result = mcp_server._handle_search_text({"query": "charniers"})
        payload = _extract_payload(result)
        self.assertEqual(payload.get("tool"), "search_text")
        self.assertIn("summary", payload)
        self.assertEqual(
            payload.get("documents"),
            [{"id": "x", "score": 0.91, "document": {"doc_id": "d1", "content": "bonjour"}}],
        )
        self.assertNotIn("search", payload)


if __name__ == "__main__":
    unittest.main()
