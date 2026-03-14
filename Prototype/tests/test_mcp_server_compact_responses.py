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

    @patch("secretarius_local.mcp_server.index_document_text")
    def test_index_text_is_compact_by_default(self, mock_index):
        mock_index.return_value = {
            "extract": {
                "expressions": ["a", "b"],
                "warning": None,
            },
            "index": {
                "collection_name": "secretarius_semantic_graph",
                "inserted_count": 2,
                "query_count": 2,
                "hits": [[], []],
                "warning": None,
            },
        }

        result = mcp_server._handle_index_text({"text": "bonjour"})
        payload = _extract_payload(result)
        self.assertEqual(payload.get("tool"), "index_text")
        self.assertIn("summary", payload)
        self.assertNotIn("extract", payload)
        self.assertNotIn("index", payload)

    @patch("secretarius_local.mcp_server.index_document_text")
    def test_index_text_debug_full_includes_details(self, mock_index):
        mock_index.return_value = {
            "extract": {"expressions": ["a"], "warning": None},
            "index": {
                "collection_name": "secretarius_semantic_graph",
                "inserted_count": 1,
                "query_count": 1,
                "hits": [[]],
                "warning": None,
            },
        }

        result = mcp_server._handle_index_text({"text": "bonjour", "debug_full": True})
        payload = _extract_payload(result)
        self.assertIn("extract", payload)
        self.assertIn("index", payload)

    @patch("secretarius_local.mcp_server.index_document_text")
    def test_index_text_uses_document_pipeline(self, mock_index):
        mock_index.return_value = {
            "document": {"doc_id": "doc:test"},
            "extract": {"expressions": ["corps documentaire"], "warning": None},
            "index": {
                "collection_name": "secretarius_semantic_graph",
                "inserted_count": 1,
                "query_count": 1,
                "hits": [[]],
                "warning": None,
            },
        }

        result = mcp_server._handle_index_text(
            {"text": "Titre\n#archive 11-03-2026 https://example.com\nCorps documentaire"}
        )
        payload = _extract_payload(result)

        self.assertEqual(mock_index.call_args.args[0], "Titre\n#archive 11-03-2026 https://example.com\nCorps documentaire")
        self.assertEqual(payload.get("summary", {}).get("inserted_count"), 1)

    @patch("secretarius_local.mcp_server.search_documents_by_text")
    def test_search_text_is_compact_by_default(self, mock_search):
        mock_search.return_value = {
            "document": {"user_fields": {"keywords": ["#charniers"]}},
            "search": {
                "collection_name": "secretarius_semantic_graph",
                "query_count": 1,
                "min_score": 0.75,
                "hits": [[{"id": "x", "score": 0.91, "entity": {"payload_json": "{\"doc_id\":\"d1\",\"content\":\"bonjour\"}"}}]],
                "warning": None,
            },
        }

        result = mcp_server._handle_search_text({"query": "charniers"})
        payload = _extract_payload(result)
        self.assertEqual(payload.get("tool"), "search_text")
        self.assertIn("summary", payload)
        self.assertEqual(payload.get("summary", {}).get("min_score"), 0.75)
        self.assertEqual(payload.get("summary", {}).get("keyword_query_count"), 1)
        self.assertEqual(
            payload.get("documents"),
            [{"id": "x", "score": 0.91, "combined_score": 0.91, "keyword_matches": [], "document": {"doc_id": "d1", "content": "bonjour"}}],
        )
        self.assertNotIn("search", payload)

    @patch("secretarius_local.mcp_server.search_documents_by_text")
    def test_search_text_counts_all_query_hashtags(self, mock_search):
        mock_search.return_value = {
            "document": {"user_fields": {"keywords": ["#psychologie", "#symbolisme", "#brouillon"]}},
            "search": {
                "collection_name": "secretarius_semantic_graph",
                "query_count": 1,
                "min_score": None,
                "hits": [],
                "warning": None,
            },
        }

        result = mcp_server._handle_search_text({"query": "jung +#psychologie #symbolisme -#brouillon"})
        payload = _extract_payload(result)

        self.assertEqual(payload.get("summary", {}).get("keyword_query_count"), 3)

    def test_extract_search_documents_dedupes_by_doc_id(self):
        hits = [
            [
                {"id": "x1", "score": 0.91, "entity": {"payload_json": "{\"doc_id\":\"d1\",\"content\":\"bonjour\"}"}},
                {"id": "x2", "score": 0.89, "entity": {"payload_json": "{\"doc_id\":\"d1\",\"content\":\"bonjour\"}"}},
                {"id": "x3", "score": 0.88, "entity": {"payload_json": "{\"doc_id\":\"d2\",\"content\":\"salut\"}"}},
            ]
        ]

        documents = mcp_server._extract_search_documents(hits)

        self.assertEqual(
            documents,
            [
                {"id": "x1", "score": 0.91, "combined_score": 0.91, "keyword_matches": [], "document": {"doc_id": "d1", "content": "bonjour"}},
                {"id": "x3", "score": 0.88, "combined_score": 0.88, "keyword_matches": [], "document": {"doc_id": "d2", "content": "salut"}},
            ],
        )

    def test_extract_search_documents_reranks_with_keyword_bonus(self):
        hits = [
            [
                {
                    "id": "x1",
                    "score": 0.84,
                    "entity": {
                        "payload_json": "{\"doc_id\":\"d1\",\"user_fields\":{\"keywords\":[\"#stoicisme\"]},\"content\":{\"text\":\"a\"}}"
                    },
                },
                {
                    "id": "x2",
                    "score": 0.9,
                    "entity": {
                        "payload_json": "{\"doc_id\":\"d2\",\"user_fields\":{\"keywords\":[\"#histoire\"]},\"content\":{\"text\":\"b\"}}"
                    },
                },
            ]
        ]

        documents = mcp_server._extract_search_documents(hits, query_keywords=["#stoicisme"], query_terms=[])

        self.assertEqual(documents[0]["document"]["doc_id"], "d1")
        self.assertEqual(documents[0]["keyword_matches"], ["#stoicisme"])
        self.assertGreater(documents[0]["combined_score"], documents[1]["combined_score"])

    @patch("secretarius_local.mcp_server.update_document_text")
    def test_update_text_is_compact_by_default(self, mock_update):
        mock_update.return_value = {
            "document": {"doc_id": "doc:test-1"},
            "extract": {"expressions": ["a"], "warning": None},
            "index": {
                "collection_name": "secretarius_semantic_graph",
                "inserted_count": 1,
                "deleted_count": 2,
                "query_count": 1,
                "hits": [[]],
                "warning": None,
            },
        }

        result = mcp_server._handle_update_text({"text": "doc_id: doc:test-1\nbonjour"})
        payload = _extract_payload(result)
        self.assertEqual(payload.get("tool"), "update_text")
        self.assertEqual(payload.get("summary", {}).get("deleted_count"), 2)
        self.assertNotIn("extract", payload)
        self.assertNotIn("index", payload)


if __name__ == "__main__":
    unittest.main()
