import unittest
from unittest.mock import patch

from secretarius_local import document_pipeline


class TestDocumentPipeline(unittest.TestCase):
    @patch("secretarius_local.document_pipeline.semantic_graph_search_milvus")
    @patch("secretarius_local.document_pipeline.embed_expressions_multilingual")
    @patch("secretarius_local.document_pipeline.extract_expressions")
    def test_index_document_uses_all_expression_embeddings_for_insert(
        self,
        mock_extract,
        mock_embed,
        mock_search,
    ):
        mock_extract.return_value = {
            "chunks": ["Corps documentaire"],
            "by_chunk": [{"id": 0, "expressions": ["alpha", "beta"]}],
            "expressions": ["alpha", "beta"],
            "warning": None,
        }
        mock_embed.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "dimension": 2,
            "model": "test-model",
            "warning": None,
        }
        mock_search.return_value = {
            "collection_name": "secretarius_semantic_graph",
            "inserted_count": 2,
            "query_count": 2,
            "hits": [[], []],
            "warning": None,
        }

        result = document_pipeline.index_document_text("Titre\nCorps documentaire")

        self.assertEqual(result["index"]["inserted_count"], 2)
        self.assertEqual(len(mock_search.call_args.args[0]), 2)
        search_kwargs = mock_search.call_args.kwargs
        self.assertEqual(len(search_kwargs["documents"]), 2)
        self.assertEqual(
            search_kwargs["documents"][0]["indexing"]["source_expression"],
            "alpha",
        )
        self.assertEqual(
            search_kwargs["documents"][1]["indexing"]["source_expression"],
            "beta",
        )


if __name__ == "__main__":
    unittest.main()
