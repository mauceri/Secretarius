import unittest
from unittest.mock import patch

from secretarius_local import document_pipeline
from secretarius_local import document_schema


class TestDocumentPipeline(unittest.TestCase):
    def test_analyse_texte_documentaire_reads_doc_id_from_text(self):
        document = document_pipeline.analyse_texte_documentaire(
            "doc_id: doc:test-1\nTitre\nCorps documentaire"
        )

        self.assertEqual(document["doc_id"], "doc:test-1")
        self.assertEqual(document["content"]["text"], "Corps documentaire")

    def test_analyse_texte_documentaire_sets_type_note_from_text(self):
        document = document_pipeline.analyse_texte_documentaire(
            "Titre\ntype_note: lecture\nCorps documentaire"
        )

        self.assertEqual(document["user_fields"]["type_note"], "lecture")
        self.assertEqual(document["content"]["text"], "Corps documentaire")

    def test_analyse_texte_documentaire_defaults_type_note_to_fugace(self):
        document = document_pipeline.analyse_texte_documentaire("Titre\nCorps documentaire")

        self.assertEqual(document["user_fields"]["type_note"], "fugace")

    def test_analyse_texte_documentaire_collects_multiple_urls(self):
        document = document_pipeline.analyse_texte_documentaire(
            "Titre\nhttps://a.example/doc\nhttps://b.example/ref\nCorps documentaire"
        )

        self.assertEqual(document["source"]["url"], "https://a.example/doc")
        self.assertEqual(
            document["source"]["urls"],
            ["https://a.example/doc", "https://b.example/ref"],
        )
        self.assertEqual(document["content"]["text"], "Corps documentaire")

    def test_parse_keyword_query_separates_text_and_operators(self):
        parsed = document_pipeline._parse_keyword_query(
            "jung +#psychologie #symbolisme -#brouillon"
        )

        self.assertEqual(parsed["text"], "jung")
        self.assertEqual(parsed["required_keywords"], ["#psychologie"])
        self.assertEqual(parsed["optional_keywords"], ["#symbolisme"])
        self.assertEqual(parsed["excluded_keywords"], ["#brouillon"])

    def test_parse_keyword_query_keeps_plain_text_without_hashtags(self):
        parsed = document_pipeline._parse_keyword_query("memoire autobiographique")

        self.assertEqual(parsed["text"], "memoire autobiographique")
        self.assertEqual(parsed["required_keywords"], [])
        self.assertEqual(parsed["optional_keywords"], [])
        self.assertEqual(parsed["excluded_keywords"], [])

    def test_normalize_document_does_not_inject_unused_defaults(self):
        document = document_schema.normalize_document({"text": "Corps documentaire"})

        self.assertNotIn("authors", document["source"])
        self.assertNotIn("tags", document["user_fields"])
        self.assertNotIn("status", document["user_fields"])
        self.assertNotIn("mode", document["content"])
        self.assertNotIn("length_chars", document["content"])
        self.assertIn("hash", document["content"])
        self.assertIn("source_id", document["source"])
        self.assertIn("updated_at", document["user_fields"])
        self.assertIn("created_at", document["user_fields"])

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

    @patch("secretarius_local.document_pipeline.semantic_graph_search_milvus")
    @patch("secretarius_local.document_pipeline.embed_expressions_multilingual")
    @patch("secretarius_local.document_pipeline.extract_expressions")
    @patch("secretarius_local.document_pipeline.semantic_graph_delete_doc")
    def test_update_document_reuses_doc_id_and_deletes_previous_rows(
        self,
        mock_delete,
        mock_extract,
        mock_embed,
        mock_search,
    ):
        mock_delete.return_value = 3
        mock_extract.return_value = {
            "chunks": ["Corps documentaire"],
            "by_chunk": [{"id": 0, "expressions": ["alpha"]}],
            "expressions": ["alpha"],
            "warning": None,
        }
        mock_embed.return_value = {
            "embeddings": [[0.1, 0.2]],
            "dimension": 2,
            "model": "test-model",
            "warning": None,
        }
        mock_search.return_value = {
            "collection_name": "secretarius_semantic_graph",
            "inserted_count": 1,
            "deleted_count": 0,
            "query_count": 1,
            "hits": [[]],
            "warning": None,
        }

        result = document_pipeline.update_document_text("doc_id: doc:test-1\nTitre\nCorps documentaire")

        mock_delete.assert_called_once()
        self.assertEqual(mock_delete.call_args.kwargs["doc_id"], "doc:test-1")
        self.assertEqual(result["document"]["doc_id"], "doc:test-1")
        self.assertEqual(result["index"]["deleted_count"], 3)

    @patch("secretarius_local.document_pipeline.semantic_graph_search_milvus")
    @patch("secretarius_local.document_pipeline.embed_expressions_multilingual")
    @patch("secretarius_local.document_pipeline.extract_expressions")
    def test_search_document_text_passes_keyword_operator_groups(
        self,
        mock_extract,
        mock_embed,
        mock_search,
    ):
        mock_extract.return_value = {
            "expressions": ["memoire autobiographique"],
            "warning": None,
        }
        mock_embed.return_value = {
            "embeddings": [[0.1, 0.2]],
            "warning": None,
        }
        mock_search.return_value = {
            "collection_name": "secretarius_semantic_graph",
            "query_count": 1,
            "hits": [[]],
            "warning": None,
        }

        document_pipeline.search_documents_by_text(
            "memoire autobiographique +#psychologie #trauma -#brouillon"
        )

        self.assertEqual(mock_extract.call_args.args[0], "memoire autobiographique")
        self.assertEqual(mock_search.call_args.kwargs["required_keywords"], ["#psychologie"])
        self.assertEqual(mock_search.call_args.kwargs["optional_keywords"], ["#trauma"])
        self.assertEqual(mock_search.call_args.kwargs["excluded_keywords"], ["#brouillon"])

    @patch("secretarius_local.document_pipeline.semantic_graph_search_milvus")
    @patch("secretarius_local.document_pipeline.embed_expressions_multilingual")
    @patch("secretarius_local.document_pipeline.extract_expressions")
    def test_index_document_surfaces_embedding_failure_without_calling_semantic_graph(
        self,
        mock_extract,
        mock_embed,
        mock_search,
    ):
        mock_extract.return_value = {
            "chunks": ["Corps documentaire"],
            "by_chunk": [{"id": 0, "expressions": ["alpha"]}],
            "expressions": ["alpha"],
            "warning": None,
        }
        mock_embed.return_value = {
            "embeddings": [],
            "dimension": 0,
            "model": "BAAI/bge-m3",
            "warning": "unable to initialize sentence-transformers model: test error",
        }

        result = document_pipeline.index_document_text("Titre\nCorps documentaire")

        mock_search.assert_not_called()
        self.assertEqual(
            result["index"]["warning"],
            "unable to initialize sentence-transformers model: test error",
        )

    @patch("secretarius_local.document_pipeline.semantic_graph_search_milvus")
    @patch("secretarius_local.document_pipeline.embed_expressions_multilingual")
    @patch("secretarius_local.document_pipeline.extract_expressions")
    def test_search_includes_late_interaction_scores(
        self,
        mock_extract,
        mock_embed,
        mock_search,
    ):
        mock_extract.return_value = {
            "expressions": ["memoire", "autobiographique"],
            "warning": None,
        }
        mock_embed.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "warning": None,
        }
        # q0 → doc:A (score 0.9), doc:B (score 0.7)
        # q1 → doc:A (score 0.8)
        mock_search.return_value = {
            "collection_name": "secretarius_semantic_graph",
            "query_count": 2,
            "hits": [
                [
                    {"score": 0.9, "entity": {"payload_json": '{"doc_id": "doc:A"}'}},
                    {"score": 0.7, "entity": {"payload_json": '{"doc_id": "doc:B"}'}},
                ],
                [
                    {"score": 0.8, "entity": {"payload_json": '{"doc_id": "doc:A"}'}},
                ],
            ],
            "warning": None,
        }

        result = document_pipeline.search_documents_by_text("memoire autobiographique")

        li_scores = result.get("late_interaction_scores", {})
        self.assertIn("doc:A", li_scores)
        self.assertIn("doc:B", li_scores)
        self.assertAlmostEqual(li_scores["doc:A"], 1.7)
        self.assertAlmostEqual(li_scores["doc:B"], 0.7)
        self.assertGreater(li_scores["doc:A"], li_scores["doc:B"])

    @patch("secretarius_local.document_pipeline.semantic_graph_search_milvus")
    @patch("secretarius_local.document_pipeline.embed_expressions_multilingual")
    @patch("secretarius_local.document_pipeline.extract_expressions")
    def test_search_document_surfaces_embedding_failure_without_calling_semantic_graph(
        self,
        mock_extract,
        mock_embed,
        mock_search,
    ):
        mock_extract.return_value = {
            "expressions": ["memoire autobiographique"],
            "warning": None,
        }
        mock_embed.return_value = {
            "embeddings": [],
            "dimension": 0,
            "model": "BAAI/bge-m3",
            "warning": "embedding encode failed: test error",
        }

        result = document_pipeline.search_documents_by_text("memoire autobiographique")

        mock_search.assert_not_called()
        self.assertEqual(result["search"]["warning"], "embedding encode failed: test error")


if __name__ == "__main__":
    unittest.main()
