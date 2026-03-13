import unittest
from unittest.mock import patch

from secretarius_local import document_pipeline


class TestDocumentPipeline(unittest.TestCase):
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
