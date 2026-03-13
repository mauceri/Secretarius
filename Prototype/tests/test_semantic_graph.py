import unittest

from secretarius_local import semantic_graph


class _FakeMilvusClient:
    def __init__(self):
        self.deleted_ids = None
        self.upsert_rows = None
        self.query_calls = []

    def query(self, *, collection_name, filter, output_fields):
        self.query_calls.append(
            {
                "collection_name": collection_name,
                "filter": filter,
                "output_fields": output_fields,
            }
        )
        return [
            {"id": 101, "expression_id": "expr:obsolete"},
            {"id": 202, "expression_id": "expr:keep"},
        ]

    def delete(self, *, collection_name, ids):
        self.deleted_ids = list(ids)
        return {"delete_count": len(ids)}

    def upsert(self, *, collection_name, data):
        self.upsert_rows = list(data)
        return {"upsert_count": len(data)}


class TestSemanticGraph(unittest.TestCase):
    def test_upsert_documents_replaces_stale_rows_for_same_doc(self):
        client = _FakeMilvusClient()
        payload = {
            "doc_id": "doc:1",
            "indexing": {"source_expression": "Keep"},
            "user_fields": {"type_note": "lecture"},
        }
        rows = [
            semantic_graph._build_row(payload=payload, embedding=[0.1, 0.2], source_idx=0),
            semantic_graph._build_row(payload=payload, embedding=[0.3, 0.4], source_idx=0),
        ]
        expected_expression_id = rows[0]["expression_id"]

        original_query = client.query

        def _query(*, collection_name, filter, output_fields):
            _ = (collection_name, filter, output_fields)
            return [
                {"id": 101, "expression_id": "expr:obsolete"},
                {"id": 202, "expression_id": expected_expression_id},
            ]

        client.query = _query
        result = semantic_graph._upsert_documents(
            client=client,
            collection_name="secretarius_semantic_graph",
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            documents=[payload, payload],
        )
        client.query = original_query

        self.assertEqual(result, {"upserted_count": 1, "deleted_count": 1})
        self.assertEqual(client.deleted_ids, [101])
        self.assertEqual(len(client.upsert_rows), 1)
        self.assertEqual(client.upsert_rows[0]["doc_id"], "doc:1")
        self.assertEqual(client.upsert_rows[0]["expression_id"], expected_expression_id)
        self.assertEqual(client.upsert_rows[0]["type_note"], "lecture")

    def test_build_row_uses_stable_ids_and_expression_norm(self):
        payload = {
            "doc_id": "doc:abc",
            "indexing": {"source_expression": "  Alpha   Beta  "},
        }

        row = semantic_graph._build_row(payload=payload, embedding=[0.1], source_idx=0)

        self.assertEqual(row["doc_id"], "doc:abc")
        self.assertEqual(row["expression_norm"], "alpha beta")
        self.assertIsInstance(row["id"], int)
        self.assertTrue(row["expression_id"].startswith("expr:"))


if __name__ == "__main__":
    unittest.main()
