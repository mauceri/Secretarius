from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from scripts import milvus_collection_io


class _FakeSchema:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fields: list[dict[str, object]] = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class _FakeIndexParams:
    def __init__(self):
        self.indexes: list[dict[str, object]] = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class _FakeClient:
    def __init__(self):
        self.created_schema: _FakeSchema | None = None
        self.created_indexes: _FakeIndexParams | None = None
        self.created_collection: dict[str, object] | None = None
        self.insert_batches: list[list[dict[str, object]]] = []
        self.dropped: list[str] = []
        self._existing = False

    def create_schema(self, **kwargs):
        self.created_schema = _FakeSchema(**kwargs)
        return self.created_schema

    def prepare_index_params(self):
        self.created_indexes = _FakeIndexParams()
        return self.created_indexes

    def has_collection(self, *, collection_name: str):
        return self._existing

    def drop_collection(self, *, collection_name: str):
        self.dropped.append(collection_name)

    def create_collection(self, **kwargs):
        self.created_collection = kwargs

    def insert(self, *, collection_name: str, data):
        self.insert_batches.append(list(data))
        return {"insert_count": len(data)}


class TestMilvusCollectionIO(unittest.TestCase):
    def test_build_schema_from_dump_uses_dynamic_fields_and_vector_dim(self):
        client = _FakeClient()
        dump = {
            "schema": {
                "auto_id": False,
                "enable_dynamic_field": True,
                "fields": [
                    {"name": "id", "type": "INT64", "is_primary": True, "description": "", "params": {}},
                    {"name": "vector", "type": "FLOAT_VECTOR", "description": "", "params": {"dim": 384}},
                ],
            }
        }

        schema = milvus_collection_io._build_schema_from_dump(client, dump)

        self.assertIs(schema, client.created_schema)
        self.assertEqual(len(schema.fields), 2)
        self.assertEqual(schema.fields[0]["field_name"], "id")
        self.assertEqual(schema.fields[1]["field_name"], "vector")
        self.assertEqual(schema.fields[1]["dim"], 384)

    def test_import_collection_recreates_collection_and_inserts_rows(self):
        client = _FakeClient()
        dump = {
            "format": "secretarius_milvus_collection_dump_v1",
            "collection_name": "source_collection",
            "schema": {
                "auto_id": False,
                "enable_dynamic_field": True,
                "fields": [
                    {"name": "id", "type": "INT64", "is_primary": True, "description": "", "params": {}},
                    {"name": "vector", "type": "FLOAT_VECTOR", "description": "", "params": {"dim": 2}},
                ],
            },
            "indexes": [{"field_name": "vector", "index_name": "vector", "index_type": "AUTOINDEX", "metric_type": "IP"}],
            "rows": [
                {"id": 1, "vector": [0.1, 0.2], "payload_json": "{}"},
                {"id": 2, "vector": [0.3, 0.4], "payload_json": "{}"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dump.json"
            path.write_text(json.dumps(dump), encoding="utf-8")
            original_client = milvus_collection_io.MilvusClient
            milvus_collection_io.MilvusClient = lambda uri, token=None: client
            try:
                imported = milvus_collection_io.import_collection(
                    uri="http://127.0.0.1:19530",
                    token=None,
                    input_path=path,
                    collection_name="restored_collection",
                    batch_size=1,
                    drop_if_exists=False,
                )
            finally:
                milvus_collection_io.MilvusClient = original_client

        self.assertEqual(imported, "restored_collection")
        self.assertIsNotNone(client.created_collection)
        self.assertEqual(client.created_collection["collection_name"], "restored_collection")
        self.assertEqual(len(client.insert_batches), 2)

    def test_drop_collection_deletes_existing_collection(self):
        client = _FakeClient()
        client._existing = True
        original_client = milvus_collection_io.MilvusClient
        milvus_collection_io.MilvusClient = lambda uri, token=None: client
        try:
            deleted = milvus_collection_io.drop_collection(
                uri="http://127.0.0.1:19530",
                token=None,
                collection_name="to_delete",
                require_exists=False,
            )
        finally:
            milvus_collection_io.MilvusClient = original_client

        self.assertTrue(deleted)
        self.assertEqual(client.dropped, ["to_delete"])


if __name__ == "__main__":
    unittest.main()
