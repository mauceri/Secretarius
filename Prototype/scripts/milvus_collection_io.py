from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pymilvus import DataType
from pymilvus import MilvusClient


DEFAULT_URI = "http://127.0.0.1:19530"
DEFAULT_BATCH_SIZE = 500


def _datatype_name(value: Any) -> str:
    if isinstance(value, DataType):
        return value.name
    if isinstance(value, int):
        try:
            return DataType(value).name
        except ValueError:
            return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def _primary_field_name(description: dict[str, Any]) -> str:
    fields = description.get("fields")
    if isinstance(fields, list):
        for field in fields:
            if not isinstance(field, dict):
                continue
            if bool(field.get("is_primary")):
                name = field.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
    return "id"


def _vector_field_name(description: dict[str, Any]) -> str:
    fields = description.get("fields")
    if isinstance(fields, list):
        for field in fields:
            if not isinstance(field, dict):
                continue
            field_type = _datatype_name(field.get("type"))
            if "VECTOR" in field_type:
                name = field.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
    return "vector"


def _vector_dimension(description: dict[str, Any]) -> int | None:
    fields = description.get("fields")
    if not isinstance(fields, list):
        return None
    for field in fields:
        if not isinstance(field, dict):
            continue
        field_type = _datatype_name(field.get("type"))
        if "VECTOR" not in field_type:
            continue
        params = field.get("params")
        if isinstance(params, dict):
            dim = params.get("dim")
            if isinstance(dim, int):
                return dim
            if isinstance(dim, str) and dim.isdigit():
                return int(dim)
    return None


def _index_info(client: MilvusClient, collection_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        index_names = client.list_indexes(collection_name=collection_name)
    except Exception:
        return out
    if not isinstance(index_names, list):
        return out
    for index_name in index_names:
        if not isinstance(index_name, str):
            continue
        try:
            info = client.describe_index(collection_name=collection_name, index_name=index_name)
        except Exception:
            continue
        if isinstance(info, dict):
            out.append(info)
    return out


def _query_batch(
    client: MilvusClient,
    *,
    collection_name: str,
    primary_field: str,
    output_fields: list[str],
    after_id: int | None,
    batch_size: int,
) -> list[dict[str, Any]]:
    filters = []
    if after_id is not None:
        filters.append(f"{primary_field} > {after_id}")
    filter_expr = " and ".join(filters)
    rows = client.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=output_fields,
        limit=batch_size,
    )
    if not isinstance(rows, list):
        return []
    valid_rows = [row for row in rows if isinstance(row, dict)]
    return sorted(valid_rows, key=lambda row: int(row.get(primary_field, 0)))


def export_collection(
    *,
    uri: str,
    token: str | None,
    collection_name: str,
    output_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    client = MilvusClient(uri=uri, token=token)
    description = client.describe_collection(collection_name=collection_name)
    if not isinstance(description, dict):
        raise ValueError(f"unable to describe collection: {collection_name}")

    primary_field = _primary_field_name(description)
    after_id: int | None = None
    rows: list[dict[str, Any]] = []

    while True:
        batch = _query_batch(
            client,
            collection_name=collection_name,
            primary_field=primary_field,
            output_fields=["*"],
            after_id=after_id,
            batch_size=batch_size,
        )
        if not batch:
            break
        rows.extend(batch)
        after_id = int(batch[-1][primary_field])
        if len(batch) < batch_size:
            break

    payload = {
        "format": "secretarius_milvus_collection_dump_v1",
        "collection_name": collection_name,
        "exported_from_uri": uri,
        "schema": description,
        "indexes": _index_info(client, collection_name),
        "row_count": len(rows),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _build_schema_from_dump(client: MilvusClient, dump: dict[str, Any]) -> Any:
    description = dump.get("schema")
    if not isinstance(description, dict):
        raise ValueError("dump schema is missing")

    schema = client.create_schema(
        auto_id=bool(description.get("auto_id", False)),
        enable_dynamic_field=bool(description.get("enable_dynamic_field", True)),
    )
    fields = description.get("fields")
    if not isinstance(fields, list):
        raise ValueError("dump fields are missing")

    for field in fields:
        if not isinstance(field, dict):
            continue
        name = field.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        data_type_name = _datatype_name(field.get("type"))
        data_type = getattr(DataType, data_type_name, None)
        if data_type is None:
            raise ValueError(f"unsupported field data type: {data_type_name}")
        kwargs: dict[str, Any] = {
            "field_name": name,
            "datatype": data_type,
            "is_primary": bool(field.get("is_primary", False)),
            "description": field.get("description", "") or "",
        }
        if "VECTOR" in data_type_name:
            params = field.get("params") if isinstance(field.get("params"), dict) else {}
            dim = params.get("dim")
            if isinstance(dim, str) and dim.isdigit():
                dim = int(dim)
            if not isinstance(dim, int):
                raise ValueError(f"missing dimension for vector field: {name}")
            kwargs["dim"] = dim
        schema.add_field(**kwargs)
    return schema


def _build_index_params(client: MilvusClient, dump: dict[str, Any]) -> Any | None:
    indexes = dump.get("indexes")
    if not isinstance(indexes, list) or not indexes:
        return None
    index_params = client.prepare_index_params()
    added = 0
    for item in indexes:
        if not isinstance(item, dict):
            continue
        field_name = item.get("field_name")
        if not isinstance(field_name, str) or not field_name.strip():
            continue
        kwargs: dict[str, Any] = {}
        for key in ("index_name", "index_type", "metric_type", "params"):
            value = item.get(key)
            if value not in (None, "", {}):
                kwargs[key] = value
        index_params.add_index(field_name=field_name, **kwargs)
        added += 1
    return index_params if added else None


def import_collection(
    *,
    uri: str,
    token: str | None,
    input_path: Path,
    collection_name: str | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    drop_if_exists: bool = False,
) -> str:
    dump = json.loads(input_path.read_text(encoding="utf-8"))
    if dump.get("format") != "secretarius_milvus_collection_dump_v1":
        raise ValueError("unsupported dump format")

    target_collection = collection_name or dump.get("collection_name")
    if not isinstance(target_collection, str) or not target_collection.strip():
        raise ValueError("missing target collection name")
    target_collection = target_collection.strip()

    client = MilvusClient(uri=uri, token=token)

    if client.has_collection(collection_name=target_collection):
        if not drop_if_exists:
            raise ValueError(
                f"collection '{target_collection}' already exists; use --drop-if-exists or another --collection-name"
            )
        client.drop_collection(collection_name=target_collection)

    schema = _build_schema_from_dump(client, dump)
    index_params = _build_index_params(client, dump)
    client.create_collection(
        collection_name=target_collection,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )

    rows = dump.get("rows")
    if not isinstance(rows, list):
        raise ValueError("dump rows are missing")

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        client.insert(collection_name=target_collection, data=batch)

    return target_collection


def drop_collection(
    *,
    uri: str,
    token: str | None,
    collection_name: str,
    require_exists: bool = False,
) -> bool:
    client = MilvusClient(uri=uri, token=token)
    exists = client.has_collection(collection_name=collection_name)
    if not exists:
        if require_exists:
            raise ValueError(f"collection '{collection_name}' does not exist")
        return False
    client.drop_collection(collection_name=collection_name)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export or import a Milvus collection to/from a JSON file.")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Milvus URI, default: http://127.0.0.1:19530")
    parser.add_argument("--token", default=None, help="Optional Milvus token")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export a collection into a JSON file")
    export_parser.add_argument("--collection-name", required=True, help="Milvus collection name to export")
    export_parser.add_argument("--output", required=True, help="JSON file to write")
    export_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Query batch size")

    import_parser = subparsers.add_parser("import", help="Import a collection from a JSON file")
    import_parser.add_argument("--input", required=True, help="JSON file to read")
    import_parser.add_argument("--collection-name", default=None, help="Override destination collection name")
    import_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Insert batch size")
    import_parser.add_argument(
        "--drop-if-exists",
        action="store_true",
        help="Drop the destination collection first if it already exists",
    )

    drop_parser = subparsers.add_parser("drop", help="Drop a collection explicitly")
    drop_parser.add_argument("--collection-name", required=True, help="Collection to drop")
    drop_parser.add_argument(
        "--require-exists",
        action="store_true",
        help="Fail if the collection does not exist",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "export":
        output_path = export_collection(
            uri=args.uri,
            token=args.token,
            collection_name=args.collection_name,
            output_path=Path(args.output),
            batch_size=args.batch_size,
        )
        print(output_path)
        return 0

    if args.command == "import":
        collection_name = import_collection(
            uri=args.uri,
            token=args.token,
            input_path=Path(args.input),
            collection_name=args.collection_name,
            batch_size=args.batch_size,
            drop_if_exists=args.drop_if_exists,
        )
        print(collection_name)
        return 0

    if args.command == "drop":
        deleted = drop_collection(
            uri=args.uri,
            token=args.token,
            collection_name=args.collection_name,
            require_exists=args.require_exists,
        )
        print("DROPPED" if deleted else "SKIPPED")
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
