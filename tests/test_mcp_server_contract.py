import secretarius.mcp_server as mcp_server

from secretarius.mcp_server import handle_mcp_message


def test_initialize_and_tools_list_contract() -> None:
    init_response = handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }
    )
    assert init_response is not None
    assert "result" in init_response
    assert init_response["result"]["serverInfo"]["name"] == "secretarius-mcp"

    list_response = handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
    )
    assert list_response is not None
    tools = list_response["result"]["tools"]
    names = [tool["name"] for tool in tools]
    assert names == [
        "extract_expressions",
        "expressions_to_embeddings",
        "semantic_graph_search",
    ]


def test_extract_tool_call_returns_expressions() -> None:
    def fake_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return {
            "chunks": ["Le camail est vert."],
            "by_chunk": [{"id": 0, "chunk": "Le camail est vert.", "expressions": ["camail", "vert"], "request_fingerprint": "abc"}],
            "expressions": ["camail", "vert"],
            "request_fingerprint": "root-fp",
            "inference_params": {"seed": 42},
            "warning": None,
        }

    mcp_server.extract_expressions_with_llama = fake_pipeline

    response = handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "extract_expressions",
                "arguments": {"text": "Le camail est vert."},
            },
        }
    )
    assert response is not None
    result = response["result"]["structuredContent"]
    assert result["status"] == "ok"
    assert result["tool"] == "extract_expressions"
    assert result["backend"] == "llama_cpp_direct"
    assert result["text_length"] > 0
    assert result["chunk_count"] == 1
    assert isinstance(result["by_chunk"], list)
    assert result["request_fingerprint"] == "root-fp"
    assert result["inference_params"]["seed"] == 42
    assert isinstance(result["expressions"], list)
    assert result["expressions"]


def test_extract_tool_supports_document_content() -> None:
    def fake_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return {
            "chunks": ["Le voile blanc contraste avec le camail vert."],
            "by_chunk": [{"id": 0, "chunk": "Le voile blanc contraste avec le camail vert.", "expressions": ["voile blanc"], "request_fingerprint": "def"}],
            "expressions": ["voile blanc"],
            "request_fingerprint": "root-fp-2",
            "inference_params": {"seed": 42},
            "warning": None,
        }

    mcp_server.extract_expressions_with_llama = fake_pipeline

    response = handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "extract_expressions",
                "arguments": {
                    "document": {
                        "name": "doc.txt",
                        "content": "Le voile blanc contraste avec le camail vert.",
                    },
                },
            },
        }
    )
    assert response is not None
    result = response["result"]["structuredContent"]
    assert result["status"] == "ok"
    assert result["received"]["has_document"] is True
    assert result["expressions"]


def test_extract_tool_rejects_empty_payload() -> None:
    response = handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "extract_expressions",
                "arguments": {},
            },
        }
    )
    assert response is not None
    assert "error" in response
    assert response["error"]["code"] == -32000


def test_extract_tool_returns_empty_list_when_pipeline_fails() -> None:
    def fake_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return {"expressions": [], "warning": "llama unavailable"}

    mcp_server.extract_expressions_with_llama = fake_pipeline

    response = handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "extract_expressions",
                "arguments": {"text": "Texte de test."},
            },
        }
    )
    assert response is not None
    result = response["result"]["structuredContent"]
    assert result["status"] == "ok"
    assert result["expressions"] == []


def test_embeddings_tool_call_returns_vectors() -> None:
    def fake_embedder(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "dimension": 2,
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "warning": None,
        }

    mcp_server.embed_expressions_multilingual = fake_embedder

    response = handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "expressions_to_embeddings",
                "arguments": {"expressions": ["camail vert", "chambre aux deniers"]},
            },
        }
    )
    assert response is not None
    result = response["result"]["structuredContent"]
    assert result["status"] == "ok"
    assert result["tool"] == "expressions_to_embeddings"
    assert result["embedding_count"] == 2
    assert result["dimension"] == 2
    assert result["model"] == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert isinstance(result["embeddings"], list)
    assert len(result["embeddings"]) == 2


def test_semantic_graph_search_calls_milvus_backend() -> None:
    def fake_semantic_backend(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return {
            "graph": {
                "nodes": [{"id": "query:0", "type": "query"}, {"id": "doc:1", "type": "document"}],
                "edges": [{"source": "query:0", "target": "doc:1", "rank": 0, "score": 0.95}],
            },
            "hits": [[{"id": 1, "score": 0.95}]],
            "inserted_count": 1,
            "query_count": 1,
            "collection_name": "secretarius_semantic_graph",
            "metric_type": "COSINE",
            "warning": None,
        }

    mcp_server.semantic_graph_search_milvus = fake_semantic_backend

    response = handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "semantic_graph_search",
                "arguments": {
                    "embeddings": [[0.1, 0.2, 0.3]],
                    "documents": [{"id": "doc-a", "text": "example"}],
                    "top_k": 5,
                },
            },
        }
    )
    assert response is not None
    result = response["result"]["structuredContent"]
    assert result["status"] == "ok"
    assert result["tool"] == "semantic_graph_search"
    assert result["backend"] == "milvus"
    assert result["inserted_count"] == 1
    assert result["query_count"] == 1
    assert isinstance(result["graph"]["nodes"], list)
    assert isinstance(result["graph"]["edges"], list)


def test_unknown_tool_returns_error() -> None:
    response = handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "does_not_exist",
                "arguments": {},
            },
        }
    )
    assert response is not None
    assert "error" in response
    assert response["error"]["code"] == -32602
