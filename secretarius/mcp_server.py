from __future__ import annotations

import json
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .expression_extractor import extract_expressions_with_llama
except ImportError:
    module_path = Path(__file__).resolve().parent / "expression_extractor.py"
    spec = importlib.util.spec_from_file_location("secretarius_expression_extractor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load expression extractor from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    extract_expressions_with_llama = module.extract_expressions_with_llama

try:
    from .embeddings import embed_expressions_multilingual
except ImportError:
    module_path = Path(__file__).resolve().parent / "embeddings.py"
    spec = importlib.util.spec_from_file_location("secretarius_embeddings", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load embeddings module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    embed_expressions_multilingual = module.embed_expressions_multilingual

try:
    from .semantic_graph import semantic_graph_search_milvus
except ImportError:
    module_path = Path(__file__).resolve().parent / "semantic_graph.py"
    spec = importlib.util.spec_from_file_location("secretarius_semantic_graph", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load semantic_graph module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    semantic_graph_search_milvus = module.semantic_graph_search_milvus


JSONRPC_VERSION = "2.0"
SERVER_NAME = "secretarius-mcp"
SERVER_VERSION = "0.1.0"
_LAST_INPUT_MODE = "framed"


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]


def _tools_catalog() -> list[MCPTool]:
    return [
        MCPTool(
            name="extract_expressions",
            description=(
                "Extrait des expressions caracteristiques a partir d'un texte "
                "ou d'un document attache via llama.cpp."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Texte brut a analyser.",
                    },
                    "document": {
                        "type": "object",
                        "description": "Document attache (meta ou contenu embarque).",
                        "properties": {
                            "name": {"type": "string"},
                            "mime_type": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "additionalProperties": True,
                    },
                    "llama_url": {
                        "type": "string",
                        "description": "Endpoint chat completions llama.cpp.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Nom logique du modele (ex: phi4).",
                    },
                    "timeout_s": {
                        "type": "number",
                        "minimum": 0.1,
                        "default": 30.0,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 512,
                    },
                    "seed": {
                        "type": "integer",
                        "default": 42,
                    },
                    "prompt_path": {
                        "type": "string",
                        "description": "Chemin vers le prompt texte (par defaut secretarius/prompts/prompt.txt).",
                    },
                    "debug_return_raw": {
                        "type": "boolean",
                        "default": False,
                        "description": "Inclure la sortie brute du modele pour debug.",
                    },
                },
                "anyOf": [{"required": ["text"]}, {"required": ["document"]}],
                "additionalProperties": True,
            },
        ),
        MCPTool(
            name="expressions_to_embeddings",
            description=(
                "Transforme une liste d'expressions caracteristiques en plongements "
                "(embeddings) multilingues."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "expressions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Expressions caracteristiques a projeter.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Modele d'embedding cible (optionnel).",
                    },
                    "normalize": {
                        "type": "boolean",
                        "default": True,
                        "description": "Normaliser les vecteurs (recommande pour cosine).",
                    },
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 32,
                    },
                },
                "required": ["expressions"],
                "additionalProperties": True,
            },
        ),
        MCPTool(
            name="semantic_graph_search",
            description=(
                "Recherche/insertions unifiees dans Milvus a partir d'une liste de "
                "plongements."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "embeddings": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "minItems": 1,
                        "description": "Liste de vecteurs d'entree.",
                    },
                    "documents": {
                        "type": "array",
                        "description": "Documents optionnels a inserer dans le meme flux.",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 10,
                    },
                    "milvus_uri": {
                        "type": "string",
                        "description": "URI Milvus (ex: http://127.0.0.1:19530).",
                    },
                    "milvus_token": {
                        "type": "string",
                        "description": "Token Milvus optionnel.",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Collection cible dans Milvus.",
                    },
                    "metric_type": {
                        "type": "string",
                        "description": "Métrique d'index (COSINE, IP, L2).",
                    },
                },
                "required": ["embeddings"],
                "additionalProperties": True,
            },
        ),
    ]


def _tool_by_name(name: str) -> Optional[MCPTool]:
    for tool in _tools_catalog():
        if tool.name == name:
            return tool
    return None


def _make_response(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "result": result,
    }


def _make_error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _tool_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False),
            }
        ],
        "structuredContent": payload,
    }


def _handle_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "extract_expressions":
        text = _resolve_extract_text(arguments)
        llama_url = arguments.get("llama_url") or os.environ.get(
            "SECRETARIUS_LLAMA_URL",
            "http://127.0.0.1:8080/v1/chat/completions",
        )
        model = arguments.get("model") or os.environ.get("SECRETARIUS_LLAMA_MODEL", "phi4")
        timeout_s = arguments.get("timeout_s", float(os.environ.get("SECRETARIUS_TIMEOUT_S", "30.0")))
        max_tokens = arguments.get("max_tokens", int(os.environ.get("SECRETARIUS_MAX_TOKENS", "512")))
        seed = arguments.get("seed", int(os.environ.get("SECRETARIUS_SEED", "42")))
        prompt_path = arguments.get("prompt_path") or os.environ.get("SECRETARIUS_PROMPT_PATH")
        debug_env = os.environ.get("SECRETARIUS_DEBUG_RETURN_RAW", "false").strip().lower()
        debug_default = debug_env in ("1", "true", "yes", "on")
        debug_return_raw = arguments.get("debug_return_raw", debug_default)
        if not isinstance(llama_url, str) or not llama_url.strip():
            raise ValueError("'llama_url' must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("'model' must be a non-empty string")
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("'timeout_s' must be a positive number")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("'max_tokens' must be an integer >= 1")
        if not isinstance(seed, int):
            raise ValueError("'seed' must be an integer")
        if prompt_path is not None and not isinstance(prompt_path, str):
            raise ValueError("'prompt_path' must be a string when provided")
        if not isinstance(debug_return_raw, bool):
            raise ValueError("'debug_return_raw' must be a boolean")

        result = extract_expressions_with_llama(
            text,
            llama_url=llama_url,
            model=model,
            timeout_s=float(timeout_s),
            max_tokens=max_tokens,
            seed=seed,
            prompt_path=prompt_path,
            debug_return_raw=debug_return_raw,
        )
        by_chunk = result.get("by_chunk", [])
        chunk_count = len(result.get("chunks", []))
        expressions = result.get("expressions", [])
        request_fingerprint = result.get("request_fingerprint")
        inference_params = result.get("inference_params")
        raw_llm_outputs = result.get("raw_llm_outputs") if debug_return_raw else None
        warning = result.get("warning")
        if not isinstance(by_chunk, list):
            by_chunk = []
        if not isinstance(expressions, list):
            expressions = []
        expressions = [e for e in expressions if isinstance(e, str)]

        return _tool_result(
            {
                "status": "ok",
                "tool": name,
                "backend": "llama_cpp_direct",
                "received": {
                    "has_text": bool(arguments.get("text")),
                    "has_document": isinstance(arguments.get("document"), dict),
                },
                "text_length": len(text),
                "chunk_count": chunk_count,
                "by_chunk": by_chunk,
                "expressions": expressions,
                "request_fingerprint": request_fingerprint,
                "inference_params": inference_params,
                **({"raw_llm_outputs": raw_llm_outputs} if debug_return_raw else {}),
                "message": "Extraction d'expressions caracteristiques.",
                "warning": warning,
            }
        )

    if name == "expressions_to_embeddings":
        expressions = arguments.get("expressions", [])
        model = arguments.get("model")
        normalize = arguments.get("normalize", True)
        batch_size = arguments.get("batch_size", 32)
        if not isinstance(expressions, list) or not expressions:
            raise ValueError("'expressions' must be a non-empty list")
        if not all(isinstance(item, str) for item in expressions):
            raise ValueError("'expressions' must contain only strings")
        if model is not None and not isinstance(model, str):
            raise ValueError("'model' must be a string when provided")
        if not isinstance(normalize, bool):
            raise ValueError("'normalize' must be a boolean")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("'batch_size' must be an integer >= 1")

        result = embed_expressions_multilingual(
            expressions,
            model=model,
            normalize=normalize,
            batch_size=batch_size,
        )
        embeddings = result.get("embeddings", [])
        dimension = result.get("dimension", 0)
        used_model = result.get("model")
        warning = result.get("warning")
        return _tool_result(
            {
                "status": "ok",
                "tool": name,
                "received_count": len(expressions) if isinstance(expressions, list) else 0,
                "embedding_count": len(embeddings) if isinstance(embeddings, list) else 0,
                "dimension": dimension,
                "normalized": normalize,
                "model": used_model,
                "embeddings": embeddings,
                "message": "Generation de plongements multilingues.",
                "warning": warning,
            }
        )

    if name == "semantic_graph_search":
        embeddings = arguments.get("embeddings", [])
        documents = arguments.get("documents", [])
        top_k = arguments.get("top_k", 10)
        milvus_uri = arguments.get("milvus_uri") or os.environ.get(
            "SECRETARIUS_MILVUS_URI",
            "http://127.0.0.1:19530",
        )
        milvus_token = arguments.get("milvus_token") or os.environ.get("SECRETARIUS_MILVUS_TOKEN")
        collection_name = arguments.get("collection_name") or os.environ.get(
            "SECRETARIUS_MILVUS_COLLECTION",
            "secretarius_semantic_graph",
        )
        metric_type = arguments.get("metric_type") or os.environ.get(
            "SECRETARIUS_MILVUS_METRIC",
            "COSINE",
        )

        if not isinstance(embeddings, list) or not embeddings:
            raise ValueError("'embeddings' must be a non-empty list of vectors")
        if documents is not None and not isinstance(documents, list):
            raise ValueError("'documents' must be a list when provided")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("'top_k' must be an integer >= 1")
        if not isinstance(milvus_uri, str) or not milvus_uri.strip():
            raise ValueError("'milvus_uri' must be a non-empty string")
        if milvus_token is not None and not isinstance(milvus_token, str):
            raise ValueError("'milvus_token' must be a string when provided")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("'collection_name' must be a non-empty string")
        if not isinstance(metric_type, str) or not metric_type.strip():
            raise ValueError("'metric_type' must be a non-empty string")

        result = semantic_graph_search_milvus(
            embeddings,
            documents=documents,
            top_k=top_k,
            uri=milvus_uri,
            token=milvus_token,
            collection_name=collection_name,
            metric_type=metric_type,
        )
        graph = result.get("graph", {"nodes": [], "edges": []})
        warning = result.get("warning")
        return _tool_result(
            {
                "status": "ok",
                "tool": name,
                "backend": "milvus",
                "received": {
                    "embedding_count": len(embeddings) if isinstance(embeddings, list) else 0,
                    "document_count": len(documents) if isinstance(documents, list) else 0,
                    "unified_query_and_insert": True,
                },
                "collection_name": result.get("collection_name", collection_name),
                "metric_type": result.get("metric_type", metric_type),
                "inserted_count": result.get("inserted_count", 0),
                "query_count": result.get("query_count", 0),
                "graph": graph,
                "hits": result.get("hits", []),
                "message": "Recherche/insertions semantiques via Milvus.",
                "warning": warning,
            }
        )

    raise ValueError(f"Unknown tool: {name}")


def _resolve_extract_text(arguments: Dict[str, Any]) -> str:
    text = arguments.get("text")
    if isinstance(text, str) and text.strip():
        return text

    document = arguments.get("document")
    if isinstance(document, dict):
        for key in ("content", "text"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value

    raise ValueError("Provide non-empty 'text' or 'document.content'/'document.text'")


def handle_mcp_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params", {})

    if not isinstance(method, str):
        if request_id is None:
            return None
        return _make_error(request_id, -32600, "Invalid Request")

    # Notifications have no id and should not receive a response.
    if request_id is None:
        return None

    if method == "initialize":
        return _make_response(
            request_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            },
        )

    if method == "ping":
        return _make_response(request_id, {})

    if method == "tools/list":
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in _tools_catalog()
        ]
        return _make_response(request_id, {"tools": tools})

    if method == "tools/call":
        if not isinstance(params, dict):
            return _make_error(request_id, -32602, "Invalid params")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not name:
            return _make_error(request_id, -32602, "Tool name must be a non-empty string")
        if not isinstance(arguments, dict):
            return _make_error(request_id, -32602, "Tool arguments must be an object")
        if _tool_by_name(name) is None:
            return _make_error(request_id, -32602, f"Unknown tool: {name}")

        try:
            result = _handle_tool_call(name, arguments)
        except Exception as exc:  # pragma: no cover
            return _make_error(request_id, -32000, f"Tool execution error: {exc}")
        return _make_response(request_id, result)

    return _make_error(request_id, -32601, f"Method not found: {method}")


def run_stdio_server() -> None:
    while True:
        message = _read_framed_message()
        if message is None:
            return

        response = handle_mcp_message(message)
        if response is None:
            continue
        _write_framed_message(response)


def _read_framed_message() -> Optional[Dict[str, Any]]:
    global _LAST_INPUT_MODE
    while True:
        first_raw = sys.stdin.buffer.readline()
        if first_raw == b"":
            return None
        first_line = first_raw.decode("utf-8", errors="replace").strip()
        if first_line:
            break

    # Fallback mode for clients that send plain JSON lines over stdio.
    if first_line.startswith("{"):
        try:
            message = json.loads(first_line)
        except (ValueError, json.JSONDecodeError):
            return None
        if not isinstance(message, dict):
            return None
        _LAST_INPUT_MODE = "line"
        _debug_log(f"recv(line): {message.get('method', '<unknown>')}")
        return message

    headers: Dict[str, str] = {}
    if ":" in first_line:
        key, value = first_line.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    while True:
        raw = sys.stdin.buffer.readline()
        if raw == b"":
            return None
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    content_length_value = headers.get("content-length")
    if content_length_value is None:
        return None

    try:
        content_length = int(content_length_value)
    except ValueError:
        return None
    if content_length <= 0:
        return None

    payload = sys.stdin.buffer.read(content_length)
    if len(payload) != content_length:
        return None

    try:
        message = json.loads(payload.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(message, dict):
        return None
    _LAST_INPUT_MODE = "framed"
    _debug_log(f"recv(framed): {message.get('method', '<unknown>')}")
    return message


def _write_framed_message(message: Dict[str, Any]) -> None:
    if _LAST_INPUT_MODE == "line":
        sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    else:
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        sys.stdout.buffer.write(header)
        sys.stdout.buffer.write(body)
        sys.stdout.buffer.flush()
    _debug_log(f"send: id={message.get('id')}")


def _debug_log(line: str) -> None:
    path = os.environ.get("SECRETARIUS_MCP_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        return
