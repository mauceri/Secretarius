from __future__ import annotations

import json
import importlib.util
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from localization import Translator

try:
    from .expression_extractor import extract_expressions
except ImportError:
    module_path = Path(__file__).resolve().parent / "expression_extractor.py"
    spec = importlib.util.spec_from_file_location("secretarius_expression_extractor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load expression extractor from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "extract_expressions"):
        raise RuntimeError(f"Unable to resolve extractor function from {module_path}")
    extract_expressions = module.extract_expressions

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

try:
    from .document_schema import (
        add_indexing_error,
        enrich_document_with_embeddings,
        enrich_document_with_extraction,
        expressions_from_document,
        normalize_document,
        resolve_text_for_extraction,
        set_indexing_state,
    )
except ImportError:
    module_path = Path(__file__).resolve().parent / "document_schema.py"
    spec = importlib.util.spec_from_file_location("secretarius_document_schema", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load document schema module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    add_indexing_error = module.add_indexing_error
    enrich_document_with_embeddings = module.enrich_document_with_embeddings
    enrich_document_with_extraction = module.enrich_document_with_extraction
    expressions_from_document = module.expressions_from_document
    normalize_document = module.normalize_document
    resolve_text_for_extraction = module.resolve_text_for_extraction
    set_indexing_state = module.set_indexing_state

try:
    from .document_pipeline import analyse_texte_documentaire, index_document_text, search_documents_by_text, update_document_text
except ImportError:
    module_path = Path(__file__).resolve().parent / "document_pipeline.py"
    spec = importlib.util.spec_from_file_location("secretarius_document_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load document pipeline module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    analyse_texte_documentaire = module.analyse_texte_documentaire
    index_document_text = module.index_document_text
    search_documents_by_text = module.search_documents_by_text
    update_document_text = module.update_document_text


JSONRPC_VERSION = "2.0"
SERVER_NAME = "secretarius-mcp"
SERVER_VERSION = "0.1.0"
_LAST_INPUT_MODE = "framed"
_WARMUP_STARTED = False
_KEYWORD_MATCH_BONUS = 0.1
_TITLE_MATCH_BONUS = 0.05
_TRANSLATOR = Translator(os.environ.get("SECRETARIUS_LOCALE", "fr"))


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]


def _tools_catalog() -> list[MCPTool]:
    return [
        MCPTool(
            name="ask_oracle",
            description="Test tool: ask the oracle a yes/no question only when explicitly requested.",
            input_schema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question for the oracle.",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        ),
        MCPTool(
            name="extract_expressions",
            description=(
                "Extrait les expressions caracteristiques d'un texte brut. "
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Texte brut a analyser.",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        MCPTool(
            name="semantic_graph_search",
            description=(
                "Outil technique de bas niveau pour rechercher ou inserer dans "
                "Milvus a partir d'embeddings, d'expressions deja extraites ou "
                "d'un document structure. Ne pas utiliser pour extraire des "
                "expressions d'un texte brut ni pour une simple requete textuelle."
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
                    "expressions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Alternative a embeddings: expressions a encoder avant recherche.",
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
                    "document": {
                        "type": "object",
                        "description": "Document Secretarius v0.1 principal.",
                        "additionalProperties": True,
                    },
                    "upsert": {
                        "type": "boolean",
                        "default": True,
                        "description": "Inserer aussi les documents fournis avant recherche.",
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
                    "model": {
                        "type": "string",
                        "description": "Modele d'embedding (si expressions sont fournies).",
                    },
                    "normalize": {
                        "type": "boolean",
                        "default": True,
                        "description": "Normaliser les embeddings calcules a partir des expressions.",
                    },
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 32,
                        "description": "Taille de lot embedding (si expressions sont fournies).",
                    },
                },
                "anyOf": [
                    {"required": ["embeddings"]},
                    {"required": ["expressions"]},
                    {"required": ["document"]},
                ],
                "additionalProperties": True,
            },
        ),
        MCPTool(
            name="index_text",
            description=(
                "Indexe un document decrit par une chaine de caracteres. "
                "Le texte peut contenir titre, date, URL ou mots-cles."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Texte documentaire a indexer.",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        MCPTool(
            name="search_text",
            description=(
                "Recherche dans la base des documents similaires a partir "
                "d'une requete textuelle."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requete textuelle a rechercher.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        MCPTool(
            name="update_text",
            description=(
                "Remplace une note existante a partir d'une chaine de caracteres "
                "contenant un doc_id explicite."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Texte documentaire a mettre a jour, avec doc_id: ...",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
    ]


def _public_tools_catalog() -> list[MCPTool]:
    hidden_tool_names = {"semantic_graph_search"}
    return [tool for tool in _tools_catalog() if tool.name not in hidden_tool_names]


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


def _handle_ask_oracle(arguments: Dict[str, Any]) -> Dict[str, Any]:
    question = str(arguments.get("question", ""))
    answer = random.choice(["OUI", "NON"])
    _debug_log(f"ask_oracle question_len={len(question)}")
    return {
        "content": [
            {
                "type": "text",
                "text": f"The oracle pondered your question: '{question}' and answered: {answer}",
            }
        ],
        "isError": False,
    }

def _handle_extract_expressions(arguments: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.monotonic()
    text, normalized_doc = resolve_text_for_extraction(arguments)
    text = text or ""
    llama_cpp_url = (
        arguments.get("llama_cpp_url")
        or arguments.get("llama_url")
        or os.environ.get("SECRETARIUS_LLAMA_CPP_URL")
        or os.environ.get("SECRETARIUS_LLAMA_URL")
        or "http://127.0.0.1:8989/v1/chat/completions"
    )
    llama_cpp_model = (
        arguments.get("llama_cpp_model")
        or arguments.get("model")
        or os.environ.get(
        "SECRETARIUS_LLAMA_CPP_MODEL",
        os.environ.get("SECRETARIUS_LLAMA_MODEL", "local-llama-cpp"),
    )
    )
    timeout_s = arguments.get("timeout_s", float(os.environ.get("SECRETARIUS_TIMEOUT_S", "30.0")))
    max_tokens = arguments.get("max_tokens", int(os.environ.get("SECRETARIUS_MAX_TOKENS", "20480")))
    seed = arguments.get("seed", int(os.environ.get("SECRETARIUS_SEED", "42")))
    prompt_path = arguments.get("prompt_path") or os.environ.get("SECRETARIUS_PROMPT_PATH")
    debug_env = os.environ.get("SECRETARIUS_DEBUG_RETURN_RAW", "false").strip().lower()
    debug_default = debug_env in ("1", "true", "yes", "on")
    debug_return_raw = arguments.get("debug_return_raw", debug_default)

    _debug_log(
        "extract.args "
        f"text_len={len(text)} "
        f"llama_cpp_url={llama_cpp_url} "
        f"llama_cpp_model={llama_cpp_model} "
        f"prompt_path={prompt_path} "
        f"arg_keys={sorted(list(arguments.keys()))}"
    )
    if not isinstance(llama_cpp_url, str) or not llama_cpp_url.strip():
        raise ValueError("'llama_cpp_url' must be a non-empty string")
    if not isinstance(llama_cpp_model, str) or not llama_cpp_model.strip():
        raise ValueError("'llama_cpp_model' must be a non-empty string")
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
    if not text and normalized_doc is None:
        raise ValueError("Provide non-empty 'text' or 'document.content.text'")

    if normalized_doc is not None:
        if not text:
            set_indexing_state(normalized_doc, "queued")
            add_indexing_error(normalized_doc, "extracting", "No inline content to extract (fetching required)")
            payload: Dict[str, Any] = {
                "expressions": [],
                "warning": "document has no inline text",
                "document": normalized_doc,
            }
            if debug_return_raw:
                payload.update(
                    {
                        "status": "ok",
                        "tool": "extract_expressions",
                        "backend": "llama_cpp_direct",
                        "received": {
                            "has_text": bool(arguments.get("text")),
                            "has_document": True,
                        },
                        "text_length": 0,
                        "chunk_count": 0,
                        "by_chunk": [],
                        "request_fingerprint": None,
                        "inference_params": None,
                        "message": "Document accepté sans contenu inline: extraction différée.",
                        "raw_llm_outputs": None,
                    }
                )
            return _tool_result(payload)
        set_indexing_state(normalized_doc, "extracting")

    result = extract_expressions(
        text,
        llama_cpp_url=llama_cpp_url,
        llama_cpp_model=llama_cpp_model,
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
    _debug_log(
        "extract_expressions timing: "
        f"elapsed_s={time.monotonic()-started_at:.3f} "
        f"text_len={len(text)} chunk_count={chunk_count}"
    )
    if normalized_doc is not None:
        normalized_doc = enrich_document_with_extraction(
            normalized_doc,
            chunks=result.get("chunks", []) if isinstance(result.get("chunks"), list) else [],
            by_chunk=by_chunk,
            expressions=expressions,
        )

    payload: Dict[str, Any] = {"expressions": expressions}
    if warning:
        payload["warning"] = warning
    if normalized_doc is not None:
        payload["document"] = normalized_doc
    if debug_return_raw:
        payload.update(
            {
                "status": "ok",
                "tool": "extract_expressions",
                "backend": "llama_cpp_direct",
                "received": {
                    "has_text": bool(arguments.get("text")),
                    "has_document": isinstance(arguments.get("document"), dict),
                },
                "text_length": len(text),
                "chunk_count": chunk_count,
                "by_chunk": by_chunk,
                "request_fingerprint": request_fingerprint,
                "inference_params": inference_params,
                "message": "Extraction d'expressions caracteristiques.",
                "raw_llm_outputs": raw_llm_outputs,
            }
        )
    return _tool_result(payload)


def _handle_expressions_to_embeddings(arguments: Dict[str, Any]) -> Dict[str, Any]:
    normalized_doc = normalize_document(arguments.get("document")) if isinstance(arguments.get("document"), dict) else None
    expressions = arguments.get("expressions", [])
    if normalized_doc is not None and not expressions:
        expressions = expressions_from_document(normalized_doc)
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
    if normalized_doc is not None:
        set_indexing_state(normalized_doc, "embedding")

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
    if normalized_doc is not None:
        normalized_doc = enrich_document_with_embeddings(
            normalized_doc,
            expressions=expressions if isinstance(expressions, list) else [],
            embeddings=embeddings if isinstance(embeddings, list) else [],
        )
    payload = {
        "status": "ok",
        "tool": "expressions_to_embeddings",
        "received_count": len(expressions) if isinstance(expressions, list) else 0,
        "embedding_count": len(embeddings) if isinstance(embeddings, list) else 0,
        "dimension": dimension,
        "normalized": normalize,
        "model": used_model,
        "embeddings": embeddings,
        "message": "Generation de plongements multilingues.",
        "warning": warning,
    }
    if normalized_doc is not None:
        payload["document"] = normalized_doc
    return _tool_result(payload)


def _handle_semantic_graph_search(arguments: Dict[str, Any]) -> Dict[str, Any]:
    embeddings = arguments.get("embeddings", [])
    expressions = arguments.get("expressions")
    documents = arguments.get("documents", [])
    normalized_doc = normalize_document(arguments.get("document")) if isinstance(arguments.get("document"), dict) else None
    upsert = arguments.get("upsert", True)
    top_k = _resolve_top_k_argument(arguments)
    embed_model = arguments.get("model")
    normalize = arguments.get("normalize", True)
    batch_size = arguments.get("batch_size", 32)
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

    if expressions is None and normalized_doc is not None:
        expressions = expressions_from_document(normalized_doc)
    if expressions is not None:
        if not isinstance(expressions, list) or not all(isinstance(e, str) for e in expressions):
            raise ValueError("'expressions' must be a list of strings when provided")
    if not isinstance(normalize, bool):
        raise ValueError("'normalize' must be a boolean")
    if embed_model is not None and not isinstance(embed_model, str):
        raise ValueError("'model' must be a string when provided")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("'batch_size' must be an integer >= 1")
    if (not isinstance(embeddings, list) or not embeddings) and isinstance(expressions, list) and expressions:
        embedded = embed_expressions_multilingual(
            expressions,
            model=embed_model,
            normalize=normalize,
            batch_size=batch_size,
        )
        embeddings = embedded.get("embeddings", [])
    if not isinstance(embeddings, list) or not embeddings:
        raise ValueError(
            "Provide non-empty 'embeddings', or pass non-empty 'expressions'/'document' so embeddings can be derived"
        )
    if documents is not None and not isinstance(documents, list):
        raise ValueError("'documents' must be a list when provided")
    if not isinstance(upsert, bool):
        raise ValueError("'upsert' must be a boolean")
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
    docs_to_insert = documents if isinstance(documents, list) else []
    upsert_source = "none"
    if normalized_doc is not None and upsert and not docs_to_insert:
        docs_to_insert = [normalized_doc]
        set_indexing_state(normalized_doc, "upserting")
        upsert_source = "document"
    elif upsert and not docs_to_insert:
        if isinstance(expressions, list) and expressions:
            count = min(len(expressions), len(embeddings))
            docs_to_insert = [
                {
                    "schema": "secretarius.document.v0.1",
                    "type": "snippet",
                    "content": {"mode": "inline", "text": str(expressions[idx])},
                    "derived": {"expressions": [{"expression": str(expressions[idx])}]},
                    "indexing": {"source": "semantic_graph_search:auto-from-expressions"},
                }
                for idx in range(count)
            ]
            upsert_source = "auto_expressions"
        elif isinstance(embeddings, list) and embeddings:
            docs_to_insert = [
                {
                    "schema": "secretarius.document.v0.1",
                    "type": "embedding_only",
                    "content": {"mode": "none", "text": None},
                    "indexing": {
                        "source": "semantic_graph_search:auto-from-embeddings",
                        "source_idx": idx,
                    },
                }
                for idx in range(len(embeddings))
            ]
            upsert_source = "auto_embeddings"

    result = semantic_graph_search_milvus(
        embeddings,
        documents=docs_to_insert if upsert else [],
        top_k=top_k,
        uri=milvus_uri,
        token=milvus_token,
        collection_name=collection_name,
        metric_type=metric_type,
    )
    graph = result.get("graph", {"nodes": [], "edges": []})
    warning = result.get("warning")
    payload = {
        "status": "ok",
        "tool": "semantic_graph_search",
        "backend": "milvus",
        "received": {
            "embedding_count": len(embeddings) if isinstance(embeddings, list) else 0,
            "expression_count": len(expressions) if isinstance(expressions, list) else 0,
            "document_count": len(docs_to_insert) if isinstance(docs_to_insert, list) else 0,
            "unified_query_and_insert": upsert,
            "upsert_source": upsert_source if upsert else "disabled",
        },
        "collection_name": result.get("collection_name", collection_name),
        "metric_type": result.get("metric_type", metric_type),
        "inserted_count": result.get("inserted_count", 0),
        "query_count": result.get("query_count", 0),
        "top_k": result.get("top_k", top_k),
        "graph": graph,
        "hits": result.get("hits", []),
        "message": "Recherche/insertions semantiques via Milvus.",
        "warning": warning,
    }
    if normalized_doc is not None:
        if warning:
            add_indexing_error(normalized_doc, "upserting", str(warning))
        else:
            set_indexing_state(normalized_doc, "done")
        payload["document"] = normalized_doc
    return _tool_result(payload)


def _handle_index_text(arguments: Dict[str, Any]) -> Dict[str, Any]:
    debug_full = bool(arguments.get("debug_full", False))
    text, normalized_doc = resolve_text_for_extraction(arguments)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("'text' must be a non-empty string")

    result = index_document_text(
        text.strip(),
        base_document=normalized_doc,
        llama_cpp_url=arguments.get("llama_cpp_url")
        or arguments.get("llama_url")
        or os.environ.get("SECRETARIUS_LLAMA_CPP_URL")
        or os.environ.get("SECRETARIUS_LLAMA_URL")
        or "http://127.0.0.1:8989/v1/chat/completions",
        llama_cpp_model=arguments.get("llama_cpp_model")
        or arguments.get("model")
        or os.environ.get("SECRETARIUS_LLAMA_CPP_MODEL")
        or os.environ.get("SECRETARIUS_LLAMA_MODEL", "local-llama-cpp"),
        timeout_s=arguments.get("timeout_s", float(os.environ.get("SECRETARIUS_TIMEOUT_S", "30.0"))),
        max_tokens=arguments.get("max_tokens", int(os.environ.get("SECRETARIUS_MAX_TOKENS", "20480"))),
        seed=arguments.get("seed", int(os.environ.get("SECRETARIUS_SEED", "42"))),
        prompt_path=arguments.get("prompt_path") or os.environ.get("SECRETARIUS_PROMPT_PATH"),
        debug_return_raw=bool(arguments.get("debug_return_raw", False)),
        embedding_model=arguments.get("model"),
        normalize_embeddings=arguments.get("normalize", True),
        batch_size=arguments.get("batch_size", 32),
        milvus_uri=arguments.get("milvus_uri") or os.environ.get("SECRETARIUS_MILVUS_URI", "http://127.0.0.1:19530"),
        milvus_token=arguments.get("milvus_token") or os.environ.get("SECRETARIUS_MILVUS_TOKEN"),
        collection_name=arguments.get("collection_name")
        or os.environ.get("SECRETARIUS_MILVUS_COLLECTION", "secretarius_semantic_graph"),
        metric_type=arguments.get("metric_type") or os.environ.get("SECRETARIUS_MILVUS_METRIC", "COSINE"),
        top_k=_resolve_top_k_argument(arguments),
    )
    extract_payload = result.get("extract", {})
    graph_payload = result.get("index", {})
    normalized_doc = result.get("document")
    extract_expressions = extract_payload.get("expressions")
    extract_count = len(extract_expressions) if isinstance(extract_expressions, list) else 0
    index_hits = graph_payload.get("hits")
    hit_lists = len(index_hits) if isinstance(index_hits, list) else 0
    warning = extract_payload.get("warning") or graph_payload.get("warning")
    payload = {
        "status": "ok",
        "tool": "index_text",
        "message": _TRANSLATOR.get("messages.index_complete"),
        "summary": {
            "expressions_count": extract_count,
            "collection_name": graph_payload.get("collection_name"),
            "inserted_count": graph_payload.get("inserted_count", 0),
            "query_count": graph_payload.get("query_count", 0),
            "hit_lists": hit_lists,
        },
        "warning": warning,
    }
    if debug_full:
        payload["extract"] = extract_payload
        payload["index"] = graph_payload
    return _tool_result(payload)


def _handle_search_text(arguments: Dict[str, Any]) -> Dict[str, Any]:
    debug_full = bool(arguments.get("debug_full", False))
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' must be a non-empty string")
    result = search_documents_by_text(
        query.strip(),
        llama_cpp_url=arguments.get("llama_cpp_url")
        or arguments.get("llama_url")
        or os.environ.get("SECRETARIUS_LLAMA_CPP_URL")
        or os.environ.get("SECRETARIUS_LLAMA_URL")
        or "http://127.0.0.1:8989/v1/chat/completions",
        llama_cpp_model=arguments.get("llama_cpp_model")
        or arguments.get("model")
        or os.environ.get("SECRETARIUS_LLAMA_CPP_MODEL")
        or os.environ.get("SECRETARIUS_LLAMA_MODEL", "local-llama-cpp"),
        timeout_s=arguments.get("timeout_s", float(os.environ.get("SECRETARIUS_TIMEOUT_S", "30.0"))),
        max_tokens=arguments.get("max_tokens", int(os.environ.get("SECRETARIUS_MAX_TOKENS", "20480"))),
        seed=arguments.get("seed", int(os.environ.get("SECRETARIUS_SEED", "42"))),
        prompt_path=arguments.get("prompt_path") or os.environ.get("SECRETARIUS_PROMPT_PATH"),
        debug_return_raw=bool(arguments.get("debug_return_raw", False)),
        embedding_model=arguments.get("model"),
        normalize_embeddings=arguments.get("normalize", True),
        batch_size=arguments.get("batch_size", 32),
        milvus_uri=arguments.get("milvus_uri") or os.environ.get("SECRETARIUS_MILVUS_URI", "http://127.0.0.1:19530"),
        milvus_token=arguments.get("milvus_token") or os.environ.get("SECRETARIUS_MILVUS_TOKEN"),
        collection_name=arguments.get("collection_name")
        or os.environ.get("SECRETARIUS_MILVUS_COLLECTION", "secretarius_semantic_graph"),
        metric_type=arguments.get("metric_type") or os.environ.get("SECRETARIUS_MILVUS_METRIC", "COSINE"),
        top_k=_resolve_top_k_argument(arguments),
        min_score=arguments.get("min_score", _read_optional_float_env("SECRETARIUS_MILVUS_MIN_SCORE")),
    )
    graph_payload = result.get("search", {})
    extract_payload = result.get("extract") if isinstance(result.get("extract"), dict) else {}
    query_document = result.get("document") if isinstance(result.get("document"), dict) else {}
    query_keywords = _extract_document_keywords(query_document)
    query_terms = _extract_query_terms(query)
    query_expressions = _extract_query_expressions(extract_payload)

    hits = graph_payload.get("hits")
    hit_lists = len(hits) if isinstance(hits, list) else 0
    documents = _extract_search_documents(
        hits,
        query_expressions=query_expressions,
        query_keywords=query_keywords,
        query_terms=query_terms,
        min_score=graph_payload.get("min_score"),
    )
    payload = {
        "status": "ok",
        "tool": "search_text",
        "message": _TRANSLATOR.get("messages.search_complete"),
        "query": query.strip(),
        "summary": {
            "collection_name": graph_payload.get("collection_name"),
            "query_count": graph_payload.get("query_count", 0),
            "hit_lists": hit_lists,
            "document_count": len(documents),
            "top_k": graph_payload.get("top_k", _resolve_top_k_argument(arguments)),
            "min_score": graph_payload.get("min_score"),
            "keyword_query_count": len(query_keywords),
        },
        "documents": documents,
        "warning": graph_payload.get("warning"),
    }
    if debug_full:
        payload["search"] = graph_payload
    return _tool_result(payload)


def _handle_update_text(arguments: Dict[str, Any]) -> Dict[str, Any]:
    debug_full = bool(arguments.get("debug_full", False))
    text, normalized_doc = resolve_text_for_extraction(arguments)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("'text' must be a non-empty string")

    result = update_document_text(
        text.strip(),
        base_document=normalized_doc,
        llama_cpp_url=arguments.get("llama_cpp_url")
        or arguments.get("llama_url")
        or os.environ.get("SECRETARIUS_LLAMA_CPP_URL")
        or os.environ.get("SECRETARIUS_LLAMA_URL")
        or "http://127.0.0.1:8989/v1/chat/completions",
        llama_cpp_model=arguments.get("llama_cpp_model")
        or arguments.get("model")
        or os.environ.get("SECRETARIUS_LLAMA_CPP_MODEL")
        or os.environ.get("SECRETARIUS_LLAMA_MODEL", "local-llama-cpp"),
        timeout_s=arguments.get("timeout_s", float(os.environ.get("SECRETARIUS_TIMEOUT_S", "30.0"))),
        max_tokens=arguments.get("max_tokens", int(os.environ.get("SECRETARIUS_MAX_TOKENS", "20480"))),
        seed=arguments.get("seed", int(os.environ.get("SECRETARIUS_SEED", "42"))),
        prompt_path=arguments.get("prompt_path") or os.environ.get("SECRETARIUS_PROMPT_PATH"),
        debug_return_raw=bool(arguments.get("debug_return_raw", False)),
        embedding_model=arguments.get("model"),
        normalize_embeddings=arguments.get("normalize", True),
        batch_size=arguments.get("batch_size", 32),
        milvus_uri=arguments.get("milvus_uri") or os.environ.get("SECRETARIUS_MILVUS_URI", "http://127.0.0.1:19530"),
        milvus_token=arguments.get("milvus_token") or os.environ.get("SECRETARIUS_MILVUS_TOKEN"),
        collection_name=arguments.get("collection_name")
        or os.environ.get("SECRETARIUS_MILVUS_COLLECTION", "secretarius_semantic_graph"),
        metric_type=arguments.get("metric_type") or os.environ.get("SECRETARIUS_MILVUS_METRIC", "COSINE"),
        top_k=_resolve_top_k_argument(arguments),
    )
    extract_payload = result.get("extract", {})
    graph_payload = result.get("index", {})
    extract_expressions = extract_payload.get("expressions")
    extract_count = len(extract_expressions) if isinstance(extract_expressions, list) else 0
    index_hits = graph_payload.get("hits")
    hit_lists = len(index_hits) if isinstance(index_hits, list) else 0
    warning = extract_payload.get("warning") or graph_payload.get("warning")
    payload = {
        "status": "ok",
        "tool": "update_text",
        "message": _TRANSLATOR.get("messages.update_complete"),
        "summary": {
            "expressions_count": extract_count,
            "collection_name": graph_payload.get("collection_name"),
            "deleted_count": graph_payload.get("deleted_count", 0),
            "inserted_count": graph_payload.get("inserted_count", 0),
            "query_count": graph_payload.get("query_count", 0),
            "hit_lists": hit_lists,
        },
        "warning": warning,
    }
    if debug_full:
        payload["extract"] = extract_payload
        payload["index"] = graph_payload
    return _tool_result(payload)


def _extract_search_documents(
    hits: Any,
    *,
    query_expressions: list[str] | None = None,
    query_keywords: list[str] | None = None,
    query_terms: list[str] | None = None,
    min_score: float | None = None,
) -> list[Dict[str, Any]]:
    if not isinstance(hits, list):
        return []

    query_keyword_set = {value.lower() for value in (query_keywords or []) if isinstance(value, str) and value.strip()}
    query_term_set = {value.lower() for value in (query_terms or []) if isinstance(value, str) and value.strip()}
    query_expression_list = [value.strip() for value in (query_expressions or []) if isinstance(value, str) and value.strip()]
    by_doc_id: dict[str, Dict[str, Any]] = {}
    anonymous_documents: list[Dict[str, Any]] = []
    for query_idx, hit_list in enumerate(hits):
        if not isinstance(hit_list, list):
            continue
        query_expression = query_expression_list[query_idx] if query_idx < len(query_expression_list) else None
        for hit in hit_list:
            if not isinstance(hit, dict):
                continue
            entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else hit
            payload_json = entity.get("payload_json")
            document = None
            if isinstance(payload_json, str):
                try:
                    document = json.loads(payload_json)
                except json.JSONDecodeError:
                    document = payload_json
            doc_id = document.get("doc_id") if isinstance(document, dict) else None
            semantic_score = _coerce_score(hit.get("score", entity.get("score", hit.get("distance", entity.get("distance")))))
            keyword_matches = _keyword_matches(document, query_keyword_set)
            title_matches = _title_matches(document, query_term_set)
            line_id = hit.get("id", entity.get("id"))
            document_expression = _extract_source_expression(document)
            match = _build_match_record(
                query_expression=query_expression,
                document_expression=document_expression,
                score=semantic_score,
                min_score=min_score,
            )
            if not isinstance(doc_id, str) or not doc_id:
                anonymous_documents.append(
                    _build_compact_search_document(
                        document=document,
                        matches=[match] if match is not None else [],
                        best_vector_score=semantic_score,
                        keyword_matches=keyword_matches,
                        title_matches=title_matches,
                        query_expression_count=len(query_expression_list),
                        query_keyword_count=len(query_keyword_set),
                    )
                )
                continue
            aggregate = by_doc_id.get(doc_id)
            if aggregate is None:
                aggregate = {
                    "document": document,
                    "best_vector_score": semantic_score,
                    "keyword_matches": set(keyword_matches),
                    "title_matches": title_matches,
                    "matches": {},
                }
                by_doc_id[doc_id] = aggregate
            else:
                if semantic_score > aggregate["best_vector_score"]:
                    aggregate["best_vector_score"] = semantic_score
                aggregate["title_matches"] = aggregate["title_matches"] or title_matches
                aggregate["keyword_matches"].update(keyword_matches)
            if match is not None:
                match_key = (match.get("query_expression"), match.get("document_expression"))
                previous_match = aggregate["matches"].get(match_key)
                if previous_match is None or match["score"] > previous_match["score"]:
                    aggregate["matches"][match_key] = match

    documents = [
        _build_compact_search_document(
            document=aggregate["document"],
            matches=list(aggregate["matches"].values()),
            best_vector_score=aggregate["best_vector_score"],
            keyword_matches=sorted(aggregate["keyword_matches"]),
            title_matches=aggregate["title_matches"],
            query_expression_count=len(query_expression_list),
            query_keyword_count=len(query_keyword_set),
        )
        for aggregate in by_doc_id.values()
    ] + anonymous_documents
    documents.sort(
        key=lambda item: (
            -item.get("global_score", 0.0),
            -item.get("best_vector_score", 0.0),
            str(item.get("doc_id") or ""),
        )
    )
    return documents


def _extract_query_expressions(extract_payload: Any) -> list[str]:
    if not isinstance(extract_payload, dict):
        return []
    expressions = extract_payload.get("expressions")
    if not isinstance(expressions, list):
        return []
    out: list[str] = []
    for value in expressions:
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return out


def _coerce_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_source_expression(document: Any) -> str | None:
    if not isinstance(document, dict):
        return None
    indexing = document.get("indexing")
    if not isinstance(indexing, dict):
        return None
    value = indexing.get("source_expression")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _build_match_record(
    *,
    query_expression: str | None,
    document_expression: str | None,
    score: float,
    min_score: float | None,
) -> Dict[str, Any] | None:
    if min_score is not None and score < min_score:
        return None
    match: Dict[str, Any] = {"score": score}
    if isinstance(query_expression, str) and query_expression.strip():
        match["query_expression"] = query_expression.strip()
    if isinstance(document_expression, str) and document_expression.strip():
        match["document_expression"] = document_expression.strip()
    return match


def _build_compact_search_document(
    *,
    document: Any,
    matches: list[Dict[str, Any]],
    best_vector_score: float,
    keyword_matches: list[str],
    title_matches: bool,
    query_expression_count: int,
    query_keyword_count: int,
) -> Dict[str, Any]:
    normalized_document = document if isinstance(document, dict) else {}
    user_fields = normalized_document.get("user_fields") if isinstance(normalized_document.get("user_fields"), dict) else {}
    source = normalized_document.get("source") if isinstance(normalized_document.get("source"), dict) else {}
    content = normalized_document.get("content") if isinstance(normalized_document.get("content"), dict) else {}
    expressions = expressions_from_document(normalized_document)
    best_match = max(matches, key=lambda item: item.get("score", 0.0), default=None)
    avg_match_score = (
        sum(_coerce_score(match.get("score")) for match in matches) / len(matches)
        if matches
        else 0.0
    )
    matched_query_expressions = {
        match.get("query_expression")
        for match in matches
        if isinstance(match.get("query_expression"), str) and match.get("query_expression")
    }
    query_coverage = (
        len(matched_query_expressions) / query_expression_count
        if query_expression_count > 0
        else 0.0
    )
    keyword_overlap = (
        len(keyword_matches) / query_keyword_count
        if keyword_matches and query_keyword_count > 0
        else 0.0
    )
    title_bonus = _TITLE_MATCH_BONUS if title_matches else 0.0
    global_score = (
        0.50 * best_vector_score
        + 0.20 * avg_match_score
        + 0.20 * query_coverage
        + 0.10 * keyword_overlap
        + title_bonus
    )

    compact_document: Dict[str, Any] = {
        "doc_id": normalized_document.get("doc_id"),
        "title": user_fields.get("title"),
        "type": normalized_document.get("type"),
        "document_date": user_fields.get("document_date"),
        "url": _extract_primary_url(source),
        "text": content.get("text"),
        "keywords": _extract_document_keywords(normalized_document),
        "expressions": expressions,
        "best_match": best_match,
        "matches": sorted(matches, key=lambda item: item.get("score", 0.0), reverse=True),
        "best_vector_score": best_vector_score,
        "global_score": global_score,
    }
    return compact_document


def _extract_primary_url(source: dict[str, Any]) -> str | None:
    url = source.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    urls = source.get("urls")
    if isinstance(urls, list):
        for value in urls:
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_document_keywords(document: Any) -> list[str]:
    if not isinstance(document, dict):
        return []
    user_fields = document.get("user_fields")
    if not isinstance(user_fields, dict):
        return []
    keywords = user_fields.get("keywords")
    if not isinstance(keywords, list):
        return []
    out: list[str] = []
    for value in keywords:
        if isinstance(value, str) and value.strip():
            normalized = value.strip()
            if normalized not in out:
                out.append(normalized)
    return out


def _extract_query_terms(query: str) -> list[str]:
    out: list[str] = []
    for raw in query.split():
        term = raw.strip(" \t\r\n,;:.!?()[]{}\"'").lower()
        if len(term) >= 3 and not term.startswith("#") and term not in out:
            out.append(term)
    return out


def _keyword_matches(document: Any, query_keyword_set: set[str]) -> list[str]:
    if not query_keyword_set or not isinstance(document, dict):
        return []
    document_keywords = {value.lower() for value in _extract_document_keywords(document)}
    return sorted(document_keywords & query_keyword_set)


def _title_matches(document: Any, query_term_set: set[str]) -> bool:
    if not query_term_set or not isinstance(document, dict):
        return False
    user_fields = document.get("user_fields")
    if not isinstance(user_fields, dict):
        return False
    title = user_fields.get("title")
    if not isinstance(title, str) or not title.strip():
        return False
    title_lower = title.lower()
    return any(term in title_lower for term in query_term_set)


def _handle_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    handlers = {
        "ask_oracle": _handle_ask_oracle,
        "extract_expressions": _handle_extract_expressions,
        "semantic_graph_search": _handle_semantic_graph_search,
        "index_text": _handle_index_text,
        "search_text": _handle_search_text,
        "update_text": _handle_update_text,
    }
    handler = handlers.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    return handler(arguments)


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
            for t in _public_tools_catalog()
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


def main() -> None:
    run_stdio_server()


def start_background_warmup() -> None:
    global _WARMUP_STARTED
    if _WARMUP_STARTED:
        return
    enabled = os.environ.get("SECRETARIUS_WARMUP_ON_START", "true").strip().lower()
    if enabled not in ("1", "true", "yes", "on"):
        return
    _WARMUP_STARTED = True

    def _worker() -> None:
        # Warmup chunker path via extraction call with tiny timeout.
        try:
            extract_expressions(
                "warmup",
                timeout_s=1.0,
                max_tokens=8,
                seed=42,
            )
        except Exception:
            pass

        # Warmup embedding model cache.
        try:
            embed_expressions_multilingual(["warmup"], batch_size=1)
        except Exception:
            pass

    thread = threading.Thread(target=_worker, name="secretarius-warmup", daemon=True)
    thread.start()


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


def _read_optional_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _read_optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _resolve_top_k_argument(arguments: Dict[str, Any]) -> Any:
    if "top_k" in arguments:
        return arguments.get("top_k")
    env_top_k = _read_optional_int_env("SECRETARIUS_MILVUS_TOP_K")
    if env_top_k is not None and env_top_k >= 1:
        return env_top_k
    return 10


if __name__ == "__main__":
    main()
