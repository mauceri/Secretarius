from __future__ import annotations

import json
import importlib.util
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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


JSONRPC_VERSION = "2.0"
SERVER_NAME = "secretarius-mcp"
SERVER_VERSION = "0.1.0"
_LAST_INPUT_MODE = "framed"
_WARMUP_STARTED = False


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
                "Analyse un texte brut ou un document fourni par l'utilisateur "
                "et en extrait des expressions caracteristiques via llama.cpp. "
                "A utiliser pour un contenu a analyser, pas pour rechercher dans "
                "Milvus ni pour lancer une recherche semantique."
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
                        "description": "Document Secretarius v0.1 (ou document attache).",
                        "properties": {
                            "schema": {"type": "string"},
                            "doc_id": {"type": ["string", "null"]},
                            "type": {"type": "string"},
                            "name": {"type": "string"},
                            "mime_type": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "additionalProperties": True,
                    },
                    "llama_cpp_url": {
                        "type": "string",
                        "description": "Endpoint chat completions llama.cpp.",
                    },
                    "llama_url": {
                        "type": "string",
                        "description": "Alias de compatibilite (deprecated) pour 'llama_cpp_url'.",
                    },
                    "llama_cpp_model": {
                        "type": "string",
                        "description": "Nom logique du modele llama.cpp.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Alias de compatibilite (deprecated) pour 'llama_cpp_model'.",
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
                        "description": "Chemin vers le prompt texte (par defaut secretarius/prompts/prompt_extracteur.txt).",
                    },
                    "debug_return_raw": {
                        "type": "boolean",
                        "default": False,
                        "description": "Inclure la sortie brute du modele pour debug.",
                    },
                    "per_chunk_llm": {
                        "type": "boolean",
                        "default": False,
                        "description": "Si true, appelle le LLM par chunk. Par defaut, un seul appel global.",
                    },
                },
                "anyOf": [{"required": ["text"]}, {"required": ["document"]}],
                "additionalProperties": True,
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
                "Pipeline metier d'indexation: prend un texte ou un document "
                "fourni par l'utilisateur, extrait les expressions puis les "
                "insere dans Milvus."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Texte brut a indexer.",
                    },
                    "document": {
                        "type": "object",
                        "description": "Document Secretarius v0.1.",
                        "additionalProperties": True,
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Collection cible Milvus.",
                    },
                    "milvus_uri": {
                        "type": "string",
                        "description": "URI Milvus.",
                    },
                    "milvus_token": {
                        "type": "string",
                        "description": "Token Milvus optionnel.",
                    },
                    "metric_type": {
                        "type": "string",
                        "description": "Metrique d'index (COSINE, IP, L2).",
                    },
                    "debug_full": {
                        "type": "boolean",
                        "default": False,
                        "description": "Retourner le detail complet extract/index (payload volumineux).",
                    },
                },
                "anyOf": [{"required": ["text"]}, {"required": ["document"]}],
                "additionalProperties": True,
            },
        ),
        MCPTool(
            name="search_text",
            description=(
                "Pipeline metier de recherche semantique dans Milvus a partir "
                "d'une requete textuelle courte. A utiliser pour chercher des "
                "resultats, pas pour analyser un texte fourni par l'utilisateur."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requete textuelle a rechercher.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 10,
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Collection cible Milvus.",
                    },
                    "milvus_uri": {
                        "type": "string",
                        "description": "URI Milvus.",
                    },
                    "milvus_token": {
                        "type": "string",
                        "description": "Token Milvus optionnel.",
                    },
                    "metric_type": {
                        "type": "string",
                        "description": "Metrique d'index (COSINE, IP, L2).",
                    },
                    "debug_full": {
                        "type": "boolean",
                        "default": False,
                        "description": "Retourner le detail complet de la recherche (payload volumineux).",
                    },
                },
                "required": ["query"],
                "additionalProperties": True,
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


def _analyse_texte_documentaire_index_text(text: str) -> dict[str, Any]:
    hashtag_re = re.compile(r"(?<!\w)#([^\s#]+)")
    date_re = re.compile(r"\b\d{2}-\d{2}-\d{4}\b")
    url_re = re.compile(r'\bhttps?://[^\s<>\"]+')

    def normalize(raw: str) -> str:
        cleaned = raw.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def remove_spans(raw: str, spans: list[tuple[int, int]]) -> str:
        if not spans:
            return raw
        merged: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        out: list[str] = []
        cursor = 0
        for start, end in merged:
            out.append(raw[cursor:start])
            cursor = end
        out.append(raw[cursor:])
        return "".join(out)

    def compute_incipit_title(body: str, max_len: int = 40) -> str | None:
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not lines:
            return None
        first = lines[0]
        if len(first) <= max_len:
            return first
        cut = first[:max_len]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        return cut.rstrip(" ,;:-") + "..."

    def detect_explicit_title(lines: list[str]) -> tuple[str | None, int | None]:
        for idx in range(len(lines) - 1):
            current = lines[idx].strip()
            if not current:
                continue
            if hashtag_re.fullmatch(current):
                continue
            if date_re.fullmatch(current):
                continue
            if url_re.fullmatch(current):
                continue

            next_line = ""
            for look_ahead in range(idx + 1, len(lines)):
                candidate = lines[look_ahead].strip()
                if not candidate:
                    continue
                next_line = candidate
                break
            if not next_line:
                continue
            if len(current) <= 120 and not current.endswith((".", "!", "?", ";")) and len(next_line) >= len(current):
                return current, idx
        return None, None

    normalized = normalize(text or "")
    spans_to_remove: list[tuple[int, int]] = []
    keywords: list[str] = []

    for match in hashtag_re.finditer(normalized):
        keywords.append(f"#{match.group(1)}")
        spans_to_remove.append((match.start(), match.end()))

    date_match = date_re.search(normalized)
    date_value = date_match.group(0) if date_match else None
    if date_match:
        spans_to_remove.append((date_match.start(), date_match.end()))

    url_match = url_re.search(normalized)
    url_value = url_match.group(0) if url_match else None
    if url_match:
        spans_to_remove.append((url_match.start(), url_match.end()))

    deduped_keywords: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        if keyword in seen:
            continue
        seen.add(keyword)
        deduped_keywords.append(keyword)

    cleaned = remove_spans(normalized, spans_to_remove)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip(" \n\t-:")

    lines = cleaned.splitlines()
    title, title_idx = detect_explicit_title(lines)
    if title is not None and title_idx is not None:
        body = "\n".join(line for idx, line in enumerate(lines) if idx != title_idx).strip()
    else:
        body = cleaned.strip()
        title = compute_incipit_title(body)

    return {
        "mots_clefs": deduped_keywords,
        "titre": title,
        "date": date_value,
        "url": url_value,
        "texte": normalize(body),
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
    max_tokens = arguments.get("max_tokens", int(os.environ.get("SECRETARIUS_MAX_TOKENS", "512")))
    seed = arguments.get("seed", int(os.environ.get("SECRETARIUS_SEED", "42")))
    prompt_path = arguments.get("prompt_path") or os.environ.get("SECRETARIUS_PROMPT_PATH")
    debug_env = os.environ.get("SECRETARIUS_DEBUG_RETURN_RAW", "false").strip().lower()
    debug_default = debug_env in ("1", "true", "yes", "on")
    debug_return_raw = arguments.get("debug_return_raw", debug_default)
    per_chunk_env = os.environ.get("SECRETARIUS_EXTRACT_PER_CHUNK", "false").strip().lower()
    per_chunk_default = per_chunk_env in ("1", "true", "yes", "on")
    per_chunk_llm = arguments.get("per_chunk_llm", per_chunk_default)

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
    if not isinstance(per_chunk_llm, bool):
        raise ValueError("'per_chunk_llm' must be a boolean")
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
        per_chunk_llm=per_chunk_llm,
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
    top_k = arguments.get("top_k", 10)
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
    extract_args = dict(arguments)
    text, normalized_doc = resolve_text_for_extraction(arguments)
    if text:
        analysed = _analyse_texte_documentaire_index_text(text)
        document_payload = normalized_doc if normalized_doc is not None else normalize_document({})
        content = document_payload.setdefault("content", {})
        if isinstance(content, dict):
            content["text"] = analysed.get("texte") or text
        source = document_payload.setdefault("source", {})
        if isinstance(source, dict) and isinstance(analysed.get("url"), str) and analysed["url"].strip():
            if not source.get("url"):
                source["url"] = analysed["url"].strip()
        user_fields = document_payload.setdefault("user_fields", {})
        if isinstance(user_fields, dict):
            if isinstance(analysed.get("titre"), str) and analysed["titre"].strip():
                if not user_fields.get("title"):
                    user_fields["title"] = analysed["titre"].strip()
            if isinstance(analysed.get("date"), str) and analysed["date"].strip():
                if not user_fields.get("document_date"):
                    user_fields["document_date"] = analysed["date"].strip()
            if isinstance(analysed.get("mots_clefs"), list) and analysed["mots_clefs"]:
                if not user_fields.get("keywords"):
                    user_fields["keywords"] = analysed["mots_clefs"]
        extract_args["document"] = normalize_document(document_payload)
        extract_args["text"] = (analysed.get("texte") or text).strip()

    # Reuse extraction pipeline, then index by expressions/doc.
    extract_res = _handle_extract_expressions(extract_args)
    extract_payload = extract_res.get("structuredContent", {}) if isinstance(extract_res, dict) else {}
    if not isinstance(extract_payload, dict):
        raise ValueError("extract_expressions returned invalid payload")

    expressions = extract_payload.get("expressions")
    normalized_doc = extract_payload.get("document")
    if not isinstance(expressions, list) or not expressions:
        raise ValueError("No expressions extracted; cannot index")

    graph_args: Dict[str, Any] = {
        "expressions": expressions,
        "upsert": True,
    }
    if isinstance(normalized_doc, dict):
        graph_args["document"] = normalized_doc
    for key in ("collection_name", "milvus_uri", "milvus_token", "metric_type", "model", "normalize", "batch_size"):
        if key in arguments:
            graph_args[key] = arguments[key]

    graph_res = _handle_semantic_graph_search(graph_args)
    graph_payload = graph_res.get("structuredContent", {}) if isinstance(graph_res, dict) else {}
    if not isinstance(graph_payload, dict):
        raise ValueError("semantic_graph_search returned invalid payload")

    extract_expressions = extract_payload.get("expressions")
    extract_count = len(extract_expressions) if isinstance(extract_expressions, list) else 0
    index_hits = graph_payload.get("hits")
    hit_lists = len(index_hits) if isinstance(index_hits, list) else 0
    warning = extract_payload.get("warning") or graph_payload.get("warning")
    payload = {
        "status": "ok",
        "tool": "index_text",
        "message": "Indexation complete (extraction + insertion semantic graph).",
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

    graph_args: Dict[str, Any] = {
        "expressions": [query.strip()],
        "upsert": False,
        "top_k": arguments.get("top_k", 10),
    }
    for key in ("collection_name", "milvus_uri", "milvus_token", "metric_type", "model", "normalize", "batch_size"):
        if key in arguments:
            graph_args[key] = arguments[key]

    graph_res = _handle_semantic_graph_search(graph_args)
    graph_payload = graph_res.get("structuredContent", {}) if isinstance(graph_res, dict) else {}
    if not isinstance(graph_payload, dict):
        raise ValueError("semantic_graph_search returned invalid payload")

    hits = graph_payload.get("hits")
    hit_lists = len(hits) if isinstance(hits, list) else 0
    documents = _extract_search_documents(hits)
    payload = {
        "status": "ok",
        "tool": "search_text",
        "message": "Recherche semantique executee.",
        "query": query.strip(),
        "summary": {
            "collection_name": graph_payload.get("collection_name"),
            "query_count": graph_payload.get("query_count", 0),
            "hit_lists": hit_lists,
            "top_k": graph_args.get("top_k", 10),
        },
        "documents": documents,
        "warning": graph_payload.get("warning"),
    }
    if debug_full:
        payload["search"] = graph_payload
    return _tool_result(payload)


def _extract_search_documents(hits: Any) -> list[Dict[str, Any]]:
    if not isinstance(hits, list):
        return []

    documents: list[Dict[str, Any]] = []
    for hit_list in hits:
        if not isinstance(hit_list, list):
            continue
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
            documents.append(
                {
                    "id": hit.get("id", entity.get("id")),
                    "score": hit.get("score", entity.get("score", hit.get("distance", entity.get("distance")))),
                    "document": document,
                }
            )
    return documents


def _handle_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    handlers = {
        "extract_expressions": _handle_extract_expressions,
        "semantic_graph_search": _handle_semantic_graph_search,
        "index_text": _handle_index_text,
        "search_text": _handle_search_text,
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
