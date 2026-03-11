from __future__ import annotations

import os
import importlib.util
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parent

try:
    from .runtime_paths import DEFAULT_SENTENCE_MODEL, resolve_sentence_model_path
except ImportError:
    _RUNTIME_PATHS = MODULE_ROOT / "runtime_paths.py"
    _RUNTIME_PATHS_SPEC = importlib.util.spec_from_file_location("secretarius_runtime_paths", _RUNTIME_PATHS)
    if _RUNTIME_PATHS_SPEC is None or _RUNTIME_PATHS_SPEC.loader is None:
        raise RuntimeError(f"unable to load runtime_paths from {_RUNTIME_PATHS}")
    _runtime_paths_module = importlib.util.module_from_spec(_RUNTIME_PATHS_SPEC)
    sys.modules[_RUNTIME_PATHS_SPEC.name] = _runtime_paths_module
    _RUNTIME_PATHS_SPEC.loader.exec_module(_runtime_paths_module)
    DEFAULT_SENTENCE_MODEL = _runtime_paths_module.DEFAULT_SENTENCE_MODEL
    resolve_sentence_model_path = _runtime_paths_module.resolve_sentence_model_path

DEFAULT_MODEL = DEFAULT_SENTENCE_MODEL

_CACHED_MODEL: Any | None = None
_CACHED_MODEL_NAME: str | None = None


def embed_expressions_multilingual(
    expressions: list[str],
    *,
    model: str | None = None,
    normalize: bool = True,
    batch_size: int = 32,
) -> dict[str, Any]:
    cleaned = [expr.strip() for expr in expressions if isinstance(expr, str) and expr.strip()]
    if not cleaned:
        return {
            "embeddings": [],
            "dimension": 0,
            "model": model or DEFAULT_MODEL,
            "normalized": normalize,
            "warning": "no non-empty expressions",
        }

    encoder, warning = _load_encoder(model)
    if encoder is None:
        return {
            "embeddings": [],
            "dimension": 0,
            "model": model or DEFAULT_MODEL,
            "normalized": normalize,
            "warning": warning or "embedding model unavailable",
        }

    try:
        vectors = encoder.encode(  # type: ignore[attr-defined]
            cleaned,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
    except Exception as exc:
        return {
            "embeddings": [],
            "dimension": 0,
            "model": model or DEFAULT_MODEL,
            "normalized": normalize,
            "warning": f"embedding encode failed: {exc}",
        }

    embeddings = vectors.tolist()
    dimension = len(embeddings[0]) if embeddings else 0
    return {
        "embeddings": embeddings,
        "dimension": dimension,
        "model": model or DEFAULT_MODEL,
        "normalized": normalize,
        "warning": None,
    }


def _load_encoder(model: str | None) -> tuple[Any | None, str | None]:
    global _CACHED_MODEL, _CACHED_MODEL_NAME
    model_name = model or DEFAULT_MODEL
    if _CACHED_MODEL is not None and _CACHED_MODEL_NAME == model_name:
        return _CACHED_MODEL, None

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        return None, f"sentence-transformers import failed: {exc}"

    kwargs: dict[str, Any] = {}
    local_model_path = resolve_sentence_model_path()
    if local_model_path is not None:
        kwargs["local_files_only"] = True
        model_name = str(local_model_path)
    elif os.environ.get("SECRETARIUS_LOCAL_FILES_ONLY", "1").strip().lower() in ("1", "true", "yes", "on"):
        kwargs["local_files_only"] = True

    try:
        encoder = SentenceTransformer(model_name_or_path=model_name, **kwargs)
    except Exception as exc:
        return None, f"unable to initialize sentence-transformers model: {exc}"

    _CACHED_MODEL = encoder
    _CACHED_MODEL_NAME = model or DEFAULT_MODEL
    return encoder, None
