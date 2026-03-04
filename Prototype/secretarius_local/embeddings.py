from __future__ import annotations

import os
from pathlib import Path
from typing import Any

HF_CACHE_MODEL_ROOT = Path(
    "/home/mauceric/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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
    local_model_path = _detect_local_model_path()
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


def _detect_local_model_path() -> Path | None:
    snapshots_dir = HF_CACHE_MODEL_ROOT / "snapshots"
    if not snapshots_dir.exists():
        return None
    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
