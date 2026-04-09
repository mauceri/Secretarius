"""
BGE-M3 multi-vector encoding (ColBERT-style).

Chaque texte → liste de vecteurs token (1024 dim, normalisés L2).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_MODEL_DIR = Path(__file__).resolve().parents[1] / "Prototype" / "secretarius_local"
_ENCODER: Any = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    from sentence_transformers import SentenceTransformer
    if str(_MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(_MODEL_DIR))
    from runtime_paths import resolve_sentence_model_path  # type: ignore
    model_path = resolve_sentence_model_path()
    device = "cpu"  # CPU-only : compatible VPS sans GPU
    if model_path:
        _ENCODER = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    else:
        _ENCODER = SentenceTransformer("BAAI/bge-m3", device=device)
    return _ENCODER


def encode_multivector(texts: list[str]) -> list[np.ndarray]:
    """
    Encode une liste de textes en multi-vecteurs ColBERT.

    Retourne une liste de tableaux numpy (n_tokens, 1024), normalisés L2.
    """
    encoder = _get_encoder()
    results = encoder.encode(
        texts,
        output_value="token_embeddings",
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    # sentence-transformers peut retourner des tenseurs — on normalise
    out = []
    for vecs in results:
        arr = np.array(vecs, dtype=np.float32)
        # Re-normaliser ligne par ligne (robustesse)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        out.append(arr / norms)
    return out


def encode_dense(texts: list[str]) -> np.ndarray:
    """Encode en vecteur dense unique (fallback / RAG classique)."""
    encoder = _get_encoder()
    vecs = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(vecs, dtype=np.float32)
