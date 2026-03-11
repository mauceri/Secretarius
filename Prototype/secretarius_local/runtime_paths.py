from __future__ import annotations

import os
from pathlib import Path


SECRETARIUS_ROOT = Path(__file__).resolve().parent
DEFAULT_SENTENCE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_HF_MODEL_CACHE_DIRNAME = "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"


def resolve_nltk_data_path() -> Path | None:
    env_value = os.environ.get("NLTK_DATA", "").strip()
    if env_value:
        for raw_path in env_value.split(os.pathsep):
            candidate = Path(raw_path).expanduser()
            if candidate.exists():
                return candidate

    candidates = (
        Path.cwd() / ".nltk_data",
        SECRETARIUS_ROOT.parent / ".nltk_data",
        Path.home() / "nltk_data",
        Path.home() / ".local" / "share" / "nltk_data",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_huggingface_hub_root() -> Path:
    env_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if env_cache:
        return Path(env_cache).expanduser()

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_sentence_model_override() -> str | None:
    configured = os.environ.get("SECRETARIUS_SENTENCE_MODEL", "").strip()
    return configured or None


def resolve_sentence_model_path(model_cache_dirname: str = DEFAULT_HF_MODEL_CACHE_DIRNAME) -> Path | None:
    configured = resolve_sentence_model_override()
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            return configured_path

    snapshots_dir = resolve_huggingface_hub_root() / model_cache_dirname / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]
