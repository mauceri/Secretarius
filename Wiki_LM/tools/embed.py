"""
Calcule et persiste les embeddings BGE-M3 pour toutes les pages du wiki.

Usage:
    python embed.py [--wiki PATH] [--force] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import frontmatter
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
EMBED_DIR = Path(__file__).resolve().parent.parent / "embeddings"
INDEX_FILE = "embeddings_index.json"
MATRIX_FILE = "embeddings.npy"

SKIP_NAMES = {"index.md", "log.md"}


def _extract_text(post: frontmatter.Post) -> str:
    title = str(post.get("title", ""))
    body = post.content
    m = re.search(r"## Résumé\n(.+?)(?=\n## |\Z)", body, re.DOTALL)
    abstract = m.group(1).strip() if m else body[:500].strip()
    return f"{title}\n\n{abstract}"


def load_pages(wiki_dir: Path) -> list[dict]:
    pages = []
    for path in sorted(wiki_dir.glob("*.md")):
        if path.name in SKIP_NAMES:
            continue
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        pages.append({"slug": path.stem, "text": _extract_text(post)})
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcule les embeddings BGE-M3 du wiki")
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM")),
    )
    parser.add_argument("--force", action="store_true", help="Recalculer tous les embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dir", default=str(EMBED_DIR))
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    if not wiki_dir.exists():
        raise FileNotFoundError(f"Répertoire wiki introuvable : {wiki_dir}")

    embed_dir = Path(args.embed_dir)
    embed_dir.mkdir(parents=True, exist_ok=True)
    index_path = embed_dir / INDEX_FILE
    matrix_path = embed_dir / MATRIX_FILE

    existing_slugs: list[str] = []
    existing_matrix: np.ndarray | None = None

    if not args.force and index_path.exists() and matrix_path.exists():
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)
        existing_slugs = idx.get("slugs", [])
        existing_matrix = np.load(matrix_path)
        print(f"[embed] Index existant : {len(existing_slugs)} pages")

    all_pages = load_pages(wiki_dir)
    existing_set = set(existing_slugs)
    new_pages = [p for p in all_pages if p["slug"] not in existing_set]

    if not new_pages:
        print("[embed] Aucune nouvelle page à traiter.")
        return

    print(f"[embed] {len(new_pages)} pages à encoder (sur {len(all_pages)} au total)...")

    model = SentenceTransformer(MODEL_NAME)
    texts = [p["text"] for p in new_pages]
    new_matrix = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    if existing_matrix is not None and existing_slugs:
        final_matrix = np.vstack([existing_matrix, new_matrix])
        final_slugs = existing_slugs + [p["slug"] for p in new_pages]
    else:
        final_matrix = new_matrix
        final_slugs = [p["slug"] for p in new_pages]

    np.save(matrix_path, final_matrix)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"slugs": final_slugs, "model": MODEL_NAME}, f, ensure_ascii=False, indent=2)

    print(f"[embed] Terminé : {len(final_slugs)} pages, shape {final_matrix.shape}")


if __name__ == "__main__":
    main()
