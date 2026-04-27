"""
Détection de doublons sémantiques dans le wiki via embeddings BGE-M3.

Usage:
    python dedup.py [--prefix src-] [--threshold 0.92] [--top-n 50]
    python dedup.py --prefix c- --threshold 0.90 --top-n 100
    python dedup.py --prefix all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EMBED_DIR = Path(__file__).resolve().parent.parent / "embeddings"
INDEX_FILE = "embeddings_index.json"
MATRIX_FILE = "embeddings.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Détection de doublons sémantiques dans le wiki")
    parser.add_argument("--embed-dir", default=str(EMBED_DIR))
    parser.add_argument(
        "--prefix",
        default="src-",
        help="Préfixe des slugs à comparer : src-, c-, e-, all (défaut : src-)",
    )
    parser.add_argument("--threshold", type=float, default=0.92, help="Seuil de similarité (défaut : 0.92)")
    parser.add_argument("--top-n", type=int, default=50, help="Nombre de paires à afficher (défaut : 50)")
    args = parser.parse_args()

    embed_dir = Path(args.embed_dir)
    matrix_path = embed_dir / MATRIX_FILE
    index_path = embed_dir / INDEX_FILE

    if not matrix_path.exists() or not index_path.exists():
        print(f"[dedup] Embeddings introuvables dans {embed_dir}. Lancez d'abord embed.py.")
        return

    matrix = np.load(matrix_path)
    with open(index_path, encoding="utf-8") as f:
        idx = json.load(f)
    all_slugs: list[str] = idx["slugs"]

    # Filtrage par préfixe
    if args.prefix == "all":
        indices = list(range(len(all_slugs)))
    else:
        indices = [i for i, s in enumerate(all_slugs) if s.startswith(args.prefix)]

    if len(indices) < 2:
        print(f"[dedup] Moins de 2 pages avec le préfixe '{args.prefix}'.")
        return

    sub_slugs = [all_slugs[i] for i in indices]
    sub_matrix = matrix[indices]  # déjà normalisé → dot product = cosine

    print(f"[dedup] {len(indices)} pages (préfixe='{args.prefix}'), seuil={args.threshold}")

    # Matrice de similarité cosinus
    sim: np.ndarray = sub_matrix @ sub_matrix.T  # (N, N)
    np.fill_diagonal(sim, 0.0)

    # Paires au-dessus du seuil (triangle supérieur)
    rows, cols = np.where(sim >= args.threshold)
    tri_mask = rows < cols
    rows, cols = rows[tri_mask], cols[tri_mask]

    if len(rows) == 0:
        print(f"[dedup] Aucune paire avec similarité >= {args.threshold}")
        return

    scores = sim[rows, cols]
    order = np.argsort(-scores)[: args.top_n]

    print(f"\n{'Sim':>6}  {'Slug A':<55}  Slug B")
    print("-" * 120)
    for k in order:
        a, b, s = sub_slugs[rows[k]], sub_slugs[cols[k]], float(scores[k])
        print(f"{s:6.3f}  {a:<55}  {b}")

    total = len(rows)
    shown = min(total, args.top_n)
    print(f"\n{total} paires au-dessus du seuil, {shown} affichées.")


if __name__ == "__main__":
    main()
