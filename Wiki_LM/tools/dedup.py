"""
Détection et nettoyage de doublons sémantiques dans le wiki via embeddings BGE-M3.

Usage :
    # Lister les doublons
    python dedup.py [--prefix src-] [--threshold 0.92] [--top-n 50]

    # Nettoyer (dry-run)
    python dedup.py --prefix src- --threshold 0.92 --clean

    # Nettoyer (effectif)
    python dedup.py --prefix src- --threshold 0.92 --clean --apply
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import frontmatter
import numpy as np

EMBED_DIR = Path(__file__).resolve().parent.parent / "embeddings"
INDEX_FILE = "embeddings_index.json"
MATRIX_FILE = "embeddings.npy"
_MANIFEST = ".ingested"
_DEFAULT_RAW = Path.home() / "Secretarius" / "Wiki_LM" / "raw"
_BAD_STATUS = {"illisible", "inaccessible"}


# ---------------------------------------------------------------------------
# Chargement embeddings
# ---------------------------------------------------------------------------

def _load_embeddings(embed_dir: Path) -> tuple[np.ndarray, list[str]]:
    matrix = np.load(embed_dir / MATRIX_FILE)
    with open(embed_dir / INDEX_FILE, encoding="utf-8") as f:
        idx = json.load(f)
    return matrix, idx["slugs"]


# ---------------------------------------------------------------------------
# Composantes connexes (Union-Find)
# ---------------------------------------------------------------------------

def _connected_components(pairs: list[tuple[str, str]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent.get(x, x)
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    groups: dict[str, set[str]] = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Sélection du canonique
# ---------------------------------------------------------------------------

def _select_canonical(slugs: set[str], wiki_dir: Path) -> str:
    """
    Priorité décroissante :
    1. Statut ni illisible ni inaccessible
    2. lien_source non vide
    3. Corps le plus long
    4. Created le plus récent
    5. Ordre alphabétique (déterministe)
    """
    def _score(slug: str) -> tuple:
        path = wiki_dir / f"{slug}.md"
        try:
            post = frontmatter.load(path)
            bad = 1 if str(post.get("status", "")) in _BAD_STATUS else 0
            no_url = 0 if post.get("lien_source") else 1
            body_len = -len(post.content)
            created = str(post.get("created", ""))
            return (bad, no_url, body_len, created, slug)
        except Exception:
            return (1, 1, 0, "", slug)

    return min(slugs, key=_score)


# ---------------------------------------------------------------------------
# Manifeste raw/
# ---------------------------------------------------------------------------

def _load_manifest(raw_dir: Path) -> dict[str, dict]:
    path = raw_dir / _MANIFEST
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        result[parts[0]] = {
            "slug": parts[1] if len(parts) > 1 else "",
            "hash": parts[2] if len(parts) > 2 else "",
        }
    return result


def _save_manifest(raw_dir: Path, manifest: dict[str, dict]) -> None:
    lines = [
        f"{fn}\t{meta['slug']}\t{meta['hash']}"
        for fn, meta in sorted(manifest.items())
    ]
    (raw_dir / _MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _raw_files_for_slug(slug: str, manifest: dict[str, dict], raw_dir: Path) -> list[Path]:
    return [
        raw_dir / fn
        for fn, meta in manifest.items()
        if meta.get("slug") == slug and (raw_dir / fn).exists()
    ]


# ---------------------------------------------------------------------------
# Index inversé : src-slug → pages c-/e- qui le référencent
# ---------------------------------------------------------------------------

def _build_sources_index(wiki_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for page in sorted(wiki_dir.glob("*.md")):
        if not (page.stem.startswith("c-") or page.stem.startswith("e-")):
            continue
        try:
            post = frontmatter.load(page)
        except Exception:
            continue
        if post.get("status") == "immuable":
            continue
        sources = post.get("sources", []) or []
        if not isinstance(sources, list):
            sources = [sources]
        for src in sources:
            index.setdefault(str(src), []).append(page)
    return index


# ---------------------------------------------------------------------------
# Mode --clean
# ---------------------------------------------------------------------------

def _clean(
    components: list[set[str]],
    wiki_dir: Path,
    raw_dir: Path,
    apply: bool,
) -> None:
    manifest = _load_manifest(raw_dir)
    print("[dedup] Construction de l'index des sources c-/e-...")
    sources_index = _build_sources_index(wiki_dir)

    total_wiki = 0
    total_raw = 0
    total_ce = 0

    for comp in sorted(components, key=lambda s: min(s)):
        canonical = _select_canonical(comp, wiki_dir)
        to_delete = sorted(comp - {canonical})

        print(f"\n  canonique : {canonical}")
        for slug in to_delete:
            print(f"  doublon   : {slug}")

            # Suppression page wiki
            wiki_page = wiki_dir / f"{slug}.md"
            if wiki_page.exists():
                if apply:
                    wiki_page.unlink()
                total_wiki += 1

            # Suppression fichiers raw + entrées manifeste
            raw_files = _raw_files_for_slug(slug, manifest, raw_dir)
            for rf in raw_files:
                print(f"    raw : {rf.name}")
                if apply:
                    rf.unlink()
                total_raw += 1
            if apply:
                keys_to_remove = [fn for fn, m in manifest.items() if m.get("slug") == slug]
                for k in keys_to_remove:
                    del manifest[k]

            # Mise à jour pages c-/e-
            for page in sources_index.get(slug, []):
                if not page.exists():
                    continue
                try:
                    post = frontmatter.load(page)
                except Exception:
                    continue
                sources = post.get("sources", []) or []
                if not isinstance(sources, list):
                    sources = [sources]
                new_sources = [s for s in sources if s != slug]
                if canonical not in new_sources:
                    new_sources.append(canonical)
                if apply:
                    post["sources"] = new_sources
                    page.write_text(frontmatter.dumps(post), encoding="utf-8")
                total_ce += 1

    if apply:
        _save_manifest(raw_dir, manifest)
        print(
            f"\n[dedup] Nettoyage effectué : {total_wiki} pages wiki supprimées, "
            f"{total_raw} fichiers raw supprimés, {total_ce} pages c-/e- mises à jour."
        )
    else:
        print(
            f"\n[dedup] Dry-run : {total_wiki} pages wiki, {total_raw} fichiers raw à supprimer, "
            f"{total_ce} pages c-/e- à mettre à jour. Relancer avec --apply pour exécuter."
        )


# ---------------------------------------------------------------------------
# Mode listing (défaut)
# ---------------------------------------------------------------------------

def _find_pairs(
    matrix: np.ndarray,
    all_slugs: list[str],
    prefix: str,
    threshold: float,
    top_n: int,
) -> list[tuple[str, str]]:
    if prefix == "all":
        indices = list(range(len(all_slugs)))
    else:
        indices = [i for i, s in enumerate(all_slugs) if s.startswith(prefix)]

    if len(indices) < 2:
        print(f"[dedup] Moins de 2 pages avec le préfixe '{prefix}'.")
        return []

    sub_slugs = [all_slugs[i] for i in indices]
    sub_matrix = matrix[indices]

    print(f"[dedup] {len(indices)} pages (préfixe='{prefix}'), seuil={threshold}")

    sim: np.ndarray = sub_matrix @ sub_matrix.T
    np.fill_diagonal(sim, 0.0)

    rows, cols = np.where(sim >= threshold)
    tri = rows < cols
    rows, cols = rows[tri], cols[tri]

    if len(rows) == 0:
        print(f"[dedup] Aucune paire avec similarité >= {threshold}")
        return []

    scores = sim[rows, cols]
    order = np.argsort(-scores)

    if top_n > 0:
        print(f"\n{'Sim':>6}  {'Slug A':<55}  Slug B")
        print("-" * 120)
        for k in order[:top_n]:
            a, b, s = sub_slugs[rows[k]], sub_slugs[cols[k]], float(scores[k])
            print(f"{s:6.3f}  {a:<55}  {b}")

    total = len(rows)
    shown = min(total, top_n) if top_n > 0 else 0
    print(f"\n{total} paires au-dessus du seuil" + (f", {shown} affichées." if top_n > 0 else "."))

    return [(sub_slugs[rows[k]], sub_slugs[cols[k]]) for k in order]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Détection et nettoyage de doublons sémantiques")
    parser.add_argument("--embed-dir", default=str(EMBED_DIR))
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM")),
    )
    parser.add_argument(
        "--raw",
        default=os.environ.get("WIKI_RAW_PATH", str(_DEFAULT_RAW)),
    )
    parser.add_argument(
        "--prefix",
        default="src-",
        help="Préfixe : src-, c-, e-, all (défaut : src-)",
    )
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Paires à afficher en mode listing (0 = aucune, -1 = toutes)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Nettoyer les doublons (dry-run par défaut)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Avec --clean : exécuter les suppressions (sans ça : dry-run)",
    )
    args = parser.parse_args()

    embed_dir = Path(args.embed_dir)
    if not (embed_dir / MATRIX_FILE).exists():
        print(f"[dedup] Embeddings introuvables dans {embed_dir}. Lancez d'abord embed.py.")
        return

    matrix, all_slugs = _load_embeddings(embed_dir)

    top_n = 0 if args.clean else args.top_n
    pairs = _find_pairs(matrix, all_slugs, args.prefix, args.threshold, top_n)

    if not pairs or not args.clean:
        return

    wiki_dir = Path(args.wiki) / "wiki"
    raw_dir = Path(args.raw)

    components = _connected_components(pairs)
    multi = [c for c in components if len(c) > 1]
    print(f"\n[dedup] {len(multi)} composante(s) connexe(s) de doublons")

    _clean(multi, wiki_dir, raw_dir, apply=args.apply)


if __name__ == "__main__":
    main()
