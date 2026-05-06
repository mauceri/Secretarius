# tools/kb_tags.py
"""
Construction du dictionnaire de tags normalisés par similarité sémantique.

Usage:
    python tools/kb_tags.py \\
        --wiki ~/Documents/Arbath/Wiki_LM/wiki_signets_05_2026 \\
        [--kb-dir ~/Secretarius/Wiki_LM/knowledge_base] \\
        [--threshold 0.90] \\
        [--min-count 2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import frontmatter
import numpy as np

_DEFAULT_KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


def collect_tags(wiki_root: Path) -> dict[str, int]:
    """Retourne {tag: count} depuis toutes les pages sources du wiki."""
    counts: dict[str, int] = {}
    sources_dir = wiki_root / "wiki" / "sources"
    if not sources_dir.exists():
        return counts
    for path in sorted(sources_dir.glob("src-*.md")):
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        for tag in post.get("tags", []):
            t = str(tag)
            counts[t] = counts.get(t, 0) + 1
    return counts


def build_tag_groups(
    tags: dict[str, int],
    vecs: dict[str, np.ndarray],
    threshold: float = 0.90,
    min_count: int = 2,
) -> dict[str, list[str]]:
    """
    Regroupe les tags synonymes par similarité cosinus.

    tags      : {tag: count}
    vecs      : {tag: vecteur L2-normalisé}
    threshold : similarité cosinus minimum pour fusionner deux tags
    min_count : count minimum pour conserver un tag

    Retourne {canonical: [variant, ...]}.
    Le canonique est le tag avec le count le plus élevé du groupe.
    """
    filtered = {t: c for t, c in tags.items() if c >= min_count and t in vecs}
    if not filtered:
        return {}

    sorted_tags = sorted(filtered.keys(), key=lambda t: -filtered[t])
    assigned: dict[str, str] = {}

    for tag in sorted_tags:
        if tag in assigned:
            continue
        assigned[tag] = tag
        v = vecs[tag]
        for other in sorted_tags:
            if other in assigned:
                continue
            sim = float(np.dot(v, vecs[other]))
            if sim >= threshold:
                assigned[other] = tag

    groups: dict[str, list[str]] = {}
    for tag, canon in assigned.items():
        groups.setdefault(canon, []).append(tag)
    return groups


def save_tag_dict(
    kb_dir: Path,
    groups: dict[str, list[str]],
    vecs: dict[str, np.ndarray],
) -> None:
    """
    Écrit tags_dict.json et tags_embeddings.npy dans kb_dir/tags/.
    tags_dict.json : {canonical: [variants...]}
    tags_embeddings.npy : (K, dim) — vecteurs des canoniques
    """
    (kb_dir / "tags").mkdir(parents=True, exist_ok=True)

    canonicals = sorted(groups.keys())
    mat_rows = [vecs[c].astype(np.float32) for c in canonicals if c in vecs]

    if mat_rows:
        np.save(kb_dir / "tags" / "tags_embeddings.npy", np.array(mat_rows))

    (kb_dir / "tags" / "tags_dict.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit le dictionnaire de tags normalisés")
    parser.add_argument("--wiki", required=True)
    parser.add_argument("--kb-dir", default=str(_DEFAULT_KB_DIR))
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--min-count", type=int, default=2)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    wiki_root = Path(args.wiki)
    kb_dir = Path(args.kb_dir)

    tags = collect_tags(wiki_root)
    print(f"[kb_tags] {len(tags)} tags trouvés")

    tag_list = [t for t, c in tags.items() if c >= args.min_count]
    print(f"[kb_tags] {len(tag_list)} tags avec count >= {args.min_count}")

    model = SentenceTransformer("BAAI/bge-m3")
    raw_vecs = model.encode(tag_list, normalize_embeddings=True, show_progress_bar=True)
    vecs = {t: raw_vecs[i].astype(np.float32) for i, t in enumerate(tag_list)}

    groups = build_tag_groups(tags, vecs, threshold=args.threshold, min_count=args.min_count)
    save_tag_dict(kb_dir, groups, vecs)
    print(f"[kb_tags] {len(groups)} groupes → {kb_dir / 'tags'}")


if __name__ == "__main__":
    main()
