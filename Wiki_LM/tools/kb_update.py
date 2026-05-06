# tools/kb_update.py
"""
Mise à jour de la base de connaissance à partir d'un wiki archivé.

Usage:
    python tools/kb_update.py \\
        --wiki ~/Documents/Arbath/Wiki_LM/wiki_signets_05_2026 \\
        --clustering clustering-embeddings-transfers-0.403 \\
        [--embed-dir ~/Secretarius/Wiki_LM/embeddings] \\
        [--kb-dir ~/Secretarius/Wiki_LM/knowledge_base] \\
        [--fusion-threshold 0.85] \\
        [--min-size 3]
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import frontmatter
import numpy as np

_DEFAULT_EMBED_DIR = Path(__file__).resolve().parent.parent / "embeddings"
_DEFAULT_KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
FUSION_THRESHOLD = 0.85
MIN_SIZE = 3


def _load_cluster_files(clustering_dir: Path) -> list[dict]:
    """
    Charge tous les cluster-*.md d'un répertoire de clustering.
    Retourne [{id, title, description, members, status}].
    """
    clusters = []
    for path in sorted(clustering_dir.glob("cluster-*.md")):
        try:
            post = frontmatter.load(path)
        except Exception:
            continue

        content = post.content

        m_title = re.search(r"^# (.+)$", content, re.MULTILINE)
        title = m_title.group(1).strip() if m_title else path.stem

        m_desc = re.search(r"^# .+\n\n(.+?)(?=\n\n|\Z)", content, re.DOTALL | re.MULTILINE)
        description = m_desc.group(1).strip() if m_desc else ""

        m_members = re.search(
            r"## Documents membres\n(.*?)(?=\n## |\Z)", content, re.DOTALL
        )
        member_slugs: list[str] = []
        if m_members:
            member_slugs = re.findall(r"\[\[([^\]]+)\]\]", m_members.group(1))

        clusters.append({
            "id": path.stem,
            "title": title,
            "description": description,
            "members": member_slugs,
            "status": str(post.get("status", "active")),
        })
    return clusters


def _collect_tags(member_slugs: list[str], sources_dir: Path, top_n: int = 10) -> list[str]:
    """Agrège les top_n tags les plus fréquents parmi les pages membres."""
    counts: dict[str, int] = {}
    for slug in member_slugs:
        path = sources_dir / f"{slug}.md"
        if not path.exists():
            continue
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        for tag in post.get("tags", []):
            counts[str(tag)] = counts.get(str(tag), 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])][:top_n]


def _load_embed_matrix(embed_dir: Path) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Charge la matrice d'embeddings et l'index des slugs."""
    index_path = embed_dir / "embeddings_index.json"
    matrix_path = embed_dir / "embeddings.npy"
    if not index_path.exists() or not matrix_path.exists():
        return None, None
    try:
        matrix = np.load(matrix_path)
        data = json.loads(index_path.read_text(encoding="utf-8"))
        slugs = data.get("slugs", [])
        if matrix.shape[0] != len(slugs):
            return None, None
        return matrix.astype(np.float32), slugs
    except Exception:
        return None, None


def _compute_centroid_and_cohesion(
    member_slugs: list[str],
    embed_matrix: np.ndarray,
    slug_to_idx: dict[str, int],
) -> tuple[np.ndarray, float] | tuple[None, float]:
    """
    Calcule le centroïde L2-normalisé et la cohésion (sim moyenne au centroïde).
    Retourne (None, 0.0) si aucun membre n'a d'embedding.
    """
    indices = [slug_to_idx[s] for s in member_slugs if s in slug_to_idx]
    if not indices:
        return None, 0.0
    vecs = embed_matrix[indices].astype(np.float32)
    centroid = vecs.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return None, 0.0
    centroid = centroid / norm
    cohesion = float(np.mean(vecs @ centroid))
    return centroid, cohesion



def _regenerate_index(kb_dir: Path) -> None:
    """Régénère knowledge_base/index.md."""
    axes = []
    for path in sorted((kb_dir / "axes").glob("axis-*.md")):
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        axes.append({
            "id": path.stem,
            "title": str(post.get("title", path.stem)),
            "description": str(post.get("description", "")),
            "members_count": int(post.get("members_count", 0)),
            "source_wikis": list(post.get("source_wikis", [])),
        })
    lines = ["# Base de connaissance — Index thématique\n\n"]
    for a in axes:
        lines.append(f"## [[{a['id']}]] {a['title']}\n\n")
        if a["description"]:
            lines.append(f"{a['description']}\n\n")
        lines.append(
            f"*{a['members_count']} pages — {', '.join(a['source_wikis'])}*\n\n"
        )
    (kb_dir / "index.md").write_text("".join(lines), encoding="utf-8")


def update_kb(
    wiki_root: Path,
    clustering_name: str,
    embed_dir: Path,
    kb_dir: Path,
    fusion_threshold: float = FUSION_THRESHOLD,
    min_size: int = MIN_SIZE,
    min_cohesion: float | None = None,
) -> dict:
    """
    Met à jour la base de connaissance depuis un clustering archivé.
    wiki_root : répertoire racine du wiki archivé (contient wiki/).
    Retourne {"created": int, "updated": int, "excluded": int}.
    """
    clustering_dir = wiki_root / "clusterings" / clustering_name
    if not clustering_dir.exists():
        raise FileNotFoundError(f"Clustering introuvable : {clustering_dir}")

    sources_dir = wiki_root / "sources"

    m_theta = re.search(r"-(\d+\.\d+)$", clustering_name)
    auto_theta = float(m_theta.group(1)) if m_theta else 0.0
    effective_min_cohesion = min_cohesion if min_cohesion is not None else auto_theta / 2

    embed_matrix, embed_slugs = _load_embed_matrix(embed_dir)
    slug_to_idx: dict[str, int] = (
        {s: i for i, s in enumerate(embed_slugs)} if embed_slugs else {}
    )

    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "axes").mkdir(exist_ok=True)
    (kb_dir / "embeddings").mkdir(exist_ok=True)

    axes_npy_path = kb_dir / "embeddings" / "axes.npy"
    axes_index_path = kb_dir / "embeddings" / "axes_index.json"

    if axes_npy_path.exists() and axes_index_path.exists():
        try:
            axis_matrix: list[np.ndarray] = list(
                np.load(axes_npy_path).astype(np.float32)
            )
            axis_ids: list[str] = json.loads(
                axes_index_path.read_text(encoding="utf-8")
            ).get("ids", [])
        except Exception:
            axis_matrix = []
            axis_ids = []
    else:
        axis_matrix = []
        axis_ids = []

    # Compteur local pour éviter N² scans disque dans _next_axis_id
    _existing_nums = []
    for _p in (kb_dir / "axes").glob("axis-*.md"):
        _m = re.match(r"axis-(\d+)\.md$", _p.name)
        if _m:
            _existing_nums.append(int(_m.group(1)))
    _next_axis_num = [max(_existing_nums) + 1 if _existing_nums else 1]

    wiki_name = wiki_root.name
    today = date.today().isoformat()
    excluded: list[dict] = []
    stats = {"created": 0, "updated": 0, "excluded": 0}

    for cluster in _load_cluster_files(clustering_dir):
        if cluster["status"] == "garbage":
            excluded.append({"cluster_id": cluster["id"], "wiki": wiki_name, "reason": "status=garbage"})
            stats["excluded"] += 1
            continue
        if len(cluster["members"]) < min_size:
            excluded.append({"cluster_id": cluster["id"], "wiki": wiki_name,
                              "reason": f"size {len(cluster['members'])} < {min_size}"})
            stats["excluded"] += 1
            continue

        if embed_matrix is None:
            excluded.append({"cluster_id": cluster["id"], "wiki": wiki_name, "reason": "no embeddings"})
            stats["excluded"] += 1
            continue

        centroid, cohesion = _compute_centroid_and_cohesion(
            cluster["members"], embed_matrix, slug_to_idx
        )
        if centroid is None:
            excluded.append({"cluster_id": cluster["id"], "wiki": wiki_name, "reason": "no embeddings for members"})
            stats["excluded"] += 1
            continue
        if cohesion < effective_min_cohesion:
            excluded.append({"cluster_id": cluster["id"], "wiki": wiki_name,
                              "reason": f"cohesion {cohesion:.3f} < {effective_min_cohesion:.3f}"})
            stats["excluded"] += 1
            continue

        tags = _collect_tags(cluster["members"], sources_dir)

        best_score = -1.0
        best_idx = -1
        if axis_matrix:
            mat = np.array(axis_matrix, dtype=np.float32)
            sims = mat @ centroid
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

        if best_score >= fusion_threshold:
            axis_id = axis_ids[best_idx]
            axis_path = kb_dir / "axes" / f"{axis_id}.md"
            post = frontmatter.load(axis_path)
            old_count = int(post.get("members_count", 1))
            new_count = old_count + len(cluster["members"])
            old_vec = axis_matrix[best_idx]
            # Approximation : old_vec est normalisé, pas la somme brute — dérive faible si fusion_threshold élevé
            fused = (old_vec * old_count + centroid * len(cluster["members"])) / new_count
            n = float(np.linalg.norm(fused))
            axis_matrix[best_idx] = (fused / n if n > 0 else fused).astype(np.float32)
            src_wikis = list(post.get("source_wikis", []))
            if wiki_name not in src_wikis:
                src_wikis.append(wiki_name)
            existing_tags = list(post.get("tags", []))
            for t in tags:
                if t not in existing_tags:
                    existing_tags.append(t)
            post["source_wikis"] = src_wikis
            post["members_count"] = new_count
            post["cohesion"] = round(cohesion, 4)
            post["tags"] = existing_tags[:10]
            post["updated"] = today
            axis_path.write_text(frontmatter.dumps(post), encoding="utf-8")
            stats["updated"] += 1
        else:
            axis_id = f"axis-{_next_axis_num[0]:04d}"
            _next_axis_num[0] += 1
            axis_path = kb_dir / "axes" / f"{axis_id}.md"
            rep_lines = "\n".join(f"- [[{s}]]" for s in cluster["members"][:3])
            new_post = frontmatter.Post(
                f"\n## Pages représentatives\n\n{rep_lines}\n",
                title=cluster["title"],
                description=cluster["description"],
                source_wikis=[wiki_name],
                updated=today,
                members_count=len(cluster["members"]),
                cohesion=round(cohesion, 4),
                tags=tags,
                status="active",
            )
            axis_path.write_text(frontmatter.dumps(new_post), encoding="utf-8")
            axis_ids.append(axis_id)
            axis_matrix.append(centroid.astype(np.float32))
            stats["created"] += 1

    if axis_matrix:
        tmp_npy = axes_npy_path.parent / (axes_npy_path.stem + ".tmp.npy")
        np.save(tmp_npy, np.array(axis_matrix, dtype=np.float32))
        tmp_npy.rename(axes_npy_path)
        tmp_idx = axes_index_path.with_suffix(".json.tmp")
        tmp_idx.write_text(json.dumps({"ids": axis_ids}), encoding="utf-8")
        tmp_idx.rename(axes_index_path)

    excluded_path = kb_dir / "excluded.json"
    existing_excl: list[dict] = (
        json.loads(excluded_path.read_text(encoding="utf-8"))
        if excluded_path.exists()
        else []
    )
    tmp_excl = excluded_path.with_suffix(".json.tmp")
    tmp_excl.write_text(
        json.dumps(existing_excl + excluded, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_excl.rename(excluded_path)

    _regenerate_index(kb_dir)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Met à jour la base de connaissance")
    parser.add_argument("--wiki", required=True, help="Racine du wiki archivé")
    parser.add_argument("--clustering", required=True,
                        help="Nom du clustering (ex: clustering-embeddings-transfers-0.403)")
    parser.add_argument("--embed-dir", default=str(_DEFAULT_EMBED_DIR))
    parser.add_argument("--kb-dir", default=str(_DEFAULT_KB_DIR))
    parser.add_argument("--fusion-threshold", type=float, default=FUSION_THRESHOLD)
    parser.add_argument("--min-size", type=int, default=MIN_SIZE)
    parser.add_argument("--min-cohesion", type=float, default=None)
    args = parser.parse_args()

    stats = update_kb(
        wiki_root=Path(args.wiki),
        clustering_name=args.clustering,
        embed_dir=Path(args.embed_dir),
        kb_dir=Path(args.kb_dir),
        fusion_threshold=args.fusion_threshold,
        min_size=args.min_size,
        min_cohesion=args.min_cohesion,
    )
    print(f"[kb_update] Créés: {stats['created']}, Mis à jour: {stats['updated']}, Exclus: {stats['excluded']}")


if __name__ == "__main__":
    main()
