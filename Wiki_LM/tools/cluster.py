"""
Clustering des pages src- du wiki.

Usage CLI :
    python tools/cluster.py --signal embeddings --param 30
    python tools/cluster.py --signal embeddings --param 10,30,60
    python tools/cluster.py --signal embeddings+colinks --param 30 --no-llm

Endpoint serveur : POST /cluster {"signal": "embeddings", "param": 30}
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from pathlib import Path

import frontmatter
import numpy as np
from sklearn.cluster import HDBSCAN

from llm import LLM
from similarity import (
    BaseSimilarity,
    CoLinkSimilarity,
    CombinedSimilarity,
    EmbeddingSimilarity,
    TagSimilarity,
)
from wiki_paths import CLUSTERING_SUBDIR

_EMBED_DIR = Path(__file__).resolve().parent.parent / "embeddings"
_DEFAULT_WIKI = os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM"))


# ---------------------------------------------------------------------------
# Chargement des pages src-
# ---------------------------------------------------------------------------

def _load_src_pages(wiki_dir: Path) -> list[dict]:
    """Retourne la liste de dicts {slug, title, abstract} pour toutes les pages src-."""
    pages = []
    for path in sorted((wiki_dir / "sources").glob("src-*.md")):
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        m = re.search(r"## Résumé[ \t]*\n(.+?)(?=\n## |\Z)", post.content, re.DOTALL)
        abstract = m.group(1).strip() if m else ""
        pages.append({
            "slug": path.stem,
            "title": str(post.get("title", path.stem)),
            "abstract": abstract,
        })
    return pages


# ---------------------------------------------------------------------------
# Construction du signal
# ---------------------------------------------------------------------------

def _build_signal(signal_str: str, wiki_dir: Path, embed_dir: Path) -> BaseSimilarity:
    parts = [p.strip() for p in signal_str.split("+")]
    signals: list[BaseSimilarity] = []
    for part in parts:
        if part == "embeddings":
            signals.append(EmbeddingSimilarity(embed_dir))
        elif part == "colinks":
            signals.append(CoLinkSimilarity(wiki_dir))
        elif part == "tags":
            signals.append(TagSimilarity(wiki_dir))
        else:
            raise ValueError(f"Signal inconnu : {part!r}. Valeurs : embeddings, colinks, tags")
    return signals[0] if len(signals) == 1 else CombinedSimilarity(signals)


# ---------------------------------------------------------------------------
# Fonctions utilitaires clustering
# ---------------------------------------------------------------------------

def _find_paragon(indices: list[int], sim: np.ndarray) -> int:
    """Retourne l'indice (dans sim) du document le plus central du cluster."""
    sub = sim[np.ix_(indices, indices)]
    return indices[int(np.argmax(sub.mean(axis=1)))]


def _nearest_clusters(
    cluster_id: int,
    centroids: dict[int, np.ndarray],
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Retourne les top_k clusters les plus proches par similarité cosinus des centroïdes."""
    ref = centroids[cluster_id]
    ref_norm = np.linalg.norm(ref)
    scores = []
    for cid, vec in centroids.items():
        if cid == cluster_id:
            continue
        denom = ref_norm * np.linalg.norm(vec)
        sim = float(ref @ vec / denom) if denom > 0 else 0.0
        scores.append((cid, sim))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


def _describe_cluster(
    paragon_slug: str,
    pages_by_slug: dict[str, dict],
    llm,  # LLM | None
) -> tuple[str, str]:
    """Retourne (titre, description) à partir du parangon via LLM, ou défauts si llm=None."""
    if llm is None:
        return "Cluster", ""

    p = pages_by_slug.get(paragon_slug, {})
    paragon_title = p.get("title", paragon_slug)
    paragon_abstract = p.get("abstract", "")[:300]

    prompt = (
        "Voici le titre et le résumé du document le plus représentatif d'un groupe thématique :\n\n"
        f"Titre : {paragon_title}\n"
        f"Résumé : {paragon_abstract}\n\n"
        "Donne :\n"
        "1. Un titre court (4-6 mots) caractérisant le thème du groupe\n"
        "2. Une description de 2-3 phrases résumant ce groupe\n\n"
        "Format de réponse strict :\nTITRE: <titre>\nDESCRIPTION: <description>"
    )
    response = llm.complete(prompt, max_tokens=200)

    title = "Cluster"
    description = ""
    lines = response.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("TITRE:"):
            title = line[6:].strip()
        elif line.startswith("DESCRIPTION:"):
            parts = [line[12:].strip()]
            for cont in lines[i + 1:]:
                if cont.startswith("TITRE:") or cont.startswith("DESCRIPTION:"):
                    break
                parts.append(cont)
            description = " ".join(p for p in parts if p)
    return title, description


# ---------------------------------------------------------------------------
# Embeddings bruts pour calcul de centroïdes
# ---------------------------------------------------------------------------

def _get_embed_rows(slugs: list[str], embed_dir: Path) -> np.ndarray | None:
    """Retourne un tableau (N, dim) des vecteurs d'embedding pour les slugs donnés."""
    index_path = embed_dir / "embeddings_index.json"
    matrix_path = embed_dir / "embeddings.npy"
    if not index_path.exists() or not matrix_path.exists():
        return None
    try:
        full = np.load(matrix_path)
        all_slugs = json.loads(index_path.read_text(encoding="utf-8"))["slugs"]
        slug_to_idx = {s: i for i, s in enumerate(all_slugs)}
        rows = np.zeros((len(slugs), full.shape[1]), dtype=np.float32)
        for i, s in enumerate(slugs):
            if s in slug_to_idx:
                rows[i] = full[slug_to_idx[s]]
            # slugs absent du fichier d'index restent à zéro → biais centroïde possible
        return rows
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Clustering principal
# ---------------------------------------------------------------------------

def run_clustering(
    wiki_dir: Path,
    embed_dir: Path,
    signal_str: str,
    param: int,
    llm: LLM | None = None,
    algo: str = "hdbscan",
) -> dict:
    """
    Lance le clustering et écrit les fichiers wiki.
    Retourne un dict de stats : clusters, noise, signal, algo, param, out_dir.
    """
    pages = _load_src_pages(wiki_dir)
    if not pages:
        return {"clusters": 0, "noise": 0, "signal": signal_str, "algo": algo, "param": param, "out_dir": ""}

    slugs = [p["slug"] for p in pages]
    pages_by_slug = {p["slug"]: p for p in pages}

    signal = _build_signal(signal_str, wiki_dir, embed_dir)
    sim = signal.compute(slugs)

    dist = (1.0 - sim).clip(0.0).astype(np.float64)
    np.fill_diagonal(dist, 0.0)

    labels = HDBSCAN(
        min_cluster_size=param,
        metric="precomputed",
        cluster_selection_method="eom",
        copy=True,
    ).fit_predict(dist)

    clusters: dict[int, list[int]] = {}
    noise: list[int] = []
    for i, label in enumerate(labels):
        if label == -1:
            noise.append(i)
        else:
            clusters.setdefault(int(label), []).append(i)

    embed_rows = _get_embed_rows(slugs, embed_dir)

    centroids: dict[int, np.ndarray] = {}
    for cid, idx_list in clusters.items():
        if embed_rows is not None:
            centroids[cid] = embed_rows[idx_list].mean(axis=0)
        else:
            centroids[cid] = sim[idx_list].mean(axis=0)

    out_dir = wiki_dir / CLUSTERING_SUBDIR / f"clustering-{signal_str}-{algo}-{param}"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    cluster_slugs = {cid: f"cluster-{signal_str}-{algo}-{param}-{cid:04d}" for cid in clusters}

    for cid, idx_list in clusters.items():
        paragon_idx = _find_paragon(idx_list, sim)
        paragon_slug = slugs[paragon_idx]
        member_slugs = [slugs[i] for i in idx_list]

        near = _nearest_clusters(cid, centroids) if len(centroids) > 1 else []
        title, description = _describe_cluster(paragon_slug, pages_by_slug, llm)
        cluster_file_slug = cluster_slugs[cid]

        lines = [
            "---",
            "category: cluster",
            f"signal: {signal_str}",
            f"algo: {algo}",
            f"param: {param}",
            f"members: {len(idx_list)}",
            f"paragon: {paragon_slug}",
            f"created: {today}",
            "---", "",
            f"# {title}", "",
        ]
        if description:
            lines += [description, ""]
        lines += ["## Parangon", ""]
        paragon_title = pages_by_slug.get(paragon_slug, {}).get("title", paragon_slug)
        lines.append(f"[[{paragon_slug}]] — {paragon_title}")
        lines += ["", "## Documents membres", ""]
        for s in member_slugs:
            t = pages_by_slug.get(s, {}).get("title", s)
            lines.append(f"- [[{s}]] — {t}")
        lines += ["", "## Clusters proches", ""]
        for near_cid, near_score in near:
            near_slug = cluster_slugs.get(near_cid, f"cluster-{near_cid}")
            lines.append(f"- [[{near_slug}]] (similarité : {near_score:.2f})")

        (out_dir / f"{cluster_file_slug}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    # unclustered.md
    unclustered_lines = [
        f"# Sources non assignées ({len(noise)} documents)", "",
        "Ces sources sont thématiquement isolées (bruit HDBSCAN).", "",
    ]
    for i in noise:
        s = slugs[i]
        t = pages_by_slug.get(s, {}).get("title", s)
        unclustered_lines.append(f"- [[{s}]] — {t}")
    (out_dir / "unclustered.md").write_text("\n".join(unclustered_lines), encoding="utf-8")

    # index.md
    index_lines = [
        f"# Index — clustering-{signal_str}-{algo}-{param}", "",
        f"- **Signal :** {signal_str}",
        f"- **Algorithme :** {algo}, param={param}",
        f"- **Clusters :** {len(clusters)}",
        f"- **Sources non assignées :** {len(noise)}",
        f"- **Généré le :** {today}", "",
        "## Clusters", "",
    ]
    for cid in sorted(clusters):
        index_lines.append(f"- [[{cluster_slugs[cid]}]]")
    (out_dir / "index.md").write_text("\n".join(index_lines), encoding="utf-8")

    return {
        "clusters": len(clusters),
        "noise": len(noise),
        "signal": signal_str,
        "algo": algo,
        "param": param,
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clustering des pages src- du wiki")
    parser.add_argument(
        "--wiki",
        default=_DEFAULT_WIKI,
        help="Chemin vers le wiki (défaut : $WIKI_PATH ou ~/Documents/Arbath/Wiki_LM)",
    )
    parser.add_argument(
        "--signal",
        default="embeddings",
        help="Signal(s) : embeddings, colinks, tags, ou combinaison ex. embeddings+colinks",
    )
    parser.add_argument(
        "--param",
        default="30",
        help="min_cluster_size(s), virgule-séparés ex. 10,30,60",
    )
    parser.add_argument("--no-llm", action="store_true", help="Ne pas appeler le LLM")
    parser.add_argument("--embed-dir", default="")
    parser.add_argument("--backend", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    embed_dir = Path(args.embed_dir) if args.embed_dir else _EMBED_DIR

    if args.no_llm:
        llm = None
    elif args.backend or args.model:
        llm = LLM(backend=args.backend, model=args.model)
    else:
        llm = LLM()

    params = [int(p.strip()) for p in args.param.split(",") if p.strip()]
    for param in params:
        print(f"[cluster] signal={args.signal} param={param} …")
        stats = run_clustering(wiki_dir, embed_dir, args.signal, param, llm=llm)
        print(
            f"[cluster] {stats['clusters']} clusters, {stats['noise']} non assignés"
            f" → {stats['out_dir']}"
        )


if __name__ == "__main__":
    main()
