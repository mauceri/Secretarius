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

_EMBED_DIR = Path(__file__).resolve().parent.parent / "embeddings"
_DEFAULT_WIKI = os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM"))


# ---------------------------------------------------------------------------
# Chargement des pages src-
# ---------------------------------------------------------------------------

def _load_src_pages(wiki_dir: Path) -> list[dict]:
    """Retourne la liste de dicts {slug, title, abstract} pour toutes les pages src-."""
    pages = []
    for path in sorted(wiki_dir.glob("src-*.md")):
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
