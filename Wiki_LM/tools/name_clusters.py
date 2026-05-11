# tools/name_clusters.py
"""
Titre thématique des clusters via LLM.

Pour chaque cluster dont le titre est générique ("Cluster"), appelle le LLM
avec les titres des pages membres pour générer un titre et une description.
Met à jour le fichier cluster en place.

Usage:
    python tools/name_clusters.py \\
        --clustering ~/Documents/Arbath/Wiki_LM/wiki_signets_05_2026/clusterings/clustering-embeddings-transfers-0.404 \\
        [--force]          # retitrer même les clusters déjà titrés
        [--dry-run]        # affiche sans écrire
        [--max-members 20] # nb max de titres envoyés au LLM
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import frontmatter

sys.path.insert(0, str(Path(__file__).parent))
from llm import LLM

_GENERIC_TITLE = "Cluster"
_MAX_MEMBERS_DEFAULT = 20


def _extract_member_titles(content: str) -> list[str]:
    """Extrait les titres des membres depuis la section '## Documents membres'."""
    m = re.search(r"## Documents membres\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
    if not m:
        return []
    return re.findall(r"\[\[[^\]]+\]\] — (.+)", m.group(1))


def _call_llm(llm: LLM, titles: list[str]) -> tuple[str, str]:
    """Appelle le LLM et retourne (title, description)."""
    titles_str = "\n".join(f"- {t}" for t in titles)
    prompt = (
        "Tu analyses un groupe de pages web thématiquement similaires. "
        "Voici leurs titres :\n\n"
        f"{titles_str}\n\n"
        "Génère un titre court (4-7 mots, en français) qui caractérise le thème commun "
        "de ces pages, et une description en une phrase.\n"
        "Réponds uniquement en JSON valide : {\"title\": \"...\", \"description\": \"...\"}"
    )
    raw = llm.complete(prompt, max_tokens=512)
    # Tente le JSON complet d'abord
    m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return str(data["title"]).strip(), str(data.get("description", "")).strip()
        except (json.JSONDecodeError, KeyError):
            pass
    # Fallback : extraction par regex si JSON tronqué
    m_title = re.search(r'"title"\s*:\s*"([^"]+)"', raw)
    m_desc = re.search(r'"description"\s*:\s*"([^"]+)"', raw)
    if m_title:
        title = m_title.group(1).strip()
        desc = m_desc.group(1).strip() if m_desc else ""
        return title, desc
    raise ValueError(f"Pas de titre dans la réponse : {raw!r}")


def name_clusters(
    clustering_dir: Path,
    force: bool = False,
    dry_run: bool = False,
    max_members: int = _MAX_MEMBERS_DEFAULT,
) -> dict:
    """
    Titre tous les clusters génériques du répertoire.
    Retourne {"titled": int, "skipped": int, "errors": int}.
    """
    if not clustering_dir.exists():
        raise FileNotFoundError(f"Répertoire introuvable : {clustering_dir}")

    files = sorted(clustering_dir.glob("cluster-*.md"))
    if not files:
        raise FileNotFoundError(f"Aucun cluster-*.md dans {clustering_dir}")

    llm = LLM()
    stats = {"titled": 0, "skipped": 0, "errors": 0}

    for path in files:
        try:
            post = frontmatter.load(path)
        except Exception as e:
            print(f"[ERREUR] {path.name} : impossible de lire ({e})", file=sys.stderr)
            stats["errors"] += 1
            continue

        content = post.content
        m_title = re.search(r"^# (.+)$", content, re.MULTILINE)
        current_title = m_title.group(1).strip() if m_title else ""

        if current_title != _GENERIC_TITLE and not force:
            stats["skipped"] += 1
            continue

        titles = _extract_member_titles(content)
        if not titles:
            print(f"[SKIP] {path.name} : aucun titre de membre trouvable", file=sys.stderr)
            stats["skipped"] += 1
            continue

        sample = titles[:max_members]

        try:
            title, description = _call_llm(llm, sample)
        except Exception as e:
            print(f"[ERREUR] {path.name} : LLM ({e})", file=sys.stderr)
            stats["errors"] += 1
            time.sleep(1)
            continue

        print(f"[{'DRY' if dry_run else 'OK'}] {path.name} → {title!r}")

        if not dry_run:
            # Remplace le titre générique dans le corps
            new_content = re.sub(
                r"^# .+$", f"# {title}", content, count=1, flags=re.MULTILINE
            )
            # Insère la description après le titre si absente ou si c'est une section ##
            m_desc = re.search(r"^# [^\n]+\n\n([^\n]+)", new_content, re.MULTILINE)
            if not m_desc or m_desc.group(1).startswith("##"):
                new_content = re.sub(
                    r"(^# .+\n)", rf"\1\n{description}\n", new_content, count=1, flags=re.MULTILINE
                )
            post.content = new_content
            path.write_text(frontmatter.dumps(post), encoding="utf-8")

        stats["titled"] += 1
        # Petite pause pour ne pas saturer l'API
        time.sleep(0.3)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Titre thématique des clusters via LLM")
    parser.add_argument("--clustering", required=True,
                        help="Répertoire de clustering (contient les cluster-*.md)")
    parser.add_argument("--force", action="store_true",
                        help="Retitrer même les clusters déjà titrés")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche les titres générés sans écrire")
    parser.add_argument("--max-members", type=int, default=_MAX_MEMBERS_DEFAULT,
                        help=f"Nb max de titres envoyés au LLM (défaut : {_MAX_MEMBERS_DEFAULT})")
    args = parser.parse_args()

    clustering_dir = Path(args.clustering).expanduser()
    stats = name_clusters(
        clustering_dir,
        force=args.force,
        dry_run=args.dry_run,
        max_members=args.max_members,
    )
    print(
        f"\n[name_clusters] Titrés: {stats['titled']}, "
        f"Ignorés: {stats['skipped']}, Erreurs: {stats['errors']}"
    )


if __name__ == "__main__":
    main()
