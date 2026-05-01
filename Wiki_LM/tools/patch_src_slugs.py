"""
Renomme les pages src- avec des slugs lisibles dérivés du titre LLM.

IMPORTANT : lancer uniquement quand aucune ingestion n'est en cours
(risque de conflit sur le manifeste .ingested et les fichiers raw/).

Opérations :
  1. Pour chaque src-*.md : calcule le nouveau slug depuis title: du frontmatter
  2. Renomme le fichier wiki
  3. Renomme le fichier raw correspondant (s'il existe)
  4. Remplace [[ancien-slug]] par [[nouveau-slug]] dans TOUT le wiki
  5. Met à jour index.md et tags.md (réécriture simple)
  6. Met à jour le manifeste raw/.ingested

Usage :
    python tools/patch_src_slugs.py              # dry-run
    python tools/patch_src_slugs.py --apply      # écriture effective
    python tools/patch_src_slugs.py --apply --wiki /chemin/vers/wiki
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import frontmatter


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:60]


_FILENAME_TITLE_RE = re.compile(
    r"^[\w\-]+\.(pdf|txt|html|htm|md|docx?)$", re.IGNORECASE
)


_FILENAME_PREFIX_RE = re.compile(
    r"^[\w\-]+\.(pdf|txt|html|htm|docx?)\s*[-–—]\s*", re.IGNORECASE
)


def _clean_title(title: str) -> str:
    """Supprime un éventuel préfixe 'fichier.ext - ' résiduel dans le titre."""
    return _FILENAME_PREFIX_RE.sub("", title).strip()


def _title_to_slug(title: str) -> str:
    base = re.sub(r"^src-", "", _slugify(_clean_title(title)))
    return f"src-{base}"


def _is_meaningful_title(title: str) -> bool:
    """Retourne False si le titre ressemble à un nom de fichier ou un ID brut."""
    t = title.strip()
    if not t or len(t) < 6:
        return False
    if _FILENAME_TITLE_RE.match(t):
        return False
    # Titre composé uniquement de chiffres et séparateurs (ID numérique)
    if re.match(r"^[\d\s\-_.]+$", t):
        return False
    # Titre qui commence par un ID arxiv (ex: "11283.pdf") ou similaire
    if re.match(r"^\d{4,}[\.\-]", t):
        return False
    return True


def _collect_renames(wiki_dir: Path) -> list[tuple[str, str]]:
    """Retourne [(ancien_slug, nouveau_slug)] pour les pages dont le slug change."""
    renames = []
    for path in sorted(wiki_dir.glob("src-*.md")):
        try:
            post = frontmatter.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        title = str(post.get("title", "")).strip()
        if not _is_meaningful_title(title):
            continue
        new_slug = _title_to_slug(title)
        old_slug = path.stem
        if new_slug != old_slug and new_slug != "src-":
            renames.append((old_slug, new_slug))
    return renames


def _apply_renames(
    wiki_dir: Path,
    raw_dir: Path,
    renames: list[tuple[str, str]],
    apply: bool,
) -> None:
    # Construire la map old → new pour les substitutions dans le texte
    slug_map: dict[str, str] = dict(renames)

    # --- 1. Renommer les fichiers wiki ---
    for old_slug, new_slug in renames:
        old_path = wiki_dir / f"{old_slug}.md"
        new_path = wiki_dir / f"{new_slug}.md"
        if not old_path.exists():
            print(f"  [SKIP] {old_slug}.md introuvable")
            continue
        if new_path.exists():
            print(f"  [SKIP] {new_slug}.md existe déjà, conflit ignoré")
            slug_map.pop(old_slug, None)
            continue
        if apply:
            old_path.rename(new_path)
        print(f"  {'[rename]' if apply else '[dry]  '} {old_slug} → {new_slug}")

    # --- 2. Renommer les fichiers raw ---
    for old_slug, new_slug in slug_map.items():
        for old_raw in raw_dir.glob(f"{old_slug}.*"):
            new_raw = old_raw.with_name(new_slug + old_raw.suffix)
            if new_raw.exists():
                continue
            if apply:
                old_raw.rename(new_raw)

    # --- 3. Mettre à jour le manifeste .ingested (format TSV : filename\tslug\thash) ---
    manifest = raw_dir / ".ingested"
    if manifest.exists():
        lines = manifest.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            if not line:
                continue
            parts = line.split("\t")
            # Mettre à jour le champ slug (colonne 1) si présent
            if len(parts) >= 2 and parts[1] in slug_map:
                parts[1] = slug_map[parts[1]]
            new_lines.append("\t".join(parts))
        if apply:
            manifest.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    if not slug_map:
        return

    # --- 4. Remplacer les [[liens]] dans TOUT le wiki ---
    # Trier du plus long au plus court pour éviter les substitutions partielles
    sorted_map = sorted(slug_map.items(), key=lambda x: -len(x[0]))

    def _replace_links(text: str) -> str:
        for old, new in sorted_map:
            text = re.sub(
                r"\[\[" + re.escape(old) + r"(\|[^\]]+)?\]\]",
                lambda m, n=new: f"[[{n}{m.group(1) or ''}]]",
                text,
            )
            # Aussi dans les champs sources: du frontmatter (liste YAML)
            text = re.sub(
                r"(?<=[- ])" + re.escape(old) + r"(?=\s*$)",
                new,
                text,
                flags=re.MULTILINE,
            )
        return text

    for md_path in wiki_dir.glob("*.md"):
        try:
            original = md_path.read_text(encoding="utf-8")
        except Exception:
            continue
        updated = _replace_links(original)
        if updated != original:
            if apply:
                md_path.write_text(updated, encoding="utf-8")


def _read_url_from_file(raw_file: Path) -> str:
    """Extrait l'URL d'un fichier .url (format bare ou 'url: https://...')."""
    for line in raw_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("url:"):
            line = line[4:].strip()
        if line.startswith("http"):
            return line
    return ""


def _inject_urls(wiki_dir: Path, raw_dir: Path, apply: bool) -> int:
    """Injecte url: dans le frontmatter des pages src- qui n'en ont pas encore.

    Fonctionne par correspondance directe : le fichier raw {slug}.url donne l'URL
    de la page wiki/{slug}.md — sans dépendre du manifeste.
    """
    patched = 0
    for raw_file in raw_dir.glob("*.url"):
        slug = raw_file.stem
        page = wiki_dir / f"{slug}.md"
        if not page.exists():
            continue
        content = page.read_text(encoding="utf-8")
        if "lien_source:" in content[:400]:
            continue
        url = _read_url_from_file(raw_file)
        if not url:
            continue
        new_content = content.replace("---\n", f"---\nlien_source: {url}\n", 1)
        if new_content != content:
            if apply:
                page.write_text(new_content, encoding="utf-8")
            patched += 1
    return patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Écrire les modifications")
    parser.add_argument(
        "--wiki",
        default=os.environ.get(
            "WIKI_PATH",
            str(Path.home() / "Documents/Arbath/Wiki_LM")
        ),
    )
    parser.add_argument(
        "--raw",
        default=os.environ.get(
            "WIKI_RAW_PATH",
            str(Path.home() / "Secretarius/Wiki_LM/raw")
        ),
    )
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    raw_dir = Path(args.raw)

    renames = _collect_renames(wiki_dir)
    print(f"Pages à renommer : {len(renames)}")

    if not renames:
        print("Rien à faire.")
        return

    if not args.apply:
        print("\nAperçu (--dry-run, pas d'écriture) :")
        for old, new in renames[:30]:
            print(f"  {old}")
            print(f"    → {new}")
        if len(renames) > 30:
            print(f"  … et {len(renames) - 30} autres")
        urls_count = _inject_urls(wiki_dir, raw_dir, apply=False)
        print(f"\nURLs à injecter : {urls_count} page(s) sans url: dans le frontmatter.")
        print("\nRelancez avec --apply pour écrire.")
        return

    _apply_renames(wiki_dir, raw_dir, renames, apply=True)
    print(f"\nTerminé : {len(renames)} pages renommées.")

    patched = _inject_urls(wiki_dir, raw_dir, apply=True)
    print(f"URLs injectées : {patched} page(s) mises à jour.")


if __name__ == "__main__":
    main()
