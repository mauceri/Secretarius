"""
Ingestion d'une source dans le wiki Wiki_LM.

Le pipeline :
  1. Lire la source (fichier local ou URL)
  2. Sauvegarder dans raw/ (immutable)
  3. Appeler le LLM pour produire une page wiki structurée (src-<slug>.md)
  4. Extraire les concepts/entités mentionnés → créer/mettre à jour leurs pages
  5. Mettre à jour index.md
  6. Appender à log.md

Usage CLI :
    python ingest.py article.pdf
    python ingest.py https://example.com/article
    python ingest.py article.txt --slug mon-article --wiki /chemin/vers/Wiki_LM

Usage module :
    from ingest import Ingestor
    ing = Ingestor("/home/mauceric/Documents/Arbath/Wiki_LM")
    ing.ingest("article.txt", slug="mon-article")
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import frontmatter

from llm import LLM


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convertit un texte en slug ASCII lowercase sans accents."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:60]


def _today() -> str:
    return datetime.date.today().isoformat()


_USER_AGENT = "WikiLM/1.0 (personal wiki ingestor; python-urllib)"


def _read_url(url: str) -> str:
    """Télécharge une page web et retourne le texte brut.

    Les URLs Wikipedia sont traitées via l'API REST (texte propre, sans HTML).
    Les autres URLs passent par un extracteur HTML léger.
    """
    import urllib.request

    # Wikipedia : API REST → texte Wikitext extrait, sans HTML
    if "wikipedia.org/wiki/" in url:
        return _read_wikipedia(url)

    import html.parser

    class _Stripper(html.parser.HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self._skip = False
            self.parts: list[str] = []

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            if tag in ("script", "style", "nav", "footer", "header"):
                self._skip = True

        def handle_endtag(self, tag: str) -> None:
            if tag in ("script", "style", "nav", "footer", "header"):
                self._skip = False

        def handle_data(self, data: str) -> None:
            if not self._skip:
                stripped = data.strip()
                if stripped:
                    self.parts.append(stripped)

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_html = resp.read().decode("utf-8", errors="replace")

    stripper = _Stripper()
    stripper.feed(raw_html)
    return "\n".join(stripper.parts)


def _read_wikipedia(url: str) -> str:
    """Récupère le texte d'un article Wikipedia via l'API REST."""
    import urllib.request
    import urllib.parse
    import json

    # Extraire le titre depuis l'URL
    # ex. https://en.wikipedia.org/wiki/Gerard_Salton → Gerard_Salton
    parts = url.split("/wiki/", 1)
    if len(parts) < 2:
        raise ValueError(f"URL Wikipedia non reconnue : {url}")

    lang = url.split("//")[1].split(".")[0]  # "en", "fr", …
    title = urllib.parse.unquote(parts[1].split("#")[0])

    api_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    req = urllib.request.Request(api_url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    # summary donne titre + extrait (~500 mots) — suffisant pour ingest
    extract = data.get("extract", "")
    page_title = data.get("title", title)
    description = data.get("description", "")

    return f"# {page_title}\n\n{description}\n\n{extract}"


def _read_source(source: str) -> tuple[str, str]:
    """Retourne (contenu texte, titre présumé).

    source peut être un chemin fichier ou une URL http(s).
    """
    if source.startswith("http://") or source.startswith("https://"):
        content = _read_url(source)
        title = source.split("/")[-1] or source
        return content, title

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source introuvable : {source}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        content = _read_pdf(path)
    else:
        content = path.read_text(errors="replace")

    return content, path.stem


def _read_pdf(path: Path) -> str:
    """Extrait le texte d'un PDF via pypdf (si disponible) ou pdfminer."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        pass
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    except ImportError:
        raise ImportError(
            "Installez pypdf ou pdfminer.six pour lire les PDF : pip install pypdf"
        )


def _truncate(text: str, max_chars: int = 12_000) -> str:
    """Tronque le texte source pour ne pas dépasser le contexte du LLM."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[… texte tronqué à {max_chars} caractères …]"


def _fix_mojibake(text: str) -> str:
    """Corrige le mojibake latin-1→UTF-8 (entitÃ© → entité).

    Se produit quand une API renvoie du UTF-8 interprété comme latin-1.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _normalize_links(content: str, known_slugs: set[str]) -> str:
    """Normalise les [[liens]] LLM vers des slugs réels du wiki.

    Stratégie :
    1. Si le lien correspond déjà à un slug connu → inchangé
    2. Sinon slugifie le texte et cherche une correspondance avec préfixes
    3. Sinon garde le texte slugifié (lien vers page future)
    """
    def _resolve(m: re.Match) -> str:
        raw = m.group(1).strip()
        slug = _slugify(raw)

        if slug in known_slugs:
            return f"[[{slug}]]"
        for prefix in ("concept-", "entity-", "src-", "synth-"):
            candidate = prefix + slug
            if candidate in known_slugs:
                return f"[[{candidate}]]"
        # Aucune correspondance — garder slugifié pour cohérence
        return f"[[{slug}]]"

    return _LINK_RE.sub(_resolve, content)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_INGEST = """\
Tu es l'assistant d'un wiki personnel (Wiki_LM). \
Tu reçois une source textuelle et tu produis des pages Markdown structurées. \
Respecte scrupuleusement les formats demandés — frontmatter YAML complet, \
pas de prose hors des blocs demandés, slugs en minuscules sans accents.

Conventions de liens internes (OBLIGATOIRE) :
- Format [[slug-exact]] avec préfixe selon la catégorie : concept-, entity-, src-, synth-
- Slugs : minuscules, tirets, sans accents, sans espaces
- Exemples corrects : [[concept-memex]], [[entity-vannevar-bush]], [[src-as-we-may-think]]
- Exemples INTERDITS : [[Memex]], [[Vannevar Bush]], [[mémex]]
- Ne pas inventer de liens vers des pages qui n'existent pas encore"""


_PROMPT_SOURCE_PAGE = """\
Voici le contenu d'une source à intégrer dans le wiki.

Source : {source_name}
Date d'ingestion : {today}

---
{content}
---

Produis une page wiki au format Markdown strict :

```yaml
---
title: <titre complet de la source>
category: source
tags: [<3 à 6 tags pertinents>]
created: {today}
sources: []
---
```

# <Titre>

## Résumé

<3 à 5 paragraphes résumant les idées principales>

## Points clés

- <point 1>
- <point 2>
- …

## Concepts et entités mentionnés

Liste les concepts abstraits et entités (personnes, outils, organisations) \
importants, un par ligne, au format :
- concept: <nom du concept>
- entité: <nom de l'entité>

## Liens internes suggérés

<Liens wiki [[slug]] vers des pages existantes si pertinent, sinon "Aucun">
"""


_PROMPT_CONCEPT_PAGE = """\
Tu enrichis la page wiki du concept "{concept}" à partir d'une nouvelle source.

Page actuelle (peut être vide si nouvelle page) :
---
{existing}
---

Source qui mentionne ce concept :
  Titre : {source_title}
  Slug  : {source_slug}
  Extrait pertinent :
---
{excerpt}
---

Produis la page wiki complète mise à jour, format Markdown strict :

```yaml
---
title: <Nom du concept>
category: concept
tags: [<tags>]
created: <date originale ou {today}>
sources: [{source_slug}<, anciens sources>]
---
```

# <Nom du concept>

<Corps de la page : définition, contexte, importance. 2 à 4 paragraphes.>

## Liens

<Liens [[slug]] vers pages connexes>
"""


_PROMPT_ENTITY_PAGE = """\
Tu enrichis la page wiki de l'entité "{entity}" à partir d'une nouvelle source.

Page actuelle (peut être vide si nouvelle page) :
---
{existing}
---

Source qui mentionne cette entité :
  Titre : {source_title}
  Slug  : {source_slug}
  Extrait pertinent :
---
{excerpt}
---

Produis la page wiki complète mise à jour, format Markdown strict :

```yaml
---
title: <Nom de l'entité>
category: entité
tags: [<tags>]
created: <date originale ou {today}>
sources: [{source_slug}<, anciens sources>]
---
```

# <Nom de l'entité>

<Corps : qui/quoi, rôle, importance dans le contexte du wiki. 1 à 3 paragraphes.>

## Liens

<Liens [[slug]] vers pages connexes>
"""


# ---------------------------------------------------------------------------
# Parseurs de la sortie LLM
# ---------------------------------------------------------------------------

def _parse_frontmatter_block(llm_output: str) -> str:
    """Extrait le contenu Markdown brut depuis la sortie du LLM.

    Le LLM peut encapsuler la page dans un bloc ```markdown … ```.
    """
    # Retirer le bloc ```markdown ... ``` si présent
    match = re.search(r"```(?:markdown|yaml)?\n(.*?)```", llm_output, re.DOTALL)
    if match:
        # Si c'était juste le frontmatter, concaténer avec le reste
        rest = llm_output[match.end():].strip()
        if rest:
            return match.group(1) + "\n" + rest
        return match.group(1)
    return llm_output.strip()


def _extract_items(llm_output: str, prefix: str) -> list[str]:
    """Extrait les lignes `- prefix: <valeur>` de la sortie LLM."""
    pattern = re.compile(rf"^-\s+{re.escape(prefix)}:\s+(.+)$", re.MULTILINE | re.IGNORECASE)
    return [m.group(1).strip() for m in pattern.finditer(llm_output)]


def _extract_title_from_page(content: str) -> str:
    """Lit le titre depuis le frontmatter YAML d'une page."""
    try:
        post = frontmatter.loads(content)
        return str(post.get("title", ""))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Moteur d'ingestion
# ---------------------------------------------------------------------------

_DEFAULT_RAW = Path.home() / "Secretarius" / "Wiki_LM" / "raw"


class Ingestor:
    def __init__(
        self,
        wiki_path: str | Path,
        llm: LLM | None = None,
        raw_path: str | Path | None = None,
    ) -> None:
        self.wiki_root = Path(wiki_path)
        self.wiki_dir = self.wiki_root / "wiki"
        # raw/ hors du vault Obsidian — non synchronisé, local uniquement
        self.raw_dir = Path(
            raw_path
            or os.environ.get("WIKI_RAW_PATH", str(_DEFAULT_RAW))
        )

        for d in (self.wiki_dir, self.raw_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.llm = llm or LLM()
        self.today = _today()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_batch(self, url_file: str | Path, max_concepts: int = 5) -> list[str]:
        """Ingère toutes les URLs listées dans un fichier texte.

        Format du fichier : une URL par ligne, les lignes vides et commençant
        par '#' sont ignorées.
        Retourne la liste des slugs créés.
        """
        path = Path(url_file)
        urls = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        print(f"[ingest] Batch : {len(urls)} URL(s) dans {path.name}")
        slugs = []
        for i, url in enumerate(urls, 1):
            print(f"\n[ingest] ({i}/{len(urls)}) {url}")
            try:
                slug = self.ingest(url, max_concepts=max_concepts)
                slugs.append(slug)
            except Exception as e:
                print(f"[ingest] ERREUR sur {url} : {e}")
        return slugs

    def ingest(self, source: str, slug: str = "", max_concepts: int = 5) -> str:
        """Ingère une source et retourne le slug de la page créée."""
        print(f"[ingest] Lecture de la source : {source}")
        content, title = _read_source(source)

        # Slug de la page source
        src_slug = f"src-{slug or _slugify(title)}"
        print(f"[ingest] Slug : {src_slug}")

        # 1. Sauvegarder dans raw/ (immutable)
        self._save_raw(source, content, src_slug)

        # 2. Générer la page source
        print("[ingest] Génération de la page source…")
        source_page_md = self._generate_source_page(content, title)
        source_page_md = _parse_frontmatter_block(source_page_md)
        source_title = _extract_title_from_page(source_page_md) or title
        self._write_wiki_page(src_slug, source_page_md)

        # 3. Extraire concepts et entités
        concepts = _extract_items(source_page_md, "concept")[:max_concepts]
        entities = _extract_items(source_page_md, "entité")[:max_concepts]
        print(f"[ingest] Concepts : {concepts}")
        print(f"[ingest] Entités  : {entities}")

        # 4. Mettre à jour / créer les pages de concepts et entités
        for concept in concepts:
            self._update_concept_page(concept, source_title, src_slug, content)

        for entity in entities:
            self._update_entity_page(entity, source_title, src_slug, content)

        # 5. Mettre à jour index.md
        self._update_index(src_slug, source_title, "source")

        # 6. Appender à log.md
        self._append_log("ingest", source_title)

        print(f"[ingest] Terminé → wiki/{src_slug}.md")
        return src_slug

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _save_raw(self, source: str, content: str, slug: str) -> None:
        """Sauvegarde la source dans raw/ (immutable).

        - URL  → fichier .url contenant l'URL et la date de récupération
        - Fichier local → copie du fichier original (ou .txt si contenu déjà extrait)
        """
        if source.startswith("http://") or source.startswith("https://"):
            raw_path = self.raw_dir / f"{slug}.url"
            if not raw_path.exists():
                raw_path.write_text(
                    f"url: {source}\nfetched: {self.today}\n", encoding="utf-8"
                )
        else:
            src_path = Path(source)
            # Copier le fichier original si possible
            raw_path = self.raw_dir / f"{slug}{src_path.suffix or '.txt'}"
            if not raw_path.exists():
                import shutil
                try:
                    shutil.copy2(src_path, raw_path)
                except Exception:
                    raw_path.with_suffix(".txt").write_text(content, encoding="utf-8")

    def _generate_source_page(self, content: str, source_name: str) -> str:
        prompt = _PROMPT_SOURCE_PAGE.format(
            source_name=source_name,
            today=self.today,
            content=_truncate(content),
        )
        return self.llm.complete(prompt, system=_SYSTEM_INGEST, max_tokens=3000)

    def _update_concept_page(
        self, concept: str, source_title: str, src_slug: str, full_content: str
    ) -> None:
        concept_slug = f"concept-{_slugify(concept)}"
        page_path = self.wiki_dir / f"{concept_slug}.md"
        existing = page_path.read_text(encoding="utf-8") if page_path.exists() else ""

        # Extrait contextuel : premières occurrences du concept dans la source
        excerpt = self._find_excerpt(full_content, concept)

        prompt = _PROMPT_CONCEPT_PAGE.format(
            concept=concept,
            existing=existing or "(nouvelle page)",
            source_title=source_title,
            source_slug=src_slug,
            excerpt=excerpt,
            today=self.today,
        )
        print(f"[ingest] Mise à jour concept : {concept_slug}")
        page_md = self.llm.complete(prompt, system=_SYSTEM_INGEST, max_tokens=1500)
        page_md = _parse_frontmatter_block(page_md)
        self._write_wiki_page(concept_slug, page_md)

        if not existing:
            self._update_index(concept_slug, concept, "concept")

    def _update_entity_page(
        self, entity: str, source_title: str, src_slug: str, full_content: str
    ) -> None:
        entity_slug = f"entity-{_slugify(entity)}"
        page_path = self.wiki_dir / f"{entity_slug}.md"
        existing = page_path.read_text(encoding="utf-8") if page_path.exists() else ""

        excerpt = self._find_excerpt(full_content, entity)

        prompt = _PROMPT_ENTITY_PAGE.format(
            entity=entity,
            existing=existing or "(nouvelle page)",
            source_title=source_title,
            source_slug=src_slug,
            excerpt=excerpt,
            today=self.today,
        )
        print(f"[ingest] Mise à jour entité : {entity_slug}")
        page_md = self.llm.complete(prompt, system=_SYSTEM_INGEST, max_tokens=1500)
        page_md = _parse_frontmatter_block(page_md)
        self._write_wiki_page(entity_slug, page_md)

        if not existing:
            self._update_index(entity_slug, entity, "entité")

    def _write_wiki_page(self, slug: str, content: str) -> None:
        content = _fix_mojibake(content)
        known_slugs = {p.stem for p in self.wiki_dir.glob("*.md")}
        content = _normalize_links(content, known_slugs)
        path = self.wiki_dir / f"{slug}.md"
        path.write_text(content, encoding="utf-8")

    def _update_index(self, slug: str, title: str, category: str) -> None:
        """Ajoute ou met à jour la ligne du slug dans index.md."""
        index_path = self.wiki_dir / "index.md"
        if not index_path.exists():
            return

        text = index_path.read_text(encoding="utf-8")
        entry = f"- [[{slug}]] | {category} | {title}"

        # Remplace l'entrée existante ou ajoute en fin de fichier
        pattern = re.compile(rf"^- \[\[{re.escape(slug)}\]\].*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(entry, text)
        else:
            text = text.rstrip() + "\n" + entry + "\n"

        index_path.write_text(text, encoding="utf-8")

    def _append_log(self, operation: str, title: str) -> None:
        log_path = self.wiki_dir / "log.md"
        if not log_path.exists():
            return
        entry = f"\n## [{self.today}] {operation} | {title}\n"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(entry)

    @staticmethod
    def _find_excerpt(text: str, term: str, window: int = 400) -> str:
        """Retourne un extrait du texte autour de la première occurrence du terme."""
        idx = text.lower().find(term.lower())
        if idx == -1:
            return text[:window]
        start = max(0, idx - window // 2)
        end = min(len(text), idx + window // 2)
        excerpt = text[start:end]
        if start > 0:
            excerpt = "…" + excerpt
        if end < len(text):
            excerpt += "…"
        return excerpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingère une source dans Wiki_LM")
    parser.add_argument("source", help="Chemin fichier ou URL https://…")
    parser.add_argument("--slug", default="", help="Slug personnalisé (sans préfixe src-)")
    parser.add_argument("--top-entities", type=int, default=5, help="Nombre max de concepts/entités à enrichir")
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM")),
        help="Chemin vers Wiki_LM (défaut : $WIKI_PATH ou ~/Documents/Arbath/Wiki_LM)",
    )
    parser.add_argument("--backend", default="", help="Backend LLM : claude | ollama | openai")
    parser.add_argument("--model", default="", help="Modèle LLM (ex: qwen2.5:7b)")
    args = parser.parse_args()

    llm = LLM(backend=args.backend, model=args.model) if (args.backend or args.model) else LLM()
    ing = Ingestor(args.wiki, llm=llm)

    # Détection automatique : fichier local dont toutes les lignes non-vides
    # sont des URLs → batch mode
    source = args.source
    if not source.startswith("http") and Path(source).is_file():
        lines = [
            l.strip() for l in Path(source).read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        if lines and all(l.startswith("http://") or l.startswith("https://") for l in lines):
            slugs = ing.ingest_batch(source, max_concepts=args.top_entities)
            print(f"\n{len(slugs)} page(s) créée(s) : {', '.join(slugs)}")
            return

    slug = ing.ingest(source, slug=args.slug, max_concepts=args.top_entities)
    print(f"\nPage créée : wiki/{slug}.md")


if __name__ == "__main__":
    main()
