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

import hashlib

import frontmatter

from llm import LLM
from wiki_lookup import WikiLookup


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


def _read_arxiv(url: str) -> str:
    """Récupère les métadonnées d'un papier ArXiv via l'API Atom."""
    import urllib.request
    import urllib.parse
    import re as _re

    # Extraire l'identifiant ArXiv depuis l'URL (pdf ou abs)
    # ex. https://arxiv.org/pdf/1204.1550.pdf  → 1204.1550
    # ex. https://arxiv.org/abs/1607.01668v2   → 1607.01668v2
    m = _re.search(r"arxiv\.org/(?:pdf|abs)/([^/?#]+?)(?:\.pdf)?$", url, _re.IGNORECASE)
    if not m:
        raise ValueError(f"Identifiant ArXiv non trouvé dans : {url}")
    arxiv_id = m.group(1)

    api_url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    req = urllib.request.Request(api_url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml = resp.read().decode("utf-8", errors="replace")

    def _tag(name: str) -> str:
        m2 = _re.search(rf"<{name}[^>]*>(.*?)</{name}>", xml, _re.DOTALL)
        return m2.group(1).strip() if m2 else ""

    title = _tag("title").replace("\n", " ").strip()
    # Le premier <title> est le titre de l'entrée Atom ("ArXiv Query…"), le second est le papier
    titles = _re.findall(r"<title[^>]*>(.*?)</title>", xml, _re.DOTALL)
    paper_title = titles[1].strip().replace("\n", " ") if len(titles) > 1 else title

    summary = _tag("summary").replace("\n", " ").strip()
    authors = ", ".join(_re.findall(r"<name>(.*?)</name>", xml))
    categories = " ".join(_re.findall(r'term="([^"]+)"', xml))
    published = _tag("published")[:10]

    return (
        f"# {paper_title}\n\n"
        f"Auteurs : {authors}\n"
        f"Publié : {published}\n"
        f"Catégories : {categories}\n\n"
        f"## Résumé\n\n{summary}\n"
    )


def _is_binary_content(text: str) -> bool:
    """Détecte un contenu PDF non décodé (binaire, flux d'objets).

    N'est significatif que pour des contenus longs (> 200 chars) — les sources
    courtes mais lisibles (notes, stubs) ne doivent pas être marquées illisibles.
    """
    if len(text.strip()) < 200:
        return False
    sample = text[:2000]
    printable = sum(1 for c in sample if c.isprintable() or c in "\n\r\t")
    return printable / len(sample) < 0.70


def _read_url(url: str) -> str:
    """Télécharge une page web et retourne le texte brut.

    Les URLs Wikipedia sont traitées via l'API REST (texte propre, sans HTML).
    Les URLs ArXiv (pdf ou abs) sont traitées via l'API Atom.
    Les autres URLs passent par un extracteur HTML léger.
    """
    import urllib.request

    # ArXiv : API Atom → métadonnées structurées
    if "arxiv.org/pdf/" in url or "arxiv.org/abs/" in url:
        return _read_arxiv(url)

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
    text = ""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(str(path))
        except ImportError:
            raise ImportError(
                "Installez pypdf ou pdfminer.six pour lire les PDF : pip install pypdf"
            )
    if _is_binary_content(text):
        raise ValueError(
            f"PDF illisible (contenu binaire ou chiffré) : {path.name}. "
            "Vérifiez que le fichier n'est pas protégé ou corrompu."
        )
    return text


def _truncate(text: str, max_chars: int = 12_000) -> str:
    """Tronque le texte source pour ne pas dépasser le contexte du LLM."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[… texte tronqué à {max_chars} caractères …]"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_url_from_file(path: Path) -> str:
    """Extrait l'URL d'un fichier .url (les deux formats)."""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("url:"):
            return line[4:].strip()
        if line.startswith("http://") or line.startswith("https://"):
            return line
    return ""


def _dedup_files(files: list[Path]) -> list[Path]:
    """Déduplique une liste de fichiers par contenu.

    Pour les .url : clé = URL normalisée (ignore paramètres de tracking).
    Pour les autres : clé = SHA256 du fichier.
    Garde le fichier le plus ancien (premier par ordre alphabétique/timestamp).
    """
    seen: dict[str, Path] = {}
    result: list[Path] = []
    for f in files:
        if f.suffix.lower() == ".url":
            key = _extract_url_from_file(f)
            # Normalisation minimale : supprimer fragment et trailing slash
            key = key.split("#")[0].rstrip("/")
        else:
            try:
                key = _file_hash(f)
            except OSError:
                result.append(f)
                continue
        if not key:
            result.append(f)
            continue
        if key in seen:
            print(f"[ingest] Doublon ignoré : {f.name} (même contenu que {seen[key].name})")
        else:
            seen[key] = f
            result.append(f)
    return result


def _fix_mojibake(text: str) -> str:
    """Corrige le mojibake latin-1→UTF-8 (entitÃ© → entité).

    Se produit quand une API renvoie du UTF-8 interprété comme latin-1.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _linkify_concepts_section(
    content: str, concepts: list[str] | None = None, entities: list[str] | None = None
) -> str:
    """Remplace les noms nus par [[c-slug]] / [[e-slug]] dans la section concepts/entités.

    Opère directement sur le texte — indépendant de la liste concepts/entities,
    ce qui évite les omissions dues à la troncature max_concepts.
    """
    def _replace(m: re.Match, prefix: str) -> str:
        leader, name = m.group(1), m.group(2).strip()
        if name.startswith("[["):
            return m.group(0)
        return f"{leader}[[{prefix}{_slugify(name)}]]"

    content = re.sub(
        r"^(-\s+concept:\s+)([^\[].+)$",
        lambda m: _replace(m, "c-"),
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^(-\s+entit[eé]:\s+)([^\[].+)$",
        lambda m: _replace(m, "e-"),
        content,
        flags=re.MULTILINE,
    )
    return content


_YAML_UNSAFE = re.compile(r":\s|[#{}[\]]")


def _fix_yaml_scalars(content: str) -> str:
    """Quote les valeurs scalaires YAML non-quotées qui contiennent des caractères spéciaux.

    Cible principalement le champ `title` que le LLM écrit souvent sans guillemets
    même quand la valeur contient `: ` (ex. "Guest Post: Enhancing...").
    """
    def _quote(m: re.Match) -> str:
        key, value = m.group(1), m.group(2).strip()
        if not value or value[0] in ('"', "'"):
            return m.group(0)
        if _YAML_UNSAFE.search(value):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f"{key}: \"{escaped}\""
        return m.group(0)

    # Appliqué uniquement aux scalaires simples (pas aux listes ni aux valeurs vides)
    return re.sub(
        r"^(title|description):\s+(.+)$",
        _quote,
        content,
        flags=re.MULTILINE,
    )


def _merge_tags(page_md: str, extra_tags: list[str]) -> str:
    """Injecte extra_tags dans le frontmatter YAML sans doublons."""
    if not extra_tags:
        return page_md
    try:
        post = frontmatter.loads(page_md)
        existing = list(post.get("tags", []))
        post["tags"] = existing + [t for t in extra_tags if t not in existing]
        return frontmatter.dumps(post)
    except Exception:
        return page_md


def _normalize_links(content: str, known_slugs: set[str]) -> str:
    """Normalise les [[liens]] LLM vers des slugs réels du wiki.

    Stratégie :
    1. Si le lien correspond déjà à un slug connu → inchangé
    2. Sinon slugifie le texte et cherche une correspondance avec préfixes
    3. Sinon garde le texte slugifié (lien vers page future)
    """
    def _resolve(m: re.Match) -> str:
        raw = m.group(1).strip()
        # Supprimer les annotations parenthétiques ajoutées par le LLM
        # ex. "concept-memex (Memex)" → "concept-memex"
        raw = re.sub(r"\s*\([^)]*\)", "", raw).strip()
        slug = _slugify(raw)

        if slug in known_slugs:
            return f"[[{slug}]]"
        for prefix in ("c-", "e-", "src-", "synth-"):
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
- Format [[slug-exact]] avec préfixe selon la catégorie : c-, e-, src-, synth-
- Slugs : minuscules, tirets, sans accents, sans espaces
- Exemples corrects : [[c-memex]], [[e-vannevar-bush]], [[src-as-we-may-think]]
- Exemples INTERDITS : [[Memex]], [[Vannevar Bush]], [[mémex]]
- Ne pas inventer de liens vers des pages qui n'existent pas encore"""


_PROMPT_SOURCE_PAGE = """\
Voici le contenu d'une source à intégrer dans le wiki.

Source : {source_name}
Date d'ingestion : {today}
{required_tags_line}
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


_WIKI_ANCHOR_RE = re.compile(
    r"\n*>\s*\*\*R[eé]f[eé]rence Wikipedia\*\*.*",
    re.DOTALL | re.IGNORECASE,
)


def _append_wiki_section(page_md: str, wp: dict) -> str:
    """Ajoute une section ## Extrait Wikipedia en bas de page."""
    url_line = f"\n*[Source Wikipedia]({wp['url']})*\n" if wp.get("url") else ""
    section = f"\n## Extrait Wikipedia\n\n{wp['abstract']}\n{url_line}"
    return page_md.rstrip() + "\n" + section


def _strip_wiki_anchor(content: str) -> str:
    """Supprime le bloc ancre Wikipedia si le LLM l'a recopié dans sa sortie."""
    return _WIKI_ANCHOR_RE.sub("", content).rstrip() + "\n"


_PROMPT_WIKI_ANCHOR = """\

Référence Wikipedia (contexte factuel — utiliser comme base, ne pas recopier verbatim) :
{abstract}
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
status: nouveau
---
```

# <Nom du concept>

<Corps de la page : définition factuelle et historique du concept, 2 à 4 paragraphes.>

RÈGLES STRICTES :
- Rester factuel et encyclopédique. Ne pas interpréter l'importance du concept pour ce wiki.
- Ne pas écrire de phrases du type "dans le contexte de ce wiki" ou "ce concept est important car".
- Ne pas spéculer sur les usages futurs ou la pertinence pour l'utilisateur.
- Si le concept a plusieurs acceptions distinctes (ex. Jung peut désigner un psychanalyste \
ou une querelle théorique), les traiter séparément sans en privilégier une.
- S'en tenir strictement à ce que l'extrait source permet d'affirmer.

## Liens

<Liens [[slug]] vers pages connexes, uniquement si clairement établis par la source>
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
status: nouveau
---
```

# <Nom de l'entité>

<Corps : identité factuelle, dates, domaine d'activité, faits établis. 1 à 3 paragraphes.>

RÈGLES STRICTES :
- Rester factuel. Ne pas écrire "dans le contexte de ce wiki" ni juger de l'importance pour l'utilisateur.
- Ne pas spéculer au-delà de ce que l'extrait source permet d'affirmer.
- Si l'entité a plusieurs facettes distinctes (ex. une personne connue pour plusieurs œuvres \
ou controverses), les mentionner toutes sans en privilégier une arbitrairement.

## Liens

<Liens [[slug]] vers pages connexes, uniquement si clairement établis par la source>
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
        self._wiki_lookup = WikiLookup(wiki_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Fichier manifeste des fichiers raw/ déjà ingérés
    _MANIFEST = ".ingested"

    def _load_manifest(self) -> set[str]:
        """Retourne l'ensemble des noms de fichiers déjà ingérés."""
        path = self.raw_dir / self._MANIFEST
        if not path.exists():
            return set()
        return set(path.read_text(encoding="utf-8").splitlines())

    def _mark_ingested(self, filename: str) -> None:
        """Ajoute un fichier au manifeste."""
        path = self.raw_dir / self._MANIFEST
        with path.open("a", encoding="utf-8") as f:
            f.write(filename + "\n")

    def _reset_wiki(self) -> None:
        """Supprime toutes les pages non-immuables et réinitialise index.md.

        Appelé avant une reconstruction complète (--raw --force).
        Les pages avec status: immuable sont préservées.
        Les fichiers meta (log.md, schema.md) sont préservés.
        """
        _META = {"index.md", "log.md", "schema.md"}
        _PAGE_PREFIXES = ("src-", "c-", "e-", "synth-")
        deleted = 0
        for page in self.wiki_dir.glob("*.md"):
            if page.name in _META:
                continue
            if not any(page.stem.startswith(p) for p in _PAGE_PREFIXES):
                continue
            try:
                post = frontmatter.loads(page.read_text(encoding="utf-8"))
                if post.get("status") == "immuable":
                    print(f"[ingest] Page immuable conservée : {page.name}")
                    continue
            except Exception:
                pass
            page.unlink()
            deleted += 1
        print(f"[ingest] reset : {deleted} page(s) supprimée(s)")
        # Réinitialiser index.md
        (self.wiki_dir / "index.md").write_text("# Index\n\n", encoding="utf-8")
        print("[ingest] index.md réinitialisé")

    def ingest_raw_dir(self, max_concepts: int = 5, force: bool = False) -> list[str]:
        """Ingère les fichiers de raw/ non encore traités (mode incrémental par défaut).

        - .url  → lit l'URL et ingère la page web
        - .md   → ingère comme note locale
        - .pdf  → ingère le PDF
        - Autres extensions reconnues (.txt, .html) → ingère comme texte

        force=True : réinitialise le wiki (pages + index) et retraite tout.
        Retourne la liste des slugs créés ou mis à jour.
        """
        if force:
            self._reset_wiki()
        already_done = set() if force else self._load_manifest()

        files = sorted(self.raw_dir.iterdir())
        processable = [
            f for f in files
            if f.is_file()
            and f.suffix.lower() in (".url", ".md", ".pdf", ".txt", ".html")
            and f.name != self._MANIFEST
        ]

        pending = _dedup_files([f for f in processable if f.name not in already_done])
        skipped = len(processable) - len(pending)

        if skipped:
            print(f"[ingest] raw/ : {len(pending)} nouveau(x) fichier(s) "
                  f"({skipped} déjà ingéré(s), ignorés)")
        else:
            print(f"[ingest] raw/ : {len(pending)} fichier(s) à traiter")

        if not pending:
            print("[ingest] Rien de nouveau à ingérer.")
            return []

        slugs = []
        for i, path in enumerate(pending, 1):
            print(f"\n[ingest] ({i}/{len(pending)}) {path.name}")
            try:
                if path.suffix.lower() == ".url":
                    url = self._parse_url_file(path)
                    if not url:
                        print(f"[ingest] Fichier .url vide ou invalide : {path.name}")
                        continue
                    user_tags = self._parse_raw_tags(path)
                    slug = self.ingest(url, max_concepts=max_concepts, extra_tags=user_tags or None)
                else:
                    user_tags = self._parse_raw_tags(path)
                    slug = self.ingest(str(path), max_concepts=max_concepts, extra_tags=user_tags or None)
                slugs.append(slug)
                self._mark_ingested(path.name)
            except Exception as e:
                print(f"[ingest] ERREUR sur {path.name} : {e}")

        return slugs

    @staticmethod
    def _parse_url_file(path: Path) -> str:
        """Lit l'URL depuis un fichier .url (deux formats supportés).

        Format capture.py : première ligne = URL nue
        Format _save_raw() : ligne "url: https://..."
        """
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("url:"):
                return line[4:].strip()
            if line.startswith("http://") or line.startswith("https://"):
                return line
        return ""

    @staticmethod
    def _parse_raw_tags(path: Path) -> list[str]:
        """Lit la ligne `tags: tag1, tag2` d'un fichier raw si présente."""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.lower().startswith("tags:"):
                raw = line[5:].strip()
                return [t.strip() for t in raw.split(",") if t.strip()]
        return []

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

    def ingest(
        self,
        source: str,
        slug: str = "",
        max_concepts: int = 5,
        extra_tags: list[str] | None = None,
    ) -> str:
        """Ingère une source et retourne le slug de la page créée."""
        print(f"[ingest] Lecture de la source : {source}")
        content, title = _read_source(source)

        # Slug de la page source (éviter le double préfixe si le titre commence déjà par src-)
        base_slug = slug or _slugify(title)
        base_slug = re.sub(r"^src-", "", base_slug)
        src_slug = f"src-{base_slug}"
        print(f"[ingest] Slug : {src_slug}")

        # 1. Sauvegarder dans raw/ (immutable)
        self._save_raw(source, content, src_slug)

        # 2. Générer la page source (ou page "illisible" si contenu dégradé)
        if _is_binary_content(content):
            print(f"[ingest] Contenu illisible → page stub")
            source_page_md = self._stub_page(title, src_slug, extra_tags)
            source_title = title
            self._write_wiki_page(src_slug, source_page_md)
            self._update_index(src_slug, source_title, "source")
            self._append_log("illisible", source_title)
            self._rebuild_tags_index()
            print(f"[ingest] Stub créé → wiki/{src_slug}.md")
            return src_slug

        print("[ingest] Génération de la page source…")
        source_page_md = self._generate_source_page(content, title, extra_tags=extra_tags)
        source_page_md = _parse_frontmatter_block(source_page_md)
        if extra_tags:
            source_page_md = _merge_tags(source_page_md, extra_tags)
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

        # 4b. Réécrire la page source avec des [[liens]] dans la section concepts/entités
        if concepts or entities:
            source_page_md = _linkify_concepts_section(source_page_md, concepts, entities)
            self._write_wiki_page(src_slug, source_page_md)

        # 5. Mettre à jour index.md
        self._update_index(src_slug, source_title, "source")

        # 6. Appender à log.md
        self._append_log("ingest", source_title)

        # 7. Reconstruire l'index des tags
        self._rebuild_tags_index()

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
            # Si le fichier source est déjà dans raw/, pas de copie nécessaire
            if src_path.parent.resolve() == self.raw_dir.resolve():
                return
            raw_path = self.raw_dir / f"{slug}{src_path.suffix or '.txt'}"
            if not raw_path.exists():
                # Ne pas copier si un fichier de même contenu existe déjà dans raw/
                src_hash = _file_hash(src_path) if src_path.exists() else None
                already_there = src_hash and any(
                    p != src_path and _file_hash(p) == src_hash
                    for p in self.raw_dir.iterdir()
                    if p.is_file() and p.suffix == src_path.suffix
                )
                if not already_there:
                    import shutil
                    try:
                        shutil.copy2(src_path, raw_path)
                    except Exception:
                        raw_path.with_suffix(".txt").write_text(content, encoding="utf-8")

    def _stub_page(self, title: str, slug: str, extra_tags: list[str] | None = None) -> str:
        """Génère une page minimale pour une source illisible (PDF chiffré, binaire…)."""
        tags = ["illisible"] + (extra_tags or [])
        tags_yaml = "[" + ", ".join(tags) + "]"
        return (
            f"---\n"
            f"title: \"{title}\"\n"
            f"category: source\n"
            f"tags: {tags_yaml}\n"
            f"created: {self.today}\n"
            f"sources: []\n"
            f"status: illisible\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"Source illisible — contenu binaire, chiffré ou corrompu.\n"
            f"À réingérer manuellement si le document devient accessible.\n"
        )

    def _generate_source_page(
        self, content: str, source_name: str, extra_tags: list[str] | None = None
    ) -> str:
        required_tags_line = ""
        if extra_tags:
            tags_str = ", ".join(extra_tags)
            required_tags_line = f"Tags requis (à inclure dans le frontmatter) : {tags_str}\n"
        prompt = _PROMPT_SOURCE_PAGE.format(
            source_name=source_name,
            today=self.today,
            content=_truncate(content),
            required_tags_line=required_tags_line,
        )
        return self.llm.complete(prompt, system=_SYSTEM_INGEST, max_tokens=3000)

    def _rebuild_tags_index(self) -> None:
        """Reconstruit tags.md : index tag → pages du wiki."""
        _META = {"index.md", "log.md", "schema.md", "tags.md"}
        tag_map: dict[str, list[tuple[str, str, str]]] = {}
        for page in sorted(self.wiki_dir.glob("*.md")):
            if page.name in _META:
                continue
            try:
                post = frontmatter.loads(page.read_text(encoding="utf-8"))
                tags = post.get("tags", [])
                title = str(post.get("title", page.stem))
                category = str(post.get("category", ""))
                for tag in tags:
                    tag_map.setdefault(str(tag), []).append((page.stem, category, title))
            except Exception:
                continue

        lines = ["# Tags\n"]
        for tag in sorted(tag_map):
            pages = tag_map[tag]
            lines.append(f"\n## {tag} ({len(pages)})\n\n")
            for slug, cat, title in sorted(pages):
                lines.append(f"- [[{slug}]] | {cat} | {title}\n")

        (self.wiki_dir / "tags.md").write_text("".join(lines), encoding="utf-8")

    def _update_concept_page(
        self, concept: str, source_title: str, src_slug: str, full_content: str
    ) -> None:
        concept_slug = f"c-{_slugify(concept)}"
        page_path = self.wiki_dir / f"{concept_slug}.md"
        existing = page_path.read_text(encoding="utf-8") if page_path.exists() else ""

        # Extrait contextuel : premières occurrences du concept dans la source
        excerpt = self._find_excerpt(full_content, concept)

        wiki_anchor = ""
        wp = self._wiki_lookup.lookup(concept)
        if wp:
            wiki_anchor = _PROMPT_WIKI_ANCHOR.format(abstract=wp["abstract"])
            print(f"[ingest] Wikipedia trouvé pour concept '{concept}' ({wp['lang']})")

        prompt = _PROMPT_CONCEPT_PAGE.format(
            concept=concept,
            existing=existing or "(nouvelle page)",
            source_title=source_title,
            source_slug=src_slug,
            excerpt=excerpt,
            today=self.today,
        ) + wiki_anchor
        print(f"[ingest] Mise à jour concept : {concept_slug}")
        page_md = self.llm.complete(prompt, system=_SYSTEM_INGEST, max_tokens=1500)
        page_md = _parse_frontmatter_block(page_md)
        page_md = _strip_wiki_anchor(page_md)
        if wp:
            page_md = _append_wiki_section(page_md, wp)
        self._write_wiki_page(concept_slug, page_md)

        if not existing:
            self._update_index(concept_slug, concept, "concept")

    def _update_entity_page(
        self, entity: str, source_title: str, src_slug: str, full_content: str
    ) -> None:
        entity_slug = f"e-{_slugify(entity)}"
        page_path = self.wiki_dir / f"{entity_slug}.md"
        existing = page_path.read_text(encoding="utf-8") if page_path.exists() else ""

        excerpt = self._find_excerpt(full_content, entity)

        wiki_anchor = ""
        wp = self._wiki_lookup.lookup(entity)
        if wp:
            wiki_anchor = _PROMPT_WIKI_ANCHOR.format(abstract=wp["abstract"])
            print(f"[ingest] Wikipedia trouvé pour entité '{entity}' ({wp['lang']})")

        prompt = _PROMPT_ENTITY_PAGE.format(
            entity=entity,
            existing=existing or "(nouvelle page)",
            source_title=source_title,
            source_slug=src_slug,
            excerpt=excerpt,
            today=self.today,
        ) + wiki_anchor
        print(f"[ingest] Mise à jour entité : {entity_slug}")
        page_md = self.llm.complete(prompt, system=_SYSTEM_INGEST, max_tokens=1500)
        page_md = _parse_frontmatter_block(page_md)
        page_md = _strip_wiki_anchor(page_md)
        if wp:
            page_md = _append_wiki_section(page_md, wp)
        self._write_wiki_page(entity_slug, page_md)

        if not existing:
            self._update_index(entity_slug, entity, "entité")

    def _write_wiki_page(self, slug: str, content: str) -> None:
        path = self.wiki_dir / f"{slug}.md"
        # Respecter le statut immuable : ne jamais écraser une page verrouillée
        if path.exists():
            try:
                existing = frontmatter.loads(path.read_text(encoding="utf-8"))
                if existing.get("status") == "immuable":
                    print(f"[ingest] Page immuable, ignorée : {slug}.md")
                    return
            except Exception:
                pass
        content = _fix_mojibake(content)
        content = _fix_yaml_scalars(content)
        known_slugs = {p.stem for p in self.wiki_dir.glob("*.md")}
        content = _normalize_links(content, known_slugs)
        # Re-sérialiser via PyYAML pour normaliser le frontmatter
        try:
            post = frontmatter.loads(content)
            content = frontmatter.dumps(post)
        except Exception:
            pass
        path.write_text(content, encoding="utf-8")

    def _update_index(self, slug: str, title: str, category: str) -> None:
        """Ajoute ou met à jour la ligne du slug dans index.md."""
        index_path = self.wiki_dir / "index.md"
        if not index_path.exists():
            index_path.write_text("# Index\n\n", encoding="utf-8")

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
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"\n## [{ts}] {operation} | {title}\n"
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
    parser.add_argument(
        "source", nargs="?", default="",
        help="Chemin fichier ou URL https://… (omis si --raw)"
    )
    parser.add_argument("--slug", default="", help="Slug personnalisé (sans préfixe src-)")
    parser.add_argument("--top-entities", type=int, default=5, help="Nombre max de concepts/entités à enrichir")
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM")),
        help="Chemin vers Wiki_LM (défaut : $WIKI_PATH ou ~/Documents/Arbath/Wiki_LM)",
    )
    parser.add_argument("--backend", default="", help="Backend LLM : claude | ollama | openai")
    parser.add_argument("--model", default="", help="Modèle LLM (ex: qwen2.5:7b)")
    parser.add_argument(
        "--raw", action="store_true",
        help="Ingère les nouveaux fichiers de raw/ (.url, .md, .pdf, .txt)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Avec --raw : retraite tous les fichiers, même déjà ingérés",
    )
    args = parser.parse_args()

    llm = LLM(backend=args.backend, model=args.model) if (args.backend or args.model) else LLM()
    ing = Ingestor(args.wiki, llm=llm)

    # Mode --raw : ingestion incrémentale (ou complète avec --force)
    if args.raw:
        slugs = ing.ingest_raw_dir(max_concepts=args.top_entities, force=args.force)
        print(f"\n{len(slugs)} page(s) créée(s) ou mises à jour : {', '.join(slugs)}")
        return

    if not args.source:
        parser.error("Fournir une source ou utiliser --raw")

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
