"""
Lookup Wikipedia avec trois backends (priorité décroissante) :

  1. ZIM local  — si un fichier *.zim est trouvé dans wiki_path/zim/ (hors ligne, instantané)
  2. Cache SQLite — résultats des appels API précédents (hors ligne après premier appel)
  3. API REST Wikipedia — premier appel réseau, résultat mis en cache

Usage :
    from wiki_lookup import WikiLookup
    wl = WikiLookup("/chemin/vers/Wiki_LM")
    result = wl.lookup("Gerard Salton")
    # -> {"title": ..., "abstract": ..., "url": ..., "lang": ...} | None

Fichiers ZIM attendus dans <wiki_path>/zim/ :
    wikipedia_fr_*.zim   → langue "fr"
    wikipedia_en_*.zim   → langue "en"
    (téléchargeables sur https://download.kiwix.org/zim/wikipedia/)
"""

from __future__ import annotations

import html.parser
import json
import re
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = "WikiLM/1.0 (personal wiki; python-urllib)"
_API = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"


# ---------------------------------------------------------------------------
# Extraction du texte brut depuis le HTML Wikipedia (ZIM)
# ---------------------------------------------------------------------------

class _TextExtractor(html.parser.HTMLParser):
    """Extrait le texte du premier paragraphe substantiel d'un article Wikipedia HTML."""

    _SKIP_TAGS = {"script", "style", "sup", "table", "figure", "figcaption"}

    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self._in_p = False
        self.paragraphs: list[str] = []
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP_TAGS:
            self._skip += 1
        elif tag == "p" and not self._skip:
            self._in_p = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag == "p" and self._in_p:
            self._in_p = False
            text = "".join(self._buf).strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) > 80:
                self.paragraphs.append(text)

    def handle_data(self, data: str) -> None:
        if self._in_p and not self._skip:
            self._buf.append(data)


def _html_to_abstract(html_bytes: bytes, max_paragraphs: int = 2) -> str:
    extractor = _TextExtractor()
    extractor.feed(html_bytes.decode("utf-8", errors="replace"))
    return " ".join(extractor.paragraphs[:max_paragraphs])


# ---------------------------------------------------------------------------
# Backend ZIM (libzim optionnel)
# ---------------------------------------------------------------------------

def _zim_files(zim_dir: Path) -> dict[str, Path]:
    """Retourne {lang: path} pour les fichiers ZIM trouvés."""
    result: dict[str, Path] = {}
    if not zim_dir.exists():
        return result
    for f in zim_dir.glob("*.zim"):
        # Conventions Kiwix : wikipedia_fr_all_mini_*.zim, wikipedia_en_all_mini_*.zim
        parts = f.stem.split("_")
        if len(parts) >= 2 and parts[0] == "wikipedia":
            result.setdefault(parts[1], f)  # parts[1] = "fr", "en", …
    return result


def _zim_lookup(title: str, lang: str, zim_path: Path) -> dict[str, Any] | None:
    try:
        from libzim.reader import Archive  # type: ignore
    except ImportError:
        return None
    try:
        archive = Archive(str(zim_path))
        # Wikipedia ZIM : articles sous "A/<Titre>" (underscores, URL-encodé)
        slug = title.replace(" ", "_")
        for candidate in (slug, slug.capitalize(), title.replace(" ", "_").title()):
            try:
                entry = archive.get_entry_by_path(f"A/{candidate}")
                item = entry.get_item()
                abstract = _html_to_abstract(bytes(item.content))
                if abstract:
                    wp_url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(candidate)}"
                    return {"title": title, "abstract": abstract, "url": wp_url, "lang": lang}
            except KeyError:
                continue
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Backend API Wikipedia
# ---------------------------------------------------------------------------

def _fetch_api(title: str, lang: str) -> dict[str, Any] | None:
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = _API.format(lang=lang, title=encoded)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        extract = (data.get("extract") or "").strip()
        if not extract:
            return None
        return {
            "title": data.get("title", title),
            "abstract": extract,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "lang": lang,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# WikiLookup
# ---------------------------------------------------------------------------

class WikiLookup:
    _DEFAULT_ZIM = Path.home() / "Secretarius" / "Wiki_LM" / "zim"

    def __init__(self, wiki_path: str | Path,
                 zim_dir: str | Path | None = None) -> None:
        self._root = Path(wiki_path)
        self._db_path = self._root / "wiki_cache.db"
        # ZIM hors vault Obsidian — dans Secretarius/Wiki_LM/zim/ par défaut
        self._zim_dir = Path(zim_dir) if zim_dir is not None else self._DEFAULT_ZIM
        self._conn: sqlite3.Connection | None = None
        self._zims: dict[str, Path] | None = None

    # -- SQLite cache --------------------------------------------------------

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS articles (
                    lang     TEXT NOT NULL,
                    title    TEXT NOT NULL,
                    abstract TEXT,
                    url      TEXT,
                    PRIMARY KEY (lang, title)
                );
                CREATE INDEX IF NOT EXISTS idx_norm
                    ON articles(lang, lower(title));
            """)
            self._conn.commit()
        return self._conn

    def _cache_get(self, title: str, lang: str) -> dict[str, Any] | None:
        row = self._db().execute(
            "SELECT title, abstract, url, lang FROM articles "
            "WHERE lang=? AND lower(title)=lower(?)",
            (lang, title),
        ).fetchone()
        return dict(row) if row and row["abstract"] else None

    def _cache_set(self, result: dict[str, Any]) -> None:
        try:
            self._db().execute(
                "INSERT OR REPLACE INTO articles(lang, title, abstract, url) "
                "VALUES (?, ?, ?, ?)",
                (result["lang"], result["title"], result["abstract"], result["url"]),
            )
            self._db().commit()
        except Exception:
            pass

    # -- ZIM -----------------------------------------------------------------

    def _get_zims(self) -> dict[str, Path]:
        if self._zims is None:
            self._zims = _zim_files(self._zim_dir)
        return self._zims

    # -- Public API ----------------------------------------------------------

    def lookup(self, title: str, langs: list[str] | None = None) -> dict[str, Any] | None:
        """Cherche title dans les trois backends, dans l'ordre ZIM → cache → API."""
        zims = self._get_zims()
        for lang in (langs or ["fr", "en"]):
            # 1. ZIM local
            if lang in zims:
                result = _zim_lookup(title, lang, zims[lang])
                if result:
                    self._cache_set(result)
                    return result
            # 2. Cache SQLite
            result = self._cache_get(title, lang)
            if result:
                return result
            # 3. API Wikipedia
            result = _fetch_api(title, lang)
            if result:
                self._cache_set(result)
                return result
        return None

    def zim_langs(self) -> list[str]:
        return list(self._get_zims().keys())

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
