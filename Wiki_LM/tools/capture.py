"""
Capture rapide vers raw/ : URLs, commentaire libre, ou fichier joint.

Usage :
    python tools/capture.py https://example.com https://autre.com
    python tools/capture.py "Réflexion sur l'article de Salton"
    python tools/capture.py --file /tmp/document.pdf
    python tools/capture.py --file /tmp/document.pdf "commentaire optionnel"

URLs   → un fichier .url par URL, slug horodaté + domaine
Texte  → un fichier .md, slug horodaté + incipit
Fichier → copie dans raw/ avec nom horodaté, extension conservée
"""

from __future__ import annotations

import re
import shutil
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

RAW_DEFAULT = Path.home() / "Secretarius/Wiki_LM/raw"


def _file_hash(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_url(url: str) -> str:
    """Supprime les paramètres de tracking courants pour la comparaison."""
    import urllib.parse
    _TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_content",
                 "utm_term", "triedRedirect", "inbox"}
    parsed = urllib.parse.urlparse(url.strip())
    qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    qs_clean = {k: v for k, v in qs.items() if k not in _TRACKING}
    clean = parsed._replace(query=urllib.parse.urlencode(qs_clean, doseq=True), fragment="")
    return urllib.parse.urlunparse(clean).rstrip("/")


def _existing_urls(raw: Path) -> set[str]:
    """Retourne l'ensemble des URLs normalisées déjà présentes dans raw/."""
    seen = set()
    for f in raw.glob("*.url"):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("url:"):
                line = line[4:].strip()
            if line.startswith("http://") or line.startswith("https://"):
                seen.add(_normalize_url(line))
                break
    return seen


def _existing_hashes(raw: Path) -> set[str]:
    """Retourne les SHA256 de tous les fichiers non-.url de raw/."""
    seen = set()
    for f in raw.iterdir():
        if f.is_file() and f.suffix.lower() not in (".url",):
            try:
                seen.add(_file_hash(f))
            except OSError:
                pass
    return seen


def slugify(text: str, max_words: int = 6) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text.lower())
    words = [w for w in text.split() if w][:max_words]
    return "-".join(words) or "capture"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def capture_urls(urls: list[str], raw: Path) -> list[Path]:
    ts = timestamp()
    existing = _existing_urls(raw)
    created = []
    for i, url in enumerate(urls):
        norm = _normalize_url(url)
        if norm in existing:
            print(f"Doublon ignoré (URL déjà dans raw/) : {url}")
            continue
        existing.add(norm)
        domain = re.sub(r"https?://", "", url).split("/")[0]
        domain_slug = slugify(domain.replace(".", "-"), max_words=3)
        suffix = f"-{i}" if len(urls) > 1 else ""
        fname = f"{ts}{suffix}-{domain_slug}.url"
        path = raw / fname
        path.write_text(url + "\n", encoding="utf-8")
        created.append(path)
    return created


def capture_comment(text: str, raw: Path) -> Path:
    ts = timestamp()
    slug = slugify(text)
    fname = f"{ts}-{slug}.md"
    path = raw / fname
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def capture_file(src: Path, raw: Path, comment: str = "") -> Path:
    ts = timestamp()
    h = _file_hash(src)
    if h in _existing_hashes(raw):
        print(f"Doublon ignoré (contenu identique déjà dans raw/) : {src.name}")
        return None, None
    stem = slugify(src.stem) or "fichier"
    ext = src.suffix.lower() or ".bin"
    fname = f"{ts}-{stem}{ext}"
    dest = raw / fname
    shutil.copy2(src, dest)
    # Si un commentaire accompagne le fichier, on le dépose en .md juxtaposé
    if comment:
        note = dest.with_suffix(".md")
        note.write_text(comment.strip() + "\n", encoding="utf-8")
        return dest, note
    return dest, None


def main() -> None:
    import os
    raw = Path(os.environ.get("WIKI_RAW_PATH", str(RAW_DEFAULT))).expanduser()
    raw.mkdir(parents=True, exist_ok=True)

    argv = sys.argv[1:]

    # Mode fichier joint : --file <chemin> [commentaire optionnel]
    if argv and argv[0] == "--file":
        if len(argv) < 2:
            print("Usage: capture.py --file <chemin> [commentaire]", file=sys.stderr)
            sys.exit(1)
        src = Path(argv[1]).expanduser()
        if not src.exists():
            print(f"Fichier introuvable : {src}", file=sys.stderr)
            sys.exit(1)
        comment = " ".join(argv[2:]).strip()
        dest, note = capture_file(src, raw, comment)
        print(f"Fichier → {dest.name}")
        if note:
            print(f"Note    → {note.name}")
        return

    args = " ".join(argv).strip()
    if not args:
        print("Usage: capture.py <url [url ...]> | <commentaire> | --file <chemin>",
              file=sys.stderr)
        sys.exit(1)

    tokens = args.split()
    urls = [t for t in tokens if re.match(r"https?://", t)]

    if urls:
        created = capture_urls(urls, raw)
        for p in created:
            print(f"URL  → {p.name}")
    else:
        path = capture_comment(args, raw)
        print(f"Note → {path.name}")


if __name__ == "__main__":
    main()
