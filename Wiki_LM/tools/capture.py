# tools/capture.py
"""
Capture rapide vers raw/ : URLs, commentaire libre, ou fichier joint.

Les tokens #motclé sont extraits comme tags et normalisés contre le
dictionnaire de tags canoniques (best-effort, silencieux si absent).

Usage :
    python tools/capture.py https://example.com
    python tools/capture.py "#memo Réflexion sur BM25"
    python tools/capture.py "#attention #transformer https://arxiv.org/abs/1706.03762"
    python tools/capture.py "#attention note: commentaire https://arxiv.org/abs/1706.03762"
    python tools/capture.py --file /tmp/document.pdf
    python tools/capture.py --file /tmp/document.pdf "commentaire optionnel"

URLs seules            → un fichier .url par URL, avec ligne tags: si tags présents
Texte seul             → un fichier .md avec frontmatter tags si tags présents
Texte + URL(s)         → un seul fichier .md contenant texte et URL(s)
Fichier joint          → copie dans raw/ + .md optionnel si commentaire
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import wiki_paths  # charge Wiki_LM/.env dans os.environ

RAW_DEFAULT = Path(os.environ.get("WIKI_PATH", str(Path.home() / "Secretarius" / "Wiki_LM"))).expanduser() / "raw"
_DEFAULT_KB_DIR = Path(os.environ.get("WIKI_PATH", str(Path.home() / "Documents" / "Arbath" / "Wiki_LM"))).expanduser() / "knowledge_base"
_TAG_NORMALIZE_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_url(url: str) -> str:
    import urllib.parse
    _TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_content",
                 "utm_term", "triedRedirect", "inbox"}
    parsed = urllib.parse.urlparse(url.strip())
    qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    qs_clean = {k: v for k, v in qs.items() if k not in _TRACKING}
    clean = parsed._replace(query=urllib.parse.urlencode(qs_clean, doseq=True), fragment="")
    return urllib.parse.urlunparse(clean).rstrip("/")


def _existing_urls(raw: Path) -> set[str]:
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


# ---------------------------------------------------------------------------
# Hashtags
# ---------------------------------------------------------------------------

def _parse_hashtags(text: str) -> tuple[list[str], str]:
    """Extrait les tokens #motclé. Retourne (tags_bruts, texte_restant).

    Le préfixe "note:" éventuel est aussi retiré du texte restant.
    """
    tags = [m.group(1) for m in re.finditer(r"#(\w+)", text)]
    remaining = re.sub(r"#\w+\s*", "", text).strip()
    remaining = re.sub(r"^note:\s*", "", remaining, flags=re.IGNORECASE).strip()
    return tags, remaining


def _normalize_tags(raw_tags: list[str], kb_dir: Path = _DEFAULT_KB_DIR) -> list[str]:
    """Normalise les tags bruts contre les tags canoniques. Best-effort silencieux."""
    if not raw_tags:
        return []
    tags_dir = kb_dir / "tags"
    dict_path = tags_dir / "tags_dict.json"
    emb_path = tags_dir / "tags_embeddings.npy"
    if not dict_path.exists() or not emb_path.exists():
        return raw_tags
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        data = json.loads(dict_path.read_text(encoding="utf-8"))
        # Les canoniques sont dans l'ordre alphabétique (cf. save_tag_dict)
        canonicals = sorted(data.keys())
        matrix = np.load(emb_path).astype(np.float32)
        model = SentenceTransformer("BAAI/bge-m3")
        normalized = []
        for tag in raw_tags:
            vec = model.encode(tag, normalize_embeddings=True).astype(np.float32)
            sims = matrix @ vec
            best_idx = int(np.argmax(sims))
            if float(sims[best_idx]) >= _TAG_NORMALIZE_THRESHOLD:
                normalized.append(canonicals[best_idx])
            else:
                normalized.append(tag)
        return normalized
    except Exception:
        return raw_tags


# ---------------------------------------------------------------------------
# Fonctions de capture
# ---------------------------------------------------------------------------

def capture_urls(urls: list[str], raw: Path, tags: list[str] | None = None,
                 note: str | None = None) -> list[Path]:
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
        content = url + "\n"
        if tags:
            content += f"tags: {', '.join(tags)}\n"
        if note and not created:            # note attachée au premier .url créé
            content += f"note: {note}\n"
        path.write_text(content, encoding="utf-8")
        created.append(path)
    return created


def _write_note(path: Path, text: str, tags: list[str] | None, refs: list[str] | None, wiki_root: Path) -> None:
    body_lines = [text.strip()] if text.strip() else []
    for ref in (refs or []):
        try:
            Path(ref).relative_to(wiki_root)
            body_lines.append(f"[[{Path(ref).name}]]")
        except ValueError:
            pass
    body = "\n".join(body_lines)
    if tags or refs:
        fm = "---\n"
        if tags:
            fm += f"tags: [{', '.join(tags)}]\n"
        if refs:
            fm += (f"ref: {refs[0]}\n" if len(refs) == 1
                   else "refs:\n" + "".join(f"  - {r}\n" for r in refs))
        fm += "---\n"
        content = fm + body + "\n"
    else:
        content = (body + "\n") if body else "\n"
    path.write_text(content, encoding="utf-8")


def capture_comment(text: str, raw: Path, tags: list[str] | None = None, refs: list[str] | None = None) -> Path:
    ts = timestamp()
    slug = slugify(text)
    fname = f"{ts}-{slug}.md"
    path = raw / fname
    _write_note(path, text, tags, refs, raw.parent)
    return path


def capture_mixed(text: str, urls: list[str], raw: Path,
                  tags: list[str] | None = None) -> Path:
    """Texte + URL(s) → un seul fichier .md."""
    ts = timestamp()
    slug = slugify(text or urls[0])
    fname = f"{ts}-{slug}.md"
    path = raw / fname
    url_block = "\n".join(urls)
    body = f"{text}\n\n{url_block}" if text else url_block
    if tags:
        content = f"---\ntags: [{', '.join(tags)}]\n---\n{body.strip()}\n"
    else:
        content = body.strip() + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def capture_file(src: Path, raw: Path, comment: str = "",
                 tags: list[str] | None = None) -> tuple[Path | None, Path | None]:
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
    if comment or tags:
        note = dest.with_suffix(".md")
        if tags:
            note_content = f"---\ntags: [{', '.join(tags)}]\n---\n{comment.strip()}\n"
        else:
            note_content = comment.strip() + "\n"
        note.write_text(note_content, encoding="utf-8")
        return dest, note
    return dest, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        raw_comment = " ".join(argv[2:]).strip()
        tags_raw, comment = _parse_hashtags(raw_comment)
        tags = _normalize_tags(tags_raw)
        dest, note = capture_file(src, raw, comment, tags)
        if dest:
            print(f"Fichier → {dest.name}")
        if note:
            print(f"Note    → {note.name}")
        return

    args = " ".join(argv).strip()
    if not args:
        print("Usage: capture.py <url|commentaire|#tags ...> | --file <chemin>",
              file=sys.stderr)
        sys.exit(1)

    tags_raw, args_clean = _parse_hashtags(args)
    tags = _normalize_tags(tags_raw)

    tokens = args_clean.split() if args_clean else []
    urls = [t for t in tokens if re.match(r"https?://", t)]
    text_tokens = [t for t in tokens if not re.match(r"https?://", t)]
    text = " ".join(text_tokens).strip()

    if urls and text:
        path = capture_mixed(text, urls, raw, tags)
        print(f"Note → {path.name}")
    elif urls:
        created = capture_urls(urls, raw, tags)
        for p in created:
            print(f"URL  → {p.name}")
    else:
        path = capture_comment(text or " ".join(f"#{t}" for t in tags_raw), raw, tags)
        print(f"Note → {path.name}")


if __name__ == "__main__":
    main()
