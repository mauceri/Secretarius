"""
Relance l'ingestion des fichiers .url.error via le pipeline actif (pas
mcp_server.py, qui est mort — abandon MCP, commit 3693a57).

En cas de nouvel échec, la raison est ajoutée en commentaire dans le
fichier .url.error (jusqu'ici silencieux — seule l'URL y était visible).

Usage :
    python retry_blocked_urls.py [--wiki PATH] [--raw PATH] [--apply]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ingest import Ingestor, _extract_url_from_file, _parse_note_from_url_file

_DEFAULT_WIKI = Path.home() / "Documents" / "Arbath" / "Wiki_LM"
_DEFAULT_RAW = Path.home() / "Documents" / "Arbath" / "Wiki_LM" / "raw"


def retry_all(ingestor: Ingestor, raw_dir: Path, dry_run: bool) -> list[dict]:
    results: list[dict] = []

    for error_file in sorted(raw_dir.glob("*.url.error")):
        url = _extract_url_from_file(error_file)
        entry = {"filename": error_file.name, "url": url}

        if dry_run:
            entry["status"] = "would-retry"
            results.append(entry)
            continue

        try:
            user_tags = ingestor._parse_raw_tags(error_file)
            simple = ingestor._parse_raw_simple(error_file)
            note = _parse_note_from_url_file(error_file)
            slug = ingestor.ingest(url, extra_tags=user_tags or None, rename_raw=False, note=note, local_note=simple)
            entry["status"] = "ingested"
            entry["slug"] = slug
            error_file.unlink()
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            content = error_file.read_text(encoding="utf-8")
            if "# échec :" not in content:
                error_file.write_text(content.rstrip("\n") + f"\n# échec : {exc}\n", encoding="utf-8")

        results.append(entry)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Relance les fichiers .url.error")
    parser.add_argument("--wiki", default=os.environ.get("WIKI_PATH", str(_DEFAULT_WIKI)))
    parser.add_argument("--raw", default=os.environ.get("WIKI_RAW_PATH", str(_DEFAULT_RAW)))
    parser.add_argument("--apply", action="store_true", help="Relancer réellement (sinon dry-run)")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    ingestor = Ingestor(args.wiki, raw_path=raw_dir)

    results = retry_all(ingestor, raw_dir, dry_run=not args.apply)

    for r in results:
        print(f"{r['filename']} → {r['status']}" + (f" ({r.get('error')})" if r.get("error") else ""))
    if not args.apply:
        print("\nRelancez avec --apply pour exécuter réellement.")


if __name__ == "__main__":
    main()
