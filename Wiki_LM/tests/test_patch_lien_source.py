from __future__ import annotations

from pathlib import Path

import frontmatter
import pytest

from patch_lien_source import _load_manifest, patch_pages


def _write_manifest(raw_dir: Path, entries: list[tuple[str, str]]) -> None:
    lines = [f"{filename}\t{slug}\thash0" for filename, slug in entries]
    (raw_dir / ".ingested").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_finds_pages_in_sources_subdir(wiki_dir: Path, raw_dir: Path):
    """Régression : les pages vivent dans wiki/sources/, pas wiki/*.md."""
    page = wiki_dir / "sources" / "src-test-page.md"
    page.write_text(
        "---\ntitle: Test\ncategory: source\ntags: [x]\n"
        "created: 2026-01-01\nsources: []\n---\n\n# Test\n",
        encoding="utf-8",
    )
    (raw_dir / "src-test-page.url").write_text("https://example.com/article\n", encoding="utf-8")
    _write_manifest(raw_dir, [("src-test-page.url", "src-test-page")])

    patched = patch_pages(wiki_dir, raw_dir, apply=False)

    assert patched == 1


def test_apply_writes_lien_source(wiki_dir: Path, raw_dir: Path):
    page = wiki_dir / "sources" / "src-test-page.md"
    page.write_text(
        "---\ntitle: Test\ncategory: source\ntags: [x]\n"
        "created: 2026-01-01\nsources: []\n---\n\n# Test\n",
        encoding="utf-8",
    )
    (raw_dir / "src-test-page.url").write_text("https://example.com/article\n", encoding="utf-8")
    _write_manifest(raw_dir, [("src-test-page.url", "src-test-page")])

    patch_pages(wiki_dir, raw_dir, apply=True)

    post = frontmatter.loads(page.read_text(encoding="utf-8"))
    assert post["lien_source"] == "https://example.com/article"


def test_relaxed_filter_extracts_from_non_url_suffix(wiki_dir: Path, raw_dir: Path):
    """Le filtre assoupli : tente l'extraction même si le fichier raw n'est pas .url."""
    page = wiki_dir / "sources" / "src-note-page.md"
    page.write_text(
        "---\ntitle: Note\ncategory: source\ntags: [x]\n"
        "created: 2026-01-01\nsources: []\n---\n\n# Note\n",
        encoding="utf-8",
    )
    # fichier raw en .md contenant quand même une URL en première ligne
    (raw_dir / "src-note-page.md").write_text(
        "https://example.com/note-source\ntags: x\n", encoding="utf-8"
    )
    _write_manifest(raw_dir, [("src-note-page.md", "src-note-page")])

    patched = patch_pages(wiki_dir, raw_dir, apply=False)

    assert patched == 1


def test_skips_page_already_patched(wiki_dir: Path, raw_dir: Path):
    page = wiki_dir / "sources" / "src-done.md"
    page.write_text(
        "---\ntitle: Done\ncategory: source\ntags: [x]\ncreated: 2026-01-01\n"
        "sources: []\nlien_source: https://already-set.example.com\n---\n\n# Done\n",
        encoding="utf-8",
    )
    (raw_dir / "src-done.url").write_text("https://different.example.com\n", encoding="utf-8")
    _write_manifest(raw_dir, [("src-done.url", "src-done")])

    patched = patch_pages(wiki_dir, raw_dir, apply=True)

    assert patched == 0
    post = frontmatter.loads(page.read_text(encoding="utf-8"))
    assert post["lien_source"] == "https://already-set.example.com"
