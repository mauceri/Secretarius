from __future__ import annotations

from pathlib import Path

from list_unsourced_pages import collect_unsourced, format_report


def _write_page(wiki_dir: Path, slug: str, lien_source: str | None, resume: str) -> Path:
    front = (
        "---\n"
        f"title: {slug.replace('src-', '').replace('-', ' ').title()}\n"
        "category: source\n"
        "tags: [test, exemple]\n"
        "created: 2026-01-01\n"
        "sources: []\n"
    )
    if lien_source:
        front += f"lien_source: {lien_source}\n"
    front += "---\n\n"
    body = f"# Titre\n\n## Résumé\n\n{resume}\n\n## Points clés\n\n- a\n"
    page = wiki_dir / "sources" / f"{slug}.md"
    page.write_text(front + body, encoding="utf-8")
    return page


def test_excludes_pages_with_lien_source(wiki_dir: Path):
    _write_page(wiki_dir, "src-a", "https://example.com", "Phrase un. Phrase deux. Phrase trois.")
    _write_page(wiki_dir, "src-b", None, "Phrase un. Phrase deux. Phrase trois.")

    result = collect_unsourced(wiki_dir)

    assert [r["slug"] for r in result] == ["src-b"]


def test_extracts_title_tags_and_two_sentence_excerpt(wiki_dir: Path):
    _write_page(wiki_dir, "src-b", None, "Phrase un. Phrase deux. Phrase trois.")

    result = collect_unsourced(wiki_dir)

    assert result[0]["title"] == "B"
    assert result[0]["tags"] == ["test", "exemple"]
    assert result[0]["excerpt"] == "Phrase un. Phrase deux."


def test_format_report_produces_markdown_table(wiki_dir: Path):
    _write_page(wiki_dir, "src-b", None, "Phrase un. Phrase deux.")
    entries = collect_unsourced(wiki_dir)

    report = format_report(entries)

    assert "| Slug | Titre | Tags | Extrait |" in report
    assert "src-b" in report
    assert "test, exemple" in report
