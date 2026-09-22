"""Tests de repair.py."""

from __future__ import annotations

from pathlib import Path

from repair import WikiRepair


def _write_page(wiki_dir: Path, subdir: str, slug: str, *, title: str = "",
                 category: str = "", body: str = "") -> Path:
    d = wiki_dir / subdir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{slug}.md"
    fm_lines = ["---"]
    if title:
        fm_lines.append(f"title: {title}")
    if category:
        fm_lines.append(f"category: {category}")
    fm_lines.append("---")
    path.write_text("\n".join(fm_lines) + f"\n\n{body}\n", encoding="utf-8")
    return path


class TestRepairBrokenLinks:
    def test_dry_run_does_not_modify_disk(self, wiki_root, wiki_dir):
        path = _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-inexistant]] et [[c-autre-inexistant]].",
        )
        original = path.read_text(encoding="utf-8")

        report = WikiRepair(wiki_root).repair_broken_links(dry_run=True)

        assert path.read_text(encoding="utf-8") == original
        assert report.dry_run is True
        assert report.family == "broken-link"

    def test_apply_strips_brackets_from_broken_links(self, wiki_root, wiki_dir):
        path = _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-inexistant]] et [[c-autre-inexistant]].",
        )

        WikiRepair(wiki_root).repair_broken_links(dry_run=False)

        content = path.read_text(encoding="utf-8")
        assert "[[c-inexistant]]" not in content
        assert "[[c-autre-inexistant]]" not in content
        assert "c-inexistant" in content
        assert "c-autre-inexistant" in content

    def test_apply_leaves_valid_links_untouched(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "concepts", "c-b", title="B", category="concept")
        path = _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-b]] et [[c-inexistant]].",
        )

        WikiRepair(wiki_root).repair_broken_links(dry_run=False)

        content = path.read_text(encoding="utf-8")
        assert "[[c-b]]" in content
        assert "[[c-inexistant]]" not in content

    def test_before_after_counts(self, wiki_root, wiki_dir):
        _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-x]] et [[c-y]].",
        )

        report = WikiRepair(wiki_root).repair_broken_links(dry_run=False)

        assert report.before_count == 2
        assert report.after_count == 0

    def test_dry_run_reports_accurate_after_count_without_writing(self, wiki_root, wiki_dir):
        path = _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-x]].",
        )
        original = path.read_text(encoding="utf-8")

        report = WikiRepair(wiki_root).repair_broken_links(dry_run=True)

        assert report.before_count == 1
        assert report.after_count == 0
        assert path.read_text(encoding="utf-8") == original  # rien écrit malgré after_count correct

    def test_no_broken_links_reports_empty_changes(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", title="A", category="source", body="Rien à réparer.")

        report = WikiRepair(wiki_root).repair_broken_links(dry_run=True)

        assert report.changes == []
        assert report.before_count == 0
        assert report.after_count == 0
