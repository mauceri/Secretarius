"""Tests du linter Wiki_LM (jamais eu de couverture avant le 21/09/2026)."""

from __future__ import annotations

from pathlib import Path

from lint import WikiLint


def _write_page(wiki_dir: Path, subdir: str, slug: str, *, title: str = "",
                 category: str = "", body: str = "") -> Path:
    d = wiki_dir / subdir
    path = d / f"{slug}.md"
    fm_lines = ["---"]
    if title:
        fm_lines.append(f"title: {title}")
    if category:
        fm_lines.append(f"category: {category}")
    fm_lines.append("---")
    path.write_text("\n".join(fm_lines) + f"\n\n{body}\n", encoding="utf-8")
    return path


class TestLoadPages:
    def test_finds_pages_across_subdirectories(self, wiki_root, wiki_dir):
        """Régression du bug réel : un glob à plat sur wiki_dir ne voyait
        plus rien depuis la restructuration en sources/concepts/entités."""
        _write_page(wiki_dir, "sources", "src-a", title="A", category="source")
        _write_page(wiki_dir, "concepts", "c-b", title="B", category="concept")
        _write_page(wiki_dir, "entités", "e-c", title="C", category="entité")

        report = WikiLint(wiki_root).run()

        assert report.checked_pages == 3

    def test_ignores_clusterings_subdirectory(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", title="A", category="source")
        (wiki_dir / "clusterings" / "cluster-x").mkdir()
        (wiki_dir / "clusterings" / "cluster-x" / "index.md").write_text(
            "---\ntitle: X\n---\n", encoding="utf-8"
        )

        report = WikiLint(wiki_root).run()

        assert report.checked_pages == 1


class TestCheckFrontmatter:
    def test_missing_title_and_category_reported(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", body="Contenu.")

        report = WikiLint(wiki_root).run()

        codes = [i.code for i in report.errors]
        assert codes.count("missing-frontmatter") == 2

    def test_unknown_category_is_a_warning(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", title="A", category="bidule")

        report = WikiLint(wiki_root).run()

        assert any(i.code == "unknown-category" for i in report.warnings)

    def test_known_category_no_warning(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", title="A", category="source")

        report = WikiLint(wiki_root).run()

        assert not any(i.code == "unknown-category" for i in report.warnings)


class TestCheckLinks:
    def test_broken_link_reported(self, wiki_root, wiki_dir):
        _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-inexistant]].",
        )

        report = WikiLint(wiki_root).run()

        broken = [i for i in report.errors if i.code == "broken-link"]
        assert len(broken) == 1
        assert broken[0].slug == "src-a"

    def test_link_to_existing_page_not_broken(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "concepts", "c-b", title="B", category="concept")
        _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-b]].",
        )

        report = WikiLint(wiki_root).run()

        assert not any(i.code == "broken-link" for i in report.errors)

    def test_link_to_meta_page_not_broken(self, wiki_root, wiki_dir):
        _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[index]].",
        )

        report = WikiLint(wiki_root).run()

        assert not any(i.code == "broken-link" for i in report.errors)

    def test_broken_link_target_is_structured(self, wiki_root, wiki_dir):
        _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-inexistant]].",
        )

        report = WikiLint(wiki_root).run()

        broken = [i for i in report.errors if i.code == "broken-link"]
        assert len(broken) == 1
        assert broken[0].target == "c-inexistant"

    def test_non_broken_link_issue_has_empty_target(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", body="Sans frontmatter valide.")

        report = WikiLint(wiki_root).run()

        assert all(i.target == "" for i in report.issues if i.code != "broken-link")


class TestCheckOrphans:
    def test_page_with_no_incoming_link_is_orphan(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", title="A", category="source")

        report = WikiLint(wiki_root).run()

        assert any(i.code == "orphan" and i.slug == "src-a" for i in report.warnings)

    def test_page_linked_from_another_page_not_orphan(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "concepts", "c-b", title="B", category="concept")
        _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-b]].",
        )

        report = WikiLint(wiki_root).run()

        assert not any(i.code == "orphan" and i.slug == "c-b" for i in report.warnings)

    def test_page_linked_from_index_not_orphan(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", title="A", category="source")
        (wiki_dir / "index.md").write_text("- [[src-a]]\n", encoding="utf-8")

        report = WikiLint(wiki_root).run()

        assert not any(i.code == "orphan" and i.slug == "src-a" for i in report.warnings)


class TestCheckIndex:
    def test_page_missing_from_index_is_a_warning(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", title="A", category="source")

        report = WikiLint(wiki_root).run()

        assert any(i.code == "not-in-index" and i.slug == "src-a" for i in report.warnings)

    def test_index_referencing_missing_file_is_an_error(self, wiki_root, wiki_dir):
        (wiki_dir / "index.md").write_text("- [[src-fantome]]\n", encoding="utf-8")

        report = WikiLint(wiki_root).run()

        assert any(
            i.code == "index-ghost" and i.slug == "src-fantome" for i in report.errors
        )

    def test_missing_index_file_is_an_error(self, wiki_root, wiki_dir):
        (wiki_dir / "index.md").unlink()

        report = WikiLint(wiki_root).run()

        assert any(i.code == "missing-index" for i in report.errors)


class TestReport:
    def test_no_issues_on_a_clean_wiki(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", title="A", category="source")
        (wiki_dir / "index.md").write_text("- [[src-a]]\n", encoding="utf-8")

        report = WikiLint(wiki_root).run()

        assert report.issues == []

    def test_to_dict_reports_counts(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", body="Sans frontmatter valide.")

        report = WikiLint(wiki_root).run()
        data = report.to_dict()

        assert data["checked_pages"] == 1
        assert data["errors"] == len(report.errors)
        assert data["warnings"] == len(report.warnings)
