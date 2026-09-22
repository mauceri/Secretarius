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


class TestExtractClosedFrontmatterBlock:
    def test_finds_well_formed_second_block(self):
        from repair import _extract_closed_frontmatter_block
        body = (
            "\nyaml\n---\ntitle: Mon titre\ncategory: concept\n---\n```\n\n"
            "# Mon titre\n\nLe corps réel."
        )
        result = _extract_closed_frontmatter_block(body)
        assert result is not None
        meta, rest = result
        assert meta["title"] == "Mon titre"
        assert meta["category"] == "concept"
        assert "Le corps réel." in rest
        assert "yaml" not in rest.split("\n")[0] if rest else True

    def test_returns_none_without_a_closed_block(self):
        from repair import _extract_closed_frontmatter_block
        body = "## Extrait Wikipedia\n\nDu texte normal, aucun bloc frontmatter."
        assert _extract_closed_frontmatter_block(body) is None

    def test_returns_none_for_an_unclosed_block(self):
        from repair import _extract_closed_frontmatter_block
        body = "---\ntitle: Titre tronqué\ncategory: concept\nsources: [src-a"
        assert _extract_closed_frontmatter_block(body) is None

    def test_returns_none_when_block_lacks_title_or_category(self):
        from repair import _extract_closed_frontmatter_block
        body = "---\ntags: [a, b]\n---\n\nCorps."
        assert _extract_closed_frontmatter_block(body) is None


class TestSlugToTitle:
    def test_strips_prefix_and_humanizes(self):
        from repair import _slug_to_title
        assert _slug_to_title("c-mon-concept") == "mon concept"
        assert _slug_to_title("e-vannevar-bush") == "vannevar bush"
        assert _slug_to_title("src-un-article") == "un article"


class TestRepairFrontmatter:
    def test_well_formed_block_is_moved_mechanically_no_llm_call(self, wiki_root, wiki_dir):
        path = wiki_dir / "concepts" / "c-x.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\n{}\n---\n\nyaml\n---\ntitle: Mon concept\ncategory: concept\n"
            "---\n```\n\n# Mon concept\n\nLe corps réel.\n",
            encoding="utf-8",
        )

        class _NoCallLLM:
            def complete(self, *a, **k):
                raise AssertionError("le LLM ne doit pas être appelé pour un bloc bien formé")

        report = WikiRepair(wiki_root, llm=_NoCallLLM()).repair_frontmatter(dry_run=False)

        content = path.read_text(encoding="utf-8")
        assert "title: Mon concept" in content
        assert content.startswith("---\n")
        assert "yaml" not in content.split("---")[0]
        assert "Le corps réel." in content
        assert report.family == "missing-frontmatter"

    def test_missing_block_regenerates_title_via_llm(self, wiki_root, wiki_dir):
        path = wiki_dir / "concepts" / "c-y.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\n{}\n---\n\n## Extrait Wikipedia\n\nDu contenu réel sur le sujet Y.\n",
            encoding="utf-8",
        )

        class _StubLLM:
            def __init__(self):
                self.calls = []

            def complete(self, prompt, system="", max_tokens=2048):
                self.calls.append(prompt)
                return "Titre régénéré"

        stub = _StubLLM()
        WikiRepair(wiki_root, llm=stub).repair_frontmatter(dry_run=False)

        content = path.read_text(encoding="utf-8")
        assert "title: Titre régénéré" in content
        assert "category: concept" in content  # dérivé du sous-répertoire, pas du LLM
        assert "Du contenu réel sur le sujet Y." in content
        assert len(stub.calls) == 1
        assert "Du contenu réel sur le sujet Y." in stub.calls[0]

    def test_category_derived_from_subdir_for_entity(self, wiki_root, wiki_dir):
        path = wiki_dir / "entités" / "e-z.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("---\n{}\n---\n\nDu contenu sur Z.\n", encoding="utf-8")

        class _StubLLM:
            def complete(self, prompt, system="", max_tokens=2048):
                return "Z"

        WikiRepair(wiki_root, llm=_StubLLM()).repair_frontmatter(dry_run=False)

        assert "category: entité" in path.read_text(encoding="utf-8")

    def test_empty_body_falls_back_to_slug_title_without_llm_call(self, wiki_root, wiki_dir):
        path = wiki_dir / "concepts" / "c-vide.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("---\n{}\n---\n", encoding="utf-8")

        class _NoCallLLM:
            def complete(self, *a, **k):
                raise AssertionError("le LLM ne doit pas être appelé sans contenu exploitable")

        WikiRepair(wiki_root, llm=_NoCallLLM()).repair_frontmatter(dry_run=False)

        content = path.read_text(encoding="utf-8")
        assert "title: vide" in content

    def test_dry_run_does_not_modify_disk_or_call_llm(self, wiki_root, wiki_dir):
        path = wiki_dir / "concepts" / "c-y.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = "---\n{}\n---\n\nDu contenu.\n"
        path.write_text(original, encoding="utf-8")

        class _NoCallLLM:
            def complete(self, *a, **k):
                raise AssertionError("essai à blanc : le LLM ne doit pas être appelé")

        report = WikiRepair(wiki_root, llm=_NoCallLLM()).repair_frontmatter(dry_run=True)

        assert path.read_text(encoding="utf-8") == original
        assert report.dry_run is True
