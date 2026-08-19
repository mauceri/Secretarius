from __future__ import annotations

from pathlib import Path

import frontmatter

from fix_corrupted_tags import clean_tags, fix_pages


class TestCleanTags:
    def test_removes_duplicate_bracket_wrapped_tag(self):
        """Motif A : tag entre crochets dupliquant un tag déjà présent."""
        result = clean_tags(["christianisme", "art", "[christianisme]"])
        assert result == ["christianisme", "art"]

    def test_rejoins_split_fragments(self):
        """Motif B : liste coupée en fragments par la virgule à l'intérieur des crochets."""
        result = clean_tags(["[documentation", "secretarius]"])
        assert result == ["documentation", "secretarius"]

    def test_leaves_clean_tags_unchanged(self):
        result = clean_tags(["documentation", "secretarius", "openclaw"])
        assert result == ["documentation", "secretarius", "openclaw"]

    def test_flattens_nested_list_and_deduplicates(self):
        """Branche isinstance(tag, list) : listes imbriquées (malformation YAML)."""
        result = clean_tags([["tailscale"], "[tailscale]"])
        assert result == ["tailscale"]


class TestFixPages:
    def test_fixes_corrupted_page_on_disk(self, wiki_dir: Path):
        page = wiki_dir / "sources" / "src-corrupted.md"
        page.write_text(
            "---\ntitle: Corrupted\ncategory: source\n"
            "tags:\n- '[documentation'\n- secretarius]\n"
            "created: 2026-01-01\nsources: []\n---\n\n# Corrupted\n",
            encoding="utf-8",
        )

        fixed = fix_pages(wiki_dir, apply=True)

        assert "src-corrupted" in fixed
        post = frontmatter.load(page)
        assert post["tags"] == ["documentation", "secretarius"]

    def test_dry_run_does_not_write(self, wiki_dir: Path):
        page = wiki_dir / "sources" / "src-corrupted.md"
        page.write_text(
            "---\ntitle: Corrupted\ncategory: source\n"
            "tags:\n- '[documentation'\n- secretarius]\n"
            "created: 2026-01-01\nsources: []\n---\n\n# Corrupted\n",
            encoding="utf-8",
        )

        fix_pages(wiki_dir, apply=False)

        post = frontmatter.load(page)
        assert post["tags"] == ["[documentation", "secretarius]"]

    def test_skips_pages_without_bracket_pattern(self, wiki_dir: Path):
        page = wiki_dir / "sources" / "src-clean.md"
        page.write_text(
            "---\ntitle: Clean\ncategory: source\ntags: [a, b]\n"
            "created: 2026-01-01\nsources: []\n---\n\n# Clean\n",
            encoding="utf-8",
        )

        fixed = fix_pages(wiki_dir, apply=True)

        assert "src-clean" not in fixed

    def test_fixes_corrupted_nested_list_tags_on_disk(self, wiki_dir: Path):
        """Branche isinstance(tag, list) : listes imbriquées (malformation YAML)."""
        page = wiki_dir / "sources" / "src-nested.md"
        # YAML qui parse en [['tailscale'], '[tailscale]']
        page.write_text(
            "---\ntitle: Nested\ncategory: source\n"
            "tags:\n- - tailscale\n- '[tailscale]'\n"
            "created: 2026-01-01\nsources: []\n---\n\n# Nested\n",
            encoding="utf-8",
        )

        fixed = fix_pages(wiki_dir, apply=True)

        assert "src-nested" in fixed
        post = frontmatter.load(page)
        assert post["tags"] == ["tailscale"]
