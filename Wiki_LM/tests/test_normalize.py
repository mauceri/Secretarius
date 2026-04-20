"""Tests de _normalize_links."""

from ingest import _normalize_links


class TestNormalizeLinks:
    def test_known_slug_unchanged(self):
        known = {"c-zettelkasten", "e-vannevar-bush"}
        result = _normalize_links("Voir [[c-zettelkasten]].", known)
        assert "[[c-zettelkasten]]" in result

    def test_resolves_with_prefix(self):
        known = {"c-zettelkasten"}
        result = _normalize_links("Voir [[zettelkasten]].", known)
        assert "[[c-zettelkasten]]" in result

    def test_unknown_slug_slugified(self):
        result = _normalize_links("Voir [[Nouveau Concept]].", set())
        assert "[[nouveau-concept]]" in result

    def test_removes_parenthetical_annotation(self):
        """Le LLM ajoute parfois [[c-memex (Memex)]] → doit devenir [[c-memex]]."""
        known = {"c-memex"}
        result = _normalize_links("Voir [[c-memex (Memex)]].", known)
        assert "[[c-memex]]" in result
        assert "Memex)" not in result

    def test_accent_in_link(self):
        known = {"c-theorie-de-l-information"}
        result = _normalize_links("Voir [[Théorie de l'information]].", known)
        assert "[[c-theorie-de-l-information]]" in result

    def test_multiple_links(self):
        known = {"c-bm25", "e-gerard-salton"}
        text = "[[bm25]] et [[gerard-salton]] sont liés."
        result = _normalize_links(text, known)
        assert "[[c-bm25]]" in result
        assert "[[e-gerard-salton]]" in result

    def test_no_links(self):
        text = "Texte sans liens."
        assert _normalize_links(text, set()) == text
