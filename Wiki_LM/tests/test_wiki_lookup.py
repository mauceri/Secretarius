"""Tests de WikiLookup (cache SQLite, sans réseau)."""

from pathlib import Path

import pytest

from wiki_lookup import WikiLookup


@pytest.fixture
def lookup(tmp_path):
    # zim_dir pointe vers un répertoire vide pour isoler des ZIM réels
    return WikiLookup(tmp_path, zim_dir=tmp_path / "zim")


class TestCache:
    def test_miss_returns_none_offline(self, lookup, monkeypatch):
        """Sans réseau, un titre inconnu retourne None."""
        monkeypatch.setattr("wiki_lookup._fetch_api", lambda *a, **kw: None)
        assert lookup.lookup("Titre Inexistant XYZ123") is None

    def test_store_and_retrieve(self, lookup):
        entry = {"lang": "fr", "title": "Test", "abstract": "Résumé test.", "url": "https://fr.wikipedia.org/wiki/Test"}
        lookup._cache_set(entry)
        result = lookup._cache_get("Test", "fr")
        assert result is not None
        assert result["abstract"] == "Résumé test."

    def test_case_insensitive_retrieval(self, lookup):
        entry = {"lang": "fr", "title": "Zettelkasten", "abstract": "Méthode de prise de notes.", "url": ""}
        lookup._cache_set(entry)
        assert lookup._cache_get("zettelkasten", "fr") is not None
        assert lookup._cache_get("ZETTELKASTEN", "fr") is not None

    def test_cache_hit_skips_api(self, lookup, monkeypatch):
        entry = {"lang": "fr", "title": "Cached", "abstract": "Depuis le cache.", "url": ""}
        lookup._cache_set(entry)
        calls = []
        monkeypatch.setattr("wiki_lookup._fetch_api", lambda *a, **kw: calls.append(a) or None)
        result = lookup.lookup("Cached", langs=["fr"])
        assert result["abstract"] == "Depuis le cache."
        assert len(calls) == 0

    def test_api_result_cached(self, lookup, monkeypatch):
        """Résultat API stocké en cache pour appels suivants."""
        fake = {"lang": "fr", "title": "Nouveau", "abstract": "Depuis l'API.", "url": ""}
        monkeypatch.setattr("wiki_lookup._fetch_api", lambda *a, **kw: fake)
        lookup.lookup("Nouveau", langs=["fr"])
        # Second appel sans API
        monkeypatch.setattr("wiki_lookup._fetch_api", lambda *a, **kw: None)
        result = lookup.lookup("Nouveau", langs=["fr"])
        assert result is not None
        assert result["abstract"] == "Depuis l'API."

    def test_separate_by_lang(self, lookup):
        fr = {"lang": "fr", "title": "Test", "abstract": "En français.", "url": ""}
        en = {"lang": "en", "title": "Test", "abstract": "In English.", "url": ""}
        lookup._cache_set(fr)
        lookup._cache_set(en)
        assert lookup._cache_get("Test", "fr")["abstract"] == "En français."
        assert lookup._cache_get("Test", "en")["abstract"] == "In English."

    def test_no_zim_without_files(self, lookup):
        assert lookup.zim_langs() == []
