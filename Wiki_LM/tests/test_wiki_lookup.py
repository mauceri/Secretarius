"""Tests de WikiLookup (cache SQLite, sans réseau)."""

from pathlib import Path

import pytest

from wiki_lookup import WikiLookup


@pytest.fixture
def lookup(tmp_path):
    # zim_dir pointe vers un répertoire vide pour isoler des ZIM réels
    return WikiLookup(tmp_path, zim_dir=tmp_path / "zim")


class TestCache:
    def test_env_zim_dir_used_when_argument_missing(self, tmp_path, monkeypatch):
        configured = tmp_path / "configured-zim"
        monkeypatch.setenv("WIKI_ZIM_DIR", str(configured))

        lookup = WikiLookup(tmp_path)

        assert lookup._zim_dir == configured

    def test_explicit_zim_dir_overrides_env(self, tmp_path, monkeypatch):
        configured = tmp_path / "configured-zim"
        explicit = tmp_path / "explicit-zim"
        monkeypatch.setenv("WIKI_ZIM_DIR", str(configured))

        lookup = WikiLookup(tmp_path, zim_dir=explicit)

        assert lookup._zim_dir == explicit

    def test_offline_mode_skips_api(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WIKI_LOOKUP_OFFLINE", "1")
        monkeypatch.setattr("wiki_lookup._zim_files", lambda zim_dir: {})
        calls = []
        monkeypatch.setattr(
            "wiki_lookup._fetch_api",
            lambda *args: calls.append(args) or {
                "lang": args[1],
                "title": args[0],
                "abstract": "api",
                "url": "",
            },
        )

        lookup = WikiLookup(tmp_path)

        assert lookup.lookup("Offline", langs=["fr"]) is None
        assert calls == []

    def test_backends_cache_only_uses_cache_without_api_or_zim(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WIKI_LOOKUP_BACKENDS", "cache")
        monkeypatch.setattr("wiki_lookup._zim_files", lambda zim_dir: {"fr": tmp_path / "fake.zim"})
        monkeypatch.setattr("wiki_lookup._zim_lookup", lambda *args: pytest.fail("ZIM should not be called"))
        monkeypatch.setattr("wiki_lookup._fetch_api", lambda *args: pytest.fail("API should not be called"))
        lookup = WikiLookup(tmp_path)
        lookup._cache_set({"lang": "fr", "title": "Cached", "abstract": "cache", "url": ""})

        result = lookup.lookup("Cached", langs=["fr"])

        assert result["abstract"] == "cache"

    def test_default_backend_order_is_zim_cache_api(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr("wiki_lookup._zim_files", lambda zim_dir: {"fr": tmp_path / "fake.zim"})
        monkeypatch.setattr("wiki_lookup._zim_lookup", lambda *args: calls.append("zim") or None)
        monkeypatch.setattr(
            "wiki_lookup._fetch_api",
            lambda title, lang: calls.append("api") or {
                "lang": lang,
                "title": title,
                "abstract": "api",
                "url": "",
            },
        )
        lookup = WikiLookup(tmp_path)
        original_cache_get = lookup._cache_get

        def cache_get(title, lang):
            calls.append("cache")
            return original_cache_get(title, lang)

        monkeypatch.setattr(lookup, "_cache_get", cache_get)

        result = lookup.lookup("Order", langs=["fr"])

        assert result["abstract"] == "api"
        assert calls == ["zim", "cache", "api"]

    def test_invalid_backend_list_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WIKI_LOOKUP_BACKENDS", "nonsense,other")
        calls = []
        monkeypatch.setattr("wiki_lookup._zim_files", lambda zim_dir: {})
        monkeypatch.setattr(
            "wiki_lookup._fetch_api",
            lambda title, lang: calls.append("api") or {
                "lang": lang,
                "title": title,
                "abstract": "api",
                "url": "",
            },
        )

        lookup = WikiLookup(tmp_path)
        result = lookup.lookup("Fallback", langs=["fr"])

        assert result["abstract"] == "api"
        assert calls == ["api"]

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
