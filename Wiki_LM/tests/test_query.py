"""Tests de WikiQuery.query() : historique horodaté + résumé bref."""

from __future__ import annotations

from search import SearchResult
from query import WikiQuery


class _StubLLM:
    def __init__(self, brief: str = "Résumé bref.", fail_brief: bool = False) -> None:
        self.calls: list[dict] = []
        self._brief = brief
        self._fail_brief = fail_brief

    def complete(self, prompt: str = "", *, messages=None, system: str = "", max_tokens: int = 2048) -> str:
        self.calls.append({"prompt": prompt, "system": system, "max_tokens": max_tokens})
        if len(self.calls) == 1:
            return "Synthèse test avec [[c-test]]."
        if self._fail_brief:
            raise RuntimeError("LLM indisponible")
        return self._brief


def _make_query(tmp_path, llm=None) -> WikiQuery:
    (tmp_path / "wiki").mkdir()
    page = tmp_path / "wiki" / "c-test.md"
    page.write_text("Contenu de test.", encoding="utf-8")
    wq = WikiQuery(tmp_path, llm=llm or _StubLLM(), mode="bm25")
    result = SearchResult(slug="c-test", path=page, title="Test", category="concept",
                          score=1.0, excerpt="Contenu de test.")
    wq._search.search = lambda q, top_k=5: [result]
    return wq


def test_query_writes_history_note(tmp_path):
    wq = _make_query(tmp_path)
    result = wq.query("Question test ?")

    assert result.history_slug
    history_files = list((tmp_path / "historique").glob("*.md"))
    assert len(history_files) == 1
    assert history_files[0].stem == result.history_slug
    content = history_files[0].read_text(encoding="utf-8")
    assert "Question test ?" in content
    assert "Synthèse test avec [[c-test]]." in content


def test_query_generates_brief(tmp_path):
    wq = _make_query(tmp_path, llm=_StubLLM(brief="Un court résumé."))
    result = wq.query("Question test ?")

    assert result.brief == "Un court résumé."


def test_query_brief_falls_back_to_truncation_on_llm_failure(tmp_path):
    wq = _make_query(tmp_path, llm=_StubLLM(fail_brief=True))
    result = wq.query("Question test ?")

    assert result.brief == result.text[:300]


def test_query_writes_second_copy_in_known_mirror_vault(tmp_path, monkeypatch):
    other_vault = tmp_path / "other-vault"
    other_vault.mkdir()
    monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={other_vault}")
    wq = _make_query(tmp_path)

    result = wq.query("Question test ?", vault_name="Arbath")

    canonical = (tmp_path / "historique" / f"{result.history_slug}.md").read_text(encoding="utf-8")
    copy = (other_vault / "Wiki_LM" / "historique" / f"{result.history_slug}.md").read_text(encoding="utf-8")
    assert copy == canonical


def test_query_ignores_unknown_vault_name(tmp_path, monkeypatch):
    """Un nom de coffre non listé dans WIKI_VAULT_MIRRORS ne doit jamais être
    traité comme un chemin (revue du 21/09/2026 : un chemin fourni par le
    client n'a de sens que sur sa propre machine)."""
    monkeypatch.delenv("WIKI_VAULT_MIRRORS", raising=False)
    wq = _make_query(tmp_path)

    wq.query("Question test ?", vault_name="Coffre inconnu")

    history_files = list((tmp_path / "historique").glob("*.md"))
    assert len(history_files) == 1


def test_query_does_not_duplicate_when_mirror_resolves_to_canonical(tmp_path, monkeypatch):
    monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Secretarius={tmp_path}")
    wq = _make_query(tmp_path)

    wq.query("Question test ?", vault_name="Secretarius")

    history_files = list((tmp_path / "historique").glob("*.md"))
    assert len(history_files) == 1


def test_query_without_vault_name_writes_only_canonical_copy(tmp_path):
    wq = _make_query(tmp_path)

    wq.query("Question test ?")

    history_files = list((tmp_path / "historique").glob("*.md"))
    assert len(history_files) == 1


def test_query_mirrors_cited_pages_to_calling_vault(tmp_path, monkeypatch):
    other_vault = tmp_path / "other-vault"
    monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={other_vault}")
    wq = _make_query(tmp_path)  # crée tmp_path/wiki/c-test.md (à plat, pour son propre stub de recherche)
    # mirror_page() résout le slug via la vraie arborescence (wiki/concepts/,
    # subdir_for_slug) — la fixture _make_query écrit à plat pour son propre
    # usage ; la page doit aussi exister au bon sous-dossier pour que la
    # copie miroir la trouve.
    concepts_dir = tmp_path / "wiki" / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / "c-test.md").write_text("Contenu de test.", encoding="utf-8")

    wq.query("Question test ?", vault_name="Arbath")  # synthèse stub cite [[c-test]]

    mirrored = other_vault / "Wiki_LM" / "wiki" / "concepts" / "c-test.md"
    assert mirrored.read_text(encoding="utf-8") == "Contenu de test."


def test_query_does_not_mirror_cited_pages_without_vault_name(tmp_path, monkeypatch):
    monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={tmp_path / 'other-vault'}")
    wq = _make_query(tmp_path)
    concepts_dir = tmp_path / "wiki" / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / "c-test.md").write_text("Contenu de test.", encoding="utf-8")

    wq.query("Question test ?")

    assert not (tmp_path / "other-vault").exists()


def test_query_save_flag_unaffected(tmp_path):
    wq = _make_query(tmp_path)
    result = wq.query("Question test ?", save=True)

    assert result.saved_slug
    assert (tmp_path / "wiki" / f"{result.saved_slug}.md").exists()


def test_query_no_results_still_writes_history(tmp_path):
    wq = _make_query(tmp_path)
    wq._search.search = lambda q, top_k=5: []
    result = wq.query("Question sans réponse ?")

    assert result.text == "_Aucune page pertinente trouvée dans le wiki._"
    assert result.history_slug
    assert result.brief
    history_files = list((tmp_path / "historique").glob("*.md"))
    assert len(history_files) == 1
