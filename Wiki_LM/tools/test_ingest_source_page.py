import ingest


def test_generate_source_page_utilise_phi4(monkeypatch):
    monkeypatch.setattr(ingest, "select_central_passages", lambda text, **k: "passages réduits")
    monkeypatch.setattr(ingest, "generate_page_content", lambda passages, **k: {
        "resume": "R.", "points_cles": ["p"], "concepts": ["c"],
        "entites": ["Napoléon"], "tags": ["histoire"]})

    ing = ingest.Ingestor.__new__(ingest.Ingestor)   # bypass __init__ (pas de LLM/wiki_dir)
    ing.today = "2026-07-09"
    md = ing._generate_source_page("un long contenu source", "Ma Source", extra_tags=["arbath"])

    assert "category: source" in md
    assert "# Ma Source" in md
    assert "- entité: Napoléon" in md
    assert "arbath" in md and "histoire" in md   # tags fusionnés
