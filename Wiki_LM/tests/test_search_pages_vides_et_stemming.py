"""Régression du 2026-09-09 : pages vides polluant BM25/sémantique, et
absence de désuffixation faisant échouer une requête au singulier alors
que le pluriel marchait.

Cause : 63 pages c-/e- au contenu ``---\\n{}\\n---`` (ni titre ni corps),
issues d'échecs d'extraction en amont, indexées comme n'importe quelle
page — dont 48 partageant un seul et même vecteur d'embedding, remontant
en tête de toute recherche sémantique peu spécifique.
"""

from __future__ import annotations

from pathlib import Path

from search import SearchResult, WikiSearch, hybrid_search, tokenize


def _make_page(wiki_dir: Path, subdir: str, slug: str, title: str = "", body: str = "") -> None:
    d = wiki_dir / "wiki" / subdir
    d.mkdir(parents=True, exist_ok=True)
    if not title and not body:
        content = "---\n{}\n---\n"
    else:
        content = f"---\ntitle: {title}\ncategory: {subdir[:-1] if subdir != 'entités' else 'entité'}\n---\n\n{body}\n"
    (d / f"{slug}.md").write_text(content, encoding="utf-8")


def test_tokenize_desuffixe_singulier_et_pluriel():
    assert tokenize("politicien") == tokenize("politiciens")


def test_build_index_exclut_les_pages_vides(tmp_path):
    _make_page(tmp_path, "concepts", "c-vide")
    _make_page(tmp_path, "sources", "src-reel", title="Un sujet réel", body="Du contenu exploitable.")

    ws = WikiSearch(tmp_path)

    slugs = {p["slug"] for p in ws._pages}
    assert slugs == {"src-reel"}


def test_recherche_singulier_trouve_ce_que_le_pluriel_trouve(tmp_path):
    _make_page(
        tmp_path, "sources", "src-politiciens",
        title="Les politiciens et le monopole étatique",
        body="Les politiciens tirent leur notoriété du monopole étatique.",
    )

    ws = WikiSearch(tmp_path)

    singulier = {r.slug for r in ws.search("politicien")}
    pluriel = {r.slug for r in ws.search("politiciens")}
    assert singulier == pluriel == {"src-politiciens"}


def test_embed_load_pages_exclut_les_pages_vides(tmp_path):
    from embed import load_pages

    wiki_dir = tmp_path / "wiki"
    _make_page(tmp_path, "concepts", "c-vide")
    _make_page(tmp_path, "sources", "src-reel", title="Un sujet réel", body="Du contenu exploitable.")

    pages = load_pages(wiki_dir)

    assert [p["slug"] for p in pages] == ["src-reel"]


def test_search_ne_remplit_pas_avec_le_plancher_bm25plus(tmp_path):
    """Régression (2026-09-10) : BM25+ (rank_bm25) ajoute delta*idf à CHAQUE
    document même sans occurrence du terme — un score positif ne veut donc
    pas dire présence réelle. Sans filtre, /r <mot rare> se voyait complété
    jusqu'à top_k par des pages totalement sans rapport, à un score plancher
    identique entre elles (observé en production : "renard" ramenait Attention
    Is All You Need, une page sur Anthropic, une commande restic…)."""
    _make_page(tmp_path, "sources", "src-renard", title="Le renard roux",
               body="Étude du renard roux et de sa domestication en laboratoire.")
    _make_page(tmp_path, "sources", "src-transformer", title="Attention Is All You Need",
               body="Architecture Transformer fondée sur les mécanismes d'attention.")
    _make_page(tmp_path, "sources", "src-restic", title="Commande restic",
               body="Lister les snapshots d'un dépôt de sauvegarde restic.")

    ws = WikiSearch(tmp_path)

    results = ws.search("renard", top_k=10)

    assert [r.slug for r in results] == ["src-renard"]


def _sr(slug: str, score: float) -> SearchResult:
    from pathlib import Path
    return SearchResult(slug=slug, path=Path(f"{slug}.md"), title=slug,
                         category="source", score=score, excerpt="")


def test_hybrid_search_bm25_reel_domine_le_bruit_semantique():
    """Régression (2026-09-10) : sur "renard", BM25 (corrigé) trouve le bon
    résultat en tête ; le sémantique brut classe trois pages sans rapport
    devant lui (BGE-M3, requête d'un mot, corpus restreint et hétérogène —
    non un bug de données, une limite inhérente au signal). Sans pondération,
    RRF classique laissait ce bruit à égalité avec un vrai second résultat
    BM25 (c-domestication). La pondération 2:1 doit les séparer nettement."""
    bm25 = [_sr("src-fox", 15.7), _sr("c-domestication", 11.6)]
    semantic = [
        _sr("c-bruit-1", 0.435),
        _sr("e-bruit-2", 0.408),
        _sr("e-bruit-3", 0.402),
        _sr("src-fox", 0.388),
    ]

    ranked = hybrid_search(bm25, semantic, top_k=5)
    slugs = [r.slug for r in ranked]

    assert slugs[0] == "src-fox"
    assert slugs[1] == "c-domestication"
    assert set(slugs[2:]) == {"c-bruit-1", "e-bruit-2", "e-bruit-3"}


def test_hybrid_search_semantique_seul_garde_son_poids_normal():
    # Sans concurrence BM25, un résultat purement sémantique n'est pas
    # pénalisé — la pondération ne joue qu'en cas de compétition.
    semantic = [_sr("c-paraphrase", 0.60)]

    ranked = hybrid_search([], semantic, top_k=5)

    assert [r.slug for r in ranked] == ["c-paraphrase"]
