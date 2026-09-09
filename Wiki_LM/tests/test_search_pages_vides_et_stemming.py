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

from search import WikiSearch, tokenize


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
