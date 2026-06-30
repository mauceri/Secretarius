# gen_corpus/tests/test_convert_seed.py
import json
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from convert_seed import extract_args, infer_variante, infer_registre, parse_seed

MINI_SEED = """\
## 1. `wiki_capture` — capturer

1. garde cet article pour moi : https://example.com #ia
2. /c #ia https://example.com
3. /c @simple #notion ma note
4. /c ref:bm25-intro note sur BM25
5. /c file:/home/user/doc.md #ia

## 2. `wiki_ingest` — ingérer

1. ingère les fichiers en attente
2. /ingest

## 10. `out_of_scope` — hors périmètre

1. commande une pizza
2. réserve un billet de train
"""

INTENTIONS = [
    {"intention": "wiki_capture",  "command": "/c",      "variantes": ["url_avec_tags","url_seule","note_sans_url","avec_directive_simple","avec_ref","avec_fichier"]},
    {"intention": "wiki_ingest",   "command": "/ingest", "variantes": ["sans_args"]},
    {"intention": "wiki_query",    "command": "/q",      "variantes": ["question_courte","question_longue"]},
    {"intention": "out_of_scope",  "command": None,      "variantes": ["action_impossible"]},
]


def test_extract_args_url_tags():
    result = extract_args("garde cet article : https://example.com #ia", "wiki_capture")
    assert "https://example.com" in result
    assert "#ia" in result

def test_extract_args_slash_command():
    result = extract_args("/c #ia https://example.com", "wiki_capture")
    assert "#ia" in result
    assert "https://example.com" in result

def test_extract_args_directive_simple():
    result = extract_args("/c @simple #notion ma note", "wiki_capture")
    assert "@simple" in result
    assert "#notion" in result

def test_extract_args_ref():
    result = extract_args("/c ref:bm25-intro note sur BM25", "wiki_capture")
    assert "ref:bm25-intro" in result

def test_extract_args_no_args_ingest():
    assert extract_args("ingère les fichiers en attente", "wiki_ingest") == ""

def test_extract_args_no_args_slash_ingest():
    assert extract_args("/ingest", "wiki_ingest") == ""

def test_infer_variante_url_tags():
    assert infer_variante("garde ce lien https://ex.com #ia", "wiki_capture") == "url_avec_tags"

def test_infer_variante_directive_simple():
    assert infer_variante("/c @simple #notion ma note", "wiki_capture") == "avec_directive_simple"

def test_infer_variante_ref():
    assert infer_variante("/c ref:bm25-intro note", "wiki_capture") == "avec_ref"

def test_infer_registre_slash():
    assert infer_registre("/c https://ex.com") == "télégraphique"

def test_infer_registre_stp():
    assert infer_registre("ingestion stp") == "familier"

def test_parse_seed_structure(tmp_path):
    seed_md = tmp_path / "seed.md"
    seed_md.write_text(MINI_SEED, encoding="utf-8")
    entries = parse_seed(str(seed_md), INTENTIONS)
    assert len(entries) > 0
    for e in entries:
        assert set(e.keys()) >= {"text", "intention", "registre", "variante", "action"}
        assert set(e["action"].keys()) == {"command", "args"}

def test_out_of_scope_null_command(tmp_path):
    seed_md = tmp_path / "seed.md"
    seed_md.write_text(MINI_SEED, encoding="utf-8")
    entries = parse_seed(str(seed_md), INTENTIONS)
    oos = [e for e in entries if e["intention"] == "out_of_scope"]
    assert len(oos) >= 2
    assert all(e["action"]["command"] is None for e in oos)

def test_wiki_ingest_no_args(tmp_path):
    seed_md = tmp_path / "seed.md"
    seed_md.write_text(MINI_SEED, encoding="utf-8")
    entries = parse_seed(str(seed_md), INTENTIONS)
    ingest = [e for e in entries if e["intention"] == "wiki_ingest"]
    assert all(e["action"]["args"] == "" for e in ingest)

def test_extract_args_backtick_wrapped():
    # Les lignes markdown avec backticks doivent être normalisées
    result = extract_args("/c https://marp.app/#get-started", "wiki_capture")
    assert "https://marp.app/#get-started" in result
    assert "`" not in result

def test_parse_seed_backtick_lines(tmp_path):
    seed_md = tmp_path / "seed.md"
    seed_md.write_text("""\
## 1. `wiki_capture` — capturer

1. `/c https://marp.app/#get-started`
2. `/c #markdown #presentation https://marp.app/#get-started`

## 4. `wiki_query` — interroger

1. `/q comment fonctionne le Zettelkasten ?`
""", encoding="utf-8")
    entries = parse_seed(str(seed_md), INTENTIONS)
    captures = [e for e in entries if e["intention"] == "wiki_capture"]
    assert all("`" not in e["action"]["args"] for e in captures)
    queries = [e for e in entries if e["intention"] == "wiki_query"]
    assert all("`" not in e["action"]["args"] for e in queries)
