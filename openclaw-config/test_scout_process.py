"""Délai d'attente du détecteur d'injection (essai réel 2026-11-18 : ~16 s sur une grosse page)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scout_process  # noqa: E402


def test_delai_du_garde_couvre_une_grosse_page_sous_le_delai_de_source():
    # /source attend scout-watcher 60 s au plus : le garde doit répondre avant.
    assert 30 <= scout_process.GUARD_TIMEOUT < 60
