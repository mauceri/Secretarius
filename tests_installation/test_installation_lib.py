"""Fonctions de l'installateur (lib/installation.sh), sans aucun effet système."""
import os
import subprocess
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent


def _bash(script, env=None):
    e = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], **(env or {})}
    return subprocess.run(
        ["bash", "-c", f'SECRETARIUS_ROOT="{RACINE}"; source lib/installation.sh; {script}'],
        cwd=RACINE, env=e, capture_output=True, text=True)


def test_profil_leger():
    r = _bash('charger_profil leger && echo "$MODULES|$MAIN_BRAIN|$WIKI_LLM"')
    assert r.stdout.strip() == "wiki scout obfuscateur|modal|obfusque"


def test_profil_inconnu():
    r = _bash("charger_profil moyen")
    assert r.returncode == 1 and "Profil inconnu" in r.stderr


def test_variable_explicite_prioritaire():
    r = _bash('charger_profil leger && echo "$MODULES"', {"MODULES": "wiki"})
    assert r.stdout.strip() == "wiki"


def test_zim_variable_valide(tmp_path):
    (tmp_path / "wikipedia_fr.zim").write_text("x")
    r = _bash("resoudre_zim", {"WIKI_ZIM_DIR": str(tmp_path)})
    assert r.returncode == 0 and r.stdout.strip() == str(tmp_path)


def test_zim_variable_invalide_echoue(tmp_path):
    r = _bash("resoudre_zim", {"WIKI_ZIM_DIR": str(tmp_path)})
    assert r.returncode == 1 and "sans fichier .zim" in r.stderr


def test_zim_absent_non_interactif(tmp_path):
    r = subprocess.run(
        ["bash", "-c", f'SECRETARIUS_ROOT="{tmp_path}"; source "{RACINE}/lib/installation.sh"; resoudre_zim'],
        capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == ""
