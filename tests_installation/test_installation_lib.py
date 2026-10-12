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


def _install(args, tmp_path):
    # HOME temporaire : même si install.sh allait plus loin que prévu, il
    # n'écrirait jamais dans la configuration de la machine (incident du 2026-10-12).
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    return subprocess.run(["bash", str(RACINE / "install.sh"), *args], capture_output=True, text=True,
                          timeout=30, stdin=subprocess.DEVNULL, env=env)


def test_install_sans_profil_refuse(tmp_path):
    r = _install([], tmp_path)
    assert r.returncode == 2 and "--profil" in r.stdout + r.stderr


def test_install_profil_inconnu_refuse(tmp_path):
    r = _install(["--profil", "moyen"], tmp_path)
    assert r.returncode != 0 and "Profil inconnu" in r.stderr
