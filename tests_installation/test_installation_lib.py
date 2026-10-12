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


def test_env_wiki_profil_leger(tmp_path):
    import shutil
    racine = tmp_path / "Secretarius"
    shutil.copytree(RACINE / "Wiki_LM" / "tools", racine / "Wiki_LM" / "tools",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(RACINE / "Wiki_LM" / ".env.template", racine / "Wiki_LM" / ".env.template")
    (racine / "Wiki_LM" / ".venv").symlink_to(Path.home() / "Secretarius" / "Wiki_LM" / ".venv")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "WIKI_PATH": "/coffre/Wiki_LM",
           "WIKI_ZIM_DIR": "/zims", "WIKI_LLM": "obfusque", "MODULES": "wiki scout obfuscateur",
           "DEEPSEEK_API_KEY": "cle-test", "FORCE": "false"}
    r = subprocess.run(["bash", "-c", f'SECRETARIUS_ROOT="{racine}"; source "{RACINE}/lib/installation.sh"; ecrire_env_wiki'],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    contenu = (racine / "Wiki_LM" / ".env").read_text()
    for ligne in ["WIKI_LLM_BACKEND=openai", "OPENAI_BASE_URL=http://172.17.0.1:8001/v1",
                  "WIKI_PATH=/coffre/Wiki_LM", "WIKI_ZIM_DIR=/zims", "WIKI_LOOKUP_OFFLINE=1",
                  "WIKI_INGEST_WEB_LLM_BACKEND=openai", "DEEPSEEK_API_KEY=cle-test",
                  "OBFUSQUE_BASE_URL=http://172.17.0.1:8001/v1"]:
        assert ligne in contenu.splitlines(), ligne


def test_verification_tout_repond():
    r = _bash('SYSTEMCTL=true; CURL=true; PAUSE=0; SERVICES_A_VERIFIER=("a|http://x" "b"); verifier_services')
    assert r.returncode == 0


def test_verification_service_muet():
    r = _bash('SYSTEMCTL=true; CURL=false; PAUSE=0; SERVICES_A_VERIFIER=("wiki-lm-server|http://x"); verifier_services')
    assert r.returncode == 1 and "wiki-lm-server" in r.stdout + r.stderr


def test_verification_unite_inactive():
    r = _bash('SYSTEMCTL=false; CURL=true; PAUSE=0; SERVICES_A_VERIFIER=("openclaw-scout"); verifier_services')
    assert r.returncode == 1 and "openclaw-scout" in r.stdout + r.stderr


def _racine_wiki(tmp_path):
    import shutil
    racine = tmp_path / "Secretarius"
    shutil.copytree(RACINE / "Wiki_LM" / "tools", racine / "Wiki_LM" / "tools",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(RACINE / "Wiki_LM" / ".env.template", racine / "Wiki_LM" / ".env.template")
    (racine / "Wiki_LM" / ".venv").symlink_to(Path.home() / "Secretarius" / "Wiki_LM" / ".venv")
    return racine


def test_env_wiki_existant_preserve_sans_force(tmp_path):
    racine = _racine_wiki(tmp_path)
    (racine / "Wiki_LM" / ".env").write_text(
        "WIKI_LLM_BACKEND=ollama\nOLLAMA_MODEL=qwen3:8b\nWIKI_PATH=/ancien\n"
        "WIKI_VAULT_MIRRORS=Arbath=/home/x/Arbath\n")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "WIKI_PATH": "/coffre/Wiki_LM",
           "WIKI_ZIM_DIR": "/zims", "WIKI_LLM": "obfusque", "MODULES": "wiki obfuscateur", "FORCE": "false"}
    r = subprocess.run(["bash", "-c", f'SECRETARIUS_ROOT="{racine}"; source "{RACINE}/lib/installation.sh"; ecrire_env_wiki'],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lignes = (racine / "Wiki_LM" / ".env").read_text().splitlines()
    assert "WIKI_VAULT_MIRRORS=Arbath=/home/x/Arbath" in lignes
    assert "WIKI_LLM_BACKEND=ollama" in lignes and "WIKI_PATH=/ancien" in lignes
    assert "WIKI_ZIM_DIR=/zims" in lignes  # clé absente : ajoutée


def test_jeton_telegram_different_refuse(tmp_path):
    f = tmp_path / "gateway.systemd.env"
    f.write_text("TELEGRAM_BOT_TOKEN=jeton-ancien\n")
    r = _bash(f'verifier_jeton_telegram "{f}"', {"TELEGRAM_BOT_TOKEN": "jeton-nouveau", "FORCE": "false"})
    assert r.returncode == 1 and "--force" in r.stderr


def test_jeton_telegram_identique_ou_force(tmp_path):
    f = tmp_path / "gateway.systemd.env"
    f.write_text("TELEGRAM_BOT_TOKEN=jeton\n")
    assert _bash(f'verifier_jeton_telegram "{f}"', {"TELEGRAM_BOT_TOKEN": "jeton"}).returncode == 0
    assert _bash(f'verifier_jeton_telegram "{f}"', {"TELEGRAM_BOT_TOKEN": "autre", "FORCE": "true"}).returncode == 0
    assert _bash(f'verifier_jeton_telegram "{tmp_path}/absent"', {"TELEGRAM_BOT_TOKEN": "x"}).returncode == 0


def test_ancien_proxy_obfuscateur_refuse(tmp_path):
    (tmp_path / "obfuscator-proxy-docker.service").write_text("[Service]\n")
    r = _bash("verifier_proxy_obfuscateur", {"SYSTEMD_USER_DIR": str(tmp_path)})
    assert r.returncode == 1 and "obfuscator-proxy-docker" in r.stderr
    assert _bash("verifier_proxy_obfuscateur", {"SYSTEMD_USER_DIR": str(tmp_path / "vide")}).returncode == 0
