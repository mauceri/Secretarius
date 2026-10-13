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
    assert r.stdout.strip() == "wiki scout obfuscateur porte|modal|obfusque"


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


def test_dependances_python_torch_cpu_et_tmpdir_sur_disque(tmp_path):
    """Ubuntu 26.04 : /tmp en tmpfs à quota ; torch CUDA par défaut (plusieurs Go)
    l'épuise (santiago, 2026-10-13). torch CPU d'abord, pip hors de /tmp."""
    journal = tmp_path / "pip.log"
    pip = tmp_path / "pip"
    pip.write_text(f'#!/bin/bash\necho "$TMPDIR|$*" >> {journal}\n')
    pip.chmod(0o755)
    r = _bash(f'installer_dependances_python {pip} /r/requirements.txt',
              env={"HOME": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    appels = journal.read_text().splitlines()
    assert len(appels) == 2
    assert "torch" in appels[0] and "download.pytorch.org/whl/cpu" in appels[0]
    assert "-r /r/requirements.txt" in appels[1]
    for a in appels:
        assert a.split("|")[0].startswith(str(tmp_path / ".cache"))


def test_nvm_charge_meme_si_enveloppe_openclaw_presente(tmp_path):
    """Relance par SSH : l'enveloppe ~/.local/bin/openclaw du premier passage
    est dans le PATH, mais npm (NVM) non — NVM doit quand même être chargé."""
    env_bin = tmp_path / "bin"
    env_bin.mkdir()
    (env_bin / "openclaw").write_text("#!/bin/sh\n")
    (env_bin / "openclaw").chmod(0o755)
    (env_bin / "bash").symlink_to("/bin/bash")  # PATH isolé : pas de /usr/bin/npm
    nvm_bin = tmp_path / ".nvm" / "bin"
    nvm_bin.mkdir(parents=True)
    (nvm_bin / "npm").write_text("#!/bin/sh\n")
    (nvm_bin / "npm").chmod(0o755)
    (tmp_path / ".nvm" / "nvm.sh").write_text(f'PATH="{nvm_bin}:$PATH"\n')
    r = _bash("charger_nvm; command -v npm",
              env={"HOME": str(tmp_path), "PATH": str(env_bin)})
    assert r.returncode == 0 and r.stdout.strip() == str(nvm_bin / "npm"), r.stderr


def _faux_openclaw(tmp_path, deepseek_present):
    journal = tmp_path / "oc.log"
    oc = tmp_path / "openclaw"
    oc.write_text(f'''#!/bin/bash
echo "$*" >> {journal}
[[ "$1" == "--version" ]] && {{ echo "OpenClaw 2026.8.2 (0965053)"; exit 0; }}
[[ "$1 $2" == "plugins info" ]] && exit {0 if deepseek_present else 1}
exit 0
''')
    oc.chmod(0o755)
    return oc, journal


def test_plugin_deepseek_installe_a_la_version_openclaw(tmp_path):
    """deepseek n'est pas livré avec OpenClaw : paquet npm à installer (santiago, 2026-10-13)."""
    oc, journal = _faux_openclaw(tmp_path, deepseek_present=False)
    r = _bash(f"installer_plugin_deepseek {oc}")
    assert r.returncode == 0, r.stderr
    assert "plugins install @openclaw/deepseek-provider@2026.8.2 --accept-capabilities" \
        in journal.read_text().splitlines()


def test_plugin_deepseek_deja_present_non_reinstalle(tmp_path):
    oc, journal = _faux_openclaw(tmp_path, deepseek_present=True)
    r = _bash(f"installer_plugin_deepseek {oc}")
    assert r.returncode == 0
    assert not any(l.startswith("plugins install") for l in journal.read_text().splitlines())


def test_preparer_wiki_cree_le_squelette(tmp_path):
    """Coffre neuf (santiago, 2026-10-13) : sans wiki/, wiki-lm-server refuse de démarrer."""
    w = tmp_path / "Publication" / "Wiki_LM"
    r = _bash(f'preparer_wiki "{w}"')
    assert r.returncode == 0, r.stderr
    assert (w / "wiki").is_dir() and (w / "raw").is_dir()


def _faux_systemd_run(tmp_path, groupes):
    sr = tmp_path / "systemd-run"
    sr.write_text(f'#!/bin/sh\necho "{groupes}"\n')
    sr.chmod(0o755)
    return sr


def test_gestionnaire_systemd_sans_groupe_docker_refuse(tmp_path):
    """santiago, 2026-10-13 : systemd --user lancé avant l'ajout au groupe docker ;
    la gateway ne pouvait pas joindre docker.sock (permission denied)."""
    sr = _faux_systemd_run(tmp_path, "mauceric sudo users")
    r = _bash(f"SYSTEMD_RUN={sr}; verifier_groupe_docker_systemd")
    assert r.returncode == 1
    assert "user@" in r.stdout + r.stderr


def test_gestionnaire_systemd_avec_groupe_docker(tmp_path):
    sr = _faux_systemd_run(tmp_path, "mauceric sudo users docker")
    r = _bash(f"SYSTEMD_RUN={sr}; verifier_groupe_docker_systemd")
    assert r.returncode == 0, r.stdout + r.stderr


def test_profil_lourd_sans_porte():
    r = _bash('charger_profil lourd && echo "$MODULES"')
    assert "porte" not in r.stdout.split()


def _faux_tailscale(tmp_path, sortie):
    t = tmp_path / "tailscale"
    t.write_text(f'#!/bin/sh\necho "{sortie}"\n')
    t.chmod(0o755)
    return t


def test_funnel_porte_actif(tmp_path):
    t = _faux_tailscale(tmp_path, "https://santiago.x.ts.net:8443 (Funnel on)\\n|-- / proxy http://127.0.0.1:5052")
    assert _bash(f"TAILSCALE={t}; etat_funnel_porte").returncode == 0


def test_funnel_porte_absent(tmp_path):
    t = _faux_tailscale(tmp_path, "No serve config")
    assert _bash(f"TAILSCALE={t}; etat_funnel_porte").returncode == 1


def test_comptes_a_lier_initialise_avant_les_modules():
    texte = (RACINE / "install.sh").read_text()
    assert texte.index("COMPTES_A_LIER=()") < texte.index("for _m in $MODULES")
