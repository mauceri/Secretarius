"""porte-pare-feu.sh, exécuté contre un faux iptables qui tient l'état des chaînes."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "porte-conteneur" / "porte-pare-feu.sh"

FAUX = r'''
import json, os, sys
etat_f = os.environ["FAUX_ETAT"]
etat = json.load(open(etat_f))
with open(os.environ["FAUX_JOURNAL"], "a") as j:
    j.write(" ".join(sys.argv[1:]) + "\n")
op, chaine, regle = sys.argv[1], sys.argv[2], sys.argv[3:]
if op == "-L":
    sys.exit(0 if chaine in etat else 1)
if op == "-N":
    etat.setdefault(chaine, [])
elif chaine not in etat:
    sys.exit(1)
elif op == "-C":
    sys.exit(0 if regle in etat[chaine] else 1)
elif op == "-I":
    etat[chaine].insert(0, regle)
elif op == "-D":
    if regle not in etat[chaine]:
        sys.exit(1)
    etat[chaine].remove(regle)
json.dump(etat, open(etat_f, "w"))
'''

EST_RETURN = "-i br-porte -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN".split()
EST_ACCEPT = "-i br-porte -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT".split()
OLLAMA = "-i br-porte -p tcp --dport 11434 -j ACCEPT".split()
DROP = "-i br-porte -j DROP".split()


@pytest.fixture
def faux(tmp_path):
    faux_py = tmp_path / "faux_iptables.py"
    faux_py.write_text(FAUX)
    iptables = tmp_path / "iptables"
    iptables.write_text(f"#!/bin/sh\nexec {sys.executable} {faux_py} \"$@\"\n")
    iptables.chmod(0o755)
    docker = tmp_path / "docker"
    docker.write_text(f"#!/bin/sh\necho \"docker $*\" >> {tmp_path / 'journal'}\n"
                      f"[ -e {tmp_path / 'temoin'} ] && echo temoin-present >> {tmp_path / 'journal'}\nexit 0\n")
    docker.chmod(0o755)
    etat = tmp_path / "etat.json"
    etat.write_text(json.dumps({"INPUT": [], "DOCKER-USER": []}))
    env = dict(os.environ, IPTABLES=str(iptables), DOCKER=str(docker), TEMOIN=str(tmp_path / "temoin"),
               FAUX_ETAT=str(etat), FAUX_JOURNAL=str(tmp_path / "journal"))

    class Faux:
        temoin = tmp_path / "temoin"
        journal = tmp_path / "journal"

        def lancer(self, action):
            return subprocess.run(["bash", str(SCRIPT), action], env=env,
                                  capture_output=True, text=True).returncode

        def etat(self):
            return json.loads(etat.read_text())

        def sans_docker_user(self):
            etat.write_text(json.dumps({"INPUT": []}))

    return Faux()


def test_poser_ecrit_les_regles_dans_l_ordre(faux):
    assert faux.lancer("poser") == 0
    assert faux.etat()["DOCKER-USER"] == [EST_RETURN, DROP]
    assert faux.etat()["INPUT"] == [EST_ACCEPT, OLLAMA, DROP]


def test_poser_deux_fois_sans_doublon(faux):
    faux.lancer("poser")
    premier = faux.etat()
    assert faux.lancer("poser") == 0
    assert faux.etat() == premier


def test_poser_cree_docker_user_si_absente(faux):
    faux.sans_docker_user()
    assert faux.lancer("poser") == 0
    assert "-N DOCKER-USER" in faux.journal.read_text().splitlines()
    assert faux.etat()["DOCKER-USER"] == [EST_RETURN, DROP]


def test_poser_ecrit_le_temoin(faux):
    faux.lancer("poser")
    assert faux.temoin.exists()
    assert faux.temoin.stat().st_mode & 0o777 == 0o644


def test_retirer_enleve_regles_et_temoin(faux):
    faux.lancer("poser")
    assert faux.lancer("retirer") == 0
    assert faux.etat() == {"INPUT": [], "DOCKER-USER": []}
    assert not faux.temoin.exists()


def test_verifier_echoue_sans_temoin(faux):
    assert faux.lancer("verifier") == 1


def test_verifier_reussit_avec_temoin(faux):
    faux.lancer("poser")
    assert faux.lancer("verifier") == 0


def test_action_inconnue(faux):
    assert faux.lancer("ouvrir") == 2


def test_retirer_arrete_le_conteneur_avant_d_ouvrir(faux):
    """Sinon le conteneur déjà lancé tourne sans filtre (revue finale 2026-11-12)."""
    faux.lancer("poser")
    faux.journal.write_text("")
    assert faux.lancer("retirer") == 0
    lignes = faux.journal.read_text().splitlines()
    arret = lignes.index("docker stop porte-grok")
    assert "temoin-present" not in lignes  # témoin effacé avant l'arrêt : pas de relance
    assert all(not l.startswith("-D") for l in lignes[:arret])
