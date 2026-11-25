"""Veille du broker Session (spec canal Session, § 5.6) : lancée toutes les 5 min par
session-broker-veille.timer. Une alerte par courriel par incident, puis un courriel
de rétablissement. Ne lit dans secrets.env que le jeton du broker : la phrase secrète
n'entre jamais dans son environnement."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SECRETS = Path.home() / ".config" / "secrets.env"
ETAT = Path.home() / ".local" / "state" / "session-broker" / "veille.json"
URL = "http://127.0.0.1:8091/health"
CLE = "TIRON_SESSION_BROKER_TOKEN"


def lire_cle(chemin: Path, cle: str) -> str:
    """Valeur de `cle` dans un fichier KEY=valeur, apostrophes retirées ; ne lit aucune autre clé."""
    try:
        lignes = chemin.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for ligne in lignes:
        if ligne.startswith(cle + "="):
            return ligne.split("=", 1)[1].strip().strip("'\"")
    return ""


def lire_jeton(chemin: Path) -> str:
    return lire_cle(chemin, CLE)


def sonde_broker(jeton: str, url: str = URL, ouvrir=urllib.request.urlopen) -> Callable[[], dict]:
    def sonder() -> dict:
        if not jeton:
            raise RuntimeError(f"jeton absent ({CLE})")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {jeton}"})
        with ouvrir(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    return sonder


def _lire_etat(chemin: Path) -> dict:
    try:
        etat = json.loads(chemin.read_text(encoding="utf-8"))
        depuis = etat.get("panne_depuis")
        if depuis is not None:
            datetime.fromisoformat(depuis)
        if not isinstance(etat.get("alerte", False), bool):
            return {}
        return etat
    except (OSError, ValueError, TypeError, AttributeError):  # illisible ou incohérent
        return {}


def _ecrire_etat(chemin: Path, etat: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_suffix(".tmp")
    tmp.write_text(json.dumps(etat), encoding="utf-8")
    tmp.replace(chemin)


def _heure(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(ZoneInfo("Europe/Paris")).strftime("%H:%M")


def veiller(sonde: Callable[[], dict], notifier: Callable[[str], None], maintenant: datetime,
            etat_chemin: Path = ETAT, seuil: timedelta = timedelta(minutes=15)) -> str:
    etat = _lire_etat(etat_chemin)
    motif, depuis, delai = "", maintenant, False
    try:
        sante = sonde()
        dernier = sante.get("lastPollOkAt")
        if not dernier:  # broker qui démarre : alerte seulement au-delà du seuil
            motif, delai = "aucune interrogation réussie du réseau", True
        elif maintenant - datetime.fromisoformat(dernier) > seuil:
            motif, depuis = "réseau Session muet", datetime.fromisoformat(dernier)
    except Exception as e:  # arrêté, injoignable, réponse illisible
        motif = f"broker injoignable : {e}"
    if motif:
        etat.setdefault("panne_depuis", None)
        etat["panne_depuis"] = etat["panne_depuis"] or depuis.isoformat()
        attendre = delai and maintenant - datetime.fromisoformat(etat["panne_depuis"]) <= seuil
        if not etat.get("alerte") and not attendre:
            try:
                notifier(f"Session : Tiron ne reçoit plus depuis {_heure(etat['panne_depuis'])} ({motif}).")
                etat["alerte"] = True
            except Exception as e:  # réessayée au passage suivant
                print(f"alerte non envoyée ({e})", file=sys.stderr)
        _ecrire_etat(etat_chemin, etat)
        return f"panne : {motif}"
    if etat.get("alerte"):
        try:
            notifier(f"Session : Tiron reçoit de nouveau (panne depuis {_heure(etat['panne_depuis'])}).")
        except Exception as e:  # réessayé au passage suivant
            print(f"rétablissement non signalé ({e})", file=sys.stderr)
            return "rétabli, avis en attente"
    _ecrire_etat(etat_chemin, {"panne_depuis": None, "alerte": False})
    return "ok"


def main() -> int:
    from tiron_mail.avis import notifier_courriel
    print(veiller(sonde_broker(lire_jeton(SECRETS)), notifier_courriel(os.environ),
                  datetime.now(timezone.utc)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
