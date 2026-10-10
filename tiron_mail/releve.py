"""Relève de tiron@secretarius.co : n'exécute que les commandes des messages de
l'utilisateur authentifiés par Proton (spec courrier, § Relève). Le corps d'un
message refusé n'est jamais transmis."""
from __future__ import annotations

import email
import email.policy
import json
import re
import subprocess
from collections.abc import Mapping
from email.message import Message
from email.utils import getaddresses, parseaddr
from pathlib import Path

from tiron_mail.boite import Boite, ErreurBoite, _connecter_imap
from tiron_mail.config import ErreurConfig
from tiron_mail.message import analyser

TRAITE, REFUSE = "Folders/Traite", "Folders/Refuse"
ETAT_DEFAUT = Path.home() / ".local" / "state" / "tiron_mail" / "releve.json"
_SEUIL_ECHECS = 10

_DMARC = re.compile(r"\bdmarc=(\w+)")


def authentifier(m: Message, proprietaires: tuple) -> str:
    """"" si le message vient de l'utilisateur et que Proton l'authentifie, sinon le motif.
    Proton remplace les X-Pm-* fournis par l'expéditeur (vérifié le 2026-10-10)."""
    froms = getaddresses([str(v) for v in m.get_all("From", [])])
    if len(froms) != 1:
        return "expéditeur absent ou multiple"
    exp = froms[0][1].lower()
    if exp not in proprietaires:
        return f"expéditeur non autorisé ({exp})"
    if parseaddr(str(m.get("Return-Path", "")))[1].lower() != exp:
        return "Return-Path différent de l'expéditeur"
    origine = [str(v).strip() for v in m.get_all("X-Pm-Origin", [])]
    chiffrement = [str(v).strip() for v in m.get_all("X-Pm-Content-Encryption", [])]
    if origine == ["internal"]:
        return "" if chiffrement == ["end-to-end"] else "message interne non chiffré de bout en bout"
    if origine != ["external"]:
        return "origine Proton inconnue"
    dmarc = [r.group(1) for v in m.get_all("Authentication-Results", [])
             if str(v).strip().startswith("mail.protonmail.ch;") and (r := _DMARC.search(str(v)))]
    if not dmarc or any(d != "pass" for d in dmarc):
        return "DMARC non validé par Proton"
    return ""


def commandes(objet: str, texte: str) -> list[str]:
    """Objet et lignes du corps qui commencent par « / » (les lignes citées commencent par « > »)."""
    return [l.strip() for l in [objet, *texte.splitlines()] if l.strip().startswith("/")]


def _lire_etat(chemin: Path) -> dict:
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"traites": [], "echecs": 0}


def _ecrire_etat(chemin: Path, etat: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_suffix(".tmp")
    tmp.write_text(json.dumps(etat), encoding="utf-8")
    tmp.replace(chemin)


def relever(cfg, env, executer, notifier, etat_chemin: Path = ETAT_DEFAUT, connecter=_connecter_imap) -> str:
    if not cfg.releve:
        raise ErreurConfig("aucune boîte à relever ([releve] boite = …)")
    compte = cfg.comptes[cfg.releve]
    etat = _lire_etat(etat_chemin)
    b = Boite(compte, env, connecter)
    try:
        conn = b._ouvrir(lecture_seule=False)
    except ErreurBoite as e:
        etat["echecs"] = etat.get("echecs", 0) + 1
        _ecrire_etat(etat_chemin, etat)
        if etat["echecs"] == _SEUIL_ECHECS:
            notifier(f"Relève de {compte.adresse} en échec depuis {_SEUIL_ECHECS} essais : {e}")
        raise
    etat["echecs"] = 0
    bilan: list[str] = []
    try:
        for dossier in (TRAITE, REFUSE):
            conn.create(dossier)  # « NO » si le dossier existe : sans effet
        typ, data = conn.uid("search", None, "ALL")
        for u in (data[0] or b"").split() if typ == "OK" else []:
            brut = b._fetch(conn, int(u), "(RFC822)")  # sans PEEK : marque le message lu
            m = email.message_from_bytes(brut, policy=email.policy.default)
            mid = str(m.get("Message-ID", "")).strip()
            exp = parseaddr(str(m.get("From", "")))[1]
            if mid and mid in etat["traites"]:
                conn.uid("move", u.decode(), TRAITE)
                continue
            motif = authentifier(m, cfg.adresses_proprietaire) or ("" if mid else "message sans Message-ID")
            if motif:
                notifier(f"Courriel refusé sur {compte.adresse} : de {exp}, {motif}. Rien n'a été exécuté.")
                conn.uid("move", u.decode(), REFUSE)
                bilan.append(f"refusé : {exp} ({motif})")
                continue
            etat["traites"].append(mid)
            _ecrire_etat(etat_chemin, etat)  # avant l'exécution : jamais deux fois
            a = analyser(brut)
            cmds = commandes(a["objet"], a["texte"])
            if not cmds:
                notifier(f"Courriel de {exp} reçu sur {compte.adresse} sans commande : "
                         "écrivez une ligne qui commence par /, par exemple /wikistatus.")
            for c in cmds:
                try:
                    executer(c)
                except Exception as e:  # les commandes suivantes passent quand même
                    notifier(f"Commande {c} (courriel de {exp}) en échec : {e}")
            conn.uid("move", u.decode(), TRAITE)
            bilan.append(f"{exp} : {len(cmds)} commande(s)")
    finally:
        _ecrire_etat(etat_chemin, etat)
        try:
            conn.logout()
        except Exception:
            pass
    return "\n".join(bilan) or "Rien à relever."


def executer_openclaw(env: Mapping[str, str]):
    """Injecte la commande comme un message Telegram ; la réponse part sur Telegram."""
    def executer(commande: str) -> None:
        exe = env.get("OPENCLAW_BIN") or str(Path.home() / ".local" / "bin" / "openclaw")
        subprocess.run([exe, "agent", "--agent", "main", "--session-key", "agent:main:courrier",
                        "--message", commande, "--deliver", "--reply-channel", "telegram",
                        "--reply-to", env["TIRON_TELEGRAM_CHAT_ID"], "--json"],
                       capture_output=True, text=True, timeout=600, check=True)
    return executer
