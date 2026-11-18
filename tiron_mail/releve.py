"""Relève de tiron@secretarius.co : n'exécute que les commandes des messages de
l'utilisateur authentifiés par Proton (spec courrier, § Relève). Le corps d'un
message refusé n'est jamais transmis."""
from __future__ import annotations

import email
import email.policy
import json
import re
import subprocess
import sys
import uuid
from collections.abc import Mapping
from email.message import Message
from email.utils import getaddresses, parseaddr
from pathlib import Path

from tiron_mail.boite import Boite, ErreurBoite, _connecter_imap
from tiron_mail.config import ErreurConfig
from tiron_mail.envoi import _smtp, construire, envoyer
from tiron_mail.message import analyser

TRAITE, REFUSE = "Folders/Traite", "Folders/Refuse"
ETAT_DEFAUT = Path.home() / ".local" / "state" / "tiron_mail" / "releve.json"
_SEUIL_ECHECS = 10

_DMARC = re.compile(r"\bdmarc=(\w+)", re.IGNORECASE)


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
    if not dmarc or any(d.lower() != "pass" for d in dmarc):
        return "DMARC non validé par Proton"
    return ""


# Commandes admises par courriel : celles de derisk-deleg, sauf confirmations,
# resynchronisation et autorisation Google (Telegram seulement). Les commandes
# natives d'OpenClaw (/model, /reset…) sont refusées.
_PAR_COURRIEL = {"/wikistatus", "/tags", "/relire", "/q", "/r", "/c", "/ingest", "/kbupdate",
                 "/verifie", "/supprimer", "/source", "/drive", "/inbox", "/chercher", "/lire",
                 "/repondre", "/ecrire", "/cron", "/consigne", "/agenda"}


def commandes(objet: str, texte: str) -> list[str]:
    """L'objet s'il commence par « / », puis les lignes « / » en tête du corps, jusqu'à la
    première autre ligne : un message transféré ou cité, qui vient après, n'est jamais lu.
    Une /consigne se poursuit sur les lignes suivantes jusqu'à une ligne vide."""
    out = [objet.strip()] if objet.strip().startswith("/") else []
    suite = False  # une /consigne continue sur les lignes suivantes, jusqu'à une ligne vide
    for ligne in texte.splitlines():
        ligne = ligne.strip()
        if not ligne:
            if suite:
                break
            continue
        if suite and not ligne.startswith(("/", ">")):
            out[-1] += " " + ligne
            continue
        if not ligne.startswith("/"):
            break
        out.append(ligne)
        suite = ligne.split()[0].lower() == "/consigne"
    return out


def _lire_etat(chemin: Path) -> tuple[dict, bool]:
    """(état, mise_en_service) : sans journal lisible, rien de ce qui est déjà dans la
    boîte ne doit être exécuté."""
    if not chemin.exists():
        return {"traites": [], "echecs": 0}, True
    try:
        return json.loads(chemin.read_text(encoding="utf-8")), False
    except (OSError, ValueError):
        chemin.replace(chemin.with_suffix(".illisible"))
        return {"traites": [], "echecs": 0}, True


def _ecrire_etat(chemin: Path, etat: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_suffix(".tmp")
    tmp.write_text(json.dumps(etat), encoding="utf-8")
    tmp.replace(chemin)


def _prevenir(notifier, texte: str) -> bool:
    """Un avis qui échoue (courriel) ne doit pas arrêter la relève."""
    try:
        notifier(texte)
        return True
    except Exception as e:
        print(f"avis non envoyé ({e}) : {texte}", file=sys.stderr)
        return False


def _ranger(conn, uid: str, dossier: str) -> None:
    typ, _ = conn.uid("move", uid, dossier)
    if typ != "OK":
        print(f"message {uid} non rangé dans {dossier}", file=sys.stderr)


def relever(cfg, env, executer, notifier, etat_chemin: Path = ETAT_DEFAUT, connecter=_connecter_imap,
            repondre=None) -> str:
    if not cfg.releve:
        raise ErreurConfig("aucune boîte à relever ([releve] boite = …)")
    compte = cfg.comptes[cfg.releve]
    etat, mise_en_service = _lire_etat(etat_chemin)
    b = Boite(compte, env, connecter)
    try:
        conn = b._ouvrir(lecture_seule=False)
    except ErreurBoite as e:
        etat["echecs"] = etat.get("echecs", 0) + 1
        if etat["echecs"] >= _SEUIL_ECHECS and not etat.get("averti"):
            etat["averti"] = _prevenir(notifier, f"Relève de {compte.adresse} en échec depuis "
                                                 f"{etat['echecs']} essais : {e}")
        _ecrire_etat(etat_chemin, etat)
        raise
    etat["echecs"], etat["averti"] = 0, False
    bilan: list[str] = []
    presents = 0
    try:
        for dossier in (TRAITE, REFUSE):
            conn.create(dossier)  # « NO » si le dossier existe : sans effet
        typ, data = conn.uid("search", None, "ALL")
        for u in (data[0] or b"").split() if typ == "OK" else []:
            uid = u.decode()
            brut = b._fetch(conn, int(u), "(RFC822)")  # sans PEEK : marque le message lu
            m = email.message_from_bytes(brut, policy=email.policy.default)
            mid = str(m.get("Message-ID", "")).strip()
            exp = parseaddr(str(m.get("From", "")))[1][:100]
            if mise_en_service or (mid and mid in etat["traites"]):
                if mise_en_service:
                    presents += 1
                    if mid:
                        etat["traites"].append(mid)
                _ranger(conn, uid, TRAITE)
                continue
            if mid:
                etat["traites"].append(mid)
                _ecrire_etat(etat_chemin, etat)  # avant l'exécution : jamais deux fois
            motif = authentifier(m, cfg.adresses_proprietaire) or ("" if mid else "message sans Message-ID")
            if motif:
                _ranger(conn, uid, REFUSE)
                _prevenir(notifier, f"Courriel refusé sur {compte.adresse} : de {exp}, {motif}. "
                                    "Rien n'a été exécuté.")
                bilan.append(f"refusé : {exp} ({motif})")
                continue
            a = analyser(brut)
            cmds = commandes(a["objet"], a["texte"])
            reponses: list[str] = []
            if not cmds:
                _prevenir(notifier, f"Courriel de {exp} reçu sur {compte.adresse} sans commande : "
                                    "écrivez une ligne qui commence par /, par exemple /wikistatus.")
            for c in cmds:
                if c.split()[0].lower().rstrip("?!") not in _PAR_COURRIEL:
                    _prevenir(notifier, f"{c.split()[0]} (courriel de {exp}) non admis par courriel : "
                                        "à taper sur Session ou Telegram.")
                    continue
                try:
                    reponse = executer(c)
                except Exception as e:  # les commandes suivantes passent quand même
                    _prevenir(notifier, f"Commande {c} (courriel de {exp}) en échec : {e}")
                    continue
                if reponse:
                    reponses.append(f"{c}\n\n{reponse}")
            if reponses and repondre:
                objet = a["objet"].strip()
                objet = objet if objet.lower().startswith("re:") else f"Re: {objet}"
                try:
                    repondre(exp, objet, "\n\n".join(reponses), mid)
                except Exception as e:  # le message reste traité : jamais réexécuté
                    print(f"réponse à {exp} non envoyée ({e})", file=sys.stderr)
            _ranger(conn, uid, TRAITE)
            bilan.append(f"{exp} : {len(cmds)} commande(s)")
        if mise_en_service and presents:
            _prevenir(notifier, f"Relève de {compte.adresse} en service : {presents} message(s) déjà "
                                "présent(s) rangé(s) dans Traite sans être exécuté(s).")
    finally:
        _ecrire_etat(etat_chemin, etat)
        try:
            conn.logout()
        except Exception:
            pass
    return "\n".join(bilan) or "Rien à relever."


def executer_openclaw(env: Mapping[str, str]):
    """Commande : agent main (crochets de derisk-deleg). /consigne : agent consignes.
    Renvoie le texte de la réponse, que la relève renvoie par courriel."""
    def executer(commande: str) -> str:
        exe = env.get("OPENCLAW_BIN") or str(Path.home() / ".local" / "bin" / "openclaw")
        mot, reste = (commande.split(None, 1) + [""])[:2]
        if mot.lower().rstrip("?!") == "/consigne":
            if not reste.strip():
                raise ValueError("usage : /consigne <texte>")
            # Session neuve à chaque consigne : pas d'historique qui biaise le modèle.
            cible = ["--agent", "consignes", "--session-key",
                     f"agent:consignes:courrier-{uuid.uuid4().hex[:8]}", "--message", reste.strip()]
        else:
            cible = ["--agent", "main", "--session-key", "agent:main:courrier", "--message", commande]
        p = subprocess.run([exe, "agent", *cible, "--json"],
                           capture_output=True, text=True, timeout=600, check=True)
        res = (json.loads(p.stdout or "{}").get("result") or {}).get("payloads") or []
        textes = [str(x.get("text") or "").strip() for x in res]
        return "\n\n".join(t for t in textes if t and t != "NO_REPLY")
    return executer


def repondeur_courriel(cfg, env: Mapping[str, str], smtp=_smtp):
    """Réponse à une commande reçue par courriel : de la boîte d'envoi par défaut,
    seulement vers une adresse du propriétaire."""
    def repondre(dest: str, objet: str, corps: str, en_reponse_a: str) -> None:
        if dest.lower() not in cfg.adresses_proprietaire:
            raise ErreurBoite(f"réponse refusée : {dest} n'est pas une de vos adresses")
        compte = cfg.comptes[cfg.envoi_defaut]
        envoyer(construire(compte, dest, objet, corps, en_reponse_a or None), compte, env, smtp)
    return repondre
