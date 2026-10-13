"""Jetons des agents de la porte Grok (spec 2026-10-13).

Une ligne « préfixe:empreinte SHA-256 » par agent ; le jeton en clair n'est
jamais stocké. Le fichier est relu à chaque requête : un retrait prend effet
immédiatement, sans redémarrer le serveur.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import secrets
import sys
from pathlib import Path

PREFIXE_RE = re.compile(r"^[a-z][a-z0-9-]{1,30}$")
FICHIER_DEFAUT = Path.home() / ".config" / "porte-grok" / "agents"


def empreinte(jeton: str) -> str:
    return hashlib.sha256(jeton.encode("utf-8")).hexdigest()


def _lire(fichier: Path) -> list[tuple[str, str]]:
    if not fichier.exists():
        return []
    lignes = []
    for ligne in fichier.read_text(encoding="utf-8").splitlines():
        prefixe, sep, emp = ligne.strip().partition(":")
        if sep and prefixe and emp:
            lignes.append((prefixe, emp))
    return lignes


def _ecrire(fichier: Path, lignes: list[tuple[str, str]]) -> None:
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.parent.chmod(0o700)
    fd = os.open(fichier, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("".join(f"{p}:{e}\n" for p, e in lignes))
    fichier.chmod(0o600)


def agent_pour(fichier: Path, en_tete: str | None) -> str | None:
    """Préfixe de l'agent dont le jeton est présenté, ou None."""
    if not en_tete or not en_tete.startswith("Bearer "):
        return None
    jeton = en_tete[len("Bearer "):].strip()
    if not jeton:
        return None
    emp = empreinte(jeton)
    trouve = None
    for prefixe, attendu in _lire(fichier):
        if hmac.compare_digest(emp, attendu):
            trouve = prefixe
    return trouve


def ajouter(fichier: Path, prefixe: str) -> str:
    if not PREFIXE_RE.match(prefixe):
        raise ValueError(f"préfixe invalide : {prefixe!r} (attendu : {PREFIXE_RE.pattern})")
    lignes = _lire(fichier)
    if any(p == prefixe for p, _ in lignes):
        raise ValueError(f"agent déjà présent : {prefixe}")
    jeton = secrets.token_urlsafe(32)
    _ecrire(fichier, lignes + [(prefixe, empreinte(jeton))])
    return jeton


def retirer(fichier: Path, prefixe: str) -> bool:
    lignes = _lire(fichier)
    restantes = [(p, e) for p, e in lignes if p != prefixe]
    if len(restantes) == len(lignes):
        return False
    _ecrire(fichier, restantes)
    return True


def lister(fichier: Path) -> list[str]:
    return [p for p, _ in _lire(fichier)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="porte-jeton", description="Jetons des agents de la porte Grok")
    parser.add_argument("--fichier", default=str(FICHIER_DEFAUT))
    sous = parser.add_subparsers(dest="action", required=True)
    for action in ("ajouter", "retirer"):
        sous.add_parser(action).add_argument("prefixe")
    sous.add_parser("lister")
    args = parser.parse_args(argv)
    fichier = Path(args.fichier)
    if args.action == "ajouter":
        try:
            jeton = ajouter(fichier, args.prefixe)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        print(f"Jeton de l'agent « {args.prefixe} » (affiché une seule fois) :")
        print(jeton)
        return 0
    if args.action == "retirer":
        if retirer(fichier, args.prefixe):
            print(f"Agent « {args.prefixe} » retiré.")
            return 0
        print(f"Agent inconnu : {args.prefixe}", file=sys.stderr)
        return 1
    for prefixe in lister(fichier):
        print(prefixe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
