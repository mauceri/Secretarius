"""Confinement des fichiers du coffre pour la porte Grok (spec 2026-10-13).

Chemins relatifs au coffre, résolus (liens symboliques suivis) puis vérifiés :
jamais hors du coffre, jamais dans Wiki_LM/ ni dans un dossier caché
(.obsidian/…), .md seulement. L'écriture exige le préfixe de l'agent.
"""
from __future__ import annotations

import re
from pathlib import Path

TAILLE_MAX = 1_000_000
ZONES_INTERDITES = {"Wiki_LM"}


# Même motif que le plugin wikilm-capture (run-commands.ts, BLOCK_RE) : un
# bloc est repéré n'importe où, pas seulement en début de ligne.
_BLOC_WIKI = re.compile(r"```wiki\r?\n([\s\S]*?)\r?\n```")
_CASE_VIDE = re.compile(r"^\s*[-*]\s+\[ \]")


class Refus(Exception):
    """Demande refusée ; le message est renvoyé tel quel à l'agent."""


def resoudre(racine: Path, chemin: str) -> Path:
    if not chemin or Path(chemin).is_absolute():
        raise Refus("chemin relatif au coffre attendu")
    base = racine.resolve()
    cible = (base / chemin).resolve()
    if cible == base or not cible.is_relative_to(base):
        raise Refus("chemin hors du coffre")
    parties = cible.relative_to(base).parts
    if parties[0] in ZONES_INTERDITES or any(p.startswith(".") for p in parties):
        raise Refus("zone interdite : Wiki_LM/ et dossiers cachés")
    return cible


def _exiger_md(cible: Path) -> None:
    if cible.suffix != ".md":
        raise Refus("seuls les fichiers .md sont permis")


def lister(racine: Path, dossier: str = "") -> list[str]:
    base = racine.resolve()
    depart = resoudre(racine, dossier) if dossier else base
    if not depart.is_dir():
        raise Refus(f"dossier introuvable : {dossier}")
    trouves = []
    for f in depart.rglob("*.md"):
        rel = f.relative_to(base).as_posix()
        try:
            if resoudre(racine, rel).is_file():
                trouves.append(rel)
        except Refus:
            continue
    return sorted(trouves)


def lire(racine: Path, chemin: str) -> str:
    cible = resoudre(racine, chemin)
    _exiger_md(cible)
    if not cible.is_file():
        raise Refus(f"fichier introuvable : {chemin}")
    return cible.read_text(encoding="utf-8")


def _verifier_blocs_wiki(contenu: str) -> None:
    """Une note d'agent ne porte que des /c derrière une case vide : sinon la
    validation de l'utilisateur dans Obsidian exécuterait le reste (revue
    finale du 2026-10-13 : /supprimer!, /mail!… venus d'une page web)."""
    for m in _BLOC_WIKI.finditer(contenu):
        lignes = [l for l in contenu[:m.start()].split("\n") if l.strip()]
        if not lignes or not _CASE_VIDE.match(lignes[-1]):
            raise Refus("chaque bloc wiki doit être précédé d'une case vide « - [ ] »")
        if m.group(1).split("\n")[0].strip() != "/c":
            raise Refus("seule la commande /c est permise dans un bloc wiki")


def ecrire(racine: Path, prefixe: str, chemin: str, contenu: str) -> str:
    cible = resoudre(racine, chemin)
    _exiger_md(cible)
    if not cible.name.startswith(f"{prefixe}-"):
        raise Refus(f"le nom du fichier doit commencer par {prefixe}-")
    if len(contenu.encode("utf-8")) > TAILLE_MAX:
        raise Refus(f"contenu trop long (maximum {TAILLE_MAX} octets)")
    _verifier_blocs_wiki(contenu)
    cible.parent.mkdir(parents=True, exist_ok=True)
    cible.write_text(contenu, encoding="utf-8")
    return cible.relative_to(racine.resolve()).as_posix()
