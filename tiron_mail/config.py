"""Description des boîtes (sans secret) : ~/.config/tiron_mail/comptes.toml.
Les mots de passe restent dans l'environnement (secret_env), cf. spec 2026-10-07."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

CHEMIN_DEFAUT = Path.home() / ".config" / "tiron_mail" / "comptes.toml"


class ErreurConfig(ValueError):
    pass


@dataclass(frozen=True)
class Compte:
    nom: str
    prefixe: str
    adresse: str
    utilisateur: str
    hote_imap: str
    port_imap: int
    securite_imap: str
    hote_smtp: str
    port_smtp: int
    securite_smtp: str
    secret_env: str


@dataclass(frozen=True)
class Config:
    comptes: dict
    envoi_defaut: str
    adresses_proprietaire: tuple


def charger(chemin: Path = CHEMIN_DEFAUT) -> Config:
    if not Path(chemin).exists():
        raise ErreurConfig(f"configuration des boîtes introuvable : {chemin}")
    d = tomllib.loads(Path(chemin).read_text(encoding="utf-8"))
    comptes = {nom: Compte(nom=nom, **c) for nom, c in d.get("comptes", {}).items()}
    envoi = d.get("envoi", {})
    defaut = envoi.get("defaut", "")
    if defaut not in comptes:
        raise ErreurConfig(f"envoi.defaut « {defaut} » ne désigne aucune boîte configurée")
    return Config(comptes, defaut, tuple(a.lower() for a in envoi.get("adresses_proprietaire", [])))


def compte_de_prefixe(cfg: Config, prefixe: str) -> Compte:
    for c in cfg.comptes.values():
        if c.prefixe == prefixe:
            return c
    raise ErreurConfig(f"aucune boîte configurée pour « {prefixe} »")
