"""Avis spontanés de Secretarius par courriel (spec canal Session, § 2) : de la
boîte d'envoi par défaut (tiron@) à l'adresse d'avis du propriétaire."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from tiron_mail.config import CHEMIN_DEFAUT, ErreurConfig, charger
from tiron_mail.envoi import _smtp, construire, envoyer


def notifier_courriel(env: Mapping[str, str], smtp=_smtp) -> Callable[[str], None]:
    def envoyer_avis(texte: str) -> None:
        cfg = charger(Path(env.get("TIRON_MAIL_CONFIG") or CHEMIN_DEFAUT))
        if not cfg.avis:
            raise ErreurConfig("adresse d'avis non configurée ([envoi] avis = …)")
        compte = cfg.comptes[cfg.envoi_defaut]
        objet = "Secretarius : " + (texte.strip().splitlines() or [""])[0][:60]
        envoyer(construire(compte, cfg.avis, objet, texte.strip()), compte, env, smtp)
    return envoyer_avis
