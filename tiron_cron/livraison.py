# tiron_cron/livraison.py
"""Livraison du résultat d'une tâche /cron dans le coffre (dossier Réception),
puis avis par courriel (spec canal Session ; spec courrier, « Livraisons vers Obsidian »). Une note par
livraison : jamais d'ajout dans une note existante (conflits de synchronisation)."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

_INTERDITS = r'[\\/:*?"<>|#^\[\]]'


def _titre(evt: dict) -> str:
    job = evt.get("job") or {}
    brut = str(job.get("displayName") or job.get("name") or evt.get("jobId") or "livraison")
    brut = " ".join(brut.split())  # un bloc cron! sur plusieurs lignes : nom de fichier sur une ligne
    brut = re.sub(r"^(consigne|commande|rappel) : ", "", brut)
    return re.sub(_INTERDITS, "", brut)[:50].strip() or "livraison"


def livrer(evt: dict, now: datetime, env: Mapping[str, str],
           notifier: Callable[[str], None]) -> Path | None:
    statut = evt.get("status")
    texte = str(evt.get("summary") or "").strip()
    if statut == "skipped" or (statut != "error" and texte in ("", "NO_REPLY")):
        return None
    titre = _titre(evt)
    if statut == "error":
        texte = f"Échec : {evt.get('error') or 'erreur inconnue'}"
    # Texte influençable (modèle, bientôt courriels) : aucun bloc exécutable par le plugin.
    texte = texte.replace("```wiki", "```text")
    d = Path(env.get("TIRON_RECEPTION_DIR") or Path.home() / "Documents" / "Arbath" / "Réception")
    d.mkdir(parents=True, exist_ok=True)
    base = f"{now:%Y-%m-%d %H%M} {titre}"
    p, n = d / f"{base}.md", 2
    while p.exists():
        p, n = d / f"{base} {n}.md", n + 1
    p.write_text(
        f"---\ntags: [livraison]\norigine: cron\ntache: {json.dumps(titre, ensure_ascii=False)}\ndate: {now:%Y-%m-%dT%H:%M}\n"
        f"statut: {'erreur' if statut == 'error' else 'ok'}\n---\n\n{texte}\n", encoding="utf-8")
    avis = (f"Échec de la tâche « {titre} » : détail dans Obsidian (Réception)" if statut == "error"
            else f"Nouvelle livraison dans Obsidian : {titre} (Réception)")
    try:
        notifier(avis)
    except Exception:  # la note est écrite : un avis manqué ne doit pas faire rejouer la livraison
        pass
    return p
