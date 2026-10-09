# tiron_cron/livraison.py
"""Livraison du résultat d'une tâche /cron dans le coffre (dossier Réception),
puis avis Telegram (spec courrier, « Livraisons vers Obsidian »). Une note par
livraison : jamais d'ajout dans une note existante (conflits de synchronisation)."""
from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

_INTERDITS = r'[\\/:*?"<>|#^\[\]]'


def _titre(evt: dict) -> str:
    job = evt.get("job") or {}
    brut = str(job.get("displayName") or job.get("name") or evt.get("jobId") or "livraison")
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
    d = Path(env.get("TIRON_RECEPTION_DIR") or Path.home() / "Documents" / "Arbath" / "Réception")
    d.mkdir(parents=True, exist_ok=True)
    base = f"{now:%Y-%m-%d %H%M} {titre}"
    p, n = d / f"{base}.md", 2
    while p.exists():
        p, n = d / f"{base} {n}.md", n + 1
    p.write_text(
        f"---\ntags: [livraison]\norigine: cron\ntache: {titre}\ndate: {now:%Y-%m-%dT%H:%M}\n"
        f"statut: {'erreur' if statut == 'error' else 'ok'}\n---\n\n{texte}\n", encoding="utf-8")
    avis = (f"Échec de la tâche « {titre} » : détail dans Obsidian (Réception)" if statut == "error"
            else f"Nouvelle livraison dans Obsidian : {titre} (Réception)")
    try:
        notifier(avis)
    except Exception:  # la note est écrite : un avis manqué ne doit pas faire rejouer la livraison
        pass
    return p


def notifier_telegram(env: Mapping[str, str]) -> Callable[[str], None]:
    def envoyer(texte: str) -> None:
        dest = env.get("TIRON_TELEGRAM_CHAT_ID")
        if not dest:
            raise OSError("TIRON_TELEGRAM_CHAT_ID non configuré")
        # Chemin complet : le PATH des services systemd n'inclut pas ~/.local/bin,
        # où se trouve l'enveloppe openclaw (qui charge gateway.systemd.env).
        exe = env.get("OPENCLAW_BIN") or str(Path.home() / ".local" / "bin" / "openclaw")
        subprocess.run([exe, "message", "send", "--channel", "telegram", "-t", dest, "-m", texte],
                       capture_output=True, text=True, timeout=60, check=True)
    return envoyer
