# tiron_cron/fallback.py
"""Repli quand l'analyseur déterministe échoue : qwen3:8b LOCAL (Ollama), jamais
un fournisseur distant. `think: false` est indispensable (sinon Qwen3 épuise
son budget en raisonnement caché, cf. mémoire du 2026-09-30). La sortie est
revalidée par parse_command : le modèle ne peut rien créer d'invalide."""
from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from datetime import datetime

from tiron_cron.syntax import Job, SyntaxeInvalide, parse_command

_JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
SYSTEM_PROMPT = """Tu convertis une demande de planification en UNE ligne de la forme :
add <quand> [<canal>] <type> <contenu>
- <quand> : expression cron 5 champs entre guillemets doubles ("0 8 * * 1"), OU date-heure locale ISO (2026-10-02T09:00), OU durée (20m, 2h, 1d).
- <canal> (facultatif, seulement si la demande le précise) : telegram, whatsapp, sms ou obsidian.
- <type> : rappel (un texte à rappeler), commande (une commande Tiron commençant par /), consigne (une tâche confiée à l'assistant).
- <contenu> : le texte du rappel, la commande, ou la consigne.
Maintenant : {now} ({jour}). Réponds par la ligne seule, sans commentaire.
Si la demande n'est pas une planification, réponds : AUCUNE"""


def _post_json(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def normalize_with_model(text: str, now: datetime, env: Mapping[str, str], http=_post_json) -> Job | None:
    url = env.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
    body = {
        "model": env.get("TIRON_CRON_MODEL", "qwen3:8b"),
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                now=now.strftime("%Y-%m-%dT%H:%M"), jour=_JOURS[now.weekday()])},
            {"role": "user", "content": text},
        ],
    }
    try:
        content = (http(url, body, 120).get("message") or {}).get("content", "")
    except Exception:
        return None
    for line in content.splitlines():
        line = line.strip().strip("`").strip()
        if line.lower().startswith("/cron "):
            line = line[6:].strip()
        if line.lower().startswith("add "):
            try:
                r = parse_command(line)
            except SyntaxeInvalide:
                return None
            return r if isinstance(r, Job) else None
    return None
