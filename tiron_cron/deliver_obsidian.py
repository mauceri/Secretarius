# tiron_cron/deliver_obsidian.py
"""Livraison d'un rappel sur le canal obsidian : une note dans le coffre,
synchronisée par obsidian-headless. Pas de notification active en V1."""
from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


def deliver(text: str, now: datetime, env: Mapping[str, str]) -> Path:
    d = Path(env.get("TIRON_RAPPELS_DIR") or Path.home() / "Documents" / "Arbath" / "Rappels")
    d.mkdir(parents=True, exist_ok=True)
    titre = re.sub(r'[\\/:*?"<>|#^\[\]]', "", text)[:50].strip() or "rappel"
    base = f"{now:%Y-%m-%d %H%M} {titre}"
    p, n = d / f"{base}.md", 2
    while p.exists():
        p, n = d / f"{base} {n}.md", n + 1
    p.write_text(f"---\ntags: [rappel]\ndate: {now:%Y-%m-%dT%H:%M}\n---\n\n{text}\n", encoding="utf-8")
    return p
