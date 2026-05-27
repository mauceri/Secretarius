#!/usr/bin/env python3
"""
injection-guard — Service Flask de détection d'injection de prompt.
Endpoint : POST /check  {type:"html"|"text", content:str}
"""
import re
import html as html_mod
import logging
from typing import List, Tuple

from bs4 import BeautifulSoup

# ─── Constantes ──────────────────────────────────────────────────────────────

MAX_CONTENT_LEN = 15_000
DEBERTA_MAX_TOKENS = 512
DEBERTA_BLOCK_THRESHOLD = 0.7
DEBERTA_MEDIUM_THRESHOLD = 0.3

# ─── HTML cleaning ────────────────────────────────────────────────────────────

_INVISIBLE_STYLE = re.compile(
    r'(display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|opacity\s*:\s*0)',
    re.I,
)


def clean_html(content: str) -> str:
    """Supprime scripts, styles, éléments invisibles ; décode les entités HTML ; retourne du texte brut."""
    if not content:
        return ''
    soup = BeautifulSoup(content, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    for tag in soup.find_all(style=True):
        if _INVISIBLE_STYLE.search(tag.get('style', '')):
            tag.decompose()
    text = soup.get_text(separator=' ')
    text = html_mod.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
