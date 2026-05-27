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


# ─── Patterns regex ───────────────────────────────────────────────────────────

HIGH_RISK_PATTERNS: List[re.Pattern] = [
    re.compile(r'ignore\s+(vos|tes|toutes\s+les)\s+instructions', re.I),
    re.compile(r'(vous\s+[eê]tes\s+maintenant|tu\s+es\s+maintenant)', re.I),
    re.compile(r'\bDAN\b'),
    re.compile(r'\bjailbreak\b', re.I),
    re.compile(r'sans\s+confirmation', re.I),
    re.compile(r'sans\s+approbation', re.I),
    re.compile(r'virement\s+urgent', re.I),
    re.compile(r'affiche\s+(la\s+cl[eé]|le\s+mot\s+de\s+passe|le\s+seed|le\s+prompt\s+syst[eè]me)', re.I),
    re.compile(r'(l[eè]ve|contourne)\s+(vos|tes|les)\s+restrictions', re.I),
    re.compile(r'ignore\s+all\s+previous\s+instructions', re.I),
    re.compile(r'disregard\s+(\w+\s+)?(all|previous|your)\s+instructions', re.I),
    re.compile(r'you\s+are\s+now\s+a[n]?\s+', re.I),
]

MEDIUM_RISK_PATTERNS: List[re.Pattern] = [
    re.compile(r'jeu\s+de\s+r[oô]le', re.I),
    re.compile(r'faites\s+semblant', re.I),
    re.compile(r"fais\s+semblant\s+d'[eê]tre", re.I),
    re.compile(r'comporte-toi\s+comme', re.I),
    re.compile(r'ignore\s+les\s+instructions\s+pr[eé]c[eé]dentes', re.I),
    re.compile(r'pretend\s+(to\s+be|you\s+are)', re.I),
    re.compile(r'\broleplay\b', re.I),
    re.compile(r'\bact\s+as\b', re.I),
]


def check_regex(text: str) -> Tuple[str, List[str]]:
    """
    Retourne (risk_level, patterns_trouvés).
    risk_level : "blocked" | "medium" | "low"
    """
    matched_high = [p.pattern for p in HIGH_RISK_PATTERNS if p.search(text)]
    if matched_high:
        return "blocked", matched_high
    matched_medium = [p.pattern for p in MEDIUM_RISK_PATTERNS if p.search(text)]
    if matched_medium:
        return "medium", matched_medium
    return "low", []
