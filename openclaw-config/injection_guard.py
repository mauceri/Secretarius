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
from flask import Flask, request, jsonify

# ─── Constantes ──────────────────────────────────────────────────────────────

MAX_CONTENT_LEN = 15_000
MAX_RAW_LEN = 200_000
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
        # Un tag déjà décomposé (descendant d'un ancêtre invisible retiré
        # plus tôt dans la boucle) a attrs=None — on l'ignore.
        if tag.attrs is None:
            continue
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


# ─── DeBERTa ─────────────────────────────────────────────────────────────────

_classifier = None
_deberta_available = False


def _load_deberta() -> None:
    global _classifier, _deberta_available
    try:
        from transformers import pipeline as hf_pipeline
        _classifier = hf_pipeline(
            "text-classification",
            model="protectai/deberta-v3-base-prompt-injection-v2",
            device=-1,
        )
        _deberta_available = True
        logging.info("DeBERTa chargé.")
    except Exception as exc:
        logging.error("Chargement DeBERTa échoué (%s) — mode regex-only.", exc)


def _deberta_risk(text: str) -> str:
    """Retourne 'blocked', 'medium' ou 'low' selon le score DeBERTa."""
    if not _deberta_available or _classifier is None:
        return "medium"
    result = _classifier(text[:DEBERTA_MAX_TOKENS])
    label = result[0]['label'].upper()
    score = result[0]['score']
    if label == 'INJECTION':
        if score >= DEBERTA_BLOCK_THRESHOLD:
            return "blocked"
        if score >= DEBERTA_MEDIUM_THRESHOLD:
            return "medium"
        return "low"
    # label == LEGIT
    return "low" if score >= DEBERTA_BLOCK_THRESHOLD else "medium"


# ─── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route('/check', methods=['POST'])
def check():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    content = str(data.get('content', '') or '')
    content_type = data.get('type', 'text')

    # Garde-fou : borne le brut avant parsing (anti-payload pathologique),
    # bien au-dessus du budget de texte nettoyé.
    if len(content) > MAX_RAW_LEN:
        content = content[:MAX_RAW_LEN]

    if content_type == 'html':
        clean_text = clean_html(content)
    else:
        clean_text = content

    # Troncature APRÈS nettoyage : le budget s'applique au texte utile,
    # pas au boilerplate <head>.
    truncated = len(clean_text) > MAX_CONTENT_LEN
    if truncated:
        clean_text = clean_text[:MAX_CONTENT_LEN]
    full_content = content[:MAX_CONTENT_LEN]

    risk, patterns = check_regex(clean_text)
    if risk == "blocked":
        return jsonify({"blocked": True, "reason": ", ".join(patterns)})

    if risk == "medium":
        try:
            deberta_risk = _deberta_risk(clean_text)
            if deberta_risk == "blocked":
                return jsonify({"blocked": True, "reason": "DeBERTa: score d'injection élevé"})
            risk = deberta_risk
        except Exception as exc:
            logging.error("DeBERTa inference error: %s", exc)

    return jsonify({
        "blocked": False,
        "risk": risk,
        "clean_text": clean_text,
        "full_content": full_content,
        "truncated": truncated,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "deberta": _deberta_available})


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _load_deberta()
    app.run(host='127.0.0.1', port=8990, debug=False)
