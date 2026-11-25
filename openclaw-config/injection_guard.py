#!/usr/bin/env python3
"""
injection-guard — Service Flask de détection d'injection de prompt.
Endpoint : POST /check  {type:"html"|"text", content:str}
Regex (seules à bloquer) + Prompt Guard 2 86M (avertit).
"""
import re
import html as html_mod
import logging
from typing import List, Tuple

from bs4 import BeautifulSoup
from flask import Flask, request, jsonify

# ─── Constantes ──────────────────────────────────────────────────────────────

MAX_CONTENT_LEN = 40_000
MAX_RAW_LEN = 200_000
PG2_MODELE = "meta-llama/Llama-Prompt-Guard-2-86M"
PG2_FENETRE = 512
PG2_SEUIL = 0.5

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
    # Motifs anglais ajoutés le 16/09/2026 (revue DSH) : les variantes
    # ci-dessus exigeaient "all"/"instructions" comme complément exact et
    # laissaient passer des formulations anglaises courantes en "low",
    # jamais vues par DeBERTa (angle mort mesuré : voir aussi la consultation
    # systématique du classifieur plus bas, qui ne dépend plus de ces regex).
    re.compile(r'ignore\s+(the\s+)?(previous|prior|above)\s+instructions', re.I),
    re.compile(r'disregard\s+(the\s+)?(above|previous|prior)\b', re.I),
    re.compile(r'(reveal|show|send|leak)\s+(me\s+)?(the\s+)?(api\s*key|password|system\s+prompt|seed)', re.I),
    re.compile(r'without\s+(confirmation|approval)', re.I),
    re.compile(r'(lift|bypass|remove)\s+(your|the)\s+restrictions', re.I),
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


# ─── Classifieur Prompt Guard 2 ──────────────────────────────────────────────
# Remplace DeBERTa (protectai v2) le 2026-11-26 : sur 22 textes sains, DeBERTa
# en signalait 9 à 11 (score ~1,0 sur du français ordinaire), PG2-86M aucun ;
# PG2 ne repère en revanche que 12 injections sur 30 (surtout les formes
# classiques). Il avertit, il ne bloque pas : seules les regex bloquent.

_classifier_available = False
_tokenizer = None
_model = None


def _load_classifier() -> None:
    global _tokenizer, _model, _classifier_available
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _tokenizer = AutoTokenizer.from_pretrained(PG2_MODELE)
        _model = AutoModelForSequenceClassification.from_pretrained(PG2_MODELE).eval()
        _classifier_available = True
        logging.info("Prompt Guard 2 chargé.")
    except Exception as exc:
        logging.error("Chargement de Prompt Guard 2 échoué (%s) — mode regex-only.", exc)


def _score_malveillant(text: str) -> float:
    """Probabilité « malveillant » maximale sur des fenêtres de 512 jetons."""
    import torch
    ids = _tokenizer(text, add_special_tokens=False)["input_ids"]
    pas = PG2_FENETRE - 2  # place pour [CLS] et [SEP]
    meilleur = 0.0
    for i in range(0, max(len(ids), 1), pas):
        fenetre = [_tokenizer.cls_token_id] + ids[i:i + pas] + [_tokenizer.sep_token_id]
        with torch.no_grad():
            logits = _model(input_ids=torch.tensor([fenetre])).logits
        meilleur = max(meilleur, torch.softmax(logits, -1)[0, 1].item())
    return meilleur


def _classifier_risk(text: str) -> str:
    """'medium' si PG2 juge le texte malveillant (ou s'il est absent), sinon 'low'."""
    if not _classifier_available:
        return "medium"
    return "medium" if _score_malveillant(text) >= PG2_SEUIL else "low"


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

    # Le classifieur est consulté systématiquement (medium ET low) depuis le
    # 16/09/2026 : les formulations anglaises passaient en "low" côté regex et
    # n'atteignaient jamais le classifieur (revue DSH). Il ne fait que relever
    # le risque : un "medium" regex reste "medium".
    try:
        if _classifier_risk(clean_text) == "medium":
            risk = "medium"
    except Exception as exc:
        logging.error("Erreur d'inférence du classifieur : %s", exc)

    return jsonify({
        "blocked": False,
        "risk": risk,
        "clean_text": clean_text,
        "full_content": full_content,
        "truncated": truncated,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "classifieur": _classifier_available})


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _load_classifier()
    app.run(host='127.0.0.1', port=8990, debug=False)
