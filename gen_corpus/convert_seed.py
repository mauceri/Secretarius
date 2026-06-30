#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convertit corpus-intentions-seed.md en seed.json."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

REQUIRES_ARGS = {"wiki_capture", "wiki_query", "source_read", "gog_mail", "gog_calendar", "gog_drive"}

# 9 exemples manuels couvrant les 3 directives de /c (3 par directive)
MANUAL_DIRECTIVES = [
    {"text": "/c @simple #notion ma note sur la productivité",
     "intention": "wiki_capture", "registre": "télégraphique", "variante": "avec_directive_simple",
     "action": {"command": "/c", "args": "@simple #notion ma note sur la productivité"}},
    {"text": "capture @simple cette note rapide #idée",
     "intention": "wiki_capture", "registre": "familier", "variante": "avec_directive_simple",
     "action": {"command": "/c", "args": "@simple #idée cette note rapide"}},
    {"text": "sauvegarde directement dans sources @simple : réflexions sur le RAG",
     "intention": "wiki_capture", "registre": "poli", "variante": "avec_directive_simple",
     "action": {"command": "/c", "args": "@simple réflexions sur le RAG"}},
    {"text": "/c file:/home/user/notes.md #ia",
     "intention": "wiki_capture", "registre": "télégraphique", "variante": "avec_fichier",
     "action": {"command": "/c", "args": "file:/home/user/notes.md #ia"}},
    {"text": "capture le contenu de ce fichier : file:/home/user/rapport.md",
     "intention": "wiki_capture", "registre": "formel", "variante": "avec_fichier",
     "action": {"command": "/c", "args": "file:/home/user/rapport.md"}},
    {"text": "ajoute ce fichier à mon wiki file:/home/user/doc.txt #documentation",
     "intention": "wiki_capture", "registre": "poli", "variante": "avec_fichier",
     "action": {"command": "/c", "args": "file:/home/user/doc.txt #documentation"}},
    {"text": "/c ref:bm25-intro note complémentaire sur BM25",
     "intention": "wiki_capture", "registre": "télégraphique", "variante": "avec_ref",
     "action": {"command": "/c", "args": "ref:bm25-intro note complémentaire sur BM25"}},
    {"text": "capture avec référence à l'article zettelkasten ref:zettelkasten-intro",
     "intention": "wiki_capture", "registre": "poli", "variante": "avec_ref",
     "action": {"command": "/c", "args": "ref:zettelkasten-intro"}},
    {"text": "note liée ref:colbert-paper : les embeddings multi-vecteurs sont plus précis",
     "intention": "wiki_capture", "registre": "abrégé", "variante": "avec_ref",
     "action": {"command": "/c", "args": "ref:colbert-paper les embeddings multi-vecteurs sont plus précis"}},
]


def extract_args(text: str, intention: str) -> str:
    if intention not in REQUIRES_ARGS:
        return ""
    slash_match = re.match(r"^/\w[\w-]*\s*(.*)", text)
    raw = slash_match.group(1).strip() if slash_match else text
    if intention == "wiki_capture":
        directives = re.findall(r"@\w+", raw)
        refs = re.findall(r"\bref:\S+", raw)
        files = re.findall(r"\bfile:\S+", raw)
        urls = re.findall(r"https?://\S+", raw)
        tags = re.findall(r"#\w+", raw)
        parts = directives + refs + files + urls + tags
        return " ".join(parts).strip()
    if intention == "source_read":
        urls = re.findall(r"https?://\S+", raw)
        return urls[0] if urls else raw
    if intention == "wiki_query":
        cleaned = re.sub(
            r"^(que dit le wiki sur|cherche dans ma base\s*[:»]?|interroge le wiki sur|"
            r"d.après mes notes[,\s]+|que sais-je déjà sur|résume ce que ma base dit sur|"
            r"retrouve mes notes sur|d.après le wiki|qu.ai-je sauvegardé à propos de|"
            r"qu.est-ce que j.ai noté sur|dans mon wiki|dans mes documents|"
            r"que contient mon wiki au sujet de|résume mes notes sur|liste mes sources sur|"
            r"quelles pages parlent de|cherche\s+|résume\s+|fais une synthèse de mes sources sur)\s*",
            "", raw, flags=re.IGNORECASE
        ).strip().rstrip("?").strip()
        return cleaned or raw
    return raw


def infer_variante(text: str, intention: str) -> str:
    if intention == "wiki_capture":
        if "@simple" in text:
            return "avec_directive_simple"
        if "ref:" in text:
            return "avec_ref"
        if "file:" in text:
            return "avec_fichier"
        if "#" in text and "http" in text:
            return "url_avec_tags"
        if "http" in text:
            return "url_seule"
        return "note_sans_url"
    if intention == "wiki_query":
        return "question_longue" if len(text) > 60 else "question_courte"
    if intention == "source_read":
        m = re.search(r"https?://\S+", text)
        return "url_avec_consigne" if m and text[m.end():].strip() else "url_seule"
    if intention == "gog_mail":
        if any(w in text.lower() for w in ["envoie", "écris", "rédige", "réponds", "transfère"]):
            return "envoi"
        if any(w in text.lower() for w in ["lis", "relève", "reçu", "mails", "boîte", "résume", "liste"]):
            return "lecture"
        return "recherche"
    if intention == "gog_calendar":
        if any(w in text.lower() for w in ["crée", "ajoute", "planifie", "bloque", "réserve", "invite"]):
            return "creation"
        if any(w in text.lower() for w in ["annule", "déplace", "supprime", "décale"]):
            return "suppression"
        return "lecture"
    if intention == "gog_drive":
        if any(w in text.lower() for w in ["retrouve", "cherche", "où est", "ouvre", "trouve"]):
            return "recherche"
        if any(w in text.lower() for w in ["liste", "quels fichiers", "contenu"]):
            return "liste"
        return "partage"
    return "sans_args"


def infer_registre(text: str) -> str:
    if text.startswith("/"):
        return "télégraphique"
    if any(w in text.lower() for w in ["veuillez", "pourriez", "auriez-vous", "je vous"]):
        return "formel"
    if any(w in text.lower() for w in ["stp", "sil te plaît", "t'", "t'as"]):
        return "familier"
    if len(text.split()) <= 4:
        return "abrégé"
    return "poli"


def parse_seed(seed_md_path: str, intentions_data: list[dict]) -> list[dict]:
    intentions_map = {item["intention"]: item for item in intentions_data}
    text = Path(seed_md_path).read_text(encoding="utf-8")
    entries: list[dict] = []
    section_re = re.compile(
        r"##\s+\d+\.\s+`(\w+)`[^\n]*\n(.*?)(?=\n##\s+\d+\.|\Z)", re.DOTALL
    )
    for m in section_re.finditer(text):
        intention = m.group(1)
        if intention not in intentions_map:
            continue
        command = intentions_map[intention]["command"]
        for line in m.group(2).splitlines():
            item = re.match(r"^\d+\.\s+(.*)", line.strip())
            if not item:
                continue
            msg = item.group(1).strip()
            msg = msg.replace("`", "")
            if not msg:
                continue
            entries.append({
                "text": msg,
                "intention": intention,
                "registre": infer_registre(msg),
                "variante": infer_variante(msg, intention),
                "action": {"command": command, "args": extract_args(msg, intention)},
            })
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-md", default="../docs/architecture/corpus-intentions-seed.md")
    parser.add_argument("--intentions", default="intentions.json")
    parser.add_argument("--output", default="seed.json")
    a = parser.parse_args()
    intentions_data = json.loads(Path(a.intentions).read_text(encoding="utf-8"))
    entries = parse_seed(a.seed_md, intentions_data)
    entries.extend(MANUAL_DIRECTIVES)
    Path(a.output).write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(entries)} entrées écrites dans {a.output}")


if __name__ == "__main__":
    main()
