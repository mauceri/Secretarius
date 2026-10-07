"""Lecture d'un message : en-têtes décodés, corps en texte seul (HTML converti,
scripts et styles retirés), pièces jointes listées mais jamais ouvertes.
Le corps est du contenu non fiable : il n'est jamais transmis comme instruction."""
from __future__ import annotations

import email
import email.policy
import re
from email.utils import parseaddr, parsedate_to_datetime
from html.parser import HTMLParser


class _Texte(HTMLParser):
    _IGNORES = {"script", "style", "head"}
    _BLOCS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.morceaux: list[str] = []
        self._ignore = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._IGNORES:
            self._ignore += 1
        elif tag in self._BLOCS:
            self.morceaux.append("\n")

    def handle_endtag(self, tag):
        if tag in self._IGNORES and self._ignore:
            self._ignore -= 1
        elif tag in self._BLOCS:
            self.morceaux.append("\n")

    def handle_data(self, data):
        if not self._ignore:
            self.morceaux.append(data)


def html_en_texte(html: str) -> str:
    p = _Texte()
    p.feed(html)
    lignes = [re.sub(r"[ \t\r\f\v]+", " ", l).strip() for l in "".join(p.morceaux).split("\n")]
    return "\n".join(l for l in lignes if l)


def analyser(brut: bytes) -> dict:
    m = email.message_from_bytes(brut, policy=email.policy.default)
    corps = m.get_body(preferencelist=("plain", "html"))
    texte = ""
    if corps is not None:
        contenu = corps.get_content()
        texte = html_en_texte(contenu) if corps.get_content_subtype() == "html" else contenu.strip()
    return {
        "expediteur": str(m.get("From", "")),
        "destinataires": str(m.get("To", "")),
        "objet": str(m.get("Subject", "")),
        "date": str(m.get("Date", "")),
        "message_id": str(m.get("Message-ID", "")),
        "texte": texte,
        "pieces_jointes": [p.get_filename() or "(sans nom)" for p in m.iter_attachments()],
        "repondre_a": parseaddr(str(m.get("Reply-To") or m.get("From", "")))[1],
    }


def resume(m: dict, ident: str) -> str:
    try:
        quand = parsedate_to_datetime(m["date"]).strftime("%d/%m %H:%M")
    except (TypeError, ValueError):
        quand = "?"
    return f"{ident}  {quand}  {m['expediteur']}  {m['objet']}"
