"""Éléments partagés par promptGenGEPA.py (optimisation) et generate_corpus.py
(génération) : un seul exemplaire de la signature, des descriptions, des juges
et de la grille de diversité. Cf. docs/superpowers/plans/2026-10-03-refonte-chaine-dspy.md."""
from __future__ import annotations

import random
import re

import dspy

INTENTION_DESCRIPTIONS = {
    "wiki_capture": "capturer une URL ou une note dans le wiki",
    "wiki_ingest": "lancer l'ingestion des captures en attente du wiki",
    "wiki_status": "consulter l'état de l'ingestion du wiki",
    "wiki_query": "poser une question au wiki ou de culture générale (connaissance, conseil, explication) et obtenir une réponse synthétisée par un LLM",
    "source_read": "lire une page web externe immédiatement, sans la sauvegarder",
    "gog_search": "rechercher des emails Gmail par mot-clé, expéditeur ou période",
    "gog_connect": "autoriser l'accès au compte Google",
    "gog_inbox": "lister les emails récents de la boîte de réception",
    "gog_reply": "préparer un brouillon de réponse à un email, sans l'envoyer",
    "gog_drive": "rechercher des fichiers sur Google Drive",
    "wiki_search": "rechercher par mots-clés dans le wiki, résultats bruts SANS synthèse ni résumé LLM (différent de wiki_query qui synthétise une réponse)",
    "wiki_tags": "lister les tags disponibles dans le wiki",
    "wiki_kb_update": "mettre à jour (rafraîchir, reconstruire) la base de connaissances du wiki — commande sans argument",
    "gog_get": "lire le contenu d'un email précis (par identifiant, ou le dernier/un email récent) — UNIQUEMENT un email, jamais une page wiki, une capture, un document Drive ou l'état du wiki",
    "wiki_delete": "supprimer une page du wiki (par son slug)",
    "wiki_reread": "obtenir la prochaine page du wiki à relire",
    "wiki_verify": "marquer une page du wiki comme vérifiée (par son slug)",
    "cron_schedule": "planifier un rappel, une commande Tiron ou une consigne à une date donnée ou de façon récurrente (chaque lundi, tous les jours, demain, dans 20 minutes), éventuellement sur un canal précis (SMS, WhatsApp, Telegram, Obsidian)",
    "out_of_scope": "demande hors périmètre de Tiron. Tiron sait UNIQUEMENT : gérer le wiki personnel (capturer, interroger, chercher, ingérer, relire, vérifier, supprimer, tags, état), lire/chercher/répondre à des emails reçus, chercher dans Google Drive, lire une page web, planifier un rappel ou une tâche (/cron). Tout le reste est hors périmètre (réserver, acheter, écrire un NOUVEAU mail, conversation, domotique…) ; une question de connaissance générale → /q, donc PAS hors périmètre ; un rappel ou une planification n'est JAMAIS hors périmètre",
}


def build_signature(prompt_text: str):
    class GenerateExample(dspy.Signature):
        __doc__ = prompt_text
        intention: str = dspy.InputField(desc="Intention Tiron à illustrer (ex: wiki_capture)")
        registre: str = dspy.InputField(desc="Registre du message : formel, familier, télégraphique, poli, abrégé")
        variante: str = dspy.InputField(desc="Type de variante (ex: url_avec_tags, question_courte, sans_args…)")
        contraintes: str = dspy.InputField(desc="Contraintes à respecter impérativement (moment, sujet, longueur…)")
        text: str = dspy.OutputField(desc="Message utilisateur réaliste en français")
        command: str = dspy.OutputField(desc="Commande Tiron ou null")
        args: str = dspy.OutputField(desc="Arguments bruts de la commande (chaîne vide si pas d'args)")
    return GenerateExample


class EvalRealisme(dspy.Signature):
    """Ce message ressemble-t-il à une vraie requête tapée par un utilisateur à son
    assistant (brève, naturelle, sans formule de politesse développée) ?"""
    text: str = dspy.InputField(desc="Message utilisateur généré")
    score: int = dspy.OutputField(desc="Entier 1..5")
    raison: str = dspy.OutputField(desc="Une phrase : ce qui le rend peu réaliste, ou « rien »")


class EvalFidelite(dspy.Signature):
    """Ce message correspond-il vraiment à l'intention décrite, et à elle seule
    (pas à une intention voisine ni à une autre commande) ? 1 si sans ambiguïté
    cette intention précise, 0 sinon — y compris si le message est plausible
    pour une AUTRE intention que celle donnée."""
    text: str = dspy.InputField(desc="Message utilisateur généré")
    intention_description: str = dspy.InputField(desc="Description de l'intention que le message doit illustrer")
    score: int = dspy.OutputField(desc="0 ou 1")
    raison: str = dspy.OutputField(desc="Une phrase : pourquoi, ou quelle autre intention il évoque")


_JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
         "septembre", "octobre", "novembre", "décembre"]
_SUJETS = ["santé", "factures", "jardin", "famille", "travail", "voiture", "courses", "sport",
           "administratif", "lecture", "cuisine", "animaux", "maison", "études", "voyage",
           "informatique", "musique", "rendez-vous médical", "anniversaire", "wiki personnel"]
GRILLES: dict[str, dict[str, list[str]]] = {
    "*": {
        "sujet": _SUJETS,
        "forme": ["question", "ordre", "phrase nominale", "constat"],
        "longueur": ["très court (≤ 5 mots)", "court (6 à 12 mots)", "moyen (13 à 20 mots)"],
        "frappe": ["sans faute", "une abréviation ou faute de frappe"],
    },
    "cron_schedule": {
        "moment": [f"chaque {j}" for j in _JOURS] + [f"tous les {j}s" for j in _JOURS]
                  + ["tous les jours", "chaque matin", "chaque soir", "en semaine (lundi à vendredi)",
                     "le week-end", "le 1er de chaque mois", "le 15 de chaque mois",
                     "demain", "après-demain", "ce soir", "dans N minutes", "dans N heures",
                     "dans N jours"] + [f"le N {m}" for m in _MOIS],
        "heure": [f"{h}h{m}" for h in range(6, 23) for m in ("", "15", "30", "45")] + ["sans heure"],
        "canal": ["non précisé"] * 4 + ["SMS", "WhatsApp", "Telegram", "Obsidian"],
        "type de tâche": ["rappel"] * 3 + ["commande Tiron (/wikistatus, /tags, /relire, /ingest…)",
                                            "consigne à l'assistant (résumer, préparer, vérifier…)"],
        "sujet": _SUJETS,
        "longueur": ["très court (≤ 5 mots)", "court (6 à 12 mots)", "moyen (13 à 20 mots)"],
    },
}


# Intentions dont la commande ne prend pas d'objet : leur imposer un sujet fait
# inventer un argument absurde (« maj la kb pr mon rdv médical », 2026-10-05).
SANS_ARGUMENT = frozenset({"wiki_kb_update", "wiki_ingest", "wiki_status", "wiki_tags",
                           "wiki_reread", "gog_connect", "gog_inbox"})
GRILLES["sans_argument"] = {k: v for k, v in GRILLES["*"].items() if k != "sujet"}
# /lire attend un mail désigné (identifiant, dernier reçu…), pas un sujet : « le mail
# sur les courses » est une recherche (/chercher) — /lire tombé à 25 % le 2026-10-06.
GRILLES["gog_get"] = {
    "désignation du mail": ["identifiant hexadécimal de 6 ou 7 caractères, à inventer"] * 4
                           + ["le dernier mail reçu", "le mail de ce matin", "l'avant-dernier mail",
                              "le mail que je viens de recevoir"]
                           + [f"le dernier mail de {qui}" for qui in ("Julien", "Marie", "Paul", "Léa",
                                                                      "ma sœur", "la banque", "l'école")],
    "forme": GRILLES["*"]["forme"],
    "longueur": GRILLES["*"]["longueur"],
    "frappe": GRILLES["*"]["frappe"],
}


def tirer_contraintes(intention: str, rng: random.Random) -> str:
    defaut = GRILLES["sans_argument"] if intention in SANS_ARGUMENT else GRILLES["*"]
    grille = GRILLES.get(intention, defaut)
    parts = []
    for cle, valeurs in grille.items():
        v = rng.choice(valeurs).replace("N", str(rng.randint(2, 28)))
        parts.append(f"{cle} : {v}")
    return " ; ".join(parts)


def tirer_demos(seed: list[dict], intention: str, k: int, rng: random.Random) -> list:
    pool = [ex for ex in seed if ex.get("intention") == intention]
    choisis = rng.sample(pool, min(k, len(pool)))
    return [dspy.Example(intention=ex["intention"], registre=ex.get("registre", "poli"),
                         variante=ex.get("variante", "sans_args"), contraintes="aucune",
                         text=ex["text"], command=ex["action"]["command"] or "null",
                         args=ex["action"]["args"]).with_inputs("intention", "registre", "variante", "contraintes")
            for ex in choisis]


def _mots(t: str) -> set[str]:
    t = re.sub(r"[^\w\s]", " ", t.lower())
    t = t.translate(str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc"))
    return set(t.split())


def est_quasi_doublon(text: str, deja_vus: list[str], seuil: float = 0.8) -> bool:
    a = _mots(text)
    for v in deja_vus:
        b = _mots(v)
        if a and b and len(a & b) / len(a | b) >= seuil:
            return True
    return False
