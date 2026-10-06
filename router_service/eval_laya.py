"""Évalue le classifieur Laya à deux niveaux (spec 2026-10-06) :
- sur le jeu de validation (gen_corpus/laya_val.jsonl), avec `classer` ;
- sur le jeu de test manuel (router_service/test_set_manuel.jsonl), avec
  `route_message`, pour que les commandes tapées suivent le vrai chemin.
Choisit le seuil de décision sur la validation (≥ 97 % de décisions justes,
≤ 15 % de demandes de précision)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

Classement = list[tuple[str | None, float]]
_GRILLE = [round(0.30 + 0.05 * k, 2) for k in range(14)]  # 0.30 … 0.95


def _norm(c: str | None) -> str:
    return "aucune" if c is None else c


def metriques(classements: list[tuple[set[str], Classement]], seuil: float) -> dict:
    """Pour chaque exemple : décidé (top ≥ seuil) juste/faux, ou question
    (pertinente si l'une des deux options est attendue)."""
    decides = justes_decides = questions = justes_ou_q = 0
    par_cmd: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for attendu, ranking in classements:
        top, p = ranking[0]
        if p >= seuil:
            decides += 1
            ok = _norm(top) in attendu
            justes_decides += ok
        else:
            questions += 1
            from router_service.server import options_proposees  # même règle que le routeur
            ok = any(_norm(o["command"]) in attendu for o in options_proposees(ranking))
        justes_ou_q += ok
        for a in attendu:
            par_cmd[a][0] += ok
            par_cmd[a][1] += 1
    n = len(classements)
    return {
        "decide_juste": justes_decides / decides if decides else 1.0,
        "taux_question": questions / n if n else 0.0,
        "juste_ou_question": justes_ou_q / n if n else 0.0,
        "par_commande": {c: ok / tot for c, (ok, tot) in sorted(par_cmd.items())},
    }


def choisir_seuil(classements: list[tuple[set[str], Classement]]) -> float:
    for s in _GRILLE:
        m = metriques(classements, s)
        if m["decide_juste"] >= 0.97 and m["taux_question"] <= 0.15:
            return s
    return max(_GRILLE, key=lambda s: metriques(classements, s)["juste_ou_question"])


def attendu_validation(row: dict) -> set[str]:
    a = row["answers"]
    for q in ("commande_wiki", "commande_google"):
        if q in a:
            return {a[q]}
    return {{"web": "/source", "planification": "/cron", "aucune": "aucune"}[a["domaine"]]}


def _route_vers_classement(r: dict) -> Classement:
    """Traduit une réponse de route_message en classement fictif, pour
    réutiliser `metriques` : ok/no_match = décision (p=1), clarify = question (p=0,05)."""
    if r["status"] == "ok":
        return [(r["command"], 1.0)]
    if r["status"] in ("no_match", "answer"):
        return [(None, 1.0)]
    # P_MIN_OPTION : sous le seuil de décision, mais assez pour que toutes les
    # options proposées soient conservées par options_proposees.
    return [(o["command"], 0.05) for o in r["options"]]


def _afficher(titre: str, m: dict) -> None:
    print(f"== {titre}")
    print(f"  décisions justes (sans question) : {m['decide_juste']:.3f}")
    print(f"  demandes de précision            : {m['taux_question']:.3f}")
    print(f"  juste ou question pertinente     : {m['juste_ou_question']:.3f}")
    for c, v in m["par_commande"].items():
        print(f"    {c:12} {v:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--val", default="gen_corpus/laya_val.jsonl")
    parser.add_argument("--manuel", default="router_service/test_set_manuel.jsonl")
    args = parser.parse_args()

    from router_service import server
    from router_service.laya_classifier import LayaClassifier

    clf = LayaClassifier(args.checkpoint)
    val = []
    for line in Path(args.val).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            val.append((attendu_validation(row), clf.classer(row["state"])))
    seuil = choisir_seuil(val)
    print(f"Seuil choisi sur la validation : {seuil}")
    _afficher(f"validation ({len(val)})", metriques(val, seuil))

    server._classifier = clf
    server._faq = None
    server.SEUIL = seuil
    manuel = []
    for line in Path(args.manuel).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            manuel.append((set(row["attendu"]), _route_vers_classement(server.route_message(row["text"]))))
    _afficher(f"jeu manuel ({len(manuel)})", metriques(manuel, 0.5))


if __name__ == "__main__":
    main()
