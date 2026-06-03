"""Pipeline d'expérience : génération+critique du corpus, courbe d'apprentissage, rapport.

Cette première moitié contient la logique pure (testable hors-ligne). La génération
réelle (build_pool) et le câblage CLI (main) sont ajoutés ensuite.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict


def subsample(train_pool: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sous-échantillonne n exemples par agent (stratifié, déterministe)."""
    by_agent: dict = defaultdict(list)
    for row in train_pool:
        by_agent[row["agent"]].append(row)
    rng = random.Random(seed)
    out: list[dict] = []
    for rows in by_agent.values():
        rows = rows[:]
        rng.shuffle(rows)
        out.extend(rows[:n])
    return out


def clamp_sizes(sizes: list[int], train_pool: list[dict]):
    """Plafonne chaque taille au minimum d'exemples disponibles parmi les agents.

    Retourne (tailles_plafonnées_triées_uniques, cap).
    """
    counts = Counter(r["agent"] for r in train_pool)
    cap = min(counts.values()) if counts else 0
    clamped = sorted({min(s, cap) for s in sizes if s > 0})
    return clamped, cap


def run_curve(train_pool: list[dict], test_set: list[dict], sizes: list[int],
              threshold: float, encode_fn, seed: int = 42) -> list[dict]:
    """Pour chaque (taille, mécanisme) : construit sur un sous-échantillon, évalue sur test_set."""
    from router_embed import EmbedRouter
    from router_clf import ClfRouter
    from eval_routing import evaluate

    mechanisms = {"prototype": EmbedRouter, "clf": ClfRouter}
    results: list[dict] = []
    for n in sizes:
        sub = subsample(train_pool, n, seed)
        for name, cls in mechanisms.items():
            router = cls.from_corpus(sub, threshold=threshold, encode_fn=encode_fn)
            report = evaluate(router, test_set)
            results.append({
                "size": n,
                "mechanism": name,
                "accuracy": report.accuracy,
                "per_agent": dict(report.per_agent),
                "clarify_recall": report.per_agent.get("clarify", 0.0),
            })
    return results


def format_experiment_report(results: list[dict], cost_summary: str,
                             min_accuracy: float, agent_names: list[str], cap: int) -> str:
    lines = [
        "# Rapport d'expérience — routage par intention",
        "",
        "⚠ Corpus SYNTHÉTIQUE (généré par DeepSeek, critiqué par Mistral).",
        "Les exactitudes sont un PLAFOND OPTIMISTE, pas la précision réelle sur de vrais",
        "utilisateurs. Ce qui transfère : la comparaison relative et la forme de la courbe.",
        "",
        f"Taille max disponible par agent (après rejets du critique) : {cap}",
        f"Seuil d'acceptabilité : {min_accuracy:.0%}",
        "",
        "| Taille/agent | Mécanisme | Exactitude | Rappel clarify | ≥ seuil |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        ok = "✓" if r["accuracy"] >= min_accuracy else ""
        lines.append(
            f"| {r['size']} | {r['mechanism']} | {r['accuracy']:.1%} "
            f"| {r['clarify_recall']:.1%} | {ok} |"
        )
    lines += ["", "## Coût réel mesuré", cost_summary]
    return "\n".join(lines)
