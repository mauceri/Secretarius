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


_DEEPSEEK_MODEL = "deepseek-chat"
_MISTRAL_MODEL = "mistralai/Mistral-Small-4-119B-2603"


def build_pool(agents, max_per_agent, clarify_k, generate_fn, critique_fn, cost):
    """Génère (DeepSeek) puis critique (Mistral) le pool. Retourne (pool_réel, pool_clarify).

    pool_réel : exemples approuvés des agents réels. pool_clarify : exemples clarify approuvés
    (destinés au test uniquement). cost est un CostTracker mis à jour au fil des appels.
    """
    from corpus_gen import build_generation_prompt, parse_candidates
    from critique import critique_candidates

    def _gen_and_critique(agent, n):
        prompt = build_generation_prompt(agent, agents, examples=[], negatives=[], n=n)
        text, usage = generate_fn(prompt)
        cost.add(_DEEPSEEK_MODEL, usage)
        cands = parse_candidates(text, agent["name"])
        kept, cusage = critique_candidates(cands, agents, critique_fn)
        cost.add(_MISTRAL_MODEL, cusage)
        return kept

    pool: list[dict] = []
    for agent in agents:
        if agent["name"] == "clarify":
            continue
        pool.extend(_gen_and_critique(agent, max_per_agent))

    clarify_pool: list[dict] = []
    clarify_agent = next((a for a in agents if a["name"] == "clarify"), None)
    if clarify_agent is not None and clarify_k > 0:
        clarify_pool.extend(_gen_and_critique(clarify_agent, clarify_k))

    return pool, clarify_pool


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from router_base import load_agents
    from eval_routing import stratified_split
    from router_embed import _default_encode
    from cost import CostTracker
    from llm_clients import deepseek_generate, mistral_critique

    parser = argparse.ArgumentParser(description="Expérience de routage (génère, critique, courbe)")
    parser.add_argument("--agents", default="agents.json")
    parser.add_argument("--max-per-agent", type=int, default=20)
    parser.add_argument("--clarify", type=int, default=15)
    parser.add_argument("--sizes", default="3,6,9,12")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="experiment_report.md")
    args = parser.parse_args()

    agents = load_agents(args.agents)
    cost = CostTracker()

    print("[experiment] Génération + critique du pool (DeepSeek → Mistral)...")
    pool, clarify_pool = build_pool(agents, args.max_per_agent, args.clarify,
                                    deepseek_generate, mistral_critique, cost)
    Path("experiment_pool.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool + clarify_pool),
        encoding="utf-8",
    )

    train_pool, test_real = stratified_split(pool, args.test_frac, args.seed)
    test_set = test_real + clarify_pool

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    clamped, cap = clamp_sizes(sizes, train_pool)

    print(f"[experiment] Courbe sur tailles {clamped} (cap={cap}), test={len(test_set)} cas...")
    results = run_curve(train_pool, test_set, clamped, args.threshold, _default_encode, args.seed)

    report = format_experiment_report(
        results, cost.summary(), args.min_accuracy, [a["name"] for a in agents], cap
    )
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
