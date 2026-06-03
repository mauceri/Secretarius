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
    """Pour chaque taille : construit un prototype sur un sous-échantillon, évalue sur test_set."""
    from router_embed import EmbedRouter
    from eval_routing import evaluate

    results: list[dict] = []
    for n in sizes:
        sub = subsample(train_pool, n, seed)
        router = EmbedRouter.from_corpus(sub, threshold=threshold, encode_fn=encode_fn)
        report = evaluate(router, test_set)
        results.append({
            "size": n,
            "accuracy": report.accuracy,
            "per_agent": dict(report.per_agent),
            "clarify_recall": report.per_agent.get("clarify", 0.0),
        })
    return results


def compute_diagnostics(train_pool: list[dict], test_set: list[dict],
                        agents: list[dict], encode_fn) -> dict:
    """Calcule trois métriques diagnostiques sur le routeur prototype final (pool complet).

    Retourne un dict avec :
    - min_correct_sim : score cosinus minimum parmi les prédictions correctes
    - inter_proto : matrice de similarité entre prototypes entraînés
    - prior_sim : matrice de similarité entre descriptions a priori (agents.json)
    """
    import numpy as np
    from router_embed import EmbedRouter
    from eval_routing import evaluate

    router = EmbedRouter.from_corpus(train_pool, encode_fn=encode_fn)
    report = evaluate(router, test_set)

    # 1. Confiance minimum pour une attribution correcte
    correct_sims = []
    for row in test_set:
        result = router.route(row["message"])
        if result.agent == row["agent"]:
            correct_sims.append(result.confidence)
    min_correct = min(correct_sims) if correct_sims else float("nan")

    # 2. Similarité inter-prototype (entre prototypes entraînés)
    proto_names = router._agents
    proto_matrix = router._matrix  # (n_agents, dim), L2-normalisé
    inter_proto = (proto_matrix @ proto_matrix.T).tolist() if len(proto_names) > 1 else []

    # 3. Similarité a priori entre descriptions (sans données d'entraînement)
    desc_names = [a["name"] for a in agents]
    descs = [a["description"] for a in agents]
    desc_vecs = encode_fn(descs)
    # L2-normaliser
    norms = np.linalg.norm(desc_vecs, axis=1, keepdims=True) + 1e-12
    desc_vecs = (desc_vecs / norms).astype(np.float32)
    prior_sim = (desc_vecs @ desc_vecs.T).tolist()

    return {
        "min_correct_sim": min_correct,
        "inter_proto": inter_proto,
        "proto_names": proto_names,
        "prior_sim": prior_sim,
        "desc_names": desc_names,
    }


def _fmt_sim_matrix(names: list[str], matrix: list[list[float]]) -> str:
    """Formate une matrice de similarité en tableau Markdown."""
    header = "| |" + "|".join(f" {n} " for n in names) + "|"
    sep = "|---|" + "|".join("---" for _ in names) + "|"
    rows = [header, sep]
    for i, row_name in enumerate(names):
        cells = "|".join(f" {matrix[i][j]:.2f} " for j in range(len(names)))
        rows.append(f"| **{row_name}** |{cells}|")
    return "\n".join(rows)


def format_experiment_report(results: list[dict], cost_summary: str,
                             min_accuracy: float, agent_names: list[str], cap: int,
                             diagnostics: dict | None = None) -> str:
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
        "| Taille/agent | Exactitude | Rappel clarify | ≥ seuil |",
        "|---|---|---|---|",
    ]
    for r in results:
        ok = "✓" if r["accuracy"] >= min_accuracy else ""
        lines.append(
            f"| {r['size']} | {r['accuracy']:.1%} "
            f"| {r['clarify_recall']:.1%} | {ok} |"
        )
    if diagnostics:
        d = diagnostics
        lines += [
            "",
            "## Diagnostics — routeur final (pool complet)",
            "",
            f"**Confiance minimum pour une attribution correcte :** {d['min_correct_sim']:.3f}",
            "(Score cosinus le plus bas parmi les prédictions correctes sur le test set.)",
            "",
            "### Similarité inter-prototype (entraîné)",
            "(Cosinus entre prototypes — proche de 1 = classes difficiles à distinguer)",
            "",
            _fmt_sim_matrix(d["proto_names"], d["inter_proto"]),
            "",
            "### Similarité a priori entre descriptions (sans données)",
            "(Cosinus entre les descriptions de agents.json encodées par BGE-M3)",
            "",
            _fmt_sim_matrix(d["desc_names"], d["prior_sim"]),
        ]
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
    from critique import critique_batch

    def _gen_and_critique(agent, n):
        prompt = build_generation_prompt(agent, agents, examples=[], negatives=[], n=n)
        text, usage = generate_fn(prompt)
        cost.add(_DEEPSEEK_MODEL, usage)
        cands = parse_candidates(text, agent["name"])
        kept, cusage = critique_batch(cands, agents, critique_fn)
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
    # Clarify : splitté entre train et test (compète comme classe normale)
    if len(clarify_pool) > 1:
        train_clarify, test_clarify = stratified_split(clarify_pool, args.test_frac, args.seed)
    else:
        train_clarify, test_clarify = [], clarify_pool
    train_pool = train_pool + train_clarify
    test_set = test_real + test_clarify

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    clamped, cap = clamp_sizes(sizes, train_pool)

    print(f"[experiment] Courbe sur tailles {clamped} (cap={cap}), test={len(test_set)} cas...")
    results = run_curve(train_pool, test_set, clamped, args.threshold, _default_encode, args.seed)

    print("[experiment] Calcul des diagnostics (routeur final sur pool complet)...")
    diag = compute_diagnostics(train_pool, test_set, agents, _default_encode)

    report = format_experiment_report(
        results, cost.summary(), args.min_accuracy, [a["name"] for a in agents], cap,
        diagnostics=diag,
    )
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
