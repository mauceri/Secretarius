"""Évaluation d'un routeur sur le corpus : split stratifié, métriques, rapport."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass

from router_base import Router, load_agents, load_corpus


def stratified_split(corpus: list[dict], test_frac: float = 0.3, seed: int = 42):
    """Découpe (train, test) en gardant la proportion par agent. Graine fixe = reproductible."""
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in corpus:
        by_agent[row["agent"]].append(row)
    rng = random.Random(seed)
    train: list[dict] = []
    test: list[dict] = []
    for rows in by_agent.values():
        rows = rows[:]
        rng.shuffle(rows)
        # Au moins 1 cas de test par agent dès qu'il y a >1 exemple
        n_test = max(1, round(len(rows) * test_frac)) if len(rows) > 1 else 0
        test.extend(rows[:n_test])
        train.extend(rows[n_test:])
    return train, test


@dataclass
class EvalReport:
    accuracy: float
    per_agent: dict          # agent -> exactitude
    confusion: dict          # (attendu, prédit) -> compte
    misroutes: list          # [{message, expected, predicted, confidence}]


def evaluate(router: Router, test_set: list[dict]) -> EvalReport:
    confusion: dict = defaultdict(int)
    total: dict = defaultdict(int)
    correct_by: dict = defaultdict(int)
    misroutes: list = []
    correct = 0
    for row in test_set:
        expected = row["agent"]
        result = router.route(row["message"])
        predicted = result.agent
        confusion[(expected, predicted)] += 1
        total[expected] += 1
        if predicted == expected:
            correct += 1
            correct_by[expected] += 1
        else:
            misroutes.append({
                "message": row["message"],
                "expected": expected,
                "predicted": predicted,
                "confidence": round(result.confidence, 3),
            })
    accuracy = correct / len(test_set) if test_set else 0.0
    per_agent = {a: correct_by[a] / total[a] for a in total}
    return EvalReport(accuracy, per_agent, dict(confusion), misroutes)


def format_report(report: EvalReport, agent_names: list[str]) -> str:
    lines = [f"Exactitude globale : {report.accuracy:.1%}", "", "Par agent :"]
    for a in agent_names:
        if a in report.per_agent:
            lines.append(f"  {a:14s} {report.per_agent[a]:.1%}")
    lines += ["", "Matrice de confusion (attendu → prédit) :"]
    for expected in agent_names:
        for predicted in agent_names:
            count = report.confusion.get((expected, predicted), 0)
            if count:
                lines.append(f"  {expected:12s} → {predicted:12s} : {count}")
    if report.misroutes:
        lines += ["", f"Erreurs ({len(report.misroutes)}) :"]
        for m in report.misroutes:
            lines.append(
                f"  [{m['expected']} → {m['predicted']} c={m['confidence']}] {m['message']}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue un routeur sur le corpus")
    parser.add_argument("--router", choices=["embed", "llm"], required=True)
    parser.add_argument("--agents", default="agents.json")
    parser.add_argument("--corpus", default="corpus.jsonl")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    agents = load_agents(args.agents)
    corpus = load_corpus(args.corpus)
    train, test = stratified_split(corpus, args.test_frac, args.seed)

    if args.router == "embed":
        from router_embed import EmbedRouter
        router = EmbedRouter.from_corpus(train, threshold=args.threshold)
    else:
        from router_llm import LlmRouter
        router = LlmRouter(agents)

    report = evaluate(router, test)
    print(format_report(report, [a["name"] for a in agents]))


if __name__ == "__main__":
    main()
