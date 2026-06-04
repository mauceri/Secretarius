"""Génération assistée du corpus de routage, itérative et few-shot, via LLM cloud.

Workflow :
  1) python corpus_gen.py --agent gog --n 20
       → génère candidates_gog.jsonl (few-shot si ≥5 exemples validés existent)
  2) revue humaine du fichier candidates_gog.jsonl (éditer/supprimer/ajouter)
  3) python corpus_gen.py --commit --agent gog
       → valide et ajoute les lignes à corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from router_base import load_agents, load_corpus

_FEWSHOT_MIN = 5


def build_generation_prompt(agent: dict, all_agents: list[dict],
                            examples: list[str], negatives: list[str], n: int) -> str:
    lines = [
        f'Génère {n} messages d\'utilisateur variés, en français, qui doivent être '
        f'routés vers l\'agent "{agent["name"]}".',
        f'Rôle de cet agent : {agent["description"]}',
        "Autres agents (NE génère PAS de messages relevant d\'eux) :",
    ]
    for a in all_agents:
        if a["name"] != agent["name"]:
            lines.append(f'- {a["name"]} : {a["description"]}')
    lines += [
        "Contraintes : registres et longueurs variés ; certains avec arguments "
        "(URL, noms, dates), d\'autres sans ; quelques cas-frontière proches d\'un "
        "autre agent mais qui restent du ressort de celui-ci.",
    ]
    if examples:
        lines.append("Exemples déjà validés pour cet agent (inspire-t\'en, varie) :")
        lines += [f"- {e}" for e in examples]
    if negatives:
        lines.append("Évite ce genre de cas (rejetés) :")
        lines += [f"- {e}" for e in negatives]
    lines.append(
        'Réponds par UNE ligne JSON par message, au format : '
        f'{{"message": "...", "agent": "{agent["name"]}"}}'
    )
    return "\n".join(lines)


def parse_candidates(text: str, agent_name: str) -> list[dict]:
    """Extrait les lignes JSON valides ; force le label = agent_name."""
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message")
        if isinstance(msg, str) and msg.strip():
            out.append({"message": msg.strip(), "agent": agent_name})
    return out


def existing_examples(corpus: list[dict], agent_name: str) -> list[str]:
    return [r["message"] for r in corpus if r["agent"] == agent_name]


def commit_candidates(candidates_path: str | Path, corpus_path: str | Path) -> int:
    """Ajoute les candidats bien formés à corpus.jsonl. Retourne le nombre ajouté."""
    valid: list[dict] = []
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg, agent = obj.get("message"), obj.get("agent")
            if isinstance(msg, str) and isinstance(agent, str) and msg.strip():
                valid.append({"message": msg.strip(), "agent": agent})
    with open(corpus_path, "a", encoding="utf-8") as f:
        for obj in valid:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return len(valid)


def _default_llm(prompt: str) -> str:
    """Appel DeepSeek (outillage hors-ligne, cloud acceptable ici)."""
    from openai import OpenAI
    client = OpenAI(base_url="https://api.deepseek.com",
                    api_key=os.environ["DEEPSEEK_API_KEY"])
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    return resp.choices[0].message.content


def main(llm=_default_llm) -> None:
    parser = argparse.ArgumentParser(description="Génération assistée du corpus de routage")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--agents", default="agents.json")
    parser.add_argument("--corpus", default="corpus.jsonl")
    parser.add_argument("--commit", action="store_true",
                        help="Valide candidates_<agent>.jsonl et l'ajoute au corpus")
    args = parser.parse_args()

    candidates_path = f"candidates_{args.agent}.jsonl"

    if args.commit:
        added = commit_candidates(candidates_path, args.corpus)
        print(f"[corpus_gen] {added} cas ajoutés à {args.corpus}")
        return

    all_agents = load_agents(args.agents)
    agent = next((a for a in all_agents if a["name"] == args.agent), None)
    if agent is None:
        raise SystemExit(f"Agent inconnu : {args.agent}")

    corpus = load_corpus(args.corpus) if Path(args.corpus).exists() else []
    examples = existing_examples(corpus, args.agent)
    examples = examples if len(examples) >= _FEWSHOT_MIN else []

    prompt = build_generation_prompt(agent, all_agents, examples, negatives=[], n=args.n)
    text = llm(prompt)
    cands = parse_candidates(text, args.agent)
    with open(candidates_path, "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    mode = "few-shot" if examples else "zéro-shot"
    print(f"[corpus_gen] {len(cands)} candidats ({mode}) écrits dans {candidates_path}")
    print("Relisez/éditez ce fichier, puis : "
          f"python corpus_gen.py --commit --agent {args.agent}")


if __name__ == "__main__":
    main()
