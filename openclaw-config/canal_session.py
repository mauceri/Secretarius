#!/usr/bin/env python3
"""Ajoute le canal Session (plugin openclaw-session) à une config openclaw.json.

Idempotent. Telegram n'est pas touché (phase 1 : les deux canaux coexistent).
Usage : canal_session.py <openclaw.json> <identifiant Session du propriétaire> <port du broker>
"""
import json
import re
import sys

ID_SESSION = re.compile(r"^05[0-9a-f]{64}$")


def ajouter_canal_session(config: dict, owner_id: str, port: int) -> dict:
    if not ID_SESSION.match(owner_id):
        raise ValueError("identifiant Session invalide (66 caractères hexadécimaux, préfixe 05)")
    config.setdefault("channels", {})["session"] = {
        "enabled": True,
        "brokerUrl": f"http://127.0.0.1:{port}",
        "dmPolicy": "allowlist",
        "allowFrom": [owner_id],
    }
    liaison = {"agentId": "main", "match": {"channel": "session", "accountId": "default"}}
    bindings = config.setdefault("bindings", [])
    if liaison not in bindings:
        bindings.append(liaison)
    owners = config.setdefault("commands", {}).setdefault("ownerAllowFrom", [])
    if f"session:{owner_id}" not in owners:
        owners.append(f"session:{owner_id}")
    config.setdefault("plugins", {}).setdefault("entries", {})["session"] = {"enabled": True}
    return config


def main() -> None:
    chemin, owner_id, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(chemin, encoding="utf-8") as f:
        config = json.load(f)
    ajouter_canal_session(config, owner_id, port)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
