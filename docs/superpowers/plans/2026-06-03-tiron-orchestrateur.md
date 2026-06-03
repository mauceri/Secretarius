# Tiron Orchestrateur (Spec 3) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformer Tiron en orchestrateur léger à prompt minuscule : il détecte l'intention via un outil MCP (`route_intent`) puis délègue à un agent spécialiste (`sessions_spawn`). Premier spécialiste : wikilm.

**Architecture:** Un nouveau service MCP `router-mcp` (port 8903, FastMCP streamable-http) expose un seul outil `route_intent(message) -> agent_name` qui charge le corpus `Wiki_LM/routing/corpus.jsonl` et interroge `EmbedRouter`. L'agent `main` (Tiron) a un prompt minimal + 2 outils : `route_intent` et `sessions_spawn`. Un nouvel agent `wikilm` a ses propres outils wiki-lm et son prompt spécialisé.

**Tech Stack:** fastmcp 3.3.1, sentence_transformers (BGE-M3 déjà chargé via wiki-lm-mcp), systemd user service, openclaw.json template.

---

## Structure des fichiers

| Opération | Fichier | Rôle |
|-----------|---------|------|
| Créer | `Wiki_LM/routing/routing_mcp.py` | Serveur MCP router-mcp, outil route_intent |
| Créer | `Wiki_LM/routing/tests/test_routing_mcp.py` | Test unitaire route_intent (hors réseau) |
| Créer | `openclaw-config/router-mcp.service` | Service systemd port 8903 |
| Modifier | `openclaw-config/openclaw.json.template` | Ajouter router-mcp, agent wikilm, Tiron minimal |
| Modifier | `openclaw-config/install.sh` | Installer router-mcp.service |
| Modifier | `openclaw-config/workspace/AGENTS.md` | Instructions routage Tiron |

**Interfaces existantes réutilisées** (ne pas réimplémenter) :
- `Wiki_LM/routing/router_base.py` : `load_agents(path)`, `load_corpus(path)`, `RouteResult`
- `Wiki_LM/routing/router_embed.py` : `EmbedRouter.from_corpus(corpus)`, `EmbedRouter.route(msg)`
- `Wiki_LM/routing/agents.json` : catalogue des agents (wikilm, gog, superpowers, clarify)
- `Wiki_LM/routing/corpus.jsonl` : corpus étiqueté validé

---

### Task 1 : Service router-mcp (routing_mcp.py + systemd)

**Files:**
- Create: `Wiki_LM/routing/routing_mcp.py`
- Create: `openclaw-config/router-mcp.service`
- Test: `Wiki_LM/routing/tests/test_routing_mcp.py`

- [ ] **Step 1 : Écrire le test qui échoue**

Créer `Wiki_LM/routing/tests/test_routing_mcp.py` :

```python
import json
from pathlib import Path

import numpy as np


def _fake_encode(texts):
    vecs = []
    for t in texts:
        low = t.lower()
        if "mail" in low:
            vecs.append([1.0, 0.0, 0.0, 0.0])
        elif "wiki" in low or "url" in low:
            vecs.append([0.0, 1.0, 0.0, 0.0])
        elif "rédige" in low or "plan" in low:
            vecs.append([0.0, 0.0, 1.0, 0.0])
        else:
            vecs.append([0.0, 0.0, 0.0, 1.0])
    return np.array(vecs, dtype=np.float32)


def _write_test_corpus(tmp_path):
    agents = tmp_path / "agents.json"
    agents.write_text(json.dumps({"agents": [
        {"name": "gog",        "description": "email et agenda"},
        {"name": "wikilm",     "description": "base de connaissances"},
        {"name": "superpowers","description": "rédaction"},
        {"name": "clarify",    "description": "intention floue"},
    ]}), encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"message": "envoie un mail", "agent": "gog"}\n'
        '{"message": "envoie un mail urgent", "agent": "gog"}\n'
        '{"message": "capture url wiki", "agent": "wikilm"}\n'
        '{"message": "note dans le wiki", "agent": "wikilm"}\n'
        '{"message": "rédige un plan", "agent": "superpowers"}\n'
        '{"message": "rédige une spec", "agent": "superpowers"}\n'
        '{"message": "aide-moi", "agent": "clarify"}\n'
        '{"message": "bla bla flou", "agent": "clarify"}\n',
        encoding="utf-8",
    )
    return agents, corpus


def test_route_intent_real_agent(tmp_path):
    import routing_mcp
    agents_path, corpus_path = _write_test_corpus(tmp_path)
    routing_mcp.AGENTS_PATH = agents_path
    routing_mcp.CORPUS_PATH = corpus_path
    routing_mcp._router = None

    result = routing_mcp._route("envoie un mail", encode_fn=_fake_encode)
    assert result == "gog"


def test_route_intent_clarify(tmp_path):
    import routing_mcp
    agents_path, corpus_path = _write_test_corpus(tmp_path)
    routing_mcp.AGENTS_PATH = agents_path
    routing_mcp.CORPUS_PATH = corpus_path
    routing_mcp._router = None

    result = routing_mcp._route("aide-moi", encode_fn=_fake_encode)
    assert result == "clarify"


def test_route_intent_wikilm(tmp_path):
    import routing_mcp
    agents_path, corpus_path = _write_test_corpus(tmp_path)
    routing_mcp.AGENTS_PATH = agents_path
    routing_mcp.CORPUS_PATH = corpus_path
    routing_mcp._router = None

    result = routing_mcp._route("capture url wiki", encode_fn=_fake_encode)
    assert result == "wikilm"
```

- [ ] **Step 2 : Lancer le test, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_routing_mcp.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'routing_mcp'`

- [ ] **Step 3 : Implémenter routing_mcp.py**

Créer `Wiki_LM/routing/routing_mcp.py` :

```python
"""Serveur MCP router-mcp — outil unique : route_intent(message) -> agent_name.

Charge le corpus Wiki_LM/routing/corpus.jsonl et agents.json au premier appel.
Utilise EmbedRouter (prototype BGE-M3) validé par l'expérience (94.7% @ 6 ex/agent).
"""
from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("router-mcp")

_HERE = Path(__file__).resolve().parent
AGENTS_PATH = _HERE / "agents.json"
CORPUS_PATH = _HERE / "corpus.jsonl"

_router = None


def _get_router(encode_fn=None):
    global _router
    if _router is None:
        from router_base import load_agents, load_corpus
        from router_embed import EmbedRouter
        agents = load_agents(AGENTS_PATH)
        corpus = load_corpus(CORPUS_PATH)
        kw = {"encode_fn": encode_fn} if encode_fn else {}
        _router = EmbedRouter.from_corpus(corpus, **kw)
    return _router


def _route(message: str, encode_fn=None) -> str:
    """Logique testable : retourne l'agent cible pour ce message."""
    router = _get_router(encode_fn)
    return router.route(message).agent


@mcp.tool()
def route_intent(message: str) -> str:
    """Détecte l'intention du message et retourne le nom de l'agent cible.

    Valeurs possibles : wikilm | gog | superpowers | clarify
    """
    return _route(message)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8903)
```

- [ ] **Step 4 : Lancer le test, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_routing_mcp.py -v`
Expected: PASS (3 tests)

Note : fastmcp décore `route_intent` avec `__wrapped__` accessible en test ; l'encodeur factice est passé pour éviter le chargement de BGE-M3.

- [ ] **Step 5 : Créer le service systemd**

Créer `openclaw-config/router-mcp.service` :

```ini
[Unit]
Description=Router MCP Server (route_intent, port 8903)
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/Secretarius/Wiki_LM/routing
ExecStart=%h/Secretarius/Wiki_LM/.venv/bin/python3 %h/Secretarius/Wiki_LM/routing/routing_mcp.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=router-mcp

[Install]
WantedBy=default.target
```

- [ ] **Step 6 : Vérifier que toute la suite routing passe**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/ -q`
Expected: tous les tests passent (50 + 3 nouveaux = 53)

- [ ] **Step 7 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/routing_mcp.py \
        Wiki_LM/routing/tests/test_routing_mcp.py \
        openclaw-config/router-mcp.service
git commit -m "feat(orchestrateur): router-mcp — outil route_intent (EmbedRouter BGE-M3, port 8903)"
```

---

### Task 2 : Configuration OpenClaw (template + install.sh)

**Files:**
- Modify: `openclaw-config/openclaw.json.template`
- Modify: `openclaw-config/install.sh`

Le template doit recevoir 4 modifications :

1. `mcp.servers` : ajouter router-mcp
2. `tools.sandbox.tools.allow` : ajouter `router-mcp__route_intent`
3. `agents.list[0]` (main) : ajouter wikilm aux allowAgents, restreindre le sandbox
4. `agents.list` : ajouter l'agent wikilm

- [ ] **Step 1 : Ajouter router-mcp aux serveurs MCP**

Dans `openclaw-config/openclaw.json.template`, section `"mcp"."servers"`, ajouter après gog :

```json
      "router-mcp": {
        "url": "http://127.0.0.1:8903/mcp",
        "transport": "http"
      }
```

- [ ] **Step 2 : Ajouter route_intent à la liste globale allow**

Dans `tools.sandbox.tools.allow`, ajouter :
```
"router-mcp__route_intent",
```

- [ ] **Step 3 : Mettre à jour l'agent main (allowAgents + sandbox)**

Remplacer le bloc `agents.list[0]` (id: "main") :

**Avant :**
```json
      {
        "id": "main",
        "subagents": {
          "allowAgents": [
            "scout"
          ]
        }
      },
```

**Après :**
```json
      {
        "id": "main",
        "subagents": {
          "allowAgents": ["scout", "wikilm"]
        },
        "sandbox": {
          "tools": {
            "allow": [
              "read",
              "sessions_list",
              "sessions_send",
              "sessions_spawn",
              "session_status",
              "group:runtime",
              "router-mcp__route_intent"
            ]
          }
        }
      },
```

- [ ] **Step 4 : Ajouter l'agent wikilm**

Dans `agents.list`, après le bloc main et avant le bloc scout, ajouter :

```json
      {
        "id": "wikilm",
        "model": "deepseek/deepseek-chat",
        "systemPrompt": "Tu es l'agent spécialisé Wiki_LM. Tu reçois une demande transmise par l'orchestrateur Tiron. Utilise les outils wiki_* pour y répondre. Sois concis et factuel. Si la demande demande de mémoriser une URL, utilise wiki_capture puis wiki_ingest.",
        "mcp": {
          "servers": {
            "wiki-lm": {
              "url": "http://127.0.0.1:8901/mcp",
              "transport": "http"
            }
          }
        },
        "sandbox": {
          "tools": {
            "allow": [
              "read",
              "sessions_send",
              "wiki-lm__wiki_capture",
              "wiki-lm__wiki_ingest",
              "wiki-lm__wiki_query",
              "wiki-lm__wiki_tags",
              "wiki-lm__wiki_ingest_status",
              "wiki-lm__wiki_list_pending",
              "wiki-lm__wiki_kb_update"
            ]
          }
        }
      },
```

- [ ] **Step 5 : Valider le JSON du template**

```bash
EURIA_PRODUCT_ID=109005 EURIA_API_KEY=x DEEPSEEK_API_KEY=x \
OPENCLAW_GATEWAY_TOKEN=x HOME=/home/mauceric HOSTNAME=sanroque \
OBSIDIAN_PATH=/x ASSISTANT_NAME=Tiron LLM_BACKEND=deepseek GOG_ACCOUNT=x \
envsubst < ~/Secretarius/openclaw-config/openclaw.json.template | python3 -m json.tool > /dev/null && echo "JSON valide ✓"
```

Expected: `JSON valide ✓`

- [ ] **Step 6 : Ajouter router-mcp.service à install.sh**

Dans `openclaw-config/install.sh`, après le bloc `# Services MCP SSE` (qui installe wiki-lm-mcp et gog-mcp), ajouter :

```bash
# Service router-mcp (routeur d'intention EmbedRouter BGE-M3)
ROUTER_SVC_DST="${SYSTEMD_USER_DIR}/router-mcp.service"
if [[ -f "$ROUTER_SVC_DST" && "$FORCE" != "true" ]]; then
  info "router-mcp.service existe déjà — ignoré"
else
  cp "${SCRIPT_DIR}/router-mcp.service" "$ROUTER_SVC_DST"
  info "router-mcp.service installé dans ${SYSTEMD_USER_DIR}"
fi
ROUTER_BIN="${HOME}/Secretarius/Wiki_LM/routing/routing_mcp.py"
if [[ -f "$ROUTER_BIN" ]]; then
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable router-mcp.service 2>/dev/null && \
    info "router-mcp.service activé au boot" || \
    warn "Activation de router-mcp.service échouée"
fi
```

- [ ] **Step 7 : Commit**

```bash
cd ~/Secretarius
git add openclaw-config/openclaw.json.template openclaw-config/install.sh
git commit -m "feat(orchestrateur): router-mcp dans openclaw, agent wikilm, Tiron minimal"
```

---

### Task 3 : AGENTS.md — instructions de routage Tiron

**Files:**
- Modify: `openclaw-config/workspace/AGENTS.md`

- [ ] **Step 1 : Ajouter la section routage en tête du fichier**

Dans `openclaw-config/workspace/AGENTS.md`, insérer AVANT la section `## Routine de session` existante :

```markdown
## Rôle principal : orchestrateur de routage

**Tiron est un routeur léger.** Il ne répond pas directement aux demandes métier.

Pour chaque message de l'utilisateur :

1. **Appeler `router-mcp__route_intent`** avec le message original, mot pour mot
2. Selon le résultat :
   - `wikilm` → `sessions_spawn` vers l'agent `wikilm`, transmettre le message original intact
   - `gog` → `sessions_spawn` vers l'agent `gog`, transmettre le message original intact
   - `superpowers` → `sessions_spawn` vers l'agent `superpowers`, message intact
   - `clarify` → **demander une précision** à l'utilisateur directement, sans spawn
3. Le sous-agent répond à l'utilisateur. Tiron ne reformule pas, ne résume pas.

**Ne jamais modifier le message avant de le transmettre.**

---

```

- [ ] **Step 2 : Vérifier que le fichier est valide (pas de doublon de section)**

```bash
grep -n "^## " ~/Secretarius/openclaw-config/workspace/AGENTS.md | head -15
```

Expected : la section `## Rôle principal` apparaît en premier, avant `## Routine de session`.

- [ ] **Step 3 : Commit**

```bash
cd ~/Secretarius
git add openclaw-config/workspace/AGENTS.md
git commit -m "feat(orchestrateur): instructions routage Tiron dans AGENTS.md"
```

---

### Task 4 : Déploiement et validation sur sanroque

- [ ] **Step 1 : Installer les services**

```bash
cd ~/Secretarius
./install.sh --force
```

Expected dans la sortie : `router-mcp.service installé`, `router-mcp.service activé au boot`

- [ ] **Step 2 : Démarrer router-mcp**

```bash
systemctl --user daemon-reload
systemctl --user start router-mcp.service
sleep 5
systemctl --user is-active router-mcp.service
```

Expected: `active`

- [ ] **Step 3 : Vérifier que route_intent répond**

```bash
curl -s -X POST http://127.0.0.1:8903/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","method":"tools/call","id":1,"params":{"name":"route_intent","arguments":{"message":"capture cette url https://exemple.fr"}}}' \
  | head -c 300
```

Expected : réponse JSON contenant `"wikilm"` dans le résultat.

- [ ] **Step 4 : Mettre à jour openclaw.json runtime depuis le template**

```bash
cd ~/Secretarius
./install.sh --force
```

(Régénère `~/.openclaw/openclaw.json` depuis le template mis à jour)

- [ ] **Step 5 : Redémarrer le gateway et vérifier les outils**

```bash
systemctl --user restart openclaw-gateway.service
sleep 15
journalctl --user -u openclaw-gateway --no-pager -n 20 | grep -E "Registered|found|ready|router-mcp"
```

Expected : `router-mcp: found 1 tools` + `Registered: route_intent`

- [ ] **Step 6 : Test Telegram — routage wiki**

Envoyer depuis Telegram : `Qu'est-ce que la base de connaissances dit sur la sobriété énergétique ?`

Expected : Tiron appelle `route_intent` → reçoit `wikilm` → spawne l'agent wikilm → wikilm répond avec le résultat de `wiki_query`.

- [ ] **Step 7 : Test Telegram — cas clarify**

Envoyer depuis Telegram : `Aide-moi`

Expected : Tiron demande une précision directement, sans spawner de sous-agent.

- [ ] **Step 8 : Commit + push**

```bash
cd ~/Secretarius
git push origin v0.2.0-dev
```

---

## Notes d'exécution

- **Ordre** : T1 → T2 → T3 → T4. T1-T3 peuvent être revus sans gateway live ; T4 exige les services démarrés.
- **`_route` vs `route_intent`** : fastmcp 3.x retourne la fonction originale sans wrapper (`__wrapped__` absent). La logique testable vit dans `_route(message, encode_fn)` ; `route_intent` est le décorateur MCP qui appelle `_route` avec l'encodeur par défaut. Les tests appellent `routing_mcp._route(...)` directement.
- **BGE-M3 non rechargé** : `router-mcp` et `wiki-lm-mcp` chargent tous deux BGE-M3, mais dans des processus Python séparés — deux instances en mémoire. Sur un iGPU à RAM partagée, surveiller la mémoire (`free -h`) si les deux sont actifs simultanément.
- **Le build llama.cpp est CPU-only** (confirmé) : `tiron-llm` (Phi-4-mini sur :8998) n'est pas activé pour ce spec. Le routeur prototype + DeepSeek pour wikilm est la config cible.
