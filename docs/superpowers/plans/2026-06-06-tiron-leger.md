# Tiron léger — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Créer une instance OpenClaw 5.12 isolée avec Tiron sur phi-4-mini, contexte réduit à ~1k tokens, et valider le prefill + conversations via gateway UI.

**Architecture:** Instance `--profile slm` dans `~/.openclaw-slm/`, port 18790, Telegram et Tailscale désactivés. Tiron garde exec (gog, cat, ls, find) mais perd tous les outils MCP. Workspace réduit (AGENTS.md ~300 tokens). Prod 6.1 non touchée.

**Tech Stack:** OpenClaw 2026.5.12 (npm local), phi-4-mini via llama.cpp port 8998 (partagé), bash, Python 3, systemd user service, pytest.

---

## Cartographie des fichiers

| Action | Fichier |
|---|---|
| Créer | `openclaw-config/openclaw-slm.json.template` |
| Créer | `openclaw-config/gateway-slm.systemd.env.template` |
| Créer | `openclaw-config/openclaw-gateway-slm.service` |
| Créer | `openclaw-config/workspace-slm/AGENTS.md` |
| Créer | `openclaw-config/workspace-slm/TOOLS.md` |
| Créer | `openclaw-config/workspace-slm/SOUL.md` (copie) |
| Créer | `openclaw-config/workspace-slm/USER.md` (copie) |
| Créer | `openclaw-config/workspace-slm/IDENTITY.md` (copie) |
| Créer | `slm/bench_prefill.py` |
| Créer | `slm/tests/__init__.py` |
| Créer | `slm/tests/test_bench_prefill.py` |
| Modifier | `openclaw-config/install.sh` |

---

## Task 1 : Templates de config (openclaw-slm.json + gateway env)

**Files:**
- Create: `openclaw-config/openclaw-slm.json.template`
- Create: `openclaw-config/gateway-slm.systemd.env.template`

- [ ] **Step 1 : Créer openclaw-slm.json.template**

```bash
cat > ~/Secretarius/openclaw-config/openclaw-slm.json.template << 'ENDOFFILE'
{
  "env": {
    "EURIA_API_KEY": "${EURIA_API_KEY}",
    "EURIA_PRODUCT_ID": "${EURIA_PRODUCT_ID}"
  },
  "auth": {
    "profiles": {
      "euria:default": {
        "provider": "euria",
        "mode": "api_key"
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "euria": {
        "baseUrl": "https://api.infomaniak.com/2/ai/${EURIA_PRODUCT_ID}/openai/v1",
        "api": "openai-completions",
        "models": [
          {
            "id": "mistralai/Mistral-Small-4-119B-2603",
            "name": "Mistral Small 4 (Euria)",
            "api": "openai-completions",
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 131072,
            "maxTokens": 8192
          }
        ]
      },
      "tiron-llm": {
        "baseUrl": "http://127.0.0.1:8998/v1",
        "api": "openai-completions",
        "models": [
          {
            "id": "phi-4-mini-instruct",
            "name": "Phi-4-mini (SLM local)",
            "api": "openai-completions",
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 32768,
            "maxTokens": 2048
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "tiron-llm/phi-4-mini-instruct"
      },
      "models": {
        "tiron-llm/phi-4-mini-instruct": { "alias": "TironSLM" },
        "euria/mistralai/Mistral-Small-4-119B-2603": { "alias": "Euria" }
      },
      "workspace": "${HOME}/.openclaw-slm/workspace",
      "compaction": {
        "mode": "safeguard",
        "reserveTokensFloor": 20000
      },
      "maxConcurrent": 1,
      "subagents": { "maxConcurrent": 2 },
      "sandbox": {
        "mode": "all",
        "workspaceAccess": "rw",
        "scope": "session"
      }
    },
    "list": [
      { "id": "main" }
    ]
  },
  "tools": {
    "exec": {
      "host": "gateway",
      "security": "allowlist",
      "ask": "on-miss",
      "safeBins": ["gog", "cat", "ls", "find"],
      "applyPatch": { "enabled": false },
      "safeBinProfiles": {
        "cat": {}, "find": {}, "gog": {}, "ls": {}
      }
    },
    "sandbox": {
      "tools": {
        "allow": [
          "gog",
          "read",
          "sessions_list",
          "sessions_spawn",
          "sessions_yield",
          "group:runtime"
        ],
        "deny": [
          "browser", "canvas", "nodes", "cron",
          "web_search", "web_fetch",
          "write", "edit", "apply_patch", "group:fs"
        ]
      }
    }
  },
  "messages": { "ackReactionScope": "group-mentions" },
  "commands": { "native": "auto", "nativeSkills": "auto", "restart": true },
  "session": { "dmScope": "per-channel-peer" },
  "channels": {
    "telegram": { "enabled": false }
  },
  "gateway": {
    "port": 18790,
    "mode": "local",
    "bind": "loopback",
    "controlUi": {
      "allowedOrigins": ["http://localhost:18790"]
    },
    "auth": {
      "mode": "token",
      "token": "${OPENCLAW_GATEWAY_TOKEN}"
    },
    "trustedProxies": ["127.0.0.1", "::1"],
    "nodes": {
      "denyCommands": [
        "camera.snap", "camera.clip", "screen.record",
        "contacts.add", "calendar.add", "reminders.add",
        "sms.send", "config.patch", "config.apply"
      ]
    }
  },
  "plugins": { "allow": [], "entries": {} },
  "skills": { "install": { "nodeManager": "npm" } }
}
ENDOFFILE
```

- [ ] **Step 2 : Valider le JSON**

```bash
python3 -m json.tool ~/Secretarius/openclaw-config/openclaw-slm.json.template > /dev/null \
  && echo "JSON valide" || echo "ERREUR : JSON invalide"
```

Résultat attendu : `JSON valide` (le template contient des `${...}` qui ne sont pas du JSON strict — s'il y a une erreur, c'est sur la structure, pas les variables).

Note : `python3 -m json.tool` peut signaler une erreur sur `${VAR}` selon la version Python. Erreur acceptable uniquement sur les valeurs `${...}` — toute autre erreur indique un problème de syntaxe à corriger.

- [ ] **Step 3 : Créer gateway-slm.systemd.env.template**

```bash
cat > ~/Secretarius/openclaw-config/gateway-slm.systemd.env.template << 'ENDOFFILE'
# gateway-slm.systemd.env — Instance SLM (chmod 600, jamais commité)
# Généré par openclaw-config/install.sh --profile slm

OPENCLAW_GATEWAY_TOKEN=${OPENCLAW_GATEWAY_TOKEN}
OPENCLAW_BIN=${OPENCLAW_BIN}
EURIA_API_KEY=${EURIA_API_KEY}
EURIA_PRODUCT_ID=${EURIA_PRODUCT_ID}
GOG_ACCOUNT=${GOG_ACCOUNT}
ENDOFFILE
```

- [ ] **Step 4 : Commit**

```bash
cd ~/Secretarius
git add openclaw-config/openclaw-slm.json.template openclaw-config/gateway-slm.systemd.env.template
git commit -m "feat(slm): templates config openclaw-slm + gateway-slm env"
```

---

## Task 2 : Service systemd openclaw-gateway-slm

**Files:**
- Create: `openclaw-config/openclaw-gateway-slm.service`

- [ ] **Step 1 : Créer le fichier service**

```bash
cat > ~/Secretarius/openclaw-config/openclaw-gateway-slm.service << 'ENDOFFILE'
[Unit]
Description=OpenClaw Gateway SLM (dev, instance isolée 5.12)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=%h/.openclaw-slm/gateway.systemd.env
ExecStart=${OPENCLAW_BIN} gateway run --profile slm
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
ENDOFFILE
```

- [ ] **Step 2 : Vérifier la syntaxe systemd**

```bash
systemd-analyze verify --user \
  ~/.config/systemd/user/openclaw-gateway-slm.service 2>&1 || \
  echo "(Fichier non encore installé — vérification après install.sh)"
```

Si le fichier n'est pas encore installé, passer à l'étape suivante sans bloquer.

- [ ] **Step 3 : Commit**

```bash
cd ~/Secretarius
git add openclaw-config/openclaw-gateway-slm.service
git commit -m "feat(slm): service systemd openclaw-gateway-slm"
```

---

## Task 3 : Workspace slm (AGENTS.md, TOOLS.md, copies)

**Files:**
- Create: `openclaw-config/workspace-slm/AGENTS.md`
- Create: `openclaw-config/workspace-slm/TOOLS.md`
- Create: `openclaw-config/workspace-slm/SOUL.md` (copie)
- Create: `openclaw-config/workspace-slm/USER.md` (copie)
- Create: `openclaw-config/workspace-slm/IDENTITY.md` (copie)

- [ ] **Step 1 : Créer le répertoire et copier les fichiers inchangés**

```bash
mkdir -p ~/Secretarius/openclaw-config/workspace-slm
cp ~/Secretarius/openclaw-config/workspace/SOUL.md \
   ~/Secretarius/openclaw-config/workspace-slm/SOUL.md
cp ~/Secretarius/openclaw-config/workspace/USER.md \
   ~/Secretarius/openclaw-config/workspace-slm/USER.md
cp ~/Secretarius/openclaw-config/workspace/IDENTITY.md \
   ~/Secretarius/openclaw-config/workspace-slm/IDENTITY.md
```

- [ ] **Step 2 : Créer AGENTS.md (prompt Tiron léger)**

```bash
cat > ~/Secretarius/openclaw-config/workspace-slm/AGENTS.md << 'ENDOFFILE'
# AGENTS.md — ${ASSISTANT_NAME} (instance slm)

## Rôle

${ASSISTANT_NAME} est un orchestrateur léger. Il traite les demandes directement
via les outils exec disponibles. Les capacités wiki, gog et sources externes
seront déléguées à des sous-agents (non disponibles dans cette instance).

## Outils exec disponibles

| Outil | Usage |
|---|---|
| `gog` | Client Google (email, agenda, drive) — voir TOOLS.md |
| `cat`, `ls`, `find` | Navigation et lecture de fichiers locaux |

## Routine de session

**AVANT de répondre au premier message**, lire obligatoirement :
1) `SOUL.md` — règles et personnalité
2) `USER.md` — préférences de l'utilisateur

## Principe fondamental : zéro initiative

Agir **uniquement sur ce qui est demandé explicitement**.
- Ne jamais enchaîner une action corrective de sa propre initiative.
- Ne jamais relancer une opération après un échec sans instruction.
- En cas de doute sur le périmètre : **demander** avant d'agir.

## Gestion des erreurs

1. Rapporter le message d'erreur **complet et exact**.
2. Si une cause probable est identifiable : l'exposer en une phrase.
3. Si une solution est envisageable : la **proposer**, jamais l'exécuter sans confirmation.

## Règles d'exécution (zéro invention)

- **Interdit** : fabriquer une sortie de commande, un ID, un lien, un résultat d'API.
- Toujours exécuter via outil et coller la **sortie réelle**.

## Politique d'actions externes (confirmation obligatoire)

Avant toute action qui écrit/envoie hors machine (email, calendar, drive) :
1) Récapitulatif : **quoi / où / qui / quand**
2) Demande de confirmation : **OUI/NON**
3) Exécution uniquement après **OUI**

## Capacités non disponibles dans cette instance

Wiki, sources web et lecture de contenu externe ne sont pas disponibles.
Les informer à l'utilisateur sans inventer de contenu.
ENDOFFILE
```

- [ ] **Step 3 : Créer TOOLS.md**

```bash
cat > ~/Secretarius/openclaw-config/workspace-slm/TOOLS.md << 'ENDOFFILE'
# TOOLS.md — Environnement local (instance slm)

## gog (client Google)

### Email

`gog email search <requête>` retourne des **identifiants de fil** — ne pas utiliser directement avec `gog email get`.

```bash
# 1. Récupérer l'identifiant de message
gog email messages search "sujet ou expéditeur" -j --results-only

# 2. Lire avec l'identifiant de message (champ "id", pas "threadId")
gog email get <identifiantMessage>
```

## Modèles disponibles

| Alias config | Modèle |
|---|---|
| `tiron-llm/phi-4-mini-instruct` | Phi-4-mini local (SLM, port 8998) — défaut |
| `euria/mistralai/Mistral-Small-4-119B-2603` | Mistral Small 4 (Euria) |

Note : `switch-model` n'est pas disponible dans cette instance (il pointe sur la prod).
Pour changer de modèle, modifier directement `~/.openclaw-slm/openclaw.json`
ou relancer `install.sh --profile slm --force`.
ENDOFFILE
```

- [ ] **Step 4 : Vérifier le nombre de mots du workspace (cible < 1200 mots)**

```bash
wc -w ~/Secretarius/openclaw-config/workspace-slm/*.md
```

Résultat attendu : total < 1200 mots (≈ 1500 tokens). Si dépassement, revoir AGENTS.md.

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add openclaw-config/workspace-slm/
git commit -m "feat(slm): workspace Tiron léger (~1k tokens)"
```

---

## Task 4 : bench_prefill.py (TDD)

**Files:**
- Create: `slm/tests/__init__.py`
- Create: `slm/tests/test_bench_prefill.py`
- Create: `slm/bench_prefill.py`

- [ ] **Step 1 : Créer l'arborescence et le fichier de test**

```bash
mkdir -p ~/Secretarius/slm/tests
touch ~/Secretarius/slm/tests/__init__.py
```

- [ ] **Step 2 : Écrire le test**

```bash
cat > ~/Secretarius/slm/tests/test_bench_prefill.py << 'ENDOFFILE'
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench_prefill import load_workspace, measure_ttft


def _fake_response(content="Bon"):
    chunk = {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
    done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.iter_lines.return_value = [
        b"data: " + json.dumps(chunk).encode(),
        b"data: " + json.dumps(done).encode(),
        b"data: [DONE]",
    ]
    return mock


def test_measure_ttft_returns_float():
    with patch("requests.post", return_value=_fake_response()):
        result = measure_ttft("system prompt", n=3)
    assert isinstance(result, float)
    assert result >= 0.0


def test_measure_ttft_makes_n_calls():
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return _fake_response()

    with patch("requests.post", side_effect=fake_post):
        measure_ttft("system prompt", n=3)
    assert len(calls) == 3


def test_load_workspace_concatenates_md_files(tmp_path):
    (tmp_path / "AGENTS.md").write_text("agents content")
    (tmp_path / "SOUL.md").write_text("soul content")
    (tmp_path / "other.txt").write_text("should be ignored")
    result = load_workspace(str(tmp_path))
    assert "agents content" in result
    assert "soul content" in result
    assert "should be ignored" not in result


def test_load_workspace_excludes_non_md(tmp_path):
    (tmp_path / "README.md").write_text("readme")
    (tmp_path / "config.json").write_text("{}")
    result = load_workspace(str(tmp_path))
    assert "{}" not in result
ENDOFFILE
```

- [ ] **Step 3 : Lancer le test — vérifier qu'il échoue (ModuleNotFoundError)**

```bash
cd ~/Secretarius
python3 -m pytest slm/tests/test_bench_prefill.py -v 2>&1 | head -20
```

Résultat attendu : `ModuleNotFoundError: No module named 'bench_prefill'`

- [ ] **Step 4 : Créer bench_prefill.py**

```bash
cat > ~/Secretarius/slm/bench_prefill.py << 'ENDOFFILE'
#!/usr/bin/env python3
"""Mesure le TTFT (time-to-first-token) de phi-4-mini sur deux prompts système.

Usage : python3 bench_prefill.py
Prérequis : service slm-llama-cpp actif (port 8998), workspaces
  ~/.openclaw/workspace/ (prod) et ~/.openclaw-slm/workspace/ (léger).
"""
import json
import os
import statistics
import time

import requests

LLAMA_URL = "http://127.0.0.1:8998/v1/chat/completions"
MODEL = "phi-4-mini-instruct"
PROD_WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
SLM_WORKSPACE = os.path.expanduser("~/.openclaw-slm/workspace")


def load_workspace(path: str) -> str:
    """Concatène tous les fichiers .md d'un workspace en un prompt système."""
    parts = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".md"):
            with open(os.path.join(path, name)) as f:
                parts.append(f.read())
    return "\n\n---\n\n".join(parts)


def measure_ttft(system_prompt: str, n: int = 3) -> float:
    """Retourne le TTFT médian (secondes) sur n appels streaming."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Bonjour."},
        ],
        "max_tokens": 10,
        "stream": True,
    }
    times = []
    for _ in range(n):
        start = time.perf_counter()
        with requests.post(LLAMA_URL, json=payload, stream=True, timeout=120) as resp:
            for line in resp.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                raw = line.decode().removeprefix("data: ")
                try:
                    chunk = json.loads(raw)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        times.append(time.perf_counter() - start)
                        break
                except json.JSONDecodeError:
                    continue
    return statistics.median(times)


if __name__ == "__main__":
    print("Chargement des workspaces...")
    prod_prompt = load_workspace(PROD_WORKSPACE)
    slm_prompt = load_workspace(SLM_WORKSPACE)
    prod_words = len(prod_prompt.split())
    slm_words = len(slm_prompt.split())
    print(f"  Prompt prod  : {prod_words} mots")
    print(f"  Prompt léger : {slm_words} mots  (ratio {prod_words/slm_words:.1f}x)")
    print()
    print("Mesure prod (3 appels, patience ~2 min)...")
    prod_ttft = measure_ttft(prod_prompt)
    print(f"  TTFT prod  (médiane) : {prod_ttft:.2f}s")
    print("Mesure léger (3 appels)...")
    slm_ttft = measure_ttft(slm_prompt)
    print(f"  TTFT léger (médiane) : {slm_ttft:.2f}s")
    if slm_ttft > 0:
        print(f"\n  Gain : {prod_ttft / slm_ttft:.1f}x")
ENDOFFILE
chmod +x ~/Secretarius/slm/bench_prefill.py
```

- [ ] **Step 5 : Lancer les tests — vérifier qu'ils passent**

```bash
cd ~/Secretarius
python3 -m pytest slm/tests/test_bench_prefill.py -v
```

Résultat attendu :
```
test_bench_prefill.py::test_measure_ttft_returns_float PASSED
test_bench_prefill.py::test_measure_ttft_makes_n_calls PASSED
test_bench_prefill.py::test_load_workspace_concatenates_md_files PASSED
test_bench_prefill.py::test_load_workspace_excludes_non_md PASSED
4 passed
```

Si `requests` n'est pas installé : `pip install requests` (ou dans le venv Wiki_LM).

- [ ] **Step 6 : Commit**

```bash
cd ~/Secretarius
git add slm/
git commit -m "feat(slm): bench_prefill.py — mesure TTFT phi-4-mini"
```

---

## Task 5 : Modification de install.sh

**Files:**
- Modify: `openclaw-config/install.sh`

Huit modifications au total dans une seule transaction. Les voici dans l'ordre d'apparition dans le fichier.

- [ ] **Step 1 : Remplacer le bloc de parsing d'arguments (lignes 8-15)**

Remplacer :
```bash
OBSIDIAN_PATH="${OBSIDIAN_PATH:-$HOME/Documents/Obsidian}"
ASSISTANT_NAME="${ASSISTANT_NAME:-Tiron}"
LLM_BACKEND="${LLM_BACKEND:-deepseek}"
OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw}"
FORCE="${FORCE:-false}"
for _arg in "$@"; do
  [[ "$_arg" == "--force" ]] && FORCE="true"
done
```

Par :
```bash
OBSIDIAN_PATH="${OBSIDIAN_PATH:-$HOME/Documents/Obsidian}"
ASSISTANT_NAME="${ASSISTANT_NAME:-Tiron}"
LLM_BACKEND="${LLM_BACKEND:-deepseek}"
FORCE="${FORCE:-false}"
PROFILE="${PROFILE:-prod}"
_i=0; _args=("$@")
while [[ $_i -lt ${#_args[@]} ]]; do
  case "${_args[$_i]}" in
    --force) FORCE="true" ;;
    --profile) _i=$((_i+1)); PROFILE="${_args[$_i]:-prod}" ;;
  esac
  _i=$((_i+1))
done
unset _i _args
if [[ "$PROFILE" == "slm" ]]; then
  OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw-slm}"
else
  OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw}"
fi
```

- [ ] **Step 2 : Ajouter l'installation d'OpenClaw 5.12 après la détection du binaire (après la ligne `export OPENCLAW_BIN`)**

Après la ligne `export OPENCLAW_BIN`, insérer :
```bash
# Instance slm : installer openclaw@2026.5.12 localement (n'affecte pas la prod)
if [[ "$PROFILE" == "slm" ]]; then
  mkdir -p "${OPENCLAW_PATH}/npm"
  info "Installation d'openclaw@2026.5.12 dans ${OPENCLAW_PATH}/npm..."
  npm install --prefix "${OPENCLAW_PATH}/npm" "openclaw@2026.5.12" --silent 2>&1 || \
    warn "npm install openclaw@2026.5.12 échoué — vérifier npm registry"
  SLM_BIN="${OPENCLAW_PATH}/npm/node_modules/.bin/openclaw"
  if [[ -x "$SLM_BIN" ]]; then
    OPENCLAW_BIN="$SLM_BIN"
    export OPENCLAW_BIN
    info "OpenClaw SLM : ${OPENCLAW_BIN}"
  else
    warn "Binaire openclaw@2026.5.12 non trouvé dans ${OPENCLAW_PATH}/npm"
  fi
fi
```

- [ ] **Step 3 : Rendre le template JSON profile-aware (bloc openclaw.json, ~ligne 96-99)**

Remplacer :
```bash
  envsubst '${HOME} ${HOSTNAME} ${OBSIDIAN_PATH} ${ASSISTANT_NAME} ${LLM_BACKEND} ${DEEPSEEK_API_KEY} ${OPENCLAW_GATEWAY_TOKEN} ${EURIA_API_KEY} ${EURIA_PRODUCT_ID}' \
    < "${SCRIPT_DIR}/openclaw.json.template" \
    > "$TARGET"
```

Par :
```bash
  _json_tpl="${SCRIPT_DIR}/openclaw.json.template"
  [[ "$PROFILE" == "slm" ]] && _json_tpl="${SCRIPT_DIR}/openclaw-slm.json.template"
  envsubst '${HOME} ${HOSTNAME} ${OBSIDIAN_PATH} ${ASSISTANT_NAME} ${LLM_BACKEND} ${DEEPSEEK_API_KEY} ${OPENCLAW_GATEWAY_TOKEN} ${EURIA_API_KEY} ${EURIA_PRODUCT_ID}' \
    < "$_json_tpl" > "$TARGET"
  unset _json_tpl
```

- [ ] **Step 4 : Rendre le template gateway env profile-aware (bloc gateway.systemd.env)**

Juste avant le bloc `# gateway.systemd.env` (la ligne `ENV_TARGET=...`), insérer :
```bash
_env_tpl="${SCRIPT_DIR}/gateway.systemd.env.template"
[[ "$PROFILE" == "slm" ]] && _env_tpl="${SCRIPT_DIR}/gateway-slm.systemd.env.template"
```

Puis remplacer les deux occurrences de `< "${SCRIPT_DIR}/gateway.systemd.env.template" \` par `< "$_env_tpl" \`.
Ajouter `unset _env_tpl` après le dernier `fi` du bloc gateway.systemd.env.

- [ ] **Step 5 : Rendre l'installation du service systemd profile-aware (bloc Service systemd user)**

Remplacer :
```bash
SERVICE_TARGET="${SYSTEMD_USER_DIR}/openclaw-gateway.service"
if [[ -f "$SERVICE_TARGET" && "$FORCE" != "true" ]]; then
  info "openclaw-gateway.service existe déjà — ignoré"
else
  envsubst '${OPENCLAW_BIN}' \
    < "${SCRIPT_DIR}/openclaw-gateway.service" \
    > "$SERVICE_TARGET"
  info "Service systemd installé dans ${SYSTEMD_USER_DIR} (${OPENCLAW_BIN})"
fi
```

Par :
```bash
_gw_svc="openclaw-gateway.service"
[[ "$PROFILE" == "slm" ]] && _gw_svc="openclaw-gateway-slm.service"
SERVICE_TARGET="${SYSTEMD_USER_DIR}/${_gw_svc}"
if [[ -f "$SERVICE_TARGET" && "$FORCE" != "true" ]]; then
  info "${_gw_svc} existe déjà — ignoré"
else
  envsubst '${OPENCLAW_BIN}' \
    < "${SCRIPT_DIR}/${_gw_svc}" \
    > "$SERVICE_TARGET"
  info "Service systemd installé dans ${SYSTEMD_USER_DIR} (${OPENCLAW_BIN})"
fi
unset _gw_svc
```

- [ ] **Step 6 : Rendre le workspace source profile-aware (bloc Workspace .md et skills)**

Remplacer :
```bash
WORKSPACE_SRC="${SCRIPT_DIR}/workspace"
```

Par :
```bash
WORKSPACE_SRC="${SCRIPT_DIR}/workspace"
[[ "$PROFILE" == "slm" ]] && WORKSPACE_SRC="${SCRIPT_DIR}/workspace-slm"
```

- [ ] **Step 7 : Ajouter deux gardes de saut pour les étapes non pertinentes en mode slm**

**Garde 1** — Juste avant le commentaire `# Services MCP SSE`, insérer :
```bash
if [[ "$PROFILE" != "slm" ]]; then
```
Fermer garde 1 juste avant le commentaire `# Workspace .md et skills` (qui doit s'exécuter pour les deux profils) :
```bash
fi  # end guard 1 (services MCP, Scout, injection-guard, Python deps)
```

Cette garde englobe : Services MCP SSE, router-mcp, slm-llama-cpp service, Scout, injection-guard, dépendances Python, openclaw-mcp-adapter, fastmcp, injection-guard restart.

**Garde 2** — Juste avant le commentaire `# Workspace scout`, insérer :
```bash
if [[ "$PROFILE" != "slm" ]]; then
```
Fermer garde 2 après la dernière ligne du bloc Docker image build :
```bash
fi  # end guard 2 (Scout workspace, wikilm workspace, Docker image)
```

Le bloc `# Workspace .md et skills` (workspace principal) reste en dehors des deux gardes : il s'exécute pour les deux profils, en utilisant `WORKSPACE_SRC` déjà positionné correctement en Step 6.

- [ ] **Step 8 : Ajouter le bloc de finalisation pour le profil slm (à la fin du fichier, après le fi fermant)**

```bash
# Finalisation instance slm
if [[ "$PROFILE" == "slm" ]]; then
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable openclaw-gateway-slm.service 2>/dev/null && \
    info "openclaw-gateway-slm.service activé" || \
    warn "Activation échouée — lancer manuellement"
  info "Installation slm terminée."
  info "Démarrer : systemctl --user start openclaw-gateway-slm.service"
  info "UI : http://localhost:18790"
fi
```

- [ ] **Step 9 : Vérifier la syntaxe bash**

```bash
bash -n ~/Secretarius/openclaw-config/install.sh && echo "Syntaxe OK"
```

Résultat attendu : `Syntaxe OK`

- [ ] **Step 10 : Commit**

```bash
cd ~/Secretarius
git add openclaw-config/install.sh
git commit -m "feat(slm): install.sh --profile slm (openclaw@2026.5.12, workspace-slm, service isolé)"
```

---

## Task 6 : Installation et démarrage de l'instance slm

*Prérequis : service slm-llama-cpp actif sur port 8998. Vérifier : `systemctl --user status slm-llama-cpp`.*

- [ ] **Step 1 : Vérifier que slm-llama-cpp tourne**

```bash
systemctl --user status slm-llama-cpp | head -5
curl -s http://127.0.0.1:8998/health 2>/dev/null && echo "OK" || echo "Service absent ou down"
```

Si le service est down : `systemctl --user start slm-llama-cpp` puis attendre 10s.

- [ ] **Step 2 : Lancer install.sh --profile slm**

```bash
cd ~/Secretarius
EURIA_API_KEY="$(grep EURIA_API_KEY ~/.openclaw/gateway.systemd.env | cut -d= -f2)" \
EURIA_PRODUCT_ID="$(grep EURIA_PRODUCT_ID ~/.openclaw/gateway.systemd.env | cut -d= -f2)" \
GOG_ACCOUNT="$(grep GOG_ACCOUNT ~/.openclaw/gateway.systemd.env | cut -d= -f2)" \
./openclaw-config/install.sh --profile slm
```

Résultat attendu :
```
[INFO] openclaw@2026.5.12 dans ~/.openclaw-slm/npm...
[INFO] openclaw.json généré dans ~/.openclaw-slm
[INFO] gateway.systemd.env généré (600)...
[INFO] openclaw-gateway-slm.service installé...
[INFO] workspace-slm/*.md installé
[INFO] openclaw-gateway-slm.service activé
[INFO] Démarrer : systemctl --user start openclaw-gateway-slm.service
```

- [ ] **Step 3 : Démarrer le gateway slm**

```bash
systemctl --user start openclaw-gateway-slm.service
sleep 5
systemctl --user status openclaw-gateway-slm.service | head -10
```

Résultat attendu : `Active: active (running)`

- [ ] **Step 4 : Vérifier que le gateway répond**

```bash
TOKEN=$(grep OPENCLAW_GATEWAY_TOKEN ~/.openclaw-slm/gateway.systemd.env | cut -d= -f2)
curl -s -H "Authorization: Bearer ${TOKEN}" http://localhost:18790/api/health 2>&1| head -5
```

Résultat attendu : réponse JSON `{"status":"ok"}` ou similaire.

- [ ] **Step 5 : Vérifier la version d'OpenClaw**

```bash
~/.openclaw-slm/npm/node_modules/.bin/openclaw --version
```

Résultat attendu : `2026.5.12` (ou `0.5.12` selon le format de version).

---

## Task 7 : Validation — Mesures et conversations

- [ ] **Step 1 : Lancer bench_prefill.py**

*Vérifier d'abord que les deux workspaces existent :*
```bash
ls ~/.openclaw/workspace/*.md 2>/dev/null | wc -l
ls ~/.openclaw-slm/workspace/*.md 2>/dev/null | wc -l
```
Les deux doivent retourner > 0.

```bash
cd ~/Secretarius
python3 slm/bench_prefill.py
```

Note : la mesure sur le prompt prod peut dépasser le TTL et expirer (`timeout=120`). Si c'est le cas, la sortie `TTFT prod (médiane)` sera absente ou l'exception `statistics.StatisticsError: no median for empty data` — c'est un résultat valide : le prompt prod est trop lourd pour phi-4-mini.

- [ ] **Step 2 : Ouvrir le gateway UI**

Ouvrir dans un navigateur : `http://localhost:18790`

S'authentifier avec le token de `~/.openclaw-slm/gateway.systemd.env`.

- [ ] **Step 3 : Checklist conversationnelle**

Envoyer ces 4 messages et noter le comportement :

1. `Bonjour, qui es-tu ?`
   - Attendu : réponse cohérente, phi-4-mini se présente, répond en français

2. `Lance : ls ~/Secretarius`
   - Attendu : résultat réel de ls, pas inventé

3. `Quelle heure est-il ?` (pas d'outil disponible pour ça)
   - Attendu : Tiron dit qu'il ne peut pas le savoir sans outil, ne fabrique pas de réponse

4. `Cherche quelque chose dans mon wiki`
   - Attendu : Tiron annonce que le wiki n'est pas disponible dans cette instance

- [ ] **Step 4 : Renseigner les résultats dans la spec**

Ouvrir `~/Secretarius/docs/superpowers/specs/2026-06-06-tiron-leger-design.md`, section 4.3 (tableau Résultats), et compléter :

| Mesure | Valeur |
|---|---|
| TTFT prompt prod (médiane) | mesure ou "TTL dépassé" |
| TTFT prompt léger (médiane) | valeur en secondes |
| Checklist UI (4/4 ?) | résultat |
| Verdict | "phi-4-mini viable / à retravailler" |

- [ ] **Step 5 : Commit final**

```bash
cd ~/Secretarius
git add docs/superpowers/specs/2026-06-06-tiron-leger-design.md
git commit -m "docs(spec): résultats validation Tiron léger"
```
