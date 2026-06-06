# Spec — Tiron léger (étape A : orchestrateur seul)

> Session superpowers du 2026-06-06.
> Périmètre : instance isolée OpenClaw 5.12, phi-4-mini orchestrateur,
> contexte ~600-800 tokens. Pas de sous-agents dans cette étape.

---

## 1. Contexte et objectif

Le prompt système de Tiron (prod 6.1) pèse ~11,7k tokens (wiki, gog, Scout,
router-mcp). Phi-4-mini est inutilisable à ce niveau : le prefill dépasse le TTL
de l'iGPU. L'objectif de cette étape est de valider que phi-4-mini devient
utilisable comme orchestrateur dès lors que ce contexte est réduit à ~600-800
tokens — sans encore construire les sous-agents.

Le bug OpenClaw #84059 (race condition sur l'annonce de sous-agent, introduit en
5.18) ne concerne pas cette étape (pas de délégation asynchrone). On travaille
néanmoins sur 5.12 car c'est la version de travail décidée pour toute
l'architecture déléguée.

---

## 2. Instance isolée slm

### 2.1 Séparation

OpenClaw `--profile slm` pointe vers `~/.openclaw-slm/`. La prod 6.1
(`~/.openclaw/`, port 18789) n'est pas touchée.

| Paramètre | Prod 6.1 | Instance slm |
|---|---|---|
| Répertoire config | `~/.openclaw/` | `~/.openclaw-slm/` |
| Port gateway | 18789 | 18790 |
| Version OpenClaw | 2026.6.1 | 2026.5.12 (épinglé) |
| Telegram | activé | désactivé |
| Tailscale | serve | désactivé |
| Modèle Tiron | Euria (Mistral Small 4) | phi-4-mini (port 8998) |
| Agents | main + scout | main seul |
| MCP | wiki-lm, gog, router-mcp | aucun |

### 2.2 Ressources partagées avec la prod

- Service `slm-llama-cpp` (port 8998, phi-4-mini, déjà actif) — aucune modification.
- Binaires `gog` et `switch-model` dans `$PATH` — inchangés.

### 2.3 Nouveaux fichiers versionnés

| Fichier | Rôle |
|---|---|
| `openclaw-config/openclaw-slm.json.template` | Config dédiée instance slm |
| `openclaw-config/openclaw-gateway-slm.service` | Service systemd instance slm |

`install.sh --profile slm` installe OpenClaw 5.12 dans `~/.openclaw-slm/` en
utilisant `openclaw-slm.json.template`. `install.sh` sans flag reste identique
(prod inchangée).

---

## 3. Prompt Tiron léger

### 3.1 Ce qui disparaît du workspace slm

- Section wiki (7 outils MCP + instructions d'usage)
- Section gog (outils MCP + politique de confirmation MCP)
- Section Scout / `sessions_spawn` (sous-agents non disponibles)
- Section router-mcp

### 3.2 Contenu cible (~300 tokens)

`AGENTS.md` (workspace `~/.openclaw-slm/workspace/`) :

```
Tiron est un orchestrateur léger. Il traite les demandes directement
via les outils exec disponibles. Les capacités wiki, gog et sources
externes seront déléguées à des sous-agents (non disponibles dans
cette instance de développement).

Outils exec disponibles : gog, switch-model, cat, ls, find.

Règles fondamentales :
- Zéro initiative : agir uniquement sur ce qui est demandé explicitement.
- Toute action qui écrit ou envoie hors machine requiert confirmation (OUI/NON).
- Ne jamais fabriquer une sortie de commande : exécuter via outil, coller la sortie réelle.

Routine de session : lire SOUL.md et USER.md avant de répondre au premier message.
```

`SOUL.md`, `USER.md`, `IDENTITY.md` — copiés tels quels depuis le workspace prod.

`TOOLS.md` — réduit aux bins exec restants, sections MCP supprimées.

### 3.3 Config openclaw-slm.json.template — différences clés

```json
"agents": {
  "defaults": { "model": { "primary": "tiron-llm/phi-4-mini-instruct" } },
  "list": [{ "id": "main" }]
},
"tools": {
  "exec": { "host": "gateway", "safeBins": ["gog","switch-model","cat","ls","find"] },
  "sandbox": {
    "tools": {
      "allow": ["gog","read","sessions_list","sessions_spawn","sessions_yield","group:runtime"],
      "deny": ["browser","canvas","nodes","cron","web_search","web_fetch","write","edit"]
    }
  }
},
"mcp": {},
"channels": { "telegram": { "enabled": false } },
"gateway": { "port": 18790 }
```

---

## 4. Validation

### 4.1 Mesure de prefill (bench_prefill.py)

Script Python minimal (`slm/bench_prefill.py`) — appelle l'API llama.cpp
(port 8998) avec le prompt prod et le prompt léger, mesure le TTFT (time-to-first-token)
sur 3 appels chacun, affiche la médiane.

Seuil qualitatif : TTFT prompt léger < 10 s sur l'iGPU. Toute valeur mesurable
sur le prompt prod est un bonus (actuellement le prefill dépasse le TTL).

### 4.2 Conversations via gateway UI

`http://localhost:18790` (controlUi localhost activé).

Checklist manuelle :
- [ ] Question de conversation générale (sans outil) → réponse cohérente
- [ ] `ls ~/Secretarius` via exec → sortie réelle, pas inventée
- [ ] `switch-model` vers un alias puis retour → gateway redémarre, Tiron confirme
- [ ] Demande impliquant wiki → Tiron annonce « non disponible » sans inventer

### 4.3 Livrable

Section « Résultats » à compléter dans ce document après exécution :

| Mesure | Valeur |
|---|---|
| TTFT prompt prod (médiane 3 appels) | — |
| TTFT prompt léger (médiane 3 appels) | — |
| Checklist UI (4/4 ?) | — |
| Verdict | — |

---

## 5. Fichiers à créer / modifier

| Action | Fichier |
|---|---|
| Créer | `openclaw-config/openclaw-slm.json.template` |
| Créer | `openclaw-config/openclaw-gateway-slm.service` |
| Créer | `openclaw-config/workspace-slm/AGENTS.md` |
| Créer | `openclaw-config/workspace-slm/TOOLS.md` |
| Copier | `workspace/SOUL.md`, `USER.md`, `IDENTITY.md` → `workspace-slm/` |
| Modifier | `install.sh` — flag `--profile slm` + épinglage `openclaw@2026.5.12` |
| Créer | `slm/bench_prefill.py` |

Le bloc `gateway.tailscale` est absent du template slm (omis = désactivé) plutôt
que `"mode": "off"` dont la validité n'est pas garantie par la doc OpenClaw.

La prod (`openclaw.json.template`, `workspace/`, `install.sh` sans flag) reste
inchangée.

---

## 6. Hors périmètre (étapes suivantes)

- Sous-agents wiki, gog, Scout
- Route_intent / BERT
- Basculement de la prod vers phi-4-mini
- Tests de charge ou benchmark GPU avancé
