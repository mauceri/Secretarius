# Design : Tiron v0.2.0 — LLM local (phi-4-mini-instruct)

**Date** : 2026-05-29
**Branche** : `v0.2.0-dev`
**Objectif** : Remplacer DeepSeek par phi-4-mini-instruct tournant localement sur sanroque, sans modifier la logique applicative de Tiron ni ses skills.

---

## Contexte

Tiron utilise actuellement DeepSeek (`deepseek/deepseek-chat`) comme LLM via l'API OpenClaw sur santiago. L'objectif de v0.2.0 est de supprimer cette dépendance cloud en faisant pointer OpenClaw vers un serveur llama.cpp local sur sanroque, accessible via Tailscale.

Le modèle `Phi-4-mini-instruct-Q6_K.gguf` est déjà présent dans `~/Modèles/` sur sanroque. llama.cpp expose une API OpenAI-compatible — aucune modification de la logique OpenClaw n'est requise, seul l'endpoint change.

Le serveur extracteur existant (port 8989, phi-4-mini LoRA Wikipedia FR) reste inchangé.

---

## Architecture

```
sanroque (GPU AMD gfx900, 30 Go RAM)
├── llama.cpp extracteur    :8989  (phi-4-mini LoRA, inchangé)
└── llama.cpp Tiron         :8990  (phi-4-mini-instruct, NOUVEAU)
        ↑ Tailscale
santiago (VPS Hetzner)
└── openclaw-gateway  →  openai/phi-4-mini-instruct @ sanroque:8990
```

Exposition : IP Tailscale de sanroque uniquement, pas sur l'internet public.

---

## Composants

### 1. Service systemd `tiron-llm.service` (sanroque)

Nouvelle unité systemd user, parallèle à l'extracteur existant.

Paramètres llama.cpp :

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `--model` | `~/Modèles/Phi-4-mini-instruct-Q6_K.gguf` | Modèle déjà disponible |
| `--port` | `8990` | Distinct du port extracteur (8989) |
| `--host` | IP Tailscale sanroque | Pas d'exposition publique |
| `--ctx-size` | `8192` | Suffisant pour orchestrateur, économise la VRAM |
| `--n-gpu-layers` | `99` | Toutes les couches sur GPU ROCm |
| `--chat-template` | `phi4` | Template spécifique phi-4 |
| `--api-key` | `<LLAMA_TIRON_API_KEY>` | Authentification minimale sur Tailscale |

Le service est géré sur sanroque — distinct du déploiement OpenClaw qui reste sur santiago.

### 2. Reconfiguration OpenClaw (santiago)

**`gateway.systemd.env`** (secrets, non versionnés) — deux variables ajoutées :
```
OPENAI_BASE_URL=http://<SANROQUE_TAILSCALE_IP>:8990/v1
OPENAI_API_KEY=<LLAMA_TIRON_API_KEY>
```

**`openclaw.json.template`** — modèle Tiron mis à jour :
```json
"model": "openai/phi-4-mini-instruct"
```
(remplace `deepseek/deepseek-chat`)

Le plugin `openai` est déjà présent dans la config — aucun ajout requis.

**`install.conf`** — nouvelle variable pour l'IP Tailscale de sanroque :
```bash
SANROQUE_TAILSCALE_IP="${SANROQUE_TAILSCALE_IP:-}"
```

### 3. Script `start-tiron-llm.sh` (sanroque)

Script de démarrage du serveur, sourcé par le service systemd. Lit `LLAMA_TIRON_API_KEY` et `SANROQUE_TAILSCALE_IP` depuis `~/.config/tiron-llm.env` (non versionné, même pattern que `gateway.systemd.env`).

---

## Flux de données

```
Utilisateur → Telegram → OpenClaw (santiago)
    → POST /v1/chat/completions (Tailscale)
        → llama.cpp Tiron (sanroque:8990)
            → phi-4-mini-instruct (GPU ROCm)
        ← réponse JSON OpenAI-compatible
    ← OpenClaw traite la réponse, appelle outils wiki si besoin
← Réponse Telegram
```

---

## Gestion des erreurs

- **sanroque injoignable** : OpenClaw retourne une erreur LLM à l'utilisateur. Tiron affiche un message d'indisponibilité. Pas de fallback automatique vers DeepSeek en v0.2.0 (à envisager en v0.2.1).
- **llama.cpp lent au démarrage** : le service systemd attend `After=network-online.target`, avec un `ExecStartPre=/bin/sleep 5` si nécessaire.
- **Contexte dépassé** : llama.cpp tronque silencieusement. Tiron a des contextes courts — 8192 tokens est conservateur et suffisant.

---

## Stratégie de test

1. **Connectivité** : `curl http://<sanroque-tailscale-ip>:8990/v1/models` depuis santiago
2. **Démarrage Tiron** : vérifier dans les logs OpenClaw que le modèle `openai/phi-4-mini-instruct` est chargé
3. **Workflows critiques** :
   - `wiki_capture` : capturer une URL, vérifier la création du fichier `.url`
   - `wiki_ingest` : lancer l'ingestion, vérifier `wiki_ingest_status`
   - `wiki_query` : poser une question, vérifier une réponse cohérente
4. **Évaluation qualitative** : 5-10 échanges réels, comparaison subjective vs DeepSeek

---

## Ce qui ne change pas

- Skills et workspace Tiron (`AGENTS.md`, `SOUL.md`, skills/)
- Scout et injection-guard
- Wiki_LM et ses outils MCP
- Le serveur extracteur llama.cpp (port 8989)
- Wiki_LM/.env et son `DEEPSEEK_API_KEY` (Wiki_LM utilise toujours DeepSeek pour l'ingestion en v0.2.0)

---

## Hors scope v0.2.0

- Fine-tuning LoRA de phi-4-mini pour Tiron (→ v0.2.1)
- Traduction des skills superpowers en français (→ v0.2.1)
- Remplacement de DeepSeek dans Wiki_LM/ingest (→ v0.2.x)
- Fallback automatique vers DeepSeek (→ v0.2.1)
- Scout BERT classification (→ v0.3.0)
