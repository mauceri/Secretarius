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
