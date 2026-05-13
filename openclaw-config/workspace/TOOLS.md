# TOOLS.md — Notes sur l'environnement local

Ce fichier documente les spécificités de la configuration locale : hôtes, alias, comportements connus des outils.
**À personnaliser après installation.**

---

## Hôtes SSH

| Alias | Rôle |
|---|---|
| `${HOSTNAME}` | Machine locale — Ollama (port 11434) + llama.cpp (port 8989) |

---

## gog (client Google)

### Email — identifiant de fil vs identifiant de message

`gog email search <requête>` retourne des **identifiants de fil** — ne pas utiliser directement avec `gog email get`.

Pour lire le contenu d'un mail :

```bash
# 1. Récupérer l'identifiant de message (pas l'identifiant de fil)
gog email messages search "sujet ou expéditeur" -j --results-only

# 2. Lire avec l'identifiant de message (champ "id", pas "threadId")
gog email get <identifiantMessage>
```

---

## Modèles locaux

- Commande de basculement : `switch-model <alias>` (voir AGENTS.md)
- Modèles Ollama disponibles : à compléter selon la machine
- Modèles llama.cpp disponibles : à compléter selon la machine
