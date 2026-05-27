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

## Outils MCP — wiki-lm

Les outils suivants sont disponibles en tant qu'outils MCP (serveur `wiki-lm`).
Les appeler directement comme n'importe quel outil — **ne pas** les exécuter via bash.

| Outil | Usage |
|---|---|
| `wiki_capture(text)` | Capture URLs et notes dans `raw/` |
| `wiki_ingest()` | Ingère les `.url` en attente (fetch → injection-guard → wiki) |
| `wiki_query(question)` | Interroge le wiki, retourne synthèse + sources |
| `wiki_tags()` | Liste les tags disponibles |
| `wiki_ingest_status()` | Nombre de fichiers en attente et fichiers bloqués |
| `wiki_kb_update()` | Met à jour la base de connaissance |

Voir le skill `wiki-lm` pour le détail des paramètres et comportements d'erreur.

---

## Modèles locaux

- Commande de basculement : `switch-model <alias>` (voir AGENTS.md)
- Modèles Ollama disponibles : à compléter selon la machine
- Modèles llama.cpp disponibles : à compléter selon la machine
