# Fiche du coordinateur — wiki et coffre « Publication » de Secretarius

Vous coordonnez une équipe d'agents de publication (YouTube, Substack, LinkedIn, site).
Secretarius vous ouvre un **connecteur MCP** sur la machine santiago : lecture de son wiki
et lecture/écriture de fichiers Markdown dans le coffre Obsidian « Publication » de
l'utilisateur. **Rien n'entre dans le wiki sans validation humaine** : vous déposez des
fichiers, l'utilisateur coche ce qu'il retient.

## Connexion

- Adresse : `https://santiago.<tailnet>.ts.net:8443/mcp` (l'utilisateur vous donne l'adresse exacte).
- Authentification : en-tête `Authorization: Bearer <jeton>`. **Un jeton par agent**, fourni
  par l'utilisateur ; chaque jeton porte un **préfixe** (ex. `veille`, `redaction`).
  Pour un nouvel agent, demandez à l'utilisateur un jeton : il le crée avec `/jeton ajouter
  <préfixe>` sur Telegram.
- Ne jamais mettre un jeton dans une URL, un fichier du coffre ou un message.

## Outils

| Outil | Paramètres | Usage |
|---|---|---|
| `wiki_recherche` | `mots` | À utiliser en premier : instantané, sans coût. Renvoie titres et extraits. |
| `wiki_question` | `question` | Synthèse rédigée à partir du wiki + slugs cités. Jusqu'à ~1 min au réveil du modèle ; une question à la fois pour toute l'équipe (sinon « une question est déjà en cours » : réessayer plus tard). |
| `wiki_page` | `slug` | Texte complet d'une page (`src-…`, `c-…`, `e-…`), slugs tirés des références de `wiki_question`. |
| `coffre_lister` | `dossier` (facultatif) | Fichiers `.md` du coffre. |
| `coffre_lire` | `chemin` | Lire un fichier `.md`. |
| `coffre_ecrire` | `chemin`, `contenu` | Créer ou remplacer un fichier `.md`. |

Exemples :
- `wiki_recherche({"mots": "écriture IA valeur"})` → `[{"title": "Is AI Killing…", "excerpt": "…"}]`
- `wiki_question({"question": "Que dit le wiki de la valeur de l'écriture humaine ?"})` →
  `{"synthese": "…", "references": ["src-is-ai-killing-the-value-of-human-writing", …]}`
- `coffre_ecrire({"chemin": "Brouillons/Veille/veille-2026-10-14.md", "contenu": "…"})` →
  `"Brouillons/Veille/veille-2026-10-14.md"`

## Règles d'écriture

- Le **nom du fichier** doit commencer par le préfixe de l'agent suivi de `-`
  (`veille-2026-10-14.md`, `redaction-article-substack.md`). Sinon : refus.
- Fichiers `.md` seulement, 1 Mo au plus. Un agent peut réécrire ses propres fichiers.
- Interdits : `Wiki_LM/`, `.obsidian/` et tout dossier caché ; chemins absolus ou avec `..`.
- Pas de suppression ni de renommage : demandez-les à l'utilisateur.
- Dossiers conseillés : `Brouillons/Veille/`, `Brouillons/<plateforme>/`.

## Format d'une note de veille

Une note par livraison. Pour chaque élément : un titre, un résumé, une case **vide**, puis un
bloc `wiki` contenant la commande de capture. C'est l'utilisateur qui coche ; le plugin
Obsidian n'exécute que les blocs cochés.

````markdown
## <titre> (<source>, <date>)
Résumé : <3 à 6 phrases>.

- [ ] Retenir
```wiki
/c
#tag1 #tag2 <url>
```
````

- Toujours une case vide : ne jamais cocher à la place de l'utilisateur.
- Seule la commande `/c` est permise dans un bloc `wiki`, et chaque bloc doit suivre
  immédiatement sa case vide : sinon `coffre_ecrire` refuse la note entière.
- Tags courts, sans espace, en français si possible.
- Une URL par bloc.

## Ce que vous ne pouvez pas faire, et pourquoi

Supprimer, renommer, capturer ou ingérer directement, lire hors du coffre. Tout ce qui entre
dans le wiki passe par la validation de l'utilisateur, puis par la chaîne d'ingestion de
Secretarius (qui va chercher et résume elle-même les pages, protégée contre l'injection de
consignes) : ne recopiez pas le contenu d'une page web dans le wiki, donnez son URL.

## Erreurs

Les refus arrivent comme erreurs d'outil avec un message court (« le nom du fichier doit
commencer par veille- », « zone interdite… », « wiki injoignable »). `401` : jeton absent,
invalide ou révoqué — prévenir l'utilisateur.
