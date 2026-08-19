# Grand ménage Wiki_LM — design

## Contexte

Le pipeline d'ingestion Wiki_LM a accumulé quatre défauts distincts au fil du temps,
identifiés en mémoire projet et confirmés par exploration directe du dépôt et des
données réelles (`~/Documents/Arbath/Wiki_LM`) le 2026-08-18 :

1. **Traçabilité des sources** — la plupart des pages `src-` n'enregistrent pas l'URL
   d'origine dans leur frontmatter (`sources: []`).
2. **Dé-ingestion incomplète** — le retrait automatique d'une page (`_sync_deletions`
   dans `ingest.py`) ne met pas à jour `index.md`, `tags.md` ni `log.md`, et il n'existe
   aucune commande pour retirer une page à la demande.
3. **Fichiers bloqués** — 4 fichiers `.url.error` restent coincés dans `raw/` depuis
   mai/juin 2026, produits par `mcp_server.py`, un composant orphelin de l'ancienne
   approche MCP (abandonnée, commit `3693a57`).
4. **Front-matter corrompu** — 36 pages ont des listes de tags YAML mal formées
   (ex. `- '[documentation'` / `- secretarius]` au lieu de `tags: [documentation, secretarius]`).

## Chiffres réels (vérifiés, pas estimés)

- `wiki/sources/` (dossier vivant) : **87 pages `src-`**, dont **79** avec `sources: []`.
- Sur ces 79 : **46 récupérables** (le manifeste `.ingested` connaît le fichier `raw/`
  d'origine et ce fichier existe encore), **6** tracées dans le manifeste mais le fichier
  `raw/` a disparu, **27** sans aucune trace (ni manifeste, ni fichier).
- `wiki_signets_05_2026/` (1892 pages) est un dossier **legacy figé depuis le 12 mai 2026**
  (aucun fichier modifié depuis), probablement l'état d'avant la migration effectuée par
  `migrate_wiki_structure.py`. **Hors périmètre de ce chantier** — à vérifier séparément
  plus tard qu'aucun outil ne le référence encore par erreur, sans y toucher ici.
- 4 fichiers `.url.error` dans `raw/`, datés du 27 mai au 12 juin 2026.
- 36 pages avec le motif de tags corrompus (recherche `^- '\[` dans `sources/`,
  `concepts/`, `entités/`).

## Périmètre

Quatre scripts indépendants dans `Wiki_LM/tools/`, suivant le patron déjà en place
(`patch_src_slugs.py`, `patch_lien_source.py`) — pas d'outil consolidé, cohérence avec
le style existant du projet plutôt qu'une nouvelle abstraction pour un usage ponctuel.

Chaque script : `--dry-run` par défaut (affiche les changements sans écrire),
`--apply` pour écrire réellement. Avant toute écriture réelle, sauvegarde légère
(liste des fichiers touchés + leur contenu avant modification, pas un dump complet).

### 1. `fix_missing_sources.py`

- Parcourt les 87 pages de `wiki/sources/` (y compris les 8 déjà remplies, au cas où
  leur valeur serait incomplète).
- Pour chaque page : cherche une correspondance dans `.ingested` (manifeste TSV
  `filename\tslug\thash`). Si trouvée et le fichier `raw/` existe → lit l'URL (première
  ligne du fichier `.url`) et l'écrit dans `sources:`.
- Pour les pages sans correspondance récupérable (jusqu'à 33) : les liste dans
  `Wiki_LM/tools/urls_a_rechercher.md` — un tableau par page avec titre, tags, et les
  deux premières phrases du résumé, extraction déterministe depuis le frontmatter/contenu
  existant (pas d'appel LLM, pas de recherche web automatique — décision explicite pour
  éviter tout risque de faux positif sur une URL devinée à tort).

### 2. `sync_deletions_full.py`

- Réutilise (sans dupliquer) la détection déjà en place dans `_sync_deletions`
  (`ingest.py`) : fichiers `raw/` disparus → pages `src-`/`c-`/`e-` correspondantes
  déplacées vers `poubelle/`.
- Ajoute ce qui manque : retrait des liens vers ces pages dans `index.md`, retrait des
  entrées correspondantes dans `tags.md`, journalisation de l'opération dans `log.md`.
- Nouvelle option `--remove <slug>` : retrait manuel et délibéré d'une page précise
  (sans exiger la suppression préalable de son fichier `raw/`), déclenchant la même
  cascade de nettoyage.

### 3. `retry_blocked_urls.py`

- Relance l'ingestion des 4 URLs actuellement en `.url.error`, via le chemin
  d'ingestion actif (`ingest.py`), pas via `mcp_server.py`.
- En cas de nouvel échec : le fichier reste en `.url.error`, avec la raison de l'échec
  ajoutée en commentaire dans le fichier (actuellement silencieux — juste l'URL).
- Suppression de `mcp_server.py` du dépôt, dans un commit séparé, une fois les 4 URLs
  traitées (pas avant, pour garder une référence disponible si une relance échoue
  autrement qu'attendu).

### 4. `fix_corrupted_tags.py`

- Répare les 36 pages au motif de tags corrompu, en reconstituant la liste de tags
  correcte à partir des fragments existants.
- Corrige la cause racine dans le pipeline d'ingestion. Hypothèse de départ (à vérifier
  précisément en implémentation, pas figée ici) : confusion entre la frontmatter réelle
  et un bloc YAML présent dans le contenu capturé lui-même (observé sur une page dont le
  sujet — un outil de normalisation documentaire — contient un exemple YAML dans son
  propre contenu).

## Hors périmètre

- `wiki_signets_05_2026/` (dossier legacy, 1892 pages) — décision séparée à prendre plus
  tard.
- Récupération via sauvegardes restic pour les pages sans trace locale — le dépôt restic
  est sur un disque externe non branché au moment de ce design ; à tenter une autre fois.
- Recherche web automatique pour retrouver les URLs manquantes — explicitement écarté au
  profit du fichier de mots-clés à usage manuel.

## Vérification

- Après chaque script en mode `--dry-run` : relecture du résumé des changements proposés
  avant `--apply`.
- Après application : échantillon de pages modifiées relu manuellement dans Obsidian pour
  confirmer un rendu correct (frontmatter, tags, liens).
- Pour `retry_blocked_urls.py` : confirmation que les 4 pages produites (ou les erreurs
  mises à jour) apparaissent correctement dans `wiki/sources/`.
