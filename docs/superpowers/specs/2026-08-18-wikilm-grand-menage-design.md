# Grand ménage Wiki_LM — design

## Contexte

Le pipeline d'ingestion Wiki_LM a accumulé quatre défauts distincts au fil du temps,
identifiés en mémoire projet et confirmés par exploration directe du dépôt et des
données réelles (`~/Documents/Arbath/Wiki_LM`) le 2026-08-18 :

1. **Traçabilité des sources** — plus de la moitié des pages `src-` n'ont pas l'URL
   d'origine dans leur frontmatter (`lien_source:` absent — **pas** `sources:`, un champ
   distinct qui sert aux références croisées entre pages, sans rapport avec la
   traçabilité). Un script dédié existe déjà (`patch_lien_source.py`) mais il est
   **cassé depuis la migration de structure** : il cherche les pages dans `wiki/*.md`
   au lieu de `wiki/sources/*.md` (0 correspondance, échec silencieux), et son chemin
   `raw/` par défaut pointe vers un dossier qui n'existe pas
   (`~/Secretarius/Wiki_LM/raw/` au lieu de `~/Documents/Arbath/Wiki_LM/raw/`).
2. **Dé-ingestion incomplète** — le retrait automatique d'une page (`_sync_deletions`
   dans `ingest.py`) ne met pas à jour `index.md`, `tags.md` ni `log.md`, et il n'existe
   aucune commande pour retirer une page à la demande.
3. **Fichiers bloqués** — 4 fichiers `.url.error` restent coincés dans `raw/` depuis
   mai/juin 2026, produits par `mcp_server.py`, un composant orphelin de l'ancienne
   approche MCP (abandonnée, commit `3693a57`).
4. **Front-matter corrompu** — 36 pages ont des listes de tags YAML mal formées
   (ex. `- '[documentation'` / `- secretarius]` au lieu de `tags: [documentation, secretarius]`).

## Chiffres réels (vérifiés, pas estimés)

- `wiki/sources/` (dossier vivant) : **87 pages `src-`**, dont **34** ont déjà
  `lien_source:` rempli et **53** ne l'ont pas.
- Sur ces 53 (chiffres obtenus en rejouant la logique réelle de `patch_lien_source.py`
  avec les bons chemins, pas une estimation) : **3 récupérables automatiquement**
  (le manifeste `.ingested` connaît le fichier `raw/` d'origine et ce fichier contient
  une URL extractable), **27** ont une entrée manifeste et un fichier `raw/` présent
  mais celui-ci ne contient aucune URL (notes texte manuelles, pas des captures d'URL),
  **23** sans aucune trace (ni manifeste, ni fichier).
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

### 1. Corriger `patch_lien_source.py` (pas un nouveau script)

- Corriger le chemin des pages : `wiki_dir.glob("src-*.md")` → chercher dans
  `wiki_dir / "sources"` (structure actuelle post-migration).
- Corriger `_DEFAULT_RAW` : `~/Secretarius/Wiki_LM/raw` (inexistant) →
  `~/Documents/Arbath/Wiki_LM/raw` (chemin réel, cohérent avec `WIKI_PATH`).
- Assouplir le filtre d'extraction : actuellement seuls les fichiers `raw/` au suffixe
  `.url` sont essayés ; tenter l'extraction sur tout fichier `raw/` associé (gain marginal
  mais gratuit, +1 page vérifié).
- Lancer en `--dry-run` puis `--apply` une fois vérifié → répare les 3 pages
  automatiquement récupérables.
- Pour les 50 pages restantes (27 sans URL extractable + 23 sans trace) : nouveau
  script `list_unsourced_pages.py` qui les liste dans
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
