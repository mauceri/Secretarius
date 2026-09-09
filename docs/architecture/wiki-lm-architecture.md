# Wiki_LM — architecture et fonctionnement

État des lieux du composant Wiki_LM de Secretarius : ce qui existe, comment
les pièces s'articulent, ce qui a été appris (et corrigé) en le construisant.
Complète [`llm-wiki-pattern.md`](llm-wiki-pattern.md) — le patron abstrait
dont ceci est une implémentation concrète — et le
[`docs/components/wiki-lm.md`](../components/wiki-lm.md) pratique (install,
variables d'environnement, commandes CLI). Ce document-ci est la carte du
territoire entre les deux.

## 1. Vue d'ensemble

Wiki_LM instancie le patron « wiki maintenu par un LLM » décrit dans
`llm-wiki-pattern.md` (lui-même une adaptation d'un texte d'Andrej Karpathy).
L'idée centrale : plutôt qu'un RAG qui redécouvre la connaissance à chaque
question, un LLM lit chaque nouvelle source une fois et la compile dans un
wiki persistant — pages de synthèse, pages de concepts, pages d'entités,
liens croisés. La question suivante ne relit pas les sources brutes, elle
interroge un artefact déjà digéré.

Ce que le patron laisse volontairement ouvert — structure exacte,
outillage, formats — a ici pris une forme précise, développée par
itérations sur plusieurs mois (avril à septembre 2026) :

- un **pipeline de capture** multi-source (URL, note libre, fichier), avec
  un point d'entrée unique côté utilisateur (`/c` sur Telegram, un plugin
  Obsidian, un CLI)
- une **ingestion** qui produit des pages structurées avec un schéma
  frontmatter fixe, extrait concepts et entités, tisse les liens
- une **recherche hybride** (BM25 + embeddings sémantiques) plutôt que le
  seul `index.md` suffisant à l'échelle envisagée par le patron abstrait
- une **base de connaissance** distincte du wiki lui-même : des « axes »
  thématiques obtenus par clustering, qui situent chaque nouvelle source
  dans une carte d'ensemble
- une **intégration OpenClaw** qui expose tout cela comme des commandes
  déterministes (`/c`, `/q`, `/ingest`…) plutôt que par jugement libre d'un
  agent conversationnel

État courant (2026-09-09) : 82 pages sources, 416 concepts, 284 entités,
85 axes de connaissance, un wiki vivant sur Obsidian synchronisé entre
sanroque et santiago.

## 2. Architecture générale

```
   /c (Telegram, Obsidian, CLI)
          │
          ▼
     raw/ (immuable)          ← fichiers .url / .md / .pdf, jamais modifiés
          │  /ingest
          ▼
     ingest.py ──────► LLM (local Qwen3-8B, repli Modal Qwen3-14B obfusqué)
          │
          ▼
  wiki/{sources,concepts,entités}/   ← pages Markdown, frontmatter + liens
          │
          ├──► embed.py ──► embeddings/ (BGE-M3)
          │                     │
          │                     ▼
          ├──► cluster.py ──► wiki/clusterings/ ──► kb_update.py ──► knowledge_base/axes/
          │
          ▼
     search.py (BM25 + sémantique) ──► query.py ──► historique/ (toute réponse, tout canal)
          │
          ▼
     server.py (Flask :5051) / wiki.py (CLI, sandbox) / derisk-deleg (OpenClaw)
```

Trois couches, comme dans le patron abstrait, mais chacune outillée :

- **`raw/`** — sources brutes, jamais modifiées après capture. Hors du
  coffre Obsidian (non synchronisé) : c'est un détail d'implémentation du
  pipeline, pas un artefact à consulter.
- **`wiki/`** — le wiki lui-même, dans le coffre Obsidian, synchronisé,
  lisible et modifiable à la main si besoin.
- **`knowledge_base/`** — une couche supplémentaire que le patron abstrait
  n'anticipait pas explicitement : une carte thématique du wiki, recalculée
  périodiquement, distincte du wiki qu'elle décrit.

## 3. Capture

Point d'entrée unique conceptuellement, deux surfaces concrètes :

- **`/c`** (Telegram, via `derisk-deleg` → `capture.py`) : URL seule,
  note libre, note + URL mêlées, `#tags`, `ref:<slug>` (lien vers une page
  existante), `file:<chemin>` (contenu d'un fichier local),
  `@simple` (contourne la synthèse LLM, écrit la note verbatim).
  Détail documenté dans `Interactions/pense-bête.md`.
- **Plugin Obsidian `wikilm-capture`** (`Wiki_LM/obsidian-wikilm-capture/`) :
  capture la note actuellement ouverte dans Obsidian, POST vers
  `/capture` sur `server.py`. Installé le 2026-09-09, jamais testé avant
  cette date faute d'accès graphique à Obsidian pour l'agent qui l'a écrit.

Toute capture (sauf `@simple`) atterrit dans `raw/` sous forme de fichier
`.url` (URL nue, ou `url:` + métadonnées) ou `.md` (note libre) —
**jamais traitée immédiatement**. `/ingest` est un geste séparé,
délibérément asynchrone : la capture est un réflexe (« je veux garder
ça »), l'ingestion une décision (« je veux que ça entre dans le wiki
maintenant »).

**Bug corrigé le 2026-09-09** : une capture texte+URL écrit un seul
fichier `.md` contenant l'URL en texte brut sur sa propre ligne.
`ingest_raw_dir()` traitait systématiquement tout `.md` comme note locale,
sans jamais relire cette ligne — l'URL d'origine (`lien_source`) était
perdue silencieusement à chaque capture mixte. `Ingestor._extract_embedded_url()`
la retrouve maintenant. Cinquante-deux pages sources historiques restent
sans `lien_source` récupérable (le fichier `raw/` correspondant a depuis
été nettoyé) ; six ont pu être réparées parce que l'URL, mal placée,
existait encore dans leur champ `sources:`.

## 4. Ingestion

`ingest.py` (module `Ingestor`) transforme une entrée de `raw/` en pages
du wiki. Pour chaque source :

1. **Génération de la page source** — le LLM lit le contenu (tronqué à
   12 000 caractères, `_truncate`), produit résumé, points clés, et une
   liste de concepts/entités mentionnés, au format
   `- concept: [[...]]` / `- entité: [[...]]`.
2. **Extraction et mise à jour des pages dérivées** — chaque concept/entité
   cité obtient ou met à jour sa propre page, avec une liste `sources:`
   pointant vers les sources qui la mentionnent (Wikipédia FR interrogé
   en complément si une entité y a une entrée).
3. **Indexation** — `index.md` et `tags.md` (ce dernier limité aux pages
   `src-`, les concepts/entités n'y figurant qu'« à rebond ») mis à jour ;
   entrée ajoutée à `log.md`.

**Deux régimes de calcul**, distincts de `WIKI_LLM_BACKEND` (utilisé par
`/q`) via `WIKI_INGEST_LLM_BACKEND` : en production, l'ingestion tourne en
CPU local sur Qwen3-8B (Ollama), avec un **repli automatique** sur
Qwen3-14B obfusqué (projet `~/obfuscator`, servi sur Modal) quand le local
échoue — ajouté le 2026-09-09 après qu'un texte particulièrement long a
mis en évidence un timeout systématique à 300 s côté Ollama CPU. Le repli
passe par un proxy local (`obfuscator-proxy-docker.service`,
`172.17.0.1:8001`, accessible depuis le sandbox Docker de l'agent wiki)
qui traduit le protocole OpenAI standard vers le service obfusqué —
aucun texte en clair ne quitte la machine vers Modal.

**Fragilités trouvées et corrigées en marge de cette session** (toutes
dans `ingest.py`/`cluster.py`, testées, poussées) :

- `_parse_frontmatter_block()` échouait silencieusement quand le LLM
  ouvrait un bloc `` ```yaml `` sans jamais le refermer — le frontmatter
  réel et le corps entier restaient piégés comme texte littéral. 146 pages
  réelles touchées (5 sources, 95 concepts, 46 entités), réparées.
- `_describe_cluster()` (génération de titre/description d'une grappe par
  le LLM, voir §7) demandait `max_tokens=200` — trop juste, réponse
  tronquée avant la fin dans jusqu'à la moitié des cas selon la verbosité
  du modèle. Porté à 600.
- 63 pages concepts/entités au contenu littéralement vide
  (`---\n{}\n---`, aucun titre ni corps) polluaient la recherche
  sémantique — 48 d'entre elles partageaient un seul et même vecteur
  d'embedding, remontant en tête de toute requête peu spécifique. Détruites
  proprement (`sync_deletions_full.py --remove`), leurs références
  retirées des 26 pages sources qui les citaient encore.

Suppression propre d'une page (source, concept ou entité) : pas de
commande `/`, un script (`sync_deletions_full.py --remove <slug>
--apply`), documenté dans `Interactions/pense-bête.md`.

## 5. Structure du wiki

```
wiki/
  sources/       src-<slug>.md
  concepts/      c-<slug>.md
  entités/       e-<slug>.md
  poubelle/      pages retirées (status: archivé, jamais détruites)
  clusterings/   partitions calculées par cluster.py (voir §7)
  index.md, tags.md, log.md
```

Schéma frontmatter d'une page source :

```yaml
---
category: source
created: 2026-09-09
lien_source: https://...        # optionnel — absent pour une note locale
sources: []                     # bibliographie CITÉE PAR ce document (rarement rempli)
tags: [...]
title: "..."
---
```

Le champ `sources:` change de sens selon la catégorie de page : sur une
page `src-`, il liste ce que le document cite lui-même ; sur une page
concept/entité, il liste les pages `src-` du wiki qui la mentionnent —
direction inverse, même nom de champ. Format hétérogène quand il est
rempli côté source (URL nue, identifiant arXiv, dictionnaire
`{titre,url}`) — non uniformisé, sans conséquence connue à ce jour.

Les liens `[[slug]]` sont la seule structure relationnelle explicite du
wiki — pas de base de données, pas de schéma de graphe séparé : Obsidian
les rend navigables et en tire sa vue graphe.

## 6. Recherche

`search.py` expose deux moteurs, combinables en mode hybride :

- **`WikiSearch`** (BM25, `rank_bm25`) — indexe titre + corps de chaque
  page, désuffixe en français (`nltk.stem.snowball.FrenchStemmer` depuis
  le 2026-09-09 : sans ça, « politicien » et « politiciens » étaient deux
  mots différents pour l'index). Cache disque (`wiki_bm25_cache.pkl`),
  invalidé par version + date de modification du répertoire.
- **`WikiSemanticSearch`** (BGE-M3, `sentence-transformers`) — vecteurs
  précalculés par `embed.py` (service systemd, toutes les 6 h,
  incrémental), stockés dans `$WIKI_PATH/embeddings/` (et non dans le
  dépôt de code — seul le coffre est monté dans le sandbox de l'agent).

Les deux excluent désormais les pages sans titre ni corps
(`wiki_paths.is_blank_page()`, ajouté le 2026-09-09 avec le nettoyage des
63 pages vides — l'exclusion protège contre toute récidive future, la
suppression n'a traité que l'existant).

**Bug corrigé le 2026-09-09** : `search.py` résolvait
`$WIKI_PATH/embeddings` tandis que `cluster.py`, `similarity.py`,
`kb_update.py` et `dedup.py` codaient en dur `<dépôt>/embeddings` —
un chemin qui n'existe même pas dans le sandbox, où seul le coffre est
monté. La recherche sémantique retombait donc silencieusement en BM25
seul depuis un temps indéterminé. `wiki_paths.embeddings_dir()`
centralise maintenant la résolution ; un test anti-dérive
(`test_embeddings_dir.py`) vérifie que les six modules pointent au même
endroit.

## 7. Interrogation

`query.py` synthétise une réponse à partir des résultats de recherche
(mode `hybrid` par défaut). Toute requête, quel que soit son origine
(Telegram, Obsidian, WebChat), produit un enregistrement horodaté dans
`wiki/historique/` — pas seulement celles explicitement sauvegardées.

**Deux régimes de restitution**, décidés par le canal d'origine
(`derisk-deleg/src/wiki-ops.ts`, décision architecturale du 2026-09-08) :

- **Obsidian et WebChat (« full »)** — la synthèse complète, rendue
  correctement en Markdown.
- **Telegram et tout canal non identifiable (« brief »)** — un court
  résumé plus le chemin (depuis la racine du coffre, en clair) vers la
  note d'historique. Un lien `obsidian://` cliquable a été tenté puis
  abandonné : la balise `<a>` bloquait la livraison Telegram sans erreur
  visible (4 min d'attente puis rien, contre 6 s en appel direct).

Le template Obsidian (`Templates/Wiki_LM.md`, Templater) reproduit le
régime « full » : interroge `server.py`, puis ouvre la note d'historique
produite dans un nouvel onglet plutôt que d'insérer la réponse dans la
note en cours — corrigeant un défaut où toute requête depuis Obsidian
écrasait le contexte de lecture en cours.

## 8. Base de connaissance et clustering

Distincte du wiki : `knowledge_base/` ne contient que des **axes**
(`axis-NNNN.md`) — régions thématiques du corpus, chacune avec titre,
description en prose, tags dominants, nombre de membres, cohésion. Le
reste (`embeddings/`, `tags/`, `index.md`) n'est que la machinerie pour
les interroger.

Chaîne de production, en trois maillons :

1. `embed.py` — plongements de toutes les pages
2. `cluster.py` — partitionne les pages **sources** uniquement (pas les
   concepts/entités) ; deux algorithmes disponibles, **transferts**
   retenu après qu'HDBSCAN a produit 76 % de bruit sur un premier essai
   (partition complète garantie, seuil θ auto-estimé, mise à jour
   incrémentale possible)
3. `kb_update.py` — transforme la partition en axes (LLM pour
   titre/description par grappe, fusion des axes trop proches,
   exclusion des grappes de moins de 3 membres)

À l'ingestion, la circulation s'inverse : `kb_query` situe chaque nouveau
document dans les trois axes les plus proches — la base de connaissance
sert de repère, elle ne se met pas à jour d'elle-même.

**Premier clustering du wiki vivant réalisé le 2026-09-09** (82 sources,
13 grappes, θ = 0,439, 9 axes créés, 1 fusionné avec un axe existant, 3
grappes trop petites exclues). Avant cette date, les 76 axes existants
dataient de mai et portaient sur `wiki_signets_05_2026`, une archive
distincte — la base de connaissance décrivait un corpus qui n'était déjà
plus celui interrogé. `run_transfers(..., dry_run=True)` mesure ensuite,
sans recalcul complet, si une nouvelle passe est justifiée
(`QUALITY_THRESHOLD = 0.20` de transferts proposés).

Détail complémentaire : `docs/components/base-connaissance.md`.

## 9. Intégration OpenClaw

`derisk-deleg` (plugin TypeScript) route les commandes `/` vers des
opérations wiki déterministes plutôt que de laisser un agent conversationnel
juger de l'action à prendre — cf. `docs/architecture/spec-architecture-par-intention.md`.
Chaque commande wiki (`/c`, `/ingest`, `/wikistatus`, `/q`, `/r`, `/tags`,
`/kbupdate`, `/source`) exécute `python3 wiki.py <op>` dans un sandbox
Docker dédié (`network: bridge`), qui monte le coffre (`/Wiki_LM`) et les
outils (`/wiki-tools`) depuis l'hôte — jamais de copie.

Piège opérationnel récurrent : le plugin **installé**
(`~/.openclaw/extensions/derisk-deleg`) est une copie indépendante du
dépôt, pas un lien — toute modification du code source exige une
réinstallation (`openclaw plugins install ... --force`) puis un
redémarrage de la gateway pour prendre effet.

## 10. État actuel et chantiers ouverts

**En place et éprouvé** : capture multi-source, ingestion avec repli
cloud confidentiel, recherche hybride, historique systématique des
requêtes, premier clustering du wiki vivant.

**Non résolu, noté pour mémoire** :

- `lint.py` — glob non récursif, ne trouve quasiment rien (1458 « erreurs »
  rapportées à tort lors du dernier audit) ; jamais corrigé.
- `kb_lint.py` — jamais écrit. Prévu pour détecter dérive des centroïdes,
  doublons d'axes, axes orphelins ; complémentaire de la mesure `dry_run`
  qui dit *qu'il y a* dérive sans dire *où*.
- **Inflation concepts/entités** — 416 + 284 = 700 pages dérivées pour 82
  sources (~8,5 par source), 5 % seulement partagées entre plusieurs
  sources. Projet distinct envisagé : utiliser le clustering pour
  regrouper ces pages, fusionner les doublons, corriger les liens des
  pages sources en conséquence.
- **Doublons de pages sources** — plusieurs captures indépendantes du même
  contenu externe ont produit deux pages distinctes (ex. le même tweet
  capturé à deux reprises, une fois par capture mixte qui a perdu son URL
  et échappé à la déduplication). Non systématiquement traité.
- **Ingérer la documentation du projet** dans le wiki, ou monter un wiki
  dédié à la documentation — évoqué, coût à évaluer, pas encore fait.
- **Généralisation multi-wiki** — le système entier suppose un wiki
  unique via `WIKI_PATH` (outils, sandbox, services systemd). Sujet
  architectural à part entière, pas encore brainstormé.

## Références

- Patron abstrait : `docs/architecture/llm-wiki-pattern.md`
- Référence pratique (install, config, CLI) : `docs/components/wiki-lm.md`
- Base de connaissance et clustering : `docs/components/base-connaissance.md`
- Architecture par intention (routage des commandes) :
  `docs/architecture/spec-architecture-par-intention.md`
- Aide-mémoire utilisateur : `Interactions/pense-bête.md`
