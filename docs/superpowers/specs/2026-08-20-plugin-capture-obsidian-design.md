# Plugin Obsidian de capture de la note courante — design

## Contexte

Second des deux sous-projets liés à la capture de documents Obsidian dans
Wiki_LM (le premier, câblage de la directive `@simple`, est livré — voir
[[2026-08-20-directive-simple-design]]). Ce document couvre le plugin
Obsidian lui-même : un déclencheur dans Obsidian qui capture la note
actuellement ouverte vers la file `raw/` de Wiki_LM, au même titre qu'une
capture `/c` classique.

**Aucune dépendance sur `@simple`.** Ce plugin capture du texte (pas une
URL) : le fichier `.md` produit par `capture_comment()` est déjà toujours
ingéré en verbatim, sans résumé LLM, par la règle préexistante et
inconditionnelle de `ingest.py` (suffixe `.md` → `local_note=True`,
indépendante de tout marqueur `simple:`). Le sous-projet 1 ne concerne que
les captures URL.

## But

Capturer la note Obsidian actuellement ouverte comme une nouvelle entrée
dans `Wiki_LM/raw/`, pour ingestion ultérieure (`/ingest`) en page wiki
(`src-`/`c-`) — équivalent de `/c` mais pour une note déjà rédigée dans le
coffre, plutôt qu'une URL ou un texte tapé sur Telegram.

## Appareils cibles

Desktop et mobile/tablette. Le plugin appelle le serveur `wiki-lm-server`
existant (Flask, port 5051, déjà actif en permanence sur sanroque via
systemd) par `requestUrl` (API Obsidian — **jamais `fetch()`**, qui est
bloqué par le CSP d'Electron ; contrainte déjà documentée et éprouvée par le
template Templater existant, `docs/components/obsidian.md`). Fonctionne
donc identiquement, que le coffre local soit synchronisé ou non, y compris
depuis un appareil sans accès au système de fichiers de sanroque.

## Contenu capturé

1. Lire le corps de la note (hors frontmatter).
2. Si le premier titre Markdown après le frontmatter est un titre de
   **niveau 1 exactement** — `# Résumé` ou `# Summary` (insensible à la
   casse) — copier tout le contenu de cette section verbatim (jusqu'au
   titre suivant, quel que soit son niveau, ou la fin de la note).
3. Sinon, prendre les 200 premiers caractères du corps, coupés au dernier
   espace avant la limite (pas de coupure en milieu de mot), suffixés
   par « … ».
4. Préfixer le résultat par `Note d'origine : <titre> (<chemin dans le
   coffre>)\n\n` — traçabilité vers la note source, dans l'esprit de ce qui
   a été résolu lors du grand ménage Wiki_LM
   ([[project_wikilm_grand_menage_20260818]]).
5. Tags frontmatter existants de la note repris automatiquement comme tags
   de la capture.

## Déclenchement

Icône dans la barre latérale (ribbon) **et** commande enregistrée dans la
palette Obsidian (permet l'assignation d'un raccourci clavier) — les deux
pointent vers la même action.

## Comportement à l'exécution

1. Pas de note active → `Notice` « Aucune note ouverte », arrêt.
2. Construire le texte selon la règle « Contenu capturé » ci-dessus.
3. Si la note a déjà `wiki_capture:` en frontmatter → `Notice`
   d'avertissement (non bloquant), la capture se poursuit quand même.
4. `requestUrl` → `POST /capture` sur l'URL du serveur configurée dans les
   réglages du plugin.
5. Succès → marquer `wiki_capture: <horodatage ISO 8601>` sur la note via
   `processFrontMatter`, `Notice` de confirmation avec le nom du fichier créé.
6. Échec → `Notice` d'erreur (message du serveur ou de l'exception), la note
   n'est **pas** marquée.

## Réglages du plugin

Un seul champ : URL du serveur (ex. `http://sanroque:5051`).

## Serveur — `POST /capture`

Nouvel endpoint sur `wiki-lm-server` (`Wiki_LM/tools/server.py`), à côté de
`/query`, `/health`, `/reload`, etc. :

```
Body  : {"text": "...", "tags": ["..."]}
Reply : {"status": "ok", "filename": "20260820-143200-....md"}
```

- `text` vide → 400 (même convention que `/query`).
- `tags` passés à `_normalize_tags()` de `capture.py` (même canonicalisation
  que `/c` — actuellement appelée seulement côté CLI dans `main()`, pas
  encore côté serveur).
- Appelle `capture_comment(text, raw_dir, tags=tags or None)` sans
  modification de `capture.py`, sauf extraction d'un petit helper
  `raw_dir() -> Path` (résolution `WIKI_RAW_PATH`/`RAW_DEFAULT`, actuellement
  dupliquée dans `main()` — évite de la dupliquer une seconde fois côté
  serveur).
- Le texte reçu est déjà entièrement construit côté plugin (règle « Contenu
  capturé », préfixe de traçabilité inclus) — le serveur ne fait
  qu'écrire, il n'a pas besoin de `title`/`path` séparés.

## Plugin — structure (TypeScript, esbuild)

- `main.ts` : au chargement, enregistre l'icône ribbon et la commande,
  toutes deux vers `captureCurrentNote()`.
- Lecture du corps/frontmatter/tags via `app.metadataCache` (accès direct
  aux positions de titres et de frontmatter, pas de reparsing manuel).
- Un seul fichier de réglages (`serverUrl`).

## Tests

- **Serveur** (`Wiki_LM/tests/test_server.py`, nouveau, client de test
  Flask) : texte manquant → 400 ; texte + tags → fichier créé dans `raw/`
  avec le contenu attendu ; nom de fichier renvoyé correct.
- **`capture.py`** : test minimal du helper `raw_dir()` extrait, ajouté à
  `test_capture.py`.
- **Plugin (TypeScript)** : pas de harnais de test automatisé pour l'API
  Obsidian dans ce dépôt — vérification **manuelle** documentée (ouvrir une
  note avec et sans section `# Résumé`, déclencher la capture, vérifier le
  fichier `raw/` produit et le marquage frontmatter de la note source), pas
  de tests unitaires fictifs.

## Installation

- Build esbuild → `main.js` + `manifest.json` dans
  `~/Documents/Arbath/.obsidian/plugins/wikilm-capture/`, activé
  manuellement dans Obsidian (Paramètres → Modules communautaires) — pas de
  publication au store communautaire, usage personnel.
- Redémarrage de `wiki-lm-server` nécessaire après déploiement du nouvel
  endpoint (`systemctl --user restart wiki-lm-server` — confirmation requise
  au moment venu, règle déjà posée dans `CLAUDE.md`).
- Synchronisation multi-appareil du plugin lui-même (dossier
  `.obsidian/plugins/`) : dépend du réglage Obsidian Sync « inclure les
  plugins communautaires » — à vérifier à l'installation, pas une décision
  de conception.

## Hors périmètre

- Fichiers joints / pièces jointes dans la note (images, embeds) — texte
  brut uniquement, comme `capture_comment()` aujourd'hui.
- Toute modification d'`ingest.py` — le traitement verbatim des captures
  texte est déjà acquis, aucun changement nécessaire.
- Authentification sur l'endpoint `/capture` — suit le même modèle que les
  endpoints existants du serveur (`/query`, `/reload`, etc.), non
  authentifiés, protégés uniquement par la frontière réseau Tailscale.
