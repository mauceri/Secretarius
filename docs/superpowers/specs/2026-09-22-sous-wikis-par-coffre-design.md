# Sous-wikis par coffre — pages produites/citées visibles hors Secretarius

- Date : 2026-09-22
- Statut : design validé, en attente d'exécution
- Auteur : Christian Mauceri + Claude

## Problème

Le mécanisme d'historique multi-coffre (2026-09-21) rend la note d'historique
elle-même lisible depuis un coffre autre que Secretarius, mais ses liens
`[[slug]]` — vers les pages `src-`/`c-`/`e-` réellement citées — ne résolvent
nulle part hors de Secretarius : seul `historique/` est miroré, jamais
`wiki/`. Cliquer sur un tel lien depuis un autre coffre fait qu'Obsidian, ne
trouvant pas la page, en crée une **vide** sur place — silencieusement, sans
erreur visible.

Constat identique, indépendamment de l'historique : une page `src-` produite
par une capture+ingestion déclenchée depuis un coffre client cite en général
des concepts/entités (section « Concepts et entités mentionnés ») qui, eux
non plus, n'existent que dans Secretarius.

## Périmètre

**Dans le périmètre** : rendre visibles, dans un coffre client, les pages que
**ses propres requêtes ou ingestions ont produites ou citées** — un
sous-ensemble en lecture seule qui grossit avec l'usage de ce coffre.

**Hors périmètre, explicitement** :
- Capture/ingestion autonome depuis un coffre client (Secretarius reste le
  seul point d'ingestion — tranché le 22/09/2026).
- Expansion transitive : une page miroitée n'entraîne pas la copie des pages
  qu'*elle* cite à son tour au-delà du premier niveau produit par
  l'événement qui a déclenché la copie (une requête copie ce qu'elle cite,
  une ingestion copie ce qu'elle produit).
- Rafraîchissement d'une page déjà miroitée si elle est modifiée plus tard
  par une activité extérieure à ce coffre (piste B de la discussion,
  réconciliation périodique — accepté comme limitation connue, à traiter
  plus tard si elle pose un problème réel, pas anticipée).
- Résolution à la demande côté client (piste C) — écartée : surface
  supplémentaire côté API Obsidian, déjà source de fragilité cette semaine.
- Les 875 liens cassés et 128 frontmatter doublés trouvés par `lint.py` le
  21/09 — chantier de réparation séparé, non traité ici.
- Le régime de `/supprimer!` (pas d'essai à blanc, joignable sans
  authentification) et la reproductibilité de `WIKI_VAULT_MIRRORS` (aucun
  script d'installation ne le pose) — décisions en attente, non traitées ici.

## Architecture

Un seul mécanisme, **événementiel et brutal** (copie complète, sans fusion,
toujours écrasée par la version canonique courante) — pas de nouveau
processus de fond, pas de nouvelle surface de synchronisation. Deux
déclencheurs, une seule fonction de copie partagée.

### Fonction partagée (`wiki_paths.py`)

- `vault_mirrors() -> dict[str, Path]` — lit `WIKI_VAULT_MIRRORS`
  (`Wiki_LM/.env`, format `Nom1=chemin1,Nom2=chemin2`). Remplace la version
  actuellement privée à `query.py` (`_vault_mirrors()`), qui migre ici pour
  être partagée avec `ingest.py`.
- `mirror_page(wiki_root: Path, vault_name: str | None, slug: str) -> None`
  — si `vault_name` est fourni et connu, copie `<wiki_root>/wiki/<sous-
  dossier>/<slug>.md` vers `<miroir>/Wiki_LM/wiki/<sous-dossier>/<slug>.md`
  (`subdir_for_slug()`, déjà dans `wiki_paths.py` — même arborescence que le
  canonique). N'écrit rien si la page source n'existe pas, si le miroir
  résout vers le même chemin que le canonique, ou si `vault_name` est
  `None`/inconnu. Ne lève jamais (`except OSError: pass`, même principe que
  `_write_history`).

### Déclencheur 1 — requête (`/q`, Templater **et** lot)

Les deux points d'entrée de `/q` partagent déjà le même moteur
(`WikiQuery.query()`) mais divergent aujourd'hui sur `vault_name` :
- `POST /query` (modèle Templater) le porte depuis le 21/09.
- `/run` + `op_query()` (bloc `` ```wiki ``) ne le porte pas — `op_query()`
  appelle `WikiQuery(...).query(question)` sans l'argument.

Après le calcul de `result.references` dans `query.py`, pour chaque slug
cité : `mirror_page(self.wiki_root, vault_name, slug)`.

### Déclencheur 2 — capture puis ingestion (`/c`, Templater absent de ce flux)

La capture et l'ingestion sont séparées dans le temps ; l'ingestion traite
toute la file `raw/` d'un coup, potentiellement des captures de plusieurs
coffres mêlées. Chaque fichier `raw/` doit donc porter son coffre d'origine :

- `op_capture(text, vault_name=None)` (wiki.py) écrit une ligne
  `vault: <nom>` dans le fichier `raw/` créé — même emplacement que
  `tags:`/`simple:` (fichiers `.url`, dans `capture_urls()`) ou dans le
  frontmatter (fichiers `.md`, dans `_write_note()`).
- `ingest.py` relit cette ligne (nouveau `_parse_raw_vault(path)`, même
  mécanisme que `_parse_raw_tags`/`_parse_raw_simple`) dans
  `ingest_raw_dir()`, pour les deux branches (`.url` et note locale).
- `ingest()` trace les slugs concept/entité qu'il vient de créer/mettre à
  jour pendant cet appel précis, via un attribut d'instance
  (`self._last_related_slugs: list[str]`, réinitialisé en tête de méthode,
  peuplé au moment où `concepts`/`entities` sont calculés — `f"c-
  {_slugify(concept)}"` / `f"e-{_slugify(entity)}"`, déjà calculables sans
  toucher `_update_concept_page`/`_update_entity_page`). Aucun changement de
  signature de retour d'`ingest()` : les autres appelants (CLI, tests,
  `ingest_batch`) ne sont pas affectés.
- `ingest_raw_dir()`, après un appel `ingest()` réussi : si le fichier `raw/`
  traité portait un `vault:` connu, `mirror_page(...)` pour le slug `src-`
  produit **et** chacun des slugs de `self._last_related_slugs` — la page
  source et les concepts/entités qu'elle vient de produire arrivent
  ensemble, pour que cliquer un lien dans la page fraîchement capturée
  fonctionne dès le premier coup (pas seulement après une requête
  ultérieure qui les citerait).

### Plomberie commune aux deux déclencheurs en lot

`/run` (server.py) ne porte aujourd'hui aucune identité de coffre
(`{command, arg}`). Le plugin (`main.ts`) ajoute `vault_name:
app.vault.getName()` au corps POST — comme le modèle Templater le fait déjà
pour `/query`. `handle_run()` lit ce champ et l'ajoute à l'appel de chaque
entrée de `_RUN_OPS`, de façon uniforme (`op(arg, vault_name)`) : seules
`/c` et `/q` s'en servent, les autres l'ignorent — un seul protocole
d'appel dans la table de dispatch plutôt que deux arités différentes.

## Gestion des erreurs

- Coffre absent de la requête, ou absent de `WIKI_VAULT_MIRRORS` : aucune
  copie, comportement identique à aujourd'hui (pas de régression).
- Échec d'écriture (droits, disque, chemin invalide) : jamais remonté à
  l'appelant — la requête/capture/ingestion aboutit normalement même si la
  copie miroir échoue.
- Une page déjà présente dans un miroir est toujours écrasée par la version
  canonique courante — pas de fusion, pas de détection de divergence
  locale (le miroir est en lecture seule côté client, par construction).
- Le statut « immuable » (protection contre l'écrasement) ne s'applique
  qu'à l'écriture canonique ; une copie miroir est toujours remplacée.

## Tests

- `wiki_paths.py` : `vault_mirrors()` (parsing, déjà couvert indirectement
  aujourd'hui, migré), `mirror_page()` (copie réussie, coffre inconnu,
  coffre absent, échec d'écriture silencieux, page source absente, miroir
  résolvant vers le canonique).
- `query.py` : citations d'une requête copiées vers le miroir du coffre
  appelant (Templater **et** lot, une fois `op_query` corrigé).
- `capture.py` : `vault:` écrit dans le fichier `raw/` produit (`.url` et
  `.md`).
- `ingest.py` : `_parse_raw_vault()`, `self._last_related_slugs` peuplé
  correctement, `ingest_raw_dir()` pousse `src-` + concepts/entités vers le
  bon miroir.
- `server.py` : `handle_run()` transmet `vault_name` à chaque entrée de
  `_RUN_OPS` ; `/c` et `/q` en tiennent compte, les autres l'ignorent sans
  erreur.
- Plugin (`main.ts`/vitest) : corps POST de `/run` porte `vault_name`.

## Auto-critique de la spec

- Portée : un seul sujet (mirroring événementiel des pages produites/
  citées), pas mélangé avec la réparation des 875 liens cassés ni avec les
  deux décisions en attente (régime `/supprimer!`, reproductibilité de
  `WIKI_VAULT_MIRRORS`) — explicitement listées hors périmètre plutôt que
  passées sous silence.
- Cohérence : `mirror_page()` unique, utilisée par les deux déclencheurs —
  pas de logique de copie dupliquée entre `query.py` et `ingest.py`.
- Pas de nouvelle surface de synchronisation de fond — cohérent avec la
  leçon de la semaine (`ob sync --continuous` qui rate une édition sur
  place, deux fois en un jour).
