# Réparation du wiki à partir des rapports de lint

- Date : 2026-09-22
- Statut : design validé, en attente d'exécution
- Auteur : Christian Mauceri + Claude

## Problème

`lint.py`, réparé le 21/09/2026, rapporte deux familles d'erreurs réelles
et nombreuses sur le wiki en production (mesure du 22/09, 863 pages) :

- **890 liens cassés**, répartis sur 366 pages (jusqu'à 26 sur une seule) —
  cause connue (`ingest.py:~1302`) : `max_concepts` limite à 5 par famille
  les pages effectivement créées, mais le corps de la page garde toutes
  les citations produites par le modèle. Toute citation au-delà de la 5e
  devient un lien mort par construction, à chaque ingestion.
- **130 champs frontmatter manquants**, répartis sur 65 pages — un motif
  systématique : `---\n{}\n---` vide en tête, parfois suivi d'un second
  bloc frontmatter jamais reparse (YAML ne lit que le premier bloc en tête
  de fichier). Trois formes, en proportions variables : bloc bien formé
  mais mal placé ; bloc tronqué (génération interrompue) ; aucun second
  bloc, rien à récupérer.

`lint.py` ne fait que rapporter — aucune correction n'existe aujourd'hui.

## Décisions

**Liens cassés** — retirer les crochets au-delà de la 5e citation plutôt
que de les supprimer ou de créer une page par citation : `[[slug]]`
devient `slug` (texte brut). L'information reste lisible, le lien mort
disparaît. Choix retenu après vérification qu'aucune page du wiki n'est
actuellement orpheline (`lint.py`, code `orphan`, 0 occurrence sur 863
pages) — aucun risque qu'une citation en trop soit la seule voie d'accès
à une page existante.

**Frontmatter manquant** — deux traitements selon la forme :
- Second bloc bien formé mais mal placé (3 pages) : déplacement mécanique,
  aucun appel LLM.
- Bloc tronqué ou absent (le reste, 62 pages) : régénération par le LLM.
  Le champ `category` n'a pas besoin d'être deviné : il est déterministe,
  dérivé du sous-répertoire du fichier (`sources/` → `source`,
  `concepts/` → `concept`, `entités/` → `entité`) — seul `title` demande
  une vraie génération. `tags`/`sources` restent vides, `created` reprend
  la date de modification du fichier (aucune trace fiable de la date
  d'origine n'existe pour ces pages).

**Granularité et sélection** — pas de fichier de sélection ligne à ligne
(checklist Markdown envisagée puis écartée par l'utilisateur : trop de
lignes pour être réellement révisable — 890 cases pour les seuls liens
cassés). Réparation par famille entière, avec essai à blanc obligatoire
avant toute écriture réelle.

**Exposition** — coffre Obsidian uniquement (lot `` ```wiki ``, comme
`/supprimer?`/`/supprimer!`), pas Telegram.

## Architecture

### `WikiLint` (`lint.py`) — extension minime

`LintIssue` gagne un champ structuré optionnel :

```python
@dataclass
class LintIssue:
    level: str
    code: str
    slug: str
    message: str
    target: str = ""   # nouveau — slug cible pour code == "broken-link", vide sinon
```

`_check_links()` peuple `target=target` en plus de `message=...` — le
message texte ne change pas, `to_dict()`/`__str__()` restent identiques
(le nouveau champ n'apparaît dans aucun des deux). Sans ce champ,
`repair.py` devrait reparser le texte du message, fragile et couplé à son
libellé français.

### `WikiRepair` (nouveau, `Wiki_LM/tools/repair.py`)

Sur le modèle de `WikiLint` : `WikiRepair(wiki_path)`, une méthode par
famille, essai à blanc par défaut.

```python
class RepairReport:
    family: str
    dry_run: bool
    changes: list[str]      # une ligne lisible par changement (page + ce qui a changé)
    before_count: int        # nombre d'erreurs de cette famille avant
    after_count: int         # nombre d'erreurs de cette famille après (dry_run : simulé)
```

- `repair_broken_links(dry_run: bool = True) -> RepairReport` — lance
  `WikiLint().run()`, filtre les `LintIssue` de code `broken-link`, groupe
  par page. Pour chaque page concernée, remplace `[[target]]` par
  `target` (toutes les occurrences) dans le corps de la page. `dry_run`
  n'écrit rien, mais calcule quand même `after_count` en simulant le
  remplacement en mémoire, pour un rapport avant/après cohérent même à
  blanc.
- `repair_frontmatter(dry_run: bool = True) -> RepairReport` — même
  principe, sur les pages de code `missing-frontmatter`. Distingue les
  deux formes (bloc bien formé mal placé vs tronqué/absent) en relisant
  chaque fichier, applique le traitement correspondant.

Aucune des deux méthodes ne modifie l'autre famille — appelées
séparément, jamais ensemble dans un même appel.

CLI : `python repair.py --broken-links [--apply]` ou
`python repair.py --frontmatter [--apply]` (mutuellement exclusifs,
`--wiki` disponible comme `lint.py`, pour tester sur une copie du coffre
avant le vrai wiki).

### Exposition Obsidian (lot, `` ```wiki ``)

Trois commandes, sur le modèle de `/supprimer?`/`/supprimer!` :
- `/lint` — lecture seule, lance `WikiLint().run()`, retourne le rapport
  (comptage par famille).
- `/repair?` — essai à blanc pour une famille. `arg` = `broken-link` ou
  `missing-frontmatter` (mêmes chaînes que les codes de `lint.py`, pas un
  troisième vocabulaire à retenir).
- `/repair!` — applique réellement, même `arg`. **Un seul appel répare
  toute la famille d'un coup** (jusqu'à 366 pages pour les liens cassés) —
  rayon d'action nettement plus large qu'un `/supprimer!` (une page). Le
  `!` tapé dans la note tient lieu de confirmation, comme pour
  `/supprimer!`, mais ce point mérite d'être su explicitement plutôt
  qu'hérité — c'est noté ici, pas juste dans le code.

`server.py` (`_RUN_OPS`), `wiki.py` (`op_lint`, `op_repair_preview`,
`op_repair`) et le plugin (`SUPPORTED_COMMANDS`, `formatWikiResult`) sont
les trois surfaces touchées — même schéma que l'ajout de
`/supprimer?`/`/supprimer!` le 21/09.

## Méthode (héritée de la revue du 21/09, contraignante ici)

- Essai à blanc par défaut partout, écriture réelle seulement sur demande
  explicite (`--apply` en CLI, `!` en lot Obsidian).
- Mesurer avant/après avec `WikiLint` à chaque exécution, dry-run compris.
- Développable et testable sur une copie du coffre (`--wiki`), jamais
  directement sur le wiki servi pendant la mise au point.
- Une famille à la fois, jamais les deux dans le même appel.

## Hors périmètre

- Le fichier de sélection ligne à ligne (checklist Markdown) — envisagé,
  écarté par l'utilisateur.
- Les 22 « index-ghost » et 2 « unknown-category » trouvés par `lint.py`
  — familles distinctes, non traitées ici (l'index-ghost devrait se
  résorber de lui-même à la reconstruction de l'index, par constat du
  21/09).
- Régénérer `tags`/`sources`/`created` par LLM pour les pages à
  frontmatter régénéré — seul `title` est généré, le reste reste vide ou
  dérivé mécaniquement (voir Décisions).

## Tests

- `lint.py` : `LintIssue.target` peuplé pour `broken-link`, vide sinon ;
  `to_dict()`/`__str__()` inchangés (pas de régression sur les tests du
  21/09).
- `repair.py` : par famille — essai à blanc ne modifie rien sur disque
  mais rapporte un `after_count` correct ; application réelle modifie
  exactement les pages concernées et seulement elles ; les deux formes de
  frontmatter (bloc bien placé vs tronqué/absent) sont testées
  séparément ; `category` dérivé du sous-répertoire, jamais du LLM.
- `wiki.py`/`server.py`/plugin : mêmes gabarits de test que
  `/supprimer?`/`/supprimer!` (dispatch, essai à blanc vs application,
  coffre Obsidian uniquement).

## Auto-critique de la spec

- Portée : deux familles de réparation, traitées indépendamment,
  explicitement hors périmètre pour les deux autres familles trouvées par
  `lint.py` (index-ghost, unknown-category).
- Cohérence : le vocabulaire des familles (`broken-link`,
  `missing-frontmatter`) est le même partout — codes `lint.py`, flags CLI,
  argument des commandes en lot — pas de troisième nommage à traduire.
- Le changement à `LintIssue` est rétrocompatible (champ optionnel, valeur
  par défaut, aucune sortie existante modifiée) — pas de risque de casser
  les tests du 21/09.
