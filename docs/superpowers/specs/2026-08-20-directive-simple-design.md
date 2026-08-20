# Directive `@simple` pour les captures URL — design

## Contexte

Ce chantier est le premier de deux sous-projets liés à la capture de documents
Obsidian dans Wiki_LM :

1. **Câblage de `@simple`** (ce document) — un indicateur explicite, reconnu par
   `capture.py`/`ingest.py`, qui force le traitement verbatim (sans résumé LLM)
   d'une capture. Backend pur, indépendant d'Obsidian.
2. **Plugin Obsidian** (à venir, dépend de (1)) — capture la note courante et
   l'envoie en mode simple systématique (jamais de résumé LLM, puisque la source
   est déjà un texte rédigé par l'utilisateur).

`@simple` figurait déjà comme jeton de vocabulaire dans le corpus d'entraînement
du routeur d'intention (`docs/superpowers/specs/2026-06-30-gen-corpus-design.md`,
juin 2026 : `/c @simple ...` classé comme intention `wiki_capture`), mais n'était
câblé nulle part en aval — ni `capture.py` ni `ingest.py` ne le reconnaissaient.
Ce chantier l'implémente réellement.

## Périmètre

**Captures URL uniquement.** Deux autres cas ont été examinés et écartés :

- **Captures texte/note** (`capture_comment`, `capture_mixed`) : déjà
  **toujours** traitées en verbatim aujourd'hui — `ingest_raw_dir()` fixe
  `is_note = True` sur simple test de suffixe `.md`, inconditionnellement.
  `@simple` n'y aurait aucun effet ; il est accepté sans erreur mais ignoré.
- **Fichiers joints** (`--file`, .pdf/.txt/.html) : le mécanisme actuel de tags
  pour ce mode n'atteint déjà que la note `.md` compagnon, jamais le fichier
  joint lui-même (un binaire ne peut pas porter de ligne `tags:`/`simple:`
  inline). Câbler `@simple` proprement ici demanderait un nouveau mécanisme de
  fichier compagnon relu par association de nom — hors périmètre, aucun besoin
  actuel.

Seule la capture URL change réellement de comportement avec `@simple` : bascule
entre page résumée par le LLM (`_generate_source_page`, comportement actuel) et
page verbatim (`_generate_note_page`, déjà utilisé pour les notes locales).

## Conception

### `capture.py`

- Nouvelle fonction `_parse_simple_directive(text: str) -> tuple[bool, str]` :
  détecte le jeton `@simple` (insensible à la casse, limite de mot `\b`)
  n'importe où dans les arguments, le retire, retourne `(présent, texte_restant)`.
  Appelée dans `main()` avant `_parse_hashtags`, sur le même modèle.
- `capture_urls(urls, raw, tags=None, note=None, simple=False)` : si `simple`,
  ajoute une ligne `simple: true` au fichier `.url` créé, aux côtés des lignes
  `url:`/`tags:`/`note:` existantes.
- Captures texte/mixte : le drapeau est extrait par `_parse_simple_directive`
  mais non transmis à `capture_comment`/`capture_mixed` — aucun changement de
  signature pour ces fonctions, aucun effet si `@simple` y est tapé par erreur.

### `ingest.py`

- Nouvelle méthode statique `_parse_raw_simple(path: Path) -> bool` (parallèle
  à `_parse_raw_tags`) : lit un fichier `.url`, retourne `True` si une ligne
  `simple: true` est présente (insensible à la casse), sinon `False`.
- `ingest_raw_dir()`, branche `.url` : calcule
  `simple = self._parse_raw_simple(path)` et le transmet en `local_note=simple`
  à `self.ingest(...)` — paramètre déjà supporté par la signature de `ingest()`
  (ligne ~1128), simplement jamais alimenté pour les URLs jusqu'ici.
- `retry_blocked_urls.py` (ligne ~39) : même ajout, pour qu'une URL bloquée
  capturée avec `@simple` reste traitée en verbatim lors d'un nouvel essai.

### Documentation

- `openclaw-config/workspace/skills/c/SKILL.md` : mention de `@simple` dans la
  description d'usage de `/c`.

## Tests

- `test_capture.py` : `_parse_simple_directive` (présence/absence, insensible à
  la casse, position dans le texte) ; `capture_urls(..., simple=True)` écrit
  bien la ligne `simple: true`.
- `test_ingest.py` : `_parse_raw_simple` (présent/absent/valeur invalide) ;
  `ingest_raw_dir` transmet `local_note=True` pour une URL marquée simple, et
  `local_note=False` (comportement actuel inchangé) sinon.

## Vérification

- Suite de tests complète verte avant/après (`pytest`, comme les chantiers
  précédents).
- Test manuel bout en bout : `python capture.py "@simple https://exemple"`,
  puis `/ingest` (ou `ingest_raw_dir` direct) → vérifier que la page générée
  est verbatim (pas de section résumé LLM), par opposition à la même URL
  capturée sans `@simple`.

## Hors périmètre

- Câblage `@simple` pour les fichiers joints (`--file`) — chantier séparé, non
  demandé.
- Toute modification du corpus `gen_corpus`/réentraînement du routeur — le
  routeur classe déjà correctement `/c @simple ...` comme `wiki_capture`,
  aucune action nécessaire de ce côté.
- Le plugin Obsidian lui-même — sous-projet 2, spec à écrire séparément une
  fois ce câblage livré.
