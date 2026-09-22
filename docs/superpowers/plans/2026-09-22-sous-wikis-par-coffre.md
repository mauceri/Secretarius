# Sous-wikis par coffre Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendre visibles, dans un coffre Obsidian autre que Secretarius, les pages (`src-`/`c-`/`e-`) que ses propres requêtes ou ingestions ont produites ou citées — un sous-ensemble local, en lecture seule, du wiki canonique.

**Architecture:** Copie brutale et événementielle, au moment de la requête (`/q`) ou de l'ingestion déclenchée par une capture (`/c`), vers le miroir local du coffre appelant, déjà configuré via `WIKI_VAULT_MIRRORS` (`Wiki_LM/.env`). Aucun processus de fond, aucune synchronisation périodique.

**Tech Stack:** Python 3.12 (Flask, pytest), TypeScript (plugin Obsidian, vitest/esbuild).

**Spec:** `docs/superpowers/specs/2026-09-22-sous-wikis-par-coffre-design.md`

## Global Constraints

- Copie toujours brutale : une page déjà présente dans un miroir est
  systématiquement écrasée par la version canonique courante — jamais de
  fusion, jamais de détection de divergence locale.
- Aucune écriture miroir ne doit jamais faire échouer l'opération
  principale (requête, capture, ingestion) : toute erreur d'écriture est
  interceptée et ignorée (`except OSError: pass`), même principe que
  `_write_history` existant.
- Un nom de coffre absent de la requête, ou absent de `WIKI_VAULT_MIRRORS`,
  ne déclenche aucune copie — comportement strictement identique à
  aujourd'hui (pas de régression).
- Pas d'expansion transitive : une page copiée n'entraîne jamais la copie
  des pages qu'elle cite à son tour.
- Périmètre étendu au-delà du texte littéral de la spec, décision prise en
  écrivant ce plan : `/capture` (bouton du ruban « Capturer dans Wiki_LM »,
  `handle_capture()`/`captureCurrentNote()`) est traité comme un second
  point d'entrée de capture équivalent à `/c` en lot — même mécanisme, même
  raison d'être (Déclencheur 2 de la spec), simplement un autre chemin
  client vers `capture_comment()`. La branche `@simple` d'`op_capture()`
  (écriture directe dans `wiki/sources/`, hors `raw/`, hors ingestion)
  reste **hors périmètre** : elle ne passe jamais par `ingest()`, donc
  jamais par le déclencheur d'ingestion décrit dans la spec.

---

### Task 1: `wiki_paths.py` — fonctions partagées de mirroring

**Files:**
- Modify: `Wiki_LM/tools/wiki_paths.py`
- Test: `Wiki_LM/tests/test_wiki_paths.py`

**Interfaces:**
- Produces: `vault_mirrors() -> dict[str, Path]`, `mirror_page(wiki_root: Path, vault_name: str | None, slug: str) -> None`

- [ ] **Step 1: Écrire les tests, en échec**

Ajouter à la fin de `Wiki_LM/tests/test_wiki_paths.py` :

```python
class TestVaultMirrors:
    def test_parses_single_entry(self, monkeypatch, tmp_path):
        from wiki_paths import vault_mirrors
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={tmp_path}")
        assert vault_mirrors() == {"Arbath": tmp_path}

    def test_parses_multiple_entries(self, monkeypatch, tmp_path):
        from wiki_paths import vault_mirrors
        a, b = tmp_path / "a", tmp_path / "b"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={a},Autre={b}")
        assert vault_mirrors() == {"Arbath": a, "Autre": b}

    def test_empty_when_unset(self, monkeypatch):
        from wiki_paths import vault_mirrors
        monkeypatch.delenv("WIKI_VAULT_MIRRORS", raising=False)
        assert vault_mirrors() == {}

    def test_ignores_malformed_pairs(self, monkeypatch, tmp_path):
        from wiki_paths import vault_mirrors
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"sans-egal,Arbath={tmp_path}")
        assert vault_mirrors() == {"Arbath": tmp_path}


class TestMirrorPage:
    def _make_wiki(self, tmp_path):
        wiki_root = tmp_path / "canonical"
        (wiki_root / "wiki" / "concepts").mkdir(parents=True)
        page = wiki_root / "wiki" / "concepts" / "c-x.md"
        page.write_text("---\ntitle: X\n---\n\nContenu.", encoding="utf-8")
        return wiki_root, page

    def test_copies_page_to_known_mirror(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, page = self._make_wiki(tmp_path)
        mirror = tmp_path / "mirror-vault"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")

        mirror_page(wiki_root, "Arbath", "c-x")

        dest = mirror / "Wiki_LM" / "wiki" / "concepts" / "c-x.md"
        assert dest.read_text(encoding="utf-8") == page.read_text(encoding="utf-8")

    def test_overwrites_existing_mirror_copy(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, page = self._make_wiki(tmp_path)
        mirror = tmp_path / "mirror-vault"
        dest_dir = mirror / "Wiki_LM" / "wiki" / "concepts"
        dest_dir.mkdir(parents=True)
        (dest_dir / "c-x.md").write_text("ancienne version", encoding="utf-8")
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")

        mirror_page(wiki_root, "Arbath", "c-x")

        assert (dest_dir / "c-x.md").read_text(encoding="utf-8") == page.read_text(encoding="utf-8")

    def test_noop_when_vault_name_is_none(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, _ = self._make_wiki(tmp_path)
        mirror = tmp_path / "mirror-vault"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")

        mirror_page(wiki_root, None, "c-x")

        assert not mirror.exists()

    def test_noop_when_vault_unknown(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, _ = self._make_wiki(tmp_path)
        monkeypatch.delenv("WIKI_VAULT_MIRRORS", raising=False)

        mirror_page(wiki_root, "Coffre inconnu", "c-x")
        # Ne lève pas — c'est le seul comportement observable ici.

    def test_noop_when_source_page_missing(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, _ = self._make_wiki(tmp_path)
        mirror = tmp_path / "mirror-vault"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")

        mirror_page(wiki_root, "Arbath", "c-absent")

        assert not (mirror / "Wiki_LM" / "wiki" / "concepts" / "c-absent.md").exists()

    def test_noop_when_mirror_resolves_to_canonical(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, page = self._make_wiki(tmp_path)
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Secretarius={wiki_root}")

        mirror_page(wiki_root, "Secretarius", "c-x")

        # Rien d'autre à vérifier que l'absence de doublon écrit hors de wiki_root :
        # aucun répertoire "Wiki_LM" imbriqué ne doit apparaître sous wiki_root.
        assert not (wiki_root / "Wiki_LM").exists()

    def test_write_failure_is_silently_ignored(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, _ = self._make_wiki(tmp_path)
        mirror = tmp_path / "mirror-vault"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")

        def boom(*a, **k):
            raise OSError("disque plein")

        monkeypatch.setattr(Path, "write_text", boom)
        mirror_page(wiki_root, "Arbath", "c-x")  # ne lève pas
```

Ajouter `from pathlib import Path` en haut du fichier si absent (déjà présent).

- [ ] **Step 2: Vérifier l'échec**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_wiki_paths.py -k "VaultMirrors or MirrorPage" -v`
Expected: FAIL — `ImportError: cannot import name 'vault_mirrors'`

- [ ] **Step 3: Implémenter**

Dans `Wiki_LM/tools/wiki_paths.py`, ajouter à la fin du fichier :

```python
def vault_mirrors() -> dict[str, Path]:
    """Coffres Obsidian, autres que le canonique, dont ce serveur tient un
    miroir local (via ob sync --continuous). Format :
    WIKI_VAULT_MIRRORS=Nom1=chemin1,Nom2=chemin2."""
    raw = os.environ.get("WIKI_VAULT_MIRRORS", "")
    mirrors: dict[str, Path] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, _, path = pair.partition("=")
        name, path = name.strip(), path.strip()
        if name and path:
            mirrors[name] = Path(path).expanduser()
    return mirrors


def mirror_page(wiki_root: Path, vault_name: str | None, slug: str) -> None:
    """Copie la page `slug` vers le miroir du coffre `vault_name`, si connu
    et différent du canonique. Copie brutale : écrase toujours la version
    déjà présente. Ne lève jamais — une erreur d'écriture ne doit jamais
    faire échouer l'appelant (requête ou ingestion)."""
    if not vault_name:
        return
    mirror = vault_mirrors().get(vault_name)
    if mirror is None:
        return
    src = find_page(wiki_root / "wiki", slug)
    if src is None:
        return
    try:
        dest = mirror.resolve() / "Wiki_LM" / "wiki" / subdir_for_slug(slug) / f"{slug}.md"
        if dest.resolve() == src.resolve():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        pass
```

- [ ] **Step 4: Vérifier le succès**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_wiki_paths.py -v`
Expected: tous PASS (les tests existants + les nouveaux)

- [ ] **Step 5: Commit**

```bash
git add Wiki_LM/tools/wiki_paths.py Wiki_LM/tests/test_wiki_paths.py
git commit -m "feat(wiki): vault_mirrors()/mirror_page() partagés dans wiki_paths.py"
```

---

### Task 2: `query.py` — migrer vers le helper partagé, miroiter les citations

**Files:**
- Modify: `Wiki_LM/tools/query.py`
- Test: `Wiki_LM/tests/test_query.py`

**Interfaces:**
- Consumes: `wiki_paths.vault_mirrors()`, `wiki_paths.mirror_page(wiki_root, vault_name, slug)` (Task 1)
- Produces: `WikiQuery.query()` continue de fonctionner à l'identique pour l'appelant ; `result.references` sont en plus copiées vers le miroir du coffre appelant.

- [ ] **Step 1: Écrire le test, en échec**

Ajouter à `Wiki_LM/tests/test_query.py`, après les tests existants sur `vault_name` :

```python
def test_query_mirrors_cited_pages_to_calling_vault(tmp_path, monkeypatch):
    other_vault = tmp_path / "other-vault"
    monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={other_vault}")
    wq = _make_query(tmp_path)  # crée tmp_path/wiki/c-test.md (à plat, pour son propre stub de recherche)
    # mirror_page() résout le slug via la vraie arborescence (wiki/concepts/,
    # subdir_for_slug) — la fixture _make_query écrit à plat pour son propre
    # usage ; la page doit aussi exister au bon sous-dossier pour que la
    # copie miroir la trouve.
    concepts_dir = tmp_path / "wiki" / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / "c-test.md").write_text("Contenu de test.", encoding="utf-8")

    wq.query("Question test ?", vault_name="Arbath")  # synthèse stub cite [[c-test]]

    mirrored = other_vault / "Wiki_LM" / "wiki" / "concepts" / "c-test.md"
    assert mirrored.read_text(encoding="utf-8") == "Contenu de test."


def test_query_does_not_mirror_cited_pages_without_vault_name(tmp_path, monkeypatch):
    monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={tmp_path / 'other-vault'}")
    wq = _make_query(tmp_path)
    concepts_dir = tmp_path / "wiki" / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / "c-test.md").write_text("Contenu de test.", encoding="utf-8")

    wq.query("Question test ?")

    assert not (tmp_path / "other-vault").exists()
```

- [ ] **Step 2: Vérifier l'échec**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_query.py -k mirrors_cited -v`
Expected: FAIL — le fichier miroité n'existe pas encore (`mirror_page` n'est pas encore appelée depuis `_finalize`)

- [ ] **Step 3: Implémenter**

Dans `Wiki_LM/tools/query.py` :

1. Modifier l'import en tête de fichier — remplacer :
```python
from capture import slugify, timestamp
from llm import LLM
from search import WikiSearch, WikiSemanticSearch, hybrid_search
```
par :
```python
from capture import slugify, timestamp
from llm import LLM
from search import WikiSearch, WikiSemanticSearch, hybrid_search
from wiki_paths import mirror_page, vault_mirrors
```

2. Supprimer entièrement la fonction `_vault_mirrors()` (section « Coffres Obsidian miroités localement (historique multi-coffre) », lignes ~118-135) — elle est remplacée par `wiki_paths.vault_mirrors()`.

3. Dans `_write_history()`, remplacer l'appel `mirror = _vault_mirrors().get(vault_name)` par `mirror = vault_mirrors().get(vault_name)` (le reste de la fonction ne change pas).

4. Dans `_finalize()`, actuellement :
```python
    def _finalize(self, result: QueryResult, vault_name: str | None = None) -> QueryResult:
        """Historique + brief systématiques — toute réponse, y compris
        "aucune page pertinente trouvée", doit produire un enregistrement."""
        result.history_slug = self._write_history(result.question, str(result), vault_name)
        result.brief = self._generate_brief(result.question, result.text)
        return result
```
remplacer par :
```python
    def _finalize(self, result: QueryResult, vault_name: str | None = None) -> QueryResult:
        """Historique + brief systématiques — toute réponse, y compris
        "aucune page pertinente trouvée", doit produire un enregistrement.
        Les pages citées sont en plus copiées vers le miroir du coffre
        appelant, s'il est connu (sous-wikis par coffre, 2026-09-22)."""
        result.history_slug = self._write_history(result.question, str(result), vault_name)
        for slug in result.references:
            mirror_page(self.wiki_root, vault_name, slug)
        result.brief = self._generate_brief(result.question, result.text)
        return result
```

- [ ] **Step 4: Vérifier le succès**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_query.py -v`
Expected: tous PASS

- [ ] **Step 5: Commit**

```bash
git add Wiki_LM/tools/query.py Wiki_LM/tests/test_query.py
git commit -m "feat(wiki): /q copie les pages citées vers le miroir du coffre appelant"
```

---

### Task 3: `capture.py` — tracer le coffre d'origine dans les fichiers `raw/`

**Files:**
- Modify: `Wiki_LM/tools/capture.py`
- Test: `Wiki_LM/tests/test_capture.py`

**Interfaces:**
- Produces: `capture_urls(..., vault_name: str | None = None)`, `_write_note(..., vault_name: str | None = None)`, `capture_comment(..., vault_name: str | None = None)` — tous rétro-compatibles (nouveau paramètre optionnel, défaut `None`, en fin de signature).

- [ ] **Step 1: Écrire les tests, en échec**

Ajouter à `Wiki_LM/tests/test_capture.py`, dans `TestCaptureUrls` :

```python
    def test_vault_name_written(self, tmp_path):
        files = capture_urls(["https://example.com"], tmp_path, vault_name="Arbath")
        assert "vault: Arbath" in files[0].read_text()

    def test_vault_name_absent_by_default(self, tmp_path):
        files = capture_urls(["https://example.com"], tmp_path)
        assert "vault:" not in files[0].read_text()
```

Et dans `TestCaptureComment` :

```python
    def test_vault_name_written_in_frontmatter(self, tmp_path):
        path = capture_comment("Une note", tmp_path, vault_name="Arbath")
        content = path.read_text(encoding="utf-8")
        assert "vault: Arbath" in content
        assert content.startswith("---\n")

    def test_vault_name_absent_by_default(self, tmp_path):
        path = capture_comment("Une note", tmp_path)
        assert "vault:" not in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Vérifier l'échec**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_capture.py -k vault_name -v`
Expected: FAIL — `TypeError: capture_urls() got an unexpected keyword argument 'vault_name'`

- [ ] **Step 3: Implémenter**

Dans `Wiki_LM/tools/capture.py` :

1. `capture_urls` (ligne 163) — signature actuelle :
```python
def capture_urls(urls: list[str], raw: Path, tags: list[str] | None = None,
                 note: str | None = None, simple: bool = False) -> list[Path]:
```
remplacer par :
```python
def capture_urls(urls: list[str], raw: Path, tags: list[str] | None = None,
                 note: str | None = None, simple: bool = False,
                 vault_name: str | None = None) -> list[Path]:
```
et, dans le corps, après le bloc :
```python
        if note and not created:            # note attachée au premier .url créé
            content += f"note: {note}\n"
```
ajouter :
```python
        if vault_name:
            content += f"vault: {vault_name}\n"
```

2. `_write_note` (ligne 191) — signature actuelle :
```python
def _write_note(path: Path, text: str, tags: list[str] | None, refs: list[str] | None, wiki_root: Path) -> None:
```
remplacer par :
```python
def _write_note(path: Path, text: str, tags: list[str] | None, refs: list[str] | None, wiki_root: Path,
                 vault_name: str | None = None) -> None:
```
puis, dans le corps :
```python
    if tags or refs:
        fm = "---\n"
        if tags:
            fm += f"tags: [{', '.join(tags)}]\n"
        if refs:
            fm += (f"ref: {refs[0]}\n" if len(refs) == 1
                   else "refs:\n" + "".join(f"  - {r}\n" for r in refs))
        fm += "---\n"
        content = fm + body + "\n"
    else:
        content = (body + "\n") if body else "\n"
```
remplacer par :
```python
    if tags or refs or vault_name:
        fm = "---\n"
        if tags:
            fm += f"tags: [{', '.join(tags)}]\n"
        if refs:
            fm += (f"ref: {refs[0]}\n" if len(refs) == 1
                   else "refs:\n" + "".join(f"  - {r}\n" for r in refs))
        if vault_name:
            fm += f"vault: {vault_name}\n"
        fm += "---\n"
        content = fm + body + "\n"
    else:
        content = (body + "\n") if body else "\n"
```

3. `capture_comment` (ligne 214) — signature actuelle :
```python
def capture_comment(text: str, raw: Path, tags: list[str] | None = None, refs: list[str] | None = None,
                    title: str | None = None) -> Path:
    ts = timestamp()
    slug = slugify(title) if title else slugify(text)
    fname = f"{ts}-{slug}.md"
    path = raw / fname
    _write_note(path, text, tags, refs, raw.parent)
    return path
```
remplacer par :
```python
def capture_comment(text: str, raw: Path, tags: list[str] | None = None, refs: list[str] | None = None,
                    title: str | None = None, vault_name: str | None = None) -> Path:
    ts = timestamp()
    slug = slugify(title) if title else slugify(text)
    fname = f"{ts}-{slug}.md"
    path = raw / fname
    _write_note(path, text, tags, refs, raw.parent, vault_name)
    return path
```

- [ ] **Step 4: Vérifier le succès**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_capture.py -v`
Expected: tous PASS

- [ ] **Step 5: Commit**

```bash
git add Wiki_LM/tools/capture.py Wiki_LM/tests/test_capture.py
git commit -m "feat(wiki): capture_urls/capture_comment tracent le coffre d'origine (vault:)"
```

---

### Task 4: `ingest.py` — pousser vers le miroir après ingestion

**Files:**
- Modify: `Wiki_LM/tools/ingest.py`
- Test: `Wiki_LM/tests/test_ingest.py`

**Interfaces:**
- Consumes: `wiki_paths.mirror_page(wiki_root, vault_name, slug)` (Task 1)
- Produces: `Ingestor._parse_raw_vault(path) -> str | None` (staticmethod, même famille que `_parse_raw_tags`/`_parse_raw_simple`) ; `Ingestor._last_related_slugs: list[str]` (attribut d'instance, peuplé par chaque appel à `ingest()`).

- [ ] **Step 1: Écrire les tests, en échec**

Ajouter à `Wiki_LM/tests/test_ingest.py` :

```python
class TestParseRawVault:
    def test_reads_vault_line(self, tmp_path: Path):
        from ingest import Ingestor
        f = tmp_path / "test.url"
        f.write_text("https://example.com\nvault: Arbath\n", encoding="utf-8")
        assert Ingestor._parse_raw_vault(f) == "Arbath"

    def test_none_when_absent(self, tmp_path: Path):
        from ingest import Ingestor
        f = tmp_path / "test.url"
        f.write_text("https://example.com\n", encoding="utf-8")
        assert Ingestor._parse_raw_vault(f) is None


class TestIngestTracksRelatedSlugs:
    def test_last_related_slugs_populated_with_concepts_and_entities(
        self, ingestor, wiki_dir, tmp_path
    ):
        note = tmp_path / "note.md"
        note.write_text("Note sur Vannevar Bush.", encoding="utf-8")
        # mock_llm (fixture ingestor) répond, pour une note personnelle :
        # "TITRE: Ma note de test\n- concept: zettelkasten\n- entité: Vannevar Bush\n"
        ingestor.ingest(str(note), local_note=True)

        assert ingestor._last_related_slugs == ["c-zettelkasten", "e-vannevar-bush"]

    def test_last_related_slugs_reset_on_stub_page(self, ingestor, wiki_dir, tmp_path):
        # Contenu binaire → page stub, retour avant le calcul concepts/entités.
        binary = tmp_path / "bin.pdf"
        binary.write_bytes(b"\x00\x01\x02\x03binaire")
        ingestor.ingest(str(binary))

        assert ingestor._last_related_slugs == []


class TestIngestRawDirMirrorsToVault:
    def test_pushes_source_and_related_pages_to_known_mirror(
        self, ingestor, wiki_dir, raw_dir, monkeypatch, tmp_path
    ):
        from wiki_paths import mirror_page
        mirror = tmp_path / "mirror-vault"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")
        (raw_dir / "note.md").write_text(
            "---\nvault: Arbath\n---\n\nNote sur Vannevar Bush.", encoding="utf-8"
        )

        ingestor.ingest_raw_dir()

        pages = list((wiki_dir / "sources").glob("src-*.md"))
        assert pages, "aucune page source créée"
        mirrored_src = mirror / "Wiki_LM" / "wiki" / "sources" / pages[0].name
        assert mirrored_src.exists()
        mirrored_entity = mirror / "Wiki_LM" / "wiki" / "entités" / "e-vannevar-bush.md"
        assert mirrored_entity.exists()

    def test_no_mirror_push_without_vault_line(
        self, ingestor, wiki_dir, raw_dir, monkeypatch, tmp_path
    ):
        mirror = tmp_path / "mirror-vault"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")
        (raw_dir / "note.md").write_text("Note sans coffre.", encoding="utf-8")

        ingestor.ingest_raw_dir()

        assert not mirror.exists()
```

Confirmé dans `Wiki_LM/tests/conftest.py` : `MockLLM.complete()` renvoie,
pour tout prompt contenant « note personnelle » (le gabarit `_PROMPT_NOTE_ITEMS`
de `_generate_note_page` commence par cette phrase), exactement
`"TITRE: Ma note de test\n- concept: zettelkasten\n- entité: Vannevar Bush\n"` —
d'où `_last_related_slugs == ["c-zettelkasten", "e-vannevar-bush"]` attendu
ci-dessus (même slug d'entité que `test_note_creates_entity_link`, déjà dans
`TestIngestLocalNote` du même fichier, qui vérifie `e-vannevar-bush.md`).

- [ ] **Step 2: Vérifier l'échec**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_ingest.py -k "ParseRawVault or TracksRelatedSlugs or MirrorsToVault" -v`
Expected: FAIL — `AttributeError: type object 'Ingestor' has no attribute '_parse_raw_vault'`

- [ ] **Step 3: Implémenter**

Dans `Wiki_LM/tools/ingest.py` :

1. Import (ligne 41) — remplacer :
```python
from wiki_paths import CONTENT_SUBDIRS, CLUSTERING_SUBDIR, iter_pages, slug_to_path
```
par :
```python
from wiki_paths import CONTENT_SUBDIRS, CLUSTERING_SUBDIR, iter_pages, mirror_page, slug_to_path
```

2. `__init__` (ligne 815-829) — après la boucle `for d in (self.wiki_dir, self.raw_dir): d.mkdir(...)`, ajouter :
```python
        self._last_related_slugs: list[str] = []
```

3. `_parse_raw_simple` (ligne 1183) — juste après cette méthode (avant `def ingest_batch`, ligne 1191), ajouter :
```python

    @staticmethod
    def _parse_raw_vault(path: Path) -> str | None:
        """Lit la ligne `vault: <nom>` d'un fichier raw si présente — coffre
        d'origine de la capture, pour repousser au bon miroir après
        ingestion (sous-wikis par coffre, 2026-09-22)."""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.lower().startswith("vault:"):
                value = line[len("vault:"):].strip()
                return value or None
        return None
```

4. `ingest()` (ligne 1215) — juste après la ligne de docstring de fermeture (`"""`, avant `print(f"[ingest] Lecture de la source : {source}")`), ajouter :
```python
        self._last_related_slugs = []
```

5. Dans `ingest()`, la section (autour de la ligne 1300) :
```python
        # 4. Mettre à jour / créer les pages de concepts et entités
        for concept in concepts:
            self._update_concept_page(concept, source_title, src_slug, content)

        for entity in entities:
            self._update_entity_page(entity, source_title, src_slug, content)
```
remplacer par :
```python
        # 4. Mettre à jour / créer les pages de concepts et entités
        for concept in concepts:
            self._update_concept_page(concept, source_title, src_slug, content)
            self._last_related_slugs.append(f"c-{_slugify(concept)}")

        for entity in entities:
            self._update_entity_page(entity, source_title, src_slug, content)
            self._last_related_slugs.append(f"e-{_slugify(entity)}")
```

6. Dans `ingest_raw_dir()` (autour de la ligne 1129), remplacer :
```python
                slugs.append(slug)
                self._mark_ingested(path.name, slug=slug, file_hash=_file_hash(path))
```
par :
```python
                vault_name = self._parse_raw_vault(path)
                if vault_name:
                    mirror_page(self.wiki_root, vault_name, slug)
                    for related_slug in self._last_related_slugs:
                        mirror_page(self.wiki_root, vault_name, related_slug)
                slugs.append(slug)
                self._mark_ingested(path.name, slug=slug, file_hash=_file_hash(path))
```

- [ ] **Step 4: Vérifier le succès**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: tous PASS

- [ ] **Step 5: Commit**

```bash
git add Wiki_LM/tools/ingest.py Wiki_LM/tests/test_ingest.py
git commit -m "feat(wiki): ingestion pousse src-/concepts/entités vers le miroir du coffre d'origine"
```

---

### Task 5: `wiki.py` — `op_capture`/`op_query` acceptent `vault_name`

**Files:**
- Modify: `Wiki_LM/tools/wiki.py`
- Test: `Wiki_LM/tests/test_wiki_cli.py`

**Interfaces:**
- Consumes: `capture_urls(..., vault_name=...)`, `capture_comment(..., vault_name=...)` (Task 3) ; `WikiQuery.query(..., vault_name=...)` (déjà existant, 2026-09-21).
- Produces: `op_capture(text: str, vault_name: str | None = None) -> dict`, `op_query(question: str, vault_name: str | None = None) -> dict`.

- [ ] **Step 1: Adapter les stubs existants, vérifier l'échec**

Dans `Wiki_LM/tests/test_wiki_cli.py`, les stubs `_Q.query` (deux occurrences, autour des lignes 90 et 108) doivent accepter le nouveau paramètre pour ne pas casser quand `op_query` sera modifié. Remplacer, dans les deux classes `_Q` :
```python
        def query(self, q, top_k=5):
            return _R()
```
par :
```python
        def query(self, q, top_k=5, vault_name=None):
            return _R()
```

Ajouter ensuite les tests suivants (nouvelle classe, en fin de fichier) :

```python
class TestOpCaptureVaultName:
    def test_forwards_vault_name_to_raw_file(self, monkeypatch, tmp_path):
        wiki = _wiki(monkeypatch, tmp_path)
        out = wiki.op_capture("note libre", vault_name="Arbath")
        raw_file = tmp_path / "raw" / out["files"][0]
        assert "vault: Arbath" in raw_file.read_text(encoding="utf-8")

    def test_no_vault_line_without_vault_name(self, monkeypatch, tmp_path):
        wiki = _wiki(monkeypatch, tmp_path)
        out = wiki.op_capture("note libre")
        raw_file = tmp_path / "raw" / out["files"][0]
        assert "vault:" not in raw_file.read_text(encoding="utf-8")


class TestOpQueryVaultName:
    def test_forwards_vault_name_to_query(self, monkeypatch, tmp_path):
        wiki = _wiki(monkeypatch, tmp_path)

        class _R:
            text = "Synthèse."
            references = []
            history_slug = "slug"
            brief = "Bref."

        calls = []

        class _Q:
            def __init__(self, *a, **k):
                pass

            def query(self, q, top_k=5, vault_name=None):
                calls.append(vault_name)
                return _R()

        monkeypatch.setattr(wiki, "WikiQuery", _Q)
        wiki.op_query("question ?", vault_name="Arbath")
        wiki.op_query("question ?")

        assert calls == ["Arbath", None]
```

Le helper `_wiki(monkeypatch, tmp_path)` (déjà en tête de `test_wiki_cli.py`, ligne 7) positionne `WIKI_PATH=tmp_path`, crée `tmp_path/raw`, recharge le module `wiki` et le retourne — le réutiliser tel quel, ne pas le redéfinir. `op_capture(..., vault_name=...)` écrit alors sous `tmp_path / "raw"`, d'où les chemins `raw_file = tmp_path / "raw" / out["files"][0]` utilisés ci-dessus.

Run : `cd Wiki_LM && .venv/bin/python -m pytest tests/test_wiki_cli.py -v`
Expected: les tests déjà existants passent toujours (stubs mis à jour) ; les deux nouvelles classes échouent — `TypeError: op_capture() got an unexpected keyword argument 'vault_name'`

- [ ] **Step 2: Implémenter**

Dans `Wiki_LM/tools/wiki.py` :

1. `op_capture` (ligne 51) — signature actuelle `def op_capture(text: str) -> dict:` devient
`def op_capture(text: str, vault_name: str | None = None) -> dict:`. Dans le corps, remplacer :
```python
    raw = _raw_dir()
    raw.mkdir(parents=True, exist_ok=True)
    created = []
    if urls:
        created.extend(capture_urls(urls, raw, tags=tags or None, note=note or None))
        if refs:
            created.append(capture_comment("", raw, tags=None, refs=refs))
    elif note or refs:
        created.append(capture_comment(note, raw, tags=tags or None, refs=refs or None))
    return {"files": [p.name for p in created if p is not None]}
```
par :
```python
    raw = _raw_dir()
    raw.mkdir(parents=True, exist_ok=True)
    created = []
    if urls:
        created.extend(capture_urls(urls, raw, tags=tags or None, note=note or None, vault_name=vault_name))
        if refs:
            created.append(capture_comment("", raw, tags=None, refs=refs, vault_name=vault_name))
    elif note or refs:
        created.append(capture_comment(note, raw, tags=tags or None, refs=refs or None, vault_name=vault_name))
    return {"files": [p.name for p in created if p is not None]}
```
La branche `@simple` (écriture directe dans `wiki/sources/`, quelques lignes plus haut) reste **inchangée** — hors périmètre (voir Global Constraints).

2. `op_query` (ligne 116) — remplacer :
```python
def op_query(question: str) -> dict:
    try:
        result = WikiQuery(_wiki_root()).query(question)
```
par :
```python
def op_query(question: str, vault_name: str | None = None) -> dict:
    try:
        result = WikiQuery(_wiki_root()).query(question, vault_name=vault_name)
```
Le reste du corps de `op_query` ne change pas.

- [ ] **Step 3: Vérifier le succès**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_wiki_cli.py -v`
Expected: tous PASS

- [ ] **Step 4: Commit**

```bash
git add Wiki_LM/tools/wiki.py Wiki_LM/tests/test_wiki_cli.py
git commit -m "feat(wiki): op_capture/op_query acceptent vault_name"
```

---

### Task 6: `server.py` — `/capture` et `/run` transportent `vault_name`

**Files:**
- Modify: `Wiki_LM/tools/server.py`
- Test: `Wiki_LM/tests/test_server.py`

**Interfaces:**
- Consumes: `op_capture(text, vault_name=None)`, `op_query(question, vault_name=None)` (Task 5).
- Produces: `POST /capture` et `POST /run` acceptent un champ optionnel `vault_name` dans le corps JSON.

- [ ] **Step 1: Écrire les tests, en échec**

Dans `Wiki_LM/tests/test_server.py`, classe `TestHandleRun` — remplacer la table et le test paramétré :
```python
    _COMMAND_TABLE = [
        ("/c", "op_capture", "texte de capture"),
        ("/q", "op_query", "une question ?"),
        ("/ingest", "op_ingest", ""),
        ("/wikistatus", "op_status", ""),
        ("/r", "op_search", "une recherche"),
        ("/tags", "op_tags", ""),
        ("/kbupdate", "op_kb_update", ""),
        ("/relire", "op_review", ""),
        ("/verifie", "op_verify", "src-test"),
        ("/supprimer?", "op_delete_preview", "src-test"),
        ("/supprimer!", "op_delete", "src-test"),
    ]
```
par (retrait de `/c` et `/q`, traités séparément ci-dessous car leur arité d'appel change) :
```python
    _COMMAND_TABLE = [
        ("/ingest", "op_ingest", ""),
        ("/wikistatus", "op_status", ""),
        ("/r", "op_search", "une recherche"),
        ("/tags", "op_tags", ""),
        ("/kbupdate", "op_kb_update", ""),
        ("/relire", "op_review", ""),
        ("/verifie", "op_verify", "src-test"),
        ("/supprimer?", "op_delete_preview", "src-test"),
        ("/supprimer!", "op_delete", "src-test"),
    ]
```
Après la méthode `test_dispatches_each_command_to_its_op`, ajouter :
```python
    def test_dispatches_c_with_vault_name(self, client, monkeypatch):
        import server
        calls = []
        monkeypatch.setattr(server, "op_capture", lambda *a: calls.append(a) or {"status": "ok"})
        client.post("/run", json={"command": "/c", "arg": "texte", "vault_name": "Arbath"})
        assert calls == [("texte", "Arbath")]

    def test_dispatches_c_without_vault_name(self, client, monkeypatch):
        import server
        calls = []
        monkeypatch.setattr(server, "op_capture", lambda *a: calls.append(a) or {"status": "ok"})
        client.post("/run", json={"command": "/c", "arg": "texte"})
        assert calls == [("texte", None)]

    def test_dispatches_q_with_vault_name(self, client, monkeypatch):
        import server
        calls = []
        monkeypatch.setattr(server, "op_query", lambda *a: calls.append(a) or {"status": "ok"})
        client.post("/run", json={"command": "/q", "arg": "question ?", "vault_name": "Arbath"})
        assert calls == [("question ?", "Arbath")]
```

Toujours dans `TestHandleRun`, le test `test_op_exception_returns_500` stub `op_capture` avec `def boom(arg): raise RuntimeError(...)` — remplacer par :
```python
    def test_op_exception_returns_500(self, client, monkeypatch):
        import server

        def boom(*a):
            raise RuntimeError("panne simulée")

        monkeypatch.setattr(server, "op_capture", boom)
        response = client.post("/run", json={"command": "/c", "arg": "texte"})

        assert response.status_code == 500
        assert "panne simulée" in response.get_json()["error"]
```

Dans `TestHandleQuery` (route `/query`, indépendante de `/run` — pas de changement de comportement mais vérifions `/capture`), ajouter une nouvelle classe :
```python
class TestHandleCapture:
    def test_forwards_vault_name(self, client, monkeypatch):
        import server
        calls = []

        def fake_capture_comment(text, raw, tags=None, title=None, vault_name=None):
            calls.append(vault_name)
            class _Path:
                name = "fake.md"
            return _Path()

        monkeypatch.setattr(server, "capture_comment", fake_capture_comment)
        client.post("/capture", json={"text": "note", "vault_name": "Arbath"})
        client.post("/capture", json={"text": "note"})

        assert calls == ["Arbath", None]
```

- [ ] **Step 2: Vérifier l'échec**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_server.py -v`
Expected: les nouveaux tests échouent (`/run` n'envoie pas encore `vault_name`, `handle_capture` non plus) ; la table réduite continue de passer.

- [ ] **Step 3: Implémenter**

Dans `Wiki_LM/tools/server.py` :

1. `handle_capture()` — remplacer :
```python
@app.post("/capture")
def handle_capture():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Paramètre 'text' manquant"}), 400
    tags_raw = [str(t) for t in (data.get("tags") or [])]
    tags = _normalize_tags(tags_raw) if tags_raw else []
    title = str(data.get("title", "")).strip()
    path = capture_comment(text, raw_dir(), tags=tags or None, title=title or None)
    return jsonify({"status": "ok", "filename": path.name})
```
par :
```python
@app.post("/capture")
def handle_capture():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Paramètre 'text' manquant"}), 400
    tags_raw = [str(t) for t in (data.get("tags") or [])]
    tags = _normalize_tags(tags_raw) if tags_raw else []
    title = str(data.get("title", "")).strip()
    vault_name = str(data.get("vault_name", "")).strip() or None
    path = capture_comment(text, raw_dir(), tags=tags or None, title=title or None, vault_name=vault_name)
    return jsonify({"status": "ok", "filename": path.name})
```

2. `_RUN_OPS` — remplacer :
```python
_RUN_OPS = {
    "/c": lambda arg: op_capture(arg),
    "/q": lambda arg: op_query(arg),
    "/ingest": lambda arg: op_ingest(),
    "/wikistatus": lambda arg: op_status(),
    "/r": lambda arg: op_search(arg),
    "/tags": lambda arg: op_tags(),
    "/kbupdate": lambda arg: op_kb_update(),
    "/relire": lambda arg: op_review(),
    "/verifie": lambda arg: op_verify(arg),
    # /supprimer (sans !) reste absente : jamais dispatchée, même demandée
    # explicitement — seule Telegram, avec essai à blanc puis /confirm,
    # peut supprimer sans le ! explicite ci-dessous.
    "/supprimer?": lambda arg: op_delete_preview(arg),
    "/supprimer!": lambda arg: op_delete(arg),
}
```
par :
```python
_RUN_OPS = {
    # Chaque entrée reçoit (arg, vault_name) de façon uniforme — un seul
    # protocole d'appel dans la table de dispatch. Seules /c et /q
    # utilisent vault_name ; les autres l'ignorent (sous-wikis par coffre,
    # 2026-09-22).
    "/c": lambda arg, vault: op_capture(arg, vault),
    "/q": lambda arg, vault: op_query(arg, vault),
    "/ingest": lambda arg, vault: op_ingest(),
    "/wikistatus": lambda arg, vault: op_status(),
    "/r": lambda arg, vault: op_search(arg),
    "/tags": lambda arg, vault: op_tags(),
    "/kbupdate": lambda arg, vault: op_kb_update(),
    "/relire": lambda arg, vault: op_review(),
    "/verifie": lambda arg, vault: op_verify(arg),
    # /supprimer (sans !) reste absente : jamais dispatchée, même demandée
    # explicitement — seule Telegram, avec essai à blanc puis /confirm,
    # peut supprimer sans le ! explicite ci-dessous.
    "/supprimer?": lambda arg, vault: op_delete_preview(arg),
    "/supprimer!": lambda arg, vault: op_delete(arg),
}
```

3. `handle_run()` — remplacer :
```python
@app.post("/run")
def handle_run():
    """Exécute une commande wiki pour le lot Obsidian (blocs ```wiki). /supprimer
    est absente de _RUN_OPS : jamais dispatchée, même demandée explicitement."""
    data = request.get_json(silent=True) or {}
    command = str(data.get("command", "")).strip()
    arg = str(data.get("arg", ""))

    op = _RUN_OPS.get(command)
    if op is None:
        return jsonify({"error": f"Commande inconnue ou non autorisée en lot : {command!r}"}), 400

    try:
        result = op(arg)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)
```
par :
```python
@app.post("/run")
def handle_run():
    """Exécute une commande wiki pour le lot Obsidian (blocs ```wiki). /supprimer
    est absente de _RUN_OPS : jamais dispatchée, même demandée explicitement."""
    data = request.get_json(silent=True) or {}
    command = str(data.get("command", "")).strip()
    arg = str(data.get("arg", ""))
    vault_name = str(data.get("vault_name", "")).strip() or None

    op = _RUN_OPS.get(command)
    if op is None:
        return jsonify({"error": f"Commande inconnue ou non autorisée en lot : {command!r}"}), 400

    try:
        result = op(arg, vault_name)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)
```

Aussi mettre à jour le docstring en tête de fichier (`POST /run`) pour mentionner `vault_name`, sur le modèle de ce qui a été fait pour `POST /query` le 21/09.

- [ ] **Step 4: Vérifier le succès**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_server.py -v`
Expected: tous PASS

- [ ] **Step 5: Commit**

```bash
git add Wiki_LM/tools/server.py Wiki_LM/tests/test_server.py
git commit -m "feat(wiki): /capture et /run transportent vault_name"
```

---

### Task 7: Plugin Obsidian — envoyer `vault_name` depuis les deux points d'entrée

**Files:**
- Modify: `Wiki_LM/obsidian-wikilm-capture/src/main.ts`

**Interfaces:**
- Consumes: `POST /capture` et `POST /run` acceptent `vault_name` (Task 6).
- Produces: aucune nouvelle interface — modification du corps des requêtes déjà émises.

Pas de nouveau test unitaire : `main.ts` n'est pas testé unitairement aujourd'hui (seules les fonctions pures de `run-commands.ts` le sont, déjà couvertes). Vérification par build + test en direct (Step 3-4).

- [ ] **Step 1: Modifier `captureCurrentNote()`**

Dans `Wiki_LM/obsidian-wikilm-capture/src/main.ts`, remplacer :
```typescript
      const response = await requestUrl({
        url: `${this.settings.serverUrl}/capture`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ text, tags, title: file.basename }),
      });
```
par :
```typescript
      const response = await requestUrl({
        url: `${this.settings.serverUrl}/capture`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ text, tags, title: file.basename, vault_name: this.app.vault.getName() }),
      });
```

- [ ] **Step 2: Modifier `runOneCommand()`**

Dans le même fichier, remplacer :
```typescript
      const response = await requestUrl({
        url: `${this.settings.serverUrl}/run`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ command: block.command, arg: block.arg }),
        throw: false,
      });
```
par :
```typescript
      const response = await requestUrl({
        url: `${this.settings.serverUrl}/run`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ command: block.command, arg: block.arg, vault_name: this.app.vault.getName() }),
        throw: false,
      });
```

- [ ] **Step 3: Construire et faire passer les tests existants**

```bash
cd Wiki_LM/obsidian-wikilm-capture
npm test -- --run
npm run build
```
Expected: 30 tests plugin PASS (inchangés — `main.ts` n'a pas de test dédié), `main.js` reconstruit sans erreur.

- [ ] **Step 4: Commit**

```bash
git add Wiki_LM/obsidian-wikilm-capture/src/main.ts
git commit -m "feat(wiki): le plugin envoie vault_name à /capture et /run"
```

(`main.js` reste hors dépôt — `.gitignore` — comme pour tous les changements précédents du plugin cette semaine.)

---

### Task 8: Vérification finale, déploiement, documentation

**Files:**
- Modify: `docs/architecture/wiki-lm-architecture.md` (section « Généralisation multi-wiki »)

- [ ] **Step 1: Suite de tests complète**

```bash
cd Wiki_LM && .venv/bin/python -m pytest tests/ -q
cd obsidian-wikilm-capture && npm test -- --run
```
Expected: tous PASS (Python : 376 + nouveaux de ce plan ; plugin : 30).

- [ ] **Step 2: Mettre à jour la doc d'architecture**

Dans `docs/architecture/wiki-lm-architecture.md`, section « 10. État actuel et chantiers ouverts », remplacer la ligne :
```
- **Généralisation multi-wiki** — le système entier suppose un wiki
  unique via `WIKI_PATH` (outils, sandbox, services systemd). Sujet
  architectural à part entière, pas encore brainstormé.
```
par :
```
- **Sous-wikis par coffre** — les pages produites/citées par les requêtes
  et ingestions d'un coffre client sont copiées vers son miroir local
  depuis le 22/09/2026 (`docs/superpowers/specs/2026-09-22-sous-wikis-par-
  coffre-design.md`). Reste : rafraîchissement des pages déjà miroitées si
  le canonique change sans nouvelle citation (piste B de la spec, reportée
  délibérément) ; `WIKI_PATH` reste supposé unique côté outils/sandbox pour
  tout le reste (services systemd, ZIM, embeddings).
```

- [ ] **Step 3: Redémarrer et vérifier en direct — demander confirmation avant**

Ceci touche `server.py`/`query.py`/`ingest.py`/`wiki.py`, servis par `wiki-lm-server` : demander la confirmation de l'utilisateur avant `systemctl --user restart wiki-lm-server`, puis déployer le plugin reconstruit dans les deux coffres (`~/Documents/Secretarius/.obsidian/plugins/wikilm-capture/main.js` et `~/Documents/Arbath/.obsidian/plugins/wikilm-capture/main.js`), comme à chaque changement de plugin cette semaine.

Test en direct suggéré, à faire nettoyer ensuite (mêmes précautions que les tests en direct précédents de cette semaine — créer puis supprimer les artefacts) :
```bash
curl -s -X POST http://127.0.0.1:5051/run -H "Content-Type: application/json" \
  -d '{"command":"/q","arg":"une question dont la réponse cite une page connue","vault_name":"Arbath"}'
# Vérifier que la page citée est apparue sous
# ~/Documents/Arbath/Wiki_LM/wiki/<sous-dossier>/<slug>.md
```

- [ ] **Step 4: Commit de la documentation**

```bash
git add docs/architecture/wiki-lm-architecture.md
git commit -m "docs(wiki): sous-wikis par coffre livrés, mise à jour de l'état des chantiers"
```
