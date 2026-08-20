# Directive `@simple` pour les captures URL — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendre la directive `@simple` (`/c @simple <url>`) réellement fonctionnelle : une capture URL marquée `@simple` doit être ingérée en verbatim (pas de résumé LLM), au lieu du résumé habituel.

**Architecture:** `capture.py` détecte le jeton `@simple` dans les arguments de `/c`, l'écrit comme marqueur `simple: true` dans le fichier `.url` créé. `ingest.py` lit ce marqueur et le transmet au paramètre `local_note` déjà existant de `Ingestor.ingest()` (qui bascule déjà entre page verbatim et page résumée pour les fichiers locaux, mais n'était jamais alimenté pour les URLs).

**Tech Stack:** Python 3, pytest, exécution via `~/Secretarius/Wiki_LM/.venv/bin/pytest` depuis `~/Secretarius/Wiki_LM/`.

## Global Constraints

- Périmètre strict : captures URL uniquement. Les captures texte/mixte (`capture_comment`, `capture_mixed`) et fichier joint (`capture_file`, `--file`) ne sont **pas** modifiées.
- Jeton `@simple` : insensible à la casse, limite de mot (`@simplement` ne doit pas matcher).
- Marqueur dans le fichier `.url` : ligne `simple: true` (même style que les lignes `url:`/`tags:`/`note:` existantes), lecture insensible à la casse.
- Aucune modification de signature sur `capture_comment`, `capture_mixed`, `capture_file` — seule `capture_urls` gagne un paramètre `simple`.
- `Ingestor.ingest()` a déjà un paramètre `local_note: bool = False` (ingest.py ligne ~1128) — ne pas le modifier, seulement l'alimenter depuis les nouveaux appelants.
- Commande de test : `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/<fichier> -v`. Baseline actuelle : 20/20 passants sur `test_capture.py` seul ; lancer la suite complète (`.venv/bin/pytest`) avant/après le plan.

---

### Task 1: `capture.py` — détection du jeton `@simple`

**Files:**
- Modify: `Wiki_LM/tools/capture.py:100-112` (section "Hashtags", après `_parse_hashtags`)
- Test: `Wiki_LM/tests/test_capture.py`

**Interfaces:**
- Produces: `_parse_simple_directive(text: str) -> tuple[bool, str]` — `(présent, texte_restant)`. Consommée par la Task 2.

- [ ] **Step 1: Write the failing tests**

Ajouter dans `Wiki_LM/tests/test_capture.py`, après l'import existant `from capture import capture_urls, capture_comment, capture_file, _normalize_url` (ligne 7) — étendre l'import :

```python
from capture import (
    capture_urls, capture_comment, capture_file, _normalize_url,
    _parse_simple_directive,
)
```

Ajouter une nouvelle classe de test, par exemple juste avant `class TestCaptureUrls:` :

```python
class TestParseSimpleDirective:
    def test_detects_and_strips(self):
        found, remaining = _parse_simple_directive("@simple https://example.com")
        assert found is True
        assert "@simple" not in remaining
        assert remaining == "https://example.com"

    def test_absent(self):
        found, remaining = _parse_simple_directive("https://example.com")
        assert found is False
        assert remaining == "https://example.com"

    def test_case_insensitive(self):
        found, remaining = _parse_simple_directive("@SIMPLE https://example.com")
        assert found is True
        assert remaining == "https://example.com"

    def test_not_matched_without_at(self):
        found, remaining = _parse_simple_directive("simple sans arobase")
        assert found is False
        assert remaining == "simple sans arobase"

    def test_not_matched_as_substring(self):
        found, remaining = _parse_simple_directive("@simplement écrit")
        assert found is False
        assert remaining == "@simplement écrit"

    def test_directive_in_middle(self):
        found, remaining = _parse_simple_directive("capture @simple ce texte")
        assert found is True
        assert remaining == "capture ce texte"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_capture.py::TestParseSimpleDirective -v`
Expected: FAIL — `ImportError: cannot import name '_parse_simple_directive'`

- [ ] **Step 3: Write minimal implementation**

Dans `Wiki_LM/tools/capture.py`, ajouter cette fonction juste après `_parse_hashtags` (après la ligne 111 `return tags, remaining`) :

```python
def _parse_simple_directive(text: str) -> tuple[bool, str]:
    """Détecte le jeton @simple. Retourne (présent, texte_restant)."""
    found = bool(re.search(r"@simple\b", text, flags=re.IGNORECASE))
    remaining = re.sub(r"@simple\b\s*", "", text, flags=re.IGNORECASE).strip()
    return found, remaining
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_capture.py -v`
Expected: PASS (20 tests précédents + 6 nouveaux = 26 passants)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/capture.py Wiki_LM/tests/test_capture.py
git commit -m "feat: détecter le jeton @simple dans les arguments de capture"
```

---

### Task 2: `capture.py` — `capture_urls` accepte `simple`, câblage dans `main()`, doc `/c`

**Files:**
- Modify: `Wiki_LM/tools/capture.py:149-172` (`capture_urls`)
- Modify: `Wiki_LM/tools/capture.py:277-297` (bloc URL de `main()`)
- Modify: `openclaw-config/workspace/skills/c/SKILL.md`
- Test: `Wiki_LM/tests/test_capture.py`

**Interfaces:**
- Consumes: `_parse_simple_directive` (Task 1).
- Produces: `capture_urls(urls, raw, tags=None, note=None, simple=False) -> list[Path]` — écrit `simple: true` dans chaque fichier `.url` créé quand `simple=True`.

- [ ] **Step 1: Write the failing tests**

Ajouter dans `class TestCaptureUrls:` (`Wiki_LM/tests/test_capture.py`), après la méthode existante `test_note_on_first_of_multiple` :

```python
    def test_simple_flag_written(self, tmp_path):
        files = capture_urls(["https://example.com"], tmp_path, simple=True)
        assert "simple: true" in files[0].read_text()

    def test_simple_flag_absent_by_default(self, tmp_path):
        files = capture_urls(["https://example.com"], tmp_path)
        assert "simple:" not in files[0].read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_capture.py::TestCaptureUrls::test_simple_flag_written -v`
Expected: FAIL — `TypeError: capture_urls() got an unexpected keyword argument 'simple'`

- [ ] **Step 3: Write minimal implementation**

Remplacer la signature et le corps de `capture_urls` (`Wiki_LM/tools/capture.py:149-172`) par :

```python
def capture_urls(urls: list[str], raw: Path, tags: list[str] | None = None,
                 note: str | None = None, simple: bool = False) -> list[Path]:
    ts = timestamp()
    existing = _existing_urls(raw)
    created = []
    for i, url in enumerate(urls):
        norm = _normalize_url(url)
        if norm in existing:
            print(f"Doublon ignoré (URL déjà dans raw/) : {url}")
            continue
        existing.add(norm)
        domain = re.sub(r"https?://", "", url).split("/")[0]
        domain_slug = slugify(domain.replace(".", "-"), max_words=3)
        suffix = f"-{i}" if len(urls) > 1 else ""
        fname = f"{ts}{suffix}-{domain_slug}.url"
        path = raw / fname
        content = url + "\n"
        if simple:
            content += "simple: true\n"
        if tags:
            content += f"tags: {', '.join(tags)}\n"
        if note and not created:            # note attachée au premier .url créé
            content += f"note: {note}\n"
        path.write_text(content, encoding="utf-8")
        created.append(path)
    return created
```

Puis, dans `main()`, repérer ce bloc actuel :

```python
    args = " ".join(argv).strip()
    if not args:
        print("Usage: capture.py <url|commentaire|#tags ...> | --file <chemin>",
              file=sys.stderr)
        sys.exit(1)

    tags_raw, args_clean = _parse_hashtags(args)
    tags = _normalize_tags(tags_raw)

    tokens = args_clean.split() if args_clean else []
    urls = [t for t in tokens if re.match(r"https?://", t)]
    text_tokens = [t for t in tokens if not re.match(r"https?://", t)]
    text = " ".join(text_tokens).strip()

    if urls and text:
        path = capture_mixed(text, urls, raw, tags)
        print(f"Note → {path.name}")
    elif urls:
        created = capture_urls(urls, raw, tags)
        for p in created:
            print(f"URL  → {p.name}")
    else:
        path = capture_comment(text or " ".join(f"#{t}" for t in tags_raw), raw, tags)
        print(f"Note → {path.name}")
```

Le remplacer par :

```python
    args = " ".join(argv).strip()
    if not args:
        print("Usage: capture.py <url|commentaire|#tags ...> | --file <chemin>",
              file=sys.stderr)
        sys.exit(1)

    simple, args = _parse_simple_directive(args)
    tags_raw, args_clean = _parse_hashtags(args)
    tags = _normalize_tags(tags_raw)

    tokens = args_clean.split() if args_clean else []
    urls = [t for t in tokens if re.match(r"https?://", t)]
    text_tokens = [t for t in tokens if not re.match(r"https?://", t)]
    text = " ".join(text_tokens).strip()

    if urls and text:
        path = capture_mixed(text, urls, raw, tags)
        print(f"Note → {path.name}")
    elif urls:
        created = capture_urls(urls, raw, tags, simple=simple)
        for p in created:
            print(f"URL  → {p.name}")
    else:
        path = capture_comment(text or " ".join(f"#{t}" for t in tags_raw), raw, tags)
        print(f"Note → {path.name}")
```

(`simple` est extrait avant `_parse_hashtags` mais reste sans effet dans les branches texte/mixte, conformément au périmètre — voir Global Constraints.)

Enfin, mettre à jour `openclaw-config/workspace/skills/c/SKILL.md` (remplacer la ligne 11) :

```markdown
`/c [@simple] [#tags] <url|texte>` capture la ressource dans le wiki, de façon
déterministe (aucune décision du modèle) : l'outil `wiki_capture` délègue
`op: capture | <args>` à l'agent wiki et relaie le résultat. `@simple`
(captures URL uniquement) force une ingestion verbatim, sans résumé LLM.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_capture.py -v`
Expected: PASS (28 tests passants)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/capture.py Wiki_LM/tests/test_capture.py openclaw-config/workspace/skills/c/SKILL.md
git commit -m "feat: capture_urls écrit le marqueur simple:, câblage /c @simple"
```

---

### Task 3: `ingest.py` — `Ingestor._parse_raw_simple`

**Files:**
- Modify: `Wiki_LM/tools/ingest.py:1082-1093` (après `_parse_raw_tags`)
- Test: `Wiki_LM/tests/test_ingest.py`

**Interfaces:**
- Produces: `Ingestor._parse_raw_simple(path: Path) -> bool` (méthode statique). Consommée par les Tasks 4 et 5.

- [ ] **Step 1: Write the failing tests**

Ajouter dans `Wiki_LM/tests/test_ingest.py`, une nouvelle classe juste après `class TestParseRawTags:` (dont le dernier test est `test_single_bracket_wrapped_tag`, ligne ~422) :

```python
class TestParseRawSimple:
    def test_true_when_marker_present(self, tmp_path: Path):
        f = tmp_path / "test.url"
        f.write_text("https://example.com\nsimple: true\n", encoding="utf-8")
        assert Ingestor._parse_raw_simple(f) is True

    def test_false_when_absent(self, tmp_path: Path):
        f = tmp_path / "test.url"
        f.write_text("https://example.com\n", encoding="utf-8")
        assert Ingestor._parse_raw_simple(f) is False

    def test_false_when_value_false(self, tmp_path: Path):
        f = tmp_path / "test.url"
        f.write_text("https://example.com\nsimple: false\n", encoding="utf-8")
        assert Ingestor._parse_raw_simple(f) is False

    def test_case_insensitive(self, tmp_path: Path):
        f = tmp_path / "test.url"
        f.write_text("https://example.com\nSimple: True\n", encoding="utf-8")
        assert Ingestor._parse_raw_simple(f) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_ingest.py::TestParseRawSimple -v`
Expected: FAIL — `AttributeError: type object 'Ingestor' has no attribute '_parse_raw_simple'`

- [ ] **Step 3: Write minimal implementation**

Dans `Wiki_LM/tools/ingest.py`, ajouter cette méthode juste après `_parse_raw_tags` (après la ligne 1093 `return []`) :

```python
    @staticmethod
    def _parse_raw_simple(path: Path) -> bool:
        """Lit la ligne `simple: true` d'un fichier raw si présente."""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.lower().startswith("simple:"):
                return line[len("simple:"):].strip().lower() == "true"
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_ingest.py -v`
Expected: PASS (tous les tests existants + 4 nouveaux passants)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/ingest.py Wiki_LM/tests/test_ingest.py
git commit -m "feat: Ingestor._parse_raw_simple lit le marqueur simple: des fichiers raw"
```

---

### Task 4: `ingest.py` — transmettre `local_note` depuis `ingest_raw_dir()`

**Files:**
- Modify: `Wiki_LM/tools/ingest.py:1042-1050` (branche `.url` de `ingest_raw_dir`)
- Test: `Wiki_LM/tests/test_ingest.py`

**Interfaces:**
- Consumes: `Ingestor._parse_raw_simple` (Task 3), `Ingestor.ingest(..., local_note: bool = False)` (existant, inchangé).

- [ ] **Step 1: Write the failing test**

Ajouter dans `Wiki_LM/tests/test_ingest.py`, dans `class TestIngestLocalNote:` (dont le dernier test est `test_md_dispatched_as_note`, ligne ~394), en suivant le patron spy déjà utilisé plus haut dans le fichier (`ingestor.ingest = boom`, cf. `test_errored_file_not_marked_for_retry`) :

```python
    def test_url_marked_simple_passes_local_note_true(self, ingestor, raw_dir):
        (raw_dir / "test.url").write_text(
            "https://example.com\nsimple: true\n", encoding="utf-8")
        captured = {}

        def fake_ingest(source, **kwargs):
            captured.update(kwargs)
            return "src-test"

        ingestor.ingest = fake_ingest
        ingestor.ingest_raw_dir()
        assert captured.get("local_note") is True

    def test_url_without_marker_passes_local_note_false(self, ingestor, raw_dir):
        (raw_dir / "test.url").write_text("https://example.com\n", encoding="utf-8")
        captured = {}

        def fake_ingest(source, **kwargs):
            captured.update(kwargs)
            return "src-test"

        ingestor.ingest = fake_ingest
        ingestor.ingest_raw_dir()
        assert captured.get("local_note") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_ingest.py::TestIngestLocalNote::test_url_marked_simple_passes_local_note_true -v`
Expected: FAIL — `assert None is True` (le paramètre `local_note` n'est pas transmis)

- [ ] **Step 3: Write minimal implementation**

Dans `Wiki_LM/tools/ingest.py`, remplacer les lignes 1042-1050 :

```python
                if path.suffix.lower() == ".url":
                    url = self._parse_url_file(path)
                    if not url:
                        print(f"[ingest] Fichier .url vide ou invalide : {path.name}")
                        self._mark_ingested(path.name, slug="", file_hash=_file_hash(path))
                        continue
                    user_tags = self._parse_raw_tags(path)
                    note = _parse_note_from_url_file(path)
                    slug = self.ingest(url, max_concepts=max_concepts, extra_tags=user_tags or None, rename_raw=False, note=note)
```

par :

```python
                if path.suffix.lower() == ".url":
                    url = self._parse_url_file(path)
                    if not url:
                        print(f"[ingest] Fichier .url vide ou invalide : {path.name}")
                        self._mark_ingested(path.name, slug="", file_hash=_file_hash(path))
                        continue
                    user_tags = self._parse_raw_tags(path)
                    simple = self._parse_raw_simple(path)
                    note = _parse_note_from_url_file(path)
                    slug = self.ingest(url, max_concepts=max_concepts, extra_tags=user_tags or None, rename_raw=False, note=note, local_note=simple)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_ingest.py -v`
Expected: PASS (tous les tests existants + 2 nouveaux passants)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/ingest.py Wiki_LM/tests/test_ingest.py
git commit -m "feat: ingest_raw_dir transmet local_note depuis le marqueur simple: des URLs"
```

---

### Task 5: `retry_blocked_urls.py` — cohérence sur les URLs bloquées relancées

**Files:**
- Modify: `Wiki_LM/tools/retry_blocked_urls.py:36-39`
- Test: `Wiki_LM/tests/test_retry_blocked_urls.py`

**Interfaces:**
- Consumes: `Ingestor._parse_raw_simple` (Task 3).

- [ ] **Step 1: Write the failing test**

Ajouter dans `Wiki_LM/tests/test_retry_blocked_urls.py`, à la suite de `test_retries_forwards_tags_and_note` :

```python
def test_retries_forwards_simple_flag(ingestor, raw_dir: Path, monkeypatch):
    (raw_dir / "20260101-000000-example-com.url.error").write_text(
        "https://example.com/article\nsimple: true\n", encoding="utf-8"
    )

    calls = []

    def fake_ingest(self, source, **kwargs):
        calls.append(kwargs)
        return "src-article"

    monkeypatch.setattr(type(ingestor), "ingest", fake_ingest)

    retry_all(ingestor, raw_dir, dry_run=False)

    assert calls[0]["local_note"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_retry_blocked_urls.py::test_retries_forwards_simple_flag -v`
Expected: FAIL — `KeyError: 'local_note'`

- [ ] **Step 3: Write minimal implementation**

Dans `Wiki_LM/tools/retry_blocked_urls.py`, remplacer (lignes 36-39) :

```python
        try:
            user_tags = ingestor._parse_raw_tags(error_file)
            note = _parse_note_from_url_file(error_file)
            slug = ingestor.ingest(url, extra_tags=user_tags or None, rename_raw=False, note=note)
```

par :

```python
        try:
            user_tags = ingestor._parse_raw_tags(error_file)
            simple = ingestor._parse_raw_simple(error_file)
            note = _parse_note_from_url_file(error_file)
            slug = ingestor.ingest(url, extra_tags=user_tags or None, rename_raw=False, note=note, local_note=simple)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_retry_blocked_urls.py -v`
Expected: PASS (tous les tests existants + 1 nouveau passants)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/retry_blocked_urls.py Wiki_LM/tests/test_retry_blocked_urls.py
git commit -m "fix: retry_blocked_urls respecte le marqueur simple: des URLs bloquées"
```

---

### Task 6: Vérification finale — suite complète et test manuel bout en bout

**Files:** aucun fichier modifié — vérification uniquement.

- [ ] **Step 1: Run full test suite**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest -q`
Expected: PASS, 0 failure (nombre total = baseline + 13 nouveaux tests des Tasks 1-5)

- [ ] **Step 2: Manual end-to-end check**

```bash
cd ~/Secretarius/Wiki_LM/tools
python capture.py "@simple https://fr.wikipedia.org/wiki/Zettelkasten"
cat "$(ls -t ../../../Documents/Arbath/Wiki_LM/raw/*.url | head -1)"
```

Expected: le fichier `.url` le plus récent contient une ligne `simple: true`.

```bash
python -c "
from ingest import Ingestor
i = Ingestor()
i.ingest_raw_dir()
"
```

Expected (dans la sortie console) : `[ingest] Note locale → page verbatim (pas de résumé)…` pour cette URL — confirmer ensuite dans `wiki/sources/src-zettelkasten*.md` l'absence de section `## Résumé`.

- [ ] **Step 3: Report**

Aucun commit pour cette tâche (vérification seule). Si un test manuel échoue, revenir à la tâche concernée avant de considérer le plan terminé.
