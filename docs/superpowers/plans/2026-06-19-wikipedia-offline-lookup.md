# Wikipedia Offline Lookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Wikipedia lookup configurable for local ZIM directories and strict offline operation without changing `WikiLookup(wiki_path)` or `lookup(title, langs=None)`.

**Architecture:** Keep all behavior in `Wiki_LM/tools/wiki_lookup.py`: resolve configuration at `WikiLookup` construction time, then have `lookup()` iterate configured backends in order per language. Tests monkeypatch ZIM/API helpers so no real ZIM, network, or LLM is required.

**Tech Stack:** Python, pathlib, sqlite3, pytest, monkeypatch.

## Global Constraints

- Preserve public API: `WikiLookup(wiki_path)` and `lookup(title, langs=None)`.
- Default behavior remains `zim,cache,api`.
- `WIKI_LOOKUP_OFFLINE=1`, `true`, `yes`, or `on` forces `zim,cache`.
- `WIKI_LOOKUP_BACKENDS` accepts comma-separated `zim`, `cache`, and `api`; unknown values are ignored.
- If parsed backends are empty, fall back to `zim,cache,api`.
- `zim_dir` argument has priority over `WIKI_ZIM_DIR`, which has priority over the existing default `~/Secretarius/Wiki_LM/zim`.
- Do not modify `ingest.py` or `build_wiki_cache.py`.

---

### Task 1: Environment And Backend Tests

**Files:**
- Modify: `Wiki_LM/tests/test_wiki_lookup.py`
- Test: `Wiki_LM/tests/test_wiki_lookup.py`

**Interfaces:**
- Consumes: `WikiLookup(wiki_path, zim_dir=None)`
- Produces: tests requiring `WikiLookup._zim_dir` and backend behavior through `lookup()`

- [ ] **Step 1: Write failing tests**

Add tests covering:

```python
def test_env_zim_dir_used_when_argument_missing(tmp_path, monkeypatch):
    configured = tmp_path / "configured-zim"
    monkeypatch.setenv("WIKI_ZIM_DIR", str(configured))

    lookup = WikiLookup(tmp_path)

    assert lookup._zim_dir == configured
```

```python
def test_explicit_zim_dir_overrides_env(tmp_path, monkeypatch):
    configured = tmp_path / "configured-zim"
    explicit = tmp_path / "explicit-zim"
    monkeypatch.setenv("WIKI_ZIM_DIR", str(configured))

    lookup = WikiLookup(tmp_path, zim_dir=explicit)

    assert lookup._zim_dir == explicit
```

```python
def test_offline_mode_skips_api(tmp_path, monkeypatch):
    monkeypatch.setenv("WIKI_LOOKUP_OFFLINE", "1")
    monkeypatch.setattr("wiki_lookup._zim_files", lambda zim_dir: {})
    calls = []
    monkeypatch.setattr("wiki_lookup._fetch_api", lambda *args: calls.append(args) or {
        "lang": args[1],
        "title": args[0],
        "abstract": "api",
        "url": "",
    })

    lookup = WikiLookup(tmp_path)

    assert lookup.lookup("Offline", langs=["fr"]) is None
    assert calls == []
```

```python
def test_backends_cache_only_uses_cache_without_api_or_zim(tmp_path, monkeypatch):
    monkeypatch.setenv("WIKI_LOOKUP_BACKENDS", "cache")
    monkeypatch.setattr("wiki_lookup._zim_files", lambda zim_dir: {"fr": tmp_path / "fake.zim"})
    monkeypatch.setattr("wiki_lookup._zim_lookup", lambda *args: pytest.fail("ZIM should not be called"))
    monkeypatch.setattr("wiki_lookup._fetch_api", lambda *args: pytest.fail("API should not be called"))
    lookup = WikiLookup(tmp_path)
    lookup._cache_set({"lang": "fr", "title": "Cached", "abstract": "cache", "url": ""})

    result = lookup.lookup("Cached", langs=["fr"])

    assert result["abstract"] == "cache"
```

```python
def test_default_backend_order_is_zim_cache_api(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("wiki_lookup._zim_files", lambda zim_dir: {"fr": tmp_path / "fake.zim"})
    monkeypatch.setattr("wiki_lookup._zim_lookup", lambda *args: calls.append("zim") or None)
    monkeypatch.setattr("wiki_lookup._fetch_api", lambda title, lang: calls.append("api") or {
        "lang": lang,
        "title": title,
        "abstract": "api",
        "url": "",
    })
    lookup = WikiLookup(tmp_path)
    original_cache_get = lookup._cache_get

    def cache_get(title, lang):
        calls.append("cache")
        return original_cache_get(title, lang)

    monkeypatch.setattr(lookup, "_cache_get", cache_get)

    result = lookup.lookup("Order", langs=["fr"])

    assert result["abstract"] == "api"
    assert calls == ["zim", "cache", "api"]
```

```python
def test_invalid_backend_list_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setenv("WIKI_LOOKUP_BACKENDS", "nonsense,other")
    calls = []
    monkeypatch.setattr("wiki_lookup._zim_files", lambda zim_dir: {})
    monkeypatch.setattr("wiki_lookup._fetch_api", lambda title, lang: calls.append("api") or {
        "lang": lang,
        "title": title,
        "abstract": "api",
        "url": "",
    })

    lookup = WikiLookup(tmp_path)
    result = lookup.lookup("Fallback", langs=["fr"])

    assert result["abstract"] == "api"
    assert calls == ["api"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest Wiki_LM/tests/test_wiki_lookup.py -q`

Expected: failures for missing environment/backend support.

### Task 2: Lookup Configuration Implementation

**Files:**
- Modify: `Wiki_LM/tools/wiki_lookup.py`
- Test: `Wiki_LM/tests/test_wiki_lookup.py`

**Interfaces:**
- Produces: `self._backends: tuple[str, ...]`
- Produces: helper functions `_env_truthy(value: str | None) -> bool` and `_parse_backends(value: str | None) -> tuple[str, ...]`

- [ ] **Step 1: Implement minimal configuration helpers**

Add:

```python
_DEFAULT_BACKENDS = ("zim", "cache", "api")
_OFFLINE_BACKENDS = ("zim", "cache")
_BACKEND_NAMES = set(_DEFAULT_BACKENDS)


def _env_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_backends(value: str | None) -> tuple[str, ...]:
    if value is None:
        return _DEFAULT_BACKENDS
    parsed = tuple(
        backend
        for backend in (part.strip().lower() for part in value.split(","))
        if backend in _BACKEND_NAMES
    )
    return parsed or _DEFAULT_BACKENDS
```

- [ ] **Step 2: Resolve env in `WikiLookup.__init__`**

Set:

```python
env_zim_dir = os.environ.get("WIKI_ZIM_DIR")
self._zim_dir = Path(zim_dir) if zim_dir is not None else Path(env_zim_dir) if env_zim_dir else self._DEFAULT_ZIM
self._backends = _OFFLINE_BACKENDS if _env_truthy(os.environ.get("WIKI_LOOKUP_OFFLINE")) else _parse_backends(os.environ.get("WIKI_LOOKUP_BACKENDS"))
```

- [ ] **Step 3: Iterate configured backends in lookup**

Replace fixed ZIM/cache/API sequence with:

```python
zims = self._get_zims() if "zim" in self._backends else {}
for lang in (langs or ["fr", "en"]):
    for backend in self._backends:
        if backend == "zim":
            if lang not in zims:
                continue
            result = _zim_lookup(title, lang, zims[lang])
            if result:
                self._cache_set(result)
                return result
        elif backend == "cache":
            result = self._cache_get(title, lang)
            if result:
                return result
        elif backend == "api":
            result = _fetch_api(title, lang)
            if result:
                self._cache_set(result)
                return result
return None
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest Wiki_LM/tests/test_wiki_lookup.py -q`

Expected: all tests in the file pass.

### Task 3: Documentation

**Files:**
- Modify: `Wiki_LM/README.md`

**Interfaces:**
- Consumes: behavior from Task 2.
- Produces: user-facing configuration reference for ZIM/offline/backends.

- [ ] **Step 1: Update README**

Document:

```markdown
Par défaut, `wiki_lookup.py` cherche les ZIM dans `~/Secretarius/Wiki_LM/zim`.
Définir `WIKI_ZIM_DIR=/chemin/zim` pour utiliser un autre répertoire.
Définir `WIKI_LOOKUP_OFFLINE=1` pour désactiver tout appel API.
Définir `WIKI_LOOKUP_BACKENDS=cache`, `zim,cache`, ou `cache,api` pour contrôler l'ordre avancé.
```

- [ ] **Step 2: Verify targeted tests**

Run: `pytest Wiki_LM/tests/test_wiki_lookup.py -q`

Expected: all tests in the file pass.

