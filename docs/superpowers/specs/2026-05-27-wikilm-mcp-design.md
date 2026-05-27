# Design — Serveur MCP Wiki_LM

**Date :** 2026-05-27  
**Statut :** approuvé  
**Périmètre :** serveur MCP FastMCP (6 outils), modification minimale `ingest.py`, enregistrement dans `openclaw.json`  
**Hors périmètre :** ingest watcher systemd, proxy Gmail MCP (phase 2 Scout)

---

## Contexte

Tiron reçoit des URLs et des notes via Telegram et ne peut pas les traiter directement : `web_fetch` est dans la deny list sandbox et les outils Wiki_LM sont inaccessibles depuis le contexte agent. Le serveur MCP Wiki_LM expose les opérations du pipeline (capture, ingest, query, tags, status, kb_update) comme outils natifs OpenClaw, via transport stdio (subprocess).

**Contrainte principale :** l'ingest ne doit pas court-circuiter la protection anti-injection. Tout contenu fetché depuis une URL passe par `injection-guard.service` (`localhost:8990`) avant d'entrer dans le wiki — et donc avant de pouvoir apparaître dans le contexte de Tiron lors d'une query.

---

## Architecture

```
Telegram → Tiron (OpenClaw sandbox)
               │
               │  MCP stdio (subprocess)
               ▼
         Wiki_LM/tools/mcp_server.py (FastMCP)
         ┌──────────────────────────────────────┐
         │ wiki_capture(text)                   │──→ raw/  (écrit .url ou .md)
         │ wiki_ingest()                        │──→ fetch URL → /check → ingest
         │ wiki_query(question)                 │──→ WikiQuery
         │ wiki_tags()                          │──→ kb_tags
         │ wiki_ingest_status()                 │──→ compte fichiers raw/ en attente
         │ wiki_kb_update()                     │──→ kb_update
         └──────────────────────────────────────┘
                          │
                          │ POST localhost:8990/check (pour wiki_ingest)
                          ▼
                 injection-guard.service
```

`mcp_server.py` est lancé comme subprocess stdio par OpenClaw à chaque appel MCP — pas de service systemd dédié. `injection-guard.service` doit être actif sur l'hôte.

---

## Composants

### `Wiki_LM/tools/mcp_server.py` (nouveau)

Fichier ~150 lignes. Importe les modules existants (`capture`, `ingest`, `query`, `kb_tags`, `kb_update`, `wiki_paths`). Utilise `fastmcp.FastMCP`.

**Outils exposés :**

| Outil | Signature | Description |
|---|---|---|
| `wiki_capture` | `(text: str) -> dict` | Parse URLs + hashtags depuis `text`, appelle `capture.capture_mixed()`, retourne `{files: [...]}` |
| `wiki_ingest` | `() -> dict` | Traite tous les `.url` en attente dans `raw/` via injection-guard, retourne `{ingested, blocked, errors}` |
| `wiki_query` | `(question: str, top_k: int = 5) -> dict` | Retourne `{synthesis, references}` |
| `wiki_tags` | `() -> dict` | Retourne `{tags: [...]}` depuis le KB |
| `wiki_ingest_status` | `() -> dict` | Retourne `{pending: N, blocked_files: [...]}` — "pending" = fichiers `.url` dans `raw/` absents du manifest ingest (non encore ingérés, non bloqués) |
| `wiki_kb_update` | `() -> dict` | Lance `kb_update`, retourne `{status}` |

### `Wiki_LM/tools/ingest.py` (modification minimale)

La méthode `Ingester.ingest()` accepte un nouveau paramètre :

```python
def ingest(self, source: str, content: str | None = None, ...) -> str | None:
```

Si `content` est fourni, le bloc `content, title = _read_source(source)` est sauté. Aucun autre changement. La logique d'embedding, de déduplication et d'écriture wiki reste intacte.

### `openclaw-config/openclaw.json.template` (modification)

Ajout du bloc `mcp` :

```json
"mcp": {
  "servers": {
    "wiki-lm": {
      "command": "${HOME}/Secretarius/Wiki_LM/.venv/bin/python3",
      "args": ["${HOME}/Secretarius/Wiki_LM/tools/mcp_server.py"]
    }
  }
}
```

### `openclaw-config/install.sh` (modification)

Après le bloc d'installation de l'injection guard, ajout :

```bash
# MCP Wiki_LM
if command -v openclaw &>/dev/null; then
  openclaw mcp set wiki-lm "{\"command\":\"${HOME}/Secretarius/Wiki_LM/.venv/bin/python3\",\"args\":[\"${HOME}/Secretarius/Wiki_LM/tools/mcp_server.py\"]}" && \
    info "MCP wiki-lm enregistré" || \
    warn "Enregistrement MCP wiki-lm échoué — relancer manuellement"
fi
```

### `openclaw-config/workspace/skills/wiki-lm/SKILL.md` (nouveau)

Skill pour Tiron : quand utiliser chaque outil, format des arguments, comportement attendu en cas de blocage ou d'erreur.

---

## Flux de données

### Capture

```
Tiron: wiki_capture("#linguistique https://example.com Note personnelle")
  → capture.capture_mixed(text, urls, raw=RAW_DIR, tags=["linguistique"])
  → crée raw/20260527-HHMMSS-example-com.url
  → retourne {files: ["20260527-HHMMSS-example-com.url"]}
```

### Ingest

```
Tiron: wiki_ingest()
  → liste raw/*.url non traités
  → pour chaque .url :
      fetch URL (urllib, 30s)
      POST localhost:8990/check {type:"html", content:html}
      si blocked → renomme .url.blocked, note la raison
      si ok      → Ingester.ingest(source=url, content=clean_text)
                   embed + stockage wiki/
                   le fichier .url reste en place (le manifest ingest prévient
                   la ré-ingestion via déduplication SHA-256/URL normalisée)
  → retourne {ingested: N, blocked: M, errors: K,
               blocked_details: [{file, reason}, ...],
               error_details: [{file, reason}, ...]}
```

### Query

```
Tiron: wiki_query("Que dit le wiki sur les langues impossibles ?")
  → WikiQuery.query(question, top_k=5)
  → retourne {synthesis: "...", references: ["slug1", "slug2", ...]}
```

---

## Gestion d'erreurs

| Situation | Comportement |
|---|---|
| `injection-guard.service` indisponible | fail-safe : fichier marqué `.url.blocked`, reason "injection-guard unavailable" — ingest continue |
| Fetch URL échoue (timeout, 404…) | fichier marqué `.url.error`, raison dans le résumé |
| `wiki_ingest()` avec `raw/` vide | retourne `{ingested:0, blocked:0, errors:0}` |
| `wiki_query()` avec KB vide | retourne `{error: "KB vide — lancer wiki_ingest d'abord"}` |
| FastMCP subprocess crash | OpenClaw relance au prochain appel (comportement stdio natif) |
| `wiki_kb_update()` appel concurrent | retourne `{status: "already_running"}` sans bloquer |

---

## Stratégie de tests

Fichier `Wiki_LM/tests/test_mcp_server.py`. Chaque outil est testé en isolation :

- **`wiki_capture`** : vérifie les fichiers créés dans `tmp_path`, hashtags parsés, déduplication URL
- **`wiki_ingest`** : injection-guard mocké (cas ok, bloqué, indisponible), fetch mocké — vérifie les compteurs et les renommages
- **`wiki_query`** : WikiQuery mocké, vérifie le format de retour
- **`wiki_tags`**, **`wiki_ingest_status`**, **`wiki_kb_update`** : modules sous-jacents mockés

**Contraintes CI :** zéro réseau, zéro modèle réel, zéro appel à `injection-guard.service` réel.

---

## Déploiement

```bash
cd ~/Secretarius
git pull
cd Wiki_LM && pip install fastmcp  # si absent du venv
./install.sh --force
```

Vérification manuelle :
```bash
# Lister les serveurs MCP enregistrés
~/.nvm/versions/node/v22.22.3/bin/openclaw mcp list

# Test capture depuis Telegram
/c wiki_capture "#test https://example.com"
```

---

## Fichiers à créer / modifier

| Fichier | Action |
|---|---|
| `Wiki_LM/tools/mcp_server.py` | Créer |
| `Wiki_LM/tests/test_mcp_server.py` | Créer |
| `Wiki_LM/tools/ingest.py` | Modifier (paramètre `content` dans `ingest()`) |
| `openclaw-config/openclaw.json.template` | Modifier (bloc `mcp`) |
| `openclaw-config/install.sh` | Modifier (enregistrement MCP) |
| `openclaw-config/workspace/skills/wiki-lm/SKILL.md` | Créer |
