# Design — Agent wiki dans Secretarius SLM (Capture + Ingest + Query)

**Date :** 2026-06-12
**Statut :** approuvé
**Périmètre :** doter l'instance SLM (`.openclaw-slm`, sanroque) d'un agent `wiki` fonctionnel offrant capture / ingest / query, dans l'architecture « Tiron léger » (image par agent + skills, **pas de MCP**). Remplace le stub de validation actuel.
**Hors périmètre :** ingestion frugale (résumés extractifs BERT + plongements, Phase 2) ; filtrage anti-injection à l'ingest (amélioration séparée, prod incluse).

---

## Contexte

Sur `.openclaw-slm`, l'agent `wiki` est aujourd'hui un **stub de validation** : son `AGENTS.md` indique que « le skill wiki définitif (orchestration capture/ingest/query, filtrage anti-injection façon Scout) n'existe pas encore ». La capture `/c` n'y est donc pas câblée (pas de skill `c`, pas de MCP côté SLM).

L'infrastructure conteneur est en revanche déjà en place (image `secretarius-wiki:latest`, étape B du 2026-06-08) :
- `tools.exec.host: "sandbox"` → l'agent exécute ses commandes **dans** le conteneur.
- Binds : `/wiki-tools` = `Wiki_LM/tools` (RO) ; `/Wiki_LM` = `~/Documents/Arbath/Wiki_LM` (**RW, KB partagée avec la prod**).
- Env : LLM d'ingest = **Euria distant** (API OpenAI-compatible Infomaniak, Mistral-Small) ; embeddings **BGE-M3 embarqués hors-ligne** (`HF_HUB_OFFLINE=1`) → CPU dans le conteneur. **Aucun passthrough GPU nécessaire.**

Le manque est donc l'**orchestration**, pas l'infrastructure.

---

## Décisions actées

1. **Périmètre** : Capture + Ingest + Query.
2. **Exécution de l'ingest** : dans le conteneur de l'agent (LLM Euria distant + embeddings BGE-M3 CPU). Pas de coût GPU nouveau (l'ingest prod actuel fait déjà LLM + embeddings).
3. **Anti-injection à l'ingest** : **parité prod** (fetch direct dans `ingest.py`) pour ce livrable. Le filtrage « façon Scout » est différé (faille présente aussi en prod aujourd'hui).
4. **Phase 2 (séparée)** : ingestion frugale (résumés extractifs BERT + plongements). L'étape de résumé reste isolée dans `ingest.py` pour être remplaçable **sans toucher** à l'orchestration.
5. **Orchestration** : façade binaire unique `wiki.py <op>` à sortie JSON (approche B), plus fiable pour un consommateur SLM que faire composer des CLI et parser du texte.

---

## Architecture / flux

```
Utilisateur → Tiron (main, Euria)
   │ skill de délégation : intention → tâche
   ▼ sessions_spawn(agentId="wiki", task="op: <capture|ingest|status|query> | <args>")
Agent wiki (Euria, conteneur secretarius-wiki)
   │ skill wiki : traduit la tâche en UN appel exec
   ▼ python /wiki-tools/wiki.py <op> [args]   → JSON sur stdout
   │ relit le JSON, le reformule
   ▼ retourne à Tiron, qui relaie à l'utilisateur
```

---

## Composants

### A. Façade `wiki.py` (nouveau — `Wiki_LM/tools/wiki.py`, monté `/wiki-tools/wiki.py`)

CLI minimal réutilisant les **primitives déjà testées** (`capture_urls`/`capture_comment`, `Ingestor`, `WikiQuery`) — **pas** les wrappers décorés `@mcp.tool()`. Sortie : `json.dumps(...)` sur stdout, code retour 0 (erreurs encodées dans le JSON, fail-safe).

| Op | Commande | Sortie JSON |
|----|----------|-------------|
| capture | `wiki.py capture "<texte>"` | `{files:[...]}` |
| ingest | `wiki.py ingest` | `{status:"started"\|"nothing_to_do"\|"already_running", queued:N}` |
| status | `wiki.py status` | `{running, last_run:{ingested,blocked,errors,...}\|null, pending, blocked_files}` |
| query | `wiki.py query "<question>"` | `{synthesis, references}` ou `{error:"KB vide…"}` |

**Ingest détaché (important).** Un exec synchrone de plusieurs minutes bloquerait l'agent (timeouts). `wiki.py ingest` lance donc l'ingestion en **sous-processus détaché** et écrit l'état dans `/Wiki_LM/.ingest_state.json` ; `wiki.py status` relit cet état. Cela reprend la discipline async « lancer puis rendre la main » déjà fiabilisée côté prod (le MCP `wiki_ingest` renvoie `{status:"started"}` et traite en tâche de fond).

### B. Skill de l'agent wiki (`~/.openclaw-slm/workspace-wiki/`, remplace le stub)

`AGENTS.md`/skill définitif : un seul outil à connaître (`wiki.py`), **une opération par tâche** reçue de Tiron. Règle async réutilisée : après `ingest`, répondre une fois « ingestion lancée (N en file) » puis **s'arrêter** ; ne pas interroger `status` ni relancer `ingest` de sa propre initiative (`pending>0`/`running:true` juste après = normal).

### C. Skill de délégation Tiron (`~/.openclaw-slm/workspace/skills/`)

Mappe les intentions utilisateur (URL/`/c`, « ingère », question) → `sessions_spawn(agentId="wiki", task="op: …")` puis `sessions_yield`. Aucune logique wiki dans le contexte permanent de Tiron (« Tiron léger »).

---

## Tests

- **Unitaires `wiki.py`** : chaque op renvoie le JSON attendu (harnais `Wiki_LM/tests/`, primitives mockées comme l'existant). Cas : capture (URL+tags), ingest (started / nothing_to_do), status (running / terminé via fichier d'état), query (synthèse / KB vide).
- **E2E manuel** : spawn de l'agent wiki → `capture` d'une URL test non hostile → `ingest` → `status` (jusqu'à terminé) → `query`, contre la KB partagée. Vérifier la chaîne complète et le retour à Tiron.

---

## Différé / connu

- **Anti-injection à l'ingest** : parité prod (fetch direct). Faille présente aussi en prod ; à traiter globalement plus tard.
- **KB partagée prod ↔ SLM** (`/Wiki_LM` RW) : assumé pour cette phase d'exploration ; risque de concurrence d'écriture non traité ici.
- **Phase 2** : ingestion frugale (BERT extractif + plongements) derrière l'interface de résumé d'`ingest.py`.
