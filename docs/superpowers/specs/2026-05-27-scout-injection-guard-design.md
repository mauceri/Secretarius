# Design — Scout : refonte injection guard

**Date :** 2026-05-27  
**Statut :** approuvé  
**Périmètre :** injection-guard.service + scout-watcher refondu + Scout LLM simplifié  
**Hors périmètre (phase 2) :** proxy Gmail MCP (garantie structurelle emails)

---

## Contexte

Scout est l'agent isolé de Secretarius chargé de lire les sources externes (pages web) à la place de Tiron. La défense anti-injection actuelle est purement comportementale : règles dans SOUL.md, regex inline dans scout-watcher, skill prompt-injection-guard (qui ne se déclenche pas sur les résultats d'outils). Scout produit des résumés qu'il ne devrait pas faire, et il n'existe pas de moteur de détection fiable.

Objectifs du redesign :
- Détection hybride déterministe (regex) + ML (DeBERTa) dans un service persistant
- Scout LLM réduit à un thin wrapper JSON — plus de résumés
- Fail-safe strict : si le guard est indisponible, tout est bloqué
- Suite de tests complète, zéro réseau, zéro GPU en CI

---

## Architecture

```
[curl / scout-watcher]        [Gmail MCP — phase 2]
         |
         v
    HTML brut / texte
         |
         v
injection-guard.service
(Flask, localhost:8990)
┌─────────────────────────┐
│ 1. nettoyage HTML        │
│ 2. regex rapide          │
│ 3. DeBERTa si ambigu     │
└─────────────────────────┘
         |
  ┌──────┴──────┐
  v             v
bloqué         OK
{blocked:true  {blocked:false,
 reason:...}    risk:"low|medium",
                clean_text:"...",
                full_content:"..."}
         |
         v
    Scout LLM
 (thin wrapper JSON)
         |
         v
       Tiron
```

Scout LLM est quasi-stateless : il reçoit le JSON structuré d'injection-guard, l'enveloppe dans son format de sortie, retourne à Tiron. Il ne résume pas, n'analyse pas.

---

## Composants

### `injection-guard.service` (nouveau)

- Script Python `injection-guard.py`, installé dans `~/.local/bin/`
- Unité systemd user `openclaw-injection-guard.service`
- Flask sur `localhost:8990`, endpoint `POST /check`
- Modèle `protectai/deberta-v3-base-prompt-injection-v2` chargé une fois au démarrage
- Regex compilées au démarrage
- Versionnés dans `openclaw-config/` (source + template systemd)
- `install.sh` : installe le service, télécharge le modèle au premier lancement

**Endpoint `POST /check`**

Requête :
```json
{
  "type": "html",
  "content": "..."
}
```

Réponse bloquée :
```json
{
  "blocked": true,
  "reason": "pattern1, pattern2"
}
```

Réponse OK :
```json
{
  "blocked": false,
  "risk": "low",
  "clean_text": "...",
  "full_content": "..."
}
```

### `scout-watcher` (modifié)

Pipeline après fetch curl :
1. `POST /check` avec le HTML brut
2. Si `blocked: true` → écrit `tasks/done/task.json = {blocked, reason}`, fin (Scout LLM n'est pas appelé)
3. Si OK → injecte `fetched_content = {risk, clean_text, full_content}` dans task.json, move to `tasks/done/`, signal Scout LLM

Gestion des emails (phase 1 — comportemental) :
- Tiron extrait le texte via Gmail MCP, passe à Scout via `sessions_spawn(task="check_email: <texte>")`
- Scout-watcher détecte le type `check_email`, appelle `/check` avec `{type:"text", content:...}`
- Même logique de retour

### `agents/scout/workspace/SOUL.md` (simplifié)

Scout LLM :
- Ne fait **aucun résumé**, aucune analyse propre
- Si `fetched_content.blocked == true` → retourne `{blocked, reason}` sans rien ajouter
- Sinon → retourne `{source, retrieved_at, risk, clean_text, full_content?, warnings[]}`
- Ne jamais inventer de contenu

Format de sortie Scout LLM :
```json
{
  "source": "URL ou chemin source",
  "retrieved_at": "ISO8601",
  "risk": "low|medium",
  "clean_text": "<UNTRUSTED> texte propre sans HTML",
  "full_content": "<UNTRUSTED> HTML verbatim (présent uniquement si demandé)",
  "warnings": []
}
```

### `openclaw-config/workspace/skills/scout/SKILL.md` (mis à jour)

- Format de retour mis à jour (suppression de `summary` et `raw_excerpt`, ajout de `risk` et `clean_text`)
- Règle : "Toujours lire `risk` et `warnings` en premier"
- Note phase 2 : proxy Gmail MCP pour garantie structurelle emails

---

## Flux de données

### Flux 1 — Page web

```
scout-watcher poll tasks/pending/
  → lit task.json : {url, instructions}
  → curl fetch (max 30s)
  → POST /check {type:"html", content:"<html>..."}
  → si bloqué : tasks/done/{blocked,reason} — fin
  → si ok     : tasks/done/{risk, clean_text, full_content}
  → signal Scout LLM
  → Scout LLM formate et retourne à Tiron
```

### Flux 2 — Email (phase 1, comportemental)

```
Tiron reçoit demande → extrait texte via Gmail MCP
  → sessions_spawn(agentId="scout", task="check_email: <texte>")
  → scout-watcher détecte type "check_email"
  → POST /check {type:"text", content:"<texte>"}
  → même logique de retour
```

### Flux 3 — Scout LLM

```
Scout LLM reçoit fetched_content depuis tasks/done/
  → si blocked:true → {blocked, reason}
  → si ok → {source, retrieved_at, risk, clean_text, full_content?, warnings[]}
```

---

## Gestion d'erreurs

| Situation | Comportement |
|---|---|
| injection-guard indisponible (timeout 3s) | fail-safe : `{blocked:true, reason:"injection-guard unavailable"}` |
| DeBERTa échec de chargement | service démarre en mode regex-only ; ambigus → `risk:"medium"` |
| Fetch curl échoue | `{error:"fetch_error", reason:"..."}` dans tasks/done/ |
| Payload >15 000 chars | troncature avant `/check`, flag `truncated:true` dans retour |
| Encoding non-UTF8 | conversion UTF-8 avec `errors='replace'` |
| Scout LLM ajoute du contenu non demandé | interdit par SOUL.md ; surface d'hallucination quasi nulle |

---

## Stratégie de test

```
injection-guard/
├── injection_guard.py
└── tests/
    ├── test_regex.py
    ├── test_html_cleaning.py
    ├── test_service.py
    ├── test_scout_watcher.py
    └── fixtures/
        ├── payloads_blocked.json
        ├── payloads_safe.json
        └── html_samples/
```

**test_regex.py**
- Chaque pattern bloquant → `blocked:true` avec reason correct
- Faux positifs courants : "ignore" / "instructions" / "rôle" dans contexte juridique normal → `blocked:false`
- Patterns ambigus → `risk:"medium"`
- Cas limites : vide, unicode, très longue chaîne

**test_html_cleaning.py**
- `<script>`, `<style>` supprimés avec contenu
- Texte invisible supprimé : `display:none`, `visibility:hidden`, `color:white`, `font-size:0`
- Entités HTML décodées
- Résultat propre sans artefacts

**test_service.py**
- DeBERTa mocké : `model.predict` patché avec scores contrôlés
- `POST /check` → structure JSON vérifiée pour cas bloqué et OK
- Timeout client → comportement fail-safe vérifié

**test_scout_watcher.py**
- Mock Flask sur port 8990
- Tâche pending → poll → appel `/check` → fichier done vérifié
- Cas bloqué : done contient `{blocked:true}`, Scout LLM non appelé
- Payload >15 000 chars : troncature + `truncated:true`

**Contraintes CI :** zéro réseau, zéro GPU, zéro modèle réel chargé.  
**Couverture cible :** >90% `injection_guard.py`, >80% partie Python scout-watcher.

---

## Phase 2 — Proxy Gmail MCP

Pour un usage commercial (traitement de la correspondance "tout venant"), la règle SOUL.md est insuffisante : Tiron a techniquement accès au corps des emails via Gmail MCP et peut l'interpréter sans passer par Scout.

Solution structurelle prévue : un **proxy MCP Gmail** Python (FastMCP) qui expose à Tiron les outils de recherche/métadonnées, mais intercale injection-guard sur `get_body(message_id)` avant de retourner le contenu. Le vrai Gmail MCP est retiré des outils de Tiron.

Prérequis : injection-guard.service opérationnel et validé.

---

## Fichiers à créer / modifier

| Fichier | Action |
|---|---|
| `openclaw-config/injection-guard.py` | Créer |
| `openclaw-config/openclaw-injection-guard.service` | Créer |
| `openclaw-config/install.sh` | Modifier (install service + téléchargement modèle) |
| `openclaw-config/scout-watcher` | Modifier (appel /check, fail-safe) |
| `openclaw-config/agents/scout/workspace/SOUL.md` | Modifier (thin wrapper, suppression résumés) |
| `openclaw-config/workspace/skills/scout/SKILL.md` | Modifier (nouveau format de retour) |
| `injection-guard/tests/` | Créer (suite complète) |
