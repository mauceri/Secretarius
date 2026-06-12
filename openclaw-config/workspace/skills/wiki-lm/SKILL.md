# wiki-lm — Skill Tiron

Utiliser les outils MCP `wiki-lm` pour gérer le pipeline documentaire Wiki_LM depuis Telegram.

## Quand utiliser

- L'utilisateur envoie une URL ou un hashtag → `wiki_capture`
- L'utilisateur demande d'ingérer les URLs en attente → `wiki_ingest`
- L'utilisateur pose une question sur le wiki → `wiki_query`
- L'utilisateur demande les tags → `wiki_tags`
- L'utilisateur demande combien d'URLs attendent → `wiki_ingest_status`
- L'utilisateur demande de mettre à jour la base de connaissance → `wiki_kb_update`

## Outils

### `wiki_capture(text)`

Capture les URLs et les notes dans `raw/` pour traitement ultérieur.

- Extrait automatiquement les `#hashtags` comme tags
- Les URLs créent des fichiers `.url` (traités par `wiki_ingest`)
- Le texte restant crée un fichier `.md` (note locale)
- Retourne `{files: ["nom-fichier.url", ...]}`

Exemple : `wiki_capture("#linguistique https://example.com Note personnelle")`

**Après `wiki_capture`, ne pas appeler `wiki_ingest` immédiatement sauf si l'utilisateur le demande explicitement.** Le pipeline est asynchrone.

### `wiki_ingest()`

Lance en arrière-plan le traitement des `.url` en attente (fetch → injection-guard → ingest) et **rend la main immédiatement**.

- Aucun paramètre
- Retourne un accusé : `{status: "started", queued: N}` (ou `{status: "already_running"}` si un run est déjà en cours, `{status: "nothing_to_do", queued: 0}` si rien à traiter)
- `queued` = nombre d'éléments mis en file. Ce n'est **pas** un compte d'éléments déjà ingérés : le traitement se poursuit en tâche de fond.

**Comportement obligatoire après `wiki_ingest` :** répondre une seule fois à l'utilisateur (ex. « Ingestion de N éléments lancée en arrière-plan. ») puis **s'arrêter**. Ne PAS interroger `wiki_ingest_status` de votre propre initiative, ne JAMAIS relancer `wiki_ingest`. Le résultat final n'est pas disponible immédiatement, c'est normal.

### `wiki_query(question, top_k=5)`

Interroge le wiki. Retourne `{synthesis: "...", references: ["slug1", ...]}`.

- Si `{error: "KB vide..."}` : suggérer de lancer `wiki_ingest` puis `wiki_kb_update`
- Reformuler la synthèse dans le style de Tiron avant de l'envoyer à l'utilisateur

### `wiki_tags()`

Liste les tags disponibles. Retourne `{tags: [...]}`.

### `wiki_ingest_status()`

État de l'ingestion. **À n'appeler que si l'utilisateur demande explicitement où en est l'ingestion** — jamais spontanément après `wiki_ingest`.

Retourne `{running: bool, last_run: {ingested, blocked, errors, ...} | null, pending: N, blocked_files: [...]}`.

- `running: true` → ingestion encore en cours : répondre « en cours » et s'arrêter (ne pas relancer).
- `running: false` avec `last_run` non nul → ingestion terminée : rapporter `last_run` (ingérés / bloqués / erreurs).
- `pending` : éléments pas encore traités. Juste après un `wiki_ingest`, `pending > 0` et `running: true` sont **normaux**, ce n'est pas un échec.
- `blocked_files` : fichiers bloqués par injection-guard (ne seront pas réingérés automatiquement)

### `wiki_kb_update()`

Met à jour la base de connaissance depuis le dernier clustering.

- Retourne `{status: "ok", clustering: "...", created: N, updated: M, excluded: K}`
- Si `{status: "already_running"}` : une mise à jour est déjà en cours, patienter
- À appeler après une ingestion importante pour mettre à jour les axes thématiques

## Flux typique

```
Utilisateur: "#nlp https://arxiv.org/abs/2406.12345"
Tiron: wiki_capture("#nlp https://arxiv.org/abs/2406.12345")
→ Réponse : "Capturé : 20260527-arxiv-org.url"

Utilisateur: "Ingère"
Tiron: wiki_ingest()
→ {status: "started", queued: 1}
→ Réponse : "Ingestion de 1 élément lancée en arrière-plan." (puis s'arrêter, ne pas vérifier le statut)

Utilisateur: "Que dit le wiki sur l'attention multi-tête ?"
Tiron: wiki_query("Que dit le wiki sur l'attention multi-tête ?")
→ Réponse : [synthèse reformulée]
```

## Comportement en cas d'erreur

- `injection-guard unavailable` dans `blocked_details` : le service injection-guard est arrêté. Signaler à l'utilisateur et lui indiquer de vérifier `systemctl --user status openclaw-injection-guard.service`.
- Erreur de fetch : URL inaccessible (timeout, 404, etc.). Signaler le fichier `.url.error`.
- `{status: "error"}` sur `wiki_kb_update` : aucun clustering disponible — lancer `wiki_ingest` d'abord pour créer des embeddings.
