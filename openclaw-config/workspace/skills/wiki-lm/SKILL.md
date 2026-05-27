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

Traite tous les `.url` en attente : fetch → injection-guard → ingest.

- Aucun paramètre
- Retourne `{ingested: N, blocked: M, errors: K, blocked_details: [...], error_details: [...]}`
- Si `blocked > 0` : signaler à l'utilisateur quels fichiers ont été bloqués et pourquoi
- Si `errors > 0` : signaler les erreurs de fetch

### `wiki_query(question, top_k=5)`

Interroge le wiki. Retourne `{synthesis: "...", references: ["slug1", ...]}`.

- Si `{error: "KB vide..."}` : suggérer de lancer `wiki_ingest` puis `wiki_kb_update`
- Reformuler la synthèse dans le style de Tiron avant de l'envoyer à l'utilisateur

### `wiki_tags()`

Liste les tags disponibles. Retourne `{tags: [...]}`.

### `wiki_ingest_status()`

État de la file d'attente. Retourne `{pending: N, blocked_files: [...]}`.

- `pending` : URLs prêtes à être ingérées au prochain `wiki_ingest()`
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
→ Réponse : "Ingéré : 1 article. Aucun blocage."

Utilisateur: "Que dit le wiki sur l'attention multi-tête ?"
Tiron: wiki_query("Que dit le wiki sur l'attention multi-tête ?")
→ Réponse : [synthèse reformulée]
```

## Comportement en cas d'erreur

- `injection-guard unavailable` dans `blocked_details` : le service injection-guard est arrêté. Signaler à l'utilisateur et lui indiquer de vérifier `systemctl --user status openclaw-injection-guard.service`.
- Erreur de fetch : URL inaccessible (timeout, 404, etc.). Signaler le fichier `.url.error`.
- `{status: "error"}` sur `wiki_kb_update` : aucun clustering disponible — lancer `wiki_ingest` d'abord pour créer des embeddings.
