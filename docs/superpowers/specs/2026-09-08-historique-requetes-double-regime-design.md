# Historique des requêtes — régime par canal (Telegram/WebChat/Obsidian) pour /q

- Date : 2026-09-08
- Statut : design validé, en attente d'exécution
- Auteur : Christian Mauceri + Claude

## Problème

Les réponses de `/q` (et de toute requête en langage naturel routée vers `wiki
query`) supposent une lecture dans Obsidian : synthèse complète en Markdown,
sans limite de longueur. Relayées telles quelles sur Telegram, elles arrivent
tronquées à 4000 caractères (`formatWikiResult` → `out.slice(0, 4000)` dans
`derisk-deleg/src/index.ts`) et affichées en Markdown brut (le canal Telegram
d'OpenClaw envoie en `parse_mode: "HTML"`, jamais converti — sujet déjà
diagnostiqué séparément, non traité ici).

Il existe donc en réalité plusieurs **régimes de lecture**, jamais distingués
dans le code : un régime Telegram (message court, consultation rapide, rendu
HTML strict), un régime WebChat (rend le Markdown correctement, mais reste un
fil de chat) et un régime Obsidian (note complète, navigable, sans contrainte
de taille). Ce design fait cette distinction explicite au lieu de servir le
même texte à tous.

## Objectif

- Sur Telegram : un résumé court + un lien vers la réponse complète, jamais
  de synthèse tronquée en plein message.
- Dans Obsidian : la réponse complète reste disponible, mais dans une note
  séparée plutôt qu'insérée brutalement dans la note en cours de lecture.
- Toute requête, quelle que soit son origine (Telegram, WebChat ou
  Obsidian), doit produire un enregistrement horodaté consultable — pas
  seulement celles passées par un flag `--save` explicite. Ceci est un
  effet de bord gratuit du point d'écriture unique (voir plus bas) : aucune
  logique supplémentaire par canal n'est nécessaire pour ce point précis.
- Sur WebChat (app web/Control UI d'OpenClaw), qui rend le Markdown
  correctement contrairement à Telegram : régime complet conservé (pas de
  brief+lien), car ce n'est pas le même problème de rendu cassé.

## Périmètre

**Inclus** : l'opération `query` (`/q` et son routage en langage naturel),
côté `Wiki_LM/tools/query.py` (source unique), `wiki.py` (façade CLI/sandbox,
utilisée par Telegram et WebChat), `server.py` (façade Flask Obsidian),
`derisk-deleg/src/wiki-ops.ts` et `index.ts` (formatage par canal),
les deux templates Templater Obsidian.

**Exclus** :
- `/r` (search) : renvoie déjà des extraits courts (5 résultats × excerpt),
  pas concerné par le problème de verbosité.
- Le mécanisme `--save`/`saved_slug`/`_save_synth` existant dans `query.py` :
  reste inchangé. C'est une fonctionnalité distincte et opt-in (page polie,
  régénérée par un second appel LLM, écrite **dans** `wiki/` donc indexée et
  permanente). Le nouveau mécanisme de cette spec est automatique, léger
  (pas de régénération de page), et écrit **hors** de `wiki/` (non indexé).
- Le format Markdown/HTML de Telegram (bug séparé, déjà diagnostiqué).
- La verbosité constatée un jour après un `/c` (« dump Hoppe ») : son origine
  exacte n'a pas pu être confirmée (`/q` ou autre chose). Ce design corrige
  `/q` avec certitude. S'il se reproduit après déploiement, c'est un autre
  mécanisme (probablement l'agent Tiron narrant lui-même le contenu ingéré
  hors du chemin déterministe) — chantier séparé.
- Rendre l'historique cherchable par `/q`/`/r` : reporté (décision explicite,
  à revisiter plus tard si besoin).

## Décisions actées

| Sujet | Décision |
|-------|----------|
| Emplacement historique | `Wiki_LM/historique/`, sibling de `wiki/` et `raw/` sous `WIKI_PATH` — hors de l'arbre indexé par `WikiSearch` (`wiki_root/wiki`) et hors de la file d'ingestion (`raw/`) |
| Écriture | Systématique à chaque `query()`, pas derrière un flag |
| Contenu Telegram | Résumé généré par un second appel LLM court (1-2 phrases, `max_tokens` réduit), avec repli sur troncature simple (~300 car.) de la synthèse si l'appel échoue |
| Lien Telegram | URI `obsidian://open?vault=<nom coffre>&file=<chemin encodé>` |
| Point d'écriture unique | `WikiQuery.query()` dans `query.py` — `wiki.py::op_query` et `server.py::handle_query` en héritent sans dupliquer la logique |
| Régime Obsidian | Le template ouvre la note d'historique déjà écrite par le serveur dans un nouvel onglet, au lieu d'insérer le texte dans la note courante |
| Détection du canal (Telegram vs WebChat) | Le hook `before_agent_reply` reçoit un second paramètre `ctx` (`PluginHookAgentContext`) exposant `ctx.messageProvider`, actuellement ignoré par `derisk-deleg`. On le lit pour choisir le régime : `"webchat"` → complet, tout le reste (`"telegram"`, absent) → bref+lien. La chaîne exacte `"webchat"` est confirmée présente dans le bundle OpenClaw comme identifiant de canal, mais pas vérifiée comme étant précisément la valeur de `ctx.messageProvider` en conditions réelles — à confirmer par un test manuel (log de `ctx.messageProvider` sur un message WebChat réel) en tout début d'implémentation, avant d'écrire la logique qui en dépend |
| Régime par défaut | Bref+lien (comportement le plus sûr) partout où le canal n'est pas identifiable — notamment le chemin des outils enregistrés (`wiki_query` etc.), que l'agent LLM peut invoquer de sa propre initiative et pour lequel le SDK n'expose aucun contexte de canal à `execute()` |

## Architecture

### `query.py` — point d'écriture unique

`QueryResult` gagne deux champs :

```python
@dataclass
class QueryResult:
    question: str
    text: str
    references: list[str] = field(default_factory=list)
    saved_slug: str = ""      # inchangé (mécanisme --save existant)
    history_slug: str = ""    # nouveau
    brief: str = ""           # nouveau
```

Dans `WikiQuery.query()`, après le calcul habituel de `synthesis`/`references`
(logique de recherche + prompt de synthèse inchangée), avant le `return` :

```python
history_slug = self._write_history(question, synthesis, references)
brief = self._generate_brief(question, synthesis)
result = QueryResult(question=question, text=synthesis, references=references,
                     history_slug=history_slug, brief=brief)
if save:
    result.saved_slug = self._save_synth(...)   # inchangé
    self._append_log("query", question)
return result
```

`_write_history(question, synthesis, references)` :
- slug = `f"{timestamp()}-{slugify(question)}"` (réutilise `timestamp()` et
  `slugify()` déjà présents dans `capture.py`, mêmes conventions que les
  autres fichiers du projet).
- crée `Wiki_LM/historique/` si absent (`mkdir(parents=True, exist_ok=True)`).
- écrit un fichier plat, même structure que `QueryResult.__str__()` actuel
  (en-tête `Q : ...` / `Sources : [[...]]` / texte complet) — pas de
  frontmatter, rien d'indexable.
- retourne le slug (sans extension).

`_generate_brief(question, synthesis)` :
- un appel `self.llm.complete(...)` avec un prompt court dédié et un
  `max_tokens` faible (ex. 120).
- `try/except` : en cas d'échec, repli sur `synthesis[:300]` — ne doit
  **jamais** faire échouer `query()`.

### `wiki.py::op_query` — façade Telegram/sandbox

```python
def op_query(question: str) -> dict:
    try:
        result = WikiQuery(_wiki_root()).query(question)
        if not result.text:
            return {"error": "KB vide — lancer ingest d'abord"}
        obsidian_uri = _build_obsidian_uri(result.history_slug)
        return {
            "synthesis": result.text,        # inchangé, gardé pour tout appelant CLI direct
            "references": result.references,
            "brief": result.brief,
            "obsidian_uri": obsidian_uri,
        }
    except Exception as exc:
        return {"error": str(exc)}
```

`_build_obsidian_uri(history_slug)` : vault root = `_wiki_root().parent`, nom
du coffre = `vault_root.name` (aujourd'hui `Secretarius`), chemin relatif =
`Wiki_LM/historique/<history_slug>`. Encodage via `urllib.parse.quote` sur le
nom du coffre et le chemin :

```python
f"obsidian://open?vault={quote(vault_root.name)}&file={quote(f'Wiki_LM/historique/{history_slug}')}"
```

### `server.py::handle_query` — façade Flask/Obsidian

Réponse JSON étendue avec les deux nouveaux champs, en plus de l'existant :

```python
return jsonify({
    "text": result.text,
    "references": result.references,
    "saved_slug": result.saved_slug,
    "history_slug": result.history_slug,
    "brief": result.brief,
})
```

### `derisk-deleg/src/wiki-ops.ts` — formatage par régime

`formatWikiResult` et `runWikiOp` gagnent un paramètre optionnel
`regime: "brief" | "full" = "brief"` (dernier paramètre, valeur par défaut
pour ne pas casser les appelants existants). Cas `"query"` :

```typescript
case "query": {
  if (regime === "full") {
    return typeof json?.synthesis === "string" && json.synthesis.trim()
      ? json.synthesis
      : "Réponse wiki vide ou inattendue.";
  }
  const brief = typeof json?.brief === "string" ? json.brief.trim() : "";
  const uri = typeof json?.obsidian_uri === "string" ? json.obsidian_uri : "";
  if (!brief && !uri) return "Réponse wiki vide ou inattendue.";
  return [brief, uri].filter(Boolean).join("\n\n");
}
```

Dans `index.ts`, la branche `before_agent_reply` qui gère `action.kind ===
"wiki"` (dispatch déterministe — c'est le chemin réel d'un `/q` tapé ou
routé en langage naturel, sur n'importe quel canal) lit `ctx.messageProvider`
et détermine `regime` avant d'appeler `runWikiOp(api, action.op, routed.args,
undefined, regime)`. Le hook doit donc être modifié pour capturer son
second paramètre (`api.on("before_agent_reply", async (event, ctx) =>
{ ... })`, actuellement `ctx` n'est jamais lu).

Les outils enregistrés (`wiki_query`, etc.), que l'agent LLM peut appeler
de sa propre initiative, n'ont pas accès à cette information (signature
SDK `execute(toolCallId, params, signal?, onUpdate?)`, sans contexte de
canal) — ils utilisent donc le régime par défaut (`"brief"`), un choix
délibéré et documenté, pas un oubli.

Remplace le comportement actuel (`json.synthesis` verbatim, envoyé sans
distinction à tous les canaux). C'est le point de correction exact du dump
verbeux sur `/q` via Telegram — `synthesis` reste dans le JSON dans tous
les cas (nécessaire au régime `"full"`, inutilisé en régime `"brief"`).

### Templates Obsidian (`obsidian_template_wikilm.md`, `_android.md`)

Remplacent la construction de `block` + `tR += block` par : appeler
`/query` comme aujourd'hui, récupérer `history_slug` dans la réponse, puis
ouvrir directement le fichier correspondant dans un nouvel onglet :

```javascript
const file = app.vault.getAbstractFileByPath(`Wiki_LM/historique/${data.history_slug}.md`);
if (file) {
    await app.workspace.getLeaf(true).openFile(file);
} else {
    new Notice("Wiki_LM : note d'historique introuvable", 5000);
}
```

Le template n'insère plus rien dans la note en cours de lecture — la note
d'historique, déjà écrite complète par `server.py` via `query.py`, est la
seule source de vérité. Pas de duplication de logique de formatage côté
JavaScript.

## Erreurs et cas limites

- Échec du second appel LLM (`brief`) : repli silencieux sur troncature,
  jamais d'exception remontée à l'appelant.
- Échec d'écriture du fichier historique (disque plein, permissions) :
  laissé remonter tel quel — un tel échec doit être visible (comme les
  autres erreurs I/O de `query.py` aujourd'hui), pas masqué.
- `historique/` absent au premier appel : créé à la volée
  (`mkdir(parents=True, exist_ok=True)`), pas de setup manuel requis.
- Note d'historique introuvable côté template (désync éventuelle
  vault/serveur) : notification d'erreur explicite, pas de silence.

## Tests

- `Wiki_LM/tests/test_query.py` (nouveau fichier — `query.py` n'a aujourd'hui
  aucun test dédié, seulement une couverture indirecte via
  `test_wiki_cli.py`) : `history_slug` non vide et fichier réellement écrit
  avec le bon contenu ; `brief` non vide en fonctionnement normal ; repli sur
  troncature si le LLM de brief échoue (LLM stub qui lève une exception) ;
  mécanisme `--save` existant non affecté (régression).
- `Wiki_LM/tests/test_wiki_cli.py` : `op_query` inclut `brief` et
  `obsidian_uri` dans le JSON retourné ; URI correctement encodée.
- `derisk-deleg/src/wiki-ops.test.ts` (fichier existant) : cas `query` avec
  `brief`+`obsidian_uri` présents (régime par défaut/`"brief"`) ; cas
  `brief` vide mais `obsidian_uri` présent (et inversement) ; cas des deux
  absents (message de repli) ; régime `"full"` renvoie `synthesis` verbatim,
  y compris quand `brief`/`obsidian_uri` sont aussi présents (ne doit pas
  les préférer par erreur).
- `derisk-deleg/src/index.test.ts` (fichier existant) : la branche wiki du
  dispatch déterministe choisit `regime: "full"` quand
  `ctx.messageProvider === "webchat"`, et `"brief"` sinon (Telegram, canal
  absent/inconnu) — régression à couvrir explicitement puisque c'est un
  comportement par défaut sur lequel toute erreur serait silencieuse.

## Documentation

`docs/components/wiki-lm.md` et `docs/components/obsidian.md` à mettre à
jour : nouveau répertoire `historique/`, nouveaux champs JSON de `/query`,
nouveau comportement des templates (ouverture en nouvel onglet plutôt
qu'insertion).
