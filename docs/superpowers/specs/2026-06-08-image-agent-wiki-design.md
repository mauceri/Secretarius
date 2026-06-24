# Spec — Image Docker de l'agent wiki (étape B)

> Session superpowers du 2026-06-08.
> Périmètre : conception de l'image spécialisée de l'agent wiki (Tiron léger,
> architecture déléguée). L'image de Tiron (gog + bind credentials) sera
> traitée dans une session ultérieure, sur le même patron.

---

## 1. Contexte et objectif

L'architecture cible (`docs/architecture/secretarius-architecture-socle.md`)
isole les agents par le **contenu de leur image Docker**, pas par la
tool-policy : ce qu'un agent peut faire dépend des binaires et bibliothèques
présents dans son image (`agents.list[].sandbox.docker.image`,
`tools.exec.host = "sandbox"`). Le mécanisme est validé (Pilier A, 2026-06-05) :
un agent dont l'image contient un binaire marqueur l'exécute, un agent sur
l'image de base échoue.

Cette session conçoit la **première image réelle** sur ce patron : celle de
l'agent wiki, qui doit pouvoir capturer, ingérer et interroger la base de
connaissances Wiki_LM (décision étape A : « Image wiki embarque query.py +
extracteur + embeddings — capture + ingest + query »).

Contrainte de fond identifiée en session : `Wiki_LM/` est un chantier de
développement actif (`CLAUDE.md` — Phase 1 Late Interaction en cours). Une
image qui embarquerait le code Python imposerait un rebuild à chaque
modification — friction incompatible avec ce rythme.

---

## 2. Contenu de l'image : embarqué vs monté

### 2.1 Embarqué dans l'image (figé, change uniquement au rebuild)

- Runtime Python + pip/venv
- Dépendances ML lourdes : `sentence-transformers` (→ torch/transformers) et
  le sous-ensemble pertinent de `Wiki_LM/requirements.txt`. À trier à
  l'implémentation : `fastmcp`, `flask`, `anthropic`, `pytest` semblent
  obsolètes dans la nouvelle architecture (pas de MCP, pas de serveur Flask,
  backend Anthropic non utilisé par le wiki)
- Poids du modèle BGE-M3 (`BAAI/bge-m3`, ~2,3 Go) pré-téléchargés au build,
  pour un démarrage sans dépendance réseau au chargement du modèle

### 2.2 Monté en bind depuis l'hôte (vivant, modifiable sans rebuild)

- **Code** : `/home/mauceric/Secretarius/Wiki_LM/tools` → `/wiki-tools` en
  lecture seule (`:ro`). Aucune raison d'écrire dans le code depuis le
  conteneur ; toute modification sur l'hôte est immédiatement active.
- **Données** : `/home/mauceric/Documents/Arbath/Wiki_LM` → `/Wiki_LM` en
  lecture-écriture (`:rw` — décision étape A, capture/ingest écrivent dans
  la base).

### 2.3 Configuration et secrets : variables d'environnement, pas fichier monté

`Wiki_LM/tools/wiki_paths._load_dotenv()` charge `Wiki_LM/.env` mais **ne
remplace jamais une variable d'environnement déjà définie**
(`if key and key not in os.environ`). On exploite cette priorité : les
variables nécessaires (`WIKI_PATH`, `WIKI_LLM_BACKEND`, `DEEPSEEK_API_KEY`,
etc.) sont injectées via `sandbox.docker.env` — **pas** un bloc `env` générique
au niveau de l'agent : la documentation OpenClaw (`gateway/sandboxing.md`) est
explicite, *« Sandbox exec ne hérite pas de `process.env` de l'hôte ; utilisez
`agents.defaults.sandbox.docker.env` (ou une image personnalisée) pour les
clés des skills »*. Comme `binds`, `docker.env` se règle globalement ou par
agent (`agents.list[].sandbox.docker.env`), les deux niveaux étant fusionnés.
Pas de fichier secret monté dans le conteneur — `Wiki_LM/.env` reste sur
l'hôte, inutilisé par l'agent.

---

## 3. Image de base

`FROM openclaw-sandbox:bookworm-slim`. Vérifié en session (`docker history`) :
cette image est une simple base Debian Bookworm slim (211 Mo) avec
`bash, ca-certificates, curl, git, jq, python3, ripgrep`, un utilisateur
`sandbox`, `WORKDIR /home/sandbox`. Rien de spécifique à OpenClaw à recréer —
`python3` y est déjà présent. Repartir d'une image Python officielle
obligerait à reconstruire cet environnement pour un gain nul.

---

## 4. Structure du Dockerfile : mono-étape

Un seul Dockerfile (`openclaw-config/Dockerfile.wiki`) :

```
FROM openclaw-sandbox:bookworm-slim
# installer pip/venv
# pip install -r <sous-ensemble de requirements.txt>
# pré-télécharger BGE-M3 :
#   python3 -c "from sentence_transformers import SentenceTransformer; \
#               SentenceTransformer('BAAI/bge-m3')"
```

**Approche retenue plutôt qu'un build multi-étapes** : le poids dominant de
l'image (~5 Go avec torch + poids du modèle) est incompressible — ni la
compilation ni les caches de build n'y contribuent significativement. Un
multi-étapes réduirait la taille finale de quelques centaines de Mo pour un
Dockerfile nettement plus complexe à maintenir : gain marginal, complexité non
justifiée (image locale, mono-machine, pas de registre à optimiser).

---

## 5. Câblage dans la configuration OpenClaw

Nouvelle entrée dans `agents.list[]`, sur le modèle du Pilier A :

```json
{
  "id": "wiki",
  "model": { "primary": "euria/mistralai/Mistral-Small-4-119B-2603" },
  "sandbox": {
    "docker": {
      "image": "secretarius-wiki:latest",
      "dangerouslyAllowExternalBindSources": true,
      "binds": [
        "/home/mauceric/Secretarius/Wiki_LM/tools:/wiki-tools:ro",
        "/home/mauceric/Documents/Arbath/Wiki_LM:/Wiki_LM:rw"
      ],
      "env": {
        "WIKI_PATH": "/Wiki_LM",
        "WIKI_LLM_BACKEND": "openai",
        "OPENAI_BASE_URL": "https://api.infomaniak.com/2/ai/${EURIA_PRODUCT_ID}/openai/v1",
        "OPENAI_API_KEY": "${EURIA_API_KEY}",
        "OPENAI_MODEL": "mistralai/Mistral-Small-4-119B-2603",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1"
      }
    }
  },
  "tools": {
    "exec": { "host": "sandbox" }
  }
}
```

**Deux découvertes faites en validation, absentes de la conception initiale :**

- **`dangerouslyAllowExternalBindSources: true`** — par défaut, OpenClaw
  n'autorise les sources de bind que sous le répertoire `workspace` de l'agent
  (`/home/mauceric/.openclaw-slm/workspace-wiki`) ; toute source en dehors est
  rejetée (« outside allowed roots »), *avant même* la liste noire des chemins
  dangereux décrite plus bas. Les deux binds de cette conception
  (`Wiki_LM/tools` et `Documents/Arbath/Wiki_LM`) sont tous deux externes au
  workspace — sans ce drapeau, aucun des deux ne peut être monté, et la
  conception entière (binds vivants, cf. §2.2) est inopérante. La liste noire
  des chemins dangereux (`/etc`, `~/.ssh`, etc.) reste active par-dessus :
  le drapeau élargit l'ensemble des sources *candidates*, pas l'ensemble des
  sources *autorisées en clair*.
- **Credential OpenClaw pour le modèle propre de l'agent** — distinct de
  `OPENAI_API_KEY` (§2.3, qui alimente uniquement les appels internes des
  outils Wiki_LM dans le sandbox), l'agent a besoin d'un credential stocké
  dans son propre magasin OpenClaw
  (`~/.openclaw-slm/agents/wiki/agent/auth-profiles.json`, format
  `{"version": 1, "profiles": {"euria:default": {"type": "api_key",
  "provider": "euria", "key": "<valeur de EURIA_API_KEY>"}}}`, miroir du
  motif déjà en place côté prod) pour que *l'agent lui-même* (son
  raisonnement, pas les outils qu'il invoque) puisse appeler son modèle
  `euria/mistralai/Mistral-Small-4-119B-2603`. Sans ce credential,
  `FailoverError: No API key found for provider "euria"` empêche même la
  création du conteneur sandbox. Pas de commande CLI non interactive pour le
  provisionner — fichier écrit directement, à l'image du motif prod.

Confirmé dans la documentation OpenClaw (les points laissés ouverts à la fin
de la session de design sont désormais tranchés) :
- **Format des binds** : `"host:container:mode"`
  (`sandbox.docker.binds`, fusion entre `agents.defaults` et `agents.list[]`)
- **Variables d'environnement du sandbox** : `sandbox.docker.env`, *pas* un
  bloc `env` générique au niveau de l'agent (cf. §2.3) — même mécanisme de
  fusion global/par-agent que `binds`
- **`tools.exec.host` par agent** : `gateway/cli-backends.md` confirme que
  *« per-agent `agents.list[].tools.exec` settings override global
  `tools.exec` for that agent »*. Le template actuel force
  `"tools.exec.host": "gateway"` globalement (nécessaire à `main`/Tiron pour
  exécuter `gog`, qui vit sur l'hôte — règle « gog chez Tiron »). Plutôt que
  de toucher ce réglage global, l'agent `wiki` porte son propre override
  `"tools": { "exec": { "host": "sandbox" } }` — local, sans effet de bord sur
  Tiron. C'est la solution retenue (cf. JSON ci-dessus).

**`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`** : ajoutés suite à un constat fait en
validation (Task 3 du plan) — sans ces variables, `sentence-transformers`
effectue une vérification réseau (`HEAD` sur `adapter_config.json`) avant
d'utiliser les poids déjà présents en cache, à chaque chargement du modèle.
En déploiement le réseau est disponible (Euria), donc cette vérification
réussirait et le chargement utiliserait quand même le cache local — mais elle
ajoute un aller-retour réseau évitable au démarrage, contraire à l'objectif du
§2.1 (« démarrage sans dépendance réseau au chargement du modèle »). Forcer le
mode hors-ligne fait sauter cette vérification : le modèle se charge
directement depuis le cache embarqué dans l'image.

**Garde-fous des binds (`gateway/sandboxing.md`)** : OpenClaw bloque les
sources dangereuses (`/etc`, `/proc`, `/sys`, `/dev`, `docker.sock`) et les
racines de credentials usuelles (`~/.config`, `~/.ssh`, `~/.aws`, etc.). Les
deux binds prévus ici (`Wiki_LM/tools` et `Documents/Arbath/Wiki_LM`) ne
touchent aucune racine bloquée.

---

## 6. Construction et maintenance

- Construction manuelle : `docker build -t secretarius-wiki:latest -f
  openclaw-config/Dockerfile.wiki .`
- **Rebuild nécessaire** si `requirements.txt` change ou changement de modèle
  d'embeddings (BGE-M3 → autre)
- **Pas de rebuild** pour les modifications dans `Wiki_LM/tools/` (bind-monté,
  vivant)
- Intégration à `install.sh` : différée. Remarque ouverte de l'utilisateur en
  session — « il faudra peut-être y penser ». Cohérent avec la décision étape A
  de différer la migration OpenClaw 6.x « quand le produit sera prêt à être
  présenté » : même horizon pour l'industrialisation du build d'image.

---

## 7. Validation

Reprend le patron de test du Pilier A (binaire marqueur exécuté par l'agent
dédié, échec pour `main`) :

1. Construire l'image, démarrer un agent `wiki` pointant dessus
2. **Test d'isolation** : l'agent `wiki` peut importer `sentence_transformers`
   et charger BGE-M3 sans accès réseau (poids déjà dans l'image) ; l'agent
   `main` ne le peut pas (dépendances absentes de son image)
3. **Test du bind code** : modifier une fonction dans `Wiki_LM/tools/` sur
   l'hôte, vérifier que l'agent `wiki` voit le changement sans rebuild ni
   redémarrage du conteneur
4. **Test du bind données** : vérifier lecture/écriture dans `/Wiki_LM`
   (capture/ingest)

### Résultats (exécutés le 2026-06-08)

| Test | Résultat |
|---|---|
| Build de l'image (`secretarius-wiki:latest`) | ✅ 10,5 Go (au-delà de l'estimation ~5 Go du §4 — torch + transformers + poids BGE-M3 ; pas un problème, image locale mono-machine) |
| Chargement hors-ligne BGE-M3 (`--network none`, `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`) | ✅ dimension 1024 — voir note ci-dessous sur ces deux variables |
| Isolation — `wiki` charge BGE-M3 hors-ligne dans son conteneur sandbox réel | ✅ dimension 1024 |
| Isolation — `main` échoue sur `import sentence_transformers` | ✅ `ModuleNotFoundError: No module named 'sentence_transformers'` |
| Bind code — fichier créé/supprimé sur l'hôte visible immédiatement dans `/wiki-tools` | ✅ visible et disparaît sans rebuild ni redémarrage |
| Bind données — écriture/lecture/suppression cohérente entre `/Wiki_LM` (conteneur) et `Documents/Arbath/Wiki_LM` (hôte) | ✅ contenu identique des deux côtés, suppression propagée |

**Note sur `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`** : le premier essai du test
de chargement hors-ligne (sans ces variables) a échoué — `sentence-transformers`
tente une vérification réseau (`HEAD` sur `adapter_config.json`) avant
d'utiliser le cache local, et cette vérification échoue brutalement sous
`--network none`. Cette découverte a conduit à ajouter ces deux variables à la
fois à la commande de test et à l'environnement de l'agent (§5) — réalisant
ainsi pleinement l'objectif du §2.1 (« démarrage sans dépendance réseau »),
y compris en déploiement réel où le réseau est disponible mais où la
vérification serait un aller-retour évitable.

---

## 8. Hors périmètre (sessions suivantes)

- Image de l'agent Tiron (binaire `gog` + bind credentials `~/.config/gogcli/`)
  — même patron, à dérouler séparément
- Conception du skill wiki lui-même (comment le skill orchestre
  capture/ingest/query, intégration du filtrage anti-injection façon Scout)
- Intégration du build d'image à `install.sh`

**Question tranchée pendant l'écriture du plan (2026-06-08)** : le backend LLM
pour les appels internes de `ingest.py` (`llm.py`, génération de
concepts/entités) passera par le modèle natif de l'agent — Euria — via le
backend générique `openai` de `llm.py` (compatible OpenAI). D'où le câblage
`WIKI_LLM_BACKEND=openai` + `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`OPENAI_MODEL`
pointant sur Euria en §5 : réutilise les secrets `EURIA_API_KEY`/
`EURIA_PRODUCT_ID` déjà présents dans `gateway-slm.systemd.env`, sans
provisionner de nouveau secret (`DEEPSEEK_API_KEY` abandonné pour ce canal).
