# Image Docker de l'agent wiki — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire l'image Docker `secretarius-wiki:latest`, l'embarquer dans un nouvel agent `wiki` câblé sur l'instance OpenClaw slm, et valider les quatre garanties du spec (isolation, bind code vivant, bind données rw, chargement du modèle hors-ligne).

**Architecture:** Mono-étape (`Dockerfile.wiki` part de `openclaw-sandbox:bookworm-slim`, installe les dépendances Python du sous-ensemble pertinent de `Wiki_LM/requirements.txt`, pré-télécharge BGE-M3). Code (`Wiki_LM/tools`) et données (`Documents/Arbath/Wiki_LM`) montés en bind — pas embarqués. Agent `wiki` ajouté à `agents.list[]` avec override `tools.exec.host = "sandbox"` local (la config globale `gateway` reste intacte pour Tiron/`gog`).

**Tech Stack:** Docker, Python 3 (pip + `--break-system-packages`, PEP 668), `sentence-transformers`/BGE-M3, OpenClaw 5.12 (instance slm, `~/.openclaw-slm/`).

**Référence :** `docs/superpowers/specs/2026-06-08-image-agent-wiki-design.md` (spec approuvé).

---

## Contexte pour l'exécutant

- L'instance slm tourne en service systemd user `openclaw-gateway-slm.service`,
  config `~/.openclaw-slm/openclaw.json` (généré depuis
  `openclaw-config/openclaw-slm.json.template`, secrets dans
  `~/.openclaw-slm/gateway.systemd.env`). `${EURIA_API_KEY}` et
  `${EURIA_PRODUCT_ID}` y sont déjà présents — pas de nouveau secret à
  provisionner.
- Le mécanisme d'isolation par image est déjà prouvé (Pilier A, 2026-06-05) :
  un agent dont l'image contient un binaire/bibliothèque l'exécute, un agent
  sur l'image de base échoue. Ce plan applique ce patron à un cas réel.
- **Convention de test retenue** : déclencher un tour de l'agent `wiki` via
  `openclaw agent --agent wiki --message "ping" --json` (cela force la
  création du conteneur sandbox dédié), puis vérifier le contenu du conteneur
  via `docker exec` direct — déterministe, ne dépend pas de la capacité du LLM
  à invoquer correctement les outils exec (calibré sur la découverte du
  06-02 : "tool-calling avec 29 outils = mur").
- **Convention workspace** : suit le patron `workspace-slm` →
  `~/.openclaw-slm/workspace` (seed versionné dans `openclaw-config/`, copie
  vivante dans `~/.openclaw-slm/`). Ici : `openclaw-config/workspace-wiki/` →
  `~/.openclaw-slm/workspace-wiki/`.
- **Note pour l'utilisateur (pas une tâche du plan)** :
  `openclaw-config/agents/wikilm/workspace/AGENTS.md` est du code mort de
  l'ancienne architecture MCP (référence des outils `wiki-lm__wiki_*` qui
  n'existent plus dans la nouvelle architecture sans MCP). Il n'est pas touché
  par ce plan — à traiter séparément si l'utilisateur le souhaite.

---

## Task 1: Dockerfile de l'image wiki

**Files:**
- Create: `openclaw-config/Dockerfile.wiki`

- [ ] **Step 1: Écrire le Dockerfile**

```dockerfile
FROM openclaw-sandbox:bookworm-slim
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
  && rm -rf /var/lib/apt/lists/*
COPY Wiki_LM/requirements.txt /tmp/wiki-requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/wiki-requirements.txt \
  && rm /tmp/wiki-requirements.txt
USER sandbox
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

- [ ] **Step 2: Commit**

```bash
git add openclaw-config/Dockerfile.wiki
git commit -m "feat(wiki-image): ajouter Dockerfile.wiki (mono-étape, deps + BGE-M3 pré-téléchargé)"
```

---

## Task 2: Construction de l'image et vérification de base

**Files:** aucun (opération Docker)

- [ ] **Step 1: Construire l'image**

Depuis la racine du dépôt (le contexte de build doit inclure `Wiki_LM/`) :

```bash
cd /home/mauceric/Secretarius
docker build -t secretarius-wiki:latest -f openclaw-config/Dockerfile.wiki .
```

Expected: le build se termine par `Successfully tagged secretarius-wiki:latest`
(ou équivalent BuildKit `naming to docker.io/library/secretarius-wiki:latest done`).
Durée attendue : plusieurs minutes (téléchargement torch/transformers + poids
BGE-M3 ~2,3 Go).

- [ ] **Step 2: Vérifier que l'image existe et sa taille**

```bash
docker images secretarius-wiki:latest --format '{{.Repository}}:{{.Tag}}\t{{.Size}}'
```

Expected: une ligne `secretarius-wiki:latest` avec une taille de l'ordre de
plusieurs Go (poids dominant = torch + poids BGE-M3, cf. spec §4 — "~5 Go").

---

## Task 3: Test de chargement hors-ligne (sanity check avant câblage)

**Files:** aucun (conteneur jetable)

Ce test vérifie — avant d'investir dans le câblage OpenClaw — que les poids
BGE-M3 sont bien embarqués dans l'image et chargeables sans réseau. C'est la
garantie centrale de la conception (§2.1 du spec : "démarrage sans dépendance
réseau au chargement du modèle").

- [ ] **Step 1: Lancer un conteneur jetable sans accès réseau et charger le modèle**

```bash
docker run --rm --network none secretarius-wiki:latest \
  python3 -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-m3'); print(m.get_sentence_embedding_dimension())"
```

Expected: la commande affiche `1024` (dimension des embeddings BGE-M3) et se
termine sans erreur réseau — preuve que les poids sont pré-téléchargés dans
l'image et que le chargement ne dépend d'aucun accès externe.

Si cette étape échoue (erreur de résolution DNS / téléchargement), le
`RUN python3 -c "... SentenceTransformer('BAAI/bge-m3')"` du Dockerfile (Task 1)
n'a pas mis les poids au bon endroit dans le cache HuggingFace — revoir avant
de poursuivre.

---

## Task 4: Workspace minimal de l'agent wiki

**Files:**
- Create: `openclaw-config/workspace-wiki/AGENTS.md`

- [ ] **Step 1: Écrire le fichier `AGENTS.md`**

```markdown
# AGENTS.md — Agent wiki (image en cours de validation)

## Rôle actuel

Vous êtes l'agent `wiki`. Votre image Docker (`secretarius-wiki:latest`) est
en cours de validation — le skill wiki définitif (orchestration
capture/ingest/query, filtrage anti-injection façon Scout) n'existe pas
encore et sera conçu dans une session ultérieure.

Pour l'instant, vous exécutez les commandes de vérification qu'on vous
demande via l'outil exec, dans votre conteneur sandbox :
- imports de bibliothèques Python (`sentence_transformers`, etc.)
- lecture/écriture dans `/Wiki_LM` (base de connaissances, montée en
  lecture-écriture)
- exécution des scripts montés en lecture seule dans `/wiki-tools`
  (correspond à `Wiki_LM/tools/` sur l'hôte)

## Conventions

- Quand on vous demande d'exécuter une commande et de rapporter le résultat,
  exécutez-la réellement avec l'outil exec et donnez la sortie verbatim —
  n'inventez jamais de résultat, même si vous pensez connaître la réponse.
- Soyez concis : pas de commentaire ni de suggestion au-delà de ce qui est
  demandé pendant cette phase de validation.
```

- [ ] **Step 2: Copier le seed vers le répertoire vivant de l'instance slm**

```bash
mkdir -p ~/.openclaw-slm/workspace-wiki
cp /home/mauceric/Secretarius/openclaw-config/workspace-wiki/AGENTS.md ~/.openclaw-slm/workspace-wiki/AGENTS.md
ls ~/.openclaw-slm/workspace-wiki/
```

Expected: `AGENTS.md` listé. (Les autres fichiers de bootstrap — `SOUL.md`,
`TOOLS.md`, `IDENTITY.md`, `USER.md`, `HEARTBEAT.md` — seront générés
automatiquement par OpenClaw au premier tour de l'agent, comme documenté dans
`gateway/config-agents.md` : "Disables automatic creation of workspace
bootstrap files".)

- [ ] **Step 3: Commit du seed versionné**

```bash
git add openclaw-config/workspace-wiki/AGENTS.md
git commit -m "feat(wiki-image): seed workspace minimal de l'agent wiki (rôle = validation d'image)"
```

---

## Task 5: Câblage de l'agent wiki dans le template

**Files:**
- Modify: `openclaw-config/openclaw-slm.json.template:71-73`

- [ ] **Step 1: Ajouter l'entrée `wiki` à `agents.list[]`**

Remplacer :

```json
    "list": [
      { "id": "main" }
    ]
```

par :

```json
    "list": [
      { "id": "main" },
      {
        "id": "wiki",
        "model": { "primary": "euria/mistralai/Mistral-Small-4-119B-2603" },
        "workspace": "${HOME}/.openclaw-slm/workspace-wiki",
        "sandbox": {
          "docker": {
            "image": "secretarius-wiki:latest",
            "binds": [
              "/home/mauceric/Secretarius/Wiki_LM/tools:/wiki-tools:ro",
              "/home/mauceric/Documents/Arbath/Wiki_LM:/Wiki_LM:rw"
            ],
            "env": {
              "WIKI_PATH": "/Wiki_LM",
              "WIKI_LLM_BACKEND": "openai",
              "OPENAI_BASE_URL": "https://api.infomaniak.com/2/ai/${EURIA_PRODUCT_ID}/openai/v1",
              "OPENAI_API_KEY": "${EURIA_API_KEY}",
              "OPENAI_MODEL": "mistralai/Mistral-Small-4-119B-2603"
            }
          }
        },
        "tools": {
          "exec": { "host": "sandbox" }
        }
      }
    ]
```

- [ ] **Step 2: Vérifier que le JSON est valide**

```bash
python3 -c "import json; json.load(open('/home/mauceric/Secretarius/openclaw-config/openclaw-slm.json.template'))" && echo "JSON valide"
```

Expected: `JSON valide`

- [ ] **Step 3: Commit**

```bash
git add openclaw-config/openclaw-slm.json.template
git commit -m "feat(wiki-image): ajouter l'agent wiki au template slm (image dédiée, binds, exec.host=sandbox)"
```

---

## Task 6: Câblage dans la config installée + redémarrage du gateway

**Files:**
- Modify: `~/.openclaw-slm/openclaw.json` (non versionné — secrets résolus)
- Modify: `~/.openclaw-slm/openclaw.json.bak` (synchronisation, pattern existant — cf. `install.sh:228`)

> Cette étape touche la config réelle de l'instance slm en cours d'exécution.
> **Demander confirmation à l'utilisateur avant le redémarrage du service**
> (Step 3) — règle CLAUDE.md "Confirmation requise avant : systemctl
> start/stop/enable".

- [ ] **Step 1: Appliquer le même ajout à `agents.list[]`, avec les chemins déjà résolus (pas de `${HOME}`)**

Remplacer dans `~/.openclaw-slm/openclaw.json` :

```json
    "list": [
      { "id": "main" }
    ]
```

par :

```json
    "list": [
      { "id": "main" },
      {
        "id": "wiki",
        "model": { "primary": "euria/mistralai/Mistral-Small-4-119B-2603" },
        "workspace": "/home/mauceric/.openclaw-slm/workspace-wiki",
        "sandbox": {
          "docker": {
            "image": "secretarius-wiki:latest",
            "binds": [
              "/home/mauceric/Secretarius/Wiki_LM/tools:/wiki-tools:ro",
              "/home/mauceric/Documents/Arbath/Wiki_LM:/Wiki_LM:rw"
            ],
            "env": {
              "WIKI_PATH": "/Wiki_LM",
              "WIKI_LLM_BACKEND": "openai",
              "OPENAI_BASE_URL": "https://api.infomaniak.com/2/ai/${EURIA_PRODUCT_ID}/openai/v1",
              "OPENAI_API_KEY": "${EURIA_API_KEY}",
              "OPENAI_MODEL": "mistralai/Mistral-Small-4-119B-2603"
            }
          }
        },
        "tools": {
          "exec": { "host": "sandbox" }
        }
      }
    ]
```

(`${EURIA_PRODUCT_ID}` et `${EURIA_API_KEY}` restent des références de
substitution résolues à l'exécution depuis `gateway.systemd.env` — comme
`models.providers.euria.baseUrl` plus haut dans le même fichier — ce ne sont
pas des secrets en clair à remplacer ici.)

- [ ] **Step 2: Synchroniser `.bak` (pattern existant, `install.sh:228`)**

```bash
python3 -c "import json; json.load(open('/home/mauceric/.openclaw-slm/openclaw.json'))" && echo "JSON valide"
cp ~/.openclaw-slm/openclaw.json ~/.openclaw-slm/openclaw.json.bak
```

Expected: `JSON valide`

- [ ] **Step 3: Redémarrer le gateway slm (confirmation utilisateur requise avant exécution)**

```bash
systemctl --user restart openclaw-gateway-slm.service
sleep 5
systemctl --user is-active openclaw-gateway-slm.service
```

Expected: `active`

- [ ] **Step 4: Vérifier que l'agent `wiki` est chargé**

```bash
~/.openclaw-slm/npm/node_modules/.bin/openclaw agents list --json | python3 -c "import json,sys; ids=[a['id'] for a in json.load(sys.stdin)]; print(ids)"
```

Expected: une liste contenant `'main'` et `'wiki'`.

---

## Task 7: Validation — isolation (l'agent wiki a accès à BGE-M3, `main` non)

**Files:** aucun (vérification runtime)

Reprend le patron de test du Pilier A : binaire/bibliothèque marqueur exécuté
par l'agent dédié, échec pour `main`.

- [ ] **Step 1: Déclencher un tour de l'agent wiki pour faire naître son conteneur sandbox**

```bash
~/.openclaw-slm/npm/node_modules/.bin/openclaw agent --agent wiki --message "ping" --json
```

Expected: une réponse JSON sans erreur (le contenu textuel importe peu — ce
tour sert uniquement à faire créer le conteneur sandbox dédié à `wiki`).

- [ ] **Step 2: Repérer les conteneurs sandbox `wiki` et `main`**

```bash
docker ps --filter "name=openclaw-sbx-agent-wiki" --format '{{.Names}}'
docker ps --filter "name=openclaw-sbx-agent-main-main" --format '{{.Names}}'
```

Expected: chaque commande retourne exactement un nom de conteneur (par
exemple `openclaw-sbx-agent-wiki-main-<hash>` et
`openclaw-sbx-agent-main-main-<hash>`).

- [ ] **Step 3: Vérifier que l'agent wiki charge BGE-M3 sans réseau**

```bash
WIKI_CTN=$(docker ps --filter "name=openclaw-sbx-agent-wiki" --format '{{.Names}}' | head -1)
docker exec -e HF_HUB_OFFLINE=1 "$WIKI_CTN" \
  python3 -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-m3'); print(m.get_sentence_embedding_dimension())"
```

Expected: `1024`. `HF_HUB_OFFLINE=1` interdit tout appel réseau HuggingFace —
le succès prouve que les poids sont bien dans l'image (pas re-téléchargés à
la demande).

- [ ] **Step 4: Vérifier que l'agent main échoue sur le même import**

```bash
MAIN_CTN=$(docker ps --filter "name=openclaw-sbx-agent-main-main" --format '{{.Names}}' | head -1)
docker exec "$MAIN_CTN" python3 -c "import sentence_transformers"
```

Expected: échec avec `ModuleNotFoundError: No module named 'sentence_transformers'`
(ou équivalent) — confirme que `main` tourne sur l'image de base, sans les
dépendances ML, et que la capacité dépend bien du contenu de l'image.

---

## Task 8: Validation — bind code vivant (modifications visibles sans rebuild)

**Files:** aucun (fichier jetable créé/supprimé pendant le test, hors arborescence suivie par git)

> On crée un fichier jetable plutôt que de modifier un fichier existant de
> `Wiki_LM/tools/` — ce répertoire est un chantier actif (Phase 1 Late
> Interaction en cours, cf. `Wiki_LM/CLAUDE.md`), on évite toute interférence
> avec le travail en cours.

- [ ] **Step 1: Créer un fichier marqueur sur l'hôte, dans `Wiki_LM/tools/`**

```bash
echo 'MARKER = "bind-mount-live"' > /home/mauceric/Secretarius/Wiki_LM/tools/_bind_test_marker.py
```

- [ ] **Step 2: Vérifier que le conteneur wiki le voit immédiatement (pas de rebuild, pas de redémarrage)**

```bash
WIKI_CTN=$(docker ps --filter "name=openclaw-sbx-agent-wiki" --format '{{.Names}}' | head -1)
docker exec "$WIKI_CTN" cat /wiki-tools/_bind_test_marker.py
```

Expected: `MARKER = "bind-mount-live"` — le fichier créé sur l'hôte après le
build de l'image et le démarrage du conteneur est visible immédiatement,
preuve que `/wiki-tools` est bien un bind-mount vivant (pas une copie figée).

- [ ] **Step 3: Supprimer le marqueur sur l'hôte et vérifier sa disparition immédiate dans le conteneur**

```bash
rm /home/mauceric/Secretarius/Wiki_LM/tools/_bind_test_marker.py
docker exec "$WIKI_CTN" test -f /wiki-tools/_bind_test_marker.py && echo "ENCORE PRÉSENT (anomalie)" || echo "absent (bind vivant confirmé dans les deux sens)"
```

Expected: `absent (bind vivant confirmé dans les deux sens)`

---

## Task 9: Validation — bind données (lecture/écriture dans `/Wiki_LM`)

**Files:** aucun (fichier jetable créé/supprimé pendant le test, hors arborescence suivie par git — `/home/mauceric/Documents/Arbath/Wiki_LM` n'est pas un dépôt Secretarius)

- [ ] **Step 1: Écrire un fichier de test depuis le conteneur wiki**

```bash
WIKI_CTN=$(docker ps --filter "name=openclaw-sbx-agent-wiki" --format '{{.Names}}' | head -1)
docker exec "$WIKI_CTN" sh -c 'echo "écriture-test-$(date +%s)" > /Wiki_LM/_bind_rw_test.txt && cat /Wiki_LM/_bind_rw_test.txt'
```

Expected: affiche `écriture-test-<timestamp>` — l'écriture depuis le
conteneur réussit (bind `:rw`).

- [ ] **Step 2: Vérifier que le fichier est visible côté hôte, au même contenu**

```bash
cat /home/mauceric/Documents/Arbath/Wiki_LM/_bind_rw_test.txt
```

Expected: le même `écriture-test-<timestamp>` qu'à l'étape précédente —
confirme que le bind pointe bien sur `/home/mauceric/Documents/Arbath/Wiki_LM`
côté hôte (lecture-écriture dans les deux sens).

- [ ] **Step 3: Nettoyer**

```bash
docker exec "$WIKI_CTN" rm /Wiki_LM/_bind_rw_test.txt
ls /home/mauceric/Documents/Arbath/Wiki_LM/_bind_rw_test.txt 2>/dev/null && echo "ENCORE LÀ (anomalie)" || echo "supprimé (confirmé des deux côtés)"
```

Expected: `supprimé (confirmé des deux côtés)`

---

## Task 10: Mise à jour du spec avec les résultats de validation et commit final

**Files:**
- Modify: `docs/superpowers/specs/2026-06-08-image-agent-wiki-design.md` (§7 — ajouter une sous-section "Résultats")

- [ ] **Step 1: Ajouter, à la fin de la section 7 ("Validation") du spec, une sous-section consignant les résultats**

Ajouter après le point 4 de la section 7 :

```markdown

### Résultats (exécutés le AAAA-MM-JJ)

| Test | Résultat |
|---|---|
| Isolation — `wiki` charge BGE-M3 hors-ligne (`HF_HUB_OFFLINE=1`) | ✅ dimension 1024 |
| Isolation — `main` échoue sur `import sentence_transformers` | ✅ `ModuleNotFoundError` |
| Bind code — fichier créé/supprimé sur l'hôte visible immédiatement dans `/wiki-tools` | ✅ |
| Bind données — écriture/lecture/suppression cohérente entre `/Wiki_LM` (conteneur) et `Documents/Arbath/Wiki_LM` (hôte) | ✅ |
```

(Remplacer `AAAA-MM-JJ` par la date réelle d'exécution, et `✅` par `❌` +
description si un test échoue — auquel cas s'arrêter et diagnostiquer avant
de committer.)

- [ ] **Step 2: Commit final**

```bash
git add docs/superpowers/specs/2026-06-08-image-agent-wiki-design.md
git commit -m "docs(wiki-image): consigner les résultats de validation de l'image agent wiki"
```

---

## Self-review (effectué par le rédacteur du plan)

- **Couverture du spec** : §2.1 (deps + BGE-M3 embarqués) → Task 1 ; §2.2
  (binds code + données) → Task 5/6 + Tasks 8/9 ; §2.3 (env via
  `sandbox.docker.env`, pas de fichier secret monté) → Task 5/6 ; §3 (image de
  base) → Task 1 ; §4 (Dockerfile mono-étape) → Task 1 ; §5 (câblage
  `agents.list[]`, binds, `tools.exec.host` par agent) → Tasks 5/6 ; §6
  (construction manuelle, `install.sh` différé — non traité, conforme) ;
  §7 (4 tests) → Tasks 7/8/9 + résultats consignés en Task 10.
- **Pas de placeholder** : chaque commande porte un résultat attendu concret ;
  le `WIKI_LLM_BACKEND`/`OPENAI_*` du spec (ex-placeholder `"..."`) a été
  tranché en amont (Euria via le backend `openai` de `llm.py`) et le spec
  corrigé en conséquence avant l'écriture de ce plan.
- **Cohérence** : le nom du conteneur (`openclaw-sbx-agent-<id>-...`) est
  vérifié empiriquement (Task 7 step 2) avant d'être utilisé dans les tests
  suivants (Tasks 8/9) — pas de nom inventé.
