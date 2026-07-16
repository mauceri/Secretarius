# Câblage Modal du cerveau Tiron + fiche composant — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** rendre le cerveau Modal de Tiron utilisable (deploy persistant, api-key, bascule local↔Modal validée E2E, variables d'installation VPS) et le documenter dans `docs/components/modal.md`.

**Architecture :** l'app Modal existante (`tiron_modal/app.py`, llama-server CUDA + GGUFs sur Volume) gagne une clé API via un Secret Modal ; les deux consommateurs (provider `tiron-llm` d'openclaw.json, routeur 8999) apprennent à envoyer un Bearer ; la bascule reste une procédure manuelle documentée.

**Tech stack :** Modal (CLI dans `~/modal-venv`), llama.cpp server, OpenClaw, systemd user units, bash/envsubst, pytest.

**Spec :** `docs/superpowers/specs/2026-07-16-modal-cablage-doc-design.md`

## Global Constraints

- Le local (`slm-llama_cpp.service`, 8998) reste l'état nominal ; toute bascule se termine par un retour au local vérifié.
- URL : le provider OpenClaw attend l'URL **avec** `/v1` ; le routeur attend l'URL **sans** `/v1`.
- Toute écriture manuelle de `~/.openclaw/openclaw.json` doit être suivie de `cp` vers `openclaw.json.bak` (anti-clobber du gateway).
- `systemctl start/stop/restart` : confirmation utilisateur requise (règle CLAUDE.md).
- Docs en français, pas d'emojis.
- Ne pas toucher `openclaw.json.sanroque.template` (consommé par personne ; référence manuelle sanroque, qui reste locale).
- Le test Telegram (Tasks 4 et 5) nécessite l'utilisateur : checkpoint explicite.

---

### Task 1 : Routeur — Bearer conditionnel

**Files:**
- Modify: `router_service/server.py:14` et `:39-41`
- Test: `router_service/test_router.py` (ajout en fin de fichier)

**Interfaces:**
- Consumes: rien de nouveau.
- Produces: `router_service.server.LLAMA_KEY` (str, module-level, défaut `""`) ; `call_adapter()` envoie `Authorization: Bearer <LLAMA_KEY>` ssi `LLAMA_KEY` non vide.

- [ ] **Step 1 : écrire les tests qui échouent**

Ajouter en fin de `router_service/test_router.py` :

```python
def _start_stub(received):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Stub(BaseHTTPRequestHandler):
        def do_POST(self):
            received["auth"] = self.headers.get("Authorization")
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            payload = json.dumps({"choices": [{"message": {
                "content": '{"command": null, "args": ""}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_call_adapter_sends_bearer_when_key_set(monkeypatch):
    received = {}
    httpd = _start_stub(received)
    try:
        monkeypatch.setattr(router_server, "LLAMA_BASE",
                            f"http://127.0.0.1:{httpd.server_address[1]}")
        monkeypatch.setattr(router_server, "LLAMA_KEY", "secret123")
        router_server.call_adapter("bonjour")
        assert received["auth"] == "Bearer secret123"
    finally:
        httpd.shutdown()


def test_call_adapter_no_header_without_key(monkeypatch):
    received = {}
    httpd = _start_stub(received)
    try:
        monkeypatch.setattr(router_server, "LLAMA_BASE",
                            f"http://127.0.0.1:{httpd.server_address[1]}")
        monkeypatch.setattr(router_server, "LLAMA_KEY", "")
        router_server.call_adapter("bonjour")
        assert received["auth"] is None
    finally:
        httpd.shutdown()
```

(`json`, `threading`, `router_server` sont déjà importés dans ce fichier.)

- [ ] **Step 2 : vérifier l'échec**

Run : `cd ~/Secretarius && python3 -m pytest router_service/test_router.py -k call_adapter -v`
Attendu : FAIL — `AttributeError: ... has no attribute 'LLAMA_KEY'` (monkeypatch d'un attribut inexistant).

- [ ] **Step 3 : implémentation minimale**

Dans `router_service/server.py`, après la ligne 14 (`LLAMA_BASE = ...`) :

```python
LLAMA_KEY = os.environ.get("TIRON_LLAMA_KEY", "")
```

Dans `call_adapter()`, remplacer la construction de la requête (lignes 39-41) par :

```python
    headers = {"Content-Type": "application/json"}
    if LLAMA_KEY:
        headers["Authorization"] = "Bearer " + LLAMA_KEY
    req = urllib.request.Request(LLAMA_BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=headers)
```

- [ ] **Step 4 : vérifier le succès**

Run : `cd ~/Secretarius && python3 -m pytest router_service/test_router.py -k call_adapter -v`
Attendu : 2 PASS.

- [ ] **Step 5 : commit**

```bash
git add router_service/server.py router_service/test_router.py
git commit -m "feat(router): en-tête Bearer conditionnel (TIRON_LLAMA_KEY) vers le cerveau Tiron"
```

---

### Task 2 : App Modal — Secret + --api-key, deploy persistant

**Files:**
- Modify: `tiron_modal/app.py:37-51`
- Hors repo : clé dans `~/.openclaw/secrets/tiron-llm-key`, Secret Modal `tiron-llm-api-key`

**Interfaces:**
- Consumes: Volume `tiron-models` existant, image `ghcr.io/ggml-org/llama.cpp:server-cuda`.
- Produces: app déployée `tiron-llm-modal`, URL stable `https://<user>--tiron-llm-modal-serve.modal.run` (notée `$MODAL_URL` ci-dessous), protégée par Bearer ; fichier local `~/.openclaw/secrets/tiron-llm-key` (600) utilisé par Tasks 4 et 6.

- [ ] **Step 1 : générer et stocker la clé**

```bash
mkdir -p ~/.openclaw/secrets && chmod 700 ~/.openclaw/secrets
openssl rand -hex 32 > ~/.openclaw/secrets/tiron-llm-key
chmod 600 ~/.openclaw/secrets/tiron-llm-key
```

- [ ] **Step 2 : créer le Secret Modal**

```bash
~/modal-venv/bin/modal secret create tiron-llm-api-key \
  LLAMA_API_KEY="$(cat ~/.openclaw/secrets/tiron-llm-key)"
```

Attendu : confirmation de création (`Created secret` ou équivalent).

- [ ] **Step 3 : modifier app.py**

Décorateur de `serve()` — ajouter le Secret :

```python
@app.function(
    image=image,
    gpu=GPU,
    volumes={MODELS_DIR: vol},
    secrets=[modal.Secret.from_name("tiron-llm-api-key")],
    timeout=3600,
    scaledown_window=300,
)
```

Corps de `serve()` — ajouter `--api-key` à la ligne de lancement :

```python
    launch = (
        "export LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH:-}; "
        f"exec /app/llama-server --model {BASE} --lora {LORA} "
        f"--host 0.0.0.0 --port {PORT} -c {CTX} -ngl 999 "
        '--api-key "$LLAMA_API_KEY"'
    )
```

- [ ] **Step 4 : déployer**

```bash
~/modal-venv/bin/modal deploy tiron_modal/app.py
```

Attendu : URL imprimée en sortie → la noter (`MODAL_URL`).

- [ ] **Step 5 : vérifier 200 avec clé / 401 sans clé**

```bash
KEY=$(cat ~/.openclaw/secrets/tiron-llm-key)
# attendre la fin du cold start (503 pendant le chargement des 3 Go)
until [ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" "$MODAL_URL/health")" = "200" ]; do sleep 5; done
# avec clé : complétion valide
curl -s "$MODAL_URL/v1/chat/completions" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' -d @tiron_modal/prompt.json | head -c 400
# sans clé : 401
curl -s -o /dev/null -w '%{http_code}\n' "$MODAL_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @tiron_modal/prompt.json
```

Attendu : JSON `choices[0].message.content` non vide, puis `401`.

- [ ] **Step 6 : commit**

```bash
git add tiron_modal/app.py
git commit -m "feat(modal): endpoint protégé par api-key (Secret tiron-llm-api-key), deploy persistant"
```

---

### Task 3 : Installation — variables TIRON_LLM_URL / TIRON_LLM_KEY

**Files:**
- Modify: `install.conf` (fin de fichier)
- Modify: `openclaw-config/install.sh:117-119` (export + liste blanche envsubst ; vérifier que ce script source bien `install.conf`, sinon ajouter les mêmes défauts en tête)
- Modify: `openclaw-config/openclaw.json.template:83-84` (provider `tiron-llm`)

**Interfaces:**
- Consumes: mécanisme envsubst existant.
- Produces: openclaw.json généré avec `baseUrl = ${TIRON_LLM_URL}/v1` et `apiKey = ${TIRON_LLM_KEY}` ; défauts = comportement local actuel.

Note (écart assumé vs spec) : `envsubst` ne sait pas omettre un champ ; `apiKey`
est donc toujours présent, avec le défaut inoffensif `local` (llama-server sans
`--api-key` ignore l'en-tête Authorization).

- [ ] **Step 1 : install.conf**

Ajouter en fin de fichier :

```bash
# Cerveau Tiron (SLM) : URL du serveur llama.cpp (SANS /v1 final) et clé API.
# Local par défaut ; pour un cerveau distant (ex. Modal) :
#   TIRON_LLM_URL=https://<user>--tiron-llm-modal-serve.modal.run
#   TIRON_LLM_KEY=<clé du Secret Modal tiron-llm-api-key>
TIRON_LLM_URL="${TIRON_LLM_URL:-http://127.0.0.1:8998}"
TIRON_LLM_KEY="${TIRON_LLM_KEY:-local}"
```

- [ ] **Step 2 : template**

Dans `openclaw-config/openclaw.json.template`, bloc provider `tiron-llm` :

```json
   "tiron-llm": {
    "baseUrl": "${TIRON_LLM_URL}/v1",
    "apiKey": "${TIRON_LLM_KEY}",
    "api": "openai-completions",
```

(le reste du bloc inchangé)

- [ ] **Step 3 : install.sh — export + liste blanche**

Dans `openclaw-config/install.sh` (bloc génération openclaw.json, ~ligne 117) :
- ajouter `TIRON_LLM_URL TIRON_LLM_KEY` à la ligne `export HOME HOSTNAME ...` ;
- ajouter `${TIRON_LLM_URL} ${TIRON_LLM_KEY}` à la liste blanche `envsubst '...'`.

- [ ] **Step 4 : vérifier la substitution (sans installer)**

```bash
cd ~/Secretarius/openclaw-config
# cas défaut
bash -c 'source ../install.conf; export TIRON_LLM_URL TIRON_LLM_KEY;
  envsubst "\${TIRON_LLM_URL} \${TIRON_LLM_KEY}" < openclaw.json.template' \
| python3 -c "import json,sys; p=json.load(sys.stdin)['models']['providers']['tiron-llm']; \
assert p['baseUrl']=='http://127.0.0.1:8998/v1' and p['apiKey']=='local'; print('défaut OK')"
# cas VPS/Modal
TIRON_LLM_URL="https://example.modal.run" TIRON_LLM_KEY="k123" bash -c \
 'source ../install.conf; export TIRON_LLM_URL TIRON_LLM_KEY;
  envsubst "\${TIRON_LLM_URL} \${TIRON_LLM_KEY}" < openclaw.json.template' \
| python3 -c "import json,sys; p=json.load(sys.stdin)['models']['providers']['tiron-llm']; \
assert p['baseUrl']=='https://example.modal.run/v1' and p['apiKey']=='k123'; print('modal OK')"
```

Attendu : `défaut OK` puis `modal OK`.

- [ ] **Step 5 : commit**

```bash
git add install.conf openclaw-config/install.sh openclaw-config/openclaw.json.template
git commit -m "feat(install): TIRON_LLM_URL/TIRON_LLM_KEY — cerveau Tiron distant (Modal) configurable à l'installation"
```

---

### Task 4 : Bascule aller (local → Modal) — E2E

Hors repo (procédure sur sanroque, celle que la fiche documentera).
Prérequis : Task 1 déployé (le routeur tourne avec le nouveau code → restart), Task 2 (`$MODAL_URL`, clé).

- [ ] **Step 1 : pointer le provider tiron-llm sur Modal (+ sync .bak)**

```bash
MODAL_URL="https://<user>--tiron-llm-modal-serve.modal.run"   # URL de la Task 2
KEY=$(cat ~/.openclaw/secrets/tiron-llm-key)
python3 - "$MODAL_URL" "$KEY" <<'EOF'
import json, shutil, sys, os
url, key = sys.argv[1], sys.argv[2]
p = os.path.expanduser('~/.openclaw/openclaw.json')
c = json.load(open(p))
prov = c['models']['providers']['tiron-llm']
prov['baseUrl'] = url.rstrip('/') + '/v1'
prov['apiKey'] = key
json.dump(c, open(p, 'w'), indent=1, ensure_ascii=False)
shutil.copy(p, p + '.bak')   # anti-clobber gateway
print('provider tiron-llm ->', prov['baseUrl'])
EOF
```

- [ ] **Step 2 : pointer le routeur sur Modal (drop-in systemd)**

```bash
mkdir -p ~/.config/systemd/user/tiron-router.service.d
cat > ~/.config/systemd/user/tiron-router.service.d/modal.conf <<EOF
[Service]
Environment=TIRON_LLAMA_BASE=$MODAL_URL
Environment=TIRON_LLAMA_KEY=$KEY
EOF
systemctl --user daemon-reload
```

Rappel : ne PAS stopper `slm-llama_cpp.service` (Requires du routeur).

- [ ] **Step 3 : restarts (avec confirmation utilisateur)**

```bash
systemctl --user restart tiron-router
systemctl --user restart openclaw-gateway   # vérifier le nom exact : systemctl --user list-units | grep -i openclaw
```

- [ ] **Step 4 : vérifier le routeur via Modal**

```bash
sleep 20   # chargement BGE-M3
curl -s http://127.0.0.1:8999/route -d '{"message":"que dit le wiki sur le projet Alpha ?"}'
~/modal-venv/bin/modal app logs tiron-llm-modal | tail -20
```

Attendu : réponse `{"status": "ok", "command": "/q", ...}` (ou `no_match`, mais sans erreur), et la requête visible dans les logs Modal. Premier appel possiblement lent/en échec (cold start) : réessayer.

- [ ] **Step 5 : CHECKPOINT utilisateur — test Telegram**

Demander à l'utilisateur d'envoyer une commande réelle (p. ex. `/q <question>`) via Telegram et de confirmer la réponse. Vérifier en parallèle `modal app logs`.

---

### Task 5 : Bascule retour (Modal → local) — E2E

- [ ] **Step 1 : restaurer le provider local (+ sync .bak)**

```bash
python3 - <<'EOF'
import json, shutil, os
p = os.path.expanduser('~/.openclaw/openclaw.json')
c = json.load(open(p))
prov = c['models']['providers']['tiron-llm']
prov['baseUrl'] = 'http://127.0.0.1:8998/v1'
prov.pop('apiKey', None)   # état d'origine : pas de champ apiKey
json.dump(c, open(p, 'w'), indent=1, ensure_ascii=False)
shutil.copy(p, p + '.bak')
print('provider tiron-llm -> local')
EOF
```

- [ ] **Step 2 : retirer le drop-in routeur**

```bash
rm ~/.config/systemd/user/tiron-router.service.d/modal.conf
rmdir ~/.config/systemd/user/tiron-router.service.d
systemctl --user daemon-reload
```

- [ ] **Step 3 : restarts (avec confirmation utilisateur)**

```bash
systemctl --user restart tiron-router
systemctl --user restart openclaw-gateway
```

- [ ] **Step 4 : vérifier le retour au local**

```bash
sleep 20
curl -s http://127.0.0.1:8999/route -d '{"message":"que dit le wiki sur le projet Alpha ?"}'
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8998/health
```

Attendu : routage OK, `200` sur 8998. CHECKPOINT utilisateur : re-test `/q` Telegram.

- [ ] **Step 5 : stopper l'app Modal (optionnel, économie)**

```bash
~/modal-venv/bin/modal app stop tiron-llm-modal
```

---

### Task 6 : Fiche `docs/components/modal.md` + renvoi README

**Files:**
- Create: `docs/components/modal.md`
- Modify: `tiron_modal/README.md` (renvoi vers la fiche)

**Interfaces:**
- Consumes: URL réelle et constats E2E des Tasks 2, 4, 5 (remplacer `<user>` par la valeur réelle).

- [ ] **Step 1 : écrire la fiche**

Contenu de `docs/components/modal.md` (compléter `<user>` et les constats E2E) :

````markdown
---
tags: [documentation, secretarius]
date: 2026-07-16
---

# Composant : modal (cerveau Tiron serverless)

## Rôle

Servir le cerveau de Tiron (phi-4-mini + adaptateur `tiron-unified`, mêmes
GGUFs qu'en local) sur un GPU serverless Modal (L4), via un endpoint
OpenAI-compatible protégé par clé API. Usage : secours, démo, ou cerveau des
installations VPS sans GPU (ex. santiago). L'état nominal reste le service
local `slm-llama_cpp.service` (port 8998).

Code : `tiron_modal/app.py`. Mesures (2026-07-15, L4 à chaud) : 252 tok/s
prompt, 63,8 tok/s génération, ~0,9 s/requête.

## Prérequis

- Compte Modal + CLI : `~/modal-venv/bin/modal` (sinon :
  `python3 -m venv ~/modal-venv && ~/modal-venv/bin/pip install -U modal`
  puis `~/modal-venv/bin/modal setup`).
- Volume `tiron-models` contenant les GGUFs (voir « Charger un nouveau LLM »).
- Secret `tiron-llm-api-key` (voir « Clé API »).

## Démarrer / stopper

```bash
~/modal-venv/bin/modal deploy tiron_modal/app.py   # déploie (URL stable, imprimée)
~/modal-venv/bin/modal app list                    # état des apps
~/modal-venv/bin/modal app logs tiron-llm-modal    # logs (requêtes llama-server)
~/modal-venv/bin/modal app stop tiron-llm-modal    # stoppe l'app
~/modal-venv/bin/modal serve tiron_modal/app.py    # mode dev éphémère (terminal ouvert)
```

URL de l'endpoint : `https://<user>--tiron-llm-modal-serve.modal.run`.

Comportement serverless : conteneur éteint après ~5 min sans requête
(scale-to-zero, facturation stoppée). La requête suivante subit le cold start
(boot + chargement 3 Go : 503 pendant le chargement) — le premier message
Tiron après une pause peut donc échouer : réessayer 1 à 2 min plus tard.
Astuce réveil avant une session d'usage :

```bash
KEY=$(cat ~/.openclaw/secrets/tiron-llm-key)
until [ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" \
  "https://<user>--tiron-llm-modal-serve.modal.run/health")" = "200" ]; do sleep 5; done
```

Coût : L4 ≈ 0,80 $/h, facturé au temps d'allumage GPU uniquement.

## Clé API

Secret partagé arbitraire (pas émis par Modal) : `llama-server --api-key`
compare simplement le Bearer reçu.

```bash
# génération + stockage local
openssl rand -hex 32 > ~/.openclaw/secrets/tiron-llm-key
chmod 600 ~/.openclaw/secrets/tiron-llm-key
# côté serveur (Secret Modal, lu par app.py)
~/modal-venv/bin/modal secret create tiron-llm-api-key \
  LLAMA_API_KEY="$(cat ~/.openclaw/secrets/tiron-llm-key)"
```

La même valeur est recopiée côté consommateurs : `apiKey` du provider
`tiron-llm` (openclaw.json), `TIRON_LLAMA_KEY` (routeur),
`TIRON_LLM_KEY` (installation). Rotation : régénérer la valeur, recréer le
Secret, redéployer, mettre à jour les consommateurs.

## Basculer le cerveau de Tiron

Deux consommateurs à pointer, puis deux restarts. Piège de format : le
provider attend l'URL **avec** `/v1`, le routeur **sans**.

Aller (local → Modal) :

1. Déployer si besoin (`modal deploy`), noter l'URL.
2. `~/.openclaw/openclaw.json`, provider `tiron-llm` :
   `baseUrl` = `https://…modal.run/v1`, `apiKey` = contenu de
   `~/.openclaw/secrets/tiron-llm-key`. **Puis synchroniser le .bak**
   (`cp ~/.openclaw/openclaw.json ~/.openclaw/openclaw.json.bak`), sinon le
   gateway peut restaurer l'ancienne config (anti-clobber).
3. Routeur — drop-in systemd :
   ```bash
   mkdir -p ~/.config/systemd/user/tiron-router.service.d
   cat > ~/.config/systemd/user/tiron-router.service.d/modal.conf <<EOF
   [Service]
   Environment=TIRON_LLAMA_BASE=https://<user>--tiron-llm-modal-serve.modal.run
   Environment=TIRON_LLAMA_KEY=<clé>
   EOF
   systemctl --user daemon-reload
   ```
4. `systemctl --user restart tiron-router openclaw-gateway`

Retour (Modal → local) : `baseUrl` = `http://127.0.0.1:8998/v1`, retirer
`apiKey`, sync `.bak` ; supprimer le drop-in (`rm …/modal.conf`,
`daemon-reload`) ; mêmes restarts ; optionnel `modal app stop tiron-llm-modal`.

Attention : `tiron-router.service` a `Requires=slm-llama_cpp.service` — ne pas
stopper le service local pendant que le cerveau est sur Modal, sinon le
routeur tombe aussi.

## Installation VPS (cerveau distant dès l'installation)

Pour une installation sans GPU (ex. santiago), passer à `install.sh` :

```bash
TIRON_LLM_URL=https://<user>--tiron-llm-modal-serve.modal.run \
TIRON_LLM_KEY=<clé> ./install.sh
```

Défauts (`install.conf`) : `TIRON_LLM_URL=http://127.0.0.1:8998`,
`TIRON_LLM_KEY=local` (valeur inoffensive : le llama-server local ignore
l'en-tête). Le routeur 8999 n'est pas géré par `install.sh` (unité manuelle,
sanroque uniquement).

## Charger un nouveau LLM

```bash
~/modal-venv/bin/modal volume put tiron-models <fichier>.gguf /<fichier>.gguf
```

Puis ajuster dans `tiron_modal/app.py` : `BASE` (et `LORA`, ou retirer
`--lora` de la ligne de lancement), `CTX`, `GPU` si besoin, et redéployer.
Vigilance : la version GGUF de l'adaptateur doit être compatible avec la
release llama.cpp de l'image (`ghcr.io/ggml-org/llama.cpp:server-cuda`) —
les incompatibilités de version GGUF ont déjà causé des pannes.

## Confidentialité

TLS (HTTPS Modal) + clé API protègent le transport et l'accès. En revanche
les prompts sont traités **en clair** dans l'infrastructure Modal (l'inférence
l'exige). Règle d'usage : pas de contenu sensible quand le cerveau est sur
Modal ; le local reste la voie du confidentiel. Piste à l'étude (backlog) :
confidentialité renforcée (confidential computing / GPU TEE, anonymisation).
````

- [ ] **Step 2 : renvoi dans tiron_modal/README.md**

Ajouter après la ligne « Plan : … » :

```markdown
Fiche d'exploitation : `docs/components/modal.md` (deploy/stop, bascule du
cerveau Tiron, clé API, nouveau LLM).
```

- [ ] **Step 3 : vérifier**

Relire la fiche : plus aucun `<user>`/`<clé>` non résolu là où la valeur réelle est connue ; commandes conformes à ce qui a été exécuté en Tasks 2/4/5.

- [ ] **Step 4 : commit**

```bash
git add docs/components/modal.md tiron_modal/README.md
git commit -m "docs: fiche composant modal — deploy/stop, bascule cerveau Tiron, clé API, nouveau LLM"
```

---

### Task 7 : Backlog (mémoire Claude, hors repo)

- [ ] Ajouter au backlog d'idées (mémoire persistante `project_ideas_backlog.md`) : « Confidentialité renforcée Secretarius↔Modal — étudier confidential computing / GPU TEE, filtrage-anonymisation des prompts avant envoi ».
