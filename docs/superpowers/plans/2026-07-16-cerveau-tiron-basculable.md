# Cerveau Tiron basculable + install.sh turnkey — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** installer le routeur `tiron-router` via `install.sh` (versionné, sans dépendance llama locale), fournir une commande `switch-brain <sanroque|modal>` de bascule à chaud, et rendre `install.sh` turnkey (build des 3 images sandbox + copie du plugin automatisées).

**Architecture :** le routeur tourne sur la machine (venv `Wiki_LM/.venv`) et lit son endpoint LLM depuis `~/.openclaw/tiron-router.env`. `switch-brain` repointe deux consommateurs (provider `openclaw.json` + `tiron-router.env`) depuis un registre `~/.openclaw/brains.env`, puis redémarre. `install.sh` pose l'état initial et automatise images + plugin.

**Tech stack :** bash, python3 (édition JSON robuste, comme `switch-model`), systemd user units, docker, pytest (test de `switch-brain` par subprocess).

**Spec :** `docs/superpowers/specs/2026-07-16-cerveau-tiron-basculable-design.md`

## Global Constraints

- Provider `openclaw.json` : `baseUrl` **avec** `/v1` ; routeur `TIRON_LLAMA_BASE` **sans** `/v1` (asymétrie établie).
- Après toute écriture de `openclaw.json` : `cp` vers `openclaw.json.bak` (anti-clobber gateway).
- `install.sh` : un échec (build image, venv) → **WARNING** non bloquant + commande manuelle, jamais d'arrêt brutal (pattern existant `WARNINGS+=(...)`).
- Le routeur : venv `Wiki_LM/.venv/bin/python -m router_service.server`, **pas** `Requires=slm-llama_cpp` (santiago n'a pas de llama local).
- `switch-brain` : nom inconnu → usage + exit 1 sans rien toucher ; cible `modal` avec clé absente → erreur + exit 1.
- Commits en français. Ne pas pousser. `systemctl`/`docker build`/`git push` : pas exécutés en test unitaire.
- Valeurs : sanroque Tailscale = `http://100.100.126.7:8998` ; images = `secretarius-{gog,tiron,wiki}:latest` depuis `openclaw-config/Dockerfile.{gog,tiron,wiki}`.

---

### Task 1 : `switch-brain.sh` — bascule à chaud (TDD)

**Files:**
- Create: `switch-brain.sh` (racine du dépôt)
- Test: `tests/test_switch_brain.py` (créer le dossier `tests/` s'il n'existe pas)

**Interfaces:**
- Produces: exécutable `switch-brain.sh <sanroque|modal>`. Chemins surchargeables par env pour la testabilité : `BRAINS_ENV`, `OPENCLAW_JSON`, `ROUTER_ENV` ; `SWITCH_BRAIN_NO_RESTART=1` saute le restart.

- [ ] **Step 1 : écrire les tests qui échouent**

`tests/test_switch_brain.py` :

```python
import json, os, subprocess, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "switch-brain.sh"


def _setup(tmp, key_file_content=None):
    brains = tmp / "brains.env"
    lines = [
        "BRAIN_SANROQUE_URL=http://100.100.126.7:8998",
        "BRAIN_SANROQUE_KEY=",
        "BRAIN_MODAL_URL=https://ex--tiron.modal.run",
        f"BRAIN_MODAL_KEY_FILE={tmp}/key",
        f"WIKI_PATH={tmp}/wiki",
    ]
    brains.write_text("\n".join(lines) + "\n")
    if key_file_content is not None:
        (tmp / "key").write_text(key_file_content)
    oj = tmp / "openclaw.json"
    oj.write_text(json.dumps({"models": {"providers": {"tiron-llm": {
        "baseUrl": "http://127.0.0.1:8998/v1", "apiKey": "local"}}}}))
    renv = tmp / "router.env"
    renv.write_text("TIRON_LLAMA_BASE=http://127.0.0.1:8998\nTIRON_LLAMA_KEY=\nWIKI_PATH=/keep\n")
    return brains, oj, renv


def _run(tmp, name, brains, oj, renv):
    env = {**os.environ, "BRAINS_ENV": str(brains), "OPENCLAW_JSON": str(oj),
           "ROUTER_ENV": str(renv), "SWITCH_BRAIN_NO_RESTART": "1"}
    return subprocess.run(["bash", str(SCRIPT), name], env=env,
                          capture_output=True, text=True)


def test_switch_modal(tmp_path):
    brains, oj, renv = _setup(tmp_path, key_file_content="SECRET123")
    r = _run(tmp_path, "modal", brains, oj, renv)
    assert r.returncode == 0, r.stderr
    prov = json.loads(oj.read_text())["models"]["providers"]["tiron-llm"]
    assert prov["baseUrl"] == "https://ex--tiron.modal.run/v1"
    assert prov["apiKey"] == "SECRET123"
    env = renv.read_text()
    assert "TIRON_LLAMA_BASE=https://ex--tiron.modal.run" in env
    assert "TIRON_LLAMA_KEY=SECRET123" in env
    assert "WIKI_PATH=/keep" in env  # ligne préservée
    assert (oj.parent / "openclaw.json.bak").exists()


def test_switch_sanroque_no_key(tmp_path):
    brains, oj, renv = _setup(tmp_path)
    r = _run(tmp_path, "sanroque", brains, oj, renv)
    assert r.returncode == 0, r.stderr
    prov = json.loads(oj.read_text())["models"]["providers"]["tiron-llm"]
    assert prov["baseUrl"] == "http://100.100.126.7:8998/v1"
    assert prov["apiKey"] == "local"  # clé vide -> "local" (inerte)
    assert "TIRON_LLAMA_BASE=http://100.100.126.7:8998" in renv.read_text()


def test_unknown_brain_touches_nothing(tmp_path):
    brains, oj, renv = _setup(tmp_path)
    before = oj.read_text()
    r = _run(tmp_path, "inconnu", brains, oj, renv)
    assert r.returncode == 1
    assert oj.read_text() == before  # rien modifié


def test_modal_missing_key_fails(tmp_path):
    brains, oj, renv = _setup(tmp_path, key_file_content=None)  # pas de fichier clé
    before = oj.read_text()
    r = _run(tmp_path, "modal", brains, oj, renv)
    assert r.returncode == 1
    assert oj.read_text() == before
```

- [ ] **Step 2 : vérifier l'échec**

Run : `cd ~/Secretarius && python3 -m pytest tests/test_switch_brain.py -v`
Attendu : FAIL — `switch-brain.sh` introuvable (les 4 tests échouent).

- [ ] **Step 3 : implémenter `switch-brain.sh`**

```bash
#!/usr/bin/env bash
# switch-brain — bascule le cerveau LLM de Tiron (provider + routeur) vers un
# endpoint nommé du registre ~/.openclaw/brains.env, puis redémarre.
set -euo pipefail

BRAINS_ENV="${BRAINS_ENV:-$HOME/.openclaw/brains.env}"
OPENCLAW_JSON="${OPENCLAW_JSON:-$HOME/.openclaw/openclaw.json}"
ROUTER_ENV="${ROUTER_ENV:-$HOME/.openclaw/tiron-router.env}"

usage() { echo "Usage: switch-brain <sanroque|modal>" >&2; exit 1; }
[ $# -eq 1 ] || usage
NAME="$1"; UP="$(printf '%s' "$NAME" | tr '[:lower:]' '[:upper:]')"

[ -f "$BRAINS_ENV" ] || { echo "brains.env introuvable : $BRAINS_ENV" >&2; exit 1; }
# shellcheck disable=SC1090
source "$BRAINS_ENV"

url_var="BRAIN_${UP}_URL"; key_var="BRAIN_${UP}_KEY"; keyfile_var="BRAIN_${UP}_KEY_FILE"
URL="${!url_var:-}"
[ -n "$URL" ] || { echo "Cerveau inconnu : $NAME" >&2; usage; }
KEY="${!key_var:-}"
KEYFILE="${!keyfile_var:-}"
if [ -z "$KEY" ] && [ -n "$KEYFILE" ]; then
  KEYFILE="${KEYFILE/#\~/$HOME}"
  [ -f "$KEYFILE" ] || { echo "Clé du cerveau $NAME absente : $KEYFILE" >&2; exit 1; }
  KEY="$(cat "$KEYFILE")"
fi
APIKEY="${KEY:-local}"

# 1. provider openclaw.json (python = robuste) + sync .bak
OPENCLAW_JSON="$OPENCLAW_JSON" URL="$URL" APIKEY="$APIKEY" python3 - <<'PY'
import json, os, shutil
p = os.environ["OPENCLAW_JSON"]
c = json.load(open(p))
prov = c["models"]["providers"]["tiron-llm"]
prov["baseUrl"] = os.environ["URL"].rstrip("/") + "/v1"
prov["apiKey"] = os.environ["APIKEY"]
json.dump(c, open(p, "w"), indent=1, ensure_ascii=False)
open(p, "a").write("\n")
shutil.copy(p, p + ".bak")
PY

# 2. routeur : maj des 2 vars, préserve le reste (WIKI_PATH…)
touch "$ROUTER_ENV"
_set() {  # fichier clé valeur
  if grep -q "^$2=" "$1"; then sed -i "s|^$2=.*|$2=$3|" "$1"; else echo "$2=$3" >> "$1"; fi
}
_set "$ROUTER_ENV" TIRON_LLAMA_BASE "$URL"
_set "$ROUTER_ENV" TIRON_LLAMA_KEY "$KEY"

echo "Cerveau actif : $NAME ($URL)"

# 3. restart (sautable en test)
if [ "${SWITCH_BRAIN_NO_RESTART:-}" != "1" ]; then
  systemctl --user restart openclaw-gateway tiron-router 2>/dev/null || \
    echo "Redémarrez manuellement : systemctl --user restart openclaw-gateway tiron-router" >&2
fi
```

Puis : `chmod +x ~/Secretarius/switch-brain.sh`.

- [ ] **Step 4 : vérifier le succès**

Run : `cd ~/Secretarius && python3 -m pytest tests/test_switch_brain.py -v`
Attendu : 4 PASS.

- [ ] **Step 5 : commit**

```bash
cd ~/Secretarius && git add switch-brain.sh tests/test_switch_brain.py
git commit -m "feat: switch-brain — bascule à chaud du cerveau Tiron (provider + routeur) via brains.env"
```

---

### Task 2 : routeur versionné + câblage `install.sh`

**Files:**
- Create: `openclaw-config/tiron-router.service`
- Modify: `openclaw-config/install.sh` (nouveau bloc, après l'install de `wiki-lm-server.service` ~ligne 280)

**Interfaces:**
- Consumes: `TIRON_LLM_URL`/`TIRON_LLM_KEY` (install.conf), `WIKI_PATH`. Produit : `~/.openclaw/tiron-router.env`, `~/.openclaw/brains.env`, unité `tiron-router.service` installée+activée.

- [ ] **Step 1 : créer l'unité versionnée**

`openclaw-config/tiron-router.service` :
```ini
[Unit]
Description=Tiron router service (BGE-M3 gate + dispatch)
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/Secretarius
EnvironmentFile=%h/.openclaw/tiron-router.env
ExecStart=%h/Secretarius/Wiki_LM/.venv/bin/python -m router_service.server
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

- [ ] **Step 2 : vérifier la validité de l'unité**

Run : `systemd-analyze --user verify openclaw-config/tiron-router.service 2>&1 | grep -v "Unknown\|Failed to prepare" || echo OK`
Attendu : pas d'erreur de syntaxe (les avertissements de chemins %h hors contexte user sont tolérés).

- [ ] **Step 3 : câbler dans `openclaw-config/install.sh`**

Après le bloc `wiki-lm-server.service` (~ligne 280), ajouter :
```bash
# Routeur Tiron (tiron-router) — versionné, câblé au cerveau via tiron-router.env.
ROUTER_DST="${SYSTEMD_USER_DIR}/tiron-router.service"
cp "${SCRIPT_DIR}/tiron-router.service" "$ROUTER_DST"
# Env du routeur : endpoint LLM initial (depuis TIRON_LLM_URL/KEY) + WIKI_PATH (FAQ).
cat > "${HOME}/.openclaw/tiron-router.env" <<EOF
TIRON_LLAMA_BASE=${TIRON_LLM_URL}
TIRON_LLAMA_KEY=${TIRON_LLM_KEY}
WIKI_PATH=${WIKI_PATH}
EOF
# Registre des cerveaux (éditable) — non écrasé s'il existe.
if [[ ! -f "${HOME}/.openclaw/brains.env" ]]; then
  cat > "${HOME}/.openclaw/brains.env" <<EOF
BRAIN_SANROQUE_URL=http://100.100.126.7:8998
BRAIN_SANROQUE_KEY=
BRAIN_MODAL_URL=${BRAIN_MODAL_URL:-}
BRAIN_MODAL_KEY_FILE=${HOME}/.openclaw/secrets/tiron-llm-key
WIKI_PATH=${WIKI_PATH}
EOF
fi
if [[ -x "${HOME}/Secretarius/Wiki_LM/.venv/bin/python" ]]; then
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable tiron-router.service 2>/dev/null && \
    info "tiron-router.service activé" || warn "Activation de tiron-router.service échouée"
else
  warn "venv Wiki_LM absent — tiron-router non activé (relancer après le venv)"
fi
```
Note : `TIRON_LLM_URL`/`TIRON_LLM_KEY`/`WIKI_PATH` sont déjà exportés/disponibles dans `install.sh` (install.conf) ; vérifier qu'ils le sont aussi dans `openclaw-config/install.sh` (sinon les sourcer), et ajouter `BRAIN_MODAL_URL="${BRAIN_MODAL_URL:-}"` dans `install.conf`.

- [ ] **Step 4 : vérifier la syntaxe bash + la présence des variables**

Run : `bash -n openclaw-config/install.sh && echo "bash OK"`
Run : `grep -q "TIRON_LLM_URL\|WIKI_PATH" openclaw-config/install.sh && echo "vars référencées"`
Attendu : `bash OK` puis `vars référencées`.

- [ ] **Step 5 : ajouter `BRAIN_MODAL_URL` à `install.conf`**

Après le bloc `TIRON_LLM_*` d'`install.conf` :
```bash
# URL du cerveau Modal pour le registre switch-brain (rempli dans brains.env).
BRAIN_MODAL_URL="${BRAIN_MODAL_URL:-}"
```

- [ ] **Step 6 : commit**

```bash
cd ~/Secretarius && git add openclaw-config/tiron-router.service openclaw-config/install.sh install.conf
git commit -m "feat(install): installe tiron-router (versionné, venv Wiki_LM, sans dép llama locale) + brains.env"
```

**Vérification réelle (E2E) déférée** : l'installation complète du routeur se valide sur une machine cible (santiago) — noter dans le rapport que Steps 3-5 sont vérifiés en syntaxe ici, E2E à l'install santiago.

---

### Task 3 : `install.sh` turnkey — copie plugin + build des 3 images

**Files:**
- Modify: `install.sh` (bloc post-install ~ligne 294 : remplacer l'impression de la copie plugin par l'exécution) et un nouveau bloc de build d'images (après Étape 5 venv, ~ligne 235)
- Modify: `openclaw-config/install.sh` (le rappel manuel de copie plugin ~ligne 361 devient superflu → le retirer ou le convertir en confirmation)

**Interfaces:**
- Consumes: `SECRETARIUS_ROOT`, `OPENCLAW_PATH`, `OPENCLAW_CONFIG_PATH`, tableau `WARNINGS`.

- [ ] **Step 1 : automatiser la copie du plugin**

Dans `install.sh`, remplacer le bloc imprimé (lignes ~294-297, `echo "  2. Copier le plugin…"` + la commande) par une exécution réelle, placée avec les autres étapes d'install (après l'Étape 5 venv) :
```bash
# Plugin derisk-deleg : copie automatique (dist committé à jour).
PLUGIN_SRC="${SECRETARIUS_ROOT}/derisk-deleg"
PLUGIN_DST="${OPENCLAW_PATH}/extensions/derisk-deleg"
if [[ -d "${PLUGIN_SRC}/dist" ]]; then
  mkdir -p "$PLUGIN_DST"
  cp -r "${PLUGIN_SRC}/dist" "${PLUGIN_SRC}/node_modules" \
        "${PLUGIN_SRC}/openclaw.plugin.json" "${PLUGIN_SRC}/package.json" "$PLUGIN_DST/" 2>/dev/null \
    && info "plugin derisk-deleg copié ✓" \
    || WARNINGS+=("copie du plugin derisk-deleg échouée\n    voir ${PLUGIN_SRC}")
else
  WARNINGS+=("derisk-deleg/dist absent — construire le plugin (npm run build) avant l'install")
fi
```
Retirer le point « 2. Copier le plugin » du bloc post-install imprimé (devenu automatique) et renuméroter les points suivants.

- [ ] **Step 2 : build des 3 images sandbox**

Dans `install.sh`, après l'Étape 5 (venv), ajouter :
```bash
# Étape 5b — Images sandbox (gog/tiron/wiki). Échec = WARNING, non bloquant.
if command -v docker &>/dev/null && docker ps &>/dev/null 2>&1; then
  for img in gog tiron wiki; do
    DF="${OPENCLAW_CONFIG_PATH}/Dockerfile.${img}"
    if docker build -q -f "$DF" -t "secretarius-${img}:latest" "${OPENCLAW_CONFIG_PATH}" &>/dev/null; then
      info "image secretarius-${img}:latest ✓"
    else
      WARNINGS+=("build image secretarius-${img} échoué\n    docker build -f ${DF} -t secretarius-${img}:latest ${OPENCLAW_CONFIG_PATH}")
    fi
  done
else
  WARNINGS+=("docker inaccessible — images sandbox non construites (gog/tiron/wiki)")
fi
```
Note : vérifier le **contexte de build** attendu par chaque Dockerfile (est-ce `openclaw-config/` ou la racine du dépôt ?) — lire l'en-tête `COPY`/`ADD` des `Dockerfile.*` et ajuster le dernier argument de `docker build` en conséquence ; le corriger si le contexte diffère.

- [ ] **Step 3 : vérifier la syntaxe bash**

Run : `cd ~/Secretarius && bash -n install.sh && echo "install.sh OK"`
Run : `grep -q "secretarius-.*:latest\|extensions/derisk-deleg" install.sh && echo "blocs présents"`
Attendu : `install.sh OK` puis `blocs présents`.

- [ ] **Step 4 : commit**

```bash
cd ~/Secretarius && git add install.sh openclaw-config/install.sh
git commit -m "feat(install): copie plugin automatique + build des 3 images sandbox (turnkey)"
```

**Vérification réelle (E2E) déférée** : le build des images et la copie du plugin se valident sur une install réelle (santiago) — Steps sont vérifiés en syntaxe ici.

---

## Vérification finale du chantier (E2E, sur machine réelle)

Hors périmètre des tâches unitaires (nécessite une install/une machine) — à faire par l'utilisateur ou à la prochaine session sur santiago :
- `switch-brain modal` puis `switch-brain sanroque` sur sanroque → `/q` répond via Modal puis via le llama local, provider + routeur repointés.
- `install.sh` sur une machine neuve → routeur actif, 3 images construites, plugin copié, sans étape manuelle pour ces points.

## Notes

- Le contexte de build Docker (Step 2 de Task 3) est le seul point à confirmer par lecture des `Dockerfile.*`.
- Sur santiago, un torch CUDA inutile peut alourdir le venv (pas de GPU) — optimisation optionnelle (`--extra-index-url` CPU), non bloquante, hors de ce plan.
