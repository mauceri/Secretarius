# Installateur à profils — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** un seul `install.sh --profil leger|lourd` qui installe Secretarius depuis zéro, par
modules, et dont le profil léger tourne sur santiago.

**Architecture :** `install.sh` charge `install.conf`, puis `profils/<p>.conf`, puis
l'environnement (`--env-file`). Le socle commun s'exécute toujours ; chaque nom de `$MODULES`
appelle une fonction `module_<nom>` de `openclaw-config/modules.sh`. `openclaw.json` est
généré depuis le gabarit complet, puis élagué par `openclaw-config/profil_config.py`
(fonction pure, testée) : un module absent n'y laisse aucune trace. Une vérification finale
bloquante interroge chaque service installé.

**Tech Stack :** Bash, Python 3.11+ (pytest, venv de Wiki_LM), TypeScript (vitest) pour
derisk-deleg, systemd utilisateur, Docker, OpenClaw.

**Spec :** `docs/superpowers/specs/2026-10-12-installateur-profils-design.md`

## Global Constraints

- Vouvoiement dans tout texte destiné à l'utilisateur ; pas d'emoji ajouté.
- Dépôt imposé à `~/Secretarius` (vérifié par les prérequis).
- Profil léger : `MODULES="wiki scout obfuscateur"` ; cerveau `main` et ingestion des notes
  sur le Qwen3-14B obfusqué ; pages web résumées par DeepSeek (`WIKI_INGEST_WEB_LLM_*`) ;
  agent `scout` sur DeepSeek ; ni Laya, ni Ollama, ni courrier, ni cron, ni gog.
- Profil lourd : `MODULES="wiki scout obfuscateur laya courrier cron gog"`, cerveau Ollama.
- Un module absent ne laisse aucune trace : ni agent, ni outil autorisé, ni unité installée.
- Un seul proxy obfuscateur, sur `172.17.0.1:8001` (joignable de l'hôte et du pont Docker).
- `install.sh` ne fait jamais `sudo` ; `amorcage-systeme.sh` le fait, une fois.
- ZIM fournis par l'utilisateur ; ordre : `WIKI_ZIM_DIR` (invalide → échec) →
  `~/Secretarius/Wiki_LM/zim/` → question en `--interactive` → aucun (annoncé). Jamais de
  repli vers l'API publique (`WIKI_LOOKUP_OFFLINE=1` toujours).
- `obsidian-sync@.service` : `ExecStartPre=ob sync`, puis `ExecStart=ob sync --continuous`.
- Confirmation de l'utilisateur requise avant tout `systemctl start/stop/enable/restart`
  lancé **sur sanroque** pendant le développement, et avant tout `git push`. Les tests
  n'exécutent jamais `systemctl`.
- Ne jamais commiter `Wiki_LM/.env.bak-*` ni `gen_corpus/laya_train_chargédnasKaggle.jsonl`.

## Review Focus

1. Relancer `install.sh` sur une machine déjà installée (idempotence) : rien n'est écrasé
   sans `--force`, en particulier `openclaw.json` déjà élagué et `gateway.systemd.env`.
2. Message tapé avec espace insécable ou tabulation après la commande (`/q x`) : reconnu
   comme commande, comme le faisait `split(None, 1)` côté routeur.
3. Profil léger avec un `--env-file` sanroque (contenant `TELEGRAM_BOT_TOKEN` de Tiron) : le
   second robot doit avoir son propre jeton, sinon les deux instances se volent les messages.
4. `WIKI_ZIM_DIR` posée vers un dossier vide ou inexistant : échec net, pas installation
   silencieuse sans Wikipédia.
5. Proxy obfuscateur démarré avant que `docker0` existe (démarrage de la machine) : il doit
   se relancer jusqu'à ce que l'adresse existe.

---

### Task 1 : derisk-deleg reconnaît les commandes tapées

**Files :**
- Modify : `derisk-deleg/src/dispatch.ts`
- Modify : `derisk-deleg/src/index.ts` (fonction `register`, branche de routage ~l. 662-708)
- Modify : `derisk-deleg/openclaw.plugin.json` (`configSchema`)
- Test : `derisk-deleg/src/dispatch.test.ts`, `derisk-deleg/src/index.test.ts`
- Rebuild : `derisk-deleg/dist/*` (committé)

**Interfaces :**
- Produces : `commandeTapee(text: string): CommandeTapee | null` dans `dispatch.ts` ;
  option de plugin `routeur` (booléen, défaut `true`) lue par `api.pluginConfig?.routeur`.
  La Task 2 écrit `plugins.entries["derisk-deleg"].config.routeur = false` hors module `laya`.

- [ ] **Step 1 : tests de `commandeTapee` (dispatch.test.ts)**

```ts
import { commandeTapee } from "./dispatch.js";

describe("commandeTapee", () => {
  it("commande connue avec argument", () => {
    expect(commandeTapee("/q quelle heure ?")).toEqual({ status: "ok", command: "/q", args: "quelle heure ?" });
  });
  it("espace insécable ou tabulation après la commande", () => {
    expect(commandeTapee("/q x")).toEqual({ status: "ok", command: "/q", args: "x" });
    expect(commandeTapee("/r\tchiens")).toEqual({ status: "ok", command: "/r", args: "chiens" });
  });
  it("argument requis manquant", () => {
    expect(commandeTapee("/q   ")).toEqual({ status: "usage", reply: "Usage : /q <argument>" });
  });
  it("commande sans argument requis", () => {
    expect(commandeTapee("/wikistatus")).toEqual({ status: "ok", command: "/wikistatus", args: "" });
  });
  it("commande inconnue ou texte libre : null", () => {
    expect(commandeTapee("/inconnue x")).toBeNull();
    expect(commandeTapee("bonjour")).toBeNull();
  });
});
```

- [ ] **Step 2 : lancer, voir échouer**

Run : `cd ~/Secretarius/derisk-deleg && npx vitest run src/dispatch.test.ts`
Expected : FAIL, `commandeTapee` n'est pas exporté.

- [ ] **Step 3 : implémenter dans `dispatch.ts` (après `commandToAction`)**

```ts
// Commandes dont l'argument est obligatoire : même liste que NEEDS_ARG de
// router_service/server.py, qui faisait ce contrôle quand tout passait par le routeur.
const ARG_REQUIS = new Set([
  "/c", "/q", "/source", "/chercher", "/repondre", "/r", "/lire",
  "/supprimer", "/verifie", "/cron", "/ecrire",
]);

export type CommandeTapee =
  | { status: "ok"; command: RouterCommand; args: string }
  | { status: "usage"; reply: string };

// Commande tapée explicitement : reconnue sans routeur (spec 2026-10-12).
// \s couvre l'espace insécable et la tabulation, comme split(None, 1) côté Python.
export function commandeTapee(text: string): CommandeTapee | null {
  const m = text.trim().match(/^(\/\S+)(?:\s+([\s\S]*))?$/);
  if (!m || commandToAction(m[1]) === null) return null;
  const args = m[2] ?? "";
  if (ARG_REQUIS.has(m[1]) && !args.trim()) return { status: "usage", reply: `Usage : ${m[1]} <argument>` };
  return { status: "ok", command: m[1] as RouterCommand, args };
}
```

- [ ] **Step 4 : lancer, voir passer** — même commande, Expected : PASS.

- [ ] **Step 5 : tests du hook (index.test.ts, dans le `describe` « routage via tiron-router »)**

Le harnais `makeApi` ne fixe pas `pluginConfig` : ajouter le paramètre
`pluginConfig?: Record<string, unknown>` à `overrides` et `pluginConfig: overrides.pluginConfig`
dans l'objet `api`.

```ts
  it("commande tapée : exécutée sans appeler le routeur", async () => {
    const fetchSpy = vi.fn(async () => { throw new Error("ne doit pas être appelé"); });
    vi.stubGlobal("fetch", fetchSpy);
    const runWikiOpSpy = vi.fn(async () => "12 pages.");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/wikistatus" });

    expect(res).toEqual({ handled: true, reply: { text: "12 pages." } });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("commande tapée sans argument requis : usage, ni routeur ni délégation", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const plugin = await freshPlugin();
    const { api, hooks, run } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/q" });

    expect(res).toEqual({ handled: true, reply: { text: "Usage : /q <argument>" } });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(run).not.toHaveBeenCalled();
  });

  it("sans routeur (routeur: false) : texte libre refusé poliment, routeur jamais appelé", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi({ pluginConfig: { routeur: false } });
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "bonjour" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Tapez une commande, par exemple /q <question> ou /c <texte>." },
    });
    expect(fetchSpy).not.toHaveBeenCalled();
  });
```

Le test existant « routeur indisponible (fetch en échec) » sur `bonjour` doit rester vert
(routeur actif par défaut).

- [ ] **Step 6 : lancer, voir échouer** — `npx vitest run src/index.test.ts` ; Expected : les
  3 nouveaux tests FAIL (le premier : fetch appelé).

- [ ] **Step 7 : implémenter dans `index.ts`**

Dans `register(api)`, avant `api.on("before_agent_reply", …)` :

```ts
    // Profil léger : pas de routeur Laya installé (spec 2026-10-12) ; option
    // écrite par l'installateur, absente = routeur présent (sanroque).
    const routeurActif = api.pluginConfig?.routeur !== false;
```

Remplacer la branche finale (`} else { routed = await callRouter(text); }`) par :

```ts
      } else {
        const tapee = commandeTapee(text);
        if (tapee?.status === "usage") return { handled: true, reply: { text: tapee.reply } };
        if (tapee) {
          routed = { status: "ok", command: tapee.command, args: tapee.args };
        } else if (!routeurActif) {
          return { handled: true, reply: { text: "Tapez une commande, par exemple /q <question> ou /c <texte>." } };
        } else {
          routed = await callRouter(text);
        }
      }
```

et ajouter `commandeTapee` à l'import depuis `./dispatch.js`. Dans `openclaw.plugin.json`,
`configSchema.properties` devient `{ "routeur": { "type": "boolean" } }`.

- [ ] **Step 8 : suite complète, build, copie** —
  `cd ~/Secretarius/derisk-deleg && npx vitest run && npm run build` ; Expected : tous verts
  (171 + 8 nouveaux), `dist/` régénéré.

- [ ] **Step 9 : commit**

```bash
git add derisk-deleg/src/dispatch.ts derisk-deleg/src/dispatch.test.ts derisk-deleg/src/index.ts \
        derisk-deleg/src/index.test.ts derisk-deleg/openclaw.plugin.json derisk-deleg/dist
git commit -m "feat(derisk-deleg): commandes tapées reconnues sans routeur ; option routeur"
```

Le déploiement sur sanroque (copie dans `~/.openclaw/extensions/derisk-deleg/`, redémarrage
de la gateway) attend l'accord de l'utilisateur.

---

### Task 2 : élagage de `openclaw.json` par profil (`profil_config.py`)

**Files :**
- Create : `openclaw-config/profil_config.py`
- Test : `openclaw-config/test_profil_config.py`

**Interfaces :**
- Consumes : gabarit `openclaw-config/openclaw.json.template` après `envsubst`.
- Produces : `appliquer_profil(config: dict, modules: set[str], cerveau: str) -> dict` (pure)
  et CLI `python3 profil_config.py <openclaw.json> --modules "wiki scout obfuscateur" --cerveau modal|ollama`
  (réécrit le fichier en place). Utilisé par la Task 5.

Correspondances (constantes du module) :

```python
AGENTS_DE = {"wiki": ["wiki"], "scout": ["scout"], "courrier": ["consignes"], "gog": ["gog"]}
OUTILS_DE = {
    "wiki": ["wiki_capture", "wiki_status", "wiki_ingest", "wiki_query", "wiki_tags",
             "wiki_kb_update", "wiki_search", "wiki_review", "wiki_verify", "wiki_delete"],
    "scout": ["source_read"],
    "gog": ["gog", "gog_connect_start", "gog_drive_search"],
    "courrier": ["mail_inbox", "mail_chercher", "mail_lire", "mail_envoyer"],
}
OBFUSQUE_URL = "http://172.17.0.1:8001/v1"
MODELE_OBFUSQUE = "obfusque/qwen3-14b-h128-a1-h02"
```

Règles :
1. Pour chaque module absent : supprimer ses agents de `agents.entries` ; retirer ses outils
   de **toutes** les listes `allow` et `deny` (globale `tools.sandbox.tools` et par agent).
2. `obfuscateur` présent : `models.providers.obfusque.baseUrl = OBFUSQUE_URL` ; absent :
   supprimer le fournisseur `obfusque` et son entrée dans `agents.defaults.models`.
3. `cerveau == "modal"` : `agents.entries.main.model = {"primary": MODELE_OBFUSQUE}` et
   `agents.defaults.model.primary = MODELE_OBFUSQUE`.
4. `laya` absent : `plugins.entries["derisk-deleg"].config = {"routeur": False}`.
5. Agent `wiki` présent et `ollama` non choisi (`cerveau != "ollama"`) : dans son
   `sandbox.docker.env`, `WIKI_LLM_BACKEND=openai`, `WIKI_INGEST_LLM_BACKEND=openai`,
   `OPENAI_BASE_URL=OBFUSQUE_URL` (le moteur `openai` d'ingestion n'a pas d'URL propre : il
   lit `OPENAI_BASE_URL`, cf. `_ingest_llm()` dans `Wiki_LM/tools/wiki.py`),
   `OPENAI_MODEL`/`WIKI_INGEST_LLM_MODEL=qwen3-14b-h128-a1-h02`, `OPENAI_API_KEY_FILE`
   supprimée, `OPENAI_API_KEY=local` ; supprimer les clés `OLLAMA_*` et `WIKI_INGEST_LLM_FALLBACK_*`.
   Retirer aussi de `sandbox.docker.binds` le montage `…/secrets/infomaniak-key:/run/infomaniak-key:ro` :
   sans clé Infomaniak (profil léger), le fichier source n'existe pas et Docker refuserait de
   lancer le conteneur. L'avertissement « INFOMANIAK_API_KEY absent » de l'installateur ne
   s'affiche plus quand le cerveau n'est pas Infomaniak.

- [ ] **Step 1 : tests**

```python
import json, re
from pathlib import Path
import pytest
from profil_config import appliquer_profil, OUTILS_DE

GABARIT = Path(__file__).parent / "openclaw.json.template"

def _config():
    texte = re.sub(r"\$\{[A-Z_]+\}", "0", GABARIT.read_text())
    return json.loads(texte)

def _toutes_listes(c):
    listes = [c["tools"]["sandbox"]["tools"].get(k, []) for k in ("allow", "deny")]
    for a in c["agents"]["entries"].values():
        t = a.get("tools", {})
        listes.append(t.get("deny", []) if isinstance(t.get("deny"), list) else [])
        st = t.get("sandbox", {}).get("tools", {})
        listes += [st.get("allow", []), st.get("deny", [])]
    return [x for l in listes for x in l]

LEGER = {"wiki", "scout", "obfuscateur"}

def test_leger_sans_agents_absents():
    c = appliquer_profil(_config(), LEGER, "modal")
    assert set(c["agents"]["entries"]) == {"main", "wiki", "scout"}

def test_leger_aucun_outil_de_module_absent():
    c = appliquer_profil(_config(), LEGER, "modal")
    absents = set(OUTILS_DE["courrier"]) | set(OUTILS_DE["gog"])
    assert absents.isdisjoint(_toutes_listes(c))

def test_leger_cerveau_et_proxy_obfusques():
    c = appliquer_profil(_config(), LEGER, "modal")
    assert c["agents"]["entries"]["main"]["model"]["primary"] == "obfusque/qwen3-14b-h128-a1-h02"
    assert c["models"]["providers"]["obfusque"]["baseUrl"] == "http://172.17.0.1:8001/v1"

def test_leger_routeur_desactive():
    c = appliquer_profil(_config(), LEGER, "modal")
    assert c["plugins"]["entries"]["derisk-deleg"]["config"] == {"routeur": False}

def test_leger_wiki_sans_ollama():
    env = appliquer_profil(_config(), LEGER, "modal")["agents"]["entries"]["wiki"]["sandbox"]["docker"]["env"]
    assert env["WIKI_INGEST_LLM_BACKEND"] == "openai"
    assert env["OPENAI_BASE_URL"] == "http://172.17.0.1:8001/v1"
    assert not any(k.startswith("OLLAMA_") for k in env)

def test_leger_sans_montage_de_cle_infomaniak():
    c = appliquer_profil(_config(), LEGER, "modal")
    binds = c["agents"]["entries"]["wiki"]["sandbox"]["docker"]["binds"]
    assert not any("infomaniak-key" in b for b in binds)

def test_lourd_garde_tout():
    tous = {"wiki", "scout", "obfuscateur", "laya", "courrier", "cron", "gog"}
    c = appliquer_profil(_config(), tous, "ollama")
    assert set(c["agents"]["entries"]) == {"main", "wiki", "scout", "consignes", "gog"}
    assert "config" not in c["plugins"]["entries"]["derisk-deleg"] or \
        c["plugins"]["entries"]["derisk-deleg"]["config"].get("routeur", True)

def test_aucune_valeur_propre_a_sanroque():
    texte = json.dumps(appliquer_profil(_config(), LEGER, "modal"))
    assert "/home/mauceric" not in texte and "Arbath" not in texte and "220537738" not in texte
```

Le dernier test échoue tant que la Task 4 n'a pas remplacé l'identifiant Telegram en dur :
le marquer `@pytest.mark.xfail(strict=True, reason="Task 4")`, que la Task 4 retire.

- [ ] **Step 2 : lancer, voir échouer** —
  `cd ~/Secretarius/openclaw-config && ../Wiki_LM/.venv/bin/python -m pytest test_profil_config.py -q` ;
  Expected : `ModuleNotFoundError: profil_config`.

- [ ] **Step 3 : implémenter `profil_config.py`** (constantes ci-dessus, `appliquer_profil` selon
  les règles 1 à 5, CLI `argparse` qui lit, applique, écrit avec `indent=1, ensure_ascii=False`
  et un saut de ligne final, puis copie vers `openclaw.json.bak` comme le fait déjà l'installateur
  pour éviter la restauration anti-clobber). `copy.deepcopy` de l'entrée : la fonction ne
  modifie pas son argument.

- [ ] **Step 4 : lancer, voir passer** — Expected : 7 passed, 1 xfailed.

- [ ] **Step 5 : commit** — `git add openclaw-config/profil_config.py openclaw-config/test_profil_config.py`
  puis `git commit -m "feat(install): élagage d'openclaw.json par profil"`.

---

### Task 3 : bibliothèque de l'installateur (profil, ZIM) et profils

**Files :**
- Create : `lib/installation.sh`, `profils/leger.conf`, `profils/lourd.conf`
- Test : `tests_installation/test_installation_lib.py`

**Interfaces :**
- Produces (Bash) : `charger_profil <nom>` (source `profils/<nom>.conf`, échoue si absent) ;
  `resoudre_zim` (imprime le dossier retenu ou une chaîne vide ; code 1 si `WIKI_ZIM_DIR`
  fixée et invalide). Utilisées par la Task 5.

`profils/leger.conf` :

```bash
# Profil léger (santiago) : tous les modèles déportés, ni Laya ni modèle local.
MODULES="${MODULES:-wiki scout obfuscateur}"
MAIN_BRAIN="${MAIN_BRAIN:-modal}"
WIKI_LLM="${WIKI_LLM:-obfusque}"
COFFRE="${COFFRE:-Publication}"
```

`profils/lourd.conf` :

```bash
# Profil lourd (type sanroque).
MODULES="${MODULES:-wiki scout obfuscateur laya courrier cron gog}"
MAIN_BRAIN="${MAIN_BRAIN:-ollama}"
WIKI_LLM="${WIKI_LLM:-local}"
COFFRE="${COFFRE:-Secretarius}"
```

`lib/installation.sh` :

```bash
# Fonctions de l'installateur, sourcées par install.sh (testées par
# tests_installation/test_installation_lib.py).

charger_profil() {  # $1 = leger | lourd
  local f="${SECRETARIUS_ROOT}/profils/$1.conf"
  [[ -f "$f" ]] || { echo "Profil inconnu : $1 (leger | lourd)" >&2; return 1; }
  # shellcheck source=/dev/null
  source "$f"
}

# Dossier des ZIM de Wikipédia, fournis par l'utilisateur (spec 2026-10-12).
resoudre_zim() {
  if [[ -n "${WIKI_ZIM_DIR:-}" ]]; then
    if compgen -G "${WIKI_ZIM_DIR}/*.zim" >/dev/null; then echo "$WIKI_ZIM_DIR"; return 0; fi
    echo "WIKI_ZIM_DIR=${WIKI_ZIM_DIR} : dossier absent ou sans fichier .zim" >&2
    return 1
  fi
  local defaut="${SECRETARIUS_ROOT}/Wiki_LM/zim"
  if compgen -G "${defaut}/*.zim" >/dev/null; then echo "$defaut"; return 0; fi
  if [[ "${INTERACTIVE:-false}" == true ]]; then
    local v; read -rp "Dossier des fichiers ZIM de Wikipédia (vide = aucun) : " v || true
    if [[ -n "$v" ]]; then WIKI_ZIM_DIR="$v" resoudre_zim; return $?; fi
  fi
  echo ""
}
```

- [ ] **Step 1 : tests** (appellent Bash, sans aucun effet système)

```python
import subprocess
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

def _bash(script, env=None):
    import os
    e = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], **(env or {})}
    return subprocess.run(["bash", "-c", f'SECRETARIUS_ROOT="{RACINE}"; source lib/installation.sh; {script}'],
                           cwd=RACINE, env=e, capture_output=True, text=True)

def test_profil_leger():
    r = _bash('charger_profil leger && echo "$MODULES|$MAIN_BRAIN|$WIKI_LLM"')
    assert r.stdout.strip() == "wiki scout obfuscateur|modal|obfusque"

def test_profil_inconnu():
    r = _bash("charger_profil moyen")
    assert r.returncode == 1 and "Profil inconnu" in r.stderr

def test_variable_explicite_prioritaire():
    r = _bash('charger_profil leger && echo "$MODULES"', {"MODULES": "wiki"})
    assert r.stdout.strip() == "wiki"

def test_zim_variable_valide(tmp_path):
    (tmp_path / "wikipedia_fr.zim").write_text("x")
    r = _bash("resoudre_zim", {"WIKI_ZIM_DIR": str(tmp_path)})
    assert r.returncode == 0 and r.stdout.strip() == str(tmp_path)

def test_zim_variable_invalide_echoue(tmp_path):
    r = _bash("resoudre_zim", {"WIKI_ZIM_DIR": str(tmp_path)})
    assert r.returncode == 1 and "sans fichier .zim" in r.stderr

def test_zim_absent_non_interactif(tmp_path):
    r = subprocess.run(["bash", "-c", f'SECRETARIUS_ROOT="{tmp_path}"; source "{RACINE}/lib/installation.sh"; resoudre_zim'],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == ""
```

- [ ] **Step 2 : voir échouer** — `~/Secretarius/Wiki_LM/.venv/bin/python -m pytest tests_installation -q` ;
  Expected : échecs (`lib/installation.sh` absent).
- [ ] **Step 3 : créer les trois fichiers ci-dessus.**
- [ ] **Step 4 : voir passer** — Expected : 6 passed.
- [ ] **Step 5 : commit** — `git commit -m "feat(install): profils léger/lourd, résolution des ZIM"`.

---

### Task 4 : gabarit et wiki sans valeurs propres à sanroque

**Files :**
- Modify : `openclaw-config/openclaw.json.template` (l. 168 bind ZIM ; l. 315 et 331 identifiant Telegram)
- Modify : `openclaw-config/install.sh` (liste `envsubst` du gabarit)
- Modify : `Wiki_LM/tools/wiki.py` (`MODEL_ALIASES["obfusque"]["base_url"]`)
- Test : `openclaw-config/test_profil_config.py` (retirer le `xfail`), `Wiki_LM/tests/test_wiki_cli.py`

**Interfaces :**
- Produces : variables de gabarit `${WIKI_ZIM_DIR}` et `${TELEGRAM_OWNER_ID}` (fournies par la
  Task 5) ; `MODEL_ALIASES["obfusque"]["base_url"]` lu dans `OBFUSQUE_BASE_URL` (défaut
  `http://127.0.0.1:8001/v1`, inchangé sur sanroque).

- [ ] **Step 1 : tests.** Retirer le `xfail` de `test_aucune_valeur_propre_a_sanroque`. Dans
  `Wiki_LM/tests/test_wiki_cli.py` :

```python
def test_alias_obfusque_suit_obfusque_base_url(monkeypatch):
    import importlib, wiki
    monkeypatch.setenv("OBFUSQUE_BASE_URL", "http://172.17.0.1:8001/v1")
    importlib.reload(wiki)
    assert wiki.MODEL_ALIASES["obfusque"]["base_url"] == "http://172.17.0.1:8001/v1"
```

- [ ] **Step 2 : voir échouer** — les deux tests FAIL.
- [ ] **Step 3 : implémenter.** Gabarit : `"${WIKI_ZIM_DIR}:/zim:ro"` ; `"telegram:${TELEGRAM_OWNER_ID}"`
  et `"approvers": [${TELEGRAM_OWNER_ID}]`. Ajouter `${WIKI_ZIM_DIR} ${TELEGRAM_OWNER_ID}` aux
  listes `export` et `envsubst` de `openclaw-config/install.sh` (l. 128-129). `wiki.py` :
  `"base_url": os.environ.get("OBFUSQUE_BASE_URL", "http://127.0.0.1:8001/v1")`. Le test
  `test_aucune_valeur_propre_a_sanroque` substitue déjà `0` à toutes les variables.
  Si aucun ZIM n'est retenu, la Task 5 passe `WIKI_ZIM_DIR` vers un dossier vide créé par
  l'installateur (`~/Secretarius/Wiki_LM/zim`), pour que le montage reste valide.
- [ ] **Step 4 : voir passer** — les deux suites concernées vertes ; puis
  `cd ~/Secretarius/Wiki_LM && .venv/bin/python -m pytest -q` (519+ verts).
- [ ] **Step 5 : commit** — `git commit -m "fix(install): ZIM et propriétaire Telegram en variables, proxy obfusqué configurable"`.

---

### Task 5 : `install.sh --profil` et découpage en modules

**Files :**
- Create : `openclaw-config/modules.sh`
- Modify : `install.sh`, `openclaw-config/install.sh`, `install.conf`

**Interfaces :**
- Consumes : `charger_profil`, `resoudre_zim` (Task 3) ; `profil_config.py` (Task 2) ;
  variables de gabarit (Task 4).
- Produces : `module_wiki`, `module_scout`, `module_obfuscateur`, `module_laya`,
  `module_courrier`, `module_cron`, `module_gog` (fonctions Bash sans argument, qui lisent
  l'environnement exporté) ; tableau `SERVICES_A_VERIFIER` (noms d'unités et URL de santé,
  forme `"unite|url"`, URL facultative) que chaque module complète, lu par la Task 9.

Répartition du code existant de `openclaw-config/install.sh` (numéros de lignes actuels) :

| Bloc | Lignes | Destination |
|---|---|---|
| secrets existants, NVM, binaire, jeton, clé Infomaniak | 1-110 | socle (inchangé) |
| génération `openclaw.json`, réconciliation du jeton | 112-160 | socle ; **ajouter** l'appel `profil_config.py --modules "$MODULES" --cerveau "$MAIN_BRAIN"` juste après la génération (seulement quand le fichier vient d'être généré) |
| auth par agent | 162-186 | socle, mais `_set_auth` ignore un agent absent d'`openclaw.json` |
| `LLM_BACKEND` → modèle `main` | 188-206 | **supprimé** (remplacé par `MAIN_BRAIN` + `profil_config.py`) |
| `gateway.systemd.env`, unité gateway, switch-model, enveloppe | 208-275 | socle |
| `wiki-lm-server` | 277-291 | `module_wiki` |
| `tiron-router`, `laya[onnx]` | 293-308 | `module_laya` |
| `MAIN_BRAIN` (auto/ollama/modal 8B) | 310-417 | socle réduit à `ollama` (fournisseur Ollama, comme l. 347-358) ; le cas `modal` passe à `module_obfuscateur` (Task 6) |
| minuteur d'embeddings | 423-438 | `module_wiki` |
| relève du courrier | 440-450 | `module_courrier` |
| espaces de travail | 452-492 | socle pour `workspace` ; la boucle des sous-agents ne copie que ceux des modules présents (`wiki`→`workspace-wiki`, `scout`→`workspace-scout`, `gog`→`workspace-gog`, `courrier`→`workspace-consignes`) |
| dossiers scout, `.gog-config` | 494-499 | `module_scout`, `module_gog` |
| garde d'injection, scout-watcher | 501-539 | `module_scout` |
| finalisation et messages | 541-564 | socle ; messages refaits en Task 9 |

Dans `install.sh` (racine) :
- options `--profil NOM` (obligatoire ; absent → usage et code 2) ; `--llm` et le mode
  interactif sur le LLM sont **supprimés** (le profil décide) ;
- ordre : `source install.conf` → `source lib/installation.sh` → `charger_profil` →
  `--env-file` (variables explicites prioritaires car les profils utilisent `${VAR:-défaut}`) ;
- prérequis : ajouter « dépôt dans `~/Secretarius` » (`[[ "$SECRETARIUS_ROOT" == "$HOME/Secretarius" ]]`),
  Docker accessible (devient **bloquant**), `TELEGRAM_BOT_TOKEN` et `TELEGRAM_OWNER_ID`
  fournis (bloquants) ;
- `OBSIDIAN_PATH` dérivé de `COFFRE` : `${OBSIDIAN_PATH:-$HOME/Documents/${COFFRE}}`, créé
  s'il manque ;
- `WIKI_ZIM_DIR="$(resoudre_zim)" || exit 1` ; vide → `mkdir -p Wiki_LM/zim`,
  `WIKI_ZIM_DIR=$SECRETARIUS_ROOT/Wiki_LM/zim`, `ZIM_ETAT="absent : enrichissement par Wikipédia désactivé"` ;
  sinon `ZIM_ETAT` = liste des fichiers `.zim` trouvés ;
- étape 4b (FAQ de faits) et démarrage de `tiron-router` déplacés dans `module_laya` ;
- images Docker : l'image de base et `secretarius-wiki` dans `module_wiki` ; `tiron` dans le
  socle ; `gog` dans `module_gog` ; tout échec de construction devient bloquant (`exit 1`
  avec la commande `docker build` à relancer) ;
- après le socle : `for m in $MODULES; do "module_$m"; done` (nom inconnu → erreur).

`install.conf` : retirer `LLM_BACKEND=deepseek` et `MAIN_BRAIN` auto (le profil fixe).

- [ ] **Step 1 : test de l'analyse des options** (ajouté à `tests_installation/test_installation_lib.py`)

```python
def test_install_sans_profil_refuse():
    r = subprocess.run(["bash", str(RACINE / "install.sh")], capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "--profil" in r.stdout + r.stderr

def test_install_profil_inconnu_refuse():
    r = subprocess.run(["bash", str(RACINE / "install.sh"), "--profil", "moyen"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and "Profil inconnu" in r.stderr
```

Ces deux tests s'arrêtent avant toute action système (l'analyse des options et le chargement
du profil précèdent les prérequis).

- [ ] **Step 2 : voir échouer.**
- [ ] **Step 3 : restructurer selon le tableau.** Chaque bloc déplacé garde son code et ses
  commentaires ; seules changent la condition d'appel et les variables de chemin. Pas de
  `systemctl` lancé depuis la machine de développement pendant cette tâche.
- [ ] **Step 4 : contrôles** — `bash -n install.sh openclaw-config/install.sh openclaw-config/modules.sh lib/installation.sh` ;
  `~/Secretarius/Wiki_LM/.venv/bin/python -m pytest tests_installation openclaw-config/test_profil_config.py -q` ;
  Expected : syntaxe valide, tout vert.
- [ ] **Step 5 : contrôle de non-trace (lecture).** `grep -n "systemctl\|cp \|docker build" openclaw-config/install.sh install.sh`
  ne montre plus aucune ligne propre à un module hors de `modules.sh`.
- [ ] **Step 6 : commit** — `git commit -m "feat(install): --profil et découpage en modules"`.

---

### Task 6 : module `obfuscateur` (un seul proxy, 14B)

**Files :**
- Create : `openclaw-config/obfuscator-proxy.service`
- Modify : `openclaw-config/modules.sh` (`module_obfuscateur`)

**Interfaces :**
- Consumes : `ALOEPRI_API_KEY` (secret) ; venv de Wiki_LM.
- Produces : proxy sur `172.17.0.1:8001` ; `OBFUSQUE_BASE_URL=http://172.17.0.1:8001/v1`
  écrit dans `Wiki_LM/.env` (Task 7) ; entrée `obfuscator-proxy|http://172.17.0.1:8001/health`
  dans `SERVICES_A_VERIFIER`.

Unité (copie de `~/obfuscator/systemd/obfuscator-proxy-docker.service`, dans le dépôt
Secretarius pour être versionnée avec l'installateur) avec, en plus :

```ini
# Le proxy écoute sur docker0 (172.17.0.1) : l'hôte comme les conteneurs l'atteignent,
# un seul processus au lieu de deux (~960 Mo chacun). Au démarrage de la machine, docker0
# peut ne pas exister encore : on attend l'adresse, Restart couvre le reste.
ExecStartPre=/bin/sh -c 'for i in $(seq 1 30); do ip -4 addr show docker0 | grep -q 172.17.0.1 && exit 0; sleep 2; done; exit 1'
Restart=always
RestartSec=5
```

`module_obfuscateur` reprend le cas `modal` des l. 359-411 actuelles (clonage de
`~/obfuscator`, dépendances du proxy, clé de permutation déterministe, `proxy.env`), mais
installe `obfuscator-proxy.service` au lieu de `obfuscator-proxy-8b.service`, attend
`http://172.17.0.1:8001/health` (6 × 5 s, avertissement si Modal dort encore) et ne touche
plus à `openclaw.json` (fait par `profil_config.py`). Sans `ALOEPRI_API_KEY` : **échec**
(le profil léger n'a pas de repli).

- [ ] **Step 1 :** `bash -n openclaw-config/modules.sh` et `systemd-analyze --user verify openclaw-config/obfuscator-proxy.service`
  (Expected : aucune erreur autre que l'absence du binaire si le venv n'existe pas).
- [ ] **Step 2 :** écrire l'unité et la fonction.
- [ ] **Step 3 :** relancer les deux contrôles ; Expected : propres.
- [ ] **Step 4 : commit** — `git commit -m "feat(install): module obfuscateur, un seul proxy 14B sur docker0"`.

---

### Task 7 : module `wiki` (`.env` par profil, miroirs, branche web)

**Files :**
- Modify : `openclaw-config/modules.sh` (`module_wiki`), `install.sh` (étape 4 actuelle)
- Modify : `Wiki_LM/.env.template`

**Interfaces :**
- Consumes : `WIKI_LLM` (profil), `WIKI_ZIM_DIR` (Task 5), `OBFUSQUE_BASE_URL` (Task 6),
  `WIKI_VAULT_MIRRORS` (environnement, vide par défaut).
- Produces : `Wiki_LM/.env` complet ; `wiki-lm-server|http://127.0.0.1:5051/health` dans
  `SERVICES_A_VERIFIER`.

`module_wiki` :
1. crée `Wiki_LM/.env` depuis le gabarit s'il manque (ou `--force`) ; pose `WIKI_PATH`,
   `WIKI_ZIM_DIR`, `WIKI_VAULT_MIRRORS`, `OBFUSQUE_BASE_URL` (si `obfuscateur` est dans
   `$MODULES`), `DEEPSEEK_API_KEY`, et pour la branche web :
   `WIKI_INGEST_WEB_LLM_BACKEND=openai` (valeurs DeepSeek par défaut du code) ;
2. applique l'alias du profil avec le code existant, sans le dupliquer :
   `(cd Wiki_LM/tools && ../.venv/bin/python -c "import sys, json, wiki; r = wiki.op_switch_model(sys.argv[1]); print(json.dumps(r)); sys.exit('error' in r)" "$WIKI_LLM")` ;
3. installe `wiki-lm-server.service` et le minuteur d'embeddings (code déplacé en Task 5) ;
4. construit l'image de base et `secretarius-wiki` (bloquant).

`Wiki_LM/.env.template` : `WIKI_LLM_BACKEND`, `OPENAI_BASE_URL`, `OPENAI_MODEL` restent, mais
le commentaire d'en-tête dit que l'installateur les fixe selon le profil ; DeepSeek n'est plus
présenté comme le défaut.

- [ ] **Step 1 : test** (dans `tests_installation/`) : exécuter les points 1 et 2 de
  `module_wiki` sur une copie temporaire (`SECRETARIUS_ROOT=tmp_path` avec `Wiki_LM/tools`
  copié par `shutil.copytree` et un lien vers le venv réel), `WIKI_LLM=obfusque`,
  `OBFUSQUE_BASE_URL=http://172.17.0.1:8001/v1` ; vérifier dans le `.env` produit :
  `WIKI_LLM_BACKEND=openai`, `OPENAI_BASE_URL=http://172.17.0.1:8001/v1`,
  `WIKI_LOOKUP_OFFLINE=1`, `WIKI_INGEST_WEB_LLM_BACKEND=openai`. Pour rendre cela testable,
  les points 1-2 forment une fonction séparée `ecrire_env_wiki` dans `lib/installation.sh`.
- [ ] **Step 2 : voir échouer.** — **Step 3 : implémenter.** — **Step 4 : voir passer.**
- [ ] **Step 5 : commit** — `git commit -m "feat(install): module wiki, .env par profil et branche web"`.

---

### Task 8 : synchronisation du coffre (`obsidian-sync@.service`)

**Files :**
- Create : `openclaw-config/obsidian-sync@.service`
- Modify : `install.sh` (socle), `docs/components/obsidian.md` (§ dépôt d'un plugin)

```ini
[Unit]
Description=Synchronisation Obsidian du coffre %i (ob sync)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# Une synchro continue déjà lancée ne voit pas toujours un plugin déposé dans
# .obsidian : chaque (re)démarrage fait d'abord une synchro ponctuelle.
ExecStartPre=/usr/bin/env ob sync --path %h/Documents/%i
ExecStart=/usr/bin/env ob sync --continuous --path %h/Documents/%i
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
```

`ob` est installé par NVM : le socle écrit, comme pour l'enveloppe `openclaw`, une
enveloppe `~/.local/bin/ob` qui met le Node de NVM en tête du `PATH`, et l'unité appelle
`%h/.local/bin/ob`. Le socle active l'instance `obsidian-sync@${COFFRE}` **seulement si**
`ob sync-status --path ~/Documents/${COFFRE}` réussit ; sinon il ajoute au résumé les
commandes de liaison (étape de compte, comme l'appairage Telegram) :
`ob login`, `ob sync-create-remote` ou `ob sync-setup --vault <nom> --path ~/Documents/${COFFRE}`,
puis relancer `install.sh`.

`docs/components/obsidian.md` : remplacer « couper (Ctrl-C) et relancer dans tmux » par
`systemctl --user restart obsidian-sync@<coffre>`.

- [ ] **Step 1 :** `systemd-analyze --user verify` sur l'unité ; `bash -n install.sh`.
- [ ] **Step 2 : écrire.** — **Step 3 : recontrôler.**
- [ ] **Step 4 : commit** — `git commit -m "feat(install): synchronisation du coffre en service systemd"`.

---

### Task 9 : vérification finale et résumé

**Files :**
- Modify : `lib/installation.sh` (fonction `verifier_services`), `install.sh` (fin),
  `openclaw-config/install.sh` (messages l. 547-563)
- Test : `tests_installation/test_installation_lib.py`

**Interfaces :**
- Consumes : `SERVICES_A_VERIFIER` (`"unite|url"`).
- Produces : `verifier_services` → code 0 si tout répond, sinon 1 et la liste des muets.
  `SYSTEMCTL` et `CURL` sont des variables (défaut `systemctl --user` et `curl -sf --max-time 10`)
  pour que le test les remplace par des faux.

- [ ] **Step 1 : tests**

```python
def test_verification_tout_repond():
    r = _bash('SYSTEMCTL=true; CURL=true; SERVICES_A_VERIFIER=("a|http://x" "b"); verifier_services')
    assert r.returncode == 0

def test_verification_service_muet():
    r = _bash('SYSTEMCTL=true; CURL=false; SERVICES_A_VERIFIER=("wiki-lm-server|http://x"); verifier_services')
    assert r.returncode == 1 and "wiki-lm-server" in r.stdout + r.stderr
```

- [ ] **Step 2 : voir échouer.**
- [ ] **Step 3 : implémenter.** `verifier_services` : pour chaque entrée,
  `$SYSTEMCTL is-active --quiet <unite>` puis, si une URL est donnée, `$CURL <url>` ; trois
  essais espacés de 10 s par URL (le proxy peut attendre le réveil de Modal). `install.sh`
  l'appelle en dernier et sort en code 1 en cas d'échec. Le résumé affiche : profil, modules,
  `ZIM_ETAT`, coffre, et les étapes de compte restantes (appairage Telegram, liaison Obsidian).
  Supprimer de `openclaw-config/install.sh` la liste des modèles Infomaniak et le paragraphe
  « Post-install — plugin derisk-deleg » (l. 548-559), devenus faux.
- [ ] **Step 4 : voir passer.** — **Step 5 : commit** — `git commit -m "feat(install): vérification finale bloquante et résumé"`.

---

### Task 10 : `amorcage-systeme.sh`

**Files :**
- Create : `amorcage-systeme.sh`

Script `sudo` idempotent pour Ubuntu (24.04+), qui affiche chaque action avant de la faire :
1. paquets : `git python3 python3-venv gettext curl docker.io` ; `usermod -aG docker $SUDO_USER` ;
2. swap de 4 Go (`/swapfile`, ajouté à `/etc/fstab`) si `swapon --show` est vide ;
3. Tailscale par le dépôt officiel (`curl -fsSL https://tailscale.com/install.sh | sh`), sans
   `tailscale up` (l'authentification reste à l'utilisateur ; le script affiche la commande
   `tailscale up --hostname=<nom> --ssh` et rappelle d'exempter la clé d'expiration) ;
4. pour `$SUDO_USER` (via `sudo -u`) : NVM, Node LTS, `npm install -g openclaw obsidian-headless` ;
5. `loginctl enable-linger $SUDO_USER` (services utilisateur sans session ouverte) ;
6. message final : se reconnecter (groupe docker), puis lancer `install.sh --profil …`.

- [ ] **Step 1 :** `bash -n amorcage-systeme.sh` ; le script refuse de tourner sans `sudo`
  (`[[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]]`), vérifié par
  `bash amorcage-systeme.sh; echo $?` sans sudo (Expected : message et code 1).
- [ ] **Step 2 : commit** — `git commit -m "feat(install): amorçage système (Docker, Node, OpenClaw, Tailscale, swap)"`.

---

### Task 11 : désinstallation par profil

**Files :**
- Modify : `openclaw-config/uninstall.sh`

Ajouter à sa liste d'unités `obfuscator-proxy.service`, `obsidian-sync@*.service`
(instances actives listées par `systemctl --user list-units 'obsidian-sync@*' --plain --no-legend`),
`tiron-mail-releve.{service,timer}`, `wiki-lm-embed.{service,timer}`. Les unités absentes sont
ignorées (déjà le cas). Ne supprime ni le coffre, ni `~/obfuscator`, ni les ZIM.

- [ ] **Step 1 :** `bash -n` ; relecture de la liste contre `SERVICES_A_VERIFIER` des modules.
- [ ] **Step 2 : commit** — `git commit -m "feat(install): désinstallation couvrant les nouveaux services"`.

---

### Task 12 : documentation

**Files :**
- Modify : `openclaw-config/INSTALL.md`, `README.md` (section installation)
- Modify (coffre, hors dépôt) : `~/Documents/Secretarius/Secretarius_dev/reprise-secretarius.md`
  — **soumettre chaque modification à l'utilisateur avant de l'écrire** (convention du document).

`INSTALL.md` : amorçage, profils, fichier `--env-file` (secrets et réglages de machine :
`TELEGRAM_BOT_TOKEN` propre à l'instance, `TELEGRAM_OWNER_ID`, `ALOEPRI_API_KEY`,
`DEEPSEEK_API_KEY`, `WIKI_ZIM_DIR`, `WIKI_VAULT_MIRRORS`), ZIM (catalogue Kiwix
`https://download.kiwix.org/zim/wikipedia/`, variante `mini` pour une petite machine,
`nopic` sinon), étapes de compte, vérification, désinstallation.

- [ ] **Step 1 : écrire.** — **Step 2 : commit** — `git commit -m "docs(install): profils, amorçage, ZIM"`.

---

### Task 13 : santiago, de bout en bout (avec l'utilisateur)

Pas de code. Chaque action sur santiago qui demande `sudo`, un secret ou un compte est faite
par l'utilisateur ou avec son accord explicite.

1. Fusion de la branche dans `main` et `git push` : **accord de l'utilisateur**.
2. Sur santiago : `git clone` dans `~/Secretarius` ; l'utilisateur lance
   `sudo ./amorcage-systeme.sh`, puis `tailscale up`.
3. L'utilisateur prépare `~/.config/secrets.env` de santiago : **nouveau** jeton de robot
   Telegram, `TELEGRAM_OWNER_ID`, `ALOEPRI_API_KEY`, `DEEPSEEK_API_KEY`,
   `WIKI_ZIM_DIR=/mnt/HC_Volume_106134083`.
4. Liaison Obsidian du coffre `Publication` (`ob login`, `ob sync-create-remote`, `ob sync-setup`).
5. `./install.sh --profil leger --env-file ~/.config/secrets.env` → vérification finale verte.
6. Essais Telegram (second robot) : `/wikistatus`, `/c <texte>`, `/ingest`, `/q <question>`,
   `/source <url>` ; la page produite apparaît dans `Publication` sur la tablette.
7. Idempotence : relancer l'étape 5, `git status` du dépôt propre, aucun fichier régénéré.
8. Mesure de mémoire (`free -h`, `ps -eo rss,cmd`) au repos puis après un `/q` (BGE-M3
   chargé), consignée dans la spec.
9. Désinstallation à blanc : non lancée sur santiago (elle détruirait l'instance) ; testée sur
   le VPS du morceau 4.
