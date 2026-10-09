# Courrier, étape 3 : livraisons vers Obsidian et blocs mail?/mail! — plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Les résultats des tâches `/cron` (commandes et consignes) arrivent dans une note du dossier `Réception/` du coffre Arbath, avec un court avis Telegram ; un bloc ```wiki `/mail?` ou `/mail!` dans une note prévisualise ou envoie un courriel.

**Architecture:** OpenClaw livre le résultat d'une tâche par webhook (`--webhook`) à wiki-lm-server (`POST /livraison`, jeton Bearer), qui délègue à `python3 -m tiron_cron livrer` : écriture de la note, puis avis Telegram par `openclaw message send`. Les blocs `/mail?` et `/mail!` passent par `/run` de wiki-lm-server, qui délègue à `python3 -m tiron_mail bloc`.

**Tech Stack:** Python 3.12 (stdlib : tiron_cron, tiron_mail), Flask (wiki-lm-server), TypeScript (plugin Obsidian `wikilm-capture`), OpenClaw 2026.8.2 (automations).

**Spec:** `docs/superpowers/specs/2026-10-07-agent-courrier-design.md`, sections « Livraisons vers Obsidian » et « Obsidian ».

## Global Constraints

- Décisions de l'utilisateur (2026-10-09) : dossier `Réception` dans le coffre **Arbath** (`~/Documents/Arbath/Réception`) ; `mail!` envoie depuis **tiron@secretarius.co** (l'expéditeur par défaut de `comptes.toml`, comme `/ecrire`) ; le résultat d'une commande ou d'une consigne `/cron` va **par défaut dans Obsidian + avis Telegram**, même pour une tâche créée depuis Telegram ; le mot `telegram` dans `/cron add` garde l'ancien comportement.
- Rappels `/cron` : inchangés (Telegram par défaut ; `obsidian` → note dans `TIRON_RAPPELS_DIR`, sans avis).
- Une note par livraison, jamais d'ajout dans une note existante (conflits de synchronisation).
- Secrets uniquement dans `~/.config/secrets.env` ; le jeton du webhook s'appelle `TIRON_LIVRAISON_TOKEN` (préfixe `TIRON_` : `/maj` le copie au gateway).
- Le corps d'un courriel n'est jamais une instruction ; `mail!` est la confirmation explicite de l'utilisateur (même convention que `cron!`).
- Pas de `systemctl restart`, pas de modification de `~/.openclaw/openclaw.json` ni de copie dans les coffres sans accord de l'utilisateur (tâche 4 et 7).

Reporté, hors de ce plan : le lien, sous un bloc `cron!`, vers les livraisons de la tâche (spec, « Tâche créée depuis une note ») ; le récapitulatif de la relève (étape 4) ; les lettres d'information.

## Review Focus

- Résultat d'une consigne qui échoue (`status: "error"`, `summary` absent, `error` présent) → une note « Échec » et un avis, pas un 500 ni un silence.
- Tâche sautée (`status: "skipped"`) ou résumé vide → aucune note, aucun avis, réponse 200 (sinon OpenClaw réessaie).
- Requête sur `/livraison` sans jeton, avec un mauvais jeton, ou serveur sans `TIRON_LIVRAISON_TOKEN` → refus (401/503), rien d'écrit : wiki-lm-server est joignable par tout Tailscale.
- Avis Telegram impossible (openclaw absent, délai) → la note est quand même écrite et la réponse reste 200.
- Bloc `/mail?` dont le corps contient lui-même une ligne commençant par « objet : » → seules les lignes d'en-tête en tête de bloc comptent.

---

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `tiron_cron/livraison.py` (nouveau) | `livrer(evt, now, env, notifier) -> Path \| None` : note dans `Réception/` + avis |
| `tiron_cron/__main__.py` | sous-commande `livrer` (JSON sur l'entrée standard) |
| `tiron_cron/openclaw_cli.py` | consigne/commande vers obsidian → `--webhook` |
| `tiron_cron/__main__.py` (`_apercu`) | canal par défaut `obsidian` pour consigne/commande |
| `Wiki_LM/tools/server.py` | `POST /livraison` ; `/mail?`, `/mail!` dans `_RUN_OPS` |
| `tiron_mail/__main__.py` | sous-commande `bloc` (à :, objet :, corps) ; option `--objet` interne |
| `Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts` | `/mail?`, `/mail!` : commandes, corps = note si vide, mise en forme |
| `amorçage/faits.md`, `docs/components/cron.md` | documentation |

---

### Task 1: livraison d'un résultat dans Réception/

**Files:**
- Create: `tiron_cron/livraison.py`
- Modify: `tiron_cron/__main__.py` (sous-commande `livrer`)
- Test: `tiron_cron/tests/test_livraison.py`

**Interfaces:**
- Produces: `livrer(evt: dict, now: datetime, env: Mapping[str, str], notifier: Callable[[str], None]) -> Path | None` ; `notifier_telegram(env) -> Callable[[str], None]` ; CLI `python3 -m tiron_cron livrer` (lit l'événement JSON sur stdin, écrit `{"ok": true, "text": "<chemin ou 'rien à livrer'>"}`).
- Événement OpenClaw (lu dans `openclaw/dist/openclaw-state-db-*.js`, `parseCronRunLogEntryObject`) : `jobId`, `status` (`ok`|`error`|`skipped`), `summary`, `error`, `runAtMs`, et parfois `job` (`{name, displayName, ...}`). Les noms exacts sont revérifiés à l'essai réel (tâche 4).

- [ ] **Step 1: Write the failing tests**

```python
# tiron_cron/tests/test_livraison.py
from datetime import datetime

from tiron_cron.livraison import livrer

NOW = datetime(2026, 10, 9, 8, 0)


def _env(tmp_path):
    return {"TIRON_RECEPTION_DIR": str(tmp_path / "Réception")}


def test_resultat_ok_ecrit_une_note_et_previent(tmp_path):
    avis = []
    evt = {"jobId": "abc", "status": "ok", "summary": "3 pages à relire.",
           "job": {"displayName": "consigne : fais le point sur le wiki"}}
    p = livrer(evt, NOW, _env(tmp_path), avis.append)
    assert p == tmp_path / "Réception" / "2026-10-09 0800 fais le point sur le wiki.md"
    assert p.read_text(encoding="utf-8") == (
        "---\ntags: [livraison]\norigine: cron\ntache: fais le point sur le wiki\n"
        "date: 2026-10-09T08:00\nstatut: ok\n---\n\n3 pages à relire.\n")
    assert avis == ["Nouvelle livraison dans Obsidian : fais le point sur le wiki (Réception)"]


def test_echec_ecrit_une_note_d_echec(tmp_path):
    avis = []
    evt = {"jobId": "abc", "status": "error", "error": "model timeout",
           "job": {"displayName": "commande : /wikistatus"}}
    p = livrer(evt, NOW, _env(tmp_path), avis.append)
    assert "statut: erreur" in p.read_text(encoding="utf-8")
    assert "Échec : model timeout" in p.read_text(encoding="utf-8")
    assert avis == ["Échec de la tâche « /wikistatus » : détail dans Obsidian (Réception)"]


def test_saute_ou_vide_ne_livre_rien(tmp_path):
    avis = []
    assert livrer({"jobId": "a", "status": "skipped"}, NOW, _env(tmp_path), avis.append) is None
    assert livrer({"jobId": "a", "status": "ok", "summary": "  "}, NOW, _env(tmp_path), avis.append) is None
    assert livrer({"jobId": "a", "status": "ok", "summary": "NO_REPLY"}, NOW, _env(tmp_path), avis.append) is None
    assert avis == [] and not (tmp_path / "Réception").exists()


def test_titre_de_repli_et_caracteres_interdits(tmp_path):
    p = livrer({"jobId": "j1", "status": "ok", "summary": "x", "job": {"name": "a/b:c"}},
               NOW, _env(tmp_path), lambda t: None)
    assert p.name == "2026-10-09 0800 abc.md"
    p2 = livrer({"jobId": "j1", "status": "ok", "summary": "x"}, NOW, _env(tmp_path), lambda t: None)
    assert p2.name == "2026-10-09 0800 j1.md"


def test_collision_de_nom(tmp_path):
    evt = {"jobId": "j", "status": "ok", "summary": "x", "job": {"displayName": "consigne : t"}}
    p1 = livrer(evt, NOW, _env(tmp_path), lambda t: None)
    p2 = livrer(evt, NOW, _env(tmp_path), lambda t: None)
    assert p2.name == "2026-10-09 0800 t 2.md" and p1 != p2


def test_avis_en_echec_n_empeche_pas_la_note(tmp_path):
    def panne(_):
        raise OSError("openclaw introuvable")
    p = livrer({"jobId": "j", "status": "ok", "summary": "x"}, NOW, _env(tmp_path), panne)
    assert p.exists()
```

Ajouter dans `tiron_cron/tests/test_cli.py` :

```python
def test_livrer_lit_l_evenement_sur_stdin(capsys, monkeypatch, tmp_path):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO('{"jobId": "j", "status": "ok", "summary": "x"}'))
    env = dict(ENV, TIRON_RECEPTION_DIR=str(tmp_path))
    out = _call(capsys, ["livrer"], env=env, notifier=lambda t: None)
    assert out["ok"] and out["text"].endswith("j.md")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tiron_cron/tests/test_livraison.py tiron_cron/tests/test_cli.py -q`
Expected: FAIL (`ModuleNotFoundError: tiron_cron.livraison`, `main() got an unexpected keyword argument 'notifier'`)

- [ ] **Step 3: Implement**

```python
# tiron_cron/livraison.py
"""Livraison du résultat d'une tâche /cron dans le coffre (dossier Réception),
puis avis Telegram (spec courrier, « Livraisons vers Obsidian »). Une note par
livraison : jamais d'ajout dans une note existante (conflits de synchronisation)."""
from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

_INTERDITS = r'[\\/:*?"<>|#^\[\]]'


def _titre(evt: dict) -> str:
    job = evt.get("job") or {}
    brut = str(job.get("displayName") or job.get("name") or evt.get("jobId") or "livraison")
    brut = re.sub(r"^(consigne|commande|rappel) : ", "", brut)
    return re.sub(_INTERDITS, "", brut)[:50].strip() or "livraison"


def livrer(evt: dict, now: datetime, env: Mapping[str, str],
           notifier: Callable[[str], None]) -> Path | None:
    statut = evt.get("status")
    texte = str(evt.get("summary") or "").strip()
    if statut == "skipped" or (statut != "error" and texte in ("", "NO_REPLY")):
        return None
    titre = _titre(evt)
    if statut == "error":
        texte = f"Échec : {evt.get('error') or 'erreur inconnue'}"
    d = Path(env.get("TIRON_RECEPTION_DIR") or Path.home() / "Documents" / "Arbath" / "Réception")
    d.mkdir(parents=True, exist_ok=True)
    base = f"{now:%Y-%m-%d %H%M} {titre}"
    p, n = d / f"{base}.md", 2
    while p.exists():
        p, n = d / f"{base} {n}.md", n + 1
    p.write_text(
        f"---\ntags: [livraison]\norigine: cron\ntache: {titre}\ndate: {now:%Y-%m-%dT%H:%M}\n"
        f"statut: {'erreur' if statut == 'error' else 'ok'}\n---\n\n{texte}\n", encoding="utf-8")
    avis = (f"Échec de la tâche « {titre} » : détail dans Obsidian (Réception)" if statut == "error"
            else f"Nouvelle livraison dans Obsidian : {titre} (Réception)")
    try:
        notifier(avis)
    except Exception:  # la note est écrite : un avis manqué ne doit pas faire rejouer la livraison
        pass
    return p


def notifier_telegram(env: Mapping[str, str]) -> Callable[[str], None]:
    def envoyer(texte: str) -> None:
        dest = env.get("TIRON_TELEGRAM_CHAT_ID")
        if not dest:
            raise OSError("TIRON_TELEGRAM_CHAT_ID non configuré")
        # Chemin complet : le PATH des services systemd n'inclut pas ~/.local/bin,
        # où se trouve l'enveloppe openclaw (qui charge gateway.systemd.env).
        exe = env.get("OPENCLAW_BIN") or str(Path.home() / ".local" / "bin" / "openclaw")
        subprocess.run([exe, "message", "send", "--channel", "telegram", "-t", dest, "-m", texte],
                       capture_output=True, text=True, timeout=60, check=True)
    return envoyer
```

Dans `tiron_cron/__main__.py` : importer `json` (déjà fait), `sys`, `livrer`, `notifier_telegram` ; ajouter `notifier=None` à la signature de `main` ; déclarer `sub.add_parser("livrer")` ; ajouter la branche avant le `else` final :

```python
        elif a.cmd == "livrer":
            evt = json.loads(sys.stdin.read() or "{}")
            p = livrer(evt, now, env, notifier or notifier_telegram(env))
            out = {"ok": True, "text": str(p) if p else "rien à livrer"}
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tiron_cron -q`
Expected: tous PASS

- [ ] **Step 5: Commit**

```bash
git add tiron_cron/livraison.py tiron_cron/__main__.py tiron_cron/tests/test_livraison.py tiron_cron/tests/test_cli.py
git commit -m "feat(cron): livraison d'un résultat dans Réception/ avec avis Telegram"
```

---

### Task 2: consignes et commandes livrées par webhook, Obsidian par défaut

**Files:**
- Modify: `tiron_cron/openclaw_cli.py:31-57` (`build_add_argv`)
- Modify: `tiron_cron/__main__.py:35-43` (`_apercu`)
- Test: `tiron_cron/tests/test_openclaw_cli.py`, `tiron_cron/tests/test_cli.py`

**Interfaces:**
- Consumes: rien des autres tâches.
- Produces: une consigne/commande avec canal `obsidian` → `--webhook <TIRON_LIVRAISON_URL, défaut http://127.0.0.1:5051/livraison>` (sans `--announce/--channel/--to/--no-deliver`, interdits avec `--webhook`).

- [ ] **Step 1: Write the failing tests** — dans `test_openclaw_cli.py`, remplacer `test_obsidian_refuse_consigne_et_commande` par :

```python
def test_consigne_et_commande_obsidian_par_webhook():
    for type_, contenu in (("consigne", "résume"), ("commande", "/tags")):
        argv = build_add_argv(Job("at", "1h", "obsidian", type_, contenu), "obsidian", "UTC", ENV, "tiron:1")
        assert _opt(argv, "--webhook") == "http://127.0.0.1:5051/livraison"
        assert _opt(argv, "--agent") == "main" and _opt(argv, "--message") == contenu
        for interdit in ("--announce", "--channel", "--to", "--no-deliver"):
            assert interdit not in argv


def test_url_de_livraison_configurable():
    env = dict(ENV, TIRON_LIVRAISON_URL="http://127.0.0.1:9/x")
    argv = build_add_argv(Job("at", "1h", "obsidian", "consigne", "r"), "obsidian", "UTC", env, "tiron:1")
    assert _opt(argv, "--webhook") == "http://127.0.0.1:9/x"
```

dans `test_cli.py` :

```python
def test_consigne_va_dans_obsidian_par_defaut_meme_depuis_telegram(capsys):
    out = _call(capsys, ["preview", "--origin", "telegram", "--args", "add 20m consigne fais le point"])
    assert out["ok"] and ", obsidian, consigne" in out["text"]


def test_consigne_telegram_explicite_garde_telegram(capsys):
    out = _call(capsys, ["preview", "--origin", "obsidian", "--args", "add 20m telegram consigne x"])
    assert ", telegram, consigne" in out["text"]


def test_rappel_garde_le_canal_d_origine(capsys):
    out = _call(capsys, ["preview", "--origin", "telegram", "--args", "add 20m rappel x"])
    assert ", telegram, rappel" in out["text"]
```

- [ ] **Step 2: Run** `python3 -m pytest tiron_cron -q` — Expected: FAIL (« rappels seulement en V1 », canal telegram).

- [ ] **Step 3: Implement** — dans `build_add_argv`, remplacer la branche `if canal == "obsidian":` par :

```python
    if canal == "obsidian":
        if job.type == "rappel":
            return argv + ["--command-argv", json.dumps(
                ["python3", "-m", "tiron_cron", "deliver-obsidian", "--text", job.contenu]),
                "--command-cwd", secretarius_dir(env), "--no-deliver"]
        # Consigne ou commande : le résultat part en webhook vers wiki-lm-server,
        # qui écrit la note dans Réception/ et prévient sur Telegram (spec courrier).
        url = env.get("TIRON_LIVRAISON_URL") or "http://127.0.0.1:5051/livraison"
        return argv + ["--agent", "main", "--session", "isolated", "--light-context",
                       "--thinking", "off", "--message", job.contenu, "--webhook", url]
```

dans `_apercu`, remplacer `canal = job.canal or _origine(origin)` par :

```python
    # Résultats de consignes et de commandes : Obsidian par défaut, avis Telegram
    # (décision du 2026-10-09) ; les rappels restent sur le canal d'origine.
    canal = job.canal or ("obsidian" if job.type in ("consigne", "commande") else _origine(origin))
```

- [ ] **Step 4: Run** `python3 -m pytest tiron_cron -q` — Expected: PASS (ajuster les tests existants qui supposaient le canal d'origine pour une consigne/commande sans canal explicite, en ajoutant `telegram` à leurs arguments).

- [ ] **Step 5: Commit** — `git commit -m "feat(cron): consignes et commandes livrées dans Obsidian par webhook, par défaut"`

---

### Task 3: point de réception `/livraison` dans wiki-lm-server

**Files:**
- Modify: `Wiki_LM/tools/server.py` (après `_tiron_cron`)
- Test: `Wiki_LM/tests/test_server.py`

**Interfaces:**
- Consumes: CLI `python3 -m tiron_cron livrer` (tâche 1).
- Produces: `POST /livraison` — `Authorization: Bearer <TIRON_LIVRAISON_TOKEN>` ; 401 si absent/faux, 503 si le serveur n'a pas de jeton ; 200 `{"text": ...}` sinon, 502 si tiron_cron échoue.

- [ ] **Step 1: Write the failing tests**

```python
class TestLivraison:
    EVT = {"jobId": "j", "status": "ok", "summary": "x"}

    def test_sans_jeton_serveur_refuse(self, client, monkeypatch):
        monkeypatch.delenv("TIRON_LIVRAISON_TOKEN", raising=False)
        assert client.post("/livraison", json=self.EVT).status_code == 503

    def test_mauvais_jeton_refuse(self, client, monkeypatch):
        monkeypatch.setenv("TIRON_LIVRAISON_TOKEN", "bon")
        r = client.post("/livraison", json=self.EVT, headers={"Authorization": "Bearer mauvais"})
        assert r.status_code == 401
        assert client.post("/livraison", json=self.EVT).status_code == 401

    def test_bon_jeton_delegue_a_tiron_cron(self, client, monkeypatch):
        import server
        monkeypatch.setenv("TIRON_LIVRAISON_TOKEN", "bon")
        appels = []
        monkeypatch.setattr(server, "_tiron_cron", lambda argv, entree=None: appels.append((argv, entree)) or {"text": "/x.md"})
        r = client.post("/livraison", json=self.EVT, headers={"Authorization": "Bearer bon"})
        assert r.status_code == 200 and r.get_json() == {"text": "/x.md"}
        assert appels[0][0] == ["livrer"] and '"summary": "x"' in appels[0][1]
```

- [ ] **Step 2: Run** `cd Wiki_LM && .venv/bin/python -m pytest tests/test_server.py -q -k Livraison` — Expected: FAIL (404).

- [ ] **Step 3: Implement** — `_tiron_cron(argv, entree=None)` passe `input=entree` à `subprocess.run` ; ajouter `import hmac` et :

```python
@app.post("/livraison")
def handle_livraison():
    """Webhook des tâches /cron (OpenClaw, cron.webhookToken) : note dans
    Réception/ et avis Telegram, par tiron_cron. Joignable par Tailscale :
    jamais sans jeton."""
    attendu = os.environ.get("TIRON_LIVRAISON_TOKEN", "")
    if not attendu:
        return jsonify({"error": "livraison non configurée (TIRON_LIVRAISON_TOKEN)"}), 503
    recu = request.headers.get("Authorization", "")
    if not hmac.compare_digest(recu.encode(), f"Bearer {attendu}".encode()):
        return jsonify({"error": "jeton refusé"}), 401
    res = _tiron_cron(["livrer"], entree=json.dumps(request.get_json(silent=True) or {}, ensure_ascii=False))
    return jsonify(res), (502 if "error" in res else 200)
```

- [ ] **Step 4: Run** `cd Wiki_LM && .venv/bin/python -m pytest tests/test_server.py -q` — Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(wiki-lm-server): /livraison, webhook des tâches /cron (jeton Bearer)"`

---

### Task 4: mise en service des livraisons (avec l'utilisateur)

**Files:** hors dépôt — `~/.config/secrets.env`, `~/Secretarius/Wiki_LM/.env`, `~/.openclaw/openclaw.json` ; dépôt : `docs/components/cron.md`, `amorçage/faits.md` (+ copie `~/Documents/Secretarius/Wiki_LM/faits/faits.md`, avec ses `\<`).

- [ ] **Step 1:** l'utilisateur ajoute dans `secrets.env` `TIRON_LIVRAISON_TOKEN=<openssl rand -hex 32>` ; ajoute dans `Wiki_LM/.env` les lignes `TIRON_LIVRAISON_TOKEN=` (valeur remplie par `/maj`) et `TIRON_TELEGRAM_CHAT_ID=<même valeur que dans gateway.systemd.env>`.
- [ ] **Step 2:** avec accord, `openclaw.json` (sauvegarde `.bak-20261009-livraison`) : `cron.webhookToken: "${TIRON_LIVRAISON_TOKEN}"` et `cron.webhookSsrfPolicy: {allowedHostnames: ["127.0.0.1"]}`.
- [ ] **Step 3:** `/maj` sur Telegram (copie le jeton au gateway et dans Wiki_LM/.env, redémarre wiki-lm-server, puis le gateway).
- [ ] **Step 4:** essai réel : `/cron add 2m consigne donne l'état du wiki` → aperçu « obsidian », `/confirm` ; vérifier la note dans `~/Documents/Arbath/Réception/` et l'avis Telegram ; relever dans le journal de wiki-lm-server le corps reçu et confirmer les champs `status`, `summary`, `job.displayName` (sinon corriger `_titre`, avec test).
- [ ] **Step 5:** essai d'échec : `/cron add 2m commande /inexistante` → note « Échec » ou résultat d'erreur, et avis.
- [ ] **Step 6:** documentation : `docs/components/cron.md` (canal obsidian, webhook, jeton, Réception) ; `faits.md` : « Où arrivent les résultats de mes tâches planifiées ? » ; commit `docs(cron): livraisons dans Réception/`.

---

### Task 5: `tiron_mail bloc` — courriel décrit par un bloc de note

**Files:**
- Modify: `tiron_mail/__main__.py`
- Test: `tiron_mail/tests/test_cli.py`

**Interfaces:**
- Produces: `python3 -m tiron_mail bloc --args=<texte du bloc> [--confirme]`. Texte : lignes d'en-tête en tête de bloc `à : <adresse>` (aussi `a :`, `à:`) et `objet : <objet>`, puis le corps. Aperçu : `{"ok": true, "text": "Courriel depuis <adresse d'envoi> à <dest>\nObjet : <objet>\n\n<corps>"}` ; avec `--confirme` : `{"ok": true, "text": "Courriel envoyé à <dest> depuis <adresse>."}`. Expéditeur : `cfg.envoi_defaut` (tiron@). Objet absent → première ligne du corps (60 caractères).

- [ ] **Step 1: Write the failing tests**

```python
BLOC = "à : marie@ex.fr\nobjet : Compte rendu\nBonjour Marie,\nobjet : ceci reste dans le corps."


def test_bloc_apercu_n_envoie_rien(capsys, env_test, faux_smtp):
    out = _call(capsys, ["bloc", "--args=" + BLOC], env_test, faux_smtp)
    assert out["ok"] and faux_smtp.envoyes == []
    assert "à marie@ex.fr" in out["text"] and "Objet : Compte rendu" in out["text"]
    assert "objet : ceci reste dans le corps." in out["text"]


def test_bloc_confirme_envoie(capsys, env_test, faux_smtp):
    out = _call(capsys, ["bloc", "--args=" + BLOC, "--confirme"], env_test, faux_smtp)
    m = faux_smtp.envoyes[0]
    assert m["To"] == "marie@ex.fr" and m["Subject"] == "Compte rendu" and m["From"] == "moi@gmail.com"
    assert m.get_content().startswith("Bonjour Marie,") and out["text"].startswith("Courriel envoyé")


def test_bloc_sans_destinataire_ou_sans_corps(capsys, env_test, faux_smtp):
    assert _call(capsys, ["bloc", "--args=objet : x\ncorps"], env_test, faux_smtp)["ok"] is False
    assert _call(capsys, ["bloc", "--args=à : m@ex.fr\nobjet : x"], env_test, faux_smtp)["ok"] is False
    assert faux_smtp.envoyes == []
```

- [ ] **Step 2: Run** `python3 -m pytest tiron_mail -q -k bloc` — Expected: FAIL (`invalid choice: 'bloc'`).

- [ ] **Step 3: Implement** — ajouter `"bloc"` aux `choices`, et la fonction + la branche :

```python
_ENTETE = re.compile(r"^\s*(à|a|objet)\s*:\s*(.*)$", re.IGNORECASE)


def _lire_bloc(texte: str) -> tuple[str, str, str]:
    """En-têtes « à : » et « objet : » en tête de bloc seulement, puis le corps."""
    dest = objet = ""
    lignes = texte.strip().splitlines()
    while lignes and (m := _ENTETE.match(lignes[0])):
        if m.group(1).lower() == "objet":
            objet = m.group(2).strip()
        else:
            dest = m.group(2).strip()
        lignes.pop(0)
    return dest, objet, "\n".join(lignes).strip()
```

```python
        elif a.commande == "bloc":
            dest, objet, corps = _lire_bloc(a.args)
            if not _ADRESSE.match(dest) or not corps:
                raise ErreurBoite("bloc mail : une ligne « à : <adresse> » et un corps sont nécessaires")
            compte = cfg.comptes[cfg.envoi_defaut]
            objet = " ".join((objet or corps.splitlines()[0][:60]).split())
            if a.confirme:
                envoyer(construire(compte, dest, objet, corps), compte, env, smtp)
                out = {"ok": True, "text": f"Courriel envoyé à {dest} depuis {compte.adresse}."}
            else:
                out = {"ok": True, "text": f"Courriel depuis {compte.adresse} à {dest}\nObjet : {objet}\n\n{corps}"}
```

- [ ] **Step 4: Run** `python3 -m pytest tiron_mail -q` — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(tiron_mail): bloc — courriel décrit par un bloc de note (à :, objet :, corps)"`

---

### Task 6: blocs `/mail?` et `/mail!` dans Obsidian

**Files:**
- Modify: `Wiki_LM/tools/server.py` (`_RUN_OPS`, `_tiron_mail`)
- Modify: `Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts`, `src/main.ts` (corps = note)
- Test: `Wiki_LM/tests/test_server.py`, `Wiki_LM/obsidian-wikilm-capture/src/run-commands.test.ts`

**Interfaces:**
- Consumes: `tiron_mail bloc` (tâche 5).
- Produces: `_RUN_OPS["/mail?"]`, `_RUN_OPS["/mail!"]` → `{"text": ...}` ou `{"error": ...}` ; côté plugin `blocArgMail(arg: string, noteText: string): string` (si le bloc n'a que des lignes d'en-tête, ajoute le texte de la note sans ses blocs ```wiki ni ses résultats).

- [ ] **Step 1: Write the failing tests**

Serveur :

```python
    def test_mail_apercu_et_envoi_delegues(self, client, monkeypatch):
        import server
        appels = []
        monkeypatch.setattr(server, "_tiron_mail", lambda argv: appels.append(argv) or {"text": "ok"})
        client.post("/run", json={"command": "/mail?", "arg": "à : m@ex.fr\nx"})
        client.post("/run", json={"command": "/mail!", "arg": "à : m@ex.fr\nx"})
        assert appels == [["bloc", "--args=à : m@ex.fr\nx"], ["bloc", "--args=à : m@ex.fr\nx", "--confirme"]]
```

Plugin (`run-commands.test.ts`) :

```ts
import { blocArgMail, formatWikiResult, SUPPORTED_COMMANDS } from "./run-commands";

it("mail? et mail! sont des commandes reconnues", () => {
  expect(SUPPORTED_COMMANDS).toContain("/mail?");
  expect(SUPPORTED_COMMANDS).toContain("/mail!");
});

it("bloc sans corps : la note, sans ses blocs, devient le corps", () => {
  const note = "Bonjour Marie,\n\n```wiki\n/mail?\nà : m@ex.fr\n```\n\n<!-- wikilm-run:start -->\nancien\n<!-- wikilm-run:end -->\nÀ jeudi.";
  expect(blocArgMail("à : m@ex.fr", note)).toBe("à : m@ex.fr\nBonjour Marie,\n\nÀ jeudi.");
});

it("bloc avec corps : inchangé", () => {
  expect(blocArgMail("à : m@ex.fr\nobjet : x\nCorps", "note")).toBe("à : m@ex.fr\nobjet : x\nCorps");
});

it("mise en forme de l'aperçu", () => {
  expect(formatWikiResult("/mail?", { text: "Courriel…" }, true))
    .toBe("Courriel…\n\nRien n'a été envoyé. Remplacez par /mail! pour envoyer.");
});
```

- [ ] **Step 2: Run** `cd Wiki_LM && .venv/bin/python -m pytest tests/test_server.py -q -k mail` et `cd Wiki_LM/obsidian-wikilm-capture && npx vitest run` — Expected: FAIL.

- [ ] **Step 3: Implement**

`server.py` :

```python
def _tiron_mail(argv: list[str]) -> dict:
    """Délègue à la CLI JSON tiron_mail (blocs mail? / mail!, spec courrier)."""
    cwd = os.environ.get("SECRETARIUS_DIR") or str(Path.home() / "Secretarius")
    p = subprocess.run(["python3", "-m", "tiron_mail", *argv], cwd=cwd,
                       capture_output=True, text=True, timeout=150)
    try:
        data = json.loads(p.stdout)
    except ValueError:
        return {"error": (p.stderr or p.stdout or "tiron_mail indisponible").strip()[:500]}
    return {"text": data.get("text", "")} if data.get("ok") else {"error": data.get("text", "")}
```

et dans `_RUN_OPS` :

```python
    # Courrier (spec 2026-10-07) : /mail? aperçu, /mail! envoi (le ! tient lieu de /confirm).
    "/mail?": lambda arg, vault: _tiron_mail(["bloc", f"--args={arg}"]),
    "/mail!": lambda arg, vault: _tiron_mail(["bloc", f"--args={arg}", "--confirme"]),
```

`tiron_mail` lit les secrets dans l'environnement : `wiki-lm-server` n'a que `Wiki_LM/.env`. Ajouter `EnvironmentFile=%h/.config/secrets.env` ? **Non** (tous les secrets dans le wiki). Ajouter dans `Wiki_LM/.env` la seule ligne `PROTON_BRIDGE_PASS_PERSO=` (remplie par `/maj`) — tâche 7, étape 1.

`run-commands.ts` : ajouter `"/mail?", "/mail!"` à `SUPPORTED_COMMANDS`, et :

```ts
const ENTETE_MAIL = /^\s*(à|a|objet)\s*:/i;

export function blocArgMail(arg: string, noteText: string): string {
  const corps = arg.split("\n").filter((l) => l.trim() && !ENTETE_MAIL.test(l));
  if (corps.length > 0) return arg;
  const note = noteText.replace(new RegExp(BLOCK_RE.source, "g"), "").replace(/\n{3,}/g, "\n\n").trim();
  return `${arg}\n${note}`;
}
```

cas dans `formatWikiResult` :

```ts
    case "/mail?":
      return `${String(data.text ?? "")}\n\nRien n'a été envoyé. Remplacez par /mail! pour envoyer.`;
    case "/mail!":
      return String(data.text ?? "");
```

`main.ts`, dans l'appel `/run` : `arg: block.command.startsWith("/mail") ? blocArgMail(block.arg, noteText) : block.arg` (où `noteText` est le contenu de la note déjà lu pour `extractWikiBlocks`).

- [ ] **Step 4: Run** les deux suites — Expected: PASS ; `npm run build` dans le plugin.
- [ ] **Step 5: Commit** — `git commit -m "feat(obsidian): blocs /mail? et /mail! (aperçu puis envoi depuis tiron@)"`

---

### Task 7: mise en service des blocs mail (avec l'utilisateur)

- [ ] **Step 1:** l'utilisateur ajoute `PROTON_BRIDGE_PASS_PERSO=` dans `Wiki_LM/.env` ; `/maj` remplit la valeur et redémarre wiki-lm-server.
- [ ] **Step 2:** avec accord, copier `main.js` et `manifest.json` du plugin dans `~/Documents/Arbath/.obsidian/plugins/wikilm-capture/` et `~/Documents/Secretarius/.obsidian/plugins/wikilm-capture/` ; l'utilisateur recharge le plugin dans Obsidian.
- [ ] **Step 3:** essai réel dans une note : bloc `/mail?` (aperçu sous le bloc), puis `/mail!` vers christian@mauceri.me (message reçu, expéditeur tiron@).
- [ ] **Step 4:** `faits.md` (« Comment envoyer un courriel depuis une note ? ») et `docs/components/obsidian.md` ; commit `docs(obsidian): blocs mail? et mail!`.
- [ ] **Step 5:** registre `.superpowers/sdd/2026-10-09-courrier-etape-3/progress.md` et fiche mémoire.
