# Planification de tâches par Tiron (`/cron`) — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** permettre à l'utilisateur de planifier rappels, commandes Tiron et consignes, livrés sur Telegram, WhatsApp, SMS ou Obsidian, via Tiron (commande `/cron` ou langage naturel) et via Obsidian (`/cron?`, `/cron!`, `/cron`).

**Architecture:** un module Python stdlib unique, `tiron_cron`, analyse la syntaxe normalisée (et le français courant, avec repli qwen3:8b local), reformule, puis pilote le planificateur natif d'OpenClaw (`openclaw cron`). derisk-deleg (canaux de messagerie) et wiki-lm-server (Obsidian) l'appellent par sa CLI JSON. Laya apprend seulement à reconnaître l'intention `/cron`.

**Tech Stack:** Python 3.12 stdlib (`zoneinfo`, `urllib`, `argparse`), pytest ; TypeScript + vitest (derisk-deleg, plugin Obsidian) ; Flask (wiki-lm-server) ; OpenClaw 2026.8.2 ; Ollama (qwen3:8b) ; Laya (classifieur) + dspy/GEPA (génération de corpus).

**Spec:** `docs/superpowers/specs/2026-10-01-cron-tiron-design.md` (révisée le 2026-10-02 : Laya classe, il ne génère pas).

## Global Constraints

- Syntaxe : `add <quand> [<canal>] <type> <contenu>`, `list`, `rm <id>`, `pause <id>`, `reprendre <id>`.
- `<quand>` : cron 5 champs entre guillemets (`"0 8 * * 1"`), ISO local `AAAA-MM-JJTHH:MM`, ou durée `\d+[mhd]`.
- Canaux : `telegram`, `whatsapp`, `sms`, `obsidian`. Canal omis → canal d'origine.
- Types : `rappel` (sans modèle), `commande` (liste blanche seulement), `consigne` (agent `main`, son modèle habituel).
- Fuseau par défaut : `Europe/Paris` (surchargeable par `TIRON_TZ`).
- Aucun shell fourni par l'utilisateur ; `--command-argv` réservé aux scripts internes.
- Toute création passe par une confirmation : `/confirm` (messagerie) ou `/cron!` (Obsidian).
- Tâches étiquetées `--declaration-key tiron:<8 hex>` ; `list`/`rm`/`pause`/`reprendre` ne voient que `tiron:*`.
- Destinataires : `TIRON_TELEGRAM_CHAT_ID`, `TIRON_WHATSAPP_TO`, `TIRON_SMS_TO` (E.164) ; rappels Obsidian dans `TIRON_RAPPELS_DIR` (défaut `~/Documents/Arbath/Rappels`).
- Repli langage naturel : Ollama local `qwen3:8b`, `think: false` ; jamais un fournisseur distant.
- Obsidian : syntaxe normalisée seulement (pas de langage naturel en V1).
- Hors V1 : canal `mail`, LN depuis Obsidian, notification Obsidian active, canal de secours, rattrapage.
- Confirmation de l'utilisateur requise avant tout `systemctl --user restart`, et avant toute dépense d'API (DeepSeek pour GEPA/génération) ou tout réentraînement Kaggle.

## Review Focus

- **Apostrophes et guillemets dans le contenu** (`rappel d'appeler l'école`) : le contenu est pris tel quel, jamais découpé par `shlex` → test dans la tâche 2.
- **Date passée en langage naturel** (« aujourd'hui à 8h » à 10h) : refusée par l'analyseur, qui passe la main au repli puis à l'erreur, jamais une tâche dans le passé → test dans la tâche 5.
- **Canal `obsidian` pour une consigne ou une commande** : refus explicite (« rappels seulement en V1 »), pas une tâche qui ne livre rien → test dans la tâche 4.
- **`rm`/`pause` d'un identifiant de tâche système** (`skill-collection-review`) : introuvable, car la résolution ne cherche que parmi `tiron:*` → test dans la tâche 4.
- **Destinataire non configuré** (`TIRON_SMS_TO` absent) : erreur dès l'aperçu, avant `/confirm`, pas un échec silencieux au déclenchement → test dans la tâche 4.

---

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `tiron_cron/__init__.py` | paquet (vide) |
| `tiron_cron/syntax.py` | `Job`, `Gestion`, `SyntaxeInvalide`, `parse_command`, `to_args`, constantes |
| `tiron_cron/describe.py` | reformulation française d'un `Job` |
| `tiron_cron/openclaw_cli.py` | construction des argv `openclaw cron …`, exécution, filtrage `tiron:*` |
| `tiron_cron/nl.py` | analyseur déterministe du français → `Job` |
| `tiron_cron/fallback.py` | repli qwen3:8b (Ollama) → `Job` |
| `tiron_cron/deliver_obsidian.py` | écrit la note de rappel |
| `tiron_cron/__main__.py` | CLI JSON : `preview`, `add`, `nl`, `manage`, `deliver-obsidian` |
| `tiron_cron/tests/` | tests pytest |
| `router_service/router.py`, `server.py` | `/cron` explicite ; `COMMAND_CRITERIA` (tâche 13) |
| `derisk-deleg/src/dispatch.ts` | `/cron` → `{kind:"cron"}` |
| `derisk-deleg/src/cron-ops.ts` | appel asynchrone de la CLI `tiron_cron` |
| `derisk-deleg/src/index.ts` | aperçu + `pending` cron + `/confirm`/`/annuler` |
| `Wiki_LM/tools/server.py` | `/cron?`, `/cron!`, `/cron` dans `_RUN_OPS` |
| `Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts` | commandes + formatage |
| `gen_corpus/*` | intention `cron_schedule`, valset GEPA séparé, prompt revu |

Tests Python : `cd ~/Secretarius && python3 -m pytest tiron_cron -v`.

---

### Task 1: Essai préalable (jetable) — comportement réel d'OpenClaw

Rien de ce qui est créé ici n'est conservé. But : lever trois inconnues dont dépendent les tâches 4 et 10.

**Files:** aucun dans le dépôt. Consigner les résultats à la fin de ce plan, section « Résultats de l'essai préalable ».

- [ ] **Step 1: Une commande planifiée livre-t-elle sa sortie par `--announce` ?**

Obtenir l'identifiant de discussion Telegram du propriétaire : `openclaw channels resolve --help`, puis la commande adaptée ; à défaut, le lire dans `~/.openclaw/openclaw.json` (bindings Telegram).

```bash
openclaw cron add --name essai-rappel --at 1m --delete-after-run \
  --command-argv '["printf","%s\n","essai rappel tiron"]' \
  --announce --channel telegram --to <CHAT_ID>
```
Attendu : « essai rappel tiron » arrive sur Telegram dans la minute. Vérifier aussi `openclaw cron runs essai-rappel` (ou par id).

- [ ] **Step 2: Un message planifié vers l'agent `main` passe-t-il par le hook `before_agent_reply` de derisk-deleg ?**

```bash
openclaw cron add --name essai-commande --at 1m --delete-after-run \
  --agent main --session isolated --message "/wikistatus" \
  --announce --channel telegram --to <CHAT_ID>
```
Attendu (oui) : la réponse déterministe de `/wikistatus` (état de l'ingestion) arrive sur Telegram. Si c'est une réponse libre du modèle de `main`, le hook n'est pas passé (non).

- [ ] **Step 3: Extensions SMS et WhatsApp**

```bash
openclaw channels list --all
openclaw channels capabilities --help
```
Lire la documentation de l'extension SMS (`openclaw plugins` / docs.openclaw.ai) : fournisseur Twilio pris en charge ? identifiant de canal exact (`sms` ?). Idem WhatsApp : mode de liaison (QR, API Business ?), identifiant de canal (`whatsapp` ?).

- [ ] **Step 4: Nettoyer et décider**

```bash
openclaw cron list
```
Supprimer toute tâche `essai-*` restante (`openclaw cron rm <id>`).

Règles de décision :
- Step 1 non → **arrêter** et revenir vers l'utilisateur (le type `rappel` doit alors passer par `--message` à un agent, ce qui change la spec).
- Step 2 non → **arrêter** et revenir vers l'utilisateur (le type `commande` doit être conçu autrement).
- Step 3 : identifiants de canal différents de `telegram`/`whatsapp`/`sms` → les reporter dans `CANAUX_OPENCLAW` (tâche 4). SMS sans Twilio → le noter ; l'envoi par script Twilio devient une tâche supplémentaire à planifier avec l'utilisateur.

---

### Task 2: `tiron_cron/syntax.py` — syntaxe normalisée

**Files:**
- Create: `tiron_cron/__init__.py` (vide), `tiron_cron/tests/__init__.py` (vide)
- Create: `tiron_cron/syntax.py`
- Test: `tiron_cron/tests/test_syntax.py`

**Interfaces:**
- Produces: `CANAUX: tuple[str, ...]`, `TYPES: tuple[str, ...]`, `COMMANDES_PLANIFIABLES: frozenset[str]`, `class SyntaxeInvalide(ValueError)`, `@dataclass(frozen=True) Job(quand_kind: str, quand: str, canal: str | None, type: str, contenu: str)` (`quand_kind` ∈ `"cron"`, `"at"`), `@dataclass(frozen=True) Gestion(action: str, id: str | None)`, `parse_command(args: str) -> Job | Gestion`, `to_args(job: Job) -> str`.

- [ ] **Step 1: Écrire les tests**

```python
# tiron_cron/tests/test_syntax.py
import pytest

from tiron_cron.syntax import Gestion, Job, SyntaxeInvalide, parse_command, to_args


def test_add_cron_avec_canal():
    assert parse_command('add "0 8 * * 1" sms rappel payer le loyer') == Job(
        "cron", "0 8 * * 1", "sms", "rappel", "payer le loyer")


def test_add_iso_sans_canal():
    assert parse_command("add 2026-10-02T09:00 rappel appeler le garagiste") == Job(
        "at", "2026-10-02T09:00", None, "rappel", "appeler le garagiste")


def test_add_duree():
    assert parse_command("add 20m telegram rappel sortir le gâteau").quand == "20m"


def test_contenu_avec_apostrophes_et_guillemets_conserve_tel_quel():
    job = parse_command("add 1h rappel d'appeler l'école \"avant midi\"")
    assert job.contenu == "d'appeler l'école \"avant midi\""


def test_commande_liste_blanche():
    assert parse_command('add "0 8 * * 1" commande /wikistatus').contenu == "/wikistatus"


def test_commande_hors_liste_blanche_refusee():
    with pytest.raises(SyntaxeInvalide, match="non planifiable"):
        parse_command('add "0 8 * * 1" commande /supprimer src-x')


def test_consigne():
    assert parse_command("add 2h consigne résume le wiki").type == "consigne"


@pytest.mark.parametrize("args", [
    'add "0 8 * *" rappel x',          # 4 champs
    "add 2026-13-02T09:00 rappel x",   # mois 13
    "add 0m rappel x",                 # durée nulle
    "add demain rappel x",             # forme libre
])
def test_quand_invalide(args):
    with pytest.raises(SyntaxeInvalide, match="quand"):
        parse_command(args)


def test_type_inconnu():
    with pytest.raises(SyntaxeInvalide, match="type inconnu"):
        parse_command("add 20m alerte x")


def test_contenu_vide():
    with pytest.raises(SyntaxeInvalide, match="contenu vide"):
        parse_command("add 20m rappel   ")


def test_guillemet_non_ferme():
    with pytest.raises(SyntaxeInvalide, match="guillemet"):
        parse_command('add "0 8 * * 1 rappel x')


def test_gestion():
    assert parse_command("list") == Gestion("list", None)
    assert parse_command("rm 1a2b3c4d") == Gestion("rm", "1a2b3c4d")
    assert parse_command("PAUSE 1a2b") == Gestion("pause", "1a2b")
    assert parse_command("reprendre 1a2b") == Gestion("reprendre", "1a2b")


def test_gestion_sans_id():
    with pytest.raises(SyntaxeInvalide, match="usage"):
        parse_command("rm")


def test_action_inconnue():
    with pytest.raises(SyntaxeInvalide, match="action inconnue"):
        parse_command("modifier 1a2b")


def test_to_args_aller_retour():
    for args in ['add "0 8 * * 1" sms rappel payer le loyer',
                 "add 2026-10-02T09:00 consigne résume le wiki",
                 "add 20m rappel d'appeler l'école"]:
        assert to_args(parse_command(args)) == args
```

- [ ] **Step 2: Vérifier l'échec**

Run: `cd ~/Secretarius && python3 -m pytest tiron_cron/tests/test_syntax.py -v`
Expected: FAIL (`ModuleNotFoundError: tiron_cron.syntax`).

- [ ] **Step 3: Implémenter**

```python
# tiron_cron/syntax.py
"""Syntaxe normalisée de /cron (cf. docs/superpowers/specs/2026-10-01-cron-tiron-design.md).
Le contenu n'est jamais découpé par shlex : il est pris tel quel (apostrophes)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

CANAUX = ("telegram", "whatsapp", "sms", "obsidian")
TYPES = ("rappel", "commande", "consigne")
# Lecture ou écriture sans risque de perte ; /inbox s'ajoutera avec l'agent courrier.
COMMANDES_PLANIFIABLES = frozenset({"/wikistatus", "/tags", "/relire", "/q", "/r", "/ingest", "/kbupdate"})

_CRON_FIELD = re.compile(r"^[0-9*/,\-]+$")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
_DUREE = re.compile(r"^(\d+)[mhd]$")
USAGE_ADD = "usage : /cron add <quand> [<canal>] <type> <contenu>"


class SyntaxeInvalide(ValueError):
    pass


@dataclass(frozen=True)
class Job:
    quand_kind: str  # "cron" | "at"
    quand: str
    canal: str | None
    type: str
    contenu: str


@dataclass(frozen=True)
class Gestion:
    action: str  # "list" | "rm" | "pause" | "reprendre"
    id: str | None


def _split_first(s: str) -> tuple[str, str]:
    s = s.lstrip()
    if not s:
        return "", ""
    if s[0] in "\"'":
        end = s.find(s[0], 1)
        if end < 0:
            raise SyntaxeInvalide("guillemet non fermé")
        return s[1:end], s[end + 1:]
    parts = s.split(None, 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def parse_quand(q: str) -> tuple[str, str]:
    fields = q.split()
    if len(fields) == 5 and all(_CRON_FIELD.match(f) for f in fields):
        return "cron", " ".join(fields)
    if _ISO.match(q):
        try:
            datetime.fromisoformat(q)
        except ValueError:
            pass
        else:
            return "at", q
    m = _DUREE.match(q)
    if m and int(m.group(1)) > 0:
        return "at", q
    raise SyntaxeInvalide(
        f"<quand> invalide : {q!r} (attendu : \"0 8 * * 1\", 2026-10-02T09:00 ou 20m)")


def parse_command(args: str) -> Job | Gestion:
    action, rest = _split_first(args)
    action = action.lower()
    if action == "list":
        return Gestion("list", None)
    if action in ("rm", "pause", "reprendre"):
        id_, _ = _split_first(rest)
        if not id_:
            raise SyntaxeInvalide(f"usage : /cron {action} <id>")
        return Gestion(action, id_)
    if action != "add":
        raise SyntaxeInvalide(f"action inconnue : {action!r} (add, list, rm, pause, reprendre)")
    quand_raw, rest = _split_first(rest)
    if not quand_raw:
        raise SyntaxeInvalide(USAGE_ADD)
    kind, quand = parse_quand(quand_raw)
    tok, rest = _split_first(rest)
    canal = None
    if tok.lower() in CANAUX:
        canal = tok.lower()
        tok, rest = _split_first(rest)
    type_ = tok.lower()
    if type_ not in TYPES:
        raise SyntaxeInvalide(f"type inconnu : {tok!r} (rappel, commande, consigne)")
    contenu = rest.strip()
    if not contenu:
        raise SyntaxeInvalide("contenu vide")
    if type_ == "commande" and contenu.split()[0] not in COMMANDES_PLANIFIABLES:
        raise SyntaxeInvalide(
            f"commande non planifiable : {contenu.split()[0]} "
            f"(autorisées : {', '.join(sorted(COMMANDES_PLANIFIABLES))})")
    return Job(kind, quand, canal, type_, contenu)


def to_args(job: Job) -> str:
    quand = f'"{job.quand}"' if job.quand_kind == "cron" else job.quand
    parts = ["add", quand] + ([job.canal] if job.canal else []) + [job.type, job.contenu]
    return " ".join(parts)
```

Note : `_split_first` sur `"rappel d'appeler l'école…"` lit `rappel` comme premier jeton (pas de guillemet en tête), puis le reste brut — d'où l'aller-retour exact.

- [ ] **Step 4: Vérifier le succès**

Run: `cd ~/Secretarius && python3 -m pytest tiron_cron/tests/test_syntax.py -v`
Expected: PASS (tous).

- [ ] **Step 5: Commit**

```bash
git add tiron_cron/__init__.py tiron_cron/tests/__init__.py tiron_cron/syntax.py tiron_cron/tests/test_syntax.py
git commit -m "feat(cron): syntaxe normalisée /cron (tiron_cron.syntax)"
```

---

### Task 3: `tiron_cron/describe.py` — reformulation

**Files:**
- Create: `tiron_cron/describe.py`
- Test: `tiron_cron/tests/test_describe.py`

**Interfaces:**
- Consumes: `Job` (tâche 2).
- Produces: `describe(job: Job, canal: str, tz: str) -> str`.

- [ ] **Step 1: Écrire les tests**

```python
# tiron_cron/tests/test_describe.py
from tiron_cron.describe import describe
from tiron_cron.syntax import Job


def test_hebdomadaire():
    job = Job("cron", "0 8 * * 1", "sms", "rappel", "payer le loyer")
    assert describe(job, "sms", "Europe/Paris") == \
        "Chaque lundi à 08:00 (Europe/Paris), sms, rappel : payer le loyer"


def test_quotidien():
    job = Job("cron", "30 7 * * *", None, "consigne", "résume le wiki")
    assert describe(job, "telegram", "Europe/Paris").startswith("Chaque jour à 07:30 (Europe/Paris), telegram")


def test_dimanche_0_et_7():
    for dow in ("0", "7"):
        assert describe(Job("cron", f"0 9 * * {dow}", None, "rappel", "x"), "telegram", "UTC") \
            .startswith("Chaque dimanche à 09:00")


def test_cron_non_humanisable():
    job = Job("cron", "*/15 * * * *", None, "rappel", "x")
    assert describe(job, "telegram", "Europe/Paris").startswith("Selon « */15 * * * * »")


def test_iso():
    job = Job("at", "2026-10-02T09:00", None, "rappel", "x")
    assert describe(job, "telegram", "Europe/Paris").startswith("Le 02/10/2026 à 09:00 (Europe/Paris)")


def test_duree():
    assert describe(Job("at", "20m", None, "rappel", "x"), "sms", "UTC").startswith("Dans 20 min (UTC)")
    assert describe(Job("at", "2h", None, "rappel", "x"), "sms", "UTC").startswith("Dans 2 h")
    assert describe(Job("at", "1d", None, "rappel", "x"), "sms", "UTC").startswith("Dans 1 j")
```

- [ ] **Step 2: Vérifier l'échec**

Run: `python3 -m pytest tiron_cron/tests/test_describe.py -v` — Expected: FAIL (module absent).

- [ ] **Step 3: Implémenter**

```python
# tiron_cron/describe.py
"""Reformulation française d'une tâche, montrée avant /confirm."""
from __future__ import annotations

from datetime import datetime

from tiron_cron.syntax import Job

_JOURS = {"0": "dimanche", "1": "lundi", "2": "mardi", "3": "mercredi",
          "4": "jeudi", "5": "vendredi", "6": "samedi", "7": "dimanche"}
_UNITES = {"m": "min", "h": "h", "d": "j"}


def _quand(job: Job) -> str:
    if job.quand_kind == "cron":
        m, h, dom, mon, dow = job.quand.split()
        if m.isdigit() and h.isdigit() and dom == "*" and mon == "*":
            heure = f"{int(h):02d}:{int(m):02d}"
            if dow == "*":
                return f"chaque jour à {heure}"
            if dow in _JOURS:
                return f"chaque {_JOURS[dow]} à {heure}"
        return f"selon « {job.quand} »"
    if "T" in job.quand:
        return datetime.fromisoformat(job.quand).strftime("le %d/%m/%Y à %H:%M")
    return f"dans {job.quand[:-1]} {_UNITES[job.quand[-1]]}"


def describe(job: Job, canal: str, tz: str) -> str:
    q = _quand(job)
    return f"{q[0].upper()}{q[1:]} ({tz}), {canal}, {job.type} : {job.contenu}"
```

- [ ] **Step 4: Vérifier le succès** — `python3 -m pytest tiron_cron/tests/test_describe.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tiron_cron/describe.py tiron_cron/tests/test_describe.py
git commit -m "feat(cron): reformulation française des tâches"
```

---

### Task 4: `tiron_cron/openclaw_cli.py` — pilotage d'`openclaw cron`

**Files:**
- Create: `tiron_cron/openclaw_cli.py`
- Test: `tiron_cron/tests/test_openclaw_cli.py`

**Interfaces:**
- Consumes: `Job`, `Gestion`, `SyntaxeInvalide` (tâche 2), `describe` (tâche 3).
- Produces:
  - `DECL_PREFIX = "tiron:"`, `CANAUX_OPENCLAW: dict[str, str]` (canal Tiron → id de canal OpenClaw ; ajuster selon la tâche 1, step 3).
  - `class OpenClawError(RuntimeError)`.
  - `build_add_argv(job: Job, canal: str | None, tz: str, env: Mapping[str, str], job_key: str) -> list[str]` (lève `SyntaxeInvalide`).
  - `add_job(argv: list[str], job_key: str, runner) -> str` → id OpenClaw complet.
  - `list_jobs(runner) -> list[dict]` (seulement `tiron:*`).
  - `resolve_id(prefix: str, jobs: list[dict]) -> str` (lève `SyntaxeInvalide`).
  - `manage(g: Gestion, runner) -> str` (texte pour l'utilisateur).
  - `runner` a la signature de `subprocess.run`.

- [ ] **Step 1: Écrire les tests**

```python
# tiron_cron/tests/test_openclaw_cli.py
import json
import subprocess

import pytest

from tiron_cron.openclaw_cli import (
    OpenClawError, add_job, build_add_argv, list_jobs, manage, resolve_id)
from tiron_cron.syntax import Gestion, Job, SyntaxeInvalide

ENV = {"TIRON_TELEGRAM_CHAT_ID": "123", "TIRON_SMS_TO": "+33600000000",
       "TIRON_WHATSAPP_TO": "+33600000001", "SECRETARIUS_DIR": "/srv/sec"}


def _opt(argv, name):
    return argv[argv.index(name) + 1]


def test_rappel_cron_sms():
    job = Job("cron", "0 8 * * 1", "sms", "rappel", "payer le loyer")
    argv = build_add_argv(job, "sms", "Europe/Paris", ENV, "tiron:abcd1234")
    assert argv[:3] == ["openclaw", "cron", "add"]
    assert _opt(argv, "--cron") == "0 8 * * 1"
    assert _opt(argv, "--tz") == "Europe/Paris"
    assert _opt(argv, "--declaration-key") == "tiron:abcd1234"
    assert json.loads(_opt(argv, "--command-argv")) == ["printf", "%s\n", "payer le loyer"]
    assert "--announce" in argv and _opt(argv, "--channel") == "sms" and _opt(argv, "--to") == "+33600000000"
    assert "--delete-after-run" not in argv


def test_tache_unique_supprimee_apres_succes():
    argv = build_add_argv(Job("at", "20m", None, "rappel", "x"), "telegram", "UTC", ENV, "tiron:1")
    assert _opt(argv, "--at") == "20m" and "--delete-after-run" in argv


def test_consigne_va_a_main_en_session_isolee():
    argv = build_add_argv(Job("cron", "0 7 * * *", None, "consigne", "résume le wiki"),
                          "telegram", "UTC", ENV, "tiron:1")
    assert _opt(argv, "--agent") == "main" and _opt(argv, "--session") == "isolated"
    assert _opt(argv, "--message") == "résume le wiki" and "--command-argv" not in argv


def test_commande_va_a_main():
    argv = build_add_argv(Job("cron", "0 8 * * 1", None, "commande", "/wikistatus"),
                          "telegram", "UTC", ENV, "tiron:1")
    assert _opt(argv, "--message") == "/wikistatus"


def test_rappel_obsidian_lance_le_script_interne_sans_livraison():
    argv = build_add_argv(Job("at", "1h", "obsidian", "rappel", "payer"), "obsidian", "UTC", ENV, "tiron:1")
    assert json.loads(_opt(argv, "--command-argv")) == [
        "python3", "-m", "tiron_cron", "deliver-obsidian", "--text", "payer"]
    assert _opt(argv, "--command-cwd") == "/srv/sec" and "--no-deliver" in argv
    assert "--announce" not in argv


def test_obsidian_refuse_consigne_et_commande():
    for type_, contenu in (("consigne", "résume"), ("commande", "/tags")):
        with pytest.raises(SyntaxeInvalide, match="rappels seulement"):
            build_add_argv(Job("at", "1h", "obsidian", type_, contenu), "obsidian", "UTC", ENV, "tiron:1")


def test_destinataire_non_configure():
    with pytest.raises(SyntaxeInvalide, match="TIRON_SMS_TO"):
        build_add_argv(Job("at", "1h", "sms", "rappel", "x"), "sms", "UTC", {}, "tiron:1")


def test_canal_inconnu():
    with pytest.raises(SyntaxeInvalide, match="canal"):
        build_add_argv(Job("at", "1h", None, "rappel", "x"), None, "UTC", ENV, "tiron:1")


def _runner(outputs):
    """Faux subprocess.run : renvoie successivement les stdout donnés, enregistre les argv."""
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        out = outputs.pop(0)
        return subprocess.CompletedProcess(argv, 0 if not isinstance(out, Exception) else 1,
                                           stdout=out if isinstance(out, str) else "",
                                           stderr=str(out) if isinstance(out, Exception) else "")
    run.calls = calls
    return run


JOBS = {"jobs": [
    {"id": "aaaa1111-x", "declarationKey": "tiron:k1", "displayName": "rappel : loyer",
     "enabled": True, "lastRunStatus": "ok"},
    {"id": "aaaa2222-y", "declarationKey": "tiron:k2", "displayName": "consigne : wiki",
     "enabled": False, "lastRunStatus": None},
    {"id": "208f659e-z", "declarationKey": "skill-collection-review:main", "displayName": "système",
     "enabled": True, "lastRunStatus": "ok"},
]}


def test_list_ne_garde_que_tiron():
    jobs = list_jobs(_runner([json.dumps(JOBS)]))
    assert [j["id"] for j in jobs] == ["aaaa1111-x", "aaaa2222-y"]


def test_resolve_id_prefixe_unique_ambigu_systeme():
    jobs = list_jobs(_runner([json.dumps(JOBS)]))
    assert resolve_id("aaaa1", jobs) == "aaaa1111-x"
    with pytest.raises(SyntaxeInvalide, match="ambigu"):
        resolve_id("aaaa", jobs)
    with pytest.raises(SyntaxeInvalide, match="introuvable"):
        resolve_id("208f659e", jobs)  # tâche système : jamais atteignable


def test_add_job_retrouve_l_id_par_declaration_key():
    run = _runner(["{}", json.dumps(JOBS)])
    assert add_job(["openclaw", "cron", "add"], "tiron:k2", run) == "aaaa2222-y"


def test_add_job_erreur_openclaw():
    with pytest.raises(OpenClawError, match="boom"):
        add_job(["openclaw", "cron", "add"], "tiron:k9", _runner([RuntimeError("boom")]))


def test_manage_list_rm_pause_reprendre():
    assert "aaaa1111" in manage(Gestion("list", None), _runner([json.dumps(JOBS)]))
    run = _runner([json.dumps(JOBS), ""])
    assert "supprimée" in manage(Gestion("rm", "aaaa1"), run)
    assert run.calls[-1] == ["openclaw", "cron", "rm", "aaaa1111-x"]
    run = _runner([json.dumps(JOBS), ""])
    manage(Gestion("pause", "aaaa1"), run)
    assert run.calls[-1] == ["openclaw", "cron", "disable", "aaaa1111-x"]
    run = _runner([json.dumps(JOBS), ""])
    manage(Gestion("reprendre", "aaaa2"), run)
    assert run.calls[-1] == ["openclaw", "cron", "enable", "aaaa2222-y"]


def test_manage_list_vide():
    assert manage(Gestion("list", None), _runner([json.dumps({"jobs": []})])) == "Aucune tâche planifiée."
```

- [ ] **Step 2: Vérifier l'échec** — `python3 -m pytest tiron_cron/tests/test_openclaw_cli.py -v` → FAIL (module absent).

- [ ] **Step 3: Implémenter**

```python
# tiron_cron/openclaw_cli.py
"""Traduction d'un Job en appel `openclaw cron`, et gestion des tâches tiron:*.
Les tâches système (ex. skill-collection-review) ne sont jamais visibles ni
modifiables d'ici : toute résolution d'id se fait parmi les tâches tiron:*."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from tiron_cron.syntax import Gestion, Job, SyntaxeInvalide

DECL_PREFIX = "tiron:"
# Canal Tiron -> identifiant de canal OpenClaw (vérifié par l'essai préalable).
CANAUX_OPENCLAW = {"telegram": "telegram", "whatsapp": "whatsapp", "sms": "sms"}
DESTINATAIRES_ENV = {"telegram": "TIRON_TELEGRAM_CHAT_ID", "whatsapp": "TIRON_WHATSAPP_TO",
                     "sms": "TIRON_SMS_TO"}


class OpenClawError(RuntimeError):
    pass


def secretarius_dir(env: Mapping[str, str]) -> str:
    return env.get("SECRETARIUS_DIR") or str(Path.home() / "Secretarius")


def build_add_argv(job: Job, canal: str | None, tz: str, env: Mapping[str, str],
                   job_key: str) -> list[str]:
    if canal is None:
        raise SyntaxeInvalide("canal non précisé et canal d'origine inconnu (telegram, whatsapp, sms, obsidian)")
    argv = ["openclaw", "cron", "add", "--json", "--name", job_key.replace(":", "-"),
            "--display-name", f"{job.type} : {job.contenu}"[:60], "--declaration-key", job_key]
    if job.quand_kind == "cron":
        argv += ["--cron", job.quand, "--tz", tz, "--exact"]
    else:
        argv += ["--at", job.quand, "--tz", tz, "--delete-after-run"]
    if canal == "obsidian":
        if job.type != "rappel":
            raise SyntaxeInvalide("canal obsidian : rappels seulement en V1")
        return argv + ["--command-argv", json.dumps(
            ["python3", "-m", "tiron_cron", "deliver-obsidian", "--text", job.contenu]),
            "--command-cwd", secretarius_dir(env), "--no-deliver"]
    var = DESTINATAIRES_ENV[canal]
    dest = env.get(var)
    if not dest:
        raise SyntaxeInvalide(f"destinataire {canal} non configuré ({var})")
    delivery = ["--announce", "--channel", CANAUX_OPENCLAW[canal], "--to", dest]
    if job.type == "rappel":
        return argv + ["--command-argv", json.dumps(["printf", "%s\n", job.contenu])] + delivery
    return argv + ["--agent", "main", "--session", "isolated", "--message", job.contenu] + delivery


def _run(argv: list[str], runner) -> str:
    p = runner(argv, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise OpenClawError((p.stderr or p.stdout or "échec openclaw").strip())
    return p.stdout


def list_jobs(runner) -> list[dict]:
    data = json.loads(_run(["openclaw", "cron", "list", "--json"], runner) or "{}")
    return [j for j in data.get("jobs", []) if str(j.get("declarationKey") or "").startswith(DECL_PREFIX)]


def add_job(argv: list[str], job_key: str, runner) -> str:
    _run(argv, runner)
    for j in list_jobs(runner):
        if j.get("declarationKey") == job_key:
            return j["id"]
    raise OpenClawError(f"tâche créée mais introuvable ({job_key})")


def resolve_id(prefix: str, jobs: list[dict]) -> str:
    matches = [j["id"] for j in jobs if j["id"].startswith(prefix)]
    if not matches:
        raise SyntaxeInvalide(f"tâche introuvable : {prefix}")
    if len(matches) > 1:
        raise SyntaxeInvalide(f"identifiant ambigu : {prefix} ({len(matches)} tâches)")
    return matches[0]


def _ligne(j: dict) -> str:
    etat = "actif" if j.get("enabled") else "en pause"
    return f"{j['id'][:8]}  {etat}  {j.get('displayName', '')}  (dernier : {j.get('lastRunStatus') or '—'})"


def manage(g: Gestion, runner) -> str:
    jobs = list_jobs(runner)
    if g.action == "list":
        return "\n".join(_ligne(j) for j in jobs) if jobs else "Aucune tâche planifiée."
    full = resolve_id(g.id or "", jobs)
    verbe = {"rm": "rm", "pause": "disable", "reprendre": "enable"}[g.action]
    _run(["openclaw", "cron", verbe, full], runner)
    return {"rm": "Tâche {} supprimée.", "pause": "Tâche {} en pause.",
            "reprendre": "Tâche {} reprise."}[g.action].format(full[:8])
```

- [ ] **Step 4: Vérifier le succès** — `python3 -m pytest tiron_cron/tests/test_openclaw_cli.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tiron_cron/openclaw_cli.py tiron_cron/tests/test_openclaw_cli.py
git commit -m "feat(cron): pilotage d'openclaw cron, tâches tiron:* seulement"
```

---

### Task 5: `tiron_cron/nl.py` — analyseur déterministe du français

**Files:**
- Create: `tiron_cron/nl.py`
- Test: `tiron_cron/tests/test_nl.py`

**Interfaces:**
- Consumes: `Job`, `COMMANDES_PLANIFIABLES` (tâche 2).
- Produces: `parse_nl(text: str, now: datetime) -> Job | None` (`now` avec fuseau ; `None` = non compris, passer au repli).

- [ ] **Step 1: Écrire les tests**

```python
# tiron_cron/tests/test_nl.py
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from tiron_cron.nl import parse_nl
from tiron_cron.syntax import Job

NOW = datetime(2026, 10, 2, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))  # vendredi


@pytest.mark.parametrize("text,expected", [
    ("rappelle-moi chaque lundi à 8h de payer le loyer par SMS",
     Job("cron", "0 8 * * 1", "sms", "rappel", "payer le loyer")),
    ("Tiron, tous les jours à 7h30 résume les nouveautés du wiki",
     Job("cron", "30 7 * * *", None, "consigne", "résume les nouveautés du wiki")),
    ("rappelle-moi demain à 9h d'appeler le garagiste",
     Job("at", "2026-10-03T09:00", None, "rappel", "appeler le garagiste")),
    ("dans 20 minutes rappelle-moi de sortir le gâteau",
     Job("at", "20m", None, "rappel", "sortir le gâteau")),
    ("le 15 octobre à 14h rappelle-moi la réunion budget sur WhatsApp",
     Job("at", "2026-10-15T14:00", "whatsapp", "rappel", "la réunion budget")),
    ("le 1 mars à 9h rappelle-moi de renouveler l'assurance",
     Job("at", "2027-03-01T09:00", None, "rappel", "renouveler l'assurance")),
    ("tous les dimanches à 18h /wikistatus",
     Job("cron", "0 18 * * 0", None, "commande", "/wikistatus")),
    ("dans 2 heures envoie-moi par texto : vérifier le four",
     Job("at", "2h", "sms", "consigne", "envoie-moi : vérifier le four")),
])
def test_formes_reconnues(text, expected):
    assert parse_nl(text, NOW) == expected


@pytest.mark.parametrize("text", [
    "rappelle-moi de payer le loyer",                 # pas de moment
    "aujourd'hui à 8h rappelle-moi de partir",        # déjà passé
    "chaque lundi à 25h rappelle-moi x",              # heure invalide
    "le 31 février à 9h rappelle-moi x",              # date invalide
    "chaque lundi à 8h",                              # contenu vide
    "rappelle-moi lundi prochain vers 8 heures de x", # forme non couverte → repli
])
def test_non_compris(text):
    assert parse_nl(text, NOW) is None
```

Note sur le cas « envoie-moi par texto : vérifier le four » : le canal est retiré, le reste n'est ni un rappel explicite ni une commande, donc consigne ; la reformulation montrée avant `/confirm` permet à l'utilisateur de corriger.

- [ ] **Step 2: Vérifier l'échec** — `python3 -m pytest tiron_cron/tests/test_nl.py -v` → FAIL.

- [ ] **Step 3: Implémenter**

```python
# tiron_cron/nl.py
"""Analyse déterministe des demandes de planification en français courant.
Renvoie None dès qu'un doute subsiste : le repli (fallback.py) prend le relais,
puis la reformulation + /confirm laisse le dernier mot à l'utilisateur."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from tiron_cron.syntax import COMMANDES_PLANIFIABLES, Job

_JOURS = {"lundi": 1, "mardi": 2, "mercredi": 3, "jeudi": 4, "vendredi": 5, "samedi": 6, "dimanche": 0}
_MOIS = {"janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
         "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
         "décembre": 12, "decembre": 12}
_H = r"(\d{1,2})\s*h\s*(\d{2})?"
_I = re.IGNORECASE
_CANAL = re.compile(r"\b(?:par|en|sur|via)\s+(sms|texto|whatsapp|telegram|obsidian)\b", _I)
_RELATIF = re.compile(r"\bdans\s+(\d+)\s*(minutes?|min|heures?|h|jours?)\b", _I)
_HEBDO = re.compile(r"\b(?:chaque|tous\s+les)\s+(" + "|".join(_JOURS) + r")s?\s+à\s+" + _H, _I)
_QUOTIDIEN = re.compile(r"\b(?:chaque\s+jour|tous\s+les\s+jours|chaque\s+matin|tous\s+les\s+matins|"
                        r"quotidiennement)\s+à\s+" + _H, _I)
_PROCHE = re.compile(r"\b(aujourd'hui|demain|après-demain)\s+à\s+" + _H, _I)
_DATE = re.compile(r"\ble\s+(\d{1,2})\s+(" + "|".join(_MOIS) + r")(?:\s+(\d{4}))?\s+à\s+" + _H, _I)
_TIRON = re.compile(r"^\s*tiron\s*[,:]?\s*", _I)
_RAPPEL = re.compile(r"^\s*(?:rappelle[- ]moi|rappelle[- ]nous|rappelez[- ]moi)\s*(?:de\s+|d'|que\s+|qu'|:\s*)?", _I)
_CMD = re.compile(r"(?<!\S)(/[a-z]+)\b")


def _hm(h: str, m: str | None) -> tuple[int, int] | None:
    hh, mm = int(h), int(m or 0)
    return (hh, mm) if 0 <= hh <= 23 and 0 <= mm <= 59 else None


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


def _quand(text: str, now: datetime) -> tuple[str, str, re.Match] | None:
    if m := _RELATIF.search(text):
        unite = m.group(2).lower()[0]
        return "at", f"{int(m.group(1))}{'d' if unite == 'j' else unite}", m
    if m := _HEBDO.search(text):
        hm = _hm(m.group(2), m.group(3))
        return ("cron", f"{hm[1]} {hm[0]} * * {_JOURS[m.group(1).lower()]}", m) if hm else None
    if m := _QUOTIDIEN.search(text):
        hm = _hm(m.group(1), m.group(2))
        return ("cron", f"{hm[1]} {hm[0]} * * *", m) if hm else None
    if m := _PROCHE.search(text):
        hm = _hm(m.group(2), m.group(3))
        if not hm:
            return None
        decalage = {"aujourd'hui": 0, "demain": 1, "après-demain": 2}[m.group(1).lower()]
        dt = (now + timedelta(days=decalage)).replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
        return ("at", _iso(dt), m) if dt > now else None
    if m := _DATE.search(text):
        hm = _hm(m.group(4), m.group(5))
        if not hm:
            return None
        annee = int(m.group(3)) if m.group(3) else now.year
        try:
            dt = now.replace(year=annee, month=_MOIS[m.group(2).lower()], day=int(m.group(1)),
                             hour=hm[0], minute=hm[1], second=0, microsecond=0)
            if dt <= now and not m.group(3):
                dt = dt.replace(year=annee + 1)
        except ValueError:
            return None
        return ("at", _iso(dt), m) if dt > now else None
    return None


def _nettoie(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" ,.;:!")


def parse_nl(text: str, now: datetime) -> Job | None:
    canal = None
    if m := _CANAL.search(text):
        canal = {"texto": "sms"}.get(m.group(1).lower(), m.group(1).lower())
        text = text[:m.start()] + " " + text[m.end():]
    q = _quand(text, now)
    if q is None:
        return None
    kind, quand, m = q
    reste = _nettoie(_TIRON.sub("", _nettoie(text[:m.start()] + " " + text[m.end():])))
    if (c := _CMD.search(reste)) and c.group(1) in COMMANDES_PLANIFIABLES:
        type_, contenu = "commande", reste[c.start():]
    elif r := _RAPPEL.match(reste):
        type_, contenu = "rappel", reste[r.end():]
    else:
        type_, contenu = "consigne", reste
    contenu = _nettoie(contenu)
    return Job(kind, quand, canal, type_, contenu) if contenu else None
```

Si un cas de test échoue sur un détail de nettoyage (espace, ponctuation), corriger `_nettoie` ou l'expression concernée, jamais le test attendu sans en parler.

- [ ] **Step 4: Vérifier le succès** — `python3 -m pytest tiron_cron/tests/test_nl.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tiron_cron/nl.py tiron_cron/tests/test_nl.py
git commit -m "feat(cron): analyseur déterministe des planifications en français"
```

---

### Task 6: `tiron_cron/fallback.py` — repli qwen3:8b local

**Files:**
- Create: `tiron_cron/fallback.py`
- Test: `tiron_cron/tests/test_fallback.py`

**Interfaces:**
- Consumes: `parse_command`, `Job`, `SyntaxeInvalide` (tâche 2).
- Produces: `normalize_with_model(text: str, now: datetime, env: Mapping[str, str], http=_post_json) -> Job | None` ; `http(url: str, body: dict, timeout: int) -> dict`.

- [ ] **Step 1: Écrire les tests**

```python
# tiron_cron/tests/test_fallback.py
from datetime import datetime
from zoneinfo import ZoneInfo

from tiron_cron.fallback import normalize_with_model
from tiron_cron.syntax import Job

NOW = datetime(2026, 10, 2, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))


def _http(content):
    calls = []

    def http(url, body, timeout):
        calls.append((url, body))
        if isinstance(content, Exception):
            raise content
        return {"message": {"content": content}}
    http.calls = calls
    return http


def test_ligne_valide():
    http = _http('add "0 8 * * 1" rappel payer le loyer')
    assert normalize_with_model("x", NOW, {}, http) == Job("cron", "0 8 * * 1", None, "rappel", "payer le loyer")
    url, body = http.calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert body["model"] == "qwen3:8b" and body["think"] is False and body["stream"] is False
    assert "2026-10-02T10:00" in body["messages"][0]["content"]


def test_prefixe_cron_et_bloc_de_code_toleres():
    assert normalize_with_model("x", NOW, {}, _http("```\n/cron add 20m rappel x\n```")) is not None


def test_env_surcharge_url_et_modele():
    http = _http("add 20m rappel x")
    normalize_with_model("x", NOW, {"OLLAMA_URL": "http://h:1", "TIRON_CRON_MODEL": "m"}, http)
    assert http.calls[0][0] == "http://h:1/api/chat" and http.calls[0][1]["model"] == "m"


def test_aucune_syntaxe_invalide_gestion_erreur_reseau():
    assert normalize_with_model("x", NOW, {}, _http("AUCUNE")) is None
    assert normalize_with_model("x", NOW, {}, _http("add demain rappel x")) is None
    assert normalize_with_model("x", NOW, {}, _http("rm 1234")) is None
    assert normalize_with_model("x", NOW, {}, _http(OSError("refused"))) is None
```

- [ ] **Step 2: Vérifier l'échec** — `python3 -m pytest tiron_cron/tests/test_fallback.py -v` → FAIL.

- [ ] **Step 3: Implémenter**

```python
# tiron_cron/fallback.py
"""Repli quand l'analyseur déterministe échoue : qwen3:8b LOCAL (Ollama), jamais
un fournisseur distant. `think: false` est indispensable (sinon Qwen3 épuise
son budget en raisonnement caché, cf. mémoire du 2026-09-30). La sortie est
revalidée par parse_command : le modèle ne peut rien créer d'invalide."""
from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from datetime import datetime

from tiron_cron.syntax import Job, SyntaxeInvalide, parse_command

_JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
SYSTEM_PROMPT = """Tu convertis une demande de planification en UNE ligne de la forme :
add <quand> [<canal>] <type> <contenu>
- <quand> : expression cron 5 champs entre guillemets doubles ("0 8 * * 1"), OU date-heure locale ISO (2026-10-02T09:00), OU durée (20m, 2h, 1d).
- <canal> (facultatif, seulement si la demande le précise) : telegram, whatsapp, sms ou obsidian.
- <type> : rappel (un texte à rappeler), commande (une commande Tiron commençant par /), consigne (une tâche confiée à l'assistant).
- <contenu> : le texte du rappel, la commande, ou la consigne.
Maintenant : {now} ({jour}). Réponds par la ligne seule, sans commentaire.
Si la demande n'est pas une planification, réponds : AUCUNE"""


def _post_json(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def normalize_with_model(text: str, now: datetime, env: Mapping[str, str], http=_post_json) -> Job | None:
    url = env.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
    body = {
        "model": env.get("TIRON_CRON_MODEL", "qwen3:8b"),
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                now=now.strftime("%Y-%m-%dT%H:%M"), jour=_JOURS[now.weekday()])},
            {"role": "user", "content": text},
        ],
    }
    try:
        content = (http(url, body, 120).get("message") or {}).get("content", "")
    except Exception:
        return None
    for line in content.splitlines():
        line = line.strip().strip("`").strip()
        if line.lower().startswith("/cron "):
            line = line[6:].strip()
        if line.lower().startswith("add "):
            try:
                r = parse_command(line)
            except SyntaxeInvalide:
                return None
            return r if isinstance(r, Job) else None
    return None
```

- [ ] **Step 4: Vérifier le succès** — `python3 -m pytest tiron_cron/tests/test_fallback.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tiron_cron/fallback.py tiron_cron/tests/test_fallback.py
git commit -m "feat(cron): repli qwen3:8b local pour les planifications non comprises"
```

---

### Task 7: `tiron_cron/deliver_obsidian.py` — rappel écrit dans le coffre

**Files:**
- Create: `tiron_cron/deliver_obsidian.py`
- Test: `tiron_cron/tests/test_deliver_obsidian.py`

**Interfaces:**
- Produces: `deliver(text: str, now: datetime, env: Mapping[str, str]) -> Path`.

- [ ] **Step 1: Écrire les tests**

```python
# tiron_cron/tests/test_deliver_obsidian.py
from datetime import datetime

from tiron_cron.deliver_obsidian import deliver

NOW = datetime(2026, 10, 6, 8, 0)


def test_ecrit_la_note(tmp_path):
    p = deliver("payer le loyer", NOW, {"TIRON_RAPPELS_DIR": str(tmp_path / "Rappels")})
    assert p == tmp_path / "Rappels" / "2026-10-06 0800 payer le loyer.md"
    assert p.read_text(encoding="utf-8") == \
        "---\ntags: [rappel]\ndate: 2026-10-06T08:00\n---\n\npayer le loyer\n"


def test_titre_sans_caracteres_interdits(tmp_path):
    p = deliver('a/b:c*?"<>|#^[x]', NOW, {"TIRON_RAPPELS_DIR": str(tmp_path)})
    assert p.name == "2026-10-06 0800 abcx.md"


def test_titre_vide_et_collision(tmp_path):
    env = {"TIRON_RAPPELS_DIR": str(tmp_path)}
    p1 = deliver("///", NOW, env)
    p2 = deliver("///", NOW, env)
    assert p1.name == "2026-10-06 0800 rappel.md" and p2.name == "2026-10-06 0800 rappel 2.md"
```

- [ ] **Step 2: Vérifier l'échec** — `python3 -m pytest tiron_cron/tests/test_deliver_obsidian.py -v` → FAIL.

- [ ] **Step 3: Implémenter**

```python
# tiron_cron/deliver_obsidian.py
"""Livraison d'un rappel sur le canal obsidian : une note dans le coffre,
synchronisée par obsidian-headless. Pas de notification active en V1."""
from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


def deliver(text: str, now: datetime, env: Mapping[str, str]) -> Path:
    d = Path(env.get("TIRON_RAPPELS_DIR") or Path.home() / "Documents" / "Arbath" / "Rappels")
    d.mkdir(parents=True, exist_ok=True)
    titre = re.sub(r'[\\/:*?"<>|#^\[\]]', "", text)[:50].strip() or "rappel"
    base = f"{now:%Y-%m-%d %H%M} {titre}"
    p, n = d / f"{base}.md", 2
    while p.exists():
        p, n = d / f"{base} {n}.md", n + 1
    p.write_text(f"---\ntags: [rappel]\ndate: {now:%Y-%m-%dT%H:%M}\n---\n\n{text}\n", encoding="utf-8")
    return p
```

- [ ] **Step 4: Vérifier le succès** — PASS.

- [ ] **Step 5: Commit**

```bash
git add tiron_cron/deliver_obsidian.py tiron_cron/tests/test_deliver_obsidian.py
git commit -m "feat(cron): livraison des rappels dans le coffre Obsidian"
```

---

### Task 8: `tiron_cron/__main__.py` — CLI JSON

**Files:**
- Create: `tiron_cron/__main__.py`
- Test: `tiron_cron/tests/test_cli.py`

**Interfaces:**
- Consumes: tâches 2 à 7.
- Produces (contrat utilisé par derisk-deleg et wiki-lm-server) — une ligne JSON sur stdout, code de sortie 0 :
  - `preview --origin O --args A` → `{"ok": true, "text": <reformulation>, "args": <args normalisés>}`
  - `add --origin O --args A` → `{"ok": true, "text": "Tâche créée (<id8>) : <reformulation>", "id": <id complet>}`
  - `nl --origin O --text T` → comme `preview`, plus `"source": "regles" | "modele"`
  - `manage --args A` (list/rm/pause/reprendre) → `{"ok": true, "text": ...}`
  - `deliver-obsidian --text T` → `{"ok": true, "text": <chemin>}`
  - toute erreur → `{"ok": false, "text": <message>}`
  - `main(argv, runner=subprocess.run, env=os.environ, now=None, http=None) -> int` (injection pour les tests).

- [ ] **Step 1: Écrire les tests**

```python
# tiron_cron/tests/test_cli.py
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from tiron_cron.__main__ import main
from tiron_cron.tests.test_openclaw_cli import ENV, JOBS, _runner

NOW = datetime(2026, 10, 2, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))


def _call(capsys, argv, **kw):
    kw.setdefault("env", ENV)
    kw.setdefault("now", NOW)
    assert main(argv, **kw) == 0
    return json.loads(capsys.readouterr().out)


def test_preview(capsys):
    out = _call(capsys, ["preview", "--origin", "telegram", "--args", 'add "0 8 * * 1" rappel payer le loyer'])
    assert out == {"ok": True, "args": 'add "0 8 * * 1" rappel payer le loyer',
                   "text": "Chaque lundi à 08:00 (Europe/Paris), telegram, rappel : payer le loyer"}


def test_preview_origine_inconnue_sans_canal(capsys):
    out = _call(capsys, ["preview", "--origin", "webchat", "--args", "add 20m rappel x"])
    assert out["ok"] is False and "canal" in out["text"]


def test_preview_refuse_la_gestion(capsys):
    out = _call(capsys, ["preview", "--origin", "telegram", "--args", "list"])
    assert out["ok"] is False


def test_add(capsys):
    run = _runner(["{}", json.dumps({"jobs": [{"id": "cccc3333-q", "declarationKey": "tiron:FIXE"}]})])
    out = _call(capsys, ["add", "--origin", "sms", "--args", "add 20m rappel x"],
                runner=run, job_key="tiron:FIXE")
    assert out["ok"] and out["id"] == "cccc3333-q" and out["text"].startswith("Tâche créée (cccc3333)")
    assert run.calls[0][:3] == ["openclaw", "cron", "add"]


def test_add_erreur_openclaw(capsys):
    out = _call(capsys, ["add", "--origin", "sms", "--args", "add 20m rappel x"],
                runner=_runner([RuntimeError("gateway arrêté")]))
    assert out == {"ok": False, "text": "Échec de la création : gateway arrêté"}


def test_nl_regles(capsys):
    out = _call(capsys, ["nl", "--origin", "telegram", "--text", "dans 20 minutes rappelle-moi de x"])
    assert out["ok"] and out["source"] == "regles" and out["args"] == "add 20m rappel x"


def test_nl_repli_modele(capsys):
    http = lambda url, body, timeout: {"message": {"content": "add 1d rappel x"}}
    out = _call(capsys, ["nl", "--origin", "telegram", "--text", "lundi prochain vers 8 heures x"], http=http)
    assert out["ok"] and out["source"] == "modele" and out["args"] == "add 1d rappel x"


def test_nl_non_compris(capsys):
    http = lambda url, body, timeout: {"message": {"content": "AUCUNE"}}
    out = _call(capsys, ["nl", "--origin", "telegram", "--text", "bonjour"], http=http)
    assert out["ok"] is False and "/cron add" in out["text"]


def test_manage(capsys):
    out = _call(capsys, ["manage", "--args", "list"], runner=_runner([json.dumps(JOBS)]))
    assert out["ok"] and "aaaa1111" in out["text"]


def test_manage_refuse_add(capsys):
    assert _call(capsys, ["manage", "--args", "add 20m rappel x"])["ok"] is False


def test_deliver_obsidian(capsys, tmp_path):
    out = _call(capsys, ["deliver-obsidian", "--text", "payer"], env={"TIRON_RAPPELS_DIR": str(tmp_path)})
    assert out["ok"] and out["text"].endswith("payer.md")
```

- [ ] **Step 2: Vérifier l'échec** — `python3 -m pytest tiron_cron/tests/test_cli.py -v` → FAIL.

- [ ] **Step 3: Implémenter**

```python
# tiron_cron/__main__.py
"""CLI JSON de tiron_cron — seul point d'entrée pour derisk-deleg et wiki-lm-server.
Toujours code 0 et une ligne JSON {"ok": bool, "text": str, ...}."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from tiron_cron import fallback
from tiron_cron.deliver_obsidian import deliver
from tiron_cron.describe import describe
from tiron_cron.nl import parse_nl
from tiron_cron.openclaw_cli import OpenClawError, add_job, build_add_argv, manage
from tiron_cron.syntax import CANAUX, USAGE_ADD, Gestion, Job, SyntaxeInvalide, parse_command, to_args


def _job_from_args(args: str) -> Job:
    r = parse_command(args)
    if not isinstance(r, Job):
        raise SyntaxeInvalide(USAGE_ADD)
    return r


def _origine(origin: str) -> str | None:
    o = (origin or "").lower()
    return o if o in CANAUX else None


def _apercu(job: Job, origin: str, tz: str, env, job_key: str) -> tuple[str, list[str]]:
    canal = job.canal or _origine(origin)
    argv = build_add_argv(job, canal, tz, env, job_key)  # valide canal + destinataire dès l'aperçu
    return describe(job, canal, tz), argv


def main(argv=None, runner=subprocess.run, env=None, now=None, http=None, job_key=None) -> int:
    env = os.environ if env is None else env
    tz = env.get("TIRON_TZ", "Europe/Paris")
    now = now or datetime.now(ZoneInfo(tz))
    job_key = job_key or f"tiron:{uuid.uuid4().hex[:8]}"
    p = argparse.ArgumentParser(prog="tiron_cron")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("preview", "add"):
        s = sub.add_parser(name)
        s.add_argument("--origin", default="")
        s.add_argument("--args", required=True)
    s = sub.add_parser("nl")
    s.add_argument("--origin", default="")
    s.add_argument("--text", required=True)
    sub.add_parser("manage").add_argument("--args", required=True)
    sub.add_parser("deliver-obsidian").add_argument("--text", required=True)
    a = p.parse_args(argv)

    try:
        if a.cmd == "preview":
            job = _job_from_args(a.args)
            text, _ = _apercu(job, a.origin, tz, env, job_key)
            out = {"ok": True, "text": text, "args": to_args(job)}
        elif a.cmd == "add":
            job = _job_from_args(a.args)
            text, cmd = _apercu(job, a.origin, tz, env, job_key)
            try:
                full = add_job(cmd, job_key, runner)
            except OpenClawError as e:
                out = {"ok": False, "text": f"Échec de la création : {e}"}
            else:
                out = {"ok": True, "id": full, "text": f"Tâche créée ({full[:8]}) : {text}"}
        elif a.cmd == "nl":
            job, source = parse_nl(a.text, now), "regles"
            if job is None:
                kw = {"http": http} if http else {}
                job, source = fallback.normalize_with_model(a.text, now, env, **kw), "modele"
            if job is None:
                out = {"ok": False, "text": "Planification non comprise. Syntaxe : /cron " + USAGE_ADD.split("/cron ", 1)[1]}
            else:
                text, _ = _apercu(job, a.origin, tz, env, job_key)
                out = {"ok": True, "text": text, "args": to_args(job), "source": source}
        elif a.cmd == "manage":
            g = parse_command(a.args)
            if not isinstance(g, Gestion):
                raise SyntaxeInvalide("manage : list, rm, pause ou reprendre")
            try:
                out = {"ok": True, "text": manage(g, runner)}
            except OpenClawError as e:
                out = {"ok": False, "text": f"Échec : {e}"}
        else:
            out = {"ok": True, "text": str(deliver(a.text, now, env))}
    except SyntaxeInvalide as e:
        out = {"ok": False, "text": str(e)}
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note : `USAGE_ADD.split("/cron ", 1)[1]` donne `add <quand> …` ; le test `test_nl_non_compris` vérifie la présence de `/cron add`.

- [ ] **Step 4: Vérifier le succès**

Run: `python3 -m pytest tiron_cron -v`
Expected: PASS (toute la suite tiron_cron).

Vérification réelle sans effet de bord :
```bash
cd ~/Secretarius && TIRON_TELEGRAM_CHAT_ID=1 python3 -m tiron_cron preview --origin telegram --args 'add "0 8 * * 1" rappel payer le loyer'
```
Expected: `{"ok": true, "text": "Chaque lundi à 08:00 (Europe/Paris), telegram, rappel : payer le loyer", "args": ...}`

- [ ] **Step 5: Commit**

```bash
git add tiron_cron/__main__.py tiron_cron/tests/test_cli.py
git commit -m "feat(cron): CLI JSON tiron_cron (preview, add, nl, manage, deliver-obsidian)"
```

---

### Task 9: router_service — `/cron` tapé explicitement

`COMMAND_CRITERIA` n'est **pas** modifié ici : le checkpoint Laya actuel n'a jamais vu `/cron`, et ajouter un choix inconnu peut perturber les autres classements. Il change avec le nouveau checkpoint (tâche 13).

**Files:**
- Modify: `router_service/router.py` (après `GOG_CMDS`)
- Modify: `router_service/server.py` (import, `route_message`, `NEEDS_ARG`)
- Test: `router_service/test_router.py` (ou le fichier qui teste `route_message` ; `grep -ln route_message router_service/test_*.py`)

**Interfaces:**
- Produces: `CRON_CMDS = {"/cron"}` ; `route_message("/cron add …")` → `{"status": "ok", "command": "/cron", "args": "add …"}`.

- [ ] **Step 1: Écrire les tests** (dans le fichier de tests de `route_message`, en suivant ses imports existants)

```python
def test_cron_explicite_honore_sans_laya():
    from router_service import server
    assert server.route_message('/cron add "0 8 * * 1" rappel x') == {
        "status": "ok", "command": "/cron", "args": 'add "0 8 * * 1" rappel x'}


def test_cron_sans_argument_renvoie_l_usage():
    from router_service import server
    assert server.route_message("/cron") == {"status": "answer", "reply": "Usage : /cron <argument>"}


def test_cron_disjoint_des_autres_ensembles():
    from router_service.router import CRON_CMDS, GOG_CMDS, WIKI_CMDS
    assert CRON_CMDS.isdisjoint(WIKI_CMDS | GOG_CMDS)
```

Si l'import de `router_service.server` échoue faute de `LAYA_CHECKPOINT`, utiliser `monkeypatch.setenv("LAYA_CHECKPOINT", "x")` avant l'import, comme le font les tests existants (vérifier leur fixture).

- [ ] **Step 2: Vérifier l'échec** — `cd ~/Secretarius && python3 -m pytest router_service -k cron -v` → FAIL.

- [ ] **Step 3: Implémenter**

Dans `router_service/router.py`, après `GOG_CMDS = {...}` :
```python
CRON_CMDS = {"/cron"}
```

Dans `router_service/server.py` :
```python
from router_service.router import WIKI_CMDS, GOG_CMDS, CRON_CMDS, embed_bge_m3
```
```python
NEEDS_ARG = {"/c", "/q", "/source", "/chercher", "/repondre", "/r", "/lire", "/supprimer", "/verifie", "/cron"}
```
```python
        if cmd in WIKI_CMDS or cmd in GOG_CMDS or cmd in CRON_CMDS:
```

- [ ] **Step 4: Vérifier le succès** — `python3 -m pytest router_service -v` → PASS (y compris les tests existants).

- [ ] **Step 5: Commit**

```bash
git add router_service/router.py router_service/server.py router_service/test_*.py
git commit -m "feat(routeur): /cron tapé explicitement"
```

---

### Task 10: derisk-deleg — `/cron` dans Tiron

**Files:**
- Modify: `derisk-deleg/src/dispatch.ts`
- Create: `derisk-deleg/src/cron-ops.ts`, `derisk-deleg/src/cron-ops.test.ts`
- Modify: `derisk-deleg/src/index.ts` (type `Pending` l.105-108, `/confirm` l.638-656, `/annuler` l.659-678, routage après l.711)
- Test: `derisk-deleg/src/dispatch.test.ts`, `derisk-deleg/src/index.test.ts`

**Interfaces:**
- Consumes: CLI JSON de la tâche 8 ; `/cron` du routeur (tâche 9) ; `ctx.messageProvider` (canal d'origine).
- Produces: `runTironCron(argv: string[]): Promise<CronResult>` avec `type CronResult = { ok: boolean; text: string; args?: string }` ; `commandToAction("/cron")` → `{ kind: "cron" }`.

- [ ] **Step 1: Tests de `dispatch` et `cron-ops`**

Dans `derisk-deleg/src/dispatch.test.ts`, ajouter :
```ts
it("/cron -> kind cron", () => {
  expect(commandToAction("/cron")).toEqual({ kind: "cron" });
});
```

Créer `derisk-deleg/src/cron-ops.test.ts` :
```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { execFile } from "node:child_process";

vi.mock("node:child_process", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:child_process")>();
  return { ...actual, execFile: vi.fn() };
});

afterEach(() => { vi.mocked(execFile).mockReset(); vi.unstubAllEnvs(); });

function stdoutIs(out: string) {
  vi.mocked(execFile).mockImplementation(((_c: any, _a: any, _o: any, cb: any) =>
    cb(null, { stdout: out, stderr: "" })) as any);
}

describe("runTironCron", () => {
  it("relaie le JSON de la CLI et lance python3 -m tiron_cron dans SECRETARIUS_DIR", async () => {
    vi.stubEnv("SECRETARIUS_DIR", "/srv/sec");
    stdoutIs('{"ok": true, "text": "Chaque lundi…", "args": "add …"}');
    const { runTironCron } = await import("./cron-ops.js");
    expect(await runTironCron(["preview", "--origin", "telegram", "--args", "add …"]))
      .toEqual({ ok: true, text: "Chaque lundi…", args: "add …" });
    const [cmd, args, opts] = vi.mocked(execFile).mock.calls[0] as any[];
    expect(cmd).toBe("python3");
    expect(args).toEqual(["-m", "tiron_cron", "preview", "--origin", "telegram", "--args", "add …"]);
    expect(opts.cwd).toBe("/srv/sec");
  });

  it("échec du processus -> ok:false avec message", async () => {
    vi.mocked(execFile).mockImplementation(((_c: any, _a: any, _o: any, cb: any) =>
      cb(new Error("ENOENT python3"))) as any);
    const { runTironCron } = await import("./cron-ops.js");
    const r = await runTironCron(["manage", "--args", "list"]);
    expect(r.ok).toBe(false);
    expect(r.text).toMatch(/^Planificateur indisponible : .*ENOENT/);
  });
});
```

- [ ] **Step 2: Vérifier l'échec** — `cd ~/Secretarius/derisk-deleg && npx vitest run src/dispatch.test.ts src/cron-ops.test.ts` → FAIL.

- [ ] **Step 3: Implémenter `dispatch.ts` et `cron-ops.ts`**

`dispatch.ts` : ajouter `| "/cron"` à `RouterCommand`, `| { kind: "cron" }` à `ActionKind`, et `"/cron": { kind: "cron" },` à `TABLE`.

```ts
// derisk-deleg/src/cron-ops.ts
// Appel de la CLI JSON tiron_cron (cf. docs/superpowers/specs/2026-10-01-cron-tiron-design.md).
// Asynchrone : le repli qwen3:8b peut prendre des dizaines de secondes, il ne
// doit jamais bloquer la boucle d'événements du gateway.
import { execFile } from "node:child_process";
import { homedir } from "node:os";
import { promisify } from "node:util";

const execFileP = promisify(execFile);

export type CronResult = { ok: boolean; text: string; args?: string };

export async function runTironCron(argv: string[]): Promise<CronResult> {
  const cwd = process.env.SECRETARIUS_DIR || `${homedir()}/Secretarius`;
  try {
    const { stdout } = await execFileP("python3", ["-m", "tiron_cron", ...argv], {
      cwd,
      timeout: 150000,
      encoding: "utf8",
    });
    const data = JSON.parse(String(stdout));
    return { ok: !!data.ok, text: String(data.text ?? ""), ...(data.args ? { args: String(data.args) } : {}) };
  } catch (e: any) {
    return { ok: false, text: `Planificateur indisponible : ${String(e?.message ?? e).slice(0, 300)}` };
  }
}
```

- [ ] **Step 4: Vérifier** — `npx vitest run src/dispatch.test.ts src/cron-ops.test.ts` → PASS.

- [ ] **Step 5: Tests d'intégration dans `index.test.ts`**

Ajouter à la fin de `index.test.ts` (même harnais `freshPlugin`/`makeApi`, routeur simulé par `vi.stubGlobal("fetch", …)`, `cron-ops` simulé par `vi.doMock` comme `wiki-ops`). Ajouter `vi.doUnmock("./cron-ops.js");` dans le `afterEach` existant.

```ts
describe("before_agent_reply — /cron", () => {
  function routerReturns(command: string, args: string) {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ status: "ok", command, args }) })));
  }

  it("/cron add tapé : aperçu puis /confirm crée la tâche", async () => {
    const runTironCron = vi.fn()
      .mockResolvedValueOnce({ ok: true, text: "Chaque lundi à 08:00 (Europe/Paris), telegram, rappel : loyer", args: 'add "0 8 * * 1" rappel loyer' })
      .mockResolvedValueOnce({ ok: true, text: "Tâche créée (aaaa1111) : …" });
    vi.doMock("./cron-ops.js", () => ({ runTironCron }));
    routerReturns("/cron", 'add "0 8 * * 1" rappel loyer');
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);
    const ctx = { messageProvider: "telegram" };

    const r1 = await hooks["before_agent_reply"].handler({ cleanedBody: '/cron add "0 8 * * 1" rappel loyer' }, ctx);
    expect(r1.reply.text).toMatch(/^Tâche à créer : Chaque lundi.*\/confirm/s);
    expect(runTironCron).toHaveBeenNthCalledWith(1, ["preview", "--origin", "telegram", "--args", 'add "0 8 * * 1" rappel loyer']);

    const r2 = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" }, ctx);
    expect(r2.reply.text).toBe("Tâche créée (aaaa1111) : …");
    expect(runTironCron).toHaveBeenNthCalledWith(2, ["add", "--origin", "telegram", "--args", 'add "0 8 * * 1" rappel loyer']);
  });

  it("langage naturel : passe par nl, puis /annuler ne crée rien", async () => {
    const runTironCron = vi.fn().mockResolvedValueOnce({ ok: true, text: "Dans 20 min …", args: "add 20m rappel x" });
    vi.doMock("./cron-ops.js", () => ({ runTironCron }));
    routerReturns("/cron", "dans 20 minutes rappelle-moi de x");
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    await hooks["before_agent_reply"].handler({ cleanedBody: "dans 20 minutes rappelle-moi de x" }, { messageProvider: "telegram" });
    expect(runTironCron).toHaveBeenCalledWith(["nl", "--origin", "telegram", "--text", "dans 20 minutes rappelle-moi de x"]);
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/annuler" }, {});
    expect(r.reply.text).toBe("Planification abandonnée. Rien n'a été créé.");
    expect(runTironCron).toHaveBeenCalledTimes(1);
  });

  it("aperçu en erreur : message relayé, rien en attente", async () => {
    const runTironCron = vi.fn().mockResolvedValueOnce({ ok: false, text: "destinataire sms non configuré (TIRON_SMS_TO)" });
    vi.doMock("./cron-ops.js", () => ({ runTironCron }));
    routerReturns("/cron", "add 20m sms rappel x");
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/cron add 20m sms rappel x" }, {});
    expect(r.reply.text).toBe("destinataire sms non configuré (TIRON_SMS_TO)");
    const c = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" }, {});
    expect(c.reply.text).toMatch(/^Rien à confirmer/);
  });

  it("/cron list tapé : exécuté directement par manage", async () => {
    const runTironCron = vi.fn().mockResolvedValueOnce({ ok: true, text: "Aucune tâche planifiée." });
    vi.doMock("./cron-ops.js", () => ({ runTironCron }));
    routerReturns("/cron", "list");
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/cron list" }, {});
    expect(r.reply.text).toBe("Aucune tâche planifiée.");
    expect(runTironCron).toHaveBeenCalledWith(["manage", "--args", "list"]);
  });
});
```

- [ ] **Step 6: Vérifier l'échec** — `npx vitest run src/index.test.ts -t "/cron"` → FAIL.

- [ ] **Step 7: Implémenter dans `index.ts`**

Import en tête :
```ts
import { runTironCron } from "./cron-ops.js";
```

Type `Pending` :
```ts
type Pending =
  | { kind: "send"; to: string; subject: string; body: string; ts: number }
  | { kind: "reply"; messageId: string; body: string; ts: number }
  | { kind: "router-write"; op: string; args: string; ts: number }
  | { kind: "cron"; args: string; origin: string; ts: number };
```

Dans `/confirm`, remplacer la chaîne `if/else if/else` par :
```ts
        if (p.kind === "send") {
          out = await delegateGog(api, "send", `to=${p.to}; subject=${p.subject}; body=${p.body}`);
        } else if (p.kind === "reply") {
          out = await delegateGog(api, "reply", `id=${p.messageId}; body=${p.body}`);
        } else if (p.kind === "cron") {
          out = (await runTironCron(["add", "--origin", p.origin, "--args", p.args])).text;
        } else {
          out = await runWikiOp(api, p.op, p.args);
        }
```

Dans `/annuler`, juste après le bloc `if (pending.kind === "router-write") { … }` :
```ts
        if (pending.kind === "cron") {
          pending = null;
          return { handled: true, reply: { text: "Planification abandonnée. Rien n'a été créé." } };
        }
```

Dans le routage, juste après le calcul de `explicitCommand` et avant `if (action.kind === "wiki")` :
```ts
      // /cron : la création passe TOUJOURS par aperçu + /confirm (tapée ou
      // inférée — cf. spec cron) ; list/rm/pause/reprendre tapés s'exécutent.
      if (action.kind === "cron") {
        const origin = String(ctx?.messageProvider ?? "");
        const sub = routed.args.trim().split(/\s+/)[0]?.toLowerCase() ?? "";
        if (explicitCommand && sub !== "add") {
          const r = await runTironCron(["manage", "--args", routed.args]);
          return { handled: true, reply: { text: r.text.slice(0, 4000) } };
        }
        const r = explicitCommand
          ? await runTironCron(["preview", "--origin", origin, "--args", routed.args])
          : await runTironCron(["nl", "--origin", origin, "--text", routed.args]);
        if (!r.ok || !r.args) {
          return { handled: true, reply: { text: r.text.slice(0, 4000) } };
        }
        pending = { kind: "cron", args: r.args, origin, ts: Date.now() };
        return {
          handled: true,
          reply: { text: `Tâche à créer : ${r.text}\n\nTapez /confirm pour la créer (valable 10 min), ou /annuler pour abandonner.` },
        };
      }
```

- [ ] **Step 8: Vérifier le succès**

Run: `cd ~/Secretarius/derisk-deleg && npm test && npm run build`
Expected: toute la suite PASS, compilation sans erreur.

- [ ] **Step 9: Commit**

```bash
git add derisk-deleg/src/dispatch.ts derisk-deleg/src/dispatch.test.ts derisk-deleg/src/cron-ops.ts derisk-deleg/src/cron-ops.test.ts derisk-deleg/src/index.ts derisk-deleg/src/index.test.ts derisk-deleg/dist
git commit -m "feat(derisk-deleg): /cron — aperçu, /confirm, langage naturel, gestion"
```
(Vérifier d'abord avec `git ls-files derisk-deleg/dist | head` que `dist/` est bien suivi ; sinon ne pas l'ajouter.)

---

### Task 11: Obsidian — `/cron?`, `/cron!`, `/cron`

**Files:**
- Modify: `Wiki_LM/tools/server.py` (`_RUN_OPS` l.136-169, + fonction `_tiron_cron`)
- Modify: `Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts` (`SUPPORTED_COMMANDS`, `formatWikiResult`)
- Test: `Wiki_LM/tests/test_server.py`, `Wiki_LM/obsidian-wikilm-capture/src/run-commands.test.ts`

**Interfaces:**
- Consumes: CLI JSON (tâche 8).
- Produces: `POST /run {"command": "/cron?"|"/cron!"|"/cron", "arg": …}` → `{"text": …}` (succès) ou `{"error": …}` (HTTP 200, affiché « **Erreur :** … » par le plugin).

Format d'un bloc dans une note :
````
```wiki
/cron!
add "0 8 * * 1" sms rappel payer le loyer
```
````

- [ ] **Step 1: Tests serveur** (dans `Wiki_LM/tests/test_server.py`, réutiliser la fixture `client`)

```python
class TestHandleRunCron:
    def _fake(self, monkeypatch, out):
        import server
        calls = []

        def run(argv, **kw):
            calls.append((argv, kw))
            import subprocess
            return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")
        monkeypatch.setattr(server.subprocess, "run", run)
        return calls

    def test_apercu(self, client, monkeypatch):
        calls = self._fake(monkeypatch, '{"ok": true, "text": "Chaque lundi…", "args": "add …"}')
        r = client.post("/run", json={"command": "/cron?", "arg": 'add "0 8 * * 1" rappel x'})
        assert r.status_code == 200 and r.get_json() == {"text": "Chaque lundi…"}
        argv, kw = calls[0]
        assert argv[:3] == ["python3", "-m", "tiron_cron"]
        assert argv[3:] == ["preview", "--origin", "obsidian", "--args", 'add "0 8 * * 1" rappel x']

    def test_creation_et_gestion(self, client, monkeypatch):
        calls = self._fake(monkeypatch, '{"ok": true, "text": "ok"}')
        client.post("/run", json={"command": "/cron!", "arg": "add 20m rappel x"})
        client.post("/run", json={"command": "/cron", "arg": "list"})
        assert calls[0][0][3:5] == ["add", "--origin"] and calls[1][0][3:] == ["manage", "--args", "list"]

    def test_erreur_relayee(self, client, monkeypatch):
        self._fake(monkeypatch, '{"ok": false, "text": "contenu vide"}')
        r = client.post("/run", json={"command": "/cron!", "arg": "add 20m rappel"})
        assert r.status_code == 200 and r.get_json() == {"error": "contenu vide"}
```

- [ ] **Step 2: Vérifier l'échec** — `cd ~/Secretarius/Wiki_LM && .venv/bin/python -m pytest tests/test_server.py -k Cron -v` → FAIL.

- [ ] **Step 3: Implémenter côté serveur**

En tête de `Wiki_LM/tools/server.py`, s'assurer que `import subprocess` et `import json` sont présents (les ajouter sinon). Avant `_RUN_OPS` :
```python
def _tiron_cron(argv: list[str]) -> dict:
    """Délègue à la CLI JSON tiron_cron (même cœur que Tiron, cf. spec cron).
    Python système (stdlib seulement), cwd = racine du dépôt Secretarius."""
    cwd = os.environ.get("SECRETARIUS_DIR") or str(Path.home() / "Secretarius")
    p = subprocess.run(["python3", "-m", "tiron_cron", *argv], cwd=cwd,
                       capture_output=True, text=True, timeout=150)
    try:
        data = json.loads(p.stdout)
    except ValueError:
        return {"error": (p.stderr or p.stdout or "tiron_cron indisponible").strip()[:500]}
    return {"text": data.get("text", "")} if data.get("ok") else {"error": data.get("text", "")}
```
(Vérifier que `os` et `Path` sont importés ; les ajouter sinon.)

Dans `_RUN_OPS`, avant `"/help"` :
```python
    # Planification (spec cron 2026-10-01) : /cron? aperçu, /cron! création
    # (le ! tient lieu de /confirm), /cron list|rm|pause|reprendre.
    "/cron?": lambda arg, vault: _tiron_cron(["preview", "--origin", "obsidian", "--args", arg]),
    "/cron!": lambda arg, vault: _tiron_cron(["add", "--origin", "obsidian", "--args", arg]),
    "/cron": lambda arg, vault: _tiron_cron(["manage", "--args", arg]),
```

- [ ] **Step 4: Vérifier** — `.venv/bin/python -m pytest tests/test_server.py -v` → PASS.

- [ ] **Step 5: Tests plugin** (dans `run-commands.test.ts`)

```ts
describe("formatWikiResult — /cron", () => {
  it("liste les trois commandes cron", () => {
    expect(SUPPORTED_COMMANDS).toEqual(expect.arrayContaining(["/cron?", "/cron!", "/cron"]));
  });
  it("aperçu : invite à remplacer par /cron!", () => {
    expect(formatWikiResult("/cron?", { text: "Chaque lundi à 08:00 (Europe/Paris), obsidian, rappel : loyer" }, true))
      .toBe("Chaque lundi à 08:00 (Europe/Paris), obsidian, rappel : loyer\n\nRien n'a été créé. Remplacez par /cron! pour créer la tâche.");
  });
  it("création et gestion : texte relayé", () => {
    expect(formatWikiResult("/cron!", { text: "Tâche créée (aaaa1111) : …" }, true)).toBe("Tâche créée (aaaa1111) : …");
    expect(formatWikiResult("/cron", { text: "Aucune tâche planifiée." }, true)).toBe("Aucune tâche planifiée.");
  });
  it("erreur", () => {
    expect(formatWikiResult("/cron!", { error: "contenu vide" }, true)).toBe("**Erreur :** contenu vide");
  });
});
```

- [ ] **Step 6: Vérifier l'échec** — `cd ~/Secretarius/Wiki_LM/obsidian-wikilm-capture && npx vitest run src/run-commands.test.ts` → FAIL.

- [ ] **Step 7: Implémenter côté plugin**

Dans `SUPPORTED_COMMANDS`, avant `"/help"` : `"/cron?", "/cron!", "/cron",`.

Dans `formatWikiResult`, avant `case "/help":` :
```ts
    case "/cron?":
      return `${String(data.text ?? "")}\n\nRien n'a été créé. Remplacez par /cron! pour créer la tâche.`;
    case "/cron!":
    case "/cron":
      return String(data.text ?? "");
```

- [ ] **Step 8: Vérifier** — `npm test && npm run build` → PASS, build OK.

- [ ] **Step 9: Commit**

```bash
git add Wiki_LM/tools/server.py Wiki_LM/tests/test_server.py Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts Wiki_LM/obsidian-wikilm-capture/src/run-commands.test.ts
git commit -m "feat(obsidian): /cron?, /cron!, /cron via wiki-lm-server"
```

---

### Task 12: Corpus et GEPA — préparation (code seulement, aucun appel d'API)

**Files:**
- Modify: `gen_corpus/intentions.json`, `gen_corpus/seed.json`, `gen_corpus/prompt-init.txt`
- Modify: `gen_corpus/generate_corpus.py` (`INTENTION_DESCRIPTIONS`)
- Modify: `gen_corpus/promptGenGEPA.py` (`COMMANDS_KNOWN`, `INTENTION_DESCRIPTIONS`, `REQUIRES_ARGS`, valset séparé)
- Test: `gen_corpus/tests/test_generate.py` (ou nouveau `gen_corpus/tests/test_gepa_split.py`)

**Interfaces:**
- Produces: intention `cron_schedule` → commande `/cron` ; `split_trainset(examples: list, n_val: int, seed: int = 0) -> tuple[list, list]` dans `promptGenGEPA.py` ; option `--valset-size` (défaut 40).

- [ ] **Step 1: Test du découpage**

```python
# gen_corpus/tests/test_gepa_split.py
from gen_corpus.promptGenGEPA import split_trainset


def test_split_disjoint_taille_et_deterministe():
    ex = list(range(100))
    train, val = split_trainset(ex, 40, seed=0)
    assert len(val) == 40 and len(train) == 60 and set(train).isdisjoint(val)
    assert split_trainset(ex, 40, seed=0) == (train, val)


def test_split_borne_si_trop_peu_d_exemples():
    train, val = split_trainset(list(range(10)), 40)
    assert len(val) == 5 and len(train) == 5
```

Vérifier comment `test_generate.py` importe `promptGenGEPA` (import de `dspy` au chargement) et suivre le même chemin.

- [ ] **Step 2: Vérifier l'échec** — `cd ~/Secretarius && python3 -m pytest gen_corpus/tests/test_gepa_split.py -v` → FAIL.

- [ ] **Step 3: Implémenter**

`promptGenGEPA.py` :
```python
import random


def split_trainset(examples: list, n_val: int, seed: int = 0) -> tuple[list, list]:
    """Valset séparé et petit : GEPA dépense sinon son budget max_metric_calls à
    réévaluer tout le trainset au lieu de réfléchir (piste du 2026-09-30)."""
    n_val = min(n_val, len(examples) // 2)
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    val_idx = set(idx[:n_val])
    train = [e for i, e in enumerate(examples) if i not in val_idx]
    val = [e for i, e in enumerate(examples) if i in val_idx]
    return train, val
```
`parse_args` : `p.add_argument("--valset-size", type=int, default=40)` et champ `valset_size: int = 40` dans `Config` (transmis comme les autres). Dans `main` :
```python
    trainset, valset = split_trainset(build_trainset(seed), cfg.valset_size)
    ...
    compiled = teleprompter.compile(generator, trainset=trainset, valset=valset)
```
Ajouter `"/cron"` à `COMMANDS_KNOWN`, `"cron_schedule"` à `REQUIRES_ARGS`, et à `INTENTION_DESCRIPTIONS` (dans **les deux** fichiers `promptGenGEPA.py` et `generate_corpus.py`) :
```python
    "cron_schedule": "planifier un rappel, une commande Tiron ou une consigne à une date donnée ou de façon récurrente (chaque lundi, tous les jours, demain, dans 20 minutes), éventuellement sur un canal précis (SMS, WhatsApp, Telegram, Obsidian)",
```

`intentions.json` : ajouter
```json
{"intention": "cron_schedule", "command": "/cron", "variantes": ["rappel_recurrent", "rappel_ponctuel_date", "rappel_relatif", "commande_recurrente", "consigne_recurrente", "avec_canal"]}
```

`prompt-init.txt` : ajouter dans la liste des commandes
```
- /cron add <quand> [<canal>] <type> <contenu> — planifier un rappel, une commande ou une consigne (récurrent ou ponctuel), éventuellement sur un canal (sms, whatsapp, telegram, obsidian)
```
et dans « Règles de génération » :
```
- Le message doit être bref : une phrase, comme un vrai message tapé sur un téléphone (en général moins de 20 mots). Pas de formule de politesse développée, même en registre formel ou poli.
- Varier les tournures d'un exemple à l'autre : ne pas commencer systématiquement par « Bonjour » ni par le nom de l'assistant.
```

`seed.json` : ajouter 12 exemples `cron_schedule` (2 par variante, registres variés), au format existant, par exemple :
```json
{"text": "rappelle-moi chaque lundi à 8h de payer le loyer", "intention": "cron_schedule", "registre": "familier", "variante": "rappel_recurrent", "action": {"command": "/cron", "args": "add \"0 8 * * 1\" rappel payer le loyer"}},
{"text": "tous les jours 7h30 : rappel vitamines", "intention": "cron_schedule", "registre": "télégraphique", "variante": "rappel_recurrent", "action": {"command": "/cron", "args": "add \"30 7 * * *\" rappel vitamines"}},
{"text": "Pourriez-vous me rappeler le 15 octobre à 14h la réunion budget ?", "intention": "cron_schedule", "registre": "formel", "variante": "rappel_ponctuel_date", "action": {"command": "/cron", "args": "add 2026-10-15T14:00 rappel la réunion budget"}},
{"text": "demain 9h rappelle garagiste", "intention": "cron_schedule", "registre": "abrégé", "variante": "rappel_ponctuel_date", "action": {"command": "/cron", "args": "add 2026-10-03T09:00 rappel appeler le garagiste"}},
{"text": "dans 20 min rappelle-moi de sortir le gâteau", "intention": "cron_schedule", "registre": "familier", "variante": "rappel_relatif", "action": {"command": "/cron", "args": "add 20m rappel sortir le gâteau"}},
{"text": "Merci de me faire penser dans deux heures à rappeler Paul.", "intention": "cron_schedule", "registre": "poli", "variante": "rappel_relatif", "action": {"command": "/cron", "args": "add 2h rappel rappeler Paul"}},
{"text": "chaque dimanche soir à 18h donne-moi l'état du wiki", "intention": "cron_schedule", "registre": "familier", "variante": "commande_recurrente", "action": {"command": "/cron", "args": "add \"0 18 * * 0\" commande /wikistatus"}},
{"text": "lance l'ingestion du wiki tous les jours à 3h", "intention": "cron_schedule", "registre": "télégraphique", "variante": "commande_recurrente", "action": {"command": "/cron", "args": "add \"0 3 * * *\" commande /ingest"}},
{"text": "Chaque vendredi à 17h, faites-moi une synthèse des nouveautés du wiki.", "intention": "cron_schedule", "registre": "formel", "variante": "consigne_recurrente", "action": {"command": "/cron", "args": "add \"0 17 * * 5\" consigne synthèse des nouveautés du wiki"}},
{"text": "tous les matins à 8h résume-moi ce qui a été ajouté au wiki", "intention": "cron_schedule", "registre": "familier", "variante": "consigne_recurrente", "action": {"command": "/cron", "args": "add \"0 8 * * *\" consigne résume ce qui a été ajouté au wiki"}},
{"text": "envoie-moi un SMS chaque lundi à 8h pour le loyer", "intention": "cron_schedule", "registre": "familier", "variante": "avec_canal", "action": {"command": "/cron", "args": "add \"0 8 * * 1\" sms rappel le loyer"}},
{"text": "rappel whatsapp demain 10h dentiste", "intention": "cron_schedule", "registre": "abrégé", "variante": "avec_canal", "action": {"command": "/cron", "args": "add 2026-10-03T10:00 whatsapp rappel dentiste"}}
```

- [ ] **Step 4: Vérifier**

Run: `python3 -m pytest gen_corpus -v` → PASS (tests existants compris).
Run: `python3 -c "import json; [json.load(open(f'gen_corpus/{f}')) for f in ('intentions.json','seed.json')]"` → aucune erreur.

- [ ] **Step 5: Commit**

```bash
git add gen_corpus/
git commit -m "feat(corpus): intention cron_schedule, valset GEPA séparé, prompt plus bref"
```

---

### Task 13: Laya — génération, réentraînement, bascule (**étapes manuelles, accord requis**)

Chaque step marqué **[accord]** attend un « oui » explicite de l'utilisateur : dépense DeepSeek, Kaggle, redémarrage de service.

**Files:**
- Modify: `router_service/router.py` (`COMMAND_CRITERIA`)
- Generated: `gen_corpus/GEPAPrompt.txt`, `gen_corpus/corpus.jsonl`, `corpus_lora*.jsonl`, `laya_train.jsonl`, `laya_val.jsonl`

- [ ] **Step 1: [accord] Relancer GEPA**

```bash
cd ~/Secretarius/gen_corpus && python3 promptGenGEPA.py --valset-size 40
```
Vérifier que `GEPAPrompt.txt` contient la ligne `/cron` et les règles de brièveté. Lire 10 lignes de `gepa_llm_calls.log` pour confirmer que les itérations de réflexion ont bien eu lieu.

- [ ] **Step 2: [accord] Générer ~250 exemples `cron_schedule`**

```bash
python3 -c "import json; d=json.load(open('intentions.json')); json.dump([i for i in d if i['intention']=='cron_schedule'], open('intentions_cron.json','w'), ensure_ascii=False)"
python3 generate_corpus.py --intentions intentions_cron.json --count 250 --output corpus_cron.jsonl
cp corpus.jsonl corpus.jsonl.avant-cron.bak
cat corpus_cron.jsonl >> corpus.jsonl
rm intentions_cron.json
python3 inspect_corpus.py corpus.jsonl | head -40
```
Relire 20 exemples `cron_schedule` tirés au hasard : brefs, variés, bien étiquetés.

- [ ] **Step 3: Ajouter `/cron` aux critères et reformater**

`router_service/router.py`, dans `COMMAND_CRITERIA` avant `"aucune"` :
```python
    "/cron": "planifier un rappel, une commande ou une consigne, à une date ou de façon récurrente",
```
Puis :
```bash
cd ~/Secretarius && python3 gen_corpus/to_lora_format.py && python3 gen_corpus/to_laya_format.py
grep -c '"command": "/cron"' gen_corpus/laya_val.jsonl
```
Expected : `train=… val=…`, et au moins 25 exemples `/cron` dans le val.

- [ ] **Step 4: [accord] Réentraîner sur Kaggle**

Notebook `laya_finetune_typed_decisions_2xT4_kaggle.ipynb` (2× T4), avec les nouveaux `laya_train.jsonl`/`laya_val.jsonl`. Pousser le checkpoint sur `Mauceric/laya-tiron-router` (ou un nouveau nom, à décider avec l'utilisateur pour garder l'ancien en repli).

- [ ] **Step 5: Évaluer**

```bash
cd ~/Secretarius && python3 -m router_service.eval_laya Mauceric/laya-tiron-router --val gen_corpus/laya_val.jsonl
```
Expected : `accuracy ≥ 0.90` global, et `/cron` ≥ 0.90 (compter à part : filtrer `laya_val.jsonl` sur `/cron` dans un fichier temporaire et relancer). Sinon : arrêter et analyser les erreurs avec l'utilisateur.

- [ ] **Step 6: [accord] Basculer**

```bash
systemctl --user restart tiron-router.service
curl -s localhost:8999/route -d '{"message":"rappelle-moi chaque lundi à 8h de payer le loyer"}'
```
Expected : `{"status": "ok", "command": "/cron", "args": "rappelle-moi chaque lundi à 8h de payer le loyer"}`.

- [ ] **Step 7: Commit**

```bash
git add router_service/router.py gen_corpus/GEPAPrompt.txt gen_corpus/corpus.jsonl gen_corpus/corpus_lora*.jsonl gen_corpus/laya_*.jsonl
git commit -m "feat(routeur): Laya reconnaît /cron (corpus + GEPA + réentraînement)"
```

---

### Task 14: Canaux et déploiement (**accord requis pour les redémarrages**)

- [ ] **Step 1: Destinataires**

Dans `~/.openclaw/gateway.systemd.env` (et le modèle `openclaw-config/gateway.systemd.env.template`, avec des valeurs vides commentées) :
```
TIRON_TELEGRAM_CHAT_ID=<id trouvé à la tâche 1>
TIRON_SMS_TO=<numéro de l'utilisateur, E.164>
TIRON_WHATSAPP_TO=<numéro, E.164>
TIRON_RAPPELS_DIR=/home/mauceric/Documents/Arbath/Rappels
SECRETARIUS_DIR=/home/mauceric/Secretarius
```
Demander à l'utilisateur ses numéros ; ne jamais les inventer. Ajouter `SECRETARIUS_DIR` aussi dans `Wiki_LM/.env`.

- [ ] **Step 2: Extensions WhatsApp et SMS**

Selon les résultats de la tâche 1 : `openclaw channels add …` / `openclaw channels login …` pour WhatsApp ; configuration Twilio (SID, jeton, numéro émetteur) pour SMS, secrets dans `~/.config/secrets.env` puis `/maj` (chaîne de propagation existante). Vérifier : `openclaw channels status`.

- [ ] **Step 3: [accord] Déployer**

```bash
cd ~/Secretarius/derisk-deleg && npm run build
```
Réinstaller derisk-deleg dans `~/.openclaw/extensions/` selon la procédure habituelle (copie séparée du dépôt, cf. mémoire du 2026-09-08), puis :
```bash
systemctl --user restart openclaw-gateway.service wiki-lm-server.service
```
Plugin Obsidian : `cd Wiki_LM/obsidian-wikilm-capture && npm run build`, copier `main.js`/`manifest.json` dans le coffre selon la procédure existante.

- [ ] **Step 4: Commit**

```bash
git add openclaw-config/gateway.systemd.env.template
git commit -m "chore(cron): variables de destinataires dans le modèle d'environnement"
```

---

### Task 15: Vérification de bout en bout (manuelle, sur sanroque)

- [ ] Telegram, tapé : `/cron add 2m rappel essai telegram` → aperçu → `/confirm` → reçu après 2 min.
- [ ] Telegram, langage naturel : « dans 2 minutes rappelle-moi de tester par SMS » → aperçu (canal sms) → `/confirm` → SMS reçu.
- [ ] WhatsApp : `/cron add 2m whatsapp rappel essai whatsapp` → reçu.
- [ ] Commande : `/cron add 2m commande /wikistatus` → l'état du wiki arrive sur Telegram.
- [ ] Consigne : `/cron add 2m consigne donne-moi une citation de Sénèque` → réponse de `main` reçue.
- [ ] Obsidian : bloc `/cron?` puis `/cron!` avec `add 2m rappel essai obsidian` → note créée dans `TIRON_RAPPELS_DIR`, visible après synchronisation.
- [ ] `/cron list` → les tâches récurrentes éventuelles, aucune tâche système. `/cron rm <id8>` → supprimée.
- [ ] Repli : « lundi prochain vers 8 heures rappelle-moi x » → aperçu (`source: modele` visible dans les journaux), texte resté local.
- [ ] Mettre à jour `docs/components/` (fiche `cron.md` : syntaxe, variables, canaux) et la FAQ `amorçage/faits.md` (une ligne : Tiron sait planifier). Commit.

---

## Résultats de l'essai préalable

(À remplir pendant la tâche 1.)

- Step 1 (`--command-argv` + `--announce`) :
- Step 2 (hook `before_agent_reply` sur message planifié) :
- Step 3 (identifiants de canal, SMS/Twilio, WhatsApp) :
