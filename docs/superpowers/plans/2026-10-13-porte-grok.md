# Porte Grok — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exposer le wiki et le coffre `Publication` de santiago aux agents Grok Bot par un serveur MCP authentifié (jeton Bearer par agent) publié via Tailscale Funnel, et laisser l'utilisateur valider la veille par cases à cocher dans Obsidian.

**Architecture:** Un paquet Python `Wiki_LM/porte/` (jetons, confinement du coffre, lecture du wiki via `wiki-lm-server`, serveur `fastmcp` en HTTP « streamable » sans état) tourne en unité systemd utilisateur sur `127.0.0.1:5052` ; Funnel le publie sur 8443. Le plugin Obsidian `wikilm-capture` ignore les blocs ` ```wiki ` précédés d'une case vide. Un module `porte` de l'installateur pose le tout sur le profil léger.

**Tech Stack:** Python 3.11+ (fastmcp 3.x, starlette, uvicorn, urllib), pytest ; TypeScript + vitest (plugin) ; bash (installateur).

**Spec:** `docs/superpowers/specs/2026-10-13-porte-grok-design.md`

## Global Constraints

- Le serveur écoute **uniquement** sur `127.0.0.1:5052` ; chemin MCP `/mcp` ; `/health` sans jeton renvoie `{"status": "ok"}` et rien d'autre.
- Jetons : fichier `~/.config/porte-grok/agents` (600, dossier 700), lignes `<préfixe>:<sha256 hex>` ; jamais de jeton en clair stocké ; préfixe `^[a-z][a-z0-9-]{1,30}$` ; comparaison `hmac.compare_digest`.
- Confinement : chemin relatif, résolu dans le coffre ; refus de `Wiki_LM/` et de tout composant commençant par `.` ; `.md` seulement ; contenu écrit ≤ 1 000 000 octets UTF-8 ; écriture seulement si le nom de fichier commence par `<préfixe>-` ; ni suppression ni renommage.
- `wiki_question` : `POST /query` de `wiki-lm-server`, verrou (une question à la fois), délai 360 s. `wiki_recherche` : `POST /run` `{"command": "/r", "arg": …}` uniquement. Aucune autre commande `/run`.
- Journal : une ligne par appel (agent, outil, chemin/slug ou question tronquée à 80 caractères, ok/motif) ; jamais le contenu des fichiers.
- Erreurs vers l'agent : `ToolError` avec message court ; jamais de trace Python.
- La porte ne va jamais chercher d'URL (contrainte « le wiki ne court-circuite pas Scout »).
- Installateur : tout test qui lance un script de l'installateur le fait avec un `HOME` temporaire (incident sanroque du 2026-10-12).
- Commentaires et messages en français, style du dépôt.

## Review Focus

1. **Lien symbolique dans le coffre pointant hors du coffre** (fichier ou dossier) : `coffre_lire`/`coffre_ecrire`/`coffre_lister` doivent refuser ou ignorer, jamais lire ni écrire dehors — test dans la tâche 2.
2. **`coffre_ecrire` vers un dossier interdit avec un nom bien préfixé** (`Wiki_LM/raw/veille-x.md`, `.obsidian/veille-x.md`) : refus malgré le préfixe correct — test dans la tâche 2.
3. **En-tête `Authorization` mal formé** (`Bearer` sans jeton, `bearer` minuscule, jeton d'un agent retiré entre deux appels) : 401, et révocation effective sans redémarrage — tests dans les tâches 1 et 4.
4. **`wiki-lm-server` arrêté ou renvoyant `{"error": …}`** : message « wiki injoignable » ou l'erreur du wiki, pas de trace, et le verrou de `wiki_question` est relâché — tests dans la tâche 3.
5. **Note avec cases et blocs mêlés au texte libre** (case vide devant un bloc, case cochée plus loin devant un autre, bloc sans case) : seuls les bons blocs exécutés, et les positions `start`/`end` des blocs retenus restent exactes pour l'écriture des résultats — test dans la tâche 5.

---

### Task 1: Jetons des agents et commande `porte-jeton`

**Files:**
- Create: `Wiki_LM/porte/__init__.py`
- Create: `Wiki_LM/porte/jetons.py`
- Modify: `Wiki_LM/tests/conftest.py` (rendre `Wiki_LM/` importable)
- Test: `Wiki_LM/tests/test_porte_jetons.py`

**Interfaces:**
- Produces: `jetons.FICHIER_DEFAUT: Path` ; `jetons.empreinte(jeton: str) -> str` ; `jetons.agent_pour(fichier: Path, en_tete: str | None) -> str | None` ; `jetons.ajouter(fichier: Path, prefixe: str) -> str` (lève `ValueError`) ; `jetons.retirer(fichier: Path, prefixe: str) -> bool` ; `jetons.lister(fichier: Path) -> list[str]` ; `jetons.main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Rendre `Wiki_LM/` importable dans les tests**

Dans `Wiki_LM/tests/conftest.py`, juste après la ligne existante `sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))`, ajouter :

```python
# Rendre le paquet porte/ importable (porte.jetons, porte.coffre…)
sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 2: Écrire les tests (échouent)**

`Wiki_LM/tests/test_porte_jetons.py` :

```python
"""Jetons des agents de la porte Grok."""
import stat

import pytest

from porte import jetons


def test_ajouter_stocke_empreinte_seulement(tmp_path):
    f = tmp_path / "porte" / "agents"
    jeton = jetons.ajouter(f, "veille")
    contenu = f.read_text()
    assert jeton not in contenu
    assert contenu == f"veille:{jetons.empreinte(jeton)}\n"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert stat.S_IMODE(f.parent.stat().st_mode) == 0o700


def test_agent_pour_jeton_valide(tmp_path):
    f = tmp_path / "agents"
    jeton = jetons.ajouter(f, "veille")
    jetons.ajouter(f, "redaction")
    assert jetons.agent_pour(f, f"Bearer {jeton}") == "veille"


@pytest.mark.parametrize("en_tete", [None, "", "Bearer", "Bearer ", "bearer x", "Basic abc", "Bearer faux"])
def test_agent_pour_refuse(tmp_path, en_tete):
    f = tmp_path / "agents"
    jetons.ajouter(f, "veille")
    assert jetons.agent_pour(f, en_tete) is None


def test_fichier_absent_refuse_tout(tmp_path):
    assert jetons.agent_pour(tmp_path / "absent", "Bearer x") is None


def test_retrait_effectif_immediatement(tmp_path):
    f = tmp_path / "agents"
    jeton = jetons.ajouter(f, "veille")
    assert jetons.retirer(f, "veille") is True
    assert jetons.agent_pour(f, f"Bearer {jeton}") is None
    assert jetons.retirer(f, "veille") is False


def test_doublon_et_prefixe_invalide_refuses(tmp_path):
    f = tmp_path / "agents"
    jetons.ajouter(f, "veille")
    with pytest.raises(ValueError):
        jetons.ajouter(f, "veille")
    for mauvais in ["Veille", "1veille", "v", "veille_x", "a" * 40]:
        with pytest.raises(ValueError):
            jetons.ajouter(f, mauvais)


def test_lister(tmp_path):
    f = tmp_path / "agents"
    jetons.ajouter(f, "veille")
    jetons.ajouter(f, "redaction")
    assert jetons.lister(f) == ["veille", "redaction"]


def test_cli_ajouter_affiche_le_jeton_une_fois(tmp_path, capsys):
    f = tmp_path / "agents"
    assert jetons.main(["--fichier", str(f), "ajouter", "veille"]) == 0
    jeton = capsys.readouterr().out.strip().splitlines()[-1]
    assert jetons.agent_pour(f, f"Bearer {jeton}") == "veille"
    assert jetons.main(["--fichier", str(f), "ajouter", "veille"]) == 1
    assert jetons.main(["--fichier", str(f), "lister"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "veille"
    assert jetons.main(["--fichier", str(f), "retirer", "veille"]) == 0
    assert jetons.main(["--fichier", str(f), "retirer", "veille"]) == 1
```

- [ ] **Step 3: Lancer, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/python -m pytest tests/test_porte_jetons.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'porte'`)

- [ ] **Step 4: Implémenter**

`Wiki_LM/porte/__init__.py` :

```python
"""Porte Grok : accès MCP public (Funnel) au wiki et au coffre de santiago."""
```

`Wiki_LM/porte/jetons.py` :

```python
"""Jetons des agents de la porte Grok (spec 2026-10-13).

Une ligne « préfixe:empreinte SHA-256 » par agent ; le jeton en clair n'est
jamais stocké. Le fichier est relu à chaque requête : un retrait prend effet
immédiatement, sans redémarrer le serveur.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import secrets
import sys
from pathlib import Path

PREFIXE_RE = re.compile(r"^[a-z][a-z0-9-]{1,30}$")
FICHIER_DEFAUT = Path.home() / ".config" / "porte-grok" / "agents"


def empreinte(jeton: str) -> str:
    return hashlib.sha256(jeton.encode("utf-8")).hexdigest()


def _lire(fichier: Path) -> list[tuple[str, str]]:
    if not fichier.exists():
        return []
    lignes = []
    for ligne in fichier.read_text(encoding="utf-8").splitlines():
        prefixe, sep, emp = ligne.strip().partition(":")
        if sep and prefixe and emp:
            lignes.append((prefixe, emp))
    return lignes


def _ecrire(fichier: Path, lignes: list[tuple[str, str]]) -> None:
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.parent.chmod(0o700)
    fd = os.open(fichier, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("".join(f"{p}:{e}\n" for p, e in lignes))
    fichier.chmod(0o600)


def agent_pour(fichier: Path, en_tete: str | None) -> str | None:
    """Préfixe de l'agent dont le jeton est présenté, ou None."""
    if not en_tete or not en_tete.startswith("Bearer "):
        return None
    jeton = en_tete[len("Bearer "):].strip()
    if not jeton:
        return None
    emp = empreinte(jeton)
    trouve = None
    for prefixe, attendu in _lire(fichier):
        if hmac.compare_digest(emp, attendu):
            trouve = prefixe
    return trouve


def ajouter(fichier: Path, prefixe: str) -> str:
    if not PREFIXE_RE.match(prefixe):
        raise ValueError(f"préfixe invalide : {prefixe!r} (attendu : {PREFIXE_RE.pattern})")
    lignes = _lire(fichier)
    if any(p == prefixe for p, _ in lignes):
        raise ValueError(f"agent déjà présent : {prefixe}")
    jeton = secrets.token_urlsafe(32)
    _ecrire(fichier, lignes + [(prefixe, empreinte(jeton))])
    return jeton


def retirer(fichier: Path, prefixe: str) -> bool:
    lignes = _lire(fichier)
    restantes = [(p, e) for p, e in lignes if p != prefixe]
    if len(restantes) == len(lignes):
        return False
    _ecrire(fichier, restantes)
    return True


def lister(fichier: Path) -> list[str]:
    return [p for p, _ in _lire(fichier)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="porte-jeton", description="Jetons des agents de la porte Grok")
    parser.add_argument("--fichier", default=str(FICHIER_DEFAUT))
    sous = parser.add_subparsers(dest="action", required=True)
    for action in ("ajouter", "retirer"):
        sous.add_parser(action).add_argument("prefixe")
    sous.add_parser("lister")
    args = parser.parse_args(argv)
    fichier = Path(args.fichier)
    if args.action == "ajouter":
        try:
            jeton = ajouter(fichier, args.prefixe)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        print(f"Jeton de l'agent « {args.prefixe} » (affiché une seule fois) :")
        print(jeton)
        return 0
    if args.action == "retirer":
        if retirer(fichier, args.prefixe):
            print(f"Agent « {args.prefixe} » retiré.")
            return 0
        print(f"Agent inconnu : {args.prefixe}", file=sys.stderr)
        return 1
    for prefixe in lister(fichier):
        print(prefixe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Lancer, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/python -m pytest tests/test_porte_jetons.py -q`
Expected: PASS (tous)

- [ ] **Step 6: Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/porte/__init__.py Wiki_LM/porte/jetons.py Wiki_LM/tests/conftest.py Wiki_LM/tests/test_porte_jetons.py
git commit -m "feat(porte): jetons des agents (empreintes, porte-jeton)"
```

---

### Task 2: Confinement des fichiers du coffre

**Files:**
- Create: `Wiki_LM/porte/coffre.py`
- Test: `Wiki_LM/tests/test_porte_coffre.py`

**Interfaces:**
- Produces: `coffre.Refus(Exception)` ; `coffre.TAILLE_MAX = 1_000_000` ; `coffre.resoudre(racine: Path, chemin: str) -> Path` ; `coffre.lister(racine: Path, dossier: str = "") -> list[str]` (chemins relatifs POSIX triés) ; `coffre.lire(racine: Path, chemin: str) -> str` ; `coffre.ecrire(racine: Path, prefixe: str, chemin: str, contenu: str) -> str` (chemin relatif écrit).

- [ ] **Step 1: Écrire les tests (échouent)**

`Wiki_LM/tests/test_porte_coffre.py` :

```python
"""Confinement des fichiers du coffre pour la porte Grok."""
import pytest

from porte import coffre
from porte.coffre import Refus


@pytest.fixture
def racine(tmp_path):
    r = tmp_path / "Publication"
    (r / "Brouillons").mkdir(parents=True)
    (r / "Brouillons" / "veille-a.md").write_text("A")
    (r / "Notes.md").write_text("perso")
    (r / "Wiki_LM" / "raw").mkdir(parents=True)
    (r / "Wiki_LM" / "wiki.md").write_text("w")
    (r / ".obsidian").mkdir()
    (r / ".obsidian" / "app.md").write_text("cfg")
    (r / "image.png").write_bytes(b"x")
    return r


@pytest.mark.parametrize("chemin", [
    "/etc/passwd", "../dehors.md", "Brouillons/../../dehors.md", "",
    "Wiki_LM/wiki.md", "Wiki_LM/raw/veille-x.md", ".obsidian/app.md", "Brouillons/.cache/veille-x.md",
])
def test_resoudre_refuse(racine, chemin):
    with pytest.raises(Refus):
        coffre.resoudre(racine, chemin)


def test_lien_symbolique_sortant_refuse(racine, tmp_path):
    (tmp_path / "secret.md").write_text("secret")
    (racine / "lien.md").symlink_to(tmp_path / "secret.md")
    (tmp_path / "ailleurs").mkdir()
    (racine / "dossier-lien").symlink_to(tmp_path / "ailleurs", target_is_directory=True)
    with pytest.raises(Refus):
        coffre.lire(racine, "lien.md")
    with pytest.raises(Refus):
        coffre.ecrire(racine, "veille", "dossier-lien/veille-x.md", "x")
    assert "lien.md" not in coffre.lister(racine)
    assert not (tmp_path / "ailleurs" / "veille-x.md").exists()


def test_lister_hors_zones_interdites(racine):
    assert coffre.lister(racine) == ["Brouillons/veille-a.md", "Notes.md"]
    assert coffre.lister(racine, "Brouillons") == ["Brouillons/veille-a.md"]


def test_lister_dossier_inexistant_refuse(racine):
    with pytest.raises(Refus):
        coffre.lister(racine, "Absent")


def test_lire(racine):
    assert coffre.lire(racine, "Notes.md") == "perso"
    with pytest.raises(Refus):
        coffre.lire(racine, "image.png")
    with pytest.raises(Refus):
        coffre.lire(racine, "Absent.md")


def test_ecrire_prefixe_impose(racine):
    with pytest.raises(Refus, match="veille-"):
        coffre.ecrire(racine, "veille", "Notes.md", "écrasé")
    assert (racine / "Notes.md").read_text() == "perso"
    assert coffre.ecrire(racine, "veille", "Brouillons/Veille/veille-2026-10-13.md", "v1") \
        == "Brouillons/Veille/veille-2026-10-13.md"
    coffre.ecrire(racine, "veille", "Brouillons/Veille/veille-2026-10-13.md", "v2")
    assert (racine / "Brouillons/Veille/veille-2026-10-13.md").read_text() == "v2"


def test_ecrire_extension_et_taille(racine):
    with pytest.raises(Refus):
        coffre.ecrire(racine, "veille", "veille-x.txt", "x")
    with pytest.raises(Refus):
        coffre.ecrire(racine, "veille", "veille-x.md", "é" * (coffre.TAILLE_MAX // 2 + 1))
```

- [ ] **Step 2: Lancer, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/python -m pytest tests/test_porte_coffre.py -q`
Expected: FAIL (`ImportError: cannot import name 'coffre'`)

- [ ] **Step 3: Implémenter**

`Wiki_LM/porte/coffre.py` :

```python
"""Confinement des fichiers du coffre pour la porte Grok (spec 2026-10-13).

Chemins relatifs au coffre, résolus (liens symboliques suivis) puis vérifiés :
jamais hors du coffre, jamais dans Wiki_LM/ ni dans un dossier caché
(.obsidian/…), .md seulement. L'écriture exige le préfixe de l'agent.
"""
from __future__ import annotations

from pathlib import Path

TAILLE_MAX = 1_000_000
ZONES_INTERDITES = {"Wiki_LM"}


class Refus(Exception):
    """Demande refusée ; le message est renvoyé tel quel à l'agent."""


def resoudre(racine: Path, chemin: str) -> Path:
    if not chemin or Path(chemin).is_absolute():
        raise Refus("chemin relatif au coffre attendu")
    base = racine.resolve()
    cible = (base / chemin).resolve()
    if cible == base or not cible.is_relative_to(base):
        raise Refus("chemin hors du coffre")
    parties = cible.relative_to(base).parts
    if parties[0] in ZONES_INTERDITES or any(p.startswith(".") for p in parties):
        raise Refus("zone interdite : Wiki_LM/ et dossiers cachés")
    return cible


def _exiger_md(cible: Path) -> None:
    if cible.suffix != ".md":
        raise Refus("seuls les fichiers .md sont permis")


def lister(racine: Path, dossier: str = "") -> list[str]:
    base = racine.resolve()
    depart = resoudre(racine, dossier) if dossier else base
    if not depart.is_dir():
        raise Refus(f"dossier introuvable : {dossier}")
    trouves = []
    for f in depart.rglob("*.md"):
        rel = f.relative_to(base).as_posix()
        try:
            if resoudre(racine, rel).is_file():
                trouves.append(rel)
        except Refus:
            continue
    return sorted(trouves)


def lire(racine: Path, chemin: str) -> str:
    cible = resoudre(racine, chemin)
    _exiger_md(cible)
    if not cible.is_file():
        raise Refus(f"fichier introuvable : {chemin}")
    return cible.read_text(encoding="utf-8")


def ecrire(racine: Path, prefixe: str, chemin: str, contenu: str) -> str:
    cible = resoudre(racine, chemin)
    _exiger_md(cible)
    if not cible.name.startswith(f"{prefixe}-"):
        raise Refus(f"le nom du fichier doit commencer par {prefixe}-")
    if len(contenu.encode("utf-8")) > TAILLE_MAX:
        raise Refus(f"contenu trop long (maximum {TAILLE_MAX} octets)")
    cible.parent.mkdir(parents=True, exist_ok=True)
    cible.write_text(contenu, encoding="utf-8")
    return cible.relative_to(racine.resolve()).as_posix()
```

Note : dans `lister`, un lien symbolique `lien.md` vers l'extérieur : `rel` vaut `lien.md`, `resoudre` le résout hors du coffre → `Refus` → ignoré. Un dossier lié vers l'extérieur n'est pas parcouru par `rglob` (Python ne suit pas les liens de dossiers par défaut).

- [ ] **Step 4: Lancer, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/python -m pytest tests/test_porte_coffre.py -q`
Expected: PASS (tous)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/porte/coffre.py Wiki_LM/tests/test_porte_coffre.py
git commit -m "feat(porte): confinement des fichiers du coffre (préfixe imposé à l'écriture)"
```

---

### Task 3: Lecture du wiki (question, recherche, page)

**Files:**
- Create: `Wiki_LM/porte/lecture_wiki.py`
- Test: `Wiki_LM/tests/test_porte_lecture_wiki.py`

**Interfaces:**
- Consumes: `coffre.Refus` (Task 2).
- Produces: `lecture_wiki.WikiIndisponible(Exception)` ; `lecture_wiki.SLUG_RE` ; `lecture_wiki.question(url: str, texte: str, delai: float = 360) -> dict` (`{"synthese": str, "references": list}`) ; `lecture_wiki.recherche(url: str, mots: str, delai: float = 30) -> list[dict]` ; `lecture_wiki.page(wiki_dir: Path, slug: str) -> str` (lève `Refus`).

- [ ] **Step 1: Écrire les tests (échouent)**

`Wiki_LM/tests/test_porte_lecture_wiki.py` :

```python
"""Lecture du wiki par la porte Grok, contre un faux wiki-lm-server."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from porte import lecture_wiki
from porte.coffre import Refus
from porte.lecture_wiki import WikiIndisponible


class _Faux(BaseHTTPRequestHandler):
    reponses = {}
    en_cours = 0
    max_simultanes = 0
    recus = []

    def do_POST(self):
        corps = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).recus.append((self.path, corps))
        type(self).en_cours += 1
        type(self).max_simultanes = max(type(self).max_simultanes, type(self).en_cours)
        time.sleep(0.2 if self.path == "/query" else 0)
        type(self).en_cours -= 1
        statut, data = type(self).reponses[self.path]
        self.send_response(statut)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def faux_wiki():
    _Faux.reponses = {
        "/query": (200, {"text": "Synthèse.", "references": ["src-a", "c-b"], "brief": "b"}),
        "/run": (200, {"results": [{"title": "A", "excerpt": "ex"}]}),
    }
    _Faux.en_cours = _Faux.max_simultanes = 0
    _Faux.recus = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Faux)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_question(faux_wiki):
    assert lecture_wiki.question(faux_wiki, "Substack ?") == {"synthese": "Synthèse.", "references": ["src-a", "c-b"]}
    assert _Faux.recus == [("/query", {"question": "Substack ?"})]


def test_questions_une_a_la_fois(faux_wiki):
    fils = [threading.Thread(target=lecture_wiki.question, args=(faux_wiki, f"q{i}")) for i in range(3)]
    for f in fils:
        f.start()
    for f in fils:
        f.join()
    assert _Faux.max_simultanes == 1


def test_question_erreur_du_wiki(faux_wiki):
    _Faux.reponses["/query"] = (500, {"error": "KB vide — lancer ingest d'abord"})
    with pytest.raises(WikiIndisponible, match="KB vide"):
        lecture_wiki.question(faux_wiki, "x")
    # le verrou est relâché : une nouvelle question passe
    _Faux.reponses["/query"] = (200, {"text": "ok", "references": []})
    assert lecture_wiki.question(faux_wiki, "y")["synthese"] == "ok"


def test_wiki_injoignable():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(WikiIndisponible, match="injoignable"):
        lecture_wiki.question(f"http://127.0.0.1:{port}", "x", delai=2)


def test_recherche_utilise_seulement_r(faux_wiki):
    assert lecture_wiki.recherche(faux_wiki, "écriture IA") == [{"title": "A", "excerpt": "ex"}]
    assert _Faux.recus == [("/run", {"command": "/r", "arg": "écriture IA"})]


def test_page(tmp_path):
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "sources" / "src-a.md").write_text("contenu A")
    (wiki / "poubelle").mkdir()
    (wiki / "poubelle" / "src-vieux.md").write_text("jeté")
    assert lecture_wiki.page(wiki, "src-a") == "contenu A"
    for slug in ["src-vieux", "src-absent", "../x", "SRC-a", "x-a", "src-a.md"]:
        with pytest.raises(Refus):
            lecture_wiki.page(wiki, slug)
```

- [ ] **Step 2: Lancer, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/python -m pytest tests/test_porte_lecture_wiki.py -q`
Expected: FAIL (`ImportError: cannot import name 'lecture_wiki'`)

- [ ] **Step 3: Implémenter**

`Wiki_LM/porte/lecture_wiki.py` :

```python
"""Lecture du wiki pour la porte Grok (spec 2026-10-13) : via wiki-lm-server
local (/query, /run /r seulement) et lecture directe des pages par slug."""
from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

from porte.coffre import Refus

SLUG_RE = re.compile(r"^(src|c|e)-[a-z0-9-]+$")
# Une question à la fois : chaque /query sollicite le modèle (Modal), et
# l'ingestion comme les questions partagent ce même modèle.
_VERROU_QUESTION = threading.Lock()


class WikiIndisponible(Exception):
    """wiki-lm-server injoignable ou en erreur ; message renvoyé à l'agent."""


def _post(url: str, chemin: str, corps: dict, delai: float) -> dict:
    req = urllib.request.Request(
        url + chemin,
        data=json.dumps(corps).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=delai) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read())
        except ValueError:
            raise WikiIndisponible(f"wiki en erreur ({e.code})") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise WikiIndisponible("wiki injoignable") from e
    if isinstance(data, dict) and data.get("error"):
        raise WikiIndisponible(str(data["error"]))
    return data


def question(url: str, texte: str, delai: float = 360) -> dict:
    with _VERROU_QUESTION:
        data = _post(url, "/query", {"question": texte}, delai)
    return {"synthese": data.get("text", ""), "references": data.get("references", [])}


def recherche(url: str, mots: str, delai: float = 30) -> list[dict]:
    return _post(url, "/run", {"command": "/r", "arg": mots}, delai).get("results", [])


def page(wiki_dir: Path, slug: str) -> str:
    if not SLUG_RE.match(slug):
        raise Refus("slug invalide (attendu : src-…, c-… ou e-…)")
    for f in wiki_dir.rglob(f"{slug}.md"):
        if "poubelle" not in f.relative_to(wiki_dir).parts:
            return f.read_text(encoding="utf-8")
    raise Refus(f"page introuvable : {slug}")
```

- [ ] **Step 4: Lancer, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/python -m pytest tests/test_porte_lecture_wiki.py -q`
Expected: PASS (tous)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/porte/lecture_wiki.py Wiki_LM/tests/test_porte_lecture_wiki.py
git commit -m "feat(porte): lecture du wiki (question verrouillée, recherche /r, page par slug)"
```

---

### Task 4: Serveur MCP (authentification, outils, journal, /health)

**Files:**
- Create: `Wiki_LM/porte/serveur.py`
- Test: `Wiki_LM/tests/test_porte_serveur.py`

**Interfaces:**
- Consumes: `jetons.agent_pour`, `coffre.*`, `lecture_wiki.*` (Tasks 1–3).
- Produces: `serveur.creer_app(racine: Path, fichier_agents: Path, wiki_url: str)` → application ASGI ; `serveur.main()` (lit `PORTE_COFFRE` obligatoire, `PORTE_AGENTS` défaut `jetons.FICHIER_DEFAUT`, `PORTE_WIKI_URL` défaut `http://127.0.0.1:5051`, écoute `127.0.0.1:5052`) ; lancement `python -m porte.serveur`.

- [ ] **Step 1: Écrire les tests (échouent)**

`Wiki_LM/tests/test_porte_serveur.py` :

```python
"""Serveur MCP de la porte Grok, en HTTP réel sur un port local libre."""
import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import uvicorn
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

from porte import jetons
from porte.serveur import creer_app


class _FauxWiki(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        data = {"text": "Synthèse.", "references": ["src-a"]} if self.path == "/query" \
            else {"results": [{"title": "A", "excerpt": "ex"}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, *a):
        pass


def _port_libre():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def porte(tmp_path):
    racine = tmp_path / "Publication"
    (racine / "Wiki_LM" / "wiki" / "sources").mkdir(parents=True)
    (racine / "Wiki_LM" / "wiki" / "sources" / "src-a.md").write_text("page A")
    (racine / "Notes.md").write_text("perso")
    agents = tmp_path / "agents"
    jeton = jetons.ajouter(agents, "veille")
    wiki = ThreadingHTTPServer(("127.0.0.1", 0), _FauxWiki)
    threading.Thread(target=wiki.serve_forever, daemon=True).start()
    app = creer_app(racine, agents, f"http://127.0.0.1:{wiki.server_address[1]}")
    port = _port_libre()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)
    yield {"url": f"http://127.0.0.1:{port}", "jeton": jeton, "racine": racine, "agents": agents}
    srv.should_exit = True
    wiki.shutdown()


def _appeler(porte, jeton, outil=None, args=None):
    async def run():
        transport = StreamableHttpTransport(f"{porte['url']}/mcp", headers={"Authorization": f"Bearer {jeton}"})
        async with Client(transport) as c:
            if outil is None:
                return sorted(t.name for t in await c.list_tools())
            return (await c.call_tool(outil, args or {})).data
    return asyncio.run(run())


def test_sans_jeton_401(porte):
    assert httpx.post(f"{porte['url']}/mcp", json={}).status_code == 401
    assert httpx.post(f"{porte['url']}/mcp", json={}, headers={"Authorization": "Bearer faux"}).status_code == 401


def test_health_sans_jeton(porte):
    r = httpx.get(f"{porte['url']}/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_six_outils(porte):
    assert _appeler(porte, porte["jeton"]) == [
        "coffre_ecrire", "coffre_lire", "coffre_lister", "wiki_page", "wiki_question", "wiki_recherche"]


def test_outils_wiki(porte):
    assert _appeler(porte, porte["jeton"], "wiki_question", {"question": "q"}) == \
        {"synthese": "Synthèse.", "references": ["src-a"]}
    assert _appeler(porte, porte["jeton"], "wiki_recherche", {"mots": "a"}) == [{"title": "A", "excerpt": "ex"}]
    assert _appeler(porte, porte["jeton"], "wiki_page", {"slug": "src-a"}) == "page A"


def test_ecriture_prefixee_et_refus(porte):
    assert _appeler(porte, porte["jeton"], "coffre_ecrire",
                    {"chemin": "Brouillons/veille-1.md", "contenu": "v"}) == "Brouillons/veille-1.md"
    assert (porte["racine"] / "Brouillons" / "veille-1.md").read_text() == "v"
    with pytest.raises(ToolError, match="veille-"):
        _appeler(porte, porte["jeton"], "coffre_ecrire", {"chemin": "Notes.md", "contenu": "x"})
    assert (porte["racine"] / "Notes.md").read_text() == "perso"
    with pytest.raises(ToolError, match="zone interdite"):
        _appeler(porte, porte["jeton"], "coffre_lire", {"chemin": "Wiki_LM/wiki/sources/src-a.md"})
    assert _appeler(porte, porte["jeton"], "coffre_lister", {}) == ["Brouillons/veille-1.md", "Notes.md"]


def test_revocation_sans_redemarrage(porte):
    assert _appeler(porte, porte["jeton"], "wiki_page", {"slug": "src-a"}) == "page A"
    jetons.retirer(porte["agents"], "veille")
    assert httpx.post(f"{porte['url']}/mcp", json={},
                      headers={"Authorization": f"Bearer {porte['jeton']}"}).status_code == 401


def test_journal_sans_contenu(porte, caplog):
    caplog.set_level("INFO", logger="porte-grok")
    _appeler(porte, porte["jeton"], "coffre_ecrire", {"chemin": "veille-j.md", "contenu": "CONTENU-SECRET"})
    lignes = [r.getMessage() for r in caplog.records if r.name == "porte-grok"]
    assert any("veille" in l and "coffre_ecrire" in l and "veille-j.md" in l and "ok" in l for l in lignes)
    assert not any("CONTENU-SECRET" in l for l in lignes)
```

- [ ] **Step 2: Lancer, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/python -m pytest tests/test_porte_serveur.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'porte.serveur'`)

- [ ] **Step 3: Implémenter**

`Wiki_LM/porte/serveur.py` :

```python
"""Serveur MCP de la porte Grok (spec 2026-10-13).

Écoute sur 127.0.0.1:5052 ; Tailscale Funnel le publie sur 8443. Chaque
requête /mcp doit porter « Authorization: Bearer <jeton> » d'un agent connu
(fichier relu à chaque requête : un retrait est immédiat). Les outils
synchrones tournent dans un pool de threads (fastmcp) : une question longue
ne bloque pas les autres agents.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request
from starlette.middleware import Middleware
from starlette.responses import JSONResponse

from porte import coffre, jetons, lecture_wiki
from porte.coffre import Refus

journal = logging.getLogger("porte-grok")

_INSTRUCTIONS = (
    "Wiki et coffre Obsidian « Publication » de Secretarius. Lecture du wiki "
    "(wiki_recherche avant wiki_question, gratuite et instantanée) et lecture/"
    "écriture de fichiers .md du coffre ; vos fichiers doivent commencer par "
    "votre préfixe d'agent. Rien n'entre dans le wiki sans validation humaine."
)


def creer_app(racine: Path, fichier_agents: Path, wiki_url: str):
    mcp = FastMCP("secretarius-porte", instructions=_INSTRUCTIONS)
    wiki_dir = racine / "Wiki_LM" / "wiki"

    def _appel(outil: str, detail: str, fn):
        agent = jetons.agent_pour(fichier_agents, get_http_request().headers.get("authorization"))
        if agent is None:  # le middleware a déjà filtré ; défense en profondeur
            raise ToolError("jeton invalide")
        try:
            resultat = fn(agent)
        except (Refus, lecture_wiki.WikiIndisponible) as e:
            journal.info("%s %s %s refus : %s", agent, outil, detail, e)
            raise ToolError(str(e)) from None
        except Exception:
            journal.exception("%s %s %s erreur interne", agent, outil, detail)
            raise ToolError("erreur interne de la porte") from None
        journal.info("%s %s %s ok", agent, outil, detail)
        return resultat

    @mcp.tool()
    def wiki_question(question: str) -> dict:
        """Pose une question au wiki ; renvoie une synthèse et les slugs cités. ~1 min si le modèle dort."""
        return _appel("wiki_question", question[:80], lambda a: lecture_wiki.question(wiki_url, question))

    @mcp.tool()
    def wiki_recherche(mots: str) -> list[dict]:
        """Recherche par mots-clés (sans LLM, instantanée) ; renvoie titres et extraits."""
        return _appel("wiki_recherche", mots[:80], lambda a: lecture_wiki.recherche(wiki_url, mots))

    @mcp.tool()
    def wiki_page(slug: str) -> str:
        """Lit une page du wiki par son slug (src-…, c-…, e-…)."""
        return _appel("wiki_page", slug[:80], lambda a: lecture_wiki.page(wiki_dir, slug))

    @mcp.tool()
    def coffre_lister(dossier: str = "") -> list[str]:
        """Liste les fichiers .md du coffre (hors Wiki_LM/ et dossiers cachés)."""
        return _appel("coffre_lister", dossier[:80], lambda a: coffre.lister(racine, dossier))

    @mcp.tool()
    def coffre_lire(chemin: str) -> str:
        """Lit un fichier .md du coffre (chemin relatif)."""
        return _appel("coffre_lire", chemin[:80], lambda a: coffre.lire(racine, chemin))

    @mcp.tool()
    def coffre_ecrire(chemin: str, contenu: str) -> str:
        """Crée ou remplace un fichier .md ; son nom doit commencer par votre préfixe suivi de « - »."""
        return _appel("coffre_ecrire", chemin[:80], lambda a: coffre.ecrire(racine, a, chemin, contenu))

    @mcp.custom_route("/health", methods=["GET"])
    async def sante(request):
        return JSONResponse({"status": "ok"})

    class _Authentification:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http" and scope["path"].startswith("/mcp"):
                en_tete = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
                if jetons.agent_pour(fichier_agents, en_tete) is None:
                    journal.info("refus 401 depuis %s", (scope.get("client") or ("?",))[0])
                    await JSONResponse({"error": "jeton absent ou invalide"}, status_code=401)(scope, receive, send)
                    return
            await self.app(scope, receive, send)

    return mcp.http_app(path="/mcp", middleware=[Middleware(_Authentification)], stateless_http=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    racine = Path(os.environ["PORTE_COFFRE"])
    agents = Path(os.environ.get("PORTE_AGENTS", str(jetons.FICHIER_DEFAUT)))
    wiki_url = os.environ.get("PORTE_WIKI_URL", "http://127.0.0.1:5051")
    app = creer_app(racine, agents, wiki_url)
    uvicorn.run(app, host="127.0.0.1", port=5052, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer, vérifier le succès, puis toute la suite Wiki_LM**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/python -m pytest tests/test_porte_serveur.py -q && .venv/bin/python -m pytest tests -q`
Expected: PASS ; suite complète verte (524 + les nouveaux)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/porte/serveur.py Wiki_LM/tests/test_porte_serveur.py
git commit -m "feat(porte): serveur MCP authentifié (6 outils, journal, /health)"
```

---

### Task 5: Cases à cocher devant les blocs `wiki` (plugin Obsidian)

**Files:**
- Modify: `Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts` (`extractWikiBlocks`)
- Modify: `Wiki_LM/obsidian-wikilm-capture/main.js` (reconstruit par `npm run build`)
- Test: `Wiki_LM/obsidian-wikilm-capture/src/run-commands.test.ts`

**Interfaces:**
- Produces: `extractWikiBlocks(noteText)` ne renvoie plus les blocs dont la dernière ligne non vide précédente est une case vide `- [ ]` ; `start`/`end` inchangés pour les autres.

- [ ] **Step 1: Écrire les tests (échouent)**

Dans `src/run-commands.test.ts`, à l'intérieur de `describe("extractWikiBlocks", …)`, ajouter :

```ts
  it("ignore un bloc précédé d'une case vide", () => {
    const note = "Résumé\n\n- [ ] Retenir\n```wiki\n/c\n#ia https://a.example\n```\n";
    expect(extractWikiBlocks(note)).toHaveLength(0);
  });

  it("exécute un bloc précédé d'une case cochée (x ou X)", () => {
    for (const c of ["x", "X"]) {
      const note = `- [${c}] Retenir\n\`\`\`wiki\n/c\n#ia https://a.example\n\`\`\`\n`;
      const blocks = extractWikiBlocks(note);
      expect(blocks).toHaveLength(1);
      expect(blocks[0].arg).toBe("#ia https://a.example");
    }
  });

  it("note de veille mêlée : seuls les blocs cochés ou sans case, positions exactes", () => {
    const note =
      "## A\nRésumé A\n\n- [ ] Retenir\n```wiki\n/c\nA\n```\n\n" +
      "## B\nRésumé B\n\n- [x] Retenir\n\n```wiki\n/c\nB\n```\n\n" +
      "Texte libre\n```wiki\n/wikistatus\n```\n";
    const blocks = extractWikiBlocks(note);
    expect(blocks.map((b) => b.arg || b.command)).toEqual(["B", "/wikistatus"]);
    for (const b of blocks) {
      expect(note.slice(b.start, b.start + 8)).toBe("```wiki\n");
      expect(note.slice(b.end - 3, b.end)).toBe("```");
    }
  });
```

- [ ] **Step 2: Lancer, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/obsidian-wikilm-capture && npx vitest run src/run-commands.test.ts`
Expected: FAIL (2 tests : case vide ignorée, note mêlée)

- [ ] **Step 3: Implémenter**

Dans `src/run-commands.ts`, ajouter au-dessus de `extractWikiBlocks` :

```ts
// Note de veille (porte Grok, 2026-10-13) : une case vide « - [ ] » sur la
// dernière ligne non vide avant un bloc l'écarte ; cochée, ou absente, le
// bloc s'exécute comme avant.
const CASE_VIDE = /^\s*[-*]\s+\[ \]/;

function precedeDUneCaseVide(noteText: string, debut: number): boolean {
  const lignes = noteText.slice(0, debut).split("\n").filter((l) => l.trim() !== "");
  return lignes.length > 0 && CASE_VIDE.test(lignes[lignes.length - 1]);
}
```

et, dans la boucle `while` de `extractWikiBlocks`, juste après `while ((match = re.exec(noteText)) !== null) {` :

```ts
    if (precedeDUneCaseVide(noteText, match.index)) continue;
```

- [ ] **Step 4: Lancer, vérifier le succès ; suite complète ; reconstruire**

Run: `cd ~/Secretarius/Wiki_LM/obsidian-wikilm-capture && npx vitest run && npm run build`
Expected: tous les tests passent ; `main.js` régénéré

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts Wiki_LM/obsidian-wikilm-capture/src/run-commands.test.ts Wiki_LM/obsidian-wikilm-capture/main.js
git commit -m "feat(obsidian): cases à cocher devant les blocs wiki (note de veille)"
```

---

### Task 6: Module `porte` de l'installateur

**Files:**
- Create: `openclaw-config/porte-grok.service`
- Modify: `openclaw-config/modules.sh` (ajout de `module_porte`)
- Modify: `lib/installation.sh` (ajout de `etat_funnel_porte`)
- Modify: `install.sh` (initialiser `COMPTES_A_LIER=()` **avant** la boucle des modules)
- Modify: `profils/leger.conf` (`MODULES` + `porte`)
- Modify: `openclaw-config/uninstall.sh` (liste `SERVICES`)
- Test: `tests_installation/test_installation_lib.py`

**Interfaces:**
- Consumes: `Wiki_LM/porte/serveur.py` (`python -m porte.serveur`), `Wiki_LM/porte/jetons.py` (`python -m porte.jetons`).
- Produces: `etat_funnel_porte` (lib) : code 0 si `${TAILSCALE:-tailscale} funnel status` mentionne `:8443`, 1 sinon ; `module_porte` (modules.sh).

- [ ] **Step 1: Écrire les tests (échouent)**

Dans `tests_installation/test_installation_lib.py`, remplacer le corps de `test_profil_leger` par :

```python
def test_profil_leger():
    r = _bash('charger_profil leger && echo "$MODULES|$MAIN_BRAIN|$WIKI_LLM"')
    assert r.stdout.strip() == "wiki scout obfuscateur porte|modal|obfusque"
```

puis ajouter :

```python
def test_profil_lourd_sans_porte():
    r = _bash('charger_profil lourd && echo "$MODULES"')
    assert "porte" not in r.stdout.split()


def _faux_tailscale(tmp_path, sortie):
    t = tmp_path / "tailscale"
    t.write_text(f'#!/bin/sh\necho "{sortie}"\n')
    t.chmod(0o755)
    return t


def test_funnel_porte_actif(tmp_path):
    t = _faux_tailscale(tmp_path, "https://santiago.x.ts.net:8443 (Funnel on)\\n|-- / proxy http://127.0.0.1:5052")
    assert _bash(f"TAILSCALE={t}; etat_funnel_porte").returncode == 0


def test_funnel_porte_absent(tmp_path):
    t = _faux_tailscale(tmp_path, "No serve config")
    assert _bash(f"TAILSCALE={t}; etat_funnel_porte").returncode == 1


def test_comptes_a_lier_initialise_avant_les_modules():
    texte = (RACINE / "install.sh").read_text()
    assert texte.index("COMPTES_A_LIER=()") < texte.index("for _m in $MODULES")
```

- [ ] **Step 2: Lancer, vérifier l'échec**

Run: `cd ~/Secretarius && python3 -m pytest tests_installation -q`
Expected: FAIL (4 : profil léger, funnel actif, funnel absent, ordre de COMPTES_A_LIER)

- [ ] **Step 3: Implémenter**

`lib/installation.sh`, ajouter en fin de fichier :

```bash

# Porte Grok (spec 2026-10-13) : Funnel publie 127.0.0.1:5052 sur le port 8443.
# TAILSCALE est remplaçable par les tests.
etat_funnel_porte() {
  "${TAILSCALE:-tailscale}" funnel status 2>/dev/null | grep -q ':8443'
}
```

`openclaw-config/porte-grok.service` :

```ini
[Unit]
Description=Porte Grok — MCP public du wiki et du coffre (127.0.0.1:5052, Funnel 8443)
After=network.target wiki-lm-server.service

[Service]
Type=simple
WorkingDirectory=%h/Secretarius/Wiki_LM
EnvironmentFile=%h/.config/porte-grok/porte.env
ExecStart=%h/Secretarius/Wiki_LM/.venv/bin/python3 -m porte.serveur
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

`openclaw-config/modules.sh`, ajouter après `module_gog` (même style que les autres modules) :

```bash
# Porte Grok (spec 2026-10-13) : MCP public du wiki et du coffre, pour les
# agents Grok Bot. Funnel n'est pas activé ici (réglages de compte Tailscale) :
# son absence devient une étape de compte, pas un échec.
module_porte() {
  local conf="${HOME}/.config/porte-grok"
  mkdir -p "$conf" && chmod 700 "$conf"
  [[ -f "${conf}/agents" ]] || { : > "${conf}/agents"; chmod 600 "${conf}/agents"; }
  printf 'PORTE_COFFRE=%s\nPORTE_AGENTS=%s\nPORTE_WIKI_URL=http://127.0.0.1:5051\n' \
    "$OBSIDIAN_PATH" "${conf}/agents" > "${conf}/porte.env"
  chmod 600 "${conf}/porte.env"

  cat > "${HOME}/.local/bin/porte-jeton" <<EOF
#!/usr/bin/env bash
# Généré par Secretarius install.sh — jetons des agents de la porte Grok.
cd "${SECRETARIUS_ROOT}/Wiki_LM" && exec .venv/bin/python3 -m porte.jetons "\$@"
EOF
  chmod +x "${HOME}/.local/bin/porte-jeton"
  info "porte-jeton installé dans ${HOME}/.local/bin"

  local dst="${SYSTEMD_USER_DIR}/porte-grok.service"
  if [[ -f "$dst" && "$FORCE" != "true" ]]; then
    info "porte-grok.service existe déjà — ignoré"
  else
    cp "${SCRIPT_DIR}/porte-grok.service" "$dst"
    info "porte-grok.service installé dans ${SYSTEMD_USER_DIR}"
  fi
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable porte-grok.service 2>/dev/null || true
  systemctl --user restart porte-grok.service 2>/dev/null \
    && info "porte-grok.service démarré" || warn "Démarrage de porte-grok.service échoué"
  SERVICES_A_VERIFIER+=("porte-grok|http://127.0.0.1:5052/health")

  if etat_funnel_porte; then
    info "Funnel publie la porte sur 8443 ✓"
  else
    COMPTES_A_LIER+=("Publier la porte Grok (une fois) : autoriser Funnel pour cette machine dans la
       console Tailscale, puis
       sudo tailscale set --operator=\$USER
       tailscale funnel --bg --https=8443 http://127.0.0.1:5052
     Jeton d'un agent : porte-jeton ajouter <préfixe>")
  fi
}
```

`install.sh` : déplacer la ligne `COMPTES_A_LIER=()` (actuellement juste après la boucle des modules) sur la ligne qui précède `for _m in $MODULES; do`. Vérifier qu'aucune autre ligne ne la réinitialise ensuite (`grep -n "COMPTES_A_LIER=()" install.sh` → une seule occurrence).

`profils/leger.conf` : `MODULES="${MODULES:-wiki scout obfuscateur porte}"`.

`openclaw-config/uninstall.sh` : dans le tableau `SERVICES`, ajouter `"porte-grok.service"` après `"openclaw-scout.service"`.

- [ ] **Step 4: Lancer, vérifier le succès**

Run: `cd ~/Secretarius && bash -n install.sh && bash -n lib/installation.sh && bash -n openclaw-config/modules.sh && bash -n openclaw-config/uninstall.sh && python3 -m pytest tests_installation openclaw-config/test_profil_config.py -q`
Expected: PASS (tous)

`profil_config.py` ignore les modules qu'il ne connaît pas (`AGENTS_DE`/`OUTILS_DE`) : `porte`, sans agent ni outil OpenClaw, n'y demande aucun changement.

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius && git add openclaw-config/porte-grok.service openclaw-config/modules.sh lib/installation.sh install.sh profils/leger.conf openclaw-config/uninstall.sh tests_installation/test_installation_lib.py
git commit -m "feat(installateur): module porte (porte-grok, porte-jeton, Funnel en étape de compte)"
```

---

### Task 7: Fiche du coordinateur Grok

**Files:**
- Create: `docs/grok/fiche-coordinateur.md`

**Interfaces:**
- Consumes: noms et paramètres des 6 outils (Task 4), règles de `coffre` (Task 2), format des cases (Task 5).

- [ ] **Step 1: Écrire la fiche**

`docs/grok/fiche-coordinateur.md` :

`````markdown
# Fiche du coordinateur — wiki et coffre « Publication » de Secretarius

Vous coordonnez une équipe d'agents de publication (YouTube, Substack, LinkedIn, site).
Secretarius vous ouvre un **connecteur MCP** sur la machine santiago : lecture de son wiki
et lecture/écriture de fichiers Markdown dans le coffre Obsidian « Publication » de
l'utilisateur. **Rien n'entre dans le wiki sans validation humaine** : vous déposez des
fichiers, l'utilisateur coche ce qu'il retient.

## Connexion

- Adresse : `https://santiago.<tailnet>.ts.net:8443/mcp` (l'utilisateur vous donne l'adresse exacte).
- Authentification : en-tête `Authorization: Bearer <jeton>`. **Un jeton par agent**, fourni
  par l'utilisateur ; chaque jeton porte un **préfixe** (ex. `veille`, `redaction`).
- Ne jamais mettre un jeton dans une URL, un fichier du coffre ou un message.

## Outils

| Outil | Paramètres | Usage |
|---|---|---|
| `wiki_recherche` | `mots` | À utiliser en premier : instantané, sans coût. Renvoie titres et extraits. |
| `wiki_question` | `question` | Synthèse rédigée à partir du wiki + slugs cités. Jusqu'à ~1 min au réveil du modèle ; une question à la fois pour toute l'équipe. |
| `wiki_page` | `slug` | Texte complet d'une page (`src-…`, `c-…`, `e-…`), slugs tirés des références de `wiki_question`. |
| `coffre_lister` | `dossier` (facultatif) | Fichiers `.md` du coffre. |
| `coffre_lire` | `chemin` | Lire un fichier `.md`. |
| `coffre_ecrire` | `chemin`, `contenu` | Créer ou remplacer un fichier `.md`. |

Exemples :
- `wiki_recherche({"mots": "écriture IA valeur"})` → `[{"title": "Is AI Killing…", "excerpt": "…"}]`
- `wiki_question({"question": "Que dit le wiki de la valeur de l'écriture humaine ?"})` →
  `{"synthese": "…", "references": ["src-is-ai-killing-the-value-of-human-writing", …]}`
- `coffre_ecrire({"chemin": "Brouillons/Veille/veille-2026-10-14.md", "contenu": "…"})` →
  `"Brouillons/Veille/veille-2026-10-14.md"`

## Règles d'écriture

- Le **nom du fichier** doit commencer par le préfixe de l'agent suivi de `-`
  (`veille-2026-10-14.md`, `redaction-article-substack.md`). Sinon : refus.
- Fichiers `.md` seulement, 1 Mo au plus. Un agent peut réécrire ses propres fichiers.
- Interdits : `Wiki_LM/`, `.obsidian/` et tout dossier caché ; chemins absolus ou avec `..`.
- Pas de suppression ni de renommage : demandez-les à l'utilisateur.
- Dossiers conseillés : `Brouillons/Veille/`, `Brouillons/<plateforme>/`.

## Format d'une note de veille

Une note par livraison. Pour chaque élément : un titre, un résumé, une case **vide**, puis un
bloc `wiki` contenant la commande de capture. C'est l'utilisateur qui coche ; le plugin
Obsidian n'exécute que les blocs cochés.

````markdown
## <titre> (<source>, <date>)
Résumé : <3 à 6 phrases>.

- [ ] Retenir
```wiki
/c
#tag1 #tag2 <url>
```
````

- Toujours une case vide : ne jamais cocher à la place de l'utilisateur.
- Tags courts, sans espace, en français si possible.
- Une URL par bloc.

## Ce que vous ne pouvez pas faire, et pourquoi

Supprimer, renommer, capturer ou ingérer directement, lire hors du coffre. Tout ce qui entre
dans le wiki passe par la validation de l'utilisateur, puis par la chaîne d'ingestion de
Secretarius (qui va chercher et résume elle-même les pages, protégée contre l'injection de
consignes) : ne recopiez pas le contenu d'une page web dans le wiki, donnez son URL.

## Erreurs

Les refus arrivent comme erreurs d'outil avec un message court (« le nom du fichier doit
commencer par veille- », « zone interdite… », « wiki injoignable »). `401` : jeton absent,
invalide ou révoqué — prévenir l'utilisateur.
`````

- [ ] **Step 2: Vérifier la cohérence avec le code**

Run: `cd ~/Secretarius && grep -o 'def \(wiki\|coffre\)_[a-z]*' Wiki_LM/porte/serveur.py | sort`
Expected: les 6 noms de la fiche, à l'identique.

- [ ] **Step 3: Commit**

```bash
cd ~/Secretarius && git add docs/grok/fiche-coordinateur.md
git commit -m "docs(grok): fiche du coordinateur (connecteur, outils, règles, format de veille)"
```

---

### Task 8: Déploiement et vérification sur santiago (avec l'utilisateur)

Effets hors du dépôt : `git push`, redémarrages et Funnel demandent l'accord de l'utilisateur (CLAUDE.md).

**Files:** aucun dans le dépôt (sauf correctifs éventuels, chacun avec son test).

- [ ] **Step 1: Pousser et mettre santiago à jour** (accord : push)

```bash
cd ~/Secretarius && git push origin main
ssh santiago 'cd ~/Secretarius && git pull -q && git log --oneline -1'
```

- [ ] **Step 2: Relancer l'installateur sur santiago** (accord : redémarrages)

```bash
ssh santiago 'cd ~/Secretarius && bash -lc "./install.sh --profil leger --env-file ~/.config/secrets.env" > ~/install-porte.log 2>&1; tail -20 ~/install-porte.log'
```
Expected: « Tous les services répondent (7). » ; étape de compte Funnel listée si Funnel n'est pas encore actif.

- [ ] **Step 3: Funnel (utilisateur)** — console Tailscale : autoriser Funnel pour santiago ; puis, sur santiago :
`sudo tailscale set --operator=$USER` et `tailscale funnel --bg --https=8443 http://127.0.0.1:5052`.

- [ ] **Step 4: Premier jeton et contrôles**

```bash
ssh santiago 'porte-jeton ajouter veille'    # l'utilisateur recopie le jeton dans Grok Bot
curl -s -o /dev/null -w "%{http_code}\n" -X POST https://santiago.<tailnet>.ts.net:8443/mcp   # → 401
curl -s https://santiago.<tailnet>.ts.net:8443/health                                         # → {"status":"ok"}
```

- [ ] **Step 5: Plugin dans le coffre Publication** — copier `main.js` et `manifest.json` reconstruits dans `~/Documents/Publication/.obsidian/plugins/wikilm-capture/` sur santiago (et dans Arbath/Secretarius sur sanroque), puis l'utilisateur active le plugin dans Obsidian.

- [ ] **Step 6: Bout en bout (utilisateur, depuis Grok Bot)** : connecteur ajouté (en-tête Bearer) ; une question au wiki ; dépôt d'une note de veille au format de la fiche ; la note apparaît dans Obsidian ; cocher un élément ; « Exécuter les commandes wiki de la note » ; vérifier la capture (`.url.md` dans `raw/`) puis `/ingest`.

- [ ] **Step 7: Révocation** : `ssh santiago 'porte-jeton retirer veille'` → l'appel suivant de l'agent reçoit 401 ; puis recréer le jeton si l'utilisateur le souhaite.
