# FAQ de faits locaux — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tiron répond verbatim aux questions couvertes par un fichier de faits édité à la main, via une recherche au plus proche voisin, avant le routage commandes.

**Architecture:** Un nouveau module `router_service/faq.py` parse `faits.md`, embarque chaque question (BGE-M3 déjà chargé par `GogGate`) et retrouve l'entrée la plus proche d'un message. `route_message` consulte la FAQ d'abord (sauf commande `/…`) et renvoie le corps de l'entrée verbatim (`status:"answer"`). Le plugin `derisk-deleg` relaie ce texte. `install.sh` amorce le fichier sans jamais l'écraser.

**Tech Stack:** Python 3 (torch, BGE-M3 via transformers), TypeScript/vitest (plugin OpenClaw), bash.

## Global Constraints

- Ne JAMAIS toucher au port prod 8998 (adaptateur de routage) ni introduire d'appel LLM dans le chemin FAQ — la réponse est le corps verbatim de l'entrée.
- Réutiliser `GogGate._embed` comme fonction d'embedding (pas de second modèle chargé).
- `route_message` ne doit jamais lever d'exception non capturée (fichier absent/vide/mal formé → dégradation silencieuse, retombe sur le routage commandes).
- Les commandes explicites commençant par `/` ne passent jamais par la FAQ.
- Le seed est non-clobber : même `install.sh --force` ne doit PAS écraser `faits.md` s'il existe.
- Tests Python lancés depuis la racine du dépôt avec `PYTHONPATH=.` (motif des tests existants `router_service/test_router.py`).
- `SEUIL_FAQ` défaut 0.6 (`FAQ_SEUIL`), `FAQ_MAX_ENTREE` défaut 2000 (`FAQ_MAX_ENTREE`), `FAQ_PATH` défaut `~/Documents/Arbath/Wiki_LM/faits/faits.md`.

---

### Task 1 : `parse_faq` — parseur du fichier de faits

**Files:**
- Create: `router_service/faq.py`
- Test: `router_service/test_faq.py`

**Interfaces:**
- Produces : `parse_faq(text: str) -> list[dict]` où chaque dict = `{"questions": list[str], "answer": str}`. Constantes module : `FAQ_PATH: Path`, `SEUIL_FAQ: float`, `FAQ_MAX_ENTREE: int`.

- [ ] **Step 1 : Écrire les tests qui échouent**

```python
# router_service/test_faq.py
from router_service.faq import parse_faq


def test_entree_simple():
    e = parse_faq("## Question ?\nRéponse.")
    assert e == [{"questions": ["Question ?"], "answer": "Réponse."}]


def test_multi_formulations():
    e = parse_faq("## Q1 ?\n## Q2 ?\nUne réponse.")
    assert e == [{"questions": ["Q1 ?", "Q2 ?"], "answer": "Une réponse."}]


def test_corps_multiligne_et_deux_entrees():
    txt = "## A ?\nligne1\nligne2\n\n## B ?\nrb"
    e = parse_faq(txt)
    assert e == [
        {"questions": ["A ?"], "answer": "ligne1\nligne2"},
        {"questions": ["B ?"], "answer": "rb"},
    ]


def test_commentaires_et_titre_h1_ignores():
    txt = "# Faits\n# Format : ...\n## Q ?\nR."
    assert parse_faq(txt) == [{"questions": ["Q ?"], "answer": "R."}]


def test_fichier_vide():
    assert parse_faq("") == []


def test_entree_sans_corps_ignoree():
    assert parse_faq("## Q sans réponse ?") == []


def test_garde_fou_entree_trop_longue(capsys):
    from router_service.faq import FAQ_MAX_ENTREE
    long = "x" * (FAQ_MAX_ENTREE + 1)
    assert parse_faq(f"## Q ?\n{long}") == []
    assert "ignorée" in capsys.readouterr().out
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius && PYTHONPATH=. python -m pytest router_service/test_faq.py -v`
Expected: FAIL (`ModuleNotFoundError: router_service.faq`).

- [ ] **Step 3 : Écrire `faq.py` (constantes + `parse_faq`)**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAQ de faits locaux : parse faits.md et retrouve l'entrée la plus proche
d'un message (single-vector nearest-neighbor sur BGE-M3). Réponse = corps
verbatim de l'entrée, aucun appel LLM."""
import os
from pathlib import Path

FAQ_PATH = Path(os.environ.get(
    "FAQ_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM/faits/faits.md")))
SEUIL_FAQ = float(os.environ.get("FAQ_SEUIL", "0.6"))
FAQ_MAX_ENTREE = int(os.environ.get("FAQ_MAX_ENTREE", "2000"))


def parse_faq(text: str) -> list[dict]:
    """Une entrée = une ou plusieurs lignes '## ...' (formulations) suivies d'un
    corps, jusqu'au prochain '## ' ou la fin. Lignes '# ...' (commentaires/H1)
    ignorées. Entrée sans corps ou dont le corps dépasse FAQ_MAX_ENTREE écartée."""
    entries: list[dict] = []
    questions: list[str] = []
    body: list[str] = []

    def flush() -> None:
        nonlocal questions, body
        answer = "\n".join(body).strip()
        if questions and answer:
            if len(answer) <= FAQ_MAX_ENTREE:
                entries.append({"questions": list(questions), "answer": answer})
            else:
                print(f"[faq] entrée ignorée (> {FAQ_MAX_ENTREE} car.): "
                      f"{questions[0]!r}", flush=True)
        questions = []
        body = []

    for line in text.splitlines():
        if line.startswith("## "):
            if body:            # corps déjà accumulé -> entrée précédente terminée
                flush()
            questions.append(line[3:].strip())
        elif line.startswith("#"):
            continue            # commentaire / titre H1
        elif line.strip() == "" and not body:
            continue            # ligne vide avant le corps
        else:
            body.append(line)
    flush()
    return entries
```

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius && PYTHONPATH=. python -m pytest router_service/test_faq.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add router_service/faq.py router_service/test_faq.py
git commit -m "feat(faq): parseur du fichier de faits (parse_faq + garde-fou par-entrée)"
```

---

### Task 2 : `FaqIndex` — embarquement + recherche au plus proche voisin

**Files:**
- Modify: `router_service/faq.py`
- Test: `router_service/test_faq.py`

**Interfaces:**
- Consumes : `parse_faq`, `FAQ_PATH`, `SEUIL_FAQ` (Task 1).
- Produces : `class FaqIndex(embed_fn, path=FAQ_PATH, seuil=SEUIL_FAQ)` avec `lookup(message: str) -> dict | None` (renvoie l'entrée matchée ou `None`). `embed_fn(texts: list[str]) -> torch.Tensor` L2-normalisé `[N, D]` (signature de `GogGate._embed`).

- [ ] **Step 1 : Écrire les tests qui échouent**

```python
# à ajouter dans router_service/test_faq.py
import os
import torch
from router_service.faq import FaqIndex


def _stub_embed(texts):
    # embeddings one-hot par mot-clé (déjà normalisés)
    vecs = []
    for t in texts:
        tl = t.lower()
        if "perroquet" in tl:
            vecs.append([1.0, 0.0, 0.0])
        elif "wiki" in tl:
            vecs.append([0.0, 1.0, 0.0])
        else:
            vecs.append([0.0, 0.0, 1.0])
    return torch.tensor(vecs)


def _ecrire(tmp_path, contenu):
    p = tmp_path / "faits.md"
    p.write_text(contenu, encoding="utf-8")
    return p


def test_lookup_match(tmp_path):
    p = _ecrire(tmp_path, "## Le perroquet de Mme Michu ?\nCoco.")
    idx = FaqIndex(_stub_embed, path=p, seuil=0.6)
    assert idx.lookup("parle-moi du perroquet")["answer"] == "Coco."


def test_lookup_sous_seuil(tmp_path):
    p = _ecrire(tmp_path, "## Le perroquet ?\nCoco.")
    idx = FaqIndex(_stub_embed, path=p, seuil=0.6)
    assert idx.lookup("quelle météo aujourd'hui") is None


def test_lookup_fichier_absent(tmp_path):
    idx = FaqIndex(_stub_embed, path=tmp_path / "absent.md", seuil=0.6)
    assert idx.lookup("le perroquet") is None


def test_reload_sur_mtime(tmp_path):
    p = _ecrire(tmp_path, "## wiki ?\nancienne")
    idx = FaqIndex(_stub_embed, path=p, seuil=0.6)
    assert idx.lookup("le wiki")["answer"] == "ancienne"
    p.write_text("## wiki ?\nnouvelle", encoding="utf-8")
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 10))
    assert idx.lookup("le wiki")["answer"] == "nouvelle"
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius && PYTHONPATH=. python -m pytest router_service/test_faq.py -k "lookup or reload" -v`
Expected: FAIL (`ImportError: cannot import name 'FaqIndex'`).

- [ ] **Step 3 : Ajouter `FaqIndex` dans `faq.py`**

```python
# ajouter en tête de faq.py :
import torch  # noqa: E402  (placé avec les autres imports en haut du fichier)

# ajouter à la fin de faq.py :
class FaqIndex:
    def __init__(self, embed_fn, path=FAQ_PATH, seuil=SEUIL_FAQ):
        self._embed_fn = embed_fn
        self._path = Path(path)
        self._seuil = seuil
        self._mtime = None
        self._qmat = None       # torch.Tensor [N_questions, D] ou None
        self._entry_of = []     # entrée correspondant à chaque question
        self._reload()

    def _current_mtime(self):
        try:
            return self._path.stat().st_mtime
        except OSError:
            return None

    def _reload(self):
        self._mtime = self._current_mtime()
        if self._mtime is None:
            self._qmat, self._entry_of = None, []
            return
        entries = parse_faq(self._path.read_text(encoding="utf-8"))
        questions, entry_of = [], []
        for e in entries:
            for q in e["questions"]:
                questions.append(q)
                entry_of.append(e)
        self._entry_of = entry_of
        self._qmat = self._embed_fn(questions) if questions else None

    def lookup(self, message: str):
        if self._current_mtime() != self._mtime:
            self._reload()
        if self._qmat is None:
            return None
        e = self._embed_fn([message])                 # [1, D], normalisé
        sims = (e @ self._qmat.T).squeeze(0)          # [N_questions]
        idx = int(sims.argmax())
        if float(sims[idx]) >= self._seuil:
            return self._entry_of[idx]
        return None
```

- [ ] **Step 4 : Lancer toute la suite `test_faq.py`, vérifier le succès**

Run: `cd ~/Secretarius && PYTHONPATH=. python -m pytest router_service/test_faq.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add router_service/faq.py router_service/test_faq.py
git commit -m "feat(faq): FaqIndex (nearest-neighbor + rechargement sur mtime)"
```

---

### Task 3 : Intégration `route_message` (FAQ d'abord) + démarrage

**Files:**
- Modify: `router_service/server.py` (import ~ligne 11 ; global `_faq` ~ligne 29 ; `route_message` lignes 46-58 ; `main` lignes 81-86)
- Test: `router_service/test_faq.py`

**Interfaces:**
- Consumes : `FaqIndex` (Task 2), `GogGate._embed`.
- Produces : `route_message` renvoie `{"status": "answer", "reply": <corps verbatim>}` quand la FAQ matche un message sans `/` initial ; sinon comportement inchangé.

- [ ] **Step 1 : Écrire les tests qui échouent**

```python
# à ajouter dans router_service/test_faq.py
from router_service import server as router_server
from router_service.faq import FaqIndex


def _install_faq(tmp_path):
    p = tmp_path / "faits.md"
    p.write_text("## Le perroquet de Mme Michu ?\nCoco.", encoding="utf-8")
    router_server._faq = FaqIndex(_stub_embed, path=p, seuil=0.6)


def test_route_faq_dabord(tmp_path):
    _install_faq(tmp_path)
    r = router_server.route_message("parle-moi du perroquet")
    assert r == {"status": "answer", "reply": "Coco."}


def test_route_slash_court_circuite_faq(tmp_path):
    _install_faq(tmp_path)
    # commence par '/' -> FAQ ignorée ; l'adaptateur (8998) est injoignable en
    # test -> call_adapter lève -> no_match. On vérifie surtout : jamais "answer".
    r = router_server.route_message("/perroquet")
    assert r["status"] != "answer"


def test_route_sans_match_retombe_routage(tmp_path):
    _install_faq(tmp_path)
    r = router_server.route_message("cherche un truc inconnu xyz")
    assert r["status"] != "answer"
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius && PYTHONPATH=. python -m pytest router_service/test_faq.py -k route -v`
Expected: FAIL (`AttributeError: module ... has no attribute '_faq'`).

- [ ] **Step 3 : Modifier `server.py`**

Ajouter l'import sous la ligne `from router_service.router import GogGate, WIKI_CMDS, GOG_CMDS` :

```python
from router_service.faq import FaqIndex, FAQ_PATH
```

Sous `_gate = None  # chargé au démarrage (Step 5)`, ajouter :

```python
_faq = None   # FaqIndex, chargé au démarrage
```

Remplacer le début de `route_message` (avant le `try:` existant) :

```python
def route_message(message: str) -> dict:
    if _faq is not None and not message.lstrip().startswith("/"):
        entry = _faq.lookup(message)
        if entry is not None:
            return {"status": "answer", "reply": entry["answer"]}
    try:
        command, args = call_adapter(message)
    except Exception:
        return {"status": "no_match"}
    # ... (suite inchangée)
```

Dans `main()`, après `_gate = GogGate()`, ajouter :

```python
    global _faq
    _faq = FaqIndex(_gate._embed)
    print(f"FAQ chargée ({FAQ_PATH})", flush=True)
```

- [ ] **Step 4 : Lancer toute la suite, vérifier le succès**

Run: `cd ~/Secretarius && PYTHONPATH=. python -m pytest router_service/test_faq.py router_service/test_router.py -v`
Expected: PASS (les tests FAQ + les tests routeur existants, inchangés).

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add router_service/server.py router_service/test_faq.py
git commit -m "feat(faq): route_message consulte la FAQ d'abord (réponse verbatim)"
```

---

### Task 4 : Plugin `derisk-deleg` — statut `answer` + message no_match

**Files:**
- Modify: `derisk-deleg/src/index.ts` (`callRouter` lignes 11-29 ; hook `before_agent_reply` lignes 446-456)

**Interfaces:**
- Consumes : réponse du routeur `{status:"answer", reply}` (Task 3).
- Produces : le hook relaie `reply` verbatim (coupé à 1800 car.).

- [ ] **Step 1 : Étendre le type de retour de `callRouter`**

Remplacer la signature et le corps de `callRouter` :

```typescript
async function callRouter(message: string): Promise<
  | { status: "ok"; command: string; args: string }
  | { status: "answer"; reply: string }
  | { status: "no_match" }
  | { status: "unavailable" }
> {
  try {
    const resp = await fetch(ROUTER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal: AbortSignal.timeout(30000),
    });
    if (!resp.ok) return { status: "unavailable" };
    const data = await resp.json();
    if (data.status === "answer") return { status: "answer", reply: data.reply ?? "" };
    return data.status === "ok"
      ? { status: "ok", command: data.command, args: data.args ?? "" }
      : { status: "no_match" };
  } catch {
    return { status: "unavailable" };
  }
}
```

- [ ] **Step 2 : Brancher le statut `answer` et reformuler no_match dans le hook**

Dans `before_agent_reply`, juste après le bloc `if (routed.status === "unavailable")`, insérer :

```typescript
      if (routed.status === "answer") {
        return { handled: true, reply: { text: routed.reply.slice(0, 1800) } };
      }
```

Puis, dans le bloc `if (routed.status === "no_match")`, remplacer le texte par :

```typescript
          reply: { text: "Je n'ai pas cette information (essayez /q <question>, /c <url>...)." },
```

- [ ] **Step 3 : Typecheck + build + validation du plugin**

Run: `cd ~/Secretarius/derisk-deleg && npm run build && npm run test`
Expected: `tsc` sans erreur ; les tests vitest existants (`dispatch.test.ts`) PASS.

Run: `cd ~/Secretarius/derisk-deleg && npm run plugin:validate`
Expected: validation OpenClaw OK.

- [ ] **Step 4 : Commit**

```bash
cd ~/Secretarius
git add derisk-deleg/src/index.ts derisk-deleg/dist
git commit -m "feat(faq): derisk-deleg relaie le statut answer (réponse FAQ verbatim)"
```

---

### Task 5 : `install.sh` — amorçage non-clobber du fichier de faits

**Files:**
- Modify: `install.sh` (insérer après le bloc « Étape 4 — Wiki_LM/.env », ~ligne 209)

**Interfaces:**
- Consumes : variables existantes `SECRETARIUS_ROOT`, `WIKI_PATH`, fonction `info`.
- Produces : `${WIKI_PATH}/faits/faits.md` amorcé depuis `amorçage/faits.md` seulement s'il n'existe pas.

- [ ] **Step 1 : Insérer l'étape d'amorçage**

Juste après le `fi` de fermeture du bloc Wiki_LM/.env (ligne 209), avant `# Étape 5` :

```bash

# Étape 4b — Amorçage de la FAQ de faits (non-clobber : jamais écrasé, même --force)
FAITS_SRC="${SECRETARIUS_ROOT}/amorçage/faits.md"
FAITS_DIR="${WIKI_PATH}/faits"
FAITS_DEST="${FAITS_DIR}/faits.md"
if [[ ! -f "$FAITS_DEST" ]]; then
  mkdir -p "$FAITS_DIR"
  cp "$FAITS_SRC" "$FAITS_DEST"
  info "FAQ de faits amorcée (${FAITS_DEST})"
else
  info "FAQ de faits déjà présente, conservée (${FAITS_DEST})"
fi
```

- [ ] **Step 2 : Vérifier la syntaxe bash**

Run: `cd ~/Secretarius && bash -n install.sh`
Expected: aucune sortie (syntaxe valide).

- [ ] **Step 3 : Test manuel du comportement non-clobber (bac à sable)**

```bash
cd /tmp && rm -rf faq_test && mkdir -p faq_test/wiki faq_test/src
printf '## Q ?\nR.\n' > faq_test/src/faits.md
# 1er passage : crée
SECRETARIUS_ROOT=/tmp/faq_test/src WIKI_PATH=/tmp/faq_test/wiki
mkdir -p "$WIKI_PATH/faits" 2>/dev/null || true
[[ ! -f "$WIKI_PATH/faits/faits.md" ]] && cp "$SECRETARIUS_ROOT/faits.md" "$WIKI_PATH/faits/faits.md"
printf 'ÉDIT UTILISATEUR\n' >> "$WIKI_PATH/faits/faits.md"
# 2e passage : ne doit PAS écraser
[[ ! -f "$WIKI_PATH/faits/faits.md" ]] && cp "$SECRETARIUS_ROOT/faits.md" "$WIKI_PATH/faits/faits.md"
grep -q "ÉDIT UTILISATEUR" "$WIKI_PATH/faits/faits.md" && echo "OK non-clobber" || echo "ÉCHEC"
```
Expected: `OK non-clobber`.

- [ ] **Step 4 : Commit**

```bash
cd ~/Secretarius
git add install.sh
git commit -m "feat(faq): amorçage non-clobber de faits.md à l'install"
```

---

## Déploiement (manuel, après validation des 5 tâches)

1. Amorcer le fichier live : `bash install.sh` (ou copier `amorçage/faits.md` vers `~/Documents/Arbath/Wiki_LM/faits/faits.md` si absent).
2. Redémarrer le service routeur : `systemctl --user restart tiron-router` (port 8999). Vérifier le log `FAQ chargée (...)`.
3. Réinstaller le plugin sur l'instance live : `openclaw plugins install ~/Secretarius/derisk-deleg --force`, puis `systemctl --user restart openclaw-gateway`.
4. E2E Telegram :
   - Question connue (« Comment s'appelle le perroquet de Madame Michu ? ») → « Le perroquet de Madame Michu s'appelle Coco. »
   - Ajouter un fait dans `faits.md`, reposer la question → réponse à chaud (sans redémarrage).
   - Question inconnue → « Je n'ai pas cette information (...) ».
   - `/q <question>` et une commande gog → fonctionnent comme avant (non-régression).
5. Calibrer `SEUIL_FAQ` (via `FAQ_SEUIL` dans l'env du service `tiron-router`) si des questions légitimes passent sous le seuil ou si du bruit passe au-dessus.

## Notes

- `FAQ_PATH` par défaut vise déjà `~/Documents/Arbath/Wiki_LM/faits/faits.md` — aucune variable d'env à ajouter au service si le vault est à l'emplacement standard.
- Le watcher `wiki` (`server.py:203`) rechargera l'index `wiki/` à chaque édition de `faits.md` (effet de bord bénin, documenté dans la spec).
