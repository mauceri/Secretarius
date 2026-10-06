# Routage Laya à deux niveaux — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** un routage en langage naturel fiable sans retouches successives. Laya décide en deux temps, le domaine puis la commande. Quand il hésite, Tiron demande « 1 ou 2 » au lieu de deviner.

**Architecture :** une seule passe Laya avec trois questions (`domaine`, `commande_wiki`, `commande_google`). Les probabilités jointes donnent un classement ; au-dessus de `SEUIL` le routeur décide, en dessous il renvoie `clarify` avec les deux meilleures options. derisk-deleg pose la question et exécute la réponse. Les données d'entraînement portent les questions par domaine. L'évaluation se fait sur le jeu de validation stratifié et sur un jeu de test écrit à la main.

**Tech Stack :** Python 3.12 (router_service, python3 système), Laya, pytest ; TypeScript + vitest (derisk-deleg) ; DSPy/DeepSeek (gen_corpus, `.venv`) ; Kaggle (manuel).

**Spec :** `docs/superpowers/specs/2026-10-06-routage-deux-niveaux-design.md`

## Global Constraints

- Domaines : `wiki`, `google`, `web`, `planification`, `aucune`.
- `wiki` : `/c /q /ingest /wikistatus /tags /kbupdate /supprimer /relire /verifie` ; `google` : `/chercher /connecter /inbox /drive /repondre /lire` ; `web` → `/source` ; `planification` → `/cron`.
- Commandes tapées (`/xxx …`) : jamais soumises à Laya (comportement actuel inchangé).
- Questions de connaissance générale → `/q` ; hors périmètre = actions impossibles et conversation.
- Jeu de validation Laya stratifié : `min(40, n // 4)` exemples par commande ; valset GEPA inchangé (40 au total).
- Modèle publié sous `Mauceric/laya-tiron-router`, sans gestion de versions.
- Seuil d'acceptation, sur le jeu manuel : ≥ 90 % « juste ou question pertinente », aucune commande < 80 %, ≤ 15 % de demandes de précision.
- Redémarrer `tiron-router` ou `openclaw-gateway`, dépenser du crédit DeepSeek ou lancer Kaggle : avec l'accord de l'utilisateur.
- Tests router : `cd ~/Secretarius && LAYA_CHECKPOINT=x python3 -m pytest router_service -q` ; gen_corpus : `gen_corpus/.venv/bin/python -m pytest gen_corpus/tests -q` ; derisk-deleg : `cd derisk-deleg && npx vitest run`.

## Review Focus

- **Hésitation entre une commande et « aucune »** : l'option « aucune » apparaît comme « rien de tout ça », et la choisir donne la réponse habituelle de non-correspondance → test dans la tâche 2.
- **Réponse « 1 » sans choix en attente** : traitée comme un message ordinaire, pas comme un choix → test dans la tâche 4.
- **`/confirm` pendant un choix en attente** : message clair, rien d'exécuté → test dans la tâche 4.
- **Choix expiré (> 10 min)** : « 1 » n'exécute plus rien → test dans la tâche 4.
- **Exemple de commande absente du mapping domaine** (`/r` dans d'anciennes données) : rejeté par `to_laya_format`, pas étiqueté au hasard → test dans la tâche 3.

---

### Task 1: `router_service/router.py` — domaines, commandes, questions

**Files:**
- Modify: `router_service/router.py` (remplacer `COMMAND_CRITERIA` et `_INSTRUCTIONS`)
- Test: `router_service/test_router.py`

**Interfaces:**
- Produces : `DOMAINES: dict[str, str]`, `COMMANDES: dict[str, dict[str, str]]` (clé `"wiki"`/`"google"` → `{commande: description}`), `DOMAINE_UNIQUE: dict[str, str]` (`"web"→"/source"`, `"planification"→"/cron"`), `DOMAINE_DE: dict[str, str]` (commande → domaine), `QUESTIONS: dict` (questions Laya `domaine`, `commande_wiki`, `commande_google`), `LIBELLES: dict[str | None, str]`.

- [ ] **Step 1: tests**

```python
def test_domaines_et_commandes_couvrent_toutes_les_commandes_nl():
    from router_service.router import DOMAINE_DE, QUESTIONS, LIBELLES, WIKI_CMDS, GOG_CMDS, CRON_CMDS
    attendues = (WIKI_CMDS - {"/r"}) | GOG_CMDS | CRON_CMDS | {"/source"}
    assert set(DOMAINE_DE) == attendues
    assert set(QUESTIONS) == {"domaine", "commande_wiki", "commande_google"}
    assert set(QUESTIONS["domaine"]["criteria"]) == {"wiki", "google", "web", "planification", "aucune"}
    assert len(QUESTIONS["commande_wiki"]["criteria"]) == 9
    assert len(QUESTIONS["commande_google"]["criteria"]) == 6
    assert LIBELLES[None] == "rien de tout ça" and all(c in LIBELLES for c in DOMAINE_DE)


def test_aucune_question_ne_depasse_le_budget_laya():
    # head_max_len = 256 : au-delà, Laya tronque chaque option (constaté à 13 jetons sur 18 options).
    from router_service.router import QUESTIONS
    for q in QUESTIONS.values():
        assert sum(len(f"{k}: {v}".split()) for k, v in q["criteria"].items()) < 120
```
(`/source` est dans `WIKI_CMDS` aujourd'hui ; l'ensemble attendu reste juste.)

- [ ] **Step 2: échec** — `LAYA_CHECKPOINT=x python3 -m pytest router_service/test_router.py -q -k "domaines or budget"` → FAIL (ImportError).

- [ ] **Step 3: implémentation** — dans `router.py`, remplacer `COMMAND_CRITERIA` et `_INSTRUCTIONS` par :

```python
DOMAINES: dict[str, str] = {
    "wiki": "le wiki personnel : capturer, interroger, chercher, ingérer, relire, vérifier, supprimer ; et toute question de connaissance",
    "google": "les emails reçus (lire, chercher, lister, répondre), Google Drive, connexion au compte Google",
    "web": "lire une page web externe maintenant, sans la sauvegarder",
    "planification": "planifier un rappel, une commande ou une consigne, à une date ou de façon récurrente",
    "aucune": "action impossible pour Tiron (réserver, acheter, domotique, nouveau mail) ou simple conversation",
}
COMMANDES: dict[str, dict[str, str]] = {
    "wiki": {
        "/c": "capturer une note ou une URL dans le wiki",
        "/q": "poser une question (au wiki ou de connaissance générale), réponse synthétisée",
        "/ingest": "lancer l'ingestion des sources en attente",
        "/wikistatus": "connaître l'état de l'ingestion du wiki",
        "/tags": "lister les tags du wiki",
        "/kbupdate": "mettre à jour la base de connaissances du wiki",
        "/supprimer": "supprimer une page du wiki",
        "/relire": "obtenir la prochaine page du wiki à relire",
        "/verifie": "marquer une page du wiki comme vérifiée",
    },
    "google": {
        "/chercher": "rechercher des emails (sujet, expéditeur, période)",
        "/connecter": "connecter le compte Google",
        "/inbox": "lister les emails récents",
        "/drive": "rechercher dans Google Drive",
        "/repondre": "répondre à un email",
        "/lire": "lire un email précis (identifiant, le dernier reçu)",
    },
}
DOMAINE_UNIQUE: dict[str, str] = {"web": "/source", "planification": "/cron"}
DOMAINE_DE: dict[str, str] = {c: d for d, cs in COMMANDES.items() for c in cs} | \
    {c: d for d, c in DOMAINE_UNIQUE.items()}
QUESTIONS: dict = {
    "domaine": {"type": "choice", "instructions": "De quoi relève ce message adressé à Tiron ?",
                "criteria": DOMAINES},
    "commande_wiki": {"type": "choice", "instructions": "Quelle commande du wiki ce message déclenche-t-il ?",
                      "criteria": COMMANDES["wiki"]},
    "commande_google": {"type": "choice", "instructions": "Quelle commande Google ce message déclenche-t-il ?",
                        "criteria": COMMANDES["google"]},
}
LIBELLES: dict = {c: d for cs in COMMANDES.values() for c, d in cs.items()} | {
    "/source": "lire une page web sans la sauvegarder",
    "/cron": "planifier un rappel ou une tâche",
    None: "rien de tout ça",
}
```
Mettre à jour le docstring du module (deux niveaux, cf. spec 2026-10-06). Ne pas toucher à `WIKI_CMDS`, `GOG_CMDS`, `CRON_CMDS`, `embed_bge_m3`.

- [ ] **Step 4: vérifier** — les deux tests passent. Les autres tests qui importent `COMMAND_CRITERIA` ou `_INSTRUCTIONS` (`laya_classifier.py`, `to_laya_format.py`, leurs tests) échoueront : c'est attendu, les tâches 2 et 3 les migrent. Ne pas commiter avant la fin de la tâche 2.

---

### Task 2: `laya_classifier.py` et `server.py` — classement joint, seuil, `clarify`

**Files:**
- Modify: `router_service/laya_classifier.py`, `router_service/server.py`
- Test: `router_service/test_laya_classifier.py` (réécrit), `router_service/test_router.py` (tests NL adaptés)

**Interfaces:**
- Consumes : tâche 1.
- Produces :
  - `classement(answers: dict) -> list[tuple[str | None, float]]`, fonction pure, triée par probabilité décroissante (`None` = aucune) ;
  - `LayaClassifier.classer(message) -> list[tuple[str | None, float]]` ;
  - `server.SEUIL: float` ;
  - `route_message` → `{"status": "clarify", "options": [{"command": str | None, "label": str}, …2], "args": message}` sous le seuil.

- [ ] **Step 1: tests** — remplacer le contenu de `test_laya_classifier.py` :

```python
from router_service.laya_classifier import classement


def _ans(dom, wiki=None, google=None):
    a = {"domaine": {"probabilities": dom}}
    if wiki:
        a["commande_wiki"] = {"probabilities": wiki}
    if google:
        a["commande_google"] = {"probabilities": google}
    return a


def test_classement_joint_trie():
    r = classement(_ans({"wiki": 0.7, "google": 0.2, "web": 0.05, "planification": 0.03, "aucune": 0.02},
                        wiki={"/q": 0.9, "/c": 0.1}, google={"/lire": 0.6, "/chercher": 0.4}))
    assert r[0] == ("/q", 0.63) and r[1] == ("/c", 0.07)
    assert ("/lire", 0.12) in r and ("/source", 0.05) in r and ("/cron", 0.03) in r and (None, 0.02) in r
    assert [p for _, p in r] == sorted((p for _, p in r), reverse=True)


def test_classement_sans_probabilites_de_niveau_2():
    r = classement(_ans({"wiki": 0.1, "google": 0.0, "web": 0.0, "planification": 0.9, "aucune": 0.0}))
    assert r[0] == ("/cron", 0.9)
```

Dans `test_router.py`, remplacer `_FakeClassifier` et les tests NL (`test_nl_*`, `test_gog_command_*`, `test_wiki_command_from_slm_does_not_use_confidence_threshold`, `test_unknown_slash_command_still_reaches_slm`, `test_route_endpoint_end_to_end`) par la nouvelle interface :

```python
class _FakeClassifier:
    def __init__(self, ranking):
        self._ranking = ranking

    def classer(self, message):
        return self._ranking


def test_nl_au_dessus_du_seuil_decide(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier([("/q", 0.95), ("/c", 0.03)]))
    assert router_server.route_message("c koi SPLADE ?") == {"status": "ok", "command": "/q", "args": "c koi SPLADE ?"}


def test_nl_aucune_au_dessus_du_seuil(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier([(None, 0.97), ("/q", 0.02)]))
    assert router_server.route_message("il fait beau") == {"status": "no_match"}


def test_nl_sous_le_seuil_demande_precision(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier([("/lire", 0.45), ("/chercher", 0.40), (None, 0.1)]))
    r = router_server.route_message("le dernier mail de ma sœur")
    assert r["status"] == "clarify" and r["args"] == "le dernier mail de ma sœur"
    assert [o["command"] for o in r["options"]] == ["/lire", "/chercher"]
    assert r["options"][0]["label"].startswith("lire un email")


def test_nl_option_aucune_libellee(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier([("/q", 0.5), (None, 0.45)]))
    r = router_server.route_message("astuce tomates ?")
    assert r["options"][1] == {"command": None, "label": "rien de tout ça"}


def test_commande_inconnue_tapee_passe_par_laya(monkeypatch):
    called = {"n": 0}

    class _Counting:
        def classer(self, message):
            called["n"] += 1
            return [(None, 1.0)]

    monkeypatch.setattr(router_server, "_classifier", _Counting())
    router_server.route_message("/inconnu bla")
    assert called["n"] == 1
```
Le test `test_route_endpoint_end_to_end` garde sa structure, avec `_FakeClassifier([(None, 1.0)])`.

- [ ] **Step 2: échec** — `LAYA_CHECKPOINT=x python3 -m pytest router_service -q` → échecs sur les tests nouveaux et migrés.

- [ ] **Step 3: implémentation**

`laya_classifier.py` :
```python
"""Classifieur Laya à deux niveaux (domaine puis commande), une seule passe,
probabilités jointes — cf. docs/superpowers/specs/2026-10-06-routage-deux-niveaux-design.md."""
from __future__ import annotations

from router_service.router import COMMANDES, DOMAINE_UNIQUE, QUESTIONS


def classement(answers: dict) -> list[tuple[str | None, float]]:
    dom = answers["domaine"]["probabilities"]
    out: list[tuple[str | None, float]] = []
    for d, p_d in dom.items():
        if d == "aucune":
            out.append((None, round(p_d, 4)))
        elif d in DOMAINE_UNIQUE:
            out.append((DOMAINE_UNIQUE[d], round(p_d, 4)))
        elif d in COMMANDES:
            sous = answers.get(f"commande_{d}", {}).get("probabilities", {})
            for c, p_c in sous.items():
                out.append((c, round(p_d * p_c, 4)))
    return sorted(out, key=lambda x: -x[1])


class LayaClassifier:
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self._agent = self._load(checkpoint)

    def _load(self, checkpoint: str):
        import laya
        return laya.load(checkpoint)

    def classer(self, message: str) -> list[tuple[str | None, float]]:
        return classement(self._agent.predict(message, QUESTIONS)["answers"])
```

`server.py` :
- import : `from router_service.router import WIKI_CMDS, GOG_CMDS, CRON_CMDS, LIBELLES, embed_bge_m3` ;
- remplacer `SEUIL_GOG = 0.50` par :
```python
# Confiance jointe (domaine × commande) au-dessous de laquelle Tiron demande
# « 1 ou 2 » au lieu de deviner. Valeur provisoire, recalibrée à la tâche 6
# sur le jeu de validation (≥ 97 % de décisions justes, ≤ 15 % de questions).
SEUIL = 0.60
```
- dans `route_message`, remplacer le bloc à partir de `if _classifier is None:` par :
```python
    if _classifier is None:
        return {"status": "no_match"}
    ranking = _classifier.classer(message)
    command, p = ranking[0]
    if p >= SEUIL:
        if command is None:
            return {"status": "no_match"}
        # args : jamais généré, toujours le message brut (décision du 2026-09-28).
        return {"status": "ok", "command": command, "args": message}
    options = [{"command": c, "label": LIBELLES[c]} for c, _ in ranking[:2]]
    return {"status": "clarify", "options": options, "args": message}
```

- [ ] **Step 4: vérifier** — `LAYA_CHECKPOINT=x python3 -m pytest router_service -q` : tout passe, sauf les tests de `test_eval_laya.py` (migrés à la tâche 5). Lister les échecs restants pour confirmer qu'il ne reste que ceux-là.

- [ ] **Step 5: commit (tâches 1 et 2)** — `git commit -m "feat(routeur): Laya à deux niveaux, classement joint, demande de précision sous le seuil"`

---

### Task 3: `to_laya_format.py` — questions par domaine, validation stratifiée

**Files:**
- Modify: `gen_corpus/to_laya_format.py`
- Test: `gen_corpus/test_to_laya_format.py` (python3 système, torch requis)

**Interfaces:**
- Consumes : tâche 1 (`QUESTIONS`, `DOMAINE_DE`).
- Produces : lignes Laya `{"state", "questions": {"domaine", [commande_<d>]}, "answers": {"domaine": d, [commande_<d>: c]}}` ; `reformat(...)` découpe la validation par commande, avec `min(40, n // 4)` exemples par commande.

- [ ] **Step 1: tests** — remplacer les tests de `test_to_laya_format.py` qui dépendent de `COMMAND_CRITERIA` (dont celui qui attend encore `/r`, en échec depuis le 30/09) par :

```python
def test_ligne_wiki_porte_deux_questions():
    from gen_corpus.to_laya_format import _to_laya_row
    r = _to_laya_row("c koi SPLADE ?", "/q")
    assert set(r["questions"]) == {"domaine", "commande_wiki"}
    assert r["answers"] == {"domaine": "wiki", "commande_wiki": "/q"}


def test_ligne_planification_et_aucune_une_seule_question():
    from gen_corpus.to_laya_format import _to_laya_row
    assert _to_laya_row("rappel 8h", "/cron")["answers"] == {"domaine": "planification"}
    assert _to_laya_row("allume la lumière", "aucune")["answers"] == {"domaine": "aucune"}


def test_commande_hors_mapping_rejetee():
    import pytest
    from gen_corpus.to_laya_format import _to_laya_row
    with pytest.raises(KeyError):
        _to_laya_row("cherche X", "/r")


def test_validation_stratifiee(tmp_path):
    import json
    from gen_corpus.to_laya_format import reformat
    src = tmp_path / "train.jsonl"
    lignes = []
    for cmd, n in (("/q", 200), ("/cron", 100), ("/lire", 20)):
        for i in range(n):
            lignes.append(json.dumps({"messages": [{"role": "system", "content": "s"},
                                                   {"role": "user", "content": f"{cmd} msg {i}"},
                                                   {"role": "assistant", "content": json.dumps({"command": cmd, "args": ""})}]}))
    src.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    reformat(src, tmp_path / "t.jsonl", tmp_path / "v.jsonl")
    from collections import Counter
    val = Counter(json.loads(l)["state"].split()[0] for l in open(tmp_path / "v.jsonl"))
    assert val == {"/q": 40, "/cron": 25, "/lire": 5}
```

- [ ] **Step 2: échec** — `LAYA_CHECKPOINT=x python3 -m pytest gen_corpus/test_to_laya_format.py -q` → FAIL.

- [ ] **Step 3: implémentation**

```python
from router_service.router import DOMAINE_DE, QUESTIONS  # noqa: E402


def _to_laya_row(user_text: str, command: str) -> dict:
    domaine = "aucune" if command == "aucune" else DOMAINE_DE[command]  # KeyError si commande inconnue
    questions = {"domaine": QUESTIONS["domaine"]}
    answers = {"domaine": domaine}
    if f"commande_{domaine}" in QUESTIONS:
        questions[f"commande_{domaine}"] = QUESTIONS[f"commande_{domaine}"]
        answers[f"commande_{domaine}"] = command
    return {"state": user_text, "questions": questions, "answers": answers}
```
Dans `reformat`, regrouper les lignes dédupliquées par commande d'origine (garder `(row, command)`). Mélanger chaque groupe avec `random.Random(seed)` et placer `min(40, len(groupe) // 4)` lignes en validation, le reste en entraînement. Retirer le paramètre `val_ratio`, devenu sans objet. Mettre à jour le docstring.

- [ ] **Step 4: vérifier** — `LAYA_CHECKPOINT=x python3 -m pytest gen_corpus/test_to_laya_format.py -q` → PASS.

- [ ] **Step 5: commit** — `git commit -m "feat(corpus): format Laya à deux niveaux, validation stratifiée par commande"`

---

### Task 4: derisk-deleg — demande de précision « 1 ou 2 »

**Files:**
- Modify: `derisk-deleg/src/index.ts` (`callRouter` l.14-37, type `Pending`, `/confirm`, `/annuler`, routage l.700-715)
- Test: `derisk-deleg/src/index.test.ts`

**Interfaces:**
- Consumes : `{"status": "clarify", "options": [{"command": string | null, "label": string}], "args": string}` (tâche 2).
- Produces : `pending = {kind: "choix", options, args, ts}`.

- [ ] **Step 1: tests** (nouveau bloc `describe("before_agent_reply — demande de précision")`, même harnais que les tests `/cron`) :

```ts
describe("before_agent_reply — demande de précision", () => {
  function routerClarify() {
    vi.stubGlobal("fetch", vi.fn(async (_u: any, init: any) => {
      const msg = JSON.parse(init.body).message;
      if (msg === "le dernier mail de ma sœur") {
        return { ok: true, json: async () => ({ status: "clarify", args: msg, options: [
          { command: "/lire", label: "lire un email précis (identifiant, le dernier reçu)" },
          { command: null, label: "rien de tout ça" }] }) };
      }
      return { ok: true, json: async () => ({ status: "no_match" }) };
    }));
  }

  it("propose les deux options numérotées", async () => {
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    expect(r.reply.text).toBe("Je ne suis pas sûr :\n1) lire un email précis (identifiant, le dernier reçu)\n2) rien de tout ça\nRépondez 1 ou 2.");
  });

  it("« 1 » exécute la commande avec le message d'origine", async () => {
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    await hooks["before_agent_reply"].handler({ cleanedBody: "1" }, {});
    expect(run).toHaveBeenCalledTimes(1);
    expect(run.mock.calls[0][0].message).toContain("le dernier mail de ma sœur");
  });

  it("« 2 » = rien de tout ça : réponse de non-correspondance", async () => {
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "2" }, {});
    expect(r.reply.text).toMatch(/^Je n'ai pas cette information/);
    expect(run).not.toHaveBeenCalled();
  });

  it("un autre message annule le choix et est traité normalement", async () => {
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
    await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "bonjour" }, {});
    expect(r.reply.text).toMatch(/^Je n'ai pas cette information/);
    const r2 = await hooks["before_agent_reply"].handler({ cleanedBody: "1" }, {});
    expect(r2.reply.text).toMatch(/^Je n'ai pas cette information/); // « 1 » n'est plus un choix
  });

  it("« 1 » sans choix en attente : message ordinaire", async () => {
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "1" }, {});
    expect(r.reply.text).toMatch(/^Je n'ai pas cette information/);
    expect(run).not.toHaveBeenCalled();
  });

  it("choix expiré après 10 min", async () => {
    vi.useFakeTimers();
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    vi.advanceTimersByTime(11 * 60 * 1000);
    await hooks["before_agent_reply"].handler({ cleanedBody: "1" }, {});
    expect(run).not.toHaveBeenCalled();
  });

  it("/confirm pendant un choix : rien n'est exécuté", async () => {
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" }, {});
    expect(r.reply.text).toBe("Rien à confirmer : répondez 1 ou 2 au choix proposé.");
    expect(run).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: échec** — `npx vitest run src/index.test.ts -t "précision"` → FAIL.

- [ ] **Step 3: implémentation**

- `callRouter` : ajouter au type de retour `| { status: "clarify"; options: { command: string | null; label: string }[]; args: string }`, et avant la ligne `return data.status === "ok"` :
```ts
    if (data.status === "clarify" && Array.isArray(data.options)) {
      return { status: "clarify", options: data.options, args: data.args ?? "" };
    }
```
- `Pending` : ajouter `| { kind: "choix"; options: { command: string | null; label: string }[]; args: string; ts: number }`.
- `/confirm` : juste après le test d'expiration, ajouter :
```ts
        if (pending.kind === "choix") {
          return { handled: true, reply: { text: "Rien à confirmer : répondez 1 ou 2 au choix proposé." } };
        }
```
- `/annuler` : ajouter un cas `pending.kind === "choix"` → `pending = null`, réponse « Choix abandonné. ».
- Routage : remplacer `const routed = await callRouter(text);` par :
```ts
      // Réponse « 1 »/« 2 » à une demande de précision (spec 2026-10-06) ; tout
      // autre message annule le choix et suit le chemin normal.
      let routed: Awaited<ReturnType<typeof callRouter>>;
      const choix = pending?.kind === "choix" && Date.now() - pending.ts <= PENDING_TTL_MS ? pending : null;
      if (pending?.kind === "choix") pending = null;
      const num = text.trim().match(/^([12])$/);
      if (choix && num) {
        const opt = choix.options[Number(num[1]) - 1];
        routed = opt && opt.command
          ? { status: "ok", command: opt.command, args: choix.args }
          : { status: "no_match" };
      } else {
        routed = await callRouter(text);
      }
      if (routed.status === "clarify") {
        pending = { kind: "choix", options: routed.options.slice(0, 2), args: routed.args, ts: Date.now() };
        const lignes = routed.options.slice(0, 2).map((o, i) => `${i + 1}) ${o.label}`).join("\n");
        return { handled: true, reply: { text: `Je ne suis pas sûr :\n${lignes}\nRépondez 1 ou 2.` } };
      }
```
Le reste (`unavailable`, `answer`, `no_match`, dispatch) est inchangé. `explicitCommand` reste faux pour « 1 », donc les écritures choisies passent par `/confirm`, comme toute commande déduite.

- [ ] **Step 4: vérifier** — `npm test && npm run build` → PASS, compilation sans erreur.

- [ ] **Step 5: commit** — `git commit -m "feat(derisk-deleg): demande de précision « 1 ou 2 » quand Laya hésite"` (avec `dist/`).

---

### Task 5: évaluation — métriques à deux niveaux, seuil, jeu manuel

**Files:**
- Modify: `router_service/eval_laya.py`, `router_service/test_eval_laya.py`
- Create: `router_service/test_set_manuel.jsonl`

**Interfaces:**
- Consumes : `LayaClassifier.classer` (tâche 2), format de validation (tâche 3).
- Produces :
  - `metriques(classements: list[tuple[set[str], list]], seuil: float) -> dict`, avec les clés `decide_juste` (part des décisions justes parmi celles prises sans question), `taux_question`, `juste_ou_question` (décision juste, ou question dont les deux options contiennent une réponse acceptable), `par_commande` (`{commande: taux}`) ;
  - `choisir_seuil(classements) -> float` : le plus petit seuil de la grille 0.30…0.95 (pas de 0.05) qui donne `decide_juste ≥ 0.97` et `taux_question ≤ 0.15`, sinon celui qui maximise `juste_ou_question` ;
  - la grille de seuils est construite avec `round(0.30 + 0.05 * k, 2)`, sinon le flottant 0.6000000000000001 fait échouer `== 0.6` ;
  - CLI : `python3 -m router_service.eval_laya <checkpoint> [--val …] [--manuel router_service/test_set_manuel.jsonl]`. Elle affiche le seuil choisi sur la validation, puis les métriques sur la validation et sur le jeu manuel.

- [ ] **Step 1: tests** — remplacer `test_eval_laya.py` :

```python
from router_service.eval_laya import choisir_seuil, metriques


def test_metriques():
    cl = [({"/q"}, [("/q", 0.9), ("/c", 0.1)]),            # décidé juste
          ({"/lire"}, [("/chercher", 0.5), ("/lire", 0.4)]),  # question pertinente
          ({"aucune"}, [("/q", 0.8), (None, 0.2)]),       # décidé faux
          ({"/c"}, [("/q", 0.5), ("/tags", 0.3)])]        # question non pertinente
    m = metriques(cl, 0.6)
    assert m["decide_juste"] == 0.5 and m["taux_question"] == 0.5 and m["juste_ou_question"] == 0.5
    assert m["par_commande"]["/q"] == 1.0 and m["par_commande"]["aucune"] == 0.0


def test_choisir_seuil():
    cl = [({"/q"}, [("/q", 0.9), ("/c", 0.1)])] * 18 + [({"/c"}, [("/q", 0.55), ("/c", 0.45)])] * 2
    assert choisir_seuil(cl) == 0.6
```
(`aucune` dans l'ensemble attendu correspond à `None` dans le classement.)

- [ ] **Step 2: échec** → FAIL (ImportError).

- [ ] **Step 3: implémentation** de `metriques`, `choisir_seuil` et de la CLI. Pour le jeu de validation, l'ensemble attendu est `{answers["commande_<d>"]}` si la question existe, sinon `{"/source"}`, `{"/cron"}` ou `{"aucune"}` selon `answers["domaine"]`. Pour le jeu manuel, c'est `set(ligne["attendu"])`.

- [ ] **Step 4: jeu de test manuel** — créer `router_service/test_set_manuel.jsonl`, une ligne JSON par message, `{"text": …, "attendu": [...]}`. Les messages sont écrits à la main, sans DeepSeek. Contenu complet :

```
{"text": "c koi le SPLADE déjà", "attendu": ["/q"]}
{"text": "qu'est-ce que dit mon wiki sur Maître Eckhart ?", "attendu": ["/q"]}
{"text": "astuce pr tomates ?", "attendu": ["/q"]}
{"text": "comment on fait une béchamel", "attendu": ["/q"]}
{"text": "rappelle moi ce que j'ai noté sur la domestication du chien", "attendu": ["/q"]}
{"text": "garde ça https://arxiv.org/abs/2401.01234", "attendu": ["/c"]}
{"text": "note : rdv chez le notaire reporté au 12, à garder dans le wiki", "attendu": ["/c"]}
{"text": "#lecture https://lemonde.fr/article-ia ajoute au wiki", "attendu": ["/c"]}
{"text": "capture cette page stp https://exemple.org/recette", "attendu": ["/c"]}
{"text": "mets dans le wiki que le code du portail a changé", "attendu": ["/c"]}
{"text": "lance l'ingestion", "attendu": ["/ingest"]}
{"text": "ingère ce qui attend", "attendu": ["/ingest"]}
{"text": "vas-y traite les captures en attente", "attendu": ["/ingest"]}
{"text": "ingestion stp", "attendu": ["/ingest"]}
{"text": "où en est le wiki ?", "attendu": ["/wikistatus"]}
{"text": "l'ingestion est finie ?", "attendu": ["/wikistatus", "/ingest"]}
{"text": "combien de fichiers en attente dans le wiki", "attendu": ["/wikistatus"]}
{"text": "état du wiki", "attendu": ["/wikistatus"]}
{"text": "quels tags j'ai dans le wiki", "attendu": ["/tags"]}
{"text": "liste des tags", "attendu": ["/tags"]}
{"text": "montre moi les étiquettes du wiki", "attendu": ["/tags"]}
{"text": "mets à jour la base de connaissances", "attendu": ["/kbupdate"]}
{"text": "maj kb", "attendu": ["/kbupdate"]}
{"text": "refais la base du wiki", "attendu": ["/kbupdate"]}
{"text": "supprime la page c-test-bidon", "attendu": ["/supprimer"]}
{"text": "vire la page src-brouillon-2024 du wiki", "attendu": ["/supprimer"]}
{"text": "efface c-domestication-des-chiens", "attendu": ["/supprimer"]}
{"text": "une page à relire ?", "attendu": ["/relire"]}
{"text": "donne-moi la prochaine page à vérifier", "attendu": ["/relire"]}
{"text": "on fait un peu de relecture", "attendu": ["/relire"]}
{"text": "c-maitre-eckhart c'est vérifié", "attendu": ["/verifie"]}
{"text": "marque src-article-ia comme relue", "attendu": ["/verifie"]}
{"text": "valide la page c-splade", "attendu": ["/verifie"]}
{"text": "cherche les mails de la banque", "attendu": ["/chercher"]}
{"text": "j'ai reçu un mail de Paul sur le chantier ?", "attendu": ["/chercher"]}
{"text": "retrouve le mail de l'école de la semaine dernière", "attendu": ["/chercher"]}
{"text": "mails sur la facture EDF", "attendu": ["/chercher"]}
{"text": "connecte mon compte google", "attendu": ["/connecter"]}
{"text": "il faut que je réautorise google", "attendu": ["/connecter"]}
{"text": "accès gmail stp", "attendu": ["/connecter"]}
{"text": "quoi de neuf dans ma boîte ?", "attendu": ["/inbox"]}
{"text": "mes 5 derniers mails", "attendu": ["/inbox"]}
{"text": "des nouveaux mails ?", "attendu": ["/inbox"]}
{"text": "trouve le devis dans le drive", "attendu": ["/drive"]}
{"text": "y a un doc budget 2026 sur google drive ?", "attendu": ["/drive"]}
{"text": "cherche le pdf du contrat dans mes documents google", "attendu": ["/drive"]}
{"text": "réponds au mail 18ab3f2 que c'est d'accord pour jeudi", "attendu": ["/repondre"]}
{"text": "réponds à Paul que je passe demain", "attendu": ["/repondre"]}
{"text": "prépare une réponse au dernier mail de l'école", "attendu": ["/repondre"]}
{"text": "lis le mail 18ab3f2", "attendu": ["/lire"]}
{"text": "ouvre le dernier mail", "attendu": ["/lire", "/inbox"]}
{"text": "le dernier mail de ma sœur", "attendu": ["/lire", "/chercher"]}
{"text": "montre-moi le contenu de 7f3e2a1", "attendu": ["/lire"]}
{"text": "lis cette page et dis-moi l'essentiel https://exemple.org/etude", "attendu": ["/source"]}
{"text": "lis ça https://blog.exemple.fr/post", "attendu": ["/source", "/c"]}
{"text": "c'est quoi ce lien ? https://exemple.org/x", "attendu": ["/source"]}
{"text": "résume moi https://exemple.org/rapport sans le garder", "attendu": ["/source"]}
{"text": "rappelle-moi demain 9h d'appeler le garagiste", "attendu": ["/cron"]}
{"text": "chaque lundi à 8h donne-moi l'état du wiki", "attendu": ["/cron"]}
{"text": "dans 20 min : sortir le gâteau", "attendu": ["/cron"]}
{"text": "tous les soirs à 21h rappel pilulier par sms", "attendu": ["/cron"]}
{"text": "le 3 novembre à 14h rappelle-moi le contrôle technique", "attendu": ["/cron"]}
{"text": "/cron list", "attendu": ["/cron"]}
{"text": "réserve-moi une table pour samedi", "attendu": ["aucune"]}
{"text": "allume la lumière du salon", "attendu": ["aucune"]}
{"text": "commande des croquettes pour le chat", "attendu": ["aucune"]}
{"text": "écris un nouveau mail à la mairie", "attendu": ["aucune"]}
{"text": "merci !", "attendu": ["aucune"]}
{"text": "tu viens dimanche ?", "attendu": ["aucune"]}
{"text": "bonne nuit Tiron", "attendu": ["aucune"]}
{"text": "achète-moi un billet de train pour Lyon", "attendu": ["aucune"]}
```
(71 messages : chaque commande en a au moins 3, et les cas ambigus ont plusieurs réponses acceptables. L'utilisateur peut ajouter les siens. `/cron list` est une commande tapée : la CLI l'évalue via `route_message`, qui la traite sans Laya.)

La CLI évalue le jeu manuel avec `route_message`, pas directement avec `classer`, pour que les commandes tapées suivent le vrai chemin. Une réponse `ok` est juste si `command ∈ attendu`, `no_match` si `"aucune" ∈ attendu`, et `clarify` est pertinente si l'une des deux options est attendue.

- [ ] **Step 5: vérifier** — `LAYA_CHECKPOINT=x python3 -m pytest router_service -q` → PASS (suite complète).

- [ ] **Step 6: commit** — `git commit -m "feat(routeur): évaluation à deux niveaux, choix du seuil, jeu de test manuel"`

---

### Task 6: données — hors périmètre et culture générale, puis réentraînement (**accords requis**)

**Files:**
- Modify: `gen_corpus/chaine.py` (descriptions `out_of_scope` et `wiki_query`), `gen_corpus/intentions.json`, `gen_corpus/prompt-init.txt`, `gen_corpus/GEPAPrompt.txt`, `gen_corpus/gepa_program.json` (ligne `null` et ligne `/q`), `gen_corpus/corpus.jsonl`, jeux reformatés
- Test: `gen_corpus/tests/test_chaine.py`

- [ ] **Step 1: test**
```python
def test_culture_generale_releve_de_q():
    from chaine import INTENTION_DESCRIPTIONS
    assert "culture générale" in INTENTION_DESCRIPTIONS["wiki_query"]
    assert "→ /q" in INTENTION_DESCRIPTIONS["out_of_scope"]
    import json
    intents = {i["intention"]: i for i in json.load(open(Path(__file__).parent.parent / "intentions.json"))}
    assert "question_culture_generale" in intents["wiki_query"]["variantes"]
    assert "aide_generale" not in intents["out_of_scope"]["variantes"]
```
- [ ] **Step 2: échec**, puis **implémentation** :
  - `wiki_query` : « poser une question au wiki ou de culture générale, réponse synthétisée par un LLM ».
  - `out_of_scope` : retirer « culture générale » de la liste et ajouter « une question de connaissance → /q ».
  - `intentions.json` : `wiki_query` reçoit la variante `question_culture_generale`. Dans `out_of_scope`, `aide_generale` est remplacée par `achat_reservation` et `domotique`.
  - Ligne `null` de l'instruction, dans les trois fichiers : retirer « culture générale » et ajouter « une question de connaissance → /q ».
  - Vérifier : test vert, suite gen_corpus verte. Commit.
- [ ] **Step 3: [accord DeepSeek] régénération ciblée**
  - Régénérer `out_of_scope` (190) dans un fichier d'intentions réduit.
  - Générer 60 exemples `wiki_query` avec la seule variante `question_culture_generale`.
  - Remplacer les `out_of_scope` dans `corpus.jsonl` et ajouter les 60 exemples `/q`.
  - Relancer `to_lora_format.py` depuis `gen_corpus/`, puis `python3 gen_corpus/to_laya_format.py` depuis la racine.
  - Vérifier qu'aucune commande n'a moins de 25 exemples en validation (lot de 2 900 environ). Commit.
- [ ] **Step 4: [utilisateur] Kaggle**
  - Nouvelles versions de `laya_train.jsonl` et `laya_val.jsonl` dans le dataset.
  - Vérifier que la cellule de préparation boucle bien sur **toutes** les questions de chaque ligne (`for qid, q in questions.items()`).
  - Remplacer la cellule d'évaluation par la cellule ci-dessous, puis publier sous le même nom.
```python
# Évaluation à deux niveaux (spec 2026-10-06) — à exécuter après la publication.
import json, glob, collections, laya
agent = laya.load("Mauceric/laya-tiron-router")
val = glob.glob("/kaggle/input/*/laya_val.jsonl")[0]
ok, tot = collections.Counter(), collections.Counter()
for line in open(val):
    r = json.loads(line)
    ans = agent.predict(r["state"], r["questions"])["answers"]
    for qid, gold in r["answers"].items():
        tot[qid] += 1
        ok[qid] += ans[qid]["choice"] == gold
for qid in tot:
    print(f"{qid:16} {ok[qid]}/{tot[qid]} = {ok[qid] / tot[qid]:.3f}")
```
- [ ] **Step 5: évaluation sur sanroque**
  - Lancer `python3 -m router_service.eval_laya Mauceric/laya-tiron-router --manuel router_service/test_set_manuel.jsonl` en arrière-plan, avec `PYTHONPATH=/home/mauceric/Secretarius`, et surveiller que le processus Python tourne vraiment.
  - Reporter le seuil choisi dans `SEUIL` (`server.py`), avec en commentaire la mesure qui l'a fixé.
  - Appliquer les critères de la spec : sur le jeu manuel, au moins 90 % « juste ou question pertinente », aucune commande sous 80 %, au plus 15 % de questions. S'ils sont tenus, commit. Sinon, analyser les erreurs et revenir vers l'utilisateur avant toute nouvelle génération.

---

### Task 7: bascule et vérification (**accord requis pour les redémarrages**)

- [ ] Réinstaller derisk-deleg dans `~/.openclaw/extensions/` (procédure habituelle).
- [ ] **[accord]** Redémarrer `systemctl --user restart tiron-router.service openclaw-gateway.service`.
- [ ] Vérifier par HTTP :
  - `curl -s localhost:8999/route -d '{"message":"le dernier mail de ma sœur"}'` → `ok` ou `clarify` cohérent ;
  - `curl -s localhost:8999/route -d '{"message":"rappelle-moi demain 9h de x"}'` → `/cron`.
- [ ] Vérifier sur Telegram, avec l'utilisateur : un message clair, un message ambigu (répondre « 1 »), un message hors périmètre.
- [ ] Mettre à jour `amorçage/faits.md` (FAQ : « Tiron demande 1 ou 2 quand il hésite ») et `docs/components/` (fiche routeur). Commit.
