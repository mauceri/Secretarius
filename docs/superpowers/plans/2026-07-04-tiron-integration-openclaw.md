# Intégration du routeur Tiron local dans OpenClaw — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Câbler le routeur SLM (BGE-M3 + adaptateur LoRA unique sur phi-4-mini) dans OpenClaw sur sanroque, en remplaçant le tool-calling natif de `main` par une interception déterministe dans `derisk-deleg`.

**Architecture:** Un service Python HTTP persistant expose `POST /route {message}` : il appelle un unique adaptateur LoRA (via llama-server ROCm) pour extraire `{command, args}`, et calcule en parallèle une classification BGE-M3 à 3 centroïdes servant uniquement de portail de confiance sur les commandes gog. Le hook `before_agent_reply` de `derisk-deleg` interroge ce service et dispatche directement vers les fonctions de délégation existantes, sans jamais solliciter le modèle de `main`.

**Tech Stack:** Python 3 (torch, transformers, peft — déjà utilisés dans `lora_slm/`), llama.cpp (`build-rocm/bin/llama-server`), TypeScript (plugin OpenClaw `derisk-deleg`, vitest), systemd `--user`.

## Global Constraints

- Spec de référence : `docs/superpowers/specs/2026-07-03-tiron-integration-openclaw-design.md` (validée par l'utilisateur, y compris révision 2026-07-04 adaptateur unique).
- Binaire llama-server : `~/llama.cpp/build-rocm/bin/llama-server`, jamais `build/bin/` (non lié ROCm/HIP, vérifié).
- Un seul adaptateur LoRA actif en permanence (`--lora`), pas de hot-swap.
- `SEUIL_GOG = 0.50`, `T_SOFTMAX = 0.05` (valeurs de `router_3way.py`, ne pas changer sans nouvelle mesure).
- Les 3 centroïdes (wiki/gog/hors_perimetre) restent nécessaires au calcul même si un seul sert de portail — ne jamais retirer le centroïde hors_perimetre (mesuré : faux accepts gog 1→18 sans lui).
- `/repondre` doit **toujours** passer par la logique de mise en attente existante de `derisk-deleg` (jamais de délégation directe) — c'est la seule commande gog qui écrit réellement (après `/confirm`).
- Aucun repli cloud automatique en cas de panne (service routeur ou llama-server indisponible) — message déterministe uniquement.
- Ne pas toucher `Secretarius/CLAUDE.md` ni les deux backends `llama.cpp`/`Ollama` de `expression_extractor.py`/`llm_ollama.py` (hors périmètre, réglé par les instructions du projet).

---

### Task 1: Réentraîner l'adaptateur unique et l'évaluer

**Files:**
- Create: `lora_slm/checkpoints/tiron-unified/` (sortie training, généré par `lora_train.py`)
- Create: `lora_slm/tiron-unified-lora-f16.gguf` (adaptateur GGUF converti)
- Create: `gen_corpus/eval_adapter.py`
- Test: `gen_corpus/eval_adapter.py` s'auto-vérifie via son mode `--check` (voir Step 4)

**Interfaces:**
- Consomme : `gen_corpus/corpus_lora_train.jsonl` (1798 exemples, colonnes `messages`, déjà réaligné sur les 10 commandes réelles + `null`, aucune modification nécessaire), `gen_corpus/corpus_lora_eval.jsonl` (198 exemples tenus à l'écart).
- Produit : `lora_slm/tiron-unified-lora-f16.gguf`, consommé par la Task 2 (`--lora` de llama-server) et le service routeur (Task 3, pour construire son prompt de test local si besoin).

- [ ] **Step 1: Lancer l'entraînement**

Reprend les hyperparamètres validés (mémoire `project_lora_slm_session_20260630` : LR 2e-4, 6 epochs, R16 — confirmés dans `lora_slm/checkpoints/tiron/adapter_config.json`, checkpoint aujourd'hui obsolète car entraîné avant le réalignement du corpus du 2026-07-02).

```bash
cd ~/Secretarius/lora_slm
mkdir -p /home/mauceric/lora_slm/checkpoints/tiron-unified
./lenv/bin/python lora_train.py \
  --model_path /home/mauceric/Modèles/phi4 \
  --data_file /home/mauceric/Secretarius/gen_corpus/corpus_lora_train.jsonl \
  --output_dir /home/mauceric/lora_slm/checkpoints/tiron-unified \
  --epochs 6 \
  --lr 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --log_file /home/mauceric/lora_slm/checkpoints/tiron-unified/training.log
```

Note : le script `lora_train.py` vit dans `Secretarius/lora_slm/` (dépôt), mais les
checkpoints volumineux vont dans `/home/mauceric/lora_slm/checkpoints/` (hors
dépôt, convention déjà en place pour `checkpoints/tiron`/`wiki`/`gog`). Utiliser
le venv `Secretarius/lora_slm/lenv/bin/python` (a `torch`+ROCm/HIP, `peft`,
`transformers` — le `python3` système ne les a pas).

**RÉEL (2026-07-04) : lancé en arrière-plan, PID 1896028, ~5h attendues**
(d'après les horodatages du précédent `checkpoints/tiron/checkpoint-*`, ~83s/pas
× 216 pas). Ne pas relancer — vérifier `tail -f
/home/mauceric/lora_slm/checkpoints/tiron-unified/training.log` avant de
redémarrer quoi que ce soit.

Expected : le script affiche `trainable_params` puis la perte décroît régulièrement dans `checkpoints/tiron-unified/training.log` (comparer l'allure à `checkpoints/tiron/training.log`, perte finale attendue < 0,2).

- [ ] **Step 2: Convertir en GGUF LoRA autonome**

```bash
cd ~/llama.cpp
python convert_lora_to_gguf.py \
  --base /home/mauceric/Modèles/phi4 \
  --outfile /home/mauceric/lora_slm/tiron-unified-lora-f16.gguf \
  /home/mauceric/lora_slm/checkpoints/tiron-unified
```

Expected : fichier `tiron-unified-lora-f16.gguf` créé, taille proche de 35 Mo (cohérent avec les précédents `wiki-lora-f16.gguf`/`gog-lora-f16.gguf`, même rang R16).

- [ ] **Step 3: Écrire le script d'évaluation**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Évalue un adaptateur LoRA unique sur corpus_lora_eval.jsonl : lance
llama-server avec l'adaptateur donné, interroge chaque exemple, compare la
commande extraite à la vérité terrain."""
import argparse, json, subprocess, sys, time, urllib.request

EVAL_PATH = "/home/mauceric/Secretarius/gen_corpus/corpus_lora_eval.jsonl"
SYSTEM_ROUTE = ('Routeur de commandes Tiron. Pour chaque message, répondre '
                'uniquement avec un objet JSON : {"command": "/commande" ou '
                'null, "args": "arguments bruts ou chaîne vide"}.')


def call_llm(base_url, msg, max_tokens=60):
    body = {"messages": [{"role": "system", "content": SYSTEM_ROUTE},
                         {"role": "user", "content": msg}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(base_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    return d["choices"][0]["message"]["content"].strip()


def wait_ready(base_url, timeout_s=60):
    for _ in range(timeout_s):
        try:
            urllib.request.urlopen(base_url + "/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("llama-server non prêt après {}s".format(timeout_s))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8996")
    p.add_argument("--check", action="store_true",
                   help="Vérifie seulement le parsing JSON sur 3 exemples (fumée).")
    args = p.parse_args()

    wait_ready(args.base_url)
    rows = [json.loads(l) for l in open(EVAL_PATH) if l.strip()]
    if args.check:
        rows = rows[:3]

    ok = 0
    for r in rows:
        msg = r["messages"][-2]["content"]
        expected = json.loads(r["messages"][-1]["content"]).get("command")
        out = call_llm(args.base_url, msg)
        try:
            got = json.loads(out).get("command")
        except Exception:
            got = "<JSON invalide>"
        good = got == expected
        ok += good
        if not good:
            print(f"!! {msg!r} attendu={expected!r} obtenu={got!r}")

    print(f"=== {ok}/{len(rows)} corrects ({100*ok/len(rows):.1f}%) ===")
    if not args.check and ok / len(rows) < 0.90:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Enregistrer dans `gen_corpus/eval_adapter.py`.

- [ ] **Step 4: Lancer llama-server sur l'adaptateur unique et vérifier (fumée)**

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 nohup ~/llama.cpp/build-rocm/bin/llama-server \
  -m /home/mauceric/Modèles/Phi-4-mini-instruct-Q6_K.gguf \
  --lora /home/mauceric/lora_slm/tiron-unified-lora-f16.gguf \
  -c 2048 -ngl 99 --host 127.0.0.1 --port 8996 \
  > /tmp/eval_server.log 2>&1 &
sleep 5
python gen_corpus/eval_adapter.py --base-url http://127.0.0.1:8996 --check
```

Expected : `=== 3/3 corrects ===` (ou proche — un échec isolé sur 3 exemples n'est pas bloquant, mais un JSON invalide indique un problème de conversion à corriger avant de continuer).

- [ ] **Step 5: Évaluation complète**

```bash
python gen_corpus/eval_adapter.py --base-url http://127.0.0.1:8996
kill %1  # arrête le llama-server de test
```

Expected : `=== N/198 corrects (≥90.0%) ===`, sortie 0. Si < 90%, ne pas continuer vers les tâches suivantes — réexaminer les hyperparamètres (`--epochs`, `--lr`) avant de reprendre.

- [ ] **Step 6: Commit**

```bash
cd ~/Secretarius
git add gen_corpus/eval_adapter.py
git commit -m "feat(lora): script d'évaluation de l'adaptateur unique tiron"
```

(Les fichiers `.gguf`/checkpoints restent hors dépôt, comme les adaptateurs précédents.)

---

### Task 2: Reconfigurer le service llama-server (ROCm + adaptateur unique)

**Files:**
- Modify: `/home/mauceric/.config/systemd/user/slm-llama_cpp.service`

**Interfaces:**
- Consomme : `lora_slm/tiron-unified-lora-f16.gguf` (Task 1), base `phi4` déjà présente localement.
- Produit : endpoint `http://127.0.0.1:8998/v1/chat/completions` avec l'adaptateur unique actif, consommé par le service routeur (Task 3).

- [ ] **Step 1: Modifier l'unité systemd**

Fichier actuel (`ExecStart` pointe `build/bin`, CPU pur, modèle `tiron-router-Q6_K.gguf` obsolète) :

```ini
[Unit]
Description=llama.cpp inference server
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/mauceric/llama.cpp/build/bin

ExecStart=/home/mauceric/llama.cpp/build/bin/llama-server \
    -m /home/mauceric/Modèles/tiron-router-Q6_K.gguf \
    -c 4096 \
    -ngl 32 \
    --host 0.0.0.0 \
    --port 8998 \
    --jinja

Restart=on-failure
RestartSec=5

Environment=HSA_OVERRIDE_GFX_VERSION=10.3.0

[Install]
WantedBy=default.target
```

Remplacer par :

```ini
[Unit]
Description=llama.cpp inference server (Tiron, adaptateur unique ROCm)
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/mauceric/llama.cpp/build-rocm/bin

ExecStart=/home/mauceric/llama.cpp/build-rocm/bin/llama-server \
    -m /home/mauceric/Modèles/Phi-4-mini-instruct-Q6_K.gguf \
    --lora /home/mauceric/lora_slm/tiron-unified-lora-f16.gguf \
    -c 2048 \
    -ngl 99 \
    --host 127.0.0.1 \
    --port 8998 \
    --jinja

Restart=on-failure
RestartSec=5

Environment=HSA_OVERRIDE_GFX_VERSION=10.3.0

[Install]
WantedBy=default.target
```

Notes sur les changements : `build-rocm/bin` (pas `build/bin`, cf. Global Constraints) ; `-ngl 99` (tous les layers, cohérent avec le benchmark de la spec) ; `--host 127.0.0.1` (pas `0.0.0.0` — le service routeur tourne sur la même machine, pas besoin d'exposer au réseau) ; `-c 2048` (le gabarit réel du routeur fait ~770 tokens de prompt + marge, pas besoin de 4096).

- [ ] **Step 2: Recharger et démarrer le service**

```bash
systemctl --user daemon-reload
systemctl --user enable --now slm-llama_cpp.service
systemctl --user status slm-llama_cpp.service --no-pager
```

Expected : `Active: active (running)`, pas d'erreur ROCm dans les logs (`journalctl --user -u slm-llama_cpp.service -n 30` doit montrer `ROCm0 KV buffer size` comme dans le test de benchmark, pas d'erreur `hipErrorNoDevice`).

- [ ] **Step 3: Vérifier avec un appel réel**

```bash
curl -s http://127.0.0.1:8998/health
python /home/mauceric/Secretarius/gen_corpus/eval_adapter.py --base-url http://127.0.0.1:8998 --check
```

Expected : `{"status":"ok"}` puis `=== 3/3 corrects ===` (ou proche), confirmant que le service de production sert bien l'adaptateur unique.

- [ ] **Step 4: Commit**

Le fichier systemd n'est pas dans le dépôt git (`~/.config/systemd/user/`) — rien à commiter pour cette tâche. Documenter le changement dans le journal de suivi si applicable (pas de fichier de suivi formel identifié — passer à la tâche suivante).

---

### Task 3: Service routeur Python

**Files:**
- Create: `router_service/router.py`
- Create: `router_service/server.py`
- Create: `router_service/__init__.py` (vide, pour les imports relatifs)
- Test: `router_service/test_router.py`
- Create: `/home/mauceric/.config/systemd/user/tiron-router.service`

**Interfaces:**
- Consomme : `http://127.0.0.1:8998/v1/chat/completions` (Task 2), `gen_corpus/corpus_lora_train.jsonl`/`corpus.jsonl` (pour construire les centroïdes BGE-M3, comme `router_3way.py`).
- Produit : `POST http://127.0.0.1:8999/route` avec body `{"message": str}`, retourne `{"status": "ok", "command": str, "args": str}` ou `{"status": "no_match"}` — consommé par `derisk-deleg` (Task 4).

- [ ] **Step 1: Écrire `router.py` (classification 3 centroïdes + portail gog)**

Reprend la logique validée de `router_3way.py` (scratchpad), inchangée sur le calcul des centroïdes.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classification BGE-M3 à 3 centroïdes, utilisée uniquement comme portail
de confiance sur les commandes gog (l'extraction de commande se fait par
l'adaptateur unique, pas par ce module)."""
import json
import torch
import torch.nn.functional as F

WIKI_CMDS = {"/c", "/q", "/ingest", "/source", "/wikistatus"}
GOG_CMDS = {"/chercher", "/connecter", "/inbox", "/drive", "/repondre"}
TRAIN_FULL = "/home/mauceric/Secretarius/gen_corpus/corpus_lora_train.jsonl"
RAW_CORPUS = "/home/mauceric/Secretarius/gen_corpus/corpus.jsonl"
NULL_VARIANTES = {"action_impossible", "aide_generale"}
T_SOFTMAX = 0.05
SEUIL_GOG = 0.50


def true_domain(cmd):
    if cmd in WIKI_CMDS:
        return "wiki"
    if cmd in GOG_CMDS:
        return "gog"
    return None


class GogGate:
    """Portail de confiance : True si la classification BGE-M3 confirme
    le domaine gog avec une confiance >= SEUIL_GOG."""

    def __init__(self, n_per_class=80):
        from transformers import AutoModel, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self._mdl = AutoModel.from_pretrained("BAAI/bge-m3").eval()

        buckets = {"wiki": [], "gog": []}
        for l in open(TRAIN_FULL):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            cmd = json.loads(r["messages"][-1]["content"]).get("command")
            dom = true_domain(cmd)
            if dom is not None and len(buckets[dom]) < n_per_class:
                buckets[dom].append(r["messages"][-2]["content"])

        null_texts = []
        for l in open(RAW_CORPUS):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            if r.get("intention") == "out_of_scope" and r.get("variante") in NULL_VARIANTES:
                null_texts.append(r["text"])

        cent_wiki = self._embed(buckets["wiki"]).mean(0, keepdim=True)
        cent_gog = self._embed(buckets["gog"]).mean(0, keepdim=True)
        cent_null = self._embed(null_texts).mean(0, keepdim=True)
        self._cmat = torch.cat([cent_wiki, cent_gog, cent_null], 0)

    def _embed(self, texts):
        enc = self._tok(texts, padding=True, truncation=True, max_length=128,
                        return_tensors="pt")
        with torch.no_grad():
            out = self._mdl(**enc).last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=1)

    def gog_confident(self, message: str) -> bool:
        e = self._embed([message])
        sims = (e @ self._cmat.T).squeeze(0)
        probs = F.softmax(sims / T_SOFTMAX, dim=0)
        gog_is_argmax = sims[1] >= sims[0] and sims[1] >= sims[2]
        return bool(gog_is_argmax and probs[1].item() >= SEUIL_GOG)
```

- [ ] **Step 2: Test du portail (cas francs)**

```python
# router_service/test_router.py
from router_service.router import GogGate, WIKI_CMDS, GOG_CMDS


def test_gog_confident_on_clear_gog_message():
    gate = GogGate()
    assert gate.gog_confident("cherche les mails de Paul cette semaine") is True


def test_gog_not_confident_on_wiki_message():
    gate = GogGate()
    assert gate.gog_confident("que dit le wiki sur le projet Alpha ?") is False


def test_command_sets_disjoint():
    assert WIKI_CMDS.isdisjoint(GOG_CMDS)
```

Run: `cd ~/Secretarius && python -m pytest router_service/test_router.py -v`
Expected : 3 tests passent (le chargement de BGE-M3 prend quelques secondes, normal).

- [ ] **Step 3: Écrire `server.py` (HTTP + appel adaptateur + assemblage)**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Service HTTP du routeur Tiron : POST /route {message} -> {status, command, args}."""
import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router_service.router import GogGate, WIKI_CMDS, GOG_CMDS

LLAMA_BASE = os.environ.get("TIRON_LLAMA_BASE", "http://127.0.0.1:8998")
SYSTEM_ROUTE = ('Routeur de commandes Tiron. Pour chaque message, répondre '
                'uniquement avec un objet JSON : {"command": "/commande" ou '
                'null, "args": "arguments bruts ou chaîne vide"}.')

_gate = None  # chargé au démarrage (Step 5)


def call_adapter(message: str):
    body = {"messages": [{"role": "system", "content": SYSTEM_ROUTE},
                         {"role": "user", "content": message}],
            "max_tokens": 60, "temperature": 0}
    req = urllib.request.Request(LLAMA_BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=30))
    raw = d["choices"][0]["message"]["content"].strip()
    parsed = json.loads(raw)
    return parsed.get("command"), parsed.get("args", "")


def route_message(message: str) -> dict:
    try:
        command, args = call_adapter(message)
    except Exception:
        return {"status": "no_match"}

    if command in GOG_CMDS:
        if not _gate.gog_confident(message):
            return {"status": "no_match"}
        return {"status": "ok", "command": command, "args": args}
    if command in WIKI_CMDS:
        return {"status": "ok", "command": command, "args": args}
    return {"status": "no_match"}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/route":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        result = route_message(body.get("message", ""))
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass  # silence les logs d'accès par défaut


def main():
    global _gate
    print("Chargement BGE-M3...", flush=True)
    _gate = GogGate()
    print("Prêt, écoute sur :8999", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8999), Handler).serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Test d'intégration du endpoint (nécessite le service llama-server de la Task 2 démarré)**

```python
# ajouter à router_service/test_router.py
import json
import threading
import time
import urllib.request

from router_service import server as router_server


def test_route_endpoint_end_to_end():
    router_server._gate = router_server.GogGate()
    httpd = __import__("http.server", fromlist=["ThreadingHTTPServer"]).ThreadingHTTPServer(
        ("127.0.0.1", 8999), router_server.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8999/route",
            data=json.dumps({"message": "cherche les mails de Paul"}).encode(),
            headers={"Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=30))
        assert resp["status"] in ("ok", "no_match")
    finally:
        httpd.shutdown()
```

Run: `cd ~/Secretarius && python -m pytest router_service/test_router.py -v`
Expected : 4 tests passent (le service llama-server de la Task 2 doit tourner sur :8998 pour ce test — sinon `call_adapter` échoue et le test vérifie bien qu'on obtient `no_match` proprement, pas une exception).

- [ ] **Step 5: Unité systemd du service routeur**

```ini
[Unit]
Description=Tiron router service (BGE-M3 gate + dispatch adaptateur unique)
After=network.target slm-llama_cpp.service
Requires=slm-llama_cpp.service

[Service]
Type=simple
WorkingDirectory=/home/mauceric/Secretarius
ExecStart=/usr/bin/python3 -m router_service.server
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Enregistrer dans `/home/mauceric/.config/systemd/user/tiron-router.service`, puis :

```bash
systemctl --user daemon-reload
systemctl --user enable --now tiron-router.service
curl -s -X POST http://127.0.0.1:8999/route -H "Content-Type: application/json" \
  -d '{"message":"cherche les mails de Paul cette semaine"}'
```

Expected : `{"status": "ok", "command": "/chercher", "args": "..."}` (ou `no_match` si le portail gog refuse — dans les deux cas, pas d'erreur HTTP).

- [ ] **Step 6: Commit**

```bash
cd ~/Secretarius
git add router_service/
git commit -m "feat(tiron): service routeur HTTP (BGE-M3 portail gog + adaptateur unique)"
```

---

### Task 4: Extension du plugin `derisk-deleg`

**Files:**
- Modify: `derisk-deleg/src/index.ts` (fonction `before_agent_reply`, lignes 368-422 de la version actuelle)
- Create: `derisk-deleg/src/dispatch.ts`
- Test: `derisk-deleg/src/dispatch.test.ts`

**Interfaces:**
- Consomme : `POST http://127.0.0.1:8999/route` (Task 3), `delegateWiki`/`delegateGog`/`delegateScout` (déjà définies dans `index.ts`), `parseReply` (déjà définie dans `parse.ts`), `pending`/`PENDING_TTL_MS` (déjà définis dans `index.ts`).
- Produit : `commandToAction(command: string): "wiki" | "gog-direct" | "gog-reply" | "scout" | null`, fonction pure testable sans dépendre de `api`.

- [ ] **Step 1: Extraire la table de correspondance en fonction pure testable**

```typescript
// derisk-deleg/src/dispatch.ts
// Correspondance commande routeur -> type d'action. Fonction pure (testable
// sans mock d'api OpenClaw) ; le câblage réel (appel des fonctions delegate*)
// reste dans index.ts, qui est la seule couche à connaître `api`.

export type RouterCommand =
  | "/c" | "/q" | "/ingest" | "/wikistatus" | "/source"
  | "/chercher" | "/connecter" | "/inbox" | "/drive" | "/repondre";

export type ActionKind =
  | { kind: "wiki"; op: "capture" | "query" | "ingest" | "status" }
  | { kind: "scout" }
  | { kind: "gog"; op: "search" | "auth_start" | "inbox" | "drive_search" }
  | { kind: "gog-reply" };

const TABLE: Record<RouterCommand, ActionKind> = {
  "/c": { kind: "wiki", op: "capture" },
  "/q": { kind: "wiki", op: "query" },
  "/ingest": { kind: "wiki", op: "ingest" },
  "/wikistatus": { kind: "wiki", op: "status" },
  "/source": { kind: "scout" },
  "/chercher": { kind: "gog", op: "search" },
  "/connecter": { kind: "gog", op: "auth_start" },
  "/inbox": { kind: "gog", op: "inbox" },
  "/drive": { kind: "gog", op: "drive_search" },
  "/repondre": { kind: "gog-reply" },
};

export function commandToAction(command: string): ActionKind | null {
  return (TABLE as Record<string, ActionKind>)[command] ?? null;
}
```

- [ ] **Step 2: Test de la table de correspondance**

```typescript
// derisk-deleg/src/dispatch.test.ts
import { describe, expect, it } from "vitest";
import { commandToAction } from "./dispatch.js";

describe("commandToAction", () => {
  it("mappe /source vers scout, pas wiki", () => {
    expect(commandToAction("/source")).toEqual({ kind: "scout" });
  });

  it("mappe /repondre vers gog-reply (jamais gog direct)", () => {
    expect(commandToAction("/repondre")).toEqual({ kind: "gog-reply" });
  });

  it("mappe /chercher vers gog search", () => {
    expect(commandToAction("/chercher")).toEqual({ kind: "gog", op: "search" });
  });

  it("retourne null pour une commande inconnue", () => {
    expect(commandToAction("/inexistant")).toBeNull();
  });
});
```

Run: `cd ~/Secretarius/derisk-deleg && npm test`
Expected : 4 tests passent.

- [ ] **Step 3: Appeler le service routeur depuis `index.ts`**

Ajouter en haut de `derisk-deleg/src/index.ts` (après les imports existants) :

```typescript
import { commandToAction } from "./dispatch.js";

const ROUTER_URL = "http://127.0.0.1:8999/route";

async function callRouter(message: string): Promise<
  { status: "ok"; command: string; args: string } | { status: "no_match" } | { status: "unavailable" }
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
    return data.status === "ok"
      ? { status: "ok", command: data.command, args: data.args ?? "" }
      : { status: "no_match" };
  } catch {
    return { status: "unavailable" };
  }
}
```

- [ ] **Step 4: Étendre `before_agent_reply` pour dispatcher via le routeur**

Le hook existant (lignes 368-422 de `index.ts`) traite déjà `/confirm`, `/annuler` et le retour OAuth, et se termine par `return;` (pas géré) quand aucun de ces cas ne matche. Remplacer ce `return;` final par le nouveau dispatch :

```typescript
      // (fin du bloc existant /confirm /annuler, juste avant le `return;` final)

      // Routage déterministe via le service Tiron : tout message qui n'est
      // pas déjà /confirm, /annuler, ou un retour OAuth.
      const routed = await callRouter(text);

      if (routed.status === "unavailable") {
        return { handled: true, reply: { text: "Routeur local indisponible, réessayez dans un instant." } };
      }
      if (routed.status === "no_match") {
        return {
          handled: true,
          reply: { text: "Je n'ai pas identifié de commande (essayez /q <question>, /c <url>...)." },
        };
      }

      const action = commandToAction(routed.command);
      if (action === null) {
        return {
          handled: true,
          reply: { text: "Je n'ai pas identifié de commande (essayez /q <question>, /c <url>...)." },
        };
      }

      if (action.kind === "wiki") {
        const out = await delegateWiki(api, action.op, routed.args);
        return { handled: true, reply: { text: out.slice(0, 1800) } };
      }
      if (action.kind === "scout") {
        const out = await delegateScout(api, routed.args.trim());
        return { handled: true, reply: { text: out.slice(0, 1800) } };
      }
      if (action.kind === "gog") {
        const out = await delegateGog(api, action.op, routed.args);
        return { handled: true, reply: { text: out.slice(0, 1800) } };
      }
      // action.kind === "gog-reply" : réutilise EXACTEMENT la logique de mise
      // en attente existante (parseReply + pending), jamais de délégation
      // directe — /repondre est la seule commande sensible atteignable ici.
      const parsed = parseReply(routed.args);
      if (!parsed) {
        return { handled: true, reply: { text: "Usage: /repondre <id> <texte>" } };
      }
      pending = { kind: "reply", messageId: parsed.messageId, body: parsed.body, ts: Date.now() };
      return {
        handled: true,
        reply: {
          text: `📧 Brouillon de réponse prêt (non envoyé) :\n• En réponse à : ${parsed.messageId}\n• Corps : ${parsed.body}\n\nTapez /confirm pour envoyer (valable 10 min), ou /annuler pour abandonner.`,
        },
      };
```

Note d'implémentation : ce bloc dupliquant intentionnellement la construction du message `pending` de l'outil `gog_reply` existant (lignes 305-329 de `index.ts`) — c'est la même logique, pas une nouvelle, appliquée simplement depuis un point d'entrée différent (le hook plutôt que l'outil). Ne pas factoriser plus loin dans cette tâche (YAGNI, la duplication est de 6 lignes).

- [ ] **Step 5: Retirer les outils `wiki_*`/`gog_*`/`source_read` de `main` (préparation Task 5)**

Pas de changement de code dans cette tâche — le retrait effectif se fait dans le template `openclaw.json` (Task 5). Ici, juste vérifier que le hook fonctionne même si `main` n'a plus accès à ces outils (le hook tourne **avant** le tour de modèle, donc ne dépend pas de `tools.sandbox.tools.allow` de `main`).

- [ ] **Step 6: Test manuel de bout en bout**

Avec les services des Tasks 2 et 3 démarrés :

```bash
cd ~/Secretarius/derisk-deleg
npm run plugin:build
npm run plugin:validate
```

Expected : build et validation passent sans erreur.

- [ ] **Step 7: Commit**

```bash
cd ~/Secretarius
git add derisk-deleg/src/dispatch.ts derisk-deleg/src/dispatch.test.ts derisk-deleg/src/index.ts
git commit -m "feat(derisk-deleg): dispatch via le service routeur Tiron dans before_agent_reply"
```

---

### Task 5: Profil `main` réduit (template sanroque)

**Files:**
- Create: `openclaw-config/openclaw.json.sanroque.template` (copie de `openclaw-config/openclaw.json.template` avec les modifications ci-dessous)

**Interfaces:**
- Consomme : structure actuelle de `openclaw-config/openclaw.json.template` (bloc `agents.list[main]`, `tools.sandbox.tools.allow` global).
- Produit : fichier de config déployable sur sanroque — le câblage `install.sh --profile` pour sélectionner ce template automatiquement est **hors périmètre de cette tâche** (noté en Risques ci-dessous, à traiter dans un chantier `install.sh` séparé).

- [ ] **Step 1: Copier le template existant**

```bash
cd ~/Secretarius/openclaw-config
cp openclaw.json.template openclaw.json.sanroque.template
```

- [ ] **Step 2: Réduire `tools.sandbox.tools.allow` de l'entrée `main`**

Dans `openclaw.json.sanroque.template`, repérer le bloc `agents.list` où `id` vaut `main` (structure actuelle : `{"id": "main", "model": {...}, "sandbox": {...}, "tools": {"exec": {"host": "sandbox"}}, "subagents": {...}}` — pas de `tools.sandbox.tools.allow` propre à `main` aujourd'hui, il hérite de la liste globale `tools.sandbox.tools.allow`). Ajouter un override explicite pour `main` :

```json
{
  "id": "main",
  "model": { "primary": "tiron-llm/tiron-router" },
  "sandbox": { "docker": { "image": "secretarius-tiron:latest", "network": "bridge", "env": { "XDG_CONFIG_HOME": "/workspace/.gog-config", "GOG_KEYRING_BACKEND": "file", "GOG_ACCOUNT": "cmauceri@gmail.com" } } },
  "tools": {
    "exec": { "host": "sandbox" },
    "sandbox": {
      "tools": {
        "allow": ["read", "sessions_list", "sessions_spawn", "sessions_yield", "group:runtime"],
        "deny": ["wiki_capture", "wiki_status", "wiki_ingest", "wiki_query", "source_read", "gog_inbox", "gog_send", "gog_connect_start", "gog_search", "gog_get", "gog_drive_search", "gog_reply", "wiki_tags", "wiki_kb_update"]
      }
    }
  },
  "subagents": { "allowAgents": ["wiki", "scout", "gog"] }
}
```

`model.primary` pointe `tiron-llm/tiron-router` (jamais réellement sollicité, cf. spec) plutôt que Euria — cohérent avec « Tiron reste purement SLM » sur ce profil.

- [ ] **Step 3: Valider le JSON**

```bash
python3 -c "import json; json.load(open('openclaw-config/openclaw.json.sanroque.template'))" && echo OK
```

Expected : `OK`, pas d'exception de parsing.

- [ ] **Step 4: Commit**

```bash
cd ~/Secretarius
git add openclaw-config/openclaw.json.sanroque.template
git commit -m "feat(config): template openclaw.json sanroque — profil main réduit (SLM local)"
```

---

## Risques / points ouverts pour l'exécution

- **Sélection automatique du template** (`install.sh --profile sanroque` ou équivalent) : non traitée par ce plan (Task 5 ne fait que créer le fichier). À faire manuellement (`cp openclaw.json.sanroque.template ~/.openclaw/openclaw.json`) tant que ce chantier n'est pas fait séparément.
- **Fidélité d'extraction de `/repondre`** (déjà noté dans la spec) : l'eval de la Task 1 (Step 5, seuil 90%) couvre ce risque globalement, mais surveiller spécifiquement les erreurs sur les exemples `/repondre` dans la sortie détaillée du script d'évaluation.
- **Redémarrage du gateway OpenClaw** après déploiement du nouveau template ou changement de `derisk-deleg` : nécessaire (mémoire `project-openclaw-gateway-reload` — skills/workspace/config chargés au démarrage du gateway), à faire après la Task 5 avant tout test réel via Telegram.
