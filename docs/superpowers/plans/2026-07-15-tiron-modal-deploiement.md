# Déploiement Tiron sur Modal (mesure de faisabilité) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Servir phi-4-mini + l'adaptateur `tiron-unified` sur un GPU Modal via un endpoint OpenAI-compatible, et mesurer TTFT/tokens-s/coût (froid + chaud) contre l'iGPU local.

**Architecture:** Une app Modal héberge `llama-server` (compilé CUDA) sur un GPU L4, chargeant les deux GGUFs depuis un Modal Volume, exposé par `@modal.web_server`. Un script client mesure la latence contre cet endpoint et contre le service local (8998).

**Tech Stack:** Modal (Volume, Image CUDA, web_server), llama.cpp (`llama-server`, tag b6257), Python (`requests`) pour le bench.

## Global Constraints

- Base GGUF : `~/Modèles/Phi-4-mini-instruct-Q6_K.gguf` (3,0 Go).
- Adaptateur GGUF : `~/lora_slm/tiron-unified-lora-f16.gguf` (35 Mo).
- Build llama.cpp CUDA figé au tag **`b6257`** (= commit `b1afcab`, la version locale qui charge déjà ces GGUFs → mitige le risque de compat GGUF/LoRA).
- GPU **L4** ; contexte **`-c 8192`** ; **`-ngl 999`** (offload GPU total).
- Endpoint OpenAI-compatible, chemin `/v1/chat/completions`.
- Mesures **froid + chaud** ; baseline locale = `http://127.0.0.1:8998`.
- Répertoire de travail : **`tiron_modal/`** à la racine du dépôt (le nom évite de masquer le paquet Python `modal`).
- Hors périmètre : bascule installateur, câblage OpenClaw (`tiron-llm`) / routeur (`TIRON_LLAMA_BASE`), auth de production.
- Branche courante : `ingestion-phi4-passages-centraux`.

---

### Task 1 : Installer et configurer le CLI Modal

Le compte Modal existe mais le CLI est **absent** de sanroque.

**Files:**
- Aucun (installation d'outil + auth).

**Interfaces:**
- Produces : commande `modal` disponible et profil authentifié (consommé par les tâches 3, 4, 5).

- [ ] **Step 1 : Vérifier que `modal` est absent**

Run: `which modal || echo ABSENT`
Expected: `ABSENT`

- [ ] **Step 2 : Installer le CLI Modal**

Dans un venv dédié pour ne rien polluer :
```bash
python3 -m venv ~/modal-venv
~/modal-venv/bin/pip install --upgrade modal
```

- [ ] **Step 3 : Authentifier contre le compte existant**

```bash
~/modal-venv/bin/modal setup
```
Suivre le lien affiché pour lier le compte (navigateur). Sur machine headless, utiliser `~/modal-venv/bin/modal token new` qui imprime l'URL à ouvrir.

- [ ] **Step 4 : Vérifier le profil**

Run: `~/modal-venv/bin/modal profile current`
Expected: le nom du workspace Modal s'affiche (pas d'erreur d'auth).

> Note : dans les tâches suivantes, `modal` = `~/modal-venv/bin/modal`.

---

### Task 2 : Scaffolding + prompt de test représentatif

**Files:**
- Create : `tiron_modal/prompt.json`
- Create : `tiron_modal/README.md`

**Interfaces:**
- Produces : `tiron_modal/prompt.json` (corps de requête OpenAI chat, consommé par le bench en tâche 5 et par les curl de vérif des tâches 4/5).

- [ ] **Step 1 : Créer le répertoire et un prompt représentatif**

```bash
mkdir -p tiron_modal
```
Écrire `tiron_modal/prompt.json` :
```json
{
  "model": "phi-4",
  "messages": [
    {"role": "system", "content": "Tu es Tiron, un orchestrateur. Tu reçois une demande de l'utilisateur et tu réponds de façon concise et directe, en français."},
    {"role": "user", "content": "Résume en trois points ce que tu peux faire pour m'aider à gérer mes notes et mes documents."}
  ],
  "max_tokens": 200,
  "temperature": 0.2
}
```

> Fidélité optionnelle : pour un prompt réellement issu de la prod, capturer le corps d'une vraie requête reçue par le cerveau (tailer les logs OpenClaw / `llama-server` pendant une interaction Telegram) et remplacer le contenu de `prompt.json`. Pour une mesure de latence, la longueur représentative suffit ; ce défaut est acceptable.

- [ ] **Step 2 : Créer le squelette README**

Écrire `tiron_modal/README.md` :
```markdown
# Tiron sur Modal — déploiement de mesure

Voir `docs/superpowers/specs/2026-07-15-tiron-modal-deploiement-design.md`.

## Commandes
- Upload modèles : voir plan tâche 3.
- Déploiement : `modal serve tiron_modal/app.py`
- Bench : `python tiron_modal/bench.py <base_url> --prompt tiron_modal/prompt.json -n 6`

## Résultats de mesure
(rempli en tâche 5)
```

- [ ] **Step 3 : Vérifier le prompt contre le cerveau local (sanity + 1er point de mesure)**

Run:
```bash
curl -s http://127.0.0.1:8998/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @tiron_modal/prompt.json | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['message']['content'][:200])"
```
Expected: un texte de réponse cohérent s'affiche (prouve que le prompt est valide et donne la réponse locale de référence).

- [ ] **Step 4 : Commit**

```bash
git add tiron_modal/prompt.json tiron_modal/README.md
git commit -m "feat(modal): scaffolding + prompt de test Tiron"
```

---

### Task 3 : Modal Volume + upload des GGUFs

**Files:**
- Aucun fichier de code (actions CLI Modal).

**Interfaces:**
- Produces : Volume Modal `tiron-models` contenant `/Phi-4-mini-instruct-Q6_K.gguf` et `/tiron-unified-lora-f16.gguf` (monté par la fonction en tâche 4).

- [ ] **Step 1 : Créer le Volume**

Run: `~/modal-venv/bin/modal volume create tiron-models`
Expected: confirmation de création (ou « already exists » si relance).

- [ ] **Step 2 : Uploader la base**

Run:
```bash
~/modal-venv/bin/modal volume put tiron-models ~/Modèles/Phi-4-mini-instruct-Q6_K.gguf /Phi-4-mini-instruct-Q6_K.gguf
```
Expected: upload de ~3 Go terminé.

- [ ] **Step 3 : Uploader l'adaptateur**

Run:
```bash
~/modal-venv/bin/modal volume put tiron-models ~/lora_slm/tiron-unified-lora-f16.gguf /tiron-unified-lora-f16.gguf
```

- [ ] **Step 4 : Vérifier le contenu du Volume**

Run: `~/modal-venv/bin/modal volume ls tiron-models`
Expected: les deux fichiers `.gguf` apparaissent avec leurs tailles.

---

### Task 4 : App Modal — image CUDA + `web_server` servant `llama-server`

**Files:**
- Create : `tiron_modal/app.py`

**Interfaces:**
- Consumes : Volume `tiron-models` (tâche 3), CLI authentifié (tâche 1).
- Produces : une URL d'endpoint OpenAI-compatible `https://<...>.modal.run/v1/chat/completions` (consommée par le bench en tâche 5).

- [ ] **Step 1 : Écrire `tiron_modal/app.py`**

```python
import subprocess
import modal

MODELS_DIR = "/models"
BASE = f"{MODELS_DIR}/Phi-4-mini-instruct-Q6_K.gguf"
LORA = f"{MODELS_DIR}/tiron-unified-lora-f16.gguf"
LLAMA_TAG = "b6257"   # = build local (commit b1afcab) qui charge déjà ces GGUFs
CTX = 8192
PORT = 8080
GPU = "L4"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "cmake", "libcurl4-openssl-dev")
    .run_commands(
        f"git clone --depth 1 --branch {LLAMA_TAG} https://github.com/ggml-org/llama.cpp /llama.cpp",
        "cmake -S /llama.cpp -B /llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=ON",
        "cmake --build /llama.cpp/build --config Release -j --target llama-server",
    )
)

vol = modal.Volume.from_name("tiron-models")
app = modal.App("tiron-llm-modal")


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODELS_DIR: vol},
    timeout=3600,
    scaledown_window=300,
)
@modal.web_server(port=PORT, startup_timeout=300)
def serve():
    subprocess.Popen(
        f"/llama.cpp/build/bin/llama-server --model {BASE} --lora {LORA} "
        f"--host 0.0.0.0 --port {PORT} -c {CTX} -ngl 999",
        shell=True,
    )
```

- [ ] **Step 2 : Vérifier l'échec avant déploiement**

Run: `curl -s -m 5 https://tiron-llm-modal.invalid/v1/chat/completions || echo INJOIGNABLE`
Expected: `INJOIGNABLE` (aucun endpoint encore).

- [ ] **Step 3 : Déployer en mode dev (build + serveur)**

Run: `~/modal-venv/bin/modal serve tiron_modal/app.py`
Expected: le build de l'image se fait (première fois, quelques minutes, mis en cache ensuite), puis une URL `https://...modal.run` s'affiche. Laisser tourner ; noter l'URL (variable `$MODAL_URL` pour la suite).

- [ ] **Step 4 : Vérifier l'endpoint (succès #1 + sanity #2)**

Dans un autre terminal :
```bash
curl -s "$MODAL_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @tiron_modal/prompt.json | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['message']['content'][:200])"
```
Expected: un texte cohérent, de même **nature** que la réponse locale (Task 2 Step 3) — même base + LoRA.

- [ ] **Step 5 : Commit**

```bash
git add tiron_modal/app.py
git commit -m "feat(modal): app llama-server CUDA (phi-4 + tiron-unified) sur GPU L4"
```

---

### Task 5 : Bench + tableau de mesures (baseline local + Modal, froid/chaud, coût)

**Files:**
- Create : `tiron_modal/bench.py`
- Modify : `tiron_modal/README.md` (section « Résultats de mesure »)

**Interfaces:**
- Consumes : `tiron_modal/prompt.json` (tâche 2), l'endpoint Modal `$MODAL_URL` (tâche 4), le service local `http://127.0.0.1:8998`.

- [ ] **Step 1 : Écrire `tiron_modal/bench.py`**

```python
import argparse
import json
import time

import requests


def bench(base_url: str, prompt_path: str, n: int, api_key: str | None):
    body = json.load(open(prompt_path))
    body["stream"] = True
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    rows = []
    for i in range(n):
        t0 = time.perf_counter()
        ttft = None
        chunks = 0
        with requests.post(
            f"{base_url}/v1/chat/completions",
            json=body, headers=headers, stream=True, timeout=600,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
                try:
                    if json.loads(data)["choices"][0]["delta"].get("content"):
                        chunks += 1  # approx : 1 chunk SSE ~ 1 token
                except Exception:
                    pass
        total = time.perf_counter() - t0
        gen = max(total - (ttft or 0), 1e-6)
        toks = chunks / gen
        rows.append((i, ttft or 0.0, total, chunks, toks))
        tag = "FROID" if i == 0 else "chaud"
        print(f"[{tag}] req{i}: TTFT={ttft:.2f}s total={total:.2f}s out~{chunks} {toks:.1f} tok/s")

    warm = rows[1:] or rows
    m = lambda k: sum(x[k] for x in warm) / len(warm)
    print(f"\nRESUME {base_url}")
    print(f"  froid : TTFT={rows[0][1]:.2f}s total={rows[0][2]:.2f}s")
    print(f"  chaud : TTFT~{m(1):.2f}s total~{m(2):.2f}s {m(4):.1f} tok/s (n={len(warm)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base_url")
    p.add_argument("--prompt", default="tiron_modal/prompt.json")
    p.add_argument("-n", type=int, default=6)
    p.add_argument("--api-key", default=None)
    a = p.parse_args()
    bench(a.base_url, a.prompt, a.n, a.api_key)
```

> Note : `out~` compte les chunks SSE (approximation du nombre de tokens, suffisante pour comparer des latences). `chaud` = moyenne des requêtes après la première.

- [ ] **Step 2 : Mesurer la baseline locale**

Run: `python3 tiron_modal/bench.py http://127.0.0.1:8998 -n 6`
Expected: un résumé « froid/chaud » avec des chiffres (référence iGPU).

- [ ] **Step 3 : Mesurer Modal (froid + chaud)**

S'assurer que l'app est scale-to-zero (attendre > `scaledown_window` ou redéployer), puis :
Run: `python3 tiron_modal/bench.py "$MODAL_URL" -n 6`
Expected: la 1re requête (FROID) inclut le cold start (boot + chargement 3 Go depuis le Volume) ; les suivantes (chaud) sont bien plus rapides.

- [ ] **Step 4 : Calculer le coût/requête et remplir le README**

Coût chaud/requête ≈ `total_chaud (s) × tarif_GPU ($/s)`. Tarif L4 Modal ≈ **0,80 $/h** au moment de l'écriture → **vérifier le tarif courant sur modal.com/pricing**. Renseigner dans `tiron_modal/README.md` :
```markdown
## Résultats de mesure (<date>, GPU L4, -c 8192)

| Cible        | TTFT froid | TTFT chaud | tok/s chaud | coût chaud/req |
|--------------|-----------:|-----------:|------------:|---------------:|
| iGPU local   |          — |            |             |             — |
| Modal L4     |            |            |             |                |
```

- [ ] **Step 5 : Commit**

```bash
git add tiron_modal/bench.py tiron_modal/README.md
git commit -m "feat(modal): bench latence + tableau de mesures (local vs Modal L4)"
```

---

### Task 6 (optionnel, si le temps le permet) : Comparaison L4 vs A10G

**Files:**
- Modify : `tiron_modal/app.py` (`GPU = "A10G"`)
- Modify : `tiron_modal/README.md`

- [ ] **Step 1 : Basculer sur A10G**

Éditer `tiron_modal/app.py` : `GPU = "A10G"`.

- [ ] **Step 2 : Redéployer et benchmarker**

```bash
~/modal-venv/bin/modal serve tiron_modal/app.py   # nouvelle URL
python3 tiron_modal/bench.py "$MODAL_URL_A10G" -n 6
```

- [ ] **Step 3 : Ajouter la ligne A10G au tableau et remettre `GPU = "L4"`**

Renseigner le README, restaurer `GPU = "L4"` (défaut retenu).

- [ ] **Step 4 : Commit**

```bash
git add tiron_modal/app.py tiron_modal/README.md
git commit -m "chore(modal): mesure A10G vs L4"
```

---

## Notes d'exécution

- La tâche 4 Step 3 (`modal serve`) doit rester active pendant les vérifs/bench (elle tient l'endpoint dev). Pour un endpoint persistant, `modal deploy` à la place — hors périmètre de mesure.
- Le premier `modal serve` déclenche le build de l'image (clone + cmake CUDA de llama.cpp) : plusieurs minutes, puis mis en cache par Modal.
- Auth de l'endpoint laissée publique (URL obscure) le temps de la mesure ; passer `requires_proxy_auth=True` sur `@modal.web_server` pour un usage réel (hors périmètre).
