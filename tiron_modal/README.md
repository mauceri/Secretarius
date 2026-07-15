# Tiron sur Modal — déploiement de mesure

Spec : `docs/superpowers/specs/2026-07-15-tiron-modal-deploiement-design.md`
Plan : `docs/superpowers/plans/2026-07-15-tiron-modal-deploiement.md`

Sert phi-4-mini + l'adaptateur `tiron-unified` (mêmes GGUFs qu'en local, llama.cpp
compilé CUDA, tag `b6257`) sur un GPU Modal L4, endpoint OpenAI-compatible.

## Marche à suivre

```bash
# 0. CLI Modal (une fois)
python3 -m venv ~/modal-venv && ~/modal-venv/bin/pip install -U modal
~/modal-venv/bin/modal setup            # auth navigateur (ou: modal token new)

# 1. Volume + upload des GGUFs (une fois, ~3 Go)
~/modal-venv/bin/modal volume create tiron-models
~/modal-venv/bin/modal volume put tiron-models ~/Modèles/Phi-4-mini-instruct-Q6_K.gguf /Phi-4-mini-instruct-Q6_K.gguf
~/modal-venv/bin/modal volume put tiron-models ~/lora_slm/tiron-unified-lora-f16.gguf /tiron-unified-lora-f16.gguf

# 2. Déployer (garde le terminal ouvert ; imprime l'URL)
~/modal-venv/bin/modal serve tiron_modal/app.py

# 3. Tester l'endpoint (autre terminal ; remplacer $MODAL_URL)
curl -s "$MODAL_URL/v1/chat/completions" -H 'Content-Type: application/json' -d @tiron_modal/prompt.json

# 4. Mesurer
python3 tiron_modal/bench.py http://127.0.0.1:8998 -n 6   # baseline locale
python3 tiron_modal/bench.py "$MODAL_URL" -n 6            # Modal (1re = froid)
```

## Résultats de mesure

(rempli après les runs)

| Cible        | TTFT froid | TTFT chaud | tok/s chaud | coût chaud/req |
|--------------|-----------:|-----------:|------------:|---------------:|
| iGPU local   |          — |            |             |             — |
| Modal L4     |            |            |             |                |
