# Déploiement Tiron (phi-4 + tiron-unified) sur Modal — mesure de faisabilité

- Date : 2026-07-15
- Statut : design validé (périmètre A), en attente d'exécution
- Auteur : Christian Mauceri + Claude

## Objectif

Répondre, chiffres en main, à : « phi-4-mini + l'adaptateur `tiron-unified`
sert-il correctement et rapidement sur un GPU Modal, via un endpoint
OpenAI-compatible, comparé à l'iGPU local ? »

Livrable : une app Modal réutilisable **+** un tableau de mesures (TTFT,
tokens/s, coût/requête, à froid et à chaud) comparé au service local (port 8998).

## Périmètre

**Inclus (objectif A)** : déploiement Modal + mesure.

**Hors périmètre** (→ septembre, selon les résultats) : bascule local/modal dans
l'installateur, auth de production, câblage OpenClaw (provider `tiron-llm`) et
routeur (`TIRON_LLAMA_BASE`). Ces deux consommateurs ne sont **pas** touchés ici.

## Contexte

- Cerveau local actuel : `llama-server` sur `127.0.0.1:8998`,
  `Phi-4-mini-instruct-Q6_K.gguf` + `tiron-unified-lora-f16.gguf`, `-c 32768`.
- Consommateurs du cerveau (inchangés) : provider `tiron-llm` d'OpenClaw
  (`baseUrl` = `…:8998/v1`) et routeur 8999 (`router_service/server.py`,
  variable `TIRON_LLAMA_BASE`). La bascule future ≈ pointer ces deux URLs.
- Greenfield : aucun code Modal dans le dépôt.

## Artefacts

- Base : `~/Modèles/Phi-4-mini-instruct-Q6_K.gguf`
- Adaptateur : `~/lora_slm/tiron-unified-lora-f16.gguf`

## Architecture

1. **Modal Volume `tiron-models`** : contient les deux GGUFs, uploadés une fois
   depuis `~/Modèles` et `~/lora_slm`.
2. **Image Modal** : base CUDA + binaire `llama-server` **CUDA pré-compilé**
   (release llama.cpp figée), pas de build source.
3. **Fonction Modal** GPU (L4 par défaut), Volume monté, exposée via
   `@modal.web_server(8080)` lançant :
   ```
   llama-server \
     --model <vol>/Phi-4-mini-instruct-Q6_K.gguf \
     --lora  <vol>/tiron-unified-lora-f16.gguf \
     --host 0.0.0.0 --port 8080 -c 8192
   ```
   → endpoint `https://<user>--<app>.modal.run/v1/chat/completions`,
   OpenAI-compatible.
4. **Auth** : token Modal (proxy auth) simple ; endpoint non public.

## Flux

Requête OpenAI-compatible → `web_server` Modal → `llama-server` (GPU) → réponse.
Cold start = boot conteneur + chargement du modèle depuis le Volume (quelques s).
Warm = requêtes suivantes sur conteneur chaud.

## Plan de mesure (cœur du livrable)

Script client `bench_modal.py` :

- **Prompt de test** : une requête Tiron représentative, figée à partir d'un cas
  réel (capturer une vraie requête de routage/orchestration, p. ex. un `/q` ou un
  appel du routeur).
- N itérations ; enregistre TTFT, tokens/s, latence totale.
- Distingue **1er appel (froid)** vs **suivants (chaud)**.
- **Baseline locale** : même prompt contre `127.0.0.1:8998`.
- **Coût/requête** estimé = temps GPU × tarif Modal du GPU.
- Optionnel si le temps le permet : comparer **L4 vs A10G**.

Sortie : tableau récapitulatif (Modal froid, Modal chaud, local) × (TTFT, tok/s,
coût).

## Décisions et compromis

- **Réplique fidèle llama.cpp** (mêmes GGUFs) plutôt que vLLM : zéro dérive de
  comportement, effort minimal. vLLM (safetensors + hot-swap LoRA) reporté.
- **`-c 8192`** au lieu de 32768 : réduit le KV cache (VRAM) ; écart assumé pour
  la mesure, à réévaluer si un prompt Tiron le dépasse.
- **Petit GPU (L4)** : phi-4-mini Q6_K ≈ 3 Go, inutile de viser plus gros
  (L4 ~0,80 $/h).
- **Volume plutôt qu'image** pour les GGUFs : évite une image lourde et les
  rebuilds.
- **Binaire CUDA pré-compilé** plutôt que build source : mise en place plus rapide.

## Risques

- **Cold start** pénalise la latence perçue → mesuré séparément ; option
  `min_containers=1` (garde chaud, coûte) réservée à un usage réel, hors mesure.
- **Compat version llama.cpp / GGUF LoRA** : le GGUF de l'adaptateur doit se
  charger avec la version du `llama-server` de l'image (les versions GGUF ont
  déjà causé des pannes) → figer une release connue et la tester.
- **VRAM** : à `-c 8192` sur L4 (24 Go), large marge ; surveiller si on remonte
  le contexte.
- **Compte Modal** : prérequis (configuré par l'utilisateur).

## Critère de succès

1. Un `curl` OpenAI-compatible sur l'endpoint Modal renvoie une complétion valide.
2. Sanity : sur un prompt connu, réponse de même **nature** qu'en local
   (même base + LoRA).
3. Tableau de mesures produit (froid / chaud / local × TTFT / tok-s / coût).

## Prérequis

- Compte Modal + CLI `modal` configuré (token).
- Accès local aux deux GGUFs pour l'upload initial.
