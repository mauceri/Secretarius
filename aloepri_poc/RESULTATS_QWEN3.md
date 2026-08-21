# Résultats — Portage AloePri sur Qwen3-8B (transformation réelle)

Transformation réelle exécutée le 2026-08-21 via `transform_streaming.py`
(variante mémoire-léger, bit-à-bit identique à `model_transform.py` —
vérifié par `tests/test_qwen3_arch.py`, 49 tests verts).

- **Modèle** : `Qwen/Qwen3-8B`, bfloat16 (8,05 G paramètres, ~16,4 Go)
- **Seed** : 0 — **α_e** = 1,0 — **α_h** = 0,2 — **β** = 8 — **γ** = 1e3 — **ζ** = 1e3
- **rope_scaling** : off automatique (Qwen3 a q_norm/k_norm, cf.
  `attention_obfuscation.py` : le scaling diagonal Ĥ ne commute pas avec une
  RMSNorm de tête — R̂ et Ẑ, orthogonaux, commutent)
- **Périmètre** : h=0 (pas de matrices clés P̂/Q̂), pas de chaînage inter-couches
  (identique au POC Qwen2.5)

## Exécution

| Étape | Durée | Notes |
|---|---|---|
| Téléchargement HF (16,4 Go) | ~31 min | vitesse mesurée ~0,55 Go/min sur cette machine |
| Transformation (streaming, CPU) | ~2 min | pic RAM ≈ 4-5 Go (machine à 13 Go) |
| Écriture | 14 shards, 16 Go | `model-00001-of-00014…00014` |

σ(embed) = 0,0221, σ(head) = 0,0260 (bruit relatif α_e = 1,0 — défaut du
papier ; levier qualité principal : α_e=0,5 ramenait +19 % → +13,6 % de
perplexité sur Qwen2.5, cf. `RESULTATS.md`).

## Vérification post-transformation (`verify_transform.py`)

Sans recharger les 16 Go en RAM, recalcule des échantillons depuis le modèle
source et les compare bit-à-bit aux fichiers écrits :

- **structure** : 399 tenseurs, index.json cohérent, dtype bf16 uniforme ;
- **config** : IDs spéciaux remappés dans l'espace permuté (eos 151645 →
  36295, bos 151643 → …), generation_config idem ;
- **échantillons** : 24 lignes d'embedding/head + 3 couches complètes
  (attention + FFN) → **bit-à-bit identiques** ;
- **client** : permutation bijective sur [0, 151936[, encode→decode
  identique (« Quelle est la capitale de la France ? »), IDs dans les bornes
  (tokenizer 151669 ≤ vocab 151936, 267 lignes de padding).

## Round-trip de bout en bout

Non mesurable sur cette machine (16 Go en RAM indisponible). À faire sur le
GPU Modal (cf. `aloepri_modal/README.md`, §6) — le POC Qwen2.5 mesurait une
perte de qualité moyenne de +19 % de perplexité avec un round-trip
qualitativement correct (texte cohérent).

### Déploiement réel sur Modal (2026-08-21, GPU L4)

Endpoint : `https://mauceri--aloepri-qwen3-modal-serve.modal.run` (protégé
par clé API Bearer, scale-to-zero après 5 min).

- Modèle obfusqué sur le Volume `aloepri-models` (`/qwen3-8b-obf`, 14 shards),
  produit par `modal run app.py::transform` (~2 min : téléchargement HF 25 s
  au débit datacenter + transform CPU).
- `/health` → 200. Sans clé / mauvaise clé → 401.
- **Round-trip** (client local, greedy) :
  - « Quelle est la capitale de la France ? » → « … La capitale de la France
    est **Paris**. … » (cohérent, correct)
  - « What is the capital of Japan? » → texte cohérent (répétitions greedy
    habituelles de Qwen3-8B, aucun effet de l'obfuscation)
- 4 correctifs nécessités par le déploiement réel (cf. commit `6078afe`) :
  disque éphémère ≥ 512 GiB ; chemin d'import du POC ; `@modal.asgi_app()`
  au lieu de `web_server` + uvicorn bloquant (303 sinon) ; auth par
  `os.environ` (valeurs de Secret injectées en variables d'environnement).

## Artefacts (hors dépôt, à ne jamais committer)

- `/home/cmauceri/deepseek-harness-ws/artifacts/obfuscated_qwen3_8b/` (16 Go)
- `/home/cmauceri/deepseek-harness-ws/artifacts/obfuscation_keys.json` (4,8 Mo
  — **secret client**)
