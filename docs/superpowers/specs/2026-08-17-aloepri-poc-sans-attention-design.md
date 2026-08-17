# POC AloePri simplifié (sans obfuscation d'attention)

- Date : 2026-08-17
- Statut : design validé, en attente d'exécution
- Auteur : Christian Mauceri + Claude
- Source : arXiv 2603.01499v2, « Towards Privacy-Preserving LLM Inference via
  Covariant Obfuscation » (ByteDance / Université de Nanjing)
- Complément : `docs/architecture/aloepri-attaques-isa-tfma-sda.md` (explication
  des attaques ISA/TFMA/SDA citées ici)

## Contexte

AloePri protège la confidentialité d'une inférence LLM en cloud non fiable en
combinant, côté client, une permutation secrète des tokens et, côté modèle, une
transformation covariante des poids (embedding/unembedding, attention, FFN) qui
rend cette permutation invisible au serveur. Résultats du papier sur
Deepseek-V3.1-Terminus (671B) : perte de précision 0–3,5 %, <5 % de tokens
récupérés par les attaques testées, compatible vLLM/SGLang.

Ce POC reproduit la mécanique **sans la couche d'obfuscation d'attention**,
la plus coûteuse à implémenter (rotations RoPE, permutation par bloc à fenêtre
dynamique, permutation inter-tête). Le Tableau 4 du papier (ablation ISA) montre
que retirer cette couche fait remonter le taux de récupération à 87,14 % contre
l'Internal State Attack (ISA) — ce POC ne prétend donc à aucune garantie de
confidentialité réelle. Son but est de trancher une question d'ingénierie :
est-ce que la mécanique (permutation + bruit + matrices clés sur
embedding/unembedding, permutation + scaling sur FFN) préserve la qualité du
modèle et reste rapide, avant de décider d'investir le temps nécessaire à
l'obfuscation d'attention (qui, elle, apporterait une garantie proche d'un TEE
logiciel).

## Objectif

1. Construire le pipeline client↔serveur de bout en bout (permutation à
   l'envoi, dépermutation à la réception) sur Qwen2.5-7B.
2. Mesurer la dégradation de qualité et l'overhead de vitesse par rapport au
   modèle non obfusqué.
3. Quantifier, séparément et sans coût GPU, la résistance de la seule couche
   embedding/unembedding à une attaque par fréquence de tokens (TFMA/SDA-style).

**Critère de décision** : si qualité et vitesse sont concluantes, la suite
(hors périmètre de ce POC) reste à trancher entre l'obfuscation d'attention
(protection mesurée par le papier contre ISA, coût de dev plus élevé) et la
rotation de clé/permutation (protection candidate mais non quantifiée contre
ISA — correction du 2026-08-17, voir doc complémentaire : ISA est une
attaque « training-based », une rotation pourrait invalider le modèle
d'inversion entraîné par l'attaquant, à condition de tourner plus vite qu'il
ne peut recalibrer, coût inconnu à ce stade). Pertinent pour le cas d'usage
motivant : confidentialité d'une question posée à l'IA, indépendamment de la
confidentialité des documents sous-jacents.

## Portée

### Étape 0 — simulation d'attaque fréquence (locale, sanroque, sans GPU)

Script Python autonome : tokenizer Qwen2.5-7B, corpus de référence en français,
permutation simulée du vocabulaire, attaque par appariement de fréquence/rang
(TFMA-style) à différents volumes de corpus. Sortie : courbe % de tokens
top-N récupérés en fonction du volume de texte observé par l'attaquant.

Ne bloque pas les étapes suivantes — informatif, sert à documenter l'ordre de
grandeur de la fuite résiduelle par trafic (distincte de la fuite par requête
unique, hors scope ici, voir doc complémentaire).

### Étape 1 — transformation des poids + pipeline mécanique (RunPod)

1. Charger Qwen2.5-7B-Instruct en HF safetensors (pas GGUF — la manipulation de
   tenseurs bruts exclut llama.cpp pour ce POC, divergence assumée par rapport
   au reste de la stack Secretarius qui sert habituellement des GGUF via
   llama.cpp).
2. **Vérifier en premier** si `embed_tokens`/`lm_head` sont liés (weight tying)
   dans Qwen2.5-7B — si oui, une seule permutation cohérente pour les deux, pas
   deux jeux indépendants.
3. Générer la permutation secrète du vocabulaire + bruit gaussien + matrices
   clés (paire inversible P/Q, P·Q = I) ; appliquer à `embed_tokens`/`lm_head`.
4. Générer permutation + matrices de scaling (avec compensation inverse) pour
   chaque couche FFN (`gate_proj`/`up_proj` en sortie, `down_proj` en entrée,
   cohérence à maintenir sur la dimension intermédiaire SwiGLU).
5. Sauvegarder le modèle obfusqué (safetensors) et les clés secrètes séparément
   (les clés ne quittent jamais le client).
6. Wrapper client : tokenize standard + remapping des IDs à l'envoi (via la
   permutation secrète), remapping inverse à la réception, détokenize.
7. Serveur d'inférence sur RunPod Pod (pas Serverless — accès SSH/Jupyter pour
   déboguer une chirurgie de poids itérative), GPU RTX A5000 (24 Go,
   0,16 $/h Community Cloud), stack `transformers` HF pour ce premier jet
   (vLLM en suivi seulement si le pipeline est validé — plus proche du papier
   mais plus dur à instrumenter pour du debug ponctuel).

### Étape 2 — mesure qualité et vitesse

- **Qualité** : perplexité et/ou exactitude sur un jeu de prompts test,
  modèle obfusqué vs baseline non obfusqué (même modèle, mêmes prompts).
- **Vitesse** : tokens/s et latence, obfusqué vs baseline, sur le même Pod.

## Flux de données

```
Client (sanroque)
  → tokenize (tokenizer Qwen standard)
  → permute les IDs (clé secrète, jamais envoyée)
  → HTTP vers le Pod RunPod
       Serveur : modèle aux poids obfusqués (embedding+FFN), ignore la clé
       → génère une séquence d'IDs (dans l'espace permuté)
  ← réponse (IDs permutés)
  → dépermute (clé secrète)
  → detokenize → texte
```

## Vérification / critères de succès

1. **Round-trip correct** : un prompt de test produit un texte cohérent après
   permutation → inférence → dépermutation (sanity check de base, condition
   nécessaire avant toute mesure).
2. **Delta qualité** : écart de perplexité/exactitude obfusqué vs baseline sur
   le jeu de test, rapporté en pourcentage (comparable à la fourchette
   0–3,5 % du papier, sans obligation de l'atteindre — c'est un point de
   mesure, pas un objectif chiffré).
3. **Overhead vitesse** : écart tokens/s et latence obfusqué vs baseline.
4. **Étape 0** : courbe % tokens récupérés / volume de corpus, documentée
   indépendamment.

## Hors scope explicite

- Obfuscation d'attention (rotations RoPE, permutation par bloc/tête).
- Rotation de clés / de permutation en cours d'usage.
- Implémentation ou simulation de l'attaque ISA (réservée à une décision
  ultérieure, voir doc complémentaire — pertinente uniquement si on envisage
  la confidentialité par requête unique, ce que ce POC ne couvre pas).
- vLLM/SGLang (Pod RunPod avec `transformers` HF suffit pour ce POC).
- Tout déploiement au-delà du Pod RunPod ponctuel (pas d'intégration Secretarius/Tiron
  à ce stade).

## Risques / hypothèses à vérifier tôt

- **Weight tying** `embed_tokens`/`lm_head` sur Qwen2.5-7B : à confirmer avant
  d'écrire le script de transformation (change la structure du code).
- **Compatibilité SwiGLU FFN** : la permutation de la dimension intermédiaire
  doit être identique sur `gate_proj` et `up_proj` (sortie) pour que
  `silu(gate) * up` reste cohérent après permutation ; `down_proj` doit
  recevoir la même permutation en entrée. Point de vigilance à l'implémentation.
- **Coût RunPod** : de l'ordre de quelques dollars pour l'ensemble du POC
  (Pod A5000 à 0,16 $/h, usage de quelques heures cumulées) — pas un facteur
  bloquant, mais à confirmer une fois le Pod effectivement loué.
