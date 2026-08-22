# Matrices clés P̂/Q̂ (h>0) avec chaînage inter-couches — design et reprise

- **Date** : 2026-08-22
- **Statut** : étape suivante à concevoir et implémenter (nouvelle session)
- **Objectif** : dépasser le `h=0` du POC et reproduire le schéma complet
  d'AloePri (arXiv 2603.01499) avec les matrices clés P̂/Q̂ de l'Algorithme 1,
  chaînées à travers toutes les couches (§5.4) — cible de sécurité mesurée :
  TTRSR hidden ≈ **0,82 %** (Tableau 4, ligne « Noise + KeyMat » ; 0,0 % avec
  la permutation tête/bloc en plus).
- **Lien** : `REPRISE.md` (état général du projet), `RESULTATS_ISA.md`
  (attaques ISA mesurées à ce jour), design précédent
  `2026-08-17-aloepri-poc-complet-design.md` §« Décision h=0 ».

## 1. Contexte — pourquoi cette étape

Le POC a assumé `h=0` (décision 2026-08-18, cf. design doc lignes 195-227) :
sans `h`, les matrices clés dégénèrent en matrices carrées inutiles et ont
été désactivées (`apply_key_matrices=False`, pas de Q̂/P̂ dans l'attention).
Mesures ISA en grandeur nature (modèle servi α_e=0.3, β=8) :

| Canal | Taux récupération ids modèle |
|---|---|
| hidden L1 (peu profond) | 90,9 % |
| hidden L18 (profond) | 4,5 % |
| attn L0 | 9,1 % |

Le papier atteint **0,82 %** sur le canal hidden **avec les matrices clés**
(et 0 % en ajoutant la permutation tête/bloc, déjà active chez nous). C'est
donc la brique manquante pour fermer le canal hidden.

## 2. Ce qui existe déjà (réutilisable, testé)

- **`aloepri_poc/key_matrix.py`** — Algorithme 1 complet :
  `init_key_matrix(d, h, λ, rng)` → `KeyMatrixBase(B, B_inv, E, F, Z, rng)`,
  `key_mat_gen` → **P̂ (d, d+2h)**, `inv_key_mat_gen` → **Q̂ (d+2h, d)**.
  Vérifié : P̂·Q̂ = I (erreur max ~2,3e-15 à d=64, h=8, λ=0.3).
- **`aloepri_poc/embedding_obfuscation.py`** — `obfuscate_embedding` a déjà
  la branche `apply_key_matrices=True` :
  `W̃_embed = Π·W*_embed·P̂_embed`, `W̃_head = Q̂_head·W*_head·Πᵀ`.
  Désactivée par `model_transform.py` ; la docstring explique pourquoi
  (sans chaînage, x·P̂ livré à une couche non transformée → bruit).
- **`aloepri_poc/attention_obfuscation.py`** — point 3 de la docstring :
  l'Algorithme 2 prévoit `Q̂_q/Q̂_k/Q̂_v/P̂_o` sur la frontière hidden_size,
  non appliqués pour la même raison (pas de chaînage).
- **Le schéma du papier** (§5.4, relu p.10-11) est résumé dans le design doc
  lignes 216-223 : un **même changement de base** traverse toutes les couches
  (`φ^attn_X(x)=xP̂`, `ψ^embed_Y(y)=yP̂` — le MÊME P̂ partout, pas un P̂_i par
  couche).

## 3. Le gap à construire

1. **Redimensionnement du réseau en `d+2h` de bout en bout** (le point qui a
   motivé le h=0) : embedding → `(V, d+2h)` ; chaque couche (attention
   q/k/v/o, FFN gate/up/down, RMSNorm) opère sur `d+2h` ; `lm_head` prend
   `d+2h` en entrée. La config `hidden_size` change ; il faut donc
   **re-dimensionner les poids**, pas seulement les transformer en place —
   `transform_streaming.py` est le bon endroit (il réécrit les shards).
2. **Chaînage du changement de base** : à valider sur le papier, mais la
   lecture actuelle (§5.4) indique un **P̂ global unique** : sortie embedding
   = x·P̂ ; chaque couche transformée pour vivre dans l'espace conjugué
   (Q̂ = P̂⁻¹ en entrée de couche, P̂ en sortie — ou directement intégré aux
   poids via les Algorithmes 1-2) ; unembedding absorbe Q̂ = P̂⁻¹ final.
   Si P̂ est unique et global, le chaînage se réduit à : transformer chaque
   couche par conjugaison (poids et biais), sans P̂_i distincts par couche.
3. **FFN** : même traitement de frontière que l'attention (actuellement les
   matrices clés y sont aussi désactivées).
4. **Clés** : décider si les matrices P̂/Q̂ (grosses : (d, d+2h) en float32)
   sont **sauvegardées dans le fichier de clés** ou **régénérées par seed**
   (le tirage est déterministe : `np.random.default_rng(seed)`) — option
   recommandée : régénération par seed + stockage de la seed (déjà dans
   `ObfuscationKeys`), pour ne pas gonfler le secret client.
5. **Serveur** : inchangé en posture (il charge le checkpoint obfusqué, qui
   a simplement des dimensions +2h) — vérifier que `AutoModelForCausalLM`
   accepte un hidden_size non standard (oui, c'est un config).

## 4. Points de design à trancher en session

- **P̂ unique global vs P̂_i par couche** : le papier semble utiliser un
  changement de base unique (§5.4). À relire (p.10-11) avant d'implémenter.
- **h** : le papier utilise **h=128** (Table 10, Appendix D.2). Coût sur
  Qwen3-8B : hidden_size 4096 → 4352 (+6,25 % de mémoire/calcul).
- **Conditionnement bf16** : λ=0.3 borne la norme de P̂ (déjà prévu par
  l'Algorithme 1) ; vérifier que le round-trip reste exact en bf16 (le POC a
  déjà dû corriger Û_vo gaussien → orthogonal pour cette raison ; P̂/Q̂ ne
  sont pas orthogonaux, mais λ=0.3 les garde bien conditionnés).
- **Interaction bruit (α_e) vs matrices clés** : le papier combine les deux ;
  l'ordre des opérations dans `obfuscate_embedding` est déjà le bon
  (bruit puis permutation puis P̂).

## 5. Stratégie de test (TDD)

1. **Modèle jouet Qwen3** (comme `tests/test_qwen3_arch.py`) : avec h>0 et
   chaînage, les logits doivent être préservés modulo la permutation de
   vocabulaire (round-trip exact) ; vérifier P̂·Q̂ = I et les dimensions
   d+2h cohérentes de bout en bout (embedding → couches → lm_head).
2. **`transform_streaming.py`** : adapté pour h>0 (config d+2h, poids
   redimensionnés) ; équivalence bit-à-bit transform local == transform
   Modal (le harnais d'égalité existe déjà).
3. **Vrai modèle** : vérification par échantillons (`verify_transform.py`),
   puis **re-mesure ISA** canal hidden L1 — cible ≈ 0,82 % (ou au moins une
   chute nette vs 90,9 % actuel).
4. Qualité/vitesse : perplexité (measure_quality.py) et tok/s — le surcoût
   d+2h attendu ~6 %.

## 6. Références

- Papier AloePri arXiv 2603.01499 : Algorithme 1 (p.8), §5.2.2 (embedding),
  §5.4 (schéma complet et chaînage, p.10-11), Tableau 4 (p.15), Table 10
  (h=128, Appendix D.2).
- `docs/superpowers/specs/2026-08-17-aloepri-poc-complet-design.md`
  (décision h=0, lignes 195-227).
- `aloepri_poc/key_matrix.py`, `embedding_obfuscation.py`,
  `attention_obfuscation.py` (point 3), `model_transform.py`,
  `transform_streaming.py`, `verify_transform.py`.
- `aloepri_poc/RESULTATS_ISA.md` (mesures avant/après β), `REPRISE.md`.

## 7. Ordre de travail suggéré

1. Relire §5.4 du papier (P̂ global ?) — trancher le point de design 4.1.
2. Tiny model : chaînage P̂ global + h>0 → round-trip exact (test).
3. Étendre attention (Q̂/P̂_o) + FFN au chaînage.
4. Adapter `transform_streaming.py` au redimensionnement d+2h.
5. Vrai modèle : transform, vérification, re-mesure ISA hidden.
6. Déployer sur Modal (volume), mettre à jour RESULTATS/REPRISE.
