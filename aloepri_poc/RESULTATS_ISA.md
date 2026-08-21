# Résultats — Attaques ISA par gradient sur Qwen3-8B obfusqué (Modal)

Méthode et résultats mesurés en grandeur nature (2026-08-21) sur le modèle
obfusqué servi sur Modal (`aloepri_modal/app.py::isa_attack`, GPU A100-40GB).

Sources : AloePri arXiv 2603.01499 (Appendix D.1, Tableau 4) et
« Depth Gives a False Sense of Privacy: LLM Internal States Inversion »,
arXiv 2507.16372 (méthode d'inversion par optimisation en deux phases).

## Modèle de menace

L'attaquant est **l'opérateur du serveur** : il possède les poids obfusqués
et observe tout ce qui se passe pendant l'inférence d'une requête (états
cachés, scores d'attention). Il **n'a pas** la clé de permutation (côté
client).

## Méthode (implémentée dans `aloepri_poc/isa_attack.py`)

1. **Capture** : pour le prompt secret (IDs permutés, l'entrée réelle du
   modèle), on enregistre un état interne — état caché d'une couche
   (`channel=hidden`) ou pondérations d'attention (`channel=attn`).
2. **Paramétrisation** : le candidat X2 est représenté par des **logits par
   position** P ∈ R^{T×vocab} ; l'entrée du modèle est
   `embeds = softmax(P/τ) @ W_embed` (différentiable → le gradient remonte
   jusqu'à P à travers la table d'embedding).
3. **Phase 1** : Adam + recuit de température (τ : 3 → 0,1) sur une perte
   **relative** MSE/variance (les états cachés d'un vrai LLM ont des
   amplitudes énormes ; une MSE brute écrase le gradient).
4. **Phase 2** (2507.16372) : ré-initialisation des logits près de l'argmax
   de la phase 1 + optimisation à température basse — corrige les choix
   discrets figés prématurément par le recuit.
5. **Résultat** : ids_récupérés = argmax(P) ; métrique = taux de
   correspondance avec les IDs réellement envoyés au modèle.

## Résultats (prompt secret de 22 tokens, modèle α_e=0.3, β=1)

| Canal | Couche | Taux de récupération des ids du modèle | Loss (rel.) |
|---|---|---|---|
| hidden | 1 (peu profonde) | **95,5 %** (21/22) | 0,020 |
| hidden | 18 (profonde) | 4,5 % (1/22) | 0,093 |
| attn | 0 | 27,3 % (6/22) | 1,05 (divergente) |

## Interprétation

1. **L'attaque par gradient fonctionne** : les états cachés PEU PROFONDS
   s'inversent presque parfaitement (95,5 %), conformément au papier
   2507.16372 (« low-depth ISs »). Les états profonds résistent (4,5 %) —
   le canal attention est faible avec cette méthode (27 %), comme sur le
   modèle jouet (sous-détermination).

2. **Ce que l'attaquant récupère, ce sont les IDs PERMUTÉS** — l'entrée que
   le modèle a réellement vue. Sans la clé de permutation (jamais sur le
   serveur), il ne peut pas les traduire en texte : **la permutation est la
   défense effective contre ISA** dans notre schéma (la récupération à
   95,5 % porte sur des ids permutés, pas sur le texte clair).

3. **Le bruit d'embedding (α_e) ne suffit pas à protéger les ids** à faible
   profondeur (95,5 % de récupération malgré α_e=0.3) — cohérent avec le
   Tableau 4 d'AloePri (le bruit seul laisse 40 % de TTRSR ; ce sont les
   protections d'attention qui font tomber le taux à 0 %).

4. **Conséquence sur notre modèle servi** : le réglage qualité retenu
   (β=1 → Ẑ_block = identité) désactive la permutation de blocs d'attention,
   la protection qui, selon AloePri, fait passer le TTRSR d'attention à 0 %.
   Pour restaurer cette défense, il faut β>1 (Ẑ actif) — au prix d'un peu de
   qualité (le POC Qwen2.5 mesurait +19,1 % de perplexité à β=8 vs +18,7 % à
   β=1 : l'écart est faible).

## Ce qui n'est pas encore mesuré

- Comparaison **baseline vs obfusqué** sur le vrai modèle (l'attaque sur la
  baseline récupérerait des ids CLAIRS — le texte fuirait ; sur l'obfusqué
  il récupère des ids permutés). À faire pour quantifier précisément le
  « coût » de la défense.
- Attaque attention multi-couches / autre loss (la méthode actuelle diverge).
- Effet de β=8 (Ẑ actif) sur la récupération via attention.
- TFMA/SDA et Attn-IA/Gate-IA (hors périmètre de cette session).

## Fichiers

- `aloepri_poc/isa_attack.py` — moteur d'attaque (module + CLI).
- `aloepri_poc/tests/test_isa_attack.py` — canal hidden robuste sur modèle
  jouet (récupère les ids du modèle ; ids permutés sur l'obfusqué).
- `aloepri_modal/app.py::isa_attack` — démonstration grandeur nature sur le
  modèle réel (ids permutés passés en argument, aucune clé sur Modal).
