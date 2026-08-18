"""Obfuscation d'attention (Algorithme 2 + « Inter-head Permutation », papier
AloePri arXiv 2603.01499v2, page 9).

Algorithme 2, tel qu'imprimé (lignes 6-7) :

    6: W̃_k^{η(i)} = Q̂_k W_k^{η(i)} R̂_qk Ĥ_qk⁻¹ Ẑ_blockᵀ ,  W̃_v^{η(i)} = Q̂_v W_v^{η(i)} Û_vo
    7: W̃_q^{(i)}  = Q̂_q W_q^{(i)}  R̂_qk Ĥ_qk  Ẑ_block  ,  W̃_o^{(i)}  = Û_vo⁻¹ W_o^{(i)} P̂_o

Conventions
-----------
Le papier écrit les poids « ligne = dimension cachée » : W_q est (d, d_head) et
q = x·W_q. Les matrices clés (Q̂/P̂, d×d ici puisque h=0) agissent donc à
gauche, sur la frontière `hidden_size`, et R̂_qk/Ĥ_qk/Ẑ_block/Û_vo à droite,
sur la dimension de tête. HuggingFace stocke la transposée
(`w_q` = (num_heads·d_head, hidden)), donc toute multiplication à droite du
papier devient ici une multiplication à gauche par la transposée du facteur.

Trois points ont dû être tranchés pour que la reparamétrisation soit exacte.

1. Facteurs partagés par GROUPE, pas par tête. Le texte dit « We use
   Algorithm 2 to obfuscate the weights of a group of attention heads
   (W_q^{(i)}, W_k^{η(i)}, W_v^{η(i)}, W_o^{(i)}) » : un tirage par groupe GQA.
   C'est une nécessité mathématique, pas un détail : toutes les têtes Q d'un
   groupe font leur produit scalaire avec la MÊME tête K, donc elles doivent
   porter le même facteur droit, sans quoi rien ne s'annule. Idem pour Û_vo,
   partagé entre la tête V du groupe et les tranches de W_o des têtes Q du
   groupe.

2. Ẑ_block du côté K : Ẑ, pas Ẑᵀ. Le score obfusqué vaut
   q·A·Bᵀ·kᵀ avec A = R̂Ĥ Ẑ (ligne 7) et B le facteur de K (ligne 6) ; il faut
   donc A·Bᵀ = I, c'est-à-dire B = A⁻ᵀ = R̂⁻ᵀ Ĥ⁻ᵀ Ẑ⁻ᵀ = R̂ Ĥ⁻¹ Ẑ (R̂ orthogonale,
   Ĥ diagonale, Ẑ permutation). La ligne 6 imprimée donne bien R̂ et Ĥ⁻¹ dans
   cet ordre, mais Ẑᵀ au lieu de Ẑ⁻ᵀ = Ẑ : le produit devient Ẑ·Ẑ, qui ne vaut
   I que si Ẑ est une involution. C'est le cas tant que BlockPerm ne tire que
   des fenêtres de taille ≤ 2 (toutes les permutations de S₁/S₂ sont des
   involutions), mais plus du tout dès qu'une fenêtre contient un 3-cycle.
   Mesuré : à m_blocks=64 et β=8 (le cas du vrai modèle, d_head=128), 20/20
   tirages donnent ẐẐ ≠ I ; le round-trip avec Ẑᵀ produit alors une erreur de
   l'ordre de l'amplitude du signal (~1.0 pour une sortie d'amplitude ~1.5),
   contre ~1e-5 avec Ẑ. Les trois autres facteurs sont repris tels quels, seule
   l'orientation de Ẑ est corrigée.

3. Matrices clés Q̂_q/Q̂_k/Q̂_v/P̂_o non appliquées. Elles agissent sur la
   frontière `hidden_size` : dans le schéma complet du papier, Q̂ annule le P̂
   de la couche précédente (x̃ = x·P̂, P̂Q̂ = I) et P̂_o prépare la couche
   suivante. Ce POC assume explicitement de ne PAS chaîner les couches
   (`docs/superpowers/specs/2026-08-17-aloepri-poc-complet-design.md`, « chaque
   couche est obfusquée et vérifiée de façon indépendante, sans transformer la
   frontière hidden_size entre couches »). Sans chaînage, appliquer Q̂_q sur
   l'entrée réelle x donne x·Q̂_q·W_q ≠ x·W_q, et aucune autre matrice de la
   couche ne peut le compenser : Task 8 poserait ces poids dans un vrai modèle
   dont la sortie deviendrait du bruit. Elles sont donc omises ici — c'est la
   même limite que celle déjà actée pour le FFN, et la protection anti-ISA
   visée par le POC vient de la permutation tête/bloc, pas des matrices clés.

Ce qui reste (R̂_qk, Ĥ_qk, Ẑ_block, Û_vo, τ_kv, τ_group) s'annule intégralement
à l'intérieur de la couche : la sortie de l'attention est inchangée.

Conditionnement de Û_vo — AVERTISSEMENT pour Task 8/9. La ligne 4 est suivie
telle quelle (Û_vo ~ N(0, 1/d_head)), mais une gaussienne carrée est parfois
très mal conditionnée : à d_head=128, une graine sur six a produit un
||W̃_o||_max de 1249 pour un ||W_o||_max d'origine de 0,45 (×2700). La
reparamétrisation reste exacte en float32 (erreur relative ~1e-3), mais en
bfloat16 — le dtype que le serveur de Task 9 utilise — la sortie de la couche
est détruite : erreur relative mesurée de 3,7 (367 %) sur cette graine, et
encore 6 % sur une graine ordinaire. À traiter dans Task 8 (par exemple en
retirant un Û_vo dont le conditionnement dépasse un seuil, ou en obfusquant
en float32), sinon Task 9 mesurera un bruit numérique et non la dégradation
propre à l'obfuscation.

RoPE — ce module produit une reparamétrisation EXACTE de l'attention sans RoPE
(ce que vérifie le test). Avec RoPE, R̂_qk (rotations 2D par paire) et Ĥ_qk
(scalaire par paire) commutent avec la rotation RoPE et restent exacts ;
Ẑ_block, lui, permute les fréquences à l'intérieur de sa fenêtre : c'est
l'approximation revendiquée par le papier (« shuffling the RoPE's 2×2 blocks …
within a limited window exerts minimal impact on model accuracy »). Deuxième
réserve pour Task 8/9 : ce découpage en paires suppose la convention RoPE
entrelacée (2i, 2i+1) du papier, alors que l'implémentation HF de Qwen2 utilise
la convention en demi-vecteurs (i, i+d_head/2) — à vérifier là-bas.
"""
from dataclasses import dataclass
import random

import torch

from rope_transform import sample_rope_rotation, sample_rope_scaling
from block_perm import block_perm


@dataclass
class ObfuscatedAttention:
    w_q_obf: torch.Tensor
    w_k_obf: torch.Tensor
    w_v_obf: torch.Tensor
    w_o_obf: torch.Tensor


def obfuscate_attention_layer(
    w_q, w_k, w_v, w_o, num_heads, num_kv_heads, d_head, beta, gamma, zeta, seed
):
    hidden_size = w_q.shape[1]
    assert w_q.shape == (num_heads * d_head, hidden_size)
    assert w_k.shape == (num_kv_heads * d_head, hidden_size)
    assert w_v.shape == (num_kv_heads * d_head, hidden_size)
    assert w_o.shape == (hidden_size, num_heads * d_head)
    assert num_heads % num_kv_heads == 0
    group_size = num_heads // num_kv_heads
    m_blocks = d_head // 2

    rng_py = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)

    q_heads = w_q.view(num_heads, d_head, hidden_size)
    k_heads = w_k.view(num_kv_heads, d_head, hidden_size)
    v_heads = w_v.view(num_kv_heads, d_head, hidden_size)
    o_heads = w_o.view(hidden_size, num_heads, d_head)

    q_obf = torch.zeros_like(q_heads)
    k_obf = torch.zeros_like(k_heads)
    v_obf = torch.zeros_like(v_heads)
    o_obf = torch.zeros_like(o_heads)

    # Permutation inter-tête : τ_kv déplace les têtes K/V (et donc les groupes
    # Q/O correspondants, sinon une tête Q n'attendrait plus la bonne tête K),
    # τ_group réordonne les têtes Q/O à l'intérieur de chaque groupe.
    tau_kv = list(range(num_kv_heads))
    rng_py.shuffle(tau_kv)
    tau_group = list(range(group_size))
    rng_py.shuffle(tau_group)

    for g in range(num_kv_heads):
        r_hat = sample_rope_rotation(d_head, seed=rng_py.randrange(2**31))
        h_hat = sample_rope_scaling(d_head, seed=rng_py.randrange(2**31))
        z_pairs = block_perm(beta, gamma, zeta, m_blocks, seed=rng_py.randrange(2**31))
        z_block = torch.kron(z_pairs, torch.eye(2))  # paires -> dimensions

        h_hat_inv = torch.diag(1.0 / torch.diagonal(h_hat))
        a_q = (r_hat @ h_hat @ z_block).to(w_q.dtype)  # ligne 7
        b_k = (r_hat @ h_hat_inv @ z_block).to(w_k.dtype)  # ligne 6, cf. point 2

        # ligne 4 : Û_vo ~ N(0, 1/d_head), inversible presque sûrement
        u_vo = torch.randn(d_head, d_head, generator=gen, dtype=w_v.dtype) / d_head**0.5
        u_vo_inv = torch.linalg.inv(u_vo)

        dst_g = tau_kv[g]
        k_obf[dst_g] = b_k.T @ k_heads[g]
        v_obf[dst_g] = u_vo.T @ v_heads[g]
        for p in range(group_size):
            src_h = g * group_size + p
            dst_h = dst_g * group_size + tau_group[p]
            q_obf[dst_h] = a_q.T @ q_heads[src_h]
            o_obf[:, dst_h, :] = o_heads[:, src_h, :] @ u_vo_inv.T

    return ObfuscatedAttention(
        q_obf.reshape(num_heads * d_head, hidden_size),
        k_obf.reshape(num_kv_heads * d_head, hidden_size),
        v_obf.reshape(num_kv_heads * d_head, hidden_size),
        o_obf.reshape(hidden_size, num_heads * d_head),
    )
