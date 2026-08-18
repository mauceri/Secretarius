"""Obfuscation embedding/unembedding : bruit + permutation + matrices clés (papier §5.2.2)."""
from dataclasses import dataclass
import random

import numpy as np
import torch

from key_matrix import init_key_matrix, key_mat_gen, inv_key_mat_gen


@dataclass
class ObfuscatedEmbedding:
    w_embed_obf: torch.Tensor
    w_head_obf: torch.Tensor
    permutation: dict  # token clair -> token permuté
    unpermute: dict  # token permuté -> token clair


def obfuscate_embedding(w_embed, w_head, alpha_e, alpha_h, lam, h, seed):
    vocab_size, d = w_embed.shape
    assert w_head.shape == (vocab_size, d)

    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    # bruit gaussien : W* = W + alpha * bruit
    noise_e = torch.randn(w_embed.shape, generator=torch.Generator().manual_seed(seed))
    noise_h = torch.randn(w_head.shape, generator=torch.Generator().manual_seed(seed + 1))
    w_embed_star = w_embed + alpha_e * noise_e
    w_head_star = w_head + alpha_h * noise_h

    # permutation secrète du vocabulaire
    clear_ids = list(range(vocab_size))
    permuted_ids = list(range(vocab_size))
    rng_py.shuffle(permuted_ids)
    permutation = dict(zip(clear_ids, permuted_ids))
    unpermute = {v: k for k, v in permutation.items()}

    # Ligne `p` (un ID permuté) de la table obfusquée doit porter les
    # données du token clair `unpermute[p]` — c'est ce token-là que le
    # serveur doit reconnaître quand le client lui envoie l'ID permuté `p`.
    # D'où l'indexation par `unpermute` (== Π du papier), pas par
    # `permutation` (== Π⁻¹) : cf. task-3-report.md pour la dérivation et
    # la vérification numérique qui a débusqué le sens inverse dans le
    # brouillon initial.
    inv_perm_index = torch.tensor([unpermute[i] for i in range(vocab_size)])

    # matrices clés (Algorithme 1) — une paire pour l'embedding, une pour le head
    base_embed = init_key_matrix(d, h, lam, rng_np)
    p_hat_embed = torch.tensor(key_mat_gen(base_embed), dtype=w_embed.dtype)

    base_head = init_key_matrix(d, h, lam, rng_np)
    q_hat_head = torch.tensor(inv_key_mat_gen(base_head), dtype=w_head.dtype)

    # W̃_embed = Π · W*_embed · P̂_embed
    w_embed_permuted_rows = w_embed_star[inv_perm_index]
    w_embed_obf = w_embed_permuted_rows @ p_hat_embed

    # W̃_head = Q̂_head · W*_head · Πᵀ
    # (la sélection de lignes et la multiplication par la matrice clé
    # commutent : l'une agit sur l'axe vocab, l'autre sur l'axe d — l'ordre
    # n'a pas d'importance mathématiquement.)
    w_head_transformed = w_head_star @ q_hat_head.T
    w_head_obf = w_head_transformed[inv_perm_index]

    return ObfuscatedEmbedding(w_embed_obf, w_head_obf, permutation, unpermute)
