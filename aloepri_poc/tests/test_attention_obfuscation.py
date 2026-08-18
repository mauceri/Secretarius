import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from attention_obfuscation import obfuscate_attention_layer
from block_perm import block_perm


def naive_gqa_attention(x, w_q, w_k, w_v, w_o, num_heads, num_kv_heads, d_head):
    seq_len, hidden_size = x.shape
    group_size = num_heads // num_kv_heads

    q = (x @ w_q.T).view(seq_len, num_heads, d_head)
    k = (x @ w_k.T).view(seq_len, num_kv_heads, d_head)
    v = (x @ w_v.T).view(seq_len, num_kv_heads, d_head)

    outputs = []
    for h in range(num_heads):
        kv_head = h // group_size
        scores = q[:, h] @ k[:, kv_head].T / (d_head**0.5)
        weights = torch.softmax(scores, dim=-1)
        outputs.append(weights @ v[:, kv_head])
    concat = torch.cat(outputs, dim=-1)
    return concat @ w_o.T


def _random_layer(hidden_size, num_heads, num_kv_heads, d_head, seq_len, seed):
    torch.manual_seed(seed)
    return (
        torch.randn(num_heads * d_head, hidden_size) * 0.1,
        torch.randn(num_kv_heads * d_head, hidden_size) * 0.1,
        torch.randn(num_kv_heads * d_head, hidden_size) * 0.1,
        torch.randn(hidden_size, num_heads * d_head) * 0.1,
        torch.randn(seq_len, hidden_size),
    )


def test_obfuscated_attention_preserves_output():
    hidden_size, num_heads, num_kv_heads, d_head = 32, 8, 2, 8
    seq_len = 5
    w_q, w_k, w_v, w_o, x = _random_layer(
        hidden_size, num_heads, num_kv_heads, d_head, seq_len, seed=0
    )

    baseline = naive_gqa_attention(x, w_q, w_k, w_v, w_o, num_heads, num_kv_heads, d_head)

    obf = obfuscate_attention_layer(
        w_q, w_k, w_v, w_o,
        num_heads=num_heads, num_kv_heads=num_kv_heads, d_head=d_head,
        beta=2, gamma=0.5, zeta=1e3, seed=0,
    )
    obf_output = naive_gqa_attention(
        x, obf.w_q_obf, obf.w_k_obf, obf.w_v_obf, obf.w_o_obf,
        num_heads, num_kv_heads, d_head,
    )

    torch.testing.assert_close(obf_output, baseline, atol=1e-3, rtol=1e-3)


def test_round_trip_holds_for_mha_mqa_and_other_seeds():
    """La composition doit tenir hors du seul cas GQA 8/2 du test principal :
    MHA (num_kv_heads == num_heads), MQA (num_kv_heads == 1), et plusieurs
    graines — une composition juste « par chance » sur un tirage ne survit pas
    à ce balayage.

    `d_head=32` (et non 8) est délibéré : avec `d_head=8` on n'a que
    `m_blocks=4` blocs RoPE, BlockPerm n'y tire que des fenêtres de taille ≤ 2
    et toutes les permutations de S₁/S₂ sont des involutions (ẐẐ = I). Dans ce
    régime Ẑ et Ẑᵀ sont interchangeables et le round-trip ne peut PAS
    distinguer l'orientation de Ẑ_block — alors que le vrai modèle
    (d_head=128 → m_blocks=64) tire des 3-cycles à tous les coups. Le test
    principal ci-dessus est donc aveugle à ce point précis ; celui-ci ne l'est
    pas (cf. l'assertion de régime plus bas)."""
    hidden_size, num_heads, d_head, seq_len = 64, 8, 32, 6

    # régime discriminant : au moins une partie des tirages BlockPerm doit
    # contenir un cycle de longueur ≥ 3, sinon ce test ne vaudrait pas mieux
    # que le précédent.
    non_involutive = sum(
        bool(
            torch.linalg.matrix_norm(
                block_perm(8, 1e3, 1e3, d_head // 2, seed=s) @ block_perm(8, 1e3, 1e3, d_head // 2, seed=s)
                - torch.eye(d_head // 2)
            )
            > 1e-6
        )
        for s in range(20)
    )
    assert non_involutive >= 10, (
        "paramètres non discriminants : BlockPerm ne tire que des involutions"
    )

    for num_kv_heads in (8, 4, 1):
        for seed in (1, 2, 3):
            w_q, w_k, w_v, w_o, x = _random_layer(
                hidden_size, num_heads, num_kv_heads, d_head, seq_len, seed
            )
            baseline = naive_gqa_attention(
                x, w_q, w_k, w_v, w_o, num_heads, num_kv_heads, d_head
            )
            obf = obfuscate_attention_layer(
                w_q, w_k, w_v, w_o,
                num_heads=num_heads, num_kv_heads=num_kv_heads, d_head=d_head,
                beta=8, gamma=1e3, zeta=1e3, seed=seed,
            )
            got = naive_gqa_attention(
                x, obf.w_q_obf, obf.w_k_obf, obf.w_v_obf, obf.w_o_obf,
                num_heads, num_kv_heads, d_head,
            )
            torch.testing.assert_close(
                got, baseline, atol=1e-3, rtol=1e-3,
                msg=f"round-trip cassé pour num_kv_heads={num_kv_heads}, seed={seed}",
            )


def _row_space_projector(m):
    """Projecteur orthogonal sur l'espace engendré par les lignes de `m`."""
    basis = torch.linalg.qr(m.T).Q  # (hidden, d_head)
    return basis @ basis.T


def test_heads_are_actually_permuted_and_transformed():
    """Le round-trip seul ne distingue pas une implémentation qui ne ferait
    rien : il passerait aussi avec l'identité. Ce test vérifie que les poids
    changent réellement ET que la permutation inter-tête est appliquée de
    façon cohérente.

    Chaque tête est transformée par multiplication à gauche par une matrice
    inversible (côté d_head), ce qui laisse l'espace des lignes (sous-espace
    de dimension d_head de R^hidden) invariant. On peut donc retrouver la
    permutation appliquée en appariant les espaces de lignes, sans connaître
    les clés — et vérifier que les têtes Q suivent bien leur groupe K/V (sans
    quoi le calcul serait faux, ce que le round-trip attraperait, mais aussi
    qu'elles sont bien déplacées, ce qu'il n'attraperait pas)."""
    hidden_size, num_heads, num_kv_heads, d_head = 64, 8, 2, 8
    group_size = num_heads // num_kv_heads
    w_q, w_k, w_v, w_o, _ = _random_layer(
        hidden_size, num_heads, num_kv_heads, d_head, 4, seed=7
    )
    obf = obfuscate_attention_layer(
        w_q, w_k, w_v, w_o,
        num_heads=num_heads, num_kv_heads=num_kv_heads, d_head=d_head,
        beta=8, gamma=1e3, zeta=1e3, seed=7,
    )

    assert not torch.allclose(obf.w_q_obf, w_q)
    assert not torch.allclose(obf.w_k_obf, w_k)
    assert not torch.allclose(obf.w_v_obf, w_v)
    assert not torch.allclose(obf.w_o_obf, w_o)

    def match(clear, obfusc, n_heads):
        clear_heads = clear.view(n_heads, d_head, -1)
        obf_heads = obfusc.view(n_heads, d_head, -1)
        mapping = []
        for src in range(n_heads):
            p_src = _row_space_projector(clear_heads[src])
            dists = [
                torch.linalg.matrix_norm(p_src - _row_space_projector(obf_heads[dst]))
                for dst in range(n_heads)
            ]
            best = int(torch.tensor(dists).argmin())
            assert dists[best] < 1e-3, "aucune tête obfusquée ne correspond"
            mapping.append(best)
        return mapping

    tau_kv = match(w_k, obf.w_k_obf, num_kv_heads)
    assert sorted(tau_kv) == list(range(num_kv_heads))
    assert match(w_v, obf.w_v_obf, num_kv_heads) == tau_kv, "K et V doivent suivre τ_kv"

    sigma = match(w_q, obf.w_q_obf, num_heads)
    assert sorted(sigma) == list(range(num_heads))
    assert sigma != list(range(num_heads)), "les têtes Q ne sont pas permutées"
    for h, dst in enumerate(sigma):
        assert dst // group_size == tau_kv[h // group_size], (
            "une tête Q a changé de groupe sans suivre sa tête K/V"
        )
