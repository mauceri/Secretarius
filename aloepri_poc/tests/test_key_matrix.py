import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from key_matrix import init_key_matrix, key_mat_gen, inv_key_mat_gen


def test_key_matrices_are_exact_inverses():
    rng = np.random.default_rng(0)
    d, h, lam = 16, 128, 0.3
    base = init_key_matrix(d, h, lam, rng)

    p_hat = key_mat_gen(base)
    q_hat = inv_key_mat_gen(base)

    # Algorithm 1 (arXiv 2603.01499v2, p.8): P_hat = [B C E] Z is d x (d+2h),
    # Q_hat = Z^T [B^-1; F; D] is (d+2h) x d. Their product is the d x d
    # identity; the factors themselves are not square. See task-2-report.md.
    assert p_hat.shape == (d, d + 2 * h)
    assert q_hat.shape == (d + 2 * h, d)
    np.testing.assert_allclose(p_hat @ q_hat, np.eye(d), atol=1e-5)


def test_two_calls_produce_different_matrices():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    d, h, lam = 16, 128, 0.3
    p1 = key_mat_gen(init_key_matrix(d, h, lam, rng1))
    p2 = key_mat_gen(init_key_matrix(d, h, lam, rng2))
    assert not np.allclose(p1, p2)
