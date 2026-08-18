# POC AloePri complet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire et mesurer un pipeline complet d'obfuscation covariante
(inspiré du papier AloePri, arXiv 2603.01499v2) sur Qwen2.5-7B-Instruct :
embedding/unembedding, FFN et attention obfusqués, servis sur un Pod RunPod,
avec mesure de la dégradation de qualité et de l'overhead de vitesse.

**Architecture:** Transformation offline des poids HF safetensors (permutation
+ bruit + matrices clés sur l'embedding, permutation + scaling sur le FFN,
rotation RoPE + permutation par blocs + permutation inter-tête sur
l'attention) ; wrapper client qui permute/dépermute les IDs de tokens ; petit
serveur HTTP `transformers` sur un Pod RunPod qui sert le modèle obfusqué.

**Tech Stack:** Python, PyTorch, `transformers` (HF), `datasets`, pytest,
RunPod (Pod GPU RTX A5000, Community Cloud).

## Global Constraints

- Modèle cible : `Qwen/Qwen2.5-7B-Instruct` (HF safetensors, pas GGUF).
- Toute transformation de poids doit être une reparamétrisation exactement
  inversible (les clés/inverses annulent la transformation) — vérifiée par
  un test de round-trip à chaque tâche, pas seulement en bout de chaîne.
- Les clés secrètes (permutations, matrices, bruit) ne quittent jamais le
  client — ne jamais les sérialiser avec le modèle obfusqué.
- Source de référence pour toute formule d'obfuscation :
  `/tmp/claude-1000/-home-mauceric/c699f98b-d23a-4918-a6ce-5f391cfa5889/scratchpad/aloepri.pdf`
  (si absent, retélécharger : `curl -sL -o aloepri.pdf "https://arxiv.org/pdf/2603.01499v2"`).
  Algorithme 1 (matrices clés) page 8, Algorithme 2 (attention intra-tête)
  page 9, permutation inter-tête texte page 9 (section « Inter-head
  Permutation »).
- Hyperparamètres de départ (Table 10 du papier) : `α_e = 1.0`, `α_h = 0.2`,
  `λ = 0.3`, `h (expansion) = 128`, `β (fenêtre max BlockPerm) = 8`,
  `γ = 1e³`.
- Tests unitaires rapides (pas de réseau, pas de GPU, pas de téléchargement
  du modèle 7B) séparés des scripts d'expérience réels (qui téléchargent le
  tokenizer/modèle/corpus et peuvent être lents) — les seconds ne sont pas
  dans la suite pytest par défaut.

---

## File Structure

```
Secretarius/
  aloepri_freq_attack/
    __init__.py
    frequency_attack.py       # Étape 0 : logique de l'attaque TFMA-style
    run_frequency_experiment.py  # script réel (tokenizer Qwen + corpus fr)
    tests/
      __init__.py
      test_frequency_attack.py

  aloepri_poc/
    __init__.py
    requirements.txt
    key_matrix.py              # Algorithme 1 : INIT, KeyMatGen, InvKeyMatGen
    embedding_obfuscation.py   # bruit + permutation + matrices clés
    ffn_obfuscation.py         # permutation + scaling (SwiGLU-aware)
    rope_transform.py          # rotation RoPE + scaling (partie d'Algo 2)
    block_perm.py              # BlockPerm (fenêtre dynamique)
    attention_obfuscation.py   # assemble rope_transform+block_perm+key_matrix
                                # + permutation inter-tête (GQA-aware)
    model_transform.py         # orchestration : charge Qwen2.5-7B, applique
                                # tout, sauvegarde poids obfusqués + clés
    client_wrapper.py          # tokenize+permute / dépermute+detokenize
    server.py                  # serveur HTTP transformers (pour le Pod)
    measure_quality.py         # perplexité/exactitude obfusqué vs baseline
    measure_speed.py           # tokens/s, latence obfusqué vs baseline
    tests/
      __init__.py
      test_key_matrix.py
      test_embedding_obfuscation.py
      test_ffn_obfuscation.py
      test_rope_transform.py
      test_block_perm.py
      test_attention_obfuscation.py
      test_client_wrapper.py
```

---

### Task 1: Étape 0 — Simulation d'attaque fréquence (locale, sans GPU)

**Files:**
- Create: `aloepri_freq_attack/__init__.py`
- Create: `aloepri_freq_attack/frequency_attack.py`
- Create: `aloepri_freq_attack/run_frequency_experiment.py`
- Test: `aloepri_freq_attack/tests/__init__.py`
- Test: `aloepri_freq_attack/tests/test_frequency_attack.py`

**Interfaces:**
- Produces: `build_frequency_ranking(token_ids: list[int]) -> list[int]`,
  `random_permutation(vocab_ids: list[int], seed: int) -> dict[int, int]`,
  `apply_permutation(token_ids: list[int], permutation: dict[int, int]) -> list[int]`,
  `tfma_recovery_rate(observed_permuted_ids: list[int], reference_token_ids: list[int], permutation: dict[int, int], top_k: int) -> float`.
  Aucune tâche ultérieure ne consomme ces fonctions (Étape 0 est indépendante
  du reste du POC) — interfaces internes à ce module uniquement.

- [ ] **Step 1: Write the failing tests**

```python
# aloepri_freq_attack/tests/test_frequency_attack.py
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from frequency_attack import (
    build_frequency_ranking,
    random_permutation,
    apply_permutation,
    tfma_recovery_rate,
)


def test_build_frequency_ranking_orders_by_descending_count():
    tokens = [1, 1, 1, 2, 2, 3]
    assert build_frequency_ranking(tokens) == [1, 2, 3]


def test_permutation_round_trip():
    vocab = list(range(20))
    perm = random_permutation(vocab, seed=0)
    inverse = {v: k for k, v in perm.items()}
    tokens = [3, 7, 19, 0]
    permuted = apply_permutation(tokens, perm)
    restored = apply_permutation(permuted, inverse)
    assert restored == tokens


def test_tfma_recovers_dominant_token_with_enough_data():
    vocab = [0, 1, 2, 3, 4]
    weights = [50, 20, 15, 10, 5]
    permutation = random_permutation(vocab, seed=1)

    rng = random.Random(42)
    reference_tokens = rng.choices(vocab, weights=weights, k=20000)
    observed_clear = rng.choices(vocab, weights=weights, k=20000)
    observed_permuted = apply_permutation(observed_clear, permutation)

    rate = tfma_recovery_rate(observed_permuted, reference_tokens, permutation, top_k=1)
    assert rate == 1.0


def test_tfma_recovery_rate_is_bounded():
    vocab = list(range(10))
    weights = [1] * 10
    permutation = random_permutation(vocab, seed=2)
    rng = random.Random(7)
    reference_tokens = rng.choices(vocab, weights=weights, k=5000)
    observed_permuted = apply_permutation(
        rng.choices(vocab, weights=weights, k=5000), permutation
    )
    rate = tfma_recovery_rate(observed_permuted, reference_tokens, permutation, top_k=5)
    assert 0.0 <= rate <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius && python -m pytest aloepri_freq_attack/tests/test_frequency_attack.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'frequency_attack'`

- [ ] **Step 3: Write minimal implementation**

```python
# aloepri_freq_attack/frequency_attack.py
"""Simulation d'attaque par fréquence de tokens (TFMA-style)."""
from collections import Counter
import random


def build_frequency_ranking(token_ids):
    """Rang de fréquence décroissante : ranking[0] = token le plus fréquent."""
    counts = Counter(token_ids)
    return [tok for tok, _ in counts.most_common()]


def random_permutation(vocab_ids, seed):
    rng = random.Random(seed)
    shuffled = list(vocab_ids)
    rng.shuffle(shuffled)
    return dict(zip(vocab_ids, shuffled))


def apply_permutation(token_ids, permutation):
    return [permutation[t] for t in token_ids]


def tfma_recovery_rate(observed_permuted_ids, reference_token_ids, permutation, top_k):
    """
    Simule TFMA : l'attaquant classe les IDs permutés observés par fréquence,
    classe un corpus de référence en clair par fréquence, et suppose que le
    rang k du classement observé correspond au rang k du classement de
    référence. Mesure le % de tokens des top_k les plus fréquents (en clair)
    correctement retrouvés.
    """
    if top_k <= 0:
        return 0.0

    observed_ranking = build_frequency_ranking(observed_permuted_ids)
    reference_ranking = build_frequency_ranking(reference_token_ids)
    inverse_permutation = {v: k for k, v in permutation.items()}

    correct = 0
    for rank in range(top_k):
        if rank >= len(observed_ranking) or rank >= len(reference_ranking):
            break
        guessed_clear_tok = reference_ranking[rank]
        true_clear_tok = inverse_permutation.get(observed_ranking[rank])
        if guessed_clear_tok == true_clear_tok:
            correct += 1
    return correct / top_k
```

Create empty `aloepri_freq_attack/__init__.py` and `aloepri_freq_attack/tests/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius && python -m pytest aloepri_freq_attack/tests/test_frequency_attack.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the real experiment script (pas de test unitaire — dépend du réseau)**

```python
# aloepri_freq_attack/run_frequency_experiment.py
"""
Mesure réelle de l'Étape 0 : volume de corpus nécessaire pour qu'un
attaquant retrouve les tokens les plus fréquents via TFMA, sur le vrai
tokenizer Qwen2.5-7B et un corpus français réel.

Usage: python run_frequency_experiment.py
Nécessite : `pip install transformers datasets`, accès réseau (télécharge
le tokenizer et un échantillon Wikipedia FR en streaming).
"""
from transformers import AutoTokenizer
from datasets import load_dataset

from frequency_attack import random_permutation, apply_permutation, tfma_recovery_rate


def load_token_stream(tokenizer, n_articles):
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    tokens = []
    for i, row in enumerate(ds):
        if i >= n_articles:
            break
        tokens.extend(tokenizer.encode(row["text"]))
    return tokens


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    vocab_ids = list(range(tokenizer.vocab_size))
    permutation = random_permutation(vocab_ids, seed=0)

    all_tokens = load_token_stream(tokenizer, n_articles=200)
    reference_tokens = all_tokens[: len(all_tokens) // 2]
    target_tokens = all_tokens[len(all_tokens) // 2 :]

    print("volume_observé\ttop10\ttop100\ttop1000")
    for n in [100, 1000, 10000, 100000, len(target_tokens)]:
        n = min(n, len(target_tokens))
        observed_clear = target_tokens[:n]
        observed_permuted = apply_permutation(observed_clear, permutation)
        rates = [
            tfma_recovery_rate(observed_permuted, reference_tokens, permutation, k)
            for k in (10, 100, 1000)
        ]
        print(f"{n}\t{rates[0]:.3f}\t{rates[1]:.3f}\t{rates[2]:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
cd ~/Secretarius
git add aloepri_freq_attack/
git commit -m "feat(aloepri): étape 0 — simulation d'attaque par fréquence de tokens"
```

---

### Task 2: Génération des matrices clés (Algorithme 1)

**Files:**
- Create: `aloepri_poc/__init__.py`
- Create: `aloepri_poc/requirements.txt`
- Create: `aloepri_poc/key_matrix.py`
- Test: `aloepri_poc/tests/__init__.py`
- Test: `aloepri_poc/tests/test_key_matrix.py`

**Interfaces:**
- Produces: `init_key_matrix(d: int, h: int, lam: float, rng: numpy.random.Generator) -> KeyMatrixBase`
  (dataclass avec champs `B, B_inv, E, F, Z`), `key_mat_gen(base: KeyMatrixBase) -> np.ndarray`
  (P̂, shape `(d, d)`), `inv_key_mat_gen(base: KeyMatrixBase) -> np.ndarray` (Q̂, shape `(d, d)`).
  Consommé par Task 3 (embedding), Task 7 (attention).

- [ ] **Step 0 (obligatoire avant de coder) : relire l'Algorithme 1 dans le PDF primaire**

```bash
# si le fichier n'existe pas déjà :
mkdir -p /tmp/claude-1000/-home-mauceric/c699f98b-d23a-4918-a6ce-5f391cfa5889/scratchpad
cd /tmp/claude-1000/-home-mauceric/c699f98b-d23a-4918-a6ce-5f391cfa5889/scratchpad
curl -sL -o aloepri.pdf "https://arxiv.org/pdf/2603.01499v2"
```

Lire la page 8 (image, pas texte extrait — `pdftotext` mélange les colonnes
sur ce document). Vérifier précisément chaque dimension de matrice
(`B, V ∈ R^{d×d}` ; `E1 ∈ R^{d×h/2}`, `E2 ∈ R^{h/2×h}`, `E = E1·E2 ∈ R^{d×h}` ;
`F1 ∈ R^{h×h/2}`, `F2 ∈ R^{h/2×d}`, `F = F1·F2 ∈ R^{h×d}` ; `Z ∈ O_{d+2h}` ;
`C ∈ R^{d×h}` colonnes dans `null(Fᵀ)` ; `D ∈ R^{h×d}` lignes dans `null(E)`)
avant d'écrire le code — c'est le point le plus dense mathématiquement du
papier, ne pas coder de mémoire sur la seule base de ce plan.

- [ ] **Step 1: Write the failing test (invariant, pas les valeurs intermédiaires)**

```python
# aloepri_poc/tests/test_key_matrix.py
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

    assert p_hat.shape == (d, d)
    assert q_hat.shape == (d, d)
    np.testing.assert_allclose(p_hat @ q_hat, np.eye(d), atol=1e-5)


def test_two_calls_produce_different_matrices():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    d, h, lam = 16, 128, 0.3
    p1 = key_mat_gen(init_key_matrix(d, h, lam, rng1))
    p2 = key_mat_gen(init_key_matrix(d, h, lam, rng2))
    assert not np.allclose(p1, p2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_key_matrix.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'key_matrix'`

- [ ] **Step 3: Implement, en suivant exactement l'Algorithme 1 relu à l'étape 0**

Implémenter `aloepri_poc/key_matrix.py` avec une dataclass `KeyMatrixBase`
et les trois fonctions `init_key_matrix`, `key_mat_gen`, `inv_key_mat_gen`,
en transcrivant fidèlement les formules vérifiées à l'étape 0 (sample U
depuis le groupe orthogonal via QR d'une matrice gaussienne ; construire
C/D via l'espace nul de F^T / E — `scipy.linalg.null_space` convient pour
cette construction). Ne pas deviner une formule alternative si la lecture
du PDF laisse un doute : itérer sur le test de l'étape 1 (l'invariant
`P̂·Q̂ = I`) jusqu'à ce qu'il passe, c'est le critère de correction, pas la
ressemblance avec ce plan.

Créer `aloepri_poc/__init__.py` (vide) et `aloepri_poc/tests/__init__.py` (vide).

`aloepri_poc/requirements.txt` :
```
torch>=2.0
transformers>=4.40
numpy
scipy
fastapi
uvicorn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_key_matrix.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add aloepri_poc/__init__.py aloepri_poc/requirements.txt aloepri_poc/key_matrix.py aloepri_poc/tests/
git commit -m "feat(aloepri): algorithme 1 — génération des matrices clés"
```

---

### Task 3: Obfuscation embedding/unembedding (bruit + permutation + matrices clés)

**Files:**
- Create: `aloepri_poc/embedding_obfuscation.py`
- Test: `aloepri_poc/tests/test_embedding_obfuscation.py`

**Interfaces:**
- Consumes: `key_matrix.init_key_matrix`, `key_matrix.key_mat_gen`, `key_matrix.inv_key_mat_gen` (Task 2).
- Produces: `obfuscate_embedding(w_embed: torch.Tensor, w_head: torch.Tensor, alpha_e: float, alpha_h: float, lam: float, h: int, seed: int) -> ObfuscatedEmbedding`
  (dataclass : `w_embed_obf, w_head_obf` obfusqués + `permutation` (dict
  clair→permuté) + `unpermute` (dict permuté→clair), gardés séparément — les
  poids obfusqués partent au serveur, `permutation`/`unpermute` restent
  côté client). Consommé par Task 8 (orchestration modèle complet).

- [ ] **Step 1: Write the failing test**

```python
# aloepri_poc/tests/test_embedding_obfuscation.py
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from embedding_obfuscation import obfuscate_embedding


def test_obfuscated_embedding_round_trip_preserves_output_up_to_noise():
    torch.manual_seed(0)
    vocab_size, d = 50, 16
    w_embed = torch.randn(vocab_size, d)
    w_head = torch.randn(vocab_size, d)

    result = obfuscate_embedding(
        w_embed, w_head, alpha_e=0.0, alpha_h=0.0, lam=0.3, h=32, seed=0
    )

    assert result.w_embed_obf.shape == w_embed.shape
    assert result.w_head_obf.shape == w_head.shape
    # sans bruit (alpha=0), l'obfuscation est une reparamétrisation exacte :
    # regarder un token clair t, le convertir en ID permuté, lire la ligne
    # correspondante de l'embedding obfusqué, et vérifier qu'elle reproduit
    # (via la matrice clé) la ligne originale.
    clear_token = 5
    permuted_token = result.permutation[clear_token]
    assert result.unpermute[permuted_token] == clear_token


def test_permutation_is_a_bijection_over_the_vocabulary():
    torch.manual_seed(1)
    vocab_size, d = 30, 8
    w_embed = torch.randn(vocab_size, d)
    w_head = torch.randn(vocab_size, d)
    result = obfuscate_embedding(
        w_embed, w_head, alpha_e=1.0, alpha_h=0.2, lam=0.3, h=16, seed=1
    )
    assert sorted(result.permutation.values()) == list(range(vocab_size))
    assert sorted(result.unpermute.values()) == list(range(vocab_size))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_embedding_obfuscation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'embedding_obfuscation'`

- [ ] **Step 3: Write implementation**

```python
# aloepri_poc/embedding_obfuscation.py
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

    perm_index = torch.tensor([permutation[i] for i in range(vocab_size)])
    inv_perm_index = torch.tensor([unpermute[i] for i in range(vocab_size)])

    # matrices clés (Algorithme 1) — une paire pour l'embedding, une pour le head
    base_embed = init_key_matrix(d, h, lam, rng_np)
    p_hat_embed = torch.tensor(key_mat_gen(base_embed), dtype=w_embed.dtype)

    base_head = init_key_matrix(d, h, lam, rng_np)
    q_hat_head = torch.tensor(inv_key_mat_gen(base_head), dtype=w_head.dtype)

    # W̃_embed = Π · W*_embed · P̂_embed  (Π = permutation des lignes)
    w_embed_permuted_rows = w_embed_star[perm_index]
    w_embed_obf = w_embed_permuted_rows @ p_hat_embed

    # W̃_head = Q̂_head · W*_head · Πᵀ  (Πᵀ permute les lignes en sens inverse)
    w_head_transformed = w_head_star @ q_hat_head.T
    w_head_obf = w_head_transformed[inv_perm_index]

    return ObfuscatedEmbedding(w_embed_obf, w_head_obf, permutation, unpermute)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_embedding_obfuscation.py -v`
Expected: PASS (2 tests). Si le premier test échoue sur la forme ou la
bijection, vérifier l'indexation `perm_index`/`inv_perm_index` avant de
toucher aux matrices clés — l'erreur la plus probable est une confusion
entre `Π` et `Πᵀ`.

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add aloepri_poc/embedding_obfuscation.py aloepri_poc/tests/test_embedding_obfuscation.py
git commit -m "feat(aloepri): obfuscation embedding/unembedding (bruit+permutation+clés)"
```

---

### Task 4: Obfuscation FFN (permutation + scaling, SwiGLU-aware)

**Files:**
- Create: `aloepri_poc/ffn_obfuscation.py`
- Test: `aloepri_poc/tests/test_ffn_obfuscation.py`

**Interfaces:**
- Produces: `obfuscate_ffn_layer(gate_proj: torch.Tensor, up_proj: torch.Tensor, down_proj: torch.Tensor, seed: int) -> ObfuscatedFFN`
  (dataclass : `gate_proj_obf, up_proj_obf, down_proj_obf`). Poids au format
  HF (`gate_proj`/`up_proj`: shape `(intermediate_size, hidden_size)`,
  `down_proj`: shape `(hidden_size, intermediate_size)`). Consommé par
  Task 8.

- [ ] **Step 1: Write the failing test**

```python
# aloepri_poc/tests/test_ffn_obfuscation.py
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from ffn_obfuscation import obfuscate_ffn_layer


def swiglu_ffn(x, gate_proj, up_proj, down_proj):
    gate = x @ gate_proj.T
    up = x @ up_proj.T
    hidden = torch.nn.functional.silu(gate) * up
    return hidden @ down_proj.T


def test_obfuscated_ffn_produces_identical_output():
    torch.manual_seed(0)
    hidden_size, intermediate_size = 16, 24
    gate_proj = torch.randn(intermediate_size, hidden_size)
    up_proj = torch.randn(intermediate_size, hidden_size)
    down_proj = torch.randn(hidden_size, intermediate_size)
    x = torch.randn(3, hidden_size)

    baseline_output = swiglu_ffn(x, gate_proj, up_proj, down_proj)

    obf = obfuscate_ffn_layer(gate_proj, up_proj, down_proj, seed=0)
    obf_output = swiglu_ffn(x, obf.gate_proj_obf, obf.up_proj_obf, obf.down_proj_obf)

    torch.testing.assert_close(obf_output, baseline_output, atol=1e-4, rtol=1e-4)


def test_obfuscated_ffn_weights_differ_from_original():
    torch.manual_seed(1)
    hidden_size, intermediate_size = 16, 24
    gate_proj = torch.randn(intermediate_size, hidden_size)
    up_proj = torch.randn(intermediate_size, hidden_size)
    down_proj = torch.randn(hidden_size, intermediate_size)

    obf = obfuscate_ffn_layer(gate_proj, up_proj, down_proj, seed=1)
    assert not torch.allclose(obf.gate_proj_obf, gate_proj)
    assert not torch.allclose(obf.up_proj_obf, up_proj)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_ffn_obfuscation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffn_obfuscation'`

- [ ] **Step 3: Write implementation**

```python
# aloepri_poc/ffn_obfuscation.py
"""
Obfuscation FFN : permutation de la dimension intermédiaire + scaling par
neurone, avec compensation inverse (papier §7.5 : "permutations and scaling
matrices"). silu(gate) * up est invariant à une permutation de la dimension
intermédiaire SI gate_proj et up_proj sont permutés identiquement en sortie,
et down_proj reçoit la même permutation en entrée. Le scaling par neurone
s_i sur gate_proj/up_proj doit être compensé par 1/s_i^2 sur down_proj (le
produit gate*up introduit s_i au carré, silu(s_i * z) != s_i * silu(z) en
général — pour rester une reparamétrisation exacte, le scaling doit
s'appliquer identiquement à gate_proj ET up_proj de sorte que
silu(s_i*gate)*s_i*up = s_i * silu(gate)*up seulement si silu est linéaire,
ce qui n'est pas le cas : on limite donc le scaling à up_proj/down_proj
uniquement, où l'invariance est exacte : silu(gate) * (s_i * up) donne un
facteur s_i, compensé par down_proj / s_i.
"""
from dataclasses import dataclass
import random

import torch


@dataclass
class ObfuscatedFFN:
    gate_proj_obf: torch.Tensor
    up_proj_obf: torch.Tensor
    down_proj_obf: torch.Tensor


def obfuscate_ffn_layer(gate_proj, up_proj, down_proj, seed):
    intermediate_size, hidden_size = gate_proj.shape
    assert up_proj.shape == (intermediate_size, hidden_size)
    assert down_proj.shape == (hidden_size, intermediate_size)

    rng_py = random.Random(seed)
    perm = list(range(intermediate_size))
    rng_py.shuffle(perm)
    perm_index = torch.tensor(perm)

    gen = torch.Generator().manual_seed(seed)
    # scaling strictement positif pour rester inversible sans changer de signe
    scale = torch.exp(torch.randn(intermediate_size, generator=gen) * 0.1)

    gate_proj_obf = gate_proj[perm_index]
    up_proj_obf = up_proj[perm_index] * scale[perm_index].unsqueeze(1)
    down_proj_obf = down_proj[:, perm_index] / scale[perm_index].unsqueeze(0)

    return ObfuscatedFFN(gate_proj_obf, up_proj_obf, down_proj_obf)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_ffn_obfuscation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add aloepri_poc/ffn_obfuscation.py aloepri_poc/tests/test_ffn_obfuscation.py
git commit -m "feat(aloepri): obfuscation FFN (permutation+scaling SwiGLU-aware)"
```

---

### Task 5: Rotation RoPE + scaling pour Q/K (composant de l'Algorithme 2)

**Files:**
- Create: `aloepri_poc/rope_transform.py`
- Test: `aloepri_poc/tests/test_rope_transform.py`

**Interfaces:**
- Produces: `sample_rope_rotation(d_head: int, seed: int) -> torch.Tensor`
  (R̂_qk, shape `(d_head, d_head)`, bloc-diagonale de rotations 2×2),
  `sample_rope_scaling(d_head: int, seed: int) -> torch.Tensor` (Ĥ_qk,
  diagonale, shape `(d_head, d_head)`). Consommé par Task 7.

- [ ] **Step 0 (obligatoire avant de coder) : relire l'Algorithme 2, lignes 1-2, page 9 du PDF**

Vérifier la formule exacte : `ρ_i` uniforme sur `(0, 2π)`, `R_i` bloc de
rotation 2×2 standard `[[cos ρ_i, -sin ρ_i], [sin ρ_i, cos ρ_i]]`, empilés
en diagonale par blocs sur `d_head/2` blocs pour former `R̂_qk`
(`Diag({R_i})`). `Ĥ_qk = Diag(s_1 I_2, ..., s_{d_head/2} I_2)` — chaque
`s_i` scalaire répété deux fois (une paire RoPE = 2 dimensions).

- [ ] **Step 1: Write the failing test**

```python
# aloepri_poc/tests/test_rope_transform.py
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from rope_transform import sample_rope_rotation, sample_rope_scaling


def test_rope_rotation_is_orthogonal():
    d_head = 8
    r_hat = sample_rope_rotation(d_head, seed=0)
    assert r_hat.shape == (d_head, d_head)
    torch.testing.assert_close(r_hat @ r_hat.T, torch.eye(d_head), atol=1e-5, rtol=1e-5)


def test_rope_rotation_is_block_diagonal_2x2():
    d_head = 8
    r_hat = sample_rope_rotation(d_head, seed=1)
    for i in range(0, d_head, 2):
        for j in range(0, d_head, 2):
            if i != j:
                block = r_hat[i : i + 2, j : j + 2]
                assert torch.allclose(block, torch.zeros(2, 2), atol=1e-6)


def test_rope_scaling_is_diagonal_and_positive():
    d_head = 8
    h_hat = sample_rope_scaling(d_head, seed=2)
    assert h_hat.shape == (d_head, d_head)
    off_diag = h_hat - torch.diag(torch.diagonal(h_hat))
    assert torch.allclose(off_diag, torch.zeros(d_head, d_head))
    assert torch.all(torch.diagonal(h_hat) > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_rope_transform.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rope_transform'`

- [ ] **Step 3: Write implementation**

```python
# aloepri_poc/rope_transform.py
"""Rotation RoPE 2D par paire + scaling (Algorithme 2, lignes 1-2)."""
import torch


def sample_rope_rotation(d_head, seed):
    assert d_head % 2 == 0
    gen = torch.Generator().manual_seed(seed)
    n_pairs = d_head // 2
    rho = torch.rand(n_pairs, generator=gen) * 2 * torch.pi

    r_hat = torch.zeros(d_head, d_head)
    for i, angle in enumerate(rho):
        c, s = torch.cos(angle), torch.sin(angle)
        r_hat[2 * i, 2 * i] = c
        r_hat[2 * i, 2 * i + 1] = -s
        r_hat[2 * i + 1, 2 * i] = s
        r_hat[2 * i + 1, 2 * i + 1] = c
    return r_hat


def sample_rope_scaling(d_head, seed):
    assert d_head % 2 == 0
    gen = torch.Generator().manual_seed(seed)
    n_pairs = d_head // 2
    # strictement positif pour rester inversible
    s = torch.exp(torch.randn(n_pairs, generator=gen) * 0.1)
    diag = s.repeat_interleave(2)
    return torch.diag(diag)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_rope_transform.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add aloepri_poc/rope_transform.py aloepri_poc/tests/test_rope_transform.py
git commit -m "feat(aloepri): rotation RoPE + scaling pour Q/K (algo 2, lignes 1-2)"
```

---

### Task 6: BlockPerm — permutation par blocs à fenêtre dynamique

**Files:**
- Create: `aloepri_poc/block_perm.py`
- Test: `aloepri_poc/tests/test_block_perm.py`

**Interfaces:**
- Produces: `block_perm(beta: int, gamma: float, zeta: float, m_blocks: int, seed: int) -> torch.Tensor`
  (Ẑ_block, matrice de permutation bloc-diagonale, shape `(m_blocks, m_blocks)`).
  Consommé par Task 7 (appliqué aux paires RoPE, donc `m_blocks = d_head/2`).

- [ ] **Step 0 (obligatoire avant de coder) : relire la fonction BlockPerm, Algorithme 2 lignes 9-19, page 9**

Vérifier la formule exacte du poids d'échantillonnage
`ζ_i = ζ^{-2(i-1)/m_blocks}` et de la boucle qui construit des fenêtres de
taille variable (`c = min(β, m_blocks - t)`, `u = softmax({ζ_{t+i} - ζ_t | 1≤i≤c})`,
tire une taille de fenêtre `w` selon `u`, puis une permutation aléatoire de
`S_w` pour cette fenêtre, jusqu'à couvrir les `m_blocks` blocs). Le résultat
final est une matrice bloc-diagonale (`BlockDiag`) de permutations de tailles
variables couvrant l'ensemble des blocs.

- [ ] **Step 1: Write the failing test (propriétés structurelles, pas les valeurs exactes)**

```python
# aloepri_poc/tests/test_block_perm.py
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from block_perm import block_perm


def test_block_perm_is_a_valid_permutation_matrix():
    m_blocks = 12
    z = block_perm(beta=8, gamma=0.5, zeta=1e3, m_blocks=m_blocks, seed=0)
    assert z.shape == (m_blocks, m_blocks)
    # chaque ligne et chaque colonne a exactement un 1, le reste des 0
    row_sums = z.sum(dim=1)
    col_sums = z.sum(dim=0)
    torch.testing.assert_close(row_sums, torch.ones(m_blocks))
    torch.testing.assert_close(col_sums, torch.ones(m_blocks))
    assert set(z.unique().tolist()) <= {0.0, 1.0}


def test_block_perm_respects_max_window_size():
    # Avec une fenêtre max de 1, chaque bloc doit rester à sa place
    # (aucune permutation possible au-delà d'un singleton).
    m_blocks = 10
    z = block_perm(beta=1, gamma=0.5, zeta=1e3, m_blocks=m_blocks, seed=1)
    torch.testing.assert_close(z, torch.eye(m_blocks))


def test_block_perm_is_reproducible_with_same_seed():
    z1 = block_perm(beta=8, gamma=0.5, zeta=1e3, m_blocks=12, seed=42)
    z2 = block_perm(beta=8, gamma=0.5, zeta=1e3, m_blocks=12, seed=42)
    torch.testing.assert_close(z1, z2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_block_perm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'block_perm'`

- [ ] **Step 3: Implement, en suivant l'algorithme relu à l'étape 0**

Implémenter `aloepri_poc/block_perm.py` avec la fonction `block_perm` en
transcrivant fidèlement la boucle `while t < m_blocks` de l'Algorithme 2
(lignes 9-19). Retourner une matrice `torch.Tensor` bloc-diagonale construite
en plaçant, pour chaque fenêtre tirée, une sous-matrice de permutation
aléatoire (`torch.eye(w)[torch.randperm(w)]`) à la bonne position diagonale.
Le test `test_block_perm_respects_max_window_size` (β=1) est le plus utile
pour valider une implémentation correcte de la boucle sans ambiguïté sur les
poids d'échantillonnage — s'assurer qu'il passe avant de se soucier de la
distribution exacte des tailles de fenêtre pour β>1.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_block_perm.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add aloepri_poc/block_perm.py aloepri_poc/tests/test_block_perm.py
git commit -m "feat(aloepri): BlockPerm — permutation par blocs à fenêtre dynamique"
```

---

### Task 7: Obfuscation d'attention complète (intra-tête + inter-tête, GQA-aware)

**Files:**
- Create: `aloepri_poc/attention_obfuscation.py`
- Test: `aloepri_poc/tests/test_attention_obfuscation.py`

**Interfaces:**
- Consumes: `key_matrix.init_key_matrix/key_mat_gen` (Task 2),
  `rope_transform.sample_rope_rotation/sample_rope_scaling` (Task 5),
  `block_perm.block_perm` (Task 6).
- Produces: `obfuscate_attention_layer(w_q, w_k, w_v, w_o, num_heads: int, num_kv_heads: int, d_head: int, beta: int, gamma: float, zeta: float, seed: int) -> ObfuscatedAttention`
  (dataclass : `w_q_obf, w_k_obf, w_v_obf, w_o_obf`). Poids au format HF GQA :
  `w_q: (num_heads*d_head, hidden_size)`, `w_k/w_v: (num_kv_heads*d_head, hidden_size)`,
  `w_o: (hidden_size, num_heads*d_head)`. Consommé par Task 8.

- [ ] **Step 1: Write the failing test**

```python
# aloepri_poc/tests/test_attention_obfuscation.py
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from attention_obfuscation import obfuscate_attention_layer


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


def test_obfuscated_attention_preserves_output():
    torch.manual_seed(0)
    hidden_size, num_heads, num_kv_heads, d_head = 32, 8, 2, 8
    seq_len = 5

    w_q = torch.randn(num_heads * d_head, hidden_size) * 0.1
    w_k = torch.randn(num_kv_heads * d_head, hidden_size) * 0.1
    w_v = torch.randn(num_kv_heads * d_head, hidden_size) * 0.1
    w_o = torch.randn(hidden_size, num_heads * d_head) * 0.1
    x = torch.randn(seq_len, hidden_size)

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
```

**Note :** ce test découvrira probablement des ajustements nécessaires sur
la façon dont `R̂_qk`/`Ĥ_qk`/`Ẑ_block` (par tête) et les matrices clés/`Û_vo`
(par groupe, Task 5/6/2) se composent — c'est le but : la boucle
implémentation→test est ce qui valide la composition correcte des 3 tâches
précédentes, pas une lecture supplémentaire du papier.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_attention_obfuscation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'attention_obfuscation'`

- [ ] **Step 3: Write implementation**

```python
# aloepri_poc/attention_obfuscation.py
"""
Obfuscation d'attention complète (Algorithme 2 + permutation inter-tête,
papier §5.2.3). Applique, par tête (Q/K, RoPE-sensibles) et par groupe
(matrices clés/Û_vo), les transformations qui laissent le calcul
d'attention invariant.
"""
from dataclasses import dataclass
import random

import numpy as np
import torch

from key_matrix import init_key_matrix, key_mat_gen, inv_key_mat_gen
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

    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    m_blocks = d_head // 2

    q_heads = w_q.view(num_heads, d_head, hidden_size)
    o_heads = w_o.view(hidden_size, num_heads, d_head)
    k_heads = w_k.view(num_kv_heads, d_head, hidden_size)
    v_heads = w_v.view(num_kv_heads, d_head, hidden_size)

    q_obf = torch.zeros_like(q_heads)
    o_obf = torch.zeros_like(o_heads)
    for h in range(num_heads):
        r_hat = sample_rope_rotation(d_head, seed=seed * 1000 + h)
        h_hat = sample_rope_scaling(d_head, seed=seed * 1000 + h + 1)
        z_block = block_perm(beta, gamma, zeta, m_blocks, seed=seed * 1000 + h + 2)
        z_block_full = torch.kron(z_block, torch.eye(2))  # étend au niveau des dimensions (pas des paires)

        base_q = init_key_matrix(d_head, 16, 0.3, rng_np)
        q_hat_q = torch.tensor(key_mat_gen(base_q), dtype=w_q.dtype)
        base_o = init_key_matrix(d_head, 16, 0.3, rng_np)
        p_hat_o = torch.tensor(key_mat_gen(base_o), dtype=w_o.dtype)

        # W̃_q = Q̂_q · W_q · R̂_qk · Ĥ_qk · Ẑ_block   (transforme les d_head lignes de sortie)
        q_obf[h] = q_hat_q @ q_heads[h] @ r_hat @ h_hat @ z_block_full
        # W̃_o = Û_vo⁻¹ · W_o · P̂_o   (Û_vo appliqué côté V, voir ci-dessous, groupé)
        o_obf[:, h, :] = o_heads[:, h, :] @ p_hat_o

    k_obf = torch.zeros_like(k_heads)
    v_obf = torch.zeros_like(v_heads)
    for kv in range(num_kv_heads):
        r_hat = sample_rope_rotation(d_head, seed=seed * 1000 + 100 + kv)
        h_hat = sample_rope_scaling(d_head, seed=seed * 1000 + 100 + kv + 1)
        z_block = block_perm(beta, gamma, zeta, m_blocks, seed=seed * 1000 + 100 + kv + 2)
        z_block_full = torch.kron(z_block, torch.eye(2))

        base_k = init_key_matrix(d_head, 16, 0.3, rng_np)
        q_hat_k = torch.tensor(key_mat_gen(base_k), dtype=w_k.dtype)

        u_vo = torch.tensor(
            rng_np.standard_normal((d_head, d_head)) / np.sqrt(d_head), dtype=w_v.dtype
        )
        u_vo_inv = torch.linalg.inv(u_vo)

        # W̃_k = Q̂_k · W_k · R̂_qk · Ĥ_qk⁻¹ · Ẑ_blockᵀ
        h_hat_inv = torch.diag(1.0 / torch.diagonal(h_hat))
        k_obf[kv] = q_hat_k @ k_heads[kv] @ r_hat @ h_hat_inv @ z_block_full.T
        # W̃_v = Q̂_v · W_v · Û_vo   (Q̂_v choisi = identité ici : la protection
        # de V vient de Û_vo, cohérent avec la note du papier "Û_vo... to
        # preserve computation correctness" — Q̂_v/P̂_o ne sont utiles qu'en
        # cas d'obfuscation ADDITIONNELLE de V, hors scope du round-trip)
        v_obf[kv] = u_vo @ v_heads[kv]

        # compenser Û_vo côté O pour le groupe correspondant : chaque tête Q
        # du groupe kv doit voir Û_vo⁻¹ appliqué à l'entrée O correspondante.
        group_size = num_heads // num_kv_heads
        for h in range(kv * group_size, (kv + 1) * group_size):
            o_obf[:, h, :] = o_obf[:, h, :] @ u_vo_inv

    w_q_obf = q_obf.reshape(num_heads * d_head, hidden_size)
    w_k_obf = k_obf.reshape(num_kv_heads * d_head, hidden_size)
    w_v_obf = v_obf.reshape(num_kv_heads * d_head, hidden_size)
    w_o_obf = o_obf.reshape(hidden_size, num_heads * d_head)

    return ObfuscatedAttention(w_q_obf, w_k_obf, w_v_obf, w_o_obf)
```

**Avertissement pour l'implémenteur** : ce code est un point de départ
raisonné (composition Q̂_q/Q̂_k pour annuler dans le produit scalaire Q·K,
Û_vo/Û_vo⁻¹ pour annuler autour de V/O), mais la composition exacte
matrices-clés/rotation-RoPE/bloc-perm sur 3 tâches assemblées ici est le
point le plus délicat de tout le POC. Si `test_obfuscated_attention_preserves_output`
échoue, déboguer terme par terme (désactiver `Ẑ_block` puis `R̂_qk/Ĥ_qk`
puis les matrices clés un par un, en mettant chacun à l'identité) pour
isoler quelle transformation casse l'invariance, plutôt que de tout
réajuster en même temps.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_attention_obfuscation.py -v`
Expected: PASS. Itérer sur l'implémentation (pas sur le test) jusqu'à
convergence — le test encode l'invariant recherché, pas une supposition.

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add aloepri_poc/attention_obfuscation.py aloepri_poc/tests/test_attention_obfuscation.py
git commit -m "feat(aloepri): obfuscation d'attention complète (algo 2 + permutation inter-tête)"
```

---

### Task 8: Orchestration du modèle complet + wrapper client

**Files:**
- Create: `aloepri_poc/model_transform.py`
- Create: `aloepri_poc/client_wrapper.py`
- Test: `aloepri_poc/tests/test_client_wrapper.py`

**Interfaces:**
- Consumes: `embedding_obfuscation.obfuscate_embedding` (Task 3),
  `ffn_obfuscation.obfuscate_ffn_layer` (Task 4),
  `attention_obfuscation.obfuscate_attention_layer` (Task 7).
- Produces: `model_transform.transform_model(model_name: str, output_dir: str, seed: int) -> ObfuscationKeys`
  (sauvegarde les poids obfusqués dans `output_dir`, retourne les clés
  côté client à sérialiser séparément) ;
  `client_wrapper.ClientCodec(keys: ObfuscationKeys, tokenizer)` avec
  méthodes `.encode(text: str) -> list[int]` (tokenize + permute) et
  `.decode(permuted_ids: list[int]) -> str` (dépermute + detokenize).

- [ ] **Step 1: Write the failing test (wrapper client, testable sans le vrai modèle 7B)**

```python
# aloepri_poc/tests/test_client_wrapper.py
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from client_wrapper import ClientCodec


class FakeTokenizer:
    """Tokenizer jouet : un caractère = un token, pour tester sans réseau."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_codec_round_trip_without_obfuscation_keys():
    permutation = {i: (i + 1) % 256 for i in range(256)}
    unpermute = {v: k for k, v in permutation.items()}
    codec = ClientCodec(permutation, unpermute, FakeTokenizer())

    original = "bonjour"
    encoded = codec.encode(original)
    assert encoded != [ord(c) for c in original]  # bien permuté, pas identité
    decoded = codec.decode(encoded)
    assert decoded == original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_client_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'client_wrapper'`

- [ ] **Step 3: Write implementation**

```python
# aloepri_poc/client_wrapper.py
"""Wrapper client : tokenize+permute à l'envoi, dépermute+detokenize à la réception."""


class ClientCodec:
    def __init__(self, permutation, unpermute, tokenizer):
        self.permutation = permutation
        self.unpermute = unpermute
        self.tokenizer = tokenizer

    def encode(self, text):
        clear_ids = self.tokenizer.encode(text)
        return [self.permutation[i] for i in clear_ids]

    def decode(self, permuted_ids):
        clear_ids = [self.unpermute[i] for i in permuted_ids]
        return self.tokenizer.decode(clear_ids)
```

```python
# aloepri_poc/model_transform.py
"""
Orchestration : charge Qwen2.5-7B-Instruct, applique embedding+FFN+attention
obfuscation, sauvegarde le modèle obfusqué (safetensors) et les clés côté
client séparément.

Usage: python model_transform.py --model Qwen/Qwen2.5-7B-Instruct --output ./obfuscated_model --seed 0
Nécessite un GPU (ou beaucoup de RAM CPU) pour charger un modèle 7B.
"""
import argparse
import json
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from embedding_obfuscation import obfuscate_embedding
from ffn_obfuscation import obfuscate_ffn_layer
from attention_obfuscation import obfuscate_attention_layer


@dataclass
class ObfuscationKeys:
    vocab_permutation: dict
    vocab_unpermute: dict
    seed: int


def transform_model(model_name, output_dir, seed, alpha_e=1.0, alpha_h=0.2, lam=0.3, h=128,
                     beta=8, gamma=1e3, zeta=1e3):
    config = AutoConfig.from_pretrained(model_name)

    # Vérification obligatoire (risque identifié dans la spec) : GQA, pas MLA.
    assert hasattr(config, "num_key_value_heads"), (
        "config.json ne définit pas num_key_value_heads — vérifier si le "
        "modèle utilise MLA (traitement différent, hors scope) plutôt que GQA."
    )
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    d_head = config.hidden_size // num_heads

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    # Weight tying : à vérifier avant de traiter embed_tokens/lm_head séparément.
    tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()

    w_embed = model.get_input_embeddings().weight.data.clone()
    w_head = model.get_output_embeddings().weight.data.clone()
    emb_result = obfuscate_embedding(w_embed, w_head, alpha_e, alpha_h, lam, h, seed)
    model.get_input_embeddings().weight.data.copy_(emb_result.w_embed_obf)
    if not tied:
        model.get_output_embeddings().weight.data.copy_(emb_result.w_head_obf)

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        obf_attn = obfuscate_attention_layer(
            attn.q_proj.weight.data, attn.k_proj.weight.data,
            attn.v_proj.weight.data, attn.o_proj.weight.data,
            num_heads=num_heads, num_kv_heads=num_kv_heads, d_head=d_head,
            beta=beta, gamma=gamma, zeta=zeta, seed=seed * 10000 + i,
        )
        attn.q_proj.weight.data.copy_(obf_attn.w_q_obf)
        attn.k_proj.weight.data.copy_(obf_attn.w_k_obf)
        attn.v_proj.weight.data.copy_(obf_attn.w_v_obf)
        attn.o_proj.weight.data.copy_(obf_attn.w_o_obf)

        mlp = layer.mlp
        obf_ffn = obfuscate_ffn_layer(
            mlp.gate_proj.weight.data, mlp.up_proj.weight.data, mlp.down_proj.weight.data,
            seed=seed * 10000 + i + 1,
        )
        mlp.gate_proj.weight.data.copy_(obf_ffn.gate_proj_obf)
        mlp.up_proj.weight.data.copy_(obf_ffn.up_proj_obf)
        mlp.down_proj.weight.data.copy_(obf_ffn.down_proj_obf)

    model.save_pretrained(output_dir)

    keys = ObfuscationKeys(emb_result.permutation, emb_result.unpermute, seed)
    with open(f"{output_dir}/../obfuscation_keys.json", "w") as f:
        json.dump(asdict(keys), f)
    return keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    transform_model(args.model, args.output, args.seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Secretarius && python -m pytest aloepri_poc/tests/test_client_wrapper.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add aloepri_poc/model_transform.py aloepri_poc/client_wrapper.py aloepri_poc/tests/test_client_wrapper.py
git commit -m "feat(aloepri): orchestration modèle complet + wrapper client"
```

---

### Task 9: Déploiement RunPod, round-trip réel, mesure qualité/vitesse

Cette tâche est opérationnelle (location de GPU, exécution sur un modèle
7B réel) — moins « TDD » que les précédentes, mais chaque étape reste
concrète et vérifiable.

**Files:**
- Create: `aloepri_poc/server.py`
- Create: `aloepri_poc/measure_quality.py`
- Create: `aloepri_poc/measure_speed.py`

- [ ] **Step 1: Écrire le serveur d'inférence**

```python
# aloepri_poc/server.py
"""Serveur HTTP minimal servant le modèle obfusqué (ou baseline) via transformers."""
import argparse

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
_model = None
_tokenizer = None


class GenerateRequest(BaseModel):
    input_ids: list[int]
    max_new_tokens: int = 100


@app.post("/generate")
def generate(req: GenerateRequest):
    input_tensor = torch.tensor([req.input_ids])
    output = _model.generate(input_tensor, max_new_tokens=req.max_new_tokens, do_sample=False)
    return {"output_ids": output[0].tolist()}


def load(model_dir):
    global _model, _tokenizer
    _model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda()
    _tokenizer = AutoTokenizer.from_pretrained(model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    load(args.model_dir)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
```

- [ ] **Step 2: Louer le Pod RunPod et transférer le code**

```bash
# Sur RunPod : créer un Pod GPU RTX A5000, Community Cloud, image PyTorch standard.
# Depuis sanroque, copier le code du POC sur le Pod (remplacer <POD_IP>) :
rsync -avz ~/Secretarius/aloepri_poc/ root@<POD_IP>:/workspace/aloepri_poc/
ssh root@<POD_IP> "pip install -r /workspace/aloepri_poc/requirements.txt"
```

- [ ] **Step 3: Transformer le modèle et vérifier les hypothèses (GQA, weight tying)**

```bash
ssh root@<POD_IP>
cd /workspace/aloepri_poc
python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); print(c.num_attention_heads, c.num_key_value_heads, hasattr(c, 'num_key_value_heads'))"
# Attendu : confirme GQA (num_key_value_heads défini et < num_attention_heads).
# Si absent ou égal à num_attention_heads (MHA) ou architecture MLA détectée,
# arrêter et réévaluer Task 7 avant de continuer.

python model_transform.py --model Qwen/Qwen2.5-7B-Instruct --output ./obfuscated_model --seed 0
```

- [ ] **Step 4: Round-trip de bout en bout (critère de succès n°1 de la spec)**

```bash
python -c "
from client_wrapper import ClientCodec
from model_transform import ObfuscationKeys
from transformers import AutoTokenizer
import json, requests

with open('obfuscation_keys.json') as f:
    keys = json.load(f)

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
codec = ClientCodec(
    {int(k): v for k, v in keys['vocab_permutation'].items()},
    {int(k): v for k, v in keys['vocab_unpermute'].items()},
    tokenizer,
)

prompt = 'Quelle est la capitale de la France ?'
permuted_ids = codec.encode(prompt)
resp = requests.post('http://localhost:8000/generate', json={'input_ids': permuted_ids, 'max_new_tokens': 50})
output_permuted = resp.json()['output_ids']
print(codec.decode(output_permuted))
"
# Lancer le serveur au préalable : python server.py --model-dir ./obfuscated_model &
```

Vérifier manuellement que la sortie est un texte français cohérent
(critère de succès n°1 de la spec — round-trip correct).

- [ ] **Step 5: Mesure qualité**

```python
# aloepri_poc/measure_quality.py
"""
Compare la perplexité du modèle obfusqué vs baseline sur un jeu de prompts
de test fixe.

Usage: python measure_quality.py --baseline Qwen/Qwen2.5-7B-Instruct --obfuscated ./obfuscated_model
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_PROMPTS = [
    "La capitale de la France est",
    "Le théorème de Pythagore énonce que",
    "En 1789, la Révolution française",
    # compléter avec ~20-30 prompts représentatifs du cas d'usage visé
    # (questions factuelles courtes, cohérent avec le scénario "question
    # posée à l'IA" qui motive le POC)
]


def perplexity(model, tokenizer, text):
    ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return torch.exp(loss).item()


def main(baseline_path, obfuscated_path):
    tokenizer = AutoTokenizer.from_pretrained(baseline_path)
    baseline = AutoModelForCausalLM.from_pretrained(baseline_path, torch_dtype=torch.bfloat16).cuda()
    obfuscated = AutoModelForCausalLM.from_pretrained(obfuscated_path, torch_dtype=torch.bfloat16).cuda()

    baseline_ppl = [perplexity(baseline, tokenizer, p) for p in TEST_PROMPTS]
    obfuscated_ppl = [perplexity(obfuscated, tokenizer, p) for p in TEST_PROMPTS]

    for prompt, b, o in zip(TEST_PROMPTS, baseline_ppl, obfuscated_ppl):
        delta_pct = (o - b) / b * 100
        print(f"{prompt[:40]:40s} baseline={b:.2f} obfusqué={o:.2f} delta={delta_pct:+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--obfuscated", required=True)
    args = parser.parse_args()
    main(args.baseline, args.obfuscated)
```

- [ ] **Step 6: Mesure vitesse**

```python
# aloepri_poc/measure_speed.py
"""
Compare tokens/s et latence entre modèle obfusqué et baseline.

Usage: python measure_speed.py --baseline Qwen/Qwen2.5-7B-Instruct --obfuscated ./obfuscated_model
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure(model, tokenizer, prompt, max_new_tokens=100):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    output = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - start
    n_generated = output.shape[1] - ids.shape[1]
    return n_generated / elapsed, elapsed


def main(baseline_path, obfuscated_path):
    tokenizer = AutoTokenizer.from_pretrained(baseline_path)
    baseline = AutoModelForCausalLM.from_pretrained(baseline_path, torch_dtype=torch.bfloat16).cuda()
    obfuscated = AutoModelForCausalLM.from_pretrained(obfuscated_path, torch_dtype=torch.bfloat16).cuda()

    prompt = "Décris en trois phrases le fonctionnement d'un transformer."
    b_tps, b_lat = measure(baseline, tokenizer, prompt)
    o_tps, o_lat = measure(obfuscated, tokenizer, prompt)

    print(f"baseline : {b_tps:.1f} tok/s, {b_lat:.2f}s")
    print(f"obfusqué : {o_tps:.1f} tok/s, {o_lat:.2f}s")
    print(f"overhead vitesse : {(b_tps - o_tps) / b_tps * 100:+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--obfuscated", required=True)
    args = parser.parse_args()
    main(args.baseline, args.obfuscated)
```

- [ ] **Step 7: Lancer les deux mesures sur le Pod et consigner les résultats**

```bash
python measure_quality.py --baseline Qwen/Qwen2.5-7B-Instruct --obfuscated ./obfuscated_model
python measure_speed.py --baseline Qwen/Qwen2.5-7B-Instruct --obfuscated ./obfuscated_model
```

Consigner les deux sorties dans un `aloepri_poc/RESULTATS.md` (nouveau
fichier, format libre : tableau delta qualité par prompt, overhead vitesse,
coût Pod effectif).

- [ ] **Step 8: Arrêter le Pod et commit final**

```bash
# Depuis RunPod : arrêter/terminer le Pod pour ne plus être facturé.
cd ~/Secretarius
git add aloepri_poc/server.py aloepri_poc/measure_quality.py aloepri_poc/measure_speed.py aloepri_poc/RESULTATS.md
git commit -m "feat(aloepri): serveur RunPod + mesures qualité/vitesse du pipeline complet"
```

---

## Self-Review Notes

- **Couverture de la spec** : Étape 0 (Task 1), Algorithme 1 (Task 2),
  embedding/unembedding (Task 3), FFN (Task 4), RoPE+scaling (Task 5),
  BlockPerm (Task 6), attention complète (Task 7), orchestration+client
  (Task 8), déploiement+mesures (Task 9). Tous les points de la spec du
  2026-08-17 sont couverts.
- **Risque assumé le plus élevé** : Task 7 (composition intra-tête/inter-tête)
  est le point du plan le moins garanti a priori — le code fourni est un
  point de départ raisonné, pas une certitude, et le plan le dit
  explicitement plutôt que de prétendre le contraire.
- **Hors scope, comme dans la spec** : rotation de clés en cours d'usage,
  implémentation des attaques (ISA/Attn-IA/Gate-IA/TFMA au-delà d'Étape 0),
  vLLM/SGLang, intégration Secretarius/Tiron.
