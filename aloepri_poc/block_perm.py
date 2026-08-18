"""Permutation par blocs à fenêtre dynamique (Algorithme 2, lignes 9-19).

Transcription de la fonction BlockPerm du papier AloePri (arXiv 2603.01499v2,
page 9) :

     9: function BlockPerm(β, γ, ζ, m_blocks)
    10:     Initialize a list U = {} and set block index t = 1.
    11:     Compute {ζ_i = ζ^{-2(i-1)/m_blocks} | 1 ≤ i ≤ m_blocks}.
    12:     while t < m_blocks do
    13:         Set c = min(β, m_blocks - t).
    14:         Compute: u = softmax({ζ_{t+i} - ζ_t | 1 ≤ i ≤ c}).
    15:         Sample window size w from [1, c] with probabilities u.
    16:         Uniformly sample a permutation matrix Z ∈ S_w and append it to U.
    17:     end while
    18:     Concatenate the matrices into a block diagonal form: Ẑ_block = BlockDiag(U).
    19:     Return Ẑ_block.

Les ζ_i sont les fréquences RoPE : décroissantes en i, très espacées aux
petits indices et quasi confondues aux grands. Comme ζ_{t+i} - ζ_t est
négatif et d'autant plus grand en valeur absolue que i est grand, le softmax
concentre la masse sur w=1 dans la zone haute fréquence (petits indices) et
devient quasi uniforme dans la zone basse fréquence (grands indices) : les
blocs de grand indice sont donc mélangés dans des fenêtres plus larges, ce
qui est exactement l'intention annoncée dans le texte du papier.

Deux points sont laissés implicites par le pseudo-code publié :
- la ligne d'avancement `t = t + w` est absente (la boucle bouclerait
  indéfiniment) ; elle est rétablie ici ;
- avec t partant de 1 et c = min(β, m_blocks - t), les fenêtres tirées
  couvrent les blocs 1..m_blocks-1 : le dernier bloc reste seul, complété
  ici par une fenêtre singleton pour que BlockDiag soit bien
  (m_blocks, m_blocks).

`gamma` est déclaré dans la signature de l'Algorithme 2 (« window sampling
parameter ») mais n'apparaît nulle part dans le corps de la fonction publiée :
il est accepté pour rester conforme à l'interface, sans effet ici.
"""
import torch


def block_perm(beta, gamma, zeta, m_blocks, seed):
    gen = torch.Generator().manual_seed(seed)

    # ligne 11 : ζ_i pour 1 ≤ i ≤ m_blocks (indexation 1-based → zeta_i[i - 1])
    i = torch.arange(1, m_blocks + 1, dtype=torch.float32)
    zeta_i = torch.pow(torch.tensor(float(zeta)), -2.0 * (i - 1) / m_blocks)

    windows = []
    t = 1  # ligne 10
    while t < m_blocks:  # ligne 12
        c = min(beta, m_blocks - t)  # ligne 13
        # ligne 14 : {ζ_{t+i} - ζ_t | 1 ≤ i ≤ c}
        u = torch.softmax(zeta_i[t:t + c] - zeta_i[t - 1], dim=0)
        # ligne 15 : w ∈ [1, c] tiré selon u
        w = int(torch.multinomial(u, 1, generator=gen).item()) + 1
        # ligne 16 : permutation uniforme de S_w
        windows.append(torch.eye(w)[torch.randperm(w, generator=gen)])
        t += w
    if t == m_blocks:
        windows.append(torch.eye(1))  # dernier bloc, laissé seul

    return torch.block_diag(*windows)  # ligne 18
