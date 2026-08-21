"""Attaques ISA (Internal State Attack) par descente de gradient.

Modèle de menace (AloePri arXiv 2603.01499, Appendix D.1 ; méthode
d'inversion du papier arXiv 2507.16372) : l'opérateur du serveur capture
l'état interne d'UNE requête réelle X1 (état caché d'une couche, ou scores
d'attention), puis optimise un candidat X2 par descente de gradient jusqu'à
ce que son état interne s'approche de l'état capturé — récupérant ainsi
l'entrée.

Implémentation (variante « soft tokens » du papier 2507.16372) :
- l'attaquant paramètre le candidat par des LOGITS PAR POSITION
  P ∈ R^{T×vocab} ; l'entrée du modèle est embeds = softmax(P/τ) @ W_embed
  (différentiable : le gradient remonte jusqu'à P à travers la table
  d'embedding) ;
- Adam + recuit de température τ (3 → 0,1) pour éviter les minima locaux ;
- à la fin, ids_récupérés = argmax(P) ;
- deux canaux : « hidden » (état caché d'une couche, MSE) et « attn »
  (pondérations d'attention d'une couche, MSE).

Métrique : taux de correspondance des IDs récupérés vs les IDs réellement
envoyés au modèle. Sur le modèle obfusqué, ces IDs sont les IDs PERMUTÉS —
les retrouver ne donne pas le texte (il faudrait la clé de permutation,
côté client). Le taux mesure donc la récupération de l'entrée « modèle »,
et la comparaison baseline vs obfusqué mesure l'effet de l'obfuscation.

Usage (module) :
    from isa_attack import attack_model_channel, token_match_rate
"""
import argparse
import random

import torch


def capture_state(model, embeds, layer, channel="hidden"):
    """État interne (cible) du modèle pour `embeds` (T, d).

    channel="hidden" : état caché de sortie de la couche `layer`
    (bsz=1 → (T, hidden)).
    channel="attn" : pondérations d'attention de la couche `layer`
    (bsz=1, heads, T, T).
    """
    kw = {}
    if channel == "hidden":
        kw["output_hidden_states"] = True
    else:
        kw["output_attentions"] = True
    with torch.no_grad():
        out = model(inputs_embeds=embeds.unsqueeze(0), **kw)
    if channel == "hidden":
        state = out.hidden_states[layer][0]          # (T, hidden)
    else:
        state = out.attentions[layer][0]             # (heads, T, T)
    return state.detach()


def _soft_embeds(logits, embed_table, tau):
    """embeds (T, d) = softmax(P/τ) @ W — différentiable en P."""
    probs = torch.softmax(logits / tau, dim=-1)
    return probs @ embed_table


def attack_model(model, embed_table, target_state, seq_len, channel,
                 layer, steps=500, lr=0.05, tau_start=3.0, tau_end=0.1,
                 tau_steps=400, seed=0, device="cuda"):
    """Descente de gradient sur des logits par position pour inverser l'état
    capturé `target_state`. Retourne (ids_récupérés (seq_len,), pertes)."""
    torch.manual_seed(seed)
    random.seed(seed)

    vocab = embed_table.shape[0]
    logits = torch.zeros(seq_len, vocab, device=device, requires_grad=True)
    # init léger hors de l'uniformité parfaite (évite le gradient nul)
    with torch.no_grad():
        logits += 0.05 * torch.randn_like(logits)

    opt = torch.optim.Adam([logits], lr=lr)
    losses = []
    for step in range(steps):
        tau = max(tau_end, tau_start * (tau_end / tau_start)
                  ** (step / max(1, tau_steps)))
        embeds = _soft_embeds(logits, embed_table, tau).to(model.dtype)
        kw = {}
        if channel == "hidden":
            kw["output_hidden_states"] = True
        else:
            kw["output_attentions"] = True
        out = model(inputs_embeds=embeds.unsqueeze(0), **kw)
        if channel == "hidden":
            state = out.hidden_states[layer][0]
        else:
            state = out.attentions[layer][0]
        loss = torch.nn.functional.mse_loss(state, target_state)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        ids = logits.argmax(dim=-1)
    return ids, losses


def token_match_rate(pred_ids, true_ids):
    """Proportion d'IDs exactement récupérés (positions alignées)."""
    assert pred_ids.shape == true_ids.shape
    return (pred_ids == true_ids).float().mean().item()


def run_channel_attack(model, token_ids, channel, layer, embed_table=None,
                       steps=500, lr=0.05, seed=0, device="cuda"):
    """Attaque complète sur un modèle : capture de l'état de `token_ids`
    puis inversion. `token_ids` : IDs DÉJÀ dans l'espace du modèle (permutés
    pour l'obfusqué). Retourne (ids_récupérés, taux, pertes)."""
    if embed_table is None:
        embed_table = model.get_input_embeddings().weight
    true_ids = torch.tensor(token_ids, device=device)
    with torch.no_grad():
        true_embeds = embed_table[true_ids]
    target = capture_state(model, true_embeds, layer, channel)
    pred, losses = attack_model(
        model, embed_table, target, len(token_ids), channel, layer,
        steps=steps, lr=lr, seed=seed, device=device,
    )
    return pred, token_match_rate(pred, true_ids), losses


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="répertoire modèle (HF)")
    ap.add_argument("--ids", required=True,
                    help="IDs (dans l'espace du modèle), séparés par des virgules")
    ap.add_argument("--channel", choices=["hidden", "attn"], default="hidden")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="eager").to(device).eval()
    ids = [int(x) for x in args.ids.split(",")]
    pred, rate, _ = run_channel_attack(
        model, ids, args.channel, args.layer,
        steps=args.steps, lr=args.lr, seed=args.seed, device=device,
    )
    print(f"ids réels   : {ids}")
    print(f"ids attaqués: {pred.tolist()}")
    print(f"taux de correspondance : {rate:.1%}")
