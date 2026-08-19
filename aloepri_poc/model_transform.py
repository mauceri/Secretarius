"""
Orchestration : charge Qwen2.5-7B-Instruct, applique embedding+FFN+attention
obfuscation, sauvegarde le modèle obfusqué (safetensors) et les clés côté
client séparément.

Usage:
    python model_transform.py --model Qwen/Qwen2.5-7B-Instruct \
        --output ./obfuscated_model --seed 0

Tourne sur CPU (aucun `.cuda()`) mais demande de la RAM : ~15 Go pour le modèle
bfloat16, plus une pointe de ~15 Go pendant l'étape embedding, qui manipule
plusieurs copies float32 de la table (152k × 3584).

Choix de dtype
--------------
Le modèle est chargé, sauvegardé et servi en **bfloat16** (le dtype natif du
checkpoint Qwen2.5 et celui qu'utilise `server.py`), mais toute l'arithmétique
d'obfuscation se fait en **float32** : chaque tenseur est remonté en float32,
transformé, puis redescendu en bfloat16 à l'écriture. On évite ainsi à la fois
la perte de précision d'un produit matriciel en bfloat16 et le doublement de
l'empreinte mémoire d'un checkpoint float32.

Ce choix n'est tenable que parce que tous les facteurs appliqués sont bien
conditionnés : Û_vo est orthogonale (cf. `attention_obfuscation.py`), R̂ est une
rotation, Ẑ et les permutations de vocabulaire/neurones sont des permutations,
et les scalings (Ĥ, FFN) valent exp(N(0, 0.1)). Aucun n'amplifie les poids,
donc l'arrondi bfloat16 final coûte la même erreur relative (~2⁻⁸) que celle
que subit déjà le modèle baseline — la comparaison de la Task 9 reste
comparable à elle-même. C'était FAUX avec le Û_vo gaussien du papier, qui
amplifiait max|W̃_o| jusqu'à ×2700.

Périmètre : ce POC obfusque chaque couche indépendamment, sans transformer la
frontière `hidden_size` entre couches (décision de design). Les matrices clés
P̂/Q̂ sont donc désactivées côté embedding comme elles le sont côté attention —
cf. la docstring de `embedding_obfuscation.py`.
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


def _write(param, value):
    """Écrit un tenseur float32 dans un paramètre en préservant son dtype."""
    param.data.copy_(value.to(param.dtype))


def obfuscate_model_in_place(model, config, seed, alpha_e=1.0, alpha_h=0.2,
                             lam=0.3, beta=8, gamma=1e3, zeta=1e3):
    """Applique l'obfuscation aux poids d'un modèle déjà chargé.

    Séparé de `transform_model` pour être testable sur un Qwen2 miniature
    (cf. `tests/test_model_transform.py`) sans télécharger 15 Go de poids.

    `h=0` n'est pas un paramètre : avec h>0 les matrices clés deviennent
    rectangulaires (d, d+2h) et tout le réseau devrait être redimensionné —
    hors périmètre du POC (décision « h=0 » du design). De toute façon les
    matrices clés sont désactivées ici.
    """
    # Vérifications obligatoires (risques identifiés dans la spec).
    assert hasattr(config, "num_key_value_heads"), (
        "config.json ne définit pas num_key_value_heads — vérifier si le "
        "modèle utilise MLA (traitement différent, hors scope) plutôt que GQA."
    )
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    d_head = getattr(config, "head_dim", None) or config.hidden_size // num_heads
    assert num_heads % num_kv_heads == 0

    # Weight tying : à vérifier avant de traiter embed_tokens/lm_head séparément.
    tied = (model.get_input_embeddings().weight.data_ptr()
            == model.get_output_embeddings().weight.data_ptr())

    w_embed = model.get_input_embeddings().weight.data.float()
    w_head = model.get_output_embeddings().weight.data.float()
    # Le bruit d'embedding est la seule dégradation NON compensée du POC : α est
    # une amplitude absolue (papier, Table 10 : α_e = 1.0) alors que les
    # coefficients d'embedding d'un modèle réel sont petits. Ce rapport dit donc
    # d'emblée si le modèle obfusqué a une chance de fonctionner (cf. --alpha-e).
    rms = max(float(w_embed.pow(2).mean().sqrt()), 1e-12)
    print(f"[obfuscation] RMS embedding = {rms:.4f}, alpha_e = {alpha_e} "
          f"-> rapport bruit/signal = {alpha_e / rms:.1f}")
    emb = obfuscate_embedding(
        w_embed, w_head, alpha_e, alpha_h, lam, h=0, seed=seed,
        apply_key_matrices=False,
    )
    _write(model.get_input_embeddings().weight, emb.w_embed_obf)
    if not tied:
        _write(model.get_output_embeddings().weight, emb.w_head_obf)
    # Si les poids sont liés, embed et head sont le MÊME tenseur : la ligne
    # ci-dessus suffit. C'est correct uniquement parce que, sans matrices clés,
    # les deux transformations se réduisent à la même permutation de lignes ;
    # seul le bruit diffère (alpha_e s'applique alors aussi au head).

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        obf_attn = obfuscate_attention_layer(
            attn.q_proj.weight.data.float(), attn.k_proj.weight.data.float(),
            attn.v_proj.weight.data.float(), attn.o_proj.weight.data.float(),
            num_heads=num_heads, num_kv_heads=num_kv_heads, d_head=d_head,
            # graines distinctes par couche ET par module (le brouillon du plan
            # faisait collisionner le FFN de la couche i avec l'attention de la
            # couche i+1, qui auraient partagé le même flux aléatoire).
            beta=beta, gamma=gamma, zeta=zeta, seed=seed * 10000 + 2 * i,
            b_q=None if attn.q_proj.bias is None else attn.q_proj.bias.data.float(),
            b_k=None if attn.k_proj.bias is None else attn.k_proj.bias.data.float(),
            b_v=None if attn.v_proj.bias is None else attn.v_proj.bias.data.float(),
            # Qwen2 applique RoPE via `rotate_half` : paires (i, i+d_head/2).
            rope_layout="half",
        )
        _write(attn.q_proj.weight, obf_attn.w_q_obf)
        _write(attn.k_proj.weight, obf_attn.w_k_obf)
        _write(attn.v_proj.weight, obf_attn.w_v_obf)
        _write(attn.o_proj.weight, obf_attn.w_o_obf)
        for proj, b_obf in ((attn.q_proj, obf_attn.b_q_obf),
                            (attn.k_proj, obf_attn.b_k_obf),
                            (attn.v_proj, obf_attn.b_v_obf)):
            if b_obf is not None:
                _write(proj.bias, b_obf)

        mlp = layer.mlp
        obf_ffn = obfuscate_ffn_layer(
            mlp.gate_proj.weight.data.float(), mlp.up_proj.weight.data.float(),
            mlp.down_proj.weight.data.float(), seed=seed * 10000 + 2 * i + 1,
        )
        _write(mlp.gate_proj.weight, obf_ffn.gate_proj_obf)
        _write(mlp.up_proj.weight, obf_ffn.up_proj_obf)
        _write(mlp.down_proj.weight, obf_ffn.down_proj_obf)

    return ObfuscationKeys(emb.permutation, emb.unpermute, seed)


def transform_model(model_name, output_dir, seed, alpha_e=1.0, alpha_h=0.2,
                    lam=0.3, beta=8, gamma=1e3, zeta=1e3):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    keys = obfuscate_model_in_place(
        model, config, seed, alpha_e=alpha_e, alpha_h=alpha_h, lam=lam,
        beta=beta, gamma=gamma, zeta=zeta,
    )

    model.save_pretrained(output_dir)
    # Les clés ne partent PAS avec le modèle : elles restent côté client.
    with open("obfuscation_keys.json", "w") as f:
        json.dump(asdict(keys), f)
    return keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    # Les α pilotent le bruit ajouté à l'embedding/unembedding, seule source de
    # dégradation non compensée : exposés en CLI pour pouvoir les balayer sur le
    # Pod (Task 9) sans rééditer le fichier.
    parser.add_argument("--alpha-e", type=float, default=1.0)
    parser.add_argument("--alpha-h", type=float, default=0.2)
    # β=1 force Ẑ_block = identité, donc supprime le mélange de fréquences RoPE
    # — la seule approximation restante côté attention. Un run à --beta 1 sert
    # de contrôle : il sépare « la reparamétrisation est-elle juste ? » de
    # « combien coûte l'approximation revendiquée par le papier ? ».
    parser.add_argument("--beta", type=int, default=8)
    args = parser.parse_args()
    transform_model(args.model, args.output, args.seed,
                    alpha_e=args.alpha_e, alpha_h=args.alpha_h, beta=args.beta)
