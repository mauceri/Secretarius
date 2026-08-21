"""AloePri sur Modal : transformation (optionnelle) et service du modèle
obfusqué Qwen3-8B.

Deux fonctions :
- `transform()` : reproduit la transformation AloePri sur le modèle source
  (Qwen/Qwen3-8B) dans un conteneur Modal (streaming mémoire-léger, cf.
  `aloepri_poc/transform_streaming.py`), écrit le modèle obfusqué sur le
  Volume `aloepri-models` et les clés sur le Volume `aloepri-keys`.
  C'est l'option « reproductible » ; le chemin rapide est d'uploader un
  modèle déjà transformé en local (`modal volume put`, cf. README).
- `serve()` : serveur HTTP qui charge le modèle obfusqué depuis
  `aloepri-models` et sert `/generate` (IDs de tokens PERMUTÉS) + `/health`.
  Il ne voit JAMAIS les clés : le client permute avant l'envoi et dépermute à
  la réception (`aloepri_poc/client_wrapper.py`).

Le Volume `aloepri-keys` n'est monté QUE par `transform()` — jamais par
`serve()` (posture de sécurité du POC : la clé de permutation reste côté
client, cf. RUNBOOK).

Déploiement : voir README.md. GPU de service : L4 (24 Go VRAM) — le modèle
pèse ~16 Go en bf16 ; passer à A100-40GB pour de très longs contextes.
"""
import os
import sys

import modal

MODELS_DIR = "/models"
KEYS_DIR = "/keys"
MODEL_VOL = "aloepri-models"
KEYS_VOL = "aloepri-keys"
SRC_MODEL = "Qwen/Qwen3-8B"          # source de la transformation
MODEL_SUBDIR = "qwen3-8b-obf"         # sous-répertoire sur le Volume service
KEYS_FILENAME = "obfuscation_keys.json"
PORT = 8000
GPU_SERVE = "L4"                       # 24 Go VRAM ; A100-40GB si contexte long
TRANSFORM_MEMORY = 12288               # MiB garantis pour transform()
# Disque éphémère Modal : minimum 512 GiB (524288 MiB), jusqu'à 3 TiB. Le
# cache HF (16 Go) + la sortie (16 Go) tiennent largement dans 512 GiB.
TRANSFORM_EPHEMERAL_DISK = 524288

_POC_DIR = os.path.join(os.path.dirname(__file__), "..", "aloepri_poc")

# Image du transform : le POC est copié DANS l'image au build (pas de Mount
# runtime), puis `transform_streaming.py` est importé depuis /pkg.
TRANSFORM_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers>=4.51", "numpy", "scipy",
        "safetensors", "huggingface_hub",
    )
    .add_local_dir(_POC_DIR, "/pkg/aloepri_poc", copy=True)
)
# Image du service : transformers + serveur web, sans le POC (inutile ici).
SERVE_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers>=4.51", "fastapi", "uvicorn",
        "safetensors",
    )
)

models_vol = modal.Volume.from_name(MODEL_VOL, create_if_missing=True)
keys_vol = modal.Volume.from_name(KEYS_VOL, create_if_missing=True)

try:
    API_SECRET = modal.Secret.from_name("aloepri-api-key")
except modal.exception.NotFoundError:  # pragma: no cover
    API_SECRET = None

app = modal.App("aloepri-qwen3-modal")


@app.function(
    image=TRANSFORM_IMAGE,
    memory=TRANSFORM_MEMORY,
    ephemeral_disk=TRANSFORM_EPHEMERAL_DISK,
    volumes={MODELS_DIR: models_vol, KEYS_DIR: keys_vol},
    timeout=3600,
    scaledown_window=300,
)
def transform(
    seed: int = 0,
    alpha_e: float = 1.0,
    alpha_h: float = 0.2,
    beta: int = 8,
    zeta: float = 1e3,
    rope_scaling: str = "auto",
):
    """Reproduit la transformation AloePri sur le modèle source.

    Sortie : modèle obfusqué sous {MODELS_DIR}/{MODEL_SUBDIR} sur le Volume
    `aloepri-models` ; clés sous {KEYS_DIR}/{KEYS_FILENAME} sur le Volume
    `aloepri-keys` (jamais monté par `serve`). Retourne le chemin des clés et
    leur empreinte SHA-256 — à comparer côté client après téléchargement.
    """
    sys.path.insert(0, "/pkg/aloepri_poc")
    from transform_streaming import transform_streaming

    out_dir = os.path.join(MODELS_DIR, MODEL_SUBDIR)
    keys_path = os.path.join(KEYS_DIR, KEYS_FILENAME)
    keys = transform_streaming(
        SRC_MODEL, out_dir, seed,
        alpha_e=alpha_e, alpha_h=alpha_h, beta=beta, zeta=zeta,
        keys_path=keys_path, rope_scaling=rope_scaling,
    )
    models_vol.commit()
    keys_vol.commit()
    import hashlib
    with open(keys_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {
        "model_dir": out_dir,
        "keys_path": keys_path,
        "keys_sha256": digest,
        "seed": seed,
        "alpha_e": alpha_e, "alpha_h": alpha_h, "beta": beta, "zeta": zeta,
    }


@app.function(
    image=SERVE_IMAGE,
    gpu=GPU_SERVE,
    volumes={MODELS_DIR: models_vol},
    secrets=[API_SECRET] if API_SECRET else [],
    timeout=3600,
    scaledown_window=300,
)
@modal.asgi_app()
def serve():
    """Serveur HTTP du modèle obfusqué (cf. `aloepri_poc/server.py`).

    Traite uniquement des IDs de tokens PERMUTÉS (aucun tokenizer, aucune
    clé) : le client permute avant l'envoi et dépermute à la réception.

    `@modal.asgi_app()` : la fonction RETOURNE l'app FastAPI (le conteneur
    est prêt dès qu'elle est retournée) — avec `@modal.web_server` il
    faudrait lancer uvicorn en sous-processus et rendre la main (pattern
    tiron), sinon la passerelle renvoie 303 tant que la fonction n'est pas
    retournée."""
    import torch
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel
    from transformers import AutoModelForCausalLM

    model_dir = os.path.join(MODELS_DIR, MODEL_SUBDIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16).cuda()
    model.eval()

    fastapi_app = FastAPI()

    class GenerateRequest(BaseModel):
        input_ids: list[int]
        max_new_tokens: int = 100

    class GenerateResponse(BaseModel):
        output_ids: list[int]

    def _authorized(authorization: str | None) -> bool:
        # Les valeurs du Secret Modal sont injectées comme variables
        # d'environnement dans le conteneur (pas de `.get()` sur l'objet
        # Secret dans ce SDK). Pas de secret monté → pas d'authentification.
        expected = os.environ.get("ALOEPRI_API_KEY")
        if not expected:
            return True
        return authorization == f"Bearer {expected}"

    @fastapi_app.get("/health")
    def health(authorization: str | None = Header(default=None)):
        if not _authorized(authorization):
            raise HTTPException(status_code=401, detail="unauthorized")
        return {"status": "ok"}

    @fastapi_app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest,
                 authorization: str | None = Header(default=None)):
        if not _authorized(authorization):
            raise HTTPException(status_code=401, detail="unauthorized")
        input_tensor = torch.tensor([req.input_ids], device=model.device)
        with torch.no_grad():
            output = model.generate(
                input_tensor, max_new_tokens=req.max_new_tokens,
                do_sample=False,
            )
        return GenerateResponse(output_ids=output[0].tolist())

    return fastapi_app


@app.function(image=TRANSFORM_IMAGE, scaledown_window=60)
def poc_diag():
    """Diagnostic : où sont les fichiers du POC dans l'image ?"""
    import os
    import subprocess
    import sys as _sys
    print("sys.path:", _sys.path)
    print("ls /pkg/aloepri_poc:", os.listdir("/pkg/aloepri_poc"))
    _sys.path.insert(0, "/pkg/aloepri_poc")
    try:
        import transform_streaming
        print("import OK:", transform_streaming.__file__)
    except Exception as e:
        print("import FAILED:", type(e).__name__, e)
    subprocess.run(
        ["sh", "-c",
         "echo '== find transform_streaming =='; "
         "find / -name 'transform_streaming.py' 2>/dev/null"],
        check=False,
    )


@app.function(volumes={MODELS_DIR: models_vol, KEYS_DIR: keys_vol},
              scaledown_window=60)
def diag():
    """État des Volumes : contenu du modèle de service et des clés."""
    for d in (MODELS_DIR, KEYS_DIR):
        print(f"== {d} ==")
        for root, _dirs, files in os.walk(d):
            for fname in files:
                path = os.path.join(root, fname)
                print(f"  {path}  ({os.path.getsize(path) / 1e6:.1f} Mo)")
    if os.path.exists(os.path.join(KEYS_DIR, KEYS_FILENAME)):
        import hashlib
        with open(os.path.join(KEYS_DIR, KEYS_FILENAME), "rb") as f:
            print("keys sha256:",
                  hashlib.sha256(f.read()).hexdigest())
