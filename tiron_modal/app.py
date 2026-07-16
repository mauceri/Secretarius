import subprocess

import modal

MODELS_DIR = "/models"
BASE = f"{MODELS_DIR}/Phi-4-mini-instruct-Q6_K.gguf"
LORA = f"{MODELS_DIR}/tiron-unified-lora-f16.gguf"
CTX = 8192
PORT = 8080
GPU = "L4"

# Image officielle llama.cpp CUDA (serveur pré-compilé, maintenu par le projet).
# .entrypoint([]) efface l'ENTRYPOINT `/app/llama-server` de l'image : sinon Modal
# le préfixe à chaque conteneur et notre code Python ne tourne jamais.
image = modal.Image.from_registry(
    "ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.11"
).entrypoint([])

vol = modal.Volume.from_name("tiron-models")
app = modal.App("tiron-llm-modal")


@app.function(image=image)
def diag():
    """Inspecte l'agencement de l'image : où sont le binaire et ses .so."""
    subprocess.run(
        ["sh", "-c",
         "echo '== ls -la /app =='; ls -la /app 2>&1 | head -60; "
         "echo; echo '== find libllama*.so / libggml*.so / libmtmd*.so =='; "
         "find / \\( -name 'libllama*.so' -o -name 'libggml*.so' -o -name 'libmtmd*.so' \\) 2>/dev/null; "
         "echo; echo '== ldd /app/llama-server =='; ldd /app/llama-server 2>&1; "
         "echo; echo '== LD_LIBRARY_PATH =='; echo \"$LD_LIBRARY_PATH\""],
        check=False,
    )


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODELS_DIR: vol},
    secrets=[modal.Secret.from_name("tiron-llm-api-key")],
    timeout=3600,
    scaledown_window=300,
)
@modal.web_server(port=PORT, startup_timeout=300)
def serve():
    launch = (
        "export LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH:-}; "
        f"exec /app/llama-server --model {BASE} --lora {LORA} "
        f"--host 0.0.0.0 --port {PORT} -c {CTX} -ngl 999 "
        '--api-key "$LLAMA_API_KEY"'
    )
    subprocess.Popen(["sh", "-c", launch])
