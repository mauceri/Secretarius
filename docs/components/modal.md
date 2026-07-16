---
tags: [documentation, secretarius]
date: 2026-07-16
---

# Composant : modal (cerveau Tiron serverless)

## Rôle

Servir le cerveau de Tiron (phi-4-mini + adaptateur `tiron-unified`, mêmes
GGUFs qu'en local) sur un GPU serverless Modal (L4), via un endpoint
OpenAI-compatible protégé par clé API. Usage : secours, démo, ou cerveau des
installations VPS sans GPU (ex. santiago). L'état nominal reste le service
local `slm-llama_cpp.service` (port 8998).

Code : `tiron_modal/app.py`. Mesures (2026-07-15, L4 à chaud) : 252 tok/s
prompt, 63,8 tok/s génération, ~0,9 s/requête.

## Prérequis

- Compte Modal + CLI : `~/modal-venv/bin/modal` (sinon :
  `python3 -m venv ~/modal-venv && ~/modal-venv/bin/pip install -U modal`
  puis `~/modal-venv/bin/modal setup`).
- Volume `tiron-models` contenant les GGUFs (voir « Charger un nouveau LLM »).
- Secret `tiron-llm-api-key` (voir « Clé API »).

## Démarrer / stopper

```bash
~/modal-venv/bin/modal deploy tiron_modal/app.py   # déploie (URL stable, imprimée)
~/modal-venv/bin/modal app list                    # état des apps
~/modal-venv/bin/modal app logs tiron-llm-modal    # logs (requêtes llama-server)
~/modal-venv/bin/modal app stop tiron-llm-modal    # stoppe l'app
~/modal-venv/bin/modal serve tiron_modal/app.py    # mode dev éphémère (terminal ouvert)
```

URL de l'endpoint : `https://mauceri--tiron-llm-modal-serve.modal.run`.

Comportement serverless : conteneur éteint après ~5 min sans requête
(scale-to-zero, facturation stoppée). La requête suivante subit le cold start
(boot + chargement 3 Go : 503 pendant le chargement) — le premier message
Tiron après une pause peut donc échouer : réessayer 1 à 2 min plus tard.
Astuce réveil avant une session d'usage :

```bash
KEY=$(cat ~/.openclaw/secrets/tiron-llm-key)
until [ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" \
  "https://mauceri--tiron-llm-modal-serve.modal.run/health")" = "200" ]; do sleep 5; done
```

Coût : L4 ≈ 0,80 $/h, facturé au temps d'allumage GPU uniquement.

## Clé API

Secret partagé arbitraire (pas émis par Modal) : `llama-server --api-key`
compare simplement le Bearer reçu.

```bash
# génération + stockage local
openssl rand -hex 32 > ~/.openclaw/secrets/tiron-llm-key
chmod 600 ~/.openclaw/secrets/tiron-llm-key
# côté serveur (Secret Modal, lu par app.py)
~/modal-venv/bin/modal secret create tiron-llm-api-key \
  LLAMA_API_KEY="$(cat ~/.openclaw/secrets/tiron-llm-key)"
```

La même valeur est recopiée côté consommateurs : `apiKey` du provider
`tiron-llm` (openclaw.json), `TIRON_LLAMA_KEY` (routeur),
`TIRON_LLM_KEY` (installation, qui **devient** l'`apiKey` du provider —
distinct de `TIRON_LLAMA_KEY` du routeur ; les deux noms ne diffèrent que par
`LLM`/`LLAMA`). Rotation : régénérer la valeur, recréer le Secret, redéployer,
mettre à jour les consommateurs.

## Basculer le cerveau de Tiron

Deux consommateurs à pointer, puis deux restarts. Piège de format : le
provider attend l'URL **avec** `/v1`, le routeur **sans**.

Aller (local → Modal) :

1. Déployer si besoin (`modal deploy`), noter l'URL.
2. `~/.openclaw/openclaw.json`, provider `tiron-llm` :
   `baseUrl` = `https://…modal.run/v1`, `apiKey` = contenu de
   `~/.openclaw/secrets/tiron-llm-key`. **Puis synchroniser le .bak**
   (`cp ~/.openclaw/openclaw.json ~/.openclaw/openclaw.json.bak`), sinon le
   gateway peut restaurer l'ancienne config (anti-clobber).
3. Routeur — drop-in systemd :
   ```bash
   mkdir -p ~/.config/systemd/user/tiron-router.service.d
   cat > ~/.config/systemd/user/tiron-router.service.d/modal.conf <<EOF
   [Service]
   Environment=TIRON_LLAMA_BASE=https://mauceri--tiron-llm-modal-serve.modal.run
   Environment=TIRON_LLAMA_KEY=<clé>
   EOF
   systemctl --user daemon-reload
   ```
   Ce fichier contient la clé en clair sur le disque (hors dépôt, mais non
   chiffré) — à supprimer au retour au local.
4. `systemctl --user restart tiron-router openclaw-gateway`

Retour (Modal → local) : `baseUrl` = `http://127.0.0.1:8998/v1`, remettre
`apiKey` = `local` (valeur inerte des installations locales) ou retirer le
champ, sync `.bak` ; supprimer le drop-in (`rm …/modal.conf`,
`daemon-reload`) ; mêmes restarts ; optionnel `modal app stop tiron-llm-modal`.

Attention : `tiron-router.service` a `Requires=slm-llama_cpp.service` — ne pas
stopper le service local pendant que le cerveau est sur Modal, sinon le
routeur tombe aussi.

## Installation VPS (cerveau distant dès l'installation)

Pour une installation sans GPU (ex. santiago), passer à `install.sh` :

```bash
TIRON_LLM_URL=https://mauceri--tiron-llm-modal-serve.modal.run \
TIRON_LLM_KEY=<clé> ./install.sh
```

Défauts (`install.conf`) : `TIRON_LLM_URL=http://127.0.0.1:8998`,
`TIRON_LLM_KEY=local` (valeur inoffensive : le llama-server local ignore
l'en-tête). Le routeur 8999 n'est pas géré par `install.sh` (unité manuelle,
sanroque uniquement).

## Charger un nouveau LLM

```bash
~/modal-venv/bin/modal volume put tiron-models <fichier>.gguf /<fichier>.gguf
```

Puis ajuster dans `tiron_modal/app.py` : `BASE` (et `LORA`, ou retirer
`--lora` de la ligne de lancement), `CTX`, `GPU` si besoin, et redéployer.
Vigilance : la version GGUF de l'adaptateur doit être compatible avec la
release llama.cpp de l'image (`ghcr.io/ggml-org/llama.cpp:server-cuda`) —
les incompatibilités de version GGUF ont déjà causé des pannes.

## Confidentialité

TLS (HTTPS Modal) + clé API protègent le transport et l'accès. En revanche
les prompts sont traités **en clair** dans l'infrastructure Modal (l'inférence
l'exige). Règle d'usage : pas de contenu sensible quand le cerveau est sur
Modal ; le local reste la voie du confidentiel. Piste à l'étude (backlog) :
confidentialité renforcée (confidential computing / GPU TEE, anonymisation).
