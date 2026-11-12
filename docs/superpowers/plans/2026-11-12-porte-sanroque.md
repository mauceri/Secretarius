# Porte Grok sur sanroque (conteneur isolé) — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** servir le coffre Publication à Grok par Funnel :10000 depuis sanroque, porte et wiki Publication enfermés dans un conteneur Docker filtré par pare-feu.

**Architecture :** un conteneur `porte-grok` (porte FastMCP :5052 + serveur wiki Flask :5051 interne) monte seulement le coffre Publication, le code (`tools/`, `porte/`, `.venv/`) et le cache Hugging Face. Une unité système pose les règles `DOCKER-USER`/`INPUT` au boot ; l'unité utilisateur du conteneur refuse de démarrer sans elles.

**Tech Stack :** Docker, systemd (utilisateur + système), iptables, Tailscale Funnel, Python 3.12 (venv Wiki_LM existant), pytest.

**Spec :** `docs/superpowers/specs/2026-11-12-porte-sanroque-design.md`

## Global Constraints

- Grok ne voit que le coffre **Publication** (`~/Documents/Publication`), jamais Secretarius ni Arbath.
- Funnel sur **10000** → `http://127.0.0.1:5052`. Ne pas toucher 443, 8443, 10443, 1443, 9443, 9444.
- Modèle du wiki Publication : `WIKI_LLM_BACKEND=ollama`, `OLLAMA_MODEL=qwen3:8b`.
- Conteneur : `--user 1000:1000`, `--read-only`, `--tmpfs /tmp`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`, port publié **uniquement** sur `127.0.0.1:5052`.
- Aucun secret dans le conteneur : ni `~/.config/secrets.env`, ni `Wiki_LM/.env` ni `Wiki_LM/.env.bak-*`.
- Config du conteneur : `~/.config/porte-grok/conteneur.env` (chmod 600), jamais `porte.env`.
- Confirmation de l'utilisateur avant chaque `systemctl start/stop/enable/restart`, `docker run`, `docker build`, commande Funnel, `git push`. Les commandes `sudo` sont données à l'utilisateur (`! …`).
- Ne jamais commiter `Wiki_LM/.env.bak-*` ni `gen_corpus/laya_train_chargédnasKaggle.jsonl` : `git add` par chemins explicites.
- Commits en français, terminés par les lignes d'attribution de la session.

## Constats faits en écrivant le plan (écarts avec la spec)

- **La porte écoute en dur sur `127.0.0.1`** (`porte/serveur.py:112`) : dans un conteneur, le port publié ne l'atteindrait pas. Tâche 1 ajoute `PORTE_HOTE` (défaut `127.0.0.1`).
- **`Wiki_LM/.env` est chargé automatiquement** par `tools/wiki_paths.py:19`, `tools/llm.py:32`, `tools/wiki.py:173` (chemin `Path(__file__).parent.parent / ".env"`), et le dossier contient aussi des `.env.bak-*` avec des clés. On ne monte donc **pas** tout `~/Secretarius/Wiki_LM` mais seulement `tools/`, `porte/`, `.venv/`, chacun à son chemin absolu : le `.env` n'existe pas dans le conteneur et les chargeurs le sautent (`if not env_path.exists(): return`).
- **Vérifier la règle sans `sudo` est impossible avec `iptables -S`** (root requis). Ruling : `porte-pare-feu.sh poser` écrit un témoin `/run/porte-pare-feu.ok` (root, lisible par tous, effacé au reboot car `/run` est en tmpfs) ; `verifier` teste ce témoin. Coût si faux : des règles retirées à la main sans `retirer` laisseraient un témoin périmé ; `retirer` efface le témoin.
- Hors périmètre, préexistant, à signaler : `tools/server.py:360,397` (`/cluster`) prend les embeddings sous le dossier du code et non `WIKI_PATH`. La porte n'appelle que `/query` et `/run /r` : sans effet ici.

## Review Focus

1. Réponses du conteneur vers l'hôte (docker-proxy → conteneur → réponse) : la règle `INPUT -i br-porte` doit laisser passer `ESTABLISHED,RELATED`, sinon `/health` expire. Test : Tâche 6 étape 6 (`curl /health` après la règle).
2. Pare-feu posé deux fois (redémarrage de l'unité) : pas de règles en double. Test : Tâche 2, `test_poser_deux_fois_sans_doublon`.
3. Docker démarré après l'unité de pare-feu : la chaîne `DOCKER-USER` créée par le script doit être réutilisée par Docker, pas remplacée. Test : Tâche 6 étape 9 (reboot, ou `systemctl restart docker` puis contrôle `iptables -S DOCKER-USER`, par l'utilisateur).
4. Témoin absent (boot sans unité de pare-feu) : le conteneur ne démarre pas. Test : Tâche 2, `test_verifier_echoue_sans_temoin`, et Tâche 6 étape 4.
5. Fichier écrit par un agent via la porte : propriétaire uid 1000 et visible par `ob sync`. Test : Tâche 6 étape 5.

---

### Task 1 : la porte écoute sur l'adresse donnée par `PORTE_HOTE`

**Files:**
- Modify: `Wiki_LM/porte/serveur.py:104-112`
- Test: `Wiki_LM/tests/test_porte_serveur.py`

**Interfaces:**
- Produces: `hote_ecoute() -> str` dans `porte/serveur.py` ; `main()` passe `host=hote_ecoute()` à `uvicorn.run`. Variable `PORTE_HOTE`, défaut `"127.0.0.1"`.

- [ ] **Step 1 : tests qui échouent**

```python
from porte.serveur import hote_ecoute

def test_hote_ecoute_defaut_boucle_locale(monkeypatch):
    monkeypatch.delenv("PORTE_HOTE", raising=False)
    assert hote_ecoute() == "127.0.0.1"

def test_hote_ecoute_lit_porte_hote(monkeypatch):
    monkeypatch.setenv("PORTE_HOTE", "0.0.0.0")
    assert hote_ecoute() == "0.0.0.0"
```

- [ ] **Step 2 :** `cd Wiki_LM && .venv/bin/python -m pytest tests/test_porte_serveur.py -k hote_ecoute -v` → FAIL (ImportError `hote_ecoute`).
- [ ] **Step 3 :** implémenter `hote_ecoute()` (`os.environ.get("PORTE_HOTE", "127.0.0.1")`) et l'utiliser dans `main()`.
- [ ] **Step 4 :** `.venv/bin/python -m pytest tests/test_porte_*.py -q` → tout passe.
- [ ] **Step 5 :** commit `feat(porte): adresse d'écoute réglable par PORTE_HOTE` (`Wiki_LM/porte/serveur.py`, `Wiki_LM/tests/test_porte_serveur.py`).

### Task 2 : script de pare-feu et unité système

**Files:**
- Create: `openclaw-config/porte-conteneur/porte-pare-feu.sh` (exécutable)
- Create: `openclaw-config/porte-conteneur/porte-pare-feu.service`
- Test: `openclaw-config/tests/test_porte_pare_feu.py`

**Interfaces:**
- Produces: `porte-pare-feu.sh poser|retirer|verifier`. Codes : 0 succès ; `verifier` → 1 si témoin absent ; 2 si action inconnue. Variables surchargeables pour les tests : `IPTABLES` (défaut `iptables`), `TEMOIN` (défaut `/run/porte-pare-feu.ok`), `PONT` (défaut `br-porte`), `PORT_OLLAMA` (défaut `11434`).
- Règles exactes posées par `poser` (chacune ajoutée seulement si `$IPTABLES -C …` échoue) :
  - `-N DOCKER-USER` si la chaîne n'existe pas (`$IPTABLES -L DOCKER-USER -n` échoue) ;
  - `DOCKER-USER -i $PONT -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN`
  - `DOCKER-USER -i $PONT -j DROP`
  - `INPUT -i $PONT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT`
  - `INPUT -i $PONT -p tcp --dport $PORT_OLLAMA -j ACCEPT`
  - `INPUT -i $PONT -j DROP`
  - Ordre : insertion (`-I`) en tête de chaîne, dans l'ordre inverse, pour que l'ordre final soit celui ci-dessus et passe avant les règles de Docker/ufw. Puis `touch $TEMOIN` et `chmod 644`.
- `retirer` : `-D` de chacune si présente (`-C`), `rm -f $TEMOIN`. Ne supprime pas la chaîne `DOCKER-USER`.
- Unité : `Type=oneshot`, `RemainAfterExit=yes`, `Before=docker.service`, `After=network-pre.target`, `ExecStart=/usr/local/sbin/porte-pare-feu.sh poser`, `ExecStop=/usr/local/sbin/porte-pare-feu.sh retirer`, `WantedBy=multi-user.target`.

- [ ] **Step 1 : tests qui échouent.** Un faux `iptables` (script Python ou bash dans `tmp_path`) tient un fichier d'état des règles, répond 0/1 à `-C`/`-L` selon cet état, applique `-I`/`-D`/`-N`, et journalise chaque appel.

```python
def test_poser_ecrit_les_regles_dans_l_ordre(faux): ...
    # état DOCKER-USER == [established RETURN, DROP] ; INPUT == [established ACCEPT, 11434 ACCEPT, DROP]
def test_poser_deux_fois_sans_doublon(faux): ...      # 2× poser → même état
def test_poser_cree_docker_user_si_absente(faux): ... # appel -N DOCKER-USER journalisé
def test_poser_ecrit_le_temoin(faux): ...             # TEMOIN existe, mode 0o644
def test_retirer_enleve_regles_et_temoin(faux): ...   # chaînes vides, TEMOIN absent
def test_verifier_echoue_sans_temoin(faux): ...       # code 1
def test_verifier_reussit_avec_temoin(faux): ...      # code 0 après poser
def test_action_inconnue(faux): ...                   # code 2
```

- [ ] **Step 2 :** `Wiki_LM/.venv/bin/python -m pytest openclaw-config/tests/test_porte_pare_feu.py -v` → FAIL (script absent).
- [ ] **Step 3 :** écrire le script (`set -euo pipefail`) et l'unité.
- [ ] **Step 4 :** tests → PASS ; `systemd-analyze verify openclaw-config/porte-conteneur/porte-pare-feu.service` → aucune erreur autre que le chemin `/usr/local/sbin` absent.
- [ ] **Step 5 :** commit `feat(porte): pare-feu du conteneur, posé au démarrage, témoin pour vérification sans sudo`.

### Task 3 : image et point d'entrée

**Files:**
- Create: `openclaw-config/porte-conteneur/Dockerfile`
- Create: `openclaw-config/porte-conteneur/demarrer.sh` (exécutable)

**Interfaces:**
- Produces: image `porte-grok:local`. `demarrer.sh` lance `$VENV/bin/python3 $TOOLS/server.py --no-public` en arrière-plan (cwd `$TOOLS`), puis `$VENV/bin/python3 -m porte.serveur` (cwd parent de `porte/`, soit `/home/mauceric/Secretarius/Wiki_LM`), puis `wait -n` et sortie avec le code du premier processus mort (l'autre est tué par `trap`). `VENV`, `TOOLS` : variables d'environnement fournies par l'unité.

- [ ] **Step 1 :** Dockerfile : `FROM ubuntu:24.04`, `apt-get install -y --no-install-recommends python3.12 curl ca-certificates` puis nettoyage des listes, `COPY demarrer.sh /usr/local/bin/`, `ENTRYPOINT ["/usr/local/bin/demarrer.sh"]`. Aucun `pip install`.
- [ ] **Step 2 :** (confirmation utilisateur) `docker build -t porte-grok:local openclaw-config/porte-conteneur` → succès ; `docker run --rm --entrypoint /usr/bin/python3.12 porte-grok:local --version` → `Python 3.12.3` (même version que `readlink -f Wiki_LM/.venv/bin/python3`).
- [ ] **Step 3 :** `bash -n openclaw-config/porte-conteneur/demarrer.sh` → OK.
- [ ] **Step 4 :** commit `feat(porte): image minimale et point d'entrée du conteneur`.

### Task 4 : unités utilisateur du conteneur et de l'embedding, désinstallation

**Files:**
- Create: `openclaw-config/porte-conteneur/porte-grok-conteneur.service`
- Create: `openclaw-config/porte-conteneur/porte-grok-embed.service`, `porte-grok-embed.timer`
- Modify: `openclaw-config/uninstall.sh:25-37` (liste `SERVICES`)

**Interfaces:**
- Consumes: `porte-pare-feu.sh verifier` (Tâche 2, installé en `/usr/local/sbin`), image `porte-grok:local` (Tâche 3), `PORTE_HOTE` (Tâche 1).
- `porte-grok-conteneur.service` :
  - `Conflicts=porte-grok.service`, `Restart=always`, `RestartSec=10`.
  - `ExecStartPre=/usr/local/sbin/porte-pare-feu.sh verifier`
  - `ExecStartPre=-/usr/bin/docker network create -o com.docker.network.bridge.name=br-porte porte-grok`
  - `ExecStartPre=-/usr/bin/docker rm -f porte-grok`
  - `ExecStart=/usr/bin/docker run --rm --name porte-grok --network porte-grok -p 127.0.0.1:5052:5052 --add-host host.docker.internal:host-gateway --env-file %h/.config/porte-grok/conteneur.env` + les options des Global Constraints + montages :
    - `-v %h/Documents/Publication:/coffre`
    - `-v %h/Secretarius/Wiki_LM/tools:%h/Secretarius/Wiki_LM/tools:ro`, idem `porte`, `.venv`
    - `-v %h/.cache/huggingface:/hf:ro`, `-e HF_HOME=/hf`
    - `-v %h/nltk_data:/nltk:ro`, `-e NLTK_DATA=/nltk` (`tools/central_passages.py:32` cherche `punkt_tab` et sinon tente un téléchargement, que le pare-feu bloque)
    - `-v %h/.config/porte-grok/agents:/etc/porte/agents:ro`
    - `-e VENV=%h/Secretarius/Wiki_LM/.venv -e TOOLS=%h/Secretarius/Wiki_LM/tools`
    - image `porte-grok:local`
  - `ExecStop=/usr/bin/docker stop porte-grok`, `WantedBy=default.target`.
- `conteneur.env` (rédigé à la Tâche 6, documenté en commentaire dans l'unité) : `PORTE_HOTE=0.0.0.0`, `PORTE_COFFRE=/coffre`, `PORTE_AGENTS=/etc/porte/agents`, `PORTE_WIKI_URL=http://127.0.0.1:5051`, `WIKI_PATH=/coffre/Wiki_LM`, `WIKI_LLM_BACKEND=ollama`, `OLLAMA_MODEL=qwen3:8b`, `OLLAMA_BASE_URL=http://host.docker.internal:11434`, `WIKI_VAULT_MIRRORS=`, `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`.
- `porte-grok-embed.service` : `Type=oneshot`, `ExecStart=/usr/bin/docker exec porte-grok %h/Secretarius/Wiki_LM/.venv/bin/python3 %h/Secretarius/Wiki_LM/tools/embed.py`, `ExecStartPost=-/usr/bin/docker exec porte-grok curl -fsS -X POST http://127.0.0.1:5051/reload`. Timer : `OnCalendar=*-*-* 00,06,12,18:00:00`, `Persistent=true`.
- `uninstall.sh` : ajouter `porte-grok-conteneur.service`, `porte-grok-embed.timer`, `porte-grok-embed.service`.

- [ ] **Step 1 :** écrire les trois unités et la modification de `uninstall.sh`.
- [ ] **Step 2 :** `systemd-analyze --user verify openclaw-config/porte-conteneur/*.service openclaw-config/porte-conteneur/*.timer` → aucune erreur de syntaxe ; `bash -n openclaw-config/uninstall.sh` → OK ; `grep -c porte-grok-conteneur openclaw-config/uninstall.sh` → 1.
- [ ] **Step 3 :** commit `feat(porte): unités du conteneur et de l'embedding Publication ; désinstallation`.

### Task 5 : fiche coordinateur

**Files:**
- Modify: `docs/grok/fiche-coordinateur.md:4,11`

- [ ] **Step 1 :** remplacer « la machine santiago » par « la machine sanroque » (l. 4) et l'adresse par `https://sanroque.<tailnet>.ts.net:10000/mcp` (l. 11). Rien d'autre.
- [ ] **Step 2 :** `grep -n "santiago\|8443" docs/grok/fiche-coordinateur.md` → aucune ligne.
- [ ] **Step 3 :** commit `docs(grok): la porte est sur sanroque, port 10000`.

### Task 6 : déploiement et vérification (avec l'utilisateur)

Chaque commande `docker run/build`, `systemctl`, `tailscale funnel` est soumise à l'utilisateur avant d'être lancée ; les `sudo` lui sont donnés.

- [ ] **Step 1 : restaurer les dérivés.** Extraire dans un dossier vide du scratchpad les 7 fichiers `Documents/Publication/Wiki_LM/…` de `~/Documents-sauvegardes/santiago-20261106/santiago.tar.gz`, puis copier avec `cp -n` vers `~/Documents/Publication/Wiki_LM/` (aucun écrasement). Vérif : nombre d'entrées de `embeddings/embeddings_index.json` = nombre de pages `wiki/**/*.md` non vides ; si écart, le noter et relancer l'embedding à l'étape 8.
- [ ] **Step 2 : config.** `mkdir -p ~/.config/porte-grok` (700) ; `conteneur.env` (600) avec les valeurs de la Tâche 4 ; `agents` vide (600). `cp` des unités utilisateur dans `~/.config/systemd/user/`, `systemctl --user daemon-reload`.
- [ ] **Step 3 : jeton.** `porte-jeton` n'existe pas sur sanroque (profil lourd) : `cd Wiki_LM && PORTE_AGENTS=~/.config/porte-grok/agents .venv/bin/python3 -m porte.jetons ajouter orchestrateur` (vérifier d'abord dans `porte/jetons.py:89-119` comment le fichier est désigné). Le jeton s'affiche une fois : le donner à l'utilisateur, ne pas l'écrire ailleurs.
- [ ] **Step 4 : fermé par défaut.** Sans témoin : `systemctl --user start porte-grok-conteneur` → échec, journal montre `verifier` en échec, aucun conteneur (`docker ps`).
- [ ] **Step 5 : lancement sans pare-feu, tests d'isolation « avant ».** Pour ce seul essai, lancer le `docker run` de l'unité à la main (sans `ExecStartPre verifier`). Vérifs :
  - `curl -fsS 127.0.0.1:5052/health` → OK ; sans jeton sur `/mcp` → 401 ; avec jeton, `wiki_recherche` sur un mot de Publication → résultats de Publication, aucun slug de Secretarius.
  - `docker exec porte-grok sh -c 'ls ~/.config/secrets.env /home/mauceric/Secretarius/Wiki_LM/.env'` → absents ; `touch /home/mauceric/Secretarius/Wiki_LM/tools/x` → refus (lecture seule).
  - `deposer` par la porte d'un fichier `orchestrateur-essai.md` → `stat -c %u` sur l'hôte = 1000 ; le supprimer ensuite.
  - `docker exec porte-grok curl -m3 http://host.docker.internal:5051/` → échec (5051 de l'hôte est en `127.0.0.1`) ; idem `:1143` (Bridge).
  - `docker exec porte-grok curl -m3 http://host.docker.internal:11434/api/tags` → OK.
  - `docker exec porte-grok sh -c 'exec 3<>/dev/tcp/host.docker.internal/22'` → **réussit** (preuve que le test voit ce que la règle bloquera) ; idem `curl -m3 https://example.org` → réussit.
  - `docker stop porte-grok`.
- [ ] **Step 6 : pare-feu (utilisateur, `sudo`).** `sudo install -m 755 openclaw-config/porte-conteneur/porte-pare-feu.sh /usr/local/sbin/`, `sudo cp …/porte-pare-feu.service /etc/systemd/system/`, `sudo systemctl daemon-reload && sudo systemctl enable --now porte-pare-feu`. Puis `systemctl --user enable --now porte-grok-conteneur`. Refaire les vérifs de l'étape 5 : SSH (22), `example.org`, Samba (445), VNC (5900) → **échec** ; Ollama → OK ; `/health` → OK ; 5051 et Bridge → échec.
- [ ] **Step 7 : `/q` réel.** Par la porte, `wiki_question` simple → réponse en < 360 s.
- [ ] **Step 8 : embedding.** `systemctl --user enable --now porte-grok-embed.timer` ; `systemctl --user start porte-grok-embed.service` → succès, `/reload` OK.
- [ ] **Step 9 : redémarrage (utilisateur).** Après un reboot de sanroque (ou, à défaut, `sudo systemctl restart docker`) : `sudo iptables -S DOCKER-USER` et `-S INPUT` montrent les règles une fois chacune ; `ls /run/porte-pare-feu.ok` ; conteneur en marche ; `/health` OK.
- [ ] **Step 10 : Funnel.** (utilisateur, si demandé : autoriser Funnel pour sanroque dans la console, `sudo tailscale set --operator=$USER`) puis `tailscale funnel --bg --https=10000 http://127.0.0.1:5052`. `tailscale serve status --json` : les autres ports inchangés. Depuis un appareil hors tailnet (l'utilisateur, par exemple sur son téléphone sans Wi-Fi) : `/mcp` sans jeton → 401.
- [ ] **Step 11 : Grok.** L'utilisateur met l'adresse et le jeton dans la config de Grok et fait un essai (`wiki_recherche`). Lire le journal : `docker logs porte-grok`.
- [ ] **Step 12 :** fiche mémoire datée (`project_v1_chantiers_20261001.md` : porte sur sanroque, conteneur, Funnel 10000) ; proposer le `git push` à l'utilisateur.

## Retour arrière

`systemctl --user disable --now porte-grok-conteneur porte-grok-embed.timer` ; `tailscale funnel --https=10000 off` ; `sudo systemctl disable --now porte-pare-feu` (retire les règles et le témoin).
