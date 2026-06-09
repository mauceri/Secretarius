# Image Docker de l'agent Tiron — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire l'image Docker `secretarius-tiron:latest`, câbler l'agent `main` (Tiron) sur `exec.host = sandbox` avec le binaire `gog` embarqué, et bootstrapper une identité OAuth gog isolée dans le workspace de l'agent.

**Architecture:** L'image dérive de `openclaw-sandbox:bookworm-slim` avec un seul ajout : le binaire statique `gog` (~21 Mo, aucune dépendance dynamique). Aucun bind externe — les credentials gog sont créés dans le workspace de l'agent (`/agent/.gog-config/gogcli/`) via `XDG_CONFIG_HOME`, sans jamais exposer les secrets de l'hôte.

**Spec de référence :** `docs/superpowers/specs/2026-06-08-image-agent-tiron-design.md`

**Tech Stack :** Docker, `openclaw-sandbox:bookworm-slim`, `gog` v0.9.0 (statiquement lié), OpenClaw 5.12 (`--profile slm`), systemd.

**⚠ Task 5 est manuelle et interactive** — elle requiert la présence de l'utilisateur (flow OAuth via navigateur). Elle ne peut pas être déléguée à un sous-agent.

---

## Fichiers créés ou modifiés

| Fichier | Action | Responsabilité |
|---|---|---|
| `openclaw-config/Dockerfile.tiron` | Créer | Image Docker — COPY du binaire gog sur la base bookworm-slim |
| `.gitignore` | Modifier | Exclure `gog-bin` (artefact temporaire de build, ne jamais versionner) |
| `openclaw-config/openclaw-slm.json.template` | Modifier | Ajouter l'entrée `main` avec image, env `XDG_CONFIG_HOME`, `tools.exec.host` |
| `~/.openclaw-slm/openclaw.json` | Modifier | Même changement sur la config live (non versionné) |

---

### Task 1 : Dockerfile et build de l'image

**Fichiers :**
- Créer : `openclaw-config/Dockerfile.tiron`
- Modifier : `.gitignore`

- [ ] **Étape 1 : Vérifier que le binaire gog est bien statiquement lié**

```bash
file $(which gog)
# Attendu : ELF 64-bit LSB executable, x86-64 ... statically linked, stripped
ldd $(which gog)
# Attendu : "n'est pas un exécutable dynamique"
```

- [ ] **Étape 2 : Écrire `openclaw-config/Dockerfile.tiron`**

```dockerfile
FROM openclaw-sandbox:bookworm-slim
COPY gog-bin /usr/local/bin/gog
```

- [ ] **Étape 3 : Ajouter `gog-bin` à `.gitignore`**

Ajouter à la fin de `.gitignore` :

```
# Artefact temporaire de build image Tiron (binaire statique copié puis supprimé)
gog-bin
```

- [ ] **Étape 4 : Copier le binaire dans le contexte de build**

```bash
cp $(which gog) gog-bin
```

- [ ] **Étape 5 : Construire l'image**

```bash
docker build -t secretarius-tiron:latest -f openclaw-config/Dockerfile.tiron .
```

Attendu : build en quelques secondes (pas de téléchargement réseau, juste un COPY), sans erreur.

- [ ] **Étape 6 : Nettoyer le binaire temporaire**

```bash
rm gog-bin
```

- [ ] **Étape 7 : Vérifier la taille de l'image**

```bash
docker images secretarius-tiron:latest
```

Attendu : taille ≈ 230-235 Mo (base 211 Mo + binaire gog ~21 Mo).

- [ ] **Étape 8 : Vérifier que le binaire est présent et exécutable dans l'image**

```bash
docker run --rm secretarius-tiron:latest gog --version
```

Attendu : une ligne du type `v0.9.0 (99d9575 ...)`.

- [ ] **Étape 9 : Vérifier l'isolation — gog absent de l'image de base**

```bash
docker run --rm openclaw-sandbox:bookworm-slim gog --version 2>&1 | head -5
```

Attendu : erreur de type `sh: 1: gog: not found` ou `gog: command not found` — le binaire est absent de l'image de base.

- [ ] **Étape 10 : Committer**

```bash
git add openclaw-config/Dockerfile.tiron .gitignore
git commit -m "feat(tiron-image): Dockerfile mono-étape + ignore artefact gog-bin"
```

---

### Task 2 : Mise à jour du template de configuration

**Fichiers :**
- Modifier : `openclaw-config/openclaw-slm.json.template` (ligne 72 : `{ "id": "main" }`)

- [ ] **Étape 1 : Remplacer l'entrée `main` vide par l'entrée complète**

Dans `openclaw-config/openclaw-slm.json.template`, remplacer :

```json
      { "id": "main" },
```

par :

```json
      {
        "id": "main",
        "sandbox": {
          "docker": {
            "image": "secretarius-tiron:latest",
            "env": {
              "XDG_CONFIG_HOME": "/agent/.gog-config"
            }
          }
        },
        "tools": {
          "exec": { "host": "sandbox" }
        }
      },
```

- [ ] **Étape 2 : Valider le JSON**

```bash
python3 -m json.tool < openclaw-config/openclaw-slm.json.template > /dev/null && echo "JSON valide"
```

Attendu : `JSON valide` (aucune erreur).

- [ ] **Étape 3 : Committer**

```bash
git add openclaw-config/openclaw-slm.json.template
git commit -m "feat(tiron-image): câbler l'agent main (image tiron, XDG_CONFIG_HOME, exec.host=sandbox)"
```

---

### Task 3 : Mise à jour de la config live et redémarrage du gateway

**Fichiers :**
- Modifier : `~/.openclaw-slm/openclaw.json` (non versionné — config live)

- [ ] **Étape 1 : Sauvegarder la config live**

```bash
cp ~/.openclaw-slm/openclaw.json ~/.openclaw-slm/openclaw.json.bak
```

- [ ] **Étape 2 : Appliquer la même modification que dans le template**

Dans `~/.openclaw-slm/openclaw.json`, remplacer :

```json
      { "id": "main" },
```

par :

```json
      {
        "id": "main",
        "sandbox": {
          "docker": {
            "image": "secretarius-tiron:latest",
            "env": {
              "XDG_CONFIG_HOME": "/agent/.gog-config"
            }
          }
        },
        "tools": {
          "exec": { "host": "sandbox" }
        }
      },
```

- [ ] **Étape 3 : Valider le JSON de la config live**

```bash
python3 -m json.tool < ~/.openclaw-slm/openclaw.json > /dev/null && echo "JSON valide"
```

Attendu : `JSON valide`.

- [ ] **Étape 4 : Redémarrer le gateway slm (confirmation utilisateur requise)**

```bash
systemctl --user restart openclaw-gateway-slm.service
sleep 3
systemctl --user status openclaw-gateway-slm.service | head -5
```

Attendu : `active (running)`.

- [ ] **Étape 5 : Vérifier que l'agent `main` est enregistré avec la bonne image**

```bash
openclaw --profile slm agents list --json
```

Attendu : tableau JSON contenant `"main"` et `"wiki"`.

```bash
openclaw --profile slm config get agents.list --json 2>/dev/null | python3 -c "
import json, sys
agents = json.load(sys.stdin)
for a in agents:
    if a.get('id') == 'main':
        print(json.dumps(a, indent=2))
" 2>/dev/null || echo "(commande config get non disponible — vérification manuelle dans openclaw.json)"
```

---

### Task 4 : Test d'isolation au niveau Docker (sans OpenClaw)

*Pas de fichiers modifiés — tests de validation uniquement.*

Ces tests vérifient le comportement de l'image directement, indépendamment d'OpenClaw. Ils reprennent les résultats de l'étape 8-9 de la Task 1 sous une forme plus explicite, pour les consigner.

- [ ] **Étape 1 : Confirmer que `secretarius-tiron:latest` contient `gog`**

```bash
docker run --rm secretarius-tiron:latest which gog
```

Attendu : `/usr/local/bin/gog`

```bash
docker run --rm secretarius-tiron:latest gog --version
```

Attendu : `v0.9.0 (99d9575 ...)`

- [ ] **Étape 2 : Confirmer que `XDG_CONFIG_HOME` est bien exposée au conteneur**

```bash
docker run --rm -e XDG_CONFIG_HOME=/agent/.gog-config secretarius-tiron:latest \
  sh -c 'gog auth status --json 2>&1 | head -20'
```

Attendu : JSON avec `"credentials_path": "/agent/.gog-config/gogcli/credentials.json"` — confirme que `gog` respecte `XDG_CONFIG_HOME` dans ce contexte.

- [ ] **Étape 3 : Confirmer que `openclaw-sandbox:bookworm-slim` ne contient pas `gog`**

```bash
docker run --rm openclaw-sandbox:bookworm-slim gog --version 2>&1
```

Attendu : `sh: 1: gog: not found` (ou équivalent selon le shell de base).

- [ ] **Étape 4 : Committer (pas de fichier modifié — commit de documentation uniquement si notes à conserver)**

Aucun commit requis à cette étape — les résultats seront consignés dans la Task 7.

---

### Task 5 : Bootstrap des credentials gog ⚠ MANUELLE ET INTERACTIVE

*Cette task ne peut pas être déléguée à un sous-agent. Elle requiert la présence de l'utilisateur pour le flow OAuth (navigateur + copier-coller d'URL).*

**Contexte :** l'agent `main` démarre maintenant avec son propre sandbox, `XDG_CONFIG_HOME=/agent/.gog-config`. À ce stade, `/agent/.gog-config/gogcli/` est vide — `gog` n'a ni credentials d'application ni token de compte. Cette task crée les deux via un flow OAuth `--manual` (sans navigateur dans le conteneur).

**Prérequis :** le gateway slm est redémarré (Task 3), `main` est actif avec la nouvelle image.

*Toutes les commandes ci-dessous sont à saisir dans une session OpenClaw avec l'agent `main` (interface web à l'adresse `http://sanroque:18790` ou `http://localhost:18790`), en demandant à `main` d'exécuter ces commandes via son outil `exec`.*

- [ ] **Étape 1 : Vérifier que `XDG_CONFIG_HOME` est active dans le sandbox**

Demander à `main` d'exécuter :

```bash
echo $XDG_CONFIG_HOME
```

Attendu : `/agent/.gog-config`

- [ ] **Étape 2 : Vérifier l'état initial (pas encore de credentials)**

Demander à `main` d'exécuter :

```bash
gog auth status --json
```

Attendu : JSON avec `"credentials_exists": false`.

- [ ] **Étape 3 : Enregistrer les credentials de l'application OAuth**

Les valeurs sont les credentials OAuth de l'application Google Cloud (identifiants d'app, pas de compte) — visibles dans `~/.config/gogcli/credentials.json` sur l'hôte :

- `client_id` : `<valeur>`
- `client_secret` : `<valeur>`

Demander à `main` d'exécuter :

```bash
echo '{"client_id":"<valeur>","client_secret":"<valeur>"}' | gog auth credentials set -
```

Attendu : confirmation `Credentials stored` (ou similaire) sans erreur.

- [ ] **Étape 4 : Lancer le flow d'autorisation OAuth (--manual)**

Demander à `main` d'exécuter :

```bash
gog auth add cmauceri@gmail.com --manual --services gmail,calendar,drive
```

`gog` affiche une URL d'autorisation Google. **Sur l'hôte (ou n'importe quel navigateur)** :
1. Ouvrir l'URL affichée
2. Autoriser l'application avec le compte `cmauceri@gmail.com`
3. Google redirige vers `localhost:...?code=...` — copier l'URL complète de redirection
4. La coller en réponse à `gog` dans la session `main`

Attendu : `gog` confirme l'autorisation et stocke le refresh token.

- [ ] **Étape 5 : Vérifier l'état post-bootstrap**

Demander à `main` d'exécuter :

```bash
gog auth status --json
```

Attendu : JSON avec `"credentials_exists": true`, `"credentials_path": "/agent/.gog-config/gogcli/credentials.json"`, et le compte `cmauceri@gmail.com` listé.

---

### Task 6 : Validation — persistance et test fonctionnel

*Tests en deux temps : persistance après recréation du conteneur, puis commande gog réelle.*

- [ ] **Étape 1 : Tester la persistance — terminer la session courante**

Dans l'interface OpenClaw : fermer la session avec `main` (ou démarrer une nouvelle session avec `main`). Cela force la recréation du conteneur sandbox (`scope: session`).

- [ ] **Étape 2 : Vérifier que les credentials persistent dans la nouvelle session**

Dans la nouvelle session avec `main`, demander d'exécuter :

```bash
gog auth status --json
```

Attendu : même résultat qu'à l'étape 5 de la Task 5 — credentials toujours présents, malgré la recréation du conteneur. Preuve que le stockage dans `/agent/.gog-config/` (workspace) survit au cycle de vie du conteneur.

- [ ] **Étape 3 : Test fonctionnel — commande gog réelle**

Demander à `main` d'exécuter :

```bash
gog gmail search 'newer_than:1d' --max 1 --account cmauceri@gmail.com --json
```

Attendu : réponse JSON de l'API Gmail (liste de messages/threads), sans erreur d'authentification. Confirme que l'identité gog isolée fonctionne contre l'API Google réelle.

---

### Task 7 : Consigner les résultats dans le spec et committer

**Fichiers :**
- Modifier : `docs/superpowers/specs/2026-06-08-image-agent-tiron-design.md` (section §7 Validation)

- [ ] **Étape 1 : Ajouter les résultats dans le spec (§7)**

Ajouter une sous-section `### Résultats (exécutés le 2026-06-09)` après le tableau de validation dans `docs/superpowers/specs/2026-06-08-image-agent-tiron-design.md`, avec le tableau suivant rempli selon les observations réelles :

```markdown
### Résultats (exécutés le 2026-06-09)

| Test | Résultat |
|---|---|
| Build de l'image (`secretarius-tiron:latest`) | ✅ X Mo |
| `gog --version` dans l'image tiron | ✅ v0.9.0 |
| Isolation — `gog` absent de `openclaw-sandbox:bookworm-slim` | ✅ `gog: not found` |
| Bootstrap credentials — `gog auth status` montre credentials sous `/agent/.gog-config/` | ✅ |
| Persistance — credentials présents après recréation du conteneur | ✅ |
| Fonctionnel — `gog gmail search` retourne des résultats | ✅ |
```

(Remplacer `X Mo` par la taille réelle observée à la Task 1 étape 7.)

- [ ] **Étape 2 : Committer**

```bash
git add docs/superpowers/specs/2026-06-08-image-agent-tiron-design.md
git commit -m "docs(tiron-image): consigner les résultats de validation de l'image agent Tiron"
```
