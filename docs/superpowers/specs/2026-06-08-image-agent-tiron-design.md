# Spec — Image Docker de l'agent Tiron / `main` (étape B, second sous-projet)

> Session superpowers du 2026-06-08, suite de la session image agent wiki
> (`docs/superpowers/specs/2026-06-08-image-agent-wiki-design.md`).
> Périmètre : conception de l'image spécialisée de l'agent `main` (Tiron),
> embarquant le binaire `gog`, sur l'instance isolée slm uniquement
> (`~/.openclaw-slm/`, port 18790). La prod 6.1 (`~/.openclaw/`, port 18789)
> n'est pas touchée. La réécriture du contenu du skill `gog` (actuellement
> obsolète — référence des outils MCP `gog__*` qui n'existent plus dans cette
> architecture) reste hors périmètre, comme le contenu du skill wiki l'était
> pour la session précédente.

---

## 1. Contexte et objectif

L'architecture cible (`docs/architecture/secretarius-slm-architecture.md`)
isole les agents par le **contenu de leur image Docker** : ce qu'un agent peut
faire dépend des binaires présents dans son image
(`agents.list[].sandbox.docker.image`, `tools.exec.host = "sandbox"`). Décision
prise à l'étape A : **gog reste chez Tiron** — binaire et skill dans l'image de
`main`, pas de sous-agent `gog` dédié.

Aujourd'hui, `main` exécute `gog` via `tools.exec.host = "gateway"` (réglage
global), avec accès direct au système hôte — y compris au keyring système
(`~/.config/gogcli/`). Pour réaliser l'objectif d'isolation par image
(« exec tourne *dans* le conteneur »), `main` doit basculer sur
`tools.exec.host = "sandbox"` avec une image embarquant le binaire `gog`.

Cette session conçoit cette image et le câblage associé — en particulier la
question, restée ouverte à l'étape A, de l'accès aux credentials OAuth gog
depuis un conteneur sandboxé qui n'a plus accès au keyring de l'hôte.

---

## 2. Contenu de l'image

### 2.1 Embarqué dans l'image (figé, change uniquement au rebuild)

- Le binaire `gog` (`/home/linuxbrew/.linuxbrew/Cellar/gogcli/0.9.0/bin/gog`,
  21 Mo) — **statiquement lié, sans dépendances dynamiques** (vérifié via
  `file` : `ELF 64-bit LSB executable ... statically linked, stripped`, et
  `ldd` : `n'est pas un exécutable dynamique`). Aucun runtime à installer,
  aucune dépendance à résoudre.

C'est tout. Contrairement à l'image wiki (torch, transformers, poids BGE-M3,
~10,5 Go), l'image Tiron n'a besoin d'aucune dépendance ML ni runtime
supplémentaire — un simple `COPY` suffit.

### 2.2 Rien à monter en bind

Aucune donnée ni code vivant à exposer pour cette image (contrairement au
wiki : pas de `query.py`, pas de base de connaissances). Les credentials gog
ne sont *pas* montés non plus — voir §4.

---

## 3. Image de base

`FROM openclaw-sandbox:bookworm-slim` — même base que l'image wiki et que le
Pilier A. Image Debian Bookworm slim (211 Mo), `python3` déjà présent (requis
par les helpers d'écriture/édition du sandbox), `bash`/`cat`/`ls`/`find`
disponibles.

---

## 4. Credentials gog — identité OAuth isolée, zéro bind

### 4.1 Le bind direct est techniquement impossible (et ce n'est pas un détail)

À l'étape A, l'option retenue était un bind mount de
`~/.config/gogcli/credentials.json` (mode `:rw`/`:ro` à trancher selon que gog
réécrit son token à l'usage). **Cette option s'est révélée inapplicable** :
le code de validation des binds OpenClaw (`validate-sandbox-security.ts`)
bloque *sans aucun override possible* toute source sous :

```js
const BLOCKED_HOME_SUBPATHS = [
  ".aws", ".cargo", ".config", ".docker",
  ".gnupg", ".netrc", ".npm", ".ssh"
];
```

Ce blocage est **distinct et antérieur** à celui contourné par
`dangerouslyAllowExternalBindSources` (qui ne gère que les sources « hors des
racines autorisées »). Pour `BLOCKED_HOME_SUBPATHS`, l'erreur est levée avant
toute vérification de flag :

> *« Mounting system directories, credential paths, or Docker socket paths
> into sandbox containers is not allowed. Use project-specific paths
> instead »*

`~/.config/gogcli/` est sous `~/.config` → bloqué, sans recours. La résolution
suit aussi les liens symboliques (`resolveSandboxHostPathViaExistingAncestor`),
donc un symlink projet pointant vers `~/.config/gogcli` serait également
détecté et bloqué.

**Pourquoi ce blocage existe** (commentaire du code source — *« Threat model:
local-trusted config, but protect against foot-guns and config injection »*) :
`~/.config`, `~/.ssh`, `~/.aws`, etc. sont les emplacements conventionnels où
des dizaines d'outils stockent des secrets (clés SSH, credentials cloud,
tokens npm, clés GPG…). Autoriser leur bind exposerait potentiellement
*l'intégralité* des secrets de l'utilisateur à un agent sandboxé — bien
au-delà du seul `gogcli`. Le choix d'OpenClaw est un filet large mais simple :
bloquer toute la racine, et orienter vers le pattern recommandé — *« Use
project-specific paths instead »*.

Vérification empirique complémentaire faite lors de cette investigation (qui
aurait de toute façon nécessité de revoir la décision « pas de copie » de
l'étape A) : `gog auth tokens` gère des **refresh tokens** OAuth2 — modèle
standard où le refresh token est long-lived et stable, et l'access token
court-lived n'est jamais persisté sur disque. Preuve : 8 jours d'usage actif
de `gog-mcp` (logs `journalctl` jusqu'au 2026-06-07) sans la moindre
réécriture du fichier de token (`~/.config/gogcli/keyring/token:...`, mtime
figé au 2026-05-30 18:42). Une copie figée n'aurait donc *pas* été aussi
fragile que redouté à l'étape A — mais l'option est de toute façon fermée par
le garde-fou ci-dessus, et une meilleure solution existe (§4.2).

### 4.2 Solution retenue : identité gog isolée, stockée dans le workspace

Plutôt que de copier ou monter les credentials de l'hôte, le sandbox de `main`
**crée et gère sa propre autorisation OAuth**, totalement indépendante de
celle de l'utilisateur — isolation complète, conforme à l'exigence
« bac à sable pur et dur » exprimée en session.

Deux faits techniques rendent cela possible sans aucun bind :

1. **`gog` respecte la variable standard `XDG_CONFIG_HOME`**
   (`os.UserConfigDir()` de Go) — vérifié en session :
   `XDG_CONFIG_HOME=/tmp/test gog auth status --json` rapporte
   `credentials_path: /tmp/test/gogcli/credentials.json`. En la définissant,
   on redirige entièrement le stockage de config + keyring de `gog`.
2. **Le workspace de l'agent est monté dans le conteneur sandbox à un chemin
   fixe : `/agent`** (constante interne `SANDBOX_AGENT_WORKSPACE_MOUNT`),
   déjà accessible en lecture-écriture (`agents.defaults.sandbox.workspaceAccess
   = "rw"`) sans configuration supplémentaire.

En définissant `XDG_CONFIG_HOME=/agent/.gog-config` dans
`sandbox.docker.env` de `main`, `gog` range sa config et son keyring dans
`/agent/.gog-config/gogcli/` — **dans le workspace**, donc :
- aucun bind à déclarer, aucun `dangerouslyAllowExternalBindSources`,
- aucun chemin bloqué touché,
- persistance garantie à travers les recréations de conteneur
  (`agents.defaults.sandbox.scope = "session"` : un nouveau conteneur est créé
  par session, mais le workspace — et donc `/agent/.gog-config` — survit côté
  hôte),
- isolation totale : aucun secret de l'hôte n'est jamais lu ni exposé au
  conteneur.

### 4.3 Procédure de configuration initiale (manuelle, une fois)

Exécutée dans le conteneur sandbox de `main` (après build de l'image et
premier démarrage avec la config ci-dessous) :

```bash
# 1. Enregistrer les credentials de l'application OAuth (client_id/client_secret —
#    identifiants d'application Google Cloud, pas des secrets utilisateur ;
#    valeurs visibles dans ~/.config/gogcli/credentials.json sur l'hôte,
#    copiées/collées à la main dans cette étape unique)
echo '{"client_id": "<valeur>", "client_secret": "<valeur>"}' | gog auth credentials set -

# 2. Autoriser le compte — flow sans navigateur (« paste redirect URL »)
gog auth add cmauceri@gmail.com --manual --services gmail,calendar,drive
```

`gog` crée alors sa **propre** autorisation OAuth (un grant Google séparé,
révocable indépendamment de celui de l'hôte), stockée uniquement dans
`/agent/.gog-config/gogcli/` — donc dans le workspace de l'agent.

**Note sur les scopes** (`--services`) : à choisir selon les besoins réels de
Tiron — `gog.md` documente `gmail, calendar, drive, contacts, sheets, docs`
pour le compte hôte ; le sandbox Tiron peut se limiter à un sous-ensemble plus
restreint (moins de surface qu'un blanket-grant). Choix exact laissé à
l'implémentation / au moment du bootstrap.

---

## 5. Câblage dans la configuration OpenClaw

Modification de l'entrée `main` dans `agents.list[]` (actuellement
`{ "id": "main" }`, pur défaut) :

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
}
```

- **`sandbox.docker.image`** : pointe vers la nouvelle image, comme pour
  `wiki`.
- **`sandbox.docker.env`** : une seule variable suffit (cf. §4.2). Pas de
  `binds`, pas de `dangerouslyAllowExternalBindSources` — contrairement à
  wiki, aucun bind externe n'est nécessaire ici.
- **`tools.exec.host = "sandbox"`** : override local à `main`, sur le modèle
  déjà utilisé pour `wiki` (« local, sans effet de bord », cf. spec wiki §5).
  Le défaut global `tools.exec.host = "gateway"` reste inchangé.

**Remarque (notée, pas corrigée ici — modification chirurgicale)** : la doc
OpenClaw (`tools/exec.md`) précise que `tools.exec.security`/`safeBins`
(config globale actuelle : `allowlist`, `["gog", "cat", "ls", "find"]`) sont
*« Ignored for normal tool calls »* en mode `host = sandbox` — seuls
`gateway`/`node` les utilisent. Une fois `main` basculé, cette config devient
inopérante pour lui en usage normal (mais reste un filet utile pour le mode
`elevated`/`host = auto` en l'absence de sandbox actif). Aucun changement
proposé : ce n'est pas cassé, et ce n'est pas l'objet de cette session.

---

## 6. Construction et maintenance

- Construction manuelle :
  `docker build -t secretarius-tiron:latest -f openclaw-config/Dockerfile.tiron .`
  (le binaire `gog` doit être copié dans le contexte de build, ou référencé
  via un chemin absolu dans un `COPY --from=` / montage de contexte — détail
  d'implémentation)
- **Rebuild nécessaire** uniquement en cas de mise à jour de la version de
  `gog` (binaire statique embarqué)
- **Pas de bind, pas de code vivant** — rien d'autre à maintenir entre deux
  rebuilds
- Intégration à `install.sh` : différée, même horizon que pour l'image wiki
  (« quand le produit sera prêt à être présenté », décision étape A)

---

## 7. Validation

Reprend le patron de test du Pilier A et de l'image wiki :

| # | Test | Critère de réussite |
|---|---|---|
| 1 | Build de l'image | `secretarius-tiron:latest` construite, taille proche de 211 Mo (base) + 21 Mo (binaire) |
| 2 | Isolation | `main` (nouvelle image) exécute `gog --version` dans son sandbox ; un agent sur l'image de base échoue (`gog: not found`) — même patron que Pilier A et l'isolation BGE-M3 du wiki |
| 3 | Bootstrap credentials | `gog auth credentials set -` puis `gog auth add ... --manual` exécutés dans le sandbox de `main` ; `gog auth status --json` confirme `credentials_path`/`keyring` sous `/agent/.gog-config/gogcli/` |
| 4 | Persistance | Recréer la session/le conteneur (`scope: session` → nouveau conteneur à chaque session) ; `gog auth status` montre toujours l'identité isolée — preuve que le stockage dans le workspace survit au cycle de vie du conteneur |
| 5 | Fonctionnel | Une commande `gog` réelle (ex. `gog gmail search 'newer_than:1d' --max 1 --account cmauceri@gmail.com`) aboutit depuis le sandbox de `main`, prouvant que l'identité isolée fonctionne contre l'API Google |

---

## 8. Hors périmètre (sessions suivantes)

- Réécriture du contenu du skill `gog` (`SKILL.md` actuel référence des
  outils MCP `gog__*` — obsolète depuis la décision « abandon MCP, binaires +
  skills »). À reprendre séparément, comme le contenu du skill wiki.
- Intégration du build d'image à `install.sh`.
- Migration de cette architecture vers la prod 6.1 — différée (décision
  étape A : épingler 5.12, migrer « quand le produit sera prêt à être
  présenté »).
- Choix final des scopes OAuth (`--services`) du grant isolé — à arbitrer au
  moment du bootstrap (§4.3), selon les besoins réels observés de Tiron.
