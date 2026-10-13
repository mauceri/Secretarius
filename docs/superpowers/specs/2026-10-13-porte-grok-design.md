# Porte Grok : accès public au wiki et au coffre de santiago — conception

Statut : à relire (2026-10-13). fastmcp 3.4.2 déjà présent dans le venv de Wiki_LM. Suite de la spec `2026-10-12-installateur-profils-design.md`
(découpage : 1 installateur à profils, 2 profil léger sur santiago — faits ; **3 porte
Funnel + fiche coordinateur Grok** — ce document ; 4 profil lourd sur VPS de test).

## Objectif

Les agents de Grok Bot (équipe de publication : YouTube, Substack, LinkedIn, site) tournent
dans le nuage et appellent leurs outils depuis les serveurs de Grok, pas depuis vos
appareils. Ils doivent pouvoir :

- **lire le wiki de santiago** : poser une question, chercher par mots-clés, lire une page ;
- **lire et écrire des fichiers Markdown dans le coffre `Publication`** (brouillons, notes de
  veille).

Rien de ce qu'ils produisent n'entre dans le wiki sans vous : ils déposent des fichiers,
vous validez dans Obsidian ce que vous retenez, et c'est votre validation qui déclenche la
capture.

**Succès** : depuis Grok Bot, un agent interroge le wiki de santiago et dépose une note de
veille ; vous la voyez dans Obsidian, cochez un élément, lancez la commande du plugin, et
l'élément est capturé puis ingéré.

## Décisions (avec l'utilisateur, 2026-10-13)

- Porte publique = **Tailscale Funnel** sur santiago, port **8443** (443 reste au tailnet).
  Funnel n'authentifie personne : l'authentification est applicative.
- Authentification par **en-tête `Authorization: Bearer`**, **un jeton par agent**, révocable
  séparément (recommandation de Grok Bot ; jeton jamais dans l'URL, qui finit dans les
  journaux). Grok déclare la porte comme **connecteur MCP** (en-tête Bearer fixe).
- **Pas de capture directe** par les agents : le serveur wiki ne change pas. Les agents
  écrivent des fichiers, vous validez.
- Écriture : **écrasement autorisé**, mais le serveur impose que le nom du fichier commence
  par le **préfixe de l'agent** (déduit de son jeton) — un agent ne peut ni par erreur ni
  volontairement écraser une de vos notes.
- Validation de la veille par **cases à cocher** devant les blocs ` ```wiki ` dans le plugin
  Obsidian `wikilm-capture`.
- Architecture : **serveur MCP séparé** (option 1), et non MCP greffé sur `wiki-lm-server`
  (qui resterait local, protections pensées pour Obsidian) ni simple API HTTP.

## Architecture

```
Agents Grok (nuage) ──HTTPS Bearer──► Funnel santiago:8443 ──► 127.0.0.1:5052 porte-grok (MCP)
                                                                  ├─► wiki-lm-server 127.0.0.1:5051 (/query, /run /r)
                                                                  ├─► lecture Wiki_LM/wiki/**/<slug>.md
                                                                  └─► lecture/écriture ~/Documents/Publication/**/*.md
```

### Service `porte-grok`

- Code : `Wiki_LM/porte/` (serveur `fastmcp`, transport HTTP « streamable », chemin `/mcp`),
  lancé avec le venv de Wiki_LM.
- Écoute **uniquement** sur `127.0.0.1:5052`. Unité systemd utilisateur `porte-grok.service`
  (`Restart=always`).
- Exposition : `tailscale funnel --bg --https=8443 http://127.0.0.1:5052`. Adresse publique :
  `https://santiago.<tailnet>.ts.net:8443/mcp`.
- Prérequis ponctuels, à la charge de l'utilisateur : autoriser Funnel pour santiago dans la
  console Tailscale (attribut `funnel`), et `sudo tailscale set --operator=<utilisateur>` pour
  que la commande `tailscale funnel` ne demande pas sudo.

### Jetons

- Fichier `~/.config/porte-grok/agents` (droits 600), une ligne par agent :
  `<préfixe>:<empreinte SHA-256 hexadécimale du jeton>`. Le jeton en clair n'est jamais stocké.
- Préfixe : `[a-z][a-z0-9-]{1,30}` (ex. `veille`, `redaction`, `coordinateur`).
- Commande `porte-jeton` (dans `Wiki_LM/porte/`, enveloppe dans `~/.local/bin/`) :
  - `porte-jeton ajouter <préfixe>` : génère un jeton aléatoire (`secrets.token_urlsafe(32)`),
    l'affiche **une seule fois**, enregistre son empreinte ; refuse un préfixe déjà présent ;
  - `porte-jeton retirer <préfixe>` : supprime la ligne ;
  - `porte-jeton lister` : préfixes seulement.
- Le serveur relit le fichier à chaque requête (quelques lignes ; une révocation prend effet
  immédiatement, sans redémarrage). Comparaison d'empreintes en temps constant.
- Requête sans en-tête, jeton inconnu ou révoqué → **401**, aucun outil exécuté.

### Outils MCP

| Outil | Paramètres | Comportement |
|---|---|---|
| `wiki_question` | `question` | `POST http://127.0.0.1:5051/query` ; renvoie la synthèse et les slugs cités. **Une seule question à la fois** (verrou ; les suivantes attendent). Délai : 6 min. |
| `wiki_recherche` | `mots` | `POST /run` `{"command": "/r", "arg": mots}` ; renvoie titres et extraits (5 au plus). Les slugs pour `wiki_page` viennent des références de `wiki_question`. |
| `wiki_page` | `slug` | Lit `Wiki_LM/wiki/**/<slug>.md`. Slug : `^(src|c|e)-[a-z0-9-]+$`, sinon refus. |
| `coffre_lister` | `dossier` (facultatif, défaut racine) | Liste les `.md` sous `dossier`, récursivement, hors zones interdites. |
| `coffre_lire` | `chemin` | Lit un `.md` du coffre. |
| `coffre_ecrire` | `chemin`, `contenu` | Crée ou écrase ; crée les dossiers intermédiaires. Nom de fichier (dernier composant) obligatoirement préfixé `<préfixe>-`. |

### Confinement des fichiers (outils `coffre_*`)

- Chemin **relatif** au coffre (`~/Documents/<COFFRE>`, `Publication` sur santiago) ; un chemin
  absolu est refusé.
- Résolution (`realpath`) puis vérification que le résultat est **dans** le coffre : `..` et
  liens symboliques sortants refusés.
- **Zones interdites** : `Wiki_LM/` et `.obsidian/` (et tout composant commençant par `.`).
- Extension **`.md` seulement** ; contenu écrit ≤ **1 Mo** (UTF-8).
- Pas de suppression, pas de renommage.

### Erreurs

Chaque refus ou échec renvoie à l'agent un message court et explicite (« chemin hors du
coffre », « le nom doit commencer par veille- », « page introuvable », « wiki injoignable »),
sous forme d'erreur d'outil MCP. Jamais de trace Python vers l'extérieur ; la trace va au
journal.

### Journal

Une ligne par appel dans le journal systemd de l'unité : horodatage, préfixe de l'agent,
outil, chemin ou slug éventuel, résultat (ok / motif du refus). **Jamais** le contenu des
fichiers ni le texte des questions au-delà de 80 caractères.

### Ce que la porte ne peut pas faire

Supprimer, renommer, capturer, ingérer, lancer une commande `/run` autre que `/r`, lire ou
écrire hors du coffre, lire `Wiki_LM/` autrement que par `wiki_page`. Elle ne va jamais
chercher une URL : la contrainte « le wiki ne court-circuite pas Scout » est respectée, toute
URL entre par votre validation puis la chaîne d'ingestion habituelle.

## Cases à cocher dans `wikilm-capture`

Seule `extractWikiBlocks` (`Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts`) change :
on regarde la **dernière ligne non vide avant** chaque bloc ` ```wiki ` :

- `- [ ] …` → bloc **ignoré** ;
- `- [x] …` ou `- [X] …` → bloc exécuté ;
- toute autre ligne (ou aucune) → bloc exécuté, comme aujourd'hui (compatibilité).

Le plugin reconstruit est redéployé dans les coffres qui l'utilisent ; dans `Publication`,
la configuration n'est pas synchronisée (« Configs: none »), il faut l'y déposer.

Format d'une note de veille (imposé par la fiche du coordinateur) :

````markdown
## <titre> (<source>, <date>)
Résumé : …

- [ ] Retenir
```wiki
/c
#tag1 #tag2 <url>
```
````

## Installateur

- Nouveau module **`porte`** dans `openclaw-config/modules.sh` : installe
  `porte-grok.service` (unité utilisateur, `enable --now`), l'enveloppe `porte-jeton`,
  crée `~/.config/porte-grok/` (700) et le fichier `agents` vide (600) s'il manque ; ajoute
  `porte-grok|http://127.0.0.1:5052/health` à `SERVICES_A_VERIFIER`.
- Funnel : si `tailscale funnel status` ne montre pas le port 8443, le module **n'échoue
  pas** ; il ajoute aux étapes de compte du résumé final : autorisation Funnel dans la
  console, `sudo tailscale set --operator=…`, puis
  `tailscale funnel --bg --https=8443 http://127.0.0.1:5052`.
- `porte` ajouté à `MODULES` du profil **léger** ; absent du profil lourd.
- `uninstall.sh` retire `porte-grok.service` (pas le dossier des jetons).
- `/health` (sans jeton) : `{"status": "ok"}`, pour la vérification finale locale ; ne révèle
  rien d'autre. Sur l'adresse publique, il est accessible aussi, ce qui est sans risque.

## Fiche du coordinateur Grok

Document `docs/grok/fiche-coordinateur.md`, en français, à coller dans les consignes de
l'agent coordinateur. Contenu :

- adresse du connecteur et déclaration (en-tête Bearer ; un jeton par agent, demandé à
  l'utilisateur ; ne jamais mettre un jeton dans une URL ni dans un fichier du coffre) ;
- les 6 outils, chacun avec un exemple d'appel et de réponse ;
- règles d'écriture : préfixe imposé, `.md` seulement, zones interdites, taille maximale,
  écrasement possible de ses propres fichiers seulement ;
- format des notes de veille (ci-dessus) : toutes les cases **vides**, c'est l'utilisateur
  qui coche ;
- ce que les agents ne peuvent pas faire et pourquoi (tout ce qui entre dans le wiki passe
  par l'utilisateur) ;
- conseils : `wiki_recherche` avant `wiki_question` (gratuite, instantanée) ; une question
  peut prendre ~1 min au réveil du modèle.

## Tests

- **Porte** (`Wiki_LM/tests/test_porte.py`, coffre temporaire, faux `wiki-lm-server`) :
  sans en-tête / jeton faux / jeton révoqué → 401 ; confinement : chemin absolu, `..`, lien
  symbolique sortant, `Wiki_LM/`, `.obsidian/`, extension autre que `.md`, contenu > 1 Mo ;
  préfixe imposé (refus et succès, écrasement d'un fichier préfixé) ; slug invalide ; page
  introuvable ; wiki injoignable → message clair ; deux `wiki_question` simultanées →
  exécutées l'une après l'autre ; `porte-jeton` ajouter / refus doublon / retirer / lister,
  empreinte seule dans le fichier.
- **Plugin** (`run-commands.test.ts`) : case vide ignorée, case cochée exécutée (`x` et
  `X`), pas de case exécutée.
- **Installateur** (`tests_installation/`, `HOME` temporaire) : module `porte` présent dans
  le profil léger, absent du lourd ; Funnel absent → étape de compte, pas d'échec.

## Vérification sur santiago

1. `curl` sur l'adresse publique sans jeton → 401 ; avec jeton → liste des 6 outils.
2. Depuis Grok Bot (connecteur ajouté avec le jeton `veille`) : une question au wiki, puis
   dépôt d'une note de veille ; elle apparaît dans Obsidian sur vos appareils.
3. Vous cochez un élément et lancez « Exécuter les commandes wiki de la note » : capture
   puis ingestion de cet élément seulement.
4. `porte-jeton retirer veille` → l'agent reçoit 401 à l'appel suivant.

## Hors périmètre

OAuth ; suppression ou renommage par les agents ; capture directe ; accès aux coffres
Secretarius et Arbath (sanroque) ; limitation de débit au-delà du verrou de `wiki_question` ;
profil lourd.
