---
tags: [documentation, LLM_Wiki, secretarius]
date: 2026-05-14
---

# Composant : obsidian

## Rôle

Obsidian est l'IDE du wiki Secretarius. Il visualise le graphe de liens entre pages,
permet la navigation et l'édition des notes, et synchronise le coffre via
`obsidian-headless` (sync headless sans interface graphique).

## Prérequis

- Compte Obsidian Sync (payant)
- Node.js (pour `obsidian-headless`)
- `npm install -g obsidian-headless` ou `npx obsidian-headless`

## Installation

### Obsidian headless sync

```bash
# Connexion au compte Obsidian
npx obsidian-headless login

# Lister les vaults distants
npx obsidian-headless sync-list-remote

# Configurer le vault local (remplacer "Mon Vault" par le nom exact)
npx obsidian-headless sync-setup \
  --path ~/Documents/Arbath \
  --remote "Mon Vault"

# Première synchronisation
npx obsidian-headless sync --path ~/Documents/Arbath
```

### Configuration Secretarius

Le chemin du coffre (`OBSIDIAN_PATH`) est configuré dans `install.conf` et propagé
dans `openclaw.json` via `envsubst`. Les outils Wiki_LM utilisent
`WIKI_PATH = ${OBSIDIAN_PATH}/Wiki_LM`.

## Désinstallation

```bash
npx obsidian-headless logout
npm uninstall -g obsidian-headless
```

## Usage courant

```bash
# Synchroniser avant de travailler
npx obsidian-headless sync --path ~/Documents/Arbath

# Vérifier l'état
npx obsidian-headless sync-status --path ~/Documents/Arbath

# Lister les fichiers synchronisés
npx obsidian-headless sync-list-local
```

## Pièges connus

- **Les réglages de synchronisation sont par coffre, jamais globaux.** Que ce
  soit `ob sync-config --configs ...` sur une machine headless (sanroque) ou
  le réglage « Sync » dans l'app Obsidian elle-même (tablette, desktop) : une
  catégorie activée (ex. « Installed community plugins ») pour un coffre ne
  s'applique **pas** automatiquement aux autres coffres du même compte, ni
  aux autres appareils. Un plugin qui apparaît bien envoyé côté serveur
  (visible dans le `sync.log` local : `Upload complete .../main.js`) mais
  absent des modules installés sur un autre appareil : vérifier ce réglage
  côté **réception**, coffre par coffre et appareil par appareil — pas
  seulement côté émission. Rencontré deux fois (Secretarius, puis Arbath,
  2026-09).
- **sanroque est headless** : pas d'application Obsidian graphique dessus,
  uniquement le CLI `ob` (`obsidian-headless`). Toute panne de synchro ou de
  plugin sur sanroque se diagnostique via `ob sync-status` / `ob sync-config`
  et les logs sous `~/.config/obsidian-headless/sync/<vaultId>/`, jamais via
  une interface graphique locale.
- **La synchro peut se déclarer « réussie » sans avoir réellement écrasé un
  fichier local divergent.** Observé avec le client Obsidian Sync natif
  (desktop, poste Ubuntu) sur un `main.js` de plugin déjà présent localement
  dans une version différente : le journal de sync affiche bien
  `Downloading complete` puis `Accepted` pour ce fichier, mais le contenu sur
  disque ne change pas — ni en relançant la synchro, ni en redémarrant
  Obsidian complètement (2026-09-16). Le correctif qui a fonctionné :
  **désinstaller le plugin depuis Obsidian** (supprime le dossier local ; la
  suppression se propage par sync aux autres appareils, y compris à
  sanroque) **puis redéployer manuellement les fichiers à jour** — plutôt que
  de compter sur la sync pour écraser un fichier existant qui diverge.
- **Une icône absente de la barre latérale ne veut pas dire que la commande
  a disparu.** Obsidian peut réordonner/replier les icônes de la barre sans
  prévenir ; avant de conclure à un problème, vérifier la commande dans la
  palette (Ctrl/Cmd-P) — si elle y est, ce n'est qu'un déplacement d'icône,
  pas un bug.
- **`ob sync --continuous` ne détecte pas toujours une modification en
  place d'un fichier déjà connu.** Observé sur sanroque (2026-09-17) : un
  `main.js` de plugin modifié sur disque, avec la synchro déjà active
  depuis avant l'édition, n'est jamais réémis — le processus continue
  d'annoncer « Fully synced » indéfiniment, y compris sur d'autres fichiers
  modifiés entre-temps (donc pas une panne totale, seulement ce fichier).
  Un simple redémarrage (Ctrl-C puis relancer `ob sync --continuous` dans
  la session tmux du coffre) force une comparaison complète et déclenche
  l'envoi immédiatement. Vérifier après tout déploiement de fichier vers un
  coffre déjà sous synchro continue : `grep <fichier> sync.log` doit
  montrer un envoi **après** l'heure de la modification, pas seulement une
  entrée ancienne.
- **Une session `ob sync --continuous` peut se déconnecter silencieusement
  et ne jamais se relancer.** La session tmux reste vivante (juste un shell
  inactif), mais plus aucun contenu ne part ni n'arrive — repéré ici après
  cinq jours d'inactivité (`Disconnected from server` dans le pane tmux,
  aucune unité systemd pour redémarrer automatiquement, voir aussi le point
  en suspens correspondant dans la synthèse). Vérifier périodiquement
  `tmux capture-pane -t <session> -p | tail` pour chaque coffre.

## Template de requête Wiki_LM (Templater)

Interroger le wiki en langage naturel **directement depuis Obsidian** (desktop ou
Android) : la réponse s'ouvre dans un **nouvel onglet** (note d'historique
horodatée), jamais insérée dans la note en cours. Le template appelle le
serveur `wiki-lm-server` (port 5051, voir `docs/components/wiki-lm.md`), qui
écrit lui-même cette note dans `Wiki_LM/historique/`. Fichier source :
`Wiki_LM/obsidian_template_wikilm_android.md`.

### Prérequis

- Service `wiki-lm-server` actif sur sanroque (`systemctl --user status wiki-lm-server`).
- L'appareil Obsidian atteint `sanroque` via le **tailnet** uniquement — le
  serveur n'écoute plus que sur la boucle locale (`--no-public`), publié par
  `tailscale serve` ;
  tester : `curl https://sanroque.tailc69141.ts.net:10443/health`.
- Plugin communautaire **Templater** installé et activé
  (Paramètres → Modules complémentaires → Templater).

### Installation dans Obsidian

1. Paramètres → Templater → **Template folder location** : choisir un dossier du
   coffre (p. ex. `Templates`).
2. Copier `obsidian_template_wikilm_android.md` dans ce dossier (p. ex.
   `Templates/Wiki_LM Query.md`). Le coffre étant synchronisé, le fichier est déjà
   présent sous `Wiki_LM/` ; il suffit de le copier dans le dossier de templates.
3. (Optionnel) Raccourci : Paramètres → Templater → **Template Hotkeys** → ajouter
   le template et lui affecter un raccourci.

### Utilisation

1. Ouvrir n'importe quelle note (son contenu ne sera pas modifié).
2. Lancer le template : via le raccourci, ou Ctrl/Cmd-P → « Templater: Open Insert
   Template modal » → choisir le template.
3. Choisir le **mode** (Hybride recommandé / Sémantique / BM25).
4. Saisir la **question** → la réponse s'ouvre dans un nouvel onglet (rien
   n'est inséré dans la note en cours).

### Dépannage

- « Wiki_LM : erreur — … » : serveur injoignable → vérifier
  `curl https://sanroque.tailc69141.ts.net:10443/health` depuis l'appareil
  (nécessite d'être sur le tailnet).
- Le template utilise **`requestUrl`** (API Obsidian), pas `fetch()`, pour
  contourner le CSP d'Electron — ne pas revenir à `fetch()`.
- « Aucune information » sur un document récent : normalement résolu par
  l'auto-reload du serveur ; sinon forcer, depuis sanroque,
  `curl -X POST http://127.0.0.1:5051/reload`.

## Plugin de capture Wiki_LM

Capturer la note actuellement ouverte dans la file `raw/` de Wiki_LM (équivalent
de `/c`, sans quitter Obsidian), depuis desktop ou mobile. Source :
`Wiki_LM/obsidian-wikilm-capture/` (projet TypeScript séparé).

### Prérequis

- Service `wiki-lm-server` actif sur sanroque (le plugin appelle le nouvel
  endpoint `POST /capture`, voir `docs/components/wiki-lm.md`).
- L'appareil Obsidian atteint `sanroque` via le tailnet (voir plus haut —
  le serveur n'est plus accessible en LAN direct).

### Installation

1. `cd Wiki_LM/obsidian-wikilm-capture && npm install && npm run build` — produit
   `main.js` à côté de `manifest.json`.
2. Copier `manifest.json` et `main.js` dans
   `<coffre>/.obsidian/plugins/wikilm-capture/`.
3. Dans Obsidian : Paramètres → Modules communautaires → activer
   « Wiki_LM Capture ».
4. Dans les réglages du plugin, renseigner l'URL du serveur (ex.
   `https://sanroque.tailc69141.ts.net:10443`) — c'est déjà la valeur par
   défaut du plugin.

### Utilisation

1. Ouvrir la note à capturer.
2. Cliquer l'icône dans la barre latérale, ou lancer la commande
   « Capturer la note courante dans Wiki_LM » (Ctrl/Cmd-P).
3. Si la note commence par un titre `# Résumé` ou `# Summary`, cette section
   est capturée intégralement ; sinon, les 200 premiers caractères de la note
   sont utilisés comme aperçu.
4. Une notification confirme la capture ; la note est marquée
   `wiki_capture: <date>` dans son frontmatter.

## Exécution en lot de commandes wiki

Exécuter une liste de commandes `/` Wiki_LM (`/q`, `/c`, `/tags`, `/ingest`,
`/wikistatus`, `/r`, `/kbupdate`, `/relire`, `/verifie` — `/supprimer` est
volontairement exclue) depuis une seule note Obsidian, en une action, avec les
résultats insérés directement dans la note. Utilise le nouvel endpoint
`POST /run` du même `wiki-lm-server` (voir `docs/components/wiki-lm.md`) —
même prérequis que le plugin de capture : le service doit tourner et être
joignable.

### Utilisation

1. Dans la note, ajouter un ou plusieurs blocs de code ` ```wiki ` : la
   première ligne est la commande (ex. `/q`), les lignes suivantes (s'il y en
   a) forment l'argument, conservé tel quel (y compris sur plusieurs lignes).
2. Cliquer l'icône « Exécuter les commandes wiki de la note » dans la barre
   latérale, ou lancer la commande du même nom (Ctrl/Cmd-P) — distincte de
   « Capturer la note courante dans Wiki_LM ».
3. Le résultat de chaque bloc est inséré juste après, entre des marqueurs
   `<!-- wikilm-run:start -->` / `<!-- wikilm-run:end -->` ; relancer sur la
   même note remplace le résultat précédent au lieu de le dupliquer.
4. Si une commande échoue, son message d'erreur est inséré à sa place et les
   blocs suivants s'exécutent quand même — un échec n'interrompt pas le lot.

## Archivage du coffre

Il est fortement recommandé d'archiver régulièrement le coffre :

```bash
# Archive complète
tar -cf ~/sauvegarde_obsidian_$(date +%Y%m%d).tar \
  -C "$(dirname ~/Documents/Arbath)" \
  "$(basename ~/Documents/Arbath)"

# Ou via le skill archivage-obsidian dans OpenClaw :
# "archiver le coffre"
```

## Notes d'architecture

Le coffre Obsidian est le répertoire racine de toutes les données Secretarius :
`Wiki_LM/wiki/`, `Wiki_LM/raw/`, etc. Obsidian offre une vue graphique des
liens internes (`[[slug]]`) qui matérialisent les connexions du patron LLM Wiki.
Ne pas modifier le dossier `.obsidian/` (config interne du vault).
