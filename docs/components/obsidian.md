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

## Template de requête Wiki_LM (Templater)

Interroger le wiki en langage naturel **directement depuis Obsidian** (desktop ou
Android) : la synthèse et les liens `[[source]]` sont insérés dans la note courante.
Le template appelle le serveur `wiki-lm-server` (port 5051, voir
`docs/components/wiki-lm.md`). Fichier source : `Wiki_LM/obsidian_template_wikilm_android.md`.

### Prérequis

- Service `wiki-lm-server` actif sur sanroque (`systemctl --user status wiki-lm-server`).
- L'appareil Obsidian atteint `sanroque:5051` (réseau local ou **Tailscale** ;
  tester : `curl http://sanroque:5051/health`).
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

1. Ouvrir/créer la note où insérer le résultat.
2. Lancer le template : via le raccourci, ou Ctrl/Cmd-P → « Templater: Open Insert
   Template modal » → choisir le template.
3. Choisir le **mode** (Hybride recommandé / Sémantique / BM25).
4. Saisir la **question** → la synthèse + les sources s'insèrent au curseur.

### Dépannage

- « Wiki_LM : erreur — … » : serveur injoignable → vérifier
  `curl http://sanroque:5051/health` depuis l'appareil.
- Le template utilise **`requestUrl`** (API Obsidian), pas `fetch()`, pour
  contourner le CSP d'Electron — ne pas revenir à `fetch()`.
- « Aucune information » sur un document récent : normalement résolu par
  l'auto-reload du serveur ; sinon forcer `curl -X POST http://sanroque:5051/reload`.

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
