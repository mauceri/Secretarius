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
