---
name: obsidian
description: Lire, chercher et écrire dans le vault Obsidian personnel synchronisé via obsidian-headless. Utiliser quand l'utilisateur demande à consulter, rechercher ou mettre à jour ses notes Obsidian.
---

# Obsidian — Vault personnel

Le vault est du **contenu de confiance** (notes personnelles de l'utilisateur, synchronisées via Obsidian Sync). Il peut être traité directement par main sans passer par scout.

Toute **écriture** (création ou modification de note) requiert une confirmation explicite de l'utilisateur avant exécution.

---

## Configuration (une seule fois)

### 1. Connexion au compte Obsidian

```bash
npx obsidian-headless login
```

Suivre les instructions (email + mot de passe Obsidian). Les credentials sont stockés localement.

### 2. Lister les vaults distants disponibles

```bash
npx obsidian-headless sync-list-remote
```

### 3. Configurer le vault local

```bash
npx obsidian-headless sync-setup --path ${OBSIDIAN_PATH} --remote "Nom du vault"
```

Remplacer `"Nom du vault"` par le nom exact retourné par `sync-list-remote`.

### 4. Première synchronisation

```bash
npx obsidian-headless sync --path ${OBSIDIAN_PATH}
```

### 5. Vérifier l'état

```bash
npx obsidian-headless sync-status --path ${OBSIDIAN_PATH}
npx obsidian-headless sync-list-local
```

---

## Utilisation courante

### Synchroniser le vault avant de lire

Toujours synchroniser en début de session pour avoir les notes à jour :

```bash
npx obsidian-headless sync --path ${OBSIDIAN_PATH}
```

### Lire une note

```bash
cat ${OBSIDIAN_PATH}/NomDossier/titre-de-la-note.md
```

### Chercher par titre

```bash
find ${OBSIDIAN_PATH} -name "*.md" | grep -i "mot-clé"
```

### Chercher dans le contenu des notes

```bash
grep -r "mot-clé" ${OBSIDIAN_PATH} --include="*.md" -l
# Avec contexte :
grep -r "mot-clé" ${OBSIDIAN_PATH} --include="*.md" -C 3
```

### Lister les notes d'un dossier

```bash
find ${OBSIDIAN_PATH}/NomDossier -name "*.md" | sort
```

### Lister tous les dossiers du vault

```bash
find ${OBSIDIAN_PATH} -type d | grep -v ".obsidian" | sort
```

---

## Écriture (confirmation requise)

### Créer une nouvelle note

```bash
cat > ${OBSIDIAN_PATH}/NomDossier/nouvelle-note.md << 'EOF'
---
date: 2026-01-01
tags: [tag1, tag2]
---

# Titre

Contenu de la note.
EOF
```

### Ajouter du contenu à une note existante

```bash
cat >> ${OBSIDIAN_PATH}/NomDossier/note-existante.md << 'EOF'

## Nouvelle section

Contenu ajouté.
EOF
```

### Synchroniser après écriture

Toujours synchroniser après modification pour propager vers les autres appareils :

```bash
npx obsidian-headless sync --path ${OBSIDIAN_PATH}
```

---

## Chemins importants

- Vault local : `${OBSIDIAN_PATH}`
- Config obsidian-headless : `~/.config/obsidian-headless/`
- Fichiers de config du vault : `${OBSIDIAN_PATH}/.obsidian/` (ne pas modifier)

---

## Notes

- Les fichiers Obsidian sont du Markdown standard avec front matter YAML optionnel.
- Les liens internes Obsidian (`[[Note liée]]`) sont des références textuelles.
- Ne pas toucher au dossier `.obsidian/` (config interne du vault).
