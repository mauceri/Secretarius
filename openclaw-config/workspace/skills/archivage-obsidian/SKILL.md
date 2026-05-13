---
name: archivage-obsidian
description: Archiver le coffre Obsidian (${OBSIDIAN_PATH}) dans ~/sauvegarde_obsidian.tar. Déclencher sur "archiver le coffre", "sauvegarder Obsidian", "backup coffre".
---

# Archivage du coffre Obsidian

## Description
Skill pour archiver le répertoire `${OBSIDIAN_PATH}` (coffre Obsidian) dans une archive tar.

## Quand l'utiliser
- Quand l'utilisateur demande d'"archiver le coffre", "sauvegarder Obsidian", "backup coffre".

## Procédure
1. Vérifier que `${OBSIDIAN_PATH}` existe.
2. Créer l'archive :
   ```sh
   tar -cf ~/sauvegarde_obsidian.tar -C "$(dirname ${OBSIDIAN_PATH})" "$(basename ${OBSIDIAN_PATH})"
   ```
3. Vérifier la création (taille, premiers fichiers) et informer l'utilisateur.

## Notes
- L'archive est créée dans le home de l'utilisateur (`~/`).
- Si une archive du même nom existe déjà, elle est écrasée.
- Pour une compression, ajouter l'option `-z` (gzip) ou `-j` (bzip2) si demandé.
- Le skill ne supprime pas le répertoire source.
