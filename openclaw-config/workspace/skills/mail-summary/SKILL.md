---
name: mail-summary
description: Gérer la liste des expéditeurs dont les newsletters sont résumées automatiquement sur Telegram (matin/midi/soir). Ajouter, supprimer ou lister des expéditeurs dans ~/.local/share/mail-summary-senders.json.
---

# Skill : mail-summary

## Rôle
Gérer la liste des expéditeurs dont les newsletters sont résumées chaque matin, midi et soir sur Telegram.

## Fichier de configuration
```
~/.local/share/mail-summary-senders.json
```
C'est un tableau JSON d'adresses email. Exemple :
```json
[
  "dailydigest@email.join1440.com",
  "news@daily.theepochtimes.com"
]
```

## Comment ajouter un expéditeur

1. Lire le fichier de configuration :
   ```
   read ~/.local/share/mail-summary-senders.json
   ```

2. Ajouter l'adresse email à la liste et réécrire le fichier.

3. Confirmer à l'utilisateur que l'adresse a été ajoutée et que le changement sera effectif dès le prochain envoi.

## Comment supprimer un expéditeur

Même procédure : lire, retirer l'entrée, réécrire.

## Comment lister les expéditeurs actuels

Lire `~/.local/share/mail-summary-senders.json` et afficher la liste.

## Planification (crontab)
Les résumés sont envoyés automatiquement à :
- 9h00 (Paris)
- 12h00 (Paris)
- 21h00 (Paris)

Le script est `${HOME}/.local/bin/mail-summary`.
Les logs sont dans `~/.local/share/mail-summary.log`.
