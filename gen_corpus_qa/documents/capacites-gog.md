# Capacités Google (gog) de Secretarius

L'agent gog donne accès au compte Google de l'utilisateur : messagerie Gmail et fichiers Google Drive.

## Commandes
- /chercher <critères> : rechercher des emails dans Gmail (par mot-clé, expéditeur ou période).
- /inbox : lister les emails récents de la boîte de réception.
- /repondre <contexte> : préparer un brouillon de réponse à un email.
- /drive <critères> : rechercher des fichiers sur Google Drive.
- /connecter : autoriser l'accès au compte Google.

## Sécurité
Aucun email n'est envoyé automatiquement : /repondre prépare seulement un brouillon, qui n'est expédié qu'après confirmation explicite par la commande /confirm.
