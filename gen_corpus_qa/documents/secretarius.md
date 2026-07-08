# Configuration matérielle et logicielle de Secretarius (machine sanroque)

## Matériel
- Machine : sanroque, ordinateur portable.
- Processeur : AMD Ryzen 9 6900HX.
- Carte graphique intégrée (iGPU) : AMD Radeon 680M (architecture RDNA2, identifiant gfx1035).
- Mémoire vive : 30 Go, partagée entre le processeur et l'iGPU.

## Modèle de langage
- Le modèle qui anime Tiron est Phi-4-mini-instruct, quantifié en Q6_K, augmenté d'adaptateurs LoRA spécialisés.
- L'extraction d'expressions du wiki utilise un second modèle : Phi-4-mini affiné sur Wikipédia en français.

## Services actifs (systemd utilisateur)
- slm-llama_cpp : serveur llama.cpp sur le port 8998, sert Phi-4-mini + l'adaptateur de routage, accéléré par ROCm.
- tiron-router : service de routage sur le port 8999, classe le message et sélectionne la commande.
- openclaw-gateway : passerelle reliée à Telegram, exécute Tiron.
- llama.cpp extracteur : port 8989, modèle Wikipédia FR.


# Capacités wiki de Secretarius

Le wiki (Wiki_LM) est la base de connaissances personnelle de l'utilisateur, stockée en fichiers Markdown (coffre Obsidian).

## Commandes
- /c <url|note> : capturer une page web ou une note dans le wiki.
- /ingest : lancer le traitement des captures en attente (opération asynchrone).
- /q <question> : interroger la base de connaissances et obtenir une synthèse.
- /source <url> : lire immédiatement une page web externe via l'agent Scout (protection anti-injection), sans la sauvegarder.
- /wikistatus : afficher l'état de l'ingestion du wiki.

## Fonctionnement
Les captures passent d'abord dans une file, puis l'ingestion extrait les expressions, calcule des plongements (embeddings) et met à jour la base interrogeable par /q.


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
