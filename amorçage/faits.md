# Faits — base de réponses directes de Tiron
# Format : un ou plusieurs titres "## question" (formulations) suivis de la réponse.
# Éditable à volonté. Seule l'entrée correspondante est injectée dans le modèle.

## Quel modèle de langage anime Tiron ?
## Quel LLM utilise Secretarius ?
## Sur quel modèle tourne Tiron ?
## Quel modèle vous anime ?
## Quel modèle t'anime ?
Tiron choisit la commande à exécuter avec Laya, un classifieur affiné sur ses propres commandes (Mauceric/laya-tiron-router). Les réponses de l'agent principal sont produites par Qwen3-8B, en local via Ollama. Le wiki utilise aussi Qwen3-8B en local. DeepSeek ne sert qu'à l'ingestion des pages web, et un modèle Qwen3-14B obfusqué sur Modal ou Infomaniak prennent le relais si le modèle local n'est pas utilisable.

## Sur quelle machine tourne Secretarius ?
## Quel est le matériel de sanroque ?
## C'est quoi la config matérielle ?
Secretarius tourne sur sanroque, un ordinateur portable : processeur AMD Ryzen 9 6900HX, iGPU AMD Radeon 680M (RDNA2, gfx1035), 30 Go de mémoire vive partagée entre le processeur et l'iGPU.

## Combien de RAM sur sanroque ?
## Combien de mémoire vive ?
30 Go de mémoire vive, partagée entre le processeur et l'iGPU.

## Quels services tournent pour Secretarius ?
## Quels sont les services actifs et leurs ports ?
Services systemd utilisateur : tiron-router (port 8999, routage des messages par le classifieur Laya), openclaw-gateway (passerelle Telegram qui exécute Tiron), openclaw-scout et openclaw-injection-guard (lecture sécurisée des contenus externes), wiki-lm-server (serveur du wiki pour Obsidian, port 10443, joignable par Tailscale seulement), et trois relais vers les modèles obfusqués sur Modal : obfuscator-proxy (Qwen3-14B, port 8001), obfuscator-proxy-docker (le même, joignable depuis Docker) et obfuscator-proxy-8b (Qwen3-8B, port 8002). Ollama (port 11434) sert Qwen3-8B en local. L'extracteur llama.cpp (port 8989) et phi-4-mini (port 8998) ne tournent plus.

## Qu'est-ce que le wiki de Secretarius ?
## C'est quoi Wiki_LM ?
Le wiki (Wiki_LM) est la base de connaissances personnelle de l'utilisateur, stockée en fichiers Markdown dans un coffre Obsidian. Les captures passent dans une file, puis l'ingestion en extrait les expressions, calcule des plongements et met à jour la base interrogeable.

## Comment interroger le wiki ?
## Quelle commande pour poser une question au wiki ?
## Comment obtenir une synthèse depuis mes connaissances ?
Utilisez /q <question> : la commande interroge la base de connaissances et renvoie une synthèse.

## Comment capturer une page ou une note dans le wiki ?
## Quelle commande pour enregistrer un lien dans le wiki ?
Utilisez /c <url|note> pour capturer une page web ou une note. Le traitement se lance ensuite avec /ingest (asynchrone).

## Comment programmer un rappel ?
## Quelle commande pour planifier une tâche ?
Écrivez simplement « rappelle-moi demain à 9h d'appeler le garagiste » ou « chaque lundi à 8h… », ou utilisez /cron add <quand> [canal] rappel <texte>. Tiron montre la tâche comprise et la crée après /confirm. /cron list affiche les tâches, /cron rm <id> en supprime une.

## Qu'est-ce qu'une consigne planifiée ?
## Quand utiliser une consigne avec /cron ?
Une consigne est la version en langage naturel d'une commande planifiée : vous décrivez un travail que Tiron fera lui-même avec ses commandes, par exemple /cron add "0 8 * * 1" consigne fais le point sur l'état du wiki et les pages à relire. Elle convient aux tâches qui s'appuient sur vos données (wiki, mails). Elle ne convient pas aux questions de culture générale (« Qui était Henri IV ? ») : le modèle local répond de mémoire et invente souvent. Pour une question, utilisez /q, qui répond à partir de votre wiki. Comptez deux à trois minutes de traitement après le déclenchement.

## Pourquoi Tiron me propose-t-il une liste numérotée ?
Quand Tiron n'est pas sûr de la commande voulue, il propose les plus probables avec leur nom. Répondez par le numéro, ou reformulez.

## Comment lire une page web externe sans l'enregistrer ?
## Quelle commande pour consulter un lien tout de suite ?
Utilisez /source <url> : la page est lue immédiatement via l'agent Scout (protection anti-injection), sans être sauvegardée.

## Comment voir l'état de l'ingestion du wiki ?
Utilisez /wikistatus.

## À quoi sert l'agent gog ?
## Qu'est-ce que Secretarius peut faire avec Google ?
L'agent gog donne accès au compte Google de l'utilisateur : messagerie Gmail et fichiers Google Drive.

## Comment chercher un email dans Gmail ?
## Quelle commande pour retrouver un mail ?
Utilisez /chercher <critères> : recherche par mot-clé, expéditeur ou période. /inbox liste les emails récents.

## Comment lire le contenu d'un email ?
## Quelle commande pour ouvrir un mail précis ?
Utilisez /lire <id> : lit le contenu d'un email par son identifiant, obtenu via /inbox ou /chercher.

## Comment répondre à un email ?
## Quelle commande pour préparer une réponse à un mail ?
Utilisez /repondre <contexte> : cela prépare un brouillon. Aucun email n'est envoyé automatiquement — il n'est expédié qu'après confirmation par /confirm.

## Comment chercher un fichier sur Google Drive ?
Utilisez /drive <critères>.

## Comment connecter mon compte Google ?
Utilisez /connecter pour autoriser l'accès au compte Google.

## Comment s'appelle le perroquet de Madame Michu ?
## Le perroquet de Mme Michu ?
Le perroquet de Madame Michu s'appelle Coco.
