# Faits — base de réponses directes de Tiron
# Format : un ou plusieurs titres "## question" (formulations) suivis de la réponse.
# Éditable à volonté. Seule l'entrée correspondante est injectée dans le modèle.

## Quel modèle de langage anime Tiron ?
## Quel LLM utilise Secretarius ?
## Sur quel modèle tourne Tiron ?
Tiron est animé par Phi-4-mini-instruct (quantifié en Q6_K), augmenté d'adaptateurs LoRA spécialisés. L'extraction d'expressions du wiki utilise un second modèle, Phi-4-mini affiné sur Wikipédia en français.

## Sur quelle machine tourne Secretarius ?
## Quel est le matériel de sanroque ?
## C'est quoi la config matérielle ?
Secretarius tourne sur sanroque, un ordinateur portable : processeur AMD Ryzen 9 6900HX, iGPU AMD Radeon 680M (RDNA2, gfx1035), 30 Go de mémoire vive partagée entre le processeur et l'iGPU.

## Combien de RAM sur sanroque ?
## Combien de mémoire vive ?
30 Go de mémoire vive, partagée entre le processeur et l'iGPU.

## Quels services tournent pour Secretarius ?
## Quels sont les services actifs et leurs ports ?
Quatre services systemd : slm-llama_cpp (port 8998, Phi-4-mini + adaptateur de routage, accéléré ROCm), tiron-router (port 8999, routage des messages), openclaw-gateway (passerelle Telegram qui exécute Tiron), et l'extracteur llama.cpp Wikipédia FR (port 8989).

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
