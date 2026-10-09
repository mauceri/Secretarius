# Faits — base de réponses directes de Tiron
# Format : un ou plusieurs titres "## question" (formulations) suivis de la réponse.
# Éditable à volonté. Seule l'entrée correspondante est injectée dans le modèle.

## Quel modèle de langage anime Tiron ?
## Quel LLM utilise Secretarius ?
## Sur quel modèle tourne Tiron ?
## Quel modèle vous anime ?
## Quel modèle t'anime ?
## Quel modèle vous anime-t-il ?
## Quel modèle utilisez-vous ?
## Tu tournes sur quel modèle ?
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
## Quels services tournent ?
## Quels sont les services de Secretarius ?
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
## Comment créer un rappel ?
## Comment planifier un rappel ?
## Comment fonctionne /cron ?
## Comment voir ou supprimer mes rappels ?
Écrivez simplement « rappelle-moi demain à 9h d'appeler le garagiste » ou « chaque lundi à 8h… », ou utilisez /cron add <quand> [canal] rappel <texte>. Tiron montre la tâche comprise et la crée après /confirm. /cron list affiche les tâches, /cron rm <id> en supprime une.

## Qu'est-ce qu'une consigne planifiée ?
## Quand utiliser une consigne avec /cron ?
## C'est quoi une consigne ?
## À quoi sert une consigne ?
Une consigne est la version en langage naturel d'une commande planifiée : vous décrivez un travail que Tiron fera lui-même avec ses commandes, par exemple /cron add "0 8 * * 1" consigne fais le point sur l'état du wiki et les pages à relire. Elle convient aux tâches qui s'appuient sur vos données (wiki, mails). Elle ne convient pas aux questions de culture générale (« Qui était Henri IV ? ») : le modèle local répond de mémoire et invente souvent. Pour une question, utilisez /q, qui répond à partir de votre wiki. Comptez deux à trois minutes de traitement après le déclenchement.

## Pourquoi Tiron me propose-t-il une liste numérotée ?
## Pourquoi tu me proposes une liste ?
## Pourquoi dois-je répondre par un numéro ?
Quand Tiron n'est pas sûr de la commande voulue, il propose les plus probables avec leur nom. Répondez par le numéro, ou reformulez.

## Comment lire une page web externe sans l'enregistrer ?
## Quelle commande pour consulter un lien tout de suite ?
Utilisez /source <url> : la page est lue immédiatement via l'agent Scout (protection anti-injection), sans être sauvegardée.

## Comment voir l'état de l'ingestion du wiki ?
Utilisez /wikistatus.

## À quoi sert l'agent gog ?
## Qu'est-ce que Secretarius peut faire avec Google ?
L'agent gog donne accès à l'agenda Google et aux fichiers Google Drive de l'utilisateur. Le courrier ne passe plus par gog : Tiron lit et envoie les emails lui-même (IMAP/SMTP).

## Quelles boîtes mail Tiron consulte-t-il ?
## Quelles adresses mail sont configurées ?
## Depuis quelle adresse Tiron envoie-t-il les mails ?
Tiron consulte cinq boîtes, chacune repérée par un préfixe dans les identifiants de message : g: Gmail (cmauceri@gmail.com), p: Proton personnel (christian@mauceri.me), t: tiron@secretarius.co, c: contact@secretarius.co, v: privacy@secretarius.co. Les boîtes Proton passent par Proton Bridge, sur sanroque. Un nouveau message (/ecrire) part de tiron@secretarius.co ; une réponse (/repondre) part de la boîte qui a reçu le message.

## Comment voir mes derniers emails ?
## Quelle commande pour lister les mails récents ?
Utilisez /inbox [n] [boîte] : /inbox affiche les 10 derniers messages de chaque boîte, /inbox 3 proton les 3 derniers de la boîte Proton. Noms des boîtes : gmail, proton, tiron, contact, privacy.

## Comment chercher un email ?
## Comment chercher un email dans Gmail ?
## Quelle commande pour retrouver un mail ?
Utilisez /chercher <critères> [boîte] : mots du message, de:<expéditeur>, objet:<mot>, depuis:<n>j — par exemple /chercher de:paul depuis:7j devis. Sans nom de boîte, la recherche porte sur toutes les boîtes.

## Comment lire le contenu d'un email ?
## Quelle commande pour ouvrir un mail précis ?
Utilisez /lire <id>, avec l'identifiant donné par /inbox ou /chercher (par exemple /lire p:42). Le contenu est affiché comme non fiable, à lire seulement ; les pièces jointes sont listées mais jamais ouvertes.

## Comment répondre à un email ?
## Quelle commande pour préparer une réponse à un mail ?
Utilisez /repondre <id> <texte> : Tiron montre l'aperçu (destinataire, boîte d'envoi, texte). Rien n'est envoyé avant /confirm ; /annuler abandonne.

## Comment écrire un nouvel email ?
## Comment envoyer un mail ?
Utilisez /ecrire <adresse> <texte> : le message part de tiron@secretarius.co, et son objet reprend le début de la première ligne. Tiron montre l'aperçu ; rien n'est envoyé avant /confirm.

## Où arrivent les résultats de mes tâches planifiées ?
## Où trouver le résultat d'une consigne /cron ?
Le résultat d'une consigne ou d'une commande planifiée arrive dans une note du dossier Réception du coffre Arbath, et Tiron vous prévient sur Telegram. Ajoutez telegram dans /cron add (par exemple /cron add "0 8 * * 1" telegram commande /relire) pour le recevoir sur Telegram à la place. Un rappel arrive là où il a été créé (Telegram ou Obsidian), sauf canal précisé. Pour l'état du wiki, planifiez une commande (/wikistatus, /relire) plutôt qu'une consigne : les consignes n'ont pas encore accès au wiki.

## Comment envoyer un courriel depuis une note ?
## Comment écrire un mail depuis Obsidian ?
Dans la note, ajoutez un bloc wiki dont la première ligne est /mail?, puis « à : <adresse> », « objet : <objet> » et le texte ; sans texte, la note elle-même sert de corps. Exécutez les commandes wiki de la note : /mail? montre l'aperçu sous le bloc, sans rien envoyer. Remplacez par /mail! et relancez pour envoyer, depuis tiron@secretarius.co.

## Comment chercher un fichier sur Google Drive ?
Utilisez /drive <critères>.

## Comment connecter mon compte Google ?
Utilisez /connecter pour autoriser l'accès au compte Google (agenda et Drive).

## Comment s'appelle le perroquet de Madame Michu ?
## Le perroquet de Mme Michu ?
Le perroquet de Madame Michu s'appelle Coco.
