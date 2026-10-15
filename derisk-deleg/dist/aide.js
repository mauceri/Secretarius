// Aide des commandes tapées sur Telegram (le /help de Telegram est celui
// d'OpenClaw ; l'aide des blocs wiki d'Obsidian est /help dans wiki.py).
export const AIDE = `Commandes Secretarius (Telegram)

Wiki
/q <question> — réponse rédigée à partir du wiki (environ 1 min)
/r <mots-clés> — recherche rapide, 5 résultats, sans synthèse
/c <texte ou URL> — capture une note ou une page (#tags acceptés)
/ingest — ingère les captures en attente
/wikistatus — état de l'ingestion
/tags — tags présents dans le wiki
/relire — page source à relire
/verifie <page> — marque une page comme vérifiée (puis /confirm)
/supprimer <page> — suppression en cascade : aperçu, puis /confirm
/kbupdate — reconstruit la base de connaissances
/source <url> — lecture d'une page web par Scout, isolé

Courrier
/inbox [compte] — derniers messages
/chercher <mots> [compte] — recherche dans les boîtes
/lire <id> — affiche un message
/repondre <id> <texte> — réponse : aperçu, puis /confirm
/ecrire <adresse> <texte> — nouveau message : aperçu, puis /confirm

Agenda et Drive (Google)
/agenda — prochains rendez-vous
/drive <requête> — recherche dans Drive
/connecter — autorise le compte Google

Planification
/cron add <quand> [<canal>] <rappel|commande|consigne> <contenu> — aperçu, puis /confirm
/cron list|rm|pause|reprendre [id] — gère les tâches

Confirmation et administration
/confirm — valide l'action en attente (10 min)
/annuler — abandonne l'action en attente
/maj — recopie les clés changées de secrets.env et relance les services
/jeton ajouter|retirer <préfixe> ou /jeton lister — jetons des agents Grok (santiago)
/aide — cette aide

Selon la machine, certaines commandes (courrier, Google, planification, jetons) peuvent être absentes.`;
