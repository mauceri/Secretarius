---
tags: [spec, secretarius, courrier, imap, smtp, proton, v1]
date: 2026-10-07
statut: à relire
---

# Spec : courrier par IMAP/SMTP, boîte de Tiron et agent « consignes »

## Contexte

Il s'agit du troisième chantier préalable à la V1 de Secretarius. gog (l'interface en ligne
de commande de Google Workspace) est abandonné pour le courrier au profit d'IMAP et SMTP
génériques. L'utilisateur veut **basculer peu à peu Secretarius vers Proton**
(décision du 2026-10-07).

Deux constats faits le 2026-10-07 orientent la conception :

- **Les requêtes gog échouent par délai dépassé.** Les sous-agents sont mis en file sur la
  voie d'exécution de `main` (`maxConcurrent: 1`), que le tour en cours occupe déjà :
  `lane wait exceeded: lane=main waitedMs=119970 activeAhead=1`. C'est un interblocage, et
  non une panne de gog.
- **Une consigne planifiée qui demande des outils échoue.** Dans ses sessions en bac à
  sable, l'agent `main` n'a aucun outil `wiki_*` ni `gog_*`
  (`agents.entries.main.tools.sandbox.tools.deny`). Le modèle répond alors de mémoire, et
  faux : « aucune page à relire », alors qu'il y en avait une.

## Décisions

| Sujet | Décision |
|---|---|
| Protocole | IMAP et SMTP génériques ; gog ne sert plus au courrier |
| Boîtes | Gmail perso et Proton perso (boîtes de l'utilisateur, consultées par Tiron) ; tiron@secretarius.co (boîte de Tiron, Proton Business) |
| Accès | Gmail : mot de passe d'application ; Proton (les deux comptes) : Proton Bridge en local sur sanroque |
| Envoi par défaut | Proton perso (notes Obsidian, `/ecrire`) ; une réponse part de la boîte qui a reçu le message |
| Construction | Module maison `tiron_mail` (Python standard, ligne de commande en JSON), sur le modèle de `tiron_cron`. Pas d'extension communautaire : vos identifiants restent dans votre code |
| tiron@secretarius.co | Sert à transmettre des instructions à Tiron. Elles sont acceptées seulement si elles viennent de vos adresses ET sont authentifiées par Proton (`Authentication-Results`). Le compte rendu arrive sur Telegram. La conversation par courriel est remise à plus tard |
| Instructions par courriel | Commandes explicites en `/`, une par ligne, comme les blocs ```wiki``` ; `/consigne <texte>` pour une instruction composée, confiée à un nouvel agent « consignes » (pas à `main`). Pas de langage naturel, pas de réentraînement de Laya (décision du 2026-10-07) |
| Envoi par un agent | Automatique vers vos adresses ; vers toute autre adresse, brouillon à confirmer sur Telegram. Règle appliquée dans `tiron_mail`, pas par le modèle |
| Obsidian | Blocs `mail?` (aperçu) et `mail!` (envoi), même convention que `cron?` et `cron!` |
| Agenda et Drive | Restent sur gog en attendant une API calendrier chez Proton |

## Architecture

```
Boîtes : Gmail perso · Proton perso · tiron@secretarius.co
             │                 └──── Proton Bridge (local) ───┐
             └──── IMAP/SMTP (mot de passe d'application) ────┤
                                                               ▼
                     tiron_mail (Python standard, CLI JSON)
           lister · chercher · lire · préparer · envoyer (règles d'envoi)
          ▲                     ▲                        ▲
  derisk-deleg            wiki-lm-server            agent « consignes »
  /inbox /chercher        blocs mail? / mail!       outils : lecture du wiki, courrier
  /lire /repondre         (Obsidian)                modèle : choisi après essai
  /ecrire, appel direct                                   ▲
                                                          │ tâche OpenClaw immédiate
                                                          │ (--agent consignes), compte rendu Telegram
             relève de tiron@secretarius.co ──────────────┤
             expéditeur = vous + authentifié par Proton    │
               ├─ lignes « /commande … » → exécutées dans l'ordre, comme tapées
               └─ « /consigne … » → agent « consignes »
             /cron … consigne … ────────────────────────────┘ (cet agent, au lieu de main)
```

### `tiron_mail`

Le module expose une ligne de commande JSON, comme `tiron_cron` : `lister`, `chercher`,
`lire`, `preparer`, `envoyer`, `deliver` (pour les livraisons `/cron` par courriel).

Configuration :
- `~/.config/tiron_mail/comptes.toml`, sans secret, versionné en modèle : serveurs, ports,
  adresses, rôle de chaque compte ;
- les secrets restent dans `~/.config/secrets.env` (`GMAIL_APP_PASSWORD`,
  `PROTON_BRIDGE_PASS_PERSO`, `PROTON_BRIDGE_PASS_TIRON`).

Identifiants courts préfixés par la boîte : `g:` pour Gmail, `p:` pour Proton perso, `t:` pour Tiron.

Le corps d'un message lu est du **contenu non fiable**. Il est rendu en texte seul et ne
part jamais vers un modèle comme une instruction. Les pièces jointes sont listées, pas ouvertes.

Règles d'envoi, appliquées par le module lui-même :
- vers vos propres adresses (liste fermée de la configuration) : envoi direct permis ;
- vers toute autre adresse : brouillon seulement, envoyé après `/confirm`.

### Commandes Telegram (derisk-deleg, appel direct, sans sous-agent)

| Commande | Comportement | Compte |
|---|---|---|
| `/inbox [n] [compte]` | n derniers messages (10 par défaut) : expéditeur, objet, date, identifiant | tous, ou `gmail`, `proton`, `tiron` |
| `/chercher <critères> [compte]` | recherche IMAP (expéditeur, objet, corps, période) | tous |
| `/lire <id>` | contenu présenté comme non fiable | déduit de l'id |
| `/repondre <id> <texte>` | brouillon depuis la boîte qui a reçu le message, puis `/confirm` | déduit de l'id |
| `/ecrire <adresse> <texte>` *(nouvelle)* | nouveau message, puis `/confirm` | Proton perso |

Ces commandes s'utilisent tapées : aucun ajout au corpus de Laya.

### Relève de tiron@secretarius.co

- Service systemd `tiron-mail-releve` : interrogation IMAP toutes les minutes, via Proton Bridge.
- Un message est accepté si son expéditeur est dans la liste de vos adresses **et** si
  `Authentication-Results` indique `dmarc=pass`, ou à défaut `spf=pass` et `dkim=pass`.
- Message accepté : chaque ligne de l'objet ou du corps qui commence par `/` est une commande.
  - Les commandes sont exécutées dans l'ordre, comme si elles étaient tapées sur Telegram
    (compte rendu et `/confirm` sur Telegram).
  - `/consigne <texte>` (nouvelle commande) confie une instruction composée à l'agent
    « consignes ».
  - Un message sans aucune ligne en `/` n'est pas exécuté : avis sur Telegram avec la liste
    des commandes.
  - Pas de langage naturel, donc pas de Laya : aucun réentraînement.
- Message refusé : rien n'est exécuté ; avis sur Telegram (expéditeur, motif) ; le corps
  ne part jamais vers un modèle.
- Après traitement, le message est marqué lu et rangé dans « Traité » ou « Refusé ». Il ne
  peut pas être exécuté deux fois.

### Agent « consignes »

- Nouvel agent OpenClaw, dans un bac à sable, sur sa propre voie d'exécution (pas
  d'interblocage avec `main`).
- Outils :
  - lecture du wiki (`wiki_query`, `wiki_status`, `wiki_review`, `wiki_search`, `wiki_tags`) ;
  - lecture et envoi de courrier par `tiron_mail`, avec ses règles d'envoi.

  Aucune écriture dans le wiki, aucune suppression.
- Remplace `main` pour les consignes `/cron` (`--agent consignes`) et pour les instructions
  composées reçues par courriel.
- Modèle choisi par essai, une fois les outils en place : qwen3:8b ; le Qwen3-14B obfusqué
  sur Modal s'il ne sait pas enchaîner les outils.

### Obsidian

- Bloc `mail?` : aperçu sous le bloc (adresse d'envoi, destinataire, objet, début du
  corps), sans envoi.
- Bloc `mail!` : envoi depuis le Proton perso ; résultat écrit sous le bloc.
- Corps : le texte du bloc après `à :` et `objet :` ; à défaut, toute la note sans ses blocs
  de commande.
- Passe par wiki-lm-server (`_RUN_OPS`), qui appelle `tiron_mail`, comme `cron?` et `cron!`.

### Livraisons vers Obsidian (décision du 2026-10-07)

L'utilisateur trouve Obsidian plus naturel que Telegram. Le **contenu** produit
périodiquement est donc livré dans le coffre ; Telegram sert à **prévenir**.

- **Dossier de réception** `Secretarius/Réception/` dans le coffre : une note par
  livraison. En-tête : origine (courrier, cron, lettre d'information, veille), date, tâche
  d'origine.
- **Écriture par wiki-lm-server**, qui reçoit :
  - les résultats des tâches `/cron`, par l'option `--webhook` d'OpenClaw ;
  - les récapitulatifs de la relève de courrier.

  Une note par livraison plutôt qu'un ajout dans une note existante : on évite ainsi les
  conflits de synchronisation quand une note est modifiée sur un autre appareil.
- **Tâche créée depuis une note** (bloc `cron!`) : la note d'origine reçoit sous le bloc
  un lien vers ses livraisons, comme les réponses écrites sous les blocs `wiki`.
- **Notification** : chaque nouvelle livraison déclenche un court message Telegram, par
  exemple « Nouvelle livraison dans Obsidian : Courrier du 07/10 (3 messages) ». Le
  réglage du son revient à l'utilisateur, sur Android et sa montre. Le SMS prendra le
  relais plus tard pour les alertes sonores.

| Livraison | Destination par défaut |
|---|---|
| Rappels `/cron` | Telegram (prévenir à l'heure) |
| Résultats de commandes et de consignes `/cron` | Obsidian, plus notification Telegram |
| Récapitulatif de la relève des boîtes de l'utilisateur | Obsidian, plus notification Telegram |
| Lettres d'information (expéditeurs listés par l'utilisateur) | Obsidian (une note chacune, contenu non fiable, texte seul), plus notification |
| Veilles (plus tard) | Obsidian, plus notification |
| Avis de sécurité (courriel refusé par la relève de tiron@…) | Telegram |

Le canal reste précisable : `/cron add … telegram consigne …`, ou `obsidian` pour un rappel.

### Interblocage de la voie `main`

Il faut le corriger pour gog, qui garde l'agenda et Drive. La voie à retenir est à
déterminer en lisant comment Scout (`/source`) échappe à ce problème : autoriser deux tâches
sur la voie de `main`, ou lancer les sous-agents sur leur propre voie.

## Erreurs

- Proton Bridge arrêté, ou mot de passe refusé : message clair, compte par compte ; les
  autres boîtes restent utilisables.
- Envoi refusé par le serveur : brouillon conservé, erreur affichée.
- Relève en échec : nouvel essai à la minute suivante ; après dix échecs de suite, un avis
  unique sur Telegram.
- Identifiant de message périmé : « message introuvable ».

## Tests

- `tiron_mail` : pytest avec un faux serveur IMAP et SMTP (recherche, lecture, identifiants,
  règles d'envoi, refus d'un envoi automatique vers une adresse tierce).
- Relève : tri (expéditeur, `Authentication-Results`), traitement unique, refus sans
  transmission du corps.
- derisk-deleg et le plugin Obsidian : nouvelles commandes et blocs `mail?` / `mail!`.
- Essai réel avec l'utilisateur :
  - Bridge installé ;
  - une lecture par boîte ;
  - une réponse confirmée ;
  - un envoi depuis Obsidian ;
  - une instruction envoyée à tiron@secretarius.co depuis Gmail, puis depuis une adresse
    non autorisée.
- Agent « consignes » : « fais le point sur le wiki et les pages à relire », réponse
  comparée à `wiki.op_review()`.

## Ordre de réalisation (chaque étape est utilisable seule)

1. Correction de l'interblocage de la voie `main` (gog : agenda, Drive).
2. `tiron_mail` et les commandes Telegram : Gmail d'abord ; Proton dès que le Bridge est
   installé (connexion initiale avec l'utilisateur, double authentification).
3. Livraisons vers Obsidian (`Réception/`, webhook des tâches `/cron`, notification Telegram), puis blocs Obsidian `mail?` / `mail!`.
4. Relève de tiron@secretarius.co.
5. Agent « consignes », puis bascule des consignes `/cron` vers lui.

## Hors périmètre

- Extension Chrome de capture d'URL depuis le navigateur (même point d'entrée `/capture` que le plugin Obsidian) : chantier à venir.
- Notifications sonores par SMS : après l'installation du canal SMS.

- Conversation par courriel avec Tiron (réponses par courriel aux instructions).
- Agenda hors gog (en attente d'une API calendrier chez Proton) ; Lumo à surveiller pour
  remplacer Infomaniak.
- Ouverture des pièces jointes.
