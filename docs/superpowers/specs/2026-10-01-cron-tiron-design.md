---
tags: [spec, secretarius, cron, tiron, v1]
date: 2026-10-01
statut: à relire
---

# Spec : planification de tâches par Tiron (`/cron`)

## Contexte

Premier des trois chantiers préalables à la V1 de Secretarius (les deux autres :
agent développeur sur DeepSeek, agent courrier IMAP/SMTP). Objectif : permettre à
l'utilisateur de planifier rappels, commandes et consignes, indépendamment du
système d'exploitation, via Tiron (langage naturel ou commande `/cron`) et via
Obsidian (commande normalisée seulement).

Pas d'ordonnanceur à écrire : le gateway OpenClaw (2026.8.2) embarque un
planificateur multiplateforme (`openclaw cron add|list|rm|enable|disable|runs`),
déjà utilisé par les tâches `skill-collection-review`.

## Décisions

| Sujet | Décision |
|---|---|
| Types de tâches | les trois : `rappel`, `commande`, `consigne` |
| Canaux de livraison V1 | Telegram, WhatsApp, SMS (Twilio), Obsidian |
| Courrier | ajouté comme canal après le chantier courrier (option 1) |
| Langage naturel | Laya classe vers `/cron` ; analyseur déterministe FR → syntaxe normalisée (A) ; repli qwen3:8b local (B) si l'analyseur échoue (révisé 2026-10-02) |
| Obsidian | syntaxe normalisée seulement en V1 ; LN en V2 |
| Création | confirmation obligatoire (`/confirm`) |
| Shell | aucun shell fourni par l'utilisateur |

## Syntaxe normalisée

```
/cron add <quand> [<canal>] <type> <contenu>
/cron list
/cron rm <id>
/cron pause <id>
/cron reprendre <id>
```

- `<quand>` :
  - expression cron entre guillemets, 5 champs (`"0 8 * * 1"`) → récurrent ;
  - date-heure ISO sans décalage (`2026-10-02T09:00`) → tâche unique ;
  - durée (`20m`, `2h`, `1d`) → tâche unique relative.
  Fuseau : `Europe/Paris` par défaut.
- `<canal>` : `telegram` | `whatsapp` | `sms` | `obsidian` (`mail` plus tard).
  Omis → canal d'origine du message (Obsidian compris).
- `<type>` :
  - `rappel` : `<contenu>` livré tel quel, sans modèle ;
  - `commande` : `<contenu>` est une commande Tiron de la liste blanche
    (`/inbox`, `/wikistatus`, `/kbupdate`, … — liste fermée définie dans
    `tiron_cron`) ;
  - `consigne` : `<contenu>` envoyé à l'agent `main` avec son modèle habituel.

Reformulation avant confirmation, ex. :
« Chaque lundi à 08:00 (Europe/Paris), SMS, rappel : payer le loyer. /confirm ? »

## Architecture

```
Telegram/WhatsApp/SMS ─► Laya ─► /cron ─► analyseur FR ─┐
                          (repli B : qwen3:8b local) ────┤
Obsidian ─► wiki-lm-server /run ──────────┤
                                          ▼
                         tiron_cron (Python, CLI + module)
                analyse · vérifie · reformule · confirmation
                                          │
                                          ▼
                          openclaw cron add/list/rm/…
                                          │ au déclenchement
             ┌──────────────┬─────────────┼───────────────┐
           rappel        commande      consigne       canal obsidian
        (--announce   (liste blanche  (--message     (script interne :
         --channel)   → derisk-deleg)  agent main)    écrit une note)
```

### `tiron_cron` (cœur unique)

Module Python + CLI dans le dépôt. Responsabilités :

1. analyser la syntaxe normalisée ;
2. valider `<quand>`, `<canal>`, `<type>`, liste blanche ;
3. produire la reformulation ;
4. traduire en appel `openclaw cron add` (`--cron`/`--at`, `--tz`,
   `--channel`, `--announce`, `--message`, `--agent`, `--delete-after-run`,
   `--command` pour les canaux scriptés) ;
5. étiqueter chaque tâche avec `--declaration-key tiron:<id>` ;
6. `list`/`rm`/`pause`/`reprendre` restreints aux tâches `tiron:*` (les tâches
   système comme `skill-collection-review` n'apparaissent pas).

Appelants : derisk-deleg (canaux de messagerie, après confirmation) et
wiki-lm-server (`/run` pour Obsidian), tous deux via la CLI
(`python3 -m tiron_cron …`, sortie JSON) — un seul protocole d'appel.

Destinataires (assistant mono-utilisateur) : variables d'environnement du
gateway — `TIRON_TELEGRAM_CHAT_ID`, `TIRON_WHATSAPP_TO`, `TIRON_SMS_TO`
(E.164) — et `TIRON_RAPPELS_DIR` (défaut `~/Documents/Arbath/Rappels`) pour le
canal `obsidian`.

### Laya et langage naturel (révisé 2026-10-02)

Laya est un classifieur (types de questions `choice`/`score`/`noul`), il ne
génère aucun texte : `router_service/server.py` transmet toujours le message
brut comme `args` (décision du 2026-09-28). Il ne peut donc pas produire la
syntaxe normalisée. Chaîne retenue :

1. **Laya** classe le message vers `/cron` (nouvelle entrée de
   `COMMAND_CRITERIA`, nouvelle intention `cron_schedule` dans le corpus).
2. **Analyseur déterministe** (`tiron_cron/nl.py`) : extrait `<quand>`,
   `<canal>`, `<type>`, `<contenu>` des formes françaises courantes (« chaque
   lundi à 8h », « tous les jours à 7h30 », « demain à 9h », « le 15 octobre à
   14h », « dans 20 minutes » ; canal par mots-clés « par SMS », « sur
   WhatsApp »… ; type par verbes « rappelle-moi » → rappel, commande Tiron
   nommée → commande, sinon consigne).
3. **Repli (B)** : si l'analyseur échoue, qwen3:8b local (Ollama) produit la
   syntaxe normalisée ; le texte ne quitte pas la machine.
4. Reformulation et `/confirm` dans tous les cas.

Canal et type ne sont pas des questions Laya supplémentaires : vocabulaire
fermé, détecté sans ambiguïté par mots-clés, et une question Laya de plus
imposerait des étiquettes propres au cron dans toute la chaîne corpus →
LoRA → Laya et un réentraînement dédié.

GEPA : relance avec un jeu de validation séparé (30–50 exemples) et
`GEPAPrompt.txt` revu (brièveté, niveau de langage), pistes reportées le
2026-09-30. Réentraînement Laya sur Kaggle (manuel), seuil ≥ 90 %.

### Obsidian

- Envoi : `/cron?` (aperçu : reformulation, rien n'est créé) et `/cron!`
  (création), même convention `?`/`!` que `/supprimer` et `/repair` — le `!`
  tient lieu de confirmation. `/cron list` est en lecture. Ajoutés à
  `SUPPORTED_COMMANDS` du plugin `obsidian-wikilm-capture` et à `_RUN_OPS` de
  wiki-lm-server, qui appelle la CLI `tiron_cron` avec canal d'origine
  `obsidian`. Le résultat s'écrit sous le bloc de commande (même mécanisme que
  les commandes wiki actuelles).
- Livraison : canal `obsidian` = script interne lancé par `--command`, qui écrit
  une note dans le coffre (`$TIRON_RAPPELS_DIR/<date> <titre>.md`), synchronisée
  par obsidian-headless. Pas de notification active en V1.

### Canaux WhatsApp et SMS

- Installation des extensions OpenClaw WhatsApp et SMS.
- SMS : vérifier la compatibilité Twilio de l'extension. À défaut, script interne
  d'envoi via l'API Twilio (même principe que le canal `obsidian`).

## Gestion des erreurs

- Syntaxe invalide → refus avec message précis, rien n'est créé.
- Échec de livraison → statut OpenClaw (`openclaw cron runs`) ; `/cron list`
  affiche l'état du dernier déclenchement. Pas de canal de secours en V1.
- Tâche unique → supprimée après succès ; conservée désactivée en cas d'échec.
- Gateway arrêté → tâches persistées par OpenClaw ; pas de rattrapage
  spécifique en V1.

## Sécurité

- Aucun shell utilisateur ; `--command` réservé aux scripts internes.
- Seules les commandes de la liste blanche sont planifiables.
- Seul le propriétaire (Telegram, Obsidian) crée des tâches ; le courrier entrant
  (futur) ne peut pas en créer, il passe par Scout.
- `consigne` garde le modèle de l'agent `main` (politique de confidentialité).

## Tests

1. **Essai préalable** : une tâche OpenClaw dont le message est `/inbox`
   déclenche-t-elle le hook `before_agent_reply` de derisk-deleg ? Oui → type
   `commande` via le message ; non → exécution directe par l'extension.
2. **pytest `tiron_cron`** : analyse (valide/invalide), liste blanche,
   reformulation, arguments `openclaw cron` (CLI simulé).
3. **Laya** : évaluation de l'intention `cron` sur jeu de validation séparé,
   seuil ≥ 90 %.
4. **Bout en bout manuel sur sanroque** : rappel à 2 min sur chaque canal
   (Telegram, WhatsApp, SMS, Obsidian), puis `/cron list`, `/cron rm`.

## Hors périmètre V1

- Langage naturel depuis Obsidian.
- Canal `mail` (après le chantier courrier).
- Notification active dans Obsidian, canal de secours, rattrapage des
  déclenchements manqués.
