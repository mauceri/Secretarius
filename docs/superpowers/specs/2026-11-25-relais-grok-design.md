# Relais des courriels de Grok Bot vers Session — conception

Statut : à relire (2026-11-25).

## Intention

L'utilisateur fait travailler ensemble Secretarius (Tiron) et son équipe Grok Bot (Rector,
Scriptor…). Depuis le 2026-11-25, Grok n'a plus aucun accès au wiki (porte réduite au coffre
Publication, 5bdcf28) : les échanges passent par l'utilisateur.

Exemple d'usage (utilisateur) : il demande à Grok « Pouvez-vous poster dans Publication la
dernière version du conte que nous écrivons avec Scriptor et me dire quand ce sera fait ? ».
Grok dépose le texte par la porte (`coffre_ecrire`) puis répond « c'est fait » par courriel à
Tiron, qui doit le relayer à l'utilisateur sur Session.

## Les deux sens

1. **Secretarius → Grok : rien à construire.** `/ecrire yc28mybv+secretarius@mail.grokbot.com
   <demande>` existe déjà : aperçu, puis envoi depuis `tiron@secretarius.co` après `/confirm`.
2. **Grok → Secretarius : l'objet de cette spec.** Aujourd'hui la relève refuse tout courriel qui
   ne vient pas d'une adresse du propriétaire (avis « Courriel refusé… », contenu jamais
   transmis). Elle doit relayer sur Session ceux de Grok Bot, **comme information, jamais comme
   instruction**.

## Objectifs

- Un courriel authentique de Grok Bot arrivant dans la boîte relevée (`tiron`) est transmis à
  l'utilisateur sur Session, après passage par le garde d'injection.
- Tout courriel de Grok Bot est relayé, réponse à une demande ou envoi spontané (choix de Claude :
  rien n'en est exécuté, le risque est le même, la règle est plus simple).

## Non-objectifs

- Exécuter quoi que ce soit venant de Grok : aucune commande `/`, aucun appel d'agent, aucun LLM
  sur le texte.
- Commande `/grok` (raccourci de `/ecrire`) : option à demander plus tard.
- Dépôt par Secretarius d'un document dans Publication : laissé en suspens (l'utilisateur
  transmet lui-même le fichier par l'interface de Grok Bot).
- Répondre automatiquement à Grok.

## Anti-objectifs (échec même si « ça marche »)

- Un texte de Grok qui déclenche une action de Secretarius.
- Un tiers qui se fait relayer en se faisant passer pour Grok (autre utilisateur de
  `mail.grokbot.com`, ou domaine usurpé).
- La phrase secrète Session (`SESSION_BOT_MNEMONIC`) lue par la relève ou placée dans son
  environnement.

## Données relevées (courriel d'essai de Rector, 2026-11-25 16:42 UTC)

- `From` = `Reply-To` = `yc28mybv@mail.grokbot.com` ; envoyé à `tiron@secretarius.co`.
- `Return-Path` : `<…@mail.mail.grokbot.com>` (Amazon SES), **différent** de l'expéditeur : la
  règle actuelle d'`authentifier` le refuserait toujours.
- `X-Pm-Origin: external` ; `Authentication-Results: mail.protonmail.ch; dmarc=pass (p=reject
  dis=none) header.from=mail.grokbot.com` ; `spf=pass smtp.mailfrom=mail.mail.grokbot.com` ;
  `dkim=pass … header.d=mail.grokbot.com` (et une seconde signature `amazonses.com`).
- Corps `text/plain; charset=utf-8`, 58 caractères ; `Message-ID` présent (`…@email.amazonses.com`).

## Décisions

### Expéditeurs relayés (configuration)

Dans `~/.config/tiron_mail/comptes.toml` :

```toml
[releve]
boite = "tiron"
relayes = ["yc28mybv@mail.grokbot.com"]
```

Une entrée `local@domaine` admet `local@domaine` et `local+<sélecteur>@domaine`, sélecteur en
`[a-z0-9-]+` (décision utilisateur 2026-11-25). Comparaison en minuscules, domaine **exact** :
ni `yc28mybvx@mail.grokbot.com`, ni `yc28mybv@autre.grokbot.com`, ni tout le domaine. Une entrée
qui est aussi une adresse du propriétaire → `ErreurConfig` (un expéditeur est soit commandant,
soit relayé). Sans `relayes`, comportement actuel inchangé.

Limite assumée : DMARC et DKIM authentifient le domaine, pas la partie locale ; on fait
confiance à Grok Bot pour qu'aucun autre de ses utilisateurs ne puisse écrire sous `yc28mybv`.
Le sélecteur n'élargit pas cette confiance.

### Authentification d'un relayé

Admis seulement si, en plus de la règle d'adresse :
- un seul `From` ;
- `X-Pm-Origin` = `external` ;
- au moins un `Authentication-Results` commençant par `mail.protonmail.ch;` portant `dmarc=…`,
  et tous ceux-là à `pass`, avec `header.from=` égal au domaine de l'expéditeur ;
- au moins un `dkim=pass` avec `header.d=` égal au domaine de l'expéditeur ;
- `Message-ID` présent.

Pas de contrôle `Return-Path` pour les relayés (SES le rend différent ; DMARC l'aligne déjà via
SPF/DKIM). Échec d'un seul point → refus exactement comme aujourd'hui (rangé dans
`Folders/Refuse`, avis « Courriel refusé… », motif précis). Les contrôles existants des courriels
du propriétaire ne changent pas.

### Contenu

- Partie `text/plain` de préférence, sinon `text/html` (nettoyée par le garde, `type: html`).
- Passage par le garde d'injection (`POST http://127.0.0.1:8990/check`), le même que pour
  `check_email` : regex + DeBERTa, **aucun LLM**.
  - `blocked` → relais de l'en-tête seul : expéditeur, objet, « contenu bloqué par le détecteur
    d'injection : <motif> ».
  - `risk: medium` → texte relayé, précédé de « Attention : contenu potentiellement suspect. »
    (même libellé que `/source`).
  - garde injoignable → en-tête seul + « contenu non vérifié (garde injoignable), consultez la
    boîte de Tiron ». Jamais de texte non vérifié sur Session.
- Texte raccourci à 3 500 caractères, avec « … (suite dans la boîte de Tiron) » si coupé (choix de
  Claude).
- Message Session :

  ```
  Courriel de Grok Bot (rector) — « <objet> »

  <texte>
  ```

  Le sélecteur entre parenthèses n'apparaît que s'il existe.
- Le message est ensuite rangé dans `Folders/Traite` (jamais relayé deux fois : la liste
  `traites` de l'état existant s'applique). La relève n'en tire **aucune commande**, même si le
  texte contient une ligne commençant par `/`.

### Envoi sur Session

- Directement au broker : `POST http://127.0.0.1:8091/send`, `Authorization: Bearer
  <TIRON_SESSION_BROKER_TOKEN>`, corps `{"to": <TIRON_SESSION_ID>, "text": …}`. Le broker refuse
  tout destinataire hors de sa liste blanche (403).
- Jeton et identifiant lus dans `~/.config/secrets.env` **ligne par ligne, seulement ces deux
  clés**, apostrophes retirées (même méthode que `veille_session.lire_jeton`, à réutiliser). Le
  service de relève garde son `EnvironmentFile` actuel (`Wiki_LM/.env`).
- Échec (broker arrêté, jeton absent, HTTP ≠ 200) → même texte envoyé par courriel d'avis
  (`notifier_courriel`, adresse `[envoi] avis`), préfixé « (Session indisponible) ». Si l'avis
  échoue aussi : stderr, message rangé quand même (jamais relayé deux fois, comme les réponses
  de la phase 2).

## Critères de réussite (essai réel)

1. Rector écrit à `tiron@secretarius.co` → dans les 2 min, message sur Session « Courriel de Grok
   Bot — « … » » avec le texte.
2. Un courriel contenant une ligne `/wikistatus` venant de Grok → relayé, rien n'est exécuté
   (journal de la relève, aucun appel d'agent).
3. Broker arrêté → le relais arrive par courriel d'avis.
4. Tests : faux `From` sur `mail.grokbot.com` avec DMARC `fail` → refus ; `yc28mybvx@…` → refus ;
   `yc28mybv+scriptor@…` authentifié → relayé avec « (scriptor) » ; garde `blocked` → en-tête
   seul ; garde injoignable → pas de texte.

## Laissé au constructeur

Découpage en fonctions et en tâches, emplacement exact des tests, forme de l'objet `Config`
(`relayes: tuple = ()`), libellés exacts du bilan de relève.
