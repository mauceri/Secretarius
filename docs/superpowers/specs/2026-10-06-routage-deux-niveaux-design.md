---
tags: [spec, secretarius, routeur, laya, v1]
date: 2026-10-06
statut: à relire
---

# Spec : routage Laya à deux niveaux, avec demande de précision

## Pourquoi

Deux réentraînements du classifieur à 18 choix (2026-10-06) plafonnent à 88,5 % puis 86,5 %.
Les causes sont mesurées :

- **Troncature.** Les 18 options partagent un budget de 256 jetons (`head_max_len`,
  `laya/common.py:187-193`) et chacune est coupée à 13 jetons. Le README de Laya et
  `laya/shortlist.py` recommandent de procéder du général au particulier.
- **Ambiguïtés d'étiquetage.** Le hors périmètre « culture générale » se confond avec `/q`
  (13 erreurs). `/source` se confond avec `/c` (8). `/lire` sans identifiant se confond avec `/chercher` (6).
- **Prédictions erratiques.** Des `/supprimer` sortent sur des messages sans rapport (8).

Objectif de l'utilisateur : « quelque chose qui marche sans que l'on ait à revenir ». La
fiabilité ne doit donc pas dépendre d'un classifieur parfait. Quand le modèle hésite, Tiron
pose une question au lieu de deviner.

## Décisions

| Sujet | Décision |
|---|---|
| Structure | Un seul modèle Laya, deux questions : `domaine` puis `commande` dans le domaine |
| Domaines | `wiki`, `google` (courriel + Drive), `web`, `planification`, `aucune` |
| Hésitation | Demande de précision avec les 2 meilleures options, réponse « 1 » ou « 2 » |
| Questions générales | Elles vont vers `/q` (le wiki répond ou dit qu'il n'a rien) ; le hors périmètre est réservé aux actions impossibles et à la conversation |
| Validation | Jeu de validation ≥ 40 exemples par commande, plus un jeu de test écrit à la main (pas par DeepSeek) |
| Seuil d'acceptation | Sur le jeu manuel : ≥ 90 % de décisions justes ou de questions pertinentes, aucune commande < 80 %, ≤ 15 % de demandes de précision |
| Modèle | Même nom `Mauceric/laya-tiron-router`, pas de gestion de versions |
| Commandes tapées (`/xxx`) | Inchangées, jamais soumises à Laya |

## Architecture

```
message libre
   │
   ▼
Laya, question « domaine » (5 choix)
   │ p_dom
   ├── web            → /source
   ├── planification  → /cron
   ├── aucune         → « Je n'ai pas cette information… »
   ├── wiki    → Laya, question « commande_wiki » (9 choix)    │ p_cmd
   └── google  → Laya, question « commande_google » (6 choix)  │
   │
   ▼
confiance = p_dom × p_cmd
   ├── ≥ SEUIL → {status: ok, command, args}
   └── <  SEUIL → {status: clarify, options: [2 meilleures (commande, libellé)], args}
```

Les deux meilleures options sont classées par probabilité jointe sur les deux domaines les
plus probables. Le cas « aucune » peut figurer parmi les deux options proposées, avec le
libellé « rien de tout ça ».

### Domaines et commandes

| Domaine | Commandes (niveau 2) |
|---|---|
| `wiki` | `/c`, `/q`, `/ingest`, `/wikistatus`, `/tags`, `/kbupdate`, `/supprimer`, `/relire`, `/verifie` |
| `google` | `/chercher`, `/connecter`, `/inbox`, `/drive`, `/repondre`, `/lire` |
| `web` | `/source` (pas de niveau 2) |
| `planification` | `/cron` (pas de niveau 2) |
| `aucune` | — |

Le futur agent courrier ne touchera qu'au domaine `google` : le niveau 1 reste valable.

### Demande de précision (derisk-deleg)

- Le routeur renvoie `{"status": "clarify", "options": [{"command", "label"}, …], "args"}`.
- derisk-deleg répond, par exemple : « Je ne suis pas sûr. 1) lire un email précis
  2) chercher dans les emails. Répondez 1 ou 2. » Il met en attente
  `pending = {kind: "choix", options, args, ts}`, avec le même délai de 10 minutes que `/confirm`.
- Une réponse « 1 » ou « 2 » exécute la commande choisie avec les `args` d'origine, comme un
  routage `ok`. Les règles existantes s'appliquent ensuite : `/confirm` pour les écritures,
  aperçu pour `/cron`.
- Toute autre réponse annule l'attente, et le message est traité normalement.

### Seuil

`SEUIL` est choisi à l'évaluation, sur le jeu de validation, de façon à obtenir :
- au moins 97 % de décisions justes parmi celles prises sans question ;
- au plus 15 % de demandes de précision.

Il remplace `SEUIL_GOG` (0,50). Sa valeur est consignée dans `router_service/server.py` avec
la mesure qui l'a fixée.

## Données

- `to_laya_format.py` écrit, pour chaque exemple, la question `domaine`. Il ajoute
  `commande_wiki` ou `commande_google` selon le domaine, chacune avec ses propres critères
  (descriptions complètes, sans troncature).
- **Hors périmètre** : la description du juge et l'instruction du générateur excluent les
  questions de connaissance générale. Les exemples `out_of_scope` sont régénérés (environ 170).
  Une variante « question de culture générale » est ajoutée à `wiki_query` (environ 60 exemples).
- **Validation** : la répartition passe à ≥ 40 exemples par commande (`to_laya_format.py`,
  découpage stratifié).
- **Jeu de test manuel** `router_service/test_set_manuel.jsonl` : environ 150 messages écrits
  à la main, dans le style des messages Telegram réels, ambiguïtés comprises (« le dernier
  mail de ma sœur », « astuce pr tomates ? », « lis ça <url> »). Chaque message porte la
  commande attendue, ou l'ensemble des commandes acceptables s'il est ambigu. L'utilisateur
  peut en ajouter.

## Entraînement (Kaggle)

- Un seul réentraînement, publié sous le même nom.
- Le notebook fabrique un exemple d'entraînement par question (`for qid, q in questions.items()`).
  Il faut vérifier que la version de l'utilisateur a gardé cette boucle.
- La cellule d'évaluation du notebook attend une question `command` et ne fonctionnera plus.
  L'évaluation qui fait foi est faite sur sanroque (`eval_laya.py`, exactitude par niveau,
  jeu manuel, taux de demandes de précision). Une cellule de remplacement, prête à coller,
  sera fournie.

## Tests

- `router_service` : décision à deux niveaux (classifieur simulé), `clarify` sous le seuil,
  classement des deux meilleures options, commandes tapées inchangées.
- `to_laya_format` : questions par domaine, découpage stratifié.
- derisk-deleg : `clarify` → message numéroté → « 1 » exécute, « 2 » exécute, autre texte
  annule, expiration.
- `eval_laya.py` : exactitude par niveau, taux de demandes de précision, score sur le jeu manuel.

## Hors périmètre de cette spec

- Sous-agents OpenClaw par domaine pour l'exécution : inchangé (agents wiki et gog existants).
- Remplacement de gog par l'agent courrier : chantier suivant. Seul le domaine `google` sera
  réentraîné.
