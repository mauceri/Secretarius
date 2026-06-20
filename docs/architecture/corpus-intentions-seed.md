# Corpus d'intentions — graine pour corpus synthétique (Tiron)

> **But.** Amorcer un corpus synthétique pour entraîner un petit modèle
> (Phi-4-mini, petit Qwen) à **reconnaître l'intention** d'un message
> utilisateur et la router vers la bonne **commande → skill → agent**.
> Ce fichier contient ~10 intentions × ~20 messages d'exemple (~200 au total),
> à **augmenter** ensuite par paraphrase/variation (voir §Augmentation).

## Modèle de routage visé

Deux voies (cf. architecture par intention) :

- **Voie commande** : un message commençant par `/x` invoque **déterministement**
  le skill `x` (aucune décision du modèle). Inclus ici car les utilisateurs tapent
  aussi la forme slash.
- **Voie conversation** : un message en langage naturel est **classé** en une
  intention, qui se **résout en une commande nommée**. Pour les intentions
  d'**action** (mail, agenda, drive, ingest…), l'exécution est **précédée d'une
  confirmation**. Le classifieur ne fait que l'**intention** ; l'action est gardée.

## Étiquettes et cibles de routage

| Intention | Commande | Agent / traitement |
|-----------|----------|--------------------|
| `wiki_capture` | `/c` | agent `wiki` (op `capture`) |
| `wiki_ingest` | `/ingest` | agent `wiki` (op `ingest`, file d'attente) |
| `wiki_status` | `/wiki-status` | agent `wiki` (op `status`) |
| `wiki_query` | `/q` | agent `wiki` (op `query`) |
| `source_read` | `/source` | agent `scout` (lecture externe anti-injection) |
| `gog_mail` | `/mail` | `gog` (email) |
| `gog_calendar` | `/agenda` | `gog` (calendrier) |
| `gog_drive` | `/drive` | `gog` (fichiers) |
| `meta_assistant` | `/help` | orchestrateur Tiron (réponse directe) |
| `out_of_scope` | — | refus déterministe « pas de skill pour cette demande » |

---

## 1. `wiki_capture` — enregistrer une URL/note dans le wiki (sans lire ni ingérer)

1. `/c https://marp.app/#get-started`
2. `/c #markdown #presentation https://marp.app/#get-started`
3. garde cet article pour moi : https://example.com/article
4. ajoute ça à mon wiki : https://arxiv.org/abs/1706.03762
5. capture cette page https://en.wikipedia.org/wiki/Zettelkasten
6. mets de côté ce lien #ia https://openai.com/research
7. note pour plus tard ce lien : https://www.lemonde.fr/un-article
8. enregistre cette URL dans ma base
9. à classer dans le wiki : https://exemple.org/page
10. `/c #productivité` un truc intéressant sur la prise de notes https://exemple.org
11. range ce lien dans mes sources
12. capture #histoire https://fr.wikipedia.org/wiki/Big_Sur
13. j'aimerais conserver cette page : https://exemple.org/x
14. garde cette adresse, je l'ingérerai plus tard : https://exemple.org/y
15. capture-moi ça stp : https://exemple.org/z
16. `/c` note : penser à relire l'article sur BM25
17. sauvegarde ce lien #nlp https://aclanthology.org/abc
18. mets ce papier dans le wiki https://arxiv.org/abs/2106.00001
19. pour mémoire, ce lien : https://exemple.org/memo
20. clip ça dans ma base de connaissances : https://exemple.org/clip

## 2. `wiki_ingest` — traiter la file des captures en attente (sans argument)

1. ingère les fichiers en attente
2. lance l'ingestion
3. `/ingest`
4. traite la file d'attente du wiki
5. ingère ce qui est en attente
6. peux-tu lancer l'ingestion maintenant ?
7. traite les captures en attente
8. démarre le traitement des sources non ingérées
9. ingestion stp
10. fais l'ingestion des fichiers capturés
11. process la file wiki
12. lance le worker d'ingestion
13. intègre les documents en attente dans le wiki
14. on peut ingérer maintenant ?
15. vide la file d'ingestion
16. ingère tout ce qui traîne
17. déclenche l'ingestion des captures
18. passe les fichiers en attente à l'ingestion
19. traite les `.url` en attente
20. lance l'indexation des sources capturées

## 3. `wiki_status` — état de l'ingestion / de la file

1. où en est l'ingestion ?
2. `/wiki-status`
3. quel est le statut du wiki ?
4. combien de fichiers en attente ?
5. l'ingestion est-elle terminée ?
6. y a-t-il des fichiers bloqués ?
7. statut de la base de connaissances
8. est-ce que l'ingestion tourne en ce moment ?
9. donne-moi l'état du traitement wiki
10. quel est l'avancement de l'ingestion ?
11. il reste des choses à ingérer ?
12. le wiki est à jour ?
13. état de la file d'ingestion
14. est-ce que tout a été ingéré ?
15. combien d'erreurs sur la dernière ingestion ?
16. dernier run d'ingestion, ça a donné quoi ?
17. y a-t-il un traitement en cours côté wiki ?
18. fais le point sur l'ingestion
19. statut ingestion stp
20. liste les fichiers en échec

## 4. `wiki_query` — interroger la base de connaissances

1. que dit le wiki sur l'attention ?
2. `/q comment fonctionne le Zettelkasten ?`
3. cherche dans ma base : qu'est-ce que BM25 ?
4. d'après mes notes, qui est Niklas Luhmann ?
5. interroge le wiki sur les transformeurs
6. qu'ai-je sauvegardé à propos de Marp ?
7. résume ce que ma base dit sur la productivité
8. dans mon wiki, qu'est-ce qui parle de Big Sur ?
9. retrouve mes notes sur l'ingestion de documents
10. que sais-je déjà sur la prise de notes ?
11. fais une synthèse de mes sources sur le RAG
12. cherche « confidential computing » dans le wiki
13. quelles pages parlent d'anti-injection ?
14. d'après le wiki, c'est quoi un MoE ?
15. interroge ma base sur les adaptateurs LoRA
16. qu'est-ce que j'ai noté sur Qwen ?
17. `/q` liste mes sources sur le markdown
18. dans mes documents, y a-t-il quelque chose sur Scout ?
19. résume mes notes sur les TEE
20. que contient mon wiki au sujet de la productivité cognitive ?

## 5. `source_read` — lire/consulter une page externe MAINTENANT (via Scout, sans sauvegarder)

1. que dit cette page : https://example.com ?
2. `/source https://news.ycombinator.com`
3. résume-moi cet article : https://exemple.org/blog
4. va lire cette page et dis-moi l'essentiel : https://exemple.org/x
5. de quoi parle ce lien ? https://exemple.org/y
6. peux-tu consulter cette URL et me résumer ? https://exemple.org/z
7. lis https://example.com et donne-moi les points clés
8. qu'y a-t-il sur cette page ? https://exemple.org/page
9. extrais l'info principale de https://exemple.org/a
10. ouvre ce lien et résume https://exemple.org/b
11. c'est quoi le contenu de https://exemple.org/c ?
12. donne-moi un aperçu de cette page https://exemple.org/d
13. `/source` résume cet article de blog https://exemple.org/e
14. vérifie ce que raconte cette page https://exemple.org/f
15. je veux savoir ce que dit https://exemple.org/g sans le garder
16. parcours cette URL et fais-moi un résumé https://exemple.org/h
17. dis-moi en deux lignes ce que contient https://exemple.org/i
18. cette page parle de quoi ? https://exemple.org/j
19. consulte https://exemple.org/k pour moi
20. lis-moi cet article (juste lire, pas l'enregistrer) https://exemple.org/l

## 6. `gog_mail` — courrier électronique

1. envoie un mail à Paul pour décaler la réunion
2. `/mail` lis mes derniers mails
3. ai-je reçu un message de la banque ?
4. réponds à Marie que c'est d'accord
5. écris un mail à l'équipe pour annoncer la livraison
6. y a-t-il des mails non lus ?
7. cherche le mail de l'assurance
8. transfère ce message à Jean
9. envoie un courriel à contact@exemple.org avec le compte-rendu
10. résume mes mails d'aujourd'hui
11. `/mail` rédige une relance pour le devis
12. relève ma boîte
13. qui m'a écrit ce matin ?
14. réponds à ce fil que je confirme ma présence
15. envoie à Sophie les coordonnées du restaurant
16. ai-je un mail d'Infomaniak ?
17. classe ce mail dans « factures »
18. écris à mon comptable pour le rendez-vous
19. montre-moi le dernier mail de Christian
20. envoie un mot de remerciement à l'organisateur

## 7. `gog_calendar` — agenda

1. qu'est-ce que j'ai demain ?
2. `/agenda` ajoute « dentiste » jeudi 15h
3. crée un rendez-vous avec Paul lundi matin
4. suis-je libre vendredi après-midi ?
5. déplace ma réunion de 10h à 14h
6. quel est mon programme de la semaine ?
7. ajoute un créneau « sport » tous les mardis 18h
8. annule mon rendez-vous de demain
9. quand ai-je mon prochain rendez-vous médical ?
10. bloque deux heures jeudi pour la présentation
11. `/agenda` mes événements d'aujourd'hui
12. planifie un appel avec l'équipe mercredi 11h
13. ai-je un conflit lundi entre 9h et 11h ?
14. rappelle-moi l'heure de la réunion de demain
15. crée un événement « anniversaire » le 20 juin
16. qu'y a-t-il à mon agenda ce week-end ?
17. décale tous mes rendez-vous d'une heure
18. invite Marie à la réunion de jeudi
19. réserve mon vendredi matin
20. liste mes rendez-vous de la semaine prochaine

## 8. `gog_drive` — fichiers / Drive

1. retrouve le document « budget 2026 » sur mon drive
2. `/drive` liste mes fichiers récents
3. partage le fichier « présentation » avec Paul
4. crée un dossier « Secretarius » sur le drive
5. où est le PDF du contrat ?
6. télécharge le tableur des dépenses
7. cherche sur mon drive les fichiers contenant « facture »
8. envoie-moi le lien du document partagé
9. `/drive` ouvre le dernier fichier modifié
10. fais une copie de ce document
11. quels fichiers ai-je ajoutés cette semaine ?
12. déplace ce fichier dans le dossier « archives »
13. donne accès en lecture à Marie sur ce dossier
14. retrouve mes notes de réunion sur le drive
15. supprime le fichier « brouillon »
16. liste le contenu du dossier « projets »
17. quel est le lien de partage de ce document ?
18. cherche la dernière version du rapport
19. crée un document vierge « compte-rendu »
20. montre-moi les fichiers partagés avec moi

## 9. `meta_assistant` — sur l'assistant lui-même (réponse directe de Tiron)

1. qui êtes-vous ?
2. quel modèle vous anime ?
3. que savez-vous faire ?
4. quelles sont vos commandes ?
5. `/help`
6. comment je capture une page dans le wiki ?
7. peux-tu m'expliquer ce que tu sais faire ?
8. à quoi sers-tu ?
9. tu tournes sur quel LLM ?
10. quelles intentions comprends-tu ?
11. bonjour
12. comment fonctionne l'ingestion ici ?
13. liste tes capacités
14. quelle est la différence entre capturer et ingérer ?
15. es-tu connecté à mon agenda ?
16. `/commands`
17. tu peux accéder à mes mails ?
18. quelles sont tes limites ?
19. comment te donner une URL à lire ?
20. merci, c'est noté

## 10. `out_of_scope` — demandes sans skill correspondant (refus déterministe)

1. commande une pizza
2. réserve un billet de train pour Lyon
3. allume les lumières du salon
4. passe un appel à Paul
5. envoie un SMS à Marie
6. quel temps fera-t-il demain ?
7. mets de la musique
8. achète ce produit sur Amazon
9. règle un minuteur de 10 minutes
10. fais un virement de 50 € à Jean
11. poste ce message sur les réseaux sociaux
12. imprime ce document
13. lance une recherche Google sur les LLM
14. prends une photo
15. ajoute un rappel sur mon téléphone
16. transcris ce fichier audio
17. réserve une table au restaurant
18. quelle heure est-il à Tokyo ?
19. télécharge cette vidéo YouTube
20. traduis et envoie ce contrat par fax

---

## Frontières / paires confusables (à sur-représenter dans l'augmentation)

Les erreurs viennent des **frontières**. Inclure beaucoup d'exemples
**contrastifs** sur ces axes :

- **`source_read` vs `wiki_capture`** : tous deux contiennent une URL. Signal
  distinctif = le **verbe d'intention** : *lire/consulter/résumer maintenant*
  (source) vs *garder/capturer/ajouter au wiki pour plus tard* (capture).
  Ex. « résume cette page <url> » → `source_read` ; « garde cette page <url> » → `wiki_capture`.
- **`wiki_capture` vs `wiki_ingest`** : capture = **une** URL/note précise ;
  ingest = traiter **toute la file** (jamais d'URL spécifique). « ajoute ce lien »
  → capture ; « ingère ce qui est en attente » → ingest. **Ne jamais** fusionner
  les deux (un `/c` ne déclenche jamais d'ingestion).
- **`gog_mail` / `gog_calendar` / `gog_drive`** : distinguer par l'**objet** —
  message/courrier vs rendez-vous/créneau vs fichier/document/dossier.
- **`meta_assistant` vs `out_of_scope`** : meta = question **sur l'assistant**
  (capacités, modèle, commandes) ; out_of_scope = **action réelle** réclamant
  une capacité absente.
- **URL nue sans verbe** (« https://exemple.org » seule) : cas **ambigu**
  (capturer ? lire ?). C'est une **décision de politique** à figer puis à encoder
  dans le corpus (p. ex. défaut = `wiki_capture`, ou demander confirmation).
  Inclure des exemples étiquetés selon la politique retenue.

## Augmentation (du seed vers le corpus)

Pour chaque message graine, générer des variantes :

- **Registre** : vouvoiement / tutoiement, formel / familier, poli / télégraphique.
- **Paraphrase** lexicale et syntaxique (synonymes, ordre des mots, ellipses).
- **Bruit réaliste** : fautes de frappe, accents manquants, ponctuation absente,
  abréviations (« stp », « rdv », « cr »).
- **Slots variables** : remplacer URL / #tags / noms de contacts / dates / titres
  de fichiers par un catalogue de valeurs (gabarits `<url>`, `<contact>`, `<date>`,
  `<fichier>`, `<tags>`).
- **Contexte** : préfixer parfois par une phrase de contexte ou un tour précédent.
- **Longueur** : versions courtes et versions verbeuses du même besoin.
- **Distracteurs** : messages multi-intentions (à étiqueter sur l'intention
  principale, ou à marquer « clarification nécessaire »).

Viser un **équilibre des classes**, et **sur-échantillonner les paires
confusables** ci-dessus. Conserver un **jeu de test** tenu à l'écart.

## Conversion en paires d'entraînement (JSONL)

Format cible, une ligne par exemple :

```json
{"text": "garde cet article pour moi : https://example.com", "intention": "wiki_capture"}
{"text": "résume-moi cet article : https://example.com", "intention": "source_read"}
{"text": "ingère les fichiers en attente", "intention": "wiki_ingest"}
{"text": "commande une pizza", "intention": "out_of_scope"}
```

Optionnel : extraire aussi des **slots** (`url`, `tags`, `contact`, `date`,
`fichier`) pour un second objectif d'entraînement (NER / remplissage de
paramètres de commande). L'intention reste la cible principale ; l'action,
elle, demeure **gardée par confirmation** côté application.
```
