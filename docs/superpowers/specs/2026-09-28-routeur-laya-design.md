# Remplacement du routeur Tiron (phi-4-mini+LoRA) par Laya

- Date : 2026-09-28
- Statut : design validé, en attente d'exécution
- Auteur : Christian Mauceri + Claude

## Problème

Le routeur Tiron (`router_service/server.py`) classe le texte libre en
commande via un appel génératif à phi-4-mini+LoRA servi par llama.cpp
(port 8998, `call_adapter()`) : le modèle produit un JSON
`{"command": ..., "args": ...}` par décodage token-par-token, contraint par
un schéma JSON pour éviter le texte parasite. Un second mécanisme
indépendant, `GogGate` (`router_service/router.py`), calcule une similarité
à 3 centroïdes BGE-M3 (wiki/gog/hors-sujet) comme garde-fou de confiance
avant d'exécuter une commande gog — deux systèmes de confiance empilés,
aucun des deux nativement calibré (une probabilité de 70 % n'a aucune
garantie de correspondre à 70 % de bonnes réponses).

Une classe de modèles distincte existe pour ce problème précis — classer
parmi un nombre fixe d'options avec probabilité calibrée, en une seule passe
non-autorégressive (pas de décodage token par token) : les « System One
Models » (Jev, TypeSafe AI, propriétaire) et leurs implémentations ouvertes.
**Laya** (github.com/NandhaKishorM/laya, Apache 2.0) est retenue :
multilingue nativement (mmBERT-base 322M, pertinent pour un routeur
francophone), fine-tuning gratuit sur 2× T4 Kaggle, format de données
directement compatible avec le corpus d'entraînement déjà constitué.

Hors périmètre de ce chantier, décidé explicitement :
- **Extraction d'argument** — reste une heuristique pure (texte après la
  commande explicite, ou message entier pour une commande inférée en
  langage naturel), comme le fait déjà `op_capture` par regex. Aucun modèle
  d'extraction (GLiFormer) construit par anticipation ; à réserver si un
  vrai cas d'échec de l'heuristique apparaît.
- **Nouvelles commandes** (ajout au routeur) — reporté à une V2 avec Modal,
  non traité ici.
- **FAQ** (`router_service/faq.py`) — système indépendant, inchangé.

## Décisions

**Remplacement de `call_adapter()`** — l'appel HTTP génératif vers
phi-4-mini:8998 est remplacé par un appel en process à un modèle Laya
fine-tuné, chargé une fois au démarrage de `router_service` (même pattern
que le chargement BGE-M3 actuel — pas de nouveau service HTTP séparé, le
modèle est assez léger, 322-421M paramètres, pour vivre dans le même
process). `route_message()` garde sa structure trois voies (commande
explicite / FAQ / texte libre), seule la troisième voie change de moteur.

**`args` toujours par heuristique, jamais généré** — commande explicite :
`parts[1]` après le mot de commande (inchangé). Commande inférée en langage
naturel : le message entier, brut, sans reformulation. Simplification
délibérée par rapport à l'existant (qui laissait phi-4-mini reformuler
l'argument en langage naturel) : les fonctions `op_*` en aval savent déjà
traiter une question ou un texte non reformulé.

**Suppression de `GogGate`** — le garde-fou de confiance gog par centroïdes
BGE-M3 disparaît, remplacé par un seuil sur la probabilité calibrée que
Laya attribue à son choix (un seul mécanisme de confiance, plus la
dépendance BGE-M3 propre au routage). Le seuil de départ reprend la valeur
actuelle de `SEUIL_GOG` (0.50), affinable ensuite via la calibration de
température déjà outillée dans le notebook Laya.

**`FaqIndex` découplé de `GogGate`** — aujourd'hui `FaqIndex` réutilise
l'instance BGE-M3 de `GogGate` (`FaqIndex(_gate._embed)`). Pour ne rien
casser côté FAQ (hors périmètre), extraire une petite fonction
`embed_bge_m3()` autonome que `FaqIndex` continue d'utiliser, indépendante
de la logique de centroïdes qui, elle, disparaît avec `GogGate`.

**Données** — le corpus existant (`gen_corpus/corpus_lora_train.jsonl`,
2509 exemples, format ChatML : system/user/assistant JSON) se reformate par
script vers le schéma Laya (`state` = contenu du message `user`,
`questions.command` = choix parmi les ~17 commandes connues + une option
« aucune » pour les cas hors-sujet, `answers.command` = commande extraite
du JSON `assistant`, ou « aucune » si `command: null`). Pas de collecte de
données neuve — seule la mise en forme change. Le champ `args` du corpus
existant n'est pas utilisé pour l'entraînement (décision ci-dessus : plus
d'extraction générée).

**Entraînement** — notebook Laya (`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`)
sur Kaggle, 2× T4 gratuits, hors de toute machine Secretarius. Calibration
des températures incluse dans le notebook.

**Évaluation et bascule** — validation hors-ligne sur le corpus de test
existant (déjà utilisé pour mesurer le LoRA actuel). Si le résultat est du
même ordre de grandeur ou meilleur : remplacement direct sur sanroque (dev),
pas de mode ombre. Santiago (prod) suit une fois validé sur sanroque, via
`ob`/déploiement habituel (hors périmètre du détail ici — machine
actuellement effacée en vue d'une réinstallation grandeur nature, cf.
mémoire projet du 2026-09-28).

**Décommissionnement de phi-4-mini/llama.cpp (port 8998)** — confirmé
n'être utilisé aujourd'hui que pour le routage (la piste « réponse libre au
no_match » n'a jamais été implémentée, seulement documentée comme
proposition). Le service devient entièrement décommissionnable une fois
Laya validé en production — à traiter comme une étape de nettoyage
explicite dans le plan d'implémentation, pas silencieusement.

## Reste à vérifier pendant l'implémentation

- Exécution de Laya en CPU pur sur sanroque (pas de test de charge fait
  ici, seulement une estimation de faisabilité — dizaines à basses
  centaines de ms/requête attendues pour un encodeur de cette taille).
  Utiliser l'extra `laya[onnx]` plutôt que PyTorch brut pour l'inférence.
- Liste exacte des ~17 commandes et leurs `criteria` (description courte
  par commande) pour le schéma `questions.command` — à dériver de
  `WIKI_CMDS`/`GOG_CMDS` (`router_service/router.py`) et des descriptions
  déjà utilisées ailleurs (ex. texte d'aide `/help` côté wiki).
- Dépendances `torch 2.14+`/`transformers 5.x` : compatibilité avec
  l'environnement Python existant de `router_service` à vérifier avant
  d'installer.
