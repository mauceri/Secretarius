# Design — FAQ de faits locaux répondue directement par Tiron

Date : 2026-07-08

## Objectif

Permettre à Tiron (assistant local, phi-4-mini sur iGPU) de répondre **directement
et localement** à des questions dont la réponse est consignée dans un fichier de
faits que l'utilisateur **édite à volonté** — aussi bien des faits sur Secretarius
lui-même (commandes, config) que des faits personnels arbitraires
(ex. « Le perroquet de Madame Michu s'appelle Coco »).

## Ce qui change par rapport au chantier « détection question-Secretarius »

Le chantier précédent (`2026-07-08-detection-question-secretarius-design.md`,
validé) détectait les questions *sur Secretarius* via un **centroïde** BGE-M3 et
répondait avec phi-4 nu + un document injecté en entier. Ce design le **remplace
pour l'intégration** (le code de validation `gen_corpus_qa/classify_secretarius.py`,
`SEUIL_SECRETARIUS`, etc. reste en place comme artefact, non intégré).

Raison : l'objectif s'est élargi aux **faits arbitraires et hétérogènes**. Un
centroïde (moyenne des embeddings) tombe dans un no man's land sémantique dès que
les faits sont hétérogènes → rappel structurellement faible. On passe à une
**recherche au plus proche voisin** (un vecteur par question connue, `max` cosinus),
qui reste fiable quelle que soit la dispersion des faits.

## Décisions actées

1. **Emplacement** : `~/Documents/Arbath/Wiki_LM/faits/faits.md` — dossier **frère
   de `raw/`** dans le vault Obsidian. Éditable dans Obsidian, **hors de l'index
   `/q`** (l'index ne lit que le sous-dossier `wiki/`, cf. `search.py:90`). Seul
   effet de bord, bénin : le watcher `server.py:203` (rglob récursif) déclenchera un
   rechargement de l'index `wiki/` à chaque édition (mêmes pages, ~30 s).
2. **Format** : markdown structuré, une entrée par fait. Un ou plusieurs titres `##`
   = formulations de la question (embarquées) ; le corps = la réponse injectée.
3. **Détection** : single-vector nearest-neighbor. Un vecteur BGE-M3 par question
   connue (modèle déjà chargé dans `router_service`) ; pour un message, `max` cosinus
   sur toutes les questions ; `≥ SEUIL_FAQ` → question de faits, entrée matchée connue.
4. **Ordre** : **FAQ d'abord**. Pour un message en langage naturel (sans `/`), la
   FAQ est consultée avant le routage commandes. Les commandes explicites `/…` ne
   sont jamais interceptées.
5. **Réponse** : phi-4 **nu** (adaptateur à scale 0 **par requête** sur le port
   prod 8998, sans toucher l'état global) + injection de **la seule entrée matchée**
   (contourne le plafond `-c 2048`).
6. **Sous le seuil** : « Je n'ai pas cette information. »
7. **Install** : seed `faits/faits.md` copié **seulement s'il n'existe pas**
   (non-clobber).

## Format du fichier `faits.md`

```markdown
## Comment s'appelle le perroquet de Madame Michu ?
## Le perroquet de Mme Michu ?
Le perroquet de Madame Michu s'appelle Coco.

## Quelle commande pour interroger le wiki ?
Utilisez /q <question>.
```

- Une **entrée** = un ou plusieurs `##` consécutifs (formulations) suivis d'un corps
  (jusqu'au `##` suivant ou la fin).
- Chaque formulation `##` est embarquée séparément et pointe vers le même corps.
- Le rappel d'un fait croît avec le nombre de formulations fournies — inhérent au
  modèle choisi (l'utilisateur curate autant de formulations qu'il souhaite).

## Composants et interfaces

### 1. `router_service/faq.py` (nouveau)

Responsabilité unique : charger `faits.md`, l'embarquer, et retrouver l'entrée la
plus proche d'un message.

- `parse_faq(text) -> list[Entry]` où `Entry = {questions: list[str], answer: str}`.
  **Garde-fou par-entrée** : une entrée dont le corps dépasse `FAQ_MAX_ENTREE`
  caractères (défaut 2000, très au-dessus d'un fait normal) est **écartée avec un
  avertissement** dans les logs — évite qu'une entrée pathologiquement longue
  déborde silencieusement le contexte `-c 2048` et produise une réponse tronquée.
- `class FaqIndex(embed_fn, path, seuil)` :
  - construit à l'init : parse + embarque toutes les questions (via `embed_fn`,
    = `GogGate._embed`, même BGE-M3, pas de second modèle) ; stocke une matrice
    `[N_questions, 1024]` + une correspondance question→entrée.
  - `lookup(message) -> Entry | None` : recharge si `mtime` du fichier a changé
    (cache par mtime) ; `max` cosinus ; renvoie l'entrée si `≥ seuil`, sinon `None`.
- `SEUIL_FAQ` : constante, surchargeable par variable d'environnement `FAQ_SEUIL`.
  Valeur de départ **0.6** (cosinus BGE-M3 CLS ; à calibrer sur de vrais messages).
- Fichier absent ou vide → index vide, `lookup` renvoie toujours `None` (dégradation
  silencieuse, jamais d'exception qui casserait le routage).

### 2. Réponse phi-4 nu — inférence à lora par-requête

`router_service` gagne une fonction d'inférence qui envoie
`"lora": [{"id": 0, "scale": 0}]` **dans le corps de la requête** vers le
`/v1/chat/completions` de 8998 (confirmé accepté par le build ROCm en place).
**Ne pas** utiliser `set_lora_scale` (POST `/lora-adapters`, état global) : cela
casserait le routage concurrent. Le contexte injecté = la question + le corps de
l'entrée matchée uniquement.

### 3. `router_service/server.py:route_message` (modifié)

```python
def route_message(message):
    if not message.lstrip().startswith("/"):
        entry = _faq.lookup(message)
        if entry is not None:
            return {"status": "answer", "reply": _answer(message, entry)}
    # ... routage commandes existant, INCHANGÉ ...
```

`_faq` (FaqIndex) est construit au démarrage à côté de `GogGate`, en réutilisant
`_gate._embed` comme `embed_fn`. Nouveau statut de sortie : `{"status": "answer",
"reply": <texte>}`.

### 4. `derisk-deleg/src/index.ts` (modifié)

- `callRouter` : propager le champ `reply` quand `status === "answer"`.
- Hook `before_agent_reply` : si `routed.status === "answer"` → répondre
  `routed.reply` directement (comme les autres branches, `handled: true`).
- Branche `no_match` : message reformulé en « Je n'ai pas cette information
  (essayez /q pour le wiki). » (couvre le cas « sous le seuil FAQ **et** aucune
  commande »).

### 5. `install.sh` (modifié)

Créer `~/Documents/Arbath/Wiki_LM/faits/` et y copier le seed
**uniquement s'il n'existe pas** (non-clobber). Source du seed dans le dépôt :
`amorçage/faits.md` (versionné) → cible live éditable :
`~/Documents/Arbath/Wiki_LM/faits/faits.md`. Le seed contient des entrées
représentatives sur Secretarius (commandes wiki/gog, config machine), au format
`##`-entrées.

## Flux (message en langage naturel, sans slash)

```
message ── derisk-deleg hook ── POST /route (8999)
                                      │
                          route_message: FAQ d'abord
                          ┌───────────┴───────────┐
              max-sim ≥ seuil                 max-sim < seuil
                    │                               │
        phi-4 nu (scale 0 par-req)        routage commandes existant
        + entrée matchée injectée          (adaptateur + portail gog)
                    │                               │
        {status:"answer", reply}         wiki/gog/scout  ou  no_match
                    │                               │
        hook répond reply           délégation  ou  « Je n'ai pas cette information »
```

## Gestion d'erreurs

- Routeur (8999) indisponible → message existant « Routeur local indisponible ».
- `faits.md` absent/vide → `lookup` renvoie `None`, on retombe sur le routage
  commandes (aucune régression).
- Erreur d'inférence phi-4 (timeout, crash 8998) → renvoyer un message d'échec
  explicite, ne jamais lever d'exception non capturée dans `route_message`.

## Tests

- `parse_faq` : entrées mono/multi-formulations, corps multi-lignes, fichier vide.
- `FaqIndex.lookup` : match au-dessus du seuil renvoie la bonne entrée ; en-dessous
  renvoie `None` ; rechargement sur changement de mtime. (BGE-M3 réel ou embeddings
  injectés selon le pattern des tests existants du routeur.)
- `route_message` : commande explicite `/…` **court-circuite** la FAQ ; message
  libre proche d'une entrée → `status:"answer"` ; message libre non couvert →
  chemin routage commandes inchangé.
- Inférence : la requête envoyée contient bien `lora:[{id:0,scale:0}]` (mock HTTP).
- `derisk-deleg` : `status:"answer"` → `reply` relayé tel quel (test TS existant).

## Déploiement (sanroque, prod `~/.openclaw`)

1. Créer `faits/` + seed via `install.sh` (non-clobber).
2. Déployer `router_service` (nouveau `faq.py` + `server.py` modifié) ; **redémarrer
   le service routeur (8999)**.
3. Rebuild + réinstaller le plugin `derisk-deleg` ; **redémarrer openclaw-gateway**.
4. E2E Telegram : ajouter un fait → poser la question → vérifier la réponse ;
   question inconnue → « Je n'ai pas cette information » ; vérifier que `/q` et les
   commandes continuent de fonctionner.

## Points à calibrer / limites

- `SEUIL_FAQ` (départ 0.6) est un seul hyperparamètre à **revalider sur de vrais
  messages Telegram**, réglable par `FAQ_SEUIL` sans redéploiement.
- Chaque message libre proche d'une entrée déclenche une génération phi-4 (quelques
  secondes + charge iGPU) ; le seuil borne cela aux vraies correspondances.
- Bug de stabilité connu du `llama-server` ROCm sous charge — risque déjà accepté.
- **Taille de la FAQ** : pas de plafond dur (seule l'entrée matchée est injectée,
  donc le mur `-c 2048` ne concerne que la taille d'une entrée, bornée par le
  garde-fou ci-dessus). À très grand N (plusieurs milliers d'entrées), le
  ré-embarquement déclenché à chaque édition devient perceptible (secondes) —
  optimisable plus tard par embarquement incrémental. YAGNI pour l'instant.
- Le centroïde `secretarius` validé est **superseded** par ce design pour
  l'intégration ; son code de validation reste en place, non branché.
