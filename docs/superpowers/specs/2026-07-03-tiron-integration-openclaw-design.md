# Intégration du routeur Tiron local (SLM + adaptateurs) dans OpenClaw — Design

## Objectif

Câbler dans OpenClaw, sur sanroque, le routeur SLM validé en prototype
(`router_3way.py` + `prototype_tiron_v3.py`, scratchpad session 2026-06-30/07-02),
**révisé le 2026-07-04** vers un adaptateur unique (voir décision ci-dessous) :
extraction JSON `{command, args}` par un adaptateur LoRA unique sur phi-4-mini,
classification BGE-M3 à 3 centroïdes (wiki / gog / hors_perimetre) conservée
uniquement comme portail de confiance sur gog, dispatch réel vers les
sous-agents existants (wiki, gog, scout).

**Décision 2026-07-04 — adaptateur unique, pas de hot-swap.** Le tout premier
test compilé validé (« adaptateur routeur », mémoire
`reference_context_economy_research`) couvrait déjà tout le périmètre en un
seul adaptateur et atteignait 94,5%. Le découpage ultérieur en 2 adaptateurs
de domaine (wiki/gog) n'était pas motivé par l'exactitude mais par la
sécurité (seuil de confiance asymétrique sur gog). On revient donc à **un
adaptateur unique** (tous les commandes wiki+gog+null), ce qui supprime le
hot-swap (`/lora-adapters` scale 0/1) et l'appel réseau qui l'accompagnait —
en gardant le classifieur BGE-M3, mais repositionné en **contrôle a
posteriori** sur la commande produite par l'adaptateur, pas en sélecteur
d'adaptateur en amont (détail au composant 2). Argument additionnel de
l'utilisateur pour accepter ce ré-entraînement : la partie logicielle/
conceptuelle (corpus, méthodologie, design du routeur, intégration
`derisk-deleg`) se réinvestira de toute façon dans un futur chantier
hyperréseau — l'adaptateur unique n'est pas vu comme un investissement figé.

## Contexte déjà tranché (2026-07-03)

- **Deux `openclaw.json` séparés par déploiement** : santiago (VPS léger, LLM
  externe uniquement, pas de MCP, bacs à sable comme lib d'outils) et sanroque
  (SLM local + adaptateurs). Ce spec ne concerne que le fichier sanroque.
- **`derisk-deleg` reste la couche de délégation/sécurité**, inchangée dans son
  rôle : elle ne dépend pas de quel mécanisme (LLM externe ou adaptateur
  compilé) a produit la décision de commande.
- Même dépôt `Secretarius`, même `install.sh` avec un flag de profil — la
  divergence se limite aux templates de config, pas au code.

## Architecture

Sur ce profil, `main` n'utilise plus le tool-calling natif d'OpenClaw pour les
messages classifiés. Le hook `before_agent_reply` de `derisk-deleg` (déjà
utilisé pour `/confirm`/`/annuler`) intercepte chaque message entrant **avant**
le tour de modèle, interroge le service routeur, et dispatche directement —
sans jamais solliciter le modèle de `main` pour ces cas.

```
Message Telegram
      │
      ▼
before_agent_reply (derisk-deleg, étendu)
      │
      ├─ /confirm, /annuler, retour OAuth → logique existante (inchangée)
      │
      └─ autre message → POST service routeur /route {message}
                                │
                                ├─ {command, args} reconnu (portail de
                                │   confiance gog passé si applicable)
                                │     → dispatch (delegateWiki / delegateGog /
                                │       delegateScout, ou mise en attente pour
                                │       /repondre) → handled:true
                                │
                                ├─ commande non reconnue
                                │     → message déterministe de repli → handled:true
                                │
                                └─ service routeur/llama-server indisponible
                                      → message déterministe d'indisponibilité
                                        → handled:true (pas de repli cloud)
```

## Composants

### 1. Service llama-server (reconfiguration)

`slm-llama_cpp.service` charge actuellement `tiron-router-Q6_K.gguf` (ancienne
architecture mono-adaptateur, abandonnée — modèle différent de celui visé
ici). À reconfigurer pour charger la base phi-4-mini + **un seul** adaptateur
LoRA, entraîné sur le corpus combiné wiki+gog+null (à ré-entraîner : les
adaptateurs du 2026-06-30 sont scindés par domaine, donc obsolètes pour cette
architecture révisée). Pas besoin de `--lora-init-without-apply` ni de
hot-swap : l'unique adaptateur est chargé actif en permanence
(`--lora <fichier>`), exposant simplement `/v1/chat/completions`.

**Binaire : `build-rocm/bin/llama-server`, pas `build/bin/llama-server`.**
Le service actuellement configuré pointe vers `build/bin/llama-server`, qui
n'est lié à aucune bibliothèque ROCm/HIP (vérifié par `ldd` le 2026-07-04) —
`-ngl`/`HSA_OVERRIDE_GFX_VERSION` y sont inertes, tout tourne CPU. Le binaire
`build-rocm/bin/llama-server` (lié `libggml-hip.so`/`librocblas`/`libamdhip64`)
donne un gain mesuré ce jour, même modèle (`Phi-4-mini-instruct-Q6_K.gguf`).

Trois mesures, du moins au plus représentatif du tour réel du routeur :
- **Gros prefill isolé** (4609 tokens, `max_tokens=5`) : 151,2 s CPU vs 36,9 s
  ROCm (`-ngl 99`) → ~4,1×. Meilleur cas (calcul massivement parallélisable),
  pas représentatif à lui seul.
- **Decode isolé** (prompt court, 200 tokens de sortie) : 13,5 tok/s CPU vs
  **11,2 tok/s ROCm — le GPU est ici plus lent** (lot de taille 1, peu de
  parallélisme, surcoût de lancement de noyau HIP probable).
- **Gabarit réel du routeur** (771 tokens de prompt + ~23 tokens de JSON en
  sortie, proche de la taille visée après compilation LoRA) : **23,63 s CPU
  vs 7,2 s ROCm → ~3,3×**. Le prefill (encore trois à quatre fois plus rapide
  en ROCm à cette taille) domine toujours le temps total ; le désavantage du
  decode ne pèse que 1-2 s sur les deux, donc le gain global reste proche du
  cas gros-prefill malgré le decode plus lent en ROCm.

Conclusion : passer à `build-rocm/bin/llama-server` reste un gain net et
mesuré (~3,3×) sur le gabarit représentatif du tour de routeur, malgré un
decode individuellement moins bon sur ROCm. Le service reconfiguré doit donc
pointer `build-rocm/bin/llama-server`, pas `build/bin/llama-server`.

Note matériel (corrige `CLAUDE.md` machine, obsolète) : l'iGPU réel est un
`gfx1035` (Radeon 680M, RDNA2, Ryzen 9 6900HX), pas un gfx900/Vega —
`HSA_OVERRIDE_GFX_VERSION=10.3.0` est le contournement pour un gfx1035 non
listé comme officiellement supporté.

### 2. Service routeur Python (nouveau)

Service HTTP persistant, un seul endpoint `POST /route {message}`. Charge
BGE-M3 une fois au démarrage. Pour chaque message :
1. Appelle l'adaptateur unique (`/v1/chat/completions`, system prompt minimal
   de routage) → `{"command", "args"}`.
2. Calcule en parallèle la classification BGE-M3 à 3 centroïdes
   (`router_3way.py`, inchangée : les 3 centroïdes restent nécessaires au
   calcul même si un seul sert de portail — cf. constat du 2026-07-02, retirer
   le centroïde hors_perimetre fait bondir les faux accepts gog de 1 à 18).
3. Si la `command` renvoyée par l'adaptateur appartient à l'ensemble gog
   (`/chercher /connecter /inbox /drive /repondre`) : n'accepte que si le
   softmax sur gog est l'argmax **et** ≥ `SEUIL_GOG` (0,50) — sinon traite
   comme non reconnu. Si la `command` est une commande wiki/`/source` :
   aucun contrôle de confiance (déjà jugé sans conséquence grave).

Retourne :
- `{"status": "ok", "command": "/cmd", "args": "..."}`
- `{"status": "no_match"}` si la commande extraite n'est dans aucun ensemble
  connu, ou si une commande gog n'a pas passé le portail de confiance

Nouveau service systemd `--user`, à côté de `slm-llama_cpp.service`.

### 3. Extension du plugin `derisk-deleg`

Pour tout message qui n'est pas déjà `/confirm`/`/annuler`/retour OAuth : appel
au service routeur ; dispatch via les **mêmes fonctions internes** que les
outils existants (pas de nouvelle logique de délégation) :

| Commande routeur | Fonction interne appelée | Remarque |
|---|---|---|
| `/c` | `delegateWiki(api, "capture", args)` | |
| `/q` | `delegateWiki(api, "query", args)` | |
| `/ingest` | `delegateWiki(api, "ingest", "")` | |
| `/wikistatus` | `delegateWiki(api, "status", "")` | |
| `/source` | `delegateScout(api, url)` | **cible scout, pas wiki** — l'adaptateur unique produit `/source` comme n'importe quelle commande, seul le dispatch d'exécution diffère |
| `/chercher` | `delegateGog(api, "search", args)` | |
| `/connecter` | `delegateGog(api, "auth_start", "")` + flux `pendingAuth` existant | |
| `/inbox` | `delegateGog(api, "inbox", "")` | |
| `/drive` | `delegateGog(api, "drive_search", args)` | |
| `/repondre` | logique de mise en attente existante (`parseReply` + `pending`, PAS de délégation directe) | seule commande sensible atteignable par le routeur ; doit obligatoirement passer par `/confirm` — `gog_send` (nouveau mail) n'est pas dans le vocabulaire du routeur (classé hors-périmètre dans le corpus) |

Comportements de repli, tous déterministes, sans appel LLM supplémentaire et
sans repli cloud (cohérent avec le principe déjà posé pour `tiron-llm` :
« pas de fallback automatique vers DeepSeek », `2026-05-29-tiron-llm-local-design.md`) :
- **Commande non reconnue** (JSON invalide, `command` hors des ensembles
  connus, message hors-périmètre pour lequel l'adaptateur unique n'a pas
  produit de commande valide, ou commande gog n'ayant pas passé le portail de
  confiance) : « Je n'ai pas identifié de commande (essayez `/q
  <question>`, `/c <url>`...) ».
- **Service routeur ou llama-server indisponible** : « Routeur local
  indisponible, réessayez dans un instant ».

### 4. Profil `main` réduit

Le tour de modèle de `main` devenant vestigial (le hook gère tous les cas),
`main.primary` peut rester n'importe quel modèle valide (y compris
`tiron-llm`, jamais réellement sollicité), et `tools.sandbox.tools.allow` de
`main` peut perdre la quasi-totalité des outils `wiki_*`/`gog_*`/`source_read`
— mécanisme concret qui réalise le « profil d'agent minimal » visé depuis le
début de cette pile de reprise.

## Hors périmètre de ce chantier

- Le futur `/q` à 3 étages (wiki KB → Wikipedia ZIM → synthèse Euria) —
  chantier wiki/SLM séparé, se règle à ce moment-là.
- Toute gestion de conversation libre au-delà du message de repli
  déterministe.
- Hyperréseau (génération d'adaptateur à la volée), wiki-sur-SLM.
- Cache KV persisté par adaptateur (piste ds4, notée backlog, optimisation de
  latence résiduelle, pas de fiabilité).

## Risques / points ouverts

- **Fidélité d'extraction des arguments pour `/repondre`** : l'adaptateur doit
  produire un `args` que `parseReply()` sait parser (id + texte) ; non
  garanti tant que non testé en conditions réelles — à vérifier en premier
  lors de l'implémentation, avant tout câblage OpenClaw (cohérent avec la
  remarque déjà faite ailleurs sur les redémarrages lents d'OpenClaw).
- **Isolation par agent de `systemPrompt`/`mcp`/`sandbox`** dans la version
  actuelle d'OpenClaw (limite notée 2026-06-03 sur 2026.4.24, état après
  migration 6.1 non revérifié) — sans incidence sur ce design puisque `main`
  n'a de toute façon presque plus d'outils déclarés, mais à garder en tête si
  la réduction du profil `main` s'avère insuffisante en pratique.
