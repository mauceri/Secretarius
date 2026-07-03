# Intégration du routeur Tiron local (SLM + adaptateurs) dans OpenClaw — Design

## Objectif

Câbler dans OpenClaw, sur sanroque, le routeur SLM validé en prototype
(`router_3way.py` + `prototype_tiron_v3.py`, scratchpad session 2026-06-30/07-02) :
classification BGE-M3 à 3 centroïdes (wiki / gog / hors_perimetre) → hot-swap
d'adaptateur LoRA sur phi-4-mini → extraction JSON `{command, args}` → dispatch
réel vers les sous-agents existants (wiki, gog, scout).

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
                                ├─ {command, args, destination} reconnu
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
architecture mono-adaptateur, abandonnée). À reconfigurer pour charger la base
phi-4-mini + les deux adaptateurs LoRA du 2026-06-30 (wiki, gog), préchargés
via `--lora-init-without-apply`, exposant `/lora-adapters` et
`/v1/chat/completions` comme l'attend déjà `prototype_tiron_v3.py`.

### 2. Service routeur Python (nouveau)

Service HTTP persistant, un seul endpoint `POST /route {message}`. Charge
BGE-M3 une fois au démarrage et enveloppe telle quelle la logique déjà validée
(`router_3way.py` pour la classification, la logique hot-swap/appel de
`prototype_tiron_v3.py` pour l'extraction). Retourne :
- `{"status": "ok", "command": "/cmd", "args": "...", "destination": "wiki"|"gog"}`
- `{"status": "no_match"}` si la commande extraite n'est dans aucun ensemble connu

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
| `/source` | `delegateScout(api, url)` | **cible scout, pas wiki** — le routeur classe `/source` dans le domaine « wiki » pour le choix d'adaptateur, mais l'exécution diverge de la classification |
| `/chercher` | `delegateGog(api, "search", args)` | |
| `/connecter` | `delegateGog(api, "auth_start", "")` + flux `pendingAuth` existant | |
| `/inbox` | `delegateGog(api, "inbox", "")` | |
| `/drive` | `delegateGog(api, "drive_search", args)` | |
| `/repondre` | logique de mise en attente existante (`parseReply` + `pending`, PAS de délégation directe) | seule commande sensible atteignable par le routeur ; doit obligatoirement passer par `/confirm` — `gog_send` (nouveau mail) n'est pas dans le vocabulaire du routeur (classé hors-périmètre dans le corpus) |

Comportements de repli, tous déterministes, sans appel LLM supplémentaire et
sans repli cloud (cohérent avec le principe déjà posé pour `tiron-llm` :
« pas de fallback automatique vers DeepSeek », `2026-05-29-tiron-llm-local-design.md`) :
- **Commande non reconnue** (JSON invalide, `command` hors des ensembles
  connus, ou `hors_perimetre` dont l'adaptateur wiki n'a pas produit de
  commande valide) : « Je n'ai pas identifié de commande (essayez `/q
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
