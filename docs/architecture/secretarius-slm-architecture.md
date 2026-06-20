# Secretarius SLM — Architecture « Tiron léger »

> Document de socle, écrit le 2026-06-05 après une séance de validation.
> But : servir de point de départ à une session de conception (superpowers) neuve.
> Il est autoporté : un agent qui le lit à froid doit comprendre la cible, ce qui
> est déjà prouvé, et ce qui reste à trancher.

---

## 1. La vision

Secretarius est un assistant personnel bâti sur OpenClaw. Aujourd'hui, l'assistant
principal (« Tiron ») porte **tous** les outils dans son contexte permanent (wiki,
gog/Google, routage…), ce qui gonfle son prompt système à ~11,7k tokens. Conséquence :
un petit modèle local (SLM, ex. phi-4-mini) est inutilisable comme cerveau de Tiron
— le prefill de ce contexte dépasse le TTL.

**La cible : un Tiron *léger*.** Tiron ne porte plus aucune définition d'outil métier.
Son contexte se réduit à l'essentiel — un prompt de routage et de quoi déléguer
(`sessions_spawn`). Les capacités migrent vers des **sous-agents spécialisés**, dont
chacun porte ses propres outils **dans l'image de son bac à sable** (binaires) et dans
ses **skills** (chargés à la demande, hors du contexte permanent).

Bénéfices visés :
- **SLM viable** : contexte de Tiron ~1k tokens → prefill court → phi-4-mini redevient
  jouable comme orchestrateur. L'architecture est **agnostique au modèle** : elle
  profite tout autant à Euria (coût et latence réduits).
- **Confidentialité** : un orchestrateur local possible.
- **Robustesse** : en abandonnant les serveurs MCP au profit de binaires+skills, on
  supprime toute une classe de fragilité liée aux montées de version d'OpenClaw.

---

## 2. Le principe architectural

```
Utilisateur ──> Tiron (SLM léger)
                  │  contexte = prompt de routage + sessions_spawn
                  │
                  ├─ sessions_spawn(agentId="wiki", …) ──> Agent wiki (Euria)
                  │                                         image: query.py + skill wiki
                  ├─ sessions_spawn(agentId="gog",  …) ──> Agent gog
                  │                                         image: binaire gog + skill gog
                  └─ (route_intent / BERT)                 binaire/skill dans l'image de Tiron
```

Règles de conception :

1. **Isolation par le contenu de l'image, pas par la tool-policy.** Tous les agents
   peuvent avoir `exec` autorisé ; ce qu'un agent peut *faire* dépend des binaires
   présents dans **son** image. L'image de Tiron n'a pas `gog` → il ne peut pas
   l'exécuter ; l'image de l'agent gog l'a → lui seul l'exécute. On ne demande jamais
   à OpenClaw de « réaccorder » un outil à un sous-agent (ce qui est interdit : chaque
   niveau ne peut que restreindre). On contourne ce mur par l'image.

2. **exec tourne *dans* le conteneur** (`tools.exec.host = "sandbox"` ou `"auto"`),
   pas sur l'hôte (`"gateway"`). Ainsi l'agent voit le `/usr/local/bin` de son image.

3. **Délégation asynchrone.** Tiron lance un sous-agent (`sessions_spawn`), se met en
   attente (`sessions_yield`), le sous-agent travaille, son résultat est **annoncé**
   (push) dans la session de Tiron qui est réveillé et **relaie** le résultat à
   l'utilisateur.

4. **Un modèle par agent.** Déjà supporté (`agents.list[].model`). Cible : Tiron =
   phi-4-mini ; agent wiki = Euria (synthèse lourde) ; etc.

5. **Pas de MCP.** Les capacités sont des binaires (dans les images) + des skills, pas
   des serveurs MCP. La matière existe déjà en partie : `query.py` (wiki, utilisé par
   Scout) et le binaire `gog`.

---

## 3. Ce qui est VALIDÉ (preuves du 2026-06-05)

Deux piliers techniques, testés sur la machine `sanroque`.

### Pilier A — Image par agent + exec dans le conteneur ✅ (marche sur OpenClaw 6.1)
- Schéma confirmé : `agents.list[].sandbox.docker.image` (type `AgentSandboxConfig.docker.image`)
  et `tools.exec.host ∈ {auto, sandbox, gateway, node}` (type `ExecHost`).
- Test : image dérivée `FROM openclaw-sandbox:bookworm-slim` + binaire marqueur `agent-tool`.
  Un agent `toolagent` pointant sur cette image exécute `agent-tool` (sortie remontée) ;
  l'agent `main` sur l'image de base échoue (`agent-tool: not found`).
- **Conclusion : l'isolation par contenu d'image fonctionne**, et un sous-agent peut
  exécuter un vrai binaire via exec-in-sandbox.

### Pilier B — Livraison du résultat d'un sous-agent ✅ (mais bloquée sur 6.1, voir §4)
- Séquence prouvée sur OpenClaw **2026.5.12** : Tiron `sessions_spawn` → `sessions_yield`
  → le sous-agent produit son résultat → l'annonce aboutit → Tiron est réveillé et
  **relaie le résultat verbatim**.
- Sur OpenClaw **2026.6.1**, la même séquence **échoue** (voir §4).
- `sessions_yield` doit être autorisé (`tools.sandbox.tools.allow`) pour que le parent
  attende proprement au lieu de fabriquer une réponse prématurée.

### Autres points confirmés
- **Modèle par agent** : opérationnel.
- **Auth Euria** : fonctionne sur 5.12 comme sur 6.1.
- **Compat de la config sur 5.12** : 5.12 parse la config 6.1 et possède tout le
  nécessaire (image par agent, exec-sandbox, auth-profiles). La compat MCP n'a **pas**
  besoin d'être vérifiée puisque l'architecture cible abandonne MCP.

---

## 4. Le bug bloquant et la stratégie de version

### Le bug : OpenClaw #84059 (P1, ouvert)
- Introduit en **2026.5.18** (via dépendance `pi-agent-core@0.75.1`).
- Un contrôle anti-falsification du fichier de session (empreinte nanoseconde) prend
  les **écritures internes d'OpenClaw** (journal de trajectoire, hooks) pour une
  intrusion et **avorte le tour**. Erreur : `EmbeddedAttemptSessionTakeoverError:
  session file changed while embedded prompt lock was released`.
- Effet sur l'architecture : l'**annonce** du résultat d'un sous-agent vers Tiron
  échoue (3 essais puis abandon), et Tiron **fabrique** alors une réponse au lieu de
  relayer. C'est une course ; un sous-agent rapide la déclenche presque à coup sûr
  (un agent lent comme Scout y échappe souvent — d'où l'illusion que « Scout marche »).
- Issues liées : #83510 (la mutation est mal classée en « échec modèle »), #84460.
  Correctifs proposés (PR #84250) **non encore livrés**. Aucun flag pour désactiver.

### Chronologie
- Secretarius v0.1.0 (~28 mai) tournait sur **2026.4.24** (avant le bug) → la délégation
  marchait, l'utilisateur l'avait testée.
- Migration silencieuse le 4 juin → **2026.6.1** (après le bug). `install.sh` épinglait
  `npm install -g openclaw` (= latest), d'où la montée non maîtrisée.

### Stratégie retenue
1. **Épingler la version d'OpenClaw** dans `install.sh` (ne plus suivre `latest`).
2. **Version de travail = 2026.5.12** (dernière stable d'avant le bug ; compat vérifiée
   pour tout ce dont l'archi a besoin). Alternative : attendre qu'une 6.x livre le
   correctif #84059 et rester sur la ligne moderne.
3. **Ne pas fragiliser la prod** : la prod quotidienne (6.1, usage en outils directs)
   n'a pas besoin du correctif et fonctionne ; on construit le SLM sur une **instance
   isolée** (`--profile`, port séparé, telegram/tailscale désactivés) avant tout
   basculement.

---

## 5. Décisions actées

- Tiron léger = priorité n°1 (l'utilisateur). Architecture agnostique au modèle (phi-4
  *ou* Euria comme orchestrateur).
- Abandon de MCP au profit de binaires-dans-l'image + skills.
- exec autorisé largement, `host = sandbox` ; isolation par image.
- `sessions_yield` autorisé pour l'attente du parent.
- Version OpenClaw épinglée < 2026.5.18 pour le build délégué (cible 2026.5.12).
- Build sur instance isolée, prod 6.1 préservée.

---

## 6. Composants à construire

| Composant | Contenu | Existant à réutiliser |
|---|---|---|
| **Tiron (orchestrateur)** | phi-4-mini, contexte = routage + `sessions_spawn` ; image de base | — |
| **Agent wiki** | Euria ; image avec `query.py` + skill wiki | `query.py` existe |
| **Agent gog** | image avec binaire `gog` + skill gog | binaire `gog` présent |
| **route_intent / BERT** | binaire/skill (détection d'intention) dans l'image de Tiron | router-mcp actuel (à reporter) |
| **Sécurité (Scout)** | à repositionner dans ce modèle (anti-injection) | agent Scout existant |

---

## 7. Questions ouvertes (pour la session superpowers)

1. **gog chez Tiron ou agent dédié ?** Avec binaire+skill, gog peut rester chez Tiron
   *sans* alourdir son contexte permanent (le binaire ne coûte rien en contexte, le
   skill ne se charge qu'au besoin). Simplicité vs isolation : à trancher.
2. **route_intent est-il nécessaire ?** Tiron (SLM) peut-il router directement d'après
   le message, ou faut-il l'outil BERT de détection d'intention ? Si oui, sous quelle
   forme (binaire/skill dans son image) ?
3. **Granularité des sous-agents** : un par capacité, ou regroupements ?
4. **Stratégie de version** : épingler 2026.5.12 maintenant, ou attendre le correctif
   #84059 sur la ligne 6.x ? Critères de décision ?
5. **Sécurité** : comment Scout / l'anti-injection s'insèrent dans l'archi déléguée ?
6. **Performance** : mesurer le prefill/latence de phi-4-mini en orchestrateur léger
   (contexte réduit) sur l'iGPU local — le GPU loué n'est pertinent que pour le chantier
   wiki (ingestion : extraction d'expressions + embeddings), pas pour l'orchestrateur.
7. **Chantier wiki (séparé)** : résumés extractifs (BERT/embeddings, léger) vs abstractifs
   (LLM, lourd) ; mono-vecteur BGE-M3 vs multi-vecteurs ColBERT (late interaction). Ces
   choix conditionnent le besoin GPU à l'ingestion. À explorer dans un chantier dédié.

*(L'utilisateur ajoutera peut-être une ou deux questions ici.)*

---

## 8. État de la machine (point de reprise au 2026-06-05)

- OpenClaw **2026.6.1** en prod, **restaurée à l'identique** après les tests
  (agents = main + scout ; `exec.host=gateway` ; exec re-denié ; aucun artefact de test).
- Services actifs : gateway (18789), wiki-lm-mcp (8901), gog-mcp (8902), router-mcp
  (8903), slm-llama-cpp (8998, phi-4-mini ROCm).
- Backups de config laissés : `~/.openclaw/openclaw.json.pre-test` (config d'origine).
- Reste un résidu sans gravité : l'extension obsolète `openclaw-mcp-adapter` qui erre
  dans les logs à chaque appel CLI (sera supprimée de fait avec l'abandon de MCP).
- Détail mémoire complet : voir la note `project_openclaw_announce_bug` (preuves, n° de
  versions, commandes de downgrade).
