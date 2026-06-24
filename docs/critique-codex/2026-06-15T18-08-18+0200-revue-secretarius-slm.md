# Résumé critique Codex — Secretarius SLM

Date locale : 2026-06-15 18:08:18 +0200  
Sujet : revue de l'architecture Secretarius, avec focus sur l'instance en construction `~/.openclaw-slm`.

## Contexte de la demande

La revue portait sur l'application Secretarius située dans `~/Secretarius`, avec une attention particulière aux documents d'architecture sous `~/Secretarius/docs` et à la différence entre :

- l'instance de production : `~/.openclaw` ;
- l'instance SLM en construction : `~/.openclaw-slm`.

Le problème de fond identifié par l'utilisateur est la saturation du contexte de l'orchestrateur Tiron lorsque trop d'outils MCP et de skills sont portés dans son contexte permanent. Cette saturation rend l'usage de petits modèles locaux, notamment `phi-4-mini-instruct`, impraticable comme orchestrateurs.

La direction architecturale étudiée consiste à alléger Tiron, déplacer les capacités vers des sous-agents spécialisés, et installer les outils dans les images Docker des bacs à sable de ces sous-agents. À ce stade, l'agent `wiki` est le nouveau sous-agent principal ; `scout` existait déjà.

## Documents et fichiers inspectés

Documents d'architecture consultés :

- `docs/architecture/secretarius-architecture-socle.md`
- `docs/architecture/spec-architecture-par-intention.md`
- `docs/architecture/diagnostic-et-observabilite.md`
- `docs/Secretarius.md`

Fichiers de configuration et d'installation inspectés :

- `openclaw-config/install.sh`
- `openclaw-config/openclaw-slm.json.template`
- `openclaw-config/Dockerfile.tiron`
- `openclaw-config/Dockerfile.wiki`
- `openclaw-config/openclaw-gateway-slm.service`
- `openclaw-config/gateway-slm.systemd.env.template`
- `install.sh`

Workspaces et instructions inspectés :

- `openclaw-config/workspace-slm/AGENTS.md`
- `openclaw-config/workspace/skills/wiki-deleg/SKILL.md`
- `openclaw-config/workspace/skills/scout/SKILL.md`
- `openclaw-config/workspace-wiki-slm/AGENTS.md`
- `~/.openclaw-slm/workspace/AGENTS.md`
- `~/.openclaw-slm/workspace/skills/wiki-deleg/SKILL.md`
- `~/.openclaw-slm/workspace/skills/scout/SKILL.md`
- `~/.openclaw-slm/workspace-wiki/AGENTS.md`

Scripts métier inspectés :

- `Wiki_LM/tools/wiki.py`
- `Wiki_LM/tools/capture.py`
- `Wiki_LM/tools/ingest.py`
- `Wiki_LM/tests/test_wiki_cli.py`
- `openclaw-config/scout-watcher`
- `openclaw-config/scout_process.py`
- `openclaw-config/injection_guard.py`

## État observé

Le binaire OpenClaw utilisé par le service SLM installé est bien :

```text
OpenClaw 2026.5.12 (f066dd2)
```

Le service systemd installé pour l'instance SLM pointe vers :

```text
/home/mauceric/.openclaw-slm/npm/node_modules/.bin/openclaw gateway run --profile slm
```

Cependant, la configuration réelle `~/.openclaw-slm/openclaw.json` a été touchée par `2026.6.1`, et diverge du template versionné. En particulier, l'agent `main` y tourne actuellement sur Qwen397 dans la config réelle, alors que la cible conceptuelle est Tiron léger sur phi-4-mini. Cette divergence n'est pas forcément une erreur pour des tests, mais elle rend l'état réel difficile à reproduire depuis le dépôt.

Le dépôt n'avait pas de diff suivi, mais plusieurs documents d'architecture étaient non suivis par Git au moment de la revue :

- `.claude/`
- `docs/architecture/corpus-intentions-seed.md`
- `docs/architecture/diagnostic-et-observabilite.md`
- `docs/architecture/secretarius-architecture-socle.md`
- `docs/architecture/spec-architecture-par-intention.md`
- `docs/superpowers/plans/2026-06-10-scout-slm.md`

## Constats principaux

### 1. Le profil SLM n'est pas encore reproductible depuis le dépôt

Le script `openclaw-config/install.sh`, en profil `slm`, installe `openclaw-config/workspace-slm` comme workspace principal. Or ce dossier ne contient pas de sous-répertoire `skills/`, et son `AGENTS.md` indique encore que les capacités wiki et sources externes ne sont pas disponibles.

Dans l'instance réelle `~/.openclaw-slm`, les skills `wiki-deleg` et `scout` existent pourtant et sont utilisés. Cela signifie qu'une partie importante de l'état SLM a été corrigée ou installée manuellement, mais n'est pas fidèlement décrite par les sources versionnées.

Conséquence : une réinstallation propre de `--profile slm` risque de produire une instance qui ne correspond pas à l'architecture actuellement testée.

### 2. Les outils wiki ne sont pas encore réellement bakés dans l'image

La décision d'architecture cible dit que les outils doivent être embarqués dans l'image Docker du sous-agent. C'est la bonne direction. Mais l'état actuel du template SLM utilise encore :

- `dangerouslyAllowExternalBindSources: true` ;
- un bind de `~/Secretarius/Wiki_LM/tools` vers `/wiki-tools:ro` ;
- un bind de la base de connaissances hôte vers `/Wiki_LM:rw`.

Le `Dockerfile.wiki`, lui, installe les dépendances Python et précharge `BAAI/bge-m3`, mais ne copie pas encore `wiki.py` ni les modules `Wiki_LM/tools` dans l'image.

Conclusion : l'architecture est encore dans un état de transition. L'image n'est pas encore autonome ; elle dépend du code source monté depuis l'hôte.

### 3. L'ingestion wiki contourne encore Scout

Le document `spec-architecture-par-intention.md` identifie déjà ce point comme ouvert : l'ingestion doit idéalement faire son fetch via Scout pour préserver la frontière anti-injection.

Dans le code actuel, `Wiki_LM/tools/ingest.py` lit les URL directement via `urllib.request.urlopen`. Donc une URL capturée puis ingérée passe directement dans le pipeline LLM wiki sans filtrage par Scout/injection-guard.

Ce n'est pas seulement une dette théorique : c'est la principale faille de sécurité restante dans l'architecture cible. Scout protège les lectures externes explicites, mais pas encore le chemin `capture -> ingest`.

### 4. Le dispatch déterministe des commandes n'est pas encore implémenté dans les skills versionnés

Le document `spec-architecture-par-intention.md` propose de remplacer le routage par jugement du LLM par un dispatch déterministe :

```text
commande -> skill user-invocable -> outil cible
```

Cette décision est solide. Mais les skills versionnés `wiki-deleg` et `scout` restent actuellement des skills advisory. Ils n'ont pas encore de frontmatter de type :

```yaml
user-invocable: true
command-dispatch: tool
command-tool: ...
command-arg-mode: raw
```

Donc `/c`, URL nue, `ingère`, `/q`, etc. restent dépendants du jugement du modèle dans l'état versionné.

Le spike documenté montre que le dispatch déterministe OpenClaw fonctionne, mais il reste à l'intégrer dans l'application.

### 5. Les instructions installées divergent des instructions versionnées

L'agent wiki réellement installé dans `~/.openclaw-slm/workspace-wiki/AGENTS.md` contient des corrections importantes :

- usage explicite de `python3`, pas `python` ;
- tolérance de format si Tiron envoie une tâche qui n'est pas exactement `op: <op> | <argument>` ;
- rappel que capture n'analyse ni ne traduit.

Le fichier versionné `openclaw-config/workspace-wiki-slm/AGENTS.md` reste marqué "Draft non déployé" et contient encore une commande `python /wiki-tools/wiki.py` dans la section "Outil unique", même si la procédure d'ingestion utilise ensuite `python3`.

Conséquence : les corrections qui rendent l'E2E plus robuste ne sont pas encore entièrement rapatriées dans le dépôt.

### 6. Le watcher scout peut facilement pointer vers la production

`openclaw-config/scout-watcher` définit par défaut :

```bash
SCOUT_WORKSPACE="${SCOUT_WORKSPACE:-${HOME}/.openclaw/agents/scout/workspace}"
```

Ce défaut correspond à la production, pas à l'instance SLM. Or le skill scout installé dans l'instance SLM mentionne un usage manuel de `scout-watcher-slm`, qui ne semble pas exister dans le dépôt inspecté.

Si le watcher est lancé à la main sans `SCOUT_WORKSPACE=...`, il peut donc traiter le mauvais workspace.

## Analyse de la décision : outils dans les images Docker

La décision d'encapsuler les outils dans les images Docker des bacs à sable est jugée bonne et cohérente avec l'objectif.

Elle répond directement au problème initial :

- Tiron ne porte plus les définitions et modes d'emploi détaillés de tous les outils métier ;
- les capacités ne coûtent plus de tokens dans le contexte permanent de l'orchestrateur ;
- chaque agent ne dispose que des binaires présents dans son image ;
- l'isolation devient une propriété d'environnement, et non une simple règle de prompt ;
- l'abandon de MCP réduit la fragilité liée aux changements de versions OpenClaw et aux serveurs intermédiaires.

Cette approche est particulièrement adaptée à un orchestrateur SLM, parce qu'elle diminue le prefill et rend phi-4-mini plus réaliste comme cerveau de Tiron.

Mais l'approche ne produit ses bénéfices que si elle est appliquée complètement. La cible propre devrait être :

- image `secretarius-tiron` : uniquement les outils que Tiron peut utiliser directement, par exemple `gog` si cette décision est maintenue ;
- image `secretarius-wiki` : `wiki.py` et tous ses modules Python embarqués ;
- image ou environnement `scout` : outils de lecture externe et guard clairement séparés ;
- volumes limités aux données métier nécessaires ;
- pas de bind du code source applicatif ;
- tests vérifiant qu'un outil existe dans l'image de l'agent attendu et est absent des autres images.

Exemple de propriété à tester :

- `wiki.py` doit être présent dans l'image `secretarius-wiki` ;
- `wiki.py` doit être absent de l'image `secretarius-tiron` ;
- `gog` doit être présent seulement là où il est explicitement décidé de l'autoriser ;
- l'image wiki doit fonctionner sans bind de `~/Secretarius/Wiki_LM/tools`.

## Analyse de la décision : véritables commandes déterministes

La décision d'introduire de véritables commandes est jugée encore plus importante que le découpage Docker.

Elle corrige un défaut structurel des agents : confier au LLM une décision qui devrait relever d'un protocole.

Des commandes comme :

- `/c`
- `/ingest`
- `/wiki-status`
- `/q`
- `/source`
- `/mail`
- `/agenda`
- `/drive`

doivent être des contrats opérationnels. Elles doivent avoir :

- une grammaire simple ;
- un handler déterministe ;
- un outil cible explicite ;
- une sortie JSON ou une erreur précise ;
- une trace observable ;
- un comportement de refus fixe en cas de commande inconnue.

La commande `/c` ne doit jamais ingérer.  
La commande `/ingest` ne doit jamais capturer.  
La commande `/q` ne doit jamais modifier la base.  
La commande `/source` ne doit jamais passer par wiki si elle relève d'une lecture externe ponctuelle.

Cette séparation réduit fortement les confabulations de statut et les mauvais routages observés.

## Commandes comme outil de désambiguïsation conversationnelle

L'idée d'utiliser les commandes comme représentation canonique des intentions dans le mode conversationnel est également jugée saine.

Le mode conversationnel ne devrait pas exécuter directement une intention inférée. Il devrait plutôt produire une commande candidate, puis demander confirmation.

Exemple :

```text
Utilisateur : Garde cette page pour le wiki : https://example.com
Tiron : Exécuter /c https://example.com ? Oui/Non
```

Après confirmation seulement, le handler déterministe de `/c` est appelé.

Le classifieur ou le petit modèle d'intention devient alors un assistant d'interface, pas une autorité d'exécution. Il aide à traduire le langage naturel en commande canonique, mais ne déclenche pas lui-même d'action irréversible.

Cette distinction est essentielle :

- voie commande : déterministe, exécution directe ;
- voie conversation : inférence, proposition, confirmation, puis exécution déterministe.

Cela permet aussi d'améliorer progressivement le classifieur sans risquer des actions non voulues.

## Risques à surveiller

### Ne pas confondre commandes et skills advisory

Si `/c` reste seulement décrit dans un skill, le routage reste fragile. Le gain de déterminisme apparaît seulement quand OpenClaw dispatch réellement la slash-command vers un outil sans tour modèle.

### Ne pas créer des handlers flous

Les handlers de commande ne doivent pas devenir de petits orchestrateurs ambigus. Ils doivent parser, valider, appeler l'opération prévue, et retourner un résultat structuré.

Un handler `/c` qui déciderait parfois de capturer, parfois d'ingérer, parfois de lire la page, réintroduirait le problème initial.

### Stabiliser peu de commandes avant d'élargir

Il vaut mieux stabiliser cinq commandes robustes que créer trop tôt une grande taxonomie.

Noyau recommandé :

- `/c` : capture seulement ;
- `/ingest` : lance l'ingestion ;
- `/wiki-status` : état wiki ;
- `/q` : requête wiki ;
- `/source` : lecture externe via Scout.

Ensuite seulement :

- `/mail`
- `/agenda`
- `/drive`
- autres commandes métier.

### Faire passer l'ingestion par Scout ou documenter consciemment l'exception

Tant que `ingest.py` fetch directement les URL, le pipeline wiki reste exposé au contenu externe non filtré. Il faut soit corriger ce chemin, soit assumer explicitement l'exception et la borner.

## Recommandations prioritaires

1. Rendre le profil SLM reproductible depuis le dépôt.
   - Installer les skills SLM depuis `openclaw-config/workspace-slm/skills`.
   - Installer les workspaces `wiki` et `scout` en profil `slm`.
   - Rapatrier les corrections manuelles de `~/.openclaw-slm` dans les sources.

2. Finir l'autonomie de `Dockerfile.wiki`.
   - Copier `Wiki_LM/tools` dans l'image.
   - Supprimer le bind du code source.
   - Ne garder qu'un volume de données métier pour la base wiki.

3. Implémenter un premier dispatch déterministe réel.
   - Commencer par `/c`.
   - Le handler doit écrire un `.url` ou `.md`, retourner JSON, et ne jamais invoquer le modèle.
   - Tester que `/c` ne passe pas par le LLM.

4. Ajouter un harnais de validation E2E.
   - Commande inconnue -> refus fixe.
   - `/c URL` -> fichier raw créé.
   - `/ingest` -> worker lancé en arrière-plan.
   - `/wiki-status` -> état lu depuis artefacts.
   - `/source URL` -> Scout utilisé, pas wiki.

5. Régler le chemin `capture -> ingest -> fetch`.
   - Soit `ingest.py` consomme du contenu déjà nettoyé par Scout ;
   - soit il délègue le fetch à Scout ;
   - soit il isole explicitement cette exception avec des garde-fous.

## Conclusion

L'orientation générale est solide.

L'usage des images Docker comme frontière de capacité est pragmatique et adapté à l'objectif SLM. Il permet de sortir les outils du contexte de Tiron et de transformer une contrainte fragile de prompt en contrainte d'environnement.

L'usage de vraies commandes est la pièce la plus structurante : il remplace le routage par jugement du LLM par un protocole testable. Les commandes peuvent aussi devenir la représentation canonique des intentions en mode conversationnel, à condition que ce mode propose et confirme une commande, sans exécuter directement une inférence.

Le principal travail restant n'est pas conceptuel mais d'intégration :

- rendre l'état SLM reproductible ;
- finir les images autonomes ;
- passer des skills advisory au dispatch déterministe ;
- fermer la frontière Scout pour l'ingestion ;
- tester chaque commande sur des artefacts vérifiables plutôt que sur les réponses des agents.
