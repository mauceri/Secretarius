# Plan De Migration Du Serveur MCP

## Objectif

Faire evoluer le serveur MCP local de Secretarius depuis une implementation monolithique et statique vers une architecture a registre d'outils, plus simple a etendre et a tester, sans changer le contrat MCP expose aux autres composants.

Le but n'est pas de lancer cette migration maintenant, mais de disposer d'un plan de reprise detaille, decoupe en etapes de faible risque.

## Etat De Depart

Aujourd'hui, le serveur MCP repose principalement sur [secretarius_local/mcp_server.py](/home/mauceric/Secretarius/Prototype/secretarius_local/mcp_server.py).

Caracteristiques actuelles :
- le catalogue d'outils est code en dur dans `_tools_catalog()`
- les handlers sont definis dans le meme fichier
- le dispatch `tools/call` repose sur une table fixe
- le contrat MCP public est stable, mais l'ajout d'un nouvel outil impose une modification directe du serveur
- les outils publics actuellement exposes reflètent des intentions metier utiles :
  - `extract_expressions`
  - `index_text`
  - `search_text`
  - `ask_oracle`

Limites principales :
- serveur central trop charge
- ajout d'un outil trop couple a l'infrastructure JSON-RPC
- descriptions, schemas, handlers et dispatch melanges au meme endroit
- tests peu modulaires a mesure que le nombre d'outils augmente

## Cible Architecturale

La cible recommande est une architecture a registre explicite d'outils MCP.

Principes :
- un outil MCP = une specification (`ToolSpec`) + un handler
- les outils vivent dans des modules separes
- le serveur MCP ne fait plus que :
  - exposer `tools/list`
  - router `tools/call`
  - normaliser les erreurs et le format MCP
- le contrat public reste identique pour le client MCP et pour le routeur

## Invariants A Preserver

Ces points ne doivent pas changer pendant la migration :
- meme protocole MCP cote `tools/list` et `tools/call`
- memes noms d'outils publics
- memes schemas publics, sauf correction explicitement voulue
- memes reponses compactes par defaut
- pas de dependance vers `../secretarius`
- pas de regression sur `tools/secretarius_server.py`
- pas de changement de comportement du routeur du seul fait de la migration de structure

## Proposition De Structure Cible

Arborescence cible suggeree :

```text
Prototype/secretarius_local/
  mcp_server.py
  mcp_tools_registry.py
  mcp_tools/
    __init__.py
    extract_expressions.py
    index_text.py
    search_text.py
    ask_oracle.py
```

Responsabilites :

`mcp_server.py`
- point d'entree JSON-RPC MCP
- appel au registre
- formatage uniforme des reponses et erreurs

`mcp_tools_registry.py`
- definition de `ToolSpec`
- enregistrement des outils
- listing des outils
- resolution d'un handler a partir du nom

`mcp_tools/*.py`
- description et schema de chaque outil
- validation locale de ses arguments
- appel a la logique metier existante

## Design Du Registre

### Option Recommandee

Utiliser un registre explicite, sans decorateur magique dans un premier temps.

Exemple conceptuel :

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]
```

API minimale du registre :
- `register_tool(spec: ToolSpec) -> None`
- `get_tool(name: str) -> ToolSpec | None`
- `list_tools() -> list[ToolSpec]`

Pourquoi cette option :
- explicite
- simple a debugger
- facile a tester
- peu de magie d'import

### Option A Reporter

Un decorateur `@register_mcp_tool(...)` peut etre envisage plus tard, mais il est preferable de commencer par un registre explicite pour limiter la complexite de chargement.

## Etapes De Migration

### Etape 1 - Extraire Le Registre Sans Changer Le Serveur

Objectif :
- introduire `mcp_tools_registry.py`
- y definir `ToolSpec`
- y creer le registre

Travail :
- creer `ToolSpec`
- creer les fonctions `register_tool`, `get_tool`, `list_tools`
- ne rien changer encore au dispatch MCP reel

Validation :
- tests unitaires du registre seul
- aucune regression sur les tests MCP existants

Risque :
- faible

### Etape 2 - Migrer Le Catalogue D'Outils

Objectif :
- sortir les definitions d'outils de `_tools_catalog()` vers des `ToolSpec`

Travail :
- conserver les descriptions et schemas publics existants
- reconstruire `tools/list` a partir du registre
- verifier que l'ordre et le contenu exposes restent conformes

Validation :
- tests de [tests/test_mcp_tools_catalog.py](/home/mauceric/Secretarius/Prototype/tests/test_mcp_tools_catalog.py)
- diff de `tools/list` avant/apres

Risque :
- faible a moyen

### Etape 3 - Extraire Les Handlers Outil Par Outil

Objectif :
- sortir progressivement les handlers du gros fichier `mcp_server.py`

Ordre recommande :
1. `ask_oracle`
2. `extract_expressions`
3. `search_text`
4. `index_text`

Raison :
- `ask_oracle` est le plus simple
- `extract_expressions` est central mais localise
- `search_text` et `index_text` sont plus riches et dependent du pipeline documentaire

Travail :
- creer un module par outil dans `mcp_tools/`
- y deplacer :
  - description
  - schema
  - handler
- faire enregistrer chaque `ToolSpec`

Validation :
- tests unitaires par outil
- conservation des reponses compactes

Risque :
- moyen

### Etape 4 - Simplifier `mcp_server.py`

Objectif :
- faire de `mcp_server.py` un adaptateur MCP mince

Contenu residuel cible :
- lecture des messages MCP
- `tools/list` depuis le registre
- `tools/call` vers le handler du `ToolSpec`
- normalisation de la reponse avec `_tool_result(...)`
- normalisation des erreurs JSON-RPC

Validation :
- test bout en bout `tools/list`
- test bout en bout `tools/call`
- smoke test via [tests/test_mcp_client.py](/home/mauceric/Secretarius/Prototype/tests/test_mcp_client.py)

Risque :
- moyen

### Etape 5 - Introduire Des Tests De Non-Regression Par Outil

Objectif :
- rendre la migration sure dans la duree

Tests a ajouter ou renforcer :
- contrat public de `tools/list`
- contrat compact de chaque outil
- comportement `debug_full` / `debug_return_raw`
- erreurs de validation d'arguments
- tests d'import du registre

Risque :
- faible

### Etape 6 - Documentation D'Extension

Objectif :
- documenter la procedure d'ajout d'un nouvel outil

Le document devra expliquer :
- comment creer un module outil
- comment definir un `ToolSpec`
- quelles reponses sont attendues
- quelles conventions de schema respecter
- comment ecrire les tests associes

## Contrat Recommande Pour Un Outil MCP

Chaque outil doit suivre ces principes :

- nom oriente intention utilisateur
- schema minimal
- pas d'arguments parasites
- handler autonome
- reponse compacte par defaut
- details supplementaires uniquement en mode debug

Convention suggeree :
- `structuredContent` compact
- champs de synthese stables
- warnings explicites mais non verbeux

## Strategie De Migration

### Strategie Recommandee

Migration progressive, outil par outil, avec coexistence temporaire.

Principe :
- pendant un temps, `mcp_server.py` peut continuer a contenir des wrappers de compatibilite
- chaque outil migre doit conserver exactement son nom public
- aucun changement de client ou de routeur pendant la migration

Pourquoi :
- minimise le risque
- facilite les comparaisons avant/apres
- permet des retours arriere simples

### Strategie A Eviter

Ne pas faire un big bang du type :
- nouvelle arborescence
- tous les outils migres en une fois
- serveur MCP entierement reecrit

Risque :
- trop de regressions difficiles a localiser

## Points Sensibles

### 1. Couplage Avec Le Pipeline Documentaire

`index_text` et `search_text` ne sont pas de simples wrappers :
- ils dependent de `document_pipeline.py`
- ils ont des comportements compacts et debug a conserver

Il faudra verifier soigneusement :
- les warnings
- les compteurs de synthese
- la forme des retours

### 2. Contrat Public Du Routeur

Le routeur s'appuie sur :
- les noms d'outils
- leurs descriptions
- leurs schemas minimaux

Toute regression ici peut modifier la selection d'outil. Il faut donc garder les descriptions stables pendant la migration initiale.

### 3. Ordre D'Import

Si les outils s'enregistrent dans un registre au chargement du module :
- il faudra garantir que les modules sont bien importes avant `tools/list`
- sinon, utiliser un bootstrap explicite dans `mcp_server.py`

Le bootstrap explicite est recommande au depart.

### 4. Tests Existants

Les tests actuels supposent parfois implicitement :
- la presence de certaines fonctions dans `mcp_server.py`
- une structure de reponse compacte donnee

Il faudra adapter les tests avec prudence, sans perdre la couverture existante.

## Plan De Fichiers Concret

### Fichier 1

`secretarius_local/mcp_tools_registry.py`

Contenu prevu :
- `ToolSpec`
- registre interne
- fonctions de registration et de lecture

### Fichier 2

`secretarius_local/mcp_tools/__init__.py`

Contenu prevu :
- bootstrap d'import des outils

### Fichier 3+

Modules outils :
- `extract_expressions.py`
- `index_text.py`
- `search_text.py`
- `ask_oracle.py`

Contenu prevu :
- description
- schema
- handler
- `TOOL_SPEC`

### Fichier Final

`secretarius_local/mcp_server.py`

Contenu cible :
- compatibilite protocolaire
- glue code minimal

## Plan De Tests Associe

Tests a conserver :
- [tests/test_mcp_client.py](/home/mauceric/Secretarius/Prototype/tests/test_mcp_client.py)
- [tests/test_mcp_tools_catalog.py](/home/mauceric/Secretarius/Prototype/tests/test_mcp_tools_catalog.py)
- [tests/test_mcp_server_compact_responses.py](/home/mauceric/Secretarius/Prototype/tests/test_mcp_server_compact_responses.py)
- [tests/test_document_pipeline.py](/home/mauceric/Secretarius/Prototype/tests/test_document_pipeline.py)

Tests a ajouter :
- `tests/test_mcp_tools_registry.py`
- `tests/test_mcp_tool_extract_module.py`
- `tests/test_mcp_tool_index_module.py`
- `tests/test_mcp_tool_search_module.py`
- `tests/test_mcp_tool_oracle_module.py`

## Definition De Termine

La migration pourra etre consideree comme terminee lorsque :
- `mcp_server.py` ne contient plus les handlers metier
- le catalogue public vient du registre
- chaque outil public vit dans son module
- tous les tests MCP passent
- la procedure d'ajout d'un nouvel outil n'impose plus de modifier un gros fichier central

## Recommandation Finale

Quand cette migration sera reprise, il faudra la traiter comme une migration de structure, pas comme une refonte fonctionnelle.

Autrement dit :
- ne pas changer en meme temps les outils, le routeur et le serveur
- geler autant que possible le contrat public
- migrer la structure d'abord
- n'ameliorer l'ergonomie d'ajout d'outils qu'une fois la base stabilisee
