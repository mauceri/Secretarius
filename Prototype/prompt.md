
### Prompt pour Antigravity )

**Rôle :** Vous êtes un ingénieur logiciel expert en Python 3.10+ et en architecture hexagonale. Votre mission est de générer le code source complet de l'agent **Secretarius** en vous basant sur les spécifications techniques et les structures de classes fournies.

**Objectif :** Créer une application modulaire, déterministe et strictement découplée utilisant le protocole **MCP (Model Context Protocol)**.

**Spécifications d'implémentation :**

1. **Cœur Logique (Core) :**
* Implémentez le `ChefDOrchestre` comme une machine à états pilotant le cycle de raisonnement ReAct.
* Appliquez une gestion rigoureuse des priorités : **Priorité 1** pour les résultats d'outils (MCP), **Priorité 2** pour l'historique de session, **Priorité 3** pour la mémoire sémantique.
* Utilisez les modèles Pydantic fournis pour garantir l'intégrité des messages.


2. **Adaptateurs d'Entrée (Guichet) :**
* Implémentez une interface `TUI_Guichet` via la bibliothèque **Textual**.
* L'interface doit isoler les entrées utilisateur et afficher les étapes de réflexion du Chef d'Orchestre.
* Prévoyez l'extensibilité pour de futurs adaptateurs comme Telegram ou une API.


3. **Adaptateurs de Sortie (MCP & LLM) :**
* Développez un `StdioMCPClient` utilisant le SDK officiel `mcp` pour la communication avec les serveurs d'outils.
* Implémentez l'interface `LLMInterface` pour communiquer avec une instance **Ollama** locale via JSON.
* **Sécurité :** Le LLM ne doit avoir aucun accès direct au réseau ou au système ; toutes les interactions doivent transiter par le Chef d'Orchestre.


4. **Serveur MCP de Test (Oracle) :**
* Générez un fichier indépendant `oracle_server.py` qui expose l'outil `ask_oracle`.
* Cet outil doit répondre "OUI" ou "NON" de manière aléatoire à toute question fermée.



**Contraintes techniques :**

* Utilisez un typage fort sur l'ensemble du projet.
* Générez un fichier `config.yaml` pour gérer les chemins des serveurs MCP et les endpoints du LLM.
* Fournissez une arborescence claire comprenant les répertoires `main.py`, `core/`, `adapters/`, et `tools/`.

---

Le répertoire de travail est :
/home/mauceric/Secretarius/Prototype