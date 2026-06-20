# Critiques Codex

Ce répertoire sert à conserver les critiques, commentaires et résumés produits par Codex sur le projet Secretarius.

## Rôle attendu

Dans ce projet, le rôle de Codex est de critiquer les choix architecturaux et les choix d'implémentation, en priorité pour :

- l'architecture Secretarius SLM ;
- l'orchestrateur Tiron léger ;
- la séparation en sous-agents ;
- l'encapsulation des outils dans les images Docker de sandbox ;
- le remplacement du routage par jugement du LLM par des commandes déterministes ;
- les frontières de sécurité entre Tiron, wiki, scout et les outils externes.

Codex doit privilégier les risques concrets, les écarts entre architecture cible et état réel, les problèmes de reproductibilité, les failles de sécurité, les incohérences de configuration, et les tests manquants.

## Destination des sorties

Les commentaires longs, résumés de session, revues et notes de critique doivent être écrits dans ce répertoire, en Markdown, avec un nom de fichier horodaté.

Format recommandé :

```text
YYYY-MM-DDTHH-MM-SS+ZZZZ-sujet.md
```

Exemple :

```text
2026-06-15T18-08-18+0200-revue-secretarius-slm.md
```
