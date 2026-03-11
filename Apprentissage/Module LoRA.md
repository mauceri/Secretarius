Voici le document **entièrement dans une seule boîte Markdown**, sans identifiants ni directives parasites.

```markdown
# Secretarius — Contexte d'utilisation du module LoRA

## Objet du document

Ce document décrit le rôle attendu du module LoRA entraîné dans le cadre du projet Secretarius.

Il sert de référence pour :
- la génération de corpus synthétique (DSPy / GEPA),
- l'entraînement LoRA,
- les tests d'évaluation,
- et l'intégration ultérieure dans le pipeline d'indexation.

Ce document ne décrit pas l'ensemble de l'architecture Secretarius, mais uniquement la fonction que doit remplir le module appris.

---

# 1. Contexte général

Secretarius est un système local d'archivage et de recherche sémantique basé sur :

- l'extraction d'expressions caractéristiques,
- la génération d'embeddings,
- l'indexation vectorielle dans Milvus,
- une recherche sémantique avec restitution de contexte.

Actuellement, l'indexation se fait principalement à partir de textes bruts fournis par l'utilisateur.

Cependant, les textes soumis peuvent avoir des formes très variées :

- fragments informels,
- notes rapides,
- citations,
- textes littéraires,
- documents structurés,
- notes contenant des métadonnées visibles.

Pour améliorer la qualité de l'indexation et la restitution des résultats, il est nécessaire d'introduire une reconnaissance légère de la morphologie documentaire.

---

# 2. Rôle du module appris

Le module LoRA doit permettre de transformer un texte brut fourni par l'utilisateur en une description morphologique minimale du document.

Il ne doit pas :

- analyser en profondeur le contenu sémantique,
- produire des résumés,
- inventer des métadonnées absentes,
- ni remplacer les modules d'extraction existants.

Son rôle est uniquement de produire un petit objet JSON structuré décrivant :

- la forme probable du texte,
- l'existence éventuelle d'un titre,
- quelques mots-clés évidents si présents.

---

# 3. Position dans le pipeline

Le module intervient après la réception d'un texte brut et avant les étapes d'extraction et d'indexation.

Pipeline simplifié :

texte utilisateur  
↓  
analyse morphologique  
↓  
création du document parent  
↓  
extraction d'expressions  
↓  
génération d'embeddings  
↓  
indexation  

L'analyse morphologique peut être réalisée :

1. par heuristiques locales,
2. puis, si nécessaire, par le module LoRA.

Le module appris doit rester optionnel et ne jamais bloquer le pipeline.

---

# 4. Contrat d'entrée

Entrée :

texte brut

Le texte peut être :

- très court,
- multiligne,
- contenir des métadonnées,
- contenir des vers ou des paragraphes.

---

# 5. Contrat de sortie

Le module doit produire uniquement un objet JSON valide.

Exemple :

{
  "morphology_class": "note_semi_structuree",
  "document_kind": "note",
  "title": "Le Dormeur du val",
  "keywords": ["soldat", "nature", "mort"],
  "confidence": 0.82
}

Contraintes :

- classes fermées,
- pas de texte explicatif,
- maximum 5 mots-clés,
- "title" uniquement si identifiable clairement.

---

# 6. Classes morphologiques

Le modèle doit choisir parmi un ensemble fermé de classes.

Exemple initial :

- fragment_informel
- note_semi_structuree
- document_structure
- poetic_or_verse_like
- list_like

Ces classes peuvent évoluer avec le projet.

---

# 7. Contraintes de conception

Le module LoRA doit rester :

- léger,
- déterministe autant que possible,
- stable sur les sorties JSON.

Il ne doit pas nécessiter de GPU puissant et doit pouvoir fonctionner localement.

---

# 8. Relation avec DSPy / GEPA

DSPy et GEPA seront utilisés pour :

- générer un corpus synthétique d'exemples,
- optimiser les prompts de génération,
- produire un dataset d'entraînement.

Ce dataset servira ensuite à entraîner un adaptateur LoRA sur un micro-modèle.

---

# 9. Modèle cible

Modèle envisagé :

Qwen3-0.6B

Objectif :

spécialiser ce modèle pour produire des sorties JSON stables pour cette tâche précise.

---

# 10. Principe directeur

Le module appris ne doit jamais remplacer les heuristiques simples lorsque celles-ci sont suffisantes.

La philosophie générale reste :

heuristiques  
↓  
micro-modèle si ambigu  

et non l'inverse.

---

# 11. Objectif final

Disposer d'un module très léger capable de reconnaître la forme documentaire d'un texte brut et de produire une structure JSON exploitable pour l'indexation.
```
