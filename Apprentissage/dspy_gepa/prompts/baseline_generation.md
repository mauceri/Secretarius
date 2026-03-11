# Baseline generation

Objectif : generer un exemple d'entrainement JSON unique a partir d'une graine source manuelle.

Contraintes de sortie :

- respecter strictement le schema `schemas/synthetic_record.schema.json`
- produire un JSON unique, sans markdown
- conserver la langue francaise
- formuler une instruction claire et exploitable pour un futur fine-tuning
- produire une reponse concise, correcte et directement utile
- renseigner `metadata.synthetic=true`

Usage :

- ce fichier sert de contrat lisible pour le script baseline et le notebook
- il ne contient pas de logique GEPA a ce stade
