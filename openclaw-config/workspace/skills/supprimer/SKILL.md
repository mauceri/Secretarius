---
name: supprimer
description: "Supprimer une page du wiki et sa cascade. Dispatch déterministe vers wiki_delete."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_delete
command-arg-mode: raw
---

`/supprimer <page>` supprime une page du wiki et sa cascade sur les pages
liées, de façon déterministe via l'agent wiki. Garde-fou renforcé, même par
rapport aux autres écritures : passe **toujours** par un essai à blanc
(annonce des pages affectées) puis exige `/confirm` (valable 10 min), même
tapée explicitement ou appelée en tant qu'outil — jamais de suppression
directe, car l'ampleur d'une cascade n'est pas prévisible depuis la
commande seule.
