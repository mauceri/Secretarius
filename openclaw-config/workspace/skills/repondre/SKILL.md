---
name: repondre
description: "Préparer une réponse à un email (brouillon + /confirm). Dispatch déterministe vers gog_reply."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: gog_reply
command-arg-mode: raw
---

`/repondre <id> <texte>` prépare un brouillon de réponse (n'envoie pas) ; taper /confirm pour envoyer, /annuler pour abandonner.
