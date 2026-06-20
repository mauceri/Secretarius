---
name: connecter
description: "Connecter (autoriser) votre compte Google. Dispatch déterministe vers gog_connect_start."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: gog_connect_start
command-arg-mode: raw
---

`/connecter` démarre l'autorisation Google : un lien vous est envoyé, vous autorisez puis recollez l'URL de redirection.
