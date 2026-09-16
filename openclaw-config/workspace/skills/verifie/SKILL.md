---
name: verifie
description: "Marquer une page du wiki comme vérifiée. Dispatch déterministe vers wiki_verify."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_verify
command-arg-mode: raw
---

`/verifie <page>` marque une page comme vérifiée, de façon déterministe via
l'agent wiki. Écriture : ne marque jamais directement, même appelée en tant
qu'outil — prépare l'action et attend `/confirm` (valable 10 min) avant de
l'exécuter.
