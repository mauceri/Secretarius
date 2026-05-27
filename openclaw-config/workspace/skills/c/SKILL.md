---
name: c
description: Capture rapide vers raw/ Wiki_LM. Déclencher sur /c [#tags] [urls|texte] ou URL nue (partage Android). Délègue à wiki_capture (MCP wiki-lm).
---

# Skill : c

## Rôle

Capture rapide depuis Telegram. Transmet le contenu brut à `wiki_capture` (outil MCP wiki-lm)
qui extrait les URLs, les hashtags et le texte libre.

## Déclencheur

Ce skill s'active dans deux cas :
1. Message commençant par `/c <argument>`
2. **Message contenant uniquement une ou plusieurs URLs nues** (commençant par `http://` ou `https://`)

## Exécution

Appeler `wiki_capture` avec tout ce qui suit `/c`, ou l'URL nue si c'est le cas 2.

```
wiki_capture("<argument>")
```

Exemples :
- `/c https://example.com` → `wiki_capture("https://example.com")`
- `/c #nlp https://arxiv.org/abs/1706.03762 article fondateur` → `wiki_capture("#nlp https://arxiv.org/abs/1706.03762 article fondateur")`
- `/c #memo Acheter du Gewurztraminer` → `wiki_capture("#memo Acheter du Gewurztraminer")`
- URL nue `https://example.com` → `wiki_capture("https://example.com")`

## Réponse

Confirmer avec les noms de fichiers retournés par `wiki_capture` :
`{files: ["20260527-HHMMSS-example-com.url"]}` → "Capturé : 20260527-HHMMSS-example-com.url"

**Ne pas appeler `wiki_ingest` après `wiki_capture` sauf si l'utilisateur le demande explicitement.**

## Fichiers joints

Pour une pièce jointe Telegram, sauvegarder le fichier joint dans un chemin temporaire,
puis appeler directement `capture.py` (hors MCP) :

```bash
cd ~/Secretarius/Wiki_LM && source .venv/bin/activate && \
  python tools/capture.py --file "<chemin_temporaire>" "[#tags] [commentaire]"
```
