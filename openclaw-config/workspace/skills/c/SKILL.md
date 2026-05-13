---
name: c
description: Capture rapide vers raw/ Wiki_LM. Déclencher sur /c [#tags] [urls|texte] ou URL nue (partage Android). Crée des fichiers .url (URLs seules) ou .md (texte, ou texte+URL) horodatés dans raw/.
---

# Skill : c

## Rôle

Capture rapide depuis Telegram : sauvegarde une ou plusieurs URLs, ou un commentaire libre,
dans `raw/` pour ingestion ultérieure dans le wiki personnel.

Les tokens `#motclé` sont extraits comme tags et normalisés contre le dictionnaire
de tags canoniques du système (best-effort).

## Commandes

### Capturer une ou plusieurs URLs

```
/c https://example.com
/c https://url1.com https://url2.com
```

Crée un fichier `.url` par URL.
Nom : `YYYYMMDD-HHMMSS[-index]-<domaine>.url`
Contenu : URL seule (+ ligne `tags:` si tags présents).

### Capturer un commentaire ou une note

```
/c Réflexion sur l'article de Salton concernant le modèle vectoriel
/c #memo Acheter une bouteille de Gewurztraminer pour ce soir
```

Crée un fichier `.md`.
Nom : `YYYYMMDD-HHMMSS-<incipit-slug>.md`
Contenu : frontmatter YAML avec tags si présents, puis texte.

### Capturer une URL avec un commentaire

```
/c #attention #transformer https://arxiv.org/abs/1706.03762
/c #attention #transformer note: cet article est fondateur de l'IA actuelle https://arxiv.org/abs/1706.03762
```

Crée un **seul fichier `.md`** contenant le texte et l'URL.
Le préfixe `note:` est optionnel et retiré du texte.

### Capturer un fichier joint

```
/c [#tags] [commentaire optionnel]   + pièce jointe Telegram
```

1. Sauvegarder le fichier joint dans un chemin temporaire accessible
2. Exécuter :

```bash
cd ~/Secretarius/Wiki_LM && source .venv/bin/activate && \
  python tools/capture.py --file "<chemin_temporaire>" "[#tags] [commentaire]"
```

## Tags `#motclé`

- Les tokens commençant par `#` sont extraits comme tags, dans n'importe quelle position.
- Ils sont normalisés contre les tags canoniques du système si le dictionnaire est disponible.
- Exemples : `#memo`, `#attention`, `#transformer`, `#todo`

## Comportement

### Détection du déclencheur

Ce skill s'active dans deux cas :
1. Message commençant par `/c <argument>`
2. **Message contenant uniquement une ou plusieurs URLs nues** (commençant par `http://` ou `https://`) sans préfixe de commande

### Exécution

```bash
cd ~/Secretarius/Wiki_LM && source .venv/bin/activate && python tools/capture.py "<argument>"
```

Remplacer `<argument>` par ce qui suit `/c`, ou par l'URL nue si c'est le cas 2.

Confirmer à l'utilisateur avec le ou les noms de fichiers créés.
