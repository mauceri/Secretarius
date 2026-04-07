# Commandes Secretarius Prototype

Ce document regroupe :
- les commandes `/` directement reconnues par le routeur local ;
- les scripts utilitaires ajoutes pour la reprise de session et la gestion de Milvus.

Documents complementaires :
- [`Prototype/memos.md`](/home/mauceric/Secretarius/Prototype/memos.md) pour l'usage du canal Memos
- [`Prototype/spec_secretarius_markdown.md`](/home/mauceric/Secretarius/Prototype/spec_secretarius_markdown.md) pour le DSL Markdown `secretarius`

## Commandes `/`

Les commandes ci-dessous sont traitees directement par le routeur dans `core/chef_orchestre.py`.
Elles contournent le passage par le LLM routeur et appellent l'outil cible de facon deterministe.

### `/exp`

Usage :
```text
/exp <texte>
```

Effet :
- appelle l'outil `extract_expressions`
- extrait les expressions caracteristiques d'un texte brut

Exemples :
```text
/exp Bonjour, je cherche les themes principaux de ce passage.
```

```text
/exp Le regiment de cavalerie progresse vers l'est en territoire sovietique.
```

### `/index`

Usage :
```text
/index <texte documentaire>
```

Effet :
- appelle l'outil `index_text`
- analyse le texte documentaire
- extrait les expressions utiles
- calcule les embeddings
- enregistre le contenu dans Milvus

Le payload peut etre sur une seule ligne ou apres un saut de ligne.

Exemples :
```text
/index Titre: Notes sur la cavalerie rouge #URSS https://exemple.org
```

```text
/index
doc_id: note-cavalerie-001
type_note: permanente
title: Cavalerie rouge
https://fr.wikipedia.org/wiki/Arm%C3%A9e_rouge
#URSS #cavalerie
Notes sur l'organisation de la cavalerie sovietique.
```

### `/req`

Usage :
```text
/req <requete>
```

Effet :
- appelle l'outil `search_text`
- transforme la requete textuelle en recherche documentaire
- renvoie les notes/documents les plus proches

Exemples :
```text
/req cavalerie #URSS
```

```text
/req organisation de l'Armee rouge
```

Exemple notebook :
```python
res = secretarius("/req cavalerie #URSS")
```

### `/update`

Usage :
```text
/update <texte documentaire avec doc_id: ...>
```

Effet :
- appelle l'outil `update_text`
- remplace completement une note deja indexee
- supprime les lignes precedentes associees au `doc_id`
- reindexe le nouveau contenu

Important :
- `doc_id: ...` est obligatoire pour `/update`

Exemples :
```text
/update
doc_id: note-cavalerie-001
type_note: permanente
title: Cavalerie rouge
#URSS #cavalerie
Version mise a jour de la note sur la cavalerie sovietique.
```

```text
/update doc_id: note-cavalerie-001
Correction rapide du contenu documentaire.
```

## Scripts utilitaires

## `session_resume.py`

Fichier :
- `Prototype/session_resume.py`

Objectif :
- reprendre la derniere session locale ;
- enregistrer un snapshot Markdown de travail ;
- produire des snapshots automatiques avec garde-fous.

Commandes :
```bash
cd /home/mauceric/Secretarius/Prototype
source /home/mauceric/Secretarius/.venv/bin/activate

python session_resume.py resume --last
python session_resume.py snapshot --title "..." --summary "..." --next-step "..." --notes "..."
python session_resume.py snapshot-auto
```

Equivalents `make` :
```bash
make resume-last
make snapshot-session TITLE="..." SUMMARY="..." NEXT="..." NOTES="..."
make snapshot-auto
```

Notes :
- `resume --last` affiche le snapshot le plus recent s'il existe ;
- sinon il fabrique un resume de secours a partir de `continuation.md` et de l'etat git ;
- `snapshot-auto` n'ecrit rien si aucun changement utile n'est detecte ou si le dernier snapshot est trop recent.

## `scripts/snapshot_auto.sh`

Fichier :
- `Prototype/scripts/snapshot_auto.sh`

Objectif :
- wrapper shell pour lancer `session_resume.py snapshot-auto`
- utilise par le timer `systemd --user`

Usage :
```bash
/home/mauceric/Secretarius/Prototype/scripts/snapshot_auto.sh
```

## `scripts/milvus_collection_io.py`

Fichier :
- `Prototype/scripts/milvus_collection_io.py`

Objectif :
- exporter une collection Milvus dans un fichier JSON ;
- recharger une collection depuis un fichier du meme format ;
- supprimer proprement une collection.

Le dump JSON contient notamment :
- le nom de collection ;
- le schema ;
- les index ;
- les lignes de la collection ;
- les champs dynamiques.

### Export

```bash
cd /home/mauceric/Secretarius/Prototype
source /home/mauceric/Secretarius/.venv/bin/activate

python scripts/milvus_collection_io.py export \
  --output /tmp/secretarius_semantic_graph_dump.json
```

Par defaut :
- si `--collection-name` n'est pas fourni, le script prend la collection courante definie dans `config.yaml` ;
- si `--output` n'est pas fourni, le script cree un fichier dans `Prototype/backups/` avec un nom derive de la collection.

Exemple implicite avec `config.yaml` :
```bash
python scripts/milvus_collection_io.py export
```

### Import

```bash
python scripts/milvus_collection_io.py import \
  --input /tmp/secretarius_semantic_graph_dump.json \
  --drop-if-exists
```

Par defaut :
- si `--collection-name` n'est pas fourni, l'import cible la collection courante definie dans `config.yaml`.

### Suppression propre

```bash
python scripts/milvus_collection_io.py drop \
  --require-exists
```

Notes :
- `--collection-name` reste prioritaire si vous voulez viser explicitement une autre collection que celle du `config.yaml` ;
- l'import recree une collection compatible avec le dump ;
- `--drop-if-exists` permet de remplacer une collection existante ;
- `drop` affiche `DROPPED` si la collection a ete supprimee, sinon `SKIPPED` si elle n'existe pas et que `--require-exists` n'est pas demande.

## Scripts d'installation `systemd --user`

Ces scripts ne sont pas des commandes `/`, mais ils sont utiles pour l'exploitation locale.

### Services Secretarius

Fichier :
- `deploy/scripts/setup_secretarius_services.sh`

Effet :
- installe et active :
  - `secretarius_ollama.service`
  - `secretarius_server.service`

Usage :
```bash
bash /home/mauceric/Secretarius/deploy/scripts/setup_secretarius_services.sh
```

### Open WebUI

Fichier :
- `deploy/scripts/setup_open_webui_service.sh`

Effet :
- installe et active `open-webui.service`

Usage :
```bash
bash /home/mauceric/Secretarius/deploy/scripts/setup_open_webui_service.sh
```

### Snapshot automatique

Fichier :
- `deploy/scripts/setup_prototype_session_snapshot_timer.sh`

Effet :
- installe et active le timer `secretarius-prototype-session-snapshot.timer`

Usage :
```bash
bash /home/mauceric/Secretarius/deploy/scripts/setup_prototype_session_snapshot_timer.sh
```

### Milvus

Fichier :
- `deploy/scripts/setup_milvus_compose_service.sh`

Effet :
- installe et active `milvus-compose.service`
- relance `docker compose -f /home/mauceric/milvus/docker-compose.yml up -d`

Usage :
```bash
bash /home/mauceric/Secretarius/deploy/scripts/setup_milvus_compose_service.sh
```
