# Reprise - Wikipedia offline pour ingest

Date locale : 2026-06-19 19:11:17 +0200

## Situation actuelle

Les fichiers ZIM Wikipedia sont telecharges et places dans le repertoire par defaut attendu par `WikiLookup` :

```text
~/Secretarius/Wiki_LM/zim/wikipedia_fr_all_mini_2026-05.zim
~/Secretarius/Wiki_LM/zim/wikipedia_en_all_mini_2026-06.zim
```

Le code a ete modifie pour permettre un mode Wikipedia offline explicite :

- `Wiki_LM/tools/wiki_lookup.py`
  - lit `WIKI_ZIM_DIR` si `zim_dir` n'est pas fourni ;
  - lit `WIKI_LOOKUP_OFFLINE` ;
  - lit `WIKI_LOOKUP_BACKENDS` ;
  - conserve par defaut l'ordre `zim,cache,api`.
- `Wiki_LM/tests/test_wiki_lookup.py`
  - couvre la priorite `zim_dir` explicite > `WIKI_ZIM_DIR` > defaut ;
  - couvre le mode offline sans appel API ;
  - couvre `WIKI_LOOKUP_BACKENDS=cache` ;
  - couvre l'ordre par defaut ;
  - couvre le fallback si la liste de backends est invalide.
- `Wiki_LM/README.md`
  - documente le placement des fichiers ZIM ;
  - documente `WIKI_ZIM_DIR`, `WIKI_LOOKUP_OFFLINE` et `WIKI_LOOKUP_BACKENDS`.
- `docs/superpowers/plans/2026-06-19-wikipedia-offline-lookup.md`
  - contient le plan d'implementation de cet increment.

## Validations deja executees

Tests unitaires :

```bash
pytest Wiki_LM/tests/test_wiki_lookup.py -q
```

Resultat observe :

```text
13 passed in 0.32s
```

Tests CLI wiki :

```bash
pytest Wiki_LM/tests/test_wiki_cli.py -q
```

Resultat observe :

```text
17 passed in 1.34s
```

Test manuel contre les vrais ZIM, sans reseau :

```bash
env WIKI_LOOKUP_OFFLINE=1 PYTHONPATH=Wiki_LM/tools python3 -c "from wiki_lookup import WikiLookup; wl=WikiLookup('Wiki_LM'); print('zims=', wl.zim_langs()); r=wl.lookup('Paris', langs=['fr']); print(r['lang'] if r else None); print(r['title'] if r else None); print((r['abstract'][:200] if r else 'NO RESULT'))"
```

Resultat observe :

```text
zims= ['fr', 'en']
fr
Paris
Paris (/pa.ʁi/ ), officiellement la Ville de Paris, est la capitale de la France, ...
```

Conclusion : la detection des ZIM et le lookup offline fonctionnent localement.

## Ce qu'il faut faire maintenant

### 1. Activer le mode offline dans le shell de travail

Pour les commandes manuelles :

```bash
cd ~/Secretarius/Wiki_LM
export WIKI_LOOKUP_OFFLINE=1
```

`WIKI_ZIM_DIR` n'est pas necessaire tant que les fichiers restent dans `~/Secretarius/Wiki_LM/zim`.

### 2. Prechauffer le cache Wikipedia

Depuis `~/Secretarius/Wiki_LM` :

```bash
export WIKI_LOOKUP_OFFLINE=1
python tools/build_wiki_cache.py
```

But : remplir `wiki_cache.db` a partir des ZIM locaux pour les pages existantes.

Si trop de titres ne resolvent pas, ne pas modifier tout de suite le pipeline d'ingestion. Le prochain increment logique serait la resolution de titres ZIM : normalisation, alias, redirects, ou index local.

### 3. Tester une ingestion reelle en offline

Depuis `~/Secretarius/Wiki_LM` :

```bash
export WIKI_LOOKUP_OFFLINE=1
python tools/ingest.py --raw
```

Ou tester d'abord une source unique connue.

Critere attendu : l'ingestion ne doit pas tenter l'API Wikipedia. Les extraits Wikipedia doivent venir de ZIM ou du cache SQLite.

### 4. Propager la variable dans l'instance SLM/OpenClaw

Point important : l'export shell ne suffit que pour les commandes lancees dans ce shell.

Pour que le sous-agent wiki SLM utilise aussi le mode offline, ajouter :

```text
WIKI_LOOKUP_OFFLINE=1
```

dans l'environnement qui lance les outils wiki. Les endroits probables a inspecter sont :

- `openclaw-config/openclaw-slm.json.template`
- `openclaw-config/gateway-slm.systemd.env.template`
- le fichier reel sous `~/.openclaw-slm/` qui configure le profil `slm`
- le service systemd reel qui lance OpenClaw SLM

Ne pas supposer que l'agent wiki herite de l'environnement du shell interactif.

### 5. Redemarrer le service SLM apres configuration

Apres modification de l'environnement systemd ou OpenClaw, recharger et redemarrer le service concerne, puis tester une commande wiki qui declenche `ingest`.

Verifier ensuite dans les logs qu'il n'y a pas d'appel reseau Wikipedia. Si l'observabilite n'est pas suffisante, ajouter temporairement une trace dans `_fetch_api` ou dans le wrapper d'outil, mais ne pas la laisser en production sans decision explicite.

## Etat Git a la coupure

Fichiers modifies par cet increment :

```text
Wiki_LM/README.md
Wiki_LM/tests/test_wiki_lookup.py
Wiki_LM/tools/wiki_lookup.py
docs/superpowers/plans/2026-06-19-wikipedia-offline-lookup.md
docs/critique-codex/2026-06-19T19-11-17+0200-reprise-wikipedia-offline.md
docs/critique-codex/MEMORY.md
```

Le worktree contenait deja d'autres fichiers non suivis lies aux docs d'architecture et aux notes Codex. Ne pas nettoyer ni revert sans verifier leur provenance.

## Prochaine reprise conseillee pour Codex

1. Lire ce fichier.
2. Lire `docs/critique-codex/MEMORY.md`.
3. Verifier `git status --short`.
4. Si l'utilisateur veut poursuivre l'exploitation : configurer `WIKI_LOOKUP_OFFLINE=1` dans l'environnement SLM/OpenClaw.
5. Si l'utilisateur veut poursuivre le code : ajouter l'integration de cette variable dans les templates versionnes et tester la reinstall/reload du profil SLM.


## Decision architecturale pour l acces aux ZIM depuis OpenClaw

Decision prise le 2026-06-19 :

- Ne pas copier les archives ZIM dans l image `secretarius-wiki`. Les archives FR et EN representent environ 16 Gio ; leur inclusion rendrait l image trop lourde a construire, stocker, transferer et mettre a jour.
- Ne pas ajouter un bind mount externe vers les ZIM dans le sandbox wiki. OpenClaw rappelle que les binds percent la frontiere du sandbox. Dans la configuration actuelle, un chemin hors workspace requiert en plus `dangerouslyAllowExternalBindSources: true`.
- Solution retenue : un petit serveur HTTP local, execute sur l hote et gere par une unite systemd utilisateur. Il ouvre les archives locales avec `libzim` et expose uniquement un contrat JSON minimal.

Contraintes du service :

- lecture seule des archives ;
- aucune route permettant de lire un chemin arbitraire ;
- aucun fallback vers Wikipedia en ligne ;
- ecoute limitee a une interface locale accessible depuis le pont Docker, jamais sur le LAN ou Internet ;
- authentification par jeton local ;
- limitation de la taille et du debit des requetes ;
- journaux disponibles avec `journalctl` ;
- reponses incluant obligatoirement `backend: "zim"`, le nom de l archive, la langue et le titre resolu.

Exemple de reponse attendue :

```json
{
  "backend": "zim",
  "archive": "wikipedia_fr_all_mini_2026-05.zim",
  "lang": "fr",
  "title": "Paris",
  "abstract": "..."
}
```

Integration attendue :

- `WikiLookup` appelle ce service depuis le sandbox au lieu d ouvrir les ZIM ;
- une absence ou une panne doit etre explicite et ne doit jamais declencher silencieusement l API Wikipedia ;
- le statut d ingestion visible depuis Telegram doit indiquer les backends utilises, idealement avec des compteurs `zim`, `cache` et `api` ;
- en mode strict offline, le compteur `api` doit rester a zero et toute tentative d appel doit etre consideree comme une erreur.

Attention : « Wikipedia offline » concerne uniquement la verification et l enrichissement Wikipedia. La recuperation de la page source et les appels au LLM EurIA peuvent toujours utiliser le reseau.

Prochaine etape :

1. Definir le contrat HTTP minimal et les erreurs.
2. Ajouter des tests du serveur contre les vrais ZIM et des doubles en tests.
3. Creer l unite systemd avec une exposition reseau minimale.
4. Ajouter un backend HTTP ZIM dans `WikiLookup`.
5. Propager les metadonnees de backend jusqu au statut d ingestion Telegram.
