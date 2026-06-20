# Memoire Codex - Secretarius

Derniere mise a jour : 2026-06-19

## Contexte

- Branche courante : `feat/recouvrement-prod-slm`.
- L'utilisateur travaille aussi avec Claude Code et souhaite limiter les pertes de contexte entre agents.
- Les skills Superpowers ont ete installes dans `~/.codex/skills/` depuis `obra/superpowers`.
- Redemarrer Codex pour que les nouveaux skills soient disponibles nativement.

## Revue de branche recente

Points principaux trouves lors de la revue :

- `derisk-deleg/src/index.ts` : `/confirm` matche meme au milieu d'un message, donc un brouillon email peut etre envoye par erreur si le message contient `/confirm`. Recommandation : matcher une commande exacte apres `trim`, par exemple `^/(confirm|annuler)\\s*$`.
- `openclaw-config/install.sh` / `openclaw-slm.json.template` : le plugin `derisk-deleg` et le workspace wiki SLM ne semblent pas deployes par l'installation SLM reproductible.
- `Wiki_LM/tools/wiki.py` / `kb_update.py` : `wiki_kb_update` utilise des chemins par defaut hors volume `/Wiki_LM` dans le conteneur SLM.
- `openclaw-config/wiki-lm-server.service` : le serveur Flask demarre sans `--no-public`, donc bind `0.0.0.0` par defaut.
- Les ops `wiki_tags` et `wiki_kb_update` ne sont pas encore documentees dans l'AGENTS wiki SLM.

Validations executees :

- `npm test` dans `derisk-deleg` : OK.
- `npm run build` dans `derisk-deleg` : OK.
- `pytest Wiki_LM/tests/test_wiki_cli.py` : OK.

## Sujet en cours : Wikipedia local pour ingest

Etat au 2026-06-19 19:11 +0200 :

- Increment implementation termine dans le worktree, pas encore commit.
- Fichiers ZIM presents dans le repertoire par defaut :
  - `Wiki_LM/zim/wikipedia_fr_all_mini_2026-05.zim`
  - `Wiki_LM/zim/wikipedia_en_all_mini_2026-06.zim`
- `Wiki_LM/tools/wiki_lookup.py` supporte maintenant :
  - `WIKI_ZIM_DIR`
  - `WIKI_LOOKUP_OFFLINE=1|true|yes|on`
  - `WIKI_LOOKUP_BACKENDS=zim,cache,api` avec valeurs inconnues ignorees
  - fallback par defaut `zim,cache,api`
- `Wiki_LM/tests/test_wiki_lookup.py` contient les tests env/backends/offline.
- `Wiki_LM/README.md` documente les variables.
- Plan cree : `docs/superpowers/plans/2026-06-19-wikipedia-offline-lookup.md`.
- Note de reprise utilisateur : `docs/critique-codex/2026-06-19T19-11-17+0200-reprise-wikipedia-offline.md`.

Validations executees apres implementation :

- `pytest Wiki_LM/tests/test_wiki_lookup.py -q` : 13 passed.
- `pytest Wiki_LM/tests/test_wiki_cli.py -q` : 17 passed.
- Test manuel offline strict :
  - commande : `env WIKI_LOOKUP_OFFLINE=1 PYTHONPATH=Wiki_LM/tools python3 -c "... WikiLookup('Wiki_LM') ... lookup('Paris', langs=['fr']) ..."`
  - resultat : ZIM detectes `['fr', 'en']`, lookup `Paris` retourne un extrait FR depuis ZIM.

Prochaines actions recommandees :

1. Pour usage manuel :
   - `cd ~/Secretarius/Wiki_LM`
   - `export WIKI_LOOKUP_OFFLINE=1`
   - `python tools/build_wiki_cache.py`
   - puis tester `python tools/ingest.py --raw` ou une source unique.
2. Pour l'instance SLM/OpenClaw :
   - propager `WIKI_LOOKUP_OFFLINE=1` dans l'environnement reel du sous-agent wiki ou du service qui lance les outils wiki ;
   - inspecter probablement `openclaw-config/openclaw-slm.json.template`, `openclaw-config/gateway-slm.systemd.env.template`, et la config reelle sous `~/.openclaw-slm/` ;
   - redemarrer/recharger le service apres modification ;
   - verifier que l'agent wiki n'appelle plus l'API Wikipedia.
3. Si la resolution ZIM est insuffisante, prochain increment code : normalisation titres / redirects / alias / index local, sans modifier d'abord `ingest.py`.


## Decision de reprise : service ZIM local dedie

Decision prise le 2026-06-19 :

- Ne pas incorporer les archives ZIM dans l image Docker `secretarius-wiki` : elles representent environ 16 Gio et alourdiraient fortement les builds, transferts et mises a jour de l image.
- Eviter les bind mounts OpenClaw pour les ZIM. Ils percent la frontiere du sandbox et une source exterieure au workspace impose `dangerouslyAllowExternalBindSources: true`.
- Preference retenue : petit service HTTP dedie, execute directement sur l hote et gere par un service utilisateur systemd.
- Le service lit les ZIM locaux en lecture seule via `libzim`. Le sandbox wiki ne recoit que les resultats JSON, jamais un acces aux archives.
- API volontairement minimale, par exemple une recherche de resume par langue et titre. Ne pas exposer l API Kiwix complete.
- Aucun fallback vers l API Wikipedia : une absence dans les ZIM doit produire une reponse explicite, pas un acces reseau silencieux.
- Chaque reponse doit inclure `backend: "zim"`, le nom de l archive utilisee, le titre et la langue resolus.
- Le service ne doit pas etre expose sur le LAN ou Internet. Il doit ecouter uniquement sur une interface locale accessible depuis le pont Docker, avec un jeton d acces, des limites de requete et des journaux systemd.
- Le statut d ingestion destine a Telegram devra rendre le backend observable, idealement avec un bilan `zim/cache/api`.

Ce dispositif rend les recherches Wikipedia offline, mais l ingestion complete peut encore utiliser le reseau pour recuperer la source et appeler le LLM EurIA.

Prochaine etape : concevoir puis implementer le contrat HTTP du service ZIM, son unite systemd, le client `WikiLookup` et les traces retournees par l ingestion.

## Coordination Claude Code / Codex

Regle simple conseillee :

- Un agent possede un lot de fichiers a la fois.
- Pour le sujet Wikipedia local, Codex peut prendre :
  - `Wiki_LM/tools/wiki_lookup.py`
  - `Wiki_LM/tests/test_wiki_lookup.py`
  - `Wiki_LM/README.md`
- L'increment sur ces trois fichiers est maintenant fait. Eviter que Claude Code les modifie sans relire le diff courant.
- Claude Code peut continuer sur les plans Superpowers, l'integration SLM, ou les scripts d'installation.

