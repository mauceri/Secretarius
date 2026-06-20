# Intégration Wikipedia offline (ZIM) — préparation bascule prod

Date : 2026-06-20. Branche : `feat/recouvrement-prod-slm`.

## État

- **Code Codex commité** (`a3fe4fa`) : `wiki_lookup.py` supporte `WIKI_ZIM_DIR`,
  `WIKI_LOOKUP_OFFLINE`, `WIKI_LOOKUP_BACKENDS` (ordre défaut `zim,cache,api`).
  Tests 13/13. ZIM (16 Go) dans `Wiki_LM/zim/` (gitignoré). Appelé par `ingest.py`.
- **Offline validé en local hôte** (lookup "Paris" depuis ZIM, sans réseau).
- **NON actif dans le sous-agent wiki SLM** : le conteneur wiki ne voit pas les ZIM.

## ⛔ DÉCISION À TRANCHER (bloque l'intégration SLM) — accès ZIM au conteneur

Le conteneur wiki monte `tools` (ro), `/Wiki_LM` (rw), `euria-key`. Pas `zim/`.
Tant que ce n'est pas résolu, l'ingestion dans le sous-agent retombe sur l'**API
Wikipedia en ligne** (backend `api`).

| Option | Effort | Pour | Contre |
|--------|--------|------|--------|
| **A. Bind mount** `zim:/zim:ro` + env | ~5 min | marche tout de suite ; la config wiki a déjà `dangerouslyAllowExternalBindSources: true` ; lecture seule | couple au chemin hôte (pas portable VPS) ; perce le sandbox (ce que Codex voulait éviter) |
| **B. Serveur HTTP libzim** (décision Codex) | conséquent, **non fait** | propre, portable VPS, sandbox intact, contrat JSON `backend:"zim"` | gros chantier (contrat, libzim, systemd, backend HTTP dans WikiLookup, auth, tests) |

**Ma recommandation** : **A maintenant** pour débloquer et passer l'ingestion SLM en
offline sur sanroque (réversible, ro) ; **B plus tard** pour la portabilité VPS (pas
urgent — Google Drive est aussi repoussé). Mais c'est **votre** arbitrage (Codex a
explicitement choisi B).

## Variables d'environnement (selon l'option retenue)

À ajouter à l'env de l'agent wiki dans `~/.openclaw-slm/openclaw.json` (+ template
versionné `openclaw-config/…`) :
- Option A : bind `"/home/mauceric/Secretarius/Wiki_LM/zim:/zim:ro"` + env
  `WIKI_ZIM_DIR=/zim`, `WIKI_LOOKUP_OFFLINE=1`.
- Option B : env `WIKI_LOOKUP_OFFLINE=1` + URL du service HTTP (à définir).

**Important** : ne PAS activer `WIKI_LOOKUP_OFFLINE=1` dans le conteneur **avant**
d'avoir résolu l'accès ZIM — sinon l'ingestion échoue (ni ZIM, ni API).

## Nettoyage des commits (proposition — à valider)

La branche `feat/recouvrement-prod-slm` mélange 3 sujets : (1) recouvrement gog/wiki
+ image, (2) serveur Obsidian (auto-reload, timer, doc), (3) Wikipedia offline.
Options : laisser tel quel (historique fin, lisible) ; ou regrouper par sujet. Le
rebase interactif n'est pas disponible dans cet environnement. À préciser : que
voulez-vous « nettoyer » exactement (squash ? messages ? branches séparées) ?

## Checklist bascule prod (à clarifier puis exécuter)

À clarifier d'abord : **« production » = quoi ?** (faire du gateway SLM la prod ?
intégrer dans le gateway prod 18789 ? merger la branche dans `main` ?)

Une fois clarifié : trancher option A/B → appliquer env/bind → restart wiki →
ingestion test offline → vérifier 0 appel API Wikipedia → merge/déploiement →
`git push` (confirmation requise).
