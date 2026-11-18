# Canal Session, phase 2 (envois spontanés) — plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** rappels `/cron` livrés sur Session ; avis de livraison, réponses et avis de la relève par courriel Proton ; alerte de panne du broker par courriel.

**Architecture:** `tiron_cron` reconnaît le canal `session` (livraison `--announce --channel session`, déjà vérifiée en réel le 2026-11-18). Un module `tiron_mail/avis.py` envoie un courriel de `tiron@secretarius.co` à l'adresse d'avis ; il remplace `notifier_telegram` partout. La relève récupère le texte de la réponse (`openclaw agent --json`, `result.payloads[].text`) au lieu de `--deliver` et le renvoie en « Re: » à l'expéditeur. Une minuterie utilisateur appelle `python3 -m tiron_mail.veille_session`.

**Tech Stack:** Python 3 (stdlib), pytest, systemd utilisateur.

**Spec:** `docs/superpowers/specs/2026-11-18-canal-session-design.md` (§2, §5.6, §5.7, §6 phase 2)

## Global Constraints

- Adresse des avis et alertes : `christian@mauceri.me` (décision utilisateur 2026-11-18), lue dans `~/.config/tiron_mail/comptes.toml`, `[envoi] avis = "…"` ; elle DOIT figurer dans `adresses_proprietaire`, sinon `ErreurConfig`.
- Expéditeur des avis : boîte `envoi.defaut` (tiron@secretarius.co).
- `/confirm` reste refusé par courriel (inchangé, `_PAR_COURRIEL`).
- Origine `/cron` inconnue ou vide → canal `session` (décision utilisateur 2026-11-18). Les rappels déjà créés ne changent pas.
- Destinataire Session : variable `TIRON_SESSION_ID` (gateway, via `/maj` : `TIRON_*` déjà en liste blanche).
- La phrase secrète (`SESSION_BOT_MNEMONIC`) n'entre jamais dans l'environnement de la veille : la veille lit seulement `TIRON_SESSION_BROKER_TOKEN` dans `~/.config/secrets.env`, ligne par ligne, apostrophes retirées.
- Alerte : seuil 15 min sur `lastPollOkAt`, minuterie toutes les 5 min, une seule alerte par incident, un courriel de rétablissement. Marqueur : `~/.local/state/session-broker/veille.json`.
- Textes : « Session : Tiron ne reçoit plus depuis HH:MM (<motif>). » et « Session : Tiron reçoit de nouveau (panne depuis HH:MM). »
- Message de la relève pour une commande non admise : « … non admis par courriel : à taper sur Session ou Telegram. »

## Review Focus

- Réponse vide de l'agent (payloads vides, `NO_REPLY`) : aucun courriel vide n'est envoyé → test `test_reponse_vide_pas_de_courriel` (tâche 3).
- Échec SMTP de la réponse : la relève continue et le message reste rangé (jamais réexécuté) → `test_reponse_en_echec_n_arrete_pas_la_releve` (tâche 3).
- Veille : fichier d'état illisible → traité comme « pas d'alerte en cours », pas de plantage → `test_etat_illisible` (tâche 4).
- Veille : échec d'envoi de l'alerte → réessai au passage suivant (marqueur non posé) → `test_alerte_reessayee_si_envoi_echoue` (tâche 4).
- Veille : `secrets.env` sans jeton → panne signalée (motif « jeton absent »), jamais d'exception non rattrapée → `test_jeton_absent` (tâche 4).

---

### Task 1: canal `session` dans `tiron_cron`

**Files:**
- Modify: `tiron_cron/syntax.py` (CANAUX), `tiron_cron/nl.py` (_CANAL), `tiron_cron/openclaw_cli.py` (CANAUX_OPENCLAW, DESTINATAIRES_ENV, message d'erreur), `tiron_cron/__main__.py` (`_origine`), `tiron_cron/fallback.py` (liste des canaux du prompt)
- Test: `tiron_cron/tests/test_openclaw_cli.py`, `test_nl.py`, `test_syntax.py`, `test_cli.py`

**Interfaces:** Produces `CANAUX` incluant `"session"` ; `_origine("")` → `"session"`.

- [ ] Step 1: tests
  - `test_rappel_session` : `build_add_argv(Job rappel, "session", tz, {"TIRON_SESSION_ID": "05ab"}, key)` se termine par `["--announce", "--channel", "session", "--to", "05ab"]`.
  - `test_session_sans_identifiant` : sans `TIRON_SESSION_ID` → `SyntaxeInvalide` contenant `TIRON_SESSION_ID`.
  - `test_nl_sur_session` : « rappelle-moi demain 9h sur session de payer » → `job.canal == "session"`.
  - `test_syntaxe_canal_session` : `/cron add <quand> session rappel x` → `canal == "session"`.
  - `test_origine_inconnue_va_sur_session` (test_cli) : `preview --origin "" ` d'un rappel avec `TIRON_SESSION_ID` dans env → `ok` et texte mentionnant session ; `--origin session` idem ; `--origin telegram` reste telegram.
- [ ] Step 2: les lancer → FAIL.
- [ ] Step 3: implémenter (ajouts de tables ; `_origine` renvoie `"session"` si l'origine n'est pas dans CANAUX).
- [ ] Step 4: `python3 -m pytest tiron_cron -q` → PASS.
- [ ] Step 5: commit `feat(cron): canal session, origine inconnue → session`.

### Task 2: avis par courriel (`tiron_mail/avis.py`)

**Files:**
- Modify: `tiron_mail/config.py` (champ `avis: str = ""` ; validation dans `charger`), `tiron_mail/comptes.toml.modele`
- Create: `tiron_mail/avis.py`
- Modify: `tiron_cron/__main__.py` (`livrer` → `notifier_courriel` ; « avis non envoyé »), `tiron_cron/livraison.py` (docstring, retrait de `notifier_telegram`), `tiron_mail/__main__.py` (relève → `notifier_courriel`)
- Test: `tiron_mail/tests/test_avis.py`, `tiron_mail/tests/test_config.py`

**Interfaces:** Produces `notifier_courriel(env: Mapping[str, str], smtp=_smtp) -> Callable[[str], None]` : charge la config (`TIRON_MAIL_CONFIG` ou défaut), envoie de `cfg.comptes[cfg.envoi_defaut]` à `cfg.avis`, objet `"Secretarius : " + première ligne[:60]`, corps = texte ; lève `ErreurConfig` si `avis` vide.

- [ ] Step 1: tests — `test_avis_hors_proprietaire_refuse` (config) ; `test_avis_envoye_a_l_adresse_d_avis` (faux smtp : To = avis, From = tiron, Subject commence par « Secretarius : ») ; `test_avis_non_configure`.
- [ ] Step 2: FAIL.
- [ ] Step 3: implémenter ; brancher `livrer` et `releve` sur `notifier_courriel`.
- [ ] Step 4: `python3 -m pytest tiron_mail tiron_cron -q` → PASS.
- [ ] Step 5: commit `feat(avis): avis de livraison et de relève par courriel Proton`.

### Task 3: réponse de la relève par courriel

**Files:**
- Modify: `tiron_mail/releve.py` (`executer_openclaw` sans `--deliver`, renvoie le texte ; `relever(…, repondre=None)` ; libellé Session), `tiron_mail/__main__.py`
- Test: `tiron_mail/tests/test_releve.py`

**Interfaces:**
- `executer_openclaw(env) -> Callable[[str], str]` : lance `openclaw agent … --json` (plus de `--deliver/--reply-channel/--reply-to`), renvoie les `result.payloads[].text` non vides joints par une ligne vide.
- `repondeur_courriel(cfg, env, smtp=_smtp) -> Callable[[str, str, str, str], None]` : `(dest, objet, corps, en_reponse_a)` ; envoie depuis `envoi_defaut`, refuse un `dest` hors `adresses_proprietaire`.
- `relever(cfg, env, executer, notifier, etat_chemin=…, connecter=…, repondre=None)` : par message accepté, réunit `"<commande>\n\n<réponse>"` de chaque commande à réponse non vide ; si au moins une, un seul courriel `Re: <objet>` à l'expéditeur, `In-Reply-To` = Message-ID. Échec d'envoi → stderr, la relève continue.

- [ ] Step 1: tests — `test_reponse_renvoyee_a_l_expediteur` (une réponse, objet « Re: … », in-reply-to) ; `test_deux_commandes_un_seul_courriel` ; `test_reponse_vide_pas_de_courriel` ; `test_reponse_en_echec_n_arrete_pas_la_releve` ; `test_non_admis_mentionne_session` ; `test_executer_openclaw_sans_livraison` (monkeypatch subprocess : argv sans `--deliver`, renvoie le texte du JSON) ; mettre à jour `test_consigne_confiee_a_l_agent_consignes`.
- [ ] Step 2: FAIL.
- [ ] Step 3: implémenter.
- [ ] Step 4: `python3 -m pytest tiron_mail -q` → PASS.
- [ ] Step 5: commit `feat(releve): réponses aux commandes par courriel`.

### Task 4: veille du broker

**Files:**
- Create: `tiron_mail/veille_session.py`, `session-broker/systemd/session-broker-veille.service`, `session-broker/systemd/session-broker-veille.timer`
- Modify: `session-broker/README.md` (installation de la minuterie)
- Test: `tiron_mail/tests/test_veille_session.py`

**Interfaces:**
- `lire_jeton(chemin: Path) -> str` : valeur de `TIRON_SESSION_BROKER_TOKEN`, apostrophes retirées, `""` si absente ; ne lit aucune autre clé.
- `sonder(jeton, url="http://127.0.0.1:8091/health", ouvrir=urlopen) -> dict`.
- `veiller(sonde: Callable[[], dict], notifier, maintenant: datetime, etat_chemin: Path, seuil=timedelta(minutes=15)) -> str` : panne si la sonde lève, si `lastPollOkAt` est nul ou plus vieux que le seuil ; état `{"panne_depuis": iso|None, "alerte": bool}` écrit atomiquement.
- `main()` : lit le jeton (`~/.config/secrets.env`), notifier = `notifier_courriel(os.environ)`.

- [ ] Step 1: tests — `test_ok_rien` ; `test_panne_une_seule_alerte` (deux passages → un avis) ; `test_retablissement` ; `test_poll_trop_ancien` (16 min) ; `test_alerte_reessayee_si_envoi_echoue` ; `test_etat_illisible` ; `test_jeton_absent` ; `test_lire_jeton_n_extrait_que_le_jeton` (fichier avec `SESSION_BOT_MNEMONIC='a b'` : seule la valeur du jeton est renvoyée).
- [ ] Step 2: FAIL.
- [ ] Step 3: implémenter ; unités : service `Type=oneshot`, `WorkingDirectory=%h/Secretarius`, `EnvironmentFile=%h/Secretarius/Wiki_LM/.env`, `ExecStart=/usr/bin/python3 -m tiron_mail.veille_session` ; timer `OnBootSec=5min`, `OnUnitActiveSec=5min`.
- [ ] Step 4: `python3 -m pytest tiron_mail -q` → PASS.
- [ ] Step 5: commit `feat(session): veille du broker et alerte par courriel`.

### Task 5: mise en service et essai réel (avec l'utilisateur)

- [ ] `avis = "christian@mauceri.me"` dans `~/.config/tiron_mail/comptes.toml`.
- [ ] `TIRON_SESSION_ID=<identifiant utilisateur>` dans `secrets.env`, puis `/maj` (accord : redémarre le gateway).
- [ ] Fusion, push, installation des unités de veille, `systemctl --user enable --now session-broker-veille.timer` (accord).
- [ ] Critères spec : rappel posé depuis Session → Session ; livraison `/cron` → courriel ; broker arrêté → alerte ≤ 20 min puis rétablissement.
