# Relais des courriels de Grok Bot vers Session — plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** la relève de `tiron@` relaie sur Session, comme information jamais exécutée, les courriels authentifiés des expéditeurs listés dans `[releve] relayes`.

**Architecture:** un module `tiron_mail/relais.py` (règle d'adresse, authentification DMARC/DKIM, garde d'injection, mise en forme, envoi au broker Session avec repli courriel) ; `relever` reçoit un paramètre `relayer` et oriente les relayés avant l'authentification du propriétaire.

**Tech Stack:** Python 3 (stdlib : `urllib.request`, `email`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-25-relais-grok-design.md`

## Global Constraints

- Entrée `local@domaine` → admet `local@domaine` et `local+<sel>@domaine`, `sel` en `[a-z0-9-]+`, minuscules, domaine exact.
- Relayé admis seulement si : un seul `From` ; `X-Pm-Origin` = `external` ; ≥ 1 `Authentication-Results` `mail.protonmail.ch;` avec `dmarc=`, tous `pass`, `header.from=<domaine>` ; (DKIM non exigé depuis la révision du 2026-11-25) ; `Message-ID` présent. Pas de contrôle `Return-Path`.
- Garde : `POST http://127.0.0.1:8990/check` `{"type": "text", "content": texte}` ; réponse `{"blocked": bool, "reason"?, "risk"?, "clean_text"?}`.
- Libellés exacts : en-tête `Courriel de Grok Bot (<sel>) — « <objet> »` (sans `(<sel>)` si pas de sélecteur) ; `Attention : contenu potentiellement suspect.` ; `contenu bloqué par le détecteur d'injection : <motif>` ; `contenu non vérifié (garde injoignable), consultez la boîte de Tiron` ; coupe à 3 500 caractères + `… (suite dans la boîte de Tiron)` ; repli `(Session indisponible) ` en tête.
- Broker : `POST http://127.0.0.1:8091/send`, `Authorization: Bearer <TIRON_SESSION_BROKER_TOKEN>`, `{"to": <TIRON_SESSION_ID>, "text": …}` ; ces deux clés lues dans `~/.config/secrets.env` ligne par ligne, apostrophes retirées ; aucune autre clé lue.
- Aucune commande extraite ni exécutée d'un courriel relayé.
- Une adresse dans `relayes` qui figure aussi dans `adresses_proprietaire` → `ErreurConfig`.

## Review Focus

- Courriel relayé dont le texte contient `/wikistatus` → relayé, `executer` jamais appelé → `test_relaye_jamais_execute` (tâche 3).
- `From` `yc28mybv@mail.grokbot.com` mais DMARC `fail` (usurpation) → refus, rangé dans Refuse → `test_relaye_dmarc_fail_refuse` (tâche 1) et `test_relaye_non_authentifie_refuse` (tâche 3).
- DKIM du domaine en échec mais DMARC `pass` → admis (révision 2026-11-25) → `test_dkim_du_domaine_en_echec_mais_dmarc_pass_admis` (tâche 1).
- Broker arrêté et avis courriel en échec → message rangé quand même, jamais relayé deux fois → `test_relais_en_echec_n_arrete_pas_la_releve` (tâche 3).
- Sous-domaine ou suffixe proche (`yc28mybvx@…`, `yc28mybv@x.mail.grokbot.com`) → pas relayé → `test_regle_adresse` (tâche 1).

---

### Task 1: règle d'adresse, configuration, authentification

**Files:**
- Modify: `tiron_mail/config.py` (`Config.relayes: tuple = ()`, lecture `[releve] relayes`, contrôle propriétaire), `tiron_mail/comptes.toml.modele`
- Create: `tiron_mail/relais.py`
- Test: `tiron_mail/tests/test_relais.py`, `tiron_mail/tests/test_config.py`

**Interfaces:** Produces
- `selecteur_relaye(adresse: str, relayes: tuple) -> str | None` : `None` si non relayé, `""` sans sélecteur, sinon le sélecteur.
- `authentifier_relaye(m: email.message.Message) -> str` : `""` si admis, sinon le motif (le contrôle d'adresse est fait avant, par `selecteur_relaye`).

- [ ] Step 1: tests
  - `test_regle_adresse` : avec `("yc28mybv@mail.grokbot.com",)` : `yc28mybv@mail.grokbot.com` → `""` ; `YC28MYBV+Rector@Mail.GrokBot.com` → `"rector"` ; `yc28mybvx@…`, `yc28mybv@x.mail.grokbot.com`, `yc28mybv+a b@…`, `yc28mybv+@…`, `autre@mail.grokbot.com` → `None`.
  - `test_relaye_authentique` : en-têtes du courriel d'essai de la spec (From, `X-Pm-Origin: external`, dmarc pass header.from=mail.grokbot.com, dkim pass header.d=mail.grokbot.com, Message-ID) → `""`.
  - `test_relaye_dmarc_fail_refuse`, `test_dkim_autre_domaine_refuse` (seulement `header.d=amazonses.com`), `test_origine_interne_refusee`, `test_sans_message_id_refuse`, `test_deux_from_refuse` → motif non vide.
  - `test_relayes_lus` et `test_relaye_proprietaire_refuse` (config) : `relayes` en minuscules ; adresse aussi propriétaire → `ErreurConfig` contenant `relayes`.
- [ ] Step 2: `python3 -m pytest tiron_mail/tests/test_relais.py tiron_mail/tests/test_config.py -q` → FAIL.
- [ ] Step 3: implémenter (regex construite par entrée avec `re.escape`).
- [ ] Step 4: `python3 -m pytest tiron_mail -q` → PASS.
- [ ] Step 5: commit `feat(relais): expéditeurs relayés, authentification DMARC/DKIM`.

### Task 2: garde, mise en forme, envoi Session

**Files:**
- Modify: `tiron_mail/veille_session.py` (`lire_cle(chemin, cle) -> str` ; `lire_jeton` l'appelle)
- Modify: `tiron_mail/relais.py`
- Test: `tiron_mail/tests/test_relais.py`, `tiron_mail/tests/test_veille_session.py` (inchangés et verts)

**Interfaces:** Produces
- `message_relais(selecteur: str, objet: str, texte: str, garde: Callable[[str], dict]) -> str` (garde qui lève = injoignable).
- `garde_injection(url="http://127.0.0.1:8990/check", ouvrir=urlopen) -> Callable[[str], dict]`.
- `envoyeur_session(secrets: Path, url="http://127.0.0.1:8091/send", ouvrir=urlopen) -> Callable[[str], None]` (lève si clé absente ou HTTP ≠ 200).
- `relayeur(session: Callable[[str], None], avis: Callable[[str], None]) -> Callable[[str], None]` : essaie Session, sinon `avis("(Session indisponible) " + texte)` ; lève si les deux échouent.

- [ ] Step 1: tests
  - `test_message_ok` : sélecteur `rector`, objet `Test`, garde `{"blocked": False, "risk": "low", "clean_text": "Bonjour"}` → `"Courriel de Grok Bot (rector) — « Test »\n\nBonjour"` ; sans sélecteur → pas de parenthèses.
  - `test_message_suspect` (risk medium → ligne d'attention avant le texte) ; `test_message_bloque` (pas de texte, motif) ; `test_message_garde_injoignable` (garde lève → pas de texte) ; `test_message_coupe` (5 000 car. → 3 500 + mention).
  - `test_envoyeur_session` : faux `secrets.env` avec `SESSION_BOT_MNEMONIC='a b'`, jeton et identifiant → requête Bearer, corps `{"to", "text"}` ; `test_envoyeur_sans_jeton` → lève sans requête.
  - `test_relayeur_repli` : session lève → avis reçoit le texte préfixé ; les deux lèvent → exception.
- [ ] Step 2: FAIL. Step 3: implémenter. Step 4: `python3 -m pytest tiron_mail -q` → PASS.
- [ ] Step 5: commit `feat(relais): garde d'injection et envoi sur Session`.

### Task 3: branchement dans la relève

**Files:**
- Modify: `tiron_mail/releve.py` (`relever(…, relayer=None)`), `tiron_mail/__main__.py`
- Test: `tiron_mail/tests/test_releve.py`

**Interfaces:** Consumes tâches 1-2. Dans la boucle, après l'enregistrement du Message-ID dans `traites` : si `cfg.relayes` et `relayer` et `selecteur_relaye(exp, cfg.relayes) is not None` → `authentifier_relaye` ; motif → refus comme aujourd'hui ; sinon `relayer(message_relais(sel, objet, texte, garde))` (exception → stderr), rangé dans Traite, bilan `relayé : <exp>`, `continue`. `relever` reçoit aussi `garde` (défaut `garde_injection()`). `__main__` : `relayer=relayeur(envoyeur_session(SECRETS), notifier_courriel(env))`.

- [ ] Step 1: tests (faux IMAP existant de `test_releve.py`)
  - `test_relaye_relaye_et_range` : courriel authentique → `relayer` appelé une fois avec l'en-tête Grok, message dans Traite.
  - `test_relaye_jamais_execute` : texte `/wikistatus` → `executer` jamais appelé.
  - `test_relaye_non_authentifie_refuse` : DMARC fail → Refuse, avis « Courriel refusé ».
  - `test_relais_en_echec_n_arrete_pas_la_releve` : `relayer` lève → message suivant traité, premier rangé, pas relayé à la passe suivante.
  - `test_sans_relayes_inchange` : `relayes` vide → refus comme avant.
- [ ] Step 2: FAIL. Step 3: implémenter. Step 4: `python3 -m pytest tiron_mail tiron_cron -q` → PASS.
- [ ] Step 5: commit `feat(releve): relais des courriels de Grok Bot sur Session`.

### Task 4: mise en service et essai réel (avec l'utilisateur)

- [ ] `relayes = ["yc28mybv@mail.grokbot.com"]` sous `[releve]` dans `~/.config/tiron_mail/comptes.toml` (sauvegarde préalable).
- [ ] Fusion, push (accord). Aucun redémarrage : la relève est lancée par minuterie.
- [ ] Critères spec : courriel de Rector → Session ≤ 2 min ; courriel avec `/wikistatus` → relayé, rien exécuté ; broker arrêté (accord) → relais par courriel d'avis.
